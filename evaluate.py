"""
Evaluation script for SG-ECCT: BER/FER sweep over SNR.

Usage:
    python evaluate.py --checkpoint experiments/BCH_31_11_<ts>/best_model.pt

The script:
    1. Loads the checkpoint and reconstructs the model + masks from the
       saved training args (code_name, num_masks, d_model, etc.)
    2. Sweeps SNR from snr_min to snr_max in snr_step increments
    3. At each SNR point, generates Monte Carlo samples until
       min_fer_events frame errors are observed (or max_samples is reached)
    4. Reports BER, FER, and -ln(BER) matching Table 9 of the paper
    5. Saves results to <checkpoint_dir>/eval_results.json and a BER plot
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mask_generation import ConductanceBasedMaskGenerator
from model import create_model
from train import load_code, build_attention_mask


# -----------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------

def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Reconstruct model from a checkpoint saved by train.py.

    Loads the code, regenerates spectral masks, rebuilds the model
    architecture, and restores weights.

    Returns:
        model   : SGECCT in eval mode
        G, H    : generator and parity-check matrices
        n, k    : code dimensions
        tr_args : dict of training arguments
    """
    ckpt    = torch.load(checkpoint_path, map_location=device)
    tr_args = ckpt['args']

    # Load code
    code = load_code(tr_args['code_name'], tr_args.get('codes_dir', 'data/codes'))
    H, G = code['H'], code['G']
    n, k = code['n'], code['k']

    # Regenerate masks (deterministic for a given H)
    mask_gen = ConductanceBasedMaskGenerator(
        pcm_matrix=H,
        num_masks=tr_args['num_masks'],
        reweight_method=tr_args.get('reweight_method', 'degree_normalized'),
        verbose=False,
    )
    mask_gen.generate_masks()
    mask_matrices = [
        build_attention_mask(Hr, n)
        for Hr in mask_gen.analysis_data['H_rowsys_all']
    ]

    # Build model and load weights
    model = create_model(
        pcm_matrix=H,
        mask_matrices=mask_matrices,
        num_masks=tr_args['num_masks'],
        d_model=tr_args['d_model'],
        num_heads=tr_args['num_heads'],
        num_layers=tr_args['num_layers'],
        dropout=tr_args.get('dropout', 0.1),
        verbose=False,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    return model, G, H, n, k, tr_args


# -----------------------------------------------------------------
# Evaluation at one SNR point
# -----------------------------------------------------------------

def evaluate_snr(
    model,
    G: np.ndarray,
    n: int,
    snr_db: float,
    batch_size: int,
    device: torch.device,
    min_fer_events: int = 100,
    max_samples: int = 1_000_000,
) -> dict:
    """Monte Carlo BER/FER estimation at a single SNR point.

    Continues sampling until `min_fer_events` frame errors are accumulated
    or `max_samples` samples have been processed, whichever comes first.
    This ensures reliable estimates at high SNR.

    Returns:
        dict with keys: ber, fer, neg_ln_ber, bit_errors, frame_errors,
                        total_bits, total_frames
    """
    k = G.shape[0]
    G_t = torch.from_numpy(G).float()

    noise_std      = 1.0 / np.sqrt(2.0 * 10 ** (snr_db / 10.0))
    bit_errors     = 0
    frame_errors   = 0
    total_bits     = 0
    total_frames   = 0

    while frame_errors < min_fer_events and total_frames < max_samples:
        bs   = min(batch_size, max_samples - total_frames)
        msgs = torch.randint(0, 2, (bs, k)).float()
        cw   = torch.matmul(msgs, G_t) % 2          # (bs, n)
        tx   = 1.0 - 2.0 * cw                        # BPSK
        rx   = tx + torch.randn(bs, n) * noise_std

        with torch.no_grad():
            out     = model(rx.unsqueeze(-1).to(device)).squeeze(-1).cpu()
        decoded = (out < 0).float()                   # 1 where model says bit=1

        errs_bit   = (decoded != cw).float()
        bit_errors   += errs_bit.sum().item()
        frame_errors += (errs_bit.sum(dim=1) > 0).sum().item()
        total_bits   += bs * n
        total_frames += bs

    ber = bit_errors   / total_bits   if total_bits   > 0 else float('nan')
    fer = frame_errors / total_frames if total_frames > 0 else float('nan')
    neg_ln_ber = -np.log(ber) if ber > 0 else float('inf')

    return {
        'ber':          ber,
        'fer':          fer,
        'neg_ln_ber':   neg_ln_ber,
        'bit_errors':   int(bit_errors),
        'frame_errors': int(frame_errors),
        'total_bits':   int(total_bits),
        'total_frames': int(total_frames),
    }


# -----------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------

def plot_ber(snr_values, ber_values, code_name: str, save_path: str):
    """Save a semilogy BER-vs-SNR plot."""
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.semilogy(snr_values, ber_values, 'o-', linewidth=2,
                markersize=6, label='SG-ECCT')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Bit Error Rate')
    ax.set_title(f'BER — {code_name}')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved → {save_path}")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main(args):
    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, G, H, n, k, tr_args = load_checkpoint(args.checkpoint, device)
    code_name = tr_args['code_name']
    print(f"Code: ({n},{k})  rate={k/n:.3f}  masks={tr_args['num_masks']}")

    snr_values = np.arange(args.snr_min, args.snr_max + args.snr_step / 2,
                           args.snr_step)
    print(f"\nSNR sweep: {snr_values} dB")
    print(f"min_fer_events={args.min_fer_events}, max_samples={args.max_samples}\n")

    records = []
    print(f"{'SNR':>6}  {'BER':>12}  {'FER':>10}  {'−ln(BER)':>10}  "
          f"{'FE':>7}  {'frames':>9}")
    print("-" * 62)

    for snr_db in snr_values:
        r = evaluate_snr(
            model, G, n, float(snr_db),
            batch_size=args.batch_size,
            device=device,
            min_fer_events=args.min_fer_events,
            max_samples=args.max_samples,
        )
        r['snr_db'] = float(snr_db)
        records.append(r)
        print(f"{snr_db:6.1f}  {r['ber']:12.6e}  {r['fer']:10.6f}  "
              f"{r['neg_ln_ber']:10.4f}  {r['frame_errors']:7d}  "
              f"{r['total_frames']:9d}")

    # Save JSON
    save_dir   = os.path.dirname(args.checkpoint)
    json_path  = os.path.join(save_dir, 'eval_results.json')
    with open(json_path, 'w') as f:
        json.dump({'code_name': code_name, 'results': records}, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # Plot
    plot_ber(
        snr_values,
        [r['ber'] for r in records],
        code_name,
        os.path.join(save_dir, 'ber_curve.png'),
    )


# -----------------------------------------------------------------
# Tests  (python evaluate.py --test)
# -----------------------------------------------------------------

def _run_tests():
    PASS = lambda s: print(f"  PASS  {s}")

    # ── Test 1: evaluate_snr returns valid BER/FER ────────────
    print("Test 1: evaluate_snr output ranges")
    d    = np.load('data/codes/BCH_31_11.npz')
    G, H = d['G'].astype(int), d['H'].astype(int)
    n, k = int(d['n']), int(d['k'])

    # Tiny random model (untrained, just checks plumbing)
    mask = torch.ones(n, n, dtype=torch.bool)
    model = create_model(H, [mask, mask], num_masks=2,
                         d_model=16, num_heads=4, num_layers=1,
                         dropout=0.0, verbose=False)
    model.eval()

    device = torch.device('cpu')
    r = evaluate_snr(model, G, n, snr_db=4.0, batch_size=64,
                     device=device, min_fer_events=5, max_samples=500)

    assert 0.0 <= r['ber'] <= 1.0,   f"BER out of range: {r['ber']}"
    assert 0.0 <= r['fer'] <= 1.0,   f"FER out of range: {r['fer']}"
    assert r['total_frames'] > 0,    "No frames evaluated"
    assert r['total_bits'] == r['total_frames'] * n, "Bit count mismatch"
    PASS(f"BER={r['ber']:.4f}, FER={r['fer']:.4f}, "
         f"frames={r['total_frames']}, FE={r['frame_errors']}")

    # ── Test 2: BER not worse than random at very low SNR ─────
    print("Test 2: BER <= 0.5 at SNR=0 dB (sanity check)")
    r_low = evaluate_snr(model, G, n, snr_db=0.0, batch_size=256,
                         device=device, min_fer_events=10, max_samples=2000)
    # Random guessing gives BER=0.5; any model should be <= 0.5
    assert r_low['ber'] <= 0.5 + 1e-6, f"BER={r_low['ber']} > 0.5 at 0 dB"
    PASS(f"BER={r_low['ber']:.4f} <= 0.5 at 0 dB")

    # ── Test 3: min_fer_events stopping criterion ─────────────
    print("Test 3: min_fer_events stopping criterion")
    r_stop = evaluate_snr(model, G, n, snr_db=1.0, batch_size=128,
                          device=device, min_fer_events=20, max_samples=10_000)
    # Either we hit min_fer_events or exhausted max_samples
    assert (r_stop['frame_errors'] >= 20 or
            r_stop['total_frames'] >= 10_000), \
        "Stopping criterion not reached"
    PASS(f"Stopped at frame_errors={r_stop['frame_errors']}, "
         f"frames={r_stop['total_frames']}")

    print("\nAll 3 tests passed.")


# -----------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SG-ECCT")
    parser.add_argument('--test', action='store_true',
                        help='Run built-in tests instead of evaluation')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to best_model.pt from train.py')

    # SNR sweep (paper: 1–7 dB, step 1)
    parser.add_argument('--snr_min',  type=float, default=1.0)
    parser.add_argument('--snr_max',  type=float, default=7.0)
    parser.add_argument('--snr_step', type=float, default=1.0)

    # Monte Carlo settings
    parser.add_argument('--min_fer_events', type=int, default=100,
                        help='Stop each SNR when this many frame errors observed')
    parser.add_argument('--max_samples',    type=int, default=1_000_000,
                        help='Hard cap on samples per SNR point')
    parser.add_argument('--batch_size',     type=int, default=256)

    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    if args.test:
        _run_tests()
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for evaluation")
        main(args)