"""
Training script for SG-ECCT.

Usage:
    python train.py --code_name BCH_31_11 --num_masks 2 --d_model 128 --num_heads 8

The script:
    1. Loads a pre-generated code from data/codes/<code_name>.npz
    2. Generates spectral attention masks via ConductanceBasedMaskGenerator
    3. Converts masks to n x n attention matrices (Algorithm 1, VN-only)
    4. Trains SGECCT with Adam + cosine annealing and early stopping
    5. Saves best model and training history to experiments/<code_name>_<timestamp>/
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mask_generation import ConductanceBasedMaskGenerator
from model import create_model


# -----------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------

def load_code(code_name: str, codes_dir: str = 'data/codes') -> dict:
    """Load a pre-generated code from an .npz file.

    Returns a dict with keys: G, H, n, k, rate, code_type.
    """
    path = os.path.join(codes_dir, f"{code_name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Code file not found: {path}\n"
            f"Run scripts/prepare_codes.py to generate it."
        )
    d = np.load(path, allow_pickle=True)
    return {
        'G':         d['G'].astype(int),
        'H':         d['H'].astype(int),
        'n':         int(d['n']),
        'k':         int(d['k']),
        'rate':      float(d['rate']),
        'code_type': str(d['code_type']),
    }


# -----------------------------------------------------------------
# Mask construction
# -----------------------------------------------------------------

def build_attention_mask(H_rowsys: np.ndarray, n: int) -> torch.Tensor:
    """Build an n x n boolean attention mask from a row-operated PCM.

    Implements the VN-only submatrix of Algorithm 1 from the ECCT paper:
    two variable nodes may attend to each other iff they share at least
    one check equation in H_rowsys. Self-attention is always allowed.

    Args:
        H_rowsys : (m, n) PCM after row operations (from ConductanceBasedMaskGenerator)
        n        : number of variable nodes
    Returns:
        (n, n) BoolTensor  — True means the pair may attend
    """
    mask = np.eye(n, dtype=bool)
    for row in H_rowsys:
        connected = np.where(row == 1)[0]
        for j in connected:
            mask[j, connected] = True
    return torch.from_numpy(mask)


# -----------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------

class ECCDataset(Dataset):
    """On-the-fly AWGN dataset for error correction code training.

    Samples are generated fresh each epoch to prevent overfitting
    to a fixed noise realisation.

    Args:
        G          : generator matrix (k, n)
        snr_range  : (snr_min, snr_max) in dB, drawn uniformly per sample
        num_samples: dataset length (samples generated on demand)
    """

    def __init__(self, G: np.ndarray, snr_range: tuple, num_samples: int):
        self.G          = torch.from_numpy(G).float()
        self.k, self.n  = G.shape
        self.snr_min, self.snr_max = snr_range
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, _):
        msg       = torch.randint(0, 2, (self.k,)).float()
        codeword  = torch.matmul(msg, self.G) % 2
        tx        = 1.0 - 2.0 * codeword                   # BPSK: 0->+1, 1->-1

        snr_db    = np.random.uniform(self.snr_min, self.snr_max)
        noise_std = 1.0 / np.sqrt(2.0 * 10 ** (snr_db / 10.0))
        rx        = tx + torch.randn(self.n) * noise_std

        return {'received': rx.unsqueeze(-1), 'transmitted': tx}


# -----------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------

def compute_ber(output: torch.Tensor, transmitted: torch.Tensor) -> float:
    """Bit error rate between decoded bits and transmitted BPSK symbols."""
    pred = (output.squeeze(-1) >= 0).float()
    true = (transmitted + 1.0) / 2.0
    return ((pred != true).float().sum() / transmitted.numel()).item()


def compute_fer(output: torch.Tensor, transmitted: torch.Tensor) -> float:
    """Frame error rate between decoded bits and transmitted BPSK symbols."""
    pred  = (output.squeeze(-1) >= 0).float()
    true  = (transmitted + 1.0) / 2.0
    errs  = (pred != true).float().sum(dim=1)
    return ((errs > 0).float().sum() / transmitted.size(0)).item()


def compute_syndrome(output: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Return H @ decoded_bits mod 2.  Zero iff codeword is valid."""
    decoded = (output.squeeze(-1) >= 0).float()
    return torch.matmul(decoded, H.T) % 2


# -----------------------------------------------------------------
# Train / validate one epoch
# -----------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    tot_loss = tot_ber = tot_fer = n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        rx = batch['received'].to(device)
        tx = batch['transmitted'].to(device)

        out  = model(rx)
        loss = criterion(out.squeeze(-1), tx)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        with torch.no_grad():
            ber = compute_ber(out, tx)
            fer = compute_fer(out, tx)

        tot_loss += loss.item()
        tot_ber  += ber
        tot_fer  += fer
        n_batches += 1
        pbar.set_postfix(loss=f'{tot_loss/n_batches:.4f}',
                         BER=f'{tot_ber/n_batches:.4f}')

    return {'loss': tot_loss / n_batches,
            'ber':  tot_ber  / n_batches,
            'fer':  tot_fer  / n_batches}


def validate(model, loader, criterion, device):
    model.eval()
    tot_loss = tot_ber = tot_fer = n_batches = 0

    with torch.no_grad():
        for batch in loader:
            rx  = batch['received'].to(device)
            tx  = batch['transmitted'].to(device)
            out = model(rx)

            tot_loss  += criterion(out.squeeze(-1), tx).item()
            tot_ber   += compute_ber(out, tx)
            tot_fer   += compute_fer(out, tx)
            n_batches += 1

    return {'loss': tot_loss / n_batches,
            'ber':  tot_ber  / n_batches,
            'fer':  tot_fer  / n_batches}


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main(args):
    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load code
    code = load_code(args.code_name, args.codes_dir)
    H, G = code['H'], code['G']
    n, k = code['n'], code['k']
    print(f"Code: ({n},{k})  rate={code['rate']:.3f}  type={code['code_type']}")

    # Generate spectral masks and convert to n x n attention matrices
    print(f"Generating {args.num_masks} spectral mask(s)...")
    mask_gen = ConductanceBasedMaskGenerator(
        pcm_matrix=H,
        num_masks=args.num_masks,
        reweight_method=args.reweight_method,
        verbose=False,
    )
    mask_gen.generate_masks()
    mask_matrices = [
        build_attention_mask(H_rowsys, n)
        for H_rowsys in mask_gen.analysis_data['H_rowsys_all']
    ]
    densities = [m.float().mean().item() for m in mask_matrices]
    print(f"Mask densities: {[f'{d:.1%}' for d in densities]}")

    # Model
    model = create_model(
        pcm_matrix=H,
        mask_matrices=mask_matrices,
        num_masks=args.num_masks,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        verbose=True,
    ).to(device)

    # Data
    snr_range    = (args.snr_min, args.snr_max)
    train_loader = DataLoader(
        ECCDataset(G, snr_range, args.train_samples),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        ECCDataset(G, snr_range, args.val_samples),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Optimiser
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    # Training loop
    best_ber    = float('inf')
    patience_ct = 0
    history     = {k: [] for k in
                   ['train_loss', 'train_ber', 'train_fer',
                    'val_loss',   'val_ber',   'val_fer']}

    print(f"\nTraining for up to {args.epochs} epochs "
          f"(early stop patience={args.patience})\n")

    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        train_m = train_epoch(model, train_loader, optimizer,
                              criterion, device, epoch)
        val_m   = validate(model, val_loader, criterion, device)
        scheduler.step()

        for key in ['loss', 'ber', 'fer']:
            history[f'train_{key}'].append(train_m[key])
            history[f'val_{key}'].append(val_m[key])

        print(f"Epoch {epoch:4d}/{args.epochs} | lr={lr:.2e} | "
              f"train BER={train_m['ber']:.4f} | "
              f"val BER={val_m['ber']:.4f} FER={val_m['fer']:.3f}",
              end='')

        if val_m['ber'] < best_ber:
            best_ber    = val_m['ber']
            patience_ct = 0
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_ber':          best_ber,
                'args':             vars(args),
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  [best]", end='')
        else:
            patience_ct += 1

        print()

        if patience_ct >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    # Save final checkpoint and history
    torch.save({'model_state_dict': model.state_dict(),
                'history': history},
               os.path.join(args.save_dir, 'final_model.pt'))
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val BER: {best_ber:.4f}")
    print(f"Saved to: {args.save_dir}")


# -----------------------------------------------------------------
# Tests  (python train.py --test)
# -----------------------------------------------------------------

def _run_tests():
    import sys
    PASS = lambda s: print(f"  PASS  {s}")

    # ── Test 1: ECCDataset shapes and value ranges ────────────
    print("Test 1: ECCDataset")
    d   = np.load('data/codes/BCH_31_11.npz')
    G   = d['G'].astype(int)
    k, n = G.shape
    ds  = ECCDataset(G, snr_range=(1.0, 6.0), num_samples=200)
    assert len(ds) == 200
    sample = ds[0]
    assert sample['received'].shape    == (n, 1), "received shape wrong"
    assert sample['transmitted'].shape == (n,),   "transmitted shape wrong"
    assert set(sample['transmitted'].tolist()).issubset({-1.0, 1.0}), \
        "transmitted values not in {-1, +1}"
    PASS(f"len={len(ds)}, received={sample['received'].shape}, "
         f"tx values in {{-1,+1}}")

    # ── Test 2: compute_ber / compute_fer on known inputs ─────
    print("Test 2: compute_ber and compute_fer")
    # Both functions expect batched inputs: tx (batch, n), out (batch, n, 1)
    tx          = torch.tensor([[ 1., -1.,  1.,  1., -1.],
                                 [-1.,  1., -1.,  1.,  1.]])
    out_perfect = tx.unsqueeze(-1)       # sign matches tx exactly
    out_wrong   = (-tx).unsqueeze(-1)    # all bits flipped

    assert compute_ber(out_perfect, tx) == 0.0, "BER should be 0 for perfect output"
    assert compute_fer(out_perfect, tx) == 0.0, "FER should be 0 for perfect output"
    assert compute_ber(out_wrong,   tx) == 1.0, "BER should be 1 for all-wrong output"
    assert compute_fer(out_wrong,   tx) == 1.0, "FER should be 1 for all-wrong output"
    PASS("BER=0/FER=0 for perfect, BER=1/FER=1 for all-wrong")

    # ── Test 3: build_attention_mask from H_rowsys ────────────
    print("Test 3: build_attention_mask")
    H = d['H'].astype(int)
    mask_gen = ConductanceBasedMaskGenerator(
        pcm_matrix=H, num_masks=2, verbose=False
    )
    mask_gen.generate_masks()
    for t, H_rowsys in enumerate(mask_gen.analysis_data['H_rowsys_all']):
        attn = build_attention_mask(H_rowsys, n)
        assert attn.shape   == (n, n),         f"mask {t} wrong shape"
        assert attn.dtype   == torch.bool,     f"mask {t} not bool"
        assert attn.diagonal().all(),           f"mask {t} diagonal not all True"
        assert attn.equal(attn.T),             f"mask {t} not symmetric"
    PASS(f"2 masks: shape=({n},{n}), bool, diagonal=True, symmetric")

    print("\nAll 3 tests passed.")


# -----------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SG-ECCT")
    parser.add_argument('--test', action='store_true',
                        help='Run built-in tests instead of training')

    # Code
    parser.add_argument('--code_name',  type=str, default='BCH_31_11')
    parser.add_argument('--codes_dir',  type=str, default='data/codes')

    # Masks
    parser.add_argument('--num_masks',       type=int, default=2)
    parser.add_argument('--reweight_method', type=str,
                        default='degree_normalized',
                        choices=['degree_normalized', 'jaccard', 'penalty'])

    # Model
    parser.add_argument('--d_model',    type=int,   default=128)
    parser.add_argument('--num_heads',  type=int,   default=8)
    parser.add_argument('--num_layers', type=int,   default=4)
    parser.add_argument('--dropout',    type=float, default=0.1)

    # Training
    parser.add_argument('--epochs',        type=int,   default=1000)
    parser.add_argument('--patience',      type=int,   default=30)
    parser.add_argument('--batch_size',    type=int,   default=256)
    parser.add_argument('--lr',            type=float, default=5e-4)
    parser.add_argument('--train_samples', type=int,   default=100000)
    parser.add_argument('--val_samples',   type=int,   default=10000)
    parser.add_argument('--snr_min',       type=float, default=1.0)
    parser.add_argument('--snr_max',       type=float, default=6.0)

    # System
    parser.add_argument('--gpu',         type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--save_dir',    type=str, default=None)

    args = parser.parse_args()

    if args.test:
        _run_tests()
    else:
        if args.save_dir is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            args.save_dir = os.path.join('experiments',
                                         f'{args.code_name}_{ts}')
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        main(args)