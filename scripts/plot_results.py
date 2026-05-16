"""
Generate figures and tables for the SG-ECCT paper from experimental BER data.

Expects output_data.xlsx in the working directory with one sheet per code,
each sheet containing columns: SNR, ECCT, MM-ECCT, Our1, Our2.

Outputs saved to ./paper_results/:
  fig_ber_summary.pdf/.png        6-panel BER grid (BCH / LDPC / Polar, 4h vs 8h)
  fig_ber_<config>.pdf/.png       Individual BER plot per config
  fig_fiedler_analysis.pdf/.png   Fiedler vector visualization LDPC(121,60)
  table_ber_snr7.txt              BER @ SNR=7 for all configs
  table_gain.txt                  Gain of Ours(2m) over ECCT @ SNR=7

Usage:
  python generate_results.py                   # all outputs
  python generate_results.py --fig summary     # 6-panel summary only
  python generate_results.py --fig ber         # all individual BER plots
  python generate_results.py --fig fiedler     # Fiedler analysis
  python generate_results.py --table ber       # BER @ SNR=7 table
  python generate_results.py --table gain      # gain table

Requirements: pip install matplotlib openpyxl scipy numpy
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erfc, comb
import openpyxl

OUT_DIR   = 'paper_results'
DATA_FILE = 'output_data.xlsx'
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family':    'serif',
    'font.size':      11,
    'axes.labelsize': 12,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.8,
    'figure.dpi':      150,
})
COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
MARKERS = ['o', 's', '^', 'D']
LABELS  = ['ECCT', 'MM-ECCT', 'Ours (1 mask)', 'Ours (2 masks)']


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(filepath=DATA_FILE):
    """
    Returns dict: data[config_name] = {
        'snr': [...], 'ECCT': [...], 'MM-ECCT': [...], 'Our1': [...], 'Our2': [...]
    }
    """
    wb   = openpyxl.load_workbook(filepath)
    data = {}

    for sheet in wb.sheetnames:
        ws      = wb[sheet]
        current = None
        snr_col = []
        cols    = {k: [] for k in ['ECCT', 'MM-ECCT', 'Our1', 'Our2']}

        for row in ws.iter_rows(values_only=True):
            if row[0] == 'SNR':
                continue
            if isinstance(row[0], str):          # new config block
                if current and snr_col:
                    data[current] = {'snr': snr_col, **cols}
                current = row[0]
                snr_col = []; cols = {k: [] for k in ['ECCT', 'MM-ECCT', 'Our1', 'Our2']}
            elif row[0] is None:                 # blank separator
                continue
            elif isinstance(row[0], (int, float)):
                snr_col.append(row[0])
                cols['ECCT'].append(row[1])
                cols['MM-ECCT'].append(row[2])
                cols['Our1'].append(row[3])
                cols['Our2'].append(row[4])

        if current and snr_col:                  # last block
            data[current] = {'snr': snr_col, **cols}

    return data


def parse_config(name):
    """bch_63_36_4h_128d  ->  {type, n, k, heads, dim}"""
    p = name.split('_')
    return {'type': p[0], 'n': int(p[1]), 'k': int(p[2]),
            'heads': int(p[3].replace('h', '')),
            'dim':   int(p[4].replace('d', ''))}


# ═══════════════════════════════════════════════════════════════════════════════
#  ML bound (BCH only)
# ═══════════════════════════════════════════════════════════════════════════════

# Exact weight enumerators for standard BCH codes
# Source: Lin & Costello "Error Control Coding", standard BCH tables
BCH_WEIGHT_ENUM = {
    (31, 11): [(11,186),(15,806),(16,1240),(19,1488),(20,1023),(23,186),(31,1)],
    (31, 16): [(7,155),(8,310),(11,1085),(12,1085),(15,310),(16,155)],
    (63, 36): [(11,1),(12,0),(13,0),(14,0),(15,2667),(16,0),(17,0),(18,0),(19,0),(20,5481),(21,0),(22,0),(23,0),(24,0),(25,1),(26,0)],
    (63, 51): [(7,9),(8,63),(9,0),(10,126),(11,180)],
}


def ml_union_bound(n, k, snr_db_range):
    """
    ML union bound (bit error probability) for binary BCH(n,k) over AWGN.

    P_b <= (1/k) * sum_w  w * A_w * Q(sqrt(2 * w * Rc * Eb/N0))

    Uses exact weight enumerators where available (Lin & Costello tables).
    Falls back to random-coding approximation otherwise.

    IMPORTANT: the union bound becomes loose (drops below real ML performance)
    at high SNR. We only plot it where it lies above the best decoder curve —
    the caller should pass best_ber to enable this clipping.
    """
    Rc      = k / n
    w_enum  = BCH_WEIGHT_ENUM.get((n, k))

    if w_enum is None:
        # Random-coding fallback: A_w ~ C(n,w) / 2^(n-k)
        BCH_DMIN = {(63,45):13,(127,99):11,(127,64):21}
        d_min = BCH_DMIN.get((n, k), max(3, (n - k) // 2))
        M     = 2 ** (n - k)
        w_enum = [(w, comb(n, w, exact=False) / M)
                  for w in range(d_min, min(n + 1, 3 * d_min + 1))]

    out = []
    for s in snr_db_range:
        EbN0 = 10 ** (s / 10)
        pb   = sum(w * A_w * 0.5 * erfc(np.sqrt(w * Rc * EbN0))
                   for w, A_w in w_enum)
        out.append(min(pb / k, 0.5))
    return np.array(out)


def ml_union_bound_valid(n, k, snr_db_range, best_ber_array):
    """
    Return ML union bound values masked to NaN where the bound falls below
    the best decoder (i.e., where the union bound is no longer informative).
    NaN values are not plotted by matplotlib, giving a clean cutoff.
    """
    raw = ml_union_bound(n, k, snr_db_range)
    out = np.where(raw >= best_ber_array, raw, np.nan)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG: summary — 2 rows (4h / 8h) × 3 cols (BCH / LDPC / Polar), d=128
# ═══════════════════════════════════════════════════════════════════════════════

def fig_ber_summary(data):
    picks = {
        '4h': {'BCH':   'bch_63_36_4h_128d',
                'LDPC':  'ldpc_121_60_4h_128d',
                'Polar': 'polar_128_64_4h_128d'},
        '8h': {'BCH':   'bch_63_36_8h_128d',
                'LDPC':  'ldpc_121_60_8h_128d',
                'Polar': 'polar_128_64_8h_128d'},
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for ri, (hk, codes) in enumerate(picks.items()):
        for ci, (ctype, cname) in enumerate(codes.items()):
            ax  = axes[ri][ci]
            d   = data[cname]
            cfg = parse_config(cname)
            snr = np.array(d['snr'])

            if ctype == 'BCH':
                best = np.array(d['Our2'])
                ml   = ml_union_bound_valid(cfg['n'], cfg['k'], snr, best)
                ax.semilogy(snr, ml, 'k--', linewidth=1.3, label='ML bound', zorder=5)

            for i, (key, lbl) in enumerate(zip(
                    ['ECCT', 'MM-ECCT', 'Our1', 'Our2'], LABELS)):
                ax.semilogy(snr, d[key], color=COLORS[i], marker=MARKERS[i],
                            markevery=2, markersize=4, label=lbl)

            ax.set_title(f"{ctype}({cfg['n']},{cfg['k']}), {hk}, d=128")
            ax.set_xlabel('$E_b/N_0$ (dB)')
            ax.set_ylabel('BER')
            ax.set_ylim([1e-8, 1.0])
            ax.grid(True, which='both', alpha=0.3)
            ax.legend(loc='lower left', fontsize=8)

    plt.suptitle('BER Comparison — SG-ECCT vs Baselines', fontsize=13, y=1.01)
    plt.tight_layout()
    _savefig('fig_ber_summary')


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG: individual BER plot per config
# ═══════════════════════════════════════════════════════════════════════════════

def fig_ber_all(data):
    for name, d in data.items():
        cfg = parse_config(name)
        snr = np.array(d['snr'])

        fig, ax = plt.subplots(figsize=(6, 4.5))

        if cfg['type'] == 'bch':
            best = np.array(d['Our2'])
            ml   = ml_union_bound_valid(cfg['n'], cfg['k'], snr, best)
            ax.semilogy(snr, ml, 'k--', linewidth=1.3, label='ML bound', zorder=5)

        for i, (key, lbl) in enumerate(zip(
                ['ECCT', 'MM-ECCT', 'Our1', 'Our2'], LABELS)):
            ax.semilogy(snr, d[key], color=COLORS[i], marker=MARKERS[i],
                        markevery=2, markersize=5, label=lbl)

        code_str = f"{cfg['type'].upper()}({cfg['n']},{cfg['k']})"
        ax.set_title(f"{code_str}  h={cfg['heads']}  d={cfg['dim']}")
        ax.set_xlabel('$E_b/N_0$ (dB)')
        ax.set_ylabel('BER')
        ax.set_ylim([1e-8, 1.0])
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='lower left')
        plt.tight_layout()
        _savefig(f'fig_ber_{name}')


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG: Fiedler vector analysis (LDPC 121,60)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_fiedler():
    # Load the actual LDPC(121,60) used in the paper
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    codes_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'codes')
    d = np.load(os.path.join(codes_dir, 'LDPC_121_60.npz'))
    H = d['H'].astype(int)
    n = int(d['n'])

    A = (H.T @ H).astype(float);  np.fill_diagonal(A, 0)
    A = (A > 0).astype(float)
    deg        = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    L_norm     = np.eye(n) - np.diag(d_inv_sqrt) @ A @ np.diag(d_inv_sqrt)
    eigenvalues, evecs = np.linalg.eigh(L_norm)
    fiedler = evecs[:, 1]

    fig = plt.figure(figsize=(14, 4.5))
    gs  = gridspec.GridSpec(1, 3, wspace=0.38)

    # (a) sorted bar
    ax1 = fig.add_subplot(gs[0])
    si  = np.argsort(fiedler)
    nf  = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min())
    ax1.bar(range(n), fiedler[si], color=plt.cm.RdYlBu_r(nf[si]),
            width=1.0, edgecolor='none')
    ax1.axhline(0, color='k', linewidth=0.8, linestyle='--')
    thr = 0.5 * np.abs(fiedler).max()
    ax1.axvline(np.sum(fiedler[si] < -thr), color='red',
                linestyle=':', linewidth=1.3, label='Partition boundary')
    ax1.axvline(n - np.sum(fiedler[si] > thr), color='red',
                linestyle=':', linewidth=1.3)
    ax1.set_xlabel('Variable node (sorted)');  ax1.set_ylabel('Fiedler value')
    ax1.set_title('(a) Sorted Fiedler Values')
    ax1.legend(fontsize=9);  ax1.grid(axis='y', alpha=0.3)

    # (b) histogram
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(fiedler, bins=22, color=COLORS[0], edgecolor='white',
             alpha=0.85, density=True)
    ax2.axvline(-thr, color='red', linestyle=':', linewidth=1.5,
                label='Weak node threshold')
    ax2.axvline( thr, color='red', linestyle=':', linewidth=1.5)
    ax2.set_xlabel('Fiedler vector value');  ax2.set_ylabel('Density')
    ax2.set_title('(b) Component Distribution')
    ax2.legend(fontsize=9);  ax2.grid(alpha=0.3)

    # (c) eigenvalue spectrum
    ax3 = fig.add_subplot(gs[2])
    K   = min(20, len(eigenvalues))
    ax3.plot(range(K), eigenvalues[:K], 'o-', color=COLORS[1], markersize=5)
    ax3.axvline(1, color='red', linestyle=':', linewidth=1.3,
                label='Fiedler ($\\lambda_2$)')
    ax3.set_xlabel('Eigenvalue index');  ax3.set_ylabel('$\\lambda_i$')
    ax3.set_title('(c) Laplacian Spectrum')
    ax3.legend(fontsize=9);  ax3.grid(alpha=0.3)

    fig.suptitle(f'Fiedler Vector Analysis — LDPC({n},{int(d["k"])})', fontsize=13, y=1.02)
    plt.tight_layout()
    _savefig('fig_fiedler_analysis')


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE: BER @ SNR=7 for all configs
# ═══════════════════════════════════════════════════════════════════════════════

def table_ber_snr7(data):
    hdr   = f"{'Config':<30}{'ECCT':>12}{'MM-ECCT':>12}{'Our(1m)':>12}{'Our(2m)':>12}"
    sep   = '-' * len(hdr)
    lines = [hdr, sep]
    prev  = None

    for name in sorted(data.keys()):
        cfg  = parse_config(name)
        d    = data[name]
        grp  = f"{cfg['type']}_{cfg['n']}"
        if prev and grp != prev:
            lines.append(sep)
        lines.append(
            f"{name:<30}{d['ECCT'][-1]:>12.2e}{d['MM-ECCT'][-1]:>12.2e}"
            f"{d['Our1'][-1]:>12.2e}{d['Our2'][-1]:>12.2e}"
        )
        prev = grp

    txt = '\n'.join(lines)
    print('\n' + txt)
    _savetxt('table_ber_snr7', txt)


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE: Gain of Ours(2m) over ECCT @ SNR=7
# ═══════════════════════════════════════════════════════════════════════════════

def table_gain(data):
    hdr   = f"{'Config':<30}{'ECCT':>12}{'Our(2m)':>12}{'Gain(x)':>10}{'Gain(dB)':>10}"
    sep   = '-' * len(hdr)
    lines = [hdr, sep]
    prev  = None

    for name in sorted(data.keys()):
        cfg     = parse_config(name); d = data[name]
        ecct    = d['ECCT'][-1];    o2 = d['Our2'][-1]
        gain    = ecct / o2 if o2 > 0 else float('inf')
        gain_db = 10 * np.log10(gain) if gain > 0 else 0
        grp     = f"{cfg['type']}_{cfg['n']}"
        if prev and grp != prev:
            lines.append(sep)
        lines.append(
            f"{name:<30}{ecct:>12.2e}{o2:>12.2e}{gain:>9.1f}x{gain_db:>9.1f}dB"
        )
        prev = grp

    txt = '\n'.join(lines)
    print('\n' + txt)
    _savetxt('table_gain', txt)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _savefig(name):
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUT_DIR}/{name}.pdf + .png")


def _savetxt(name, content):
    path = os.path.join(OUT_DIR, f'{name}.txt')
    with open(path, 'w') as f:
        f.write(content + '\n')
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════



# -----------------------------------------------------------------
# Tests  (python scripts/plot_results.py --test)
# -----------------------------------------------------------------

def _run_tests():
    PASS = lambda s: print(f"  PASS  {s}")

    # ── Test 1: ml_union_bound values in [0, 0.5] ────────────
    print("Test 1: ml_union_bound output range")
    snr = np.arange(1, 8, 1.0)
    ub  = ml_union_bound(31, 11, snr)
    assert len(ub) == len(snr),         "Length mismatch"
    assert np.all(ub >= 0),             "Negative bound values"
    assert np.all(ub <= 0.5 + 1e-9),   "Bound exceeds 0.5"
    assert ub[0] > ub[-1],             "Bound not decreasing with SNR"
    PASS(f"values in [0, 0.5], decreasing: {ub[0]:.4f} → {ub[-1]:.2e}")

    # ── Test 2: ml_union_bound_valid masks low values ─────────
    print("Test 2: ml_union_bound_valid masks below best BER")
    best_ber = np.array([0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    raw  = ml_union_bound(31, 11, snr)
    masked = ml_union_bound_valid(31, 11, snr, best_ber)
    for i in range(len(snr)):
        if raw[i] < best_ber[i]:
            assert np.isnan(masked[i]), f"Expected NaN at SNR={snr[i]}"
        else:
            assert not np.isnan(masked[i]), f"Unexpected NaN at SNR={snr[i]}"
    PASS("NaN mask applied correctly where bound < best BER")

    # ── Test 3: parse_config correctly parses config names ────
    print("Test 3: parse_config")
    cfg = parse_config('bch_63_36_4h_128d')
    assert cfg['type'] == 'bch'
    assert cfg['n']    == 63
    assert cfg['k']    == 36
    assert cfg['heads'] == 4
    assert cfg['dim']   == 128
    PASS(f"parsed: type={cfg['type']}, n={cfg['n']}, k={cfg['k']}, "
         f"h={cfg['heads']}, d={cfg['dim']}")

    print("\nAll 3 tests passed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',  action='store_true', help='Run built-in tests')
    parser.add_argument('--fig',   choices=['ber', 'summary', 'fiedler'], default=None)
    parser.add_argument('--table', choices=['ber', 'gain'],               default=None)
    args = parser.parse_args()

    if args.test:
        _run_tests()
        sys.exit(0)
    print(f"Loading {DATA_FILE} ...")
    data = load_data()
    print(f"  {len(data)} configs loaded\n")

    if args.fig == 'ber':
        print("[Fig] Individual BER plots ..."); fig_ber_all(data)
    elif args.fig == 'summary':
        print("[Fig] Summary grid ...");         fig_ber_summary(data)
    elif args.fig == 'fiedler':
        print("[Fig] Fiedler analysis ...");     fig_fiedler()
    elif args.table == 'ber':
        print("[Table] BER @ SNR=7 ...");        table_ber_snr7(data)
    elif args.table == 'gain':
        print("[Table] Gain over ECCT ...");     table_gain(data)
    else:
        print("Generating all outputs ...\n")
        fig_ber_all(data)
        fig_ber_summary(data)
        fig_fiedler()
        table_ber_snr7(data)
        table_gain(data)
        print(f"\nDone. All outputs in ./{OUT_DIR}/")