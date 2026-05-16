"""
Reproduces Figure 2 of the paper: spectral analysis of BCH(31,11)
via the Fiedler vector (Tanner graph, component distribution, conductance sweep).

Usage:
    python scripts/plot_fiedler.py                        # saves fiedler_BCH31_11.pdf/png
    python scripts/plot_fiedler.py --code LDPC_121_60     # any code in data/codes/
    python scripts/plot_fiedler.py --out results/         # custom output dir
    python scripts/plot_fiedler.py --test                 # run built-in tests

Requirements: pip install matplotlib networkx scipy
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman'] + plt.rcParams['font.serif']

CODES_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'codes')


# -----------------------------------------------------------------
# Spectral helpers
# -----------------------------------------------------------------

def compute_fiedler(H: np.ndarray):
    """Return (fiedler_vec, fiedler_val, A, deg) for the V×V graph of H."""
    A = (H.T @ H).astype(float)
    np.fill_diagonal(A, 0)
    deg       = A.sum(axis=1)
    d_safe    = np.where(deg > 0, deg, 1.0)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d_safe))
    L_norm    = np.eye(H.shape[1]) - D_inv_sqrt @ A @ D_inv_sqrt
    vals, vecs = eigh(L_norm)
    return vecs[:, 1], vals[1], A, deg


def sweep_cut(A: np.ndarray, fvec: np.ndarray, deg: np.ndarray):
    """Return (S_sw, k_star, conductances) from the Fiedler sweep."""
    n      = len(fvec)
    order  = np.argsort(fvec)
    conds  = []
    for k in range(1, n):
        S      = set(order[:k])
        cut    = sum(A[i, j] for i in S for j in range(n) if j not in S)
        vol_S  = sum(deg[i] for i in S)
        vol_Sc = sum(deg[i] for i in range(n) if i not in S)
        denom  = min(vol_S, vol_Sc)
        conds.append(cut / denom if denom > 0 else np.inf)
    k_star = int(np.argmin(conds))
    S_sw   = set(order[:k_star + 1])
    return S_sw, k_star, conds


# -----------------------------------------------------------------
# Plot
# -----------------------------------------------------------------

def plot_fiedler(H: np.ndarray, code_name: str, out_dir: str):
    m, n = H.shape
    fvec, lambda2, A, deg = compute_fiedler(H)
    S_sw, k_star, conds   = sweep_cut(A, fvec, deg)

    # Build Tanner graph
    G_nx = nx.Graph()
    vns  = [f'v{i}' for i in range(n)]
    cns  = [f'c{j}' for j in range(m)]
    G_nx.add_nodes_from(vns, ntype='VN')
    G_nx.add_nodes_from(cns, ntype='CN')
    for j in range(m):
        for i in range(n):
            if H[j, i] == 1:
                G_nx.add_edge(f'v{i}', f'c{j}')

    # Sort VNs by Fiedler value for cleaner partition view
    pos = {}
    vn_sorted = [f'v{i}' for i in np.argsort(fvec)]
    for rank, v in enumerate(vn_sorted):
        pos[v] = (rank / (n - 1), 1.0)
    for j, c in enumerate(cns):
        pos[c] = (j / max(m - 1, 1), 0.0)

    cmap     = plt.cm.RdBu_r
    norm_vn  = mcolors.TwoSlopeNorm(vmin=fvec.min(), vcenter=0.0, vmax=fvec.max())
    vn_colors = [cmap(norm_vn(fvec[i])) for i in range(n)]
    threshold = np.sort(fvec)[k_star]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    # ── Panel A: Tanner graph ─────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('#f9f9f9')
    nx.draw_networkx_edges(G_nx, pos, ax=ax,
                           alpha=0.25, width=0.7, edge_color='#999999')
    nx.draw_networkx_nodes(G_nx, pos, nodelist=cns,
                           node_color='#bbbbbb', node_shape='s',
                           node_size=120, ax=ax)
    nx.draw_networkx_labels(G_nx, pos,
                            labels={c: c[1:] for c in cns},
                            font_size=5.5, ax=ax)
    nx.draw_networkx_nodes(G_nx, pos, nodelist=vns,
                           node_color=vn_colors, node_size=180, ax=ax)
    weak_vns = [f'v{i}' for i in range(n) if i in S_sw]
    nx.draw_networkx_nodes(G_nx, pos, nodelist=weak_vns,
                           node_color=[cmap(norm_vn(fvec[i]))
                                       for i in range(n) if i in S_sw],
                           node_size=220, linewidths=2.0,
                           edgecolors='#d62728', ax=ax)
    nx.draw_networkx_labels(G_nx, pos,
                            labels={f'v{i}': str(i) for i in range(n)},
                            font_size=5.5, ax=ax)
    boundary_x = k_star / (n - 1)
    ax.axvline(boundary_x, ymin=0.05, ymax=0.95,
               color='#d62728', linestyle='--', linewidth=1.8, alpha=0.85)
    ax.text(boundary_x - 0.02, 0.5, 'Weak',   fontsize=9, color='#d62728',
            ha='right', va='center', transform=ax.transAxes,
            rotation=90, fontweight='bold')
    ax.text(boundary_x + 0.02, 0.5, 'Strong', fontsize=9, color='#4393c3',
            ha='left',  va='center', transform=ax.transAxes,
            rotation=90, fontweight='bold')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_vn)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                        fraction=0.04, pad=0.06, aspect=30)
    cbar.set_label('Fiedler value $u_2[i]$', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.legend(handles=[
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#888',
               markersize=8, label='Variable node (VN)'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor='#bbb',
               markeredgecolor='k', markersize=8, label='Check node (CN)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#d62728',
               markeredgecolor='#d62728', markersize=9,
               label=r'Weak cluster $\mathcal{S}_{sw}$'),
    ], fontsize=8, loc='lower right', framealpha=0.9)
    ax.set_title(f'(a) Tanner graph: {code_name}\ncolored by $u_2$'
                 f'  ($\\lambda_2={lambda2:.4f}$)', fontsize=10, pad=8)
    ax.axis('off')

    # ── Panel B: Histogram ────────────────────────────────────
    ax2  = axes[1]
    ax2.set_facecolor('#f9f9f9')
    bins = np.linspace(fvec.min() - 0.01, fvec.max() + 0.01, 20)
    ax2.hist(fvec[[i for i in range(n) if i not in S_sw]],
             bins=bins, color='#4393c3', alpha=0.75,
             label=r'Strong region $\bar{\mathcal{S}}_{sw}$',
             edgecolor='white', linewidth=0.5)
    ax2.hist(fvec[list(S_sw)],
             bins=bins, color='#d62728', alpha=0.80,
             label=r'Weak cluster $\mathcal{S}_{sw}$',
             edgecolor='white', linewidth=0.5)
    ax2.axvline(threshold, color='#333', linestyle='--', linewidth=1.8,
                label=f'Threshold $u_2^*={threshold:.4f}$')
    ax2.axvline(0.0, color='green', linestyle=':', linewidth=1.2,
                label='Zero crossing', alpha=0.8)
    ax2.set_xlabel('Fiedler vector component $u_2[i]$', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_title(f'(b) Distribution of Fiedler\ncomponents — {code_name}',
                  fontsize=10, pad=8)
    ax2.legend(fontsize=8.5, framealpha=0.9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.text(0.97, 0.97,
             f'$|\\mathcal{{S}}_{{sw}}|={len(S_sw)}$\n'
             f'$|\\bar{{\\mathcal{{S}}}}_{{sw}}|={n-len(S_sw)}$',
             transform=ax2.transAxes, fontsize=9, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))

    # ── Panel C: Conductance sweep ────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor('#f9f9f9')
    k_vals = np.arange(1, n)
    ax3.plot(k_vals, conds, color='#4393c3', linewidth=1.8,
             label='$\\phi(\\mathcal{S}_k)$')
    ax3.scatter(k_star + 1, conds[k_star], color='#d62728', zorder=5, s=80,
                label=f'$\\phi^*\\approx{conds[k_star]:.4f}$  at $k={k_star+1}$')
    ax3.axvline(k_star + 1, color='#d62728', linestyle='--',
                linewidth=1.4, alpha=0.7)
    lam2_half  = lambda2 / 2
    sqrt_2lam2 = np.sqrt(2 * lambda2)
    ax3.axhline(lam2_half,  color='#888', linestyle=':', linewidth=1.2,
                label=f'$\\lambda_2/2={lam2_half:.4f}$')
    ax3.axhline(sqrt_2lam2, color='#bbb', linestyle=':', linewidth=1.2,
                label=f'$\\sqrt{{2\\lambda_2}}={sqrt_2lam2:.4f}$')
    ax3.fill_between(k_vals, lam2_half, sqrt_2lam2,
                     alpha=0.12, color='gray', label='Cheeger bound')
    ax3.set_ylim(bottom=conds[k_star] * 0.85,
                 top=min(sqrt_2lam2 * 1.2, np.percentile(conds, 90)))
    ax3.set_xlabel('Sweep index $k$', fontsize=10)
    ax3.set_ylabel('Conductance $\\phi(\\mathcal{S}_k)$', fontsize=10)
    ax3.set_title('(c) Fiedler sweep — conductance\nprofile and optimal cut',
                  fontsize=10, pad=8)
    ax3.legend(fontsize=8.5, framealpha=0.9, loc='upper right')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.text(0.03, 0.10,
             r'$\frac{\lambda_2}{2} \leq \phi^* \leq \sqrt{2\lambda_2}$',
             transform=ax3.transAxes, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))

    fig.suptitle(
        f'Spectral analysis of {code_name}: Fiedler vector identifies '
        'structural bottlenecks associated with trapping sets',
        fontsize=11, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f'fiedler_{code_name}')
    for ext in ('pdf', 'png'):
        plt.savefig(f'{stem}.{ext}', dpi=300,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {stem}.pdf  +  .png")


# -----------------------------------------------------------------
# Tests  (python scripts/plot_fiedler.py --test)
# -----------------------------------------------------------------

def _run_tests():
    PASS = lambda s: print(f"  PASS  {s}")

    d = np.load(os.path.join(CODES_DIR, 'BCH_31_11.npz'))
    H = d['H'].astype(int)
    n = int(d['n'])

    # ── Test 1: Fiedler value is positive ─────────────────────
    print("Test 1: Fiedler value lambda2 > 0")
    fvec, lambda2, A, deg = compute_fiedler(H)
    assert lambda2 > 0,          f"lambda2={lambda2} <= 0"
    assert len(fvec) == n,       f"Fiedler vec length {len(fvec)} != {n}"
    PASS(f"lambda2={lambda2:.4f}, fvec length={len(fvec)}")

    # ── Test 2: Sweep-cut gives valid partition ────────────────
    print("Test 2: Sweep-cut partition is non-trivial")
    S_sw, k_star, conds = sweep_cut(A, fvec, deg)
    assert 0 < len(S_sw) < n,   f"|S_sw|={len(S_sw)} not in (0,{n})"
    assert conds[k_star] > 0,   "Minimum conductance is 0"
    assert conds[k_star] < 1,   "Minimum conductance >= 1 (unexpected)"
    PASS(f"|S_sw|={len(S_sw)}, phi*={conds[k_star]:.4f}")

    # ── Test 3: Cheeger bound holds ────────────────────────────
    print("Test 3: Cheeger bound  lambda2/2 <= phi* <= sqrt(2*lambda2)")
    phi_star   = conds[k_star]
    lower      = lambda2 / 2
    upper      = np.sqrt(2 * lambda2)
    assert phi_star >= lower - 1e-9, \
        f"phi*={phi_star:.4f} < lambda2/2={lower:.4f}"
    assert phi_star <= upper + 1e-9, \
        f"phi*={phi_star:.4f} > sqrt(2*lambda2)={upper:.4f}"
    PASS(f"{lower:.4f} <= {phi_star:.4f} <= {upper:.4f}")

    print("\nAll 3 tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=str, default='BCH_31_11',
                        help='Code name in data/codes/')
    parser.add_argument('--out',  type=str, default='paper_results',
                        help='Output directory for saved figures')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.test:
        _run_tests()
    else:
        path = os.path.join(CODES_DIR, f'{args.code}.npz')
        if not os.path.exists(path):
            print(f"Code file not found: {path}")
            sys.exit(1)
        d = np.load(path)
        H = d['H'].astype(int)
        plot_fiedler(H, args.code, args.out)
