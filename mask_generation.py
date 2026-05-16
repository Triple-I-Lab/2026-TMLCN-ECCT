"""
Spectral-guided mask generation for SG-ECCT.

Pipeline:
    1. Build V*V adjacency graph A = H^T H
    2. Apply edge reweighting
    3. Compute Fiedler vector of the normalised Laplacian
    4. Sweep-cut to identify the weak cluster S_sw
    5. Rank-aware greedy selection of up to m weak columns (GF(2))
    6. Repair if fewer than m columns were selected
    7. Row-ops only: enforce scattered identity at the selected weak columns
    8. Return boolean mask tensor over the n variable nodes

Public API:
    ConductanceBasedMaskGenerator(pcm_matrix, num_masks, reweight_method)
        .generate_masks()  ->  List[torch.BoolTensor]  (length n each)
"""

import numpy as np
import torch
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian


# -----------------------------------------------------------------
# 1. GF(2) Linear Algebra
# -----------------------------------------------------------------

class GF2Echelon:
    """Incremental row echelon form over GF(2) for rank tracking.

    Maintains a set of pivot vectors so that new columns can be tested
    for linear independence and inserted in O(m) time.
    """

    def __init__(self, m: int):
        self.m = m
        self.pivots = []         # list of (pivot_row, pivot_vector)
        self.pivot_rows = set()  # rows already used as pivots
        self.rank = 0

    def reduce(self, v: np.ndarray) -> np.ndarray:
        """Reduce v by current pivot vectors over GF(2)."""
        v_red = v.copy()
        for pivot_row, pivot_vec in self.pivots:
            if v_red[pivot_row] == 1:
                v_red = (v_red + pivot_vec) % 2
        return v_red

    def would_increase_rank(self, v: np.ndarray) -> bool:
        """Return True if v is linearly independent of current pivots."""
        return np.any(self.reduce(v) != 0)

    def insert_if_independent(self, v: np.ndarray) -> bool:
        """Insert v if independent; return True on success."""
        v_red = self.reduce(v)
        if not np.any(v_red != 0):
            return False
        pivot_row = np.argmax(v_red)
        if pivot_row in self.pivot_rows:
            for i in np.where(v_red == 1)[0]:
                if i not in self.pivot_rows:
                    pivot_row = i
                    break
        self.pivots.append((pivot_row, v_red.copy()))
        self.pivot_rows.add(pivot_row)
        self.rank += 1
        return True

    def get_rank(self) -> int:
        return self.rank


# -----------------------------------------------------------------
# 2. V*V Graph Construction and Edge Reweighting
# -----------------------------------------------------------------

def build_vv_adjacency(H: np.ndarray) -> np.ndarray:
    """Build weighted V*V adjacency matrix A = H^T H (diagonal zeroed).

    A[i, j] counts the number of check nodes shared by variable nodes i and j.
    """
    A = H.T @ H
    np.fill_diagonal(A, 0)
    return A.astype(float)


def apply_edge_reweighting(
    A: np.ndarray,
    method: str = 'degree_normalized',
    epsilon: float = 1e-9,
    alpha: float = 1.0,
    tau: int = 2,
) -> np.ndarray:
    """Reweight edges of the V*V graph before spectral analysis.

    Args:
        method: One of
            'degree_normalized' -- w'_ij = w_ij / sqrt(d_i * d_j)
            'jaccard'           -- w'_ij = w_ij / (d_i + d_j - w_ij)
            'penalty'           -- w'_ij = w_ij / (1 + alpha * max(0, w_ij - tau))
    """
    if method == 'degree_normalized':
        deg = A.sum(1)
        A_new = A / (np.sqrt(np.outer(deg, deg)) + epsilon)
    elif method == 'jaccard':
        deg = A.sum(1)
        A_new = A / (deg[:, None] + deg[None, :] - A + epsilon)
    elif method == 'penalty':
        A_new = A / (1 + alpha * np.maximum(0, A - tau))
    else:
        raise ValueError(
            f"Unknown reweighting method '{method}'. "
            "Choose from 'degree_normalized', 'jaccard', 'penalty'."
        )
    A_new = (A_new + A_new.T) / 2
    np.fill_diagonal(A_new, 0)
    return A_new


# -----------------------------------------------------------------
# 3. Spectral Analysis: Fiedler Vector and Sweep-Cut
# -----------------------------------------------------------------

def compute_fiedler_vector(A: np.ndarray, verbose: bool = False):
    """Compute the Fiedler vector of the normalised Laplacian of A.

    Falls back to the degree vector as a proxy when the graph has no
    positive eigenvalues (empty or extremely sparse graph).

    Returns:
        fiedler_vec : np.ndarray of shape (n,)
        fiedler_val : float, corresponding eigenvalue lambda_2
    """
    L_norm = laplacian(A, normed=True)
    eigenvals, eigenvecs = eigh(L_norm)
    idx = np.argsort(eigenvals)
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    positive = eigenvals > 1e-10
    if not np.any(positive):
        if verbose:
            print("  [Fiedler] No positive eigenvalues -- using degree proxy.")
        return A.sum(1).astype(float), 0.0

    fiedler_val = eigenvals[positive][0]
    fiedler_vec = eigenvecs[:, positive][:, 0]
    if verbose:
        print(f"  Fiedler eigenvalue lambda_2 = {fiedler_val:.6f}")
    return fiedler_vec, fiedler_val


def _conductance(A: np.ndarray, S_idx: list) -> float:
    """Normalised conductance of cut (S, V\\S) in graph A."""
    n = A.shape[0]
    mask = np.zeros(n, dtype=bool)
    mask[S_idx] = True
    vol_S    = A[mask].sum()
    vol_comp = A[~mask].sum()
    denom = min(vol_S, vol_comp)
    if denom < 1e-10:
        return np.inf
    return A[mask][:, ~mask].sum() / denom


def fiedler_sweep_cut(A: np.ndarray, fiedler_vec: np.ndarray, verbose: bool = False):
    """Identify the weak cluster by sweeping thresholds over the Fiedler vector.

    Sorts nodes by fiedler_vec and evaluates conductance for each prefix
    subset S_k = {v : fiedler_vec[v] <= threshold_k}. Returns the subset
    with minimum conductance (the weakest structural cluster S_sw).

    Returns:
        weak   : list of node indices in S_sw
        strong : remaining node indices
        phi    : conductance of the cut
    """
    order = np.argsort(fiedler_vec)
    best_phi, best_k = np.inf, 1
    for k in range(1, len(fiedler_vec)):
        phi = _conductance(A, order[:k].tolist())
        if phi < best_phi:
            best_phi, best_k = phi, k

    weak   = order[:best_k].tolist()
    strong = order[best_k:].tolist()

    phi_strong = _conductance(A, strong)
    if phi_strong < best_phi:
        weak, strong = strong, weak
        best_phi = phi_strong

    if verbose:
        print(f"  Sweep-cut: |weak|={len(weak)}, |strong|={len(strong)}, phi={best_phi:.6f}")
    return weak, strong, best_phi


# -----------------------------------------------------------------
# 4. Rank-Aware Column Selection and Repair
# -----------------------------------------------------------------

def rank_aware_greedy_select(
    H: np.ndarray,
    candidate_nodes: list,
    target_size: int,
    echelon: GF2Echelon = None,
    verbose: bool = False,
):
    """Greedily select up to target_size columns from candidate_nodes
    that are linearly independent over GF(2).

    Returns:
        selected : list of accepted column indices
        echelon  : updated GF2Echelon tracker
        info     : dict with selection statistics
    """
    m = H.shape[0]
    if echelon is None:
        echelon = GF2Echelon(m)

    selected, rejected = [], 0
    for node in candidate_nodes:
        if len(selected) >= target_size:
            break
        if echelon.would_increase_rank(H[:, node]):
            echelon.insert_if_independent(H[:, node])
            selected.append(node)
        else:
            rejected += 1

    if verbose:
        print(f"  Rank-aware: {len(selected)}/{target_size} selected "
              f"({rejected} rejected), rank={echelon.get_rank()}")
    return selected, echelon, {
        'selected_count': len(selected),
        'rejected_count': rejected,
        'final_rank':     echelon.get_rank(),
        'target_size':    target_size,
    }


def pad_to_m_with_repair(
    H: np.ndarray,
    current_selected: list,
    target_size: int,
    degrees_local: np.ndarray = None,
    available_indices: np.ndarray = None,
    excluded_nodes: set = None,
    verbose: bool = False,
):
    """Supplement current_selected with additional independent columns
    if fewer than target_size were found in the weak cluster.

    Prioritises columns with low degree (peripheral nodes).

    Returns:
        final   : extended list of selected column indices
        echelon : updated GF2Echelon tracker
        info    : dict with repair statistics
    """
    m, n = H.shape
    if excluded_nodes is None:
        excluded_nodes = set()

    ech = GF2Echelon(m)
    for node in current_selected:
        ech.insert_if_independent(H[:, node])

    need = target_size - len(current_selected)
    if need <= 0:
        return current_selected, ech, {
            'padded_count': 0, 'needed': 0,
            'final_count': len(current_selected), 'final_rank': ech.get_rank(),
        }

    if verbose:
        print(f"  Repair: need {need} more columns")

    available = list(set(range(n)) - set(current_selected) - excluded_nodes)
    if degrees_local is not None and available_indices is not None:
        local_pos = {g: i for i, g in enumerate(available_indices)}
        available.sort(key=lambda g: degrees_local[local_pos[g]]
                       if g in local_pos else np.inf)
    else:
        available.sort()

    added, ech, _ = rank_aware_greedy_select(H, available, need, echelon=ech, verbose=verbose)
    final = current_selected + added
    if verbose:
        print(f"  Repair: added {len(added)}, total {len(final)}/{target_size}, "
              f"rank={ech.get_rank()}")
    return final, ech, {
        'padded_count': len(added), 'needed': need,
        'final_count':  len(final), 'final_rank': ech.get_rank(),
    }


# -----------------------------------------------------------------
# 5. Row-Only Identity Enforcement at Weak Columns
# -----------------------------------------------------------------

def enforce_identity_at_columns(
    H: np.ndarray,
    target_cols: list,
    verbose: bool = False,
):
    """Use GF(2) row operations only to create a standard basis vector e_r
    at each target column in its original position (no column swaps).

    Returns:
        pivot_cols : subset of target_cols that successfully formed e_r
        H_rowsys   : H after row operations
        success    : True if all target columns were processed
    """
    H_work = H.copy().astype(int)
    m = H_work.shape[0]
    used_rows, pivot_cols = set(), []

    for col in target_cols:
        pivot_row = next(
            (r for r in range(m) if r not in used_rows and H_work[r, col] == 1),
            None,
        )
        if pivot_row is None:
            if verbose:
                print(f"  No available pivot row for column {col} -- skipped")
            continue
        for r in range(m):
            if r != pivot_row and H_work[r, col] == 1:
                H_work[r] = (H_work[r] + H_work[pivot_row]) % 2
        used_rows.add(pivot_row)
        pivot_cols.append(col)

    success = len(pivot_cols) == len(target_cols)
    if verbose:
        print(f"  Identity enforced: {len(pivot_cols)}/{len(target_cols)} columns")
    return pivot_cols, H_work, success


# -----------------------------------------------------------------
# 6. Main Pipeline
# -----------------------------------------------------------------

class ConductanceBasedMaskGenerator:
    """Generates spectral-guided attention masks for SG-ECCT.

    For each mask the pipeline:
        (a) restricts the V*V graph to columns not yet used as pivots,
        (b) applies edge reweighting and Fiedler sweep to find S_sw,
        (c) selects up to m linearly independent columns from S_sw,
        (d) repairs if fewer than m are found,
        (e) enforces row-only scattered identity at the selected columns,
        (f) returns a boolean mask tensor of length n.

    Masks are constructed sequentially; pivot columns from mask t are
    excluded when building mask t+1, ensuring non-overlapping coverage.

    Args:
        pcm_matrix      : binary H matrix, shape (m, n)
        num_masks       : number of masks to generate (default 2)
        reweight_method : edge reweighting passed to apply_edge_reweighting
                          (default 'degree_normalized')
        verbose         : print per-step diagnostics (default False)
    """

    def __init__(
        self,
        pcm_matrix: np.ndarray,
        num_masks: int = 2,
        reweight_method: str = 'degree_normalized',
        verbose: bool = False,
    ):
        if not np.all(np.isin(pcm_matrix, [0, 1])):
            raise ValueError("pcm_matrix must be binary (entries 0 or 1).")
        self.pcm_matrix      = pcm_matrix
        self.m, self.n       = pcm_matrix.shape
        self.num_masks       = num_masks
        self.reweight_method = reweight_method
        self.verbose         = verbose
        self.H_rowsys_all    = []
        self.analysis_data   = None

    def _generate_one_mask(self, excluded_nodes: set, mask_idx: int):
        v = self.verbose

        # (a) Available columns
        avail_mask = np.ones(self.n, dtype=bool)
        if excluded_nodes:
            avail_mask[list(excluded_nodes)] = False
        avail_idx = np.where(avail_mask)[0]
        H_sub = self.pcm_matrix[:, avail_idx]
        A_sub = build_vv_adjacency(H_sub)
        if v:
            print(f"\n-- Mask {mask_idx + 1} | available={len(avail_idx)}")

        # (b) Reweight + spectral
        A_rw = apply_edge_reweighting(A_sub, method=self.reweight_method)
        fiedler_vec, fiedler_val = compute_fiedler_vector(A_rw, verbose=v)
        weak_local, _, phi = fiedler_sweep_cut(A_rw, fiedler_vec, verbose=v)
        weak_global = [avail_idx[i] for i in weak_local]

        # (c) Rank-aware selection
        selected, echelon, _ = rank_aware_greedy_select(
            self.pcm_matrix, weak_global, self.m, verbose=v,
        )

        # (d) Repair if needed
        if len(selected) < self.m:
            selected, echelon, _ = pad_to_m_with_repair(
                self.pcm_matrix, selected, self.m,
                degrees_local=A_rw.sum(1),
                available_indices=avail_idx,
                excluded_nodes=excluded_nodes,
                verbose=v,
            )

        # (e) Enforce row-only identity
        pivot_cols, H_rowsys, success = enforce_identity_at_columns(
            self.pcm_matrix, selected, verbose=v,
        )
        self.H_rowsys_all.append(H_rowsys)

        # (f) Boolean mask
        mask = torch.zeros(self.n, dtype=torch.bool)
        mask[pivot_cols] = True

        info = {
            'conductance':       float(phi),
            'weak_set_size':     int(len(weak_global)),
            'selected_count':    int(len(selected)),
            'pivot_count':       int(len(pivot_cols)),
            'rank':              int(len(pivot_cols)),
            'identity_success':  bool(success),
            'fiedler_eigenvalue': float(fiedler_val),
        }
        if v:
            print(f"  pivot={info['pivot_count']}/{self.m}, "
                  f"success={success}, phi={phi:.6f}")
        return mask, pivot_cols, info

    def generate_masks(self):
        """Generate num_masks spectral attention masks.

        Returns:
            masks : list of torch.BoolTensor, each of length n
        """
        masks, pivots_all, infos = [], [], []
        excluded: set = set()

        for i in range(self.num_masks):
            mask, pivots, info = self._generate_one_mask(excluded, i)
            masks.append(mask)
            pivots_all.append(pivots)
            infos.append(info)
            excluded.update(pivots)

        self.analysis_data = {
            'masks':          masks,
            'pivot_cols_all': pivots_all,
            'info_all':       infos,
            'pcm_matrix':     self.pcm_matrix,
            'H_rowsys_all':   self.H_rowsys_all,
        }
        return masks


# -----------------------------------------------------------------
# Tests  (python mask_generation.py)
# -----------------------------------------------------------------

if __name__ == "__main__":
    import sys
    _np = np
    _torch = torch

    PASS = lambda s: print(f"  PASS  {s}")

    # ── Test 1: GF2Echelon rank tracking ─────────────────────
    print("Test 1: GF2Echelon")
    ech = GF2Echelon(m=4)
    v1 = _np.array([1, 0, 0, 0])
    v2 = _np.array([0, 1, 0, 0])
    v3 = _np.array([1, 1, 0, 0])   # dependent: v1 + v2 mod 2
    v4 = _np.array([0, 0, 1, 0])

    assert ech.insert_if_independent(v1) == True  and ech.get_rank() == 1
    assert ech.insert_if_independent(v2) == True  and ech.get_rank() == 2
    assert ech.insert_if_independent(v3) == False and ech.get_rank() == 2
    assert ech.insert_if_independent(v4) == True  and ech.get_rank() == 3
    assert ech.would_increase_rank(v3)   == False
    PASS("inserts independent vectors, rejects dependent ones, rank correct")

    # ── Test 2: Spectral pipeline on BCH(31,11) ──────────────
    print("Test 2: Spectral pipeline")
    d = _np.load("data/codes/BCH_31_11.npz")
    H = d["H"].astype(int)           # (20, 31)
    A = build_vv_adjacency(H)

    assert A.shape == (31, 31)
    assert _np.all(_np.diag(A) == 0)
    assert _np.allclose(A, A.T)

    A_rw = apply_edge_reweighting(A, method="degree_normalized")
    assert _np.allclose(A_rw, A_rw.T)

    fvec, fval = compute_fiedler_vector(A_rw)
    assert fval > 0 and len(fvec) == 31

    weak, strong, phi = fiedler_sweep_cut(A_rw, fvec)
    assert len(weak) + len(strong) == 31
    assert len(weak) > 0 and len(strong) > 0
    assert 0 < phi < _np.inf
    PASS(f"|weak|={len(weak)}, phi={phi:.4f}, lambda2={fval:.4f}")

    # ── Test 3: ConductanceBasedMaskGenerator end-to-end ─────
    print("Test 3: ConductanceBasedMaskGenerator")
    gen = ConductanceBasedMaskGenerator(pcm_matrix=H, num_masks=2, verbose=False)
    masks = gen.generate_masks()

    assert len(masks) == 2
    assert all(m.dtype == _torch.bool for m in masks)
    assert all(m.shape == (31,) for m in masks)
    assert masks[0].sum() > 0 and masks[1].sum() > 0

    # Pivot columns must be disjoint
    p0 = set(gen.analysis_data["pivot_cols_all"][0])
    p1 = set(gen.analysis_data["pivot_cols_all"][1])
    assert len(p0 & p1) == 0, f"pivot overlap: {p0 & p1}"

    # Scattered identity check on H_rowsys
    for t, piv in enumerate(gen.analysis_data["pivot_cols_all"]):
        Hr, rows_seen = gen.analysis_data["H_rowsys_all"][t], set()
        for col in piv:
            col_vec = Hr[:, col]
            assert col_vec.sum() == 1
            r = int(_np.argmax(col_vec))
            assert r not in rows_seen, f"mask {t+1} row {r} reused"
            rows_seen.add(r)

    PASS(f"masks={[m.sum().item() for m in masks]} pivots, disjoint, identity verified")

    print("\nAll 3 tests passed.")