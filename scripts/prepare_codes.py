"""
Generate and save all 7 codes used in SG-ECCT experiments.

Generates BCH and Polar codes and saves them to data/codes/ as .npz files.
LDPC codes require separate generation scripts:
    - data/codes/LDPC_121_60.npz : parse from alist  (gen_missing_codes.py)
    - data/codes/LDPC_144_72.npz : random regular     (gen_ldpc_144_192.py)
    - data/codes/LDPC_192_96.npz : random regular     (gen_ldpc_144_192.py)

Usage:
    python scripts/prepare_codes.py          # generate BCH + Polar
    python scripts/prepare_codes.py --check  # verify all 10 codes
"""

import argparse
import os
import sys

import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from code_generators import get_bch_code, get_polar_code

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'codes')

BCH_CODES   = ['BCH_31_11', 'BCH_31_16', 'BCH_63_36', 'BCH_63_51']
POLAR_CODES = ['POLAR_64_32', 'POLAR_128_64', 'POLAR_128_86']
LDPC_CODES  = ['LDPC_121_60', 'LDPC_144_72', 'LDPC_192_96']
ALL_CODES   = BCH_CODES + POLAR_CODES + LDPC_CODES


def save_code(name: str, G: np.ndarray, H: np.ndarray,
              n: int, k: int, code_type: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.npz")
    np.savez_compressed(path,
                        G=G.astype(np.uint8), H=H.astype(np.uint8),
                        n=n, k=k, rate=np.float32(k / n),
                        code_type=code_type)
    return path


def generate_bch_polar():
    """Generate and save the 4 BCH and 3 Polar codes."""
    print("Generating BCH codes...")
    for name in BCH_CODES:
        code = get_bch_code(name)
        G, H = code.get_matrices()
        path = save_code(name, G, H, code.n, code.k, 'BCH')
        print(f"  {name}: G={G.shape}  H={H.shape}  → {path}")

    print("\nGenerating Polar codes...")
    for name in POLAR_CODES:
        code = get_polar_code(name)
        G, H = code.get_matrices()
        path = save_code(name, G, H, code.n, code.k, 'POLAR')
        print(f"  {name}: G={G.shape}  H={H.shape}  → {path}")


def check_all_codes():
    """Verify all 10 codes exist and are algebraically valid."""
    print(f"\n{'Code':<22} {'G shape':>10} {'H shape':>10} {'rate':>6}  Status")
    print("-" * 65)
    missing, failed = [], []

    for name in ALL_CODES:
        path = os.path.join(OUT_DIR, f"{name}.npz")
        if not os.path.exists(path):
            print(f"  {name:<20}  {'MISSING':>30}")
            missing.append(name)
            continue
        d = np.load(path)
        G, H = d['G'].astype(int), d['H'].astype(int)
        n, k = int(d['n']), int(d['k'])
        ok = (G.shape == (k, n) and H.shape == (n - k, n)
              and np.all((G @ H.T) % 2 == 0)
              and np.all(H.sum(axis=1) > 0)
              and np.all(H.sum(axis=0) > 0))
        status = 'OK' if ok else 'FAIL'
        print(f"  {name:<20}  {str(G.shape):>10}  {str(H.shape):>10}  "
              f"{k/n:>5.3f}  {status}")
        if not ok:
            failed.append(name)

    print()
    if missing:
        print(f"Missing ({len(missing)}): {missing}")
        if any(c in missing for c in LDPC_CODES):
            print("  Run gen_missing_codes.py and gen_ldpc_144_192.py "
                  "to generate LDPC codes.")
    if failed:
        print(f"Failed  ({len(failed)}): {failed}")
    if not missing and not failed:
        print(f"All {len(ALL_CODES)} codes OK.")
    return not missing and not failed


# -----------------------------------------------------------------
# Tests  (python scripts/prepare_codes.py --test)
# -----------------------------------------------------------------

def _run_tests():
    PASS = lambda s: print(f"  PASS  {s}")

    # ── Test 1: All BCH and Polar codes load and are valid ────
    print("Test 1: BCH and Polar codes are valid")
    for name in BCH_CODES + POLAR_CODES:
        path = os.path.join(OUT_DIR, f"{name}.npz")
        assert os.path.exists(path), f"{name}.npz not found — run prepare_codes.py first"
        d = np.load(path)
        G, H = d['G'].astype(int), d['H'].astype(int)
        n, k = int(d['n']), int(d['k'])
        assert G.shape == (k, n),           f"{name}: G shape {G.shape}"
        assert H.shape == (n - k, n),       f"{name}: H shape {H.shape}"
        assert np.all((G @ H.T) % 2 == 0), f"{name}: G@H.T != 0"
    PASS(f"all {len(BCH_CODES + POLAR_CODES)} BCH+Polar codes valid")

    # ── Test 2: Encoding consistency ─────────────────────────
    print("Test 2: Random codewords satisfy H for BCH and Polar")
    for name in BCH_CODES + POLAR_CODES:
        d = np.load(os.path.join(OUT_DIR, f"{name}.npz"))
        G, H = d['G'].astype(int), d['H'].astype(int)
        k = int(d['k'])
        for _ in range(5):
            msg = np.random.randint(0, 2, k)
            cw  = (msg @ G) % 2
            assert np.all((H @ cw) % 2 == 0), f"{name}: syndrome != 0"
    PASS("5 random codewords verified for each BCH and Polar code")

    # ── Test 3: LDPC codes exist and are valid (if present) ──
    print("Test 3: LDPC codes validity (skipped if not yet generated)")
    found = 0
    for name in LDPC_CODES:
        path = os.path.join(OUT_DIR, f"{name}.npz")
        if not os.path.exists(path):
            print(f"    {name}: not found — skipping")
            continue
        d = np.load(path)
        G, H = d['G'].astype(int), d['H'].astype(int)
        n, k = int(d['n']), int(d['k'])
        assert G.shape == (k, n)
        assert H.shape == (n - k, n)
        assert np.all((G @ H.T) % 2 == 0), f"{name}: G@H.T != 0"
        found += 1
    PASS(f"{found}/{len(LDPC_CODES)} LDPC codes found and valid")

    print("\nAll 3 tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true',
                        help='Verify all 10 codes without regenerating')
    parser.add_argument('--test',  action='store_true',
                        help='Run built-in tests')
    args = parser.parse_args()

    if args.test:
        _run_tests()
    elif args.check:
        ok = check_all_codes()
        sys.exit(0 if ok else 1)
    else:
        generate_bch_polar()
        print()
        check_all_codes()