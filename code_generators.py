"""
Code generators for SG-ECCT experiments.

Supports BCH and Polar codes used in the paper.
LDPC codes are loaded directly from pre-generated .npz files
(see data/codes/ and gen_ldpc_144_192.py).

Dependencies:
    pip install galois       # BCH codes
    pip install numpy scipy  # Polar codes (no extra library needed)
"""

import numpy as np


# -----------------------------------------------------------------
# Shared utility
# -----------------------------------------------------------------

def _h_from_g(G: np.ndarray) -> np.ndarray:
    """Derive a valid parity-check matrix H from G over GF(2).

    Puts G in systematic form [I_k | P] via row reduction, then
    returns H = [P^T | I_{n-k}], which satisfies G @ H.T = 0 mod 2.
    """
    k, n = G.shape
    Gw = G.copy().astype(int)
    pivot_cols = []
    row = 0
    for col in range(n):
        pivot = next((r for r in range(row, k) if Gw[r, col] % 2 == 1), None)
        if pivot is None:
            continue
        Gw[[row, pivot]] = Gw[[pivot, row]]
        for r in range(k):
            if r != row and Gw[r, col] % 2 == 1:
                Gw[r] = (Gw[r] + Gw[row]) % 2
        pivot_cols.append(col)
        row += 1
        if row == k:
            break
    free_cols = [c for c in range(n) if c not in pivot_cols]
    P = Gw[:, free_cols]                        # (k, n-k)
    H = np.zeros((n - k, n), dtype=int)
    H[:, pivot_cols] = P.T
    H[:, free_cols]  = np.eye(n - k, dtype=int)
    return H


# -----------------------------------------------------------------
# BCH Codes  (requires: pip install galois)
# -----------------------------------------------------------------

try:
    import galois as _galois
    _GALOIS_AVAILABLE = True
except ImportError:
    _GALOIS_AVAILABLE = False


# Paper codes only
_BCH_CODES = {
    'BCH_31_11': (31, 11),
    'BCH_31_16': (31, 16),
    'BCH_63_36': (63, 36),
    'BCH_63_51': (63, 51),
}


class BCHCodeGenerator:
    """BCH code generator using the galois library.

    Args:
        n : codeword length (must be 2^m - 1 for some m)
        k : message length
    """

    def __init__(self, n: int, k: int):
        if not _GALOIS_AVAILABLE:
            raise ImportError("galois library required. Install: pip install galois")
        try:
            self._bch = _galois.BCH(n, k)
        except Exception as e:
            raise ValueError(f"Cannot create BCH({n},{k}): {e}")
        self.n    = n
        self.k    = k
        self.rate = k / n

    def get_matrices(self):
        """Return (G, H) as numpy int arrays, shapes (k, n) and (n-k, n)."""
        return np.array(self._bch.G, dtype=int), np.array(self._bch.H, dtype=int)

    def encode(self, message: np.ndarray) -> np.ndarray:
        """Encode a binary message vector of length k."""
        return np.array(self._bch.encode(_galois.GF2(message)), dtype=int)


def get_bch_code(code_name: str) -> BCHCodeGenerator:
    """Return a BCHCodeGenerator for one of the paper's four BCH codes.

    Args:
        code_name : one of 'BCH_31_11', 'BCH_31_16', 'BCH_63_36', 'BCH_63_51'
    """
    if code_name not in _BCH_CODES:
        raise ValueError(f"Unknown BCH code '{code_name}'. "
                         f"Available: {list(_BCH_CODES)}")
    n, k = _BCH_CODES[code_name]
    return BCHCodeGenerator(n, k)


# -----------------------------------------------------------------
# Polar Codes  (no extra library required)
# -----------------------------------------------------------------

# Partial 3GPP TS 38.212 reliability sequence (indices valid up to N=1024).
# Values >= n are filtered out when constructing codes of length n < 1024.
_5G_SEQUENCE = [
    0, 1, 2, 4, 8, 16, 32, 3, 5, 64, 9, 6, 17, 10, 18, 128, 12, 33,
    65, 20, 256, 34, 24, 36, 7, 129, 66, 512, 11, 40, 68, 130, 19,
    13, 48, 14, 72, 257, 21, 132, 35, 258, 26, 513, 80, 37, 25, 22,
    136, 260, 264, 38, 514, 96, 67, 41, 144, 28, 69, 42, 516, 49,
    74, 272, 160, 520, 288, 528, 192, 544, 70, 44, 131, 81, 50, 73,
    15, 320, 133, 52, 23, 134, 384, 76, 137, 82, 56, 27, 97, 39,
    259, 84, 138, 145, 261, 29, 43, 98, 515, 88, 140, 30, 146, 71,
    262, 265, 161, 576, 45, 100, 640, 51, 148, 46, 75, 266, 273,
    517, 104, 162, 53, 193, 152, 77, 164, 768, 268, 274, 518, 54,
    83, 57, 112, 135, 78, 289, 521, 194, 85, 276, 522, 58, 168, 139,
    99, 86, 60, 280, 89, 290, 529, 524, 196, 141, 101, 147, 176,
    142, 530, 321, 31, 200, 90, 545, 292, 322, 532, 263, 149, 102,
    105, 304, 296, 163, 92, 47, 267, 385, 546, 324, 208, 386, 150,
    153, 165, 106, 55, 328, 536, 577, 195, 113, 154, 79, 269, 108,
    578, 224, 166, 519, 548, 641, 275, 580, 291, 584, 169, 114,
    156, 197, 87, 198, 277, 592, 523, 642, 116, 170, 201, 281,
    278, 526, 91, 769, 644, 202, 531, 298, 172, 120, 103,
    282, 648, 525, 93, 209, 143, 284, 151, 210, 204, 293, 107,
    177, 323, 656, 294, 325, 178, 155, 297, 672, 109, 180,
    533, 115, 326, 157, 225, 167, 300, 212, 329, 548, 184,
    534, 704, 158, 537, 117, 330, 171, 226, 305, 547, 199,
    538, 540, 387, 279, 228, 173, 121, 332, 203, 306, 549, 295, 118,
    388, 336, 581, 174, 232, 550, 211, 389, 179, 122, 583, 240,
    205, 390, 324, 552, 181, 585, 213, 600, 527, 327,
    417, 597, 556, 206, 124, 214, 182, 643, 298, 593, 564, 228,
    331, 391, 645, 185, 418, 227, 580, 216, 531, 594, 535, 333,
    646, 649, 159, 596, 392, 229, 307, 186, 568, 539, 334, 220,
    598, 649, 188, 230, 338, 650, 233, 175, 601, 541, 552,
    394, 540, 337, 234, 652, 542, 602, 583, 327, 241, 554, 191,
    123, 604, 400, 339, 558, 660, 189, 553, 242, 125, 665,
    420, 543, 340, 244, 585, 190, 215, 562, 608, 673, 583, 342,
    557, 424, 394, 396, 586, 217, 676, 565, 231, 559, 192, 248,
    654, 335, 218, 674, 680, 560, 603, 235, 588, 126,
    344, 221, 605, 566, 688, 658, 236, 569,
    348, 243, 222, 570, 641, 572, 395, 245,
    225, 704, 609, 397, 238, 610,
    398, 356, 246, 401, 249, 612, 189, 402,
    421, 193, 250, 404, 616, 219,
    408, 544, 561, 252, 675, 563, 425, 422,
    677, 426, 587, 223, 237, 567,
    428, 681, 396, 239, 356, 589, 432,
    682, 571, 247, 399, 573, 689,
    440, 574, 403, 251, 605, 590,
    456, 611, 253, 405,
    613, 488, 406, 614, 254, 409,
    617, 552, 618, 410, 423,
    255, 620, 427, 429,
    624, 430, 433, 632,
]

# Paper codes only
_POLAR_CODES = {
    'POLAR_64_32':  (64,  32),
    'POLAR_128_64': (128, 64),
    'POLAR_128_86': (128, 86),
}


class PolarCodeGenerator:
    """Polar code generator using the 5G NR reliability sequence (3GPP TS 38.212).

    Constructs the full polarisation matrix F^{⊗log2(n)} via Kronecker products,
    selects the k most reliable positions as information bits, and derives a
    valid H from G using systematic form over GF(2).

    Args:
        n : codeword length (must be a power of 2, n <= 1024)
        k : message length
    """

    def __init__(self, n: int, k: int):
        assert n & (n - 1) == 0 and 1 < n <= 1024, \
            "n must be a power of 2 in the range (1, 1024]"
        self.n    = n
        self.k    = k
        self.rate = k / n
        self._build()

    def _build(self):
        # Polar transform kernel
        F = np.array([[1, 0], [1, 1]], dtype=int)
        G_full = F
        for _ in range(int(np.log2(self.n)) - 1):
            G_full = np.kron(G_full, F)             # (n, n)

        # Select k most reliable positions from the 5G sequence
        reliable = [i for i in _5G_SEQUENCE if i < self.n]
        self.info_bits   = sorted(reliable[-self.k:])
        self.frozen_bits = [i for i in range(self.n) if i not in self.info_bits]

        self.G = G_full[self.info_bits, :]          # (k, n)
        self.H = _h_from_g(self.G)                  # (n-k, n), guaranteed valid

    def get_matrices(self):
        """Return (G, H) as numpy int arrays, shapes (k, n) and (n-k, n)."""
        return self.G.copy(), self.H.copy()

    def encode(self, message: np.ndarray) -> np.ndarray:
        """Encode a binary message vector of length k."""
        return (message @ self.G) % 2


def get_polar_code(code_name: str) -> PolarCodeGenerator:
    """Return a PolarCodeGenerator for one of the paper's three Polar codes.

    Args:
        code_name : one of 'POLAR_64_32', 'POLAR_128_64', 'POLAR_128_86'
    """
    if code_name not in _POLAR_CODES:
        raise ValueError(f"Unknown Polar code '{code_name}'. "
                         f"Available: {list(_POLAR_CODES)}")
    n, k = _POLAR_CODES[code_name]
    return PolarCodeGenerator(n, k)


# -----------------------------------------------------------------
# Tests  (python code_generators.py)
# -----------------------------------------------------------------

if __name__ == "__main__":
    import sys

    PASS = lambda s: print(f"  PASS  {s}")

    # ── Test 1: BCH G @ H.T = 0 and encoding ─────────────────
    print("Test 1: BCH codes — G@H.T=0 and syndrome check")
    for name in _BCH_CODES:
        code = get_bch_code(name)
        G, H = code.get_matrices()
        n, k = code.n, code.k
        assert G.shape == (k, n),       f"{name}: G shape {G.shape}"
        assert H.shape == (n - k, n),   f"{name}: H shape {H.shape}"
        assert np.all((G @ H.T) % 2 == 0), f"{name}: G@H.T != 0"
        # Encode a random message and verify zero syndrome
        msg = np.random.randint(0, 2, k)
        cw  = code.encode(msg)
        assert np.all((H @ cw) % 2 == 0), f"{name}: syndrome != 0"
    PASS(f"all {len(_BCH_CODES)} BCH codes valid")

    # ── Test 2: Polar G @ H.T = 0 and shapes ─────────────────
    print("Test 2: Polar codes — G@H.T=0 and shape check")
    for name in _POLAR_CODES:
        code = get_polar_code(name)
        G, H = code.get_matrices()
        n, k = code.n, code.k
        assert G.shape == (k, n),           f"{name}: G shape {G.shape}"
        assert H.shape == (n - k, n),       f"{name}: H shape {H.shape}"
        assert np.all((G @ H.T) % 2 == 0), f"{name}: G@H.T != 0"
        assert np.all(H.sum(axis=1) > 0),  f"{name}: H has zero rows"
        assert np.all(H.sum(axis=0) > 0),  f"{name}: H has zero cols"
    PASS(f"all {len(_POLAR_CODES)} Polar codes valid")

    # ── Test 3: Encoding consistency — codeword satisfies H ──
    print("Test 3: Encoding consistency for BCH and Polar")
    for name in list(_BCH_CODES) + list(_POLAR_CODES):
        if name.startswith('BCH'):
            code = get_bch_code(name)
        else:
            code = get_polar_code(name)
        G, H = code.get_matrices()
        for _ in range(10):
            msg = np.random.randint(0, 2, code.k)
            cw  = code.encode(msg)
            assert cw.shape == (code.n,),          f"{name}: codeword shape wrong"
            assert np.all((H @ cw) % 2 == 0),     f"{name}: codeword not in code"
            assert np.all(np.isin(cw, [0, 1])),   f"{name}: codeword not binary"
    PASS(f"10 random codewords verified for all BCH and Polar codes")

    print("\nAll 3 tests passed.")