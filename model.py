"""
SG-ECCT model: multi-mask transformer decoder for error correction codes.

Architecture:
    - HighDimBitEmbedding     : embeds received LLRs with reliability features
    - TannerGraphEncoder      : adds graph-structural positional encoding
    - ElementWiseReliabilityPath : parallel per-bit refinement path
    - MaskBranch              : one transformer stack per spectral mask
    - ConfidenceWeightedFusion : learned weighted combination of branch outputs
    - SGECCT                  : top-level model (b=1 or b=2 mask configurations)

Input  : (batch, n, 1)  received LLRs
Output : (batch, n, 1)  decoded soft bits in [-1, 1]
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List


# -----------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------

class HighDimBitEmbedding(nn.Module):
    """Embed received LLRs into d_model-dimensional features.

    Combines a direct LLR projection with reliability features
    (magnitude, sign, soft saturation) and a confidence projection.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.llr_embed        = nn.Linear(1, d_model // 2)
        self.reliability_net  = nn.Sequential(
            nn.Linear(3, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 4),
        )
        self.confidence_embed = nn.Linear(1, d_model // 4)
        self.norm             = nn.LayerNorm(d_model)

    def forward(self, llr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            llr : (batch, n, 1)
        Returns:
            (batch, n, d_model)
        """
        llr_feats = self.llr_embed(llr)
        rel_input = torch.cat([
            torch.abs(llr),
            torch.sign(llr),
            torch.tanh(llr / 5.0),
        ], dim=-1)
        rel_feats  = self.reliability_net(rel_input)
        conf_feats = self.confidence_embed(torch.abs(llr))
        return self.norm(torch.cat([llr_feats, rel_feats, conf_feats], dim=-1))


class TannerGraphEncoder(nn.Module):
    """Graph-structural positional encoding derived from the PCM.

    Encodes per-variable-node features: degree, local structure,
    a degree-normalised spectral proxy, and learned position embedding.
    All four sub-embeddings have dimension d_model // 4 and are concatenated.
    """

    def __init__(self, H: np.ndarray, d_model: int, max_degree: int = 20):
        super().__init__()
        self.n           = H.shape[1]
        self.d_model     = d_model
        self.var_degrees = H.sum(axis=0)   # (n,)  variable-node degrees
        m                = H.shape[0]

        self.degree_embed   = nn.Embedding(max_degree + 1, d_model // 4)
        self.structure_embed = nn.Linear(4, d_model // 4)
        self.spectral_embed  = nn.Linear(1, d_model // 4)
        self.position_embed  = nn.Embedding(self.n, d_model // 4)

        # Pre-compute fixed structural features
        deg_t = torch.from_numpy(self.var_degrees).float()
        struct = torch.stack([
            deg_t / m,
            torch.ones(self.n),
            deg_t.sqrt(),
            torch.full((self.n,), 0.5),
        ], dim=1)                               # (n, 4)
        self.register_buffer('_struct', struct)
        spectral = (deg_t / deg_t.max()).unsqueeze(-1)  # (n, 1)
        self.register_buffer('_spectral', spectral)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Returns:
            (batch, n, d_model) positional encoding
        """
        degrees      = torch.from_numpy(self.var_degrees).long().clamp(0, 20).to(device)
        deg_feats    = self.degree_embed(degrees)
        struct_feats = self.structure_embed(self._struct.to(device))
        spec_feats   = self.spectral_embed(self._spectral.to(device))
        pos_feats    = self.position_embed(torch.arange(self.n, device=device))
        enc = torch.cat([deg_feats, struct_feats, spec_feats, pos_feats], dim=-1)
        return enc.unsqueeze(0).expand(batch_size, -1, -1)


# -----------------------------------------------------------------
# Transformer Components
# -----------------------------------------------------------------

class ElementWiseReliabilityPath(nn.Module):
    """Parallel per-bit refinement path (no cross-bit attention).

    Processes each bit position independently through an MLP
    with residual connection and layer norm.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, n, d_model)
        Returns:
            (batch, n, d_model)
        """
        return self.norm(x + self.net(x))


class MaskedMultiHeadAttention(nn.Module):
    """Multi-head self-attention with a structural boolean mask.

    Args:
        mask : (n, n) BoolTensor where True = attend, False = block.
               If None, full attention is used.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj  = nn.Linear(d_model, d_model)
        self.k_proj  = nn.Linear(d_model, d_model)
        self.v_proj  = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x    : (batch, n, d_model)
            mask : (n, n) BoolTensor  [True = allow]
        Returns:
            (batch, n, d_model)
        """
        B, n, _ = x.shape
        def reshape(t):
            return t.view(B, n, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(self.q_proj(x)), reshape(self.k_proj(x)), reshape(self.v_proj(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, h, n, n)

        if mask is not None:
            scores = scores.masked_fill(
                ~mask.unsqueeze(0).unsqueeze(0), float('-inf')
            )

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out  = torch.matmul(attn, v)
        out  = out.transpose(1, 2).contiguous().view(B, n, -1)
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer: masked attention + feed-forward."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MaskedMultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class MaskBranch(nn.Module):
    """Stack of transformer layers associated with one spectral mask."""

    def __init__(self, d_model: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


# -----------------------------------------------------------------
# Fusion
# -----------------------------------------------------------------

class ConfidenceWeightedFusion(nn.Module):
    """Learned confidence-weighted combination of mask branch outputs.

    Computes a per-position softmax weight over branches, takes the
    weighted sum, adds the reliability path output, and projects
    through the concatenated features with a residual connection.
    """

    def __init__(self, d_model: int, num_masks: int):
        super().__init__()
        self.num_masks = num_masks
        self.confidence_net = nn.Sequential(
            nn.Linear(d_model * num_masks, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_masks),
            nn.Softmax(dim=-1),
        )
        self.fusion_proj = nn.Linear(d_model * num_masks, d_model)
        self.norm        = nn.LayerNorm(d_model)

    def forward(
        self,
        branch_outputs: List[torch.Tensor],
        reliability_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            branch_outputs     : list of (batch, n, d_model), one per mask
            reliability_output : (batch, n, d_model)
        Returns:
            (batch, n, d_model)
        """
        cat   = torch.cat(branch_outputs, dim=-1)          # (B, n, d*num_masks)
        w     = self.confidence_net(cat)                    # (B, n, num_masks)
        fused = sum(w[..., i:i+1] * branch_outputs[i]
                    for i in range(self.num_masks))         # (B, n, d)
        fused = fused + reliability_output
        return self.norm(fused + self.fusion_proj(cat))


# -----------------------------------------------------------------
# Top-Level Model
# -----------------------------------------------------------------

class SGECCT(nn.Module):
    """Spectral-Guided Error Correction Code Transformer (SG-ECCT).

    Implements the full decoding pipeline:
        1. High-dimensional LLR embedding with reliability features
        2. Tanner graph positional encoding
        3. Parallel element-wise reliability path
        4. num_masks independent masked attention branches
        5. Confidence-weighted fusion
        6. Output projection to soft bits in [-1, 1]

    Args:
        H            : parity-check matrix (m, n) as numpy array
        mask_matrices: list of num_masks boolean tensors, each (n, n)
        num_masks    : number of spectral masks (b=1 or b=2 in the paper)
        d_model      : embedding dimension
        num_heads    : attention heads per layer
        num_layers   : transformer layers per mask branch
        dropout      : dropout probability
    """

    def __init__(
        self,
        H: np.ndarray,
        mask_matrices: List[torch.Tensor],
        num_masks: int = 2,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.m, self.n  = H.shape
        self.num_masks  = num_masks
        self.d_model    = d_model

        self.bit_embedding    = HighDimBitEmbedding(d_model)
        self.tanner_encoder   = TannerGraphEncoder(H, d_model)
        self.reliability_path = ElementWiseReliabilityPath(d_model, dropout)
        self.mask_branches    = nn.ModuleList([
            MaskBranch(d_model, num_heads, num_layers, dropout)
            for _ in range(num_masks)
        ])
        self.fusion           = ConfidenceWeightedFusion(d_model, num_masks)
        self.output_net       = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask_matrices', torch.stack(mask_matrices))

    def forward(self, llr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            llr : (batch, n, 1)  received log-likelihood ratios
        Returns:
            (batch, n, 1)  decoded soft bits in [-1, 1]
        """
        B = llr.shape[0]
        x = self.bit_embedding(llr)
        x = self.dropout(x + self.tanner_encoder(B, llr.device))

        reliability_out = self.reliability_path(x)
        branch_outputs  = [
            self.mask_branches[i](x, self.mask_matrices[i])
            for i in range(self.num_masks)
        ]
        return self.output_net(self.fusion(branch_outputs, reliability_out))


# -----------------------------------------------------------------
# Factory
# -----------------------------------------------------------------

def create_model(
    pcm_matrix: np.ndarray,
    mask_matrices: List[torch.Tensor],
    num_masks: int = 2,
    d_model: int = 128,
    num_heads: int = 8,
    num_layers: int = 3,
    dropout: float = 0.1,
    verbose: bool = True,
) -> SGECCT:
    """Instantiate and optionally summarise an SGECCT model.

    Args:
        pcm_matrix   : binary H matrix (m, n)
        mask_matrices: list of num_masks boolean tensors (n, n)
        num_masks    : number of spectral mask branches
        d_model      : embedding dimension
        num_heads    : attention heads per layer
        num_layers   : transformer layers per branch
        dropout      : dropout rate
        verbose      : print parameter count summary
    Returns:
        SGECCT instance
    """
    model = SGECCT(
        H=pcm_matrix,
        mask_matrices=mask_matrices,
        num_masks=num_masks,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
    if verbose:
        m, n   = pcm_matrix.shape
        k      = n - m
        total  = sum(p.numel() for p in model.parameters())
        print(f"SGECCT | code=({n},{k}) | masks={num_masks} | "
              f"d={d_model} h={num_heads} L={num_layers} | "
              f"params={total:,}")
    return model


# -----------------------------------------------------------------
# Tests  (python model.py)
# -----------------------------------------------------------------

if __name__ == "__main__":
    import sys
    _np = np

    PASS = lambda s: print(f"  PASS  {s}")

    d = _np.load("data/codes/BCH_31_11.npz")
    H = d["H"].astype(int)   # (20, 31)
    m, n = H.shape

    mask_full = torch.ones(n, n, dtype=torch.bool)
    mask_diag = torch.eye(n, dtype=torch.bool)

    # ── Test 1: Forward pass shape and output range ──────────
    print("Test 1: Forward pass shape and output range")
    model = create_model(H, [mask_full, mask_full],
                         num_masks=2, d_model=32, num_heads=4,
                         num_layers=2, dropout=0.0, verbose=False)
    model.eval()
    llr = torch.randn(8, n, 1)
    with torch.no_grad():
        out = model(llr)
    assert out.shape == (8, n, 1),      f"wrong output shape: {out.shape}"
    assert out.min() >= -1 - 1e-5,     "output below -1 (tanh violated)"
    assert out.max() <=  1 + 1e-5,     "output above +1 (tanh violated)"
    PASS(f"output shape {out.shape}, range [{out.min():.3f}, {out.max():.3f}]")

    # ── Test 2: Mask affects output ──────────────────────────
    print("Test 2: Mask affects output")
    model_full = create_model(H, [mask_full, mask_full],
                              num_masks=2, d_model=32, num_heads=4,
                              num_layers=2, dropout=0.0, verbose=False)
    model_diag = create_model(H, [mask_diag, mask_diag],
                              num_masks=2, d_model=32, num_heads=4,
                              num_layers=2, dropout=0.0, verbose=False)
    # Copy weights so only the mask differs
    model_diag.load_state_dict(model_full.state_dict(), strict=False)
    model_diag.mask_matrices = torch.stack([mask_diag, mask_diag])

    llr = torch.randn(4, n, 1)
    with torch.no_grad():
        out_full = model_full(llr)
        out_diag = model_diag(llr)
    assert not torch.allclose(out_full, out_diag), \
        "different masks produced identical outputs — mask has no effect"
    PASS("full-attention and diagonal-mask models produce different outputs")

    # ── Test 3: num_masks=1 vs num_masks=2 parameter counts ─
    print("Test 3: Parameter count scales with num_masks")
    m1 = create_model(H, [mask_full],
                      num_masks=1, d_model=32, num_heads=4,
                      num_layers=2, dropout=0.0, verbose=False)
    m2 = create_model(H, [mask_full, mask_full],
                      num_masks=2, d_model=32, num_heads=4,
                      num_layers=2, dropout=0.0, verbose=False)
    p1 = sum(p.numel() for p in m1.parameters())
    p2 = sum(p.numel() for p in m2.parameters())
    assert p2 > p1, f"b=2 should have more params than b=1 ({p2} vs {p1})"
    llr = torch.randn(4, n, 1)
    with torch.no_grad():
        assert m1(llr).shape == (4, n, 1)
        assert m2(llr).shape == (4, n, 1)
    PASS(f"b=1: {p1:,} params, b=2: {p2:,} params — both produce correct shapes")

    print("\nAll 3 tests passed.")