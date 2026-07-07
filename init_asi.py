#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""init_asi.py — Active Subspace Initialization for SAEs (OpenMOSS).

Random SAE init is isotropic in R^d_in, but LM activations concentrate in a
lower-dimensional 'active subspace'. Initializing feature directions INSIDE that
subspace (top principal directions of a data sample) starts every latent near
real data variance, sharply reducing dead latents (OpenMOSS: 87% -> ~1% at 1M
features on low-rank activations; helps Qwen/Llama/Gemma). ChemDFM is a Qwen2
finetune, so this is the on-target dead-feature fix.

IMPORTANT (residual stream): resid_post activations have a dominant "rogue"
dimension that alone holds ~99% of the variance, so a naive cumulative-99%
threshold collapses the rank to 1 and destroys reconstruction. We therefore use
a rank FLOOR (min_rank) and cap the rank at the numerically non-null rank, so the
active subspace stays rich. min_rank is the main knob; tune it on real data.

W_dec rows (feature directions) are drawn in span(V_r) where V_r = top-r
eigenvectors of the centered activation covariance; W_enc is tied to W_dec^T;
b_dec = data mean. Works for any SAELens SAE exposing W_enc/W_dec/b_enc/b_dec.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import torch


@torch.no_grad()
def compute_active_subspace(X: torch.Tensor, var_threshold: float = 0.999,
                            min_rank: int = 1024, max_rank: Optional[int] = None
                            ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return (V, mu, r): V=(d_in, r) top principal directions, mu=(d_in,) mean.

    r = clamp( max(min_rank, cumulative-variance rank), 1, numerical_rank ) and
    capped by max_rank if given. The numerical-rank cap avoids seeding features
    into pure-noise eigen-directions.
    """
    mu = X.mean(0)
    Xc = X - mu
    cov = (Xc.t() @ Xc) / X.shape[0]                 # (d_in, d_in)
    evals, evecs = torch.linalg.eigh(cov)            # ascending eigen-pairs
    evals = evals.flip(0).clamp_min(0)
    evecs = evecs.flip(1)
    total = evals.sum().clamp_min(1e-12)
    cum = torch.cumsum(evals, 0) / total
    r_thresh = int((cum < var_threshold).sum().item()) + 1
    # numerically non-null directions (eigenvalue well above float noise)
    num_rank = int((evals > 1e-7 * evals[0].clamp_min(1e-12)).sum().item())
    num_rank = max(1, num_rank)
    r = max(int(min_rank), r_thresh)
    r = min(r, num_rank, evecs.shape[1])
    if max_rank:
        r = min(r, int(max_rank))
    r = max(1, r)
    V = evecs[:, :r].contiguous()                    # (d_in, r)
    return V, mu, r


@torch.no_grad()
def asi_init_sae(sae, X: torch.Tensor, decoder_init_norm: float = 0.1,
                 var_threshold: float = 0.999, min_rank: int = 1024,
                 max_rank: Optional[int] = None, seed: int = 0) -> Dict[str, Any]:
    """Initialize a SAELens SAE's W_enc/W_dec/b_enc/b_dec in the active subspace."""
    X = X.to(torch.float32)
    device = X.device
    d_in = X.shape[1]
    d_sae = int(sae.W_dec.shape[0])
    V, mu, r = compute_active_subspace(X, var_threshold, min_rank, max_rank)
    gen = torch.Generator(device=device).manual_seed(seed)
    g = torch.randn(d_sae, r, generator=gen, device=device)          # (d_sae, r)
    W_dec = g @ V.t()                                                # (d_sae, d_in) rows in span(V)
    W_dec = W_dec / W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8) * decoder_init_norm
    sae.W_dec.data.copy_(W_dec.to(sae.W_dec.dtype))
    sae.W_enc.data.copy_(W_dec.t().to(sae.W_enc.dtype))              # tied init
    sae.b_enc.data.zero_()
    sae.b_dec.data.copy_(mu.to(sae.b_dec.dtype))
    return {"init": "asi", "active_rank": int(r), "d_in": int(d_in), "d_sae": d_sae,
            "var_threshold": var_threshold, "min_rank": int(min_rank),
            "active_rank_frac": round(r / d_in, 4)}
