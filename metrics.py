#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small metric helpers shared by training and evaluation."""
from __future__ import annotations

import torch


class FVUAccumulator:
    """Accumulate reconstruction FVU using a global feature-wise mean."""

    def __init__(self, device: str | torch.device | None = None):
        self.device = device
        self.sse = torch.zeros((), device=device)
        self.sum_x: torch.Tensor | None = None
        self.sum_x_sq = torch.zeros((), device=device)
        self.n_rows = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor, recon: torch.Tensor) -> None:
        x_f = x.detach().float()
        recon_f = recon.detach().float()
        if self.sum_x is None:
            self.sum_x = torch.zeros(x_f.shape[-1], device=x_f.device, dtype=torch.float32)
            self.sse = self.sse.to(x_f.device)
            self.sum_x_sq = self.sum_x_sq.to(x_f.device)
        self.sse += (recon_f - x_f).pow(2).sum()
        self.sum_x += x_f.sum(dim=0)
        self.sum_x_sq += x_f.pow(2).sum()
        self.n_rows += int(x_f.shape[0])

    def fvu(self) -> float:
        if self.n_rows <= 0 or self.sum_x is None:
            return 0.0
        den = self.sum_x_sq - self.sum_x.pow(2).sum() / max(self.n_rows, 1)
        return float((self.sse / den.clamp_min(1e-8)).item())


@torch.no_grad()
def reconstruction_fvu(x: torch.Tensor, recon: torch.Tensor) -> float:
    acc = FVUAccumulator(device=x.device)
    acc.update(x, recon)
    return acc.fvu()
