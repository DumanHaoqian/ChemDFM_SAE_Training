#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sparsemax Attention SAE for ChemDFM activation rows.

ChemDFM-sized Sparsemax SAE variant. Selection is sparsemax attention over
learned keys; reconstruction uses a decoder dictionary with learned feature and
output scales. The module intentionally exposes the small SAELens-compatible
surface used by ``train_sae.py`` and ``eval_sae.py``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SparsemaxAttentionSAEConfig:
    d_in: int
    d_sae: int
    dtype: str = "float32"
    device: str = "cuda"
    decoder_init_norm: float = 1.0
    key_dim: int | None = None
    preselect_k: int | None = 2048
    activation_mode: str = "probs"  # "probs" or "masked_scores"
    use_input_norm: bool = True
    use_idf_mask: bool = False
    idf_threshold: float = 0.1
    mse_loss_scale: float = 1.0
    score_scale: float = 2.0
    l0_target: float | None = 32.0
    l0_coefficient: float = 10.0
    cosine_loss_coefficient: float = 0.0
    norm_loss_coefficient: float = 0.0
    value_scale_init: float = 1.0
    global_output_scale_init: float = 1.0
    key_init_std: float | None = None
    b_dec_init: str = "zero"
    b_dec_init_rows: int = 0
    b_dec_init_norm: float = 0.0
    eps: float = 1e-8
    architecture: str = "sparsemax_attention"


class SparsemaxAttentionSAE(nn.Module):
    cfg: SparsemaxAttentionSAEConfig

    def __init__(self, cfg: SparsemaxAttentionSAEConfig):
        super().__init__()
        if cfg.activation_mode not in {"probs", "masked_scores"}:
            raise ValueError("activation_mode must be 'probs' or 'masked_scores'")
        cfg.key_dim = int(cfg.key_dim or min(1024, cfg.d_in))
        if cfg.preselect_k is not None:
            cfg.preselect_k = int(cfg.preselect_k)
            if cfg.preselect_k <= 0:
                cfg.preselect_k = None
        self.cfg = cfg
        dtype = getattr(torch, cfg.dtype)

        self.W_dec = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in, device=cfg.device, dtype=dtype))
        self.W_q = nn.Parameter(torch.empty(cfg.d_in, cfg.key_dim, device=cfg.device, dtype=dtype))
        self.W_key = nn.Parameter(torch.empty(cfg.d_sae, cfg.key_dim, device=cfg.device, dtype=dtype))
        self.value_scale = nn.Parameter(torch.full((cfg.d_sae,), float(cfg.value_scale_init), device=cfg.device, dtype=dtype))
        self.global_output_scale = nn.Parameter(torch.tensor(float(cfg.global_output_scale_init), device=cfg.device, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae, device=cfg.device, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in, device=cfg.device, dtype=dtype))
        self.register_buffer("idf_score", torch.zeros(cfg.d_sae, device=cfg.device, dtype=torch.float32))
        self.register_buffer("num_encode_calls", torch.zeros((), device=cfg.device, dtype=torch.long))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.W_dec)
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1) * float(self.cfg.decoder_init_norm)
        q_std = 1.0 / (self.cfg.d_in ** 0.5)
        k_std = float(self.cfg.key_init_std or (1.0 / (self.cfg.key_dim ** 0.5)))
        nn.init.normal_(self.W_q, std=q_std)
        nn.init.normal_(self.W_key, std=k_std)
        self.value_scale.data.fill_(float(self.cfg.value_scale_init))
        self.global_output_scale.data.fill_(float(self.cfg.global_output_scale_init))
        self.b_enc.data.zero_()
        self.b_dec.data.zero_()

    @property
    def W_enc(self):
        # Compatibility for ASI-style initializers; Sparsemax selection uses W_q/W_key.
        return self.W_dec.t()

    def get_coefficients(self):
        return {}

    def _input_norm(self, x: torch.Tensor):
        if not self.cfg.use_input_norm:
            return x, None
        norm = x.norm(dim=-1, keepdim=True).clamp_min(self.cfg.eps)
        return x / norm, norm

    @staticmethod
    def sparsemax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Martins & Astudillo sparsemax over ``dim``."""
        z = scores - scores.max(dim=dim, keepdim=True).values
        z_sorted = torch.sort(z, descending=True, dim=dim).values
        z_cumsum = torch.cumsum(z_sorted, dim=dim)
        r = torch.arange(1, z_sorted.size(dim) + 1, device=z.device, dtype=z.dtype)
        view = [1] * z.dim()
        view[dim] = -1
        r = r.view(*view)
        support = 1 + r * z_sorted > z_cumsum
        k = support.sum(dim=dim, keepdim=True).clamp_min(1)
        tau = ((z_sorted * support).sum(dim=dim, keepdim=True) - 1) / k
        return torch.clamp(z - tau, min=0)

    def _scores(self, x: torch.Tensor):
        x_normed, input_norm = self._input_norm(x)
        query = F.normalize(x_normed @ self.W_q, dim=-1, eps=self.cfg.eps)
        keys = F.normalize(self.W_key, dim=-1, eps=self.cfg.eps)
        scores = (query @ keys.transpose(0, 1)) * float(self.cfg.score_scale)
        if self.cfg.use_idf_mask:
            dense_mask = (self.idf_score > self.cfg.idf_threshold).to(scores.dtype)
            scores = scores.masked_fill(dense_mask.unsqueeze(0).bool(), torch.finfo(scores.dtype).min)
        return scores, input_norm

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.W_dec.device, dtype=self.W_dec.dtype)
        scores, input_norm = self._scores(x)
        if self.cfg.preselect_k is not None and self.cfg.preselect_k < self.cfg.d_sae:
            top_scores, top_idx = torch.topk(scores, k=self.cfg.preselect_k, dim=-1)
            top_probs = self.sparsemax(top_scores, dim=-1)
            if self.cfg.activation_mode == "masked_scores":
                top_acts = top_scores * (top_probs > 0).to(top_scores.dtype)
            else:
                top_acts = top_probs
            acts = torch.zeros_like(scores)
            acts.scatter_(dim=-1, index=top_idx, src=top_acts)
        else:
            probs = self.sparsemax(scores, dim=-1)
            if self.cfg.activation_mode == "masked_scores":
                acts = scores * (probs > 0).to(scores.dtype)
            else:
                acts = probs

        if input_norm is not None:
            acts = acts * input_norm.to(device=acts.device, dtype=acts.dtype)

        if self.training:
            with torch.no_grad():
                fired = (acts.detach() > 0).float().mean(dim=0).to(self.idf_score.dtype)
                n = int(self.num_encode_calls.item())
                self.idf_score.mul_(n / (n + 1)).add_(fired / (n + 1))
                self.num_encode_calls.add_(1)
        return acts

    def decode(self, feature_acts: torch.Tensor, input_norm: torch.Tensor | None = None) -> torch.Tensor:
        feature_acts = feature_acts.to(device=self.W_dec.device, dtype=self.W_dec.dtype)
        scaled_acts = feature_acts * self.value_scale.unsqueeze(0)
        recon = scaled_acts @ self.W_dec
        recon = recon * self.global_output_scale + self.b_dec
        if input_norm is not None:
            recon = recon * input_norm.to(device=recon.device, dtype=recon.dtype)
        return recon

    def forward(self, x: torch.Tensor):
        x = x.to(device=self.W_dec.device, dtype=self.W_dec.dtype)
        acts = self.encode(x)
        recon = self.decode(acts)
        return recon, acts

    def _participation_l0(self, feature_acts: torch.Tensor) -> torch.Tensor:
        weights = feature_acts.abs().float()
        return weights.sum(dim=-1).pow(2) / weights.pow(2).sum(dim=-1).clamp_min(float(self.cfg.eps))

    def training_forward_pass(self, step_input):
        sae_in = step_input.sae_in.to(device=self.W_dec.device, dtype=self.W_dec.dtype)
        sae_out, feature_acts = self.forward(sae_in)
        mse_loss = (sae_out.float() - sae_in.float()).pow(2).mean() * float(self.cfg.mse_loss_scale)
        l0_proxy = self._participation_l0(feature_acts).mean()
        if self.cfg.l0_target is not None and float(self.cfg.l0_coefficient) > 0:
            target = torch.tensor(float(self.cfg.l0_target), device=sae_in.device, dtype=torch.float32)
            l0_loss = float(self.cfg.l0_coefficient) * ((l0_proxy - target) / target.clamp_min(1.0)).pow(2)
        else:
            l0_loss = torch.zeros((), device=sae_in.device)
        sae_in_f = sae_in.float()
        sae_out_f = sae_out.float()
        cosine_raw = 1.0 - F.cosine_similarity(
            sae_out_f, sae_in_f, dim=-1, eps=float(self.cfg.eps)
        ).mean()
        cosine_loss = float(self.cfg.cosine_loss_coefficient) * cosine_raw
        in_norm = sae_in_f.norm(dim=-1).clamp_min(float(self.cfg.eps))
        out_norm = sae_out_f.norm(dim=-1)
        norm_rel_error = ((out_norm - in_norm) / in_norm).pow(2).mean()
        norm_loss = float(self.cfg.norm_loss_coefficient) * norm_rel_error
        loss = mse_loss + l0_loss + cosine_loss + norm_loss
        losses = {
            "mse_loss": mse_loss,
            "l0_target_loss": l0_loss,
            "l0_proxy": l0_proxy.detach(),
            "cosine_loss": cosine_loss,
            "cosine_raw": cosine_raw.detach(),
            "norm_loss": norm_loss,
            "norm_rel_error": norm_rel_error.detach(),
            "l1_loss": torch.zeros((), device=sae_in.device),
            "loss": loss,
        }
        return SimpleNamespace(
            loss=loss,
            losses=losses,
            feature_acts=feature_acts,
            sae_in=sae_in,
            sae_out=sae_out,
        )

    @torch.no_grad()
    def set_decoder_bias_from_data(self, rows: torch.Tensor):
        rows = rows.to(device=self.b_dec.device, dtype=self.b_dec.dtype)
        mu = rows.mean(dim=0)
        self.b_dec.copy_(mu)
        self.cfg.b_dec_init = "data_mean"
        self.cfg.b_dec_init_rows = int(rows.shape[0])
        self.cfg.b_dec_init_norm = float(mu.float().norm().item())
        return {
            "b_dec_init": self.cfg.b_dec_init,
            "b_dec_init_rows": self.cfg.b_dec_init_rows,
            "b_dec_init_norm": self.cfg.b_dec_init_norm,
        }

    def save_inference_model(self, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        weights_path = out / "sparsemax_attention_sae.pt"
        safe_weights_path = out / "sparsemax_attention_sae.safetensors"
        cfg_path = out / "cfg.json"
        torch.save({"state_dict": self.state_dict(), "cfg": asdict(self.cfg)}, weights_path)
        try:
            from safetensors.torch import save_file

            save_file(self.state_dict(), safe_weights_path)
        except ImportError:
            pass
        cfg_path.write_text(json.dumps(asdict(self.cfg), indent=2), encoding="utf-8")
        return str(safe_weights_path if safe_weights_path.exists() else weights_path), str(cfg_path)

    @classmethod
    def load_from_disk(cls, run_dir: str | Path, device: str = "cuda"):
        run_dir = Path(run_dir)
        cfg_dict = json.loads((run_dir / "cfg.json").read_text(encoding="utf-8"))
        cfg_dict["device"] = device
        cfg = SparsemaxAttentionSAEConfig(**cfg_dict)
        model = cls(cfg)
        safe_weights_path = run_dir / "sparsemax_attention_sae.safetensors"
        if safe_weights_path.exists():
            from safetensors.torch import load_file

            state_dict = load_file(safe_weights_path, device=device)
        else:
            ckpt = torch.load(run_dir / "sparsemax_attention_sae.pt", map_location=device)
            state_dict = ckpt["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        return model

    @torch.no_grad()
    def set_concepts_from_rows(self, rows: torch.Tensor, init_norm: float | None = None):
        rows = rows.to(device=self.W_dec.device, dtype=self.W_dec.dtype)
        if rows.shape != self.W_dec.shape:
            raise ValueError(f"expected rows {tuple(self.W_dec.shape)}, got {tuple(rows.shape)}")
        scale = float(init_norm if init_norm is not None else self.cfg.decoder_init_norm)
        self.W_dec.copy_(F.normalize(rows, dim=1) * scale)
