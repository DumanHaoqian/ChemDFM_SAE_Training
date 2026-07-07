#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_sae.py — train a Sparse Autoencoder on ChemDFM residual-stream activations.

Uses SAELens' SAE implementations (BatchTopK is the main line, §2.2) inside a
custom training loop that reads the Stage-3 disk cache. This keeps full control
over the 14B HF model / chemistry tokenizer / layer sweep without forcing
ChemDFM through TransformerLens.

Architectures (``--arch``):
  * ``batchtopk``  (default) — batch-level TopK; saved as JumpReLU for inference.
  * ``topk``       — per-sample TopK.
  * ``jumprelu``   — JumpReLU (Gemma-Scope style baseline).
  * ``matryoshka`` — Matryoshka-BatchTopK with nested prefixes.

Key knobs match the design guidance: expansion 8x-32x, k (L0) ~= 32-64,
lr ~1e-4 with warmup, unit-normalised decoder, dead-feature revival via the
TopK auxiliary loss.

Usage::

    source /home/haoqian/Data/SAERAG/venvs/chemdfm/bin/activate
    cd /home/haoqian/Data/SAERAG/v3_Chem_SAE/Stage4_sae_training
    python train_sae.py --layer 26 --arch batchtopk --expansion 16 --k 32 \
        --total-steps 30000 --no-wandb
    python train_sae.py --smoke --no-wandb        # tiny end-to-end check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch

import paths
from data import ActivationStore
from init_asi import asi_init_sae


# ---------------------------------------------------------------------------
# SAE construction
# ---------------------------------------------------------------------------
def build_sae(arch: str, d_in: int, d_sae: int, k: int, device: str,
              decoder_init_norm: float, matryoshka_widths=None):
    from sae_lens import (
        BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig,
        TopKTrainingSAE, TopKTrainingSAEConfig,
        JumpReLUTrainingSAE, JumpReLUTrainingSAEConfig,
        MatryoshkaBatchTopKTrainingSAE, MatryoshkaBatchTopKTrainingSAEConfig,
    )
    common = dict(
        d_in=d_in, d_sae=d_sae, dtype="float32", device=device,
        normalize_activations="none",        # Stage-3 already rescales inputs
        decoder_init_norm=decoder_init_norm,  # Anthropic "heuristic" init
    )
    if arch == "batchtopk":
        cfg = BatchTopKTrainingSAEConfig(k=float(k), rescale_acts_by_decoder_norm=True, **common)
        return BatchTopKTrainingSAE(cfg), cfg
    if arch == "topk":
        cfg = TopKTrainingSAEConfig(k=int(k), rescale_acts_by_decoder_norm=True, **common)
        return TopKTrainingSAE(cfg), cfg
    if arch == "jumprelu":
        cfg = JumpReLUTrainingSAEConfig(l0_coefficient=2.0, l0_warm_up_steps=1000, **common)
        return JumpReLUTrainingSAE(cfg), cfg
    if arch == "matryoshka":
        widths = matryoshka_widths or _default_matryoshka_widths(d_sae)
        cfg = MatryoshkaBatchTopKTrainingSAEConfig(
            k=float(k), rescale_acts_by_decoder_norm=True,
            matryoshka_widths=widths, use_matryoshka_aux_loss=True, **common)
        return MatryoshkaBatchTopKTrainingSAE(cfg), cfg
    raise ValueError(f"unknown arch: {arch}")


def _default_matryoshka_widths(d_sae: int):
    # nested prefixes {1/16, 1/8, 1/2, full} of the dictionary
    ws = sorted({max(128, d_sae // 16), d_sae // 8, d_sae // 2, d_sae})
    return [w for w in ws if w <= d_sae]


# ---------------------------------------------------------------------------
# lr schedule + coefficient warmup
# ---------------------------------------------------------------------------
def make_lr_lambda(total_steps: int, warmup: int, decay: int):
    def fn(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if decay > 0 and step >= total_steps - decay:
            remaining = max(0, total_steps - step)
            return remaining / decay
        return 1.0
    return fn


def resolve_coefficients(coeff_cfg: Dict[str, Any], step: int) -> Dict[str, float]:
    from sae_lens.saes.sae import TrainCoefficientConfig
    out: Dict[str, float] = {}
    for name, c in coeff_cfg.items():
        if isinstance(c, TrainCoefficientConfig):
            warm = c.warm_up_steps
            scale = min(1.0, (step + 1) / warm) if warm > 0 else 1.0
            out[name] = c.value * scale
        else:
            out[name] = float(c)
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def batch_metrics(sae_in: torch.Tensor, sae_out: torch.Tensor,
                  feature_acts: torch.Tensor) -> Dict[str, float]:
    resid = sae_out - sae_in
    num = resid.pow(2).sum()
    den = (sae_in - sae_in.mean(0, keepdim=True)).pow(2).sum().clamp_min(1e-8)
    fvu = (num / den).item()
    l0 = (feature_acts > 0).float().sum(-1).mean().item()
    return {"fvu": fvu, "l0": l0}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train(
    layer: int,
    arch: str = "batchtopk",
    expansion: int = 16,
    d_sae: int | None = None,
    k: int = 32,
    init: str = "default",
    lr: float = 1e-4,
    total_steps: int = 30_000,
    batch_size: int = 4096,
    warmup_steps: int | None = None,
    decay_frac: float = 0.2,
    decoder_init_norm: float = 0.1,
    dead_feature_window: int = 2_000_000,   # in tokens
    holdout_rows: int = 50_000,
    train_block_rows: int = 32_768,
    log_every: int = 100,
    ckpt_every: int = 0,
    run_name: str | None = None,
    log_to_wandb: bool = True,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    paths.ensure_dirs()

    store = ActivationStore(
        paths.layer_acts_dir(layer), device=device,
        apply_input_scale=True, dtype=torch.float32,
        holdout_rows=holdout_rows, seed=seed, train_block_rows=train_block_rows)
    d_in = store.d_model
    d_sae = d_sae or expansion * d_in
    run_name = run_name or f"chemdfm_L{layer:02d}_{arch}_x{d_sae // d_in}_k{k}"
    out_dir = paths.run_dir(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    sae, sae_cfg = build_sae(arch, d_in, d_sae, k, device, decoder_init_norm)
    sae.to(device)
    coeff_cfg = sae.get_coefficients()

    asi_meta = None
    if init == "asi":
        n_fit = min(10_000, store.n_train_rows)
        x_fit = store.sample_train_blocks(n_fit, block_rows=1024)
        asi_meta = asi_init_sae(sae, x_fit, decoder_init_norm=decoder_init_norm)
        print(f"[train] ASI init: active_rank={asi_meta['active_rank']}/{d_in} "
              f"frac={asi_meta['active_rank_frac']} on {n_fit} rows")
        del x_fit

    warmup_steps = warmup_steps if warmup_steps is not None else max(1, total_steps // 20)
    decay_steps = int(total_steps * decay_frac)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_lambda(total_steps, warmup_steps, decay_steps))

    # dead-feature tracking (in tokens since last activation)
    tokens_since_fired = torch.zeros(d_sae, device=device)

    if log_to_wandb:
        import wandb
        wandb.init(project="chem_sae", name=run_name, config={
            "layer": layer, "arch": arch, "d_in": d_in, "d_sae": d_sae,
            "expansion": d_sae // d_in, "k": k, "lr": lr,
            "total_steps": total_steps, "batch_size": batch_size,
            "input_scale": store.input_scale, "n_train_rows": store.n_train_rows,
            "train_block_rows": train_block_rows,
        })

    print(f"[train] run={run_name} arch={arch} d_in={d_in} d_sae={d_sae} k={k} "
          f"| train_rows={store.n_train_rows} eval_rows={store.n_eval_rows} "
          f"block_rows={train_block_rows} input_scale={store.input_scale:.4f}")

    t0 = time.time()
    last = {}
    for step in range(total_steps):
        sae_in = store.next_train_batch(batch_size)
        dead_mask = tokens_since_fired > dead_feature_window

        from sae_lens.saes.sae import TrainStepInput
        step_input = TrainStepInput(
            sae_in=sae_in,
            coefficients=resolve_coefficients(coeff_cfg, step),
            dead_neuron_mask=dead_mask,
            n_training_steps=step,
            is_logging_step=(step % log_every == 0),
        )
        out = sae.training_forward_pass(step_input)
        loss = out.loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        sched.step()

        # update dead-feature counters
        with torch.no_grad():
            fired = (out.feature_acts > 0).any(0)
            tokens_since_fired[fired] = 0
            tokens_since_fired[~fired] += sae_in.shape[0]

        if step % log_every == 0 or step == total_steps - 1:
            m = batch_metrics(out.sae_in, out.sae_out, out.feature_acts)
            n_dead = int((tokens_since_fired > dead_feature_window).sum().item())
            rec = {
                "loss": float(loss.item()),
                "mse": float(out.losses["mse_loss"].item()),
                "fvu": m["fvu"], "l0": m["l0"],
                "dead_frac": n_dead / d_sae,
                "lr": sched.get_last_lr()[0],
                "tok_per_s": (step + 1) * batch_size / max(time.time() - t0, 1e-6),
            }
            last = rec
            print(f"[train] step={step:>6} loss={rec['loss']:.3f} fvu={rec['fvu']:.4f} "
                  f"l0={rec['l0']:.1f} dead={rec['dead_frac']:.3f} lr={rec['lr']:.2e} "
                  f"({rec['tok_per_s']:.0f} tok/s)")
            if log_to_wandb:
                import wandb
                wandb.log(rec, step=step)

        if ckpt_every and step > 0 and step % ckpt_every == 0:
            _save(sae, out_dir / f"checkpoint_{step}", store, sae_cfg, layer, arch, k, last, init, asi_meta)

    final = _save(sae, out_dir, store, sae_cfg, layer, arch, k, last, init, asi_meta)
    print(f"[train] done in {time.time() - t0:.0f}s -> {out_dir}")
    if log_to_wandb:
        import wandb
        wandb.finish()
    return final


def _save(sae, out_dir: Path, store: ActivationStore, sae_cfg, layer, arch, k, last_metrics,
          init="default", asi_meta=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path, cfg_path = sae.save_inference_model(str(out_dir))
    meta = {
        "layer": layer,
        "arch": arch,
        "k": k,
        "d_in": store.d_model,
        "d_sae": sae_cfg.d_sae,
        "expansion": sae_cfg.d_sae // store.d_model,
        "input_scale": store.input_scale,
        "acts_meta": store.meta,
        "weights_file": str(Path(weights_path).name),
        "cfg_file": str(Path(cfg_path).name),
        "final_metrics": last_metrics,
        "init": init,
        "asi": asi_meta,
        "note": "SAE trained on Stage-3 activations pre-scaled by input_scale; "
                "apply x * input_scale before encoding raw ChemDFM activations.",
    }
    (out_dir / "training_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a Chemistry SAE (SAELens) on ChemDFM activations")
    ap.add_argument("--layer", type=int, default=paths.MAIN_SAE_LAYER)
    ap.add_argument("--arch", choices=["batchtopk", "topk", "jumprelu", "matryoshka"],
                    default="batchtopk")
    ap.add_argument("--expansion", type=int, default=16, help="d_sae = expansion * d_in")
    ap.add_argument("--d-sae", type=int, default=None, help="override d_sae directly")
    ap.add_argument("--k", type=int, default=32, help="target L0 (avg active latents)")
    ap.add_argument("--init", choices=["default", "asi"], default="default",
                    help="asi = Active Subspace Init (OpenMOSS dead-feature fix)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--total-steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--decoder-init-norm", type=float, default=0.1)
    ap.add_argument("--dead-window", type=int, default=2_000_000,
                    help="tokens without firing before a latent is 'dead'")
    ap.add_argument("--holdout-rows", type=int, default=50_000)
    ap.add_argument("--train-block-rows", type=int, default=32_768)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run for validation")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.smoke:
        args.expansion = 8
        args.k = 32
        args.total_steps = 200
        args.batch_size = 256
        args.dead_window = 50_000
        args.holdout_rows = 500
        args.train_block_rows = 1024
        args.log_every = 20
        args.no_wandb = True

    train(
        layer=args.layer,
        arch=args.arch,
        expansion=args.expansion,
        d_sae=args.d_sae,
        k=args.k,
        init=args.init,
        lr=args.lr,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        decoder_init_norm=args.decoder_init_norm,
        dead_feature_window=args.dead_window,
        holdout_rows=args.holdout_rows,
        train_block_rows=args.train_block_rows,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        run_name=args.run_name,
        log_to_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
