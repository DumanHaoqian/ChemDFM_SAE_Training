#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_layer_sweep.py — train + reconstruction-eval a Chemistry SAE per layer.

The design (§2.4) calls for a *layer sweep* rather than committing to a single
layer: train one small SAE on each candidate middle layer, then pick the layer
by downstream detection AUC (a later stage). This orchestrator trains a SAE per
layer that has a Stage-3 activation cache, runs the cheap reconstruction eval,
and writes a combined ``sweep_summary.json``.

Usage::

    python run_layer_sweep.py --layers 12 18 24 26 30 --arch batchtopk \
        --expansion 16 --k 32 --total-steps 30000 --no-wandb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import paths
from train_sae import train
from eval_sae import recon_eval


def main() -> None:
    ap = argparse.ArgumentParser(description="SAE layer sweep (train + recon eval)")
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 18, 24, 26, 30])
    ap.add_argument("--arch", default="batchtopk")
    ap.add_argument("--expansion", type=int, default=16)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--total-steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow overwriting existing per-layer run directories")
    args = ap.parse_args()

    paths.ensure_dirs()
    summary = {}
    for layer in args.layers:
        acts_dir = paths.layer_acts_dir(layer)
        if not (acts_dir / "meta.json").exists():
            print(f"[sweep] SKIP layer {layer}: no Stage-3 cache at {acts_dir}")
            continue
        print(f"\n{'='*60}\n[sweep] layer {layer}\n{'='*60}")
        run_name = f"chemdfm_L{layer:02d}_{args.arch}_x{args.expansion}_k{args.k}"
        train(
            layer=layer, arch=args.arch, expansion=args.expansion, k=args.k,
            lr=args.lr, total_steps=args.total_steps, batch_size=args.batch_size,
            log_to_wandb=not args.no_wandb, run_name=run_name,
            overwrite=args.overwrite,
        )
        run_dir = paths.run_dir(run_name)
        recon = recon_eval(run_dir, layer, batch_size=args.batch_size)
        summary[layer] = {
            "run": str(run_dir),
            "fvu": recon["fvu"], "l0": recon["l0"],
            "dead_frac": recon["dead_frac"], "dense_frac": recon["dense_frac"],
        }
        print(f"[sweep] layer {layer}: fvu={recon['fvu']:.4f} l0={recon['l0']:.1f} "
              f"dead={recon['dead_frac']:.3f}")

    out = paths.OUT_DIR / "sweep_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[sweep] wrote {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
