#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate a trained Chemistry SAE.

Two layers of evaluation:

1. Reconstruction / sparsity (cheap, no LM): FVU / normalised MSE, L0,
   dead & dense feature fractions, activation-frequency histogram. These are
   computed on held-out activation rows from the Stage-3 cache.

2. Delta LM loss (needs the 14B LM): splice the SAE reconstruction back into
   the residual stream at the SAE's layer and measure how much the model's
   next-token cross-entropy degrades vs. clean and vs. zero-ablation. This is
   the strongest fidelity metric here because it measures the causal effect of
   SAE reconstruction error on final LM predictions.

Usage:

    python eval_sae.py --run output/chemdfm_L26_batchtopk_x16_k32
    python eval_sae.py --run output/chemdfm_L26_batchtopk_x16_k32 --delta-lm-loss --n-eval-seqs 32

The legacy flag --delta-ce is kept as an alias for --delta-lm-loss.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

import paths
from data import ActivationStore


def load_sae(run_dir: Path, device: str):
    meta_path = run_dir / "training_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("arch") == "sparsemax_attention":
            from sparsemax_attention_sae import SparsemaxAttentionSAE

            return SparsemaxAttentionSAE.load_from_disk(run_dir, device=device)

    from sae_lens import SAE

    sae = SAE.load_from_disk(str(run_dir))
    sae.to(device)
    sae.eval()
    return sae


def load_training_meta(run_dir: Path) -> Dict[str, Any]:
    return json.loads((run_dir / "training_meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1) reconstruction / sparsity metrics on held-out activations
# ---------------------------------------------------------------------------
@torch.no_grad()
def recon_eval(
    run_dir: Path,
    layer: int,
    device: str = "cuda",
    batch_size: int = 4096,
    max_batches: int | None = None,
    dense_threshold: float = 0.5,
) -> Dict[str, Any]:
    meta = load_training_meta(run_dir)
    sae = load_sae(run_dir, device)
    d_sae = int(sae.cfg.d_sae)

    store = ActivationStore(
        paths.layer_acts_dir(layer),
        device=device,
        apply_input_scale=True,
        dtype=torch.float32,
        holdout_rows=max(batch_size * 4, 20_000),
        seed=42,
    )

    num = torch.zeros((), device=device)
    den = torch.zeros((), device=device)
    l0_sum = torch.zeros((), device=device)
    n_rows = 0
    fired_count = torch.zeros(d_sae, device=device)

    for x in store.eval_batches(batch_size, max_batches=max_batches):
        feats = sae.encode(x)
        recon = sae.decode(feats)
        num += (recon - x).pow(2).sum()
        den += (x - x.mean(0, keepdim=True)).pow(2).sum()
        l0_sum += (feats > 0).float().sum()
        fired_count += (feats > 0).float().sum(0)
        n_rows += x.shape[0]

    fvu = (num / den.clamp_min(1e-8)).item()
    l0 = (l0_sum / max(n_rows, 1)).item()
    freq = (fired_count / max(n_rows, 1)).cpu().numpy()
    dead_frac = float((freq == 0).mean())
    dense_frac = float((freq > dense_threshold).mean())

    hist_edges = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.01]
    hist = np.histogram(freq, bins=hist_edges)[0].tolist()

    return {
        "run": str(run_dir),
        "layer": layer,
        "arch": meta.get("arch"),
        "d_sae": d_sae,
        "k": meta.get("k"),
        "n_eval_rows": n_rows,
        "fvu": fvu,
        "explained_variance": 1.0 - fvu,
        "l0": l0,
        "dead_frac": dead_frac,
        "dense_frac": dense_frac,
        "freq_hist_edges": hist_edges,
        "freq_hist": hist,
        "input_scale": meta.get("input_scale"),
    }


# ---------------------------------------------------------------------------
# 2) Delta LM loss via residual-stream patching (needs the 14B LM)
# ---------------------------------------------------------------------------
@torch.no_grad()
def delta_lm_loss_eval(
    run_dir: Path,
    layer: int,
    device: str = "cuda",
    n_eval_seqs: int = 32,
    max_length: int = 128,
    batch_size: int = 1,
    model: str = "chemdfm",
) -> Dict[str, Any]:
    """Measure LM CE increase caused by replacing layer activations with SAE recon.

    clean: normal LM forward pass.
    recon: hook the target decoder block output and replace hidden_states by
        SAE.decode(SAE.encode(hidden_states * input_scale)) / input_scale.
    zero: replace the same hidden_states by zero, used as the denominator for
        CE-loss recovery.
    """
    meta = load_training_meta(run_dir)
    input_scale = float(meta.get("input_scale", 1.0))

    from model_config import HFHookedModel, get_model_config

    hk = HFHookedModel(get_model_config(model), device=device)
    target = hk.layer_module(layer)
    sae = load_sae(run_dir, device)

    corpus_path = paths.STAGE3_DIR / "data" / "corpus" / "sae_corpus.jsonl"
    texts: List[str] = []
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            texts.append(item["text"])
            if len(texts) >= n_eval_seqs:
                break
    if not texts:
        raise ValueError(f"No eval texts found in {corpus_path}")

    def ce_for(mode: str) -> float:
        total_loss, total_tok = 0.0, 0
        handle = None

        def hook(_module, _inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if mode == "zero":
                new = torch.zeros_like(hs)
            elif mode == "recon":
                x_scaled = hs.to(torch.float32) * input_scale
                recon_scaled = sae.decode(sae.encode(x_scaled))
                new = (recon_scaled / input_scale).to(hs.dtype)
            else:
                raise ValueError(f"Unexpected hook mode: {mode}")
            if isinstance(output, tuple):
                return (new,) + tuple(output[1:])
            return new

        if mode != "clean":
            handle = target.register_forward_hook(hook)
        try:
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                enc = hk.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(device)
                logits = hk.model(**enc).logits
                ids = enc["input_ids"]
                attn = enc["attention_mask"]
                shift_labels = ids[:, 1:]
                shift_mask = attn[:, 1:].bool()
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)).float(),
                    shift_labels.reshape(-1),
                    reduction="none",
                )
                loss = loss[shift_mask.reshape(-1)]
                total_loss += float(loss.sum().item())
                total_tok += int(shift_mask.sum().item())
                del logits, loss, enc
        finally:
            if handle is not None:
                handle.remove()
        return total_loss / max(total_tok, 1)

    ce_clean = ce_for("clean")
    ce_recon = ce_for("recon")
    ce_zero = ce_for("zero")
    delta = ce_recon - ce_clean
    zero_delta = ce_zero - ce_clean
    recovered = 1.0 - delta / max(zero_delta, 1e-8)

    return {
        "layer": layer,
        "hook_point": "resid_post",
        "hook_module": hk.cfg.layer_module_fmt.format(layer=layer),
        "n_eval_seqs": len(texts),
        "max_length": max_length,
        "batch_size": batch_size,
        "ce_clean": ce_clean,
        "ce_recon": ce_recon,
        "ce_zero": ce_zero,
        "delta_lm_loss": delta,
        "delta_ce": delta,
        "ce_loss_recovered": recovered,
    }


def delta_ce_eval(*args, **kwargs) -> Dict[str, Any]:
    """Backward-compatible alias for delta_lm_loss_eval."""
    return delta_lm_loss_eval(*args, **kwargs)


def maybe_log_wandb(
    args: argparse.Namespace,
    run_dir: Path,
    layer: int,
    recon: Dict[str, Any],
    delta_report: Dict[str, Any] | None,
) -> None:
    if not args.wandb:
        return
    import wandb

    name = args.wandb_run_name or f"eval_{run_dir.name}"
    wandb.init(
        project=args.wandb_project,
        name=name,
        config={
            "eval_run_dir": str(run_dir),
            "layer": layer,
            "model": args.model,
            "n_eval_seqs": args.n_eval_seqs,
            "max_length": args.max_length,
            "delta_batch_size": args.delta_batch_size,
        },
    )
    metrics = {
        "eval/fvu": recon["fvu"],
        "eval/explained_variance": recon["explained_variance"],
        "eval/l0": recon["l0"],
        "eval/dead_frac": recon["dead_frac"],
        "eval/dense_frac": recon["dense_frac"],
    }
    if delta_report is not None:
        metrics.update({
            "eval/ce_clean": delta_report["ce_clean"],
            "eval/ce_recon": delta_report["ce_recon"],
            "eval/ce_zero": delta_report["ce_zero"],
            "eval/delta_lm_loss": delta_report["delta_lm_loss"],
            "eval/delta_ce": delta_report["delta_ce"],
            "eval/ce_loss_recovered": delta_report["ce_loss_recovered"],
        })
    wandb.log(metrics)
    wandb.finish()


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained Chemistry SAE")
    ap.add_argument("--run", type=str, required=True, help="path to output/<run_name>")
    ap.add_argument("--layer", type=int, default=None,
                    help="activation layer (default: read from training_meta.json)")
    ap.add_argument("--batch-size", type=int, default=4096,
                    help="activation-row batch size for reconstruction eval")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--delta-lm-loss", action="store_true",
                    help="run Delta LM loss eval by splicing SAE recon into the LM")
    ap.add_argument("--delta-ce", action="store_true",
                    help="legacy alias for --delta-lm-loss")
    ap.add_argument("--n-eval-seqs", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--delta-batch-size", type=int, default=1,
                    help="text batch size for Delta LM loss eval")
    ap.add_argument("--model", default="chemdfm", help="model_config registry name")
    ap.add_argument("--wandb", action="store_true", help="log eval metrics to W&B")
    ap.add_argument("--wandb-project", default="chem_sae")
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    meta = load_training_meta(run_dir)
    layer = args.layer if args.layer is not None else int(meta["layer"])

    recon = recon_eval(
        run_dir,
        layer,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )
    print("[recon]", json.dumps(
        {k: v for k, v in recon.items() if k not in ("freq_hist_edges", "freq_hist")},
        indent=2,
    ))
    report: Dict[str, Any] = {"recon": recon}

    delta_report = None
    if args.delta_lm_loss or args.delta_ce:
        delta_report = delta_lm_loss_eval(
            run_dir,
            layer,
            n_eval_seqs=args.n_eval_seqs,
            max_length=args.max_length,
            batch_size=args.delta_batch_size,
            model=args.model,
        )
        print(
            "[delta-lm-loss] "
            f"clean={delta_report['ce_clean']:.6f} "
            f"recon={delta_report['ce_recon']:.6f} "
            f"zero={delta_report['ce_zero']:.6f} "
            f"delta={delta_report['delta_lm_loss']:.6f} "
            f"recovered={delta_report['ce_loss_recovered']:.4f}"
        )
        print("[delta-lm-loss-json]", json.dumps(delta_report, indent=2))
        report["delta_lm_loss"] = delta_report
        report["delta_ce"] = delta_report

    maybe_log_wandb(args, run_dir, layer, recon, delta_report)

    (run_dir / "eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[eval] wrote {run_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
