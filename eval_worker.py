#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Persistent Delta LM loss worker for Stage-4 SAE training.

The trainer writes JSON requests containing checkpoint directories. This worker
runs in a separate process/GPU, preloads the 14B LM once, then repeatedly loads
requested SAE checkpoints and writes Delta LM loss responses. The trainer polls
those responses and logs them to its own W&B run.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import torch

import paths
from eval_sae import load_sae, load_training_meta
from model_config import HFHookedModel, get_model_config


def load_eval_texts(n_eval_seqs: int) -> List[str]:
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
    return texts


@torch.no_grad()
def delta_lm_loss_with_loaded_lm(
    run_dir: Path,
    layer: int,
    hk: HFHookedModel,
    texts: List[str],
    max_length: int,
    batch_size: int,
    device: str = "cuda",
) -> Dict[str, Any]:
    meta = load_training_meta(run_dir)
    input_scale = float(meta.get("input_scale", 1.0))
    sae = load_sae(run_dir, device)
    target = hk.layer_module(layer)

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

    del sae
    torch.cuda.empty_cache()

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


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Persistent Delta LM loss worker")
    ap.add_argument("--request-dir", required=True)
    ap.add_argument("--response-dir", required=True)
    ap.add_argument("--stop-file", required=True)
    ap.add_argument("--model", default="chemdfm")
    ap.add_argument("--n-eval-seqs", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--poll-interval", type=float, default=5.0)
    args = ap.parse_args()

    request_dir = Path(args.request_dir)
    response_dir = Path(args.response_dir)
    stop_file = Path(args.stop_file)
    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[eval-worker] loading LM model={args.model} n_eval_seqs={args.n_eval_seqs} "
        f"max_length={args.max_length} batch_size={args.batch_size}",
        flush=True,
    )
    hk = HFHookedModel(get_model_config(args.model), device="cuda")
    texts = load_eval_texts(args.n_eval_seqs)
    print("[eval-worker] ready", flush=True)

    seen: set[str] = set()
    while True:
        if stop_file.exists() and not list(request_dir.glob("*.json")):
            print("[eval-worker] stop requested; no pending requests", flush=True)
            break

        requests = sorted(request_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not requests:
            time.sleep(args.poll_interval)
            continue

        for req_path in requests:
            if req_path.name in seen:
                continue
            seen.add(req_path.name)
            try:
                req = json.loads(req_path.read_text(encoding="utf-8"))
                request_id = req["request_id"]
                step = int(req["step"])
                run_dir = Path(req["run_dir"])
                layer = int(req["layer"])
                print(f"[eval-worker] step={step} loading SAE {run_dir}", flush=True)
                t0 = time.time()
                metrics = delta_lm_loss_with_loaded_lm(
                    run_dir=run_dir,
                    layer=layer,
                    hk=hk,
                    texts=texts,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    device="cuda",
                )
                metrics["eval_seconds"] = time.time() - t0
                response = {
                    "request_id": request_id,
                    "status": "ok",
                    "step": step,
                    "run_dir": str(run_dir),
                    "metrics": metrics,
                }
                print(
                    f"[eval-worker] step={step} delta={metrics['delta_lm_loss']:.6f} "
                    f"recovered={metrics['ce_loss_recovered']:.4f} "
                    f"seconds={metrics['eval_seconds']:.1f}",
                    flush=True,
                )
            except Exception as exc:
                response = {
                    "request_id": req_path.stem,
                    "status": "error",
                    "step": -1,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                print("[eval-worker] ERROR", response["error"], flush=True)
                print(response["traceback"], flush=True)
            write_json_atomic(response_dir / f"{response['request_id']}.json", response)
            try:
                req_path.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
