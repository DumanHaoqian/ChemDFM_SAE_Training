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
from typing import Any, Dict

import torch

from delta_lm import delta_lm_loss_with_loaded_lm, load_eval_texts
from eval_sae import load_sae, load_training_meta
from model_config import HFHookedModel, get_model_config


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_text_atomic(path: Path, payload: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Persistent Delta LM loss worker")
    ap.add_argument("--request-dir", required=True)
    ap.add_argument("--response-dir", required=True)
    ap.add_argument("--stop-file", required=True)
    ap.add_argument("--ready-file", required=True)
    ap.add_argument("--heartbeat-file", required=True)
    ap.add_argument("--fatal-file", required=True)
    ap.add_argument("--eval-texts-path", default=None)
    ap.add_argument("--model", default="chemdfm")
    ap.add_argument("--n-eval-seqs", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--poll-interval", type=float, default=5.0)
    args = ap.parse_args()

    request_dir = Path(args.request_dir)
    response_dir = Path(args.response_dir)
    stop_file = Path(args.stop_file)
    ready_file = Path(args.ready_file)
    heartbeat_file = Path(args.heartbeat_file)
    fatal_file = Path(args.fatal_file)
    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    def heartbeat() -> None:
        write_text_atomic(heartbeat_file, f"{time.time():.6f}\n")

    try:
        print(
            f"[eval-worker] loading LM model={args.model} n_eval_seqs={args.n_eval_seqs} "
            f"max_length={args.max_length} batch_size={args.batch_size}",
            flush=True,
        )
        hk = HFHookedModel(get_model_config(args.model), device="cuda")
        texts, text_meta = load_eval_texts(args.eval_texts_path, args.n_eval_seqs)
        ready_payload = {
            "status": "ready",
            "ready_at": time.time(),
            "model": args.model,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            **text_meta,
        }
        write_json_atomic(ready_file, ready_payload)
        heartbeat()
        print("[eval-worker] ready", flush=True)
    except Exception as exc:
        fatal_payload = {
            "status": "fatal",
            "failed_at": time.time(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(fatal_file, fatal_payload)
        print("[eval-worker] FATAL", fatal_payload["error"], flush=True)
        print(fatal_payload["traceback"], flush=True)
        raise

    seen: set[str] = set()
    while True:
        heartbeat()
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
                meta = load_training_meta(run_dir)
                input_scale = float(meta.get("input_scale", 1.0))
                sae = load_sae(run_dir, device="cuda")
                metrics = delta_lm_loss_with_loaded_lm(
                    sae=sae,
                    input_scale=input_scale,
                    layer=layer,
                    hk=hk,
                    texts=texts,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    device="cuda",
                )
                metrics.update(text_meta)
                metrics["eval_seconds"] = time.time() - t0
                del sae
                torch.cuda.empty_cache()
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
