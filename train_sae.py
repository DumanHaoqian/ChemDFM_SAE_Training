#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_sae.py — train a Sparse Autoencoder on ChemDFM residual-stream activations.

Uses SAELens' SAE implementations plus local SAELens-compatible modules (BatchTopK is the main line, §2.2) inside a
custom training loop that reads the Stage-3 disk cache. This keeps full control
over the 14B HF model / chemistry tokenizer / layer sweep without forcing
ChemDFM through TransformerLens.

Architectures (``--arch``):
  * ``batchtopk``  (default) — batch-level TopK; saved as JumpReLU for inference.
  * ``topk``       — per-sample TopK.
  * ``jumprelu``   — JumpReLU (Gemma-Scope style baseline).
  * ``matryoshka`` — Matryoshka-BatchTopK with nested prefixes.
  * ``sparsemax_attention`` -- sparsemax attention over a learned SAE dictionary.

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
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch


class Tee:
    """Write stdout/stderr both to the terminal/nohup stream and a configured log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def setup_file_logging(log_dir: str | None, log_file: str | None, run_name: str | None) -> Path | None:
    """Optionally tee stdout/stderr to a config-defined log file."""
    if not log_dir:
        return None
    log_root = Path(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    if log_file:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = log_root / log_path
    else:
        safe_name = run_name or time.strftime("train_%Y%m%d_%H%M%S")
        log_path = log_root / f"{safe_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, fh)
    sys.stderr = Tee(sys.stderr, fh)
    print(f"[log] tee stdout/stderr -> {log_path}")
    return log_path

import paths
from data import ActivationStore
from init_asi import asi_init_sae
from metrics import reconstruction_fvu


# ---------------------------------------------------------------------------
# SAE construction
# ---------------------------------------------------------------------------
def build_sae(arch: str, d_in: int, d_sae: int, k: int, device: str,
              decoder_init_norm: float, jumprelu_l0_coefficient: float = 2.0,
              jumprelu_l0_warm_up_steps: int = 1000, matryoshka_widths=None,
              sparsemax_key_dim: int | None = None,
              sparsemax_preselect_k: int | None = 2048,
              sparsemax_activation_mode: str = "probs",
              sparsemax_use_input_norm: bool = True,
              sparsemax_use_idf_mask: bool = False,
              sparsemax_idf_threshold: float = 0.1,
              sparsemax_mse_loss_scale: float = 1.0,
              sparsemax_score_scale: float = 2.0,
              sparsemax_l0_target: float | None = 32.0,
              sparsemax_l0_coefficient: float = 10.0,
              sparsemax_cosine_loss_coefficient: float = 0.0,
              sparsemax_norm_loss_coefficient: float = 0.0,
              sparsemax_value_scale_init: float = 1.0,
              sparsemax_global_output_scale_init: float = 1.0):
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
        cfg = JumpReLUTrainingSAEConfig(
            l0_coefficient=float(jumprelu_l0_coefficient),
            l0_warm_up_steps=int(jumprelu_l0_warm_up_steps),
            **common)
        return JumpReLUTrainingSAE(cfg), cfg
    if arch == "matryoshka":
        widths = matryoshka_widths or _default_matryoshka_widths(d_sae)
        cfg = MatryoshkaBatchTopKTrainingSAEConfig(
            k=float(k), rescale_acts_by_decoder_norm=True,
            matryoshka_widths=widths, use_matryoshka_aux_loss=True, **common)
        return MatryoshkaBatchTopKTrainingSAE(cfg), cfg
    if arch == "sparsemax_attention":
        from sparsemax_attention_sae import SparsemaxAttentionSAE, SparsemaxAttentionSAEConfig

        cfg = SparsemaxAttentionSAEConfig(
            d_in=d_in,
            d_sae=d_sae,
            dtype="float32",
            device=device,
            decoder_init_norm=decoder_init_norm,
            key_dim=sparsemax_key_dim,
            preselect_k=sparsemax_preselect_k,
            activation_mode=sparsemax_activation_mode,
            use_input_norm=sparsemax_use_input_norm,
            use_idf_mask=sparsemax_use_idf_mask,
            idf_threshold=sparsemax_idf_threshold,
            mse_loss_scale=sparsemax_mse_loss_scale,
            score_scale=sparsemax_score_scale,
            l0_target=sparsemax_l0_target,
            l0_coefficient=sparsemax_l0_coefficient,
            cosine_loss_coefficient=sparsemax_cosine_loss_coefficient,
            norm_loss_coefficient=sparsemax_norm_loss_coefficient,
            value_scale_init=sparsemax_value_scale_init,
            global_output_scale_init=sparsemax_global_output_scale_init,
        )
        return SparsemaxAttentionSAE(cfg), cfg
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
    fvu = reconstruction_fvu(sae_in, sae_out)
    l0 = (feature_acts > 0).float().sum(-1).mean().item()
    return {"fvu": fvu, "l0": l0}




# ---------------------------------------------------------------------------
# periodic Delta LM loss evaluation on a second GPU
# ---------------------------------------------------------------------------
class PeriodicDeltaLmEvaluator:
    """Launch and coordinate a persistent Delta LM loss worker process."""

    def __init__(
        self,
        out_dir: Path,
        layer: int,
        eval_gpu: int,
        eval_per_step: int,
        eval_batch_size: int,
        eval_n_eval_seqs: int,
        eval_max_length: int,
        eval_model: str,
        eval_timeout_sec: int,
        eval_startup_timeout_sec: int,
        eval_request_timeout_sec: int,
        eval_poll_interval: float,
        eval_keep_checkpoints: bool,
        eval_texts_path: str | None,
    ):
        self.out_dir = Path(out_dir)
        self.layer = int(layer)
        self.eval_gpu = int(eval_gpu)
        self.eval_per_step = int(eval_per_step)
        self.eval_timeout_sec = int(eval_timeout_sec)
        self.eval_startup_timeout_sec = int(eval_startup_timeout_sec)
        self.eval_request_timeout_sec = int(eval_request_timeout_sec)
        self.eval_keep_checkpoints = bool(eval_keep_checkpoints)
        self.root = self.out_dir / "delta_lm_eval"
        self.request_dir = self.root / "requests"
        self.response_dir = self.root / "responses"
        self.stop_file = self.root / "STOP"
        self.ready_file = self.root / "READY.json"
        self.heartbeat_file = self.root / "HEARTBEAT"
        self.fatal_file = self.root / "FATAL.json"
        self.log_path = self.root / "eval_worker.log"
        self.pending: Dict[str, Dict[str, Any]] = {}
        self.best_delta_lm_loss = float("inf")
        self.best_dir = self.out_dir / "best_delta_lm_loss"
        self.best_meta_path = self.out_dir / "best_delta_lm_loss.json"
        self.proc: subprocess.Popen | None = None

        self.root.mkdir(parents=True, exist_ok=True)
        self.request_dir.mkdir(parents=True, exist_ok=True)
        self.response_dir.mkdir(parents=True, exist_ok=True)
        for d in (self.request_dir, self.response_dir):
            for stale in list(d.glob("*.json")) + list(d.glob("*.tmp")):
                stale.unlink(missing_ok=True)
        for stale in (self.stop_file, self.ready_file, self.heartbeat_file, self.fatal_file):
            stale.unlink(missing_ok=True)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.eval_gpu)
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("eval_worker.py")),
            "--request-dir", str(self.request_dir),
            "--response-dir", str(self.response_dir),
            "--stop-file", str(self.stop_file),
            "--ready-file", str(self.ready_file),
            "--heartbeat-file", str(self.heartbeat_file),
            "--fatal-file", str(self.fatal_file),
            "--model", eval_model,
            "--n-eval-seqs", str(eval_n_eval_seqs),
            "--max-length", str(eval_max_length),
            "--batch-size", str(eval_batch_size),
            "--poll-interval", str(eval_poll_interval),
        ]
        if eval_texts_path:
            cmd.extend(["--eval-texts-path", str(eval_texts_path)])
        self._log_fh = self.log_path.open("a", encoding="utf-8")
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).parent),
            env=env,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(f"[eval] worker pid={self.proc.pid} gpu={self.eval_gpu} log={self.log_path}")
        try:
            self._wait_until_ready()
        except Exception:
            self.close()
            raise

    def _read_json(self, path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _worker_tail(self, n_chars: int = 4000) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-n_chars:]
        except FileNotFoundError:
            return ""

    def _raise_fatal_if_present(self) -> None:
        if self.fatal_file.exists():
            try:
                fatal = self._read_json(self.fatal_file)
                detail = fatal.get("error") or fatal
            except Exception:
                detail = self.fatal_file.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"Delta LM eval worker fatal error: {detail}\n{self._worker_tail()}")

    def _check_worker_alive(self) -> None:
        self._raise_fatal_if_present()
        if self.proc is not None and self.proc.poll() is not None and not self.stop_file.exists():
            raise RuntimeError(
                f"Delta LM eval worker exited with code {self.proc.returncode}; log={self.log_path}\n"
                f"{self._worker_tail()}"
            )

    def _wait_until_ready(self) -> None:
        start = time.time()
        while time.time() - start < self.eval_startup_timeout_sec:
            self._check_worker_alive()
            if self.ready_file.exists():
                ready = self._read_json(self.ready_file)
                print(
                    f"[eval] worker ready texts={ready.get('n_eval_texts')} "
                    f"sha256={str(ready.get('eval_texts_sha256', ''))[:12]}"
                )
                return
            time.sleep(1)
        raise TimeoutError(
            f"Delta LM eval worker was not ready after {self.eval_startup_timeout_sec}s; "
            f"log={self.log_path}\n{self._worker_tail()}"
        )

    def _check_request_timeouts(self) -> None:
        now = time.time()
        expired = [
            request_id for request_id, payload in self.pending.items()
            if now - float(payload["created_at"]) > self.eval_request_timeout_sec
        ]
        if expired:
            raise TimeoutError(
                f"Delta LM eval request timeout after {self.eval_request_timeout_sec}s: {expired}; "
                f"log={self.log_path}\n{self._worker_tail()}"
            )

    def enqueue(self, step: int, run_dir: Path) -> None:
        self._check_worker_alive()
        request_id = f"step_{int(step):06d}"
        payload = {
            "request_id": request_id,
            "step": int(step),
            "run_dir": str(Path(run_dir).resolve()),
            "layer": self.layer,
            "created_at": time.time(),
        }
        tmp = self.request_dir / f"{request_id}.tmp"
        final = self.request_dir / f"{request_id}.json"
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(final)
        self.pending[request_id] = {
            "step": int(step),
            "run_dir": str(run_dir),
            "created_at": time.time(),
        }
        print(f"[eval] queued Delta LM loss step={step} checkpoint={run_dir}")

    def poll(self, wandb_module=None) -> None:
        self._check_worker_alive()
        self._check_request_timeouts()
        for resp_path in sorted(self.response_dir.glob("*.json")):
            try:
                resp = json.loads(resp_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            request_id = resp.get("request_id", resp_path.stem)
            pending = self.pending.pop(request_id, None)
            step = int(resp.get("step", pending.get("step", -1) if pending else -1))
            if resp.get("status") == "ok":
                metrics = resp["metrics"]
                log_metrics = {
                    "eval_step": step,
                    "eval/ce_clean": metrics["ce_clean"],
                    "eval/ce_recon": metrics["ce_recon"],
                    "eval/ce_zero": metrics["ce_zero"],
                    "eval/delta_lm_loss": metrics["delta_lm_loss"],
                    "eval/delta_ce": metrics["delta_ce"],
                    "eval/ce_loss_recovered": metrics["ce_loss_recovered"],
                    "eval/eval_seconds": metrics.get("eval_seconds", 0.0),
                }
                delta = float(metrics["delta_lm_loss"])
                is_best = delta < self.best_delta_lm_loss
                log_metrics["eval/is_best_delta_lm_loss"] = int(is_best)
                if is_best:
                    self.best_delta_lm_loss = delta
                    source_dir = Path(resp.get("run_dir") or (pending or {}).get("run_dir", ""))
                    if source_dir.exists():
                        if self.best_dir.exists():
                            shutil.rmtree(self.best_dir, ignore_errors=True)
                        shutil.copytree(source_dir, self.best_dir)
                    best_payload = {
                        "step": step,
                        "source_checkpoint": str(source_dir),
                        "best_dir": str(self.best_dir),
                        "metrics": metrics,
                        "saved_at": time.time(),
                    }
                    self.best_meta_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
                    if self.best_dir.exists():
                        (self.best_dir / "best_delta_lm_loss.json").write_text(
                            json.dumps(best_payload, indent=2), encoding="utf-8")
                    print(f"[eval] new best Delta LM loss step={step} delta={delta:.6f} -> {self.best_dir}")
                print(
                    f"[eval] step={step} delta_lm_loss={metrics['delta_lm_loss']:.6f} "
                    f"ce_recovered={metrics['ce_loss_recovered']:.4f}"
                )
                if wandb_module is not None and step >= 0:
                    wandb_module.log(log_metrics)
            else:
                raise RuntimeError(f"[eval] request {request_id} failed: {resp.get('error')}")
            if pending and not self.eval_keep_checkpoints:
                shutil.rmtree(pending["run_dir"], ignore_errors=True)
            try:
                resp_path.unlink()
            except FileNotFoundError:
                pass

    def drain(self, wandb_module=None) -> None:
        start = time.time()
        while self.pending and time.time() - start < self.eval_timeout_sec:
            self.poll(wandb_module)
            if self.pending:
                time.sleep(5)
        if self.pending:
            raise TimeoutError(
                f"[eval] timeout with pending requests after {self.eval_timeout_sec}s: "
                f"{sorted(self.pending)}"
            )

    def close(self) -> None:
        self.stop_file.write_text("stop\n", encoding="utf-8")
        if self.proc is not None:
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=10)
        try:
            self._log_fh.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train(
    layer: int,
    arch: str = "batchtopk",
    expansion: int = 16,
    d_sae: int | None = None,
    k: int = 32,
    jumprelu_l0_coefficient: float = 2.0,
    jumprelu_l0_warm_up_steps: int = 1000,
    matryoshka_widths: list[int] | None = None,
    sparsemax_key_dim: int | None = None,
    sparsemax_preselect_k: int | None = 2048,
    sparsemax_activation_mode: str = "probs",
    sparsemax_use_input_norm: bool = True,
    sparsemax_use_idf_mask: bool = False,
    sparsemax_idf_threshold: float = 0.1,
    sparsemax_mse_loss_scale: float = 1.0,
    sparsemax_score_scale: float = 2.0,
    sparsemax_l0_target: float | None = 32.0,
    sparsemax_l0_coefficient: float = 10.0,
    sparsemax_cosine_loss_coefficient: float = 0.0,
    sparsemax_norm_loss_coefficient: float = 0.0,
    sparsemax_value_scale_init: float = 1.0,
    sparsemax_global_output_scale_init: float = 1.0,
    sparsemax_init_b_dec_from_data: bool = False,
    sparsemax_b_dec_init_rows: int = 10_000,
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
    output_dir: str | None = None,
    checkpoint_dir: str | None = None,
    wandb_project: str = "chem_sae",
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    log_to_wandb: bool = True,
    device: str = "cuda",
    seed: int = 42,
    eval_enabled: bool = False,
    eval_gpu: int = 7,
    eval_per_step: int = 500,
    eval_batch_size: int = 2,
    eval_n_eval_seqs: int = 32,
    eval_max_length: int = 128,
    eval_model: str = "chemdfm",
    eval_timeout_sec: int = 3600,
    eval_startup_timeout_sec: int = 600,
    eval_request_timeout_sec: int = 1800,
    eval_poll_interval: float = 5.0,
    eval_keep_checkpoints: bool = False,
    eval_texts_path: str | None = None,
    overwrite: bool = False,
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
    output_root = Path(output_dir) if output_dir else paths.OUT_DIR
    out_dir = output_root / run_name
    if out_dir.exists() and (out_dir / "training_meta.json").exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing run directory {out_dir}. "
            "Pass --overwrite for an intentional rerun with the same name."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = (Path(checkpoint_dir) / run_name) if checkpoint_dir else out_dir
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    sae, sae_cfg = build_sae(
        arch, d_in, d_sae, k, device, decoder_init_norm,
        jumprelu_l0_coefficient=jumprelu_l0_coefficient,
        jumprelu_l0_warm_up_steps=jumprelu_l0_warm_up_steps,
        matryoshka_widths=matryoshka_widths,
        sparsemax_key_dim=sparsemax_key_dim,
        sparsemax_preselect_k=sparsemax_preselect_k,
        sparsemax_activation_mode=sparsemax_activation_mode,
        sparsemax_use_input_norm=sparsemax_use_input_norm,
        sparsemax_use_idf_mask=sparsemax_use_idf_mask,
        sparsemax_idf_threshold=sparsemax_idf_threshold,
        sparsemax_mse_loss_scale=sparsemax_mse_loss_scale,
        sparsemax_score_scale=sparsemax_score_scale,
        sparsemax_l0_target=sparsemax_l0_target,
        sparsemax_l0_coefficient=sparsemax_l0_coefficient,
        sparsemax_cosine_loss_coefficient=sparsemax_cosine_loss_coefficient,
        sparsemax_norm_loss_coefficient=sparsemax_norm_loss_coefficient,
        sparsemax_value_scale_init=sparsemax_value_scale_init,
        sparsemax_global_output_scale_init=sparsemax_global_output_scale_init,
    )
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

    if arch == "sparsemax_attention" and sparsemax_init_b_dec_from_data:
        n_bias = min(int(sparsemax_b_dec_init_rows), store.n_train_rows)
        x_bias = store.sample_train_blocks(n_bias, block_rows=1024)
        bias_meta = sae.set_decoder_bias_from_data(x_bias)
        print(f"[train] Sparsemax b_dec init: rows={bias_meta['b_dec_init_rows']} "
              f"norm={bias_meta['b_dec_init_norm']:.4f}")
        del x_bias

    warmup_steps = warmup_steps if warmup_steps is not None else max(1, total_steps // 20)
    decay_steps = int(total_steps * decay_frac)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_lambda(total_steps, warmup_steps, decay_steps))

    # dead-feature tracking (in tokens since last activation)
    tokens_since_fired = torch.zeros(d_sae, device=device)

    wandb_started = False
    if log_to_wandb:
        import wandb
        wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name or run_name, config={
            "layer": layer, "arch": arch, "d_in": d_in, "d_sae": d_sae,
            "expansion": d_sae // d_in, "k": k,
            "jumprelu_l0_coefficient": jumprelu_l0_coefficient,
            "jumprelu_l0_warm_up_steps": jumprelu_l0_warm_up_steps,
            "matryoshka_widths": matryoshka_widths,
            "sparsemax_key_dim": sparsemax_key_dim,
            "sparsemax_preselect_k": sparsemax_preselect_k,
            "sparsemax_activation_mode": sparsemax_activation_mode,
            "sparsemax_use_input_norm": sparsemax_use_input_norm,
            "sparsemax_use_idf_mask": sparsemax_use_idf_mask,
            "sparsemax_idf_threshold": sparsemax_idf_threshold,
            "sparsemax_mse_loss_scale": sparsemax_mse_loss_scale,
            "sparsemax_score_scale": sparsemax_score_scale,
            "sparsemax_l0_target": sparsemax_l0_target,
            "sparsemax_l0_coefficient": sparsemax_l0_coefficient,
            "sparsemax_cosine_loss_coefficient": sparsemax_cosine_loss_coefficient,
            "sparsemax_norm_loss_coefficient": sparsemax_norm_loss_coefficient,
            "sparsemax_value_scale_init": sparsemax_value_scale_init,
            "sparsemax_global_output_scale_init": sparsemax_global_output_scale_init,
            "sparsemax_init_b_dec_from_data": sparsemax_init_b_dec_from_data,
            "sparsemax_b_dec_init_rows": sparsemax_b_dec_init_rows,
            "lr": lr,
            "total_steps": total_steps, "batch_size": batch_size,
            "input_scale": store.input_scale, "n_train_rows": store.n_train_rows,
            "train_block_rows": train_block_rows,
            "output_dir": str(out_dir),
            "checkpoint_root": str(checkpoint_root),
            "training_tokens_total": total_steps * batch_size,
            "wandb_project": wandb_project,
            "wandb_entity": wandb_entity,
            "wandb_run_name": wandb_run_name or run_name,
            "eval_enabled": eval_enabled,
            "eval_gpu": eval_gpu,
            "eval_per_step": eval_per_step,
            "eval_batch_size": eval_batch_size,
            "eval_n_eval_seqs": eval_n_eval_seqs,
            "eval_max_length": eval_max_length,
            "eval_model": eval_model,
            "eval_texts_path": eval_texts_path or str(paths.EVAL_TEXTS_PATH),
            "eval_startup_timeout_sec": eval_startup_timeout_sec,
            "eval_request_timeout_sec": eval_request_timeout_sec,
            "overwrite": overwrite,
        })
        wandb_started = True
        wandb.define_metric("eval_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    print(f"[train] run={run_name} arch={arch} d_in={d_in} d_sae={d_sae} k={k} "
          f"| train_rows={store.n_train_rows} eval_rows={store.n_eval_rows} "
          f"block_rows={train_block_rows} input_scale={store.input_scale:.4f}")

    eval_mgr = None
    t0 = time.time()
    last = {}
    try:
        if eval_enabled:
            if not log_to_wandb:
                print("[eval] WARNING: eval_enabled=True but W&B is disabled; metrics will print but not upload")
            eval_mgr = PeriodicDeltaLmEvaluator(
                out_dir=out_dir,
                layer=layer,
                eval_gpu=eval_gpu,
                eval_per_step=eval_per_step,
                eval_batch_size=eval_batch_size,
                eval_n_eval_seqs=eval_n_eval_seqs,
                eval_max_length=eval_max_length,
                eval_model=eval_model,
                eval_timeout_sec=eval_timeout_sec,
                eval_startup_timeout_sec=eval_startup_timeout_sec,
                eval_request_timeout_sec=eval_request_timeout_sec,
                eval_poll_interval=eval_poll_interval,
                eval_keep_checkpoints=eval_keep_checkpoints,
                eval_texts_path=eval_texts_path,
            )

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
                    "tokens_seen": (step + 1) * batch_size,
                    "tokens_total": total_steps * batch_size,
                    "token_progress": (step + 1) / max(total_steps, 1),
                    "train_epoch_equiv": ((step + 1) * batch_size) / max(store.n_train_rows, 1),
                    "tok_per_s": (step + 1) * batch_size / max(time.time() - t0, 1e-6),
                }
                for loss_name, loss_value in out.losses.items():
                    if loss_name == "loss":
                        continue
                    if hasattr(loss_value, "detach"):
                        rec[f"loss/{loss_name}"] = float(loss_value.detach().float().item())
                last = rec
                aux = ""
                if "loss/l0_proxy" in rec:
                    aux += f" l0p={rec['loss/l0_proxy']:.1f}"
                if "loss/cosine_raw" in rec:
                    aux += f" cos={rec['loss/cosine_raw']:.3f}"
                if "loss/norm_rel_error" in rec:
                    aux += f" normerr={rec['loss/norm_rel_error']:.3f}"
                print(f"[train] step={step:>6} loss={rec['loss']:.3f} fvu={rec['fvu']:.4f} "
                      f"l0={rec['l0']:.1f}{aux} dead={rec['dead_frac']:.3f} lr={rec['lr']:.2e} "
                      f"tokens={rec['tokens_seen'] / 1e6:.1f}M/{rec['tokens_total'] / 1e6:.1f}M "
                      f"({rec['tok_per_s']:.0f} tok/s)")
                if log_to_wandb:
                    import wandb
                    wandb.log(rec, step=step)

            if ckpt_every and step > 0 and step % ckpt_every == 0:
                _save(sae, checkpoint_root / f"checkpoint_{step}", store, sae_cfg, layer, arch, k, last, init, asi_meta)

            completed_steps = step + 1
            wandb_module = None
            if log_to_wandb:
                import wandb as wandb_module
            if eval_mgr is not None:
                eval_mgr.poll(wandb_module)
                if eval_per_step > 0 and completed_steps % eval_per_step == 0:
                    eval_ckpt = out_dir / "eval_checkpoints" / f"step_{completed_steps:06d}"
                    _save(sae, eval_ckpt, store, sae_cfg, layer, arch, k, last, init, asi_meta)
                    eval_mgr.enqueue(completed_steps, eval_ckpt)

        final = _save(sae, out_dir, store, sae_cfg, layer, arch, k, last, init, asi_meta)
        if eval_mgr is not None:
            wandb_module = None
            if log_to_wandb:
                import wandb as wandb_module
            eval_mgr.drain(wandb_module)
        print(f"[train] done in {time.time() - t0:.0f}s -> {out_dir}")
        return final
    finally:
        if eval_mgr is not None:
            eval_mgr.close()
        if log_to_wandb and wandb_started:
            import wandb
            wandb.finish()


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
        "training_tokens_seen": last_metrics.get("tokens_seen") if isinstance(last_metrics, dict) else None,
        "training_tokens_total": last_metrics.get("tokens_total") if isinstance(last_metrics, dict) else None,
        "init": init,
        "asi": asi_meta,
        "note": "SAE trained on Stage-3 activations pre-scaled by input_scale; "
                "apply x * input_scale before encoding raw ChemDFM activations.",
    }
    (out_dir / "training_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    import yaml
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {str(k).replace("-", "_"): v for k, v in data.items()}


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="configs/train.yaml",
                     help="YAML config path; CLI flags override values")
    pre_args, _ = pre.parse_known_args()
    cfg = _load_config(pre_args.config)

    def c(name: str, default: Any) -> Any:
        return cfg.get(name, default)

    ap = argparse.ArgumentParser(
        description="Train a Chemistry SAE (SAELens) on ChemDFM activations",
        parents=[pre],
    )
    ap.add_argument("--layer", type=int, default=c("layer", paths.MAIN_SAE_LAYER))
    ap.add_argument("--arch", choices=["batchtopk", "topk", "jumprelu", "matryoshka", "sparsemax_attention"],
                    default=c("arch", "batchtopk"))
    ap.add_argument("--expansion", type=int, default=c("expansion", 16), help="d_sae = expansion * d_in")
    ap.add_argument("--d-sae", type=int, default=c("d_sae", None), help="override d_sae directly")
    ap.add_argument("--k", type=int, default=c("k", 32), help="target L0 (avg active latents)")
    ap.add_argument("--jumprelu-l0-coefficient", type=float,
                    default=c("jumprelu_l0_coefficient", 2.0),
                    help="JumpReLU L0 sparsity penalty coefficient; higher is sparser")
    ap.add_argument("--jumprelu-l0-warm-up-steps", type=int,
                    default=c("jumprelu_l0_warm_up_steps", 1000),
                    help="steps used to warm up the JumpReLU L0 penalty")
    ap.add_argument("--matryoshka-widths", default=c("matryoshka_widths", None),
                    help="comma-separated nested prefix widths for Matryoshka SAE; default is d_sae/16,d_sae/8,d_sae/2,d_sae")
    ap.add_argument("--sparsemax-key-dim", type=int, default=c("sparsemax_key_dim", None),
                    help="attention key/query dimension for Sparsemax Attention SAE; null uses min(1024, d_in)")
    ap.add_argument("--sparsemax-preselect-k", type=int, default=c("sparsemax_preselect_k", 2048),
                    help="top score candidates before sparsemax; <=0 means full dictionary sparsemax")
    ap.add_argument("--sparsemax-activation-mode", choices=["probs", "masked_scores"],
                    default=c("sparsemax_activation_mode", "probs"),
                    help="Sparsemax activations: sparse probabilities or sparse masked logits")
    ap.add_argument("--sparsemax-use-input-norm", dest="sparsemax_use_input_norm", action="store_true",
                    default=bool(c("sparsemax_use_input_norm", True)),
                    help="encode activation direction and multiply sparse coefficients by input norm")
    ap.add_argument("--sparsemax-no-input-norm", dest="sparsemax_use_input_norm", action="store_false",
                    help="disable input-norm factorization for Sparsemax Attention SAE")
    ap.add_argument("--sparsemax-use-idf-mask", dest="sparsemax_use_idf_mask", action="store_true",
                    default=bool(c("sparsemax_use_idf_mask", False)),
                    help="mask overly frequent Sparsemax features using the running idf_score buffer")
    ap.add_argument("--sparsemax-no-idf-mask", dest="sparsemax_use_idf_mask", action="store_false",
                    help="disable Sparsemax frequent-feature mask even if config enables it")
    ap.add_argument("--sparsemax-idf-threshold", type=float, default=c("sparsemax_idf_threshold", 0.1),
                    help="feature firing-frequency threshold for Sparsemax idf masking")
    ap.add_argument("--sparsemax-mse-loss-scale", type=float, default=c("sparsemax_mse_loss_scale", 1.0),
                    help="multiplier on Sparsemax SAE reconstruction MSE")
    ap.add_argument("--sparsemax-score-scale", type=float, default=c("sparsemax_score_scale", 2.0),
                    help="fixed scale applied to normalized attention scores before sparsemax")
    ap.add_argument("--sparsemax-l0-target", type=float, default=c("sparsemax_l0_target", 32.0),
                    help="target participation-ratio L0 for Sparsemax activations; <=0 disables")
    ap.add_argument("--sparsemax-l0-coefficient", type=float, default=c("sparsemax_l0_coefficient", 10.0),
                    help="coefficient for the Sparsemax L0 target loss")
    ap.add_argument("--sparsemax-cosine-loss-coefficient", type=float, default=c("sparsemax_cosine_loss_coefficient", 0.0),
                    help="coefficient for activation-direction cosine reconstruction loss")
    ap.add_argument("--sparsemax-norm-loss-coefficient", type=float, default=c("sparsemax_norm_loss_coefficient", 0.0),
                    help="coefficient for relative activation-norm reconstruction loss")
    ap.add_argument("--sparsemax-value-scale-init", type=float, default=c("sparsemax_value_scale_init", 1.0),
                    help="initial learned per-feature value scale")
    ap.add_argument("--sparsemax-global-output-scale-init", type=float, default=c("sparsemax_global_output_scale_init", 1.0),
                    help="initial learned scalar output scale")
    ap.add_argument("--sparsemax-init-b-dec-from-data", dest="sparsemax_init_b_dec_from_data", action="store_true",
                    default=bool(c("sparsemax_init_b_dec_from_data", False)),
                    help="initialize Sparsemax decoder bias from activation mean")
    ap.add_argument("--sparsemax-no-init-b-dec-from-data", dest="sparsemax_init_b_dec_from_data", action="store_false",
                    help="disable Sparsemax decoder-bias data mean init")
    ap.add_argument("--sparsemax-b-dec-init-rows", type=int, default=c("sparsemax_b_dec_init_rows", 10000),
                    help="number of contiguous sampled activation rows for Sparsemax b_dec mean init")
    ap.add_argument("--init", choices=["default", "asi"], default=c("init", "default"),
                    help="asi = Active Subspace Init (OpenMOSS dead-feature fix)")
    ap.add_argument("--lr", type=float, default=c("lr", 1e-4))
    ap.add_argument("--total-steps", type=int, default=c("total_steps", 30_000))
    ap.add_argument("--batch-size", type=int, default=c("batch_size", 4096))
    ap.add_argument("--decoder-init-norm", type=float, default=c("decoder_init_norm", 0.1))
    ap.add_argument("--dead-window", type=int, default=c("dead_window", 2_000_000),
                    help="tokens without firing before a latent is 'dead'")
    ap.add_argument("--holdout-rows", type=int, default=c("holdout_rows", 50_000))
    ap.add_argument("--train-block-rows", type=int, default=c("train_block_rows", 32_768))
    ap.add_argument("--train-gpu", type=int, default=c("train_gpu", None),
                    help="physical GPU id for SAE training; sets CUDA_VISIBLE_DEVICES before CUDA init")
    ap.add_argument("--log-every", type=int, default=c("log_every", 100))
    ap.add_argument("--ckpt-every", type=int, default=c("ckpt_every", 0),
                    help="save a persistent checkpoint every N steps; 0 disables")
    ap.add_argument("--output-dir", default=c("output_dir", str(paths.OUT_DIR)),
                    help="root directory for final run outputs")
    ap.add_argument("--checkpoint-dir", default=c("checkpoint_dir", None),
                    help="root directory for persistent checkpoints; default is output/<run>")
    ap.add_argument("--log-dir", default=c("log_dir", None),
                    help="directory for tee logs written by train_sae.py; null disables internal tee logging")
    ap.add_argument("--log-file", default=c("log_file", None),
                    help="optional explicit log file name/path; relative paths live under log_dir")
    ap.add_argument("--run-name", type=str, default=c("run_name", None))
    ap.add_argument("--wandb-project", default=c("wandb_project", "chem_sae"))
    ap.add_argument("--wandb-entity", default=c("wandb_entity", None))
    ap.add_argument("--wandb-run-name", default=c("wandb_run_name", None))
    ap.add_argument("--wandb", dest="no_wandb", action="store_false",
                    help="enable W&B logging even if config sets wandb: false")
    ap.add_argument("--no-wandb", dest="no_wandb", action="store_true",
                    default=not bool(c("wandb", True)),
                    help="disable W&B logging even if config sets wandb: true")
    ap.add_argument("--eval", dest="eval_enabled", action="store_true", default=bool(c("eval", False)),
                    help="enable periodic Delta LM loss eval on a second GPU")
    ap.add_argument("--no-eval", dest="eval_enabled", action="store_false",
                    help="disable periodic Delta LM loss eval")
    ap.add_argument("--eval-gpu", type=int, default=c("eval_gpu", 7),
                    help="physical GPU id for the persistent LM eval worker")
    ap.add_argument("--eval-per-step", type=int, default=c("eval_per_step", 500),
                    help="run Delta LM loss every N completed training steps")
    ap.add_argument("--eval-batch-size", type=int, default=c("eval_batch_size", 2),
                    help="text batch size for Delta LM loss eval worker")
    ap.add_argument("--eval-n-eval-seqs", type=int, default=c("eval_n_eval_seqs", 32),
                    help="number of fixed holdout texts used by Delta LM loss")
    ap.add_argument("--eval-max-length", type=int, default=c("eval_max_length", 128),
                    help="max token length for Delta LM loss text batches")
    ap.add_argument("--eval-model", default=c("eval_model", "chemdfm"),
                    help="model_config registry name for Delta LM loss")
    ap.add_argument("--eval-timeout-sec", type=int, default=c("eval_timeout_sec", 3600),
                    help="max seconds to wait for pending eval jobs at training end")
    ap.add_argument("--eval-startup-timeout-sec", type=int, default=c("eval_startup_timeout_sec", 600),
                    help="max seconds to wait for the persistent eval worker to become ready")
    ap.add_argument("--eval-request-timeout-sec", type=int, default=c("eval_request_timeout_sec", 1800),
                    help="max seconds to wait for one Delta LM eval request")
    ap.add_argument("--eval-poll-interval", type=float, default=c("eval_poll_interval", 5.0),
                    help="seconds between eval worker filesystem polls")
    ap.add_argument("--eval-keep-checkpoints", action="store_true", default=bool(c("eval_keep_checkpoints", False)),
                    help="keep intermediate eval checkpoints instead of deleting after eval result")
    ap.add_argument("--eval-texts-path", default=c("eval_texts_path", None),
                    help="fixed JSONL holdout for Delta LM eval; defaults to paths.EVAL_TEXTS_PATH")
    ap.add_argument("--overwrite", action="store_true", default=bool(c("overwrite", False)),
                    help="allow writing into an existing run directory with training_meta.json")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run for validation")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    setup_file_logging(args.log_dir, args.log_file, args.run_name)

    if args.train_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.train_gpu)
        print(f"[train] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (train_gpu)")

    if isinstance(args.matryoshka_widths, str) and args.matryoshka_widths.strip():
        args.matryoshka_widths = [int(x) for x in args.matryoshka_widths.split(",") if x.strip()]
    elif not args.matryoshka_widths:
        args.matryoshka_widths = None

    if args.sparsemax_key_dim is not None and args.sparsemax_key_dim <= 0:
        args.sparsemax_key_dim = None
    if args.sparsemax_preselect_k is not None and args.sparsemax_preselect_k <= 0:
        args.sparsemax_preselect_k = None
    if args.sparsemax_l0_target is not None and args.sparsemax_l0_target <= 0:
        args.sparsemax_l0_target = None

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
        args.eval_enabled = False
        if args.arch == "sparsemax_attention":
            args.sparsemax_key_dim = args.sparsemax_key_dim or 256
            args.sparsemax_preselect_k = min(int(args.sparsemax_preselect_k or 512), 512)

    train(
        layer=args.layer,
        arch=args.arch,
        expansion=args.expansion,
        d_sae=args.d_sae,
        k=args.k,
        jumprelu_l0_coefficient=args.jumprelu_l0_coefficient,
        jumprelu_l0_warm_up_steps=args.jumprelu_l0_warm_up_steps,
        matryoshka_widths=args.matryoshka_widths,
        sparsemax_key_dim=args.sparsemax_key_dim,
        sparsemax_preselect_k=args.sparsemax_preselect_k,
        sparsemax_activation_mode=args.sparsemax_activation_mode,
        sparsemax_use_input_norm=args.sparsemax_use_input_norm,
        sparsemax_use_idf_mask=args.sparsemax_use_idf_mask,
        sparsemax_idf_threshold=args.sparsemax_idf_threshold,
        sparsemax_mse_loss_scale=args.sparsemax_mse_loss_scale,
        sparsemax_score_scale=args.sparsemax_score_scale,
        sparsemax_l0_target=args.sparsemax_l0_target,
        sparsemax_l0_coefficient=args.sparsemax_l0_coefficient,
        sparsemax_cosine_loss_coefficient=args.sparsemax_cosine_loss_coefficient,
        sparsemax_norm_loss_coefficient=args.sparsemax_norm_loss_coefficient,
        sparsemax_value_scale_init=args.sparsemax_value_scale_init,
        sparsemax_global_output_scale_init=args.sparsemax_global_output_scale_init,
        sparsemax_init_b_dec_from_data=args.sparsemax_init_b_dec_from_data,
        sparsemax_b_dec_init_rows=args.sparsemax_b_dec_init_rows,
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
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        log_to_wandb=not args.no_wandb,
        eval_enabled=args.eval_enabled,
        eval_gpu=args.eval_gpu,
        eval_per_step=args.eval_per_step,
        eval_batch_size=args.eval_batch_size,
        eval_n_eval_seqs=args.eval_n_eval_seqs,
        eval_max_length=args.eval_max_length,
        eval_model=args.eval_model,
        eval_timeout_sec=args.eval_timeout_sec,
        eval_startup_timeout_sec=args.eval_startup_timeout_sec,
        eval_request_timeout_sec=args.eval_request_timeout_sec,
        eval_poll_interval=args.eval_poll_interval,
        eval_keep_checkpoints=args.eval_keep_checkpoints,
        eval_texts_path=args.eval_texts_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
