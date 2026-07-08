#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared Delta LM loss helpers for CLI eval and the persistent worker."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

import paths


def resolve_eval_texts_path(eval_texts_path: str | Path | None = None) -> Path:
    return Path(eval_texts_path).expanduser() if eval_texts_path else paths.EVAL_TEXTS_PATH


def load_eval_texts(
    eval_texts_path: str | Path | None,
    n_eval_seqs: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """Load a fixed eval JSONL file and return selected texts plus provenance."""
    path = resolve_eval_texts_path(eval_texts_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fixed Delta LM eval text file not found: {path}. "
            "Create this holdout file before running Delta LM loss, or pass --eval-texts-path."
        )

    texts: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "text" not in item:
                raise ValueError(f"{path}:{line_no} is missing required key 'text'")
            texts.append(str(item["text"]))
            if len(texts) >= int(n_eval_seqs):
                break
    if not texts:
        raise ValueError(f"No eval texts found in {path}")

    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    provenance = {
        "eval_texts_path": str(path),
        "eval_texts_source": "fixed_jsonl",
        "eval_texts_policy": "first_n_from_fixed_file",
        "eval_texts_seed": None,
        "eval_texts_sha256": digest.hexdigest(),
        "n_eval_texts": len(texts),
        "n_eval_seqs_requested": int(n_eval_seqs),
    }
    return texts, provenance


@torch.no_grad()
def delta_lm_loss_with_loaded_lm(
    sae,
    input_scale: float,
    layer: int,
    hk,
    texts: List[str],
    max_length: int,
    batch_size: int,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Measure CE degradation from replacing a layer output with SAE reconstruction."""
    target = hk.layer_module(layer)

    def ce_for(mode: str) -> float:
        total_loss, total_tok = 0.0, 0
        handle = None

        def hook(_module, _inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if mode == "zero":
                new = torch.zeros_like(hs)
            elif mode == "recon":
                x_scaled = hs.to(torch.float32) * float(input_scale)
                recon_scaled = sae.decode(sae.encode(x_scaled))
                new = (recon_scaled / float(input_scale)).to(hs.dtype)
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
