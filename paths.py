#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Central path constants for Stage 4 (SAE training) — A6000 deployment.

The 5090 is unreachable from the A6000; the real 113.4M-token layer-26 dump lives
on the shared NAS under /home/haoqian/Data/Graph. These paths point there.
"""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("SAERAG_GRAPH_DIR", "/home/haoqian/Data/Graph")).expanduser()
STAGE3_DIR = Path(
    os.environ.get("SAERAG_STAGE3_DIR", str(PROJECT_DIR / "SAERAG_Stage3"))
).expanduser()
ACTS_DIR = STAGE3_DIR / "data" / "acts"
INFER_HOOK_DIR = STAGE3_DIR
STAGE1_DIR = STAGE3_DIR
MODEL_PATH = os.environ.get("CHEMDFM_MODEL_PATH", str(PROJECT_DIR / "ChemDFM-v2.0-14B"))
D_MODEL = 5120
N_LAYERS = 48
MAIN_SAE_LAYER = 26
STAGE4_DIR = Path(
    os.environ.get("SAERAG_STAGE4_DIR", str(PROJECT_DIR / "SAERAG_Stage4"))
).expanduser()
OUT_DIR = STAGE4_DIR / "output"
CONFIG_DIR = STAGE4_DIR / "configs"
LOG_DIR = STAGE4_DIR / "logs"
EVAL_TEXTS_PATH = STAGE3_DIR / "data" / "corpus" / "sae_eval_corpus.jsonl"

def ensure_dirs() -> None:
    for d in (OUT_DIR, CONFIG_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)

def layer_acts_dir(layer: int) -> Path:
    return ACTS_DIR / f"layer_{layer:02d}"

def run_dir(run_name: str) -> Path:
    return OUT_DIR / run_name
