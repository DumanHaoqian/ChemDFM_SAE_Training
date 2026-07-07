#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Central path constants for Stage 4 (SAE training) — A6000 deployment.

The 5090 is unreachable from the A6000; the real 113.4M-token layer-26 dump lives
on the shared NAS under /home/haoqian/Data/Graph. These paths point there.
"""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_DIR = Path("/home/haoqian/Data/Graph")
STAGE3_DIR = PROJECT_DIR / "SAERAG_Stage3"
ACTS_DIR = STAGE3_DIR / "data" / "acts"
INFER_HOOK_DIR = STAGE3_DIR
STAGE1_DIR = STAGE3_DIR
MODEL_PATH = "/home/haoqian/Data/Graph/ChemDFM-v2.0-14B"
D_MODEL = 5120
N_LAYERS = 48
MAIN_SAE_LAYER = 26
STAGE4_DIR = PROJECT_DIR / "SAERAG_Stage4"
OUT_DIR = STAGE4_DIR / "output"
CONFIG_DIR = STAGE4_DIR / "configs"

def ensure_dirs() -> None:
    for d in (OUT_DIR, CONFIG_DIR):
        os.makedirs(d, exist_ok=True)

def layer_acts_dir(layer: int) -> Path:
    return ACTS_DIR / f"layer_{layer:02d}"

def run_dir(run_name: str) -> Path:
    return OUT_DIR / run_name
