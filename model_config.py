#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""model_config.py — model-agnostic config + generic HF hooked-model wrapper.

Decouples Stage-4 from ChemDFM so the same training/eval code runs on any
HuggingFace AutoModelForCausalLM (Qwen2 / Llama / Gemma / GPT-2 ...). A
ModelConfig records residual-stream geometry (d_model, n_layers) and how to reach
a decoder block BY NAME; HFHookedModel loads the model and exposes
layer_module(L) for resid-stream patching in delta-CE eval.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch

import paths

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_path: str
    d_model: int
    n_layers: int
    default_sae_layer: int
    layer_module_fmt: str = "model.layers.{layer}"   # AutoModelForCausalLM Qwen2/Llama
    dtype: str = "bfloat16"


REGISTRY: Dict[str, ModelConfig] = {}


def register(cfg: ModelConfig) -> ModelConfig:
    REGISTRY[cfg.name] = cfg
    return cfg


register(ModelConfig(
    name="chemdfm",
    model_path=paths.MODEL_PATH,
    d_model=5120, n_layers=48, default_sae_layer=26,
    layer_module_fmt="model.layers.{layer}", dtype="bfloat16"))

# Templates for other architectures (fill model_path to enable):
# register(ModelConfig("qwen2.5-7b", "<path>", 3584, 28, 14, "model.layers.{layer}"))
# register(ModelConfig("llama3-8b",  "<path>", 4096, 32, 16, "model.layers.{layer}"))
# register(ModelConfig("gpt2",       "gpt2",    768, 12,  6, "transformer.h.{layer}"))


def get_model_config(name: str) -> ModelConfig:
    if name not in REGISTRY:
        raise KeyError("unknown model " + repr(name) + "; registered: " + str(list(REGISTRY)))
    return REGISTRY[name]


class HFHookedModel:
    """Generic HF AutoModelForCausalLM wrapper for residual-stream hooking.

    layer_module(L) resolves the decoder block by name (layer_module_fmt) so the
    same delta-CE patching works across architectures without ChemDFM-specific code.
    """

    def __init__(self, cfg: ModelConfig, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg = cfg
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, torch_dtype=_DTYPES.get(cfg.dtype, torch.bfloat16)).to(device).eval()

    def layer_module(self, layer: int):
        mod = self.model
        for part in self.cfg.layer_module_fmt.format(layer=layer).split("."):
            mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
        return mod
