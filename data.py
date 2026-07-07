#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load Stage-3 sharded activations for SAE training.

Stage 3 writes, per layer, a set of ``shard_*.f16.npy`` files of shape
``[rows, d_model]`` plus a ``meta.json``. This module memmaps those shards and
serves mini-batches of activation rows. For NAS-backed shards, train batches are
read as shuffled contiguous blocks instead of global random point reads; this
keeps I/O mostly sequential while still shuffling block order and rows in block.

Activations are optionally rescaled by ``meta["input_scale"]`` so the SAE trains
on well-conditioned inputs (mean L2 norm ~= sqrt(d_model)); the scale is
recorded so downstream code can reproduce it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch


class ActivationStore:
    def __init__(
        self,
        layer_dir: Path,
        device: str = "cuda",
        apply_input_scale: bool = True,
        dtype: torch.dtype = torch.float32,
        holdout_rows: int = 50_000,
        seed: int = 0,
        train_block_rows: int = 32_768,
    ):
        self.layer_dir = Path(layer_dir)
        self.device = device
        self.dtype = dtype
        meta_path = self.layer_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No meta.json in {layer_dir}. Run Stage 3 dump_activations.py first.")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.d_model = int(self.meta["d_model"])
        self.input_scale = float(self.meta.get("input_scale", 1.0)) if apply_input_scale else 1.0

        self.shards: List[np.memmap] = []
        self.shard_sizes: List[int] = []
        n_shards = int(self.meta["n_shards"])
        for i in range(n_shards):
            arr = np.load(self.layer_dir / f"shard_{i:05d}.f16.npy", mmap_mode="r")
            self.shards.append(arr)
            self.shard_sizes.append(int(arr.shape[0]))
        self.total_rows = int(sum(self.shard_sizes))
        self.shard_offsets = np.cumsum([0] + self.shard_sizes, dtype=np.int64)

        self._rng = np.random.default_rng(seed)
        holdout = min(int(holdout_rows), max(0, self.total_rows // 5))
        self.eval_rows = holdout
        self.train_rows = self.total_rows - holdout
        self.eval_start = self.train_rows

        self.train_block_rows = max(1, int(train_block_rows))
        self.train_blocks: List[Tuple[int, int, int]] = self._build_train_blocks(
            self.train_block_rows)
        if not self.train_blocks:
            raise ValueError(f"No train blocks available in {layer_dir}")
        self._block_order = np.empty(0, dtype=np.int64)
        self._block_cursor = 0
        self._batch_buffer: Optional[np.ndarray] = None
        self._batch_cursor = 0
        self._reshuffle_blocks()

    # ------------------------------------------------------------------
    def _to_tensor(self, out: np.ndarray) -> torch.Tensor:
        out = np.ascontiguousarray(out, dtype=np.float32)
        t = torch.from_numpy(out).to(self.device, self.dtype)
        if self.input_scale != 1.0:
            t = t * self.input_scale
        return t

    def _build_train_blocks(self, block_rows: int) -> List[Tuple[int, int, int]]:
        blocks: List[Tuple[int, int, int]] = []
        for s, _ in enumerate(self.shards):
            shard_start = int(self.shard_offsets[s])
            shard_end = int(self.shard_offsets[s + 1])
            local_train_end = max(0, min(shard_end, self.train_rows) - shard_start)
            for start in range(0, local_train_end, block_rows):
                end = min(start + block_rows, local_train_end)
                if end > start:
                    blocks.append((s, start, end))
        return blocks

    def _reshuffle_blocks(self) -> None:
        self._block_order = self._rng.permutation(len(self.train_blocks))
        self._block_cursor = 0

    def _load_next_train_block(self) -> None:
        if self._block_cursor >= len(self._block_order):
            self._reshuffle_blocks()
        block_id = int(self._block_order[self._block_cursor])
        self._block_cursor += 1

        s, start, end = self.train_blocks[block_id]
        out = np.asarray(self.shards[s][start:end], dtype=np.float32)
        if out.shape[0] > 1:
            out = out[self._rng.permutation(out.shape[0])]
        self._batch_buffer = np.ascontiguousarray(out)
        self._batch_cursor = 0

    def _gather(self, global_indices: np.ndarray) -> torch.Tensor:
        """Fetch rows at the given global indices from the memmapped shards."""
        global_indices = np.asarray(global_indices, dtype=np.int64)
        shard_of = np.searchsorted(self.shard_offsets, global_indices, side="right") - 1
        out = np.empty((len(global_indices), self.d_model), dtype=np.float32)
        for s in np.unique(shard_of):
            mask = shard_of == s
            local = global_indices[mask] - self.shard_offsets[s]
            out[mask] = np.asarray(self.shards[int(s)][local], dtype=np.float32)
        return self._to_tensor(out)

    def next_train_batch(self, batch_size: int) -> torch.Tensor:
        """Return a shuffled train batch from sequentially-read shard blocks."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            return torch.empty((0, self.d_model), device=self.device, dtype=self.dtype)

        chunks = []
        remaining = batch_size
        while remaining > 0:
            if self._batch_buffer is None or self._batch_cursor >= self._batch_buffer.shape[0]:
                self._load_next_train_block()
            assert self._batch_buffer is not None
            available = self._batch_buffer.shape[0] - self._batch_cursor
            take = min(remaining, available)
            chunks.append(self._batch_buffer[self._batch_cursor:self._batch_cursor + take])
            self._batch_cursor += take
            remaining -= take

        out = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
        if len(chunks) > 1 and out.shape[0] > 1:
            out = out[self._rng.permutation(out.shape[0])]
        return self._to_tensor(out)

    def sample_train_rows(self, n: int) -> torch.Tensor:
        """NAS-friendly random-ish subset for data stats."""
        return self.sample_train_blocks(n)

    def sample_train_blocks(self, n: int, block_rows: int = 1024) -> torch.Tensor:
        """Sample rows by reading contiguous blocks from train shard ranges."""
        n = min(int(n), self.train_rows)
        if n <= 0:
            return torch.empty((0, self.d_model), device=self.device, dtype=self.dtype)

        block_rows = max(1, min(int(block_rows), n))
        train_sizes = np.maximum(
            0,
            np.minimum(self.shard_offsets[1:], self.train_rows) - self.shard_offsets[:-1],
        ).astype(np.int64)
        eligible = np.flatnonzero(train_sizes > 0)
        probs = train_sizes[eligible] / train_sizes[eligible].sum()

        chunks = []
        remaining = n
        while remaining > 0:
            s = int(self._rng.choice(eligible, p=probs))
            take = min(block_rows, remaining, int(train_sizes[s]))
            high = int(train_sizes[s]) - take + 1
            start = int(self._rng.integers(0, high)) if high > 1 else 0
            chunks.append(np.asarray(self.shards[s][start:start + take], dtype=np.float32))
            remaining -= take
        out = np.concatenate(chunks, axis=0)
        if out.shape[0] > 1:
            out = out[self._rng.permutation(out.shape[0])]
        return self._to_tensor(out)

    def iter_train_batches(self, batch_size: int, n_steps: int) -> Iterator[torch.Tensor]:
        for _ in range(n_steps):
            yield self.next_train_batch(batch_size)

    def eval_batches(self, batch_size: int, max_batches: Optional[int] = None) -> Iterator[torch.Tensor]:
        b = 0
        for start in range(0, self.eval_rows, batch_size):
            if max_batches is not None and b >= max_batches:
                break
            end = min(start + batch_size, self.eval_rows)
            if end <= start:
                break
            idx = np.arange(self.eval_start + start, self.eval_start + end, dtype=np.int64)
            yield self._gather(idx)
            b += 1

    @property
    def n_train_rows(self) -> int:
        return int(self.train_rows)

    @property
    def n_eval_rows(self) -> int:
        return int(self.eval_rows)
