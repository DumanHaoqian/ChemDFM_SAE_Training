# Stage 4 — SAE Training

Train Sparse Autoencoders on the ChemDFM residual-stream activations dumped by
Stage 3. We use **SAELens' SAE implementations** (BatchTopK is the main line,
design §2.2) inside a **custom training loop** that reads the Stage-3 disk cache.

Why a custom loop instead of `LanguageModelSAETrainingRunner`? SAELens' runner
drives a TransformerLens `HookedTransformer` and re-runs the LM to stream
activations. Forcing a 14B custom Qwen fine-tune through TransformerLens (and
re-running it every step) is fragile and slow; caching activations once (Stage
3) and training directly on them is cheaper and keeps full control over the
chemistry tokenizer, chat template, and layer sweep. We still use the genuine
SAELens `BatchTopKTrainingSAE` module, so the trained SAE **saves as a JumpReLU
inference model** (consistent across batch sizes — friendly to the detector
stage) exactly as the design intends.

- **Env:** `source /home/haoqian/Data/SAERAG/venvs/chemdfm/bin/activate`
- **Architectures (`--arch`):** `batchtopk` (default) · `topk` · `jumprelu`
  (baseline) · `matryoshka` (Matryoshka-BatchTopK, nested prefixes).
- **Defaults (§2.2):** expansion 16× (d_sae = 81,920), k (L0) = 32, lr 1e-4 with
  warmup + linear decay, unit-ish decoder init (`decoder_init_norm=0.1`),
  dead-feature revival via the TopK auxiliary loss.

## Files

```
data.py           memmap Stage-3 shards -> shuffled activation mini-batches
train_sae.py      build SAELens SAE + custom training loop + save (JumpReLU)
eval_sae.py       reconstruction metrics (FVU/L0/dead%) + Delta LM loss patching
run_layer_sweep.py train+eval one SAE per cached layer -> sweep_summary.json
```

## Train

```bash
cd /home/haoqian/Data/Graph/SAERAG_Stage4

# Config-first main line. configs/train.yaml defaults to train_gpu=6, eval=true,
# eval_gpu=7, eval_per_step=500, and eval_batch_size=2. CLI flags override.
python train_sae.py --config configs/train.yaml

# Example override for a short run while keeping the config defaults.
python train_sae.py --config configs/train.yaml --total-steps 1000 --run-name debug_1000

# tiny end-to-end validation; smoke disables periodic Delta LM eval.
python train_sae.py --smoke --layer 26 --no-wandb
```

Output (`output/<run_name>/`): `sae_weights.safetensors` + `cfg.json`
(JumpReLU inference model, load with `sae_lens.SAE.load_from_disk`) and
`training_meta.json` (layer, arch, k, d_sae, **`input_scale`**, final metrics).
The local Sparsemax Attention SAE writes `sparsemax_attention_sae.safetensors`
when `safetensors` is installed, and keeps backward-compatible `.pt` loading for
older runs.

By default, `train_sae.py` refuses to overwrite an existing run directory that
already has `training_meta.json`; pass `--overwrite` only for intentional reruns.
Logs live under `logs/`, while `output/`, `checkpoints/`, `wandb/`, `logs/`,
`.nfs*`, and platform sidecars are ignored by Git.

> The SAE is trained on activations pre-scaled by `input_scale`. To encode
> **raw** ChemDFM activations downstream, multiply by `input_scale` first
> (recorded in `training_meta.json`).

## Evaluate (§2.5)

```bash
# cheap: reconstruction + sparsity on held-out activations (no LM)
python eval_sae.py --run output/chemdfm_L26_batchtopk_x16_k32

# + Delta LM loss / CE-loss recovery (loads the 14B LM; patches recon into resid @ L)
python eval_sae.py --run output/chemdfm_L26_batchtopk_x16_k32 \
    --delta-lm-loss --n-eval-seqs 32 --max-length 128 --delta-batch-size 1
```

Reports **FVU / explained variance**, **L0**, **dead & dense feature fractions**,
activation-frequency histogram, and **Delta LM loss** (`ce_clean` vs `ce_recon`
vs `ce_zero`, `delta_lm_loss`, and fraction of CE loss recovered). Written to
`output/<run>/eval.json`.

Delta LM loss now requires a fixed held-out JSONL at
`/home/haoqian/Data/Graph/SAERAG_Stage3/data/corpus/sae_eval_corpus.jsonl`
unless `--eval-texts-path` / `eval_texts_path` is provided. The eval report logs
the file path, number of texts, policy, and SHA256 digest of the selected text
slice. This avoids treating the first rows of the training corpus as a formal
holdout metric.

### Metric: Delta LM loss

Delta LM loss is the primary SAE fidelity metric: splice the SAE into the LM
forward pass, replace the target layer's `resid_post` hidden states with SAE
reconstruction, and measure the next-token CE increase:

```text
delta_lm_loss = ce_recon - ce_clean
ce_loss_recovered = 1 - (ce_recon - ce_clean) / (ce_zero - ce_clean)
```

A small `delta_lm_loss` means reconstruction error has little causal effect on
the LM's final predictions. This is stronger than reconstruction MSE/FVU alone
because it evaluates behavioural fidelity, not just numerical closeness.

Use `--wandb` during eval to log `eval/delta_lm_loss`, `eval/ce_clean`,
`eval/ce_recon`, `eval/ce_zero`, and `eval/ce_loss_recovered`.

The persistent periodic eval worker writes `READY.json`, `HEARTBEAT`, and
`FATAL.json` under `output/<run>/delta_lm_eval/`. Training fails fast if the
worker cannot load, exits unexpectedly, or a request exceeds
`eval_request_timeout_sec`.

The *ultimate* selection metric is downstream RAGLens detection AUC (a later
stage), not MSE; this stage produces the Pareto inputs (L0 vs FVU / Delta LM loss).

## Layer sweep + baselines

```bash
# train + recon-eval a SAE per cached layer
python run_layer_sweep.py --layers 12 18 24 26 30 --arch batchtopk --no-wandb
# -> output/sweep_summary.json  (pick layer later by detection AUC)
```

Suggested comparisons: `--arch matryoshka` (feature absorption/splitting),
`--arch jumprelu` (Gemma-Scope-style baseline), and an off-the-shelf Qwen2.5 SAE
ablation to demonstrate the domain SAE is necessary.

## Paths

Default A6000 paths can be overridden without editing code:

```bash
export SAERAG_GRAPH_DIR=/home/haoqian/Data/Graph
export SAERAG_STAGE3_DIR=/home/haoqian/Data/Graph/SAERAG_Stage3
export SAERAG_STAGE4_DIR=/home/haoqian/Data/Graph/SAERAG_Stage4
export CHEMDFM_MODEL_PATH=/home/haoqian/Data/Graph/ChemDFM-v2.0-14B
```

## Validation performed

Smoke: BatchTopK x8, k=32, 200 steps on the layer-26 smoke cache trained
end-to-end (FVU → ~1e-3 on the tiny set), saved & reloaded as `JumpReLUSAE`,
`eval_sae.py` recon + delta-CE both ran (delta-CE `ce_clean=2.27`,
`ce_recon=14.3`, recovered≈0.25 — expected low for a 3k-token smoke SAE). The
large smoke weight file was deleted; `training_meta.json`/`eval.json` kept.
```
