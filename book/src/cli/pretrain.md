<!-- PCU: cli-pretrain | contract: contracts/apr-page-cli-pretrain-v1.yaml -->

# apr pretrain

Pretraining loop driver (SHIP-TWO-001 MODEL-2).

**Category**: Training

## Synopsis

```text
apr pretrain [OPTIONS]
```

## Example

```bash
apr pretrain config.yaml
```

## What this does

`apr pretrain` drives a from-scratch pretraining loop per
`contracts/training-loop-pretrain-v1.yaml` — the SHIP-TWO-001 MODEL-2 workflow.
Unlike `apr finetune` (small LoRA adapter on a frozen base), pretrain updates
every parameter from a random init or a checkpoint. Defaults atomically flip
between `finetune` mode (post-divergence remedy, lr=5e-5, warmup=100) and
`from-scratch` mode (lr=3e-4, warmup=1000) so you can't accidentally apply
finetune LRs to a cold start.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--dataset PATH` | Tokenized shard index or raw corpus | `--dataset ./data/shards.json` |
| `--tokenizer DIR` | `vocab.json` + `merges.txt` | `--tokenizer ./tok/` |
| `--run-dir DIR` | Output dir for ckpts + metadata | `--run-dir ./runs/v1/` |
| `--mode M` | `finetune` or `from-scratch` (default `finetune`) | `--mode from-scratch` |
| `--lr LR` | Peak learning rate (overrides mode default) | `--lr 3e-4` |
| `--num-steps N` | Warmup + cosine decay total | `--num-steps 100000` |
| `--warmup-steps N` | Warmup steps (overrides mode default) | `--warmup-steps 1000` |
| `--batch-size N` | Micro-batch size (default 16) | `--batch-size 32` |
| `--seq-length N` | Tokens per example | `--seq-length 2048` |

## Common workflows

**From-scratch 370M cold start (MODEL-2 lane).**

```bash
apr pretrain --mode from-scratch \
    --dataset ./data/csn-python-shards.json \
    --tokenizer ./tokenizers/csn-bpe-50257/ \
    --run-dir ./runs/csn-370m-cold/ \
    --num-steps 50000 --batch-size 32 --seq-length 2048
```

**Continual pretraining (warm start from a checkpoint).**

```bash
apr pretrain --mode finetune \
    --dataset ./data/domain-shards.json \
    --tokenizer ./tokenizers/qwen-bpe/ \
    --run-dir ./runs/qwen-continual/ \
    --lr 5e-5 --warmup-steps 100 --num-steps 5000
```

## Troubleshooting

- **val_loss plateaus at the Chinchilla floor** — you're undertrained for the
  model size. Add more tokens (10-20 tok/param is the Chinchilla optimum) or
  reduce the model. See [a-priori theoretical falsification](https://github.com/paiml/aprender/issues/700).
- **NaN loss at step 1** — usually a learning-rate-too-high crash. Drop `--lr`
  by 5x, or rely on the mode default.
- **Throughput collapses after 500 steps** — memory fragmentation. Check
  `apr gpu --verbose` for VRAM growth; restart with `apr train watch`.

## See also

- Source: [`crates/apr-cli/src/commands/pretrain.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/pretrain.rs)
- Contract: [`contracts/apr-page-cli-pretrain-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-pretrain-v1.yaml)

