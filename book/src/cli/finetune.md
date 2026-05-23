<!-- PCU: cli-finetune | contract: contracts/apr-page-cli-finetune-v1.yaml -->

# apr finetune

Fine-tune model with LoRA/QLoRA (GH-244)

**Category**: Training

## Synopsis

```text
apr finetune [OPTIONS]
```

## Example

```bash
apr finetune qwen2.5-coder-0.5b --data train.jsonl --epochs 3
```

## What this does

`apr finetune` runs LoRA / QLoRA / full-finetune on a base model with a JSONL
training set. Method is `auto` by default — it inspects available VRAM and
model size and picks the right configuration. With `--quantize-nf4` it
activates QLoRA (frozen-weight 4-bit + LoRA adapters, ~8x less VRAM). Multi-GPU
data-parallel works via `--gpus 0,1`. Output is a PEFT-compatible adapter
directory (or merged model when `--merge` is set).

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `-m, --method M` | `auto`, `full`, `lora`, `qlora` | `--method qlora` |
| `-d, --data FILE` | Training JSONL | `--data train.jsonl` |
| `-o, --output PATH` | Adapter dir or merged model | `-o ./adapter/` |
| `-r, --rank N` | LoRA rank (auto-selected by default) | `--rank 16` |
| `--epochs N` | Epoch count (default 3) | `--epochs 5` |
| `--learning-rate LR` | Learning rate (default 2e-4) | `--learning-rate 5e-5` |
| `--quantize-nf4` | Activate QLoRA (4-bit frozen base) | `--quantize-nf4` |
| `--gpus LIST` | Data-parallel device list | `--gpus 0,1` |
| `--merge` | Merge adapter into base when done | `--merge` |

## Common workflows

**LoRA finetune Qwen2.5-Coder-0.5B on a custom JSONL.**

```bash
apr finetune qwen2.5-coder-0.5b.apr \
    --data ./data/train.jsonl --method lora --rank 8 \
    --epochs 3 -o ./qwen-ft-adapter/
apr eval ./qwen-ft-adapter/ --dataset wikitext-2
```

**QLoRA on a 7B model with a single 16GB GPU.**

```bash
apr finetune qwen2.5-coder-7b.apr \
    --data ./data/train.jsonl --method qlora --quantize-nf4 \
    --vram 16 --rank 32 -o ./qwen7b-qlora/
```

## Troubleshooting

- **CUDA OOM mid-epoch** — bump `--quantize-nf4` (QLoRA), or drop
  `--max-seq-len`, or reduce `--rank`. The auto-planner uses `--vram`; if you
  pass too high a value it'll try too aggressive a config.
- **Loss diverges to NaN** — drop `--learning-rate` by 5x. Default 2e-4 is
  for LoRA; QLoRA usually wants 1e-4.
- **Adapter doesn't load post-merge** — check `--checkpoint-format`
  (default `apr,safetensors`); the adapter dir needs both files for PEFT
  compatibility.

## See also

- Source: [`crates/apr-cli/src/commands/finetune.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/finetune.rs)
- Contract: [`contracts/apr-page-cli-finetune-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-finetune-v1.yaml)

