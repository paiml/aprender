<!-- PCU: cli-import | contract: contracts/apr-page-cli-import-v1.yaml -->

# apr import

Import from external formats (hf://org/repo, local files, URLs)

**Category**: Model Transform

## Synopsis

```text
apr import [OPTIONS]
```

## Example

```bash
apr import hf://openai/whisper-tiny -o whisper.apr --arch whisper
```

## What this does

`apr import` is the all-in-one inbound pipeline: download from a `hf://` URL
(authenticated via `HF_TOKEN` for gated repos), parse SafeTensors / GGUF / URL
inputs, apply the LAYOUT-001/002 transpose, infer or accept the architecture,
optionally preserve Q4K, and write a `.apr`. Unlike `apr convert` (which works
on a local file), `apr import` understands `hf://` and provenance.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--arch ARCH` | `llama`, `qwen2`, `qwen3`, `whisper`, `auto`, ... | `--arch qwen2` |
| `-o, --output PATH` | Output `.apr` path | `-o qwen.apr` |
| `--quantize FMT` | Quantize during import | `--quantize int4` |
| `--preserve-q4k` | Keep GGUF Q4K quantization | `--preserve-q4k` |
| `--strict` | Reject unverified archs / fail on warnings | `--strict` |
| `--enforce-provenance` | Reject pre-baked GGUF (only SafeTensors) | `--enforce-provenance` |
| `--allow-no-config` | Allow import without `config.json` (risky) | `--allow-no-config` |

## Common workflows

**Standard HF -> APR import.**

```bash
HF_TOKEN=hf_xxx apr import hf://Qwen/Qwen2.5-Coder-1.5B-Instruct \
    -o qwen2.5-coder-1.5b.apr --arch qwen2
apr inspect qwen2.5-coder-1.5b.apr --quality
```

**Single-provenance ship pipeline (rejects pre-baked GGUF).**

```bash
apr import hf://Qwen/Qwen2.5-Coder-7B-Instruct -o qwen-7b.apr \
    --arch qwen2 --enforce-provenance --strict
```

## Troubleshooting

- **"refusing to import without config.json"** — without `config.json`,
  hyperparameters like `rope_theta` are inferred from tensor shapes and may
  be wrong (GH-223). Either upload a `config.json` or, only if you've
  verified the inference, pass `--allow-no-config`.
- **HF download 401 / 403** — set `HF_TOKEN` (it's already in our environment
  per the memory note). Gated repos require accepting the license on the HF
  website first.
- **"unknown architecture"** — pass `--arch` explicitly; `auto` only works on
  well-known archs (llama, qwen2/3, whisper, gpt2, ...).

## See also

- Source: [`crates/apr-cli/src/commands/import.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/import.rs)
- Contract: [`contracts/apr-page-cli-import-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-import-v1.yaml)

