<!-- PCU: cli-compare-hf | contract: contracts/apr-page-cli-compare-hf-v1.yaml -->

# apr compare-hf

Compare APR model against HuggingFace source

**Category**: Quality & Evaluation

## Synopsis

```text
apr compare-hf [OPTIONS]
```

## Example

```bash
apr compare-hf model.apr --hf-repo Qwen/Qwen2.5-Coder-1.5B-Instruct
```

## What this does

`apr compare-hf` downloads the original SafeTensors from a HuggingFace repo and
runs an element-wise diff against your local APR conversion. It's the gold
standard for "did my converter actually preserve the weights" — if the maximum
absolute difference exceeds `--threshold` (default 1e-5), the converter has a
bug. Use this as the discharge gate for any new arch in the import pipeline.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--hf REPO` | HuggingFace repo ID | `--hf Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| `--tensor PAT` | Filter to one tensor pattern | `--tensor "lm_head"` |
| `--threshold N` | Max allowed abs diff (default 1e-5) | `--threshold 1e-4` |
| `--json` | JSON output with per-tensor diff stats | `--json` |
| `-v, --verbose` | Print every tensor (not just failures) | `--verbose` |

## Common workflows

**Verify a fresh import.**

```bash
apr import hf://Qwen/Qwen2.5-Coder-1.5B-Instruct -o qwen.apr
apr compare-hf qwen.apr --hf Qwen/Qwen2.5-Coder-1.5B-Instruct --json | \
    jq '.tensors[] | select(.max_abs_diff > 1e-5)'
```

**Drill into the LM head specifically (where transpose bugs surface).**

```bash
apr compare-hf qwen.apr --hf Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --tensor "lm_head" --threshold 1e-6 --verbose
```

## Troubleshooting

- **Large diff on `lm_head` only** — classic LAYOUT-001 bug. The converter
  forgot to transpose. Fix in `crates/aprender-core/src/format/converter/`.
- **All tensors diverge equally** — quantization mismatch. APR Q4K vs HF F16
  will always diverge; compare F32 APR against HF SafeTensors instead.
- **`--hf` download is slow** — set `HF_HUB_ENABLE_HF_TRANSFER=1` to use the
  Rust-native downloader. Auth via `HF_TOKEN` for gated models.

## See also

- Source: [`crates/apr-cli/src/commands/compare_hf.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/compare_hf.rs)
- Contract: [`contracts/apr-page-cli-compare-hf-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-compare-hf-v1.yaml)

