<!-- PCU: cli-validate | contract: contracts/apr-page-cli-validate-v1.yaml -->

# apr validate

Validate model integrity and quality

**Category**: Inspection

## Synopsis

```text
apr validate [OPTIONS]
```

## Example

```bash
apr validate qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --quality
```

## What this does

`apr validate` verifies a model file's structural integrity (magic bytes, header,
tensor offsets, shape contract) and optionally emits a 100-point quality score.
Use it as a CI gate — it returns non-zero on any error and, with `--strict`, on
any warning. Where `apr inspect --quality` summarizes the score, `apr validate
--quality` blocks the pipeline when the score is below threshold.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--quality` | Compute 100-point quality assessment | `--quality` |
| `--strict` | Treat warnings as errors | `--strict` |
| `--min-score N` | Fail if quality score < N | `--min-score 90` |
| `--json` | JSON output for CI parsing | `--json` |

## Common workflows

**CI gate before pushing to crates.io / HF.**

```bash
apr validate qwen2.5-coder-1.5b.apr --quality --min-score 90 --strict
# exit 0 = ship; non-zero = block release
```

**Audit every model in your cache.**

```bash
for m in ~/.cache/aprender/models/*.apr; do
    echo "=== $m ==="
    apr validate "$m" --quality --json | jq '{score, errors: .errors|length}'
done
```

## Troubleshooting

- **`magic bytes mismatch`** — file is corrupted or not an APR/GGUF file. Re-pull
  with `apr pull` or verify the SHA256 against the source.
- **Score lower than `apr inspect --quality` reported** — `validate` runs full
  tensor-layout contracts, not just metadata. The two diverge when LAYOUT-001/002
  catches a row-major / column-major mismatch.
- **`tensor offset out of bounds`** — partial download. Delete from cache
  (`apr rm <model>`) and re-pull.

## See also

- Source: [`crates/apr-cli/src/commands/validate.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/validate.rs)
- Contract: [`contracts/apr-page-cli-validate-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-validate-v1.yaml)

