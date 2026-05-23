<!-- PCU: cli-lint | contract: contracts/apr-page-cli-lint-v1.yaml -->

# apr lint

Check for best practices and conventions

**Category**: Inspection

## Synopsis

```text
apr lint [OPTIONS]
```

## Example

```bash
apr lint qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

## What this does

`apr lint` is to model files what clippy is to Rust — it flags conventions, style
issues, and footguns that aren't strictly invalid but should be fixed before
shipping. Examples: missing metadata fields, non-canonical tensor names, untagged
quantization formats, or vocabulary anomalies. Use it as a pre-publish gate that
complements `apr validate` (which checks integrity, not style).

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--json` | Machine-readable lint output | `--json` |
| `-v, --verbose` | Show rationale for each lint | `--verbose` |
| `--skip-contract` | Skip the tensor-layout contract | `--skip-contract` |
| `--offline` | Suppress all network access | `--offline` |

## Common workflows

**Pre-publish lint sweep.**

```bash
apr lint qwen2.5-coder-1.5b.apr --verbose
# Review warnings, fix metadata, re-run until clean.
```

**Lint every cached APR file.**

```bash
find ~/.cache/aprender/models -name "*.apr" -print0 | \
    xargs -0 -I{} apr lint {} --json
```

## Troubleshooting

- **"non-canonical tensor name detected"** — usually a stale converter. Re-run
  `apr convert` on the original source; the current writer normalizes
  `model.layers.N.*` to the canonical form.
- **"missing data_source / data_license"** — provenance fields are not strictly
  required but lint flags them. Use `apr stamp` to add them.
- **Lint hangs on a huge model** — the model is being mmap-walked; this is
  expected for 30B+ checkpoints. Add `--quiet` if you only care about exit code.

## See also

- Source: [`crates/apr-cli/src/commands/lint.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/lint.rs)
- Contract: [`contracts/apr-page-cli-lint-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-lint-v1.yaml)

