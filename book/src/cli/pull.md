<!-- PCU: cli-pull | contract: contracts/apr-page-cli-pull-v1.yaml -->

# apr pull

Download and cache model OR HuggingFace dataset (Ollama-like UX)

**Category**: Registry & Resources

## Synopsis

```text
apr pull [OPTIONS]
```

## Example

```bash
apr pull hf://Qwen/Qwen2.5-Coder-0.5B-Instruct
```

## What this does

`apr pull` is `ollama pull` for `apr` — give it a short alias (`qwen2.5-coder-1.5b`),
an `hf://` URI, or an `org/repo` ID, and it downloads + caches under
`~/.cache/aprender/`. It also pulls datasets when invoked as `apr pull dataset
<repo>`. Auth via `HF_TOKEN` (already in the environment per the project memory
note). Use `--revision` to pin to a tag / SHA for reproducibility.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--force` | Re-download even if cached | `--force` |
| `--dry-run` | Resolve to canonical URL, don't download | `--dry-run` |
| `--revision REV` | Pin to branch / tag / SHA | `--revision v1.0` |
| `--offline` | Forbid network I/O | `--offline` |
| `--include GLOB` | (dataset) shard glob filter | `--include "train-00*"` |
| `-o, --output DIR` | (dataset) output directory | `-o ./mydata/` |

## Common workflows

**Pull a model and verify integrity.**

```bash
apr pull qwen2.5-coder-1.5b
apr validate ~/.cache/aprender/models/qwen2.5-coder-1.5b.apr --quality
```

**Reproducibility-pin a model to a specific commit.**

```bash
apr pull hf://Qwen/Qwen2.5-Coder-1.5B-Instruct --revision a1b2c3d4
# Record the SHA in CI; future pulls with the same --revision are byte-identical
```

**Pull a dataset subset.**

```bash
apr pull dataset HuggingFaceH4/no_robots --include "train-*.parquet" -o ./data/
```

## Troubleshooting

- **HTTP 401 / 403** — set `HF_TOKEN`; for gated models accept the license on
  the HF website first.
- **"alias not in registry"** — short names live in `configs/aliases.yaml`.
  Use the full `hf://org/repo` URI, or `apr registry aliases` to see what's
  registered.
- **Partial / corrupted download** — `apr pull --force` re-fetches. If the
  underlying disk is full, `apr rm` old models first.

## See also

- Source: [`crates/apr-cli/src/commands/pull.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/pull.rs)
- Contract: [`contracts/apr-page-cli-pull-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-pull-v1.yaml)

