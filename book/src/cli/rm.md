<!-- PCU: cli-rm | contract: contracts/apr-page-cli-rm-v1.yaml -->

# apr rm

Remove model from cache

**Category**: Registry & Resources

## Synopsis

```text
apr rm [OPTIONS]
```

## Example

```bash
apr rm <model-id>          # from \`apr list\` output
```

## What this does

`apr rm` deletes a cached model from `~/.cache/aprender/`. It's the cleanup
counterpart to `apr pull` — useful when disk is tight, when a corrupted
download needs to be wiped, or when you want a fresh fetch to verify
reproducibility. The model ID is exactly what `apr list` shows in the first
column.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--json` | Machine-readable confirmation | `--json` |
| `-v, --verbose` | Show byte count + path freed | `--verbose` |
| `-q, --quiet` | Silent success (errors only) | `--quiet` |

## Common workflows

**Wipe a corrupted download and re-pull.**

```bash
apr rm qwen2.5-coder-1.5b
apr pull qwen2.5-coder-1.5b
apr validate ~/.cache/aprender/models/qwen2.5-coder-1.5b.apr
```

**Bulk-clean stale models.**

```bash
apr list --json | jq -r '.[] | select(.last_used_days_ago > 60) | .name' | \
    xargs -I{} apr rm {} --verbose
```

## Troubleshooting

- **"model not found"** — the ID must match `apr list` exactly. Names are
  case-sensitive and include the format suffix (e.g. `qwen2.5-coder-1.5b.apr`).
- **`apr rm` doesn't free disk** — Linux page cache. Run `sync; echo 3 > 
  /proc/sys/vm/drop_caches` (root) or just wait; the disk really is free.
- **Removed model still appears in `apr list`** — stale index. Run any
  `apr pull` or `apr inspect`; the index is rebuilt on next use.

## See also

- Source: [`crates/apr-cli/src/commands/rm.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/rm.rs)
- Contract: [`contracts/apr-page-cli-rm-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-rm-v1.yaml)

