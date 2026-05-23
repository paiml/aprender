<!-- PCU: cli-list | contract: contracts/apr-page-cli-list-v1.yaml -->

# apr list

List cached models

**Category**: Registry & Resources

## Synopsis

```text
apr list [OPTIONS]
```

## Example

```bash
apr list --json | jq length
```

## What this does

`apr list` enumerates models in the local cache (`~/.cache/aprender/`). Output
includes name, size on disk, last-used time, and format. Combined with `--json`
and `jq` it's the inventory query for "what models do I have, sorted by size"
or "which model haven't I used in 30 days?" Use `apr rm` to delete unused
checkpoints when disk gets tight.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--json` | Machine-readable output | `--json` |
| `-v, --verbose` | Include extra metadata (arch, dtype) | `--verbose` |
| `-q, --quiet` | Names only | `--quiet` |

## Common workflows

**Find your biggest cached models.**

```bash
apr list --json | jq -r '.[] | "\(.size_mb) \(.name)"' | sort -rn | head -10
```

**Garbage-collect anything older than 30 days.**

```bash
apr list --json | jq -r '.[] | select(.last_used_days_ago > 30) | .name' | \
    xargs -I{} apr rm {}
```

## Troubleshooting

- **Empty list, yet `apr run` works** — your `APR_CACHE_DIR` may differ from
  the default `~/.cache/aprender/`. Confirm with `apr list --verbose`.
- **`last_used_days_ago` missing for some entries** — older cached models
  pre-date the access-time tracking. They'll start populating once used.
- **Sizes don't match `du -sh ~/.cache/aprender`** — `apr list` shows logical
  model size; `du` includes shard metadata + temporary download artifacts.

## See also

- Source: [`crates/apr-cli/src/commands/list.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/list.rs)
- Contract: [`contracts/apr-page-cli-list-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-list-v1.yaml)

