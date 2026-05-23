<!-- PCU: cli-registry | contract: contracts/apr-page-cli-registry-v1.yaml -->

# apr registry

Registry operations (CRUX-A-01): inspect alias map, etc

**Category**: Registry & Resources

## Synopsis

```text
apr registry [OPTIONS]
```

## Example

```bash
apr registry status
```

## What this does

`apr registry` exposes the alias map at `configs/aliases.yaml` — the table that
lets you write `apr run qwen2.5-coder-1.5b` instead of `apr run
hf://Qwen/Qwen2.5-Coder-1.5B-Instruct`. Aliases are version-controlled, so two
developers running the same short name will pull the same canonical revision.
Use `registry aliases` to dump the table; future subcommands will add health
and provenance checks.

## Key flags

| Subcommand | What it does | Example |
|-----------|-------------|---------|
| `registry aliases` | List short-name -> canonical-URL pairs | `apr registry aliases --json` |
| `--json` | Machine-readable output | `--json` |
| `-v, --verbose` | Include revision pins + descriptions | `--verbose` |

## Common workflows

**Audit which short names point to which HF repos.**

```bash
apr registry aliases --json | jq '.[] | {alias: .name, hf: .canonical_url}'
```

**Verify a `apr run` resolves to the expected canonical URL.**

```bash
apr pull qwen2.5-coder-0.5b --dry-run
# Prints resolved hf://... URL without downloading
```

## Troubleshooting

- **"unknown alias"** — the short name isn't in `configs/aliases.yaml`. Use
  the full `hf://org/repo` URI, or open a PR adding the alias.
- **Two short names map to the same URL** — that's a registry duplicate.
  File an issue; the lint will flag this once
  [the registry-lint contract](https://github.com/paiml/aprender/blob/main/contracts/apr-registry-aliases-v1.yaml)
  ships.
- **Alias resolves but `apr pull` fails** — revision pin in the alias map
  may point to a deleted SHA. Update `configs/aliases.yaml`.

## See also

- Source: [`crates/apr-cli/src/commands/registry.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/registry.rs)
- Contract: [`contracts/apr-page-cli-registry-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-registry-v1.yaml)

