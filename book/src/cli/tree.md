<!-- PCU: cli-tree | contract: contracts/apr-page-cli-tree-v1.yaml -->

# apr tree

Model architecture tree view

**Category**: Inspection

## Synopsis

```text
apr tree [OPTIONS]
```

## Example

```bash
apr tree qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

## What this does

`apr tree` renders the architecture as a `tree(1)`-style hierarchy: model ->
blocks -> sub-modules -> tensors. With `--sizes` it sums tensor bytes at each
node, so you can see exactly where memory goes (typically embeddings + LM head
dominate 1B-3B models). Output formats include ASCII, Graphviz `dot`, Mermaid,
and JSON.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--format FMT` | `ascii` (default), `dot`, `mermaid`, `json` | `--format mermaid` |
| `--sizes` | Show tensor sizes at each node | `--sizes` |
| `--filter PAT` | Filter by component pattern | `--filter "attn"` |
| `--depth N` | Cap tree depth | `--depth 3` |

## Common workflows

**Generate a Mermaid diagram for an architecture spec doc.**

```bash
apr tree qwen2.5-coder-1.5b.apr --format mermaid --depth 4 > qwen-arch.mmd
```

**Find the largest tensors at a glance.**

```bash
apr tree qwen2.5-coder-7b.apr --sizes --depth 2 | sort -k 2 -h | tail -10
```

## Troubleshooting

- **Tree is flat (one level)** — naming convention doesn't match a known
  architecture. Run `apr inspect --json | jq .arch` and confirm the arch is
  detected; fall back to `apr tensors` for a raw view.
- **`--sizes` shows 0** — model uses sentinel-only tensors (rare). Check
  `apr tensors --stats` to see if shapes are real.
- **Mermaid output too large to render** — clip with `--depth 3` and
  `--filter` to one block, then expand layer-by-layer.

## See also

- Source: [`crates/apr-cli/src/commands/tree.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/tree.rs)
- Contract: [`contracts/apr-page-cli-tree-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-tree-v1.yaml)

