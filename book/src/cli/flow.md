<!-- PCU: cli-flow | contract: contracts/apr-page-cli-flow-v1.yaml -->

# apr flow

Data flow visualization

**Category**: Inspection

## Synopsis

```text
apr flow [OPTIONS]
```

## Example

```bash
apr flow qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

## What this does

`apr flow` renders the model's data-flow graph — how a token embedding becomes
logits — as a directed graph of operators (matmul, attn, RMSNorm, SwiGLU, etc.)
rather than a tensor-storage tree (which is `apr tree`'s job). Use this to
understand WHERE in the pipeline a fused kernel will apply, or to author a
roofline plan for a new architecture.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--component C` | `full`, `encoder`, `decoder` | `--component decoder` |
| `--layer PAT` | Filter to specific layer pattern | `--layer "blk.0"` |
| `--json` | JSON output for graph tools | `--json` |
| `-v, --verbose` | Include per-edge tensor stats | `--verbose` |

## Common workflows

**Print the decoder data-flow for SwiGLU fusion analysis.**

```bash
apr flow qwen2.5-coder-1.5b.apr --component decoder --layer "blk.0" --verbose
# Shows: rmsnorm -> attn_qkv (fused) -> attn_o -> rmsnorm -> ffn_gate+up (fused) -> ffn_down
```

**Export the graph for an external visualizer.**

```bash
apr flow qwen2.5-coder-7b.apr --json | jq '.nodes | length'
```

## Troubleshooting

- **`unknown architecture`** — flow uses an arch-specific graph. If `apr inspect`
  reports `arch: unknown`, the model lacks the `general.architecture` metadata
  field; stamp it with `apr stamp` or re-convert.
- **Output is huge for 30B models** — scope with `--layer "blk.0"` first and
  expand from there.
- **Verbose stats show zeros** — `--verbose` requires the model to be in a
  format with embedded tensor stats; otherwise it falls back to shape-only.

## See also

- Source: [`crates/apr-cli/src/commands/flow.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/flow.rs)
- Contract: [`contracts/apr-page-cli-flow-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-flow-v1.yaml)

