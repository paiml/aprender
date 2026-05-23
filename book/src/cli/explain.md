<!-- PCU: cli-explain | contract: contracts/apr-page-cli-explain-v1.yaml -->

# apr explain

Explain errors, architecture, tensors, and kernel dispatch

**Category**: Inspection

## Synopsis

```text
apr explain [OPTIONS]
```

## Example

```bash
apr explain qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

## What this does

`apr explain` is the human-readable companion to the rest of the inspection
suite. Pass an error code (like `LAYOUT-001`), a model file, an architecture
family name, or a tensor — and it returns a paragraph explaining what it is,
why it matters, and where to look next. With `--kernel` it walks the kernel
dispatch pipeline (which fused kernel handles `attn_q.weight` on this backend?
why?).

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--tensor NAME` | Explain a specific tensor | `--tensor lm_head.weight` |
| `--kernel` | Walk the kernel dispatch tree | `--kernel` |
| `--proof-status` | Per-kernel proof status from contracts | `--proof-status` |
| `-v, --verbose` | Include contract details + obligations | `--verbose` |

## Common workflows

**Look up an error code from a stack trace.**

```bash
apr explain LAYOUT-001
# "GGUF column-major data was loaded without transpose..."
```

**Audit which kernel will run for a given tensor on this backend.**

```bash
apr explain qwen2.5-coder-1.5b.apr --tensor "blk.0.ffn_gate.weight" --kernel --verbose
# Shows: fused_q4k_parallel_matvec via Trueno SIMD (row-major), proof status: VERIFIED
```

## Troubleshooting

- **"no such error code"** — the code may be from a downstream crate (trueno,
  realizar). Cross-check with `pmat query "<code>"`.
- **`--proof-status` reports `UNVERIFIED`** — the kernel is in the dispatch
  table but its falsification suite has gaps. File a contract bump rather than
  shipping.
- **Architecture name not recognized** — pass a model file directly so explain
  can read `general.architecture` from metadata.

## See also

- Source: [`crates/apr-cli/src/commands/explain.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/explain.rs)
- Contract: [`contracts/apr-page-cli-explain-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-explain-v1.yaml)

