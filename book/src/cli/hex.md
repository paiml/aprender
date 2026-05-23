<!-- PCU: cli-hex | contract: contracts/apr-page-cli-hex-v1.yaml -->

# apr hex

Format-aware binary forensics (10X better than xxd)

**Category**: Inspection

## Synopsis

```text
apr hex [OPTIONS]
```

## Example

```bash
apr hex qwen2.5-coder-1.5b-instruct-q4_k_m.gguf | head -20
```

## What this does

`apr hex` is xxd that understands model formats. It overlays APR/GGUF/SafeTensors
structure on top of the byte view: header fields, Q4K/Q6K/Q8_0 super-block layout,
per-region entropy, value distribution, and the layout contract from
`contracts/tensor-layout-v1.yaml`. Use it to debug new format support or to prove
a transpose was applied correctly.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--header` | Annotated header (magic, version, tensor count) | `--header` |
| `--blocks` | Q4K/Q6K/Q8_0 super-block structure | `--blocks --tensor lm_head` |
| `--distribution` | Value histogram + entropy + kurtosis | `--distribution` |
| `--contract` | Overlay tensor-layout contract per tensor | `--contract` |
| `--tensor PAT` | Filter to one tensor | `--tensor "blk.0.attn"` |
| `--raw` | xxd-style raw view with ASCII column | `--raw --offset 0x100` |

## Common workflows

**Verify GGUF -> APR import transposed correctly.**

```bash
apr hex qwen2.5-coder-1.5b.gguf --blocks --tensor "blk.0.attn_q.weight" > gguf.txt
apr hex qwen2.5-coder-1.5b.apr  --blocks --tensor "blk.0.attn_q.weight" > apr.txt
# Row-major APR view should show the col-major GGUF rows as columns
```

**Audit the file header without loading the whole model.**

```bash
apr hex unknown-model.gguf --header
# Confirms magic bytes, version, tensor count before you try `apr run`
```

## Troubleshooting

- **`magic bytes mismatch` on a known-good file** — corrupted download or wrong
  format extension. Cross-check with `file <model>` (POSIX) and re-pull if
  needed.
- **Block view looks random** — you're probably looking at an F32 tensor, not
  Q4K. Confirm with `apr tensors --filter <name>` to see the dtype.
- **Slow on 30B models** — `--raw` reads sequentially; scope with `--offset` and
  `--limit` to inspect just the region of interest.

## See also

- Source: [`crates/apr-cli/src/commands/hex.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/hex.rs)
- Contract: [`contracts/apr-page-cli-hex-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-hex-v1.yaml)

