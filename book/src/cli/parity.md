<!-- PCU: cli-parity | contract: contracts/apr-page-cli-parity-v1.yaml -->

# apr parity

GPU/CPU parity check (PMAT-232: genchi genbutsu — see where GPU diverges)

**Category**: Quality & Evaluation

## Synopsis

```text
apr parity [OPTIONS]
```

## Example

```bash
apr parity model.gguf --backends cpu,gpu
```

## What this does

`apr parity` runs the same prompt through CPU and GPU and compares outputs
token-by-token. When CPU and GPU disagree, the divergence is almost always a
quantization kernel that lost precision differently on each backend, or a
layout-001/002 transpose error. With `--assert` it exits non-zero on divergence,
making it a CI gate.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `-p, --prompt TEXT` | Prompt to compare (default "What is 2+2?") | `--prompt "fn main()"` |
| `--assert` | Exit non-zero on divergence | `--assert` |
| `--json` | JSON output with per-token diff | `--json` |
| `-v, --verbose` | Per-layer divergence details | `--verbose` |

## Common workflows

**Block PR if a kernel change introduces GPU divergence.**

```bash
apr parity qwen2.5-coder-1.5b.gguf --prompt "def hello():" --assert --json
```

**Drill into the layer that diverges (genchi genbutsu).**

```bash
apr parity qwen2.5-coder-1.5b.gguf --verbose 2>&1 | grep DIVERGE
# Then surgically trace that layer:
apr trace qwen2.5-coder-1.5b.gguf --layer "blk.7" --save-tensor all
```

## Troubleshooting

- **Parity fails on first token** — likely tokenizer or embedding bug, not a
  kernel bug. Confirm with `apr tokenize <model> <prompt>` on both backends.
- **Token 50 diverges but tokens 1-49 match** — slow precision drift from
  accumulator differences. Often acceptable for quantized models; investigate
  if it crosses a sampling boundary.
- **"no GPU detected"** — `apr parity` requires CUDA. Build with `--features
  cuda` or use `apr trace` for offline comparison via saved tensors.

## See also

- Source: [`crates/apr-cli/src/commands/parity.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/parity.rs)
- Contract: [`contracts/apr-page-cli-parity-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-parity-v1.yaml)

