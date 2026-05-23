<!-- PCU: cli-trace | contract: contracts/apr-page-cli-trace-v1.yaml -->

# apr trace

Layer-by-layer trace analysis

**Category**: Inspection

## Synopsis

```text
apr trace [OPTIONS]
```

## Example

```bash
apr trace qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --prompt "What is 2+2?" --max-tokens 4
```

## What this does

`apr trace` runs a prompt through the model and emits per-layer tensor stats
(or full F32 payloads) at every stage: embedding, qkv, attn output, FFN gate/up,
FFN down, RMSNorm, LM head. It's the surgical tool for CPU/GPU divergence
hunting — diff a trace from `--backend cpu` against `--backend cuda` and the
first row to disagree tells you which kernel diverged.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--layer PAT` | Filter to one layer pattern | `--layer "blk.0"` |
| `--payload` | Trace full F32 tensor payloads | `--payload` |
| `--diff` | Diff mode (against `--reference`) | `--diff` |
| `--save-tensor STAGES` | Save per-stage F32 tensors to disk | `--save-tensor embedding,qkv_matmul` |
| `--save-tensor-layers RANGE` | Layer range (default `0..1`) | `--save-tensor-layers 0..2` |
| `--reference PATH` | Reference model for `--diff` | `--reference golden.apr` |

## Common workflows

**Find the layer where GPU diverges from CPU on a quantized model.**

```bash
apr trace qwen2.5-coder-1.5b.apr --backend cpu  --save-tensor all --save-tensor-dir /tmp/cpu
apr trace qwen2.5-coder-1.5b.apr --backend cuda --save-tensor all --save-tensor-dir /tmp/gpu
diff <(ls /tmp/cpu) <(ls /tmp/gpu)
# Then numpy-diff each stage to find the first divergent kernel
```

**Diff two quantization levels at the same prompt.**

```bash
apr trace qwen2.5-coder-1.5b-q4k.apr --diff --reference qwen2.5-coder-1.5b-q6k.apr \
    --prompt "fn main() {" --max-tokens 8
```

## Troubleshooting

- **`--save-tensor` writes nothing** — confirm the stage names match
  `contracts/apr-cli-trace-save-tensor-v1.yaml` (e.g. `embedding`, not
  `embed`). Use `all` to capture every stage.
- **Trace is slow for 7B+ models** — that's the F32 payload write. Drop
  `--payload` if you only need stats, or restrict to one layer with `--layer`.
- **Diff shows divergence at layer 0** — likely a tokenizer or embedding bug,
  not a kernel bug. Confirm with `apr tokenize <model>` against the reference.

## See also

- Source: [`crates/apr-cli/src/commands/trace.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/trace.rs)
- Contract: [`contracts/apr-page-cli-trace-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-trace-v1.yaml)

