<!-- PCU: cli-check | contract: contracts/apr-page-cli-check-v1.yaml -->

# apr check

Model self-test: 10-stage pipeline integrity check (APR-TRACE-001)

**Category**: Quality & Evaluation

## Synopsis

```text
apr check [OPTIONS]
```

## Example

```bash
apr check qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

## What this does

`apr check` is the 10-stage end-to-end pipeline self-test from APR-TRACE-001:
tokenize, embed, RMSNorm, QKV, attention, FFN gate+up, FFN down, residual, LM
head, sample. Each stage must produce numerically sane output (no NaN/Inf,
within distribution band). A passing `apr check` means the model runs; it
doesn't mean it's accurate (use `apr eval` / `apr qa` for that).

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--no-gpu` | Run on CPU (Trueno SIMD) | `--no-gpu` |
| `--json` | One JSON envelope per stage | `--json` |
| `-v, --verbose` | Per-stage tensor stats | `--verbose` |

## Common workflows

**Quick post-convert sanity check.**

```bash
apr check qwen2.5-coder-0.5b.apr --json | \
    jq '.stages[] | select(.status != "PASS")'
```

**Compare CPU and GPU pipelines stage-by-stage.**

```bash
apr check qwen2.5-coder-1.5b.apr --no-gpu --json > cpu.json
apr check qwen2.5-coder-1.5b.apr         --json > gpu.json
jq -s '.[0].stages - .[1].stages' cpu.json gpu.json
```

## Troubleshooting

- **NaN at attention stage** — almost always a softmax numerical issue caused
  by a missing causal mask. Cross-check with `apr trace --layer "blk.0.attn"`.
- **"unsupported architecture"** — `apr check` needs a known dispatch table.
  Confirm `apr inspect --json | jq .arch`; if `unknown`, add the arch to
  the model dispatch contract.
- **All stages pass but `apr run` is gibberish** — the 10 stages cover
  numerical sanity, not tokenizer correctness. Run `apr tokenize <model>
  <prompt>` to verify the tokenizer round-trips.

## See also

- Source: [`crates/apr-cli/src/commands/check.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/check.rs)
- Contract: [`contracts/apr-page-cli-check-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-check-v1.yaml)

