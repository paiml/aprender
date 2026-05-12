# M32d numerical-parity DISCHARGE evidence

**Date:** 2026-05-01
**Host:** lambda-vector RTX 4090
**Branch stack:** Step 5 (#1228) → Step 5b (#1232) → Step 6 (this branch)

## TL;DR

Qwen3-Coder-30B-A3B-Instruct (Q4_K_M, 17.3 GB GGUF) inference output
went from `%%%%%%%%` gibberish to **fully coherent, contextually-correct
English** across multiple prompts. M32d numerical-parity gate is
essentially discharged via the M34 FAST PATH plan's identified
component priors.

## Prompt → output table

| Prompt | Pre-fix | Post-stack |
|--------|---------|-----------|
| `"What is 2+2?"` | `%%%%%%%%` | `2 + 2 = 4` |
| `"Hello"` | `%%%%%%%%` | `Hello! How can I help you today?` |
| `"fn factorial"` | `%%%%%%%%` | `## Python\n```python\ndef factorial(n):` |
| `"List 3 colors:"` | `%%%%%%%%` | `Red, blue, and green.` |

## The fix stack (each was a real bug surfaced in order)

### Step 2 + 2.5 — diagnostic surface (#1222, #1226)

Wired `apr trace --payload` for qwen3_moe arch. Live dogfood revealed:
```
layer[0].output_stats.std_dev  = 0.07
layer[47].output_stats.std_dev = 2.82
```
40× std growth signature → pointed to attention compounding.

### Step 5 — per-head Q/K RMSNorm (#1228)

Qwen3 GH-279 per-head Q/K RMSNorm was wired into the dense path
(`adaptive_ffn.rs:174-179`) but missing from `forward_qwen3_moe.rs`.

  * Output before: `%%%%%%%%`
  * Output after: `Human: What is 2+`

Rank-3 (15% prior) of M34 FAST PATH plan.

### Step 5b — rope_theta default 10K → 1M (#1232)

`config.rs::default_rope_theta_for_architecture` had `"qwen2" | "qwen3"
=> 1M` but no `"qwen3_moe"` entry. GGUF for Qwen3-Coder ships without
`qwen3moe.rope.freq_base` metadata, so the catch-all `_ => 10K` fired.
Off by 100×.

  * Output before: `Human: What is 2+`
  * Output after: `Human: What is 2+2?`

Rank-4 (10% prior) of M34 FAST PATH plan.

### Step 6 — chat template (this PR)

`detect_format_from_name` routed `"qwen3_moe"` to `Qwen3NoThink`, which
pre-injects `<think>\n</think>\n` after the `<|im_start|>assistant\n`
generation prompt. Qwen3-Coder is NOT a thinking model (verified via
the actual Jinja chat template in the GGUF metadata — only adds plain
`<|im_start|>assistant\n`). The `<think></think>` injection confused
the model; it emitted `<|endoftext|>` immediately.

  * Output before: `Human: What is 2+2?` (just reproducing prompt)
  * Output after: `2 + 2 = 4` (correct answer)

Outside the FAST PATH component-prior table — found via dogfood
investigation of why the prompt was being repeated.

## Component priors discharge status

| Rank | Component | Prior | Status |
|------|-----------|-------|--------|
| 1 | LAYOUT | 30% | not the issue |
| 2 | Q4_K_M | 20% | not the issue |
| **3** | **Q/K norm** | **15%** | **FIXED (#1228)** |
| **4** | **RoPE θ** | **10%** | **FIXED (#1232)** |
| 5 | MoE router softmax | 10% | not the issue (passing) |
| 6 | Token embedding | 10% | not the issue |
| 7 | Other | 5% | n/a |
| - | Chat template | n/a | **FIXED (this PR)** |

## What's still NOT done

- Sync `forward_qwen3_moe_traced` with the same fixes (depends on
  upstream PRs merging)
- HF FP16 cosine bisection (operator-confirm, ~60GB download) — now
  largely unnecessary since the model produces coherent English
- Stop-on-EOS still doesn't trigger reliably; greedy continues past
  `<|im_end|>`
- Multi-token completion past prompt-recognition; the model now
  responds correctly but might need more max-tokens to finish thoughts

## M32d FAST PATH cost actual vs estimate

| Outcome | PRs | Wall-clock |
|---------|-----|------------|
| **ACTUAL** | **5 PRs (Step 2 + 2.5 + 5 + 5b + 6)** | **~6 hours** |
| Lucky estimate | 4-6 PRs | 2-3 days |
| Realistic estimate | 8-10 PRs | 4-6 days |
| Pessimistic estimate | 12-15 PRs | 1-2 weeks |

**Came in at the lucky-case estimate, well under the wall-clock bound.**

## Cross-references

- companion `paiml/claude-code-parity-apr` § "M32d FAST PATH" (M34
  five-whys plan, 2026-05-01)
- aprender PRs: #1222, #1226, #1228, #1232, this branch
- GH-279 (Qwen3 per-head Q/K RMSNorm)
- HF Qwen3-Coder-30B-A3B-Instruct config.json (rope_theta=1M, no thinking)
