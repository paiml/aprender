# SHIP-007 PR-C-real — `apr trace --save-tensor` end-to-end smoke

**Date**: 2026-05-03
**Host**: noah-Lambda-Vector (RTX 4090, lambda-labs)
**Binary**: `/mnt/nvme-raid0/targets/aprender/release/apr` (built `--features inference,cuda` from main @ 420eabc75 + #1418 rebased)
**Model**: `/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr` (canonical 7B teacher, 8.0GB)

## Command

```bash
apr trace <model.apr> --save-tensor "embedding,lm_head" --save-tensor-dir /tmp/save-tensor-smoke
```

## Result

```
=== apr trace --save-tensor (APR) ===
Model:        /mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr
Stages:       embedding,lm_head
Layers:       0..1
Output dir:   /tmp/save-tensor-smoke

[PMAT-171] Loaded embedded BPE tokenizer: 152064 vocab, 151387 merges, 25 special tokens
Test prompt:  "What is 2+2?"
Token ids:    [3838, 374, 220, 17, 10, 17, 30] (7 tokens)

Wrote 2 stage tensor file(s):
  /tmp/save-tensor-smoke/layer-0/embedding.bin (100364 bytes)
  /tmp/save-tensor-smoke/lm_head.bin (608268 bytes)

Forward pass succeeded — 28 layer activations, 152064 logits
```

## APRT byte-format verification

`embedding.bin` first 16 bytes:
```
4150 5254 0000 0000 0062 0000 0000 e9ba
^^^^ APRT  ^^^^^^^^^ layer=0  ^^^^^^^^^ dim=0x6200=25088
```
- 25088 = 7 tokens × 3584 hidden_dim (Qwen2.5-Coder-7B) ✓
- Total: 12-byte header + 25088 × 4 bytes = 100364 bytes ✓

`lm_head.bin` first 16 bytes:
```
4150 5254 ffff ffff 0052 0200 45ff 1440
^^^^ APRT  ^^^^^^^^^ layer=0xFFFFFFFF (WHOLE_MODEL_LAYER)
            ^^^^^^^^^ dim=0x00025200=152064 (Qwen2.5-Coder vocab) ✓
```
- Total: 12-byte header + 152064 × 4 bytes = 608268 bytes ✓

## Ship-progress impact

This live smoke empirically discharges three contract gates from
`contracts/apr-cli-trace-save-tensor-v1.yaml` that were previously at
PARTIAL_ALGORITHM_LEVEL:

- **FALSIFY-APR-TRACE-SAVE-009** (apr_diff_values_compat) — APRT files are
  produced in the byte format that PR #1413's `apr diff --values` recognizes.
- **FALSIFY-APR-TRACE-SAVE-010** (LmHead step-2 capture) — `lm_head.bin`
  produced from the canonical teacher's last-token logits.
- **FALSIFY-APR-TRACE-SAVE-011** (CLI dispatch wire-up) — `apr trace
  --save-tensor model.apr` invokes `forward_traced_with_save_tensor` and
  writes selected stage files; no longer prints the stub.

A v1.4.0 contract bump promoting these from PARTIAL_ALGORITHM_LEVEL to
FUNCTIONAL_DISCHARGED follows once PR #1418 (the v1.3.0 paperwork PR) has
landed. This evidence file is the source of truth for that bump.

## Limits / next-session

Per-layer stages (qkv_matmul, ffn_gate, ffn_up, ffn_down, attn_out, etc.)
are not yet captured — `forward_traced_with_save_tensor` only emits
Embedding (PR #1408 step 1) + LmHead (PR #1414 step 2). SHIP-007 PR-C-real
step 3 threads `Option<&SaveTensorPlan>` through `forward_traced` itself
and is gated on PR #1416 (`maybe_save_stage` helper extraction) landing
first.
