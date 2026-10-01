---
type: defect
severity: high
created: 2026-09-07
found_in: phase-05-plan-11-task-1-checkpoint
resolves_phase:
---

# Qwen3.5-9B hybrid forward path is unimplemented — blocks the LoRA comparison arm

## What

`TransformerConfig` cannot express the Qwen3.5-9B architecture, so `--model-size 9B` can be
selected but never loaded. Three gaps, each independently fatal: no `layer_types` field (the
checkpoint is 24 `linear_attention` + 8 `full_attention` over 32 layers, `full_attention_interval:
4`), no `attn_output_gate`, and a flat text-only config against a multimodal
`Qwen3_5ForConditionalGeneration` checkpoint whose text hyperparameters are nested under
`text_config`. The only hybrid-forward artifact lives in `aprender-contracts-staging`, which has no
`Cargo.toml` and never compiles.

Weights are not the obstacle: `Qwen/Qwen3.5-9B` is public and ungated (revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, 19.31 GB bf16, 4 shards).

## Impact

Descoped the 9B LoRA arm out of EVAL-02 and EVAL-04 and out of the Phase 5 goal and success
criteria 2 and 4 (`05-CONTEXT.md` D-19). Phase 5 now publishes a SetFit-only claim set.

## Before picking this up

Run the falsifier first — it is free and needs no GPU. D-19(b) rests on structural absence, not an
observed loader failure (Verification Discipline rule 6): point aprender's loader at the real
`config.json` and see whether it accepts or rejects the checkpoint. If it loads cleanly, the
diagnosis is wrong and this is only a host-access problem.

## Full record

`.planning/phases/05-benchmark-and-claims-gate/deferred-items.md` § `D-ITEM-05-15`
