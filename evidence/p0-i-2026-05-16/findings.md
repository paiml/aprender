# P0-I Verdict: PARTIAL — P0-G verified, P0-H gated on fresh training

**Ticket:** PMAT-679
**Date:** 2026-05-16
**Spec:** [albor-370m-roadmap.md §4 P0](../../docs/specifications/aprender-train/albor-370m-roadmap.md)

## What P0-I tested

Build apr from post-merge main; re-export `epoch-020.apr` → GGUF → `llama-cli` to verify both P0-G (PR #1706, vocab pad) and P0-H (PR #1709, arch-from-init) fixes work end-to-end.

## What actually ran

System memory was critically low (3 GB free of 125 GB; 127 GB swap exhausted); `cargo build` was blocked. Used the pre-existing canonical binary at `/mnt/nvme-raid0/targets/aprender/release/apr` (`apr 0.33.0 (864d69a75)`) which already contains the P0-G fix from the local working-tree build before that merge.

## Findings

### P0-G verified LIVE ✅

The export log (`export-log.txt`) shows the P0-G code path executing as expected:

```
[BUG-EXPORT-004] Warning: No tokenizer.json found near …
[P0-G] Padding APR-fallback tokenizer.ggml.tokens: 151643 + 293 placeholders = 151936
```

The GGUF metadata (`gguf-metadata.txt`) confirms the result:

```
tokenizer.ggml.tokens     [len=151936]   (was 151643 pre-pad)
llama.vocab_size          151936
token_embd.weight shape   [896, 151936]
```

llama-cli's `check_tensor_dims` would now accept this token tensor first-dim (151936 matches across all three). The previous PR-1706-pre symptom (`expected 896, 151643, got 896, 151936`) does NOT appear in the load log.

### P0-H NOT verified on this checkpoint ⚠️

llama-cli (`llamacli-load.txt`) still fails — but with a DIFFERENT error than the pre-P0-G state:

```
done_getting_tensors: wrong number of tensors; expected 291, got 219
```

This is the Qwen2-bias-leak symptom P0-H is designed to fix. The reason it still appears: the `epoch-020.apr` checkpoint was trained BEFORE the P0-H code landed on main, so its APR metadata still has the hardcoded `architecture = "LlamaForCausalLM"`. When re-exported under the llama-family GGUF mapper, the 72 Qwen2 `q_proj/k_proj/v_proj.bias` tensors fall through as passthrough names (counted in the header at 291, rejected by llama-cli at 219).

The P0-H code change (`apr-cli/src/commands/pretrain.rs`'s `checkpoint_name_and_arch` helper) only affects checkpoints **emitted after** the fix landed. To verify P0-H end-to-end, we need a fresh `apr pretrain --init qwen-0.5b ...` run whose APR carries `architecture = "Qwen2ForCausalLM"`.

## Discharge plan

P0-I is **PARTIAL — discharged for P0-G; deferred for P0-H to P2-C**. Rationale:

1. P0-G is verified independently from any new training run (current export works).
2. P0-H verification will happen automatically as a side-effect of P2-C (corpus widening + retraining), since P2-C produces a fresh checkpoint emitted by the post-P0-H code path.
3. There is no cheaper way to exercise P0-H end-to-end without producing a new checkpoint, and dispatching P2-A2 just to test P0-H would burn 3-8h GPU on a pre-falsified run (per §83 audit).

## Evidence files

- `export-log.txt` — full `apr export --format gguf` output (P0-G pad message line 4)
- `gguf-metadata.txt` — `apr inspect` showing vocab=151936 alignment
- `llamacli-load.txt` — llama-cli error showing remaining P0-H symptom (291 vs 219)

## Next actions

- **PMAT-679 (P0-I):** Mark complete with partial verdict; close the P0-G half; defer P0-H half to PMAT-681 (P2-C).
- **PMAT-681 (P2-C):** When P2-C dispatches, include an llama-cli load test against the new checkpoint as part of acceptance.
- **No retraining is sensible here** — wait for P2-C.
