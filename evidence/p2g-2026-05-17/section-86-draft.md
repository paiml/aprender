# §86 draft — `apr pretrain --init` silently fails to load arch-mismatched checkpoints (2026-05-17)

**Draft section for `docs/specifications/aprender-train/ship-model-2-spec.md`.**
Will be applied as a PR after #1754 (§85) merges to avoid stacking churn.

## Discovery

P2-G v1 was dispatched to test the §85.4 marginal-gain decay prediction by resuming P2-E ep49's checkpoint for 10,000 more steps. The init eval at step 0 produced **val_loss = 8.60** — higher than P2-E's ep0 init eval (val_loss = 7.43) and ~85% higher than P2-E ep49's actual val_loss (4.62). The trained weights weren't loading.

## Root cause

P2-E's ep49 checkpoint has:
- `architecture: "LlamaForCausalLM"` (the P0-H fallback when `init_arch.hf_architecture` is None)
- 291 tensors with Qwen2 naming convention (`model.layers.N.self_attn.{q,k,v,o}_proj.weight` etc.)

When `apr pretrain --init <P2-E-ep49.apr>` reads this:
1. Architecture extraction sees `architecture: "LlamaForCausalLM"` and builds a Llama-shaped trainer
2. Tensor load tries to map the APR's Qwen2 names onto the Llama trainer's expected names
3. Mismatch → silent fallback to random init for failed tensors
4. Training proceeds from essentially-random weights with val_loss ≈ 8.60

This is a SECOND symptom of the §81–§84 cascade root cause: pre-P0-K APRs lack `hf_architecture`, the P0-H stamp falls back to "LlamaForCausalLM" by default, and the trained checkpoint inherits the wrong arch family stamp. The checkpoint is then non-resumable.

## Implications

- All training checkpoints produced before PR #1742 landed (timestamp ~2026-05-17T13:32:08Z) have `architecture = "LlamaForCausalLM"` regardless of their actual tensor structure. They are non-resumable via `apr pretrain --init`.
- The 50 P2-E checkpoints (`epoch-000.apr` … `epoch-049.apr`, ~125 GB total) cannot be used for continuation training.
- P2-E ep49's empirical result (val_loss = 4.62) stands as a single-shot benchmark but cannot be extended without re-training from scratch.

## Workarounds (in order of preference)

1. **Re-import the source Qwen2.5-Coder-0.5B-Instruct via post-P0-K `apr convert`** — produces an init APR with `hf_architecture = "Qwen2ForCausalLM"` correctly stamped. Trained checkpoints from THIS init will have correct arch family and be self-resumable. Requires re-downloading the safetensors (HF cache currently has config.json only, no .safetensors).

2. **Restamp existing pre-P0-K APRs in-place** — write a small `apr stamp --hf-architecture Qwen2ForCausalLM` tool that patches metadata only. The +128-byte size delta from PR #1050 (apr stamp helper) is precedent. This salvages the existing P2-E checkpoints. Estimated work: ~50 LOC + contract.

3. **Treat P2-E's result as final** — accept that the trained checkpoints are non-resumable, use ep49 as a single benchmark for §85's marginal-gain analysis, and direct future dispatches at fresh-from-import inits.

P2-G v2 (the re-dispatched run currently in flight at PID 2063155) takes approach #3: it runs 10,000 steps from the same pre-P0-K `qwen2.5-coder-0.5b-instruct-imported.apr` that P2-E used. This effectively doubles P2-E's training length but does NOT validate the marginal-gain extrapolation cleanly because the init is fresh-from-import, not P2-E's ep49.

## Failure mode classification

| Aspect | Value |
|---|---|
| Class | Class 4 (Silent Incorrect Behavior) |
| Detection latency | 1 epoch (~55s) once init eval prints — easy to spot if you compare against expected loss |
| Symptom | val_loss at ep 0 disagrees with init checkpoint's last recorded val_loss by orders of magnitude |
| Fix scope | Producer-side: P0-K already shipped. Existing pre-P0-K artifacts: need either re-import or restamp tool |

## Related contracts

- `contracts/apr-convert-hf-arch-v1.yaml` — producer-side stamping (already shipped via P0-K)
- `contracts/apr-pretrain-from-init-v1.yaml` — would benefit from a new INV-INIT-ARCH-MATCH-001 invariant: "if init.architecture is set, the architecture-family inferred from tensor names MUST match"

## Recommended follow-up

`apr stamp` tool (workaround #2) — small, high-EV, salvages all pre-P0-K artifacts AND establishes a pattern for in-place metadata patching that could discharge AC-SHIP1-009 (MODEL-1 teacher provenance restamp) via the same code path.

Effort: ~100 LOC + integration test + contract. EV: unblocks resume training for the existing ~125 GB of P2-E checkpoints without re-running the training.
