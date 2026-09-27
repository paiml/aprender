# 0.72 "Train What You Serve" — top-5 row specs (L2 shaping)

**Status:** PROPOSED (look-ahead L2, APR-LOOKAHEAD-001 §4). Nothing here is minted; the cop mints tickets.
**Epic:** #4000 · **Author:** la-72 · **Date:** 2026-09-27 · **Tree read:** origin/main `aca6f2d7f6`
**Exit criteria (operator ruling 2026-09-27, APR-LOOKAHEAD-001 §2a):** T1 verbs end to end on Qwen 3.5 with 0 refusals ·
T2 fine-tune throughput ≥ 0.8× Unsloth · T3 Prometheus B2 challenger · T4 dogfood model rc on Hugging Face · T5 round-trip cosine ≥ 0.98.
Marks: `[V]` verified in tree · `[C]` computed · `[A]` asserted · `[U]` unmeasured.

## §0 The finding that shapes the train `[V]`

T1 reads "the qwen35 refusals are removed". The training verbs have **no** qwen35 refusal. They mis-model it silently:

| Where | What |
|---|---|
| `crates/aprender-train/src/train/pretrain_real.rs:230` | `"qwen3_5" \| "qwen3.5"` → family `qwen3` (dense) |
| `crates/aprender-train/src/transformer/config.rs:246` | `qwen3_5_9b()` is a dense `Decoder`: no Gated DeltaNet (GDN) layers |
| `crates/aprender-train/src/transformer/config.rs:297-301` | any arch prefixed `qwen3` gets head_dim = 128; the 3.5 preset's own comment says 256 |
| `crates/aprender-train*` | no GDN implementation; GDN exists only in aprender-serve (`forward_qwen35*`, `gdn_ops`) |

So "0 refusals" means **GDN forward and backward in the training crate**. LoRA over a frozen base still back-propagates
through every GDN layer, so no adapter-only shortcut avoids the backward. `apr merge` and `apr quantize` are
tensor-name keyed and architecture-independent `[V]`; they need qwen35 cells, not new modelling.

## §1 Critical path `[C]`

```
R1 honesty gate ─► R2 GDN forward (= serve) ─► R3 GDN backward ─► R4 QLoRA e2e ─┬─► R5 Unsloth baseline ─► R14 perf
                                                                                ├─► R6 distill, R7 merge
                                                  #4418 GGUF name map (0.71) ───┴─► R10 round-trip ─► R13 HF rc
```

## §2 Rows

### R1 — Training-arch honesty gate · contract `train-arch-honesty-v1` · K̂ 30 `[A]`
- **Change:** aprender-train refuses any architecture it doesn't model (`UnsupportedArch{arch, missing}`) before any
  weight is read. Delete the `qwen3_5 → qwen3` alias. head_dim comes from `contracts/model-families/<f>.yaml`, not a prefix match.
- **Falsifiers:** TAH-001 (named refusal at the verb) · TAH-002 planted: re-adding the alias turns TAH-001 RED ·
  TAH-003 head_dim table over every family YAML · TAH-004 refusal precedes I/O.
- **Why first:** it is small, it lands under L1/L2 rules as a normal PR, and it stops wrong training today.
  R2/R3 later remove the refusal. A refusal is the honest interim state, not the goal.

### R2 — GDN forward in aprender-train, identical to serve · contract `qwen35-train-gdn-v1` (QTG-001/002/005) · K̂ 90 `[A]`
- **Change:** add a GDN layer (causal conv1d, gated delta recurrence, gated RMSNorm, per-head L2 norm) and a hybrid
  layer schedule to the training transformer. Reuse the `gated-delta-net-v1` equations; don't re-derive them.
- **Gate:** per-position cosine ≥ 0.9999 and equal argmax between aprender-train and aprender-serve logits on Qwen3.5-4B,
  both loading the same .apr bytes (CPU, dequantized f32). This check runs on CPU, so it may run while train-active.
- **Planted:** the sigmoid-gate mutant turns it RED. The same mutant scores 0.7656 on the serving side (qwen35-hybrid-forward-v1).
- **Spike S-R2 (desk, 2026-09-27) — share the kernel, or run a parallel implementation? Answer: parallel training impl, serve as the oracle. No ruling needed.** `[V]`
  - The crate DAG already allows train → serve. `crates/aprender-train/Cargo.toml:84` has
    `realizar = { workspace = true, optional = true }` (implicit feature `realizar`), and aprender-serve does not depend on
    aprender-train, so there is no cycle. `realizar::gguf::forward_qwen35` is `pub` (`crates/aprender-serve/src/gguf/mod.rs:147`)
    and exports `causal_conv1d`, `delta_rule_recurrence` and `delta_rule_recurrence_gqa` (`forward_qwen35.rs:89,130,221`).
  - Those functions cannot be the training forward. They step one token at a time, update the conv/recurrent state in
    place, and save no activations, so autograd has nothing to differentiate through. A training GDN has to be a
    sequence-level (chunked) forward on the tape that keeps S_t (or recomputes it per chunk).
  - So: aprender-train implements its own sequence GDN forward+backward. The parity test QTG-001 runs under
    `--features realizar` and calls serve's `forward_qwen35` on the same .apr bytes as the ORACLE. "train = serve" holds
    by test, not by shared code, which is the same guarantee without a refactor.
  - Moving the recurrence into aprender-compute is a refactor with no extra guarantee; deferred (not a 0.72 row).
  - `[U]` Build cost: `cargo check -p aprender-train --features realizar` was not run (heavy slots were saturated). The
    first R2 PR must show it green.

### R3 — GDN backward + gradcheck · contract `qwen35-train-gdn-v1` (QTG-003/004) · K̂ 120 `[A]`
- **Change:** analytic backward for every GDN parameter, plus the carried state S₀ (chunked training passes state across chunks).
- **Gate:** f64 central-difference gradcheck, rel err ≤ 1e-3, on a tiny layer (2 heads, d = 8, T = 16).
- **Planted:** zeroing the read-out gradient (through r_t = S_tᵀk_t) turns it RED.
- **Risk K1:** no in-tree training-side reference exists; this is the train's largest schedule risk.

### R4 — QLoRA end to end on Qwen3.5-4B (CUDA) · contract `qwen35-qlora-e2e-v1` · K̂ 90 `[A]`
- **Cell:** NF4 base; LoRA r16 on attention, MLP and GDN projections; 1,000-sample pinned set; seed 42; 200 steps; RTX 4090.
- **Gates:** loss(last 10) ≤ 0.9 × loss(first 10), all finite. Served base+adapter equals the training-side merged forward
  (cos ≥ 0.999, equal argmax).
- **Planted:** lr = 0 must FAIL the loss gate. The receipt names the device from a trace line (CLAUDE.md verification rule 2).
- **Existing surface `[V]`:** QLoRA path at `crates/apr-cli/src/commands/finetune.rs:270-336`. The contract it cites,
  `qlora-training-loop-v1` (`finetune.rs:349`), has no file (row R16).

### R5 — Unsloth fine-tune throughput harness · contract `beat-unsloth-finetune-throughput-v1` · K̂ 60 `[A]`
- **Today `[V]`:** there is no training-throughput harness. The sibling `beat-unsloth-coldstart-speed-v1` explicitly
  *concedes* GPU in-loop QLoRA throughput to Unsloth. T2 reverses that concession.
- **Harness:** pinned Unsloth (uv lock, versions recorded) vs `apr finetune`, same GPU, same recipe, same data sha.
  Median of 3 runs. Trained tokens/s over 200 timed steps, excluding load.
- **Same-work guard (planted):** trainable-parameter counts must match within 1%. Halving apr's targets must FAIL.
- **First run = baseline.** The 0.8 threshold is fixed by operator ruling. The gap feeds R14.
- **`[U]` first check:** does Unsloth support Qwen 3.5 GDN layers? If not, both sides target attention + MLP only, and the
  receipt says so.

## §3 Remaining ranked rows (R6–R20)
See the L2 handoff (`docs/lookahead/0.72.md` once LA-00 lands). In brief: R6 distill 27B→4B at batch > 1 · R7 merge cells ·
R8 quantize policy for GDN tensors · R9 #4418 (0.71 dependency) · R10 T5 round-trip gate (none exists `[V]`) ·
R11 B2 wiring (#4367) · R12 training receipts on APR-OBS identity · R13 HF rc · R14 throughput work · R15 LoRA on CUDA ·
R16 dangling `qlora-training-loop-v1` · R17 4B memory plan · R18 teacher/student vocab alignment · R19 ROADMAP PMAT-711 stale ·
R20 declarative recipe (only if pulled from E8 0.75).

## §4 Rulings requested (S-4)
RQ-1 epic #4000 body is still "Agent Ready" · RQ-2 T1 scope = GDN training · RQ-3 T2 vs E8 0.75 (#4002) overlap ·
RQ-4 #4418 stays in 0.71. Full text is in the handoff file.
