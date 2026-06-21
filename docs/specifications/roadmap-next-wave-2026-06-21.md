# Roadmap — Next Wave (2026-06-21, post-v0.50.0)

**Supersedes the day-by-day in `beat-campaign-10day-schedule-2026-06-12.md` (its Day-10 milestone
is met) and is the live continuation of `autonomous-continuous-operation-2026-06-15.md` (mandate
runs to 2026-06-25, never-stop).**

## Milestone closed: v0.50.0 SHIPPED 2026-06-21
- **50 contract-backed correctness beats (PMAT-827..876)** across all four pillars + format/eval + CI-determinism.
- **68 crates published** to crates.io (full cascade; flagship `aprender`/`apr-cli` live, not yanked).
- Tag `v0.50.0` + GH release live. **Exhaustive post-publish QA = GO (8/8 dimensions)**: clean
  install, fresh-consumer build from crates.io, functional smoke, end-to-end model ops, release
  integrity, CB-510 content, dep-graph consistency.
- The first-ever full cascade exposed publish blockers (sibling path-deps missing `version`; unused
  version-pinned dev-deps closing cycles). Root-fixed; recovery = PR #2164. See
  `memory/feedback_crates_io_devdep_publish_cycles.md`.

## Standing operating model (unchanged, reaffirmed)
The **standing-pool invariant**: every turn ends with ≥1 background agent ACTIVELY producing work
(scouting/drafting/building/verifying), gx10 holding a live GPU job, and the main loop harvesting →
shipping → refilling. **A passive Monitor watching CI is NOT "work in flight"** — it must be paired
with an active producer agent, or the pool is idle. No dead gaps; never end a turn on a question.

## Next wave: PMAT-877+ → v0.51.0
Latest shipped ticket = PMAT-876; next wave starts at **PMAT-877**. BEAT half of the mission
(CI-gated falsifiable benchmarks + correctness beats). Tracks, always ≥1 in flight per lane:

### 🌟 cuda-oxide migration (gx10 / GB10 Blackwell — marquee north-star)
Pure-Rust `#[kernel]`→PTX is proven + optimized (Q4K dequant-matvec 22.9×, bit-exact on Blackwell).
NEXT: A/B the pure-Rust Q4K matvec vs trueno hand-PTX; if competitive, **migrate trueno's hand-PTX
hot kernels (dequant-matvec, FFN fusion, attention) to `#[kernel]`** — escapes Blackwell JIT pain.

### Pillar-4 (Ollama / llama.cpp)
- Decode marquee 1.43×→**1.5×** (gated at current number; closer = **DP4A Q8_1 kernel**).
- More fail-closed correctness classes, each adversarially vetted zero-false-positive on real models
  (quant-scale-near-zero candidate).

### Pillar-1 (scikit-learn)
Robust **compute-bound** beats only (algorithmic edge that survives the OpenBLAS CI host —
elementwise scalers were pulled because they lose to OpenBLAS). NEXT: LDA/QDA, SIMD scalers that
genuinely beat OpenBLAS, plus continued correctness-parity beats.

### Pillar-2 (PyTorch)
Training-step correctness + the PyTorch-CPU training beat (wall ≤ PyTorch+20% AND MSE≤0.05); GPU
PyTorch-CUDA baseline off idle Blackwell.

### Pillar-3 (Unsloth / PEFT)
QLoRA pipeline parity, single-command `apr finetune --qlora --export gguf`, NF4/bitsandbytes
equivalence.

## Release-engineering hardening (from the v0.50.0 cascade)
- **Add a release-branch CI gate**: `cargo publish -p aprender --dry-run --no-verify` so a
  version-pinned sibling dev-dep (the cycle class) is caught BEFORE a cascade, not mid-publish.
  (Operator-grade `.github/workflows` change — propose as a one-line addition.)

## Cadence
GitHub release per increment; crates.io batched. Next cascade after the next beat wave lands → 0.51.0.

## Stop conditions (only these)
Operator says stop; an architectural decision needs a human; or a destructive/irreversible op needs
sign-off. Otherwise: never stop, never idle, both boxes + agents always working.
