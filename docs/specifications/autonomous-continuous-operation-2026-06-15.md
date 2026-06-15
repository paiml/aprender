# Autonomous Continuous Operation — 10-Day Plan (2026-06-15 → 2026-06-25)

**Mandate (operator, 2026-06-15):** work autonomously for the next 10 days, **never stop**,
**always use BOTH compute boxes** (lambda-labs RTX 4090 + gx10 GB10 Blackwell), and **always have
agent(s) working**. The prior failure mode — "do a burst → `ScheduleWakeup` → sleep 20 min with
nothing running" — is FORBIDDEN. There must be **no dead gaps**.

## Operating model: the standing pool (how "never stop" actually works)

The harness re-invokes the main loop whenever a background task finishes (Bash bg, `Agent`
`run_in_background`, `Workflow`) **or** a `ScheduleWakeup`/Monitor fires. Therefore:

> **INVARIANT: at the end of EVERY turn there must be ≥1 background agent running, gx10 must have a
> live job, and (when work exists) a Workflow or background build in flight.** As long as ≥1
> background task is running, its completion re-invokes the loop → dispatch more → repeat, with no
> 20-minute idle sleep. `ScheduleWakeup` is only a *short safety net* (~600s) for the rare case the
> pool fully drains.

Three concurrent lanes, always full:

1. **gx10 lane (GB10 Blackwell, `ssh gx10`)** — always one long GPU job: cuda-oxide kernel
   R&D (port/optimize/A-B trueno hot kernels → pure-Rust→PTX), training/distill, or eval sweeps.
   Re-dispatch the instant one finishes. gx10 disk is tight (~853G of operator distill data; keep
   ≥10G free by cleaning only `/tmp/*_spike`, `/tmp/apr-gxval/target`, my logs — never operator data).
2. **lambda-vector lane (this box, RTX 4090 + CPU)** — always ≥1 background `Agent` doing high-EV
   work: beat scouting + measurement, kernel/impl drafts in `isolation: worktree`, adversarial
   verification, coverage/contract co-evolution. The local RTX 4090 runs serve/inference GPU work.
3. **Main loop (coordinator)** — on each re-invoke: harvest finished results → ship the PR /
   re-dispatch → **refill the pool** → end the turn with the pool still full. Reviews what agents
   produce before merging (quality gate; agents scout+draft, main ships).

Standing rules carried in: full autonomy / never ask; main protected (PRs + auto-merge);
**verify beats on the host that gates them, not the dev box** (PMAT-733 lesson); honest-by-design
(ship only measured wins); Rule 7 (every fix co-evolves a contract + falsifier).

## Current state (Day-4 checkpoint, 2026-06-15)

**Pillar-1 (replace+beat sklearn) — speed beats SHIPPED + nightly-gated (Intel/OpenBLAS CI host):**
LinReg, GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, GMM. Elementwise scalers
(StandardScaler/MinMaxScaler) were shipped then REMOVED — they win on a fast dev box but LOSE to
OpenBLAS-numpy on the canonical CI host (no algorithmic edge). **Only compute-bound beats with a
genuine algorithmic edge are robust across hosts.** 3 real perf fixes landed (ln/det-hoist:
GaussianNB, BernoulliNB, GMM).

**Pillar-4 (beat Ollama/llama.cpp) — fail-closed correctness beats:** apr provably rejects
semantically-broken models the incumbents run silently. Headline (all-zero/NaN/Inf/L2~0/constant) +
extreme-magnitude (F-DATA-QUALITY-005, #2039) shipped.

**cuda-oxide north-star (pure-Rust→PTX on Blackwell) — PROVEN + OPTIMIZED:** LLVM-21 + cargo-oxide
provisioned on gx10; 5 `#[kernel]`s working bit-exact on GB10 incl. the **Q4K dequant-matvec hot
kernel**, optimized 22.9× (1738µs→76µs via T=32 threads/row + atomic-reduction). Escape from hand-PTX
Blackwell JIT pain. See `memory/reference_cuda_oxide_rust_to_ptx.md`.

## 10-day work tracks (always ≥1 in flight per lane)

- **cuda-oxide adopt (gx10):** A/B the pure-Rust Q4K matvec vs trueno hand-PTX → if competitive,
  migrate trueno's hand-PTX hot kernels (dequant matvec, FFN fusion, attention) to `#[kernel]`.
- **Pillar-1 (robust beats only):** compute-bound sklearn algos with algorithmic edge, each
  measured on an OpenBLAS profile before shipping (NaiveBayes family done; assess LDA/QDA, robust
  scalers via a trueno-NEON/AVX SIMD optimization that would beat OpenBLAS elementwise).
- **Pillar-4 (correctness):** more fail-closed classes that hold zero-false-positive on real models
  (quant-scale-near-zero candidate; adversarially vet each via a scout, as with #2039).
- **Pillar-2/3 (PyTorch / Unsloth):** scout falsifiable beat candidates (training-step correctness,
  QLoRA pipeline parity) — measure-first.
- **Continuous hygiene:** keep main green; nightly beat-gate stays green on the CI host; merge-armed
  PRs; rebase chains; gx10 disk hygiene.

## Stop conditions (only these)
Operator says stop; an architectural decision needs a human; or a destructive/irreversible op needs
sign-off. Otherwise: never stop, never idle, both boxes + agents always working.
