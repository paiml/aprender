# 0.70.0 "Fast Train" — execution plan for EPIC #3998

Status: DRAFT for operator review. Written by aprender-19, 2026-09-23, on the
assignment of aprender-cf (planning cop), relaying the operator: "quorum decide. now
use quorum and claud plan for each epic". The plan answers the epic's open questions
**Q1–Q7** below. The quorum that reviews this file decides them; its receipt is
[`.quorum/PMAT-3998/`](../../.quorum/PMAT-3998/).
Nothing here creates an issue, moves a milestone or touches `release/*`. Rows become
child issues only after operator review.

## Exit bar (from the epic, made measurable)

| # | Bar | Definition used here | First-green proof |
|---|---|---|---|
| X1 | freeze → publish ≤ 4 h, measured on 0.70 itself | **freeze** = the timestamp of the commit that cuts `release/0.70.0*` (the T-5 reconcile starts from it). **publish** = the later of: `aprender` 0.70.0 visible on crates.io (`cargo search aprender --limit 1` reports 0.70.0), and the GitHub release object's `publishedAt`. Both are recorded in the train ledger by the automation, never retyped. | the 0.70.0 train ledger carries both timestamps, and `scripts/check_release_train_duration.sh` (FT-10) computes ≤ 4 h from them, and exits non-zero at > 4 h |
| X2 | lock idle-GPU fraction falls by > half | #3986's method: while `/tmp/apr-gpu.lock` is held, sample `nvidia-smi --query-gpu=utilization.gpu` at 1 Hz; idle fraction = samples at ≤ 5 % / all samples. The baseline window and the 0.70 window are each a full pre-freeze sweep on lambda | `scripts/gpu_lock_idle_sample.sh` (FT-2) writes one JSON per window; the 0.70 value ≤ ½ × the baseline |

**Baselines, measured, with their source:**
- **X2 baseline: ≈ 97 % idle GPU while the lock was held.** One 25-min window on lambda, 2026-09-23 09:20Z (#3986, aprender-cf); 6 jobs queued behind it. **It is one sample.** FT-2's first action re-measures it over a whole sweep, and that number replaces this one.
- **Queueing share of cell time ≈ 80 %.** In the 09:23–09:29Z window, yoga's idle sm_89 card finished 3 cells at 0.7–3.3 min each, while lambda finished 0 in 45 min behind the suite (#3986 comment).
- **X1 baseline: [U].** 0.69.1 froze at 13:00Z 2026-09-23 and has not published. Its freeze→publish time is the baseline, recorded from the 0.69.1 ledger the moment it publishes.
  - The only complete prior datum is 0.69.0: the release PR #3698 was opened 12:58Z and merged 13:47Z, and the GitHub release was published 13:56:55Z (2026-09-21).
  - Its freeze time is not recorded anywhere this plan could find. That gap is itself FT-10's reason to exist.

## Rows

Every row has a `done_when` that names no person, a baseline, and a **first-green proof**: a command or artifact that fails when the row is not done. "Green" never means "no error printed".

### FT-1 · GPU lock scoped to GPU work (#3986 P1)
- **done_when:** gpu-q refuses, with a non-zero exit and before taking the lock, a locked command whose argv is a bare `cargo test` / `cargo build` / `cargo nextest run` without a prior `--no-run` build. The fleet's CUDA test recipe is:
  1. `cargo test --no-run` outside the lock;
  2. then the prebuilt test binary under the lock, filtered to CUDA tests.
- **baseline:** gpu-q accepts any argv. The lock was held 1.5 min for compiling inside a 25-min suite (#3986).
- **first-green proof:**
  - a gpu-q case-table row that runs `gpu-q -- cargo test -p x` must exit with the refusal code, and must never have touched the lock file (flock `-n` probe from a sibling process);
  - its mutant (the refusal deleted) must be killed.
- **ownership boundary:** gpu-q is fleet state (infra/forjar). The aprender side is the recipe and a `scripts/gpu_test.sh` that implements it.

### FT-2 · Lock idle sampling (the X2 instrument)
- **done_when:** `scripts/gpu_lock_idle_sample.sh` samples utilization at 1 Hz while the lock is held, and writes `{host, window, samples, idle_fraction, holders[]}`.
  - A window with 0 samples is an error, never 0 % idle.
  - `holders[]` comes from the lock's `fuser`, so the cause of idle time is attributable.
- **baseline:** no instrument exists; #3986's 97 % was taken by hand.
- **first-green proof:**
  - a fake `nvidia-smi` emitting a scripted utilization series gives the exact fraction;
  - a 0-sample window exits non-zero.

### FT-3 · CPU work off the lock (#3986 P3)
- **done_when:** `scripts/model_ladder.sh`'s CPU legs (`--no-gpu`, `-ngl 0 -dev none`) and its sha256 pass run without `apr_locked`/gpu-q. GPU legs are unchanged.
- **baseline:** every ladder leg, including sha hashing of multi-GB files, runs under the lock (#3986 P3).
- **first-green proof:** a ladder self-test row with a fake lock asserts two things:
  - the CPU leg and the hash run while a sibling holds the lock (they would block otherwise);
  - a GPU leg still waits.

  Mutant: the CPU leg wrapped back in the lock ⇒ that row is RED.

### FT-4 · Idle same-arch pre-screen, a standard pre-freeze step (#3986 P5)
- **done_when:**
  - The release train's pre-freeze stage runs the **risk cells** on an idle card of the same compute capability as each release host (sm_89 for lambda, sm_121 for gx10). The risk cells are:
    - non-Q4_K_M quants;
    - `.apr`;
    - hybrid and new architectures;
    - anything the previous sweep had RED.
  - It uses the release host's exact binary (sha256-matched) and a private model dir.
  - A pre-screen RED is reproduced on the release host before it is ticketed.
  - A pre-screen GREEN is never release evidence.
- **baseline:** done once by hand (#3986 comment). yoga: 3 cells green in 6 min, while lambda: 0 cells in 45 min.
- **first-green proof:**
  - the pre-freeze stage's receipt names the pre-screen host, its `compute_cap`, and the binary sha;
  - the stage refuses when the pre-screen card's compute capability differs from the release host's;
  - the stage refuses when the binary sha differs.

  A must-RED fixture exists for each.

### FT-5 · Sharded sweep across hosts (#3986 P5)
- **done_when:** the ladder sweep partitions its cells across the hosts that can run them, and one reducer merges the shards into ONE receipt per release host. The hosts are lambda, gx10 and yoga (CUDA), plus the Mac host (CPU/Metal legs only; it has no CUDA).
  - Each cell is measured on the host its evidence must come from.
  - Only CPU legs, and cells whose evidence is architecture-independent, may move.
- **baseline:** one host runs its whole column serially.
- **first-green proof:** the reducer refuses:
  - a missing shard;
  - a duplicated cell;
  - a cell measured on a host whose arch does not match the column it fills.

  Three must-RED fixtures. A real 0.70 sweep receipt shows > 1 host contributing.

### FT-6 · VRAM-budget sharing, pilot only (#3986 P4)
- **done_when:**
  - On yoga, a known-green model run executes concurrently with a CUDA correctness suite as separate processes (NOT MPS: an MPS client fault can kill the others).
  - The model run's outputs are byte-identical to a solo run of the same binary and model.
  - The suite's verdicts are identical to a solo run.
  - Benchmarks and A/Bs stay exclusive.
- **baseline:** a binary mutex; no concurrent run has ever been compared.
- **first-green proof:**
  - the pilot receipt carries solo and concurrent output sha256s side by side, and they are equal;
  - the harness is RED when they differ, proven by one planted perturbation.

  The pilot does NOT change the lock's policy (see Q4).

### FT-7 · Sweep hygiene
- **done_when:**
  - (a) one sha256 pass per sweep: each file is hashed once and reused by every leg;
  - (b) the serve probe keeps its own evidence (#3949, already merged; this row is its regression guard staying wired);
  - (c) the serve health wait's clock starts after the lock is acquired, never while the cell queues for it. This is the gx10 q4k.apr false-RED shape of 2026-09-23.
- **baseline:**
  - (a) hashed per leg;
  - (c) the health wait counts queue time: the q4k.apr cell went RED while it was still waiting for the lock.
- **first-green proof:**
  - (a) a ladder self-test counts `sha256sum` invocations per file, and requires exactly 1;
  - (c) a fixture holds the lock for longer than the health-wait ceiling, then releases it, and the cell must be GREEN.

  The mutant (the clock started before the lock) turns (c) RED.

### FT-8 · Dark test targets into CI (#3984, #3982)
- **done_when:** every integration-test target is compiled by some CI job, or deleted.
  - "Compiled" includes the targets behind a `cuda`/`gpu` feature: a CPU-only `cargo check -p aprender-serve --features cuda --tests` job compiles them with no device.
  - Every target that needs no device also runs.
  - `driver_cuda_gguf` compiles: #3984's fields fixed via `..Default` so the next field addition cannot break it.
  - `cuda_combinatorial_coverage::test_tqa023` derives its expectation from the admitted-type registry (#3982).
- **baseline (measured at `origin/main` 49fe19c28):**
  - **28** `tests/*.rs` targets are gated at file level on `feature = "cuda"|"gpu"` (20 in aprender-serve, 8 in aprender-gpu).
  - `cargo check --workspace --all-targets --locked` (ci.yml:840) compiles no feature-gated target.
  - `[[test]] required-features` appears 96 times in aprender-serve/Cargo.toml and 36 times in aprender-train/Cargo.toml. How many of those are dark is FT-8's first measurement.
- **first-green proof:** `scripts/check_test_targets_compiled.sh`, a census wired into a workflow per `check_guards_are_wired.sh`.
  - It enumerates every test target from `cargo metadata` (never a hand list), with its `required-features`.
  - It fails when a target is neither compiled by a named CI job nor listed in a shrink-only baseline.
  - The baseline starts at the measured dark count, and can only go down.

### FT-9 · Fail-fast release gates (#3892, #3974)
- **done_when:**
  - (a) #3892's follow-ups:
    - a guard simulates the publish cascade's drain over the current `cargo metadata` graph and requires completion;
    - `TIERS[]` is corrected to the real order: 0 strictly-later-tier edges, from 44.
  - (b) #3974: each of the release surface's verdict lines is classified, and every line is closed. Classes: gate (exits non-zero), informational (prints no verdict word), or named post-publish obligation.
    - `apr qa gpu_speedup` propagates the generation error;
    - `apr validate --quality` F exits non-zero or is documented as not a gate;
    - model_ladder.sh, check_multiplatform_dogfood.sh, crux_inference_dogfood.sh and `make publish`'s post-publish check each gain a must-RED self-test.
- **baseline:**
  - (a) the drain was fixed in `cab272e5d`; there is no simulation guard; 44 strictly-later-tier edges;
  - (b) `grep -cE '\b(WARN|SKIP|REPORT|INFO|MANUAL)\b' scripts/dogfood.sh` = **58** lines at 49fe19c28. #3974 lists 22 of them as verdicts; the rest are to be classified. There are 141 `scripts/check_*.sh`.
- **first-green proof:**
  - (a) the drain guard over a planted strictly-later cycle must be RED;
  - (b) a ratchet file of unclassified verdict lines, shrink-only, whose zero is required at freeze;
  - each new self-test proven RED on its own mutant.

### FT-10 · The train measures itself
- **done_when:** the train ledger records `freeze_at` (the release branch cut) and `published_at` (crates.io + the GH release, the later of the two) from the automation. `scripts/check_release_train_duration.sh` computes X1 and fails a > 4 h release in the post-publish stage.
- **baseline:** no ledger field carries a freeze time; 0.69.0's is not recoverable (see baselines).
- **first-green proof:** a fixture ledger with a 4 h 01 min span must be RED; one with a missing `freeze_at` must be RED, never "unknown ⇒ pass".

### FT-11 · 0.69.1 known REDs, carried with "no defer" (subject to operator confirmation; see Q1)
- **#3951 Qwen3.5-0.8B-IQ4_XS:** already ruled by the operator ("proceed with (a)"): a permanent `RED-MODEL` cell, attribution re-proven every sweep (#3957 F9). The row is F9's oracle landing and running every sweep; the model file is never "fixed".
  - done_when: F9's three must-RED cases pass in the ladder judge's case table, and the cell prints `RED-MODEL` with its oracle and control in the 0.70 capability report.
- **#3987 qwen3moe chat/serve/code on CUDA:** done_when: chat, serve and code on the qwen3moe file are GREEN with `used_gpu=true` and `fell_back=false`. Its must-RED: each verb is RED on fallback.
- **Low-bit admission, #3963 IQ3_XXS / #3953 IQ2_S / #3960 Q2_K:** done_when: each type's device A/B at every (k,n) shape the model uses, plus real bytes, ≤ 1e-5 per row, NaN-prefilled, planted faults RED first; the whitelist flips in the order #3950 → #3953 → #3963/#3960.
- **baseline:** all open at 0.69.1's freeze (13:00Z 2026-09-23). Which of them land in 0.69.1 is settled by 0.69.1's own receipt. This row carries only what 0.69.1 did not close.

### FT-12 · Debt-ratchet slice 1 (#3997: 0.70 of 0.70–0.74, plus 0.75's 6th)
- The slice is sized by aprender-cb, and this plan cites it rather than inventing it: "5 releases to clear 80%" with a 6th slice for 0.75 (operator, #3997 comment).
  - `done_when`: the 0.70.0 thresholds for pillars A–D exactly as aprender-cb's sizing states them. Pillars: A coverage, B pv depth, C ONT-001 rows bound, D backlog.
  - Each pillar's gate refuses a level below 0.69.x's.
- **baselines (from #3997, to be re-derived by aprender-cb's commands):**
  - A: 88.78 % line (`COV_FLOOR := 88`);
  - C: 13/27 ONT rows bound;
  - D: 715 open issues (121 with no milestone), 57 open PRs, 2,594 remote branches (2026-09-23 09:50Z).
- **first-green proof:** each pillar's gate fails a fixture one unit below its 0.70 threshold.
- **[U] until aprender-cb's sizing lands.** The thresholds are copied into this row verbatim in a follow-up commit on this PR.

## Out (epic's own list)
New backends and silicon (0.71, #3994), and serve/agentic features (0.72, #4000). FT-6 does not change the lock policy.

## Open questions — the quorum decides these

- **Q1** Should FT-11 (the 0.69.1 known REDs) be a must-carry of 0.70, or of 0.71 "Don't Leave Behind"? The epic says "subject to operator confirmation". The quorum recommends; the operator confirms.
- **Q2** Where is FT-1's refusal enforced?
  - (a) in gpu-q itself, fleet-wide via forjar;
  - (b) in an aprender wrapper only (`scripts/gpu_test.sh`), with gpu-q unchanged;
  - (c) both.

  A bare-flock user (seen today: `gpu_exclusive_run.sh`, `apr_locked`) bypasses gpu-q entirely.
- **Q3** Is X2's "idle fraction" the right bar? The alternative is the jobs-waiting × idle-time integral (wasted GPU-minutes), which also rewards shorter holds.
- **Q4** Does FT-6 (the VRAM-budget pilot) belong in a *speed* release at all, given that its benefit only lands if the policy changes afterwards, and a failed pilot proves nothing about speed?
- **Q5** For FT-5, may a CPU leg be measured on a different host from the one whose column it fills, when the binary sha and ISA match? Or is every leg host-bound?
- **Q6** Is ≤ 4 h (X1) achievable with a full two-host ladder, given today's per-cell time once queueing is removed? Or should X1 be split: ≤ 4 h for the automated path, with an explicit budget for fix-and-re-sweep cycles?
- **Q7** Order: which rows must land BEFORE 0.70's own freeze so that X1/X2 can be measured on 0.70 itself? The draft says FT-1, FT-2, FT-3, FT-7 and FT-10 at least.
