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
- **X1 baseline: the history, reconstructed (aprender-cb, 2026-09-23), with the 0.69.1 figure to be read from its ledger once it publishes.** No ledger records a freeze time, so this uses a proxy: T-0 = the release PR was **opened** (`gh pr list --search '"0.N.M" in:title'`), and publish = `aprender` `created_at` on crates.io (`https://crates.io/api/v1/crates/aprender/versions`).

  | Release | T-0 (release PR opened) | crates.io | Span |
  |---|---|---|---|
  | 0.67.0 | #3145, 09-12 10:51Z | 09-13 08:36Z | **21.7 h** |
  | 0.68.0 | #3406, 09-16 23:05Z | **never published** | ∞ |
  | 0.68.1 | #3450, 09-17 12:24Z | 09-17 22:40Z | **10.3 h** |
  | 0.68.2 | #3498, 09-18 18:55Z | 09-20 07:56Z | **37.0 h** |
  | 0.69.0 | #3698, 09-21 12:58Z | **never published** (tagged 09-21 13:56Z; still absent at 09-23 ~11:30Z) | ∞ |

  Of the last five trains, 3 reached crates.io, taking between 10.3 h and 37.0 h. X1's ≤ 4 h is a **3–9× cut** on the best of them.
  - 0.69.0's GitHub release was published 13:56:55Z (2026-09-21), but its crates are **not** on crates.io. So X1 must define "publish" as the **later** of the two, as it does. A GitHub release alone is not a publish.
  - No train's freeze time is recorded anywhere. That gap is itself FT-10's reason to exist.

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
  - **in-tree bare-flock callers are in scope** (quorum fix): every caller that takes `/tmp/apr-gpu.lock` without gpu-q is enumerated from the tree (`model_ladder.sh`'s `apr_locked`, `gpu_exclusive_run.sh` (not in this repo: a fleet script outside the tree), …). Each one either routes through the same refusal or is listed in a shrink-only census. A new bare `flock /tmp/apr-gpu.lock` caller turns the census RED.
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
- **baseline (corrected by the quorum, re-read by aprender-cb at `9c17b6560`):** `apr_locked` (`scripts/model_ladder.sh:77`) wraps **`apr` calls only**. `sha256sum` (`:287`, `:302`) already runs **outside** the lock. What holds the lock needlessly is the CPU `apr run` legs: `:222` passes `$flag` through `apr_locked` for every backend. The sha pass costs serial wall time (FT-7a), not lock time.
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
  - (b) (quorum fix) the #3949 regression case stays in the ladder self-test: a failed serve probe whose evidence file is deleted before the receipt is written must still carry its evidence inline. Reverting #3949's change turns that case RED;
  - (c) a fixture holds the lock for longer than the health-wait ceiling, then releases it, and the cell must be GREEN.

  The mutant (the clock started before the lock) turns (c) RED.

### FT-8 · Dark test targets into CI (#3984, #3982)
- **done_when:** every integration-test target is compiled by some CI job, or deleted.
  - "Compiled" includes the targets behind a `cuda`/`gpu` feature: a CPU-only `cargo check -p aprender-serve --features cuda --tests` job compiles them with no device.
  - Every target that needs no device also runs.
  - `driver_cuda_gguf` compiles: #3984's fields fixed via `..Default` so the next field addition cannot break it.
  - `cuda_combinatorial_coverage::test_tqa023` derives its expectation from the admitted-type registry (#3982).
- **baseline (measured at `origin/main` 49fe19c28):**
  - **44** `crates/*/tests/*.rs` targets are gated at file level on `feature = "cuda"|"gpu"` (`grep -rlE '#!\[cfg\(.*feature *= *"(cuda|gpu)"' crates/*/tests/*.rs | wc -l` at `9c17b6560`). The draft's 28 was an undercount, corrected by the quorum and re-derived.
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
    - `apr validate --quality` F exits non-zero (quorum fix: the "or is documented as not a gate" escape is removed);
    - model_ladder.sh, check_multiplatform_dogfood.sh, crux_inference_dogfood.sh (absent on `main`: this clause applies only if it lands) and `make publish`'s post-publish check each gain a must-RED self-test.
- **baseline:**
  - (a) **the drain fix `cab272e5d` is on NO remote branch** (`git for-each-ref --contains cab272e5d refs/remotes/origin` is empty). `scripts/cascade-publish.sh` on `main` @ 49fe19c28 still runs exactly one `=== RETRY ROUND ===` (`grep -c` = 1). #3892 is OPEN in 0.69.1. So FT-9(a)'s first job is to get the fix onto `main` (and onto `release/0.69.1*` if 0.69.1 publishes with this script). There is no simulation guard; 44 strictly-later-tier edges;
  - (b) `grep -cE '\b(WARN|SKIP|REPORT|INFO|MANUAL)\b' scripts/dogfood.sh` = **58** lines at 49fe19c28. #3974 lists 22 of them as verdicts; the rest are to be classified. There are 141 `scripts/check_*.sh`.
- **first-green proof:**
  - (a) the drain guard over a planted strictly-later cycle must be RED;
  - (b) a ratchet file of unclassified verdict lines, shrink-only, whose zero is required at freeze;
  - each new self-test proven RED on its own mutant.

### FT-10 · The train measures itself
- **done_when:** the train ledger records `freeze_at` (the release branch cut) and `published_at` (crates.io + the GH release, the later of the two) from the automation. `scripts/check_release_train_duration.sh` computes X1. **Quorum fix:** a post-publish failure cannot un-publish, so the gate runs at **T-3, before the tag**, on the elapsed time so far plus the measured T-4 cascade time (0.68.1: 41 min, ledger `cascade.wall_s` = 2458). After publish, it records the final figure.
- **baseline:** no ledger field carries a freeze time; 0.69.0's is not recoverable (see baselines).
- **first-green proof:** a fixture ledger with a 4 h 01 min span must be RED; one with a missing `freeze_at` must be RED, never "unknown ⇒ pass".

### FT-11 · 0.69.1 known REDs, carried with "no defer" (subject to operator confirmation; see Q1)
- **#3951 Qwen3.5-0.8B-IQ4_XS:** already ruled by the operator ("proceed with (a)"): a permanent `RED-MODEL` cell, attribution re-proven every sweep (#3957 F9). The row is F9's oracle landing and running every sweep; the model file is never "fixed".
  - done_when: F9's three must-RED cases pass in the ladder judge's case table, and the cell prints `RED-MODEL` with its oracle and control in the 0.70 capability report.
- **#3987 qwen3moe chat/serve/code on CUDA:** done_when: chat, serve and code on the qwen3moe file are GREEN with `used_gpu=true` and `fell_back=false`. Its must-RED: each verb is RED on fallback.
- **Low-bit admission, #3963 IQ3_XXS / #3953 IQ2_S / #3960 Q2_K:** done_when: each type's device A/B at every (k,n) shape the model uses, plus real bytes, ≤ 1e-5 per row, NaN-prefilled, planted faults RED first; the whitelist flips in the order #3950 → #3953 → #3963/#3960.
- **baseline:** all open at 0.69.1's freeze (13:00Z 2026-09-23). Which of them land in 0.69.1 is settled by 0.69.1's own receipt. This row carries only what 0.69.1 did not close.
- **first-green proof (quorum fix):** per carried issue, the ladder cell for that exact model file is GREEN on lambda **and** gx10 in a receipt from the release SHA. The same cell on the 0.69.1 receipt is the RED control. A cell with no RED control on record is not evidence the fix did anything.

### FT-12 · Debt-ratchet slice 1 (of 5 to the ≥80% target at 0.74, then the 0.75 sixth slice) (#3997; sized by aprender-cb in PR #4003, PROPOSED until the operator rules on #4003 §6)
The 0.75 sixth slice exists by operator ruling (#3997 comment, 10:46Z: *"ALL releases in .7 have some rachet"*). Every
release refuses a level below the previous tag's. Baselines were measured 2026-09-23 on `main` @ 49fe19c28 (commands:
#4003 §7). **They supersede #3997's table**: coverage is 88.20%, not 88.78%, and there are 353 live remote branches, not 2,594.

| Pillar | Unit | Baseline | **0.70 threshold** | Gate |
|---|---|---|---|---|
| A | P₀ line coverage in bp (regex **and** excluded-file list pinned) | 8,819 (run 33245815502, 08-29; 0 green in 25 days since) | **≥ 8,928**, plus one green coverage-nightly on the release SHA | `make coverage` with `COV_FLOOR_BP` (**does not exist yet**, the first child issue) |
| B-1 | E2 call sites, `pv coverage --binding contracts/aprender/binding.yaml --enforcement <crate>` summed; total call sites never < 510 | 115 (of 510: E0 267 / E1 128 / E2 115) | **≥ 179** | pv enforcement sum at T-5 |
| B-2 | contracts with obligations > 0 and falsifiers = 0 (`pv coverage`) | 19 | **≤ 15** | same |
| B-4 | `pv lint contracts/` as a required PR check | reachable from no workflow | **wired and green** | ci.yml (**a workflow change: operator check-in**) |
| C | ONT-001 rows bound (infra `precondition-lint.sh --ledger`) | 13/27 | **≥ 16** | the lint's `bound=` at the tag |
| D-1 | open issues > 24 h old and not in an open release milestone ≥ the current one | 428 | **≤ 342** | `check_reconcile.sh` R6 (new) |
| D-2 | PRs > 7 d old with no author activity in 72 h | 19 | **≤ 15** | `check_reconcile.sh` R7 (new) |
| D-3 | live remote branches with no open PR (excluding `main` and `release/*`) | 297 | **≤ 237** | count at T-5 |

- **first-green proof:** each gate passes on the live repo at its threshold and fails at threshold − 1 unit (a count
  pillar) or + 1 unit (a ceiling pillar). R6/R7 do not exist yet, so their proofs are step-2 acceptance tests
  (#4003 §5). **No proof has run.**
- **B-3 (bound equations) and the §6 decisions of #4003 are the operator's.** This row carries no B-3 number.

## Out (epic's own list)
New backends and silicon (0.71, #3994), and serve/agentic features (0.72, #4000). FT-6 does not change the lock policy.

## Open questions — the quorum decides these

- **Q1** Should FT-11 (the 0.69.1 known REDs) be a must-carry of 0.70, or of 0.71 "Don't Leave Behind"? The epic says "subject to operator confirmation". The quorum recommends; the operator confirms.
- **Q2** Where is FT-1's refusal enforced?
  - (a) in gpu-q itself, fleet-wide via forjar;
  - (b) in an aprender wrapper only (`scripts/gpu_test.sh`), with gpu-q unchanged;
  - (c) both.

  A bare-flock user (seen today: `gpu_exclusive_run.sh` (not in this repo: a fleet script outside the tree), `apr_locked`) bypasses gpu-q entirely.
- **Q3** Is X2's "idle fraction" the right bar? The alternative is the jobs-waiting × idle-time integral (wasted GPU-minutes), which also rewards shorter holds.
- **Q4** Does FT-6 (the VRAM-budget pilot) belong in a *speed* release at all, given that its benefit only lands if the policy changes afterwards, and a failed pilot proves nothing about speed?
- **Q5** For FT-5, may a CPU leg be measured on a different host from the one whose column it fills, when the binary sha and ISA match? Or is every leg host-bound?
- **Q6** Is ≤ 4 h (X1) achievable with a full two-host ladder, given today's per-cell time once queueing is removed? Or should X1 be split: ≤ 4 h for the automated path, with an explicit budget for fix-and-re-sweep cycles?
- **Q7** Order: which rows must land BEFORE 0.70's own freeze so that X1/X2 can be measured on 0.70 itself? The draft says FT-1, FT-2, FT-3, FT-7 and FT-10 at least.

## Quorum record: decision quorum, 2026-09-23 (aprender-cb)

aprender-19's earlier 3-lane grillme (`.quorum/PMAT-3998/`) was **not counted**, because every lane exited 3. This is
the second quorum.

**Lanes (ADVISORY: single family, all gemini):** gemini-3.1-pro-high, gemini-3.8-flash-high, gemini-3.7-flash-high,
all returning PASS-with-changes. gpt-oss-120b-medium was tried once and returned 429 (reset about 95 h). 3/3 exited 3
on foreign fleet ref motion, with every clone byte-identical and every tree witness verified. Conversations:
`6c6c787e`, `367fca8d`, `25769f6f`.

| Q | Decision (tally) | Plan now says |
|---|---|---|
| Q1 | **FT-11 is a must-carry of 0.70**, 2/3. Lane 2 would split: the low-bit issues and #3987 to 0.71, keeping the #3951 oracle in 0.70. aprender-19's quorum split the same way, 2/3 | 0.70 carries FT-11. **Two quorums disagree on this, so it goes to the operator.** 0.71's plan drops whatever 0.70 carries |
| Q2 | **(c) both gpu-q and the in-tree wrappers**, 3/3 (the same in both quorums) | FT-1 enforces in both |
| Q3 | **keep the idle fraction and also record the waiting integral**, 2/3 | X2 gates on the idle fraction; FT-2 also writes `Σ(jobs_waiting × idle_s)` |
| Q4 | **FT-6 out of 0.70**, 2/3 (3/3 in aprender-19's quorum) | FT-6 moves to 0.71 as a non-gating pilot |
| Q5 | **a CPU leg may run on another host when binary sha and ISA match**, 3/3 | FT-5 allows it, recording sha and ISA |
| Q6 | **split X1**, 2/3 (3/3 in aprender-19's): ≤ 4 h for the automated path, plus a declared budget for fix-and-re-sweep | X1 is split |
| Q7 | **FT-1, 2, 3, 7, 10** land before the 0.70 freeze, 3/3, plus **FT-9(a)** (the cascade fix is not on `main`) | the pre-freeze set is FT-1, 2, 3, 7, 9(a), 10 |

**Must-fix items applied:**
- FT-3's baseline is corrected (hashing is already outside the lock).
- FT-8's count is 44, not 28.
- The two absent scripts are marked.
- FT-10 is moved before the tag.
- FT-9(b)'s escape is removed.
- The FT-12 header is fixed.

**Must-fix items carried to step 2**, as the acceptance of each child issue:
- negative controls on the FT-4 anti-vacuity row and plan lines 56, 106, 138 and 161;
- proof that FT-2's sampler is invoked by the sweep.

**Not applied:** "open a ticket for the broken coverage-nightly". It is already #4003 A-5, the first 0.70 child
issue, and step 2 files it.

