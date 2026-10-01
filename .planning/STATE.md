---
gsd_state_version: "1.0"
milestone: v1.0
current_phase: 08
current_phase_name: "Laya Decision Model: Local Fine-Tune and Thin MCP Server"
status: executing
stopped_at: "Completed 08-33-PLAN.md (contract-hygiene repair, local commits, NOT pushed). Next: plan 08-34 (CI audit, push, maintainer approval, CI evidence)"
last_updated: "2026-09-29T21:46:01.311Z"
last_activity: 2026-09-28
last_activity_desc: Phase 08 execution started
state_head: 3b75bfd7450ffcc5d2ac052c3575152bac343207
progress:
  total_phases: 9
  completed_phases: 1
  total_plans: 126
  completed_plans: 125
milestone_name: milestone
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-07)

**Core value:** A small labeled dataset can produce an accurate, fast, reproducible classifier that trains and runs entirely through Aprender's native Rust and APR lifecycle.
**Current focus:** Phase 08 — Laya Decision Model: Local Fine-Tune and Thin MCP Server

## Current Position

Phase: 08 (Laya Decision Model: Local Fine-Tune and Thin MCP Server) — EXECUTING
Plan: 33 of 34 executed (08-01..08-33; 08-25 ran before 08-24 by wave order; Wave 23 done) — gap round 08-19..08-32 executed (`--gaps-only`, waves 17-22, sequential on the main tree). 08-32's must-have "workspace-test = success on the pushed head" is UNMET; plan 08-33 (contract-hygiene repair, wave 23) is EXECUTED locally (committed, NOT pushed; upstream's hygiene gates green at their baselines, see 08-33-SUMMARY.md); plan 08-34 (CI audit + push + maintainer-approved CI evidence, wave 24, autonomous: false) is planned and NOT yet executed; Phase 8 is not complete
Status: Phase 08 gap round EXECUTED, phase NOT complete — 08-33 DONE locally (2026-09-29, local commits e465421c4..3b75bfd74, NOT pushed): 13 contracts repaired, formal_prose 1569 -> 1464 and contracts_without_valid_under 398 -> 386 (both at their shrink-only baselines, none touched), pv validate 0 of 1850 failed, graph regenerated from a clean clone (this working tree's untracked .claude/worktrees pollute pv extract), aprender-contracts-cli 394 passed 0 failed in a clean clone, laya-claims OK 133 rows; next 08-34 (CI-order audit, push, maintainer approval, CI evidence). 08-32 done with an unmet must-have: Task 1 `just laya-gap-regression` PASS (classes A-E + regression, pre-merge; real-artifact legs OK on the merged tree: laya-verify-suite max_abs 3.841e-6, laya-verify deploy_eligible true sha 24a44d7e seed 17); Task 2: history scrubbed (backup ref backup/pre-scrub-08-32, map 08-SCRUB-COMMIT-MAP.tsv), upstream main merged (0d24db9c5, 301 commits, 46 conflicts), draft PR paiml/aprender#4634 opened (head 30bc2baaa at the runs) — all its workflow runs are action_required (fork PR needs a paiml maintainer approval; pull-only access) and pr-review-quorum fails on a missing review receipt (paiml process). workspace-test has NOT run; locally upstream's shrink-only contract-hygiene gates fail (389 passed, 5 failed: formal vocabulary +105 of which Phase 8 25, valid_under +12, neon-blis-v1 / spectral-indices-v1 pv errors, stale contract graph, binding.yaml book page) and would stop CI before decide fragments 510/520. Owner decisions: "Repair all (new gap plan)" -> decided; planned as 08-33; "Defer to deferred-items" for the 17 SetFit CI legs (D-ITEM-08-32-C); "Keep one draft for CI" (#4634 stays one draft; split into reviewable PRs after phase verification). Evidence: 08-CI-RUN-EVIDENCE.json. Next: plan 08-33, maintainer approval, then Phase 8 verification
Phase 05 is PLANNED — 13 plans in 8 waves, verification passed, then REPLANNED 2026-08-17
against `05-REVIEWS.md` (codex + gemini). The replan is targeted, not from scratch: eight
consensus findings were incorporated and six of Gemini's were rejected with in-plan rationale
(its `f1_average_for_classes(&[1,2])` "off-by-one fix" would have INVERTED the phase's headline
metric — the ordered labels really are `["none","against","favor"]`). Structural changes worth
knowing before execution: (a) 05-06 gains a tracer task that must PROVE the LoRA
adapter-save → fresh-process-reload → ordered-probability-vector route before any 9B compute is
spent — the code confirms the gap (`ClassifyPipeline::from_apr` builds FRESH LoRA layers,
`forward_only_tokenized` returns `(loss, class)` not a probability vector); (b) 05-11 and 05-12
now depend on 05-10, so the 80 expensive cells cannot be generated before the gate that judges
them exists (waves 5→6, 6→7, 7→8); (c) cold latency and inference peak RSS move to a dedicated
fresh child process with a true kernel high-water mark on both platforms, and train peak becomes
a separate, separately-labelled field.
Last activity: 2026-09-28 — Phase 08 execution started

**Phase 04 UAT ran 2026-08-16 at `b3f816c25` (macOS/arm64): 12 tests, 12 passed, 0 issues —
see `04-UAT.md`.** Every gate was executed in-session, not read off a SUMMARY: codec 17,
reload 17, lifecycle 5, spawned CLI ladder 3+1, parity 20/20, artifact 86, eval 20, serve 11,
feature matrix (core 246 / train 311 / apr-cli delta 18→79 / serve 11) with RUN legs and
two-sided negatives, `pv validate` 0 errors 0 warnings, `contract-audit-phase4` 15/15 bound
with **zero BIND- lines**, `setfit-api-boundary` PASSED with an executed MUST-MATCH control,
`bashrs-lint-makefile` honest at bashrs rc=2 with 1 control-refuted false positive.

**`04-VERIFICATION.md` was stale and is now reconciled, not rewritten.** It was written at
`0fb47958f` and never re-run; the five gap-closure plans it triggered (04-18..04-22) landed
afterwards, and `roadmap.update-plan-progress` had already flipped Phase 4 to Complete on
summary-count parity alone — the SIXTH occurrence of that false-completion defect. Status is
now `gaps_acknowledged` with the original verdict and its measurements UNAMENDED, plus an
`## Acknowledged Gaps` section. Closed since the report: SC4+SC5's shared BIND-004 (04-19),
WR-08/WR-09 (04-18), WR-10 (04-20), both D-04-11-A production survivors (04-21), F-07 (04-22).

**Four items are OPEN by explicit human ruling, and Phase 4 does NOT close them:**

  1. **F-10** — `CALIBRATED_REGIMES` admits only the phase-3 MiniLM slice, whose 97-row
     vocabulary cannot compute `probe_unicode`, so no user-reachable path produces a
     `setfit-apr-v1`. SC1 and SC3 remain UNMET; SC2's policy has only run over a substituted
     encoder/head; SC4's parity fixture is synthetic. Deferred to **Phase 5**, which must
     calibrate on production `all-MiniLM-L6-v2` and add its fingerprint to
     `contracts/setfit-train-lifecycle-v1.yaml` by deliberate `pv diff`-flagged contract edit
     per Phase 3 D-10(c) — never an inline relaxation.

  2. **04-11 must-have 4** — per-crate cargo-mutants baselines and an aggregate adjusted score.
     1 of 4 crates attempted, interrupted at ~68 min, no score produced. Human ruled this out
     of the phase sequence and into a **standalone compute ticket** (≥10 h for 890 mutants is a
     compute-budget decision CLAUDE.md reserves for the human). **No mutation score exists for
     Phase 4 and none may be inferred.** Owner unassigned; successor to D-04-11-B.

  3. **WR-01** — the APR write path is not race-free. `fs::rename` in `atomic_write` replaces
     its destination unconditionally, so a file created between check and rename is destroyed
     without `--force`. 04-20 narrowed the window; closing it needs `O_CREAT|O_EXCL`.

  4. **SAFE-02 "in CI"** — the 16 setfit legs at `.github/workflows/ci.yml:378-397` (applied
     `57f7823ab`) have NEVER EXECUTED. All Phase 4 evidence is macOS/arm64 local. The arch
     asymmetry is two-way and only one direction has been measured: `make tier2` is known RED
     on arm64 with 24 clippy errors in arch-gated SIMD that X64-Linux CI never lints.

**Phase 03 is COMPLETE.** Its five `human_needed` items were adjudicated 2026-08-14 —
`03-HUMAN-UAT.md` is `status: complete`, `03-VERIFICATION.md` reads `passed`. The ROADMAP
progress table still showed `human_needed` for Phase 3 as of `b3f816c25`; that is table lag,
not an open item. The four Phase 3 code-review blockers below are retained as the historical
record of what was found and why it was not an implementation defect:

  - CR-01: Phase 3's ~2900 lines of unit tests and all seven trybuild cases run in NO tier and NO CI
    job. `setfit` is declared (aprender-train/Cargo.toml:79) but not default; no workspace member
    enables it; tier3's `cargo test --all` and CI's nextest both compile the module out. tier3 only
    `cargo check`s the feature (setfit-feature-matrix, Makefile:338) and runs 2 repro tests.

  - CR-02: `setfit_repro_recorded_matches_expected_replay` matches neither Makefile filter, so the
    one test separating "reproducible" from "correct" never runs — and libtest exits 0 on a
    zero-match filter, so both repro gates go vacuous on a rename while printing success.

  - CR-03: serde_json renders non-finite f64 as `null`, so UpdateEvidence::table_hash is not
    injective over inf/NaN; the contract precondition demanding a typed failure before hashing has
    no implementation.

  - CR-04: thresholds_match_the_contract checks entry COUNT then a SUBSET test, so widening
    CALIBRATED_REGIMES keeps it green while admitting uncalibrated runs.

**Two of 03-10's must_haves were unmet at the time of that verification, both blocked on the
same compute-budget decision CLAUDE.md reserves for the human. They were adjudicated in
`03-HUMAN-UAT.md` on 2026-08-14 — and Phase 4's mutation deferral (item 2 above) is the same
shape, now routed to a standalone compute ticket:**

  1. Scoped cargo-mutants adjusted score >= 85% — NOT RUN. 1181 mutants inventoried;
     ~44.6 h projected single-job. Two measured tooling blockers: `--in-place` conflicts
     with `--jobs` in cargo-mutants 25.3.1, and the plan's mandated `--timeout 20` kills
     the BASELINE (aprender-core's 14285-test binary has not finished LINKING in 20 s;
     `elapsed=20.050001083s -> Timeout`). No score is claimed.

  2. `make coverage` — deferred; needs the same uncontended target dir.

TRN-07 is deliberately left unchecked: its compile-time negatives landed, but no
out-of-crate or `apr` path exercises create_selection_lock -> mint_test_token -> grant,
so nothing demonstrates a user REACHING the lock.

REPAIRED BY HAND after the GSD state/roadmap handlers (recurring defect — 5th occurrence at
Phase 3, and the first where the damage was a FALSE COMPLETION CLAIM. **It recurred a 6th time
at Phase 4**: `roadmap.update-plan-progress` marked Phase 4 Complete on summary-count parity
while `04-VERIFICATION.md` still read `gaps_found`, and the two artifacts sat in contradiction
for 25 commits until the 2026-08-16 UAT reconciled them. The rule stands: diff STATE.md and
ROADMAP.md after EVERY handler call and revert any completion claim the verifier has not
earned; the handler's own `"complete": true` is not evidence of anything):

  - `state.begin-phase` wrote `Plan: 1 of 10` on a resume at plan 10, left `stopped_at` on
    the Phase 2 value, wrote the phase percentage (40) into a field the body renders as a
    plan percentage, and left Phase 2's `human_needed` sentence reading as if it described
    Phase 3.

  - `roadmap.update-plan-progress 03 03-10 complete` then marked the WHOLE PHASE complete
    (`[x] Phase 3 ... (completed 2026-08-12)`, Progress table -> Complete) purely because
    summary_count reached plan_count — before the verifier ran and with two must_haves
    unmet. Reverted to `[ ]` / "Awaiting verification".
Phase 2's outstanding items live in `02-HUMAN-UAT.md` (status: partial, 4 human decisions
incl. the publish cascade) and are unchanged by these repairs.

Working branch: `gsd/phase-2-contract-gate` @ b3f816c25. Phases 2, 3 and 4 all ride this one
branch — see 02-01-SUMMARY.md for the branch/PR policy. **No PR has been opened yet; per
02-01 that is the human's call after the verifier runs.** This is now also why SAFE-02's
"in CI" clause is unproven: with no PR, the 16 setfit legs in ci.yml have never executed.

Progress: [████████████████████] 100% (50 of 50 PLANNED plans executed across phases 1-4;
Phase 5 is not yet planned, so this is NOT milestone completion. Phase 4 is executed and
UAT-passed but NOT CLOSED — `/gsd:secure-phase 04` has not run, and SC1/SC3 remain unmet
pending F-10 in Phase 5.)

## Performance Metrics

**Velocity:**

- Total plans completed: 9 (this milestone's execution log; Phase 1 predates metric capture)
- Average duration: ~1h40m
- Total execution time: ~14.8 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| Phase 02 P01 | 1 | 1h20m | 1h20m |
| Phase 02 P02 | 1 | ~35m | ~35m |

**Recent Trend:**

- Last 5 plans: 02-05 (~2h10m, 3 tasks, 15 files), 02-06 (~1h50m, 3 tasks, 6 files), 02-07 (~2h45m, 3 tasks, 11 files), 02-08 (~1h45m, 3 tasks, 15 files), 02-09 (~2h45m, 3 tasks, 6 files)
- Trend: the long plans are the ones that had to derive evidence independently — a second implementation, an induced mutation, a scaling measurement — rather than assert it. 02-09 matches 02-07's length for a different reason: it is the first plan whose evidence is a real CLI run against live pinned data rather than a test, and roughly 20 minutes of it was an ENOSPC stop-and-report (the phase's second; both were `target/debug/incremental` at ~25 GB)

*Updated after each plan completion*
| Phase 02 P03 | 55m | 3 tasks | 6 files |
| Phase 02 P04 | ~50m | 2 tasks | 15 files |
| Phase 02 P05 | ~2h10m | 3 tasks | 15 files |
| Phase 02 P06 | ~1h50m | 3 tasks | 6 files |
| Phase 02 P07 | ~2h45m | 3 tasks | 11 files |
| Phase 02 P08 | ~1h45m | 3 tasks | 15 files |
| Phase 02 P09 | ~2h45m | 3 tasks | 6 files |
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 06 P01 | 37 min | 2 tasks | 40 files |
| Phase 06 P02 | 20 min | 3 tasks | 4 files |
| Phase 06 P03 | 24 min | 2 tasks | 3 files |
| Phase 06 P04 | 23 min | 3 tasks | 4 files |
| Phase 06 P05 | 46 min | 4 tasks | 13 files |
| Phase 06 P06 | 49 min | 3 tasks | 8 files |
| Phase 06 P07 | 1h 54m | 3 tasks | 10 files |
| Phase 06 P08 | 32 min | 3 tasks | 5 files |
| Phase 06 P09 | 25 min | 3 tasks | 20 files |
| Phase 06 P10 | 23 min | 3 tasks | 5 files |
| Phase 06 P11 | 1h 21m | 2 tasks | 5 files |
| Phase 06 P12 | 18 min | 3 tasks | 4 files |
| Phase 06 P13 | 20 min | 3 tasks | 3 files |
| Phase 06 P14 | 2h 5m | 3 tasks | 7 files |
| Phase 06 P15 | 3h 5m | 3 tasks | 6 files |
| Phase 06 P16 | 96 min | 3 tasks | 14 files |
| Phase 06 P17 | 31 min | 3 tasks | 6 files |
| Phase 05 P11 | session | 3 tasks | 17 files |
| Phase 05 P12 | 6h 56m | 3 tasks | 126 files |
| Phase 05 P13 | 4h 0m | 3 tasks | 8 files |
| Phase 05 P15 | 113 min | 3 tasks | 6 files |
| Phase 05 P16 | 61 min | 3 tasks | 6 files |
| Phase 05 P17 | 71 min | 3 tasks | 11 files |
| Phase 06.1 P01 | 95 | 3 tasks | 14 files |
| Phase 06.1 P04 | 78m | 3 tasks | 1 files |
| Phase 06.1 P08 | 1h 20m | 3 tasks | 7 files |
| Phase 08 P01 | 17min | 3 tasks | 8 files |
| Phase 08 P02 | 15min | 2 tasks | 27 files |
| Phase 08 P03 | 26min | 2 tasks | 10 files |
| Phase 08 P04 | 24min | 2 tasks | 16 files |
| Phase 08 P05 | 38min | 3 tasks | 19 files |
| Phase 08 P06 | 21 min | 2 tasks | 11 files |
| Phase 08 P07 | 24min | 3 tasks | 14 files |
| Phase 08 P08 | 42min | 2 tasks | 11 files |
| Phase 08 P09 | 45 min | 3 tasks | 15 files |
| Phase 08 P10 | 22 min | 2 tasks | 4 files |
| Phase 08 P11 | 3min | 2 tasks | 2 files |
| Phase 08 P13 | 11min | 3 tasks | 6 files |
| Phase 08 P14 | 18 min | 3 tasks | 12 files |
| Phase 08 P15 | 36min | 3 tasks | 18 files |
| Phase 08 P16 | 43min | 2 tasks | 3 files |
| Phase 08 P18 | 6 min | 2 tasks | 3 files |
| Phase 08 P12 | 86min | 3 tasks | 14 files |
| Phase 08 P19 | 24 min | 3 tasks | 4 files |
| Phase 08 P20 | 15min | 3 tasks | 5 files |
| Phase 08 P21 | 50min | 3 tasks | 7 files |
| Phase 08 P22 | 102min | 3 tasks | 7 files |
| Phase 08 P23 | 17min | 3 tasks | 7 files |
| Phase 08 P25 | 12min | 3 tasks | 3 files |
| Phase 08 P24 | 16min | 3 tasks | 8 files |
| Phase 08 P26 | 41min | 3 tasks | 9 files |
| Phase 08 P27 | 55min | 3 tasks | 12 files |
| Phase 08 P29 | 25 min | 3 tasks | 9 files |
| Phase 08 P28 | 22min | 3 tasks | 7 files |
| Phase 08 P30 | 3h50m | 3 tasks | 17 files |
| Phase 08 P31 | 2h14m | 3 tasks | 13 files |
| Phase 08 P32 | ~12h wall | 2 tasks (CI must-have unmet) | 10 files own + 46 merge conflicts |
| Phase 08 P33 | 39 min | 4 tasks | 19 files |

## Accumulated Context

### Phase 06 gap-closure round 3 (06-14..06-17) — planned 2026-09-06, scoped to the CLASS

Three `--gaps` rounds on one phase is a symptom, not bad luck. Every Phase 6 gap is one
instance of a single class: **the forecast door accepts a request whose cost or whose
semantics it never checks.** Rounds 1 and 2 each fixed exactly the probe the verifier
measured (06-10 the `cap` knob, 06-11/12 the holiday design-cost product, 06-13 the Poisson
sampler's domain), and the next adversarial pass found the next instance. `06-REVIEW.md`'s
CR-01 is *caused by* round 2's own fix: 06-13 correctly removed Knuth's saturation near
lambda=745, which had been an accidental hard cap, so the simulated changepoint count now
tracks an unbounded lambda — 0.157 s -> 2.334 s (14.8x) on a 1 132-byte accepted request,
over the same 2 s SC1 bar 06-11/06-12 were built to defend.

Round 3 closes the class instead. `contracts/forecast-tool-boundary-v1.yaml` gains a
`door_surface:` block enumerating every caller-settable knob and every cost axis, checked in
BOTH directions by `types::tests::every_request_knob_is_enumerated` (derived from
`schemars::schema_for!`, so a new struct field with no entry AND a phantom entry both go red)
and `every_cost_axis_names_a_real_bound`. Doing the enumeration by inspection rather than by
symptom found two axes no review finding pointed at: **C-07** `holidays[].name`, unenforced
anywhere, ~2 000 owned clones plus a byte-wise `sort_by` at `MAX_HOLIDAY_COLUMNS`; and
**C-08** the NeuralProphet path, which reads no budget at all (`FIT_BUDGET_SECS` is read at
exactly one site, `fit.rs:112`, inside `fit_prophet`) while `forecast.rs:359-362` runs 2-3
full `np::train` calls per request.

**C-07 has no structural maximum to measure** — `holidays[].name` is an unbounded `String`
and there is no `DefaultBodyLimit`/`max_body`/`content_length` in either crate, nor any
framing cap on stdio. It is closable only by a bound, never by a measurement.

**A declared-red gate spans waves 12-13, deliberately.** `no_cost_axis_is_pending` ships
FAILING at the close of 06-14, naming C-07 and C-08, and only 06-15 turns it green. Blast
radius, verified against the files: no git hooks exist and `make tier1/tier2` touch only the
root facade, but `make tier3` (`cargo test --all`, Makefile:313) and CI's `workspace-test`
(`.github/workflows/ci.yml:289`, which does NOT exclude `aprender-forecast`) both go red —
and `workspace-test` is a required check on protected `main`, so the window blocks the PR.
`-- --skip` is a libtest idiom with no nextest equivalent short of a filterset. **The
forbidden repair is deleting the assertion** — autonomous mode's re-run-CI licence and the
"repair, don't bypass" rule both point at it, and taking it restores the guard-that-cannot-
fail contradiction. The only legitimate repair is landing 06-15.

Statistical bars were re-derived, not inherited: adding lambda=3.0 weakened the pre-existing
mean bar to 2.45 sigma at N=20 000, so N moved to 60 000 (mean 4.24, variance 8.02, zero-mass
4.44 sigma at their weakest points). The review's suggested 1-2 % variance bar was REFUSED as
a 1.9-sigma flake, and the refusal survived two revision passes.

**This is the second phase to hit the stale-VERIFICATION trap** (see Phase 4's reconciliation
note above). `--gaps` feeds the planner `VERIFICATION.md` + UAT only — `NN-REVIEW.md` from
`/gsd-code-review` is not in its reading list — and nothing invalidates `VERIFICATION.md` when
gap plans execute. Phase 6's was dated 2026-09-07T00:26Z, predating all four of 06-10..06-13,
so a straight `--gaps` run would have replanned closed work and missed the live Critical.
Check the VERIFICATION timestamp against the newest executed gap-plan SUMMARY before trusting
it.

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Use five hard capability gates in dependency order; later phases retain earlier invariants as regression contracts.
- [Phase 1]: Use `aprender-core::autograd::Tensor` as the only SetFit graph and prove the pinned MiniLM path before exposing training.
- [Phase 3]: SetFit identity requires encoder-update evidence followed by a separate unique-row multinomial head fit.
- [Phase 4]: Only a closed, production-reloaded, parity-verified F32 APR may reach evaluation, CLI prediction, benchmarking, or serving.
- [Phase 5]: Claims require all 40 shot/seed cells and identical sampled IDs for SetFit and 9B LoRA.
- [Phase 2]: Phase 2 ships as exactly two PRs — the D-06 baseline PR (#1, stacked on the Phase 1 branch) and one Phase 2 PR from gsd/phase-2-contract-gate after wave 6; plans 02-02..02-09 commit to that single branch and open no PRs of their own.
- [Phase 2]: An as-is baseline landing is attested by SHA-256 recorded before staging and re-verified against the committed blobs via git cat-file — git status alone cannot prove byte-identity for an untracked file.
- [Phase 2]: Declared Kani harnesses must state in-contract that they are not executed and name an identically bounded runnable proptest; cargo-kani is absent repo-wide, including for Phase 1's setfit contract.
- [Phase 2]: $(CONTRACTS) in the Makefile is an explicit list, not a glob — a contract file in contracts/ is validated by nothing until it is appended there.
- [Phase 2]: `git status --porcelain` is rewritten by the rtk hook and prints "ok" on a clean path, so every porcelain-emptiness assertion in this phase must run through `rtk proxy`.
- [Phase 2]: cargo package -p apr-cli is KNOWN-RED from wave 2 until the publish cascade — and so is --no-verify; --no-verify skips the packaged-crate BUILD, not the manifest resolution that rewrites the path dep into a registry dep. Control-verified: removing the dep line makes the identical command exit 0 with 581 files.
- [Phase 2]: Both CB-510 guard scripts pass VACUOUSLY on macOS — they use GNU grep -P, BSD grep exits 2, the trailing || true swallows it, and they report 0 include!() files where the true count is 1768. Logged as D-ITEM-01; compensating direct evidence taken for the new crate.
- [Phase 2]: The binding registry ACCEPTS module_path: aprender_contrastive_data::* under target_crate: aprender (bound 0->1, BIND-001 24->23, no namespace complaint), so plan 02-08 proceeds as written. Traps: contract: must be the BARE filename (a ../ prefix parses and binds nothing) and status: accepts only implemented|partial|not_implemented|pending.
- [Phase 2]: D-04 is enforced by two POSITIVE checks — a dependency allowlist compared against the resolved cargo tree closure, and a src/-wide fs/net/path symbol ban with NO cfg(test) exemption. All four failure modes were induced, observed and reverted before the gate was trusted.
- [Phase 02]: 02-03: DatasetProfile carries an associated type Splits, so PreparedDataset<Compatibility> has no validation field at all rather than an Option+expect() — Makes D-19 structural rather than a runtime invariant, and makes 02-08's trybuild non-constructibility gate provable: proving a field is always None needs whole-program reasoning, proving it does not exist is a type error.
- [Phase 02]: 02-03: SplitFingerprintInput ordering is discharged by BTreeMap iteration order via Split::exact_hash_pairs(), not a caller-side sort — An ordering obligation left to callers can be silently omitted, and a wrong order yields a plausible-looking wrong digest. There is now no unsorted path to construct the input from.
- [Phase 02]: 02-04: reference fixtures record MEASURED pinned-setfit behavior in one family and Aprender's contracted closed forms in another; every number must agree three ways (measurement, closed form read out of sampler.py, contract literal) or the generator aborts
- [Phase 02]: 02-04: fixtures are keyed by fixture_id rather than by class layout, because 8_4_8 and 8_4_8_maxpairs100 share [8,4,8] and a layout-keyed map would silently drop one
- [Phase 02]: 02-05: permutation invariance is claimed for the SELECTION, not the semantic hash — the payload embeds a dataset fingerprint that digests each split's JSONL in ingest order, so a permuted file is legitimately a different dataset; the test asserts both halves
- [Phase 02]: 02-05: the selection AccessRecord carries the DATASET fingerprint, not the validation-split digest the plan named, so one ledger field does not mean two things depending on which code path wrote it; D-19 evidence is discharged via validation_witness().dataset_fingerprint_hex() and profile
- [Phase 02]: 02-05: RNG byte-encoding and ordered-selection goldens are ALGORITHM-DERIVED (independent Python from the contract text, cross-checked with shasum); the four payload.json goldens are capture-and-blessed byte forms, and the SUMMARY labels which is which
- [Phase 02]: 02-06: from_attested_bytes routes through Split::from_jsonl_bytes rather than re-using from_labeled_rows, which required extracting a pub(crate) from_validated_splits in prepared.rs — the plan forbade touching prepared.rs, but that constraint's stated reason (wave-4 parallelism with 02-05) had already expired, and without the change the attested path would not have gone through the byte-ingest door and the cross-path fingerprint test would have been a tautology
- [Phase 02]: 02-06: the setfit compatibility profile's merged rows now carry source_split compatibility_test instead of validation/test — an unavoidable consequence of D-19's distinct role plus the crate's role gate, declared by the schema_version 1->2 bump and pinned by a test; canonical row bytes are unchanged and proven byte-for-byte
- [Phase 02]: 02-06: write_outputs re-opens the directory it just wrote through PreparedDataset::from_attested_bytes and rolls back on rejection — without a production caller the attestation read path and the schema-version gate would have been test-only dead code
- [Phase 02]: 02-06: ContrastiveDataError gains UnsupportedNormalizationVersion (a String tag cannot ride in UnsupportedSchemaVersion's u32), with OBLIG-CPP-ERROR-TAXONOMY extended per the process error.rs itself mandates
- [Phase 02]: 02-06: every crate-level test command in the tweet-eval contract carries --lib, because the bare filter form emits a 'test result: ok' line from a suite that ran ZERO matching tests and would satisfy the expected_output grep vacuously
- [Phase 02]: 02-07: PairLayout is a separate PUBLIC type holding the sampler's whole retained state, because a Selection always carries shots_per_class in EVERY class and therefore cannot express the K = N all-singleton layout DATA-05 must survive — without it plan 02-08's headline capacity gate would be unwritable from outside the crate
- [Phase 02]: 02-07: negative_capacity accumulates the running prefix rather than (S^2 - sum n^2)/2 — same O(K), same quantity, but it overflows exactly when the RESULT does; a test pins the two derivations against each other wherever the second is computable
- [Phase 02]: 02-07: NoPairCapacity is checked BEFORE the default budget resolves, following the degenerate equation's invariant prose rather than its formula line, which would have reported ZeroBudget for a layout with no pair space at all
- [Phase 02]: 02-07: the pair manifest hash refuses a record whose selection_hash or budget does not describe the sampler it is handed; hashing stream X under record Y would be the exact failure the header-inside-the-digest design exists to prevent
- [Phase 02]: 02-07: both untrusted_pair_ingest and split_span_fail_closed are STACKED on the public validate_pair_records — the contract macro accepts two attributes (verified by compiling both forms), and a binding that names a private helper is harder to audit
- [Phase 02]: 02-07: the GSD SDK state handlers damaged STATE.md more severely than env note 7 records — state.update-progress reported percent 89 while writing 20 into the frontmatter and leaving the body bar at 83, state.record-metric flipped status to 'completed' mid-phase, state.advance-plan clobbered last_activity to a bare date and left stopped_at on the previous plan, and add-decision tagged all five entries [Phase ?]. Every field was repaired by hand and read back. Plans 02-08/02-09: run the handlers, then READ THE FILE and repair
- [Phase 02]: 02-08: the capacity bound is the contract's c*(examples+classes), not the plan's c*(examples+budget) — a bound containing the budget cannot express the same obligation's budget-independence clause and grows the allowance exactly when the pair space grows
- [Phase 02]: 02-08: the blocking tier3 binding gate is the SCOPED contract-audit-phase2; the repo-wide contract-audit prints 132 BIND-001 errors across 38 of 44 contracts and exits 0 anyway (its loop never reads the audit status) — logged as D-ITEM-04, not fixed
- [Phase 02]: 02-08: cargo-mutants ran at --timeout 20 rather than the planned 60, from a measured sample — 9 percent of mutants hang, and 48 hangs x 60 s alone exceeds the whole 2700 s wall budget; the full 529-mutant run then COMPLETED
- [Phase 02]: 02-08: 10 of 22 surviving mutants are individually justified as unobservable (disjoint-bit OR/XOR, union-by-size balancing, single-variant enums, an accessor no constructible Selection can make non-zero) and were re-run to confirm they still survive; 12 were killed and a targeted re-run reported 14/14 caught
- [Phase 02]: 02-08: official_f_avg binds to entrenar::eval::classification::metrics::f1_average_for_classes with NO #[contract] attribute added to aprender-train; the plan's ClassificationMetrics does not exist, the type is MultiClassMetrics
- [Phase 02]: 02-09: apr data select and apr data pairs are pure filesystem adapters — 0 occurrences of Sha256, swap(, unrank or json! in non-comment lines, exactly ONE fs::rename site, zero File::create, zero unwrap(); all semantics including the on-disk manifest envelope, budget resolution and the manifest->Selection path stay in the crate
- [Phase 02]: 02-09: the seed MODE is DERIVED from the recorded root_seed rather than stored beside it — SelectionPayload is crate-owned and deny_unknown_fields, adding a field would need a schema bump invalidating 02-05's goldens, and a stored mode could only ever disagree with the seed printed next to it
- [Phase 02]: 02-09: cli.offline is deliberately NOT threaded into either command — neither opens a socket (the crate cannot; make contrastive-data-boundary enforces it), so an offline switch would advertise a capability that does not exist
- [Phase 02]: 02-09: the split ROLE SET is read out of the attestation rather than hardcoded as a canonical triple, so a compatibility directory is refused by PROFILE instead of dying on a missing filename — a true statement about the wrong problem
- [Phase 02]: 02-09: atomic_write takes a FILL CLOSURE (atomic_write_with) rather than a byte slice, so --dump streams dump_pairs into the temp file instead of buffering up to ~60 MB; one rename site, so Task 2's three write-safety proofs still cover both artifacts
- [Phase 02]: 02-09: the plan's Task 3 replay rejections are unreachable from the CLI because SelectionManifest::from_bytes verifies the digest BEFORE returning — the tests reseal forged payloads with the crate's public hash::exact_hash, which reaches the membership, row-hash and recomputation rungs without naming Sha256 in apr-cli
- [Phase 02]: 02-09: Task 3 had NO executable RED (its tests could not compile until the interface existed, the same produced-before-consumed constraint the checker found in Task 1); its gates are falsified by three induced mutations instead, and the SUMMARY says so rather than manufacturing a RED after the fact
- [Phase 02]: 02-09: scripts/check_apr_bin_pinned.sh does NOT scan docs — measured with a two-sided control (a bare apr added to the doc is silent; a bare @apr Makefile recipe fires BARE-APR Makefile:1282), both reverted; nobody should later assume documentation is covered
- [Phase 03]: 03-10: the GSD tracking handlers damaged STATE.md/ROADMAP.md a FIFTH time, and this time the damage was a FALSE COMPLETION CLAIM, not a cosmetic field: `roadmap.update-plan-progress <phase> <plan> complete` marks the WHOLE PHASE `[x] (completed <date>)` and flips the Progress table to Complete as soon as summary_count == plan_count — before the verifier runs and regardless of unmet must_haves. `state.begin-phase` separately wrote `Plan: 1 of 10` on a resume at plan 10 and put the phase percentage in a field the body renders as a plan percentage. The orchestrator MUST diff both files after every handler call and revert any completion claim the verifier has not earned; reading the handler's own JSON (`"complete": true`) is not evidence of anything
- [Phase 02]: 02-09: the GSD state handlers corrupted STATE.md a THIRD time and in a NEW way — update-progress reported percent 100 while writing 40 into the frontmatter (and 94/20 on the earlier call: it writes the PHASE percentage into a field the body renders as a PLAN percentage), record-session silently ignored its positional stopped-at argument, record-metric REJECTS the documented positional form and needs --phase/--plan/--duration flags, and advance-plan clobbered last_activity to a bare date. Every field repaired by hand and read back
- [Phase 06]: FALSIFY-MONO-011 treats thin MCP deployment units as a SECOND, separately-ratcheted category (human decision `deployment-unit-class`, 06-02 Task 1) — Human answered the gate="blocking-human" checkpoint with "deployment-unit-class", no wording changes. `allowed_bins` and ALLOWLIST_BASELINE = 27 stay byte-identical so "migration debt" keeps meaning migration debt; a new `deployment_unit_bins` set of six names (the four SetFit crates plus aprender-mcp-forecast and the not-yet-created aprender-mcp-chronos) gets its own DEPLOYMENT_UNIT_BASELINE = 6 shrink-only assert and its own stale-entry check. Policy sentence: "publish = false thin MCP servers whose capability IS a protocol surface; adding one requires a CONTEXT decision, never a same-PR edit." The existing ratchet's claim that every entry is a capability awaiting `apr <subcommand>` migration stays true because CONTEXT explicitly defers `apr forecast`, so these six are not awaiting migration. Rejected: single-list-33 (blurs debt with deployment unit), phase6-only-29 (leaves the gate RED, SC5 unreachable), halt (contradicts D-06 and the `apr forecast` deferral).
- [Phase 06]: NeuralProphet-lite ships behind the one forecast tool with the D-10 training rules as invariant tests: graph-connected Huber (core's own smooth-L1 loss is detached and returns None gradients), strict mini-batches (full batch collapses the fit to MAE 2.6), clear_graph() per step, and lr selected by TRAIN loss — Selecting by test error would have reported 0.4112 instead of 0.4463 on this very dataset - a fitted curve presented as a forecast. The bars are one-sided thresholds rather than value-parity residuals because NeuralProphet's lr-finder and torch RNG are not reproducible from the committed oracle.
- [Phase 06]: Contract bars must be read with the contract name written IN FULL at each call site, never through a Rust constant — The acceptance criterion is a STATIC link check (grep -c). A Rust constant reads the right file at runtime but makes the link invisible to the grep, so the guard silently stops guarding. Fixed at 12 call sites in 06-04.
- [Phase 06]: D-13 memory clause AMENDED (blocking human decision `amend-memory-clause`): the shipped Bolt keeps BOTH weight layouts — dot8 needs contiguous [out, in] rows for D-14 single rows, gemm_blis needs the [in, out] transpose for multi-row. Cost asserted and printed: 69.18 MB resident vs 34.61 MB, exactly 2.00x. NOT the SC4 < 30 MB binary bar. — The frozen quantiles_abs_f32 = 1.0e-6 was measured under this routing; deleting the untransposed copies would move every single-row product onto gemv and change the float accumulation order, so the bar would stop being evidence. Re-confirmed in-tree at 9.5367e-7.
- [Phase 06]: The enforced Chronos f16 sha256 is f5dc2ef53533c8896bcb120a754c52c39d8917c15750a9e845192014dfa74a67, NOT the f9a033b4... RESEARCH A3 predicted: safetensors 0.8.0 orders the two __metadata__ keys differently than the spike-007 writer did. Same 101 tensors, byte-identical 17,305,344-byte body; only the header differs. — Re-pinning to the value this toolchain reproduces keeps the check fail-closed; loosening it to a warning would have deleted the property REVIEW-06-03 asked for. Both values are documented in the justfile, the crate README and contracts/chronos-bolt-parity-v1.yaml.
- [Phase 06]: The forecast tool boundary is pinned in ONE contract (contracts/forecast-tool-boundary-v1.yaml) that BOTH servers are held to; the six Rust bounds are asserted EQUAL to it at test time, proven by inducing RED from the YAML alone. — D-03/D-15: a bound written twice can be loosened in one place. Reading it from the contract makes the YAML the source and the constant the mirror, and a bound edit becomes a pv diff-visible change rather than an inline relaxation.
- [Phase 06]: pmcp 2.19.3 emits JSON-RPC -32603 (the INTERNAL error code) for Error::validation, so a refusal test pins BOTH the code and the "Validation error: " message prefix. — error_code() returns None for both the Validation and Internal variants, so the numeric code cannot distinguish a caller fault from a server fault. The thiserror-rendered prefix is the only discriminator the SDK ships. Read off a live reply, not assumed; breaking map_error turns 20 of 21 refusal cases red.
- [Phase 06]: REVIEW-06-04: the pool claim is split by instrument. Equality under load is a unit-test correctness claim in CI; the >= 2.0 speed-up is a benchmark claim owned by just forecast-pool-ratio (06-08). mod pool_equality contains ZERO timing assertions. — A wall-clock ratio under four Tokio workers and eight heterogeneous blocking fits moves with CPU throttling independently of the router serialisation the pool removes. A flaky bar in the correctness suite trains readers to ignore the suite, including the equality failure that would matter.
- [Phase 06]: The router pool is evidenced by a CONTROL pair, not a single number: POOL=1 measures 1.002x and POOL=8 measures 2.070x on the same host with the same eight requests, reproducing spike-010 in-tree. — CLAUDE.md Verification Discipline #2: never label a run by intent. Without the 1.002x control, 2.070x could have come from anything; with it, the pmcp router mutex is demonstrably what the pool removes. Equality held in BOTH configurations.
- [Phase 06]: 06-07: aprender-mcp-chronos ships WITHOUT a router pool — the D-12 pool answers a seconds-long fit holding pmcp's server mutex, and an 18 ms Chronos forward against an immutable Arc<Model> is not that — The D-12 pool is a remedy for a MEASURED serialisation (spike 010: 8 concurrent fits at 1.0x on one router). Copying it to a server whose call is 18 ms would ship K copies of a bottleneck that is not present.
- [Phase 06]: 06-07: the four D-18 gates carry #[rustfmt::skip] so cfg_attr stays on one line, and the ignore reason was shortened to 'CHRONOS_MODEL_DIR unset; just fetch-chronos-tiny' — rustfmt's attr_fn_like_width (70) splits the attribute vertically on every fmt run, and a split gate is harder to audit than the counted-skip claim deserves. The shortened reason still names the variable and the recipe.
- [Phase 06]: CI decision measure-x86-first: no ci.yml edit in phase 06; the gating work is one x86_64 `just chronos-gate` run that turns the PROVISIONAL quantiles_abs_f32_nonaarch64 bar into a measurement — Hunk (a) needs two unprovisioned runner prerequisites (weights mount AND a uv cache/baked packages for the network-less clean-room), so wiring it blind produces a step that dies at the weight-hash check before running a test. Collapses to `defer` with D-18 clause 2 named as an open CI gap if no x86_64 host is reachable.
- [Phase 06]: The SC5 >= 2.0 speed-up bar is asserted by `just forecast-pool-ratio` (best-of-3, aarch64 release) and by no unit test; measured 5.146/5.150/5.162 — REVIEW-06-04, both reviewers: a wall-clock ratio inside libtest moves with CPU throttling independently of the router serialisation, so a suite that cries wolf gets its real failures ignored. pool_equality asserts bit-identical responses only and prints the ratio.
- [Phase 06]: 06-EVIDENCE.md records the uncommitted working-tree delta's sha256 beside the commit hash, because the measurements ran on adc8a560a PLUS 18 uncommitted paths this plan did not create — A commit hash alone would have described code that was not what ran (CLAUDE.md Verification Discipline #2). 06-09 must establish that delta's provenance and then commit it, so the numbers become reproducible from a real commit.
- [Phase 06]: Applied the pv-suggested MINOR bump (forecast-tool-boundary 1.0.0 -> 1.1.0) rather than hand-choosing MAJOR; recorded that pv diff sees only structural additions and cannot see the door narrowing its accepted-input surface
- [Phase 06]: Asserted four cap-off-logistic refusal shapes plus a positive control, not the two the verifier named — one failing input is an anecdote (CLAUDE.md rule 6)
- [Phase 06]: Every GSD verification command in this repo must run through 'rtk proxy': the rtk hook replaces libtest's 'test result:' line with a summary, which makes plan <verify> greps fail vacuously
- [Phase 06]: 06-11: the 16.113 s in-bounds forecast wall is attributed to the FIT (99.8 percent), not to prophet::feature_row — the design build is 13 ms of it — Measured with an ignored release-profile harness splitting total wall by the response own fit_seconds/predict_seconds at five configurations plus a no-holiday control. The verifier attributed it from a code reading; removing the linear scan entirely moved the wall not at all.
- [Phase 06]: 06-11: MAX_HOLIDAY_DESIGN_COST=50000 bounds (points+horizon) x holiday_columns at the door — it caps WORK, not WALL — Largest round value whose three at-the-bound compositions all clear 2 s (1.174 / 1.692 / 0.106 s); 100000 fails two of three. A 25000-cell request still walls at 4.2 s reproducibly because the L-BFGS iteration count is data-dependent, so no payload statistic can guarantee a wall. SC1 for holiday-carrying requests stays open.
- [Phase 06]: The OVER/UNDER e2e pair is at the BOUNDARY — 62+7 vs 61+7 rows x 731 columns, 50 439 vs 49 708 cells against MAX_HOLIDAY_DESIGN_COST 50 000 — so one history point separates refusal from acceptance
- [Phase 06]: The RED observation needed two temporary perturbations (the constant AND the test's own geometry pre-assert); both are reported rather than only the second
- [Phase 06]: forecast-tool-boundary-v1.yaml bumped 1.1.0 -> 1.2.0 on pv diff's own 'minor' suggestion; the equation is what finally made 06-11's behavioural narrowing visible to pv diff
- [Phase 06]: The 2 s bar is scoped in the recipe header to the worst ACCEPTED shape and is NOT a general SC1 guarantee; WINDOWS.md entry 7 / 06-11 D7 stays OPEN and this plan picked none of its options
- [Phase 06]: Threshold 30.0 KEPT because the measurement allowed it: the only logistic parity fixture reaches lambda 3.1239, ~1/10 of the threshold, so no parity rung can take the new branch
- [Phase 06]: Outcome taken for the out-of-domain changepoint sampler: the VALID BRANCH (normal approximation above lambda 30), not a refusal and not a reported clamp - the regime is legal input inside every door bound
- [Phase 06]: poisson_sampler_domain float_tolerance = 0.01: 3.16 sigma at the noisiest sweep point (lambda 5), >= 7.62 sigma elsewhere, against a 17.2% pre-fix failure at lambda 900
- [Phase 06]: Cost axis C-07 (holidays[].name) can be closed ONLY by a bound, never by a measurement: the field is an unbounded String and the router construction applies no DefaultBodyLimit / max_body / content-length layer, with no stdio framing cap — so it has no structural maximum to measure against.
- [Phase 06]: A plan whose thesis is that a guard which cannot fail is theater must not reduce its own open items to a note. types::tests::no_cost_axis_is_pending SHIPS RED naming C-07 and C-08 instead of giving C-07 a measured disposition it cannot honestly carry.
- [Phase 05]: D-19 contract narrowing APPROVED at a blocking human checkpoint: setfit-benchmark-claims-v1 goes 1.0.0 -> 2.0.0, active expectation set 80 two-method cells -> 40 SetFit cells, with the two-method design RETAINED as an explicitly deferred scope pointing at D-ITEM-05-15. pv diff suggested major; major taken. — Nothing describing the two-method design was deleted (equations 10->10, obligations 9->9, falsification tests 10->10, machine-checked), so D-ITEM-05-15 is a restoration rather than a re-derivation, and two of the six doctored negatives keep their meaning.
- [Phase 05]: The narrowing lands on a NEW constant ACTIVE_METHODS (expectation-set domain), not on BENCH_METHODS (row-validity domain), which keeps both methods. — Narrowing BENCH_METHODS would have made a second method row fail row validity and silently DELETED the two deferred-scope negatives by making the rows they doctor unbuildable - a gate that looks tighter while proving less.
- [Phase 05]: A single-cell verification door (verify_cell / apr setfit bench verify-cell) is verify_run steps 1+4+6 over ONE declared cell, excluding the set-level steps, and emits no statistic. — At pilot time the manifest declares 40 cells with 39 pending - exactly what step 3 sweep refuses on - so a door inheriting the set-level checks could never pass on the cell it exists to check. Equivalence with verify_run is proven behaviourally by a per-defect variant-tag table, never by asserting the door calls the two functions.
- [Phase 05]: The active claims report must not name the deferred method BELOW section level: 05-11's case table carried only section-level literals, so a row label, a size footnote and the header's provenance clause survived the retarget. Case table grown 6 -> 9 rows plus a token-level control. — Found by running the plan's own automated verify rather than by review. A section-level absence table cannot refuse a sub-section literal nobody thought to add, so the report is now gated on the rendered TEXT as well as on the constants.
- [Phase 05]: EVAL-02, EVAL-04 and EVAL-05 recorded as met by a NARROWER deliverable under the 2026-09-07 amendment, with D-ITEM-05-15 named; no requirement checkbox flipped and requirements.mark-complete not run. — Filing a narrower deliverable against an unamended promise is the false-completion defect this project has hit six times. Flipping requirement or phase-completion state is the verifier's act.
- [Phase 05]: WR-06 taken as option (a): read_evidence takes an EvidenceKind, so a missing lock or ledger names its own artifact instead of telling the operator to restore a row file — Minting a correctly-typed path-escape variant beside a mis-typed missing-file variant, inside one function, in one edit, would be incoherent
- [Phase 05]: resolve_committed_evidence_path returns the JOINED path, not the canonical one; canonicalization is used only for the containment check — On macOS a temp dir canonicalizes /var to /private/var, so returning the canonical form would rewrite every downstream refusal message and show operators a path they never typed
- [Phase 05]: throughput_batch_size deliberately left class (iii) while three sibling contract-pinned constants were closed — The contract pins no value for it and the writer constant lives in apr-cli as a producer choice, so comparing against it would refuse a legitimately different batch size while calling it a contract violation
- [Phase 05]: 05-16 closes verifier gap 2 (EVAL-02): verify_selection_binding recomputes the row's selection_manifest_hash from the manifest at selections/s{shots}-seed{seed}/, a path built from the CELL KEY and taking no BenchRow, and compares the manifest's own shots_per_class/root_seed against the cell key so a transplanted manifest is refused even when the row's hash is doctored to match it.
- [Phase 05]: The digest is NOT recomputed in bench_gate: SelectionManifest::from_bytes verifies it before returning, so a second recomputation would be a second definition of a manifest's identity (OPS-03). A structural guard goes red on the three obvious ways to write one.
- [Phase 05]: The binding is step 6b, AFTER verify_pairing (step 5), which is byte-unchanged. Earlier would pre-empt the deferred-scope unpaired_selection negative and silently change which rule it observes; a test asserts that negative still fires on its own variant.
- [Phase 05]: contracts/setfit-benchmark-claims-v1.yaml bumped 2.0.0 -> 3.0.0, the bump pv diff suggested. Additive in shape but the ACCEPTANCE SET NARROWS: a tree that verified at 2.0.0 can be refused now. pairing_rule keeps status: deferred and D-ITEM-05-15.
- [Phase 05]: The setfit-bench-gate floor is the MEASURED 52, not the plan's 'floor plus 9'. That arithmetic contradicted the plan's own single-swept-table criterion; the gap is closed instead by pinning the 11-row table length inside the test, which a test-FUNCTION floor cannot see.
- [Phase 05]: Re-measured for the selection manifests: compact JSON in FILE/DECLARATION order reproduces 40/40 committed semantic_hash values, key-SORTED reproduces 0/40 — 05-15's bench_row.rs:37-43 finding confirmed on a second artifact. Harmless here because SelectionPayload serializes as a struct, but the comment's stated reason is still false.
- [Phase 05]: The acceptance band for the closed-form quality cross-check is EXACT IEEE-754 bit equality, chosen from a measurement taken first: 40/40 bit-identical on every field over the committed rows, max deviation 0e0. No epsilon was needed and none was fitted. — Exactness is structural rather than lucky: the recomputation expands the counts and routes to the SAME integer-sourced surfaces assemble_quality_block used, so agreement is bit-identical by construction and no tolerance can be justified.
- [Phase 05]: Contract 4.0.0: pv diff suggested MINOR, the suggestion was recorded verbatim and NOT taken. — The acceptance set narrows again - the exact tree verifier spot-check D built verified at 3.0.0 and is refused now. pv scores the shape of the edit; a strengthened guarantee is a breaking change to every producer relying on the old acceptance, which is the reasoning the 3.0.0 metadata block itself records.
- [Phase 05]: The report residual, the gate module header and the contract residual_risk are ONE statement written down three times, and all three were corrected together. — Three attacks the sentence conceded are now refused (path escape, pairing key, quality metric). An understated disclosure teaches a reader to trust real evidence less than it warrants, which is over-claiming with the sign flipped.
- [Phase 06.1]: RegressorArg wire shape frozen as proposed (name, values, mode, prior_scale, standardize); map-keyed rejected because serde_json::Map is key-sorted and would break the oracle insertion-ordered column tail
- [Phase 06.1]: Regressor columns identified by the structural trailing index range base_k..k, never by name membership; Column.holiday keeps its two-valued discriminator (add-alongside)
- [Phase 06.1]: regressor_prior_scale_min kept at 1e-153 as planned, but MEASURED to be a representability bound not a usability bound - the fit runs zero iterations and silently zeroes the regressor for any prior_scale below ~1e-8; no NaN ever appears
- [Phase 06.1]: budget_hit stays inside the D-19 signature; the masking result is reported as a triage pointer with both readings, never as a subtraction
- [Phase 06.1]: Part C of the invariance gate establishes splice inertness only; the independent historical claim belongs to the committed pre-change baseline alone
- [Phase 06.1]: Part C fits once per series and reuses the same Params for both predict calls, removing L-BFGS budget behaviour from a test whose claim is about the design
- [Phase 06.1]: Verifies t1c and t3b kept verbatim and recorded RED with intent-preserving substitutes, rather than removing capture_baseline #[ignore] to make them pass
- [Phase 06.1]: SC6 is NOT fully green on this host and the phase does not claim it is; the sweep is recorded gate by gate with what each command printed
- [Phase 06.1]: The SC2 no-argument invariance gate is red under the --workspace --lib build CI runs and green under -p aprender-forecast; the variable is cargo feature unification, isolated to apr-cli by one-variable control
- [Phase 06.1]: The AR-absorption caution reaches FOUR surfaces, because schemars supplies the input SCHEMA and nothing supplies the tool-level description an MCP client reads in tools/list
- [Phase 06.1]: The compatibility promise is published as SCOPED to JSON/MCP callers, with both halves observed by a compile probe rather than asserted
- [Phase 08]: 08-01: classify token budget 1024 built-row tokens, derived DOWN from (30000-12000-1100-4000)/11 = 1172; probe_budget_ms 1100 = 2 x 48 x 11 ms
- [Phase 08]: 08-01: Laya gate pass decided on Rust-recomputed metrics from probabilities verified against Rust re-scores; reported metrics checked, never trusted; ECE is the house top-label ECE
- [Phase 08]: 08-01: probes use a contract-resident synthetic K=2 task and store Python F16-reload values; task refusal is marker LOSS after Laya's shrink, never raw option length
- [Phase 08]: 08-03: modernbert primitives (Linear::forward, layer_norm, attention, rope_rotate_half) return Result with typed ModernBertError; aprender-decide consumers propagate with the question-mark operator — no primitive may index-panic on a bad buffer from an untrusted .apr or a caller
- [Phase 08]: 08-03: layer_norm rounds the configured norm_eps to f32 before widening, as torch CPU LayerNorm casts eps to the f32 accumulation type — keeps spike-025 numerics bit-identical while honouring norm_eps
- [Phase 08]: 08-03: the plan clippy gate is not vacuous despite the pre-existing arm64 unreachable_code error; an induced needless_return in modernbert is still reported (base count 1) — ptr_arg probes are invalid on pub fns because of avoid-breaking-exported-api
- [Phase 08]: 08-04: aprender-decide criteria order is read from raw bytes by an order-preserving visitor; the crate never enables serde_json/preserve_order, and both backings are observed (standalone OFF, -p aprender-mcp-setfit ON)
- [Phase 08]: 08-04: Laya::from_parts refuses MarkersLost once at load (empty-state build); options longer than head_max_len are shrunk as Laya does and served
- [Phase 08]: 08-04: contract test: lines must cite concrete test fns - pv's strict-binding resolver reads the last :: segment, so a module filter like --lib task:: is invisible (a mutated name was not flagged)
- [Phase 08]: 08-05: Laya probes run via Laya::classify_for_task on the already-loaded weights (no second widening of the checkpoint per cold start)
- [Phase 08]: 08-05: decide-apr-v1 manifest gains a variant field and stores blobs as an array of {name, sha256}; contract updated, schema_version stays 1
- [Phase 08]: 08-05: the trybuild private-mint proof is two cases, because rustc skips the privacy pass after a type error and hid E0451
- [Phase 08]: 08-05: apr-format caps the tensor-index reservation at remaining/20 for every APR consumer, not only decide
- [Phase 08]: 08-06: FALSIFY-MONO-011 DEPLOYMENT_UNIT_BASELINE raised 7 -> 8 for aprender-mcp-decide under CONTEXT D-15 (D-15's 'no baseline change is expected' was wrong: the register length is the ratcheted quantity); 08-07's lambda takes it to 9 — needs human confirmation
- [Phase 08]: 08-06: ClassifyService::served is the only public handler constructor and pins ClassifyLimits::CONTRACTED; tokenizer error detail is withheld from responses (may quote caller text)
- [Phase 08]: 08-06: TOOL-004 binds aprender-decide stance order under both serde backings plus aprender-mcp-decide's response order; KANI-DECIDE-TOOL-001 evidence is the bound_order_is_count_bytes_tokens proptest
- [Phase 08]: 08-07: FALSIFY-MONO-011 DEPLOYMENT_UNIT_BASELINE 8 -> 9 for aprender-mcp-decide-lambda under D-15 (same pattern as 08-06's 7 -> 8); needs human confirmation
- [Phase 08]: 08-07: S3 cold-start loader writes each ranged GET straight into a disjoint slice of one pre-sized buffer (RangeFetcher::fetch_range(start, &mut [u8])); a hand-rolled bounded FuturesUnordered replaces buffer_unordered(closure), which broke the lambda_http handler's Send bound
- [Phase 08]: 08-07: the decide deploy template does NOT prescribe --manifest-path crates/aprender-mcp-decide-lambda: cargo-pmcp's resolver would ship another *-lambda binary from that root; 08-10 settles the deploy root
- [Phase 08]: 08-08 OPTION 3 (user, 2026-09-26): the D-19 stance demo's outcome of record is GATE FAIL under both declared recipes (fixed_epochs d0f4e40d ece_post 0.3773 T clamped 5.0; early_stopping 3d4b91da ece_post 0.2224 T 3.14; margin passes both). Both run dirs (models/decide/tweet-stance-16-fixed-epochs, models/decide/tweet-stance-16) are FAIL-CLOSED TEST VECTORS that 08-09 pack/verify must refuse; the D-18 live stance deploy is DEFERRED until a declared run passes. Recorded in laya-finetune-gate-v1 1.2.0 demo.*; gate_max_ece, T bounds, seeds and data unchanged; no further recipe tried
- [Phase 08]: 08-08: the same recipe and seed on MPS is not bitwise reproducible (the 3-seed variance run's seed-13 leg under recipe 3d4b91da gave ece_post 0.2202 vs the recorded 0.2224); a fail-closed vector is identified by its run-dir files and recipe_id, never by re-running it. Variance over seeds 13/17/23: macro_f1 0.451 +- 0.023, F_avg 0.480 +- 0.014, ece_post 0.226 +- 0.115; no seed passes
- [Phase 08]: 08-08: --seeds N keeps the data and the calibration split fixed (declared seed 13); variance seeds vary only the training RNG, run in deleted temp dirs, and the declared seed's checkpoint (byte-identical to a single-seed run's) is the only one kept
- [Phase 08]: 08-09 OPTION A (user, 2026-09-26): pack_rescore_probs_abs stays 1e-5; early_stopping 3d4b91da is the vector that demonstrates GateFailed[ece_post] (exit 3, rescore 6.7e-6 / zs 6.8e-6, argmax 280/280); fixed_epochs d0f4e40d is refused with RescoreDrift which=fine_tuned row=59 max_abs=4.667e-5 (exit 2, nothing written) because torch's own fp32 is up to 3.7e-5 from a float64 reference on it. FALSIFY-LAYA-GATE-010 (laya-finetune-gate-v1 1.3.0) binds the per-vector refusal; demo.fail_closed_rule prose left unchanged (D-ITEM-08-09-C)
- [Phase 08]: 08-09: the stdio server real-model leg is DEFERRED (D-ITEM-08-09-A, user approval): no artifact pack_laya verify accepts exists; verify_path on real-weights bytes (all eight rungs) and full-model parity (ids 14/14, |dp| 3.841e-6) cover the gap until the first eligible artifact
- [Phase 08]: 08-09: pack_laya verify is the ONLY deploy-eligibility check (exact file, full ladder, manifest bound to run/data dirs, every pack check re-run); inspect is identity-only; pack-fixture writes only synthetic-fixture artifacts (models/decide/selftest/laya_tiny.apr, golden 37d65159, refused by verify) for 08-10
- [Phase 08]: 08-10 SHARED-CRATES-ROOT (user, 2026-09-26): the decide server deploys with `cargo pmcp deploy --manifest-path crates` and server name aprender-mcp-decide. cargo-pmcp 0.24.3's own find_lambda_package_dir, executed via `just laya-resolver-proof` on a git-archive of SDK e0561f8c9 (builder.rs unchanged since c04fb4ccb; the installed binary records no build commit), returns crates/aprender-mcp-decide-lambda; the per-crate root returns crates/aprender-mcp-chronos-lambda (Pitfall 1 confirmed). Limits: one decide model per workspace (D-ITEM-08-10-A); the upstream cargo-pmcp fix is recommended future SDK work, not done (D-ITEM-08-10-B)
- [Phase 08]: 08-10: the shared root's setfit-train state (.pmcp/deploy.toml, deployment.toml, active-target AND deploy/ with its rendered stack.ts) is swapped out and restored byte-identically on every exit path (_laya-crates-root-swap); deploy/ must go too because cargo-pmcp PRESERVES an existing stack.ts, which would synthesize setfit-train's stack without the decide [environment]; laya-deploy passes --regenerate-stack
- [Phase 08]: 08-10: laya-verify precedes the first AWS call in laya-deploy and laya-upload; the selftest proves placeholder, sha-pin, resolver-proof, deploy- and upload-eligibility refusals on the synthetic artifact with an aws recorder (positive control) at AWS CALLS 0; the live halves have never run (D-ITEM-08-10-C, option 3)
- [Phase 08]: 08-11 D-18 go/no-go: HOLD with no AWS call (hold-no-aws), decided by the user 2026-09-26; 08-DEPLOY-EVIDENCE.json is the hold record 08-12 reads; D-ITEM-08-11-A preserves the live deploy; next: calibration spike, then a declared gate run
- [Phase 08]: 08-13: user option 1 (2026-09-27) declared before any s64 run - laya-parity-v1 2.0.0 A1 bar max(1e-5, 4 x torch32-vs-f64 noise), ceiling 1.0e-3, absent record = floor; laya-finetune-gate-v1 AMENDMENT 1.4.0 published as 2.0.0 (pv diff major): A2 in-distribution eval set + reported shift probe, A3 median-ECE seed of 13/17/23 (floor(ece_post x 10000), ties to smaller seed, selected with eval labels)
- [Phase 08]: Median-ECE seed rank key is floor(ece_post x rank_scale) read from the contract; seeds.label stays the contract literal 'mean ± sd over 3 seeds' (D-ITEM-08-14-A) [superseded 2026-09-27 by 08-15: the contract now declares 'median-ECE seed of N seeds' for a median run]
- [Phase 08]: Every Laya run writes rescore-noise.json only after a manual fp32 forward reproduces the Scorer's logits exactly; otherwise exit 2, no record, no gate report
- [Phase 08]: data/decide/tweet-stance-64 built by eval_set.demo_rule: 459 rows [111, 291, 57] + 280 shift rows; byte-identical to spike 028's held-out set
- [Phase 08]: 08-15: D-ITEM-08-14-A/B settled by contract text before 08-16 - gate-report seeds.label is 'median-ECE seed of N seeds' under seed selection; GATE-006's legacy multi-seed clause retired with its reason; pv diff identical, no bump
- [Phase 08]: 08-15: why_quantized implemented as refuse when Rust's rank key of the recomputed ECE differs from Python's reported rank_key (not within 1e-5 of a grid line, which would refuse about 20% of seeds)
- [Phase 08]: 08-15: SeedPolicyMissing decided after the re-scores and the gate; ThresholdMismatch now fires before the re-scores (check_thresholds precedes the seed check)
- [Phase 08]: 08-16: the one declared s64 gate run PASSED the unchanged gate (median seed 17, margin 0.2217, ece_post 0.0443 on 459 in-distribution rows); artifact 24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a deploy_eligible, stdio leg passed; claim is in-distribution only (shift probe ece_post 0.1896, not a gate clause)
- [Phase 08]: 08-18: live record FINAL deployed-passed; the external 31.05 s client-time cold call (HTTP 200, in-function 28008 ms) is risk evidence in D-ITEM-08-17-E, not a relabel, because the rule is defined on the laya-deploy-verify samples
- [Phase 08]: 08-18: decide function left RUNNING with auth off by user decision (D-ITEM-08-18-A); containment recorded, not run; 0 AWS writes
- [Phase 08]: 08-18: laya-deploy-verify elapsed_ms is client-side through the edge (client minus REPORT 753-785 ms), and the external curl showed about 3042 ms, so the contract's 895 ms gateway+client term does not bound client overhead
- [Phase 08]: 08-19: decide-apr-v1 bumped 1.0.0 -> 2.0.0 (pv diff: major, blob_integrity formula changed), not the plan's 1.1.0
- [Phase 08]: 08-19: rung 4 (e) binds all 53 manifest leaves; the sweep forges ONE binding at a time (59 bindings) so a two-source leaf cannot lose either comparison unnoticed; 23/23 comparisons mutation-verified
- [Phase 08]: 08-19: verify_checks[2] states what check_base enforces today; field-by-field base equality at verify is plan 08-21, NOT checked yet
- [Phase 08]: 08-20: shared APR v2 index must be strictly increasing (duplicate names refused by both readers); index reservation counted in in-memory entry units; class-B test hooks read back what was actually sized (Vec::capacity after allocation, RoPE-table counter) so sizing mutants turn RED
- [Phase 08]: 08-21: one typed contract-to-policy mapping (VerifyPolicy::from_contract_views over serde views) for CLI and tests; verify binds every recipe.json / gate-report.json leaf per laya-finetune-gate-v1 run_field_bindings (86 bound, 3 report_only, 5 f_avg expected-open for 08-27), enforced by a mutate-one-leaf sweep; sweep-found leaves bound (nll, t_fitted clamp relation, device_is_cpu, seeds.label, shipped per_seed.t_applied); verify_run is cfg(test)
- [Phase 08]: 08-22: laya gate enumeration is fail-closed (exit/FAIL/exec/verdict token); an exec-ing recipe cannot be not-a-gate; heavy must-pass cases are a named FULL tier (LAYA_GATES_FULL=1) listed in the default OK line
- [Phase 08]: 08-23: decide-tool-boundary-v1 4.0.0 (pv diff MAJOR on the classify_count_bound invariant); served_fields and untrusted_input_bounds are top-level tables pv ignores and a crate test sweeps row by row (unknown id or row without a case fails)
- [Phase 08]: 08-23: frame_stdio accepted (pmcp 2.19.3 has no stdio framing cap; local transport); the sweep asserts the count bound on a 100000-element parsed array and pins pmcp 2.19.3 in Cargo.lock so a bump re-opens it
- [Phase 08]: 08-25: the deploy probe's ok() is the AND of four checks against the local artifact (one classify tool, exact labels segment, response labels, sha256); MaximalError::OverBudget carries the server's check_token_budget refusal verbatim (one budget function)
- [Phase 08]: 08-24: the Lambda bootstrap's decisions are pure lib functions (route: only POST loads, 405 otherwise; health: 503 ok:false with config_error on an unparseable APRENDER_DECIDE_* config; proxied_headers: one CORS origin); a dead loopback exits the process via watch_loopback
- [Phase 08]: 08-24: Lambda crate unwrap ban is crate-wide again (item-level json! allows only; planted unwraps in lib.rs, probe.rs, main.rs fail clippy); its two untrusted_input_bounds rows are swept by lambda_request_rows_are_swept (frame_http exactly 413 with an at-bound 200 control)
- [Phase 08]: 08-26: decide-apr-v1 max_criteria = 510 (max_len 512 - 2), not max_len/4 = 128 — measured on Laya-en one-token names serve to K = 253
- [Phase 08]: 08-26: load-time probe rows are checked against probe_max_row_tokens BEFORE the replay forward; tokenizer_pipeline is an accepted class-B row (only verify binds the tokenizer)
- [Phase 08]: 08-27: Rust-Python agreement by construction — every gate quantity (macro-F1, f_avg, top-label ECE, margin, rank key) is one f64 computation with exactly-rounded sums (aprender-core metrics::fsum = Python math.fsum), declared in laya-finetune-gate-v1 4.0.0 numeric_agreement and replayed bit for bit from scripts/laya_train/numeric_cases.json in both languages
- [Phase 08]: 08-27: the f64 gate metrics are ADDED beside aprender-core's f32 f1_score / expected_calibration_error_top_label, never swapped in — the f32 bits are frozen by metrics::f32_bits_tests (committed alone); f_avg is recomputed by verify::check_f_avg under train.py's null rule, so all 5 f_avg run_field_bindings rows are bound (bound=91, expected_open=0)
- [Phase 08]: 08-29: the early-stopping rule of record is best-anchored (m_e < m_best - min_delta), re-derived from the code that trained the 08-16 run (gate.py 24cea2a63); its seed-17 trace alone cannot separate the candidate rules, so contract text is corrected to the code, never the reverse (laya-finetune-gate-v1 5.0.0)
- [Phase 08]: 08-29: every contract number the Python back office reads goes through contract.number (no coercion); rescore-noise.json k is written as a float, the value Rust compares bit for bit
- [Phase 08]: 08-28: owner decision A-derive B-iserror (D-09 amendment 'gap round 08-28'): the served truncation sentence is derived from the artifact's agent max_len vs classify_max_total_tokens, and every bound refusal is pmcp::Error::tool_rejected (an isError tool result on the wire, measured over stdio and the Lambda HTTP loopback); model/internal failures stay JSON-RPC -32603; effective only at the next deploy (plan 08-30)
- [Phase 08]: 08-28: refusal_names_bound quantifies over distinctive caller texts of at least refusal_echo_min_chars (12, a decide-tool-boundary-v1 constant the test reads); the old any-non-empty-text clause was unsatisfiable (A4-7)
- [Phase 08]: 08-30: owner keep-800-apply-10739 - decide-tool-boundary-v1 keeps 8 texts / 800 tokens at 10,240 MB on the acceptance rule (8 proven-cold samples 22.6-27.0 s < 30 s) although the per-term-max rule gives 648; contract 9.0.0 shows priced vs measured; s3::DOWNLOAD_DEADLINE 10739 ms derived from the samples; download_budget(remaining, reserve) honours the invocation deadline (V4-b); accepted_region_cold implemented; live at 10,240 MB serving b615d8244
- [Phase 08]: 08-31: owner decision publish-false (D-14, 2026-09-29) - aprender-decide is publish = false until the crate name is confirmed; not in the cascade TIERS; kept by publish_is_false_until_the_crate_name_is_confirmed and gate row cascade-guard; nothing published
- [Phase 08]: 08-31: class D ledger scripts/laya_claims.tsv (133 rows) derives its FALSIFY and untrusted_input_bounds rows from the contracts' own cargo commands, one row per owner target, each named test matched exactly (never cargo's substring filter); a corrected claim needs a ledger row in the same commit
- [Phase 08]: 08-32: owner decisions "Scrub, then push", "Yes, draft PR", "Merge upstream main in", "Keep one draft for CI" (#4634 one draft purely for CI evidence; split into reviewable PRs after phase verification), "Defer to deferred-items" (17 SetFit CI legs, D-ITEM-08-32-C) and "Repair all (new gap plan)" (upstream contract-hygiene gates; decided; planned as 08-33). pmcp stays at 2.19.3 (D-ITEM-08-32-G). The plan's CI must-have (workspace-test success on the pushed head) is UNMET and recorded as such in 08-CI-RUN-EVIDENCE.json

### Pending Todos

- [Phase 8 — QUEUED 2026-09-26 by the 08-08 option-3 decision]: `.planning/todos/pending/spike-laya-calibration-slice-and-temperature-cap.md` (`/gsd-spike`). Before any third Laya stance gate attempt: measure whether a larger calibration slice (the s64-seed13 cell, 48-row slice) and/or a higher served temperature cap can bring ece_post under 0.10, over seeds 13/17/23. Whatever it recommends is declared in laya-finetune-gate-v1 BEFORE that run is read; gate_max_ece does not move; a cap change touches laya-parity-v1 and the Rust clamp and is the user's call. The D-18 live stance deploy waits on a passing declared run.

- [Phase 2 — CLOSED BY 02-09, the policy held]: DATA-01 through DATA-06 were deliberately left
  UNCHECKED after 02-01 and 02-02, because those plans shipped a crate of `//!`-doc stubs and
  checking the boxes would have put five false claims in the traceability table. The policy was
  to mark each at the plan that actually closes it, and that is what happened: 02-06 closed
  DATA-01/02, 02-05 closed DATA-03, 02-07 closed DATA-04/05, 02-08 closed DATA-06. **02-09
  re-audited all six against the shipped behaviour before letting the table stand**, and each is
  genuinely delivered at the "a user can…" tier this milestone demands — every one was
  demonstrated in 02-09 by a real `apr` run against the live pinned TweetEval revision
  (587/66/280 with provenance; ten contracted seeds each replaying its own hash; a 256-pair dump
  audited endpoint-by-endpoint against the splits; a fixed budget holding while examples grow 8x;
  and the compatibility, mixed, forged and stale directories each refused fail-closed on BOTH
  commands). `requirements mark-complete DATA-03 DATA-04 DATA-05` returned
  `updated: false, already_complete` and REQUIREMENTS.md is byte-unchanged. **Nothing was closed
  to make the table look finished, and nothing is left open.**

- [Phase 2 — PARTIALLY CLOSED BY 02-08]: `official_f_avg` and all 24
  contrastive-pair-protocol equations are now BOUND, and `make contract-audit-phase2` (blocking,
  in tier3) keeps them bound. The REPO-WIDE `make contract-audit` is not fixed and is worse than
  this entry recorded: measured, it reports **132 BIND-001 errors across 38 of the 44 contracts**
  — 10 of them Phase 1's setfit equations — and **exits 0 anyway**, because its loop body never
  reads the audit's status. It is a target that prints failures and reports success. Logged as
  D-ITEM-04 in the phase's `deferred-items.md`; the honest fix is to make it read its status and
  then either bind the 132 or mark them `status: pending` (a BIND-004 warning, not a BIND-001
  error). Still worth a dedicated binding-registry pass.

- [Repo-wide]: No `#[kani::proof]` harness exists anywhere in `crates/` and `cargo-kani` is not
  installed, yet contracts declare harnesses. 02-01's and 02-02's contracts now say so explicitly
  in-prose and name their runnable proptest backing; Phase 1's
  `setfit-encoder-conformance-v1.yaml` still does not.

- [Repo-wide]: Both CB-510 packaging guards (`scripts/check_include_files.sh`,
  `scripts/check_package_includes.sh`) pass VACUOUSLY on macOS. They use GNU `grep -oP`; BSD grep
  exits 2 with `invalid option -- P`, the trailing `|| true` swallows it, and both print
  "All 0 include!() files" and exit 0. True count via `ggrep`: 1768. CI runs on Linux so this is a
  local false-green, but `make tier3` tells a developer something untrue. Surfaced by 02-02 as
  D-ITEM-01 in the phase's `deferred-items.md`; fix is a repo-wide shell-portability change with
  its own must-match/must-not-match case table (CLAUDE.md rule 7). Worth a dedicated ticket.

- [Repo-wide — SURFACED BY 02-09]: `scripts/check_apr_bin_pinned.sh` does **not** scan
  documentation. Measured with a two-sided control rather than read: a bare `apr data select …`
  added to `docs/examples/tweet-eval-stance.md` leaves it green (rc=0, 28 files), while a bare
  `@apr qa model.apr` added as a Makefile recipe fires `BARE-APR Makefile:1282`. Both probes
  reverted. This is the guard's stated design ("the invariant is what CI executes, not that
  nobody may ever type apr"), so it is a scope note rather than a defect — but a reader who
  assumes docs are covered will be wrong, and the docs are where a user copies commands from.
  Worth deciding deliberately whether user-facing docs should be in scope.

### Blockers/Concerns

- [Phase 1]: Freeze numerical tolerances from pinned reference fixtures before examining Rust discrepancies; validate the real-weight mixed-batch graph before committing the full BERT refactor.
- [Host — RECURRING, hit TWICE in this phase]: `target/debug/incremental` regrows to ~25 GB and
  fills the volume; 02-09 stopped mid-plan with `ld: write() failed, errno=28` at 546 MiB free.
  Nothing was deleted by the executor (standing instruction) and the run was reported as a
  checkpoint with measured numbers; the coordinator reclaimed ~21 GB. **Mitigation adopted for
  the rest of the plan and recommended for Phase 3: `export CARGO_INCREMENTAL=0`** — the cache
  did not regrow and 15 GB was still free at plan end.
- [Host — THIRD occurrence, 2026-08-17, BLOCKING Phase 05 wave 2]: the volume is 100% full —
  **791 MiB free of 926 GiB**. `target/` is 138 GB, of which `target/debug/incremental` is
  **90 GB** (~3.6x the ~25 GB recorded above), `target/debug/deps` 22 GB and
  `target/llvm-cov-target` 19 GB. Measured by the orchestrator, not inferred: `du -sh target`
  and `df -h /System/Volumes/Data`. Both wave-2 plans failed on it — 05-05 could not even be
  dispatched (`git worktree add` died with `No space left on device`), and 05-03's executor
  halted before Task 1 (see `05-03-SUMMARY.md`, `status: halted`). `CARGO_INCREMENTAL=0` was
  NOT in effect, so the mitigation adopted above has lapsed.
  **The orchestrator deleted nothing** — the user declined reclamation when asked. It was
  nevertheless reclaimed from outside this session at 12:05 local: `target/debug` (117 GB),
  `target/llvm-cov-target` (19 GB) and `target/coverage` were removed, leaving `target/` at
  1.7 GB and **138 GiB free**, so the `MIN_FREE_GIB = 10` preflight now clears with wide margin
  and wave 2 was re-dispatched on that basis. Note the reclaim also took `target/release`'s
  siblings but left `target/release` itself, so release builds are warm-ish, not cold.
  **`CARGO_INCREMENTAL=0` is still NOT exported** — the mitigation adopted after occurrence #2
  has lapsed, which is why the cache reached 90 GB. Re-export it before the next long build or
  this recurs a fourth time.

- [Phase 2]: Decide and version singleton-class and bounded-oversampling behavior during phase planning.
- [Phase 2 — KNOWN-RED, EXPECTED, NOT A REGRESSION — **WIDENED BY MEASUREMENT IN 02-02**]: `pre-release` Gate 5 fails from Phase 2 wave 2 through phase exit. Cause: `apr-cli` gains a dependency on the new `aprender-contrastive-data` crate, which is not on crates.io until the human-approved publish cascade lands it (RESEARCH Pitfall 8 / Finding F5). **CORRECTION (02-02, measured):** it is NOT only the verifying form. `cargo package --no-verify -p apr-cli` ALSO fails — `--no-verify` skips the packaged-crate BUILD, not the MANIFEST RESOLUTION that rewrites the path dep into a registry dep, and resolution is where it breaks (`no matching package named 'aprender-contrastive-data' found`). Control-verified: with the dependency line temporarily removed the identical command exits 0 and packages 581 files. So ANY `cargo package -p apr-cli`, verifying or not, is red. What IS gated and must stay green: `cargo package --no-verify -p aprender-contrastive-data` (rc=0, 19 files). Exit condition unchanged: publish `aprender-contrastive-data` BEFORE `apr-cli` — a human-approved release action; CLAUDE.md forbids self-serving the publish. `/gsd:verify-work` must read a red Gate 5 as this expected state. Mirrored in `must_haves.caveats` of plans 02-02 and 02-08 and in 02-VALIDATION.md; plan 02-08's acceptance criterion "both `cargo package --no-verify` runs exit 0" is falsified and should be read as the crate-only form.
- [Cross-cutting — HOST-SPECIFIC, PRE-EXISTING, NOT A REGRESSION]: `cargo check --workspace` cannot exit 0 on Darwin. `crates/aprender-profile/src/main.rs:6` is a `compile_error!("renacer requires Linux (ptrace syscall tracing)")` under `#[cfg(not(target_os = "linux"))]`, and the consequent E0601 (`main` not found) is its shadow rather than a second defect. Control measured at 02-08: `cargo check --workspace --exclude aprender-profile` exits **0**. Any plan whose acceptance criterion names a bare `cargo check --workspace` should be read as the `--exclude aprender-profile` form on this host.
- [Phase 5]: Choose validation-only calibration and uncertainty estimators before collecting benchmark results.
- [Cross-cutting]: Preserve CPU-only package/MSRV/feature combinations and executable contract conventions from the repository's pre-release and APR dogfood skills.
- make tier2 is RED on arm64: pre-existing clippy errors across 5 crates untouched by phase 2. Re-measured at 02-08: **24 errors, 44 locations** — aprender-compute 38, zram-core 3, present-terminal 1, core 1, serve 1; **zero in aprender-contrastive-data**, whose only appearance in the tier2 log is its `Checking` line. All arch-gated SIMD; CI runs X64-Linux-only so these aarch64-live arms are never linted. Proven independent of 02-03. See deferred-items D-ITEM-02.
- contracts/chronos-bolt-parity-v1.yaml quantiles_abs_f32_nonaarch64 = 5.0e-6 is PROVISIONAL AND UNMEASURED: no x86_64 run has happened, and every CI job here is [self-hosted, X64]. FALSIFY-CHRONOS-002 obliges the first x86_64 run to record its measured max|delta| and tighten the bar in a pv diff-visible edit.
- 06-07: REVIEW-06-02's provisional non-aarch64 f32 bar (5.0e-6) is STILL unmeasured — this host is aarch64, so quantiles_abs_f32 (1e-6, measured 9.5367e-7) is what ran. The server test now prints what a first x86_64 run needs.
- DECLARED-RED WINDOW OPEN (06-14 -> 06-15): types::tests::no_cost_axis_is_pending fails on purpose, turning `make tier3` and CI `workspace-test` red. `workspace-test` is a REQUIRED check on protected `main`. Mitigation is push sequencing — land waves 12 and 13 in ONE push. Do NOT delete/weaken/ignore/CI-filter the test.
- Workspace lint debt blocks the plan-level clippy line: cargo clippy -p aprender-train --features setfit -- -D warnings exits 101 on pre-existing findings in aprender-compute and aprender-present-terminal. Zero are in aprender-train (--no-deps is rc=0). CLAUDE.md #2370 class; owned by toolchain-ceiling.yml, not by phase 5.
- bench_row.rs:37-43 claims canonical bytes are key-SORTED because no workspace crate enables preserve_order. MEASURED false: sorting reproduces neither the row nor the manifest digest, file order reproduces both. Not a correctness defect; the stated reason and the field-reordering property it claims are wrong. For 05-16 or 05-17, which touch that module.
- FINDING for 05-17: apr setfit bench verify-cell's printed 'scope:' prose (apr-cli/src/commands/setfit_bench.rs:2931) and bench report's 'verified:' header no longer enumerate the selection binding both doors now perform. Under-claiming, not a false attestation; assigned to 05-17 by threat T-05-16-05.
- D-ITEM-05-17-A: the bench row seal is BUILD-GRAPH DEPENDENT. serde_json/preserve_order (via pmcp v2.19.3) is in apr-cli graph and absent from aprender-train, so the same committed row verifies under apr and is refused as row_digest_mismatch under cargo test -p aprender-train. A cargo feature-unification change with no code change can flip the whole committed evidence set. Needs its own plan: the fix re-seals 40 rows, 40 selection manifests and the run manifest.
- RESOLVED (06.1-03, wave 2): regressor_prior_scale_min stays at 1e-153. The ~1e-7 usability-floor hypothesis was REFUTED by a 5-shape x 12-decade release campaign: at 1e-7 the seasonal series contributes 2.8e-12 relative to yhat, as negligible as at 1e-9 — so a floor there would refuse requests behaving exactly like ones it accepts — and there is no cliff, contribution decaying continuously and saturating between 1e-4 and 1e-2, shape-dependently, because prior_scale is a regularisation STRENGTH and shrinking the coefficient is what it is for. D-21 is instead closed at the OUTCOME: when a fitted coefficient is exactly zero (the optimiser starts at zero, so that means L-BFGS never moved it) the response says so. Threshold-free, shape-independent, and it refuses nothing previously accepted. See 06.1-03-SUMMARY.md.
- Verify t2c in 06.1-04-PLAN.md cannot see an #[ignore] added to part A or part B (it anchors bodies at fn, and the attribute precedes fn). Measured green on a gated part A. The in-tree Rust test invariance::parts_a_and_b_are_unconditional is the real guard; a later plan should replace the python body with a call to it.
- SC2 no-argument invariance gate (crates/aprender-forecast/src/invariance.rs:718) is RED under cargo nextest run --workspace --lib, the invocation CI runs; green under -p aprender-forecast. Deterministic, hash-stable, flipped by adding apr-cli to the package set. See 06.1 deferred-items.md item 5.
- 08-09..08-12 revised for option 3 (commit 2b75c14dc, 2026-09-26): 08-09 refuses both demo vectors; 08-10 offline-only; 08-11 records HOLD; D-18 live deploy deferred to the post-spike re-run.
- RESOLVED 2026-09-27 (was: 08-09 HALTED at its tracer on RescoreDrift zero_shot row 49 1.028e-5): debug session laya-rescore-drift fixed the RoPE inv_freq rounding (8e55e0bed); user decision option A keeps pack_rescore_probs_abs 1e-5, early_stopping 3d4b91da demonstrates GateFailed[ece_post] (exit 3), fixed_epochs d0f4e40d is refused with RescoreDrift fine_tuned row 59 4.667e-5 (exit 2). 08-09 complete (87c957bff). OPEN CONCERN, queued not acted on: the 1e-5 bar has ~1.5x headroom on early_stopping and the ladder final block 1.09x; x86_64 unmeasured (D-ITEM-08-09-B, spike-laya-calibration-slice-and-temperature-cap todo).
- [RESOLVED 2026-09-27] 08-17 deploy-refused blocker: the user chose option 1 ([iam] statements in the deploy config); redeploy loaded the model on pmcp.run's own post-deploy call, identity == H, Task 3 deployed-passed (08-17-SUMMARY.md). Open follow-ups: D-ITEM-08-17-D (pmcp.run platform issue, recommend only) and D-ITEM-08-17-E (cold margin 650 ms; 08-18/08-12)

### Roadmap Evolution

- Phase 06.1 inserted after Phase 6: Forecast exogenous inputs (Prophet regressors, NeuralProphet events, tier-safe cost bounds) — sequenced ahead of Phase 7 by user decision 2026-09-20; accepted cost is that the C-08 event-column term lands on a hard constant and is re-expressed as tier policy in Phase 7 (URGENT)

## Deferred Items

Items acknowledged and carried forward from project scope:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Encoder/objectives | Additional encoder families and contrastive losses | v2 | Project definition |
| Optimization | Accelerator support and quantization beyond the CPU/F32 lifecycle | v2 | Project definition |
| Tasks | Multilabel, hierarchical, explanation, and persistent-cache workflows | v2 | Project definition |

## Session Continuity

Last session: 2026-09-29T21:46:01.136Z
Stopped at: Completed 08-33-PLAN.md (contract-hygiene repair, local commits, NOT pushed). Next: plan 08-34 (CI audit, push, maintainer approval, CI evidence)
Resume file: None

## Accumulated Context

### Roadmap Evolution

- Phase 7 added: Tier-Resolved Door Limits — CR-01 disposition (UAT item 4, decided 2026-09-07): door bounds become a resolved profile, not hard constants, so one binary serves four deployment envelopes
- Phase 8 added (2026-09-25): Laya Decision Model — local fine-tune and thin MCP server, productising spikes 024–026; independent track
