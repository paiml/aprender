# PMAT-3451 — receipt

**Ticket:** PMAT-3451 (issue #3451) — ONT-6: one Verdict lattice, Kani-proved; per-repo arming; exit-vocabulary mapping.
**Spec:** paiml/infra `docs/specifications/paiml-ontology.md` (ONT-001 v4.4, merged as infra `8d3cd16bb8`) §3.4, §3.9, §5 ONT-6, R-2.
**Kind:** code (`kind-gate.sh PMAT-3451 docs/roadmaps/roadmap.yaml --base origin/main` → `kind=code ticket=PMAT-3451 files=30`).
**Branch:** `PMAT-3451-ont-6-verdict-lattice`, base merge-base `dc6fb687c`. Built in an independent clone
(`~/.cache/paiml-implement/wt/aprender-PMAT-3451`, its own `.git`); `~/src/aprender` was not modified.

orch_model: opus-5 [V]   orch_class: opus   orch_decision: admit   orch_basis: file
fable_binding: false   quota_age_h: absent   quota_mark: ?   k_measured_at_set: [U]

`model-gate.sh PMAT-3451 docs/roadmaps/roadmap.yaml --session 9e561dfb…` at the receipt: `model=opus-5 class=opus
decision=admit basis=file` (tier 2 meets required tier 1; ONT-001 R-22 admits Opus 5 on a code row).
`k_measured_at_set` is `[U]`: the `goal.sh set` state lived under `/run/user` and was wiped by a session restart
mid-ticket (the same restart that cost the first P3 GREEN, below). The session also carried PMAT-218 (paiml-implement)
and infra#649 before this ticket, so its measured turn count (529) is not this ticket's.

## Operator rulings (2026-09-17, verbatim from AskUserQuestion answers)

- armed set: "8: drop reverse-coverage (Recommended)"
- fleet labels: "Existing reasons (Recommended)"
- exit 3: "Only the new error (Recommended)"
- the pre-commit complexity refusal: "Refactor the offenders (Recommended)"

## Plan (v3, grilled; the plan file is replaced by this receipt)

| P | Scope | A_i |
|---|---|---|
| 1 | `ontology/{mod,verdict}.rs` | `cargo test -p aprender-contracts --lib ontology::verdict` + `cargo kani --harness kani_ont_6_{1,2,3}` + `grep -q KANI-ONT-6-2` |
| 2 | `ontology/arming.rs` | `cargo test -p aprender-contracts --lib ontology::arming` |
| 3 | `lint/*`, CLI `contract_walk.rs`, `commands/lint*.rs`, `cli.rs`, `lib.rs`, `lint-baseline.json`, `check_ont_ratchet.sh`, `tests/ont6_lint_verdict.rs`, `ci/explicit-test-commands.d/335-…cmd` | `cargo test -p aprender-contracts-cli --test ont6_lint_verdict && check_explicit_test_commands.sh && check_ont_ratchet.sh --self-test && pv lint contracts/` rc 0 |
| 4 | `contracts/ont-verdict-lattice-v1.yaml`, `census.json`, `README.md` | `pv validate … && make contracts && make readme-sync-check` |
| 5 | paiml-implement issue; probe | the ONT-6 probe's conjuncts except `merged` |

routes:
  ph1  class=plan           route=agy-plan w=1.00 basis=absent effort=1[U]
  ph2  class=impl           route=agy-goal w=1.00 basis=absent note=fable-binding effort=1[U] bucket_collision=true
  ph3  class=impl           route=agy-goal w=1.00 basis=absent note=fable-binding effort=1[U] bucket_collision=true
  ph4  class=impl           route=agy-goal w=1.00 basis=absent note=fable-binding effort=1[U] bucket_collision=true
  ph5  class=orchestration  route=self w=100.00 basis=absent
  ph6  class=review         route=agy-quorum w=1.00 basis=absent effort=1[U]

**ph4 deviation, recorded.** The P3 goal lane (`3128f1ca-…`) self-reported `achieved`; the orchestrator's review
found out-of-scope edits (aprender-core `xet.rs`, aprender-serve `batched_qkv.rs`, aprender-train
`cuda_trainer.rs`, a stray `replace.py`), a false "fmt clean" claim, and four defects in its `lint.rs` (an explicit
comparand that failed OPEN, a wrong comparand label, `LintRejected` counting every passed gate instead of the armed
ones, git logic at cyclomatic 32 / cognitive 96). Only its in-scope patch was applied, and that verified tree was
then lost uncommitted to the session restart. P3 GREEN was rewritten by the orchestrator in the persistent clone
(`route=self` in practice), fixing all four, and pushed at the first green.

## Dispatch ledger

| Phase | Mode | Agent | Lanes / models (measured) | agy conversations | Outcome |
|---|---|---|---|---|---|
| ph1 r1 | grillme ×3 | paiml-agy-delegate (opus) | gemini-3.1-pro-high FAIL · gemini-3.8-flash-high FAIL · gemini-3.7-flash-high PASS | 84e70168, 1f69515f, 08857bc1 | not agreed → plan v2 |
| ph1 r2 | grillme ×3 | paiml-agy-delegate (opus) | all three lanes exit 3 (isolation: a linked worktree shares `.git`; peer sessions' refs moved) | — | VOID, paiml-implement#227 |
| ph1 r3 | grillme ×3 | paiml-agy-delegate (opus) | gemini-3.1-pro-high FAIL · gemini-3.8-flash-high FAIL · gemini-3.7-flash-high PASS | 119d2aa6, 240abb4a, 32c0b48e | not agreed; findings folded → plan v3 |
| ph2 | goal ×1 writes | paiml-agy-delegate (opus) | gemini-3.1-pro-high | 7d650b83 | achieved; kani_ont_6_3 bound raised 16→30 by the orchestrator |
| ph3 | goal ×1 writes | paiml-agy-delegate (opus) | gemini-3.1-pro-high | 8de30355 | achieved |
| ph4 | goal ×1 writes | paiml-agy-delegate (opus) | gemini-3.1-pro-high | 3128f1ca | self-reported achieved; see deviation |
| ph6 r1 | quorum ×3 (`quorum-review.sh --base origin/main`) on `61fa95aab` | paiml-agy-delegate (opus) | gemini-3.1-pro-high FAIL · gemini-3.8-flash-high FAIL · gemini-3.7-flash-high PASS | 252f06b1, cbaa220d, a43a4af5 | not agreed; the delegate hit maxTurns 30 after the artifact was written (read from disk, not resumed) |

Slots: one Claude subagent at a time for this ticket (`running_peak=1`), `slots=3`. The events file for this
session holds 2 `SessionStart` + 1 `SubagentStop` after the restart; the pre-restart events were under the wiped
`/run/user` directory — a host hard crash at 15:30Z (a peer session's resume note records the 4th unclean reboot in
24 h). I-3: `attempted=7 denied=0 running_peak=1 slots=3`.

## Diff quorum round 1 → what changed

Both FAILs (lanes 1 and 2) named one thing, cited at `crates/aprender-contracts/src/lint/mod.rs:255`: `61fa95aab`
let an opt-in flag arm its own gate for that run (`LintConfig::requested_gates` + `ArmedGates::arm_requested`), a
design choice beyond the ticket, flagged as such in the receipt. Lane 2 added a real inconsistency it caused:
`contracts/ont-verdict-lattice-v1.yaml`'s `armed_meet` precondition reads A from `lint-baseline.json` only.
Spec §3.9 agrees with the lanes ("only armed gates enter that repo's meet"; arming is the per-repo declaration),
so the extension is **removed**: both functions, their test, and the CLI parameter. In its place
`a_gate_a_flag_ran_is_reported_but_not_armed` pins the spec behaviour: `--strict-test-binding` runs the gate, the
gate line prints `→ Unknown(NotArmed) [gate: <its verdict>]`, and it stays out of the meet. Round 1's artifact is
kept outside the tree at `~/.cache/paiml-implement/ont6-evidence/quorum-r1/`.

The consequence, measured rather than argued: `scripts/dogfood.sh` runs `pv lint contracts --binding
contracts/binding.yaml --crate-dir .` and grades the exit code. reverse-coverage PASSES there today (67.3% bound ≥
50% threshold), so no present verdict changes; a future reverse-coverage failure under those flags will no longer
fail that run unless `contracts/lint-baseline.json` arms `reverse-coverage` — and arming it makes every flagless
`pv lint` decline (Skip). That trade-off is a spec question for ONT-001 §3.9 (a gate that is only meaningful when
requested), recorded under Gaps.

## What changed

| File | Change |
|---|---|
| `crates/aprender-contracts/src/ontology/verdict.rs` | `Verdict`, `Reason` (15), `meet`/`arm`/`exit_code`/`decline_line`, `from_gate`, `from_shapes_report`, `FLEET_LABELS`/`from_label`, `Display` + `Serialize` as `Pass`/`Fail`/`Unknown(<Reason>)`; `kani_ont_6_1..3` |
| `crates/aprender-contracts/src/ontology/arming.rs` | `ArmedGates` (`DEFAULT_ARMED` = the ruled 8, `from_baseline`), `check_monotone` → `ArmedGatesShrank`, `meet_armed` |
| `crates/aprender-contracts/src/lint/{mod,gates,gates_extended,composition_gate,duplicate_stems,strict_test_binding,trend}.rs` | `GateResult.verdict` in all 11 constructors; `LintReport.{verdict, armed_gates, not_armed, armed_monotone}` + `LintReport::arm`; `run_lint` arms the default set |
| `crates/aprender-contracts-cli/src/commands/lint_arming.rs` (new) | the comparand: merge-base(HEAD, origin/main) → origin/main tip → `--armed-baseline-ref`; explicit ref fails closed; `GIT_DIR`/`GIT_WORK_TREE`/`GIT_INDEX_FILE` dropped |
| `crates/aprender-contracts-cli/src/commands/lint.rs` | shrink check before the lint run; the exit is `meet_exit` over the armed meet; watch mode arms from the declared baseline each tick |
| `crates/aprender-contracts-cli/src/commands/lint_render.rs` | per-gate `→ <verdict>` (unarmed: `Unknown(NotArmed) [gate: …]`), `armed meet:` and `armed_gates monotone:` lines, `Result:` follows the meet |
| `crates/aprender-contracts-cli/src/contract_walk.rs` | `LintDeclined` (2, `decline`), `LintRejected` (1, `reject`), `ArmedGatesShrank` (3, `error`) |
| `crates/aprender-contracts-cli/src/{cli,lib}.rs` | `--armed-baseline-ref` |
| `crates/aprender-contracts-cli/tests/ont6_lint_verdict.rs` + `ci/explicit-test-commands.d/335-…cmd` | 7 end-to-end cases |
| `contracts/lint-baseline.json` | `armed_gates` = the 8 |
| `scripts/check_ont_ratchet.sh` | `armed_gates_of` + `measure()` carry the declaration; `--write` writes via a temp file (`measure > "$BASELINE"` truncated the file before reading it); 2 self-test rows |
| `crates/aprender-contracts/src/lint/gates_extended.rs`, `lint_render.rs`, `scripts/complexity_baseline.txt` | ruled refactor: `run_verify_gate` 31/110 → 5/7, `collect_test_fns` 9/32 → 5/11, `print_findings_grouped` 15/36 → 6/9 (same class, in a touched file); exactly their 3 rows removed |
| `docs/audits/surface_audit.csv` | 38 `cli.rs:N` citations +4, each checked against origin/main's line |
| `contracts/ont-verdict-lattice-v1.yaml`, `contracts/census.json`, `README.md` | the kernel contract; census 1791 → 1792; README count regenerated |
| `docs/roadmaps/entries/PMAT-3451.yaml`, `docs/roadmaps/roadmap.yaml` | spec + acceptance criteria written from the diff; aggregate regenerated |

## Evidence — every command re-run by the orchestrator

verification:
  cmd=cargo test -p aprender-contracts-cli --test ont6_lint_verdict  claimed_exit=0  rerun_exit=0  log_path=~/.cache/paiml-implement/ont6-evidence/test-contracts-cli.log  sha256=0
  cmd=cargo test -p aprender-contracts --lib  claimed_exit=0  rerun_exit=0  log_path=~/.cache/paiml-implement/ont6-evidence/test-contracts-lib.log  sha256=0
  cmd=cargo test -p aprender-contracts-cli  claimed_exit=0  rerun_exit=0  log_path=~/.cache/paiml-implement/ont6-evidence/test-contracts-cli.log  sha256=0
  cmd=cargo clippy -p aprender-contracts -p aprender-contracts-cli --all-targets -- -D warnings  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3451-receipt.md  sha256=0
  cmd=cargo kani -p aprender-contracts --harness kani_ont_6_1  claimed_exit=0  rerun_exit=0  log_path=~/.cache/paiml-implement/ont6-evidence/kani-kani_ont_6_1.log  sha256=0
  cmd=cargo kani -p aprender-contracts --harness kani_ont_6_2  claimed_exit=0  rerun_exit=0  log_path=~/.cache/paiml-implement/ont6-evidence/kani-kani_ont_6_2.log  sha256=0
  cmd=cargo kani -p aprender-contracts --harness kani_ont_6_3  claimed_exit=0  rerun_exit=0  log_path=~/.cache/paiml-implement/ont6-evidence/kani-kani_ont_6_3.log  sha256=0
  cmd=bash scripts/check_ont_ratchet.sh --self-test  claimed_exit=0  rerun_exit=0  log_path=~/.cache/paiml-implement/ont6-evidence/ont-ratchet-selftest.log  sha256=0
  cmd=MUTATION check_ont_ratchet.sh --write back to measure > "$BASELINE"  claimed_exit=1  rerun_exit=1  log_path=~/.cache/paiml-implement/ont6-evidence/ont-ratchet-selftest-mutant.log  sha256=0
  cmd=bash scripts/check_ont_ratchet.sh  claimed_exit=0  rerun_exit=0  log_path=~/.cache/paiml-implement/ont6-evidence/g-ont-ratchet-check.log  sha256=0
  cmd=bash scripts/check_explicit_test_commands.sh  claimed_exit=0  rerun_exit=0  log_path=~/.cache/paiml-implement/ont6-evidence/g-explicit-test-cmds.log  sha256=0
  cmd=bash scripts/check_dogfood_coverage.sh  claimed_exit=0  rerun_exit=0  log_path=~/.cache/paiml-implement/ont6-evidence/g-dogfood-coverage.log  sha256=0
  cmd=PATH=<pmat 3.40.1> bash scripts/check_complexity_ratchet.sh  claimed_exit=0  rerun_exit=0  log_path=~/.cache/paiml-implement/ont6-evidence/complexity-ratchet-3.40.1.log  sha256=0
  cmd=make contracts  claimed_exit=0  rerun_exit=0  log_path=~/.cache/paiml-implement/ont6-evidence/make-contracts.log  sha256=0
  cmd=pv lint contracts/ --format json (before d3a22ae21 vs after, normalised)  claimed_exit=0  rerun_exit=0  log_path=~/.cache/paiml-implement/ont6-evidence/lint-after.norm.json  sha256=0
  cmd=MUTATIONS M1..M6 (committed tree 5a2c96d88, each restored)  claimed_exit=101  rerun_exit=101  log_path=~/.cache/paiml-implement/ont6-evidence/mut-M1.log  sha256=0

What each line observed:

- ont6_lint_verdict: 7 passed. contracts lib: 1554 passed, 0 failed, 5 ignored. aprender-contracts-cli: every
  suite `ok`. clippy `-D warnings`: clean. Pre-commit hook on GREEN and P4: format, complexity, SATD, docs ✅.
- Kani 0.67.0 at `81d8f725e`: `VERIFICATION:- SUCCESSFUL` ×3 (4.94 s, 1.07 s, 2.79 s). P1's RED record:
  `kani_ont_6_2` FAILED against the stubbed arm; `kani_ont_6_3` FAILED at unwind 16 (memcmp unwinding assertion).
- `pv lint contracts/` JSON, before (`d3a22ae21`) vs after, with timings, the four new report fields, per-gate
  `verdict` and run-history `is_new` removed: **byte-identical** — `passed: true`, 9 gates with identical `detail`
  (verify 26 refs / 26 found), 1198 findings. After: `verdict: Pass`, `not_armed: [reverse-coverage]`,
  `armed_monotone: OK against merge-base(HEAD, origin/main) dc6fb687c639 (0 committed, 8 declared)`.
- Complexity ratchet under the pinned pmat 3.40.1: `PASS (D2): dc6fb687c vs 5a2c96d88 — none new, none grown`,
  the three refactored functions `RESOLVED`. Under this host's `~/.cargo/bin/pmat` 3.40.2 it FAILs on the
  instrument check (recorded 3.40.1) before judging anything, and `--update` under 3.40.2 also dropped six
  unrelated rows — instrument drift on this host, not this change, left out of the diff.
- Mutations, each on the committed tree and restored with `git checkout --`:
  M1 `meet_exit` Unknown → Ok: `explicit_empty_armed_set_declines_r2` FAILED (exit 0).
  M2 `check_monotone` ignored: `dropping_a_committed_armed_gate_is_exit_3` FAILED (exit 0).
  M3 explicit ref outside a work tree → NOT CHECKED: `an_explicit_comparand_that_cannot_be_read_is_an_error_not_a_skip` FAILED (exit 0).
  M4 `exit_code_for` loses `LintDeclined`: `explicit_empty_armed_set_declines_r2` FAILED (exit 1).
  M5 (after round 1) `run_lint` arms every gate that ran: `a_gate_a_flag_ran_is_reported_but_not_armed` FAILED.
  M6 `skipped_gate` verdict → Pass: `every_gate_verdict_agrees_with_passed_and_skipped_on_the_real_corpus` FAILED (gate reverse-coverage).
  P4: `kani_harnesses[1]` without `obligation` → `pv validate` rc 1, `missing field obligation`.
- `pv validate contracts/ont-verdict-lattice-v1.yaml`: 0 errors, 0 warnings. `pv lint contracts/ --strict-test-binding`:
  0 findings on the new file. `make contracts`: rc 0 (census 1792, README count, provenance self-test, 1555 engine tests at `81d8f725e`).
- ONT-6 probe (infra main `scripts/pvl/lib.sh`, bash), every conjunct except `merged`: `tracked` 0 · `rc_is 0 pv validate` 0 ·
  `json_object lint-baseline.json` 0 · `armed_gates|length ≥ 8` 0 · `present KANI-ONT-6-2 verdict.rs` 0.

## Jidoka

| Defect | Owner | Whys → fix |
|---|---|---|
| `check_ont_ratchet.sh --write` truncated `lint-baseline.json` before reading `armed_gates` | aprender, this PR | the preservation read and the write were the same path in one redirection → temp file + `mv`; self-test row runs `--write` in place (mutant 12/13) |
| a linked-worktree lane shares `.git`; peer refs voided plan grill round 2 | paiml-implement | filed paiml-implement#227; rounds after it used an independent clone |
| scratchpad and `/run/user` wiped at a session restart: lost the hand-off and a verified uncommitted P3 GREEN | harness | persistent clone under `~/.cache`; commit and push at the first green |
| `cargo` in the Bash tool is a zsh function overwriting `CARGO_TARGET_DIR` with the shared `/mnt/nvme-raid0/targets/aprender` | host shell | `command cargo` + an inline private target dir on every call; a stale `pv` was caught by mtime |
| `~/.cargo/bin/pmat` is 3.40.2 on lambda-labs while aprender and infra pin 3.40.1 | fleet pin drift | not fixed here; the ratchet was run with forjar's cached 3.40.1 |

## Estimates

K̂ 90 (ONT-001 row). Actual turns for this ticket are not separable from the session's 529 (`[U]`, see identity).

## Gaps

- No aprender CI lane runs `cargo kani`; L3 is local evidence, re-checked in CI only through the L2 tests.
- RAH-005's binding lives in paiml/paiml-implement; this PR files the issue and does not implement it.
- ONT-001 §3.9 does not say how a gate that is only meaningful when a flag requests it (reverse-coverage, strict-test-binding) should arm; see "Diff quorum round 1".
- The ONT-6 probe's `merged ONT-6` conjunct turns true only after merge and an infra ledger row binding it.

verdict: PARTIAL(review-pending) — the pre-PR diff quorum has not yet run on this head.
