# PMAT-3451 — ONT-6 plan (for the plan grill; replaced by the receipt at PR time)

Spec: paiml/infra `docs/specifications/paiml-ontology.md` v4.3 (sha 512a16d5e09c) §5 ONT-6, §3.4, §3.9 (v3.1 §3.5 text: "`armed_gates` … monotone against the committed value. Every `pv lint` gate is computed and printed everywhere; only armed gates enter that repo's meet; unarmed print `Unknown{NotArmed}` and are excluded. The eight pre-existing gates are armed by default."). Issue #3451. Base aprender main dc6fb687c.

## Operator rulings (2026-09-17, verbatim)
- armed set: "8: drop reverse-coverage (Recommended)" — validate, audit, score, verify, enforce, enforcement-level, duplicate-stems, composition. `shapes` is appended by ONT-4b's own PR.
- fleet labels: "Existing reasons (Recommended)" — MANUAL, DEFER, NO-VERDICT → Unknown{NotRun}; do-not-implement-as-written → Fail; SKIP → Unknown{Skip}; REPORT → Unknown{Report}; WARN → Unknown{Warn}; pv `error` → Unknown{ToolAbsent}.
- exit 3: "Only the new error (Recommended)" — typed `ArmedGatesShrank` → 3 in `contract_walk::exit_code_for`; every existing error keeps 1.

## Measured facts
- `contracts/lint-baseline.json` = `{_spec, armed_gates: [], ont}`.
- `run_lint` (crates/aprender-contracts/src/lint/mod.rs) runs 9 always-on gates + opt-in strict-test-binding; `GateResult{name, passed, skipped, duration_ms, detail, extra}`; `GateDetail` is FROZEN at eight variants (0.3.1 compat corpus) — new payloads go in `extra`, never a new variant.
- `contract_walk::{exit_code_for, verdict_for}` is the one exit-vocabulary definition (EV-1): ZeroContracts → 2 decline, ParseErrors → 1 reject, else → 1 error.
- Kani 0.67.0 runs on the author host; no aprender workflow runs `cargo kani` (only `--test kani_harness_generation`).

## Phases and acceptance commands
| P | Scope | Change | A_i (a command) | Mutation (must go RED) |
|---|---|---|---|---|
| 1 | `crates/aprender-contracts/src/ontology/{mod,verdict}.rs`, `src/lib.rs` | `Verdict {Pass, Unknown(Reason), Fail}` with the 15 §3.4 reasons; `meet` (= min; K3); `arm(v) = v == Pass`; `exit_code` 0/1/2; `from_label(&str) -> Option<Verdict>` over the closed fleet vocabulary above; `from_gate(passed, skipped)`; `from_shapes_report{conforms, violations, warnings, shapes_n, focus_n}`; `#[cfg(kani)]` harnesses KANI-ONT-6-1 (meet laws), KANI-ONT-6-2 (`arm(v) ⇒ v == Pass`), KANI-ONT-6-3 (every (label, reason) → exactly one element) | `cargo test -p aprender-contracts --lib ontology::verdict && cargo kani -p aprender-contracts --harness kani_ont_6_2` | `arm(Unknown)=true` → `cargo kani --harness kani_ont_6_2` VERIFICATION:- FAILED |
| 2 | `.../ontology/arming.rs` | `ArmedGates` from lint-baseline.json (absent/empty file → the default 8); `check_monotone(committed, current)` → `ArmedGatesShrank` naming the dropped gates; `meet_armed(results, armed)`: unarmed → printed `Unknown{NotArmed}`, excluded; an armed gate that did not run → `Unknown{NotRun}` **in** the meet | `cargo test -p aprender-contracts --lib ontology::arming` | drop `composition` from `armed_gates` in a fixture → shrank test RED |
| 3 | `crates/aprender-contracts/src/lint/mod.rs`, `crates/aprender-contracts-cli/src/{contract_walk.rs, commands/lint.rs, commands/lint_render.rs}`, `contracts/lint-baseline.json`, new `crates/aprender-contracts-cli/tests/ont6_lint_verdict.rs`, one `--test` line in `.github/workflows/ci.yml` | `GateResult::verdict()` (method; no struct or `GateDetail` change); report prints per-gate verdict, the armed meet, and NotArmed gates; JSON adds report-level `verdict`, `armed_gates`, `not_armed`; exit Pass→0, Fail→1 `reject:`, Unknown→2 `decline:`; committed `armed_gates` read from `git show <ref>:contracts/lint-baseline.json` (`--armed-baseline-ref`, default `HEAD`) → shrank → 3 `error: armed_gates shrank: <names>`; `lint-baseline.json.armed_gates` = the 8 | `cargo test -p aprender-contracts-cli --test ont6_lint_verdict` | exit mapping `Unknown→0` → test RED; `exit_code_for(ArmedGatesShrank)→1` → test RED |
| 4 | `contracts/ont-verdict-lattice-v1.yaml`, `contracts/census.json`, `README.md` | kernel-kind contract: equations (meet table), proof_obligations, falsification_tests, kani_harnesses KANI-ONT-6-1/2/3; regenerate census + README count | `. scripts/pv_bin.sh && "$PV" validate contracts/ont-verdict-lattice-v1.yaml && make readme-sync-check` | remove a kani_harnesses entry's `obligation` → validate rc≠0 |
| 5 | paiml/paiml-implement issue; ONT-6 probe | RAH-005 issue (`prah gate reduce` binds to `ont-verdict-lattice-v1` by id); run ONT-6's probe minus `merged` | probe conjuncts rc 0 in this worktree | — |

## Routing (route.sh, 2026-09-17): cross/impl → agy-goal · review → agy-quorum · plan → agy-plan (basis=absent). Orchestration, A_i re-runs, push, PR → self.

## Open design points for the grill (not ruled)
1. Meet over an **empty** armed set (explicit `armed_gates: []` in a repo that opts out): Pass (the identity) contradicts R-2 ("zero is a decline"). Proposal: empty explicit set → `Unknown{NotArmed}` → exit 2.
2. "Committed value" for monotonicity: `HEAD` catches a working-tree shrink but not a PR that commits the shrink; `origin/<default>` catches the PR but needs a fetched ref in CI. Proposal: flag defaulting to `HEAD`, aprender CI passes the PR base.
3. `pv lint` today exits 0/1 from `passed`; after P3 an armed gate that is Skipped (only when validation failed — then validate is Fail and the meet is Fail anyway) cannot introduce a new exit 2 on aprender. Is there any path where a default `pv lint contracts/` run on aprender main changes exit code? Measure before P3 merges.
4. `from_label` is case- and spelling-sensitive (`NotRun` vs `NOT-RUN` vs `not_run`): accept an explicit alias table, refuse anything else (None), and Kani-prove the table has no label mapping to two elements.
