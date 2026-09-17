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

- Compat: the frozen 0.3.1 corpus (`crates/facades/provable-contracts/compat/0.3.1/examples/lint.rs`) READS `GateResult` (`.passed`, `.skipped`, `.name`, `.detail`) and constructs no literal (`grep -c 'GateResult {'` → 0), so a new `GateResult` field compiles there; 11 literal constructors inside `crates/aprender-contracts/src/lint/` set it (measured; v2 said 13), and the two `LintReport` literals in `lint/trend.rs`'s `#[cfg(test)]` module (lines 242, 360) must compile too.
- `scripts/lib_baseline_ratchet.sh` comparand: merge-base(HEAD, origin/main) preferred, origin/main tip fallback, `BASELINE_RATCHET_BASE_REF` override.
- `scripts/check_ont_ratchet.sh:123` writes `"armed_gates": []` and `--write` overwrites the whole baseline.
- CI test wiring is `ci/explicit-test-commands.d/NNN-*.cmd` (ruling PMAT-3313, `.github/workflows/ci.yml:542-555`); ordinal 335 is free.
- ONT R-2 (verbatim): "Zero is a decline, never an accept. `n_files==0`, `checkable_n==0`, `shapes_n==0`, `focus_nodes_n==0`, a missed positive control → `Unknown{<reason>}`, exit 2, stderr `decline: <reason>`."

## v2 — what the round-1 grill changed (2 FAIL / 1 PASS, 2026-09-17)
- ci.yml `--test` line → a `ci/explicit-test-commands.d/335-…cmd` fragment (3/3 lanes).
- P1 acceptance runs all three Kani harnesses, not only 6-2; the three ids appear literally in `verdict.rs`.
- `Reason` has a total order (derive `Ord`, §3.4 declaration order); `meet(Unknown a, Unknown b) = Unknown(min(a, b))`.
- `GateResult` gains a `verdict` FIELD set by every gate constructor (was a derived method).
- Unknown reaches exit 2 through a typed `LintDeclined` in `contract_walk`, not through the error class.
- Comparand = `lib_baseline_ratchet.sh`'s resolution, not bare `HEAD` (which compares a commit with itself).
- `check_ont_ratchet.sh` preserves `armed_gates`; A_4 includes `make contracts`.
- Empty armed set → `Unknown{NotArmed}`, exit 2, on R-2 (lane 1 argued Pass as the K3 identity; R-2 forbids an accept on zero, so the spec decides it).

## v3 — what the round-3 grill changed (2 FAIL / 1 PASS, isolation clean; round 2 was voided by peer-session refs in a shared .git, paiml-implement#227)
- `check_ont_ratchet.sh`'s `self_test()` never mentions `armed_gates` (measured: 0), so "`--write` drops armed_gates → self-test RED" was vacuous: P3 adds that assertion to the self-test and A_3 runs `bash scripts/check_ont_ratchet.sh --self-test`.
- P3 scope adds `crates/aprender-contracts/src/lint/trend.rs` (test-module `LintReport` literals).
- `Fail` reaches `reject:` through a typed `LintRejected` (not the error class); `Unknown` through `LintDeclined`; shrink through `ArmedGatesShrank`.
- Open point 1 ruled by the grill: no resolvable comparand → print `armed_gates monotone: NOT CHECKED (no comparand)`, exit unchanged.
- Open point 2: `reverse-coverage` is unarmed by ruling, so a default `pv lint contracts/` stays rc 0 once `armed_gates` holds the 8 (measured in A_3, after `lint-baseline.json` is written).

## Phases and acceptance commands
| P | Scope | Change | A_i (a command) | Mutation (must go RED) |
|---|---|---|---|---|
| 1 | `crates/aprender-contracts/src/ontology/{mod,verdict}.rs`, `src/lib.rs` | `Verdict {Pass, Unknown(Reason), Fail}`; `Reason` = the 15 §3.4 reasons, `#[derive(Ord)]` in §3.4 order; `meet`: Fail absorbing, Pass identity, `Unknown(a)∧Unknown(b) = Unknown(min(a,b))`; `arm(v) = v == Pass`; `exit_code` 0/1/2; `from_label(&str) -> Option<Verdict>` over an explicit table — PASS→Pass, FAIL→Fail, SKIP→Unknown{Skip}, REPORT→Unknown{Report}, WARN→Unknown{Warn}, MANUAL/DEFER/NO-VERDICT→Unknown{NotRun}, do-not-implement-as-written→Fail, pv `accept`→Pass, `reject`→Fail, `decline`→Unknown{NotRun}, `error`→Unknown{ToolAbsent}; anything else → None; `from_shapes_report{violations, warnings, shapes_n, focus_n}` per §3.4 (shapes_n==0 → NoShapes before focus_n==0 → NoFocus); `#[cfg(kani)]` harnesses `kani_ont_6_1` (KANI-ONT-6-1: commutative, associative, idempotent, Pass identity, Fail absorbing, meet ≤ both), `kani_ont_6_2` (KANI-ONT-6-2: `arm(v) ⇒ v == Pass`), `kani_ont_6_3` (KANI-ONT-6-3: every table label maps to exactly one element; every `Reason` is reachable) | `cargo test -p aprender-contracts --lib ontology::verdict && cargo kani -p aprender-contracts --harness kani_ont_6_1 && cargo kani -p aprender-contracts --harness kani_ont_6_2 && cargo kani -p aprender-contracts --harness kani_ont_6_3 && grep -q KANI-ONT-6-2 crates/aprender-contracts/src/ontology/verdict.rs` | `arm(Unknown)=true` → `kani_ont_6_2` VERIFICATION FAILED; `meet` returns the left Unknown's reason → `kani_ont_6_1` FAILED |
| 2 | `.../ontology/arming.rs` | `ArmedGates` parsed from `lint-baseline.json`; file absent → the default 8; `armed_gates: []` explicit → meet is `Unknown{NotArmed}` (R-2); `check_monotone(comparand, current)` → `ArmedGatesShrank{dropped}`; `meet_armed(results, armed)`: unarmed printed `Unknown{NotArmed}` and excluded; armed but not run → `Unknown{NotRun}` IN the meet | `cargo test -p aprender-contracts --lib ontology::arming` | drop `composition` from a fixture's `armed_gates` vs its comparand → shrink test RED; empty-set → Pass → R-2 test RED |
| 3 | `crates/aprender-contracts/src/lint/{mod.rs, gates.rs, gates_extended.rs, composition_gate.rs, duplicate_stems.rs, strict_test_binding.rs}`, `crates/aprender-contracts-cli/src/{contract_walk.rs, commands/lint.rs, commands/lint_render.rs}`, `contracts/lint-baseline.json`, `scripts/check_ont_ratchet.sh`, new `crates/aprender-contracts-cli/tests/ont6_lint_verdict.rs`, new `ci/explicit-test-commands.d/335-aprender-contracts-cli-ont6-lint-verdict.cmd` | `GateResult.verdict` set by all 13 constructors, plus a test that `verdict` agrees with `passed`/`skipped` for every gate of a real run; `LintReport` gains `verdict`, `armed_gates`, `not_armed`, `armed_comparand` (serde-default); text output prints per-gate verdict + the armed meet; exit: meet Pass→0, Fail→1 `reject:`, Unknown→2 `decline: <reason>` via typed `LintDeclined` in `exit_code_for`/`verdict_for`; comparand resolved as `lib_baseline_ratchet.sh` (merge-base → origin tip → `--armed-baseline-ref` override); shrink → 3 `error: armed_gates shrank: <names>` via typed `ArmedGatesShrank`; **no comparand resolvable (no git, no ref)** → the monotone check prints `armed_gates monotone: NOT CHECKED (no comparand)` and does not change the exit (open point, below); `lint-baseline.json.armed_gates` = the 8; `check_ont_ratchet.sh` reads and preserves the current `armed_gates` | `cargo test -p aprender-contracts-cli --test ont6_lint_verdict && bash scripts/check_explicit_test_commands.sh && bash scripts/check_ont_ratchet.sh --self-test && . scripts/pv_bin.sh && "$PV" lint contracts/ >/dev/null; echo rc=$?` (rc 0 on this branch, as on main before P3) | `exit_code_for(LintDeclined)→0` → test RED; a `check_ont_ratchet.sh --write` that drops `armed_gates` → its self-test RED |
| 4 | `contracts/ont-verdict-lattice-v1.yaml`, `contracts/census.json`, `README.md` | kernel-kind contract: equations (the meet table), proof_obligations, falsification_tests, `kani_harnesses` KANI-ONT-6-1/2/3 naming `kani_ont_6_1..3`; regenerate census + README | `. scripts/pv_bin.sh && "$PV" validate contracts/ont-verdict-lattice-v1.yaml && make contracts && make readme-sync-check` | drop a `kani_harnesses` entry's `obligation` → `pv validate` rc≠0 |
| 5 | paiml/paiml-implement issue; ONT-6 probe | RAH-005 issue (`prah gate reduce` binds to `ont-verdict-lattice-v1` by id); run ONT-6's probe conjuncts except `merged` | each conjunct rc 0 in this worktree | — |

## Routing (route.sh, 2026-09-17): cross/impl → agy-goal · review → agy-quorum · plan → agy-plan (basis=absent). Orchestration, A_i re-runs, push, PR → self.

## Open points for the grill
1. No resolvable comparand (a tarball, no git, a shallow clone without origin/main): print NOT CHECKED and leave the exit alone, or decline (exit 2)? Declining makes every non-git consumer of `pv lint` exit 2; CI always has the ref.
2. Is there any path where a default `pv lint contracts/` on aprender main changes exit code after P3 (an armed gate reporting Skipped while validate passes)? Measured before P3 merges.
