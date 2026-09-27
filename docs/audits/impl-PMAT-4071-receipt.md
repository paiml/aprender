# impl receipt — PMAT-4071 (ONT-2c, aprender#4071)

- ticket: PMAT-4071 · kind: code · branch: PMAT-4071-ont-2c-owl-writer · base: origin/main 49fe19c28
- spec: paiml/infra docs/specifications/paiml-ontology.md @ 948ae923, row ONT-2c (line 626), §3.8
- ruling (cop aprender-cf, 2026-09-23, refined per infra-83): (B) told-closure in-tree, precondition-guarded,
  ADVISORY (never arms, R-7); (A) ELK 0.4.3 via a JVM is the independent oracle, agreement required at the
  RELEASE gate only; no JVM ⇒ `decline:` NOT MEASURED (RED at release, never skipped); a planted unintended
  subsumption must go RED in both.
- session model: claude-opus-5-5 (model-gate.sh measured)

## What the row asks → where it is

| row RED clause | where | evidence |
|---|---|---|
| writer output re-parsed by the oracle equals the fixture's axiom set | tests/oracle/owl (horned-owl 3.0.0) `roundtrip` vs HAND-WRITTEN tests/fixtures/ont/owl/axioms.txt | 13 == 13, rc 0 |
| two writes byte-identical | owl.rs BTreeSet; `ont2c_two_writes_are_byte_identical`; `pv ontology export --owl contracts/ontology.yaml \| cmp - contracts/ontology.ofn` | equal |
| every Σ key mapped or in `not_expressible` | exhaustive destructure of `Sigma` (compile-time); `OwlError::Unexpressed`; Σ declares 6 keys | `ont2c_undeclared_unexpressed_key_is_refused` |
| `acyclic` yields no axiom | owl.rs; `ont2c_acyclic_yields_no_axiom_and_is_accounted` | pass |
| `tbox-report.json` has `advisory: true` | contracts/tbox-report.json | tracked, fresh (lib test) |
| gate maps it to `Unknown{Advisory}` | lint/tbox_gate.rs (no Pass arm) + CLI | `pv lint contracts/ --gate tbox` → `decline: Advisory`, rc 2 |
| inferred subsumption between distinct Σ concepts → `unintended_subsumptions` non-empty | owl::tbox; ELK arm positive control | `ont2c_positive_control_a_planted_subsumption_is_unintended`; ELK planted SubClassOf(Contract Symbol) → differential RED |
| mutation: write `acyclic` as transitive + irreflexive → oracle round-trip RED | oracle roundtrip + kinds | M1 (writer mutant) rc 1; M2 (literal Transitive+Irreflexive) rc 1; kinds rc 1 |

## Measured (lambda, this branch)

- aprender-contracts --lib: 1717 passed, 0 failed. aprender-contracts-cli: all test targets green.
- clippy -D warnings (aprender-contracts, aprender-contracts-cli, --lib --bins): clean. fmt: clean.
- check_complexity_ratchet: PASS 49fe19c28 vs 635fe04e5 (the first commit's `classify` at cognitive 54 was
  refactored). check_include_files, check_guards_are_wired: rc 0. roadmap aggregate idempotent, sorted.
- oracle (make oracle-owl arms): roundtrip rc 0; kinds 36/36 admitted; elk agree (consistent, 0 entailed),
  positive control RED, ignored kinds {ObjectPropertyRange, SymmetricObjectProperty} only; 4 oracle unit tests.
- must-RED: M1 writer mutant (acyclic emits an axiom) roundtrip rc 1 · M2 literal Transitive+Irreflexive rc 1 ·
  M3 kinds on it rc 1 · M4 tampered tbox-report → elk rc 1 · M5 no JVM → rc 2 `decline: NOT MEASURED`.

## Findings raised (not fixed here)

- `make oracle` (the SHACL differential) and now `make oracle-owl` are invoked by NO workflow and not by
  scripts/dogfood.sh — "release gate only" is stated, not wired. Reported to the cop; release-surface change.
- ELK 0.4.3 ignores ObjectPropertyRange and SymmetricObjectProperty (measured). Recorded, bounded, RED on any
  other ignored kind. OWL 2 EL excludes symmetric properties; §3.8's mapping writes them anyway (kept per spec).
- JVM on lambda needs a forjar declaration + infra debt-census row (infra-83). ELK pin sent to infra-83.

## Gaps

- The oracle is not run per PR by design (R-13); its green is this receipt's measurement on lambda.
- The in-tree precondition is structural (the writer emits only admitted kinds); the oracle's `kinds` arm is the
  independent confirmation on the live file.

## Re-verification and re-quorum (aprender-75, 2026-09-24; the author's session had ended)

The prior quorum was at `27315ed18`, and `9fad260dc`, the clippy `--all-targets` fix, came after it. The fix only
moves `owl_tests.rs` inline as `mod tests` and rustfmts the CLI test. Lines 1–374 of `owl.rs` are byte-identical
across the two heads.

Re-run in a fresh worktree at `9fad260dc` on lambda:
- fmt clean; clippy `--all-targets -D warnings` clean;
- `aprender-contracts --lib` **1717 passed**; `ont2c_owl_tbox` **9/9**;
- the row's probe: `ontology.ofn` and `tbox-report.json` tracked, the jq clause holds, and export is byte-identical;
  `pv lint --gate tbox` gives `decline: Advisory` (rc 2);
- `make oracle-owl-check` rc 0: roundtrip 13 == fixture, kinds 36 admitted, ELK agrees, the positive control RED,
  and `tbox-differential.json` shows no drift;
- the row's mutation, planted independently (`TransitiveObjectProperty` + `IrreflexiveObjectProperty` on `refines`):
  roundtrip rc 1 (`written but not expected`), kinds rc 1 (`outside the told-closure precondition`).

Re-quorum at `9fad260dc` (`docs/audits/quorum-PMAT-4071.json`): **3/3 PASS**, agy gemini-3.1-pro-high,
gemini-3.8-flash-high and gemini-3.7-flash-high, no dissent. All lanes exited 3 on isolation, and every cause is
attributed to foreign worktrees; the flag stays `partial: true`.

**Open dependency:** paiml/infra#965 (the ELK 0.4.3 pin in `ont-oracle-ledger.md`, and forjar's
`ont-oracle-jvm`) is still an OPEN PR. The oracle code pins ELK by sha256 on its own; the ledger row is where the
spec records the pin.

verdict: DONE (code) — quorum receipt at `9fad260dc`; not armed.

