---
status: partial
ticket: PMAT-4075
row: ONT-4e
issue: 4075
model: "claude-opus-5-5 (infra ONT-001 session): schema, checker, reasoner, gate, contract, wiring"
---
# impl-PMAT-4075 — ONT-4e · requires/ensures as first-class; the Liskov rule on refines, witness-checked (#4075; ONT-001 §3.5, §5 ONT-4e, R-20)

Stacked on #4248 (ONT-5, `PMAT-4074-ont5-pv-sat` @ 0f13c6ef) and #4137 (ONT-4d, `PMAT-4070-ont-4d-subsumption` @ ed66b5dc), merged at c0435711a.

## What lands
- `schema::Clause` `{id, statement, formal, formal_status}`; `Contract.requires` / `Contract.ensures` (skipped when empty); `CONTRACT_TOP_LEVEL_FIELDS` 16 → 18.
- `ontology::liskov`: three directed obligations. **pre** has premise B.requires and conclusion A.requires; **post** has premise A.ensures and conclusion B.ensures; **inv** has premise A.invariants and conclusion B.invariants. Atoms are opaque (`formal`, whitespace collapsed). A certificate is either a `chain` or a `counter_model`. `check` refuses a chain from a different atom, a counter-model that hides or invents a violation, and a witness for other pairs. `pc_checker` refuses `fixtures/liskov-corrupt.json` on every run.
- `lint::refines_gate` (gate 18, `refines`):
  - rules: PV-ONT-024 (Liskov violation; the message names the clause), PV-ONT-025 (the witness does not check), PV-ONT-026 (malformed clause, with its file), PV-ONT-027 (`liskov_prose` above `lint-baseline.json`).
  - verdicts: prose gives `Unknown{Prose}` naming the clause; legacy gives `Pass` with `liskov_pairs_checked` 0.
  - declines: NoCheckable, WitnessStale (names `make contracts`), PositiveControlFailed; a malformed Σ exits 3.
  - `pv lint --gate refines` exits 1 with `reject: <messages>` on Fail.
  - **computed, NOT armed.**
- `pv-sat/liskov.rs`: the untrusted reasoner, with its own direction table. It writes `contracts/witness/liskov/<liskov_sha256>.json` only after its `pc_liskov_reasoner` plant (a strengthened PRE-2) is proved by the library checker. The pv-sat self-test now covers 5 controls.
- `readme_gen` prints `**Requires:** N | **Ensures:** N | **Invariants:** N`. The `liskov_prose` baseline is 0, and `check_ont_ratchet.sh` carries it as a foreign key.
- Corpus: `online-softmax-v1` refines `softmax-kernel-v1`, with parsed PRE-1/POST-1 on both and POST-2 on the refinement.
- `contracts/ont-refines-v1.yaml` (kind `pattern`, 5 obligations, FALSIFY-ONT4E-001..006). CI: `ci/explicit-test-commands.d/600-…ont4e-refines-gate.cmd`; `scripts/tree_reader_tests.txt` +3.
- `make contracts` runs `pv lint --gate refines` after pv-sat.

## Probe (§5 ONT-4e), this tree
`pv lint contracts/ --gate refines --format json` gives:
- `verdict Pass`
- `refines_pairs 1`, `liskov_pairs_checked 1`
- `requires_n 2`, `ensures_n 3`, `invariants_n 0`
- `pc_checker fired`, `violations 0`
- `witness.pc_reasoner fired`

`jq -e '.liskov_pairs_checked>0 and .pc_checker=="fired" and .verdict=="Pass"'` returns true.

## Mutations (row's Mutation line), measured 2026-09-24, source restored after each
- **Checker accepts any implication** (the atom-equality test in `check_chain` skipped): `pc_checker_fires_on_the_shipped_fixture` goes RED. 10 lib tests went red in total, all 8 `refines_gate` tests decline PositiveControlFailed, and the pv-sat reasoner test is also red. This was re-measured after the complexity split.
- **Precondition direction swapped** (`Kind::sides` Pre): the fixture tests went RED. That covers `a_strengthened_precondition_is_named` and `a_weakened_precondition_is_liskov` in the lib, plus 2 pv-sat tests.

## dogfood:
- aprender: cmd="pv lint contracts/ --gate refines" pv=c8ca2b4e1 exit=0 verdict="Pass — refines_pairs 1, liskov_pairs_checked 1, pc_checker fired, violations 0"
- rmedia: cmd="pv lint crates/rmedia-core/contracts/ --gate refines" pv=c8ca2b4e1 exit=0 verdict="Pass — refines_pairs 0, liskov_pairs_checked 0 (nothing to check, said so), pc_checker fired" (rmedia @ 1f933c6, read-only)

## Named residuals
- **Gate NOT armed.** `refines` is absent from `armed_gates` (11 armed). Arming it is a separate PR.
- **The corpus had no `refines` pair**, so the probe's `liskov_pairs_checked>0` could not hold. One genuine pair was added: online-softmax is a refinement of softmax.
- **Spec vs corpus:** the RED says `invariants[]` already has the Clause shape. In the corpus it does not: contracts carry legacy prose lists, and 4 carry a mapping. So only invariants that carry `formal_status` count as clauses; a legacy list or mapping is not malformed. `invariants_n` is 0 on this corpus.
- **Atoms are opaque:** a precondition that is textually different but logically weaker is reported as strengthened. That is sound (no false Pass) but incomplete.
- **Zero pairs is `Pass`, not a decline** (rmedia above). The row makes a legacy pair `Pass` with the count on the line; R-2's decline is kept for "no Σ".
- The row's Change names `witness.rs`. The checker lives in `ontology/liskov.rs` beside it, sharing `FIRED` and the witness directory.
- **Pre-existing, fixed here:** the Makefile is `.ONESHELL` with `-o pipefail` and no `-e`, so a bare failing command in the middle of `make contracts` did not stop it. `pv extract --check` printed `reject` and the target still exited 0. That line and the pv-sat/refines line now end `|| exit 1`. With a stale `contracts.nt`, the target exits 2; with a fresh one, it exits 0. Other recipes with the same shape were not audited.
- `docs/roadmaps/roadmap.yaml` was assembled by hand (the aggregator is Python, which is banned in this session). PMAT-4074 was moved after PMAT-4071, where the stack merge had left it out of order, and `check_roadmap_sorted.sh` PASSes. `make roadmap-aggregate-check` was not run.

## Verification (this tree)
| check | result |
|---|---|
| `cargo test -p aprender-contracts --lib` | 1765 passed, 0 failed |
| `--test ont4e_refines_gate` / `ont5_consistency_gate` / `ont6_lint_verdict` / `--bin pv-sat` | 10/10, 5/5, 7/7, 5/5 |
| `cargo clippy -p aprender-contracts -p aprender-contracts-cli --all-targets -D warnings` · `cargo fmt --check` · `cargo deny check advisories` | clean · 0 · ok |
| `pv lint contracts/` | PASS, 11/11 armed |
| `make contracts` on the committed tree | exit 0, tree clean |
| check_ont_ratchet · complexity_ratchet · tree_reader_tests · explicit_test_commands · readme_claims · `pv extract --check` · readme_sync · check_roadmap_sorted | all PASS |
