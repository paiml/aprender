---
status: partial
partial_reason: "L3 is DECLARED, NOT EXECUTED: KANI-ONT-9-1 (witness.rs, #[kani::proof] kani_ont_9_1) has never been run by `cargo kani` — the authoring host is not cleared for Kani. Everything else ONT-9 owes (contract, L2 tests, probe, mutations) is done and green. Flip to complete, and proof.status to proved with l3_kani_proved 1, only in the commit that records a passing `cargo kani --harness kani_ont_9_1` run."
ticket: PMAT-4078
row: ONT-9
issue: 4078
base: "fold/b3-contracts-ont-docs @ c3153e552 (PR #4317)"
model: "claude-opus-5-5 (1M context), direct"
---
# impl-PMAT-4078 — ONT-9 · the ontology's contract on itself (`contracts/ont-self-v1.yaml`)

## What lands
- `contracts/ont-self-v1.yaml` — kind `kernel`, `entity {type: pv-contract, ref: contracts/}`, `valid_under {world: committed}`,
  `relations.depends_on` + `metadata.depends_on` = [ont-consistency-v1, ont-relations-v1, ont-verdict-lattice-v1].
  Seven obligations (the spec's v3 RED list): ONTSELF-INV-001 unique ids · 002 `depends_on ∪ supersedes` acyclic ·
  003 kind totality · 004 n_files > 0 · 005 witness sha == census sha · 006 armed_gates monotone · 007 checker
  soundness. Ten falsifiers, each naming a test fn that exists (enforced by `ont9_self_contract`).
  `proof {status: declared, kani {harnesses: [KANI-ONT-9-1]}}`, `verification_summary.l3_kani_proved: 0`.
- `ontology/witness.rs` — `planted::ont_planted` (proptest, 512 seeds; F-12: verdict == construction) and
  `#[cfg(kani)] kani_proofs::kani_ont_9_1` (`KANI-ONT-9-1`, unwind 12).
- `ontology/witness_small.rs` (cfg(test|kani)) — the 3-id universe, the brute-force relation-semantics oracle
  (reads clause flags, never the checker's sets), `CORE_BOUND = 3` (measured: the pv-sat plant core is 3 steps and the
  committed corpus witness is a model), and `kani_ont_9_1_twin` (proptest, 2048 cases, the harness body).
- `ontology/witness_planted.rs` (cfg(test)) — consistent-by-construction and planted-contradiction generators,
  a 2ⁿ oracle cross-checking each construction, adversarial cores (valid derivations closed on a non-conflict pair,
  on a conflict with an underived side, random step sequences) and adversarial models (all 2ⁿ for n ≤ 6).
- **A real gap, closed:** `lint/relations_gate.rs` `cycle_sweep` checked each acyclic role alone, so a cycle
  alternating `depends_on` and `supersedes` passed. It now also sweeps the union (PV-ONT-009, message names
  `depends_on ∪ supersedes` and the path), reported only when neither role closed a cycle itself.
  Fixture `tests/fixtures/ont/relations-mixed-cycle/`.
- `crates/aprender-contracts-cli/tests/ont9_self_contract.rs` (5 tests) + `ci/explicit-test-commands.d/575-…cmd`.
- Regenerated: `contracts/census.json` (1836), `contracts/contracts.nt`, the consistency witness
  (`contracts/witness/66d5fefd….json` replaces `6e578423….json` — new relations change relations_sha256), README
  count (readme_sync --write).

## RED first (pre-fix, measured 2026-09-24)
- `lint::relations_gate::tests::a_cycle_through_depends_on_and_supersedes_together_is_rejected` — FAILED against the
  per-role-only sweep (findings `[]`); green after the union sweep.
- `ont9_self_contract` without the contract — 4 of 5 FAILED (probe, anchoring, falsifier binding, Kani rung).

## Mutations (each applied, run, reverted; all RED)
| id | mutation | RED |
|---|---|---|
| M1 | checker accepts any two ids (NoSuchConflict check off) | `ont_planted`, `kani_ont_9_1_twin` |
| M2 | ConflictSideNotDerived check off | `ont_planted`, `kani_ont_9_1_twin` |
| M3 | PremiseNotDerived check off | `ont_planted`, `kani_ont_9_1_twin` |
| M4 | model check skips conflict clauses | `ont_planted`, `kani_ont_9_1_twin` |
| M5 | union sweep removed | lib mixed-cycle test; CLI `a_cycle_in_the_corpus_turns_the_relations_gate_red` |
| M6 | `l3_kani_proved: 1` while declared / `status: proved` with 0 | `the_kani_rung_is_declared_not_claimed` |
| M7 | a falsifier names a ghost fn | `every_falsifier_names_a_test_function_that_exists` |
| M8 | introduce a cycle in the corpus (ont-consistency-v1 supersedes ont-self-v1) | `a_cycle_in_the_corpus_turns_the_relations_gate_red` |

## Kani (L3) — pending
Not run (`cargo kani` is not cleared on this host). What was checked instead: the harness body, with the
`#[kani::…]` attributes stripped and `kani::any`/`assume` stubbed, was compiled and executed as a unit test against
the real `witness_small`/`check` (a scratch append, reverted), so the harness type-checks against the code it names.
The L2 twin runs the identical body on 2048 samples every `cargo test`.

## Checks
`cargo fmt --all -- --check` · `cargo clippy -p aprender-contracts -p aprender-contracts-cli --all-targets -D warnings` ·
`cargo test -p aprender-contracts --lib` (1908 passed) · CLI `ont9_self_contract`, `ont4_relations_gate`,
`ont5_consistency_gate`, `ont6b_kind_default`, `pvl_zero_contracts` · `pv lint contracts` 0 errors, armed meet Pass ·
`pv extract --check` · pv-sat + `--gate ont-consistency` + `--gate refines` · `readme_sync --check` ·
`cargo deny check advisories` · `make contracts`.

## Residuals (named, not hidden)
1. **L3 not executed** — see frontmatter. Until the run, proof.status stays `declared`; the CLI test enforces the pair.
2. **The Kani bound is 3 ids** (cores ≤ 3 steps, complete at that bound); nothing is claimed for 4+. `ont_planted`
   (2–8 ids) is L2 evidence beyond it, not proof.
3. **Spec-level common mode** (stated by the spec itself): checker, oracle and generators share the §3.5 Horn reading of
   the relations; a mistake in that reading is invisible to every rung.
4. **Deviation from the B.16 example: no `shape:` block.** The pv-contract extractor emits none of
   `ont:censusN / armedGate / witnessSha`, so a closed shape over them would either Fail the armed `shapes` gate on
   every contract or be vacuous. The row's `produces`/probe do not require it. Recorded as `ont-delta: shape`.
5. **INV-1 is baselined, not zero.** The duplicate-stems gate (PV-DUP-001) measures 48 ambiguous stems today, all 48
   baselined (0 unbaselined): a NEW shared stem fails, the 48 do not. The contract binds that behaviour, not a zero count.
