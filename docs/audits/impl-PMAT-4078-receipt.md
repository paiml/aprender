---
status: partial
partial_reason: "L3 is DECLARED, NO VERDICT: KANI-ONT-9-1 (witness.rs, #[kani::proof] kani_ont_9_1) was run by `cargo kani` on 2026-09-24 and was cut off at the 2400 s cap in CBMC symex (BTreeSet<String> unwinding), with no VERIFICATION line. See the Kani section. Everything else ONT-9 owes (contract, L2 tests, probe, mutations) is done and green. Flip to complete, and proof.status to proved with l3_kani_proved 1, only in the commit that records a passing `cargo kani --harness kani_ont_9_1` run."
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

## Kani (L3) — RUN, NO VERDICT (cut off)
Run measured 2026-09-24 under the cop ruling (flock + nice -n19 + ionice -c3, target on /mnt/nvme-raid0):
`timeout 2400 cargo kani -p aprender-contracts --lib --harness kani_ont_9_1` (Kani 0.67.0), started 19:40:14Z.
The build finished in 21.66s and `Checking harness ontology::witness::kani_proofs::kani_ont_9_1...` started, then the
2400 s cap killed it at about 20:20Z, still in CBMC symbolic execution. No `VERIFICATION:` line was printed, so there
was neither SUCCESSFUL nor FAILED; this is **not** evidence either way. By the time it was killed the log held ~5000
`Unwinding loop` lines, nearly all in `alloc::collections::btree` over `String` keys and `memcmp`: the harness reaches the checker
through `SmallGraph::clause_set` → `BTreeSet<String>`, and CBMC unrolls the B-tree code for every symbolic clause
combination. The bound was not widened, because 3 ids does not finish in 40 minutes.
The earlier checks still stand: the harness body was stubbed and executed against the real `check` (scratch,
reverted), and the L2 twin runs it on 2048 samples every `cargo test`.

## Checks
`cargo fmt --all -- --check` · `cargo clippy -p aprender-contracts -p aprender-contracts-cli --all-targets -D warnings` ·
`cargo test -p aprender-contracts --lib` (1908 passed) · CLI `ont9_self_contract`, `ont4_relations_gate`,
`ont5_consistency_gate`, `ont6b_kind_default`, `pvl_zero_contracts` · `pv lint contracts` 0 errors, armed meet Pass ·
`pv extract --check` · pv-sat + `--gate ont-consistency` + `--gate refines` · `readme_sync --check` ·
`cargo deny check advisories` · `make contracts`.

## Residuals (named, not hidden)
1. **L3 has no verdict**: the run was cut off at 40 min in symex. The fix is a harness path that does not symex `BTreeSet<String>` (e.g. an index-backed clause set that `check` also accepts); that belongs to the #4078 keep-open. Until the run, proof.status stays `declared`; the CLI test enforces the pair.
2. **The Kani bound is 3 ids** (cores ≤ 3 steps, complete at that bound) and was not widened, because 3 does not finish in 40 min; nothing is claimed for 4+. `ont_planted`
   (2–8 ids) is L2 evidence beyond it, not proof.
3. **Spec-level common mode** (stated by the spec itself): checker, oracle and generators share the §3.5 Horn reading of
   the relations; a mistake in that reading is invisible to every rung.
4. **Deviation from the B.16 example: no `shape:` block.** The pv-contract extractor emits none of
   `ont:censusN / armedGate / witnessSha`, so a closed shape over them would either Fail the armed `shapes` gate on
   every contract or be vacuous. The row's `produces`/probe do not require it. Recorded as `ont-delta: shape`.
5. **INV-1 is baselined, not zero.** The duplicate-stems gate (PV-DUP-001) measures 48 ambiguous stems today, all 48
   baselined (0 unbaselined): a NEW shared stem fails, the 48 do not. The contract binds that behaviour, not a zero count.
