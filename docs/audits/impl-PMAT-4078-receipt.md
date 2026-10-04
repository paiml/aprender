---
status: complete
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
  `proof {status: proved, kani {harnesses: [KANI-ONT-9-1, -LEN1, -LEN2, -MODEL]}}`, `verification_summary.l3_kani_proved: 1`.
- `ontology/witness.rs` — `planted::ont_planted` (proptest, 512 seeds; F-12: verdict == construction) and
  `#[cfg(kani)] kani_proofs::{kani_ont_9_1, _len1, _len2, _model}` (`KANI-ONT-9-1`, unwind 9, cadical). `check` is now
  `check_with` over `ClauseView` + `IdSet`, generic in the id type (String in production, u8 under Kani).
- `ontology/witness_small.rs` (cfg(test|kani)) — the 3-id universe, the brute-force relation-semantics oracle
  (reads clause flags, never the checker's sets), `CORE_BOUND = 3` (measured: the pv-sat plant core is 3 steps and the
  committed corpus witness is a model), and `kani_ont_9_1_twin` (proptest, 2048 cases, the harness body over BOTH instantiations, verdicts asserted identical); the
  heap-free `u8` view `ArrayClauses` / `Bounded<N>` / `Pairs<N>` Kani runs.
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

## Kani (L3) — PROVED at 3 ids
Kani 0.67.0 under the cop ruling (flock + nice -n19 + ionice -c3, target on /mnt/nvme-raid0, 30-min cap per harness),
`#[kani::unwind(9)]`, `#[kani::solver(cadical)]`, 2026-09-24:

| harness | covers | symex | VCCs (after simpl.) | result | time |
|---|---|---|---|---|---|
| `kani_ont_9_1_len1` | every graph × every 1-step core × conflict pair | 9 s | 1 357 | VERIFICATION:- SUCCESSFUL | 20 s |
| `kani_ont_9_1_len2` | … 2-step cores | 255 s | 1 720 | VERIFICATION:- SUCCESSFUL | 278 s |
| `kani_ont_9_1` | … 3-step cores (CORE_BOUND) | 22 s | 2 087 | VERIFICATION:- SUCCESSFUL | 52 s |
| `kani_ont_9_1_model` | every graph × every model | 284 s | 15 596 | VERIFICATION:- SUCCESSFUL | 406 s |

**How it became tractable (measured, in order):**
1. `String` ids through `BTreeSet<String>` were cut off at 2400 s in symbolic execution, with no verdict.
2. `u8` ids over `Vec`, one harness: stuck in `memchr`, because `<[u8]>::contains` specializes to it.
3. After switching to `iter().any`, it was still cut off at 1800 s.
4. Split into models and a core with symbolic length: the model harness had 18 046 checks after simplification,
   mostly `Vec` pointer checks, and was cut off at 1800 s in the solver. The core harness had 25 298 checks and was
   also cut off at 1800 s.
5. Fixed-capacity arrays (`Bounded<N>` and `Pairs<N>`, where overflow panics and is therefore also proved absent),
   with one harness per concrete core length: all four verified.

**One checker, two instantiations.** `check` is now `check_with::<String, BTreeSet<String>, ClauseSet>`. It is the
same function Kani runs as `check_with::<u8, Bounded<3>, ArrayClauses>`, behind `ClauseView` (clause lookups) and
`IdSet` (the derived and false sets). Production behaviour is unchanged: lib tests 1909 ok, and the old tests run
unmodified except for one `Model::<String>` type annotation. `kani_ont_9_1_twin` runs both instantiations on 2048
samples and asserts the relabelled verdicts, error included, are identical.
`both_instantiations_agree_on_the_plant` pins an accept and two different refusals.

**Kani discriminates** (each mutant applied, run and reverted; FAILED on the property assertion):

| id | mutation in `check_with` | harness | result |
|---|---|---|---|
| K1 | PremiseNotDerived check off | `kani_ont_9_1` | VERIFICATION:- FAILED (`assertion failed: accepted_verdict_is_semantic_u8`) |
| K2 | NoSuchConflict check off (any pair accepted) | `kani_ont_9_1` | VERIFICATION:- FAILED (same assertion) |
| K3 | model check skips conflict clauses | `kani_ont_9_1_model` | VERIFICATION:- FAILED (same assertion) |

## Checks
`cargo fmt --all -- --check` · `cargo clippy -p aprender-contracts -p aprender-contracts-cli --all-targets -D warnings` ·
`cargo test -p aprender-contracts --lib` (1909 passed) · CLI `ont9_self_contract`, `ont4_relations_gate`,
`ont5_consistency_gate`, `ont6b_kind_default`, `pvl_zero_contracts` · `pv lint contracts` 0 errors, armed meet Pass ·
`pv extract --check` · pv-sat + `--gate ont-consistency` + `--gate refines` · `readme_sync --check` ·
`cargo deny check advisories` · `make contracts`.

## Residuals (named, not hidden)
1. **L3 is proved for the `u8` instantiation**. The `String` instantiation that production runs is tied to it by
   sharing `check_with` and by the L2 twin's identical-verdict assertion, not by Kani. The `ClauseSet` impl of
   `ClauseView` is covered by L2 only.
2. **The Kani bound is 3 ids** (cores ≤ 3 steps, complete at that bound) and was not widened, since the four
   harnesses already take ~12.6 min and 4 ids doubles the clause flags and assignments; nothing is claimed for 4+. `ont_planted`
   (2–8 ids) is L2 evidence beyond it, not proof.
3. **Spec-level common mode** (stated by the spec itself): checker, oracle and generators share the §3.5 Horn reading of
   the relations; a mistake in that reading is invisible to every rung.
4. **Deviation from the B.16 example: no `shape:` block.** The pv-contract extractor emits none of
   `ont:censusN / armedGate / witnessSha`, so a closed shape over them would either Fail the armed `shapes` gate on
   every contract or be vacuous. The row's `produces`/probe do not require it. Recorded as `ont-delta: shape`.
5. **INV-1 is baselined, not zero.** The duplicate-stems gate (PV-DUP-001) measures 48 ambiguous stems today, all 48
   baselined (0 unbaselined): a NEW shared stem fails, the 48 do not. The contract binds that behaviour, not a zero count.
