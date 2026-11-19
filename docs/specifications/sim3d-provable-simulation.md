---
kind: docs
---
# SIM3D: Provable Simulation (phases 0 and 1)

**Status**: PROPOSED. Phases 0 and 1 of epic aprender#7244 (milestone 0.77.0).
**Rows**: aprender#7252 (this document), aprender#7245 (ontology, shapes, contracts).
**Contracts**: `contracts/sim3d-spec-v1.yaml`, `contracts/sim3d-receipt-v1.yaml`,
`contracts/simular/simulation-determinism-v1.yaml`, `contracts/simular/simulation-step-v1.yaml`.
**Out of scope here**: phases 2 to 5 (viewer, scenario library, publishing), and the
`check-symbolic` grammar extension (stretch row aprender#7253).

## 1. The defect

A simulation spec used to be a YAML file that nothing graded. `ieee_strict` is a flag with
no effect, a seed is optional, and the invariants in the simular contracts are prose:
`pv check-symbolic` parses none of them. A spec with no seed, or with no invariant a run
could be held to, passed every gate. A run that nobody can reproduce, and that no monitor
judged, proves nothing.

## 2. Phase 0: the spec is an ontology instance

A simulation is written as a `.sim.yaml`, emitted as canonical JSON (sorted keys, no
whitespace), and graded by pv through the `extract:json` path (the `type: json` extractor,
the same pattern as `contracts/findings-ledger-v1.yaml`). No new reader exists: a closed
SHACL shape is the validator.

Σ (`contracts/ontology.yaml`) gains seven concepts: `Simulation`, `SimEntity`, `Law`,
`Invariant`, `Seed`, `Trajectory`, `Receipt`. The entity concept is `SimEntity` because
`Entity` already names a different thing in Σ. The new contract kind is `simulation`
(`ContractKind::Simulation`); the validator requires a declared shape (SIM-001) and at
least one falsification test (SIM-002), and exempts the kind from the kernel provability
rule, as for the other non-kernel kinds.

### 2.1 What a spec states

| Key | Rule |
|-----|------|
| `schema` | exactly `sim3d-spec/v1` |
| `name` | a slug |
| `time_unit` | exactly `ns`: time is integer nanoseconds, never a float clock |
| `horizon_ns` | a positive integer written without sign or leading zero |
| `seed` | required, exactly one: `value` (integer) and `algorithm` (`pcg64` or `counter-based`) |
| `entities` | at least one: `id`, `kind` (closed set), optional `rate_per_s`, `capacity` |
| `laws` | at least one: `id` (`LAW-nnn`), `statement`, `invariant` (`INV-nnn`) |
| `invariants` | at least one: `id` (`INV-nnn`), `formula`, `unit` (closed set), optional `tolerance` |

The shape is closed at every level: a key outside the table is refused. The worked example
is the existing M/M/1 experiment (seed 42, arrival rate 5.0, service rate 6.0) in
`tests/fixtures/ont/sim3d/mm1-littles-law.sim.yaml`.

### 2.2 Verification line

`pv lint --gate sigma --gate shapes` is green on the valid M/M/1 spec. The planted
falsifiers each go red, naming the path: a missing seed, a spec with no invariants, an
empty spec, an extra key, a unit outside the closed set. The case table is
`crates/aprender-contracts-cli/tests/sim3d_spec.rs`, which grades each fixture under
`tests/fixtures/ont/sim3d/` against the real contract.

### 2.3 What the shape cannot say

1. A law's `invariant` id is well formed, but that an invariant with that id is declared is
   a join, and the SHACL subset has no join. Closing it belongs to the typed-invariants row.
2. A positive-number bound (such as a maximum horizon) needs `minInclusive` and
   `maxInclusive`, which the subset lacks. The horizon is checked by its lexical form only.
3. A spec that conforms is well formed, not correct. Nothing in phase 0 runs a simulation.

## 3. Phase 1: the receipt and the kernel obligations

A run states a receipt: the spec sha256, the kernel git sha, the seed, the step count, the
final hash of the hash-chained trajectory, and one verdict per invariant monitor. A receipt
with no verdict is refused. The kernel that writes it is the phase-1 deliverable; the shape
it must write exists first (`contracts/sim3d-receipt-v1.yaml`), so a receipt can be graded
before the kernel does. The shape checks that the final hash is 64 lowercase hex, never that
it is the right hash: recomputing it from the trajectory is the kernel's and the viewer's
test.

The kernel obligations reuse the simular contracts rather than restating them: step
determinism, time monotonicity and audit retrieval are already stated in
`contracts/simular/simulation-determinism-v1.yaml`.

## 4. Invariant map

Every invariant this specification names maps to a contract and an obligation id in that
contract. The LAB checker `scripts/lab/check_sim3d_spec_contracts.sh` reads this table,
requires the contract to exist, to pass `pv validate`, and to contain the obligation id. A
row naming a contract or an id that does not exist turns it red.

| Invariant | Statement | Contract | Obligation |
|-----------|-----------|----------|------------|
| SIM3D-INV-001 | one step from one state is one result | `contracts/simular/simulation-determinism-v1.yaml` | `SD-INV-001` |
| SIM3D-INV-002 | simulated time strictly increases | `contracts/simular/simulation-determinism-v1.yaml` | `SD-INV-002` |
| SIM3D-INV-003 | a recorded step is retrievable by its id | `contracts/simular/simulation-determinism-v1.yaml` | `SD-INV-003` |
| SIM3D-INV-004 | a spec carries exactly one seed | `contracts/sim3d-spec-v1.yaml` | `SIM-SPEC-INV-001` |
| SIM3D-INV-005 | a spec declares at least one invariant | `contracts/sim3d-spec-v1.yaml` | `SIM-SPEC-INV-002` |
| SIM3D-INV-006 | every invariant states a unit from the closed set | `contracts/sim3d-spec-v1.yaml` | `SIM-SPEC-INV-003` |
| SIM3D-INV-007 | every law names the invariant that monitors it | `contracts/sim3d-spec-v1.yaml` | `SIM-SPEC-INV-004` |
| SIM3D-INV-008 | a receipt carries a 64-hex final trajectory hash | `contracts/sim3d-receipt-v1.yaml` | `SIM-RCPT-INV-001` |
| SIM3D-INV-009 | a receipt carries at least one monitor verdict | `contracts/sim3d-receipt-v1.yaml` | `SIM-RCPT-INV-002` |

## 5. Gate and LAB status

Nothing here is a gate. The shapes are not armed for corpus verdicts (that is a change to the
lint baseline, and a check becomes a gate only after three green nights and one gate retired).
The case-table test and the invariant-map checker are LAB checks: they run at night or on
demand, never inside a pull-request, merge-queue or release job, and a red opens or updates
one ticket with one owner.

## 6. Not done in this change

- The `check-symbolic` grammar extension (aprender#7253, stretch).
- Cross-checking that a law's invariant id is declared in the same spec.
- Positive-number and range bounds, pending SHACL subset support.
- The kernel and its receipts (phase-1 rows aprender#7246 and aprender#7251).
