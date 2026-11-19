# SIM3D: provable simulation, aprender-side spec (phases 0 and 1)

Status: draft. Master ticket #7244, this spec #7252. Phases 2 to 5 (3D rasterizer,
f64 structural kernel, WASM viewer, executed Kani and Lean) are out of scope here.

## Principle

Prove the trajectory, not the pixels. A simulation is one spec file; `pv` gates its
shape; a deterministic kernel runs it with per-step invariant monitors and emits a
hash-chained trajectory plus a receipt.

## Phase 0: ontology and shapes

Phase 0 adds Simulation concepts (spec, scenario, step, receipt) to the `pv`
ontology so a spec file is shape-checked like any other instance. `pv` shape-gates
and never executes. Existing ontology: `contracts/ontology.yaml`.

## Phase 1: deterministic kernel and load-balancer scenario

Seeded `Pcg64` replay, `f64` state, a queueing network (arrivals, one balancer,
N servers) advanced by a fixed timestep. The kernel checks each invariant below
after every step and appends the state hash to the trajectory chain.

## Invariant map

Each row names one invariant, the equation that states it, and the contract file
that carries it. `scripts/check_spec_invariants_pv.sh` reads these rows and runs
`pv validate` on every contract path; a row naming a contract `pv` rejects, or an
equation absent from that file, fails the check.

| Invariant | Equation | Contract |
|---|---|---|
| INV-SIM-1 same state and dt give the same next state | step_determinism | contracts/simular/simulation-determinism-v1.yaml |
| INV-SIM-2 time advances monotonically by timestep | time_advancement | contracts/simular/simulation-determinism-v1.yaml |
| INV-SIM-3 every step is recorded in the audit trail | audit_trail | contracts/simular/simulation-determinism-v1.yaml |
| INV-SIM-4 step count increments by one per step | step_monotonicity | contracts/simular/simulation-step-v1.yaml |
| INV-SIM-5 audit trail has no gaps | audit_completeness | contracts/simular/simulation-step-v1.yaml |

## Honest scope

This proves the model is deterministic, self-consistent and obeys its declared
laws on every run. It does not prove a real load balancer behaves this way.
`pv check-symbolic` parses 0 of 9 invariants in the determinism contract today;
mapping here is by contract ID and equation name, not symbolic proof.

## Checker

`scripts/check_spec_invariants_pv.sh [SPEC]` is a LAB check: it runs nightly, on no
PR, merge-queue or release path. Case table and planted defects:
`scripts/test_check_spec_invariants_pv.sh`.
