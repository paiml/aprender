# codegen-dispatch-v1

**Version:** 1.0.0

Codegen dispatch completeness — every Phase 1 resource type handled

## References

- Forjar spec §6.3 Shell Generation Pipeline

## Equations

### apply_script

$$
apply_script(r) = dispatch(r.type) where dispatch covers {Package, File, Service, Mount}
$$

**Domain:** $r \in Resource where r.type \in {Package, File, Service, Mount}$

**Codomain:** $Result<String, String>$

**Invariants:**

- $Returns Ok for all Phase 1 types$
- $Returns Err for non-Phase-1 types$
- $Output is non-empty shell script$

### check_script

$$
check_script(r) = dispatch(r.type) where dispatch covers {Package, File, Service, Mount}
$$

**Domain:** $r \in Resource where r.type \in {Package, File, Service, Mount}$

**Codomain:** $Result<String, String>$

**Invariants:**

- $Returns Ok for all Phase 1 types$
- $Returns Err for non-Phase-1 types$
- $Output is non-empty shell script$

### state_query_script

$$
state_query_script(r) = dispatch(r.type) where dispatch covers {Package, File, Service, Mount}
$$

**Domain:** $r \in Resource where r.type \in {Package, File, Service, Mount}$

**Codomain:** $Result<String, String>$

**Invariants:**

- $Returns Ok for all Phase 1 types$
- $Returns Err for non-Phase-1 types$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | completeness | All Phase 1 types dispatched | $\forall t \in {Package, File, Service, Mount}: check_script(r{type=t}) = Ok(_)$ |
| 2 | symmetry | Dispatch is symmetric across three functions | $\forall t: check_script(r{type=t}).is_ok() ⟺ apply_script(r{type=t}).is_ok() ⟺ state_query_script(r{type=t}).is_ok()$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-CD-001 | Completeness | check_script, apply_script, state_query_script return Ok for Package, File, Service, Mount | Missing match arm in dispatch |
| FALSIFY-CD-002 | Symmetry | If check_script(r).is_ok() then apply_script(r).is_ok() and state_query_script(r).is_ok() | Asymmetric dispatch — one function handles type that others don't |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-CD-001 | All Phase 1 types dispatched | 4 | exhaustive |
| KANI-CD-002 | Dispatch is symmetric across three functions | 4 | exhaustive |

## QA Gate

**Codegen Dispatch Contract** (F-CD-001)

Script generation dispatch completeness quality gate

**Checks:** completeness, symmetry

**Pass criteria:** All 2 falsification tests pass

