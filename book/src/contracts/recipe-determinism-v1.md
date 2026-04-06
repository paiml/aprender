# recipe-determinism-v1

**Version:** 1.0.0

Recipe determinism — deterministic expansion and input validation

## References

- Dolstra (2006) The Purely Functional Software Deployment Model (Nix thesis)

## Equations

### expand_recipe

$$
expand(id, recipe, machine, inputs, ext_deps) = namespaced resources with resolved templates
$$

**Domain:** $recipe_id \in String, recipe_file \in RecipeFile, machine \in MachineTarget, inputs \in HashMap, ext_deps \in [String]$

**Codomain:** $Result<IndexMap<String, Resource>, String>$

**Invariants:**

- $Deterministic: same inputs \to same expanded resources$
- $All resource IDs are namespaced as '{recipe_id}/{resource_name}'$
- $External deps only injected into first resource$
- $Machine target propagated to all inner resources$

### validate_input_type

$$
validate_type(name, type, value, decl) = Ok(string_val) | Err(msg)
$$

**Domain:** $name \in String, type \in {string, int, bool, path, enum}, value \in Value, decl \in RecipeInput$

**Codomain:** $Result<String, String>$

**Invariants:**

- $int with value < min \to Err$
- $int with value > max \to Err$
- $path not starting with / \to Err$
- $enum value not in non-empty choices \to Err$

### validate_inputs

$$
validate(recipe, provided) = type-checked resolved inputs or Err
$$

**Domain:** $recipe \in RecipeMetadata, provided \in HashMap<String, Value>$

**Codomain:** $Result<HashMap<String, String>, String>$

**Invariants:**

- $Missing required input \to Err$
- $Default values used when input not provided$
- $Type validation: int respects min/max, path starts with /, enum in choices$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Expansion determinism | $\forall inputs: expand(id, r, m, inputs, deps) = expand(id, r, m, inputs, deps)$ |
| 2 | bound | Integer bounds enforced | $\forall n, decl: decl.min \leq n \leq decl.max ⟹ Ok(_); n < decl.min ∨ n > decl.max ⟹ Err(_)$ |
| 3 | invariant | Path validation | $\forall s: validate_input_type('path', s) = Ok(_) ⟹ s.starts_with('/')$ |
| 4 | invariant | External deps placement | $Only first resource in expansion receives external_depends_on$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-RD-001 | Determinism | expand_recipe(same args) = expand_recipe(same args) | Non-deterministic iteration in IndexMap or HashMap |
| FALSIFY-RD-002 | Integer bounds | validate_int rejects values outside [min, max] | Bound comparison uses wrong operator |
| FALSIFY-RD-003 | Path validation | validate_input_type('path', s) rejects non-absolute paths | Missing starts_with('/') check |
| FALSIFY-RD-004 | External deps placement | Only first expanded resource has external deps | External deps injected into wrong resource |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-RD-001 | Expansion determinism | 8 | bounded_int |
| KANI-RD-002 | Integer bounds enforced | 16 | bounded_int |
| KANI-RD-003 | Path validation | 8 | bounded_int |
| KANI-RD-004 | External deps placement | 8 | bounded_int |

## QA Gate

**Recipe Determinism Contract** (F-RD-001)

Deterministic recipe expansion quality gate

**Checks:** determinism, int_bounds, path_validation, external_deps

**Pass criteria:** All 4 falsification tests pass

