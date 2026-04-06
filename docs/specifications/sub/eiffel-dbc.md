# Sub-spec: Eiffel Design by Contract Extensions

**Parent:** [pv-spec.md](../pv-spec.md) Section 3

---

## 1. Motivation

Bertrand Meyer's Design by Contract (DbC) framework, introduced in
*Object-Oriented Software Construction* (1988, 2nd ed. 1997) and
implemented in the Eiffel language, defines a richer contract vocabulary
than our current proof obligation types capture. Our 19 obligation types
express *properties of functions* (invariant, monotonicity, bound, etc.)
but do not model the **caller/callee contract pair** that is central to
Meyer's framework.

This sub-spec defines seven new obligation types derived from the Eiffel
DbC tradition. These extend the existing type system without replacing
it — every current obligation type remains valid.

### What Already Exists

The codebase has already adopted Eiffel DbC concepts at two levels:

1. **Equation-level pre/postconditions.** The `Equation` struct has
   `preconditions: Vec<String>` and `postconditions: Vec<String>` fields
   containing Rust expressions that compile to `debug_assert!()` via
   `build.rs` codegen (see [escape-proof-enforcement.md](escape-proof-enforcement.md)).

2. **Proc macro enforcement.** Three attribute macros implement Meyer's
   core keywords directly:
   - `#[requires(pred)]` → Eiffel `require` → `debug_assert!` at entry
   - `#[ensures(pred)]` → Eiffel `ensure` → `debug_assert!` before return
   - `#[invariant(pred)]` → Eiffel `invariant` → assert pre + post

3. **`codegen.rs` module.** Generates `contract_pre_*!()` and
   `contract_post_*!()` assertion macros from YAML equations.

4. **Gate 5 (Enforce).** `pv lint` checks that equations have
   `preconditions`, `postconditions`, and `lean_theorem` fields.

### The Remaining Gap

The existing implementation puts pre/postconditions on **equations**
(implementation-facing, Rust expressions, `debug_assert!` enforcement).
This sub-spec proposes them as **proof obligation types** (specification-
facing, formal predicates, Kani/Lean verifiable). These are
complementary layers:

| Layer | Where | What | Verified by |
|---|---|---|---|
| Equation pre/post | `equations.<name>.preconditions` | Rust expressions | `debug_assert!` (runtime, debug) |
| Obligation pre/post | `proof_obligations[].type: precondition` | Formal predicates | Kani (L4), Lean (L5), probar (L3) |

The equation-level pre/postconditions are *enforcement*. The obligation-
level pre/postconditions are *specification*. Meyer's framework requires
both: the formal spec (what we prove) and the runtime check (what we
enforce). The five remaining types (`frame`, `loop_invariant`,
`loop_variant`, `old_state`, `subcontract`) have no equivalent at
either level today.

---

## 2. New Obligation Types

### 2.1. `precondition`

**Eiffel keyword:** `require`

**Pattern:** `P(input) — conditions that must hold before the kernel executes`

A precondition defines the caller's responsibility. If the caller
violates the precondition, the kernel's behavior is undefined. This is
distinct from a `bound` (which constrains the *output*) and from an
`invariant` (which asserts a property of the *function itself*).

```yaml
proof_obligations:
  - type: precondition
    property: "Input vector is finite and non-empty"
    formal: "∀i: ¬isNaN(x_i) ∧ ¬isInf(x_i) ∧ len(x) > 0"
    if_fails: "Kernel may produce NaN or panic on empty slice"
```

**Codegen implications:**
- probar: generate `proptest` that asserts the kernel *does* panic or
  return `Err` when the precondition is violated (negative testing)
- Kani: assume the precondition (`kani::assume`) and verify the
  postcondition
- Lean: precondition becomes a hypothesis in the theorem statement

**Relationship to `equations.<name>.preconditions`:** The equation-
level field contains *Rust expressions* for runtime `debug_assert!`.
The obligation-level `precondition` type contains a *formal predicate*
for static verification. They encode the same intent at different
abstraction levels. `pv generate` should emit both: the formal
predicate for Kani/Lean, and a Rust translation for `debug_assert!`.

**Relationship to existing types:** A `bound` obligation like
`0 <= f(x)_i <= 1` is implicitly a postcondition. Adding explicit
`precondition` makes the conditional structure visible:
`require P(x) ensure 0 <= f(x)_i <= 1`.

### 2.2. `postcondition`

**Eiffel keyword:** `ensure`

**Pattern:** `P(input) → Q(output) — what the kernel guarantees if preconditions hold`

A postcondition defines the kernel's responsibility. Unlike our current
`invariant` type (which asserts `∀x: P(f(x))`), a postcondition is
explicitly conditional on the precondition holding.

```yaml
proof_obligations:
  - type: postcondition
    property: "Output sums to 1.0 within tolerance"
    formal: "|Σ softmax(x)_i - 1.0| < ε"
    tolerance: 1.0e-6
    requires: "SM-PRE-001"   # Links to a precondition obligation
    if_fails: "Normalization numerically unstable for this input"
```

**The `requires` field:** A postcondition MAY reference one or more
precondition obligation IDs. This makes the `require → ensure` pair
explicit and machine-traceable.

**Relationship to `equations.<name>.postconditions`:** Same layering
as preconditions. The equation-level field holds Rust expressions for
`debug_assert!` (enforced by `#[ensures]` / `#[contract]` macros). The
obligation-level `postcondition` type holds the formal predicate for
Kani/Lean verification. Both are generated from the same contract.

**Codegen implications:**
- probar: generate test that first asserts precondition, then checks
  postcondition
- Kani: `kani::assume(precondition); kani::assert(postcondition);`
- Lean: `theorem softmax_sums_to_one (h : precondition x) : postcondition (softmax x)`

### 2.3. `frame`

**Eiffel keyword:** `only` (Eiffel 2005+ frame specification)

**Pattern:** `modifies(S) ∧ preserves(T \ S) — what state the kernel may change`

A frame condition specifies what the kernel is *allowed* to modify.
Everything not listed is implicitly preserved. This is critical for
kernels that operate on shared buffers (KV cache, weight tensors,
activation buffers).

```yaml
proof_obligations:
  - type: frame
    property: "Only output buffer is modified; input and weights unchanged"
    formal: "modifies(output) ∧ preserves(input, weights)"
    if_fails: "Kernel corrupts input buffer or weight tensor"
```

**Why this matters for ML kernels:**
- In-place operations (LayerNorm, RMSNorm) must not corrupt input
- KV cache append must not modify existing entries
- Quantization must not modify the original weights
- SIMD kernels with overlapping loads must not write beyond bounds

**Codegen implications:**
- probar: snapshot input/weights before call, assert equality after
- Kani: clone inputs, run kernel, assert originals unchanged
- Lean: frame condition as a conjunction in the postcondition

### 2.4. `loop_invariant`

**Eiffel keyword:** `invariant` (within `from...until...loop...end`)

**Pattern:** `∀ iteration i: P(state_i) — property maintained across iterations`

Distinct from our top-level `invariant` (a property of the function),
a loop invariant is a property of the *iterative state* at each step.
Relevant for iterative kernels: AdamW, LBFGS, CMA-ES, PageRank,
online softmax.

```yaml
proof_obligations:
  - type: loop_invariant
    property: "Running max is true max of elements seen so far"
    formal: "∀k ≤ i: running_max ≥ x_k"
    applies_to: "online_softmax.find_max"
    if_fails: "Online softmax numerically unstable — max tracking diverges"
```

**The `applies_to` field:** References a specific phase from the
contract's `kernel_structure.phases[]`, anchoring the invariant to a
concrete loop in the implementation.

**Codegen implications:**
- probar: instrument the loop body, assert invariant after each iteration
  for a bounded number of steps
- Kani: unroll loop to `bound`, assert invariant at each unrolling step
- Lean: induction proof over loop iterations

### 2.5. `loop_variant`

**Eiffel keyword:** `variant`

**Pattern:** `V(state_i) ∈ ℕ ∧ V(state_{i+1}) < V(state_i) — strictly decreasing natural number expression`

A loop variant is a witness function that proves termination. It must
be a non-negative integer expression that strictly decreases with each
iteration. This is more precise than our existing `termination` type,
which asserts termination without providing a witness.

```yaml
proof_obligations:
  - type: loop_variant
    property: "Remaining elements decreases each iteration"
    formal: "V(state) = n - i, V ≥ 0, V decreases"
    applies_to: "online_softmax.accumulate"
    if_fails: "Infinite loop — iteration counter not advancing"
```

**Relationship to `termination`:** A `loop_variant` *implies*
`termination` but is strictly stronger — it provides the proof witness.
Contracts MAY use `termination` when no clean variant expression exists
(e.g., convergence-based loops where termination depends on input
properties), but SHOULD prefer `loop_variant` when one exists.

**Codegen implications:**
- probar: compute variant before and after each iteration, assert
  strictly decreasing and non-negative
- Kani: assert `variant_after < variant_before` and `variant_after >= 0`
  at each unrolling step
- Lean: well-founded recursion or `Nat.lt_wfRel`

### 2.6. `old_state`

**Eiffel keyword:** `old` (expression)

**Pattern:** `Q(old(state), new(state)) — postcondition referencing pre-call values`

An old-state obligation relates the output to the input's *original*
value. This is more expressive than `conservation` (which asserts
`Q(before) = Q(after)` for a single quantity). Old-state obligations
can express arbitrary relationships between pre-state and post-state.

```yaml
proof_obligations:
  - type: old_state
    property: "KV cache length increases by exactly seq_len"
    formal: "new(cache.len) = old(cache.len) + seq_len"
    if_fails: "Cache append wrote wrong number of entries"
```

**Relationship to `conservation`:** Conservation is a special case of
old-state where the relationship is equality of a derived quantity.
Old-state is more general:

| Obligation | Expressible? |
|---|---|
| `conservation`: `sum(before) = sum(after)` | `old_state` can express this |
| `old_state`: `len(after) = len(before) + n` | `conservation` cannot |
| `old_state`: `∀i < old(len): after[i] = before[i]` | `conservation` cannot |

**Codegen implications:**
- probar: snapshot state before call, compare with state after
- Kani: clone state, run kernel, assert relationship between clone and result
- Lean: `old` values become universally quantified variables in theorem

### 2.7. `subcontract`

**Eiffel keyword:** Inheritance-based subcontracting rules

**Pattern:** `weaken(pre) ∧ strengthen(post) — contract refinement under substitution`

A subcontract obligation asserts that a derived contract is a valid
behavioral subtype of its parent. This enforces Meyer's subcontracting
rules, which align with the Liskov Substitution Principle:

1. Preconditions may only be **weakened** (OR'd with parent)
2. Postconditions may only be **strengthened** (AND'd with parent)
3. Invariants are **accumulated** (AND'd with parent)

```yaml
proof_obligations:
  - type: subcontract
    property: "GQA attention is a valid refinement of MHA attention"
    formal: "pre(MHA) → pre(GQA) ∧ post(GQA) → post(MHA)"
    parent_contract: "attention-kernel-v1"
    if_fails: "GQA cannot be substituted for MHA — contract violation"
```

**The `parent_contract` field:** References the contract stem that this
contract refines. Must be present in `metadata.depends_on`.

**Relationship to `depends_on`:** The existing `depends_on` field
declares a dependency. A `subcontract` obligation goes further: it
asserts that the dependency is a *behavioral subtyping* relationship,
not just a compositional one.

**Codegen implications:**
- probar: load both contracts, generate test that any input satisfying
  the parent's precondition also satisfies the child's, and the child's
  postcondition implies the parent's
- Kani: verify the weakening/strengthening relationship symbolically
- Lean: subtyping proof via implication chains

---

## 3. Schema Changes

### 3.1. Existing Pre/Post Infrastructure

The `Equation` struct already has pre/postcondition fields (added in
the escape-proof enforcement work):

```rust
pub struct Equation {
    pub formula: String,
    pub preconditions: Vec<String>,    // Rust exprs → debug_assert!
    pub postconditions: Vec<String>,   // Rust exprs → debug_assert!
    pub lean_theorem: Option<String>,  // Lean 4 proof reference
    // ...
}
```

These are *implementation-level* assertions (Rust code, runtime checks).
The new `ObligationType` variants add *specification-level* predicates
(formal logic, static verification via Kani/Lean).

### 3.2. New `ObligationType` Variants

Add to the `ObligationType` enum in `crates/provable-contracts/src/schema/types.rs`:

```rust
pub enum ObligationType {
    // ... existing 19 variants ...
    Precondition,
    Postcondition,
    Frame,
    LoopInvariant,
    LoopVariant,
    OldState,
    Subcontract,
}
```

### 3.3. New Optional Fields on `ProofObligation`

```yaml
proof_obligations:
  - type: precondition | postcondition | frame | ...
    property: "..."           # REQUIRED (existing)
    formal: "..."             # Formal predicate (existing)
    tolerance: 1.0e-6         # Numerical tolerance (existing)
    # --- new fields ---
    requires: "OB-ID"         # Postcondition: links to precondition
    applies_to: "phase_name"  # Loop invariant/variant: kernel_structure phase
    parent_contract: "stem"   # Subcontract: contract being refined
```

All new fields are optional and only meaningful for their respective
obligation types. Validation (`pv validate`) will reject:
- `requires` on any type other than `postcondition`
- `applies_to` on any type other than `loop_invariant` or `loop_variant`
- `parent_contract` on any type other than `subcontract`
- `parent_contract` values not present in `metadata.depends_on`

### 3.4. Backward Compatibility

All new types and fields are additive. Existing contracts are
unaffected. The `ObligationType` enum already uses `#[serde(rename)]`
for `state_machine`; new multi-word types follow the same pattern:

```rust
#[serde(rename = "loop_invariant")]
LoopInvariant,
#[serde(rename = "loop_variant")]
LoopVariant,
#[serde(rename = "old_state")]
OldState,
```

---

## 4. Proof Obligation Type Reference (Updated)

Extends [schema.md Section 2](schema.md#2-proof-obligation-type-reference)
with the Eiffel DbC types:

| Type | Pattern | Example |
|---|---|---|
| `precondition` | P(input) must hold before call | input is finite, non-empty |
| `postcondition` | P(in) → Q(out) conditional guarantee | given finite input, output sums to 1 |
| `frame` | modifies(S), preserves(T\S) | only output buffer written |
| `loop_invariant` | ∀ iter i: P(state_i) | running max tracks true max |
| `loop_variant` | V(state) ∈ N, strictly decreasing | remaining elements = n - i |
| `old_state` | Q(old(state), new(state)) | cache.len grows by seq_len |
| `subcontract` | weaken(pre) ∧ strengthen(post) | GQA refines MHA |

---

## 5. Codegen Strategy

**Integration with existing pipeline.** The codegen for new obligation
types plugs into the existing six-stage escape-proof enforcement
pipeline (see [escape-proof-enforcement.md](escape-proof-enforcement.md)):
- Stage D (`build.rs` codegen) already generates `debug_assert!` from
  equation-level `preconditions`/`postconditions`
- Stage E (`#[contract]` macro) already reads `CONTRACT_*_PRE_*` and
  `CONTRACT_*_POST_*` env vars and injects assertions
- The new obligation types extend **Stages C (validation), D (codegen),
  and F (test execution)** with generators for the five types not yet
  covered: `frame`, `loop_invariant`, `loop_variant`, `old_state`,
  `subcontract`

**Integration with Lean-Kani composition.** The `stub_float` bridge
(see [lean-kani-composition.md](lean-kani-composition.md)) applies to
the new types: Lean proves the obligation over ℝ, Kani verifies the
Rust code over f32, and `#[kani::stub_verified]` connects them. For
example, a `loop_invariant` can be proved by induction in Lean and
verified for bounded iterations in Kani.

### 5.1. Probar (Property Tests)

| Type | Test Strategy |
|---|---|
| `precondition` | Negative test: violate precondition, assert panic/Err |
| `postcondition` | Conditional test: assume pre, assert post |
| `frame` | Snapshot-compare: clone inputs, run kernel, assert inputs unchanged |
| `loop_invariant` | Instrumented loop: assert property at each iteration |
| `loop_variant` | Instrumented loop: assert decreasing and non-negative |
| `old_state` | Snapshot-compare: clone state, run, assert relationship |
| `subcontract` | Cross-contract: load parent, verify pre weakening + post strengthening |

### 5.2. Kani (Bounded Model Checking)

| Type | Kani Pattern |
|---|---|
| `precondition` | `kani::assume(pre); /* verify postcondition */` |
| `postcondition` | `kani::assume(pre); kani::assert(post);` |
| `frame` | Clone inputs; run kernel; `kani::assert(inputs_unchanged)` |
| `loop_invariant` | Unroll to bound; assert invariant at each step |
| `loop_variant` | Unroll to bound; `kani::assert(v_after < v_before && v_after >= 0)` |
| `old_state` | Clone state; run kernel; assert pre/post relationship |
| `subcontract` | `kani::assume(parent_pre); kani::assert(child_pre);` + reverse for post |

### 5.3. Lean 4 (Theorem Proving)

| Type | Lean Pattern |
|---|---|
| `precondition` | Hypothesis `h : P x` in theorem statement |
| `postcondition` | `theorem name (h : pre x) : post (f x)` |
| `frame` | Conjunction: `∧ ∀ i, old_input[i] = new_input[i]` |
| `loop_invariant` | Induction: `∀ n, inv (iter^n state)` |
| `loop_variant` | Well-founded recursion via `Nat.lt_wfRel` |
| `old_state` | Universally quantified pre-state variable |
| `subcontract` | Implication chain: `parent_pre → child_pre` |

---


---

## Continuation

The remaining sections are in separate files:

- [eiffel-dbc-type-invariants.md](eiffel-dbc-type-invariants.md) — §6 Type Invariants, §7 Coq Integration
- [eiffel-dbc-domains.md](eiffel-dbc-domains.md) — §8 Domain Applicability (Part 1)
- [eiffel-dbc-domains-2.md](eiffel-dbc-domains-2.md) — §8 Domain Applicability (Part 2)
- [eiffel-dbc-explain.md](eiffel-dbc-explain.md) — §9 Migration Path, §10 pv explain
- [eiffel-dbc-enforcement.md](eiffel-dbc-enforcement.md) — §11 Escape-Proof Enforcement (Part 1)
- [eiffel-dbc-enforcement-2.md](eiffel-dbc-enforcement-2.md) — §11 Escape-Proof Enforcement (Part 2)
- [eiffel-dbc-references.md](eiffel-dbc-references.md) — §12 Falsification, §13 References
