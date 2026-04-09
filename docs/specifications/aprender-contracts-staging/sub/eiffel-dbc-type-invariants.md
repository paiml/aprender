# Sub-spec: Eiffel DbC — Type Invariants and Coq Integration

Version: 1.0
Status: proposed
Date: 2026-04-09

**Parent:** [eiffel-dbc.md](eiffel-dbc.md) Sections 6-7

---

## 6. Type Invariants

Meyer's Design by Contract places *class invariants* as the third pillar
alongside preconditions and postconditions. A class invariant is a
predicate that must hold for every instance of a type at every stable
state — after construction and after every public method returns.

Our current `invariant` obligation type asserts properties of
*functions*. Type invariants assert properties of *data structures*.

### 6.1. YAML Schema Extension

```yaml
type_invariants:
  - name: tensor_validity
    type: "ValidatedTensor"
    predicate: "self.dims.iter().product::<usize>() == self.data.len()"
    description: "Data length equals product of dimensions"
  - name: tensor_non_empty
    type: "ValidatedTensor"
    predicate: "!self.dims.is_empty()"
    description: "At least one dimension"
```

Add `type_invariants: Vec<TypeInvariant>` to the `Contract` struct:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInvariant {
    pub name: String,
    #[serde(rename = "type")]
    pub type_name: String,
    pub predicate: String,
    #[serde(default)]
    pub description: Option<String>,
}
```

### 6.2. Rust Implementation Paths

**Path A: Stable Rust — `Invariant` trait pattern.**

This is the seL4 Rust verification pattern. Works today with Kani:

```rust
pub trait Invariant {
    fn is_valid(&self) -> bool;
}

impl Invariant for ValidatedTensor {
    fn is_valid(&self) -> bool {
        !self.dims.is_empty()
            && self.dims.iter().product::<usize>() == self.data.len()
    }
}
```

`pv scaffold` generates the `Invariant` impl from `type_invariants`.
`pv kani` generates preservation harnesses:

```rust
#[kani::proof]
fn verify_tensor_invariant_preserved_by_reshape() {
    let tensor: ValidatedTensor = kani::any();
    kani::assume(tensor.is_valid());
    let reshaped = tensor.reshape(&new_dims);
    assert!(reshaped.is_valid());
}
```

**Path B: Nightly Rust — `#[contracts::invariant]`.**

RFC #128044 adds native type invariants to the compiler. When stable,
`pv scaffold --nightly-invariants` would generate:

```rust
#![feature(contracts)]

#[contracts::invariant(self.dims.iter().product::<usize>() == self.data.len())]
#[contracts::invariant(!self.dims.is_empty())]
pub struct ValidatedTensor {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}
```

### 6.3. Codegen: `pv invariants`

New CLI command `pv invariants <contract.yaml>` generates:

1. `Invariant` trait impl for each type with declared invariants
2. Kani preservation harnesses for every function in the contract's
   binding that takes or returns the invariant-bearing type
3. `debug_assert!(self.is_valid())` calls in constructors (via
   `#[contract]` macro integration)

### 6.4. Relationship to Existing Obligations

| Concept | Scope | Checked when |
|---|---|---|
| `invariant` obligation | Function property | After function returns |
| `type_invariants` | Data structure property | After construction + every public method |
| `precondition` obligation | Function input | Before function body |
| `postcondition` obligation | Function output | Before function returns |
| `frame` obligation | Mutation scope | After function returns |

Type invariants are *orthogonal* to function-level obligations. A
function can satisfy all its pre/postconditions while violating a type
invariant on the return value. The preservation harness catches this:
`kani::assume(input.is_valid()); /* operation */; assert!(output.is_valid())`.

### 6.5. Escape-Proof Enforcement for Type Invariants

The six-stage pipeline extends naturally:

| Stage | Type Invariant Enforcement |
|---|---|
| A (YAML) | `type_invariants[]` section in contract |
| B (Lean) | Invariant as a `Prop` on the type; preservation theorems |
| C (Lint) | `pv lint`: every type with invariants has preservation harnesses |
| D (build.rs) | Generate `Invariant` trait impl |
| E (Macro) | `#[contract]` inserts `debug_assert!(result.is_valid())` |
| F (Tests) | Kani preservation harnesses + proptest |

---

## 7. Coq Integration

The current verification ladder has Lean 4 at L5 (unbounded proof) and
Kani at L4 (bounded model checking). Adding Coq provides a second
path to unbounded proofs with different strengths:

| Prover | Strengths | Ecosystem |
|---|---|---|
| **Lean 4** | Mathlib (analysis, algebra), tactic automation, `sorry` tracking | Active, growing |
| **Coq** | CompCert (verified C compiler), Fiat Crypto, `coq-of-rust` bridge | Mature, battle-tested |

The two are NOT interchangeable — they verify different aspects:
- **Lean** excels at mathematical properties (softmax sums to 1 over R)
- **Coq** excels at implementation verification (`coq-of-rust` translates
  actual Rust code to Coq, proving properties of *the code itself*)

### 7.1. YAML Schema Extension

```yaml
coq_spec:
  module: "SoftmaxSpec"
  imports:
    - "Require Import Reals."
    - "Require Import List."
  definitions:
    - name: "softmax_sum_to_one"
      statement: |
        Theorem softmax_partition_of_unity : forall (xs : list R),
          xs <> [] ->
          fold_left Rplus (map softmax_fn xs) 0 = 1.
  obligations:
    - links_to: "SM-INV-001"
      coq_lemma: "softmax_partition_of_unity"
      status: "proved|admitted|stub"
```

### 7.2. The `coq-of-rust` Bridge

`coq-of-rust` translates Rust code directly to Coq, giving a formal
model of the *actual implementation* — not a parallel specification.
This means Kani-verified bounds and Coq proofs refer to the *same
code*:

```bash
# In Makefile
coq-verify:
    coq-of-rust translate crates/aprender/src/kernels/softmax.rs \
        --output generated/coq/softmax.v
    coqc generated/coq/softmax.v
```

### 7.3. Codegen: `pv coq`

New CLI command `pv coq <contract.yaml>` generates:

1. `.v` file with `Require Import` statements from `coq_spec.imports`
2. Definitions from equations (translated to Coq `Definition`)
3. Theorem stubs from proof obligations (with `admit.` placeholders)
4. Obligation cross-references via `(** Obligation: SM-INV-001 *)`

```coq
(* Generated from softmax-kernel-v1 v1.0.0 *)
Require Import Reals.
Require Import List.

(** Equation: softmax *)
Definition softmax (xs : list R) : list R :=
  (* TODO: formalize *) nil.

(** Obligation: Output sums to 1 [invariant] *)
(** Paper: Bridle (1990) *)
Theorem softmax_partition_of_unity :
  forall (xs : list R),
    xs <> [] ->
    fold_left Rplus (softmax xs) 0 = 1.
Proof.
  admit. (* replace with proof *)
Qed.
```

### 7.4. Tiered Proof Strategy

Full Coq proofs require mathematician time. The practical approach
(CompCert, seL4, Fiat Cryptography) is tiered:

```
Tier 1: Kani (automated, bounded)      <- current, L4
Tier 2: Lean 4 (semi-automated, R)     <- current, L5
Tier 3: Coq stubs (generated, admit)   <- new: pv coq
Tier 4: Coq proofs (human-verified)    <- manual, L5+
Tier 5: coq-of-rust (implementation)   <- automated translation
```

### 7.5. Audit Integration

`pv audit --coq` reports which obligations have:

| Status | Meaning |
|---|---|
| `kani_only` | Bounded verification (L4), no proof |
| `lean_proved` | Lean theorem over R (L5) |
| `coq_stub` | Coq theorem generated but unproved (`admit`) |
| `coq_proved` | Coq theorem fully discharged |
| `coq_of_rust` | Implementation translated and verified |

### 7.6. Relationship to Lean-Kani Composition

The `stub_float` bridge works identically with Coq:
- Coq proves `exp > 0` (over R)
- Kani stubs `f32::exp()` with `kani::any()` constrained by Coq's proof
- Kani verifies the surrounding Rust code preserves the invariant

The choice between Lean and Coq is per-obligation:
- Use Lean when Mathlib has the required analysis lemmas
- Use Coq when `coq-of-rust` can directly verify the implementation
- Use both when maximum assurance is needed (defense-in-depth)

### 7.7. Migration Path

1. Add `coq_spec` and `type_invariants` as optional schema sections
2. Implement `pv coq` command (generates `.v` stubs)
3. Implement `pv invariants` command (generates `Invariant` trait)
4. Extend `pv audit` with `--coq` flag
5. Extend `pv explain` to render type invariants and Coq status
6. Write exemplar Coq proofs for Tier 1 contracts (softmax, matmul)
7. Integrate `coq-of-rust` for implementation-level verification

### 7.8. References

1. Leroy, X. (2009). "Formal verification of a realistic compiler." *CACM* 52(7). (CompCert)
2. Klein, G. et al. (2009). "seL4: Formal Verification of an OS Kernel." *SOSP 2009.*
3. Erbsen, A. et al. (2019). "Simple High-Level Code For Cryptographic Arithmetic." *S&P 2019.* (Fiat Crypto)
4. `coq-of-rust` — github.com/formal-land/coq-of-rust
5. Rust RFC #128044 — `core::contracts` type invariants (nightly tracking)
6. Lattuada, A. et al. (2023). "Verus: Verifying Rust Programs using Linear Ghost Types." arXiv:2303.05491.
