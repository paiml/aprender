# Sub-spec: Contract Schema

**Parent:** [pv-spec.md](../pv-spec.md) Section 3

---

## 1. YAML Contract Schema

Every contract follows this structure. Fields marked REQUIRED must be
present; others are recommended.

```yaml
metadata:
  version: "1.0.0"                    # REQUIRED: Semantic version
  created: "2026-MM-DD"               # REQUIRED: Creation date
  author: "PAIML Engineering"         # REQUIRED: Author
  description: "..."                  # REQUIRED: One-line description
  references:                         # REQUIRED: Paper citations
    - "Author et al. (YYYY). Title. arXiv:XXXX.XXXXX"
  depends_on:                          # OPTIONAL: Contract dependencies
    - "silu-kernel-v1"

equations:
  <equation_name>:
    formula: "..."                    # REQUIRED: LaTeX-like formula
    domain: "..."                     # Input space
    codomain: "..."                   # Output space
    invariants:                       # Mathematical properties
      - "..."

proof_obligations:
  - type: "invariant|equivalence|bound|...|precondition|postcondition|frame|..."
    property: "..."                   # Human-readable description
    formal: "..."                     # Formal predicate
    tolerance: 1.0e-6                 # Numerical tolerance
    applies_to: "all|scalar|simd"     # Implementation scope
    requires: "OB-ID"                 # postcondition only: links to precondition
    parent_contract: "stem"           # subcontract only: contract being refined
    lean:                             # Phase 7 Lean 4 metadata
      theorem: "softmax_partition_unity"
      module: "ProvableContracts.Softmax"
      status: "proved|sorry|wip|not-applicable"
      depends_on: []
      mathlib_imports: []

kernel_structure:
  phases:
    - name: "find_max"
      description: "Compute max(x)"
      invariant: "result = max(input)"

simd_dispatch:
  softmax:
    scalar: "softmax_scalar"
    avx2: "softmax_avx2"
    avx512: "softmax_avx512"
    neon: "softmax_neon"
    ptx: "softmax_ptx"

enforcement:
  <rule_name>:
    description: "..."
    check: "..."
    severity: "ERROR|WARNING"

falsification_tests:
  - id: "FALSIFY-<PREFIX>-NNN"
    rule: "..."                       # Which enforcement rule
    prediction: "..."                 # What correct impl guarantees
    test: "..."                       # How to test
    if_fails: "..."                   # Root cause diagnosis

kani_harnesses:
  - id: "KANI-<PREFIX>-NNN"
    obligation: "..."                 # Which proof obligation
    bound: 16                         # kani::unwind value
    strategy: "exhaustive|stub_float|compositional|bounded_int"
    harness: "verify_<name>"          # Rust function name
    solver: "cadical|kissat|z3"       # SAT/SMT solver

verification_summary:
  total_obligations: 7
  proved: 3
  sorry: 2
  wip: 1
  not_applicable: 1

qa_gate:
  id: "F-<PREFIX>-NNN"
  name: "..."
  checks: ["..."]
  pass_criteria: "..."
  falsification: "..."
```

---

## 2. Proof Obligation Type Reference

| Type | Pattern | Example |
|---|---|---|
| `invariant` | For-all x: P(f(x)) | softmax sums to 1 |
| `equivalence` | f(x) = g(x) within tolerance | SIMD matches scalar |
| `bound` | a <= f(x)_i <= b | sigmoid in (0,1) |
| `monotonicity` | x_i > x_j implies f(x)_i > f(x)_j | softmax order |
| `idempotency` | f(f(x)) = f(x) | relu(relu(x)) = relu(x) |
| `linearity` | f(alpha*x) = alpha*f(x) | relu positive scaling |
| `symmetry` | dot(a,b) = dot(b,a) | dot product |
| `associativity` | (a+b)+c = a+(b+c) | integer arithmetic |
| `conservation` | Q(before) = Q(after) | attention mass = 1.0 |
| `ordering` | f preserves total/partial order | sorted output |
| `completeness` | all cases covered | dispatch exhaustive |
| `soundness` | no false positives | validation rejects bad |
| `involution` | f(f(x)) = x | encode/decode roundtrip |
| `determinism` | f(x) = f(x) always | sampling with fixed seed |
| `roundtrip` | decode(encode(x)) = x | serialization fidelity |
| `state_machine` | S x A -> S valid transitions | cache state FSM |
| `classification` | f(x) in C | output in valid class set |
| `independence` | P(A∩B) = P(A)*P(B) | feature independence |
| `termination` | algorithm halts | convergence loop exits |
| `precondition` | P(input) before call | input finite, non-empty |
| `postcondition` | P(in) -> Q(out) guarantee | given valid input, output in range |
| `frame` | modifies(S), preserves(rest) | only output buffer written |
| `loop_invariant` | for-all iter i: P(state_i) | running max tracks true max |
| `loop_variant` | V(state) in N, decreasing | remaining = n - i |
| `old_state` | Q(old, new) state relation | cache.len grows by seq_len |
| `subcontract` | weaken(pre), strengthen(post) | GQA refines MHA |

See **[eiffel-dbc.md](eiffel-dbc.md)** for full definitions of the
Eiffel DbC types (last 7 rows).

---

## 3. Tolerance Selection

Tolerances are derived from the arithmetic, not guessed:

| Operation | Source of Error | Typical Tolerance |
|---|---|---|
| f32 addition (n terms) | Catastrophic cancellation | `n * f32::EPSILON` |
| f32 multiply-accumulate | Rounding per FMA | `sqrt(n) * f32::EPSILON` |
| Quantized dot product | Dequantization error | `ULP_TOLERANCE * f32::EPSILON` |
| Softmax normalization | Exp + division | `1e-6` absolute on sum |
| RMSNorm | Sqrt + division | `1e-4` absolute |
| SIMD vs scalar | Reassociation | `ULP_TOLERANCE` (format-specific) |

---

## 4. Naming Conventions

### Contract Files

```
<operation>-kernel-v<version>.yaml      # Kernel contracts
<operation>-v<version>.yaml             # Non-kernel contracts
```

### Falsification Test IDs

`FALSIFY-<PREFIX>-NNN` where PREFIX maps to:

| Contract | Prefix |
|---|---|
| Softmax | SM |
| RMSNorm | RMS |
| Attention | ATTN |
| FlashAttention | FATTN |
| RoPE | ROPE |
| MatMul | MM |
| SwiGLU/GeGLU | ACT |
| Quantized Dot | QDOT |
| Tensor Layout | LAYOUT |
| Layer Parity | PARITY |

### Kani Harness IDs

`KANI-<PREFIX>-NNN` using the same prefix table.

### QA Gate IDs

`F-<PREFIX>-NNN` (matches certeza format).

---

## 5. Versioning Rules

Contracts follow semantic versioning:

| Bump | Trigger |
|---|---|
| **MAJOR** | Removed equation, tightened tolerance, new required obligation |
| **MINOR** | New optional obligation, new SIMD entry, new test |
| **PATCH** | Typo fix, clarification, new reference |

The `pv diff` command auto-detects the required bump type.

---

## 6. Contract Classification — `metadata.kind`

Every YAML file in `contracts/` declares a `kind`, which determines which
validation rules apply. The default is `kernel` (unstated = kernel).

| `kind`         | Proofs? | Purpose                                          |
|----------------|---------|--------------------------------------------------|
| `kernel`       | required | mathematical kernel contract                    |
| `registry`     | exempt  | lookup tables, enum definitions, config bounds  |
| `model-family` | exempt  | architecture metadata, size variants, vendors   |
| `pattern`      | exempt  | cross-cutting verification patterns             |
| `schema`       | exempt  | generic reference/schema documents              |

Non-kernel kinds are first-class pv artifacts: `pv query` finds them,
they are scored, and they appear in composition chains when referenced
via `depends_on`. They are exempt only from the provability invariant
and kernel-specific lint gates.

**Back-compat:** `metadata.registry: true` is equivalent to
`metadata.kind: registry` and remains supported.

Kernel contracts MUST have the full chain:

```
equations → proof_obligations → falsification_tests → kani_harnesses
```

**Provability Invariant (enforced by `pv validate` and test suite):**

```
∀ contract C where C.kind = kernel:
  |C.proof_obligations| > 0                                    # MUST have obligations
  |C.kani_harnesses|   > 0                                    # MUST have Kani harnesses
  |C.falsification_tests| >= |C.proof_obligations|            # every obligation falsified
  ∀ h ∈ C.kani_harnesses: h.obligation ∈ C.proof_obligations  # harnesses trace to obligations
```

**Example `kind: model-family`:**

```yaml
metadata:
  version: "1.0.0"
  description: "Google BERT architecture family metadata"
  kind: model-family
  references:
    - "https://arxiv.org/abs/1810.04805"
# Custom top-level fields consumed by downstream crates
family: bert
architectures: [BertModel, BertForMaskedLM]
size_variants:
  base: { parameters: "110M", hidden_dim: 768 }
```

**Known data registries:** `arch-constraints-v1`, `model-metadata-bounds-v1`,
`special-tokens-registry-v1`, `tensor-names-v1`, `codebert-tokenizer-validation-v1`.

---

## 7. Validation Rules

`pv validate` checks:

1. All REQUIRED sections present
2. Provability invariant enforced (kernel contracts only)
3. Every falsification test references a valid enforcement rule
4. Every Kani harness references a valid proof obligation
5. No duplicate IDs across falsification tests or harnesses
6. `depends_on` entries reference existing contract stems
7. Version field is valid semver
8. At least one paper reference in metadata

Violations are reported with severity (ERROR, WARNING, INFO).
