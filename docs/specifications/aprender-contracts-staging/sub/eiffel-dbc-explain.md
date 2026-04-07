# Eiffel DbC — Migration & pv explain

*See also: [eiffel-dbc.md](eiffel-dbc.md) (sections 1-5)*

## 9. Migration Path

### Phase 1: Schema + Validation

1. Add 7 new variants to `ObligationType`
2. Add optional `requires`, `applies_to`, `parent_contract` fields to
   `ProofObligation`
3. Update `pv validate` to enforce field/type constraints
4. Update `pv query --obligation` to filter by new types

**Already done (partial):** Equation-level `preconditions`/
`postconditions` fields exist, `#[requires]`/`#[ensures]`/
`#[invariant]` proc macros exist, `codegen.rs` generates assertion
macros, Gate 5 checks for pre/post on equations. What remains is
adding the `ObligationType` variants and the obligation-level fields.

### Phase 2: Codegen

5. Add probar test generators for each new type (extend `probar_gen/`)
6. Add Kani harness generators for each new type (extend `kani_gen/`)
7. Add Lean 4 theorem generators for each new type (extend `lean_gen/`)
8. Wire obligation-level pre/post into the existing escape-proof
   enforcement pipeline (Stages D-E) alongside equation-level pre/post

### Phase 3: Kernel Adoption

8. Retrofit `precondition`/`postcondition` pairs onto Tier 1 contracts
   (softmax, rmsnorm, rope, silu) as exemplars
9. Add `frame` obligations to KV cache and in-place buffer contracts
10. Add `loop_invariant`/`loop_variant` to iterative contracts
    (online-softmax, adamw, lbfgs, cma-es, pagerank)
11. Add `subcontract` obligations where `depends_on` represents
    behavioral subtyping (GQA→attention, flash-attention→attention)

### Phase 4: Cross-Domain Expansion

12. **simular (Tier 8):** Retrofit existing 3 contracts (checkpoint,
    gradient, loss-functions) with `precondition`/`postcondition` pairs.
    Add `frame` obligations to integration steps. Add `loop_invariant`/
    `loop_variant` to simulation loops. Add `old_state` for energy
    drift tracking.
13. **forjar (Tier 9):** Retrofit existing `#[contract]` annotations
    with full YAML contracts. Add `frame` to resource apply (only
    target resource changes). Add `precondition` to transport safety
    (bashrs validation gate). Add `old_state` to drift detection
    (BLAKE3 hash comparison). Add `subcontract` for pepita→SSH
    transport refinement. Add `loop_invariant`/`loop_variant` to
    DAG wave execution.
14. **batuta (Tier 10):** Write contracts for the transpilation
    pipeline: `postcondition` (output compiles), `equivalence`
    (transpiled semantics match source on test suite), `frame`
    (source files unchanged), `subcontract` (PyTorchConverter
    refines TranspilerPlugin). Write contracts for BackendSelector
    cost model (`bound` on latency, `postcondition` on correctness).
15. **pmat (Tier 11):** Formalize the uniform contracts pattern as
    YAML: `subcontract` (each specialized contract refines base),
    `frame` (analysis is read-only), `determinism` (same input →
    same grade), `invariant` (TDG monotonic). Write contracts for
    the TDG scoring formula and mutation testing threshold.
16. **rmedia (Tier 12):** Extract render pipeline score dimensions as
    YAML contracts. Write contracts for codec parity (`equivalence`
    with melt), determinism (`determinism` — identical hash), SRT lock
    (`old_state` — SHA-256 match), SVG quality floors (`bound` on
    opacity/stroke/font), animation timing (`bound` on SRT alignment
    within ±2 frames), frame pipeline (`loop_invariant` — channel
    depth ≤ 16, `loop_variant` — remaining frames decreasing),
    compositing correctness (`invariant` — YUV420P maintained).
17. **probar (Tier 13):** Extract Brick Architecture implicit contracts
    into YAML. Write contracts for locator resolution (precondition:
    element exists), visual regression (frame: reference image
    unchanged), and playbook replay (determinism, roundtrip).
18. Write exemplar presentation contracts (layout, accessibility,
    animation) for presentar — demonstrate pre/post/frame/old-state
19. Write exemplar data pipeline and API contracts for future consumers
20. Update `pv scaffold` to generate domain-appropriate test skeletons
    (e.g., state snapshot tests for simulation, lock-file assertions
    for IaC, equivalence tests for transpilation, ffprobe assertions
    for media, DOM snapshot tests for presentation)
21. Update `pv generate` to produce deterministic README.md and
    GitHub Actions workflow files for consumer projects
22. Add Tier 8-17 to the contract registry classification

---

## 10. `pv explain` — Contract Narrative Command

Before implementing the 7 new DbC obligation types, we need the
ability to *explain* any contract in detail. The existing commands
provide counts (`pv status`), gap detection (`pv audit`), formulas
(`pv equations`), and reference pages (`pv generate` → `_book.md`).
None produces a narrative explanation of *what the contract means,
why each obligation exists, and how the verification chain works*.

### 10.1. Command Interface

```bash
pv explain <contract.yaml>                              # text narrative
pv explain <contract.yaml> --format markdown             # markdown with LaTeX
pv explain <contract.yaml> --format json                 # structured JSON
pv explain <contract.yaml> --binding binding.yaml        # include binding status
```

### 10.2. Output Structure

`pv explain` renders a **chain-of-thought narrative** organized into
these sections:

**1. What this contract specifies** — One-paragraph summary derived
from `metadata.description` and `metadata.references`. Names the
governing paper(s) and the domain.

**2. Governing equations** — For each equation in `equations`:
- Formula with domain/codomain
- Mathematical invariants (prose, not table)
- Preconditions (if present): what the caller must guarantee
- Postconditions (if present): what the kernel guarantees
- `lean_theorem` reference (if present): link to the L5 proof

**3. Proof obligations** — For each obligation in `proof_obligations`:
- Obligation type with its mathematical pattern
  (e.g., "invariant: ∀x ∈ Domain: P(f(x))")
- Property and formal predicate
- Tolerance (if numeric)
- **Why this matters** — prose explaining the obligation's purpose
- Lean proof status (proved/sorry/wip/not-applicable) with theorem
  name, dependencies, and mathlib imports
- Cross-references to falsification tests and Kani harnesses that
  verify this obligation

**4. Verification ladder** — Summary of verification coverage:
- L5 (Lean): N of M proved (percentage)
- L4 (Kani): N harnesses with strategy breakdown
- L3 (probar): N property tests
- L2 (falsification): N tests
- Overall proof level (L1-L5)

**5. Falsification tests** — For each test:
- Prediction (what should hold)
- Method (how it's tested)
- What it catches (root cause if it fails)
- Explain the Popperian structure: each test tries to *refute* the
  contract, not confirm it

**6. Kani bounded model checking** — For each harness:
- Strategy explanation:
  - `exhaustive`: verify for ALL inputs within bound
  - `stub_float`: assume Lean-proved postconditions on transcendentals,
    verify surrounding code
  - `compositional`: verify sub-kernels separately, compose proofs
  - `bounded_int`: integer-only verification
- Bound and solver

**7. Kernel execution phases** — If `kernel_structure` present:
- Sequential walkthrough of phases with invariants
- Explain how each phase's invariant feeds into the next

**8. SIMD dispatch** — If present: list dispatch targets

**9. Enforcement rules** — If present: list with severity

**10. Quality gate** — If present: explain pass criteria and mutation test

**11. Binding status** — If `--binding` provided:
- Which equations are implemented/partial/missing
- Which project(s) consume this contract

### 10.3. Example: Softmax

```
softmax-kernel-v1 (v1.0.0)
Softmax kernel — numerically stable exponential normalization

What this contract specifies
  This contract governs the softmax function, which normalizes a vector
  of real numbers into a probability distribution. It derives from
  Bridle (1990) and Milakov & Gimelshein (2018).

Governing equations

  softmax
    σ(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
    Domain: x ∈ ℝ^n, n ≥ 1
    Range:  σ(x) ∈ (0,1)^n

    The equation uses the max-subtraction trick for numerical stability:
    subtracting max(x) before exponentiation prevents overflow while
    preserving the result (translation invariance).

    Invariants:
      1. Σ σ(x)_i = 1.0 — outputs form a probability distribution
      2. σ(x)_i > 0 — all outputs strictly positive (exp > 0)
      3. argmax(σ(x)) = argmax(x) — largest input → largest output

Proof obligations (6)

  1. [invariant] Output sums to 1
     Pattern: ∀x ∈ Domain: P(f(x)) — property holds for all inputs
     Formal:  |Σ σ(x)_i - 1.0| < ε  (tolerance: 1e-6)
     Why: The defining property of a probability distribution. If this
       fails, downstream sampling and cross-entropy loss produce garbage.
     Lean: Softmax.partition_of_unity (proved)
       Depends: Real.exp_pos, Finset.sum_div_distrib
       Note: Proof over reals; f32 gap addressed by error-bound lemma.
     Verified at: L2 (FALSIFY-SM-001), L4 (KANI-SM-001), L5 (Lean)

  2. [invariant] All outputs strictly positive
     Pattern: ∀x ∈ Domain: P(f(x))
     Formal:  σ(x)_i > 0 for all i
     Why: Logarithm of softmax output must be defined (log-softmax in
       cross-entropy). Zero outputs cause -inf and NaN propagation.
     Lean: Softmax.softmax_pos (proved)
     Verified at: L2 (FALSIFY-SM-002), L4 (KANI-SM-002), L5 (Lean)

  ...

Verification ladder
  L5 (Lean):  5/6 proved (83%)
  L4 (Kani):  3 harnesses (stub_float strategy)
  L2 (Tests): 6 falsification tests
  Level: L4

Falsification tests (Popperian)
  Each test tries to refute the contract. Survival = evidence for,
  not proof of, correctness.

  FALSIFY-SM-001: Normalization
    Predicts: sum(softmax(x)) ≈ 1.0 for random x ∈ [-1000, 1000]^n
    Method:   proptest with 10000 random vectors, dim 1..128
    Catches:  Missing or incorrect max-subtraction trick

  FALSIFY-SM-004: SIMD equivalence
    Predicts: |softmax_avx2(x) - softmax_scalar(x)| < 8 ULP
    Method:   proptest comparing scalar vs SIMD output
    Catches:  SIMD reduction order differs from scalar

Kani bounded model checking
  KANI-SM-001: verify_softmax_normalization (bound: 8, stub_float)
    Assumes Lean-proved postconditions on exp() (positive, finite),
    then verifies the sum-to-1 invariant holds for ALL vectors of
    length ≤ 8 regardless of exp's exact return value.

Kernel phases
  1. find_max      — Compute max(x) for stability [max ≥ x_i ∀i]
  2. exp_subtract  — exp(x_i - max) per element   [result ∈ (0,1]]
  3. sum_exp       — Σ exp(x_i - max)             [sum > 0]
  4. normalize     — Divide each exp by sum        [output_i = exp_i/sum]

SIMD dispatch
  softmax: scalar → softmax_scalar | avx2 → softmax_avx2 | ptx → softmax_ptx

Enforcement
  normalization — Output must sum to 1.0 (ERROR) → FALSIFY-SM-001
  positivity    — All outputs positive   (ERROR) → FALSIFY-SM-002

Quality gate: F-SM-001 Softmax Contract
  Pass: All 6 falsification tests + Kani harnesses verify
  Mutation: Introduce off-by-one in max reduction loop
```

### 10.4. Implementation

**Library module:** `crates/provable-contracts/src/explain.rs`

```rust
pub fn explain_contract(
    contract: &Contract,
    stem: &str,
    binding: Option<&BindingRegistry>,
) -> String
```

Reuses existing building blocks:
- `schema::parse_contract()` — parse YAML
- `proof_status::compute_proof_level()` — L1-L5 level
- `probar_gen::obligation_pattern()` — pattern strings per type
- `binding::parse_binding()` — binding registry
- `latex::math_to_latex()` — math rendering (markdown format)

**CLI handler:** `crates/provable-contracts-cli/src/commands/explain.rs`

Adds `Explain` variant to `Commands` enum with `contract: PathBuf`,
`--format text|markdown|json`, and `--binding` option.

### 10.5. Relationship to DbC Types

When the 7 new obligation types are implemented, `pv explain` will
render their narratives using the same pattern:

| DbC Type | Explain Narrative |
|---|---|
| `precondition` | "Caller must guarantee: [formal]. If violated, kernel behavior is undefined." |
| `postcondition` | "Kernel guarantees: [formal], conditional on precondition [requires] holding." |
| `frame` | "This operation modifies [modifies set] only. All other state is preserved." |
| `loop_invariant` | "At every iteration of [applies_to]: [formal]. This is maintained inductively." |
| `loop_variant` | "Termination witness: [formal]. Strictly decreasing, proving the loop exits." |
| `old_state` | "Relates pre-call state to post-call state: [formal]." |
| `subcontract` | "This contract refines [parent_contract]. Preconditions weakened, postconditions strengthened." |

This makes `pv explain` the primary user-facing tool for understanding
contracts — including the new Eiffel DbC types as they are adopted.

### 10.6. Generated Artifacts: README.md and CI Workflows

`pv generate` already produces per-contract Rust artifacts (`_scaffold.rs`,
`_kani.rs`, `_probar.rs`, `_book.md`). It should also generate project-
level artifacts that consumer projects can adopt:

**`pv generate --readme`** — generates a `README.md` for a consumer project
that documents its contract coverage:

```bash
pv generate contracts/ --readme --binding contracts/aprender/binding.yaml \
    --output aprender/
```

Produces `aprender/CONTRACT-README.md` containing:

1. **Contract coverage badge** — "28/31 contracts bound (90.3%)"
2. **Bound contracts table** — contract stem, equation, binding function,
   proof level (L1-L5), Lean status
3. **Verification ladder summary** — how many obligations at each level
4. **Gap list** — unbound equations with priority and suggested action
5. **Build integration** — `build.rs` setup instructions for
   `#[contract]` enforcement
6. **CI integration** — reference to generated workflow file

**`pv generate --ci`** — generates a GitHub Actions workflow for contract
validation in the consumer project:

```bash
pv generate contracts/ --ci --output .github/workflows/
```

Produces `.github/workflows/contracts.yml`:

```yaml
name: Contract Validation
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  validate:
    name: Contract Validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/checkout@v4
        with:
          repository: paiml/provable-contracts
          path: provable-contracts

      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2

      - name: Install pv
        run: cargo install --path provable-contracts/crates/provable-contracts-cli

      - name: Validate contracts
        run: pv validate provable-contracts/contracts/*.yaml

      - name: Lint contracts
        run: pv lint provable-contracts/contracts/ --min-score 0.60

      - name: Verify bindings
        run: pv lint provable-contracts/contracts/
             --binding provable-contracts/contracts/$PROJECT/binding.yaml

      - name: Check Lean theorems (no sorry)
        run: |
          cd provable-contracts/lean
          lake build
          # Verify no sorry in proved theorems
          grep -r "sorry" ProvableContracts/Theorems/ && exit 1 || true

      - name: Run falsification tests
        run: cargo test --workspace
```

The workflow enforces the escape-proof pipeline at CI level:
- Gate C (YAML validation): `pv validate` + `pv lint`
- Gate D (binding check): `pv lint --binding`
- Gate B (Lean no-sorry): `lake build` + sorry grep
- Gate F (tests): `cargo test`

This makes contract enforcement automatic for any project that consumes
provable-contracts — no manual CI configuration needed.

**Determinism invariant:** All generated artifacts (README.md, CI
workflows, Rust files) must be **fully deterministic**. Same inputs →
identical byte-for-byte output. No timestamps, no git hashes, no
random IDs, no `created:` dates, no environment-dependent values.
This is a `determinism` obligation on the generator itself:

```yaml
proof_obligations:
  - type: determinism
    property: "pv generate output is deterministic"
    formal: "generate(contracts, binding) = generate(contracts, binding)"
```

Running `pv generate` twice with the same contracts and binding
must produce identical files. This enables `git diff` to detect real
changes (not regeneration noise) and makes CI caching effective.

**Cross-project CI matrix** — for the full stack, the generated workflow
can include a matrix strategy testing all bound projects:

```yaml
strategy:
  matrix:
    project: [aprender, entrenar, realizar, trueno, forjar, simular]
```

Each matrix entry validates that project's binding.yaml against the
contracts, ensuring no project ships with broken bindings.

---

