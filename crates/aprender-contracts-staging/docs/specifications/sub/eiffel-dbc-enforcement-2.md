# Eiffel DbC — Escape-Proof Enforcement (Part 2)

*See also: [eiffel-dbc-enforcement.md](eiffel-dbc-enforcement.md) (Part 1)*

### 9.5. Chain of Thought: Orchestration (batuta)

**Domain axiom:** `eval(transpile(source), input) = eval(source, input)` (semantic equivalence)

**Stage A — Equation YAML:**
```yaml
equations:
  transpile_equivalence:
    formula: "∀ input: eval(transpile(py_source), input) = eval(py_source, input)"
    preconditions:
      - "source.is_valid_python()"
      - "source.uses_only_supported_ops()"
    postconditions:
      - "output.compiles_as_rust()"
      - "output.function_count() == source.function_count()"
    lean_theorem: "ProvableContracts.Theorems.Transpile.Equivalence"
```
**Chain of thought:** Transpilation is the domain where postconditions
are *hardest to enforce* because semantic equivalence is undecidable
in general. The precondition narrows the domain to supported ops, and
the postcondition checks *structural* properties (compiles, same
function count) that are decidable. The full semantic equivalence
is verified at Stage F via test-suite comparison, not compile-time
assertion.

**NEW: Frame for transpilation:**
```yaml
proof_obligations:
  - type: frame
    property: "Transpilation does not modify source files"
    formal: "modifies(output_dir) ∧ preserves(source_dir)"
```
**Stage D codegen:**
```rust
let source_hash_before = blake3_hash_dir(&source_dir);
let result = transpile(&source_dir, &output_dir);
debug_assert!(
    blake3_hash_dir(&source_dir) == source_hash_before,
    "Frame violated: transpilation modified source directory"
);
```
**Chain of thought:** This is defensive but critical. A transpiler bug
that *modifies the source while reading it* would be catastrophic —
the user loses their original code. The frame obligation makes this
impossible to ship: `debug_assert!` catches it in testing, Kani can
verify it for bounded inputs.

**NEW: Loop invariant for pipeline stages:**
```yaml
proof_obligations:
  - type: loop_invariant
    property: "Pipeline context valid at each stage transition"
    formal: "∀k ≤ i: context.validate() == Ok(()) after stage k"
    applies_to: "pipeline.execute"
```
**Stage E** — the pipeline's `execute()` method:
```rust
#[contract("pipeline-v1", equation = "pipeline_stages")]
pub async fn execute(&self, mut ctx: PipelineContext) -> Result<PipelineContext> {
    for stage in &self.stages {
        ctx = stage.execute(ctx).await?;
        // Macro inserts: debug_assert!(ctx.validate().is_ok(),
        //     "Loop invariant violated after stage {}", stage.name());
    }
    Ok(ctx)
}
```
**Chain of thought:** Batuta's `PipelineStage` trait already has a
`validate()` method — the loop invariant obligation makes the call to
`validate()` a *contractual requirement*, not an optional check. If
a new stage is added that breaks the context between stages, the
debug assertion fires immediately. This is Meyer's seamless development:
the Jidoka "stop on error" principle is *formalized as a loop invariant*.

### 9.6. Chain of Thought: Code Quality (pmat)

**Domain axiom:** `analyze(codebase) → metrics` where `metrics` are
deterministic, complete, and read-only.

**Stage A — Equation YAML:**
```yaml
equations:
  tdg_score:
    formula: "TDG = w₁·cyclomatic + w₂·cognitive + w₃·coverage + w₄·docs + w₅·satd + w₆·duplication"
    preconditions:
      - "path.exists()"
      - "path.is_dir()"
    postconditions:
      - "score >= 0.0 && score <= 100.0"
      - "grade.is_valid()"
    lean_theorem: "ProvableContracts.Theorems.TDG.BoundedScore"
```
**Chain of thought:** Code quality analysis has a surprising
*mathematical* foundation — the TDG formula is a weighted linear
combination. The postcondition (`score ∈ [0, 100]`) is provable
by Lean if each component is bounded. The precondition (`path exists`)
is the caller's responsibility.

**NEW: Frame for read-only analysis:**
```yaml
proof_obligations:
  - type: frame
    property: "Analysis never modifies the analyzed codebase"
    formal: "preserves(codebase_dir)"
```
**Stage D codegen:**
```rust
let codebase_hash_before = blake3_hash_dir(&path);
let result = analyze_tdg(&path);
debug_assert!(
    blake3_hash_dir(&path) == codebase_hash_before,
    "Frame violated: analysis modified codebase"
);
```
**Chain of thought:** An analysis tool that modifies the code it
analyzes would be a catastrophic trust violation. The frame obligation
makes this *contractually impossible* in debug builds. This is the
Hippocratic principle: "first, do no harm" — formalized as a DbC
frame condition.

**NEW: Subcontract for uniform interface contracts:**
```yaml
proof_obligations:
  - type: subcontract
    property: "AnalyzeComplexityContract refines BaseAnalysisContract"
    formal: "fields(Base) ⊂ fields(Complexity) ∧ semantics(Base) preserved"
    parent_contract: "base-analysis-v1"
```
**Stage F test:**
```rust
#[test]
fn complexity_contract_refines_base() {
    // Every field in BaseAnalysisContract exists in AnalyzeComplexityContract
    let base_fields = BaseAnalysisContract::field_names();
    let complexity_fields = AnalyzeComplexityContract::field_names();
    for field in &base_fields {
        assert!(complexity_fields.contains(field),
            "Subcontract violated: base field {} missing from complexity contract", field);
    }
}
```
**Chain of thought:** PMAT's uniform contracts pattern — where every
specialized contract `#[serde(flatten)]`s the base — is a structural
invariant that could break silently if someone removes a base field
from a specialization. The `subcontract` obligation makes this a
tested property. If a developer removes `format` from
`AnalyzeComplexityContract`, the test fails immediately.

### 9.7. Chain of Thought: Testing (probar)

**Domain axiom:** `assert(locator.find(element)) → element.exists_in_DOM`

**Stage A — Equation YAML:**
```yaml
equations:
  locator_resolution:
    formula: "find(selector, DOM) → element | timeout"
    preconditions:
      - "browser.is_connected()"
      - "!selector.is_empty()"
    postconditions:
      - "result.is_some() → result.unwrap().matches(selector)"
      - "result.is_none() → elapsed >= timeout"
```
**Chain of thought:** Testing framework contracts are *meta-contracts*
— they specify the behavior of the tool that checks other tools'
behavior. The precondition (browser connected) is the caller's
responsibility. The postcondition says: either we found the element
and it matches, or we timed out. There is no third state.

**NEW: Frame for visual regression:**
```yaml
proof_obligations:
  - type: frame
    property: "Visual regression comparison preserves reference image"
    formal: "modifies(diff_buffer) ∧ preserves(reference_image)"
```
**Stage D codegen:**
```rust
let ref_hash_before = blake3_hash(&reference_path);
let diff = visual_compare(&reference_path, &screenshot);
debug_assert!(
    blake3_hash(&reference_path) == ref_hash_before,
    "Frame violated: visual regression modified reference image"
);
```
**Chain of thought:** A visual regression tool that accidentally
overwrites the reference image when comparing would silently accept
all future regressions. The frame obligation prevents this: the
reference image must be byte-identical before and after comparison.

### 9.8. Chain of Thought: Infrastructure as Code — Forjar Transport

**Domain axiom:** `∀ script: SSH.accepts(script) → Pepita.accepts(script)` (Liskov)

This is a deeper trace through `subcontract` enforcement specifically,
showing how Meyer's subcontracting rules map to the compile pipeline.

**Stage A — Two YAML contracts with parent-child relationship:**
```yaml
# ssh-transport-v1.yaml
equations:
  exec_script:
    preconditions:
      - "!script.is_empty()"
      - "script.is_valid_posix()"
    postconditions:
      - "exit_code.is_some()"

# pepita-transport-v1.yaml
metadata:
  depends_on: ["ssh-transport-v1"]
equations:
  exec_script:
    preconditions:
      - "!script.is_empty()"         # Same (not strengthened)
      # NOTE: script.is_valid_posix() NOT required — Pepita accepts more
    postconditions:
      - "exit_code.is_some()"        # Same (not weakened)
      - "namespace.is_isolated()"    # Added (strengthened)
```
**Chain of thought:** Pepita *weakens* the precondition (doesn't
require POSIX validation — it runs in a namespace so non-POSIX is
safe) and *strengthens* the postcondition (adds namespace isolation
guarantee). This is exactly Meyer's `require else` / `ensure then`
pattern.

**Stage C — Validation:** `pv validate` checks:
1. `pepita-transport-v1.yaml` declares `depends_on: [ssh-transport-v1]`
2. The `subcontract` obligation references `parent_contract: ssh-transport-v1`
3. The parent contract exists in the registry

**Stage D — build.rs:** Generates cross-contract assertions:
```
CONTRACT_PEPITA_TRANSPORT_V1_EXEC_SCRIPT_PARENT=ssh-transport-v1
```

**Stage E — #[contract] macro** on pepita's `exec_script`:
```rust
#[contract("pepita-transport-v1", equation = "exec_script")]
pub fn exec_script(script: &str) -> ExecOutput {
    // Macro injects pepita's weaker preconditions
    // AND verifies parent contract binding exists
}
```

**Stage F — Cross-contract falsification test:**
```rust
#[test]
fn pepita_never_rejects_what_ssh_accepts() {
    proptest!(|(script in ssh_valid_scripts())| {
        // If SSH accepts this script, Pepita MUST also accept it
        assert!(pepita_accepts(&script),
            "Liskov violation: Pepita rejects '{}' but SSH accepts it", script);
    });
}
```

**Escape analysis for subcontracting:**

| Attempted Escape | Gate That Catches It |
|---|---|
| Pepita adds a precondition SSH doesn't have | Stage F: cross-contract proptest fails |
| Pepita removes a postcondition SSH guarantees | Stage F: cross-contract proptest fails |
| Developer removes `depends_on` | Stage C: `pv validate` — `parent_contract` references missing dep |
| Developer removes subcontract obligation | Stage C: `pv lint` — consistency check fails |

### 9.9. Summary: Escape Prevention by Stage

| Stage | What It Prevents | Enforcement Mechanism | Runtime Cost |
|---|---|---|---|
| A (YAML) | Missing specification | Schema validation, required fields | Zero |
| B (Lean) | Unproved mathematics | `sorry` detection → `compile_error!` | Zero |
| C (Lint) | Inconsistent contracts | `pv lint` non-zero exit → CI blocks | Zero |
| D (build.rs) | Missing assertions | Env var generation from YAML | Zero |
| E (Macro) | Unbound functions | `option_env!` check → `compile_error!` | Zero (debug_assert) |
| F (Tests) | Incorrect behavior | Falsification tests → CI blocks | Test-time only |

**Total runtime cost in release binary: zero.** The proof exists
in the build artifacts, not the shipped code. Like SPARK/Ada's
proof discharge, the verification is *consumed* during compilation
and leaves no trace in the deployed binary.

---

