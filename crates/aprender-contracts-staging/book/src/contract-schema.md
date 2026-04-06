# Contract Schema

Every YAML contract follows this schema. Fields marked `REQUIRED` must be
present; others are recommended.

```yaml
# <kernel-name>-v<version>.yaml

# REQUIRED: Metadata block
metadata:
  version: "1.0.0"                    # Semantic version
  created: "2026-MM-DD"               # Creation date
  author: "PAIML Engineering"         # Author
  description: "..."                  # One-line description
  references:                         # REQUIRED: Paper citations
    - "Author et al. (YYYY). Title. arXiv:XXXX.XXXXX"
    - "..."

# REQUIRED: Mathematical equations
equations:
  <equation_name>:
    formula: "..."                    # LaTeX-like formula
    domain: "..."                     # Input space
    codomain: "..."                   # Output space
    invariants:                       # Mathematical properties that MUST hold
      - "..."

# REQUIRED: Proof obligations extracted from equations
proof_obligations:
  - type: "invariant|equivalence|bound|monotonicity|idempotency|linearity"
    property: "..."                   # Human-readable description
    formal: "..."                     # Formal predicate
    tolerance: 1.0e-6                 # Numerical tolerance (if applicable)
    applies_to: "all|scalar|simd"     # Which implementations

# RECOMMENDED: Kernel structure (phase decomposition)
kernel_structure:
  phases:
    - name: "..."
      description: "..."
      invariant: "..."                # What must hold after this phase

# RECOMMENDED: SIMD dispatch table
simd_dispatch:
  <operation>:
    scalar: "fn_name"
    avx2: "fn_name"
    avx512: "fn_name"
    neon: "fn_name"                   # ARM

# REQUIRED: Enforcement rules
enforcement:
  <rule_name>:
    description: "..."
    check: "..."                      # How to verify
    severity: "ERROR|WARNING"

# REQUIRED: Falsification tests
falsification_tests:
  - id: "FALSIFY-<PREFIX>-NNN"
    rule: "..."                       # Which enforcement rule this tests
    prediction: "..."                 # What the correct implementation guarantees
    test: "..."                       # How to test
    if_fails: "..."                   # Root cause diagnosis

# REQUIRED: Kani verification harnesses
kani_harnesses:
  - id: "KANI-<PREFIX>-NNN"
    obligation: "..."                 # Which proof obligation this verifies
    bound: 16                         # Max input size (kani::unwind)
    strategy: "exhaustive|stub_float" # Exhaustive or stub transcendentals
    harness: "verify_<name>"          # Rust function name
    solver: "cadical|kissat|z3"       # SAT/SMT solver (optional, default cadical)

# REQUIRED: QA gate definition
qa_gate:
  id: "F-<PREFIX>-NNN"
  name: "..."
  checks:
    - "..."
  pass_criteria: "..."
  falsification: "..."               # Meta-test: introduce bug, gate must catch
```
