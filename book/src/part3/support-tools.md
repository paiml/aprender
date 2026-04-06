# Support Tools

The Sovereign AI Stack includes essential support tools for scripting, quality analysis, and system tracing. These tools integrate with Batuta's orchestration workflow.

## Tool Overview

| Tool | Purpose | Integration Point |
|------|---------|-------------------|
| **Ruchy** | Rust scripting language | Embedded scripting, automation |
| **PMAT** | Quality analysis (TDG scoring) | Phase 1: Analysis, CI/CD gates |
| **APR-QA** | APR model validation | Model quality assurance |
| **Renacer** | Syscall tracing | Phase 4: Validation |
| **Provable Contracts** | YAML → Kani formal verification | Kernel correctness proofs |
| **Tiny Model Ground Truth** | Popperian model parity tests | Conversion validation |

## Ruchy: Rust Scripting

Ruchy provides a scripting language that compiles to Rust, enabling:

- **Automation scripts:** Build, deployment, data processing
- **Embedded scripting:** In Presentar apps (Section 8)
- **REPL development:** Interactive exploration

```ruchy
// Ruchy script for data processing
let data = load_dataset("transactions")
let filtered = data.filter(|row| row.amount > 100)
let aggregated = filtered.group_by("category").sum("amount")
save_dataset(aggregated, "output.ald")
```

**Security (in Presentar):**
- Max 1M instructions per script
- Max 16MB memory allocation
- 10ms time slices (cooperative yielding)

## PMAT: Quality Analysis

PMAT computes Technical Debt Grade (TDG) scores for projects:

- **0-100 scale:** F, D, C-, C, C+, B-, B, B+, A-, A, A+
- **Multi-language:** Rust, Python, C/C++, Shell
- **Metrics:** Complexity, coverage, duplication, dependencies

```bash
# Analyze a project
pmat analyze ./myproject --output report.json

# CI gate (fail if below B+)
pmat gate ./myproject --min-grade B+
```

**Integration with Batuta:**
- Phase 1 (Analysis): Initial TDG assessment
- Phase 4 (Validation): Post-transpilation quality check
- CI/CD: Gate enforcement

## Renacer: Syscall Tracing

Renacer captures system call traces for validation:

- **Deterministic replay:** Ensures transpiled code matches original behavior
- **Golden trace comparison:** Baseline vs current
- **Cross-platform:** Linux, macOS, Windows

```bash
# Capture baseline trace
renacer capture ./original_binary -- args > baseline.trace

# Compare against transpiled
renacer compare baseline.trace ./transpiled_binary -- args
```

**Integration with Batuta:**
- Phase 4 (Validation): Behavioral equivalence testing

## APR-QA: Model Quality Assurance

APR-QA provides a comprehensive QA playbook for APR models:

- **Test Generation:** Automatic QA test generation for APR models
- **Model Validation:** Verify model correctness and integrity
- **Benchmark Runner:** Performance benchmarks on APR models
- **Coverage Reports:** Model coverage analysis and reporting

```bash
# Generate QA tests for an APR model
apr-qa gen model.apr --output tests/

# Run QA suite
apr-qa run tests/ --report report.html

# Quick validation
apr-qa validate model.apr
```

**Integration with Batuta:**
- Stack quality gates for APR model artifacts
- Integration with certeza for CI/CD pipelines
- Works with aprender (training) and realizar (inference)

## Provable Contracts: Formal Verification

Provable Contracts provides a YAML contract → Kani verification pipeline for ML kernels:

- **Contract Parsing:** YAML specifications for kernel pre/post conditions
- **Scaffold Generation:** Automatic Kani harness generation from contracts
- **Probar Integration:** Generate property-based tests from the same contracts
- **Traceability Audit:** Full contract-to-proof audit trail

```yaml
# Example YAML contract for a SIMD kernel
contract:
  name: fused_q4k_matmul
  preconditions:
    - input.len() % 256 == 0
    - output.len() == input.len() / 256 * out_dim
  postconditions:
    - result.is_ok()
    - output values within [-1e6, 1e6]
```

**Integration with Batuta:**
- Quality gates via Kani verification
- Integration with trueno (SIMD kernels) and realizar (Q4K/Q6K kernels)
- Contract-to-probar property test generation

## Tiny Model Ground Truth: Parity Validation

Popperian falsification test suite for model conversion parity:

- **Oracle Generation:** Generate reference outputs from HuggingFace models
- **Parity Checking:** Validate realizar inference matches HuggingFace oracle
- **Quantization Drift:** Measure accuracy loss across format conversions
- **Roundtrip Validation:** Verify GGUF → APR → inference fidelity

```bash
# Generate oracle outputs from HuggingFace
python -m tiny_model_ground_truth generate --model tiny-llama

# Validate realizar inference against oracle
python -m tiny_model_ground_truth validate --oracle outputs/ --engine realizar
```

**Integration with Batuta:**
- Validates realizar and aprender conversion pipelines
- Popperian methodology: attempts to falsify, not just verify
- Part of stack quality gates for model format changes

## Additional Support Tools

### Trueno-RAG (v0.1.0)

Retrieval-Augmented Generation pipeline built on Trueno:

- Vector similarity search
- Document chunking
- Embedding generation

### Trueno-Graph

Graph data structures and algorithms:

- Property graphs
- Traversal operations
- Connected component analysis

### Trueno-DB

Embedded database with Trueno compute:

- Column-store backend
- SQL-like query interface
- ACID transactions

## Tool Ecosystem Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    Batuta (Orchestration)                       │
├─────────────────────────────────────────────────────────────────┤
│  Transpilers          │  Support Tools         │  Data/ML         │
│  ├── Depyler          │  ├── Ruchy             │  ├── Alimentar   │
│  ├── Decy             │  ├── PMAT              │  ├── Aprender    │
│  └── Bashrs           │  ├── APR-QA            │  └── Realizar    │
│                       │  ├── Provable Contracts │                  │
│                       │  ├── Tiny Model GT      │                  │
│                       │  └── Renacer            │                  │
├─────────────────────────────────────────────────────────────────┤
│  Visualization        │  Extensions         │  Registry        │
│  ├── Trueno-Viz       │  ├── Trueno-RAG     │  └── Pacha       │
│  └── Presentar        │  ├── Trueno-Graph   │                  │
│                       │  └── Trueno-DB      │                  │
└─────────────────────────────────────────────────────────────────┘
```

## Further Reading

- [Ruchy: Rust Scripting](./ruchy.md)
- [PMAT: Quality Analysis](./pmat.md)
- [Renacer: Syscall Tracing](./renacer.md)

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Foundation Libraries](./foundation-libs.md)
