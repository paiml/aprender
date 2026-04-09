# Eiffel DbC — Falsification & References

Version: 1.0
Status: proposed
Date: 2026-04-09

*See also: [eiffel-dbc.md](eiffel-dbc.md)*

## 12. Falsification

Every claim in this spec must be falsifiable. The following tests can
refute the spec's core hypotheses. They follow the project's standard
Popperian pattern: prediction → test → if_fails.

### 12.1. Hypothesis: DbC Types Add Verification Power

**H1: Precondition/postcondition obligation types produce Kani
harnesses that catch bugs undetectable by property-only types.**

```yaml
- id: FALSIFY-DBC-001
  rule: "Pre/post catches what invariant misses"
  prediction: >
    A kernel with a precondition obligation (input finite, non-empty)
    produces a Kani harness with kani::assume(pre) that catches an
    empty-slice panic unreachable via invariant-only harnesses
  test: >
    Generate Kani harnesses for softmax-kernel-v1 with and without
    precondition obligations. Introduce a deliberate empty-slice bug.
    Verify the precondition harness catches it, invariant-only does not.
  if_fails: >
    Precondition obligation type adds no verification power beyond
    existing invariant type — remove from spec
```

**H2: Frame obligations detect state corruption undetectable by
conservation obligations.**

```yaml
- id: FALSIFY-DBC-002
  rule: "Frame detects what conservation misses"
  prediction: >
    A frame obligation (modifies output only) on kv-cache-equivalence-v1
    produces a probar test that catches input buffer corruption, while
    conservation (sum preserved) does not detect it
  test: >
    Generate tests for kv-cache with frame vs conservation-only
    obligations. Introduce a bug that corrupts input but preserves
    sum. Verify frame test catches it, conservation does not.
  if_fails: >
    Frame obligation adds no detection power beyond conservation —
    merge into conservation or remove
```

**H3: Loop invariant/variant obligations produce stronger termination
proofs than bare termination type.**

```yaml
- id: FALSIFY-DBC-003
  rule: "Loop variant proves termination with witness"
  prediction: >
    A loop_variant obligation on adamw-kernel-v1 produces a Kani
    harness that verifies decreasing iteration count, while a bare
    termination obligation produces only an assertion that the loop
    exits (no witness)
  test: >
    Generate Kani harnesses for adamw with loop_variant vs termination.
    Introduce a bug where the loop counter wraps around (non-decreasing).
    Verify loop_variant catches it, termination does not.
  if_fails: >
    Loop variant witness provides no advantage over bare termination
    assertion — simplify to termination only
```

### 12.2. Hypothesis: Two-Layer Pre/Post Model Is Necessary

**H4: Obligation-level pre/postconditions provide value beyond
equation-level debug_assert pre/postconditions.**

```yaml
- id: FALSIFY-DBC-004
  rule: "Formal predicates vs Rust expressions"
  prediction: >
    At least one obligation-level precondition (formal predicate)
    can be verified by Kani but cannot be expressed as a valid
    Rust debug_assert! expression in the equation-level field
  test: >
    Attempt to express all proposed precondition formal predicates
    as valid Rust expressions. Count how many require quantifiers
    (∀, ∃), mathematical notation, or cross-equation references
    that have no Rust equivalent.
  if_fails: >
    Every formal predicate is expressible as Rust debug_assert —
    obligation-level pre/post is redundant, equation-level suffices
```

### 12.3. Hypothesis: DbC Types Apply to Non-Kernel Domains

**H5: The frame obligation type is useful for at least 3 non-kernel
stack projects (simular, forjar, probar).**

```yaml
- id: FALSIFY-DBC-005
  rule: "Frame applicability across domains"
  prediction: >
    Writing a frame obligation for each of simular (integration step
    preserves masses), forjar (resource apply preserves other resources),
    and probar (visual regression preserves reference image) produces
    meaningful probar tests that detect real mutation bugs
  test: >
    Write 3 frame contracts (one per project). Run mutation testing
    (pmat mutate). Verify each frame test kills at least 1 mutant
    that no existing test catches.
  if_fails: >
    Frame obligations in non-kernel domains are tautological — they
    catch no bugs beyond what existing tests already cover
```

**H6: The subcontract obligation type detects Liskov violations in
the stack's transport/plugin abstractions.**

```yaml
- id: FALSIFY-DBC-006
  rule: "Subcontract detects substitution bugs"
  prediction: >
    A subcontract obligation (pepita refines SSH transport in forjar)
    produces a test that catches a pepita-only bug where the precondition
    is accidentally strengthened (rejecting scripts SSH accepts)
  test: >
    Write subcontract obligation for forjar pepita→SSH. Generate
    cross-contract test. Introduce a pepita bug that rejects valid
    SSH scripts. Verify the subcontract test catches it.
  if_fails: >
    Subcontract obligations are documentation-only — they produce
    no tests that detect real behavioral subtyping violations
```

### 12.4. Hypothesis: The Heat Map Predicts Adoption Value

**H7: Domains rated "High" for a DbC type benefit measurably more
from that type than domains rated "Low" or "Medium".**

```yaml
- id: FALSIFY-DBC-007
  rule: "Heat map predicts bug detection"
  prediction: >
    After implementing DbC types across 3+ projects, the number of
    unique bugs caught per obligation (bugs / contract) is at least
    2× higher in domains rated "High" vs domains rated "Low/Medium"
    for that obligation type
  test: >
    Track bugs caught by each new DbC obligation type across projects.
    After 6 months of adoption, compute bugs/contract ratio per
    domain-type pair. Compare High vs Low/Medium groups.
  if_fails: >
    The heat map does not predict adoption value — obligation type
    utility is domain-independent, and the per-domain recommendations
    should be removed
```

---

## 13. References

### Design by Contract Foundations

1. Meyer, B. (1988). *Object-Oriented Software Construction.* Prentice Hall.
2. Meyer, B. (1992). "Applying Design by Contract." *IEEE Computer* 25(10).
3. Meyer, B. (1997). *Object-Oriented Software Construction.* 2nd ed.
   Prentice Hall. Ch. 11 (DbC), Ch. 16 (Inheritance and contracts),
   Ch. 25 (GUI contracts via EiffelVision), Ch. 30 (Concurrency).
4. Meyer, B. (2009). *Touch of Class: Learning to Program Well.* Springer.
5. Meyer, B. (2022). "The Dependent Delegate Dilemma." *CACM* 65(4).
6. Liskov, B. & Wing, J. (1994). "A Behavioral Notion of Subtyping." *ACM TOPLAS* 16(6).
7. Findler, R.B. & Felleisen, M. (2002). "Contracts for Higher-Order Functions." *ICFP 2002.*
8. Hoare, C.A.R. (1969). "An Axiomatic Basis for Computer Programming." *CACM* 12(10).
9. Parnas, D.L. (1972). "On the Criteria to Be Used in Decomposing Systems into Modules." *CACM* 15(12).

### Domain-Specific Contracts

10. Barnett, M. et al. (2004). "Spec#: A Language for API Contracts." *CASSIS 2004.*
11. Meyer, B. (2003). "The Grand Challenge of Trusted Components." *ICSE 2003.*
12. Dagstuhl Seminar 26031 (2026). "Software Contracts Meet System Contracts."

### Rust Verification and Bounded Model Checking

13. Le Blanc, A. & Lam, P. (2024). "Surveying the Rust Verification Landscape." arXiv:2410.01981.
14. Lattuada, A. et al. (2023). "Verus: Verifying Rust Programs using Linear Ghost Types." arXiv:2303.05491.
15. Ayoun, S.-E. et al. (2024). "A Hybrid Approach to Semi-automated Rust Verification." arXiv:2403.15122.
16. Le Blanc, A. & Lam, P. (2025). "Lessons Learned from Verifying the Rust Standard Library." arXiv:2510.01072.
17. Kroening, D. et al. (2023). "CBMC: The C Bounded Model Checker." arXiv:2302.02384.
18. Amusuo, P.C. et al. (2025). "Do Unit Proofs Work? Compositional Bounded Model Checking." arXiv:2503.13762.

### Frame Conditions and Separation Logic

19. Eilers, M. et al. (2024). "Verification Algorithms for Automated Separation Logic Verifiers." arXiv:2405.10661.
20. Jacobs, B. (2025). "VeriFast's Separation Logic." arXiv:2505.04500.
21. Fasse, J. & Jacobs, B. (2022). "Modular Termination Verification with Higher-Order Concurrent Separation Logic." arXiv:2212.14126.

### Loop Invariants and Termination

22. Sarita, Y. et al. (2024). "Syndicate: Efficient Ranking Function-Based Termination Analysis." arXiv:2404.05951.
23. Liu, R. et al. (2024). "Enhancing Automated Loop Invariant Generation with Large Language Models." arXiv:2412.10483.
24. Liu, C. et al. (2023). "LIG-MM: Towards General Loop Invariant Generation." arXiv:2311.10483.
25. Akhond, M.R. et al. (2025). "LLM For Loop Invariant Generation: How Far Are We?" arXiv:2511.06552.

### Behavioral Subtyping and Refinement

26. Haehnle, R. et al. (2023). "Context-aware Trace Contracts." arXiv:2310.04384.
27. Dominguez, F. & Spiwack, A. (2025). "Refinement-Types Driven Development." arXiv:2509.15005.

### Contracts for ML and Infrastructure

28. Wong, S. et al. (2023). "MLGuard: Defend Your Machine Learning Model!" arXiv:2309.01379.
29. Jakeman, J.D. et al. (2025). "V&V for Trustworthy Scientific Machine Learning." arXiv:2502.15496.
30. Chiari, M. et al. (2022). "Static Analysis of Infrastructure as Code: A Survey." arXiv:2206.10344.
31. Jana, P. et al. (2026). "TerraFormer: Automated IaC with LLMs via Policy-Guided Verifier Feedback." arXiv:2601.08734.

### Pre/Postcondition Inference and LLM-Assisted Verification

32. Richter, C. & Wehrheim, H. (2025). "Beyond Postconditions: Can LLMs Infer Formal Contracts?" arXiv:2510.12702.
33. Faria, J.P. et al. (2026). "Automatic Generation of Formal Specification Using LLMs and Test Oracles." arXiv:2601.12845.
34. Wen, C. et al. (2024). "Enchanting Program Specification Synthesis by LLMs." arXiv:2404.00762.
35. Yang, A.Z.H. et al. (2024). "VERT: Verified Equivalent Rust Transpilation with LLMs." arXiv:2404.18852.
36. Liu, Y. et al. (2024). "PropertyGPT: LLM-driven Formal Verification." arXiv:2405.02580.
37. Lim, S. et al. (2025). "ContractEval: Evaluating Contract-Satisfying Assertions." arXiv:2510.12047.
38. Councilman, A. et al. (2025). "Towards Formal Verification of LLM-Generated Code." arXiv:2507.13290.

### Internal Cross-References

- [escape-proof-enforcement.md](escape-proof-enforcement.md) — Six-stage
  compile-time enforcement pipeline (equation pre/post → build.rs →
  `#[contract]` macro → test execution)
- [lean-kani-composition.md](lean-kani-composition.md) — How Lean (ℝ)
  and Kani (f32) compose via `stub_float` bridge
- [pytorch-extraction.md](pytorch-extraction.md) — `pv extract-pytorch`
  infers pre/postconditions from PyTorch docstrings
