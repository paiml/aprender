# Code Review: AutoML with Synthetic Data Specification
**Target:** `docs/specifications/automl-with-synthetic-data.md`
**Version:** 1.0.0
**Review Date:** 2025-11-26
**Reviewers:** Gemini Agent (Persona: Toyota Production System Engineer | NASA Systems Engineer | AI Startup CTO)

---

## 1. Executive Summary

This review evaluates the specification against three distinct philosophies:
1.  **The Toyota Way:** Focus on continuous flow, waste elimination (*Muda*), and error-proofing (*Poka-Yoke*).
2.  **NASA Systems Engineering:** Focus on robustness, failure mode analysis, and verification.
3.  **Lean AI Startup:** Focus on velocity, scalability, and return on investment.

**Overall Verdict:** **Approved with Comments**. The specification is well-structured and theoretically grounded. However, specific implementation details regarding "failure boundaries" (NASA) and "feedback loops" (Toyota) need refinement before v0.14.0.

---

## 2. The Toyota Way Assessment

### 2.1. *Muda* (Waste) Reduction
*   **Observation:** The `generate` function creates a batch and *then* filters by `quality_threshold`.
*   **Critique:** Generating expensive samples (e.g., via BackTranslation) only to discard them is *Muda*.
*   **Recommendation:** Implement "Early Stopping" in the generation loop or use lightweight pre-filtering before expensive operations.
    *   *Ref:* **Ratner et al. [7]** suggests heuristic rules (weak supervision) are cheap; use these to filter seeds *before* expensive mutations.

### 2.2. *Jidoka* (Automation with Human Touch)
*   **Observation:** The system is fully automated.
*   **Critique:** If the `quality_score` metric drifts (e.g., the `embedder` is biased), the system will silently produce "garbage".
*   **Recommendation:** Add an `Andon Cord` mechanism. If the rejection rate exceeds X% (e.g., >90% of generated samples are rejected), the pipeline should halt and alert a human.

### 2.3. *Heijunka* (Leveling)
*   **Observation:** `generate_parallel` uses `par_chunks`.
*   **Support:** This aligns well with leveling the workload.
*   **Refinement:** Ensure the `SyntheticCache` (Section 8.1) is thread-safe and doesn't become a bottleneck (contention point).

---

## 3. NASA Systems Engineering Assessment

### 3.1. Failure Mode & Effects Analysis (FMEA)
*   **Risk:** **Model Poisoning**. Synthetic data that passes the `quality_threshold` (high semantic similarity) but contains subtle logic errors (especially in Python-to-Rust translation).
*   **Analysis:** Paper **[10] Chen et al. (Codex)** notes that even powerful models hallucinate subtle bugs. A simple "compiles" check (`rust_parser.compiles`) is insufficient for functional correctness.
*   **Requirement:** Add a "Unit Test Generation" phase or "Execution Sandbox" for the Code Oracle. The generated Rust code must not only compile but *pass tests* derived from the Python source.

### 3.2. Out-of-Distribution (OOD) Anomalies
*   **Risk:** The `DiversityMonitor` checks for collapse but not for "drift" into valid but irrelevant regions.
*   **Mitigation:** Establish "Guardrails" based on **Feng et al. [9]**. Define explicit boundaries for augmentation (e.g., "cannot change variable names to reserved keywords" - though the parser catches this, semantic drift is harder).

### 3.3. Verification & Validation (V&V)
*   **Critique:** The weighting in `quality_score` (0.4 vs 0.2) is arbitrary.
*   **Recommendation:** These weights should not be hardcoded constants. They should be **learnable parameters** or derived from a "Golden Set" evaluation, aligning with **Cubuk et al. [1] (AutoAugment)** which treats augmentation policies as searchable.

---

## 4. New AI Startup Assessment (Lean/Agile)

### 4.1. Velocity & MVP
*   **Critique:** Section 4 (Shell SLM) and Section 5 (Code Oracle) are massive features.
*   **Recommendation:** **Decouple them.** Ship `ShellSyntheticGenerator` first (Release v0.14.0). Push `CodeTranslationGenerator` to v0.15.0. The Code Oracle is an "AI-Complete" problem; Shell Autocomplete is a "Structured Prediction" problem (easier).
*   **Pivot Potential:** The `SyntheticGenerator` trait is the core asset. If the Code Oracle fails, the framework remains valid.

### 4.2. Differentiation
*   **Insight:** Most AutoML tools only do hyperparameter tuning. Adding *Data Engineering* as a hyperparameter is a strong moat.
*   **Support:** **Xie et al. [6] (Self-Training)** shows massive gains from noisy student training. Marketing this as "AutoML that fixes your data" is a winning value proposition.

---

## 5. Peer-Reviewed Annotation Support

The specification's validity is supported by the following mapping of features to literature:

1.  **Searchable Data Augmentation:**
    *   *Feature:* `SyntheticParam` (AugmentationRatio, Strategy).
    *   *Evidence:* **Cubuk et al. [1]** proved that augmentation strategies are domain-dependent and must be learned, not heuristic.

2.  **Text Augmentation Robustness:**
    *   *Feature:* `GenerationStrategy::EDA`.
    *   *Evidence:* **Wei & Zou [2]** demonstrated that simple operations (swap/delete) prevent overfitting in low-resource text tasks (Shell Autocomplete).

3.  **Translation Quality:**
    *   *Feature:* `GenerationStrategy::BackTranslation`.
    *   *Evidence:* **Sennrich et al. [3]** established back-translation as the standard for improving NMT with monolingual data (crucial for the sparse Python-Rust corpus).

4.  **Interpolation Consistency:**
    *   *Feature:* `GenerationStrategy::MixUp`.
    *   *Evidence:* **Zhang et al. [4]** showed that training on convex combinations stabilizes models, critical when the "Shell Command" space is sparse.

5.  **Grammar-Guided Generation:**
    *   *Feature:* `ShellGrammar` / `GenerationStrategy::GrammarBased`.
    *   *Evidence:* **Wang & Yang [5]** successfully used frame-semantic embeddings, validating the approach of using structure (EBNF) to guide synthesis.

6.  **Self-Training Loop:**
    *   *Feature:* `GenerationStrategy::SelfTraining`.
    *   *Evidence:* **Xie et al. [6]** confirms that using the model's own predictions (with noise) scales performance logarithmically with unlabeled data.

7.  **Weak Supervision:**
    *   *Feature:* `GenerationStrategy::WeakSupervision`.
    *   *Evidence:* **Ratner et al. [7] (Snorkel)** validates the use of "labeling functions" (e.g., simple regex checks on shell commands) to create massive training sets cheaply.

8.  **Generative Augmentation:**
    *   *Feature:* `ShellSyntheticGenerator`.
    *   *Evidence:* **Anaby-Tavor et al. [8] (LAMBDA)** showed that generative models (like the one proposed) outperform simple heuristic augmentation for class-conditional generation.

9.  **Taxonomy of Strategies:**
    *   *Feature:* `GenerationStrategy` Enum.
    *   *Evidence:* **Feng et al. [9]** provides the taxonomy (Rule-based vs. Model-based) that this spec correctly implements.

10. **Code Generation Complexity:**
    *   *Feature:* `CodeTranslationGenerator`.
    *   *Evidence:* **Chen et al. [10] (Codex)** provides the baseline difficulty. It suggests that while synthesis is possible, *evaluation* (correctness checking) is the hardest part, supporting the NASA-style critique above.

---

## 6. Action Items

| # | Source | Action | Status |
|---|--------|--------|--------|
| 1 | NASA | Add `sandbox_execution` check to `CodeTranslationGenerator` | ✅ **RESOLVED** (v1.1.0) |
| 2 | Toyota | Implement `Andon` alert for high rejection rates in `generate()` | ✅ **RESOLVED** (v1.1.0) |
| 3 | Startup | Mark `CodeTranslationGenerator` as "Experimental" in v0.14.0 | ✅ **RESOLVED** (v1.1.0) |

### Resolution Summary (2025-11-26)

**[NASA]** Added `SandboxExecutor` field to `CodeTranslationGenerator`. Updated `quality_score()` to:
- Generate unit tests from Python behavior
- Execute tests against generated Rust code
- Weight functional correctness at 40% (primary signal)

**[Toyota]** Added `AndonHandler` trait and `DefaultAndon` implementation:
- Halts pipeline if rejection rate >90%
- Alerts on quality drift below baseline
- Configurable via `SyntheticConfig.andon_*` fields

**[Startup]** Decoupled roadmap:
- Shell SLM: v0.14.0 (MVP, tractable structured prediction)
- Code Oracle: v0.15.0 (experimental, AI-Complete problem)
- Added EXPERIMENTAL warning to `CodeTranslationGenerator` docstring
