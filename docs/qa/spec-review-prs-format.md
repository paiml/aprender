# Critical Review: Presentar Scene Format (.prs) Specification

**Reviewer**: Gemini Agent (Codebase Investigator)
**Date**: 2025-12-06
**Target**: `docs/specifications/sharing-format-file-type.md`
**Philosophy**: Toyota Way (TPS) & Sovereign AI Stack Principles

## 1. Executive Summary

The proposed `.prs` format represents a strong shift towards **Jidoka** (built-in quality) and **Standardization** by decoupling visualization logic from runtime implementation. By moving away from "Python-as-config" (Gradio/Streamlit) to a declarative schema, it eliminates the **Muda** (waste) of heavy runtime dependencies and enables true edge/WASM portability.

However, the "flat widget list" architecture introduces potential **Cognitive Load** (a form of overburden or *Muri* for the developer). This review proposes refinements to balance machine-readability with developer ergonomics, supported by 10 peer-reviewed citations.

## 2. Toyota Way Analysis

### 2.1 Muri (Overburden) in Layout Definition
**Observation**: The spec separates `layout` (structure) from `widgets` (content).
```yaml
layout:
  type: grid
widgets:
  - id: chart
    position: { row: 1, col: 0 }
```
**Critique**: This forces the developer to mentally map `row: 1, col: 0` back to the grid definition. In complex dashboards, this "split attention" effect increases the risk of defects (*Muda* of correction).
**Recommendation**: Allow hierarchical nesting for simple layouts (Flex/Column) while keeping the flat ID-based reference for advanced grids. This aligns with **Poka-Yoke** (error-proofing) by making the structure visible.

### 2.2 Jidoka (Automation) in Expressions
**Observation**: Jinja-like expressions `{{ inference.model | select('val') }}` are string-typed.
**Critique**: Stringly-typed logic is a major source of runtime errors.
**Recommendation**: Enforce a strict subset of expression logic that can be statically analyzed (linted) before runtime. The spec mentions schema validation; it should explicitly require expression validation (e.g., checking if `inference.model` actually has a `val` field based on the referenced `.apr` model card).

### 2.3 Heijunka (Level Loading) via Resources
**Observation**: Resources are declared upfront with hashes.
**Critique**: This is excellent. It allows the runtime to pre-fetch/cache assets (leveling the load) rather than spiking network usage when a user clicks a button.
**Validation**: Supports the "offline-first" capability required by the Sovereign AI Stack.

## 3. Technical Critique

### 3.1 Long-Term Success & Extensibility
The use of `prs_version` and SemVer is standard. However, the spec should explicitly define **forward compatibility rules**.
*   *Question*: What happens if a v1.0 runtime encounters a v1.1 widget?
*   *Proposal*: Add a `fallback` field to widgets, allowing a newer `.prs` to define "If `3d_scatter` is not available, render `2d_scatter`".

### 3.2 Interoperability with Stack (.apr/.ald)
The `resources` section effectively links to `.apr` and `.ald` files.
*   **Improvement**: The spec should mandate that the `hash` field in `.prs` matches the *integrity* of the `.apr` file. This creates a "Chain of Custody" from data collection (`.ald`) -> model training (`.apr`) -> visualization (`.prs`), essential for AI safety.

### 3.3 Simplicity vs. Power
The flat list is simpler for diffs (Git) but harder to read.
*   **Verdict**: Stick to the flat list for the *Canonical* format (machine-optimized), but strictly require the `id` field to be semantic (e.g., `sentiment_chart`, not `widget_1`) to aid human review.

## 4. Supporting Citations

1.  **Heer, J., & Bostock, M. (2010). Declarative language design for interactive visualization.** *IEEE Transactions on Visualization and Computer Graphics.*
    *   *Relevance*: Validates the core premise that separating specification from execution enables optimization and portability.
2.  **Fowler, M. (2010). Domain-Specific Languages.** *Addison-Wesley Professional.*
    *   *Relevance*: Supports the creation of the `.prs` DSL as a way to handle the specific complexity of UI layout without general-purpose programming overhead.
3.  **Womack, J. P., Jones, D. T., & Roos, D. (1990). The Machine That Changed the World.** *Rawson Associates.*
    *   *Relevance*: Foundational text on Lean/TPS. Justifies the removal of "Muda" (runtime overhead) in the `.prs` design.
4.  **Satyanarayan, A., et al. (2017). Vega-Lite: A Grammar of Interactive Graphics.** *IEEE TVCG.*
    *   *Relevance*: Cited in spec, but critical here to support the "Interaction Bindings" section as the gold standard for reactive grammars.
5.  **Hunt, A., & Thomas, D. (1999). The Pragmatic Programmer.** *Addison-Wesley.*
    *   *Relevance*: "DRY" (Don't Repeat Yourself) and "Orthogonality". The `.prs` spec's separation of data (`resources`) from view (`widgets`) follows this strictly.
6.  **Fielding, R. T. (2000). Architectural Styles and the Design of Network-based Software Architectures.** *University of California, Irvine.*
    *   *Relevance*: The RESTful/Resource-oriented approach of `pacha://` URIs in the spec aligns with web-scale architectural best practices.
7.  **Peng, R. D. (2011). Reproducible research in computational science.** *Science, 334(6060).*
    *   *Relevance*: Strong support for the `hash` (BLAKE3) pinning. Without exact versioning of data/models, visualization is not reproducible science.
8.  **Norman, D. A. (2013). The Design of Everyday Things.** *Basic Books.*
    *   *Relevance*: Highlights the importance of "Mapping". The critique regarding Layout/Widget separation addresses the mapping problem for the developer.
9.  **Spinellis, D. (2012). Version Control Systems.** *IEEE Software.*
    *   *Relevance*: Supports the text-based, diff-friendly YAML format over binary blobs, enabling better collaboration.
10. **Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). Design Patterns: Elements of Reusable Object-Oriented Software.** *Addison-Wesley.*
    *   *Relevance*: The "Observer" pattern is the theoretical basis for the `bindings` section (Event -> Action), ensuring decoupled interactivity.

## 5. Conclusion & Action Items

The `.prs` format is architecturally sound and aligns with the Sovereign AI Stack's goals.

**Action Items:**
1.  **Refine**: Update spec to explicitly mention "Static Analysis" for expressions (Jidoka).
2.  **Add**: Add a `fallback` mechanism for widgets (Long-term robustness).
3.  **Approve**: The move to BLAKE3 hashing and WASM-first design is approved.

This review confirms the specification is ready for the "Prototyping" phase, pending the minor adjustments above.
