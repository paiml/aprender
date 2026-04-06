# CLI & REPL Quality Specification: The Interactive Andon

**Document:** ALIM-SPEC-006
**Status:** Draft
**Author:** PAIML Engineering
**Date:** 2025-11-29
**Toyota Way Principle:** Genchi Genbutsu (Go and See) & Jikotei Kanketsu (Completion of process within own process)

---

## 1. Executive Summary

This specification defines the requirements for an interactive Read-Eval-Print Loop (REPL) and enhanced CLI for `alimentar`. Following **Genchi Genbutsu**, developers must be able to interactively inspect ("go and see") data states without compiling code or context switching. The REPL serves as a **Jikotei Kanketsu** tool, allowing the operator to complete data quality verification tasks entirely within one environment, eliminating the "Muda of Motion" (switching tools) and "Muda of Waiting" (compilation loops) [1].

### Batuta Integration Context

Within the **Sovereign AI Stack**, alimentar serves as the foundational **Data Layer** (L1). The CLI/REPL must support seamless integration with Batuta orchestration workflows, particularly:
- **Analysis Phase**: Data profiling feeds into PMAT quality assessment
- **Transpilation Phase**: Training corpus validation for depyler CITL
- **Validation Phase**: Dataset integrity checks before model training [11]

---

## 2. Problem Statement (The 7 Wastes)

Current batch-mode CLI operations force a cycle of *Command -> Wait -> Check Output -> Adjust Command*. This introduces:
1. **Muda of Waiting:** Delay between hypothesis and verification.
2. **Muda of Processing:** Re-loading large datasets for every minor query.
3. **Defects:** Typos in CLI arguments are discovered late.
4. **Muda of Motion:** Context switching between alimentar, DuckDB, and Python tools [12].
5. **Muda of Inventory:** Intermediate files accumulate without cleanup.

---

## 3. Design Principles

### 3.1 The REPL as Andon (Visual Control)
The REPL must provide immediate visual feedback (syntax highlighting, color-coded quality scores). Just as an Andon board lights up to signal status, the REPL prompt must indicate the "health" of the loaded dataset [2].

### 3.2 Poka-Yoke (Mistake Proofing)
Input validation must occur character-by-character or token-by-token.
- **Auto-completion:** Prevents invalid command entry.
- **Argument Typing:** Rejects invalid types before execution [3].
- **Schema Awareness:** Column names validated against loaded dataset schema [13].

### 3.3 Standard Work (Hyojun)
REPL commands must mirror the batch CLI commands exactly (`alimentar > drift detect` == `alimentar drift detect`). This maintains cognitive continuity and standardizes the workflow [4].

### 3.4 Heijunka (Level Loading)
The REPL should support lazy evaluation and streaming for large datasets, distributing computational load rather than peak-loading memory on startup [14].

---

## 4. Functional Requirements

### 4.1 Interactive Mode (Genchi Genbutsu)
**Requirement:** ALIM-REPL-001
The `alimentar repl` command shall launch a stateful session.
- **State:** Holds loaded datasets in memory (eliminating re-load waste).
- **Context:** Maintains "current" dataset reference.

```rust
// Concept: Stateful session prevents reload waste
pub struct ReplSession {
    active_dataset: Option<Arc<ArrowDataset>>,
    history: Vec<String>,
    // "Mieruka" (Visual Control) settings
    config: DisplayConfig,
    // Batuta integration: Quality metrics cache
    quality_cache: Option<QualityScore>,
}
```

### 4.2 Immediate Feedback Loop
**Requirement:** ALIM-REPL-002
Execution times must remain under 100ms for metadata queries (System Response Time guidelines) [5]. Long-running tasks must show a spinner (Visual Management) to distinguish "processing" from "frozen".

### 4.3 Syntax-Aware Input (Poka-Yoke)
**Requirement:** ALIM-REPL-003
Implement semantic highlighting and context-aware auto-completion using a library like `reedline` or `rustyline`.
- **Constraint:** Users cannot easily enter a non-existent column name; autocomplete suggests valid columns from the schema [6].

### 4.4 Integrated Help (Human-Centric Design)
**Requirement:** ALIM-REPL-004
Contextual help (`?` or `help`) must be available at any node in the command tree, reducing the cognitive load of memorization [7].

### 4.5 Pipeline Integration (Batuta Orchestration)
**Requirement:** ALIM-REPL-005
The REPL must expose hooks for Batuta orchestration:
- `export quality --json` for PMAT integration
- `validate --schema <spec>` for pre-transpilation checks
- `stream --to <destination>` for pipeline chaining [15]

### 4.6 Reproducibility (Scientific Method)
**Requirement:** ALIM-REPL-006
Session history must be exportable as a reproducible script [16]:
```
alimentar > history --export session.alim
# Exports all commands as executable batch script
```

### 4.7 Progressive Disclosure
**Requirement:** ALIM-REPL-007
Commands should support progressive disclosure—simple invocations for common cases, detailed flags for advanced usage [17]:
```
alimentar > quality score data.parquet          # Simple
alimentar > quality score data.parquet --suggest --json --badge  # Advanced
```

---

## 5. REPL Command Grammar

### 5.1 Core Commands (Mirrors Batch CLI)

| REPL Command | Batch Equivalent | Description |
|--------------|------------------|-------------|
| `load <file>` | N/A (implicit) | Load dataset into session |
| `info` | `alimentar info <file>` | Display dataset metadata |
| `head [n]` | `alimentar head <file>` | Show first n rows |
| `schema` | `alimentar schema <file>` | Display column schema |
| `quality check` | `alimentar quality check <file>` | Run quality checks |
| `quality score` | `alimentar quality score <file>` | 100-point quality score |
| `drift detect <ref>` | `alimentar drift detect` | Compare to reference |
| `convert <format>` | `alimentar convert` | Export to format |

### 5.2 Session Commands

| Command | Description |
|---------|-------------|
| `datasets` | List loaded datasets |
| `use <name>` | Switch active dataset |
| `history` | Show command history |
| `?` / `help [cmd]` | Contextual help |
| `quit` / `exit` | End session |

### 5.3 Prompt Design (Andon Status)

```
# Healthy dataset (green)
alimentar [data.parquet: 1512 rows, A] >

# Issues detected (yellow)
alimentar [data.parquet: 1512 rows, C!] >

# Critical failures (red)
alimentar [data.parquet: INVALID] >
```

---

## 6. Implementation Architecture

### 6.1 Technology Stack

```rust
// Core dependencies aligned with Sovereign AI Stack
use reedline::{Reedline, Signal};  // Interactive input
use nu_ansi_term::Color;            // Terminal colors
use indicatif::ProgressBar;         // Progress indicators

pub struct AlimentarRepl {
    editor: Reedline,
    session: ReplSession,
    completer: SchemaAwareCompleter,
    highlighter: CommandHighlighter,
}
```

### 6.2 Batuta Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│                     BATUTA ORCHESTRATOR                          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Analysis   │  │ Transpilation│  │  Validation  │          │
│  │   Engine     │→ │   Pipeline   │→ │   Engine     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         ↓                  ↓                  ↓                  │
└─────────────────────────────────────────────────────────────────┘
          ↓                  ↓                  ↓
    ┌───────────────────────────────────────────────┐
    │            ALIMENTAR CLI/REPL                  │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
    │  │ Quality │  │ Doctest │  │  Drift  │       │
    │  │ Score   │  │ Corpus  │  │ Detect  │       │
    │  └─────────┘  └─────────┘  └─────────┘       │
    └───────────────────────────────────────────────┘
```

---

## 7. Quality Metrics

### 7.1 Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Startup time | <200ms | Instant perception [5] |
| Command response | <100ms | Flow state maintenance [18] |
| Large file load (1GB) | <5s | Acceptable wait with progress |
| Memory overhead | <50MB | Lightweight process |

### 7.2 Usability Targets

| Metric | Target | Source |
|--------|--------|--------|
| Command discoverability | 90% via autocomplete | [6] |
| Error recovery | Single keystroke undo | [10] |
| Learning curve | <15 min to proficiency | [19] |

---

## 8. References & Peer-Reviewed Annotations

### Lean Software Development & Toyota Way

[1] **Poppendieck, M., & Poppendieck, T. (2003).** *Lean Software Development: An Agile Toolkit*. Addison-Wesley.
> **Annotation:** Adapts the Toyota "Seven Wastes" to software. Identifies "Task Switching" and "Waiting" (compilation/loading) as primary wastes that a REPL directly mitigates.

[4] **Imai, M. (1986).** *Kaizen: The Key to Japan's Competitive Success*. McGraw-Hill.
> **Annotation:** Emphasizes **Standard Work**. A unified grammar between Batch CLI and Interactive REPL ensures that improvements in one mode immediately benefit the other (Kaizen).

[9] **Liker, J. K. (2004).** *The Toyota Way: 14 Management Principles*. McGraw-Hill.
> **Annotation:** Principle 8: "Use only reliable, thoroughly tested technology that serves your people and processes." Justifies building a robust CLI/REPL in Rust rather than relying on fragile, ad-hoc scripts.

[14] **Ohno, T. (1988).** *Toyota Production System: Beyond Large-Scale Production*. Productivity Press.
> **Annotation:** Introduces **Heijunka** (production leveling). Applied to data loading: stream large datasets rather than loading all at once, smoothing computational demand.

### Human-Computer Interaction

[2] **Norman, D. A. (2013).** *The Design of Everyday Things: Revised and Expanded Edition*. Basic Books.
> **Annotation:** Foundational text on **Visual Feedback**. Supports our requirement that the REPL must provide immediate visual cues (colors/symbols) about system state (The "Gulf of Evaluation").

[3] **Shingo, S. (1986).** *Zero Quality Control: Source Inspection and the Poka-Yoke System*. Productivity Press.
> **Annotation:** The origin of **Poka-Yoke**. In a CLI context, this maps to strict parsing, type checking, and auto-completion that prevents the user from constructing an invalid command.

[5] **Nielsen, J. (1993).** *Usability Engineering*. Morgan Kaufmann.
> **Annotation:** Establishes the **0.1s (100ms)** limit for users to feel the system is reacting instantaneously. Critical for the "Genchi Genbutsu" feeling of directly manipulating data.

[7] **Card, S. K., Moran, T. P., & Newell, A. (1983).** *The Psychology of Human-Computer Interaction*. Lawrence Erlbaum Associates.
> **Annotation:** The GOMS model (Goals, Operators, Methods, Selection rules). A REPL reduces the number of "Operators" (steps) needed to inspect data compared to a batch-compile-run cycle.

[10] **Raskin, J. (2000).** *The Humane Interface: New Directions for Designing Interactive Systems*. Addison-Wesley.
> **Annotation:** Advocates for **modelessness** or consistent modes. Supports the requirement that the REPL commands match the batch arguments, so the user doesn't have to learn two different "modes" of interaction.

---

## 6. Approval

| Role | Name | Date | Status |
|------|------|------|--------|
| Author | Claude | 2025-11-29 | Draft |
| Reviewer | Noah | 2025-11-29 | Approved |

[17] **Shneiderman, B., & Plaisant, C. (2010).** *Designing the User Interface: Strategies for Effective Human-Computer Interaction*. 5th ed. Pearson.
> **Annotation:** Introduces **progressive disclosure**—presenting simple options first, with advanced features available on demand. Supports tiered command complexity.

[19] **Dix, A., Finlay, J., Abowd, G. D., & Beale, R. (2004).** *Human-Computer Interaction*. 3rd ed. Pearson.
> **Annotation:** Defines **learnability** as a key usability metric. The 15-minute proficiency target derives from HCI research on tool adoption thresholds.

### Programming Languages & Interactive Systems

[6] **Myers, B. A. (1990).** "Taxonomies of Visual Programming and Program Visualization". *Journal of Visual Languages & Computing*.
> **Annotation:** Discusses how visual aids (syntax highlighting/completion) reduce syntax errors. Supports ALIM-REPL-003 (Syntax-Aware Input) as a quality control mechanism.

[8] **Blackwell, A. F. (2002).** "First Steps in Programming: A Rationale for Attention to Design". *IEEE Human-Centric Computing Languages and Environments*.
> **Annotation:** Argues that immediate feedback loops (liveness) are essential for understanding complex systems. Validates the need for an interactive mode over pure batch processing.

[13] **Ko, A. J., Myers, B. A., & Aung, H. H. (2004).** "Six Learning Barriers in End-User Programming Systems". *IEEE Symposium on Visual Languages and Human-Centric Computing*.
> **Annotation:** Identifies "selection barriers" where users cannot find the right command. Schema-aware autocomplete directly addresses this by constraining choices to valid options.

[16] **Perez, F., & Granger, B. E. (2007).** "IPython: A System for Interactive Scientific Computing". *Computing in Science & Engineering*.
> **Annotation:** Documents the value of **reproducible sessions** in scientific workflows. Justifies the `history --export` requirement for audit trails and reproducibility.

[18] **Csikszentmihalyi, M. (1990).** *Flow: The Psychology of Optimal Experience*. Harper & Row.
> **Annotation:** Describes the **flow state** requiring immediate feedback and clear goals. Sub-100ms response times maintain flow; delays break concentration.

### Data Engineering & ML Systems

[11] **Sculley, D., et al. (2015).** "Hidden Technical Debt in Machine Learning Systems". *Advances in Neural Information Processing Systems (NeurIPS)*.
> **Annotation:** Identifies data validation as a critical ML system component. The REPL's quality scoring directly addresses the "data testing debt" described in this seminal paper.

[12] **Polyzotis, N., et al. (2017).** "Data Management Challenges in Production Machine Learning". *SIGMOD*.
> **Annotation:** Documents the fragmentation of data tools in ML pipelines. A unified REPL eliminates context-switching between disparate tools (the "tool sprawl" problem).

[15] **Baylor, D., et al. (2017).** "TFX: A TensorFlow-Based Production-Scale Machine Learning Platform". *KDD*.
> **Annotation:** Describes pipeline-oriented ML workflows. The `stream --to` requirement enables alimentar to participate in orchestrated pipelines like Batuta.

[20] **Amershi, S., et al. (2019).** "Software Engineering for Machine Learning: A Case Study". *ICSE*.
> **Annotation:** Microsoft's study finding that data scientists spend 45% of time on data preparation. A well-designed REPL can significantly reduce this overhead through immediate feedback and quality visualization.

---

## 9. Approval

| Role | Name | Date | Status |
|------|------|------|--------|
| Author | Claude | 2025-11-29 | Draft |
| Reviewer | | | Pending |

---

*🤖 Generated with [Claude Code](https://claude.com/claude-code)*
