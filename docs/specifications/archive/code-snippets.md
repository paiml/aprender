# Code Snippet Generation Specification v1.0

**Status:** Approved
**Authors:** Sovereign AI Stack Team
**Date:** 2026-02-02
**Refs:** oracle-mode-spec.md, popperian-falsification-checklist.md

---

## Part I: Theoretical Framework

### 1. Motivation (Genchi Genbutsu)

Developers using `batuta oracle` need raw, pipeable Rust code -- not ANSI-decorated
documentation. The existing `--format text|json|markdown` outputs serve human reading
and machine parsing respectively, but none emits code alone. A fourth format, `code`,
closes this gap:

```bash
batuta oracle "train a model" --format code | rustfmt | pbcopy
batuta oracle --recipe training-lora --format code > example.rs
```

The Genchi Genbutsu principle ("go and see") demands that code examples come from
component experts embedded in the knowledge graph, not from templates generated
at display time.

### 2. Toyota Way as Code Generation Epistemology

| Principle | Application |
|-----------|-------------|
| **Jidoka** (Stop-on-error) | `exit(1)` when no code is available. Never emit garbage. |
| **Poka-Yoke** (Error-proofing) | No ANSI escapes in output prevents pipe corruption. |
| **Heijunka** (Level loading) | Normalize all 3 code sources (NL query, recipe, integration) to identical format. |
| **Genchi Genbutsu** (Go and see) | Code examples originate from `Recommender::generate_code_example` and `Recipe.code`, not from display wrappers. |
| **Kaizen** (Continuous improvement) | Track which components lack code examples; expand coverage over time. |

### 3. Literature Foundations

The design draws on three bodies of work:

1. **Sculley et al. (2015)** "Hidden Technical Debt in Machine Learning Systems," NeurIPS.
   Code examples serve as executable documentation that reduces the configuration debt
   identified by Sculley et al. when integrating ML pipelines.

2. **Bacchelli & Bird (2013)** "Expectations, Outcomes, and Challenges of Modern Code
   Review," ICSE. Their finding that reviewers value concrete code examples over abstract
   descriptions motivates emitting compilable snippets.

3. **Gregg & Hazelwood (2011)** "PCIe Transfer Overhead," HPCA. The backend selector
   in `Recommender` uses the 5x PCIe rule from this work; code examples that include
   backend selection demonstrate this principle in context.

---

## Part II: Architecture

### 4. Code Source Taxonomy

The `--format code` output has three possible sources:

| Source | CLI Path | Data Field |
|--------|----------|------------|
| Natural language query | `batuta oracle "query" --format code` | `OracleResponse.code_example` |
| Cookbook recipe | `batuta oracle --recipe ID --format code` | `Recipe.code` |
| Integration pattern | `batuta oracle --integrate "A,B" --format code` | `IntegrationPattern.code_template` |

When the requested source has no code available, the process exits with code 1 and
a stderr diagnostic.

### 5. Output Specification

| Property | Requirement |
|----------|-------------|
| Encoding | UTF-8 |
| ANSI escapes | Prohibited (`\x1b` must never appear) |
| Trailing newline | Single `\n` at end (println! semantics) |
| Exit code (success) | 0 |
| Exit code (no code) | 1 |
| Stderr on failure | Human-readable message suggesting `--format text` |
| Stdout on failure | Empty |

### 6. Component Coverage Matrix

Components with code examples in `Recommender::generate_code_example`:

| Component | Has Example | Source |
|-----------|-------------|--------|
| aprender | Yes | Algorithm-specific path-resolved |
| trueno | Yes | SIMD tensor operations |
| depyler | Yes | Transpilation workflow |
| realizar | Yes | Model registry + serving |
| whisper-apr | Yes | ASR transcription |
| repartir | Yes | Pool + distributed |
| entrenar | No | (via cookbook recipes) |
| alimentar | No | (via cookbook recipes) |
| pacha | No | (via cookbook recipes) |
| batuta | No | Fallback component |

All 33+ cookbook recipes have non-empty `code` fields (enforced by
`test_all_recipes_have_code`).

### 7. Pipeline Design

The output is designed for Unix pipeline composition:

```bash
# Extract, format, copy
batuta oracle --recipe training-lora --format code | rustfmt | pbcopy

# Extract to file
batuta oracle "random forest" --format code > example.rs

# Count lines (no junk)
batuta oracle --recipe ml-serving --format code | wc -l

# Validate UTF-8
batuta oracle --recipe ml-random-forest --format code | iconv -f utf-8 -t utf-8

# Combine multiple recipes
batuta oracle --cookbook --format code > all_recipes.rs
```

---

## Part III: Falsification Protocol (Popper)

### 8. Null Hypotheses

Following Popper's falsificationist epistemology, we define six testable null hypotheses.
Each can be refuted by a single counterexample.

| ID | Null Hypothesis | Falsification Method |
|----|----------------|---------------------|
| H01 | Output contains ANSI escapes | `grep -P '\x1b'` on stdout |
| H02 | A recipe with non-empty code exits != 0 | Test all 33+ recipes with `--format code` |
| H03 | `--list`/`--capabilities`/`--rag` exits 0 with `--format code` | Assert exit code 1 |
| H04 | Output is not valid UTF-8 | `iconv -f utf-8 -t utf-8` check |
| H05 | Code references non-existent APIs | `cargo check` on extracted snippet (future work) |
| H06 | Code snippet count regresses | Track recipe count in CI |

### 9. Falsification Test Mapping

| Hypothesis | Test | Makefile Target |
|-----------|------|-----------------|
| H01 | `test_oracle_format_code_no_ansi` | `make test-format-code` |
| H02 | `test_oracle_format_code_recipe` | `make test-format-code` |
| H03 | `test_oracle_format_code_no_code_exits_1` | `make test-format-code` |
| H04 | Implicit (Rust `String` is UTF-8) | N/A |
| H05 | Future: `make validate-snippets-compile` | Not yet implemented |
| H06 | `test_all_recipes_have_code` | `make validate-snippets` |

---

## Part IV: Best Practices for Sovereign Stack Code

### 10. Code Example Conventions

All code examples in the knowledge graph and cookbook follow these conventions:

1. **Imports first** -- Every example starts with `use` statements
2. **Error handling** -- Use `?` operator; show `Result` return types
3. **Comments** -- Brief comments for non-obvious operations only
4. **No main wrapping** -- Examples show the core logic, not boilerplate `fn main()`
5. **Real types** -- Use actual crate types, not pseudo-code

### 11. Self-Referential Queries

Batuta can query itself about its own patterns:

```bash
# How does the oracle work?
batuta oracle "stack orchestration" --format code

# Integration between components
batuta oracle --integrate "aprender,realizar" --format code
```

### 12. Cross-Component Recipe Composition

The `--cookbook --format code` output concatenates all recipes with delimiter comments:

```
// --- ml-random-forest ---
use aprender::prelude::*;
...

// --- ml-serving ---
use realizar::prelude::*;
...
```

This enables grep-based extraction:

```bash
batuta oracle --cookbook --format code | sed -n '/^\/\/ --- training-lora ---$/,/^\/\/ ---/p'
```

---

## Part V: Future Work

### 13. Compilability Validation

Extract snippets and run `cargo check` against them:

```bash
batuta oracle --recipe ml-random-forest --format code > /tmp/snippet.rs
# Wrap in fn main() { ... }
# cargo check
```

This requires generating compilable wrapper code with correct dependency declarations.

### 14. Extend Coverage to All 27 Components

Current coverage: 6/27 components have direct code examples in the recommender.
Cookbook recipes cover additional components. Target: every component queryable
by name returns a code example.

### 15. `--format code --lang python` for Ground Truth Corpora

Cross-language code output for the Python ground truth corpora:

```bash
batuta oracle --rag "tokenization" --format code --lang python
```

This would emit Python code from `hf-ground-truth-corpus` when relevant.

---

## References

1. Liker, J. K. (2004). *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. McGraw-Hill. ISBN 978-0071392310.
2. Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale Production*. Productivity Press. ISBN 978-0915299140.
3. Gregg, C. & Hazelwood, K. (2011). "Where is the Data? Why You Cannot Debate CPU vs. GPU Performance Without the Answer." *HPCA*.
4. Amdahl, G. M. (1967). "Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities." *AFIPS*. DOI: 10.1145/1465482.1465560.
5. Sculley, D. et al. (2015). "Hidden Technical Debt in Machine Learning Systems." *NeurIPS*.
6. Bacchelli, A. & Bird, C. (2013). "Expectations, Outcomes, and Challenges of Modern Code Review." *ICSE*.
7. Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge.
8. Shingo, S. (1986). *Zero Quality Control: Source Inspection and the Poka-Yoke System*. Productivity Press. ISBN 978-0915299072.
