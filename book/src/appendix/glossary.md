# Glossary

Essential terms and concepts used throughout the Batuta framework.

## Core Concepts

| Term | Definition |
|------|------------|
| **Batuta** | Orchestration framework for the Sovereign AI Stack. From Spanish "baton" - the conductor's wand. |
| **Sovereign AI Stack** | 20-component pure Rust ML infrastructure for privacy-preserving AI. |
| **Toyota Way** | Lean manufacturing principles (Jidoka, Kaizen, Muda, etc.) applied to software. |

## Toyota Way Principles

| Principle | Japanese | Meaning |
|-----------|----------|---------|
| **Jidoka** | 自働化 | Built-in quality: stop-the-line on defects |
| **Kaizen** | 改善 | Continuous improvement |
| **Muda** | 無駄 | Waste elimination |
| **Heijunka** | 平準化 | Level scheduling |
| **Kanban** | 看板 | Visual workflow management |
| **Andon** | 行灯 | Problem visualization (red/yellow/green) |
| **Mieruka** | 見える化 | Visual control dashboards |
| **Genchi Genbutsu** | 現地現物 | Go and see for yourself |

## Stack Components

| Component | Layer | Description |
|-----------|-------|-------------|
| **Trueno** | Compute | SIMD/GPU tensor primitives |
| **Aprender** | ML | First-principles ML algorithms |
| **Realizar** | Inference | LLM inference runtime |
| **Depyler** | Transpiler | Python to Rust conversion |
| **Batuta** | Orchestration | Workflow coordination |
| **Certeza** | Quality | Validation framework |
| **PMAT** | Quality | Code quality metrics |

## Quality Metrics

| Term | Definition |
|------|------------|
| **Demo Score** | PMAT quality metric (0-100 scale) |
| **TDG** | Technical Debt Grade |
| **Quality Gate** | A- (85) minimum for production |
| **Coverage** | Test code coverage percentage |
| **Mutation Score** | Mutation testing kill rate |

## Transpilation Terms

| Term | Definition |
|------|------------|
| **AST** | Abstract Syntax Tree |
| **HIR** | High-level Intermediate Representation |
| **MIR** | Mid-level Intermediate Representation |
| **FFI** | Foreign Function Interface |
| **Zero-copy** | Memory operations without data copying |

---

**Navigate:** [Table of Contents](../SUMMARY.md)
