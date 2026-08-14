<!-- PCU: introduction | contract: contracts/apr-page-introduction-v1.yaml -->
<!-- Example: cargo run -p aprender-core --example none -->
<!-- Status: enforced -->

# Introduction

Welcome to **The Aprender Guide** — documentation for the Aprender ML framework.

## Quick Start

```bash
cargo install aprender
apr pull qwen2.5-coder-1.5b
apr run qwen2.5-coder-1.5b "What is 2+2?"
```

## What is Aprender?

Aprender is a next-generation ML framework in pure Rust — 78 workspace crates,
105 CLI commands, 1771 provable contracts. Install once, get everything:
inference, training, serving, profiling, model operations.

(Counts are samples, not gates — nothing checks the numbers on this page. Re-derive
with `cargo metadata --no-deps`, `apr --help`, and `find contracts -name '*.yaml'`.)

## This Book

- **Getting Started** — Install, run your first model, fine-tune, serve
- **CLI Reference** — All 105 `apr` commands with `--help` output
- **Architecture** — Monorepo layout, crate map, provable contracts
- **Cookbook** — Recipes and workflows (separate repo: [apr-cookbook](https://github.com/paiml/apr-cookbook))
- **EXTREME TDD** — The methodology used to build aprender (296 chapters)

---

This book also documents the **EXTREME TDD methodology** — the practices used to achieve
80,604 passing lib tests across 78 workspace crates (CI `workspace-test`, run
`31631488466`, `main` @ `d40756541`, 2026-08-12).

## What You'll Learn

This book is your complete guide to implementing EXTREME TDD in production codebases:

- **The RED-GREEN-REFACTOR Cycle**: How to write tests first, implement minimally, and refactor with confidence
- **Advanced Testing Techniques**: Property-based testing, mutation testing, and fuzzing strategies
- **Quality Gates**: Automated enforcement of zero-tolerance quality standards
- **Toyota Way Principles**: Applying Kaizen, Jidoka, and PDCA to software development
- **Real-World Examples**: Actual implementation cycles from building aprender's ML algorithms
- **Anti-Hallucination**: Ensuring every example is test-backed and verified

## Why EXTREME TDD?

Traditional TDD is valuable, but **EXTREME TDD** takes it further:

| Standard TDD | EXTREME TDD |
|--------------|-------------|
| Write tests first | Write tests first (**NO exceptions**) |
| Make tests pass | Make tests pass (**minimally**) |
| Refactor as needed | Refactor **comprehensively** with full test coverage |
| Unit tests | Unit + Integration + **Property-Based + Mutation** tests |
| Some quality checks | **Zero-tolerance quality gates** (all must pass) |
| Code coverage goals | **>90% coverage + 80%+ mutation score** |
| Manual verification | **Automated CI/CD enforcement** |

## The Philosophy

> **"Test EVERYTHING. Trust NOTHING. Verify ALWAYS."**

EXTREME TDD is built on these core principles:

1. **Tests are written FIRST** - Implementation follows tests, never the reverse
2. **Minimal implementation** - Write only the code needed to pass tests
3. **Comprehensive refactoring** - With test safety nets, improve fearlessly
4. **Property-based testing** - Cover edge cases automatically
5. **Mutation testing** - Verify tests actually catch bugs
6. **Zero tolerance** - All tests pass, zero warnings, always

## Real-World Results

This methodology has produced exceptional results in aprender:

- **80,604 passing lib tests** across the workspace (CI `workspace-test`, 2026-08-12)
- **88.78% line coverage** — measured 2026-07-29 by coverage-nightly on `95145584f`.
  The "~97%" this page used to quote predates the coverage pipeline ever working:
  it reported 0/0 until #2333, so no run ever produced that number.
- **Zero clippy warnings** at all times
- **<0.01s test-fast time** for rapid feedback

## How This Book is Organized

### Part 1: Core Methodology
Foundational concepts of EXTREME TDD, the RED-GREEN-REFACTOR cycle, and test-first philosophy.

### Part 2: The Three Phases
Deep dives into RED (failing tests), GREEN (minimal implementation), and REFACTOR (comprehensive improvement).

### Part 3: Advanced Testing
Property-based testing, mutation testing, fuzzing, and benchmarking strategies.

### Part 4: Quality Gates
Automated enforcement through pre-commit hooks, CI/CD, linting, and complexity analysis.

### Part 5: Toyota Way Principles
Kaizen, Genchi Genbutsu, Jidoka, PDCA, and their application to software development.

### Part 6: Real-World Examples
Actual implementation cycles from aprender: Cross-Validation, Random Forest, Serialization, and more.

### Part 7: Sprints and Process
Sprint-based development, issue management, and anti-hallucination enforcement.

### Part 8: Tools and Best Practices
Practical guides to cargo test, clippy, mutants, proptest, and PMAT.

### Part 9: Metrics and Pitfalls
Measuring success and avoiding common TDD mistakes.

## Who This Book is For

- **Software engineers** wanting production-quality TDD practices
- **ML practitioners** building reliable, testable ML systems
- **Teams** adopting Toyota Way principles in software
- **Quality-focused developers** seeking zero-defect methodologies
- **Rust developers** building libraries and frameworks

## Anti-Hallucination Guarantee

Every code example in this book is:
- ✅ **Test-backed** - Validated by actual passing tests in aprender
- ✅ **CI-verified** - Automatically tested in GitHub Actions
- ✅ **Production-proven** - From a real, working codebase
- ✅ **Reproducible** - You can run the same tests and see the same results

**If an example cannot be validated by tests, it will not appear in this book.**

## Getting Started

Ready to master EXTREME TDD? Start with:
1. [What is EXTREME TDD?](./methodology/what-is-extreme-tdd.md) - Core concepts
2. [The RED-GREEN-REFACTOR Cycle](./methodology/red-green-refactor.md) - The fundamental workflow
3. [Case Study: Cross-Validation](./examples/cross-validation.md) - A complete real-world example

Or dive into Development Environment Setup to start practicing immediately.

## Contributing to This Book

This book is open source and accepts contributions. See Contributing to This Book for guidelines.

All book content follows the same EXTREME TDD principles it documents:
- Every example must be test-backed
- All code must compile and run
- Zero tolerance for hallucinated examples
- Continuous improvement through Kaizen

---

**Let's build software with zero defects. Let's master EXTREME TDD.**
