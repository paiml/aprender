# Migration Strategy

A successful migration from Python, C, or Shell to Rust follows a disciplined cycle: **Assess, Plan, Execute, Validate**. Batuta orchestrates each phase, applying Toyota Production System principles to prevent waste and ensure quality at every step.

## The Migration Cycle

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Assess  │────>│   Plan   │────>│ Execute  │────>│ Validate │
│          │     │          │     │          │     │          │
│ TDG scan │     │ Priority │     │ Transpile│     │ renacer  │
│ pmat     │     │ schedule │     │ optimize │     │ tests    │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
      ^                                                  │
      └──────────────── Kaizen feedback ─────────────────┘
```

## Phase 1: Assess

Run batuta's analysis phase to understand the codebase before writing any Rust:

```bash
batuta analyze --languages --tdg /path/to/project
```

This produces a TDG (Technical Debt Grade) per file, language breakdown, dependency map, and ML framework detection results.

## Phase 2: Plan

Use risk-based prioritization to order the migration. High-value, low-risk modules go first:

| Priority | Criteria | Example |
|----------|----------|---------|
| P0 | Pure functions, no I/O | Math utilities, parsers |
| P1 | Isolated modules, clear interfaces | Data transformers |
| P2 | Stateful but well-tested | Service handlers |
| P3 | Complex dependencies, unsafe code | FFI layers, kernel modules |

## Phase 3: Execute

Batuta coordinates transpilers (depyler, decy, bashrs) and applies optimization passes:

```bash
batuta transpile --source ./src --target ./rust_out
batuta optimize --backend auto ./rust_out
```

## Phase 4: Validate

Semantic preservation is verified through syscall tracing and output comparison:

```bash
batuta validate --trace --compare ./rust_out
```

## Risk-Based Prioritization

Score each module on two axes and migrate the high-value, low-risk quadrant first:

```
        High Value
            │
     P1     │     P0
  (plan     │  (migrate
   carefully)│   first)
────────────┼────────────
     P3     │     P2
  (defer or │  (migrate
   wrap FFI)│   second)
            │
        Low Value
```

Batuta's `stack quality` command generates these scores automatically from TDG data, cyclomatic complexity, and test coverage.

## Key Principles

- **Jidoka**: Stop the migration if validation fails at any phase. Never proceed with broken output.
- **Kaizen**: Each cycle improves the migration playbook. Feed validation results back into assessment.
- **Muda**: Avoid migrating dead code. Use `batuta analyze` to identify unused modules.
- **Poka-Yoke**: Enforce type safety early. Let the Rust compiler catch errors that tests missed.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
