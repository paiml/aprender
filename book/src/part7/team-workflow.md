# Team Workflow

Migrating a codebase to Rust is a team effort. This chapter covers workflow practices that keep the team productive while maintaining quality standards during the transition.

## Workflow Overview

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│   Develop  │───>│   Review   │───>│   Validate │───>│   Merge    │
│            │    │            │    │            │    │            │
│ Write code │    │ PR review  │    │ Tier 3/4   │    │ Quality    │
│ Tier 1/2   │    │ pmat check │    │ CI pipeline│    │ gate pass  │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
```

## Role Allocation During Migration

| Role | Responsibility | Tools |
|------|---------------|-------|
| Migration Lead | Prioritization, risk assessment | `batuta analyze`, `batuta stack quality` |
| Transpilation Engineer | Running and tuning transpilers | `batuta transpile`, `batuta optimize` |
| Validation Engineer | Testing parity and performance | `batuta validate`, `renacer`, Criterion |
| Rust Mentor | Code review, idiom guidance | `cargo clippy`, `pmat query` |

Small teams combine roles. The key is that no migration step skips validation.

## Daily Workflow

```bash
# Morning: check stack health
batuta stack check

# Development: write and test
make tier1           # On every save
make tier2           # Before each commit

# Afternoon: integration
make tier3           # Before pushing

# CI/CD: automated
make tier4           # Runs on every push
```

## Communication Practices

### Migration Status Board

Track module migration status visually:

```
Module            Status       Owner    Risk
─────────────────────────────────────────────
data_loader       [DONE]       Alice    Low
api_server        [IN PROGRESS] Bob     Medium
ml_pipeline       [PLANNED]    Carol    High
legacy_ffi        [DEFERRED]   --       Critical
```

Use `batuta stack status` for the TUI dashboard equivalent.

### Decision Log

Document every non-obvious decision during migration:

- Why a module was deferred instead of migrated
- Why FFI was chosen over rewrite for a specific boundary
- Why a particular Rust pattern was preferred over another

This prevents re-litigating decisions and helps onboard new team members.

## Quality Enforcement

The pre-commit hook enforces quality gates automatically:

- Formatting must pass (`cargo fmt`)
- No clippy warnings (`cargo clippy -- -D warnings`)
- Complexity thresholds: cyclomatic <= 30, cognitive <= 25
- Commit messages must reference a work item

These gates apply equally to migration code and new development, ensuring the migrated codebase maintains high quality from day one.

See [Code Review Process](./code-review.md) and [Knowledge Transfer](./knowledge-transfer.md) for detailed guidance on team practices.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
