# Greenfield vs Brownfield

When migrating to Rust, the first architectural decision is whether to start a new Rust project from scratch (greenfield) or wrap and incrementally replace existing code (brownfield). The right choice depends on codebase size, risk tolerance, and timeline.

## Decision Matrix

| Factor | Greenfield (Rewrite) | Brownfield (Wrap + Replace) |
|--------|---------------------|-----------------------------|
| Codebase size | < 10K lines | > 10K lines |
| Test coverage | < 50% (tests unreliable) | > 70% (tests guide migration) |
| Timeline | 3+ months available | Incremental delivery needed |
| Dependencies | Few, well-understood | Many, deeply coupled |
| Team Rust experience | Intermediate+ | Any level |
| Risk tolerance | Higher | Lower |

## Greenfield: New Rust Project

Best when the original code is small, poorly tested, or architecturally flawed.

```bash
# Generate a fresh Rust project from analysis
batuta init --from-analysis ./legacy_python_project
```

Batuta analyzes the source, generates a Cargo.toml with mapped dependencies, and creates module stubs matching the original structure.

### When to Rewrite

- The original has no tests and unclear behavior
- Architecture needs fundamental changes (e.g., single-threaded to async)
- The codebase is small enough to rewrite in one sprint
- You want to leverage trueno SIMD from the ground up

## Brownfield: Wrap with FFI

Best when the system is large, in production, and must keep running during migration.

```rust
// Wrap existing C library via FFI
extern "C" {
    fn legacy_compute(data: *const f32, len: usize) -> f32;
}

// Rust wrapper with safety boundary
pub fn compute(data: &[f32]) -> f32 {
    unsafe { legacy_compute(data.as_ptr(), data.len()) }
}
```

### When to Wrap

- The system is in production with live traffic
- Individual modules can be replaced behind stable interfaces
- You need to validate Rust output against the original at each step
- Team is still learning Rust idioms

## Hybrid Approach

Most real migrations use a hybrid. Batuta supports this with its gradual migration mode:

```bash
# Transpile one module at a time
batuta transpile --module data_loader --source ./src --target ./rust_out

# Validate the single module
batuta validate --module data_loader --compare
```

### Progression Pattern

```
Week 1-2:  [Python] [Python] [Python] [Python]
Week 3-4:  [Rust  ] [Python] [Python] [Python]
Week 5-6:  [Rust  ] [Rust  ] [Python] [Python]
Week 7-8:  [Rust  ] [Rust  ] [Rust  ] [Python]
Week 9-10: [Rust  ] [Rust  ] [Rust  ] [Rust  ]
```

Each replacement is validated independently before proceeding. This is the Jidoka principle applied to migration: stop and fix before moving forward.

## Common Pitfall: The Big Bang Rewrite

Avoid rewriting everything at once. Even small projects benefit from incremental validation. Batuta's 5-phase pipeline enforces this discipline by requiring validation after each transpilation.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
