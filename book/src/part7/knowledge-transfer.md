# Knowledge Transfer

Migration projects create knowledge silos if not managed deliberately. This chapter covers documentation-driven development, Oracle mode as a knowledge base, and cross-training on Rust idioms.

## Documentation-Driven Development

Every migrated module should have a doc comment explaining its origin:

```rust
//! # Data Loader
//!
//! Migrated from `src/data_loader.py`.
//!
//! ## Key Changes
//! - `load_csv()` returns `Result<DataFrame>` instead of raising exceptions
//! - NumPy operations replaced with trueno `Vector`
//! - File I/O uses `BufReader` for 3x throughput improvement
```

## Oracle Mode as Knowledge Base

Batuta's Oracle provides natural language access to stack knowledge:

```bash
batuta oracle "How do I load a model with quantization?"
batuta oracle --recipe ml-random-forest --format code
batuta oracle --rag "tokenization pipeline"
```

Re-index after adding documentation:

```bash
batuta oracle --rag-index
```

## Cross-Training on Rust Idioms

### Python-to-Rust Mental Model Shifts

| Python Concept | Rust Equivalent | Key Difference |
|---------------|-----------------|----------------|
| `try/except` | `Result<T, E>` + `?` | Errors are values |
| `None` checks | `Option<T>` + `.map()` | Compiler-enforced null safety |
| `class` | `struct` + `impl` | No inheritance; use traits |
| List comprehension | `.iter().map().collect()` | Lazy evaluation |
| `with` context manager | `Drop` trait | Automatic cleanup on scope exit |

### Recommended Learning Path

1. **Week 1-2**: Rust Book chapters 1-10 (ownership, borrowing, traits)
2. **Week 3-4**: Read stack code with `pmat query --include-source`
3. **Week 5-6**: Pair-program on a low-risk migration
4. **Week 7+**: Independent migration with mentored review

## Knowledge Artifacts

| Artifact | Location | Purpose |
|----------|----------|---------|
| CLAUDE.md | Project root | Machine-readable project context |
| Oracle recipes | `batuta oracle --cookbook` | Code patterns with tests |
| mdBook | `book/src/` | Comprehensive reference |
| API docs | `cargo doc --no-deps` | Generated from doc comments |

---

**Navigate:** [Table of Contents](../SUMMARY.md)
