# Community Resources

The Sovereign AI Stack is an open ecosystem of Rust crates. This chapter lists the primary resources for learning, contributing, and getting support.

## GitHub Repositories

| Repository | Purpose |
|-----------|---------|
| batuta | Orchestration framework |
| trueno | SIMD/GPU compute primitives |
| aprender | ML algorithms, APR v2 format |
| realizar | Inference engine |
| repartir | Distributed computing |
| depyler / decy / bashrs | Language transpilers |
| renacer | Syscall tracing |
| pmat | Static analysis and TDG scoring |

## Documentation

| Resource | Access |
|----------|--------|
| API docs (local) | `cargo doc --no-deps --open` |
| API docs (published) | `https://docs.rs/<crate>` |
| This book (local) | `make book-serve` (localhost:3000) |
| Oracle mode | `batuta oracle "your question"` |
| Oracle RAG | `batuta oracle --rag "topic"` |
| Cookbook recipes | `batuta oracle --cookbook --format code` |

## Crates.io

All production-ready stack components are published on crates.io:

```bash
# Check latest versions
batuta stack versions

# JSON output for automation
batuta stack versions --format json
```

## Learning Path

| Stage | Resources |
|-------|-----------|
| Getting started | This book, Parts I-II |
| Practical examples | This book, Part IV |
| ML workflows | `batuta oracle --cookbook` |
| Deep internals | This book, Part IX, and `cargo doc` |
| Contributing | [Appendix J: Contributing Guide](../appendix/contributing.md) |

## Staying Updated

Subscribe to crates.io RSS feeds for release notifications:

```
https://crates.io/api/v1/crates/trueno/versions.rss
https://crates.io/api/v1/crates/aprender/versions.rss
https://crates.io/api/v1/crates/realizar/versions.rss
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
