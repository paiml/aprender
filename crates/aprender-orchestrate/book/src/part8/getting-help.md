# Getting Help

When debugging and documentation are not enough, here is how to get assistance with Batuta and the Sovereign AI Stack.

## Self-Service Resources

Before reaching out, check these resources in order:

| Resource | URL / Command | Best For |
|----------|---------------|----------|
| This book | `make book-serve` | Concepts, architecture, examples |
| API documentation | `cargo doc --no-deps --open` | Function signatures, type details |
| Oracle mode | `batuta oracle "your question"` | Natural language queries about the stack |
| Oracle RAG | `batuta oracle --rag "topic"` | Searching indexed documentation |
| Error codes | [Appendix E](../appendix/error-codes.md) | Specific error code explanations |
| CLI help | `batuta --help`, `batuta <cmd> --help` | Command flags and options |

## Diagnostic Self-Check

Run these commands and include the output in any help request:

```bash
# Environment info
rustc --version
cargo --version
batuta --version

# Tool availability
batuta analyze --check-tools

# Stack health
batuta stack check

# Pipeline state (if relevant)
batuta status --verbose
```

## Escalation Path

```
┌────────────────────┐
│ 1. Read the docs   │  This book, cargo doc, oracle mode
├────────────────────┤
│ 2. Search issues   │  GitHub issues (existing solutions)
├────────────────────┤
│ 3. File an issue   │  See Issue Reporting chapter
├────────────────────┤
│ 4. Community help  │  See Community Resources chapter
└────────────────────┘
```

## Common Resolution Paths

| Problem Type | First Step |
|-------------|------------|
| Build failure | `cargo build 2>&1` -- read the compiler error carefully |
| Test failure | `cargo test -- --nocapture test_name` -- see the full output |
| Pipeline failure | `batuta status --verbose` -- check which phase failed |
| Performance issue | `cargo bench` -- measure before diagnosing |
| Transpilation error | `RUST_LOG=debug batuta transpile` -- check the logs |

## Stack Component Documentation

Each component in the Sovereign AI Stack has its own documentation:

| Component | docs.rs | Source |
|-----------|---------|--------|
| trueno | [docs.rs/trueno](https://docs.rs/trueno) | SIMD/GPU compute |
| aprender | [docs.rs/aprender](https://docs.rs/aprender) | ML algorithms |
| realizar | [docs.rs/realizar](https://docs.rs/realizar) | Inference engine |
| repartir | [docs.rs/repartir](https://docs.rs/repartir) | Distributed compute |
| renacer | [docs.rs/renacer](https://docs.rs/renacer) | Syscall tracing |

See [Issue Reporting](./issue-reporting.md) for how to file effective bug reports, and [Community Resources](./community.md) for additional support channels.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
