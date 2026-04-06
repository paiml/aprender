# Introduction

Welcome to **Renacer** (Spanish: "to be reborn"), a next-generation pure Rust system call tracer with source-aware correlation for Rust binaries.

## What is Renacer?

Renacer is a binary inspection and tracing framework that allows you to observe and analyze system calls made by programs. Unlike traditional tools like `strace`, Renacer provides:

- **Pure Rust implementation** - Type-safe, memory-safe tracing
- **DWARF debug info correlation** - See which source file and line triggered each syscall
- **Function-level profiling** - Understand I/O bottlenecks and hot paths
- **Advanced filtering** - Powerful syscall selection with regex patterns and negation
- **Statistical analysis** - SIMD-accelerated percentile analysis and anomaly detection
- **OpenTelemetry integration** - Export traces to Jaeger, Grafana Tempo, and more
- **Distributed tracing** - W3C Trace Context propagation across services
- **Transpiler support** - Map transpiled code (Python→Rust, C→Rust) back to original source
- **Performance optimized** - <5% overhead with memory pooling and zero-copy strings
- **Multiple output formats** - JSON, CSV, HTML for integration with other tools
- **Chaos engineering** - Test system resilience with controlled fault injection
- **Fuzz testing** - Coverage-guided fuzzing for robustness

## Why Renacer?

**For Developers:**
- Debug performance issues by seeing exactly which functions cause slow I/O
- Understand your program's system-level behavior
- Correlate syscalls with source code locations

**For DevOps:**
- Monitor production processes with minimal overhead (3-4% vs strace's 8-12%)
- Detect anomalies in real-time with configurable thresholds
- Export traces to OpenTelemetry backends (Jaeger, Tempo, Honeycomb)
- Build end-to-end observability with distributed tracing

**For Security Researchers:**
- Observe program behavior at the syscall level
- Trace multi-process applications with fork/clone tracking
- Analyze syscall patterns with HPU-accelerated correlation matrices

## Current Status

**Version:** 0.5.0
**Status:** Production-Ready + Performance Optimization (Sprint 36)
**Test Coverage:** 400+ tests (all passing)
**TDG Score:** 95.1/100 (A+ grade)

Renacer is built following Toyota Way principles and EXTREME TDD methodology, ensuring every feature is thoroughly tested and production-ready.

## Quick Example

```bash
# Basic syscall tracing
$ renacer -- ls -la
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
openat(AT_FDCWD, ".", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
getdents64(3, [...], 32768) = 1024
write(1, "total 128\n", 10) = 10
...

# With source correlation (requires debug symbols)
$ renacer --source -- ./my-program
read(3, buf, 1024) = 42          [src/main.rs:15 in my_function]
write(1, "result", 6) = 6        [src/main.rs:20 in my_function]
...

# Function profiling with I/O bottleneck detection
$ renacer --function-time --source -- cargo test
Function Profiling Summary:
========================
Top 10 Hot Paths (by total time):
  1. cargo::build_script  - 45.2% (1.2s, 67 syscalls) ⚠️ SLOW I/O
  2. rustc::compile       - 32.1% (850ms, 45 syscalls)
  ...
```

## Who Built Renacer?

Renacer is developed by [Pragmatic AI Labs](https://paiml.com) using:
- **Toyota Way** quality principles
- **EXTREME TDD** methodology (every feature test-driven)
- **Zero tolerance** for defects (all 400+ tests pass, zero warnings)
- **Property-based testing** (670+ test cases via proptest)
- **Mutation testing** (80%+ mutation score via cargo-mutants)
- **Fuzz testing** (coverage-guided fuzzing via cargo-fuzz)
- **Performance benchmarking** (Criterion.rs with regression detection)
- **Tiered TDD** (fast/medium/slow test tiers for rapid development)

## Next Steps

- **New to Renacer?** Start with [Quick Start](./getting-started/quick-start.md)
- **Want to understand concepts?** Read [Core Concepts](./core-concepts/syscall-tracing.md)
- **Ready for advanced features?** Explore [Function Profiling](./advanced/function-profiling.md) or [Anomaly Detection](./advanced/anomaly-detection.md)
- **Contributing?** See [EXTREME TDD](./contributing/extreme-tdd.md), [Fuzz Testing](./contributing/fuzz-testing.md), and [Chaos Engineering](./contributing/chaos-engineering.md)
- **Want faster development?** Check out [Tiered TDD Workflow](./contributing/tiered-tdd.md)

Let's get started!
