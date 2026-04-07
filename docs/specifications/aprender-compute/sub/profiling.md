# Sub-spec: Profiling & Tracing

**Parent:** [trueno-spec.md](../trueno-spec.md) Section 14

---

## 1. Renacer (v0.5.0+)

Syscall tracing, function profiling, flamegraphs, and OTLP export.

```bash
make profile                              # benchmark profiling
make profile-flamegraph                   # flamegraph visualization
make profile-bench BENCH=vector_ops       # specific benchmark
make profile-test                         # profile test suite
```

Advanced:
```bash
renacer --function-time --source -- cargo bench vector_ops    # I/O bottleneck >1ms
renacer --function-time --source -- cargo bench | flamegraph.pl > flame.svg
```

## 2. OTLP Distributed Tracing

Export syscall traces to Jaeger, Grafana Tempo, or any OTLP-compatible collector.

```bash
make profile-otlp-jaeger       # traces → localhost:16686
make profile-otlp-tempo         # traces → Grafana (localhost:3000, admin/admin)
```

Features: span hierarchy (process→syscall), rich attributes (name, result, duration, source location), works with all Renacer flags.

### Pre-Release Validation Workflow

```bash
# Baseline
make profile-otlp-jaeger
curl "localhost:16686/api/traces?service=trueno-benchmarks" > traces-before.json

# After changes
make profile-otlp-jaeger
curl "localhost:16686/api/traces?service=trueno-benchmarks" > traces-after.json

# Compare
python3 scripts/compare_traces.py traces-before.json traces-after.json
```

## 3. Golden Trace Validation

Syscall-level performance regression detection via `renacer.toml` assertions.

```bash
./scripts/capture_golden_traces.sh                                    # capture baselines
renacer --assert renacer.toml -- ./target/release/examples/backend_detection  # validate
```

**Captured operations** (v0.7.0): backend_detection (0.73ms/87 syscalls), matrix_operations (1.56ms/168), activation_functions (1.30ms/159), performance_demo (1.51ms/138), ml_similarity (0.82ms/109).

**Assertions** in `renacer.toml`:
- `example_startup_latency`: max 100ms, CI fails on violation
- `max_syscall_budget`: max 500 spans, CI fails on violation
- `detect_pcie_bottleneck`: anti-pattern warning (threshold 0.7)

## 4. Key Empirical Insights

- Futex overhead dominates for <1us operations (up to 22x slowdown)
- Cargo test harness adds 0.9ms overhead (1600x for 547ns operation)
- Zero mmap/munmap confirmed in hot path (zero-allocation validated)
- Use raw binaries for micro-benchmarks, avoid async for <10us ops

## 5. Documentation

- Integration report: `docs/integration-report-golden-trace.md`
- Book chapter: `book/src/performance/golden-trace-validation.md`
