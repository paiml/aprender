# Integration Tests for Sprints 32-33 Specification

**Version:** 1.0
**Date:** 2025-11-20
**Status:** Specification - Ready for Implementation
**Sprint Target:** 34 (Integration Test Coverage)

## Executive Summary

This specification defines **comprehensive integration tests** for Sprint 32 (Block-Level Compute Tracing) and Sprint 33 (W3C Trace Context Propagation) to ensure production readiness and validate end-to-end observability stack functionality.

**Business Value:**
- **Production Confidence**: Verify features work with actual Jaeger/Tempo backends
- **Regression Prevention**: Catch breaking changes early
- **Documentation**: Tests serve as executable examples
- **Quality Assurance**: Validate complex interactions between features

**Key Principle (Toyota Way - Jidoka):**
> *"Build quality in at the source. Stop and fix problems immediately."* - Comprehensive testing prevents defects from reaching production.

---

## 1. Goals and Requirements

### 1.1 Primary Goals

**G1: Sprint 32 Integration Coverage**
- Verify compute tracing with actual OTLP backend (Jaeger)
- Validate adaptive sampling thresholds (100μs default)
- Test span hierarchy: process → compute_block → syscalls
- Verify feature-gating (builds without OTLP)
- Test all CLI flag combinations

**G2: Sprint 33 Integration Coverage**
- Verify W3C Trace Context propagation end-to-end
- Validate parent-child span relationships in Jaeger
- Test trace-id propagation across process boundaries
- Verify invalid context handling (all-zero IDs, malformed)
- Test environment variable extraction

**G3: Combined Stack Testing**
- Test full observability stack (Sprints 30-33 combined)
- Verify: syscalls + decisions + compute + distributed tracing
- Validate span hierarchy in actual trace backend
- Performance regression testing

### 1.2 Non-Goals

**NG1: Load Testing**
We will NOT perform high-volume load testing (deferred to Sprint 36)

**NG2: Chaos Testing**
We will NOT inject failures/chaos (already covered in Sprint 29)

**NG3: Cross-Platform Testing**
We will NOT test on ARM64/Windows (deferred to v1.0)

---

## 2. Test Categories

### 2.1 Sprint 32: Compute Tracing Tests

**Category A: Adaptive Sampling**
1. `test_compute_adaptive_sampling_default` - Verify 100μs threshold
2. `test_compute_adaptive_sampling_custom` - Verify custom threshold (50μs)
3. `test_compute_trace_all_flag` - Verify --trace-compute-all bypasses sampling
4. `test_compute_sampling_below_threshold` - Verify fast blocks (<100μs) NOT traced
5. `test_compute_sampling_above_threshold` - Verify slow blocks (>=100μs) traced

**Category B: Span Attributes**
6. `test_compute_span_attributes` - Verify compute.operation, duration_us, elements, is_slow
7. `test_compute_resource_attributes` - Verify compute.library, version, abstraction at Resource level
8. `test_compute_span_hierarchy` - Verify process → compute_block parent-child

**Category C: Feature Integration**
9. `test_compute_with_statistics` - Verify --trace-compute -c --stats-extended
10. `test_compute_with_decisions` - Verify --trace-compute --trace-transpiler-decisions
11. `test_compute_with_filtering` - Verify --trace-compute -e trace=write
12. `test_compute_feature_gating` - Verify build without OTLP feature

**Category D: Jaeger Verification**
13. `test_compute_jaeger_export` - Verify spans appear in Jaeger UI
14. `test_compute_span_timing` - Verify duration_us matches actual execution time
15. `test_compute_multiple_blocks` - Verify multiple compute blocks in single trace

### 2.2 Sprint 33: Distributed Tracing Tests

**Category E: W3C Context Parsing**
16. `test_w3c_valid_traceparent_cli` - Verify --trace-parent flag parsing
17. `test_w3c_valid_traceparent_env` - Verify TRACEPARENT env var extraction
18. `test_w3c_invalid_traceparent_fallback` - Verify graceful fallback on invalid context
19. `test_w3c_all_zero_trace_id_rejection` - Verify all-zero trace-id rejected
20. `test_w3c_all_zero_parent_id_rejection` - Verify all-zero parent-id rejected

**Category F: Span Propagation**
21. `test_distributed_parent_child_relationship` - Verify Renacer span is child of app span
22. `test_distributed_trace_id_propagation` - Verify same trace-id across boundaries
23. `test_distributed_span_context_remote_flag` - Verify is_remote=true for propagated context
24. `test_distributed_sampling_flag` - Verify trace_flags sampling bit honored

**Category G: Jaeger Verification**
25. `test_distributed_jaeger_hierarchy` - Verify span hierarchy in Jaeger UI
26. `test_distributed_trace_search` - Verify can find traces by trace-id
27. `test_distributed_service_attribution` - Verify service name appears correctly

### 2.3 Combined Stack Tests

**Category H: Full Observability**
28. `test_full_stack_syscalls_decisions_compute_distributed` - All features together
29. `test_full_stack_span_hierarchy` - Verify 4-level hierarchy in Jaeger
30. `test_full_stack_performance_overhead` - Verify <5% overhead with all features

---

## 3. Test Infrastructure

### 3.1 Jaeger Backend Setup

**Docker Compose for Tests:**
```yaml
version: '3'
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    ports:
      - "16686:16686"  # Jaeger UI
      - "4317:4317"    # OTLP gRPC
      - "4318:4318"    # OTLP HTTP
```

**Test Harness:**
```rust
// Helper to start Jaeger for tests
fn start_jaeger_container() -> Result<JaegerHandle> {
    // Use testcontainers-rs to manage Jaeger lifecycle
    let container = clients::Cli::default()
        .run(GenericImage::new("jaegertracing/all-in-one", "latest")
            .with_exposed_port(16686)
            .with_exposed_port(4317));

    // Wait for Jaeger to be ready
    wait_for_jaeger_ready(&container)?;

    Ok(JaegerHandle { container })
}

// Helper to query Jaeger API for traces
fn query_jaeger_traces(service: &str, trace_id: Option<&str>) -> Result<Vec<Trace>> {
    let url = format!("http://localhost:16686/api/traces?service={}", service);
    // ... HTTP request to Jaeger API
}

// Helper to verify span hierarchy
fn verify_span_hierarchy(trace_id: &str, expected_spans: &[ExpectedSpan]) -> Result<()> {
    let traces = query_jaeger_traces("renacer", Some(trace_id))?;
    // ... validation logic
}
```

### 3.2 Test Program

**Simple test binary with debug symbols:**
```rust
// tests/fixtures/test_program.rs
use std::fs;

fn main() {
    // Generate syscalls for testing
    fs::write("/tmp/test.txt", "Hello").unwrap();
    let _ = fs::read_to_string("/tmp/test.txt").unwrap();

    // Trigger compute tracing (if statistics enabled)
    // This will call calculate_extended_statistics()
}
```

**Compile:**
```bash
rustc -g tests/fixtures/test_program.rs -o tests/fixtures/test_program
```

### 3.3 Test Utilities

```rust
// tests/utils/mod.rs

/// Wait for Jaeger to be ready
pub fn wait_for_jaeger_ready(endpoint: &str) -> Result<()> {
    for _ in 0..30 {
        if reqwest::blocking::get(format!("{}/", endpoint)).is_ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    Err(anyhow!("Jaeger not ready"))
}

/// Extract trace-id from Renacer stderr output
pub fn extract_trace_id_from_output(stderr: &str) -> Option<String> {
    // Look for trace-id in OTLP export logs
    // Example: "[renacer: OTLP trace-id: abc123...]"
    let re = regex::Regex::new(r"trace-id: ([0-9a-f]{32})").unwrap();
    re.captures(stderr)?.get(1).map(|m| m.as_str().to_string())
}

/// Verify span exists in Jaeger
pub fn verify_span_exists(
    trace_id: &str,
    span_name: &str,
    attributes: HashMap<String, String>,
) -> Result<()> {
    let traces = query_jaeger_traces("renacer", Some(trace_id))?;

    for trace in traces {
        for span in trace.spans {
            if span.operation_name == span_name {
                // Verify attributes
                for (key, expected_value) in &attributes {
                    let actual = span.tags.get(key)
                        .ok_or_else(|| anyhow!("Missing attribute: {}", key))?;

                    if actual != expected_value {
                        return Err(anyhow!(
                            "Attribute mismatch: {} = {} (expected {})",
                            key, actual, expected_value
                        ));
                    }
                }
                return Ok(());
            }
        }
    }

    Err(anyhow!("Span not found: {}", span_name))
}
```

---

## 4. Detailed Test Specifications

### 4.1 Sprint 32 Tests (Compute Tracing)

#### Test 1: Adaptive Sampling Default (100μs)

**Test:** `test_compute_adaptive_sampling_default`

**Setup:**
```bash
# Start Jaeger
docker-compose -f docker-compose-test.yml up -d jaeger

# Run Renacer with compute tracing
renacer --otlp-endpoint http://localhost:4317 \
        --trace-compute \
        -c --stats-extended \
        -- cargo build
```

**Validation:**
1. Query Jaeger for compute_block spans
2. Verify only spans with duration >= 100μs are present
3. Count total spans, ensure < 50% of all operations (sampling working)
4. Check span attributes: `compute.duration_us >= 100`, `compute.is_slow = true`

**Expected Result:**
- ✅ Only slow compute blocks (>=100μs) appear in Jaeger
- ✅ Fast blocks (<100μs) are NOT in Jaeger
- ✅ Attributes correct

#### Test 2: Custom Sampling Threshold (50μs)

**Test:** `test_compute_adaptive_sampling_custom`

**Setup:**
```bash
renacer --otlp-endpoint http://localhost:4317 \
        --trace-compute \
        --trace-compute-threshold 50 \
        -c --stats-extended \
        -- cargo build
```

**Validation:**
1. Query Jaeger for compute_block spans
2. Verify spans with duration >= 50μs are present
3. Verify spans with duration < 50μs are NOT present
4. Check `compute.duration_us >= 50`

**Expected Result:**
- ✅ Threshold lowered to 50μs
- ✅ More spans than default (100μs)

#### Test 3: Trace All Flag (Bypass Sampling)

**Test:** `test_compute_trace_all_flag`

**Setup:**
```bash
renacer --otlp-endpoint http://localhost:4317 \
        --trace-compute \
        --trace-compute-all \
        -c --stats-extended \
        -- ./small_workload
```

**Validation:**
1. Query Jaeger for compute_block spans
2. Count spans, should be ~100% of compute operations
3. Verify fast spans (<100μs) ARE present
4. Check `compute.is_slow` varies (true/false based on actual duration)

**Expected Result:**
- ✅ ALL compute blocks traced (no sampling)
- ✅ Both fast and slow spans present

#### Test 13: Jaeger Export Verification

**Test:** `test_compute_jaeger_export`

**Setup:**
```bash
# Start Jaeger
docker-compose -f docker-compose-test.yml up -d

# Run with compute tracing
renacer --otlp-endpoint http://localhost:4317 \
        --trace-compute \
        -c --stats-extended \
        -- ./test_program
```

**Validation:**
1. **Jaeger UI Check:**
   - Open http://localhost:16686
   - Search for service: "renacer"
   - Verify traces appear within 5 seconds
2. **API Check:**
   ```bash
   curl "http://localhost:16686/api/traces?service=renacer&limit=1"
   ```
3. **Span Verification:**
   - Verify `process: ./test_program` span exists
   - Verify `compute_block: calculate_statistics` child span
   - Verify Resource attributes: `compute.library=trueno`

**Expected Result:**
- ✅ Spans visible in Jaeger UI
- ✅ Correct span hierarchy
- ✅ Resource attributes present

### 4.2 Sprint 33 Tests (Distributed Tracing)

#### Test 21: Parent-Child Relationship

**Test:** `test_distributed_parent_child_relationship`

**Setup:**
```bash
# Simulate app span with known trace context
TRACE_ID="0af7651916cd43dd8448eb211c80319c"
PARENT_SPAN_ID="b7ad6b7169203331"

# Run Renacer with injected context
renacer --otlp-endpoint http://localhost:4317 \
        --trace-parent "00-${TRACE_ID}-${PARENT_SPAN_ID}-01" \
        -- ./test_program
```

**Validation:**
1. Query Jaeger for trace-id: `0af7651916cd43dd8448eb211c80319c`
2. Find Renacer's root span (`process: ./test_program`)
3. Verify:
   - `span.trace_id == TRACE_ID`
   - `span.parent_span_id == PARENT_SPAN_ID`
   - `span.context.is_remote == true`

**Expected Result:**
- ✅ Renacer span has correct parent
- ✅ Same trace-id propagated
- ✅ Remote flag set

#### Test 25: Jaeger Hierarchy Visualization

**Test:** `test_distributed_jaeger_hierarchy`

**Setup:**
```bash
# Create mock app span first (using OTLP directly or mock)
# Then run Renacer as child

renacer --otlp-endpoint http://localhost:4317 \
        --trace-parent "00-abc123-def456-01" \
        -- ./test_program
```

**Validation:**
1. Open Jaeger UI: http://localhost:16686
2. Search for trace-id: `abc123`
3. **Visual Hierarchy Check:**
   ```
   └─ process: ./test_program (Renacer)
       ├─ syscall: write
       ├─ syscall: read
       └─ compute_block: calculate_statistics
   ```
4. Verify indentation shows parent-child relationship
5. Verify timeline alignment

**Expected Result:**
- ✅ Hierarchy visible in Jaeger UI
- ✅ Spans aligned on timeline
- ✅ Parent-child indentation correct

### 4.3 Combined Stack Tests

#### Test 28: Full Observability Stack

**Test:** `test_full_stack_syscalls_decisions_compute_distributed`

**Setup:**
```bash
# Full stack: Sprints 30-33
renacer --otlp-endpoint http://localhost:4317 \
        --trace-parent "00-fullstack-parentspan-01" \
        --trace-compute \
        --trace-transpiler-decisions \
        -c --stats-extended \
        -T \
        -- ./transpiled_app
```

**Validation:**
1. Query Jaeger for trace-id: `fullstack`
2. Verify 4-layer hierarchy:
   ```
   └─ process: ./transpiled_app (Renacer root, child of external)
       ├─ Span Event: decision: type_inference::infer_type (Sprint 31)
       ├─ compute_block: calculate_statistics (Sprint 32)
       │   └─ duration: 150μs, elements: 1024
       ├─ syscall: connect (Sprint 30)
       ├─ syscall: write (Sprint 30)
       └─ syscall: read (Sprint 30)
   ```
3. Verify all attributes present across all layers
4. Check Resource attributes: service.name, compute.library, etc.

**Expected Result:**
- ✅ All 4 sprint features working together
- ✅ Complete span hierarchy
- ✅ No conflicts/errors

#### Test 30: Performance Overhead

**Test:** `test_full_stack_performance_overhead`

**Setup:**
```bash
# Baseline: no tracing
time ./benchmark_app

# With full stack tracing
time renacer --otlp-endpoint http://localhost:4317 \
              --trace-parent "00-perf-test-01" \
              --trace-compute \
              --trace-transpiler-decisions \
              -c --stats-extended \
              -- ./benchmark_app
```

**Validation:**
1. Measure baseline execution time (3 runs, average)
2. Measure with tracing (3 runs, average)
3. Calculate overhead: `(traced_time - baseline_time) / baseline_time * 100`
4. **Acceptance Criteria:** Overhead < 5%

**Expected Result:**
- ✅ Overhead < 5% for full stack
- ✅ Performance regression testing

---

## 5. Implementation Phases

### Phase 1: Infrastructure Setup (Day 1)

**Tasks:**
1. Create `docker-compose-test.yml` with Jaeger
2. Create test utilities in `tests/utils/mod.rs`
3. Create test fixture programs in `tests/fixtures/`
4. Set up Jaeger API client for validation

**Deliverables:**
- Docker Compose file
- Test utilities (wait_for_jaeger, query_traces, verify_span)
- Test programs (simple_program, compute_heavy, transpiled_app)

### Phase 2: Sprint 32 Tests (Days 1-2)

**Tasks:**
1. Write 15 integration tests for compute tracing
2. Validate with actual Jaeger backend
3. Document test setup and verification steps

**Deliverables:**
- `tests/sprint34_compute_integration_tests.rs` (15 tests)
- Test documentation

### Phase 3: Sprint 33 Tests (Day 2)

**Tasks:**
1. Write 12 integration tests for distributed tracing
2. Validate parent-child relationships in Jaeger
3. Test invalid context handling

**Deliverables:**
- `tests/sprint34_distributed_integration_tests.rs` (12 tests)
- Test documentation

### Phase 4: Combined Stack Tests (Day 3)

**Tasks:**
1. Write 3 full-stack integration tests
2. Performance regression baseline
3. Documentation updates

**Deliverables:**
- `tests/sprint34_full_stack_tests.rs` (3 tests)
- Performance benchmarks
- README and CHANGELOG updates

---

## 6. Success Criteria

**Functional:**
- ✅ 30/30 integration tests passing
- ✅ All tests run against actual Jaeger backend
- ✅ Span hierarchies verified in Jaeger UI
- ✅ Invalid input handling validated
- ✅ Feature combinations tested

**Performance:**
- ✅ Full stack overhead < 5%
- ✅ Tests complete in < 5 minutes
- ✅ Jaeger container starts in < 10 seconds

**Documentation:**
- ✅ Test setup documented
- ✅ Jaeger verification steps documented
- ✅ README updated with integration test info
- ✅ CHANGELOG entry for Sprint 34

**Quality:**
- ✅ Zero clippy warnings
- ✅ All tests use actual OTLP backend (no mocks)
- ✅ Tests are deterministic and reproducible

---

## 7. Test Execution

### 7.1 Running Tests Locally

```bash
# Start Jaeger
docker-compose -f docker-compose-test.yml up -d

# Wait for Jaeger to be ready
curl --retry 10 --retry-delay 1 http://localhost:16686

# Run integration tests
cargo test --test sprint34_compute_integration_tests
cargo test --test sprint34_distributed_integration_tests
cargo test --test sprint34_full_stack_tests

# Stop Jaeger
docker-compose -f docker-compose-test.yml down
```

### 7.2 CI/CD Integration

```yaml
# .github/workflows/integration-tests.yml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest

    services:
      jaeger:
        image: jaegertracing/all-in-one:latest
        ports:
          - 16686:16686
          - 4317:4317
        env:
          COLLECTOR_OTLP_ENABLED: true

    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1

      - name: Wait for Jaeger
        run: |
          timeout 30 bash -c 'until curl -f http://localhost:16686; do sleep 1; done'

      - name: Run integration tests
        run: |
          cargo test --test sprint34_compute_integration_tests
          cargo test --test sprint34_distributed_integration_tests
          cargo test --test sprint34_full_stack_tests
```

---

## 8. Risk Mitigation

**Risk 1: Flaky Tests (Network/Timing)**
- **Mitigation:** Retry logic, generous timeouts, deterministic test data

**Risk 2: Jaeger Container Startup**
- **Mitigation:** Health checks, explicit wait logic, fallback to local Jaeger

**Risk 3: Test Duration**
- **Mitigation:** Parallel test execution, Docker layer caching

**Risk 4: False Positives**
- **Mitigation:** Strict assertions, validate multiple attributes, check Jaeger UI manually

---

## 9. Future Enhancements (Post-Sprint 34)

- **Sprint 35:** Load testing (1M spans/sec)
- **Sprint 36:** Multi-backend testing (Tempo, Zipkin, Honeycomb)
- **Sprint 37:** Cross-platform integration tests (ARM64, macOS)
- **Sprint 38:** Chaos testing integration (network failures, backend outages)

---

**Status:** Ready for implementation
**Estimated Effort:** 2-3 days
**Dependencies:** Sprints 30-33 complete ✅
**Blocks:** Production deployment, v1.0 release
