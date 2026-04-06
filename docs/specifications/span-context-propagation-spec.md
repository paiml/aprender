# Span Context Propagation Specification for Renacer

**Version:** 1.0
**Date:** 2025-11-20
**Status:** Specification - Ready for Implementation
**Sprint Target:** 33 (Span Context Propagation)

## Executive Summary

This specification defines **distributed tracing integration** between **Renacer** (external syscall tracer) and **traced applications** (in-process OpenTelemetry instrumentation) to provide unified end-to-end observability. Following **W3C Trace Context** standard, Renacer will extract trace context from applications and create syscall spans as children of application spans.

**Business Value:**
- **End-to-End Tracing**: Follow a request from HTTP handler → business logic → syscall → kernel
- **Root Cause Analysis**: Correlate slow application operations with underlying syscalls
- **Unified Timeline**: Single trace view combining app-level and syscall-level spans
- **Cross-Process Observability**: Connect distributed traces across service boundaries

**Key Principle (W3C Standard):**
> *"Trace context propagation enables distributed tracing by passing trace identifiers across service boundaries."*

---

## 1. Goals and Requirements

### 1.1 Primary Goals

**G1: W3C Trace Context Compliance**
- Support W3C Trace Context standard (traceparent header format)
- Extract `trace-id`, `parent-id`, and `trace-flags` from traced applications
- Make Renacer's root span a child of the application's active span

**G2: Multiple Context Injection Methods**
- **Method 1**: Environment variables (`TRACEPARENT`, `OTEL_TRACEPARENT`)
- **Method 2**: CLI flag (`--trace-parent`) for explicit injection
- **Method 3**: Auto-detection from application's OpenTelemetry SDK

**G3: Backward Compatibility**
- If no trace context provided → Renacer creates new root trace (existing behavior)
- Feature-gated behind existing `otlp` feature flag
- Zero impact when `--otlp-endpoint` not specified

**G4: Observability Backend Agnostic**
- Works with Jaeger, Tempo, Zipkin, Honeycomb, Datadog, etc.
- Standard W3C format ensures universal compatibility

### 1.2 Non-Goals

**NG1: In-Process Span Extraction**
We will NOT extract spans from application memory (too complex, fragile)

**NG2: Automatic Context Detection**
We will NOT automatically detect trace context from HTTP headers in network traffic (Sprint 34 candidate)

**NG3: B3 Propagation Format**
Initial version supports W3C only (B3 format is Sprint 35 candidate)

---

## 2. W3C Trace Context Standard

### 2.1 Traceparent Header Format

```
traceparent: version-trace-id-parent-id-trace-flags
```

**Example:**
```
traceparent: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
```

**Field Breakdown:**
- `version` (2 hex chars): `00` (current W3C spec version)
- `trace-id` (32 hex chars): 128-bit trace identifier (16 bytes)
- `parent-id` (16 hex chars): 64-bit parent span identifier (8 bytes)
- `trace-flags` (2 hex chars): 8-bit flags (01 = sampled, 00 = not sampled)

### 2.2 Validation Rules

**Valid traceparent:**
- Version must be `00`
- trace-id must not be all zeros (`00000000000000000000000000000000`)
- parent-id must not be all zeros (`0000000000000000`)
- trace-flags must be valid hex

**Invalid traceparent:**
- Malformed format → Reject and create new trace
- All-zero IDs → Reject and create new trace
- Unknown version → Reject and create new trace (forward compatibility)

---

## 3. Architecture Overview

### 3.1 Span Hierarchy

**Without Context Propagation (Current - Sprint 30):**
```
Trace A (Application)                    Trace B (Renacer - disconnected)
  └─ HTTP Handler Span                     └─ process: ./app (ROOT)
      └─ Database Query Span                   ├─ syscall: connect
          └─ (syscalls invisible)              ├─ syscall: write
                                               └─ syscall: read
```

**With Context Propagation (Sprint 33):**
```
Trace A (Unified)
  └─ HTTP Handler Span (app)
      └─ Database Query Span (app)
          └─ process: ./app (renacer - CHILD of DB query)
              ├─ syscall: connect
              ├─ syscall: write
              └─ syscall: read
```

### 3.2 Context Injection Flow

```
┌─────────────────────┐
│  Application        │
│  (OpenTelemetry)    │
│                     │
│  Active Span:       │
│  trace-id: abc123   │
│  span-id: def456    │
└──────────┬──────────┘
           │
           │ (1) Export via ENV or CLI
           ▼
┌─────────────────────┐
│  TRACEPARENT        │
│  Environment Var    │
│  or CLI Flag        │
└──────────┬──────────┘
           │
           │ (2) Extract at startup
           ▼
┌─────────────────────┐
│  Renacer            │
│  Parse traceparent  │
│  Extract:           │
│   - trace-id        │
│   - parent-id       │
│   - trace-flags     │
└──────────┬──────────┘
           │
           │ (3) Set parent context
           ▼
┌─────────────────────┐
│  Root Span          │
│  "process: ./app"   │
│                     │
│  trace-id: abc123   │ ← Same as app
│  span-id: ghi789    │ ← New span ID
│  parent-id: def456  │ ← App's span ID
└─────────────────────┘
```

---

## 4. Implementation Design

### 4.1 CLI Flags

```rust
/// CLI additions (src/cli.rs)

/// Inject W3C trace context (traceparent) for distributed tracing (Sprint 33)
///
/// Format: version-trace_id-parent_id-trace_flags
/// Example: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
#[arg(long = "trace-parent", value_name = "TRACEPARENT")]
pub trace_parent: Option<String>,
```

### 4.2 Context Extraction Module

**New file:** `src/trace_context.rs` (~200 lines)

```rust
/// W3C Trace Context parser (Sprint 33)

use std::fmt;

/// W3C Trace Context (traceparent header)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceContext {
    pub version: u8,
    pub trace_id: [u8; 16],      // 128-bit trace ID
    pub parent_id: [u8; 8],      // 64-bit parent span ID
    pub trace_flags: u8,         // 8-bit flags (01 = sampled)
}

impl TraceContext {
    /// Parse traceparent string: "00-{trace_id}-{parent_id}-{flags}"
    pub fn parse(traceparent: &str) -> Result<Self, TraceContextError> {
        // Parse format: version-trace_id-parent_id-trace_flags
        // Validate all-zero IDs
        // Validate version = 00
    }

    /// Extract from environment variable (TRACEPARENT or OTEL_TRACEPARENT)
    pub fn from_env() -> Option<Self> {
        std::env::var("TRACEPARENT")
            .or_else(|_| std::env::var("OTEL_TRACEPARENT"))
            .ok()
            .and_then(|s| Self::parse(&s).ok())
    }

    /// Check if trace is sampled
    pub fn is_sampled(&self) -> bool {
        self.trace_flags & 0x01 != 0
    }

    /// Convert trace_id to OpenTelemetry TraceId
    pub fn otel_trace_id(&self) -> opentelemetry::trace::TraceId {
        opentelemetry::trace::TraceId::from_bytes(self.trace_id)
    }

    /// Convert parent_id to OpenTelemetry SpanId
    pub fn otel_parent_id(&self) -> opentelemetry::trace::SpanId {
        opentelemetry::trace::SpanId::from_bytes(self.parent_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceContextError {
    InvalidFormat,
    InvalidVersion,
    InvalidTraceId,
    InvalidParentId,
    InvalidTraceFlags,
    AllZeroTraceId,
    AllZeroParentId,
}
```

### 4.3 OTLP Exporter Integration

**Modified file:** `src/otlp_exporter.rs`

```rust
/// Modified OtlpExporter::new() to accept optional trace context

impl OtlpExporter {
    pub fn new(
        config: OtlpConfig,
        trace_context: Option<TraceContext>,  // NEW parameter
    ) -> Result<Self> {
        #[cfg(feature = "otlp")]
        {
            // ... existing setup ...

            // NEW: Create tracer with remote parent context
            let tracer = if let Some(ctx) = trace_context {
                // Create SpanContext from W3C trace context
                let span_context = opentelemetry::trace::SpanContext::new(
                    ctx.otel_trace_id(),
                    ctx.otel_parent_id(),
                    opentelemetry::trace::TraceFlags::new(ctx.trace_flags),
                    true,  // is_remote = true
                    opentelemetry::trace::TraceState::default(),
                );

                // Store for later use in start_root_span()
                provider.tracer_with_context("renacer", span_context)
            } else {
                provider.tracer("renacer")
            };

            // ... rest of initialization ...
        }
    }

    /// Modified to use remote parent context
    pub fn start_root_span(&mut self, command: &str, pid: i32) {
        #[cfg(feature = "otlp")]
        {
            let mut span_builder = self.tracer
                .span_builder(format!("process: {}", command))
                .with_kind(SpanKind::SERVER);

            // NEW: If we have remote parent, set it
            if let Some(ref remote_context) = self.remote_parent_context {
                span_builder = span_builder.with_parent_context(remote_context.clone());
            }

            let span = span_builder.start(&self.tracer);
            // ... rest of method ...
        }
    }
}
```

### 4.4 Integration Points

**Modified:** `src/tracer.rs`

```rust
/// Extract trace context and pass to OTLP exporter

#[cfg(feature = "otlp")]
if let Some(ref endpoint) = config.otlp_endpoint {
    // Extract trace context from CLI flag or environment
    let trace_context = config.trace_parent
        .as_ref()
        .and_then(|s| TraceContext::parse(s).ok())
        .or_else(|| TraceContext::from_env());

    if trace_context.is_some() {
        eprintln!("[renacer: Distributed tracing enabled - joining parent trace]");
    }

    let otlp_config = OtlpConfig {
        endpoint: endpoint.clone(),
        service_name: config.otlp_service_name.clone(),
    };

    match OtlpExporter::new(otlp_config, trace_context) {  // Pass context
        Ok(exporter) => otlp_exporter = Some(exporter),
        Err(e) => eprintln!("[renacer: OTLP initialization failed: {}]", e),
    }
}
```

---

## 5. Usage Examples

### 5.1 Example 1: Environment Variable Injection

**Application code (instrumented with OpenTelemetry):**
```rust
// app.rs
use opentelemetry::trace::Tracer;

fn main() {
    let tracer = /* ... initialize OpenTelemetry ... */;

    let span = tracer.start("database_query");

    // Execute database operation (generates syscalls)
    let result = database.query("SELECT * FROM users");

    span.end();
}
```

**Trace the application:**
```bash
# Application exports trace context via TRACEPARENT env var
export TRACEPARENT="00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

# Renacer automatically detects and joins the trace
renacer --otlp-endpoint http://localhost:4317 -- ./app
```

**Result in Jaeger:**
```
Trace ID: 0af7651916cd43dd8448eb211c80319c

  └─ database_query (span-id: b7ad6b7169203331)  ← Application span
      └─ process: ./app (span-id: <new>)         ← Renacer root span (CHILD)
          ├─ syscall: connect
          ├─ syscall: write
          └─ syscall: read
```

### 5.2 Example 2: Explicit CLI Injection

```bash
# Parent application exports trace context
TRACE_PARENT=$(otel_cli span get-traceparent)

# Explicitly pass to Renacer
renacer --otlp-endpoint http://localhost:4317 \
        --trace-parent "$TRACE_PARENT" \
        -- ./app
```

### 5.3 Example 3: HTTP Request Tracing

```bash
# Extract trace context from incoming HTTP request
TRACE_PARENT=$(curl -v https://api.example.com 2>&1 | grep -i traceparent | cut -d: -f2)

# Trace backend service with same trace ID
renacer --otlp-endpoint http://localhost:4317 \
        --trace-parent "$TRACE_PARENT" \
        -- ./backend-service
```

### 5.4 Example 4: Multi-Service Distributed Trace

```
API Gateway (trace-id: abc123)
  └─ Auth Service (trace-id: abc123, parent: gateway-span)
      └─ renacer ./auth-service (trace-id: abc123, parent: auth-span)
          ├─ syscall: open (read JWT keys)
          └─ syscall: read
```

---

## 6. Testing Strategy (EXTREME TDD)

### 6.1 Phase 1: RED (15 Tests)

**Unit Tests (src/trace_context.rs):**
1. `test_parse_valid_traceparent` - Parse well-formed traceparent
2. `test_parse_invalid_format` - Reject malformed format
3. `test_parse_all_zero_trace_id` - Reject all-zero trace ID
4. `test_parse_all_zero_parent_id` - Reject all-zero parent ID
5. `test_parse_invalid_version` - Reject unknown version
6. `test_parse_invalid_hex` - Reject non-hex characters
7. `test_is_sampled_flag_set` - Check trace_flags & 0x01
8. `test_is_sampled_flag_unset` - Check trace_flags & 0x00
9. `test_from_env_traceparent` - Extract from TRACEPARENT env
10. `test_from_env_otel_traceparent` - Extract from OTEL_TRACEPARENT env
11. `test_from_env_missing` - Return None if no env var

**Integration Tests (tests/sprint33_span_context_propagation_tests.rs):**
12. `test_trace_parent_cli_flag` - Accept --trace-parent flag
13. `test_trace_parent_env_detection` - Auto-detect from environment
14. `test_trace_parent_creates_child_span` - Verify parent-child relationship
15. `test_trace_parent_same_trace_id` - Verify trace ID propagates
16. `test_trace_parent_invalid_fallback` - Invalid context → new root trace
17. `test_trace_parent_with_otlp_export` - End-to-end with Jaeger
18. `test_backward_compatibility_no_context` - Works without trace context

### 6.2 Phase 2: GREEN

Implement minimum code to pass all 17 tests:
1. Create `src/trace_context.rs` with TraceContext parser
2. Add `--trace-parent` CLI flag
3. Modify `OtlpExporter::new()` to accept trace context
4. Modify `start_root_span()` to set remote parent
5. Integrate with `src/tracer.rs`

### 6.3 Phase 3: REFACTOR

- Extract common parsing logic
- Add comprehensive error messages
- Document edge cases
- Performance optimization (parse once at startup)

---

## 7. Success Criteria

**Functional:**
- ✅ Parse valid W3C traceparent format (17-field validation)
- ✅ Extract from TRACEPARENT or OTEL_TRACEPARENT env vars
- ✅ Accept --trace-parent CLI flag
- ✅ Create Renacer root span as child of application span
- ✅ Propagate same trace-id across process boundary
- ✅ Reject invalid trace contexts gracefully
- ✅ Backward compatible (no context = new root trace)

**Testing:**
- ✅ 17/17 integration tests passing
- ✅ End-to-end test with Jaeger showing parent-child relationship
- ✅ Invalid context handling verified

**Documentation:**
- ✅ README updated with distributed tracing examples
- ✅ CHANGELOG entry for Sprint 33
- ✅ docs/otlp-integration.md updated

**Quality:**
- ✅ Zero clippy warnings
- ✅ All functions ≤10 complexity
- ✅ Build passes with --no-default-features

---

## 8. Implementation Phases

### Phase 1: Foundation (Days 1-2)
- Create `src/trace_context.rs` with TraceContext struct
- Write 11 unit tests (RED)
- Implement parser (GREEN)
- Add CLI flag

### Phase 2: OTLP Integration (Days 2-3)
- Modify `OtlpExporter::new()` signature
- Implement remote parent context
- Write 6 integration tests
- End-to-end testing with Jaeger

### Phase 3: Documentation (Day 3)
- Update README with examples
- Update CHANGELOG
- Update docs/otlp-integration.md
- Create end-to-end demo

---

## 9. Open Questions

**Q1: Should we support tracestate header?**
A1: Deferred to Sprint 34. tracestate is optional for basic propagation.

**Q2: What if application uses B3 format instead of W3C?**
A2: Deferred to Sprint 35. W3C is standard for OpenTelemetry.

**Q3: Should we auto-detect trace context from process environment?**
A3: YES - check TRACEPARENT and OTEL_TRACEPARENT env vars automatically.

**Q4: What if --trace-parent and TRACEPARENT both set?**
A4: CLI flag takes precedence over environment variable.

---

## 10. References

**Standards:**
- [W3C Trace Context](https://www.w3.org/TR/trace-context/) - Official W3C specification
- [W3C Trace Context Level 2](https://www.w3.org/TR/trace-context-2/) - Next version

**OpenTelemetry:**
- [OpenTelemetry Rust Context Propagation](https://docs.rs/opentelemetry/latest/opentelemetry/trace/)
- [Span Context API](https://docs.rs/opentelemetry/latest/opentelemetry/trace/struct.SpanContext.html)

**Industry Examples:**
- Google Dapper - First large-scale distributed tracing system
- Zipkin - Twitter's distributed tracing
- Jaeger - Uber's distributed tracing (now CNCF)

---

## Appendix A: Traceparent Format Examples

**Valid:**
```
00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01  ✅ Sampled
00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00  ✅ Not sampled
```

**Invalid:**
```
00-00000000000000000000000000000000-b7ad6b7169203331-01  ❌ All-zero trace-id
00-0af7651916cd43dd8448eb211c80319c-0000000000000000-01  ❌ All-zero parent-id
01-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01  ❌ Invalid version
00-INVALID-b7ad6b7169203331-01                          ❌ Malformed
```

---

**Status:** Ready for implementation
**Estimated Effort:** 3-4 days
**Dependencies:** Sprint 30 (OTLP Export) ✅ Complete
**Blocks:** Sprint 34 (Advanced Context Propagation)
