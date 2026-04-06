# ComputeBrick Tracing

This chapter covers renacer's integration with trueno's ComputeBrick infrastructure,
enabling automatic escalation from measurement to deep tracing when performance
anomalies are detected.

## Overview

The `brick_tracer` module provides syscall-level visibility into ComputeBrick
execution when cbtop detects performance problems:

1. **cbtop** measures ComputeBrick performance (CV%, efficiency)
2. When **CV > 15%** or **efficiency < 25%**, escalate to renacer
3. **renacer** captures syscall breakdown for root cause analysis
4. Export spans to Jaeger/Tempo for visualization

This follows Mace et al. (2015) Pivot Tracing: "Always-on tracing for high-frequency
events degrades system throughput significantly." We only trace when anomalies are detected.

## Key Structures

### BrickEscalationThresholds

Controls when tracing is triggered:

```rust
use renacer::brick_tracer::BrickEscalationThresholds;

let thresholds = BrickEscalationThresholds::default()
    .with_cv(15.0)        // CV threshold (default: 15%)
    .with_efficiency(25.0) // Efficiency threshold (default: 25%)
    .with_rate_limit(100); // Max traces per second
```

### BrickTracer

The main entry point for brick tracing:

```rust
use renacer::brick_tracer::BrickTracer;

// Local tracer (no OTLP export)
let tracer = BrickTracer::new_local();

// With OTLP export to Jaeger/Tempo
let tracer = BrickTracer::new("http://localhost:4317")?;
```

### SyscallBreakdown

Categorizes time spent in different syscall categories:

- `mmap_us` - Memory allocation (mmap, munmap, mprotect, brk)
- `futex_us` - Thread synchronization
- `ioctl_us` - Device control (CUDA driver)
- `read_us` - Read operations
- `write_us` - Write operations
- `other_us` - Other syscalls
- `compute_us` - Actual computation (total - syscall overhead)

## Basic Usage

### Check if Tracing Should Occur

```rust
use renacer::brick_tracer::BrickTracer;

let tracer = BrickTracer::new_local();

// From cbtop measurements
let cv_percent = 18.5;      // Measured CV
let efficiency = 22.0;      // Measured efficiency

if tracer.should_trace(cv_percent, efficiency) {
    // Escalate to deep tracing
}
```

### Trace a Brick Execution

```rust
use renacer::brick_tracer::BrickTracer;

let tracer = BrickTracer::new_local();

let result = tracer.trace("MatMulBrick", 1000, || {
    // Your compute work here
    perform_matrix_multiply()
});

println!("Duration: {} us", result.duration_us);
println!("Efficiency: {:.2}%", result.metadata.unwrap().efficiency * 100.0);
println!("Dominant syscall: {}", result.syscall_breakdown.dominant_syscall());
```

### Get Escalation Reason

```rust
use renacer::brick_tracer::{BrickTracer, EscalationReason};

let tracer = BrickTracer::new_local();

match tracer.escalation_reason(cv_percent, efficiency) {
    EscalationReason::CvExceeded => println!("High variance detected"),
    EscalationReason::EfficiencyLow => println!("Low efficiency detected"),
    EscalationReason::Both => println!("Both CV and efficiency problematic"),
    EscalationReason::Manual => println!("Manual trace requested"),
}
```

## Integration with cbtop

The typical workflow:

```rust
use renacer::brick_tracer::{BrickTracer, BrickEscalationThresholds};

// Create tracer with thresholds matching cbtop config
let thresholds = BrickEscalationThresholds::default()
    .with_cv(15.0)
    .with_efficiency(25.0);
let tracer = BrickTracer::new("http://localhost:4317")?
    .with_thresholds(thresholds);

// Called from cbtop when brick metrics are collected
fn on_brick_measured(
    tracer: &BrickTracer,
    brick_name: &str,
    budget_us: u64,
    cv_percent: f64,
    efficiency: f64,
) {
    if tracer.should_trace(cv_percent, efficiency) {
        let reason = tracer.escalation_reason(cv_percent, efficiency);

        // Re-run brick with tracing
        let result = tracer.trace_with_reason(brick_name, budget_us, reason, || {
            execute_brick()
        });

        // Log syscall breakdown for diagnosis
        let breakdown = &result.syscall_breakdown;
        if breakdown.futex_us > breakdown.compute_us {
            eprintln!("WARNING: Thread contention dominates compute");
        }
        if breakdown.mmap_us > 100 {
            eprintln!("WARNING: Memory allocation overhead detected");
        }
    }
}
```

## OTLP Export

When using the OTLP-enabled tracer, spans are exported to Jaeger or Grafana Tempo:

```bash
# Start Jaeger
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/jaeger:2.6.0

# Run your application with brick tracing
cargo run --example brick_trace_demo

# View traces at http://localhost:16686
```

Exported spans include:

- `brick.name` - Brick identifier
- `brick.budget_us` - Expected execution time
- `brick.actual_us` - Actual execution time
- `brick.efficiency` - Efficiency percentage
- `brick.over_budget` - Whether budget was exceeded
- `syscall.overhead_percent` - Syscall overhead
- `syscall.dominant` - Dominant syscall category
- `escalation.reason` - Why tracing was triggered

## Scientific Foundations

- **Popper (1959)**: Falsifiable assertions - brick claims must be testable
- **Sigelman et al. (2010)**: Dapper adaptive sampling for high-frequency tracing
- **Mace et al. (2015)**: Pivot Tracing - trace only anomalies
- **Curtsinger & Berger (2013)**: CV < 5% indicates stable measurements
- **Williams et al. (2009)**: Roofline efficiency for bottleneck detection

## Toyota Way Alignment

- **Genchi Genbutsu**: Trace real ComputeBrick execution, not simulated workloads
- **Jidoka**: Stop-the-line when brick assertions fail
- **Muda Elimination**: Only trace when anomaly detected
- **Mieruka**: Visual brick timeline in Jaeger/Tempo

## Running the Example

```bash
cargo run --example brick_trace_demo
```

Output:

```
=== Renacer Brick Tracing Demo ===

1. Testing escalation thresholds

   Default thresholds: CV > 15%, efficiency < 25%
   Should trace (CV=20%, eff=80%): true
   Should trace (CV=5%, eff=10%): true
   Should trace (CV=5%, eff=80%): false

2. Tracing compute-bound brick

   Result: 499999500000
   Duration: 7041 us
   Budget: 1000 us
   Efficiency: 14.20%
   Over budget: true
   Dominant syscall: none
...
```
