# ADR-004: Toyota Production System Principles

## Status

Accepted

## Context

Software quality in simulation engines is critical - incorrect results can lead to:
- Wrong scientific conclusions
- Engineering failures
- Wasted computational resources

We need a systematic approach to quality that:
1. Prevents defects at the source
2. Stops propagation of errors
3. Enables continuous improvement
4. Reduces waste in development

## Decision

We adopt principles from the **Toyota Production System (TPS)** adapted for software:

### 1. Jidoka (自働化) - Autonomation

**Principle**: Stop the line when defects occur.

**Implementation**:
```rust
// Jidoka guard - halt on invariant violation
fn jidoka_check(state: &SimState) -> Result<(), JidokaError> {
    if state.energy.is_nan() {
        return Err(JidokaError::NaN("energy"));
    }
    if (state.energy - state.initial_energy).abs() > DRIFT_THRESHOLD {
        return Err(JidokaError::EnergyDrift(state.energy));
    }
    Ok(())
}
```

**Metrics**:
- Energy drift threshold: 1e-9
- Angular momentum threshold: 1e-12
- NaN/Inf detection: immediate halt

### 2. Poka-Yoke (ポカヨケ) - Mistake-Proofing

**Principle**: Design systems that prevent errors.

**Implementation**:
```rust
// Type-safe units prevent unit confusion
use uom::si::f64::{Length, Mass, Time};

fn kinetic_energy(mass: Mass, velocity: Velocity) -> Energy {
    // Cannot accidentally pass Length for Mass
    0.5 * mass * velocity * velocity
}
```

**Techniques**:
- Type-safe units via `uom` crate
- Newtype wrappers for domain concepts
- Builder pattern with validation
- Const assertions for invariants

### 3. Heijunka (平準化) - Load Leveling

**Principle**: Smooth workflow to prevent overburden.

**Implementation**:
```rust
struct HeijunkaScheduler {
    budget_ms: f64,    // Time budget per frame
    used_ms: f64,      // Time consumed
    quality: Quality,  // Full, Reduced, Minimal
}

impl HeijunkaScheduler {
    fn tick(&mut self, work: impl FnOnce()) {
        let start = Instant::now();
        if self.used_ms < self.budget_ms {
            work();
        }
        self.used_ms += start.elapsed().as_secs_f64() * 1000.0;
        self.adjust_quality();
    }
}
```

### 4. Kaizen (改善) - Continuous Improvement

**Principle**: Small, incremental improvements.

**Implementation**:
- PMAT quality tracking over time
- Automated baseline comparison
- Pre-commit quality gates
- Technical debt grading

### 5. Muda (無駄) - Waste Elimination

**Principle**: Remove non-value-adding activities.

**Types of waste in simulation code**:
| Waste Type | Example | Mitigation |
|------------|---------|------------|
| Over-processing | Unnecessary precision | Adaptive tolerance |
| Waiting | Blocking I/O | Async operations |
| Defects | Runtime panics | Result types |
| Motion | Excessive allocation | Arena allocators |

### 6. Genchi Genbutsu (現地現物) - Go and See

**Principle**: Observe the actual work.

**Implementation**:
- Chrome trace export for profiling
- Flame graph visualization
- Real-time TUI dashboards
- Detailed logging with tracing

## Consequences

### Positive
- Defects caught at source (Jidoka)
- Class of bugs eliminated by types (Poka-Yoke)
- Predictable performance (Heijunka)
- Continuous quality improvement (Kaizen)
- Efficient resource usage (Muda elimination)

### Negative
- More verbose code (type safety overhead)
- Learning curve for team
- Initial development slower
- Not all principles map perfectly to software

### Metrics

| Principle | Metric | Target |
|-----------|--------|--------|
| Jidoka | Mean time to detect | < 1ms |
| Poka-Yoke | Type-related bugs | 0 |
| Heijunka | Frame budget adherence | > 95% |
| Kaizen | TDG score trend | Increasing |
| Muda | Unnecessary allocations | 0 in hot path |

## References

- Ohno, Taiichi. "Toyota Production System" (1988)
- Liker, Jeffrey. "The Toyota Way" (2004)
- Poppendieck. "Lean Software Development" (2003)
