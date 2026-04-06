# Anti-Pattern Detection

Renacer includes an architectural anti-pattern detector that analyzes system traces to identify common performance and design issues. This feature is part of the Quality Gates (Jidoka) system specified in Section 27 of the architecture specification.

## Overview

The anti-pattern detector examines syscall traces and identifies four types of architectural anti-patterns:

| Anti-Pattern | Description | Default Threshold |
|-------------|-------------|-------------------|
| **GodProcess** | Single process dominates syscall activity | >80% of syscalls |
| **TightLoop** | Syscalls occurring at very short intervals | <10ms between calls |
| **ExcessiveIO** | I/O operations exceed sustainable rate | >10,000 ops/sec |
| **BlockingMainThread** | Long-running operations block main thread | >100ms duration |

## Quick Start

```rust,ignore
use renacer::unified_trace::UnifiedTrace;

// Analyze a trace for anti-patterns
let trace = UnifiedTrace::new(1234, "my_app".to_string());
// ... add syscalls to trace ...

if let Some(quality) = trace.architectural_quality() {
    println!("Quality Score: {:.2}", quality.score);

    for pattern in &quality.anti_patterns {
        println!("Anti-pattern: {:?}", pattern);
    }

    for recommendation in &quality.recommendations {
        println!("Recommendation: {}", recommendation);
    }
}
```

## API Reference

### AntiPatternDetector

The main entry point for anti-pattern detection:

```rust,ignore
use renacer::analysis::anti_pattern::{
    AntiPatternDetector,
    AntiPatternThresholds,
    ArchitecturalQuality,
};

// Create detector with default thresholds
let detector = AntiPatternDetector::default();

// Or customize thresholds
let thresholds = AntiPatternThresholds {
    god_process_syscall_percent: 90.0,  // More lenient
    tight_loop_threshold_ms: 5,          // More strict
    excessive_io_ops_per_sec: 5000,      // More strict
};
let detector = AntiPatternDetector::new(thresholds);

// Analyze a trace
let quality = detector.analyze(&trace);
```

### ArchitecturalQuality

The result of anti-pattern analysis:

```rust,ignore
pub struct ArchitecturalQuality {
    /// Overall quality score (0.0 - 1.0, where 1.0 is perfect)
    pub score: f64,

    /// Detected anti-patterns
    pub anti_patterns: Vec<AntiPattern>,

    /// Actionable recommendations for improvement
    pub recommendations: Vec<String>,
}
```

### AntiPattern Enum

```rust,ignore
pub enum AntiPattern {
    /// Single process dominates syscall activity
    GodProcess {
        process_id: u32,
        syscall_percent: f64
    },

    /// Syscalls occurring at very short intervals
    TightLoop {
        location: String,
        interval_ms: u64
    },

    /// Excessive I/O operations
    ExcessiveIO {
        ops_per_sec: u64
    },

    /// Main thread blocked for extended duration
    BlockingMainThread {
        duration_ms: u64
    },
}
```

## Anti-Pattern Details

### God Process

A "God Process" is detected when a single process accounts for more than the threshold percentage of all syscalls. This indicates:

- **Problem**: Single point of failure, limits parallelization
- **Symptoms**: One process doing 80%+ of all syscalls
- **Fix**: Decompose into microservices, use load balancing

**Example Detection:**
```text
Anti-pattern: GodProcess { process_id: 1234, syscall_percent: 95.0 }
Recommendation: Consider decomposing the dominant process into smaller
services or distributing work across multiple processes.
```

### Tight Loop

A "Tight Loop" is detected when syscalls occur at intervals below the threshold. This indicates polling or busy-waiting behavior:

- **Problem**: High CPU usage, wasted resources
- **Symptoms**: Same syscall repeated at <10ms intervals
- **Fix**: Use event-driven I/O, async operations, or proper sleep intervals

**Example Detection:**
```text
Anti-pattern: TightLoop { location: "syscall: futex", interval_ms: 2 }
Recommendation: Consider using vectorized I/O (readv/writev), buffering,
or async I/O to reduce syscall frequency.
```

### Excessive I/O

Detected when the rate of I/O syscalls exceeds a sustainable threshold:

- **Problem**: I/O subsystem saturation, latency spikes
- **Symptoms**: >10,000 I/O operations per second
- **Fix**: Batch operations, use buffered I/O, implement throttling

**I/O syscalls monitored:** `read`, `write`, `pread64`, `pwrite64`, `readv`, `writev`, `preadv`, `pwritev`, `sendfile`, `splice`, `tee`

**Example Detection:**
```text
Anti-pattern: ExcessiveIO { ops_per_sec: 15000 }
Recommendation: Consider batching I/O operations, using buffered I/O,
or implementing I/O throttling.
```

### Blocking Main Thread

Detected when any syscall duration exceeds 100ms, indicating blocking operations:

- **Problem**: UI freezes, request timeouts, poor responsiveness
- **Symptoms**: Single syscall taking >100ms
- **Fix**: Use async I/O, move to background thread, add timeouts

**Example Detection:**
```text
Anti-pattern: BlockingMainThread { duration_ms: 500 }
Recommendation: Consider moving blocking operations to a background
thread or using async I/O to avoid blocking the main thread.
```

## Quality Score Calculation

The quality score is calculated as:

```text
score = 1.0 - (num_anti_patterns * 0.2)
```

| Anti-Patterns Found | Quality Score |
|---------------------|---------------|
| 0 | 1.0 (Perfect) |
| 1 | 0.8 |
| 2 | 0.6 |
| 3 | 0.4 |
| 4 | 0.2 |
| 5+ | 0.0 |

## Integration with Trace Analysis

The `UnifiedTrace` type provides a convenience method:

```rust,ignore
impl UnifiedTrace {
    /// Analyze trace for architectural quality and anti-patterns
    pub fn architectural_quality(&self) -> Option<ArchitecturalQuality> {
        // Returns None for empty traces
        // Returns Some(quality) for traces with data
    }
}
```

## CI/CD Integration

Combine anti-pattern detection with build-time assertions:

```toml
# renacer.toml
[[assertion]]
name = "no_god_process"
type = "AntiPattern"
max_patterns = 0
fail_on_violation = true

[[assertion]]
name = "quality_gate"
type = "Custom"
expression = "quality_score >= 0.8"
fail_on_violation = true
```

## Toyota Way: Jidoka (Quality Gates)

This feature implements the Toyota Way principle of Jidoka - building quality into the process by detecting defects as they occur. The anti-pattern detector acts as an automated "andon cord" that can stop the build when architectural quality degrades.

**Key Principles:**
1. **Stop and Fix**: When anti-patterns are detected, fix them immediately
2. **Root Cause Analysis**: Use recommendations to address underlying issues
3. **Continuous Improvement**: Track quality scores over time
4. **Prevention**: Integrate into CI/CD to prevent regressions

## Peer-Reviewed Foundation

The anti-pattern detection algorithms are based on established research:

- **Sambasivan et al. (2016)** "So, you want to trace your distributed system?" - 5 common anti-patterns account for 70% of performance bugs
- **Gunawi et al. (2014)** "Why Does the Cloud Stop Computing?" SOSP - 60% of cloud failures from resource exhaustion patterns
- **Jeon et al. (2019)** "Analysis of Large-Scale Multi-Tenant GPU Clusters" - PCIe bandwidth saturation patterns

## Related Topics

- [Build-Time Assertions](../contributing/quality-gates.md) - CI/CD integration
- [Regression Detection](./regression-detection.md) - Statistical regression analysis
- [Sequence Mining](./sequence-mining.md) - N-gram pattern detection
- [Toyota Way Principles](../contributing/toyota-way.md) - Quality philosophy
