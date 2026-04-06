# Performance Profiling

> **Toyota Way**: Muda (Waste Elimination) - Identify performance bottlenecks

Capture performance metrics during test execution for optimization and regression detection.

## Running the Example

```bash
cargo run --example performance_profile
```

## Quick Start

```rust
use probar::performance::{PerformanceProfiler, MetricType};

// Create a profiler
let mut profiler = PerformanceProfiler::new();

// Start profiling
profiler.start();

// Record measurements
profiler.measure("page_load", MetricType::Duration, 250.0);
profiler.measure("frame_time", MetricType::Duration, 16.67);

// Get summary
let summary = profiler.summary();
println!("Average frame time: {}ms", summary.average("frame_time"));
```

## Performance Profiler

```rust
use probar::performance::{PerformanceProfiler, PerformanceProfilerBuilder};

// Build with configuration
let profiler = PerformanceProfilerBuilder::new()
    .with_name("game_test")
    .with_sample_rate(60.0)  // 60 samples per second
    .capture_memory(true)
    .capture_cpu(true)
    .capture_gpu(true)
    .build();

// Or use defaults
let default = PerformanceProfiler::new();
```

## Metric Types

```rust
use probar::performance::MetricType;

// Available metric types
let types = [
    MetricType::Duration,    // Time measurements (ms)
    MetricType::Count,       // Counters
    MetricType::Gauge,       // Current values
    MetricType::Rate,        // Per-second rates
    MetricType::Percent,     // Percentages (0-100)
    MetricType::Bytes,       // Memory sizes
];

// Use appropriate types
// profiler.measure("render_time", MetricType::Duration, 8.5);
// profiler.measure("draw_calls", MetricType::Count, 150.0);
// profiler.measure("fps", MetricType::Rate, 60.0);
// profiler.measure("cpu_usage", MetricType::Percent, 45.0);
// profiler.measure("heap_size", MetricType::Bytes, 52428800.0);
```

## Recording Measurements

```rust
use probar::performance::{PerformanceProfiler, MetricType, Measurement};

let mut profiler = PerformanceProfiler::new();

// Single measurement
profiler.measure("startup_time", MetricType::Duration, 450.0);

// Multiple measurements for the same metric
for frame in 0..100 {
    let frame_time = 16.0 + (frame % 5) as f64;  // Simulate variation
    profiler.measure("frame_time", MetricType::Duration, frame_time);
}

// Measurements with metadata
let measurement = Measurement::new("api_call", MetricType::Duration, 125.0)
    .with_tag("endpoint", "/api/users")
    .with_tag("method", "GET");
profiler.record(measurement);
```

## Performance Monitor

```rust
use probar::performance::PerformanceMonitor;

// Continuous monitoring
let monitor = PerformanceMonitor::new();

// Start monitoring
monitor.start();

// ... run game/test ...

// Get current metrics
let metrics = monitor.current_metrics();
println!("FPS: {}", metrics.fps);
println!("Frame time: {}ms", metrics.frame_time_ms);
println!("Memory: {} MB", metrics.memory_mb);

// Stop monitoring
monitor.stop();
```

## Performance Summary

```rust
use probar::performance::{PerformanceProfiler, PerformanceSummary};

let profiler = PerformanceProfiler::new();
// ... record measurements ...

let summary = profiler.summary();

// Access statistics
println!("Total duration: {}ms", summary.total_duration_ms);
println!("Measurements: {}", summary.measurement_count);

// Get metric statistics
if let Some(stats) = summary.get_stats("frame_time") {
    println!("Frame time statistics:");
    println!("  Min: {}ms", stats.min);
    println!("  Max: {}ms", stats.max);
    println!("  Average: {}ms", stats.average);
    println!("  Median: {}ms", stats.median);
    println!("  P95: {}ms", stats.p95);
    println!("  P99: {}ms", stats.p99);
    println!("  Std dev: {}ms", stats.std_dev);
}
```

## Metric Statistics

```rust
use probar::performance::MetricStats;

// Statistics for a metric
let stats = MetricStats {
    count: 1000,
    min: 14.5,
    max: 32.1,
    sum: 16500.0,
    average: 16.5,
    median: 16.2,
    p95: 18.5,
    p99: 24.0,
    std_dev: 2.3,
};

// Check against thresholds
let threshold = 20.0;  // 20ms frame time budget
if stats.p95 > threshold {
    println!("WARNING: 5% of frames exceed {}ms budget", threshold);
}
```

## Performance Thresholds

```rust
use probar::performance::{PerformanceThreshold, PerformanceProfiler};

// Define thresholds
let thresholds = vec![
    PerformanceThreshold::new("frame_time")
        .max_average(16.67)  // 60 FPS
        .max_p95(20.0)
        .max_p99(33.33),     // Never drop below 30 FPS

    PerformanceThreshold::new("startup_time")
        .max_value(500.0),   // 500ms max startup

    PerformanceThreshold::new("memory_mb")
        .max_value(256.0),   // 256 MB limit
];

// Validate against thresholds
let profiler = PerformanceProfiler::new();
// ... record measurements ...

for threshold in &thresholds {
    let result = profiler.validate_threshold(threshold);
    if !result.passed {
        println!("FAILED: {} - {}", threshold.metric_name, result.reason);
    }
}
```

## Performance Profile

```rust
use probar::performance::PerformanceProfile;

// Create a performance profile
let profile = PerformanceProfile::new("game_benchmark")
    .with_duration_secs(60)
    .with_warmup_secs(5);

// Run profiled code
// profile.run(|| {
//     // Game loop or test code
// })?;

// Get results
// let results = profile.results();
// println!("Sustained FPS: {}", results.sustained_fps);
// println!("Frame drops: {}", results.frame_drops);
```

## Frame Time Analysis

```rust
use probar::performance::PerformanceProfiler;

fn analyze_frame_times(profiler: &PerformanceProfiler) {
    if let Some(stats) = profiler.summary().get_stats("frame_time") {
        // 60 FPS target = 16.67ms per frame
        let target_60fps = 16.67;

        // Check consistency
        let jitter = stats.max - stats.min;
        println!("Frame time jitter: {}ms", jitter);

        // Check percentiles
        if stats.p99 < target_60fps {
            println!("Excellent: 99% of frames at 60+ FPS");
        } else if stats.p95 < target_60fps {
            println!("Good: 95% of frames at 60+ FPS");
        } else if stats.average < target_60fps {
            println!("Fair: Average at 60+ FPS but with spikes");
        } else {
            println!("Poor: Cannot maintain 60 FPS");
        }
    }
}
```

## Memory Profiling

```rust
use probar::performance::{PerformanceProfiler, MetricType};

fn profile_memory(profiler: &mut PerformanceProfiler) {
    // Record memory at different points
    profiler.measure("memory_startup", MetricType::Bytes, 50_000_000.0);

    // After loading assets
    profiler.measure("memory_loaded", MetricType::Bytes, 120_000_000.0);

    // During gameplay
    profiler.measure("memory_gameplay", MetricType::Bytes, 150_000_000.0);

    // Check for leaks
    let startup = 50_000_000.0;
    let current = 150_000_000.0;
    let growth = current - startup;
    println!("Memory growth: {} MB", growth / 1_000_000.0);
}
```

## Export Results

```rust
use probar::performance::PerformanceProfiler;

fn export_results(profiler: &PerformanceProfiler) {
    let summary = profiler.summary();

    // Export to JSON
    // let json = serde_json::to_string_pretty(&summary)?;
    // fs::write("performance_results.json", json)?;

    // Print summary
    println!("=== Performance Summary ===");
    for (metric, stats) in summary.all_stats() {
        println!("{}: avg={:.2}, p95={:.2}, p99={:.2}",
            metric, stats.average, stats.p95, stats.p99);
    }
}
```

## Regression Detection

```rust
use probar::performance::{PerformanceProfiler, MetricStats};

fn check_regression(current: &MetricStats, baseline: &MetricStats) -> bool {
    // Allow 10% regression
    let threshold = 1.1;

    if current.average > baseline.average * threshold {
        println!("REGRESSION: Average increased by {:.1}%",
            (current.average / baseline.average - 1.0) * 100.0);
        return true;
    }

    if current.p99 > baseline.p99 * threshold {
        println!("REGRESSION: P99 increased by {:.1}%",
            (current.p99 / baseline.p99 - 1.0) * 100.0);
        return true;
    }

    false
}
```

## Best Practices

1. **Warmup Period**: Always exclude warmup from measurements
2. **Multiple Runs**: Average across multiple test runs
3. **Consistent Environment**: Control for background processes
4. **Percentiles**: Use P95/P99 for user experience, not just averages
5. **Thresholds**: Set clear pass/fail criteria
6. **Baseline**: Compare against known-good baselines
7. **Memory**: Monitor for leaks over time
