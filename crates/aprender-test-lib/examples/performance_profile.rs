//! Example: Performance Profiling (Feature 10)
//!
//! Demonstrates: Capturing performance metrics during test execution
//!
//! Run with: `cargo run --example performance_profile`
//!
//! Toyota Way: Muda (Waste Elimination) - Identify performance bottlenecks

use jugar_probar::prelude::*;

fn main() -> ProbarResult<()> {
    println!("=== Performance Profiling Example ===\n");

    // 1. Create performance profiler
    println!("1. Creating performance profiler...");
    let mut profiler = PerformanceProfiler::new("game_test");

    println!("   Profiler created for 'game_test'");
    println!("   Target frame time: {:.2}ms (60 FPS)", 1000.0 / 60.0);

    // 2. Metric types
    println!("\n2. Metric types...");
    let metric_names = [
        "FrameTime",
        "ScriptTime",
        "LayoutTime",
        "PaintTime",
        "NetworkTime",
    ];

    for name in &metric_names {
        println!("   {}", name);
    }

    // 3. Create measurements
    println!("\n3. Recording measurements...");
    let measurements = vec![
        Measurement::new(MetricType::FrameTime, "frame_1", 16.5, "ms"),
        Measurement::new(MetricType::FrameTime, "frame_2", 16.2, "ms"),
        Measurement::new(MetricType::FrameTime, "frame_3", 17.1, "ms"),
        Measurement::new(MetricType::FrameTime, "frame_4", 15.8, "ms"),
        Measurement::new(MetricType::FrameTime, "frame_5", 16.7, "ms"),
    ];

    for m in &measurements {
        println!("   {}: {:.2}{}", m.name, m.value, m.unit);
        profiler.record(m.clone());
    }

    // 4. Calculate manual statistics
    println!("\n4. Calculating statistics...");
    let values: Vec<f64> = measurements.iter().map(|m| m.value).collect();
    let count = values.len();
    let sum: f64 = values.iter().sum();
    let mean = sum / count as f64;
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("   Count: {}", count);
    println!("   Min: {:.2}ms", min);
    println!("   Max: {:.2}ms", max);
    println!("   Mean: {:.2}ms", mean);
    println!("   FPS: {:.1}", 1000.0 / mean);

    // 5. Performance thresholds
    println!("\n5. Performance thresholds...");
    let threshold = PerformanceThreshold::new("frame_budget");

    println!("   Threshold: {}", threshold.name);

    // 6. Create performance monitor
    println!("\n6. Creating performance monitor...");
    let mut monitor = PerformanceMonitor::new();

    // Record some frame times
    for i in 0..10 {
        let frame_time = 16.0 + (i as f64 * 0.5);
        monitor.record_frame_time(frame_time);
    }

    println!("   Frames recorded: {}", monitor.frame_count());
    println!("   Current FPS: {:.1}", monitor.current_fps());

    // 7. Check frame budget
    println!("\n7. Frame budget analysis...");
    let budget_ms = 16.67; // 60 FPS target
    let over_budget = values.iter().filter(|&&v| v > budget_ms).count();

    println!("   Target: {:.2}ms (60 FPS)", budget_ms);
    println!("   Frames over budget: {}/{}", over_budget, values.len());
    println!(
        "   Budget compliance: {:.1}%",
        (1.0 - over_budget as f64 / values.len() as f64) * 100.0
    );

    // 8. Performance profile
    println!("\n8. Creating performance profile...");
    let profile = PerformanceProfile::new("test_run");

    println!("   Profile: {}", profile.test_name);

    // 9. Summary
    println!("\n9. Performance analysis summary...");
    println!("   Test: game_test");
    println!("   Frames analyzed: {}", measurements.len());
    println!("   Average frame time: {:.2}ms", mean);
    println!("   Target: 60 FPS (16.67ms)");
    println!(
        "   Compliance: {:.1}%",
        (1.0 - over_budget as f64 / values.len() as f64) * 100.0
    );

    println!("\n✅ Performance profiling example completed!");
    Ok(())
}
