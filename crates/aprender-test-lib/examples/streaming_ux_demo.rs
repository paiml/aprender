//! Streaming UX Validation Demo
//!
//! Demonstrates the streaming UX validators from PROBAR-SPEC-011:
//! - StreamingUxValidator for latency/buffer/FPS validation
//! - VuMeterConfig for audio level validation
//! - TestExecutionStats for compression monitoring
//! - ScreenshotContent for entropy-based classification
//!
//! Run with: cargo run --example streaming_ux_demo -p jugar-probar

use jugar_probar::validators::{
    CompressionAlgorithm, ScreenshotContent, StreamingMetric, StreamingUxValidator,
    TestExecutionStats, VuMeterConfig, VuMeterSample,
};
use std::time::Duration;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Streaming UX Validation Demo (PROBAR-SPEC-011)           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    demo_streaming_validator();
    demo_vu_meter_validation();
    demo_test_execution_stats();
    demo_screenshot_classification();
}

/// Demonstrate StreamingUxValidator for real-time audio/video
fn demo_streaming_validator() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  1. StreamingUxValidator Demo");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create validator for audio streaming (strict latency requirements)
    let mut validator = StreamingUxValidator::for_audio();
    println!("Created audio streaming validator:");
    println!("  - Max latency: 100ms");
    println!("  - Buffer underrun threshold: 3");
    println!("  - Initial state: {:?}\n", validator.state());

    // Simulate streaming session
    validator.start();
    println!("Started streaming session...");
    println!("  State: {:?}", validator.state());

    // Record first byte received
    validator.record_metric(StreamingMetric::FirstByteReceived);
    println!("  Received first byte");

    // Record audio chunks
    for i in 0..10 {
        validator.record_metric(StreamingMetric::AudioChunk {
            samples: 1024,
            sample_rate: 16000,
        });
        validator.record_metric(StreamingMetric::Latency(Duration::from_millis(50 + i * 5)));
    }
    println!("  Processed 10 audio chunks");
    println!("  State: {:?}", validator.state());

    // Record some frame renders (for video/visualization)
    for i in 0..30 {
        validator.record_metric(StreamingMetric::FrameRendered {
            timestamp: i * 33, // ~30fps
        });
    }
    println!("  Rendered 30 frames");
    println!("  Average FPS: {:.1}", validator.average_fps());

    // Simulate buffer underrun
    validator.record_metric(StreamingMetric::BufferUnderrun);
    println!("  Buffer underrun occurred!");
    println!("  State: {:?}", validator.state());

    // Recovery
    validator.record_metric(StreamingMetric::FrameRendered { timestamp: 1000 });
    println!("  Recovered from stall");
    println!("  State: {:?}", validator.state());

    // Complete session
    validator.complete();
    println!("  Session completed");

    // Validate
    match validator.validate() {
        Ok(result) => {
            println!("\n  Validation PASSED:");
            println!("    - Buffer underruns: {}", result.buffer_underruns);
            println!("    - Dropped frames: {}", result.dropped_frames);
            println!("    - Average FPS: {:.1}", result.average_fps);
            println!("    - Max latency: {:?}", result.max_latency_recorded);
            println!("    - Total frames: {}", result.total_frames);
        }
        Err(e) => {
            println!("\n  Validation FAILED: {e}");
        }
    }
    println!();
}

/// Demonstrate VuMeterConfig for audio level validation
fn demo_vu_meter_validation() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  2. VU Meter Validation Demo");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create VU meter config
    let config = VuMeterConfig::default()
        .with_min_level(0.1)
        .with_max_level(0.9)
        .with_update_rate_hz(30.0)
        .with_max_stale_ms(100);

    println!("VU Meter Configuration:");
    println!("  - Min level: {:.1}", config.min_level);
    println!("  - Max level: {:.1}", config.max_level);
    println!("  - Update rate: {:.0} Hz", config.update_rate_hz);
    println!("  - Max stale: {}ms\n", config.max_stale_ms);

    // Simulate VU meter samples
    let samples = vec![
        VuMeterSample {
            timestamp_ms: 0.0,
            level: 0.2,
        },
        VuMeterSample {
            timestamp_ms: 33.3,
            level: 0.5,
        },
        VuMeterSample {
            timestamp_ms: 66.6,
            level: 0.7,
        },
        VuMeterSample {
            timestamp_ms: 100.0,
            level: 0.4,
        },
    ];

    println!("Validating VU meter samples:");
    for sample in &samples {
        match config.validate_sample(sample.level) {
            Ok(()) => println!(
                "  [{:6.1}ms] Level {:.2} - OK",
                sample.timestamp_ms, sample.level
            ),
            Err(e) => println!(
                "  [{:6.1}ms] Level {:.2} - ERROR: {e}",
                sample.timestamp_ms, sample.level
            ),
        }
    }

    // Test edge cases
    println!("\nEdge case validation:");
    println!(
        "  Level -0.1: {:?}",
        config.validate_sample(-0.1).map_err(|e| e.to_string())
    );
    println!(
        "  Level 1.5: {:?}",
        config.validate_sample(1.5).map_err(|e| e.to_string())
    );
    println!(
        "  Level 0.95: {:?}",
        config.validate_sample(0.95).map_err(|e| e.to_string())
    );
    println!();
}

/// Demonstrate TestExecutionStats for compression monitoring
fn demo_test_execution_stats() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  3. Test Execution Stats Demo (trueno-zram pattern)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut stats = TestExecutionStats::new();
    stats.start();

    println!("Recording game state captures...\n");

    // Simulate capturing game states with various compression ratios
    let captures = [
        (4096, 1024, "Normal game state"),
        (4096, 512, "Sparse game state (high compression)"),
        (4096, 200, "Same-fill page (>90% compression)"),
        (4096, 2048, "Complex game state"),
        (4096, 100, "Nearly empty (same-fill)"),
        (8192, 4096, "Large state (50% compression)"),
    ];

    for (raw, compressed, description) in captures {
        stats.record_state_capture(raw, compressed);
        let ratio = raw as f64 / compressed as f64;
        println!("  {description}");
        println!("    Raw: {raw} bytes, Compressed: {compressed} bytes, Ratio: {ratio:.1}x");
    }

    stats.stop();

    println!("\n  Summary Statistics:");
    println!("  ─────────────────────────────────────────");
    println!("    States captured: {}", stats.states_captured);
    println!("    Total raw bytes: {} bytes", stats.bytes_raw);
    println!("    Total compressed: {} bytes", stats.bytes_compressed);
    println!("    Compression ratio: {:.2}x", stats.compression_ratio());
    println!("    Efficiency: {:.1}%", stats.efficiency() * 100.0);
    println!("    Storage savings: {:.2} MB", stats.storage_savings_mb());
    println!("    Same-fill pages: {}", stats.same_fill_pages);
    println!(
        "    Same-fill ratio: {:.1}%",
        stats.same_fill_ratio() * 100.0
    );
    println!();
}

/// Demonstrate ScreenshotContent classification
fn demo_screenshot_classification() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  4. Screenshot Content Classification Demo");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Test different content types
    let test_cases: Vec<(&str, Vec<u8>)> = vec![
        ("Uniform white (blank screen)", vec![255u8; 1000]),
        ("Uniform black (loading screen)", vec![0u8; 1000]),
        ("UI-dominated (mostly background with some elements)", {
            let mut pixels = vec![240u8; 900];
            pixels.extend(vec![50u8; 50]);
            pixels.extend(vec![100u8; 50]);
            pixels
        }),
        (
            "Game world (varied content)",
            (0..1000).map(|i| ((i * 7 + 13) % 128) as u8).collect(),
        ),
        (
            "High entropy (random noise)",
            (0..1000).map(|i| ((i * 127 + 37) % 256) as u8).collect(),
        ),
    ];

    for (description, pixels) in test_cases {
        let content = ScreenshotContent::classify(&pixels);

        let content_type = match &content {
            ScreenshotContent::Uniform { fill_value } => format!("Uniform (fill: {fill_value})"),
            ScreenshotContent::UiDominated { entropy } => {
                format!("UI-Dominated (entropy: {entropy:.2})")
            }
            ScreenshotContent::GameWorld { entropy } => {
                format!("Game World (entropy: {entropy:.2})")
            }
            ScreenshotContent::HighEntropy { entropy } => {
                format!("High Entropy (entropy: {entropy:.2})")
            }
        };

        let algorithm = match content.recommended_algorithm() {
            CompressionAlgorithm::Rle => "RLE",
            CompressionAlgorithm::Png => "PNG",
            CompressionAlgorithm::Zstd => "Zstd",
            CompressionAlgorithm::Lz4 => "LZ4",
        };

        println!("  {description}:");
        println!("    Classification: {content_type}");
        println!("    Recommended algorithm: {algorithm}");
        println!("    Expected ratio: {}", content.expected_ratio_hint());
        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Demo complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
