# Streaming UX Validation

Probar provides comprehensive validation for real-time streaming interfaces including audio/video players, live transcription, and interactive media applications.

## Overview

The streaming validation system (PROBAR-SPEC-011) includes:

- **StreamingUxValidator**: Latency, buffer, and FPS monitoring
- **VuMeterConfig**: Audio level validation with staleness detection
- **TestExecutionStats**: Compression monitoring for state capture
- **ScreenshotContent**: Entropy-based content classification

## StreamingUxValidator

Validates real-time audio/video streaming interfaces.

```rust
use jugar_probar::validators::{StreamingUxValidator, StreamingMetric};
use std::time::Duration;

// Create validator with audio streaming requirements
let mut validator = StreamingUxValidator::for_audio();

// Start tracking
validator.start();

// Record metrics during playback
validator.record_metric(StreamingMetric::FirstByteReceived);
validator.record_metric(StreamingMetric::AudioChunk {
    samples: 1024,
    sample_rate: 16000,
});
validator.record_metric(StreamingMetric::Latency(Duration::from_millis(50)));

// Validate results
match validator.validate() {
    Ok(result) => println!("Passed! FPS: {:.1}", result.average_fps),
    Err(e) => println!("Failed: {}", e),
}
```

### Presets

| Preset | Max Latency | Underrun Threshold | Min FPS |
|--------|-------------|-------------------|---------|
| `for_audio()` | 100ms | 3 | - |
| `for_video()` | 500ms | - | 30 |
| `new()` (default) | 200ms | 5 | 24 |

### Streaming Metrics

```rust
pub enum StreamingMetric {
    Latency(Duration),
    FrameRendered { timestamp: u64 },
    FrameDropped,
    BufferUnderrun,
    FirstByteReceived,
    BufferLevel(f32),
    AudioChunk { samples: usize, sample_rate: u32 },
}
```

### State Machine

The validator tracks streaming state transitions:

```
Idle -> Buffering -> Streaming <-> Stalled -> Completed
                  \-> Error
```

## VU Meter Validation

Validates audio level indicators (VU meters) for responsiveness and accuracy.

```rust
use jugar_probar::validators::{VuMeterConfig, VuMeterSample};

let config = VuMeterConfig::default()
    .with_min_level(0.1)
    .with_max_level(0.9)
    .with_update_rate_hz(30.0)
    .with_max_stale_ms(100);

// Validate a sample
match config.validate_sample(0.5) {
    Ok(()) => println!("Level OK"),
    Err(e) => println!("Error: {}", e),
}
```

### Error Types

| Error | Description |
|-------|-------------|
| `NegativeLevel` | Level is below 0.0 |
| `Clipping` | Level exceeds max + tolerance |
| `Stale` | No updates within threshold |
| `SlowUpdateRate` | Updates slower than expected |
| `NotAnimating` | Constant value detected |

## Test Execution Stats

Tracks compression efficiency during test runs, based on trueno-zram patterns.

```rust
use jugar_probar::validators::TestExecutionStats;

let mut stats = TestExecutionStats::new();
stats.start();

// Record state captures
stats.record_state_capture(4096, 1024);  // Raw, Compressed
stats.record_state_capture(4096, 512);   // High compression

stats.stop();

println!("Compression ratio: {:.2}x", stats.compression_ratio());
println!("Efficiency: {:.1}%", stats.efficiency() * 100.0);
println!("Same-fill pages: {}", stats.same_fill_pages);
```

### Metrics

- `compression_ratio()`: Raw bytes / compressed bytes
- `efficiency()`: 1 - (compressed / raw)
- `storage_savings_mb()`: Savings in megabytes
- `same_fill_ratio()`: Ratio of highly compressible pages

## Screenshot Classification

Classify screenshot content for optimal compression strategy.

```rust
use jugar_probar::validators::ScreenshotContent;

let pixels: Vec<u8> = capture_screenshot();
let content = ScreenshotContent::classify(&pixels);

match content {
    ScreenshotContent::Uniform { fill_value } => {
        println!("Blank screen, use RLE");
    }
    ScreenshotContent::UiDominated { entropy } => {
        println!("UI content, use PNG (entropy: {:.2})", entropy);
    }
    ScreenshotContent::GameWorld { entropy } => {
        println!("Game content, use Zstd (entropy: {:.2})", entropy);
    }
    ScreenshotContent::HighEntropy { entropy } => {
        println!("High entropy, use LZ4 (entropy: {:.2})", entropy);
    }
}
```

### Classification Thresholds

| Type | Entropy Range | Recommended Algorithm |
|------|---------------|----------------------|
| Uniform | N/A (>95% same value) | RLE |
| UI-Dominated | < 3.0 | PNG |
| Game World | 3.0 - 6.0 | Zstd |
| High Entropy | > 6.0 | LZ4 |

## Example: Complete Streaming Test

```rust
use jugar_probar::validators::*;
use std::time::Duration;

async fn test_streaming_ui() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize validators
    let mut stream_validator = StreamingUxValidator::for_audio();
    let vu_config = VuMeterConfig::default();
    let mut stats = TestExecutionStats::new();

    stats.start();
    stream_validator.start();

    // Simulate streaming session
    for i in 0..100 {
        // Record audio chunk
        stream_validator.record_metric(StreamingMetric::AudioChunk {
            samples: 1024,
            sample_rate: 16000,
        });

        // Validate VU meter
        let level = get_vu_level().await?;
        vu_config.validate_sample(level)?;

        // Record state for compression stats
        let state = capture_game_state();
        stats.record_state_capture(state.raw_size, state.compressed_size);
    }

    stream_validator.complete();
    stats.stop();

    // Validate all metrics
    let result = stream_validator.validate()?;
    assert!(result.buffer_underruns < 3);
    assert!(stats.efficiency() > 0.5);

    Ok(())
}
```

## Running the Demo

```bash
cargo run --example streaming_ux_demo -p jugar-probar
```

## See Also

- [Performance Profiling](./performance-profiling.md)
- [Media Recording](./media-recording.md)
- [Accessibility Testing](./accessibility.md)
