# GIF Recording

> **Toyota Way**: Mieruka (Visibility) - Animated test recordings

Record animated GIF recordings of test execution for visual review and debugging.

## Basic Usage

```rust
use probar::media::{GifConfig, GifRecorder, GifFrame};

let config = GifConfig::new(320, 240);
let mut recorder = GifRecorder::new(config);

// Add frames during test execution
for screenshot in screenshots {
    let frame = GifFrame::new(screenshot.pixels, 100); // 100ms delay
    recorder.add_frame(frame);
}

let gif_data = recorder.encode()?;
```
