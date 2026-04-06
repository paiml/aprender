# MP4 Video

> **Toyota Way**: Genchi Genbutsu (Go and See) - Full motion capture of tests

Record full motion MP4 video of test execution with configurable quality settings.

## Basic Usage

```rust
use probar::media::{VideoConfig, VideoRecorder, VideoCodec};

let config = VideoConfig::new(640, 480)
    .with_fps(30)
    .with_bitrate(2_000_000)
    .with_codec(VideoCodec::H264);

let mut recorder = VideoRecorder::new(config);
recorder.start()?;

// Capture frames during test
for frame in frames {
    recorder.capture_raw_frame(&pixels, width, height, timestamp_ms)?;
}

let video_data = recorder.stop()?;
```
