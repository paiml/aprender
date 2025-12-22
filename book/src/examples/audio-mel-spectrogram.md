# Case Study: Audio Mel Spectrogram Processing

This case study demonstrates Aprender's audio module for mel spectrogram computation, the foundation for speech recognition and voice processing.

## Overview

The audio module provides:
- **Mel Filterbank**: Whisper and TTS-compatible mel spectrogram computation
- **Resampling**: Sample rate conversion (e.g., 44.1kHz to 16kHz)
- **Validation**: Clipping detection, NaN/Inf checking
- **Streaming**: Chunked processing for real-time applications
- **Capture**: Platform-specific audio input (ALSA, CoreAudio, WASAPI)

## Basic Mel Spectrogram

```rust
use aprender::audio::mel::{MelFilterbank, MelConfig};

fn main() {
    // Create filterbank with Whisper-compatible settings
    let config = MelConfig::whisper();
    let filterbank = MelFilterbank::new(&config);

    // Generate 1 second of 440Hz sine wave at 16kHz
    let sample_rate = 16000.0;
    let freq = 440.0;
    let audio: Vec<f32> = (0..16000)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin())
        .collect();

    // Compute mel spectrogram
    let mel_spec = filterbank.compute(&audio).unwrap();

    // Output: 98 frames x 80 mel channels = 7840 values
    let n_frames = mel_spec.len() / config.n_mels;
    println!("Frames: {}, Mel channels: {}", n_frames, config.n_mels);
    println!("Total values: {}", mel_spec.len());

    // Frame calculation: (16000 - 400) / 160 + 1 = 98
    assert_eq!(n_frames, 98);
}
```

## Configuration Presets

### Whisper (Speech Recognition)

```rust
use aprender::audio::mel::MelConfig;

// OpenAI Whisper parameters
let config = MelConfig::whisper();
assert_eq!(config.n_mels, 80);        // 80 mel channels
assert_eq!(config.n_fft, 400);        // 25ms window
assert_eq!(config.hop_length, 160);   // 10ms hop
assert_eq!(config.sample_rate, 16000); // 16kHz required
```

### TTS (Text-to-Speech)

```rust
use aprender::audio::mel::MelConfig;

// VITS-style TTS parameters
let config = MelConfig::tts();
assert_eq!(config.n_mels, 80);
assert_eq!(config.n_fft, 1024);       // Larger window for TTS
assert_eq!(config.hop_length, 256);
assert_eq!(config.sample_rate, 22050);
```

### Custom Configuration

```rust
use aprender::audio::mel::MelConfig;

let config = MelConfig::custom(
    128,    // n_mels
    2048,   // n_fft
    512,    // hop_length
    48000,  // sample_rate
    20.0,   // fmin (Hz)
    20000.0 // fmax (Hz)
);
```

## Sample Rate Conversion

```rust
use aprender::audio::resample::resample;

// Convert from 44.1kHz to 16kHz (Whisper requirement)
let samples_44k: Vec<f32> = (0..44100)
    .map(|i| (i as f32 / 44100.0).sin())
    .collect();

let samples_16k = resample(&samples_44k, 44100, 16000).unwrap();

// Output length: ceil(44100 * 16000 / 44100) = 16000
println!("Original: {} samples", samples_44k.len());
println!("Resampled: {} samples", samples_16k.len());
```

## Audio Validation

### Clipping Detection

```rust
use aprender::audio::mel::detect_clipping;

// Audio with clipping
let samples = vec![0.5, 0.8, 1.5, -0.3, -1.2, 0.9];

let report = detect_clipping(&samples);
println!("Has clipping: {}", report.has_clipping);
println!("Positive clipped: {}", report.positive_clipped);
println!("Negative clipped: {}", report.negative_clipped);
println!("Max value: {:.2}", report.max_value);
println!("Min value: {:.2}", report.min_value);
println!("Clipping %: {:.1}%", report.clipping_percentage());

// Output:
// Has clipping: true
// Positive clipped: 1
// Negative clipped: 1
// Max value: 1.50
// Min value: -1.20
// Clipping %: 33.3%
```

### NaN and Infinity Detection

```rust
use aprender::audio::mel::{has_nan, has_inf, validate_audio};

// Check for invalid values
let samples = vec![0.5, f32::NAN, 0.3];
assert!(has_nan(&samples));

let samples = vec![0.5, f32::INFINITY, 0.3];
assert!(has_inf(&samples));

// Full validation (clipping + NaN + Inf + empty)
let valid_samples = vec![0.5, -0.3, 0.8];
assert!(validate_audio(&valid_samples).is_ok());

let invalid_samples = vec![0.5, 1.5, -0.3]; // Clipping
assert!(validate_audio(&invalid_samples).is_err());
```

## Stereo to Mono Conversion

```rust
use aprender::audio::mel::stereo_to_mono;

// Interleaved stereo: [L0, R0, L1, R1, ...]
let stereo = vec![0.8, 0.6, 0.4, 0.2, 0.0, -0.2];

let mono = stereo_to_mono(&stereo);

// Output: [(0.8+0.6)/2, (0.4+0.2)/2, (0.0-0.2)/2]
//       = [0.7, 0.3, -0.1]
assert_eq!(mono.len(), 3);
println!("Mono samples: {:?}", mono);
```

## Streaming Audio Processing

```rust
use aprender::audio::stream::{AudioChunker, ChunkConfig};

// Configure for real-time processing
let config = ChunkConfig {
    chunk_size: 16000 * 5,  // 5 seconds at 16kHz
    overlap: 8000,          // 0.5 second overlap
    sample_rate: 16000,
};

let mut chunker = AudioChunker::new(config);

// Simulate incoming audio stream
for _ in 0..10 {
    // Receive 1 second of audio
    let incoming: Vec<f32> = vec![0.0; 16000];
    chunker.push(&incoming);

    // Check for complete chunks
    while let Some(chunk) = chunker.pop() {
        println!("Processing chunk: {} samples", chunk.len());
        // Process chunk with mel filterbank...
    }
}

// Flush remaining audio at end of stream
let remaining = chunker.flush();
if !remaining.is_empty() {
    println!("Final partial chunk: {} samples", remaining.len());
}
```

## Real-Time Chunk Configuration

```rust
use aprender::audio::stream::ChunkConfig;

// Default: 30-second chunks (batch processing)
let batch_config = ChunkConfig::default();
assert_eq!(batch_config.chunk_duration_ms(), 30000);

// Real-time: 5-second chunks (low latency)
let realtime_config = ChunkConfig::realtime();
assert_eq!(realtime_config.chunk_duration_ms(), 5000);
```

## Complete ASR Preprocessing Pipeline

```rust
use aprender::audio::mel::{MelFilterbank, MelConfig, validate_audio, stereo_to_mono};
use aprender::audio::resample::resample;

fn preprocess_for_whisper(
    audio: &[f32],
    sample_rate: u32,
    is_stereo: bool,
) -> Result<Vec<f32>, String> {
    // Step 1: Convert stereo to mono
    let mono = if is_stereo {
        stereo_to_mono(audio)
    } else {
        audio.to_vec()
    };

    // Step 2: Validate audio
    validate_audio(&mono)
        .map_err(|e| format!("Audio validation failed: {}", e))?;

    // Step 3: Resample to 16kHz
    let resampled = resample(&mono, sample_rate, 16000)
        .map_err(|e| format!("Resampling failed: {}", e))?;

    // Step 4: Compute mel spectrogram
    let config = MelConfig::whisper();
    let filterbank = MelFilterbank::new(&config);

    let mel_spec = filterbank.compute(&resampled)
        .map_err(|e| format!("Mel computation failed: {}", e))?;

    Ok(mel_spec)
}

fn main() {
    // Example: 1 second of 440Hz stereo at 44.1kHz
    let left: Vec<f32> = (0..44100)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect();
    let right = left.clone();

    // Interleave for stereo
    let stereo: Vec<f32> = left.into_iter()
        .zip(right.into_iter())
        .flat_map(|(l, r)| vec![l, r])
        .collect();

    // Preprocess
    let mel = preprocess_for_whisper(&stereo, 44100, true).unwrap();

    // Ready for Whisper model!
    let n_frames = mel.len() / 80;
    println!("Mel spectrogram: {} frames x 80 channels", n_frames);
}
```

## Mel Scale Utilities

```rust
use aprender::audio::mel::MelFilterbank;

// Convert between Hz and mel scale
let hz = 1000.0;
let mel = MelFilterbank::hz_to_mel(hz);
let recovered_hz = MelFilterbank::mel_to_hz(mel);

println!("1000 Hz = {:.1} mel", mel);
println!("Roundtrip: {:.1} Hz", recovered_hz);

// The mel scale is approximately linear below 1000 Hz
// and logarithmic above 1000 Hz
for freq in [100, 500, 1000, 2000, 4000, 8000] {
    let mel = MelFilterbank::hz_to_mel(freq as f32);
    println!("{:5} Hz = {:6.1} mel", freq, mel);
}
```

## Filterbank Inspection

```rust
use aprender::audio::mel::{MelFilterbank, MelConfig};

let config = MelConfig::whisper();
let filterbank = MelFilterbank::new(&config);

// Inspect filterbank properties
println!("Mel channels: {}", filterbank.n_mels());
println!("FFT size: {}", filterbank.n_fft());
println!("Frequency bins: {}", filterbank.n_freqs());
println!("Hop length: {}", filterbank.hop_length());
println!("Sample rate: {} Hz", filterbank.sample_rate());

// Calculate frames for given audio length
let audio_samples = 16000 * 10; // 10 seconds
let n_frames = filterbank.num_frames(audio_samples);
println!("10 seconds = {} frames", n_frames);
```

## Audio Capture (Linux ALSA)

```rust,ignore
// Requires: cargo add aprender --features audio-alsa
use aprender::audio::capture::{AlsaBackend, CaptureBackend, CaptureConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // List available devices
    let devices = AlsaBackend::list_devices()?;
    for device in &devices {
        println!("{}: {} (default: {})",
            device.id, device.name, device.is_default);
    }

    // Open default capture device
    let config = CaptureConfig::whisper();
    let mut backend = AlsaBackend::open(None, &config)?;

    // Capture 1 second of audio
    let mut buffer = vec![0.0f32; 16000];
    let n = backend.read(&mut buffer)?;
    println!("Captured {} samples", n);

    backend.close()?;
    Ok(())
}
```

## Running the Examples

```bash
# Mel spectrogram (no extra features needed)
cargo run --features audio --example mel_spectrogram

# Audio capture (Linux only)
cargo run --features audio-alsa --example audio_capture
```

## Feature Flags

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `audio` | Mel spectrogram, resampling | rustfft, thiserror |
| `audio-capture` | Base capture infrastructure | audio |
| `audio-alsa` | Linux ALSA capture | alsa (C library) |
| `audio-playback` | Audio output (stub) | audio |
| `audio-codec` | Format decoding (stub) | audio |

## Test Coverage

The audio module includes comprehensive tests:
- 40+ unit tests for mel spectrogram computation
- Property-based tests for mel scale conversion
- Edge case tests (empty audio, short audio, clipping)
- Validation tests (NaN, Infinity, clipping detection)
- Streaming/chunking tests with overlap handling

## References

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [librosa](https://librosa.org/) - Python audio analysis library (reference implementation)
- [VITS](https://github.com/jaywalnut310/vits) - TTS system mel configuration
