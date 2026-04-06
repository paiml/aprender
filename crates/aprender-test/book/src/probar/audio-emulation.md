# Audio Emulation

Probar provides deterministic audio emulation for testing streaming ASR (Automatic Speech Recognition) and audio processing applications without requiring real microphone access.

## Overview

The `AudioEmulator` mocks `navigator.mediaDevices.getUserMedia` with controlled audio sources, enabling:

- Deterministic test inputs
- No microphone permission prompts
- Reproducible audio scenarios
- CI/CD-friendly testing

## Audio Source Types

### Sine Wave

Generate pure tones at specific frequencies:

```rust
use jugar_probar::emulation::{AudioEmulator, AudioSource};

let mut emulator = AudioEmulator::new(AudioSource::SineWave {
    frequency: 440.0,  // A4 note
    amplitude: 0.5,
});

let samples = emulator.generate_samples(0.1); // 100ms of audio
```

### Speech Pattern

Generate speech-like audio with harmonics:

```rust
let emulator = AudioEmulator::new(AudioSource::SpeechPattern {
    fundamental_hz: 150.0,      // Typical male voice
    harmonics: vec![0.5, 0.3, 0.2, 0.1],
    variation_hz: 20.0,         // Pitch variation
});
```

### Silence with Noise Floor

Generate silence with optional background noise:

```rust
// Complete silence
let silent = AudioEmulator::new(AudioSource::Silence {
    noise_floor_db: -100.0,
});

// Ambient noise level
let ambient = AudioEmulator::new(AudioSource::Silence {
    noise_floor_db: -40.0,
});
```

### Custom Callback

Generate any waveform programmatically:

```rust
// Square wave at 200 Hz
let emulator = AudioEmulator::new(AudioSource::Callback(Box::new(|t| {
    if (t * 200.0 * 2.0 * std::f32::consts::PI).sin() > 0.0 {
        0.3
    } else {
        -0.3
    }
})));
```

### File-Based

Load pre-recorded audio:

```rust
let emulator = AudioEmulator::new(AudioSource::File {
    path: PathBuf::from("test_audio.wav"),
    loop_: true,
});
```

## Browser Injection

The emulator generates JavaScript to override `getUserMedia`:

```rust
let mut emulator = AudioEmulator::new(AudioSource::SineWave {
    frequency: 440.0,
    amplitude: 0.5,
});

let samples = emulator.generate_samples(0.1);
let mock_js = emulator.generate_mock_js(&samples);

// Inject into page via CDP
page.evaluate(&mock_js).await?;
```

## Testing Streaming ASR

Example test for a speech recognition application:

```rust
#[tokio::test]
async fn test_speech_recognition() {
    let browser = Browser::new().await?;
    let page = browser.new_page().await?;

    // Inject audio emulator
    let mut emulator = AudioEmulator::new(AudioSource::SpeechPattern {
        fundamental_hz: 150.0,
        harmonics: vec![0.5, 0.3, 0.2, 0.1],
        variation_hz: 20.0,
    });

    let samples = emulator.generate_samples(2.0); // 2 seconds
    let mock_js = emulator.generate_mock_js(&samples);
    page.evaluate(&mock_js).await?;

    // Navigate to app
    page.goto("http://localhost:8080").await?;

    // Start recording
    page.click("#start-recording").await?;

    // Wait for transcription
    page.wait_for_selector("#transcription").await?;

    // Verify processing occurred
    let result = page.text_content("#transcription").await?;
    assert!(!result.is_empty());
}
```

## Example

Run the audio emulation demo:

```bash
cargo run --example audio_emulation -p jugar-probar
```
