//! Audio Emulation Demo
//!
//! Demonstrates the AudioEmulator from PROBAR-SPEC-011:
//! - Deterministic audio source generation
//! - Mock JavaScript for getUserMedia injection
//! - Various audio source types (sine wave, speech pattern, silence)
//!
//! Run with: cargo run --example audio_emulation -p jugar-probar

use jugar_probar::emulation::{AudioEmulator, AudioSource};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        Audio Emulation Demo (PROBAR-SPEC-011)                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    demo_sine_wave();
    demo_speech_pattern();
    demo_silence();
    demo_white_noise();
    demo_mock_js_generation();
}

/// Demonstrate sine wave audio generation
fn demo_sine_wave() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  1. Sine Wave Audio Source");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut emulator = AudioEmulator::new(AudioSource::SineWave {
        frequency: 440.0,
        amplitude: 0.5,
    });

    println!("Created sine wave audio source:");
    println!("  - Frequency: 440 Hz (A4 note)");
    println!("  - Amplitude: 0.5");
    println!("  - Sample rate: {} Hz", emulator.sample_rate());

    // Generate 100ms of audio
    let samples = emulator.generate_samples(0.1);
    println!("\nGenerated {} samples (100ms)", samples.len());

    // Analyze samples
    let min = samples.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

    println!("  - Min value: {:.4}", min);
    println!("  - Max value: {:.4}", max);
    println!("  - RMS level: {:.4}", rms);

    // Show first few samples
    println!("\nFirst 10 samples:");
    for (i, sample) in samples.iter().take(10).enumerate() {
        println!("  [{:2}] {:.6}", i, sample);
    }
    println!();
}

/// Demonstrate speech-like audio pattern
fn demo_speech_pattern() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  2. Speech Pattern Audio Source");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut emulator = AudioEmulator::new(AudioSource::SpeechPattern {
        fundamental_hz: 150.0,
        harmonics: vec![0.5, 0.3, 0.2, 0.1],
        variation_hz: 20.0,
    });

    println!("Created speech pattern audio source:");
    println!("  - Fundamental: 150 Hz (typical male voice)");
    println!("  - Harmonics: [0.5, 0.3, 0.2, 0.1]");
    println!("  - Pitch variation: ±20 Hz");

    let samples = emulator.generate_samples(0.05);
    println!("\nGenerated {} samples (50ms)", samples.len());

    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    println!("  - RMS level: {:.4}", rms);
    println!();
}

/// Demonstrate silence with noise floor
fn demo_silence() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  3. Silence with Noise Floor");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Complete silence
    let mut silent = AudioEmulator::new(AudioSource::Silence {
        noise_floor_db: -100.0,
    });

    let samples = silent.generate_samples(0.01);
    let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    println!("Complete silence (-100dB noise floor):");
    println!("  - Max absolute value: {:.8}", max_abs);

    // Ambient noise
    let mut noisy = AudioEmulator::new(AudioSource::Silence {
        noise_floor_db: -40.0,
    });

    let samples = noisy.generate_samples(0.01);
    let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    println!("\nAmbient noise (-40dB noise floor):");
    println!("  - Max absolute value: {:.6}", max_abs);
    println!();
}

/// Demonstrate white noise for VAD testing
fn demo_white_noise() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  4. White Noise Audio Source");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // White noise for VAD (Voice Activity Detection) testing
    let mut emulator = AudioEmulator::new(AudioSource::WhiteNoise { amplitude: 0.3 });

    println!("Created white noise source:");
    println!("  - Amplitude: 0.3");
    println!("  - Purpose: VAD testing (should NOT be classified as speech)");

    let samples = emulator.generate_samples(0.01);
    println!("\nGenerated {} samples (10ms)", samples.len());

    // Analyze noise distribution
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    let variance = samples.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / samples.len() as f32;
    let std_dev = variance.sqrt();

    println!("  - Mean: {:.6} (expected ~0.0)", mean);
    println!("  - Std dev: {:.4} (relates to amplitude)", std_dev);
    println!();
}

/// Demonstrate mock JavaScript generation
fn demo_mock_js_generation() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  5. Mock JavaScript Generation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let emulator = AudioEmulator::new(AudioSource::SineWave {
        frequency: 440.0,
        amplitude: 0.5,
    });

    // Generate samples for injection
    let mut emulator_mut = emulator;
    let samples = emulator_mut.generate_samples(0.05);

    // Generate mock JS
    let mock_js = emulator_mut.generate_mock_js(&samples);

    println!("Generated JavaScript mock for getUserMedia injection:");
    println!("  - Code length: {} bytes", mock_js.len());
    println!(
        "  - Contains getUserMedia override: {}",
        mock_js.contains("getUserMedia")
    );
    println!(
        "  - Contains AudioContext: {}",
        mock_js.contains("AudioContext")
    );

    println!("\nFirst 500 characters of generated JS:");
    println!("─────────────────────────────────────────");
    let preview: String = mock_js.chars().take(500).collect();
    println!("{}", preview);
    if mock_js.len() > 500 {
        println!("...");
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Demo complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
