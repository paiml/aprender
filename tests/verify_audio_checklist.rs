#![allow(clippy::disallowed_methods)]
#![cfg(feature = "audio")]

use aprender::audio::mel::{
    detect_clipping, stereo_to_mono, validate_audio, MelConfig, MelFilterbank,
};
use aprender::audio::resample::resample;
use aprender::audio::stream::{AudioChunker, ChunkConfig};

#[test]
fn verify_a1_mel_bins_80() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);
    assert_eq!(mel.n_mels(), 80, "A1: Mel spectrogram must produce 80 bins");
}

#[test]
fn verify_a2_slaney_normalization() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);
    let max_filter_val = mel.filters().iter().fold(0.0f32, |a, &b| f32::max(a, b));
    assert!(
        max_filter_val < 0.1,
        "A2: Filterbank must use Slaney normalization (max < 0.1), got {}",
        max_filter_val
    );
}

#[test]
fn verify_a3_silence_negative_mean() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);
    let silence = vec![0.0; 16000];
    let spec = mel.compute(&silence).unwrap();
    let mean: f32 = spec.iter().sum::<f32>() / spec.len() as f32;
    assert!(
        mean < 0.0,
        "A3: Silence input must produce negative mel mean"
    );
}

#[test]
fn verify_a4_resample_preserves_duration() {
    let sr_in = 44100;
    let sr_out = 16000;
    let duration_sec = 1.0;
    let input_len = (duration_sec * sr_in as f32) as usize;
    let input = vec![0.0; input_len];
    let output = resample(&input, sr_in, sr_out).unwrap();
    let output_duration = output.len() as f32 / sr_out as f32;
    assert!(
        (output_duration - duration_sec).abs() < 0.001,
        "A4: Resample must preserve duration"
    );
}

#[test]
fn verify_a5_16khz_supported() {
    let config = MelConfig::whisper();
    assert_eq!(
        config.sample_rate, 16000,
        "A5: 16kHz must be supported sample rate"
    );
}

#[test]
fn verify_a6_streaming_chunker() {
    let chunk_config = ChunkConfig {
        chunk_size: 16000,
        overlap: 1600,
        sample_rate: 16000,
    };
    let mut chunker = AudioChunker::new(chunk_config);
    let audio = vec![0.0; 32000];
    chunker.push(&audio);
    assert!(chunker.has_chunk(), "A6: Streaming chunker should work");
}

#[test]
fn verify_a7_determinism() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);
    let audio = vec![0.0; 16000];
    let spec1 = mel.compute(&audio).unwrap();
    let spec2 = mel.compute(&audio).unwrap();
    assert_eq!(spec1, spec2, "A7: Mel computation must be deterministic");
}

#[test]
fn verify_a8_fft_window_400() {
    let config = MelConfig::whisper();
    assert_eq!(config.n_fft, 400, "A8: FFT window size must be 400");
}

#[test]
fn verify_a9_hop_length_160() {
    let config = MelConfig::whisper();
    assert_eq!(config.hop_length, 160, "A9: Hop length must be 160");
}

#[test]
fn verify_a10_mel_range_0_8000() {
    let config = MelConfig::whisper();
    assert_eq!(config.fmin, 0.0, "A10: Mel min freq must be 0");
    assert_eq!(config.fmax, 8000.0, "A10: Mel max freq must be 8000");
}

#[test]
fn verify_a11_clipping_detection() {
    let samples = vec![1.1];
    let report = detect_clipping(&samples);
    assert!(report.has_clipping, "A11: Clipping detection failed");
}

#[test]
fn verify_a12_stereo_to_mono() {
    let stereo = vec![1.0, 0.0];
    let mono = stereo_to_mono(&stereo);
    assert_eq!(mono[0], 0.5, "A12: Stereo to mono failed");
}

#[test]
fn verify_a13_zero_length_error() {
    let result = validate_audio(&[]);
    assert!(result.is_err(), "A13: Zero-length audio must return error");
}

#[test]
fn verify_a14_nan_detection() {
    let result = validate_audio(&[f32::NAN]);
    assert!(result.is_err(), "A14: NaN in audio must be detected");
}

#[test]
fn verify_a15_inf_detection() {
    let result = validate_audio(&[f32::INFINITY]);
    assert!(result.is_err(), "A15: Inf in audio must be detected");
}
