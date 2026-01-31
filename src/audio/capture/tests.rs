use super::*;

#[test]
fn test_capture_config_default() {
    let config = CaptureConfig::default();
    assert_eq!(config.sample_rate, 16000);
    assert_eq!(config.channels, 1);
    assert!(config.validate().is_ok());
}

#[test]
fn test_capture_config_whisper() {
    let config = CaptureConfig::whisper();
    assert_eq!(config.sample_rate, 16000);
    assert_eq!(config.channels, 1);
}

#[test]
fn test_capture_config_stereo() {
    let config = CaptureConfig::stereo();
    assert_eq!(config.channels, 2);
}

#[test]
fn test_capture_config_validation() {
    let mut config = CaptureConfig::default();
    config.sample_rate = 0;
    assert!(config.validate().is_err());

    config.sample_rate = 16000;
    config.channels = 0;
    assert!(config.validate().is_err());

    config.channels = 1;
    config.buffer_size = 0;
    assert!(config.validate().is_err());
}

#[test]
#[cfg(not(all(target_os = "linux", feature = "audio-alsa")))]
fn test_list_devices_stub() {
    // Stub returns empty list when no native backend
    let devices = list_devices().unwrap();
    assert!(devices.is_empty());
}

#[test]
#[cfg(all(target_os = "linux", feature = "audio-alsa"))]
fn test_list_devices_alsa() {
    // ALSA returns devices (may be empty on systems without audio hardware)
    let devices = list_devices();
    assert!(
        devices.is_ok(),
        "list_devices should succeed: {:?}",
        devices
    );
}

#[test]
#[cfg(not(all(target_os = "linux", feature = "audio-alsa")))]
fn test_default_device_stub() {
    let device = default_device().unwrap();
    assert!(device.is_none());
}

#[test]
#[cfg(not(all(target_os = "linux", feature = "audio-alsa")))]
fn test_audio_capture_open_not_implemented() {
    let config = CaptureConfig::default();
    let result = AudioCapture::open(None, &config);
    assert!(result.is_err());
}

#[test]
#[cfg(all(target_os = "linux", feature = "audio-alsa"))]
fn test_audio_capture_open_alsa() {
    // ALSA capture may fail on systems without audio hardware, but shouldn't panic
    let config = CaptureConfig::default();
    let result = AudioCapture::open(None, &config);
    // Result can be Ok or Err depending on hardware availability
    // The important thing is it doesn't panic
    let _ = result;
}

// GH-130: Mock capture source tests

#[test]
fn test_mock_signal_default() {
    let signal = MockSignal::default();
    assert_eq!(signal, MockSignal::Silence);
}

#[test]
fn test_mock_capture_silence() {
    let config = CaptureConfig::default();
    let mut mock = MockCaptureSource::silence(config);

    let mut buffer = vec![0.5f32; 100]; // Non-zero initial values
    let n = mock.read(&mut buffer).expect("read should succeed");

    assert_eq!(n, 100);
    for sample in &buffer {
        assert_eq!(*sample, 0.0, "Silence should produce zeros");
    }
}

#[test]
fn test_mock_capture_sine_bounded() {
    let config = CaptureConfig::default();
    let mut mock = MockCaptureSource::new(
        config,
        MockSignal::Sine {
            frequency: 440.0,
            amplitude: 0.8,
        },
    );

    let mut buffer = vec![0.0f32; 1000];
    mock.read(&mut buffer).expect("read should succeed");

    // All samples should be within amplitude bounds
    for sample in &buffer {
        assert!(
            sample.abs() <= 0.8 + 1e-6,
            "Sample {} exceeds amplitude 0.8",
            sample
        );
    }

    // Sine wave should have non-zero samples (not all silence)
    let non_zero_count = buffer.iter().filter(|s| s.abs() > 1e-6).count();
    assert!(
        non_zero_count > 0,
        "Sine wave should produce non-zero samples"
    );
}

#[test]
fn test_mock_capture_sine_deterministic() {
    let config = CaptureConfig::default();
    let mut mock1 = MockCaptureSource::a440(config.clone());
    let mut mock2 = MockCaptureSource::a440(config);

    let mut buffer1 = vec![0.0f32; 100];
    let mut buffer2 = vec![0.0f32; 100];

    mock1.read(&mut buffer1).expect("read should succeed");
    mock2.read(&mut buffer2).expect("read should succeed");

    // Same configuration should produce identical output
    for (s1, s2) in buffer1.iter().zip(buffer2.iter()) {
        assert!(
            (s1 - s2).abs() < 1e-6,
            "Deterministic signal should be reproducible"
        );
    }
}

#[test]
fn test_mock_capture_white_noise_bounded() {
    let config = CaptureConfig::default();
    let mut mock = MockCaptureSource::white_noise(config, 0.5);

    let mut buffer = vec![0.0f32; 1000];
    mock.read(&mut buffer).expect("read should succeed");

    // All samples should be within bounds
    for sample in &buffer {
        assert!(
            sample.abs() <= 0.5 + 0.01,
            "Noise sample {} exceeds amplitude 0.5",
            sample
        );
    }

    // Should have variance (not all same value)
    let mean: f32 = buffer.iter().sum::<f32>() / buffer.len() as f32;
    let variance: f32 =
        buffer.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / buffer.len() as f32;
    assert!(
        variance > 0.001,
        "White noise should have variance, got {}",
        variance
    );
}

#[test]
fn test_mock_capture_impulse() {
    let config = CaptureConfig::default();
    let mut mock = MockCaptureSource::new(config, MockSignal::Impulse);

    let mut buffer = vec![0.0f32; 100];
    mock.read(&mut buffer).expect("read should succeed");

    // First sample should be 1.0, rest should be 0.0
    assert_eq!(buffer[0], 1.0, "First sample should be impulse");
    for sample in &buffer[1..] {
        assert_eq!(*sample, 0.0, "Non-first samples should be zero");
    }
}

#[test]
fn test_mock_capture_square_wave() {
    let config = CaptureConfig::default();
    let mut mock = MockCaptureSource::new(
        config,
        MockSignal::Square {
            frequency: 100.0,
            amplitude: 1.0,
        },
    );

    let mut buffer = vec![0.0f32; 320]; // Multiple cycles at 16kHz
    mock.read(&mut buffer).expect("read should succeed");

    // Square wave should only have +1 or -1 values
    for sample in &buffer {
        assert!(
            (*sample - 1.0).abs() < 1e-6 || (*sample + 1.0).abs() < 1e-6,
            "Square wave sample {} should be ±1",
            sample
        );
    }
}

#[test]
fn test_mock_capture_reset() {
    let config = CaptureConfig::default();
    let mut mock = MockCaptureSource::a440(config);

    let mut buffer1 = vec![0.0f32; 100];
    let mut buffer2 = vec![0.0f32; 100];

    mock.read(&mut buffer1).expect("read should succeed");
    mock.reset();
    mock.read(&mut buffer2).expect("read should succeed");

    // After reset, should produce same output
    for (s1, s2) in buffer1.iter().zip(buffer2.iter()) {
        assert!(
            (s1 - s2).abs() < 1e-6,
            "Reset should restart signal from beginning"
        );
    }
}

#[test]
fn test_mock_capture_position() {
    let config = CaptureConfig::default();
    let mut mock = MockCaptureSource::silence(config);

    assert_eq!(mock.position(), 0);

    let mut buffer = vec![0.0f32; 100];
    mock.read(&mut buffer).expect("read should succeed");

    assert_eq!(mock.position(), 100);

    mock.reset();
    assert_eq!(mock.position(), 0);
}

#[test]
fn test_mock_capture_set_signal() {
    let config = CaptureConfig::default();
    let mut mock = MockCaptureSource::silence(config);

    assert_eq!(mock.signal(), MockSignal::Silence);

    mock.set_signal(MockSignal::Impulse);
    assert_eq!(mock.signal(), MockSignal::Impulse);
}

#[test]
fn test_buffer_capture_source_basic() {
    let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let mut source = BufferCaptureSource::new(samples);

    let mut buffer = vec![0.0f32; 3];
    let n = source.read(&mut buffer).expect("read should succeed");

    assert_eq!(n, 3);
    assert_eq!(buffer, vec![0.1, 0.2, 0.3]);
    assert_eq!(source.position(), 3);
}

#[test]
fn test_buffer_capture_source_exhausted() {
    let samples = vec![0.1, 0.2, 0.3];
    let mut source = BufferCaptureSource::new(samples);

    let mut buffer = vec![0.0f32; 5];
    let n = source.read(&mut buffer).expect("read should succeed");

    // Only 3 samples available
    assert_eq!(n, 3);
    assert!(source.is_exhausted());
}

#[test]
fn test_buffer_capture_source_loop() {
    let samples = vec![0.1, 0.2, 0.3];
    let mut source = BufferCaptureSource::new(samples).with_loop(true);

    let mut buffer = vec![0.0f32; 7];
    let n = source.read(&mut buffer).expect("read should succeed");

    assert_eq!(n, 7);
    // Should loop: [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1]
    assert!((buffer[0] - 0.1).abs() < 1e-6);
    assert!((buffer[3] - 0.1).abs() < 1e-6);
    assert!((buffer[6] - 0.1).abs() < 1e-6);
    assert!(!source.is_exhausted());
}

#[test]
fn test_buffer_capture_source_empty() {
    let mut source = BufferCaptureSource::new(vec![]);

    let mut buffer = vec![0.0f32; 10];
    let n = source.read(&mut buffer).expect("read should succeed");

    assert_eq!(n, 0);
}

#[test]
fn test_buffer_capture_source_reset() {
    let samples = vec![0.1, 0.2, 0.3];
    let mut source = BufferCaptureSource::new(samples);

    let mut buffer = vec![0.0f32; 2];
    source.read(&mut buffer).expect("read should succeed");
    assert_eq!(source.position(), 2);

    source.reset();
    assert_eq!(source.position(), 0);
}

// ========================================================================
// C5-C10: Backend Detection Tests
// ========================================================================

#[test]
fn test_available_backend_returns_string() {
    // Should always return a valid backend name or "None" message
    let backend = available_backend();
    assert!(!backend.is_empty());
}

#[test]
fn test_has_native_backend_returns_bool() {
    // Should compile and return a boolean
    let _has_backend: bool = has_native_backend();
}

#[test]
fn test_capture_backend_trait_object_safety() {
    // Verify CaptureBackend is object-safe by checking it can be used as trait bound
    fn _accepts_backend<T: CaptureBackend>(_b: T) {}
}

#[test]
fn test_backend_names_documented() {
    // Verify backend availability message is informative
    let backend = available_backend();
    // Should either be a real backend name or explain how to enable
    assert!(
        backend == "ALSA"
            || backend == "CoreAudio"
            || backend == "WASAPI"
            || backend == "WebAudio"
            || backend.contains("enable")
    );
}

// ========================================================================
// C7: ALSA Backend Tests (Linux only)
// ========================================================================

#[cfg(all(target_os = "linux", feature = "audio-alsa"))]
mod alsa_tests {
    use super::*;

    #[test]
    fn test_alsa_i16_to_f32_zero() {
        let result = AlsaBackend::i16_to_f32(0);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_alsa_i16_to_f32_max() {
        let result = AlsaBackend::i16_to_f32(i16::MAX);
        assert!(
            (result - 1.0).abs() < 1e-4,
            "Max i16 should map to ~1.0, got {}",
            result
        );
    }

    #[test]
    fn test_alsa_i16_to_f32_min() {
        let result = AlsaBackend::i16_to_f32(i16::MIN);
        assert!(
            (result - (-1.0)).abs() < 1e-4,
            "Min i16 should map to ~-1.0, got {}",
            result
        );
    }

    #[test]
    fn test_alsa_i16_to_f32_positive_range() {
        // All positive i16 should map to [0, 1]
        for val in [1, 100, 1000, 10000, 32767_i16] {
            let result = AlsaBackend::i16_to_f32(val);
            assert!(
                result >= 0.0 && result <= 1.0,
                "Positive {} mapped to {}",
                val,
                result
            );
        }
    }

    #[test]
    fn test_alsa_i16_to_f32_negative_range() {
        // All negative i16 should map to [-1, 0]
        for val in [-1, -100, -1000, -10000, -32768_i16] {
            let result = AlsaBackend::i16_to_f32(val);
            assert!(
                result >= -1.0 && result <= 0.0,
                "Negative {} mapped to {}",
                val,
                result
            );
        }
    }

    #[test]
    fn test_alsa_i16_to_f32_symmetric() {
        // Symmetric values should map to approximately symmetric results
        let positive = AlsaBackend::i16_to_f32(16384);
        let negative = AlsaBackend::i16_to_f32(-16384);
        assert!(
            (positive + negative).abs() < 0.001,
            "Symmetric values should cancel: {} + {} = {}",
            positive,
            negative,
            positive + negative
        );
    }

    #[test]
    fn test_alsa_backend_name() {
        assert_eq!(AlsaBackend::backend_name(), "ALSA");
    }

    #[test]
    fn test_alsa_list_devices() {
        // This test requires ALSA to be available on the system
        // It may return an empty list if no audio devices are present
        let result = AlsaBackend::list_devices();
        assert!(
            result.is_ok(),
            "list_devices should not error: {:?}",
            result
        );
    }
}

// Test i16 to f32 conversion logic (can run without ALSA feature)
#[test]
fn test_i16_to_f32_conversion_logic() {
    // Test the conversion formula: i16 to f32 [-1.0, 1.0]
    fn i16_to_f32(sample: i16) -> f32 {
        if sample >= 0 {
            f32::from(sample) / 32767.0
        } else {
            f32::from(sample) / 32768.0
        }
    }

    assert!((i16_to_f32(0) - 0.0).abs() < 1e-6);
    assert!((i16_to_f32(32767) - 1.0).abs() < 1e-4);
    assert!((i16_to_f32(-32768) - (-1.0)).abs() < 1e-4);
    assert!((i16_to_f32(16384) - 0.5).abs() < 0.01);
    assert!((i16_to_f32(-16384) - (-0.5)).abs() < 0.01);
}
