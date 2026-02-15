
// ============================================================================
// Backend Selection Helper
// ============================================================================

/// Get the name of the currently available audio backend
///
/// Returns the backend name based on platform and enabled features,
/// or "None" if no backend is available.
#[must_use]
pub fn available_backend() -> &'static str {
    #[cfg(all(target_os = "linux", feature = "audio-alsa"))]
    {
        return "ALSA";
    }
    #[cfg(all(target_os = "macos", feature = "audio-coreaudio"))]
    {
        return "CoreAudio";
    }
    #[cfg(all(target_os = "windows", feature = "audio-wasapi"))]
    {
        return "WASAPI";
    }
    #[cfg(all(target_arch = "wasm32", feature = "audio-webaudio"))]
    {
        return "WebAudio";
    }
    #[allow(unreachable_code)]
    "None (enable audio-alsa, audio-coreaudio, audio-wasapi, or audio-webaudio feature)"
}

/// Check if a native audio backend is available
#[must_use]
pub fn has_native_backend() -> bool {
    #[cfg(any(
        all(target_os = "linux", feature = "audio-alsa"),
        all(target_os = "macos", feature = "audio-coreaudio"),
        all(target_os = "windows", feature = "audio-wasapi"),
        all(target_arch = "wasm32", feature = "audio-webaudio")
    ))]
    {
        return true;
    }
    #[allow(unreachable_code)]
    false
}

// ============================================================================
// Configuration
// ============================================================================

/// Audio capture configuration
#[derive(Debug, Clone)]
pub struct CaptureConfig {
    /// Sample rate in Hz (default: 16000)
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u16,
    /// Buffer size in samples
    pub buffer_size: usize,
    /// Enable exclusive mode (lower latency)
    pub exclusive: bool,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000, // Whisper native rate
            channels: 1,        // Mono for ASR
            buffer_size: 1600,  // 100ms at 16kHz
            exclusive: false,
        }
    }
}

impl CaptureConfig {
    /// Create config for Whisper-compatible capture
    #[must_use]
    pub fn whisper() -> Self {
        Self::default()
    }

    /// Create config for stereo capture
    #[must_use]
    pub fn stereo() -> Self {
        Self {
            channels: 2,
            ..Self::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), AudioError> {
        if self.sample_rate == 0 {
            return Err(AudioError::InvalidConfig(
                "sample_rate must be > 0".to_string(),
            ));
        }
        if self.channels == 0 {
            return Err(AudioError::InvalidConfig(
                "channels must be > 0".to_string(),
            ));
        }
        if self.buffer_size == 0 {
            return Err(AudioError::InvalidConfig(
                "buffer_size must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Audio Device
// ============================================================================

/// Audio input device information
#[derive(Debug, Clone)]
pub struct AudioDevice {
    /// Device ID (platform-specific)
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Maximum supported sample rate
    pub max_sample_rate: u32,
    /// Number of input channels
    pub input_channels: u16,
    /// Whether this is the default device
    pub is_default: bool,
}

/// List available audio input devices
///
/// # Returns
/// Vector of available devices, or error if audio system unavailable
///
/// # Platform Support
/// - Linux: Uses ALSA (requires `audio-alsa` feature)
/// - macOS: Uses CoreAudio (requires `audio-coreaudio` feature)
/// - Windows: Uses WASAPI (requires `audio-wasapi` feature)
pub fn list_devices() -> Result<Vec<AudioDevice>, AudioError> {
    #[cfg(all(target_os = "linux", feature = "audio-alsa"))]
    {
        return AlsaBackend::list_devices();
    }

    // Stub for other platforms
    #[allow(unreachable_code)]
    Ok(vec![])
}

/// Get the default audio input device
///
/// # Returns
/// Default device if available
pub fn default_device() -> Result<Option<AudioDevice>, AudioError> {
    Ok(list_devices()?.into_iter().find(|d| d.is_default))
}

// ============================================================================
// Audio Capture
// ============================================================================

/// Audio capture handle for streaming input
///
/// # Lifecycle
/// 1. Open with `AudioCapture::open()`
/// 2. Read samples with `read()`
/// 3. Close with `close()` (or Drop)
///
/// # Platform Support
/// - Linux: Uses ALSA (requires `audio-alsa` feature)
/// - macOS: Uses CoreAudio (requires `audio-coreaudio` feature)
/// - Windows: Uses WASAPI (requires `audio-wasapi` feature)
///
/// # Example
///
/// ```rust,ignore
/// use aprender::audio::capture::{AudioCapture, CaptureConfig};
///
/// let config = CaptureConfig::whisper();
/// let mut capture = AudioCapture::open(None, &config)?;
///
/// let mut buffer = vec![0.0f32; 1600]; // 100ms at 16kHz
/// while let Ok(n) = capture.read(&mut buffer) {
///     process_audio(&buffer[..n]);
/// }
///
/// capture.close()?;
/// ```
pub struct AudioCapture {
    #[cfg(all(target_os = "linux", feature = "audio-alsa"))]
    backend: AlsaBackend,

    #[cfg(not(all(target_os = "linux", feature = "audio-alsa")))]
    config: CaptureConfig,
    #[cfg(not(all(target_os = "linux", feature = "audio-alsa")))]
    #[allow(dead_code)]
    device_id: Option<String>,
    #[cfg(not(all(target_os = "linux", feature = "audio-alsa")))]
    running: bool,
}

impl AudioCapture {
    /// Open audio capture stream
    ///
    /// # Arguments
    /// * `device` - Device ID, or None for default
    /// * `config` - Capture configuration
    ///
    /// # Errors
    /// Returns error if device cannot be opened or configuration is invalid.
    pub fn open(device: Option<&str>, config: &CaptureConfig) -> Result<Self, AudioError> {
        config.validate()?;

        #[cfg(all(target_os = "linux", feature = "audio-alsa"))]
        {
            let backend = AlsaBackend::open(device, config)?;
            return Ok(Self { backend });
        }

        // Stub for other platforms
        #[allow(unreachable_code)]
        {
            let _ = device;
            Err(AudioError::NotImplemented(
                "Audio capture requires platform-specific backend (enable audio-alsa on Linux)"
                    .to_string(),
            ))
        }
    }

    /// Read audio samples into buffer
    ///
    /// # Arguments
    /// * `buffer` - Output buffer for samples (f32 normalized to [-1.0, 1.0])
    ///
    /// # Returns
    /// Number of samples read, or error
    pub fn read(&mut self, buffer: &mut [f32]) -> Result<usize, AudioError> {
        #[cfg(all(target_os = "linux", feature = "audio-alsa"))]
        {
            return self.backend.read(buffer);
        }

        #[allow(unreachable_code)]
        {
            let _ = buffer;
            Err(AudioError::NotImplemented("read".to_string()))
        }
    }

    /// Get capture configuration
    #[must_use]
    #[allow(clippy::needless_return)] // cfg blocks require explicit return
    pub fn config(&self) -> &CaptureConfig {
        #[cfg(all(target_os = "linux", feature = "audio-alsa"))]
        {
            return &self.backend.config;
        }

        #[cfg(not(all(target_os = "linux", feature = "audio-alsa")))]
        &self.config
    }

    /// Check if capture is running
    #[must_use]
    #[allow(clippy::needless_return)] // cfg blocks require explicit return
    pub fn is_running(&self) -> bool {
        #[cfg(all(target_os = "linux", feature = "audio-alsa"))]
        {
            return true; // ALSA backend is always "running" once opened
        }

        #[cfg(not(all(target_os = "linux", feature = "audio-alsa")))]
        self.running
    }

    /// Close the capture stream
    pub fn close(self) -> Result<(), AudioError> {
        #[cfg(all(target_os = "linux", feature = "audio-alsa"))]
        {
            let mut backend = self.backend;
            return backend.close();
        }

        #[allow(unreachable_code)]
        Ok(())
    }
}

#[cfg(all(target_os = "linux", feature = "audio-alsa"))]
impl std::fmt::Debug for AudioCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioCapture")
            .field("backend", &"AlsaBackend")
            .field("config", &self.backend.config)
            .finish()
    }
}

#[cfg(not(all(target_os = "linux", feature = "audio-alsa")))]
impl std::fmt::Debug for AudioCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioCapture")
            .field("config", &self.config)
            .field("running", &self.running)
            .finish()
    }
}

// ============================================================================
// GH-130: Mock Audio Capture Source
// ============================================================================

/// Signal type for mock audio generation
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MockSignal {
    /// Silence (all zeros)
    #[default]
    Silence,
    /// Sine wave at given frequency
    Sine { frequency: f32, amplitude: f32 },
    /// White noise with given amplitude
    WhiteNoise { amplitude: f32 },
    /// Impulse (single sample at 1.0, rest zeros)
    Impulse,
    /// Square wave at given frequency
    Square { frequency: f32, amplitude: f32 },
}

/// Mock audio capture source for testing (GH-130)
///
/// Generates deterministic test signals without requiring real audio hardware.
/// Useful for:
/// - Unit testing audio pipelines
/// - Integration testing without microphone
/// - Benchmarking audio processing code
///
/// # Example
///
/// ```rust
/// use aprender::audio::capture::{MockCaptureSource, MockSignal, CaptureConfig};
///
/// let config = CaptureConfig::default();
/// let mut mock = MockCaptureSource::new(config, MockSignal::Sine {
///     frequency: 440.0,
///     amplitude: 0.5,
/// });
///
/// let mut buffer = vec![0.0f32; 160];
/// let n = mock.read(&mut buffer).expect("mock read always succeeds");
/// assert_eq!(n, 160);
/// assert!(buffer[0].abs() <= 0.5); // Amplitude bounded
/// ```
#[derive(Debug)]
pub struct MockCaptureSource {
    config: CaptureConfig,
    signal: MockSignal,
    sample_index: u64,
    /// Random state for white noise (LCG)
    rng_state: u64,
}
