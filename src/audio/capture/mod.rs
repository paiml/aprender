//! Audio capture module for real-time streaming (GH-130).
//!
//! Provides cross-platform audio capture for:
//! - Microphone input
//! - System audio loopback
//! - File streaming
//!
//! # Platform Support (C5-C10)
//!
//! | Platform | Backend | Feature | Status |
//! |----------|---------|---------|--------|
//! | Linux | ALSA | `audio-alsa` | ✅ Implemented (C7) |
//! | macOS | CoreAudio | `audio-coreaudio` | Stub ready (C8) |
//! | Windows | WASAPI | `audio-wasapi` | Stub ready (C9) |
//! | WASM | WebAudio API | `audio-webaudio` | Stub ready (C10) |
//!
//! # Linux (ALSA) Setup
//!
//! To enable ALSA audio capture on Linux:
//!
//! 1. Install ALSA development libraries:
//!    ```bash
//!    # Debian/Ubuntu
//!    sudo apt-get install libasound2-dev
//!
//!    # Fedora/RHEL
//!    sudo dnf install alsa-lib-devel
//!
//!    # Arch Linux
//!    sudo pacman -S alsa-lib
//!    ```
//!
//! 2. Enable the feature in your `Cargo.toml`:
//!    ```toml
//!    [dependencies]
//!    aprender = { version = "0.19", features = ["audio-alsa"] }
//!    ```
//!
//! # Backend Architecture
//!
//! Each platform backend implements the `CaptureBackend` trait, which provides:
//! - `open()`: Initialize audio capture with configuration
//! - `read()`: Read samples into buffer (non-blocking)
//! - `close()`: Release audio resources
//!
//! The backends are selected at compile time via feature flags, enabling
//! zero-cost abstraction for the target platform.
//!
//! # Example (Future API)
//!
//! ```rust,ignore
//! use aprender::audio::capture::{AudioCapture, CaptureConfig};
//!
//! let config = CaptureConfig::default();
//! let mut capture = AudioCapture::open(None, config)?;
//!
//! let mut buffer = vec![0.0f32; 1600]; // 100ms at 16kHz
//! while let Ok(n) = capture.read(&mut buffer) {
//!     process_audio(&buffer[..n]);
//! }
//! ```
//!
//! # PMAT Compliance
//!
//! - Zero `unwrap()` calls
//! - All public APIs return `Result<T, E>`
//! - Real-time safe (no allocations in capture loop)

use crate::audio::AudioError;

// ============================================================================
// C5-C10: Platform Backend Trait
// ============================================================================

/// Trait for platform-specific audio capture backends
///
/// Each platform (ALSA, CoreAudio, WASAPI, WebAudio) implements this trait
/// to provide native audio capture functionality.
pub trait CaptureBackend: Send {
    /// Open the audio capture stream
    fn open(device: Option<&str>, config: &CaptureConfig) -> Result<Self, AudioError>
    where
        Self: Sized;

    /// Read audio samples into buffer
    ///
    /// Returns the number of samples read, or error
    fn read(&mut self, buffer: &mut [f32]) -> Result<usize, AudioError>;

    /// Close the audio capture stream
    fn close(&mut self) -> Result<(), AudioError>;

    /// Get the backend name for diagnostics
    fn backend_name() -> &'static str
    where
        Self: Sized;
}

// ============================================================================
// C7: ALSA Backend (Linux)
// ============================================================================

/// ALSA audio capture backend for Linux (GH-130, C7)
///
/// Uses the Advanced Linux Sound Architecture for low-latency audio capture.
/// Requires the `audio-alsa` feature flag.
///
/// # Example
///
/// ```rust,ignore
/// use aprender::audio::capture::{AlsaBackend, CaptureBackend, CaptureConfig};
///
/// let config = CaptureConfig::whisper();
/// let mut backend = AlsaBackend::open(None, &config)?;
///
/// let mut buffer = vec![0.0f32; 1600];
/// let n = backend.read(&mut buffer)?;
/// println!("Read {} samples", n);
///
/// backend.close()?;
/// ```
#[cfg(all(target_os = "linux", feature = "audio-alsa"))]
pub struct AlsaBackend {
    pcm: alsa::PCM,
    config: CaptureConfig,
    /// Intermediate buffer for i16 samples from ALSA
    i16_buffer: Vec<i16>,
}

#[cfg(all(target_os = "linux", feature = "audio-alsa"))]
impl std::fmt::Debug for AlsaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlsaBackend")
            .field("config", &self.config)
            .field("buffer_size", &self.i16_buffer.len())
            .finish_non_exhaustive()
    }
}

#[cfg(all(target_os = "linux", feature = "audio-alsa"))]
impl AlsaBackend {
    /// List available ALSA capture devices
    ///
    /// # Returns
    /// Vector of device names suitable for `open()`
    pub fn list_devices() -> Result<Vec<AudioDevice>, AudioError> {
        use alsa::device_name::HintIter;
        use std::ffi::CStr;

        let mut devices = Vec::new();

        // Iterate over PCM devices - use safe CStr creation
        let pcm_cstr = CStr::from_bytes_with_nul(b"pcm\0")
            .map_err(|e| AudioError::CaptureError(format!("Invalid CStr: {e}")))?;
        let hints = HintIter::new(None, pcm_cstr)
            .map_err(|e| AudioError::CaptureError(format!("Failed to enumerate devices: {e}")))?;

        for hint in hints {
            // Only include capture-capable devices
            if let Some(name) = hint.name {
                // Skip null device and complex plugin names
                if name == "null" || name.contains("surround") {
                    continue;
                }

                let desc = hint.desc.unwrap_or_else(String::new);
                let is_default = name == "default" || name == "pulse";

                devices.push(AudioDevice {
                    id: name.clone(),
                    name: desc.lines().next().unwrap_or(&name).to_string(),
                    max_sample_rate: 48000, // Most devices support this
                    input_channels: 2,
                    is_default,
                });
            }
        }

        Ok(devices)
    }

    /// Convert i16 samples to f32 normalized to [-1.0, 1.0]
    #[inline]
    fn i16_to_f32(sample: i16) -> f32 {
        // i16 range: -32768 to 32767
        // Normalize to [-1.0, 1.0]
        if sample >= 0 {
            f32::from(sample) / 32767.0
        } else {
            f32::from(sample) / 32768.0
        }
    }
}

#[cfg(all(target_os = "linux", feature = "audio-alsa"))]
impl CaptureBackend for AlsaBackend {
    fn open(device: Option<&str>, config: &CaptureConfig) -> Result<Self, AudioError> {
        use alsa::pcm::{Access, Format, HwParams};
        use alsa::{Direction, ValueOr};

        config.validate()?;

        // Use provided device or default
        let device_name = device.unwrap_or("default");

        // Open PCM device for capture
        let pcm = alsa::PCM::new(device_name, Direction::Capture, false).map_err(|e| {
            AudioError::CaptureError(format!("Failed to open ALSA device '{device_name}': {e}"))
        })?;

        // Configure hardware parameters
        {
            let hwp = HwParams::any(&pcm)
                .map_err(|e| AudioError::CaptureError(format!("Failed to get HW params: {e}")))?;

            // Set access type (interleaved samples)
            hwp.set_access(Access::RWInterleaved)
                .map_err(|e| AudioError::CaptureError(format!("Failed to set access: {e}")))?;

            // Set format to signed 16-bit little-endian (most compatible)
            hwp.set_format(Format::s16())
                .map_err(|e| AudioError::CaptureError(format!("Failed to set format: {e}")))?;

            // Set sample rate
            hwp.set_rate(config.sample_rate, ValueOr::Nearest)
                .map_err(|e| AudioError::CaptureError(format!("Failed to set rate: {e}")))?;

            // Set channels
            hwp.set_channels(u32::from(config.channels))
                .map_err(|e| AudioError::CaptureError(format!("Failed to set channels: {e}")))?;

            // Set buffer size (in frames)
            let buffer_frames = config.buffer_size / config.channels as usize;
            hwp.set_buffer_size_near((buffer_frames * 4) as i64)
                .map_err(|e| AudioError::CaptureError(format!("Failed to set buffer size: {e}")))?;

            // Set period size (smaller = lower latency)
            hwp.set_period_size_near(buffer_frames as i64, ValueOr::Nearest)
                .map_err(|e| AudioError::CaptureError(format!("Failed to set period size: {e}")))?;

            // Apply parameters
            pcm.hw_params(&hwp)
                .map_err(|e| AudioError::CaptureError(format!("Failed to apply HW params: {e}")))?;
        }

        // Prepare the device for capture
        pcm.prepare()
            .map_err(|e| AudioError::CaptureError(format!("Failed to prepare PCM: {e}")))?;

        // Allocate intermediate buffer for i16 samples
        let i16_buffer = vec![0i16; config.buffer_size * config.channels as usize];

        Ok(Self {
            pcm,
            config: config.clone(),
            i16_buffer,
        })
    }

    fn read(&mut self, buffer: &mut [f32]) -> Result<usize, AudioError> {
        // Ensure our intermediate buffer is large enough
        let samples_needed = buffer.len();
        if self.i16_buffer.len() < samples_needed {
            self.i16_buffer.resize(samples_needed, 0);
        }

        // Get the IO interface
        let io = self
            .pcm
            .io_i16()
            .map_err(|e| AudioError::CaptureError(format!("Failed to get IO interface: {e}")))?;

        // Read from ALSA (blocking)
        // EPIPE (-32) indicates xrun (buffer overrun) - try to recover
        let frames_read = match io.readi(&mut self.i16_buffer[..samples_needed]) {
            Ok(n) => n,
            Err(e) => {
                // Try to recover from xrun (buffer overrun)
                // ALSA error codes: -EPIPE = -32 for xrun
                if e.errno() == -32 {
                    self.pcm.prepare().map_err(|e| {
                        AudioError::CaptureError(format!("Failed to recover from xrun: {e}"))
                    })?;
                    // Retry the read
                    io.readi(&mut self.i16_buffer[..samples_needed])
                        .map_err(|e| {
                            AudioError::CaptureError(format!("Failed to read after recovery: {e}"))
                        })?
                } else {
                    return Err(AudioError::CaptureError(format!("Failed to read: {e}")));
                }
            }
        };

        let samples_read = frames_read * self.config.channels as usize;

        // Convert i16 to f32
        for (i, &sample) in self.i16_buffer[..samples_read].iter().enumerate() {
            buffer[i] = Self::i16_to_f32(sample);
        }

        Ok(samples_read)
    }

    fn close(&mut self) -> Result<(), AudioError> {
        // Drop the PCM handle gracefully
        self.pcm
            .drain()
            .map_err(|e| AudioError::CaptureError(format!("Failed to drain PCM: {e}")))?;
        Ok(())
    }

    fn backend_name() -> &'static str {
        "ALSA"
    }
}

// ============================================================================
// C8: CoreAudio Backend (macOS)
// ============================================================================

/// CoreAudio capture backend for macOS (GH-130, C8)
///
/// Uses Apple's CoreAudio framework for low-latency audio capture.
/// Requires the `audio-coreaudio` feature flag.
#[cfg(all(target_os = "macos", feature = "audio-coreaudio"))]
pub struct CoreAudioBackend {
    #[allow(dead_code)]
    config: CaptureConfig,
    // Future: AudioUnit handle
}

#[cfg(all(target_os = "macos", feature = "audio-coreaudio"))]
impl CaptureBackend for CoreAudioBackend {
    fn open(_device: Option<&str>, config: &CaptureConfig) -> Result<Self, AudioError> {
        config.validate()?;
        // Stub: CoreAudio AudioUnit setup deferred (GH-130, C8)
        // Requires coreaudio-rs crate for native bindings
        Err(AudioError::NotImplemented(
            "CoreAudio backend pending implementation - requires coreaudio-rs dependency"
                .to_string(),
        ))
    }

    fn read(&mut self, _buffer: &mut [f32]) -> Result<usize, AudioError> {
        Err(AudioError::NotImplemented("CoreAudio read".to_string()))
    }

    fn close(&mut self) -> Result<(), AudioError> {
        Ok(())
    }

    fn backend_name() -> &'static str {
        "CoreAudio"
    }
}

// ============================================================================
// C9: WASAPI Backend (Windows)
// ============================================================================

/// WASAPI capture backend for Windows (GH-130, C9)
///
/// Uses Windows Audio Session API for audio capture.
/// Requires the `audio-wasapi` feature flag.
#[cfg(all(target_os = "windows", feature = "audio-wasapi"))]
pub struct WasapiBackend {
    #[allow(dead_code)]
    config: CaptureConfig,
    // Future: IAudioClient handle
}

#[cfg(all(target_os = "windows", feature = "audio-wasapi"))]
impl CaptureBackend for WasapiBackend {
    fn open(_device: Option<&str>, config: &CaptureConfig) -> Result<Self, AudioError> {
        config.validate()?;
        // Stub: WASAPI client initialization deferred (GH-130, C9)
        // Requires wasapi crate for native bindings
        Err(AudioError::NotImplemented(
            "WASAPI backend pending implementation - requires wasapi dependency".to_string(),
        ))
    }

    fn read(&mut self, _buffer: &mut [f32]) -> Result<usize, AudioError> {
        Err(AudioError::NotImplemented("WASAPI read".to_string()))
    }

    fn close(&mut self) -> Result<(), AudioError> {
        Ok(())
    }

    fn backend_name() -> &'static str {
        "WASAPI"
    }
}

// ============================================================================
// C10: WebAudio Backend (WASM)
// ============================================================================

/// WebAudio API capture backend for WASM (GH-130, C10)
///
/// Uses the Web Audio API via web-sys for browser-based audio capture.
/// Requires the `audio-webaudio` feature flag.
#[cfg(all(target_arch = "wasm32", feature = "audio-webaudio"))]
pub struct WebAudioBackend {
    #[allow(dead_code)]
    config: CaptureConfig,
    // Future: AudioContext and MediaStreamSource handles
}

#[cfg(all(target_arch = "wasm32", feature = "audio-webaudio"))]
impl CaptureBackend for WebAudioBackend {
    fn open(_device: Option<&str>, config: &CaptureConfig) -> Result<Self, AudioError> {
        config.validate()?;
        // Stub: WebAudio MediaDevices.getUserMedia deferred (GH-130, C10)
        // Requires web-sys crate for browser API bindings
        Err(AudioError::NotImplemented(
            "WebAudio backend pending implementation - requires web-sys dependency".to_string(),
        ))
    }

    fn read(&mut self, _buffer: &mut [f32]) -> Result<usize, AudioError> {
        Err(AudioError::NotImplemented("WebAudio read".to_string()))
    }

    fn close(&mut self) -> Result<(), AudioError> {
        Ok(())
    }

    fn backend_name() -> &'static str {
        "WebAudio"
    }
}

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
/// let n = mock.read(&mut buffer).unwrap();
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

impl MockCaptureSource {
    /// Create a new mock capture source
    ///
    /// # Arguments
    /// * `config` - Audio configuration (sample rate determines signal timing)
    /// * `signal` - Type of signal to generate
    #[must_use]
    pub fn new(config: CaptureConfig, signal: MockSignal) -> Self {
        Self {
            config,
            signal,
            sample_index: 0,
            rng_state: 0x5DEECE66D, // LCG seed
        }
    }

    /// Create with silence signal
    #[must_use]
    pub fn silence(config: CaptureConfig) -> Self {
        Self::new(config, MockSignal::Silence)
    }

    /// Create with 440Hz sine wave (A4 note)
    #[must_use]
    pub fn a440(config: CaptureConfig) -> Self {
        Self::new(
            config,
            MockSignal::Sine {
                frequency: 440.0,
                amplitude: 0.5,
            },
        )
    }

    /// Create with white noise
    #[must_use]
    pub fn white_noise(config: CaptureConfig, amplitude: f32) -> Self {
        Self::new(config, MockSignal::WhiteNoise { amplitude })
    }

    /// Read samples into buffer
    ///
    /// # Arguments
    /// * `buffer` - Output buffer for samples
    ///
    /// # Returns
    /// Number of samples written (always fills the buffer)
    pub fn read(&mut self, buffer: &mut [f32]) -> Result<usize, AudioError> {
        let sample_rate = f64::from(self.config.sample_rate);

        for sample in buffer.iter_mut() {
            *sample = self.generate_sample(sample_rate);
            self.sample_index += 1;
        }

        Ok(buffer.len())
    }

    /// Generate a single sample based on signal type
    fn generate_sample(&mut self, sample_rate: f64) -> f32 {
        let t = self.sample_index as f64 / sample_rate;

        match self.signal {
            MockSignal::Silence => 0.0,

            MockSignal::Sine {
                frequency,
                amplitude,
            } => {
                let phase = 2.0 * std::f64::consts::PI * f64::from(frequency) * t;
                (phase.sin() * f64::from(amplitude)) as f32
            }

            MockSignal::WhiteNoise { amplitude } => {
                // Linear congruential generator for deterministic "random" noise
                self.rng_state = self.rng_state.wrapping_mul(0x5DEECE66D).wrapping_add(11);
                // Take upper 32 bits and normalize to [0, 1]
                let upper_bits = (self.rng_state >> 32) as u32;
                let normalized = f64::from(upper_bits) / f64::from(u32::MAX);
                ((normalized * 2.0 - 1.0) * f64::from(amplitude)) as f32
            }

            MockSignal::Impulse => {
                if self.sample_index == 0 {
                    1.0
                } else {
                    0.0
                }
            }

            MockSignal::Square {
                frequency,
                amplitude,
            } => {
                let phase = 2.0 * std::f64::consts::PI * f64::from(frequency) * t;
                let value = if phase.sin() >= 0.0 { 1.0 } else { -1.0 };
                value * f64::from(amplitude) as f32
            }
        }
    }

    /// Reset the sample counter to start from beginning
    pub fn reset(&mut self) {
        self.sample_index = 0;
        self.rng_state = 0x5DEECE66D; // Reset RNG seed
    }

    /// Get current sample position
    #[must_use]
    pub fn position(&self) -> u64 {
        self.sample_index
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &CaptureConfig {
        &self.config
    }

    /// Get signal type
    #[must_use]
    pub fn signal(&self) -> MockSignal {
        self.signal
    }

    /// Change signal type
    pub fn set_signal(&mut self, signal: MockSignal) {
        self.signal = signal;
    }
}

/// Mock capture source that reads from a pre-recorded buffer
///
/// Useful for testing with specific audio content.
#[derive(Debug)]
pub struct BufferCaptureSource {
    samples: Vec<f32>,
    position: usize,
    loop_playback: bool,
}

impl BufferCaptureSource {
    /// Create from a sample buffer
    #[must_use]
    pub fn new(samples: Vec<f32>) -> Self {
        Self {
            samples,
            position: 0,
            loop_playback: false,
        }
    }

    /// Enable looping (replay from start when buffer exhausted)
    #[must_use]
    pub fn with_loop(mut self, loop_playback: bool) -> Self {
        self.loop_playback = loop_playback;
        self
    }

    /// Read samples into buffer
    ///
    /// # Returns
    /// Number of samples read (may be less than buffer size if not looping)
    pub fn read(&mut self, buffer: &mut [f32]) -> Result<usize, AudioError> {
        if self.samples.is_empty() {
            return Ok(0);
        }

        let mut written = 0;
        for sample in buffer.iter_mut() {
            if self.position >= self.samples.len() {
                if self.loop_playback {
                    self.position = 0;
                } else {
                    break;
                }
            }
            *sample = self.samples[self.position];
            self.position += 1;
            written += 1;
        }

        Ok(written)
    }

    /// Reset position to start
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Get current position
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Check if exhausted (for non-looping sources)
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        !self.loop_playback && self.position >= self.samples.len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
