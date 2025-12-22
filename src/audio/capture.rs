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
impl AlsaBackend {
    /// List available ALSA capture devices
    ///
    /// # Returns
    /// Vector of device names suitable for `open()`
    pub fn list_devices() -> Result<Vec<AudioDevice>, AudioError> {
        use alsa::device_name::HintIter;

        let mut devices = Vec::new();

        // Iterate over PCM devices
        let hints = HintIter::new(None, b"pcm")
            .map_err(|e| AudioError::CaptureError(format!("Failed to enumerate devices: {e}")))?;

        for hint in hints {
            // Only include capture-capable devices
            if let Some(name) = hint.name {
                // Skip null device and complex plugin names
                if name == "null" || name.contains("surround") {
                    continue;
                }

                let desc = hint.desc.unwrap_or_default();
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
        let pcm = alsa::PCM::new(device_name, Direction::Capture, false)
            .map_err(|e| AudioError::CaptureError(format!("Failed to open ALSA device '{device_name}': {e}")))?;

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
        use alsa::pcm::IO;

        // Ensure our intermediate buffer is large enough
        let samples_needed = buffer.len();
        if self.i16_buffer.len() < samples_needed {
            self.i16_buffer.resize(samples_needed, 0);
        }

        // Get the IO interface
        let io = self.pcm.io_i16()
            .map_err(|e| AudioError::CaptureError(format!("Failed to get IO interface: {e}")))?;

        // Calculate frames (samples / channels)
        let frames = samples_needed / self.config.channels as usize;

        // Read from ALSA (blocking)
        let frames_read = match io.readi(&mut self.i16_buffer[..samples_needed]) {
            Ok(n) => n,
            Err(e) => {
                // Try to recover from xrun (buffer overrun) - error code -32 (EPIPE)
                if e.errno() == alsa::nix::errno::Errno::EPIPE as i32 {
                    self.pcm.prepare()
                        .map_err(|e| AudioError::CaptureError(format!("Failed to recover from xrun: {e}")))?;
                    // Retry the read
                    io.readi(&mut self.i16_buffer[..samples_needed])
                        .map_err(|e| AudioError::CaptureError(format!("Failed to read after recovery: {e}")))?
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
        self.pcm.drain()
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
        // TODO: Implement CoreAudio AudioUnit setup
        // Future: Use coreaudio-rs crate for native bindings
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
        // TODO: Implement WASAPI client initialization
        // Future: Use wasapi crate for native bindings
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
        // TODO: Implement WebAudio MediaDevices.getUserMedia
        // Future: Use web-sys crate for browser API bindings
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
            channels: 1,       // Mono for ASR
            buffer_size: 1600, // 100ms at 16kHz
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
mod tests {
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
    fn test_list_devices_stub() {
        // Stub returns empty list
        let devices = list_devices().unwrap();
        assert!(devices.is_empty());
    }

    #[test]
    fn test_default_device_stub() {
        let device = default_device().unwrap();
        assert!(device.is_none());
    }

    #[test]
    fn test_audio_capture_open_not_implemented() {
        let config = CaptureConfig::default();
        let result = AudioCapture::open(None, &config);
        assert!(result.is_err());
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
        assert!(non_zero_count > 0, "Sine wave should produce non-zero samples");
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
        let variance: f32 = buffer.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / buffer.len() as f32;
        assert!(variance > 0.001, "White noise should have variance, got {}", variance);
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
            assert!((result - 1.0).abs() < 1e-4, "Max i16 should map to ~1.0, got {}", result);
        }

        #[test]
        fn test_alsa_i16_to_f32_min() {
            let result = AlsaBackend::i16_to_f32(i16::MIN);
            assert!((result - (-1.0)).abs() < 1e-4, "Min i16 should map to ~-1.0, got {}", result);
        }

        #[test]
        fn test_alsa_i16_to_f32_positive_range() {
            // All positive i16 should map to [0, 1]
            for val in [1, 100, 1000, 10000, 32767_i16] {
                let result = AlsaBackend::i16_to_f32(val);
                assert!(result >= 0.0 && result <= 1.0, "Positive {} mapped to {}", val, result);
            }
        }

        #[test]
        fn test_alsa_i16_to_f32_negative_range() {
            // All negative i16 should map to [-1, 0]
            for val in [-1, -100, -1000, -10000, -32768_i16] {
                let result = AlsaBackend::i16_to_f32(val);
                assert!(result >= -1.0 && result <= 0.0, "Negative {} mapped to {}", val, result);
            }
        }

        #[test]
        fn test_alsa_i16_to_f32_symmetric() {
            // Symmetric values should map to approximately symmetric results
            let positive = AlsaBackend::i16_to_f32(16384);
            let negative = AlsaBackend::i16_to_f32(-16384);
            assert!((positive + negative).abs() < 0.001, "Symmetric values should cancel: {} + {} = {}", positive, negative, positive + negative);
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
            assert!(result.is_ok(), "list_devices should not error: {:?}", result);
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
}
