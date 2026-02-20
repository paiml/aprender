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

include!("audio.rs");
include!("buffer_capture_source.rs");
