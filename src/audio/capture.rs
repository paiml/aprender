//! Audio capture module for real-time streaming (GH-130).
//!
//! Provides cross-platform audio capture for:
//! - Microphone input
//! - System audio loopback
//! - File streaming
//!
//! # Platform Support
//!
//! | Platform | Backend | Status |
//! |----------|---------|--------|
//! | Linux | ALSA / PulseAudio | Planned |
//! | macOS | CoreAudio | Planned |
//! | Windows | WASAPI | Planned |
//! | WASM | WebAudio API | Planned |
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
/// # Note
/// This is a stub - actual implementation requires platform-specific code.
pub fn list_devices() -> Result<Vec<AudioDevice>, AudioError> {
    // Stub: Return empty list until platform backends implemented
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
#[derive(Debug)]
pub struct AudioCapture {
    config: CaptureConfig,
    #[allow(dead_code)] // For future platform implementations
    device_id: Option<String>,
    running: bool,
}

impl AudioCapture {
    /// Open audio capture stream
    ///
    /// # Arguments
    /// * `device` - Device ID, or None for default
    /// * `config` - Capture configuration
    ///
    /// # Note
    /// This is a stub - returns NotImplemented error.
    pub fn open(_device: Option<&str>, config: &CaptureConfig) -> Result<Self, AudioError> {
        config.validate()?;

        // Stub: Platform implementation needed
        Err(AudioError::NotImplemented(
            "Audio capture requires platform-specific backend (ALSA/CoreAudio/WASAPI)"
                .to_string(),
        ))
    }

    /// Read audio samples into buffer
    ///
    /// # Arguments
    /// * `buffer` - Output buffer for samples (f32 normalized to [-1.0, 1.0])
    ///
    /// # Returns
    /// Number of samples read, or error
    pub fn read(&mut self, buffer: &mut [f32]) -> Result<usize, AudioError> {
        if !self.running {
            return Err(AudioError::NotRunning);
        }

        // Stub: Platform implementation needed
        let _ = buffer;
        Err(AudioError::NotImplemented("read".to_string()))
    }

    /// Get capture configuration
    #[must_use]
    pub fn config(&self) -> &CaptureConfig {
        &self.config
    }

    /// Check if capture is running
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Close the capture stream
    pub fn close(mut self) -> Result<(), AudioError> {
        self.running = false;
        Ok(())
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
}
