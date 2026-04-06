//! GIF Recording for Test Documentation (Feature 1)
//!
//! Record test execution as animated GIF for documentation, bug reports,
//! and visual verification.
//!
//! ## EXTREME TDD: Tests written FIRST per spec
//!
//! ## Toyota Way Application
//!
//! - **Poka-Yoke**: Type-safe frame capture prevents format mismatches
//! - **Muda**: Lazy frame encoding reduces memory pressure

use crate::driver::Screenshot;
use crate::result::{ProbarError, ProbarResult};
use gif::{Encoder, Frame, Repeat};
use image::{DynamicImage, GenericImageView, ImageFormat};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Configuration for GIF recording
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GifConfig {
    /// Frames per second (10-30 typical)
    pub fps: u8,
    /// Output width in pixels
    pub width: u32,
    /// Output height in pixels
    pub height: u32,
    /// Quality level (1-100, affects palette quantization)
    pub quality: u8,
    /// Loop count (0 = infinite)
    pub loop_count: u16,
}

impl Default for GifConfig {
    fn default() -> Self {
        Self {
            fps: 10,
            width: 800,
            height: 600,
            quality: 80,
            loop_count: 0, // Infinite loop
        }
    }
}

impl GifConfig {
    /// Create a new GIF configuration
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Set frames per second
    #[must_use]
    pub fn with_fps(mut self, fps: u8) -> Self {
        self.fps = fps.clamp(1, 60);
        self
    }

    /// Set quality (1-100)
    #[must_use]
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = quality.clamp(1, 100);
        self
    }

    /// Set loop count (0 = infinite)
    #[must_use]
    pub fn with_loop_count(mut self, count: u16) -> Self {
        self.loop_count = count;
        self
    }

    /// Calculate frame delay in centiseconds (GIF standard)
    #[must_use]
    pub fn frame_delay_cs(&self) -> u16 {
        // GIF delay is in centiseconds (1/100th of a second)
        // fps=10 -> 10 centiseconds delay
        // fps=30 -> ~3 centiseconds delay
        (100 / u16::from(self.fps.max(1))).max(1)
    }
}

/// A single frame in the GIF recording
#[derive(Debug, Clone)]
pub struct GifFrame {
    /// RGBA pixel data
    pub data: Vec<u8>,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Timestamp when frame was captured (relative to start)
    pub timestamp_ms: u64,
}

impl GifFrame {
    /// Create a new frame from RGBA data
    #[must_use]
    pub fn new(data: Vec<u8>, width: u32, height: u32, timestamp_ms: u64) -> Self {
        Self {
            data,
            width,
            height,
            timestamp_ms,
        }
    }

    /// Create a frame from a Screenshot
    pub fn from_screenshot(screenshot: &Screenshot, timestamp_ms: u64) -> ProbarResult<Self> {
        // Decode PNG data from screenshot
        let img = image::load_from_memory_with_format(&screenshot.data, ImageFormat::Png).map_err(
            |e| ProbarError::ImageProcessing {
                message: format!("Failed to decode screenshot: {e}"),
            },
        )?;

        let (width, height) = img.dimensions();
        let rgba = img.to_rgba8();

        Ok(Self {
            data: rgba.into_raw(),
            width,
            height,
            timestamp_ms,
        })
    }
}

/// GIF Recorder for capturing test execution
///
/// ## Example
///
/// ```ignore
/// let mut recorder = GifRecorder::new(GifConfig::new(800, 600));
/// recorder.start()?;
///
/// // Capture frames during test execution
/// for screenshot in screenshots {
///     recorder.capture_frame(&screenshot)?;
/// }
///
/// let gif_data = recorder.stop()?;
/// recorder.save(Path::new("test_recording.gif"))?;
/// ```
#[derive(Debug)]
pub struct GifRecorder {
    config: GifConfig,
    frames: Vec<GifFrame>,
    recording: bool,
    start_time_ms: u64,
    encoded_data: Option<Vec<u8>>,
}

impl GifRecorder {
    /// Create a new GIF recorder with the given configuration
    #[must_use]
    pub fn new(config: GifConfig) -> Self {
        Self {
            config,
            frames: Vec::new(),
            recording: false,
            start_time_ms: 0,
            encoded_data: None,
        }
    }

    /// Get the current configuration
    #[must_use]
    pub fn config(&self) -> &GifConfig {
        &self.config
    }

    /// Check if recording is in progress
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Get the number of captured frames
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Start recording
    ///
    /// # Errors
    ///
    /// Returns error if already recording
    pub fn start(&mut self) -> ProbarResult<()> {
        if self.recording {
            return Err(ProbarError::InvalidState {
                message: "GIF recording already in progress".to_string(),
            });
        }

        self.frames.clear();
        self.encoded_data = None;
        self.recording = true;
        self.start_time_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Ok(())
    }

    /// Capture a frame from a screenshot
    ///
    /// # Errors
    ///
    /// Returns error if not recording or image processing fails
    pub fn capture_frame(&mut self, screenshot: &Screenshot) -> ProbarResult<()> {
        if !self.recording {
            return Err(ProbarError::InvalidState {
                message: "GIF recording not started".to_string(),
            });
        }

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let timestamp_ms = current_time.saturating_sub(self.start_time_ms);

        let frame = GifFrame::from_screenshot(screenshot, timestamp_ms)?;
        self.frames.push(frame);

        Ok(())
    }

    /// Add a raw frame directly
    ///
    /// # Errors
    ///
    /// Returns error if not recording
    pub fn add_frame(&mut self, frame: GifFrame) -> ProbarResult<()> {
        if !self.recording {
            return Err(ProbarError::InvalidState {
                message: "GIF recording not started".to_string(),
            });
        }

        self.frames.push(frame);
        Ok(())
    }

    /// Stop recording and encode the GIF
    ///
    /// # Errors
    ///
    /// Returns error if not recording or encoding fails
    pub fn stop(&mut self) -> ProbarResult<Vec<u8>> {
        if !self.recording {
            return Err(ProbarError::InvalidState {
                message: "GIF recording not started".to_string(),
            });
        }

        self.recording = false;

        if self.frames.is_empty() {
            return Err(ProbarError::InvalidState {
                message: "No frames captured".to_string(),
            });
        }

        let encoded = self.encode_gif()?;
        self.encoded_data = Some(encoded.clone());

        Ok(encoded)
    }

    /// Get the encoded GIF data (if available)
    #[must_use]
    pub fn encoded_data(&self) -> Option<&[u8]> {
        self.encoded_data.as_deref()
    }

    /// Save the GIF to a file
    ///
    /// # Errors
    ///
    /// Returns error if no encoded data or file write fails
    pub fn save(&self, path: &Path) -> ProbarResult<()> {
        let data = self
            .encoded_data
            .as_ref()
            .ok_or_else(|| ProbarError::InvalidState {
                message: "No encoded GIF data. Call stop() first.".to_string(),
            })?;

        std::fs::write(path, data)?;

        Ok(())
    }

    /// Encode all frames into a GIF
    fn encode_gif(&self) -> ProbarResult<Vec<u8>> {
        let mut output = Vec::new();

        // Use the configured dimensions
        let width = self.config.width as u16;
        let height = self.config.height as u16;

        {
            let mut encoder = Encoder::new(&mut output, width, height, &[]).map_err(|e| {
                ProbarError::ImageProcessing {
                    message: format!("Failed to create GIF encoder: {e}"),
                }
            })?;

            // Set loop behavior
            let repeat = if self.config.loop_count == 0 {
                Repeat::Infinite
            } else {
                Repeat::Finite(self.config.loop_count)
            };
            encoder
                .set_repeat(repeat)
                .map_err(|e| ProbarError::ImageProcessing {
                    message: format!("Failed to set GIF repeat: {e}"),
                })?;

            let frame_delay = self.config.frame_delay_cs();

            for gif_frame in &self.frames {
                // Resize frame if needed
                let rgba_data = self.resize_frame(gif_frame)?;

                // Convert RGBA to indexed color
                let mut frame = Frame::from_rgba_speed(
                    width,
                    height,
                    &mut rgba_data.clone(),
                    self.quality_to_speed(),
                );
                frame.delay = frame_delay;

                encoder
                    .write_frame(&frame)
                    .map_err(|e| ProbarError::ImageProcessing {
                        message: format!("Failed to write GIF frame: {e}"),
                    })?;
            }
        }

        Ok(output)
    }

    /// Resize a frame to match the configured dimensions
    fn resize_frame(&self, frame: &GifFrame) -> ProbarResult<Vec<u8>> {
        if frame.width == self.config.width && frame.height == self.config.height {
            return Ok(frame.data.clone());
        }

        // Create image from frame data
        let img = DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(frame.width, frame.height, frame.data.clone()).ok_or_else(
                || ProbarError::ImageProcessing {
                    message: "Invalid frame data dimensions".to_string(),
                },
            )?,
        );

        // Resize to target dimensions
        let resized = img.resize_exact(
            self.config.width,
            self.config.height,
            image::imageops::FilterType::Triangle,
        );

        Ok(resized.to_rgba8().into_raw())
    }

    /// Convert quality (1-100) to GIF encoder speed (1-30)
    fn quality_to_speed(&self) -> i32 {
        // Higher quality = lower speed (more processing)
        // quality 100 -> speed 1 (slowest, best quality)
        // quality 1 -> speed 30 (fastest, worst quality)
        let normalized = i32::from(100 - self.config.quality);
        (normalized * 29 / 100 + 1).clamp(1, 30)
    }
}

// ============================================================================
// EXTREME TDD: Tests written FIRST per spec
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use image::{ImageFormat, Rgba};
    use std::io::Cursor;

    mod gif_config_tests {
        use super::*;

        #[test]
        fn test_default_config() {
            let config = GifConfig::default();
            assert_eq!(config.fps, 10);
            assert_eq!(config.width, 800);
            assert_eq!(config.height, 600);
            assert_eq!(config.quality, 80);
            assert_eq!(config.loop_count, 0);
        }

        #[test]
        fn test_new_config() {
            let config = GifConfig::new(1920, 1080);
            assert_eq!(config.width, 1920);
            assert_eq!(config.height, 1080);
        }

        #[test]
        fn test_with_fps() {
            let config = GifConfig::default().with_fps(30);
            assert_eq!(config.fps, 30);
        }

        #[test]
        fn test_fps_clamping() {
            let config = GifConfig::default().with_fps(100);
            assert_eq!(config.fps, 60); // Clamped to max

            let config = GifConfig::default().with_fps(0);
            assert_eq!(config.fps, 1); // Clamped to min
        }

        #[test]
        fn test_with_quality() {
            let config = GifConfig::default().with_quality(50);
            assert_eq!(config.quality, 50);
        }

        #[test]
        fn test_quality_clamping() {
            let config = GifConfig::default().with_quality(150);
            assert_eq!(config.quality, 100);

            let config = GifConfig::default().with_quality(0);
            assert_eq!(config.quality, 1);
        }

        #[test]
        fn test_with_loop_count() {
            let config = GifConfig::default().with_loop_count(3);
            assert_eq!(config.loop_count, 3);
        }

        #[test]
        fn test_frame_delay_calculation() {
            let config = GifConfig::default().with_fps(10);
            assert_eq!(config.frame_delay_cs(), 10); // 10 fps = 100ms = 10cs

            let config = GifConfig::default().with_fps(20);
            assert_eq!(config.frame_delay_cs(), 5); // 20 fps = 50ms = 5cs

            let config = GifConfig::default().with_fps(1);
            assert_eq!(config.frame_delay_cs(), 100); // 1 fps = 1000ms = 100cs
        }
    }

    mod gif_frame_tests {
        use super::*;

        #[test]
        fn test_new_frame() {
            let data = vec![255, 0, 0, 255]; // One red pixel RGBA
            let frame = GifFrame::new(data.clone(), 1, 1, 100);

            assert_eq!(frame.data, data);
            assert_eq!(frame.width, 1);
            assert_eq!(frame.height, 1);
            assert_eq!(frame.timestamp_ms, 100);
        }

        #[test]
        fn test_frame_from_screenshot() {
            // Create a simple 2x2 PNG
            let mut img = image::RgbaImage::new(2, 2);
            img.put_pixel(0, 0, Rgba([255, 0, 0, 255])); // Red
            img.put_pixel(1, 0, Rgba([0, 255, 0, 255])); // Green
            img.put_pixel(0, 1, Rgba([0, 0, 255, 255])); // Blue
            img.put_pixel(1, 1, Rgba([255, 255, 255, 255])); // White

            let mut png_data = Vec::new();
            img.write_to(&mut Cursor::new(&mut png_data), ImageFormat::Png)
                .unwrap();

            let screenshot = Screenshot::new(png_data, 2, 2);
            let frame = GifFrame::from_screenshot(&screenshot, 500).unwrap();

            assert_eq!(frame.width, 2);
            assert_eq!(frame.height, 2);
            assert_eq!(frame.timestamp_ms, 500);
            assert_eq!(frame.data.len(), 16); // 2x2 pixels * 4 bytes RGBA
        }
    }

    mod gif_recorder_tests {
        use super::*;

        fn create_test_screenshot(width: u32, height: u32, color: [u8; 4]) -> Screenshot {
            let mut img = image::RgbaImage::new(width, height);
            for pixel in img.pixels_mut() {
                *pixel = Rgba(color);
            }

            let mut png_data = Vec::new();
            img.write_to(&mut Cursor::new(&mut png_data), ImageFormat::Png)
                .unwrap();

            Screenshot::new(png_data, width, height)
        }

        #[test]
        fn test_new_recorder() {
            let config = GifConfig::new(800, 600);
            let recorder = GifRecorder::new(config);

            assert_eq!(recorder.config().width, 800);
            assert_eq!(recorder.config().height, 600);
            assert!(!recorder.is_recording());
            assert_eq!(recorder.frame_count(), 0);
        }

        #[test]
        fn test_start_recording() {
            let mut recorder = GifRecorder::new(GifConfig::default());

            assert!(recorder.start().is_ok());
            assert!(recorder.is_recording());
        }

        #[test]
        fn test_start_recording_twice_fails() {
            let mut recorder = GifRecorder::new(GifConfig::default());

            recorder.start().unwrap();
            let result = recorder.start();

            assert!(result.is_err());
        }

        #[test]
        fn test_capture_frame() {
            let mut recorder = GifRecorder::new(GifConfig::new(100, 100));
            recorder.start().unwrap();

            let screenshot = create_test_screenshot(100, 100, [255, 0, 0, 255]);
            let result = recorder.capture_frame(&screenshot);

            assert!(result.is_ok());
            assert_eq!(recorder.frame_count(), 1);
        }

        #[test]
        fn test_capture_frame_without_start_fails() {
            let mut recorder = GifRecorder::new(GifConfig::default());
            let screenshot = create_test_screenshot(100, 100, [255, 0, 0, 255]);

            let result = recorder.capture_frame(&screenshot);
            assert!(result.is_err());
        }

        #[test]
        fn test_add_frame() {
            let mut recorder = GifRecorder::new(GifConfig::new(100, 100));
            recorder.start().unwrap();

            let frame = GifFrame::new(vec![255, 0, 0, 255], 1, 1, 0);
            let result = recorder.add_frame(frame);

            assert!(result.is_ok());
            assert_eq!(recorder.frame_count(), 1);
        }

        #[test]
        fn test_stop_recording() {
            let mut recorder = GifRecorder::new(GifConfig::new(10, 10));
            recorder.start().unwrap();

            let screenshot = create_test_screenshot(10, 10, [255, 0, 0, 255]);
            recorder.capture_frame(&screenshot).unwrap();

            let result = recorder.stop();
            assert!(result.is_ok());
            assert!(!recorder.is_recording());
            assert!(recorder.encoded_data().is_some());
        }

        #[test]
        fn test_stop_without_frames_fails() {
            let mut recorder = GifRecorder::new(GifConfig::default());
            recorder.start().unwrap();

            let result = recorder.stop();
            assert!(result.is_err());
        }

        #[test]
        fn test_stop_without_start_fails() {
            let mut recorder = GifRecorder::new(GifConfig::default());

            let result = recorder.stop();
            assert!(result.is_err());
        }

        #[test]
        fn test_save_gif() {
            let mut recorder = GifRecorder::new(GifConfig::new(10, 10));
            recorder.start().unwrap();

            let screenshot = create_test_screenshot(10, 10, [255, 0, 0, 255]);
            recorder.capture_frame(&screenshot).unwrap();
            recorder.stop().unwrap();

            let temp_dir = tempfile::tempdir().unwrap();
            let path = temp_dir.path().join("test.gif");

            let result = recorder.save(&path);
            assert!(result.is_ok());
            assert!(path.exists());

            // Verify it's a valid GIF
            let data = std::fs::read(&path).unwrap();
            assert_eq!(&data[0..6], b"GIF89a");
        }

        #[test]
        fn test_save_without_encoding_fails() {
            let recorder = GifRecorder::new(GifConfig::default());
            let temp_dir = tempfile::tempdir().unwrap();
            let path = temp_dir.path().join("test.gif");

            let result = recorder.save(&path);
            assert!(result.is_err());
        }

        #[test]
        fn test_multiple_frames() {
            let mut recorder = GifRecorder::new(GifConfig::new(10, 10).with_fps(10));
            recorder.start().unwrap();

            // Add multiple frames with different colors
            for color in [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]] {
                let screenshot = create_test_screenshot(10, 10, color);
                recorder.capture_frame(&screenshot).unwrap();
            }

            assert_eq!(recorder.frame_count(), 3);

            let gif_data = recorder.stop().unwrap();
            assert!(!gif_data.is_empty());
            assert_eq!(&gif_data[0..6], b"GIF89a");
        }

        #[test]
        fn test_frame_resizing() {
            let mut recorder = GifRecorder::new(GifConfig::new(50, 50));
            recorder.start().unwrap();

            // Add a frame with different dimensions
            let screenshot = create_test_screenshot(100, 100, [255, 0, 0, 255]);
            recorder.capture_frame(&screenshot).unwrap();

            let gif_data = recorder.stop().unwrap();
            assert!(!gif_data.is_empty());

            // Verify GIF has correct dimensions (bytes 6-9 contain width and height)
            let width = u16::from_le_bytes([gif_data[6], gif_data[7]]);
            let height = u16::from_le_bytes([gif_data[8], gif_data[9]]);
            assert_eq!(width, 50);
            assert_eq!(height, 50);
        }
    }

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_config_dimensions_preserved(width in 1u32..4096, height in 1u32..4096) {
                let config = GifConfig::new(width, height);
                prop_assert_eq!(config.width, width);
                prop_assert_eq!(config.height, height);
            }

            #[test]
            fn prop_fps_always_valid(fps in 0u8..=255) {
                let config = GifConfig::default().with_fps(fps);
                prop_assert!(config.fps >= 1);
                prop_assert!(config.fps <= 60);
            }

            #[test]
            fn prop_quality_always_valid(quality in 0u8..=255) {
                let config = GifConfig::default().with_quality(quality);
                prop_assert!(config.quality >= 1);
                prop_assert!(config.quality <= 100);
            }

            #[test]
            fn prop_frame_delay_always_positive(fps in 1u8..=60) {
                let config = GifConfig::default().with_fps(fps);
                prop_assert!(config.frame_delay_cs() >= 1);
            }
        }
    }
}
