//! Image buffer type for structured pixel data.
//!
//! Wraps flat pixel buffers with width, height, and channel metadata.

use crate::error::ImageError;

/// Pixel data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit float (default for processing).
    F32,
    /// 8-bit unsigned integer (display/storage).
    U8,
}

/// Structured image buffer with metadata.
///
/// Stores pixels in row-major order: `data[y * width * channels + x * channels + c]`.
#[derive(Debug, Clone)]
pub struct ImageBuf {
    data: Vec<f32>,
    width: usize,
    height: usize,
    channels: usize,
    dtype: DType,
}

impl ImageBuf {
    /// Create a new image buffer.
    ///
    /// # Errors
    ///
    /// Returns error if data length doesn't match `width * height * channels`.
    pub fn new(
        data: Vec<f32>,
        width: usize,
        height: usize,
        channels: usize,
    ) -> Result<Self, ImageError> {
        let expected = width * height * channels;
        if data.len() != expected {
            return Err(ImageError::DimensionMismatch { expected, got: data.len() });
        }
        Ok(Self { data, width, height, channels, dtype: DType::F32 })
    }

    /// Create a zero-filled image buffer.
    pub fn zeros(width: usize, height: usize, channels: usize) -> Self {
        Self {
            data: vec![0.0; width * height * channels],
            width,
            height,
            channels,
            dtype: DType::F32,
        }
    }

    /// Image width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Number of channels (1=gray, 3=RGB, 4=RGBA).
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Pixel data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Raw pixel data as slice.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable raw pixel data.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Total number of elements (`width * height * channels`).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Extract a single channel as a new grayscale `ImageBuf`.
    ///
    /// # Errors
    ///
    /// Returns error if `channel >= self.channels()`.
    pub fn channel(&self, channel: usize) -> Result<Self, ImageError> {
        if channel >= self.channels {
            return Err(ImageError::InvalidChannel { channel, max: self.channels });
        }
        let mut out = Vec::with_capacity(self.width * self.height);
        for i in 0..self.width * self.height {
            out.push(self.data[i * self.channels + channel]);
        }
        Ok(Self {
            data: out,
            width: self.width,
            height: self.height,
            channels: 1,
            dtype: self.dtype,
        })
    }
}
