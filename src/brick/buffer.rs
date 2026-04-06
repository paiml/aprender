//! Buffer Management with Flow Control
//!
//! Two-tier watermarks for back-pressure control.

// ----------------------------------------------------------------------------
// AWP-01: Two-Tier Buffer Watermarks
// ----------------------------------------------------------------------------

/// Two-tier buffer watermarks for back-pressure control.
///
/// # Example
/// ```rust
/// use trueno::brick::BufferWatermarks;
///
/// let wm = BufferWatermarks::default();
/// assert!(!wm.should_backpressure(1000));  // Below high watermark
/// assert!(wm.should_backpressure(10000));  // Above high watermark
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BufferWatermarks {
    /// Low watermark: resume writing when buffer drops below this
    pub low: usize,
    /// High watermark: apply back-pressure when buffer exceeds this
    pub high: usize,
}

impl Default for BufferWatermarks {
    fn default() -> Self {
        Self {
            low: 1024,      // 1KB
            high: 8 * 1024, // 8KB
        }
    }
}

impl BufferWatermarks {
    /// Create new watermarks.
    ///
    /// # Panics
    /// Panics if low >= high.
    pub fn new(low: usize, high: usize) -> Self {
        assert!(low < high, "low watermark must be less than high");
        Self { low, high }
    }

    /// Check if back-pressure should be applied.
    #[must_use]
    pub fn should_backpressure(&self, current: usize) -> bool {
        current >= self.high
    }

    /// Check if writing can resume.
    #[must_use]
    pub fn can_write(&self, current: usize) -> bool {
        current < self.low
    }

    /// Get pressure level (0.0 = empty, 1.0 = at high watermark).
    #[must_use]
    pub fn pressure_level(&self, current: usize) -> f64 {
        (current as f64 / self.high as f64).min(1.0)
    }
}

/// Buffer with watermark-based flow control.
#[derive(Debug)]
pub struct WatermarkedBuffer {
    data: Vec<u8>,
    watermarks: BufferWatermarks,
}

impl WatermarkedBuffer {
    /// Create a new watermarked buffer.
    pub fn new(watermarks: BufferWatermarks) -> Self {
        Self { data: Vec::with_capacity(watermarks.high), watermarks }
    }

    /// Check if back-pressure should be applied.
    #[must_use]
    pub fn should_backpressure(&self) -> bool {
        self.watermarks.should_backpressure(self.data.len())
    }

    /// Check if writing can resume.
    #[must_use]
    pub fn can_write(&self) -> bool {
        self.watermarks.can_write(self.data.len())
    }

    /// Get current buffer length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Write data to the buffer.
    pub fn write(&mut self, data: &[u8]) {
        contract_pre_write!();
        self.data.extend_from_slice(data);
    }

    /// Drain data from the buffer.
    pub fn drain(&mut self, amount: usize) -> Vec<u8> {
        let amount = amount.min(self.data.len());
        self.data.drain(..amount).collect()
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get the watermarks configuration.
    #[must_use]
    pub fn watermarks(&self) -> BufferWatermarks {
        self.watermarks
    }

    /// Get current pressure level.
    #[must_use]
    pub fn pressure_level(&self) -> f64 {
        self.watermarks.pressure_level(self.data.len())
    }
}

impl Default for WatermarkedBuffer {
    fn default() -> Self {
        Self::new(BufferWatermarks::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_watermarks_default() {
        let wm = BufferWatermarks::default();
        assert_eq!(wm.low, 1024);
        assert_eq!(wm.high, 8 * 1024);
    }

    #[test]
    fn test_buffer_watermarks_new() {
        let wm = BufferWatermarks::new(100, 1000);
        assert_eq!(wm.low, 100);
        assert_eq!(wm.high, 1000);
    }

    #[test]
    #[should_panic(expected = "low watermark must be less than high")]
    fn test_buffer_watermarks_invalid() {
        BufferWatermarks::new(1000, 100);
    }

    #[test]
    fn test_buffer_watermarks_backpressure() {
        let wm = BufferWatermarks::new(100, 1000);
        assert!(!wm.should_backpressure(500));
        assert!(!wm.should_backpressure(999));
        assert!(wm.should_backpressure(1000));
        assert!(wm.should_backpressure(1500));
    }

    #[test]
    fn test_buffer_watermarks_can_write() {
        let wm = BufferWatermarks::new(100, 1000);
        assert!(wm.can_write(50));
        assert!(wm.can_write(99));
        assert!(!wm.can_write(100));
        assert!(!wm.can_write(500));
    }

    #[test]
    fn test_buffer_watermarks_pressure_level() {
        let wm = BufferWatermarks::new(100, 1000);
        assert!((wm.pressure_level(0) - 0.0).abs() < 0.001);
        assert!((wm.pressure_level(500) - 0.5).abs() < 0.001);
        assert!((wm.pressure_level(1000) - 1.0).abs() < 0.001);
        assert!((wm.pressure_level(2000) - 1.0).abs() < 0.001); // Capped at 1.0
    }

    #[test]
    fn test_watermarked_buffer_new() {
        let wm = BufferWatermarks::new(100, 1000);
        let buffer = WatermarkedBuffer::new(wm);
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_watermarked_buffer_write() {
        let mut buffer = WatermarkedBuffer::default();
        buffer.write(&[1, 2, 3, 4, 5]);
        assert_eq!(buffer.len(), 5);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_watermarked_buffer_drain() {
        let mut buffer = WatermarkedBuffer::default();
        buffer.write(&[1, 2, 3, 4, 5]);
        let drained = buffer.drain(3);
        assert_eq!(drained, vec![1, 2, 3]);
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_watermarked_buffer_drain_more_than_available() {
        let mut buffer = WatermarkedBuffer::default();
        buffer.write(&[1, 2, 3]);
        let drained = buffer.drain(10);
        assert_eq!(drained, vec![1, 2, 3]);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_watermarked_buffer_clear() {
        let mut buffer = WatermarkedBuffer::default();
        buffer.write(&[1, 2, 3, 4, 5]);
        buffer.clear();
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_watermarked_buffer_backpressure() {
        let wm = BufferWatermarks::new(10, 100);
        let mut buffer = WatermarkedBuffer::new(wm);

        assert!(!buffer.should_backpressure());
        assert!(buffer.can_write());

        buffer.write(&[0u8; 50]);
        assert!(!buffer.should_backpressure());
        assert!(!buffer.can_write());

        buffer.write(&[0u8; 50]);
        assert!(buffer.should_backpressure());
    }

    #[test]
    fn test_watermarked_buffer_pressure_level() {
        let wm = BufferWatermarks::new(10, 100);
        let mut buffer = WatermarkedBuffer::new(wm);

        assert!((buffer.pressure_level() - 0.0).abs() < 0.001);

        buffer.write(&[0u8; 50]);
        assert!((buffer.pressure_level() - 0.5).abs() < 0.001);

        buffer.write(&[0u8; 50]);
        assert!((buffer.pressure_level() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_watermarked_buffer_watermarks_getter() {
        let wm = BufferWatermarks::new(100, 1000);
        let buffer = WatermarkedBuffer::new(wm);
        let retrieved = buffer.watermarks();
        assert_eq!(retrieved.low, 100);
        assert_eq!(retrieved.high, 1000);
    }

    /// FALSIFICATION TEST (Section 4.3)
    ///
    /// Verifies the buffer's behavior when pushed past its high watermark.
    /// The WatermarkedBuffer is a SIGNALING mechanism, not a blocking mechanism.
    /// It will accept writes beyond the limit but MUST correctly signal backpressure.
    ///
    /// Failure mode: If this test passes but the buffer doesn't signal,
    /// the caller would never know to stop writing, leading to OOM.
    #[test]
    fn test_falsify_buffer_limit_signaling() {
        let wm = BufferWatermarks::new(10, 100);
        let mut buffer = WatermarkedBuffer::new(wm);

        // Write incrementally and verify signaling at each step
        let mut backpressure_signaled_at = None;

        for i in 0..200 {
            let chunk = [i as u8; 10]; // 10 bytes per write
            buffer.write(&chunk);

            if buffer.should_backpressure() && backpressure_signaled_at.is_none() {
                backpressure_signaled_at = Some(buffer.len());
            }
        }

        // CRITICAL ASSERTION: Backpressure MUST have been signaled
        assert!(
            backpressure_signaled_at.is_some(),
            "FALSIFICATION FAILED: Buffer accepted 2000 bytes without signaling backpressure"
        );

        // Verify it was signaled at the right threshold (at or above high watermark)
        let signal_point = backpressure_signaled_at.unwrap();
        assert!(
            signal_point >= 100,
            "FALSIFICATION FAILED: Backpressure signaled too early at {} bytes (high=100)",
            signal_point
        );

        // Verify buffer actually accepted all the data (it's non-blocking)
        assert_eq!(buffer.len(), 2000, "Buffer should accept all writes (signaling is advisory)");

        // Verify pressure level is capped at 1.0 even when way over
        assert!(
            (buffer.pressure_level() - 1.0).abs() < 0.001,
            "Pressure level should cap at 1.0 when over limit"
        );
    }

    /// FALSIFICATION TEST: Verify drain restores write capability
    #[test]
    fn test_falsify_drain_restores_writability() {
        let wm = BufferWatermarks::new(10, 100);
        let mut buffer = WatermarkedBuffer::new(wm);

        // Fill past high watermark
        buffer.write(&[0u8; 150]);
        assert!(buffer.should_backpressure());
        assert!(!buffer.can_write());

        // Drain to below low watermark
        buffer.drain(145); // Should leave 5 bytes
        assert_eq!(buffer.len(), 5);

        // CRITICAL: Must restore write capability
        assert!(
            buffer.can_write(),
            "FALSIFICATION FAILED: Draining below low watermark did not restore writability"
        );
        assert!(
            !buffer.should_backpressure(),
            "FALSIFICATION FAILED: Draining below low watermark did not clear backpressure"
        );
    }
}
