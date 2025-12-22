//! Chunked streaming primitives
//!
//! Provides utilities for processing audio in chunks for real-time applications.
//! This is essential for live transcription where audio arrives continuously.
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::audio::stream::{AudioChunker, ChunkConfig};
//!
//! let config = ChunkConfig::default();
//! let mut chunker = AudioChunker::new(config);
//!
//! // Feed audio as it arrives
//! chunker.push(&incoming_samples);
//!
//! // Process complete chunks
//! while let Some(chunk) = chunker.pop() {
//!     process_chunk(&chunk);
//! }
//! ```

/// Configuration for audio chunking
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Chunk size in samples
    pub chunk_size: usize,
    /// Overlap between chunks in samples (for smooth processing)
    pub overlap: usize,
    /// Sample rate in Hz
    pub sample_rate: u32,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            // 30 seconds at 16kHz (Whisper default)
            chunk_size: 16000 * 30,
            // 1 second overlap
            overlap: 16000,
            sample_rate: 16000,
        }
    }
}

impl ChunkConfig {
    /// Create config for real-time low-latency processing
    #[must_use]
    pub fn realtime() -> Self {
        Self {
            // 5 seconds
            chunk_size: 16000 * 5,
            // 0.5 second overlap
            overlap: 8000,
            sample_rate: 16000,
        }
    }

    /// Duration of each chunk in milliseconds
    #[must_use]
    pub fn chunk_duration_ms(&self) -> u64 {
        if self.sample_rate > 0 {
            (self.chunk_size as u64 * 1000) / u64::from(self.sample_rate)
        } else {
            0
        }
    }
}

/// Audio chunker for streaming processing
#[derive(Debug, Clone)]
pub struct AudioChunker {
    config: ChunkConfig,
    buffer: Vec<f32>,
}

impl AudioChunker {
    /// Create a new audio chunker
    #[must_use]
    pub fn new(config: ChunkConfig) -> Self {
        Self {
            config,
            buffer: Vec::new(),
        }
    }

    /// Push audio samples to the buffer
    pub fn push(&mut self, samples: &[f32]) {
        self.buffer.extend_from_slice(samples);
    }

    /// Pop a complete chunk if available
    ///
    /// Returns None if not enough samples have accumulated
    pub fn pop(&mut self) -> Option<Vec<f32>> {
        if self.buffer.len() >= self.config.chunk_size {
            let chunk: Vec<f32> = self.buffer.drain(..self.config.chunk_size).collect();

            // Keep overlap samples for next chunk
            if self.config.overlap > 0 && self.config.overlap < chunk.len() {
                let overlap_start = chunk.len() - self.config.overlap;
                let overlap: Vec<f32> = chunk[overlap_start..].to_vec();
                // Prepend overlap to remaining buffer
                let mut new_buffer = overlap;
                new_buffer.append(&mut self.buffer);
                self.buffer = new_buffer;
            }

            Some(chunk)
        } else {
            None
        }
    }

    /// Flush remaining samples (for end of stream)
    ///
    /// Returns the remaining buffer even if incomplete
    pub fn flush(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.buffer)
    }

    /// Check if buffer has a complete chunk ready
    #[must_use]
    pub fn has_chunk(&self) -> bool {
        self.buffer.len() >= self.config.chunk_size
    }

    /// Get current buffer length
    #[must_use]
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &ChunkConfig {
        &self.config
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_config_default() {
        let config = ChunkConfig::default();
        assert_eq!(config.chunk_size, 16000 * 30);
        assert_eq!(config.overlap, 16000);
        assert_eq!(config.sample_rate, 16000);
    }

    #[test]
    fn test_chunk_config_realtime() {
        let config = ChunkConfig::realtime();
        assert_eq!(config.chunk_size, 16000 * 5);
        assert_eq!(config.chunk_duration_ms(), 5000);
    }

    #[test]
    fn test_chunker_no_chunk() {
        let config = ChunkConfig {
            chunk_size: 100,
            overlap: 0,
            sample_rate: 16000,
        };
        let mut chunker = AudioChunker::new(config);
        chunker.push(&[0.0; 50]);
        assert!(!chunker.has_chunk());
        assert!(chunker.pop().is_none());
    }

    #[test]
    fn test_chunker_exact_chunk() {
        let config = ChunkConfig {
            chunk_size: 100,
            overlap: 0,
            sample_rate: 16000,
        };
        let mut chunker = AudioChunker::new(config);
        chunker.push(&[1.0; 100]);
        assert!(chunker.has_chunk());
        let chunk = chunker.pop();
        assert!(chunk.is_some());
        assert_eq!(chunk.map_or(0, |c| c.len()), 100);
    }

    #[test]
    fn test_chunker_with_overlap() {
        let config = ChunkConfig {
            chunk_size: 100,
            overlap: 20,
            sample_rate: 16000,
        };
        let mut chunker = AudioChunker::new(config);
        chunker.push(&[1.0; 100]);

        let chunk = chunker.pop();
        assert!(chunk.is_some());
        assert_eq!(chunk.map_or(0, |c| c.len()), 100);

        // Buffer should now have 20 overlap samples
        assert_eq!(chunker.buffer_len(), 20);
    }

    #[test]
    fn test_chunker_flush() {
        let config = ChunkConfig {
            chunk_size: 100,
            overlap: 0,
            sample_rate: 16000,
        };
        let mut chunker = AudioChunker::new(config);
        chunker.push(&[1.0; 50]);

        let remaining = chunker.flush();
        assert_eq!(remaining.len(), 50);
        assert_eq!(chunker.buffer_len(), 0);
    }

    #[test]
    fn test_chunker_clear() {
        let config = ChunkConfig {
            chunk_size: 100,
            overlap: 0,
            sample_rate: 16000,
        };
        let mut chunker = AudioChunker::new(config);
        chunker.push(&[1.0; 50]);
        chunker.clear();
        assert_eq!(chunker.buffer_len(), 0);
    }
}
