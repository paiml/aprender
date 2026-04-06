
/// Quantized dot product for Q8_0 blocks (M23 - IMP-052)
///
/// Computes dot product directly on Q8_0 quantized data without full dequantization.
/// Q8_0 format: 2 bytes (f16 scale) + 32 bytes (32 x i8 values)
#[must_use]
pub fn quantized_dot_q8(block_a: &[u8], block_b: &[u8]) -> f32 {
    if block_a.len() < 34 || block_b.len() < 34 {
        return 0.0;
    }

    // Extract scales (f16 little-endian)
    let scale_a = half::f16::from_le_bytes([block_a[0], block_a[1]]).to_f32();
    let scale_b = half::f16::from_le_bytes([block_b[0], block_b[1]]).to_f32();

    // Accumulate dot product over i8 values
    let mut acc = 0i32;

    for i in 0..32 {
        let a_val = block_a[2 + i] as i8 as i32;
        let b_val = block_b[2 + i] as i8 as i32;
        acc += a_val * b_val;
    }

    // Apply combined scale
    (acc as f32) * scale_a * scale_b
}

/// Quantized matrix-vector multiply for Q4_0 weights (M23 - IMP-053)
/// Q4_0 quantization constants
const Q4_BLOCK_SIZE: usize = 18; // 2 bytes scale + 16 bytes data
const Q4_BLOCK_VALUES: usize = 32;

/// Process a Q4_0 block for matvec, returning dot product contribution
#[inline]
fn process_q4_block(
    weights: &[u8],
    block_offset: usize,
    input: &[f32],
    input_offset: usize,
    cols: usize,
) -> f32 {
    let scale =
        half::f16::from_le_bytes([weights[block_offset], weights[block_offset + 1]]).to_f32();
    let mut acc = 0.0f32;

    for i in 0..16 {
        let byte = weights[block_offset + 2 + i];
        let val_lo = (byte & 0x0F) as i32 - 8;
        let val_hi = ((byte >> 4) & 0x0F) as i32 - 8;
        let in_idx_lo = input_offset + i * 2;
        let in_idx_hi = input_offset + i * 2 + 1;

        if in_idx_lo < cols {
            acc += (val_lo as f32) * scale * input[in_idx_lo];
        }
        if in_idx_hi < cols {
            acc += (val_hi as f32) * scale * input[in_idx_hi];
        }
    }
    acc
}

///
/// Computes y = W @ x where W is Q4_0 quantized without full dequantization.
/// Each row of W consists of ceil(cols/32) Q4_0 blocks.
#[must_use]
pub fn quantized_matvec_q4(weights: &[u8], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let blocks_per_row = cols.div_ceil(Q4_BLOCK_VALUES);
    let row_bytes = blocks_per_row * Q4_BLOCK_SIZE;
    let mut output = vec![0.0f32; rows];

    for (row, out_val) in output.iter_mut().enumerate().take(rows) {
        let row_offset = row * row_bytes;
        let mut acc = 0.0f32;

        for block_idx in 0..blocks_per_row {
            let block_offset = row_offset + block_idx * Q4_BLOCK_SIZE;
            if block_offset + Q4_BLOCK_SIZE > weights.len() {
                break;
            }
            acc += process_q4_block(
                weights,
                block_offset,
                input,
                block_idx * Q4_BLOCK_VALUES,
                cols,
            );
        }

        *out_val = acc;
    }

    output
}

/// Quantized matrix-vector multiply for Q8_0 weights (M23 - IMP-053)
///
/// Computes y = W @ x where W is Q8_0 quantized without full dequantization.
/// Each row of W consists of ceil(cols/32) Q8_0 blocks.
#[must_use]
pub fn quantized_matvec_q8(weights: &[u8], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    const Q8_BLOCK_SIZE: usize = 34; // 2 bytes scale + 32 bytes data
    const Q8_BLOCK_VALUES: usize = 32;

    let blocks_per_row = cols.div_ceil(Q8_BLOCK_VALUES);
    let row_bytes = blocks_per_row * Q8_BLOCK_SIZE;

    let mut output = vec![0.0f32; rows];

    for (row, out_val) in output.iter_mut().enumerate().take(rows) {
        let row_offset = row * row_bytes;
        let mut acc = 0.0f32;

        for block_idx in 0..blocks_per_row {
            let block_offset = row_offset + block_idx * Q8_BLOCK_SIZE;

            if block_offset + Q8_BLOCK_SIZE > weights.len() {
                break;
            }

            // Extract scale
            let scale =
                half::f16::from_le_bytes([weights[block_offset], weights[block_offset + 1]])
                    .to_f32();

            // Process 32 values in this block
            let input_offset = block_idx * Q8_BLOCK_VALUES;

            for i in 0..32 {
                let val = weights[block_offset + 2 + i] as i8 as i32;
                let in_idx = input_offset + i;

                if in_idx < cols {
                    acc += (val as f32) * scale * input[in_idx];
                }
            }
        }

        *out_val = acc;
    }

    output
}

/// Mixed precision accumulator for quantized computations (M23 - IMP-054)
///
/// Accumulates values in f32 precision while processing quantized data,
/// ensuring numerical accuracy during block-wise operations.
#[derive(Debug, Clone, Default)]
pub struct QuantizedAccumulator {
    /// Running sum in f32 precision
    sum: f32,
}

impl QuantizedAccumulator {
    /// Create a new zeroed accumulator
    #[must_use]
    pub fn new() -> Self {
        Self { sum: 0.0 }
    }

    /// Get the current accumulated sum
    #[must_use]
    pub fn sum(&self) -> f32 {
        self.sum
    }

    /// Reset the accumulator to zero
    pub fn reset(&mut self) {
        self.sum = 0.0;
    }

    /// Add a scaled value to the accumulator
    #[inline]
    pub fn add_scaled(&mut self, value: f32, scale: f32) {
        self.sum += value * scale;
    }

    /// Add a block contribution (block_sum * block_scale)
    #[inline]
    pub fn add_block(&mut self, block_sum: f32, block_scale: f32) {
        self.sum += block_sum * block_scale;
    }
}

// =============================================================================
// M24: Streaming & Pipelining (Phase 15)
// =============================================================================

/// Double buffer for overlapping compute with memory operations (M24 - IMP-055)
///
/// Enables loading next layer weights while computing current layer.
/// Front buffer is read-only for compute, back buffer is writable for loading.
#[derive(Debug)]
pub struct DoubleBuffer<T> {
    front: Vec<T>,
    back: Vec<T>,
}

impl<T: Default + Clone> DoubleBuffer<T> {
    /// Create a new double buffer with given capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            front: vec![T::default(); capacity],
            back: vec![T::default(); capacity],
        }
    }

    /// Get the capacity of each buffer
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.front.len()
    }

    /// Get immutable reference to front buffer (for reading/compute)
    #[must_use]
    pub fn front(&self) -> &[T] {
        &self.front
    }

    /// Get mutable reference to back buffer (for writing/loading)
    pub fn back_mut(&mut self) -> &mut [T] {
        &mut self.back
    }

    /// Swap front and back buffers
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.front, &mut self.back);
    }
}

/// Chunked token processor for improved cache utilization (M24 - IMP-056)
///
/// Processes tokens in configurable chunks to improve memory locality
/// and cache efficiency during batch processing.
#[derive(Debug, Clone)]
pub struct ChunkedProcessor {
    chunk_size: usize,
}

impl ChunkedProcessor {
    /// Create a new chunked processor with given chunk size
    #[must_use]
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }

    /// Get the chunk size
    #[must_use]
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Calculate number of chunks needed for given input length
    #[must_use]
    pub fn num_chunks(&self, total_len: usize) -> usize {
        if total_len == 0 {
            return 0;
        }
        total_len.div_ceil(self.chunk_size)
    }

    /// Get bounds (start, end) for a specific chunk index
    #[must_use]
    pub fn chunk_bounds(&self, chunk_idx: usize, total_len: usize) -> (usize, usize) {
        let start = chunk_idx * self.chunk_size;
        let end = (start + self.chunk_size).min(total_len);
        (start, end)
    }

    /// Process data in chunks, accumulating results
    pub fn process_chunks<T, F>(&self, data: &[T], mut process: F) -> f32
    where
        F: FnMut(&[T]) -> f32,
    {
        let mut total = 0.0f32;
        let num_chunks = self.num_chunks(data.len());

        for chunk_idx in 0..num_chunks {
            let (start, end) = self.chunk_bounds(chunk_idx, data.len());
            total += process(&data[start..end]);
        }

        total
    }
}

/// Inference pipeline stages (M24 - IMP-057)
///
/// Represents the different stages of transformer inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GpuPipelineStage {
    /// Token embedding lookup
    Embed = 0,
    /// Self-attention computation
    Attention = 1,
    /// Feed-forward network
    FFN = 2,
    /// Output projection and sampling
    Output = 3,
}

/// Inference pipeline coordinator (M24 - IMP-057)
///
/// Manages multi-stage inference pipeline with timing tracking.
#[derive(Debug)]
pub struct InferencePipeline {
    num_stages: usize,
    stage_times: std::collections::HashMap<GpuPipelineStage, f32>,
}

impl InferencePipeline {
    /// Create a new pipeline with given number of stages
    #[must_use]
    pub fn new(num_stages: usize) -> Self {
        Self {
            num_stages,
            stage_times: std::collections::HashMap::new(),
        }
    }

    /// Get number of stages in the pipeline
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    /// Record timing for a pipeline stage (in milliseconds)
    pub fn record_stage_time(&mut self, stage: GpuPipelineStage, time_ms: f32) {
        self.stage_times.insert(stage, time_ms);
    }

    /// Get total pipeline latency (sum of all stage times)
    #[must_use]
    pub fn total_latency(&self) -> f32 {
        self.stage_times.values().sum()
    }

    /// Get breakdown of stage timings
    #[must_use]
    pub fn stage_breakdown(&self) -> &std::collections::HashMap<GpuPipelineStage, f32> {
        &self.stage_times
    }

    /// Reset pipeline for new forward pass
    pub fn reset(&mut self) {
        self.stage_times.clear();
    }
}
// M29: Error Recovery & Graceful Degradation (Phase 20)
// ============================================================================

/// Error classification for recovery decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClassification {
    /// Transient error - may succeed on retry
    Transient,
    /// Fatal error - should not retry
    Fatal,
    /// GPU-specific error - may fallback to CPU
    GpuFailure,
}

/// Recovery action to take
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Retry the operation with a delay
    Retry {
        /// Delay before retry
        delay: Duration,
    },
    /// Fallback to CPU inference
    FallbackToCpu,
    /// Give up and propagate error
    Fail,
}

/// Error recovery strategy with exponential backoff
pub struct ErrorRecoveryStrategy {
    max_retries: u32,
    base_delay: Duration,
    max_delay: Duration,
    jitter: f64,
}
