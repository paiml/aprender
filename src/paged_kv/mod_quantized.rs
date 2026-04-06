
impl QuantizedKvData {
    /// Create new quantized KV data with given precision
    pub fn new(
        quant_type: KvQuantType,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        let total_size = block_size * num_heads * head_dim;
        let num_quant_blocks = total_size.div_ceil(KV_QUANT_BLOCK_SIZE);

        match quant_type {
            KvQuantType::FP32 => Self::FP32 {
                keys: vec![0.0; total_size],
                values: vec![0.0; total_size],
            },
            KvQuantType::Q8 => Self::Q8 {
                key_blocks: vec![Q8KvBlock::new(); num_quant_blocks],
                value_blocks: vec![Q8KvBlock::new(); num_quant_blocks],
            },
            KvQuantType::Q4 => Self::Q4 {
                key_blocks: vec![Q4KvBlock::new(); num_quant_blocks],
                value_blocks: vec![Q4KvBlock::new(); num_quant_blocks],
            },
        }
    }

    /// Get quantization type
    pub fn quant_type(&self) -> KvQuantType {
        match self {
            Self::FP32 { .. } => KvQuantType::FP32,
            Self::Q8 { .. } => KvQuantType::Q8,
            Self::Q4 { .. } => KvQuantType::Q4,
        }
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        match self {
            Self::FP32 { keys, values } => (keys.len() + values.len()) * 4,
            Self::Q8 {
                key_blocks,
                value_blocks,
            } => {
                // Scale (4 bytes) + quants (32 bytes) = 36 bytes per block
                (key_blocks.len() + value_blocks.len()) * (4 + KV_QUANT_BLOCK_SIZE)
            },
            Self::Q4 {
                key_blocks,
                value_blocks,
            } => {
                // Scale (4 bytes) + quants (16 bytes) = 20 bytes per block
                (key_blocks.len() + value_blocks.len()) * (4 + KV_QUANT_BLOCK_SIZE / 2)
            },
        }
    }

    /// Write keys at given offset
    pub fn write_keys(&mut self, offset: usize, data: &[f32]) {
        match self {
            Self::FP32 { keys, .. } => {
                let end = (offset + data.len()).min(keys.len());
                keys[offset..end].copy_from_slice(&data[..end - offset]);
            },
            Self::Q8 { key_blocks, .. } => {
                write_quantized_q8(key_blocks, offset, data);
            },
            Self::Q4 { key_blocks, .. } => {
                write_quantized_q4(key_blocks, offset, data);
            },
        }
    }

    /// Write values at given offset
    pub fn write_values(&mut self, offset: usize, data: &[f32]) {
        match self {
            Self::FP32 { values, .. } => {
                let end = (offset + data.len()).min(values.len());
                values[offset..end].copy_from_slice(&data[..end - offset]);
            },
            Self::Q8 { value_blocks, .. } => {
                write_quantized_q8(value_blocks, offset, data);
            },
            Self::Q4 { value_blocks, .. } => {
                write_quantized_q4(value_blocks, offset, data);
            },
        }
    }

    /// Read keys at given offset
    pub fn read_keys(&self, offset: usize, length: usize) -> Vec<f32> {
        match self {
            Self::FP32 { keys, .. } => {
                let end = (offset + length).min(keys.len());
                keys[offset..end].to_vec()
            },
            Self::Q8 { key_blocks, .. } => read_quantized_q8(key_blocks, offset, length),
            Self::Q4 { key_blocks, .. } => read_quantized_q4(key_blocks, offset, length),
        }
    }

    /// Read values at given offset
    pub fn read_values(&self, offset: usize, length: usize) -> Vec<f32> {
        match self {
            Self::FP32 { values, .. } => {
                let end = (offset + length).min(values.len());
                values[offset..end].to_vec()
            },
            Self::Q8 { value_blocks, .. } => read_quantized_q8(value_blocks, offset, length),
            Self::Q4 { value_blocks, .. } => read_quantized_q4(value_blocks, offset, length),
        }
    }
}

// Helper: Write to Q8 blocks
fn write_quantized_q8(blocks: &mut [Q8KvBlock], offset: usize, data: &[f32]) {
    let start_block = offset / KV_QUANT_BLOCK_SIZE;
    let start_offset = offset % KV_QUANT_BLOCK_SIZE;

    let mut data_idx = 0;
    let mut block_idx = start_block;
    let mut in_block_offset = start_offset;

    while data_idx < data.len() && block_idx < blocks.len() {
        // Read existing block, modify, re-quantize
        let mut values = blocks[block_idx].dequantize();

        while in_block_offset < KV_QUANT_BLOCK_SIZE && data_idx < data.len() {
            values[in_block_offset] = data[data_idx];
            in_block_offset += 1;
            data_idx += 1;
        }

        blocks[block_idx] = Q8KvBlock::quantize(&values);
        block_idx += 1;
        in_block_offset = 0;
    }
}

// Helper: Write to Q4 blocks
fn write_quantized_q4(blocks: &mut [Q4KvBlock], offset: usize, data: &[f32]) {
    let start_block = offset / KV_QUANT_BLOCK_SIZE;
    let start_offset = offset % KV_QUANT_BLOCK_SIZE;

    let mut data_idx = 0;
    let mut block_idx = start_block;
    let mut in_block_offset = start_offset;

    while data_idx < data.len() && block_idx < blocks.len() {
        let mut values = blocks[block_idx].dequantize();

        while in_block_offset < KV_QUANT_BLOCK_SIZE && data_idx < data.len() {
            values[in_block_offset] = data[data_idx];
            in_block_offset += 1;
            data_idx += 1;
        }

        blocks[block_idx] = Q4KvBlock::quantize(&values);
        block_idx += 1;
        in_block_offset = 0;
    }
}

// Helper: Read from Q8 blocks
fn read_quantized_q8(blocks: &[Q8KvBlock], offset: usize, length: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(length);
    let start_block = offset / KV_QUANT_BLOCK_SIZE;
    let start_offset = offset % KV_QUANT_BLOCK_SIZE;

    let mut block_idx = start_block;
    let mut in_block_offset = start_offset;
    let mut remaining = length;

    while remaining > 0 && block_idx < blocks.len() {
        let values = blocks[block_idx].dequantize();

        while in_block_offset < KV_QUANT_BLOCK_SIZE && remaining > 0 {
            result.push(values[in_block_offset]);
            in_block_offset += 1;
            remaining -= 1;
        }

        block_idx += 1;
        in_block_offset = 0;
    }

    result
}

// Helper: Read from Q4 blocks
fn read_quantized_q4(blocks: &[Q4KvBlock], offset: usize, length: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(length);
    let start_block = offset / KV_QUANT_BLOCK_SIZE;
    let start_offset = offset % KV_QUANT_BLOCK_SIZE;

    let mut block_idx = start_block;
    let mut in_block_offset = start_offset;
    let mut remaining = length;

    while remaining > 0 && block_idx < blocks.len() {
        let values = blocks[block_idx].dequantize();

        while in_block_offset < KV_QUANT_BLOCK_SIZE && remaining > 0 {
            result.push(values[in_block_offset]);
            in_block_offset += 1;
            remaining -= 1;
        }

        block_idx += 1;
        in_block_offset = 0;
    }

    result
}

/// Quantized KV page for memory-efficient cache
#[derive(Debug, Clone)]
pub struct QuantizedKvPage {
    /// Page identifier
    pub id: PageId,
    /// Quantized KV data
    pub data: QuantizedKvData,
    /// Number of tokens currently stored
    pub num_tokens: usize,
    /// Reference count for COW
    pub ref_count: usize,
    /// Block size (tokens per page)
    block_size: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
}

impl QuantizedKvPage {
    /// Create new quantized KV page
    pub fn new(
        id: PageId,
        quant_type: KvQuantType,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            id,
            data: QuantizedKvData::new(quant_type, block_size, num_heads, head_dim),
            num_tokens: 0,
            ref_count: 0, // Pages start in free pool with ref_count 0
            block_size,
            num_heads,
            head_dim,
        }
    }

    /// Get quantization type
    pub fn quant_type(&self) -> KvQuantType {
        self.data.quant_type()
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.data.memory_bytes()
    }

    /// Check if page is full
    pub fn is_full(&self) -> bool {
        self.num_tokens >= self.block_size
    }

    /// Check if page is shared (COW)
    pub fn is_shared(&self) -> bool {
        self.ref_count > 1
    }

    /// Remaining capacity in tokens
    pub fn remaining_capacity(&self) -> usize {
        self.block_size.saturating_sub(self.num_tokens)
    }

    /// Write keys for a token position
    pub fn write_keys(&mut self, token_pos: usize, keys: &[f32]) {
        let offset = token_pos * self.num_heads * self.head_dim;
        self.data.write_keys(offset, keys);
    }

    /// Write values for a token position
    pub fn write_values(&mut self, token_pos: usize, values: &[f32]) {
        let offset = token_pos * self.num_heads * self.head_dim;
        self.data.write_values(offset, values);
    }

    /// Read keys for a token position
    pub fn read_keys(&self, token_pos: usize) -> Vec<f32> {
        let offset = token_pos * self.num_heads * self.head_dim;
        let length = self.num_heads * self.head_dim;
        self.data.read_keys(offset, length)
    }

    /// Read values for a token position
    pub fn read_values(&self, token_pos: usize) -> Vec<f32> {
        let offset = token_pos * self.num_heads * self.head_dim;
        let length = self.num_heads * self.head_dim;
        self.data.read_values(offset, length)
    }
}

/// Quantized PagedKvCache with configurable precision
pub struct QuantizedPagedKvCache {
    /// Physical pages with quantized storage
    physical_pages: Vec<QuantizedKvPage>,
    /// Page tables (same as regular PagedKvCache)
    page_tables: HashMap<SeqId, Vec<PageId>>,
    /// Free page list
    free_pages: VecDeque<PageId>,
    /// Quantization type
    quant_type: KvQuantType,
    /// Tokens per page
    block_size: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Total pages
    total_pages: usize,
    /// Statistics
    stats: PagedCacheStats,
}
