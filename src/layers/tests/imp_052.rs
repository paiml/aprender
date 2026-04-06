
// ============================================================================
// Phase 14: Quantized Compute Kernels (M23) - EXTREME TDD
// ============================================================================

/// IMP-052: Quantized dot product
/// Target: Compute dot product on Q4/Q8 data without full dequantization
#[test]
#[cfg(feature = "gpu")]
#[allow(clippy::similar_names)] // scale_a_f16/scale_b_f16, block_a_q8/block_b_q8 are intentionally paired
fn test_imp_052_quantized_dot() {
    use crate::gpu::{quantized_dot_q4, quantized_dot_q8};

    // Q4_0 format: 32 values per block, 2 values per byte + f16 scale
    // Block size = 2 (scale) + 16 (data) = 18 bytes

    // Test 1: Q4 dot product - create test blocks
    // Each block has scale (f16 as 2 bytes) + 16 bytes of packed 4-bit values
    let scale_a: f32 = 0.5;
    let scale_b: f32 = 0.25;

    // Create Q4 blocks: [scale_lo, scale_hi, packed_data...]
    let mut block_a = vec![0u8; 18];
    let mut block_b = vec![0u8; 18];

    // Set scales (f16 little-endian)
    let scale_a_f16 = half::f16::from_f32(scale_a);
    let scale_b_f16 = half::f16::from_f32(scale_b);
    block_a[0..2].copy_from_slice(&scale_a_f16.to_le_bytes());
    block_b[0..2].copy_from_slice(&scale_b_f16.to_le_bytes());

    // Set packed values: each byte has two 4-bit values (low nibble, high nibble)
    // Values are stored as unsigned 0-15, centered at 8
    // Use simple test pattern: all 8s (which is 0 after centering)
    for i in 2..18 {
        block_a[i] = 0x99; // Two 9s: (9-8)*scale = scale per element
        block_b[i] = 0x99;
    }

    let result_q4 = quantized_dot_q4(&block_a, &block_b);

    // Expected: 32 elements, each (1*scale_a) * (1*scale_b) = 0.5 * 0.25 = 0.125
    // Sum = 32 * 0.125 = 4.0
    assert!(
        (result_q4 - 4.0).abs() < 0.5,
        "IMP-052: Q4 dot product result ({}) should be ~4.0",
        result_q4
    );

    // Test 2: Q8 dot product
    // Q8_0 format: 32 values per block, 1 byte per value + f16 scale
    // Block size = 2 (scale) + 32 (data) = 34 bytes
    let mut block_a_q8 = vec![0u8; 34];
    let mut block_b_q8 = vec![0u8; 34];

    block_a_q8[0..2].copy_from_slice(&scale_a_f16.to_le_bytes());
    block_b_q8[0..2].copy_from_slice(&scale_b_f16.to_le_bytes());

    // Q8 values are signed i8, use value 1 for simplicity
    for i in 2..34 {
        block_a_q8[i] = 1i8 as u8;
        block_b_q8[i] = 1i8 as u8;
    }

    let result_q8 = quantized_dot_q8(&block_a_q8, &block_b_q8);

    // Expected: 32 elements, each (1*scale_a) * (1*scale_b) = 0.5 * 0.25 = 0.125
    // Sum = 32 * 0.125 = 4.0
    assert!(
        (result_q8 - 4.0).abs() < 0.5,
        "IMP-052: Q8 dot product result ({}) should be ~4.0",
        result_q8
    );

    // Test 3: Zero blocks should give zero result
    let zero_block_q4 = vec![0u8; 18];
    let zero_result = quantized_dot_q4(&zero_block_q4, &zero_block_q4);
    assert!(
        zero_result.abs() < 1e-6,
        "IMP-052: Zero blocks should give zero dot product"
    );
}

/// IMP-053: Quantized matrix-vector multiply
/// Target: MatVec on quantized weights without full dequantization
#[test]
#[cfg(feature = "gpu")]
fn test_imp_053_quantized_matvec() {
    use crate::gpu::{quantized_matvec_q4, quantized_matvec_q8};

    // Test matrix: 2 rows x 32 cols (1 block per row)
    let rows = 2;
    let cols = 32;

    // Create Q4 weight matrix (2 blocks, 18 bytes each)
    let scale: f32 = 0.1;
    let scale_f16 = half::f16::from_f32(scale);

    let mut weights_q4 = vec![0u8; rows * 18];
    for row in 0..rows {
        let offset = row * 18;
        weights_q4[offset..offset + 2].copy_from_slice(&scale_f16.to_le_bytes());
        // Fill with 9s (value 1 after centering at 8)
        for i in 2..18 {
            weights_q4[offset + i] = 0x99;
        }
    }

    // Input vector: 32 f32 values, all 1.0
    let input: Vec<f32> = vec![1.0; cols];

    let result_q4 = quantized_matvec_q4(&weights_q4, &input, rows, cols);

    assert_eq!(
        result_q4.len(),
        rows,
        "IMP-053: Q4 matvec should produce {} outputs",
        rows
    );

    // Each row: sum of 32 * (1 * scale) * 1.0 = 32 * 0.1 = 3.2
    for (i, &val) in result_q4.iter().enumerate() {
        assert!(
            (val - 3.2).abs() < 0.5,
            "IMP-053: Q4 matvec row {} ({}) should be ~3.2",
            i,
            val
        );
    }

    // Test Q8 matvec
    let mut weights_q8 = vec![0u8; rows * 34];
    for row in 0..rows {
        let offset = row * 34;
        weights_q8[offset..offset + 2].copy_from_slice(&scale_f16.to_le_bytes());
        // Fill with 1s (signed i8)
        for i in 2..34 {
            weights_q8[offset + i] = 1i8 as u8;
        }
    }

    let result_q8 = quantized_matvec_q8(&weights_q8, &input, rows, cols);

    assert_eq!(
        result_q8.len(),
        rows,
        "IMP-053: Q8 matvec should produce {} outputs",
        rows
    );

    for (i, &val) in result_q8.iter().enumerate() {
        assert!(
            (val - 3.2).abs() < 0.5,
            "IMP-053: Q8 matvec row {} ({}) should be ~3.2",
            i,
            val
        );
    }
}

/// IMP-054: Mixed precision accumulation
/// Target: Accumulate in f32 while reading quantized data
#[test]
#[cfg(feature = "gpu")]
fn test_imp_054_mixed_precision() {
    use crate::gpu::QuantizedAccumulator;

    // Test 1: Create accumulator
    let mut acc = QuantizedAccumulator::new();
    assert_eq!(
        acc.sum(),
        0.0,
        "IMP-054: New accumulator should have zero sum"
    );

    // Test 2: Add scaled values
    acc.add_scaled(1.0, 0.5); // 1.0 * 0.5 = 0.5
    acc.add_scaled(2.0, 0.5); // 2.0 * 0.5 = 1.0
    acc.add_scaled(3.0, 0.5); // 3.0 * 0.5 = 1.5

    assert!(
        (acc.sum() - 3.0).abs() < 1e-6,
        "IMP-054: Accumulator sum ({}) should be 3.0",
        acc.sum()
    );

    // Test 3: Reset accumulator
    acc.reset();
    assert_eq!(
        acc.sum(),
        0.0,
        "IMP-054: Reset accumulator should have zero sum"
    );

    // Test 4: Add block contribution (simulates quantized block processing)
    let block_sum: f32 = 10.0;
    let block_scale: f32 = 0.1;
    acc.add_block(block_sum, block_scale);

    assert!(
        (acc.sum() - 1.0).abs() < 1e-6,
        "IMP-054: Block contribution ({}) should be 1.0",
        acc.sum()
    );

    // Test 5: Multiple block accumulation
    acc.reset();
    for _ in 0..10 {
        acc.add_block(5.0, 0.2); // 5.0 * 0.2 = 1.0 per block
    }

    assert!(
        (acc.sum() - 10.0).abs() < 1e-5,
        "IMP-054: 10 blocks should sum to 10.0, got {}",
        acc.sum()
    );
}

/// IMP-055: Double-buffered weight loading
/// Target: Load next layer weights while computing current layer
#[test]
#[cfg(feature = "gpu")]
fn test_imp_055_double_buffer() {
    use crate::gpu::DoubleBuffer;

    // Test 1: Create double buffer with given capacity
    let buffer: DoubleBuffer<f32> = DoubleBuffer::new(1024);
    assert_eq!(
        buffer.capacity(),
        1024,
        "IMP-055: Double buffer should have requested capacity"
    );

    // Test 2: Access front buffer for reading
    let front = buffer.front();
    assert_eq!(
        front.len(),
        1024,
        "IMP-055: Front buffer should have full capacity"
    );

    // Test 3: Access back buffer for writing
    let mut buffer = DoubleBuffer::new(256);
    {
        let back = buffer.back_mut();
        for (i, val) in back.iter_mut().enumerate() {
            *val = i as f32;
        }
    }

    // Test 4: Swap buffers - back becomes front
    buffer.swap();
    let front_after_swap = buffer.front();
    assert!(
        (front_after_swap[0] - 0.0).abs() < 1e-6,
        "IMP-055: After swap, front[0] should be 0.0"
    );
    assert!(
        (front_after_swap[255] - 255.0).abs() < 1e-6,
        "IMP-055: After swap, front[255] should be 255.0"
    );

    // Test 5: Multiple swaps maintain data integrity
    {
        let back = buffer.back_mut();
        for val in back.iter_mut() {
            *val = 42.0;
        }
    }
    buffer.swap();
    let front_again = buffer.front();
    assert!(
        (front_again[0] - 42.0).abs() < 1e-6,
        "IMP-055: After second swap, front should have 42.0 values"
    );
}

/// IMP-056: Chunked token processing
/// Target: Process tokens in chunks to improve cache utilization
#[test]
#[cfg(feature = "gpu")]
fn test_imp_056_chunked_processing() {
    use crate::gpu::ChunkedProcessor;

    // Test 1: Create processor with chunk size
    let processor = ChunkedProcessor::new(64);
    assert_eq!(
        processor.chunk_size(),
        64,
        "IMP-056: Processor should have requested chunk size"
    );

    // Test 2: Calculate number of chunks for input
    assert_eq!(
        processor.num_chunks(100),
        2,
        "IMP-056: 100 items with chunk_size=64 needs 2 chunks"
    );
    assert_eq!(
        processor.num_chunks(64),
        1,
        "IMP-056: 64 items with chunk_size=64 needs 1 chunk"
    );
    assert_eq!(
        processor.num_chunks(0),
        0,
        "IMP-056: 0 items needs 0 chunks"
    );

    // Test 3: Get chunk bounds
    let (start, end) = processor.chunk_bounds(0, 100);
    assert_eq!(start, 0, "IMP-056: First chunk starts at 0");
    assert_eq!(end, 64, "IMP-056: First chunk ends at chunk_size");

    let (start, end) = processor.chunk_bounds(1, 100);
    assert_eq!(start, 64, "IMP-056: Second chunk starts at 64");
    assert_eq!(end, 100, "IMP-056: Second chunk ends at total length");

    // Test 4: Process chunks with accumulator function
    let data: Vec<f32> = (0..128).map(|x| x as f32).collect();
    let sum = processor.process_chunks(&data, |chunk| chunk.iter().sum::<f32>());

    // Sum of 0..127 = 127 * 128 / 2 = 8128
    assert!(
        (sum - 8128.0).abs() < 1e-3,
        "IMP-056: Chunked sum ({}) should equal 8128.0",
        sum
    );

    // Test 5: Small input (single chunk)
    let small_data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let small_sum = processor.process_chunks(&small_data, |chunk| chunk.iter().sum::<f32>());
    assert!(
        (small_sum - 6.0).abs() < 1e-6,
        "IMP-056: Small chunked sum ({}) should equal 6.0",
        small_sum
    );
}

/// IMP-057: Pipeline stage management
/// Target: Coordinate multi-stage inference pipeline
#[test]
#[cfg(feature = "gpu")]
fn test_imp_057_pipeline_stages() {
    use crate::gpu::{GpuPipelineStage, InferencePipeline};

    // Test 1: Create pipeline stages enum
    let embed = GpuPipelineStage::Embed;
    let attention = GpuPipelineStage::Attention;
    let ffn = GpuPipelineStage::FFN;
    let output = GpuPipelineStage::Output;

    // Test 2: Pipeline stage ordering
    assert!(
        (embed as u8) < (attention as u8),
        "IMP-057: Embed should come before Attention"
    );
    assert!(
        (attention as u8) < (ffn as u8),
        "IMP-057: Attention should come before FFN"
    );
    assert!(
        (ffn as u8) < (output as u8),
        "IMP-057: FFN should come before Output"
    );

    // Test 3: Create inference pipeline
    let mut pipeline = InferencePipeline::new(4); // 4-stage pipeline
    assert_eq!(
        pipeline.num_stages(),
        4,
        "IMP-057: Pipeline should have 4 stages"
    );

    // Test 4: Record stage timing
    pipeline.record_stage_time(GpuPipelineStage::Embed, 1.0);
    pipeline.record_stage_time(GpuPipelineStage::Attention, 5.0);
    pipeline.record_stage_time(GpuPipelineStage::FFN, 3.0);
    pipeline.record_stage_time(GpuPipelineStage::Output, 0.5);

    // Test 5: Get total pipeline latency
    let total = pipeline.total_latency();
    assert!(
        (total - 9.5).abs() < 1e-6,
        "IMP-057: Total latency ({}) should be 9.5ms",
        total
    );

    // Test 6: Get stage breakdown
    let breakdown = pipeline.stage_breakdown();
    assert!(
        (breakdown[&GpuPipelineStage::Attention] - 5.0).abs() < 1e-6,
        "IMP-057: Attention stage should be 5.0ms"
    );

    // Test 7: Reset pipeline for new forward pass
    pipeline.reset();
    assert!(
        pipeline.total_latency() < 1e-6,
        "IMP-057: Reset pipeline should have zero latency"
    );
}
