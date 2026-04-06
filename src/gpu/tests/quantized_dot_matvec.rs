
#[test]
fn test_quantized_dot_q8_zeros() {
    let block = vec![0u8; 34];
    let result = quantized_dot_q8(&block, &block);
    assert_eq!(result, 0.0);
}

// ============================================================================
// Quantized MatVec Tests
// ============================================================================

#[test]
fn test_quantized_matvec_q4_basic() {
    // 2 rows, 32 cols (1 block per row)
    let rows = 2;
    let cols = 32;
    let block_size = 18;

    // Create weights with some pattern
    let mut weights = vec![0u8; rows * block_size];
    // Set scales to 1.0 (f16)
    for row in 0..rows {
        weights[row * block_size] = 0x00;
        weights[row * block_size + 1] = 0x3c;
    }

    let input = vec![1.0f32; cols];

    let result = quantized_matvec_q4(&weights, &input, rows, cols);

    assert_eq!(result.len(), rows);
}

#[test]
fn test_quantized_matvec_q4_empty() {
    let result = quantized_matvec_q4(&[], &[], 0, 0);
    assert!(result.is_empty());
}

#[test]
fn test_quantized_matvec_q8_basic() {
    // 2 rows, 32 cols (1 block per row)
    let rows = 2;
    let cols = 32;
    let block_size = 34;

    let mut weights = vec![0u8; rows * block_size];
    // Set scales to 1.0 (f16)
    for row in 0..rows {
        weights[row * block_size] = 0x00;
        weights[row * block_size + 1] = 0x3c;
    }

    let input = vec![1.0f32; cols];

    let result = quantized_matvec_q8(&weights, &input, rows, cols);

    assert_eq!(result.len(), rows);
}

#[test]
fn test_quantized_matvec_q8_empty() {
    let result = quantized_matvec_q8(&[], &[], 0, 0);
    assert!(result.is_empty());
}

// ============================================================================
// QuantizedAccumulator Tests
// ============================================================================

#[test]
fn test_quantized_accumulator_new() {
    let acc = QuantizedAccumulator::new();
    assert_eq!(acc.sum(), 0.0);
}

#[test]
fn test_quantized_accumulator_default() {
    let acc = QuantizedAccumulator::default();
    assert_eq!(acc.sum(), 0.0);
}

#[test]
fn test_quantized_accumulator_add_scaled() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_scaled(2.0, 3.0);
    assert!((acc.sum() - 6.0).abs() < 1e-6);

    acc.add_scaled(1.0, 4.0);
    assert!((acc.sum() - 10.0).abs() < 1e-6);
}

#[test]
fn test_quantized_accumulator_add_block() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_block(5.0, 2.0);
    assert!((acc.sum() - 10.0).abs() < 1e-6);
}

#[test]
fn test_quantized_accumulator_reset() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_scaled(10.0, 5.0);
    assert!(acc.sum() > 0.0);

    acc.reset();
    assert_eq!(acc.sum(), 0.0);
}

#[test]
fn test_quantized_accumulator_clone() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_scaled(3.0, 4.0);

    let cloned = acc.clone();
    assert_eq!(acc.sum(), cloned.sum());
}

// ============================================================================
// DoubleBuffer Tests
// ============================================================================

#[test]
fn test_double_buffer_new() {
    let buf: DoubleBuffer<f32> = DoubleBuffer::new(100);
    assert_eq!(buf.capacity(), 100);
    assert_eq!(buf.front().len(), 100);
}

#[test]
fn test_double_buffer_front() {
    let buf: DoubleBuffer<f32> = DoubleBuffer::new(10);
    let front = buf.front();
    assert_eq!(front.len(), 10);
    assert!(front.iter().all(|&x| x == 0.0));
}

#[test]
fn test_double_buffer_back_mut() {
    let mut buf: DoubleBuffer<f32> = DoubleBuffer::new(5);
    {
        let back = buf.back_mut();
        back[0] = 1.0;
        back[1] = 2.0;
    }

    // Back values should be set
    // After swap, they should appear in front
    buf.swap();

    let front = buf.front();
    assert!((front[0] - 1.0).abs() < 1e-6);
    assert!((front[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_double_buffer_swap() {
    let mut buf: DoubleBuffer<i32> = DoubleBuffer::new(3);

    // Set front and back differently
    buf.back_mut().fill(1);
    buf.swap();

    // Now front should have 1s
    assert!(buf.front().iter().all(|&x| x == 1));

    // Set new back
    buf.back_mut().fill(2);
    buf.swap();

    assert!(buf.front().iter().all(|&x| x == 2));
}

#[test]
fn test_double_buffer_capacity() {
    let buf: DoubleBuffer<u8> = DoubleBuffer::new(256);
    assert_eq!(buf.capacity(), 256);
}

// ============================================================================
// ChunkedProcessor Tests
// ============================================================================

#[test]
fn test_chunked_processor_new() {
    let processor = ChunkedProcessor::new(64);
    assert_eq!(processor.chunk_size(), 64);
}

#[test]
fn test_chunked_processor_num_chunks() {
    let processor = ChunkedProcessor::new(10);

    assert_eq!(processor.num_chunks(0), 0);
    assert_eq!(processor.num_chunks(5), 1);
    assert_eq!(processor.num_chunks(10), 1);
    assert_eq!(processor.num_chunks(11), 2);
    assert_eq!(processor.num_chunks(25), 3);
}

#[test]
fn test_chunked_processor_chunk_bounds() {
    let processor = ChunkedProcessor::new(10);

    // Total length 25, chunk 0
    let (start, end) = processor.chunk_bounds(0, 25);
    assert_eq!(start, 0);
    assert_eq!(end, 10);

    // Chunk 1
    let (start, end) = processor.chunk_bounds(1, 25);
    assert_eq!(start, 10);
    assert_eq!(end, 20);

    // Chunk 2 (partial)
    let (start, end) = processor.chunk_bounds(2, 25);
    assert_eq!(start, 20);
    assert_eq!(end, 25);
}

#[test]
fn test_chunked_processor_process_chunks() {
    let processor = ChunkedProcessor::new(3);
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    // Sum all chunks
    let total = processor.process_chunks(&data, |chunk| chunk.iter().sum());

    // 1+2+3 + 4+5+6 + 7 = 28
    assert!((total - 28.0).abs() < 1e-6);
}

#[test]
fn test_chunked_processor_process_empty() {
    let processor = ChunkedProcessor::new(10);
    let data: Vec<f32> = vec![];

    let total = processor.process_chunks(&data, |chunk| chunk.iter().sum());
    assert_eq!(total, 0.0);
}

// ============================================================================
// InferencePipeline Tests
// ============================================================================

#[test]
fn test_inference_pipeline_new() {
    let pipeline = InferencePipeline::new(4);
    assert_eq!(pipeline.num_stages(), 4);
    assert_eq!(pipeline.total_latency(), 0.0);
}

#[test]
fn test_inference_pipeline_record_stage_time() {
    let mut pipeline = InferencePipeline::new(4);

    pipeline.record_stage_time(GpuPipelineStage::Embed, 1.0);
    pipeline.record_stage_time(GpuPipelineStage::Attention, 5.0);
    pipeline.record_stage_time(GpuPipelineStage::FFN, 3.0);
    pipeline.record_stage_time(GpuPipelineStage::Output, 2.0);

    assert!((pipeline.total_latency() - 11.0).abs() < 1e-6);
}

#[test]
fn test_inference_pipeline_stage_breakdown() {
    let mut pipeline = InferencePipeline::new(4);

    pipeline.record_stage_time(GpuPipelineStage::Embed, 1.5);
    pipeline.record_stage_time(GpuPipelineStage::Attention, 4.5);

    let breakdown = pipeline.stage_breakdown();

    assert!(breakdown.contains_key(&GpuPipelineStage::Embed));
    assert!(breakdown.contains_key(&GpuPipelineStage::Attention));

    let embed_time = breakdown.get(&GpuPipelineStage::Embed).expect("test value should be present");
    assert!((*embed_time - 1.5).abs() < 1e-6);
}

#[test]
fn test_inference_pipeline_reset() {
    let mut pipeline = InferencePipeline::new(2);

    pipeline.record_stage_time(GpuPipelineStage::Embed, 5.0);
    assert!(pipeline.total_latency() > 0.0);

    pipeline.reset();

    assert_eq!(pipeline.total_latency(), 0.0);
    assert!(pipeline.stage_breakdown().is_empty());
}

#[test]
fn test_gpu_pipeline_stage_values() {
    // Test all stage variants
    assert_eq!(GpuPipelineStage::Embed as u8, 0);
    assert_eq!(GpuPipelineStage::Attention as u8, 1);
    assert_eq!(GpuPipelineStage::FFN as u8, 2);
    assert_eq!(GpuPipelineStage::Output as u8, 3);
}

// ============================================================================
// ErrorRecoveryStrategy Tests
// ============================================================================

#[test]
fn test_error_recovery_strategy_new() {
    let strategy = ErrorRecoveryStrategy::new();
    assert_eq!(strategy.max_retries(), 3);
}

#[test]
fn test_error_recovery_strategy_default() {
    let strategy = ErrorRecoveryStrategy::default();
    assert_eq!(strategy.max_retries(), 3);
}

#[test]
fn test_error_recovery_strategy_with_max_retries() {
    let strategy = ErrorRecoveryStrategy::new().with_max_retries(5);
    assert_eq!(strategy.max_retries(), 5);
}

#[test]
fn test_error_recovery_strategy_with_base_delay() {
    let strategy = ErrorRecoveryStrategy::new().with_base_delay(Duration::from_millis(200));

    let delay = strategy.calculate_delay(0);
    assert!(delay.as_millis() >= 200);
}

#[test]
fn test_error_recovery_strategy_with_max_delay() {
    let strategy = ErrorRecoveryStrategy::new()
        .with_base_delay(Duration::from_secs(1))
        .with_max_delay(Duration::from_secs(2));

    // After many retries, delay should be capped
    let delay = strategy.calculate_delay(10);
    assert!(delay.as_secs() <= 2);
}

#[test]
fn test_error_recovery_strategy_with_jitter() {
    let strategy = ErrorRecoveryStrategy::new().with_jitter(0.5);
    // Just verify it doesn't panic
    let _ = strategy.calculate_delay(1);
}

#[test]
fn test_error_recovery_strategy_classify_transient() {
    let strategy = ErrorRecoveryStrategy::new();

    let error = Error::new(ErrorKind::TimedOut, "timeout");
    assert_eq!(
        strategy.classify_error(&error),
        ErrorClassification::Transient
    );

    let error = Error::new(ErrorKind::ConnectionReset, "reset");
    assert_eq!(
        strategy.classify_error(&error),
        ErrorClassification::Transient
    );

    let error = Error::new(ErrorKind::Interrupted, "interrupted");
    assert_eq!(
        strategy.classify_error(&error),
        ErrorClassification::Transient
    );
}

#[test]
fn test_error_recovery_strategy_classify_fatal() {
    let strategy = ErrorRecoveryStrategy::new();

    let error = Error::new(ErrorKind::NotFound, "not found");
    assert_eq!(strategy.classify_error(&error), ErrorClassification::Fatal);

    let error = Error::new(ErrorKind::PermissionDenied, "denied");
    assert_eq!(strategy.classify_error(&error), ErrorClassification::Fatal);
}

#[test]
fn test_error_recovery_strategy_classify_gpu_failure() {
    let strategy = ErrorRecoveryStrategy::new();

    let error = Error::other("GPU memory exhausted");
    assert_eq!(
        strategy.classify_error(&error),
        ErrorClassification::GpuFailure
    );

    let error = Error::other("CUDA error");
    assert_eq!(
        strategy.classify_error(&error),
        ErrorClassification::GpuFailure
    );

    let error = Error::other("wgpu device lost");
    assert_eq!(
        strategy.classify_error(&error),
        ErrorClassification::GpuFailure
    );
}

#[test]
fn test_error_recovery_strategy_determine_action_retry() {
    let strategy = ErrorRecoveryStrategy::new();

    let error = Error::new(ErrorKind::TimedOut, "timeout");
    let action = strategy.determine_action(&error, 0);

    assert!(matches!(action, RecoveryAction::Retry { .. }));
}

#[test]
fn test_error_recovery_strategy_determine_action_fail() {
    let strategy = ErrorRecoveryStrategy::new().with_max_retries(3);

    let error = Error::new(ErrorKind::TimedOut, "timeout");
    let action = strategy.determine_action(&error, 3); // At max

    assert!(matches!(action, RecoveryAction::Fail));
}

#[test]
fn test_error_recovery_strategy_determine_action_fallback() {
    let strategy = ErrorRecoveryStrategy::new();

    let error = Error::other("GPU error");
    let action = strategy.determine_action(&error, 0);

    assert!(matches!(action, RecoveryAction::FallbackToCpu));
}

#[test]
fn test_error_recovery_strategy_determine_action_with_fallback() {
    let strategy = ErrorRecoveryStrategy::new();

    let error = Error::other("GPU unavailable");
    let action = strategy.determine_action_with_fallback(&error, 0);

    assert!(matches!(action, RecoveryAction::FallbackToCpu));
}
