
#[test]
fn test_double_buffer_swap_cov() {
    let mut buf: DoubleBuffer<i32> = DoubleBuffer::new(5);
    buf.back_mut()[0] = 1;
    buf.swap();
    buf.back_mut()[0] = 2;
    assert_eq!(buf.front()[0], 1);
    buf.swap();
    assert_eq!(buf.front()[0], 2);
}

// --- ChunkedProcessor tests ---
#[test]
fn test_chunked_processor_new_cov() {
    let proc = ChunkedProcessor::new(64);
    assert_eq!(proc.chunk_size(), 64);
}

#[test]
fn test_chunked_processor_num_chunks_cov() {
    let proc = ChunkedProcessor::new(10);
    assert_eq!(proc.num_chunks(0), 0);
    assert_eq!(proc.num_chunks(10), 1);
    assert_eq!(proc.num_chunks(11), 2);
    assert_eq!(proc.num_chunks(20), 2);
    assert_eq!(proc.num_chunks(21), 3);
}

#[test]
fn test_chunked_processor_bounds_cov() {
    let proc = ChunkedProcessor::new(10);
    assert_eq!(proc.chunk_bounds(0, 25), (0, 10));
    assert_eq!(proc.chunk_bounds(1, 25), (10, 20));
    assert_eq!(proc.chunk_bounds(2, 25), (20, 25));
}

#[test]
fn test_chunked_processor_process_cov() {
    let proc = ChunkedProcessor::new(5);
    let data: Vec<f32> = vec![1.0; 12];
    let result = proc.process_chunks(&data, |chunk| chunk.iter().sum());
    assert!((result - 12.0).abs() < 1e-5);
}

// --- GpuPipelineStage tests ---
#[test]
fn test_pipeline_stage_values_cov() {
    assert_eq!(GpuPipelineStage::Embed as u8, 0);
    assert_eq!(GpuPipelineStage::Attention as u8, 1);
    assert_eq!(GpuPipelineStage::FFN as u8, 2);
    assert_eq!(GpuPipelineStage::Output as u8, 3);
}

// --- InferencePipeline tests ---
#[test]
fn test_inference_pipeline_new_cov() {
    let pipe = InferencePipeline::new(4);
    assert_eq!(pipe.num_stages(), 4);
    assert_eq!(pipe.total_latency(), 0.0);
}

#[test]
fn test_inference_pipeline_record_cov() {
    let mut pipe = InferencePipeline::new(4);
    pipe.record_stage_time(GpuPipelineStage::Embed, 1.0);
    pipe.record_stage_time(GpuPipelineStage::Attention, 5.0);
    pipe.record_stage_time(GpuPipelineStage::FFN, 3.0);
    pipe.record_stage_time(GpuPipelineStage::Output, 1.0);
    assert!((pipe.total_latency() - 10.0).abs() < 1e-5);
}

#[test]
fn test_inference_pipeline_reset_cov() {
    let mut pipe = InferencePipeline::new(4);
    pipe.record_stage_time(GpuPipelineStage::Embed, 5.0);
    pipe.reset();
    assert_eq!(pipe.total_latency(), 0.0);
    assert!(pipe.stage_breakdown().is_empty());
}

// --- TokenBatch tests ---
#[test]
fn test_token_batch_new_deep2() {
    let batch = TokenBatch::new(32);
    assert_eq!(batch.capacity(), 32);
    assert_eq!(batch.len(), 0);
    assert!(batch.is_empty());
}

#[test]
fn test_token_batch_push_deep2() {
    let mut batch = TokenBatch::new(5);
    assert!(batch.push(1).is_none());
    assert!(batch.push(2).is_none());
    assert!(batch.push(3).is_none());
    assert_eq!(batch.len(), 3);
    assert!(!batch.is_full());
}

#[test]
fn test_token_batch_full_deep2() {
    let mut batch = TokenBatch::new(3);
    batch.push(1);
    batch.push(2);
    let result = batch.push(3); // Should return Some with tokens
    assert!(result.is_some());
    let tokens = result.expect("test");
    assert_eq!(tokens, vec![1, 2, 3]);
}

#[test]
fn test_token_batch_flush_deep2() {
    let mut batch = TokenBatch::new(5);
    batch.push(10);
    batch.push(20);
    batch.push(30);
    let tokens = batch.flush();
    assert_eq!(tokens, vec![10, 20, 30]);
    assert!(batch.is_empty());
}

// --- quantized_matvec_q4 deep tests ---
#[test]
fn test_quantized_matvec_q4_basic_deep2() {
    // Q4_0 block: 2 bytes scale + 16 bytes (32 4-bit values)
    let block_size = 18;
    let cols = 32; // One block
    let rows = 2;
    let weights = vec![0u8; rows * (cols / 32) * block_size];
    let input = vec![1.0f32; cols];
    let result = quantized_matvec_q4(&weights, &input, rows, cols);
    assert_eq!(result.len(), rows);
}

#[test]
fn test_quantized_matvec_q4_empty_deep2() {
    let weights: Vec<u8> = vec![];
    let input: Vec<f32> = vec![];
    let result = quantized_matvec_q4(&weights, &input, 0, 0);
    assert!(result.is_empty());
}

// --- quantized_matvec_q8 deep tests ---
#[test]
fn test_quantized_matvec_q8_basic_deep2() {
    // Q8_0 block: 2 bytes scale + 32 bytes values
    let block_size = 34;
    let cols = 32; // One block
    let rows = 2;
    let weights = vec![0u8; rows * (cols / 32) * block_size];
    let input = vec![1.0f32; cols];
    let result = quantized_matvec_q8(&weights, &input, rows, cols);
    assert_eq!(result.len(), rows);
}

// --- QuantizedAccumulator tests ---
#[test]
fn test_quantized_accumulator_new_deep2() {
    let acc = QuantizedAccumulator::new();
    assert_eq!(acc.sum(), 0.0);
}

#[test]
fn test_quantized_accumulator_add_scaled_deep2() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_scaled(2.0, 0.5); // 2.0 * 0.5 = 1.0
    acc.add_scaled(4.0, 0.5); // 4.0 * 0.5 = 2.0
    assert!((acc.sum() - 3.0).abs() < 1e-5);
}

#[test]
fn test_quantized_accumulator_add_block_deep2() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_block(10.0, 0.5); // 10.0 * 0.5 = 5.0
    acc.add_block(6.0, 0.5); // 6.0 * 0.5 = 3.0
    assert!((acc.sum() - 8.0).abs() < 1e-5);
}

#[test]
fn test_quantized_accumulator_reset_deep2() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_scaled(10.0, 1.0);
    acc.reset();
    assert_eq!(acc.sum(), 0.0);
}

// --- ContiguousAttentionBuffer tests ---
#[test]
fn test_contiguous_attention_buffer_new_deep2() {
    // Constructor: new(max_seq_len, num_heads, head_dim)
    let buf = ContiguousAttentionBuffer::new(8, 4, 64);
    assert_eq!(buf.max_seq_len(), 8);
}

#[test]
fn test_contiguous_attention_buffer_is_contiguous_deep2() {
    let buf = ContiguousAttentionBuffer::new(10, 2, 32);
    assert!(buf.is_contiguous());
}

#[test]
fn test_contiguous_attention_buffer_get_views_deep2() {
    let buf = ContiguousAttentionBuffer::new(4, 2, 16);
    let (q, k, v, o) = buf.get_views();
    assert_eq!(q.len(), 4 * 2 * 16); // max_seq_len * num_heads * head_dim
    assert_eq!(k.len(), q.len());
    assert_eq!(v.len(), q.len());
    assert_eq!(o.len(), q.len());
}

#[test]
fn test_contiguous_attention_buffer_reset_deep2() {
    let mut buf = ContiguousAttentionBuffer::new(4, 2, 16);
    {
        let (q, _, _, _) = buf.get_views_mut();
        q[0] = 42.0;
    }
    buf.reset();
    let (q, _, _, _) = buf.get_views();
    assert_eq!(q[0], 0.0);
}

// --- batch_embed tests ---
#[test]
fn test_batch_embed_empty_cov() {
    let embedding_table = vec![1.0f32; 100];
    let tokens: Vec<usize> = vec![];
    let result = batch_embed(&embedding_table, &tokens, 10);
    assert!(result.is_empty());
}

#[test]
fn test_batch_embed_multiple_tokens_cov() {
    let hidden_dim = 4;
    let vocab_size = 10;
    let mut embedding_table = vec![0.0f32; vocab_size * hidden_dim];
    // Set specific embeddings
    for i in 0..vocab_size {
        for j in 0..hidden_dim {
            embedding_table[i * hidden_dim + j] = (i * 10 + j) as f32;
        }
    }
    let tokens = vec![0, 5, 9];
    let result = batch_embed(&embedding_table, &tokens, hidden_dim);
    assert_eq!(result.len(), 3 * hidden_dim);
    // Check token 0 embedding
    assert_eq!(result[0], 0.0);
    // Check token 5 embedding
    assert_eq!(result[4], 50.0);
    // Check token 9 embedding
    assert_eq!(result[8], 90.0);
}

// --- sequential_ffn and parallel_ffn tests ---
#[test]
fn test_sequential_ffn_zero_weights_cov() {
    let hidden_dim = 4;
    let inter_dim = 8;
    let hidden = vec![1.0f32; hidden_dim];
    let w1 = vec![0.0f32; hidden_dim * inter_dim];
    let w2 = vec![0.0f32; inter_dim * hidden_dim];
    let result = sequential_ffn(&hidden, &w1, &w2, hidden_dim, inter_dim);
    assert_eq!(result.len(), hidden_dim);
    for val in &result {
        assert_eq!(*val, 0.0);
    }
}

#[test]
fn test_parallel_ffn_matches_sequential_cov() {
    let hidden_dim = 8;
    let inter_dim = 16;
    let hidden: Vec<f32> = (0..hidden_dim).map(|x| x as f32 * 0.1).collect();
    let w1: Vec<f32> = (0..hidden_dim * inter_dim)
        .map(|x| x as f32 * 0.01)
        .collect();
    let w2: Vec<f32> = (0..inter_dim * hidden_dim)
        .map(|x| x as f32 * 0.01)
        .collect();

    let seq = sequential_ffn(&hidden, &w1, &w2, hidden_dim, inter_dim);
    let par = parallel_ffn(&hidden, &w1, &w2, hidden_dim, inter_dim);

    assert_eq!(seq.len(), par.len());
    for i in 0..seq.len() {
        assert!(
            (seq[i] - par[i]).abs() < 1e-3,
            "Mismatch at {}: seq={}, par={}",
            i,
            seq[i],
            par[i]
        );
    }
}

// --- standard_layernorm and fused_layernorm deep tests ---
#[test]
fn test_standard_layernorm_basic_deep2() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0, 1.0, 1.0, 1.0];
    let beta = vec![0.0, 0.0, 0.0, 0.0];
    let result = standard_layernorm(&input, &gamma, &beta, 1e-5);
    assert_eq!(result.len(), 4);
    // Result should be normalized (mean ~0, std ~1)
    let mean: f32 = result.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-4);
}

#[test]
fn test_standard_layernorm_with_scale_shift_deep2() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![2.0, 2.0, 2.0, 2.0];
    let beta = vec![1.0, 1.0, 1.0, 1.0];
    let result = standard_layernorm(&input, &gamma, &beta, 1e-5);
    // After scaling by 2 and shifting by 1
    let mean: f32 = result.iter().sum::<f32>() / 4.0;
    assert!((mean - 1.0).abs() < 1e-4); // Mean should be close to beta
}

#[test]
fn test_fused_layernorm_matches_standard_deep2() {
    let input: Vec<f32> = (0..32).map(|x| x as f32 * 0.1).collect();
    let gamma = vec![1.0f32; 32];
    let beta = vec![0.0f32; 32];

    let standard = standard_layernorm(&input, &gamma, &beta, 1e-5);
    let fused = fused_layernorm(&input, &gamma, &beta, 1e-5);

    assert_eq!(standard.len(), fused.len());
    for i in 0..standard.len() {
        assert!(
            (standard[i] - fused[i]).abs() < 1e-4,
            "Mismatch at {}: standard={}, fused={}",
            i,
            standard[i],
            fused[i]
        );
    }
}

// --- ForwardArena tests ---
#[test]
fn test_forward_arena_new_cov() {
    let arena = ForwardArena::new(1024);
    assert_eq!(arena.capacity(), 1024);
}

#[test]
fn test_forward_arena_alloc_cov() {
    let mut arena = ForwardArena::new(1000);
    let buf1 = arena.alloc(100);
    assert_eq!(buf1.len(), 100);
    let buf2 = arena.alloc(200);
    assert_eq!(buf2.len(), 200);
}

#[test]
fn test_forward_arena_reset_cov() {
    let mut arena = ForwardArena::new(500);
    arena.alloc(100);
    arena.alloc(200);
    arena.reset();
    // After reset, should be able to allocate full capacity again
    let buf = arena.alloc(400);
    assert_eq!(buf.len(), 400);
}

// --- ScratchBuffer tests ---
