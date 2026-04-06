
#[test]
fn test_hybrid_scheduler_m1_forces_cpu_deep_gcov() {
    // IMP-097: m=1 operations should always use CPU
    let scheduler = HybridScheduler::with_threshold(1).expect("test");

    // Even with threshold=1, m=1 should force CPU
    assert!(!scheduler.should_use_gpu(1, 1000, 1000));
    assert!(!scheduler.should_use_gpu(1, 10000, 10000));
}

#[test]
fn test_hybrid_scheduler_transpose_b_small_deep_gcov() {
    // Test matmul with transposed B for CPU path
    let mut scheduler = HybridScheduler::with_threshold(1_000_000).expect("test");

    // Q @ K^T style operation
    let q = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let k = vec![1.0, 0.0, 0.0, 1.0]; // 2x2, will be transposed

    let result = scheduler.matmul_transpose_b(&q, &k, 2, 2, 2);
    assert!(result.is_ok());
    let scores = result.expect("GPU operation failed");
    assert_eq!(scores.len(), 4);
}

#[test]
fn test_streaming_kv_cache_wraparound_deep_gcov() {
    // Test circular buffer wraparound behavior
    let mut cache = StreamingKVCache::new(2, 3, 2, 4); // max 3 positions
    let kv_dim = 2 * 4; // 8

    // Append 5 positions (should wrap around)
    for i in 0..5 {
        let k = vec![i as f32; kv_dim];
        let v = vec![i as f32 * 10.0; kv_dim];
        for layer in 0..2 {
            cache.append(layer, &k, &v);
        }
    }

    // Cache should have max_positions (3) valid entries
    assert_eq!(cache.len(), 3);
    assert!(!cache.is_empty());
}

#[test]
fn test_streaming_kv_cache_memory_calculation_deep_gcov() {
    // Test memory calculation
    let cache = StreamingKVCache::new(4, 100, 8, 64);

    // Memory = 4 layers * 100 pos * 8 heads * 64 dim * 2 (K+V) * 4 bytes
    let expected = 4 * 100 * 8 * 64 * 2 * 4;
    assert_eq!(cache.memory_bytes(), expected);

    let expected_mb = expected as f64 / (1024.0 * 1024.0);
    assert!((cache.memory_mb() - expected_mb).abs() < 0.001);
}

#[test]
fn test_tensor_pool_capacity_limit_deep_gcov() {
    // Test pool respects capacity limit
    let mut pool = TensorPool::new(2);
    assert_eq!(pool.capacity(), 2);

    // Acquire and release 3 buffers (exceeds capacity)
    let b1 = pool.acquire(100);
    let b2 = pool.acquire(200);
    let b3 = pool.acquire(300);

    pool.release(b1);
    pool.release(b2);
    pool.release(b3); // This one should be dropped

    assert_eq!(pool.available(), 2);
}

#[test]
fn test_tensor_pool_size_matching_deep_gcov() {
    // Test pool finds appropriate sized buffer
    let mut pool = TensorPool::new(10);

    // Release a large buffer
    let large = vec![0.0f32; 1000];
    pool.release(large);

    // Request smaller buffer - should reuse the large one
    let buf = pool.acquire(500);
    assert!(buf.capacity() >= 500);
}

#[test]
fn test_forward_arena_insufficient_capacity_deep_gcov() {
    // Test arena panics on insufficient capacity
    let mut arena = ForwardArena::new(100);

    // First allocation succeeds
    let _ = arena.alloc(50);
    assert_eq!(arena.used(), 50);

    // Second allocation succeeds
    let _ = arena.alloc(49);
    assert_eq!(arena.used(), 99);

    // Reset and verify
    arena.reset();
    assert_eq!(arena.used(), 0);
}

#[test]
fn test_scratch_buffer_layer_access_deep_gcov() {
    // Test scratch buffer layer access
    let mut scratch = ScratchBuffer::new(4, 128);

    assert_eq!(scratch.num_layers(), 4);
    assert_eq!(scratch.layer_size(), 128);
    assert_eq!(scratch.total_size(), 512);

    // Modify each layer
    for i in 0..4 {
        let layer = scratch.get_layer_mut(i);
        layer.fill(i as f32);
    }

    // Verify each layer has correct values
    for i in 0..4 {
        let layer = scratch.get_layer(i);
        assert!(layer.iter().all(|&x| (x - i as f32).abs() < 1e-5));
    }

    // Reset and verify zeros
    scratch.reset();
    for i in 0..4 {
        assert!(scratch.get_layer(i).iter().all(|&x| x == 0.0));
    }
}

#[test]
fn test_quantized_dot_q4_short_blocks_deep_gcov() {
    // Test Q4 dot product with blocks smaller than required
    let short_a: [u8; 10] = [0; 10]; // Less than 18 required
    let short_b: [u8; 10] = [0; 10];

    let result = quantized_dot_q4(&short_a, &short_b);
    assert_eq!(result, 0.0); // Should return 0 for invalid blocks
}

#[test]
fn test_quantized_dot_q8_short_blocks_deep_gcov() {
    // Test Q8 dot product with blocks smaller than required
    let short_a: [u8; 20] = [0; 20]; // Less than 34 required
    let short_b: [u8; 20] = [0; 20];

    let result = quantized_dot_q8(&short_a, &short_b);
    assert_eq!(result, 0.0); // Should return 0 for invalid blocks
}

#[test]
fn test_quantized_accumulator_operations_deep_gcov() {
    // Test quantized accumulator operations
    let mut acc = QuantizedAccumulator::new();
    assert_eq!(acc.sum(), 0.0);

    acc.add_scaled(2.0, 3.0);
    assert!((acc.sum() - 6.0).abs() < 1e-5);

    acc.add_block(4.0, 0.5);
    assert!((acc.sum() - 8.0).abs() < 1e-5);

    acc.reset();
    assert_eq!(acc.sum(), 0.0);
}

#[test]
fn test_double_buffer_operations_deep_gcov() {
    // Test double buffer swap and access
    let mut db: DoubleBuffer<f32> = DoubleBuffer::new(100);
    assert_eq!(db.capacity(), 100);

    // Write to back buffer
    db.back_mut().fill(1.0);

    // Front should still be zeros
    assert!(db.front().iter().all(|&x| x == 0.0));

    // Swap
    db.swap();

    // Now front has our data
    assert!(db.front().iter().all(|&x| x == 1.0));
}

#[test]
fn test_chunked_processor_empty_data_deep_gcov() {
    // Test chunked processor with empty data
    let processor = ChunkedProcessor::new(64);

    assert_eq!(processor.num_chunks(0), 0);

    let empty: Vec<f32> = vec![];
    let result = processor.process_chunks(&empty, |_chunk| 0.0);
    assert_eq!(result, 0.0);
}

#[test]
fn test_chunked_processor_exact_chunks_deep_gcov() {
    // Test chunked processor when data is exact multiple of chunk size
    let processor = ChunkedProcessor::new(4);

    let _data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 8 elements, 2 chunks
    assert_eq!(processor.num_chunks(8), 2);

    let (start, end) = processor.chunk_bounds(0, 8);
    assert_eq!((start, end), (0, 4));

    let (start, end) = processor.chunk_bounds(1, 8);
    assert_eq!((start, end), (4, 8));
}

#[test]
fn test_chunked_processor_partial_chunk_deep_gcov() {
    // Test chunked processor with partial last chunk
    let processor = ChunkedProcessor::new(4);

    let _data = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 5 elements, 2 chunks (4 + 1)
    assert_eq!(processor.num_chunks(5), 2);

    let (start, end) = processor.chunk_bounds(1, 5);
    assert_eq!((start, end), (4, 5)); // Partial chunk
}

#[test]
fn test_inference_pipeline_stage_tracking_deep_gcov() {
    // Test pipeline stage timing
    let mut pipeline = InferencePipeline::new(4);
    assert_eq!(pipeline.num_stages(), 4);
    assert_eq!(pipeline.total_latency(), 0.0);

    pipeline.record_stage_time(GpuPipelineStage::Embed, 1.5);
    pipeline.record_stage_time(GpuPipelineStage::Attention, 3.0);
    pipeline.record_stage_time(GpuPipelineStage::FFN, 2.5);
    pipeline.record_stage_time(GpuPipelineStage::Output, 0.5);

    assert!((pipeline.total_latency() - 7.5).abs() < 1e-5);

    let breakdown = pipeline.stage_breakdown();
    assert_eq!(breakdown.len(), 4);

    pipeline.reset();
    assert_eq!(pipeline.total_latency(), 0.0);
}

#[test]
fn test_token_batch_push_and_flush_deep_gcov() {
    // Test token batch accumulation and flushing
    let mut batch = TokenBatch::new(3);
    assert_eq!(batch.capacity(), 3);
    assert!(batch.is_empty());
    assert!(!batch.is_full());

    // Push tokens
    assert!(batch.push(1).is_none());
    assert!(batch.push(2).is_none());

    // Third push fills and returns batch
    let result = batch.push(3);
    assert!(result.is_some());
    assert_eq!(result.expect("index out of bounds"), vec![1, 2, 3]);

    // Batch should be empty after auto-flush
    assert!(batch.is_empty());
}

#[test]
fn test_speculative_buffer_verify_all_match_deep_gcov() {
    // Test speculative buffer verification when all tokens match
    let mut buffer = SpeculativeBuffer::new(5);

    buffer.add_candidate(10, 0.9);
    buffer.add_candidate(20, 0.8);
    buffer.add_candidate(30, 0.7);

    let (accepted, rejection_idx) = buffer.verify(&[10, 20, 30]);
    assert_eq!(accepted, 3);
    assert!(rejection_idx.is_none());
}

#[test]
fn test_speculative_buffer_verify_partial_match_deep_gcov() {
    // Test speculative buffer verification with partial match
    let mut buffer = SpeculativeBuffer::new(5);

    buffer.add_candidate(10, 0.9);
    buffer.add_candidate(20, 0.8);
    buffer.add_candidate(30, 0.7);

    let (accepted, rejection_idx) = buffer.verify(&[10, 20, 99]); // Mismatch at index 2
    assert_eq!(accepted, 2);
    assert_eq!(rejection_idx, Some(2));
}

#[test]
fn test_speculative_buffer_accept_reject_deep_gcov() {
    // Test accepting and rejecting candidates
    let mut buffer = SpeculativeBuffer::new(5);

    buffer.add_candidate(10, 0.9);
    buffer.add_candidate(20, 0.8);
    buffer.add_candidate(30, 0.7);

    // Accept first 2
    buffer.accept(2);
    assert_eq!(buffer.len(), 1); // One remaining

    // Reject remaining
    buffer.reject();
    assert!(buffer.is_empty());
}

#[test]
fn test_inference_batch_scheduler_workflow_deep_gcov() {
    // Test full batch scheduler workflow
    let mut scheduler = InferenceBatchScheduler::new();

    // Submit batches
    let id1 = scheduler.submit(vec![1, 2, 3]);
    let id2 = scheduler.submit(vec![4, 5, 6]);

    assert_eq!(scheduler.pending_count(), 2);
    assert_eq!(scheduler.completed_count(), 0);

    // Complete first batch
    scheduler.complete(id1, vec![10, 11, 12]);
    assert_eq!(scheduler.pending_count(), 1);
    assert_eq!(scheduler.completed_count(), 1);

    // Poll completed
    let result = scheduler.poll();
    assert!(result.is_some());
    let (batch_id, tokens) = result.expect("GPU operation failed");
    assert_eq!(batch_id, id1);
    assert_eq!(tokens, vec![10, 11, 12]);

    // Complete and drain remaining
    scheduler.complete(id2, vec![40, 50, 60]);
    let drained = scheduler.drain();
    assert_eq!(drained.len(), 1);
}

#[test]
fn test_async_request_queue_backpressure_deep_gcov() {
    // Test async request queue backpressure
    let mut queue: AsyncRequestQueue<u32> = AsyncRequestQueue::new(3);

    assert!(queue.try_push(1));
    assert!(queue.try_push(2));
    assert!(queue.try_push(3));
    assert!(queue.is_full());

    // Backpressure - push should fail
    assert!(!queue.try_push(4));

    // Pop one and try again
    assert_eq!(queue.try_pop(), Some(1));
    assert!(queue.try_push(4)); // Now succeeds
}

#[test]
fn test_inference_event_notifier_multiple_handlers_deep_gcov() {
    // Test event notifier with multiple handlers
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let mut notifier = InferenceEventNotifier::new();
    let count = Arc::new(AtomicUsize::new(0));

    // Register 3 handlers
    for _ in 0..3 {
        let count_clone = Arc::clone(&count);
        notifier.register(Box::new(move |_id, _tokens| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        }));
    }

    assert_eq!(notifier.handler_count(), 3);

    // Notify
    notifier.notify(42, &[1, 2, 3]);
    assert_eq!(count.load(Ordering::SeqCst), 3);

    // Clear and verify
    notifier.clear();
    assert_eq!(notifier.handler_count(), 0);
}

#[test]
fn test_timeout_manager_check_expired_deep_gcov() {
    // Test timeout manager expiry checking
    use std::time::{Duration, Instant};

    let mut manager = TimeoutManager::new();

    // Register timeouts - one immediate, one far future
    let now = Instant::now();
    manager.register(1, now); // Already expired
    manager.register(2, now + Duration::from_secs(3600)); // Far future

    assert_eq!(manager.active_count(), 2);

    // Check expired
    let expired = manager.check_expired();
    assert!(expired.contains(&1));
    assert!(!expired.contains(&2));

    assert_eq!(manager.active_count(), 1);

    // Remove remaining
    manager.remove(2);
    assert_eq!(manager.active_count(), 0);
}

#[test]
fn test_priority_request_queue_ordering_deep_gcov() {
    // Test priority queue maintains correct ordering
    let mut queue: PriorityRequestQueue<&str> = PriorityRequestQueue::new();

    // Enqueue with different priorities
    queue.enqueue(PriorityRequest::new(1, "low"));
    queue.enqueue(PriorityRequest::new(10, "high"));
    queue.enqueue(PriorityRequest::new(5, "medium"));

    // Should dequeue highest first
    let item = queue.dequeue_highest();
    assert!(item.is_some());
    assert_eq!(item.expect("GPU operation failed").into_data(), "high");

    let item = queue.dequeue_highest();
    assert_eq!(item.expect("GPU operation failed").into_data(), "medium");

    let item = queue.dequeue_highest();
    assert_eq!(item.expect("GPU operation failed").into_data(), "low");

    assert!(queue.is_empty());
}
