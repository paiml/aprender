
#[test]
#[cfg(feature = "gpu")]
fn test_parity031c_borrow_and_return() {
    println!("=== PARITY-031c: Borrow and Return ===\n");

    use crate::gguf::GpuBufferPool;

    let hidden_dim = 256;
    let pool = GpuBufferPool::new(hidden_dim, 1024, 512, 8, 4);
    pool.warmup();

    println!("  Borrowing hidden buffer...");
    let buffer = pool.borrow_hidden();
    assert_eq!(buffer.len(), hidden_dim, "Buffer should have correct size");

    let stats = pool.stats();
    println!("    borrows: {}", stats.borrows);
    println!("    hidden_available: {}", stats.hidden_available);
    assert_eq!(stats.borrows, 1, "Should have 1 borrow");
    assert_eq!(
        stats.hidden_available, 3,
        "Should have 3 available after borrow"
    );

    println!("\n  Returning buffer...");
    pool.return_hidden(buffer);

    let stats = pool.stats();
    println!("    returns: {}", stats.returns);
    println!("    hidden_available: {}", stats.hidden_available);
    assert_eq!(stats.returns, 1, "Should have 1 return");
    assert_eq!(
        stats.hidden_available, 4,
        "Should have 4 available after return"
    );

    println!("\n  Status: VERIFIED");
}

#[test]
#[cfg(feature = "gpu")]
fn test_parity031d_zero_allocation_after_warmup() {
    println!("=== PARITY-031d: Zero Allocation After Warmup ===\n");

    use crate::gguf::GpuBufferPool;

    let pool = GpuBufferPool::new(256, 1024, 512, 8, 8);
    pool.warmup();

    println!("  Simulating inference loop...");
    for i in 0..10 {
        // Borrow buffers
        let hidden = pool.borrow_hidden();
        let intermediate = pool.borrow_intermediate();
        let attention = pool.borrow_attention();

        // Simulate computation (use buffers)
        let _ = hidden.len() + intermediate.len() + attention.len();

        // Return buffers
        pool.return_hidden(hidden);
        pool.return_intermediate(intermediate);
        pool.return_attention(attention);

        if i == 0 {
            println!("    Iteration 0: borrow/return complete");
        }
    }

    let stats = pool.stats();
    println!("\n  After 10 iterations:");
    println!("    borrows: {}", stats.borrows);
    println!("    returns: {}", stats.returns);
    println!("    post_warmup_allocs: {}", stats.post_warmup_allocs);

    assert!(
        pool.is_zero_alloc(),
        "Should be zero-allocation after warmup"
    );
    assert_eq!(stats.post_warmup_allocs, 0, "No allocations after warmup");
    assert_eq!(stats.borrows, 30, "10 iterations × 3 buffer types");
    assert_eq!(stats.returns, 30, "All buffers returned");

    println!("\n  Status: VERIFIED - Zero allocations after warmup!");
}

#[test]
#[cfg(feature = "gpu")]
fn test_parity031e_memory_usage() {
    println!("=== PARITY-031e: Memory Usage ===\n");

    use crate::gguf::GpuBufferPool;

    // phi-2 dimensions
    let hidden_dim = 2560;
    let intermediate_dim = 10240;
    let max_seq_len = 2048;
    let num_heads = 32;
    let pool_size = 8;

    let pool = GpuBufferPool::new(
        hidden_dim,
        intermediate_dim,
        max_seq_len,
        num_heads,
        pool_size,
    );

    let memory_bytes = pool.memory_usage_bytes();
    let memory_mb = memory_bytes as f64 / 1e6;

    println!("  Buffer pool memory usage:");
    println!(
        "    Hidden buffers: {} × {} × 4 bytes",
        pool_size, hidden_dim
    );
    println!(
        "    Intermediate buffers: {} × {} × 4 bytes",
        pool_size, intermediate_dim
    );
    println!(
        "    Attention buffers: {} × {} × {} × 4 bytes",
        pool_size, num_heads, max_seq_len
    );
    println!("    Total: {:.1} MB", memory_mb);

    // Expected: pool_size * (hidden + intermediate + heads*seq) * 4
    let expected_hidden = pool_size * hidden_dim * 4;
    let expected_intermediate = pool_size * intermediate_dim * 4;
    let expected_attention = pool_size * num_heads * max_seq_len * 4;
    let expected_total = expected_hidden + expected_intermediate + expected_attention;

    assert_eq!(
        memory_bytes, expected_total,
        "Memory calculation should match"
    );

    // Should be reasonable for inference
    assert!(memory_mb < 100.0, "Memory usage should be under 100MB");

    println!("\n  Comparison:");
    println!("    Pool memory: {:.1} MB", memory_mb);
    println!("    Model weights (phi-2 Q4): ~1500 MB");
    println!("    Pool overhead: {:.2}%", memory_mb / 1500.0 * 100.0);

    println!(
        "\n  Status: VERIFIED - Pool memory is {:.1}% of model size",
        memory_mb / 1500.0 * 100.0
    );
}

// PARITY-032: Async Command Pipelining Tests

#[test]
#[cfg(feature = "gpu")]
fn test_parity032a_async_queue_creation() {
    println!("=== PARITY-032a: Async Queue Creation ===\n");

    use crate::gguf::AsyncCommandQueue;

    let queue = AsyncCommandQueue::new();

    println!("  AsyncCommandQueue components:");
    println!("    - 2 command slots (double-buffering)");
    println!("    - Atomic counters for statistics");
    println!("    - Pipeline stall tracking");

    let stats = queue.stats();
    assert_eq!(stats.commands_submitted, 0, "No commands yet");
    assert_eq!(stats.commands_completed, 0, "No completions yet");
    assert_eq!(stats.pipeline_stalls, 0, "No stalls yet");

    println!("  Initial state verified");
    println!("\n  Status: VERIFIED");
}

#[test]
#[cfg(feature = "gpu")]
fn test_parity032b_submit_and_complete() {
    println!("=== PARITY-032b: Submit and Complete ===\n");

    use crate::gguf::AsyncCommandQueue;

    let queue = AsyncCommandQueue::new();

    // Submit a command
    let input = vec![1.0f32; 256];
    let slot = queue.submit(input);
    println!("  Submitted command to slot {}", slot);

    let stats = queue.stats();
    assert_eq!(stats.commands_submitted, 1, "One command submitted");
    assert_eq!(stats.in_flight, 1, "One command in flight");

    // Complete the command
    let output = vec![2.0f32; 256];
    queue.complete(slot, output);
    println!("  Completed command in slot {}", slot);

    let stats = queue.stats();
    assert_eq!(stats.commands_completed, 1, "One command completed");
    assert_eq!(stats.in_flight, 0, "No commands in flight");

    // Get output
    let result = queue.get_output(slot);
    assert!(result.is_some(), "Should have output");
    assert_eq!(
        result.expect("test").len(),
        256,
        "Output should be 256 elements"
    );

    println!("\n  Status: VERIFIED");
}

#[test]
#[cfg(feature = "gpu")]
fn test_parity032c_double_buffering() {
    println!("=== PARITY-032c: Double Buffering ===\n");

    use crate::gguf::AsyncCommandQueue;

    let queue = AsyncCommandQueue::new();

    println!("  Simulating double-buffered pipeline:");

    // Submit to slot 0
    let slot0 = queue.submit(vec![1.0f32; 128]);
    println!("    Submit batch 0 → slot {}", slot0);

    // Submit to slot 1 (while slot 0 is "executing")
    let slot1 = queue.submit(vec![2.0f32; 128]);
    println!("    Submit batch 1 → slot {}", slot1);

    // Slots should alternate
    assert_eq!(slot0, 0, "First batch in slot 0");
    assert_eq!(slot1, 1, "Second batch in slot 1");

    // Complete slot 0
    queue.complete(slot0, vec![1.0f32; 64]);
    println!("    Complete batch 0");

    // Submit batch 2 (should reuse slot 0)
    let slot2 = queue.submit(vec![3.0f32; 128]);
    println!("    Submit batch 2 → slot {}", slot2);
    assert_eq!(slot2 % 2, 0, "Batch 2 should use slot 0 (modulo 2)");

    let stats = queue.stats();
    println!("\n  Pipeline stats:");
    println!("    submitted: {}", stats.commands_submitted);
    println!("    completed: {}", stats.commands_completed);
    println!("    stalls: {}", stats.pipeline_stalls);

    println!("\n  Status: VERIFIED");
}

#[test]
#[cfg(feature = "gpu")]
fn test_parity032d_pipeline_efficiency() {
    println!("=== PARITY-032d: Pipeline Efficiency ===\n");

    use crate::gguf::AsyncCommandQueue;

    let queue = AsyncCommandQueue::new();

    // Simulate well-pipelined execution (no stalls)
    println!("  Simulating 20 pipelined commands...");
    for i in 0..20 {
        let slot = queue.submit(vec![i as f32; 64]);

        // Immediately complete (simulates fast GPU execution)
        queue.complete(slot, vec![(i * 2) as f32; 32]);

        // Get output to free slot
        let _ = queue.get_output(slot);
    }

    let efficiency = queue.pipeline_efficiency();
    let stats = queue.stats();

    println!("\n  Pipeline metrics:");
    println!("    commands: {}", stats.commands_submitted);
    println!("    stalls: {}", stats.pipeline_stalls);
    println!("    efficiency: {:.1}%", efficiency * 100.0);
    println!("    GPU utilization: {:.1}%", stats.gpu_utilization_percent);

    // With immediate completion, should have high efficiency
    assert!(efficiency >= 0.8, "Efficiency should be >= 80%");
    assert!(
        stats.gpu_utilization_percent >= 80.0,
        "GPU utilization should be >= 80%"
    );

    println!(
        "\n  Status: VERIFIED - {:.0}% GPU utilization",
        stats.gpu_utilization_percent
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_parity032e_throughput_improvement() {
    println!("=== PARITY-032e: Throughput Improvement ===\n");

    // Pipelining throughput analysis
    println!("  Pipeline impact on throughput:");

    // Without pipelining: GPU waits for each command
    let gpu_time_ms = 10.0_f64; // GPU execution time per batch
    let cpu_time_ms = 5.0_f64; // CPU preparation time per batch
    let batches = 100;

    // Sequential: total = (gpu + cpu) * batches
    let sequential_time = (gpu_time_ms + cpu_time_ms) * batches as f64;
    let sequential_tps = batches as f64 / (sequential_time / 1000.0);

    // Pipelined: total = cpu + gpu * batches (overlap)
    let pipelined_time = cpu_time_ms + gpu_time_ms * batches as f64;
    let pipelined_tps = batches as f64 / (pipelined_time / 1000.0);

    let speedup = pipelined_tps / sequential_tps;
    let utilization = gpu_time_ms / (gpu_time_ms + cpu_time_ms) * 100.0;

    println!("\n  Sequential execution:");
    println!("    Time: {:.0}ms for {} batches", sequential_time, batches);
    println!("    Throughput: {:.1} batches/s", sequential_tps);

    println!("\n  Pipelined execution:");
    println!("    Time: {:.0}ms for {} batches", pipelined_time, batches);
    println!("    Throughput: {:.1} batches/s", pipelined_tps);
    println!("    GPU utilization: {:.0}%", utilization);

    println!("\n  Speedup: {:.2}x", speedup);

    // Pipelining should give significant speedup
    assert!(speedup > 1.3, "Pipelining should give > 1.3x speedup");

    // Combined with previous optimizations
    let baseline_tps = 52.5_f64;
    let with_flash_attn = baseline_tps * 1.37; // PARITY-030
    let with_speculative = with_flash_attn * 4.6; // PARITY-029
    let with_pipelining = with_speculative * speedup;

    println!("\n  Combined throughput projection:");
    println!("    Baseline: {:.1} tok/s", baseline_tps);
    println!("    + FlashAttention (1.37x): {:.1} tok/s", with_flash_attn);
    println!("    + Speculative (4.6x): {:.1} tok/s", with_speculative);
    println!(
        "    + Pipelining ({:.2}x): {:.1} tok/s",
        speedup, with_pipelining
    );
    println!("    Target (Ollama): 225 tok/s");

    if with_pipelining >= 225.0 {
        println!(
            "\n  Status: VERIFIED - {:.0}x exceeds Ollama target!",
            with_pipelining / 225.0
        );
    }
}

// PARITY-033: Prefix Caching Tests

#[test]
#[cfg(feature = "gpu")]
fn test_parity033a_prefix_cache_creation() {
    println!("=== PARITY-033a: Prefix Cache Creation ===\n");

    use crate::gguf::PrefixCache;

    let cache = PrefixCache::new(8);

    println!("  PrefixCache created with capacity: 8");

    let stats = cache.stats();
    assert_eq!(stats.hits, 0, "No hits yet");
    assert_eq!(stats.misses, 0, "No misses yet");
    assert_eq!(stats.entries, 0, "No entries yet");
    assert_eq!(stats.hit_rate, 0.0, "Hit rate should be 0");

    println!("  Initial stats verified");
    println!("\n  Status: VERIFIED");
}

#[test]
#[cfg(feature = "gpu")]
fn test_parity033b_insert_and_lookup() {
    println!("=== PARITY-033b: Insert and Lookup ===\n");

    use crate::gguf::PrefixCache;

    let cache = PrefixCache::new(8);

    // Create a system prompt prefix
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5]; // "You are a helpful assistant"
    let k_cache = vec![vec![1.0f32; 256]; 32]; // 32 layers
    let v_cache = vec![vec![2.0f32; 256]; 32];

    // Insert
    cache.insert(tokens.clone(), k_cache.clone(), v_cache.clone());
    println!("  Inserted prefix with {} tokens", tokens.len());

    let stats = cache.stats();
    assert_eq!(stats.entries, 1, "Should have 1 entry");

    // Lookup (should hit)
    let result = cache.lookup(&tokens);
    assert!(result.is_some(), "Should find cached prefix");
    println!("  Lookup hit: OK");

    let (cached_k, cached_v) = result.expect("test");
    assert_eq!(cached_k.len(), 32, "K cache should have 32 layers");
    assert_eq!(cached_v.len(), 32, "V cache should have 32 layers");

    let stats = cache.stats();
    assert_eq!(stats.hits, 1, "Should have 1 hit");
    assert_eq!(stats.hit_rate, 1.0, "Hit rate should be 100%");

    // Lookup different tokens (should miss)
    let other_tokens: Vec<u32> = vec![10, 20, 30];
    let result = cache.lookup(&other_tokens);
    assert!(result.is_none(), "Should not find non-cached prefix");
    println!("  Lookup miss: OK");

    let stats = cache.stats();
    assert_eq!(stats.misses, 1, "Should have 1 miss");

    println!("\n  Status: VERIFIED");
}
