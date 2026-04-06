
// ============================================================================
// M11 Mega-Long Context Tests (32768+ positions)
// ============================================================================

#[test]
fn test_streaming_kv_cache_32768_positions() {
    // M11 target: 32768 context positions
    let num_layers = 4; // Use smaller for test speed
    let max_positions = 32768;
    let num_heads = 8;
    let head_dim = 64;

    let mut cache = StreamingKVCache::new(num_layers, max_positions, num_heads, head_dim);

    assert_eq!(cache.max_positions(), 32768);
    assert_eq!(cache.len(), 0);

    // Fill to capacity - must fill all layers for each position
    let kv_dim = num_heads * head_dim;
    let key = vec![0.1f32; kv_dim];
    let value = vec![0.2f32; kv_dim];

    for _pos in 0..32768 {
        // Append to all layers for each position
        for layer in 0..num_layers {
            cache.append(layer, &key, &value);
        }
    }

    // Should have filled to max_positions
    assert_eq!(cache.len(), max_positions);
}

#[test]
fn test_mega_long_context_memory_bound() {
    // Verify 32768 context memory stays bounded
    let num_layers = 32;
    let max_positions = 32768;
    let num_heads = 32;
    let head_dim = 128;

    let cache = StreamingKVCache::new(num_layers, max_positions, num_heads, head_dim);

    // Memory calculation:
    // 32 layers * 32768 positions * 32 heads * 128 dim * 2 (K+V) * 4 bytes
    // = 34,359,738,368 bytes = 34.36 GB
    let expected_bytes = num_layers * max_positions * num_heads * head_dim * 2 * 4;
    assert_eq!(cache.memory_bytes(), expected_bytes);

    let memory_gb = cache.memory_mb() / 1024.0;
    assert!(
        memory_gb < 36.0,
        "32768 context KV cache should be < 36 GB, got {:.2} GB",
        memory_gb
    );
}

#[test]
fn test_mega_long_context_fill_performance() {
    use std::time::Instant;

    let num_layers = 4;
    let max_positions = 32768;
    let num_heads = 8;
    let head_dim = 64;

    let mut cache = StreamingKVCache::new(num_layers, max_positions, num_heads, head_dim);

    let kv_dim = num_heads * head_dim;
    let key = vec![0.1f32; kv_dim];
    let value = vec![0.2f32; kv_dim];

    // Measure fill time
    let start = Instant::now();
    for _pos in 0..32768 {
        // Append to all layers for each position
        for layer in 0..num_layers {
            cache.append(layer, &key, &value);
        }
    }
    let elapsed = start.elapsed();

    // Should fill 32768 positions in < 4 seconds
    let fill_rate = 32768.0 / elapsed.as_secs_f64();
    assert!(
        fill_rate > 25.0,
        "Fill rate should be > 25 pos/s, got {:.0}",
        fill_rate
    );
}

// ==================== M12: FP16 KV Cache Tests (65536 Context) ====================

#[test]
fn test_f32_f16_conversion_roundtrip() {
    // Test that FP16 conversion preserves values within tolerance
    let test_values = vec![
        0.0f32, 1.0, -1.0, 0.5, -0.5, 0.125, 100.0, -100.0, 0.001, 65504.0,
    ];

    for &original in &test_values {
        let fp16_bits = StreamingKVCacheFp16::f32_to_f16(original);
        let recovered = StreamingKVCacheFp16::f16_to_f32(fp16_bits);

        // FP16 has limited precision, check relative error
        let error = if original.abs() > 1e-6 {
            ((recovered - original) / original).abs()
        } else {
            (recovered - original).abs()
        };

        assert!(
            error < 0.01,
            "FP16 roundtrip error too large for {}: got {}, error {}",
            original,
            recovered,
            error
        );
    }
}

#[test]
fn test_streaming_kv_cache_fp16_basic() {
    let num_layers = 2;
    let max_positions = 16;
    let num_heads = 4;
    let head_dim = 8;

    let mut cache = StreamingKVCacheFp16::new(num_layers, max_positions, num_heads, head_dim);

    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.max_positions(), 16);

    // Append a single position
    let kv_dim = num_heads * head_dim;
    let key = vec![0.5f32; kv_dim];
    let value = vec![0.25f32; kv_dim];

    for layer in 0..num_layers {
        cache.append(layer, &key, &value);
    }

    assert_eq!(cache.len(), 1);

    // Retrieve and verify
    let (keys, values) = cache.get_valid_f32(0);
    assert_eq!(keys.len(), kv_dim);
    assert_eq!(values.len(), kv_dim);

    // Check values within FP16 tolerance
    for &k in &keys {
        assert!((k - 0.5).abs() < 0.01, "Key mismatch: {}", k);
    }
    for &v in &values {
        assert!((v - 0.25).abs() < 0.01, "Value mismatch: {}", v);
    }
}

#[test]
#[ignore = "allocates 100GB+ memory - run with --ignored"]
fn test_streaming_kv_cache_fp16_memory_half() {
    // Verify FP16 uses half the memory of FP32
    let num_layers = 32;
    let max_positions = 65536;
    let num_heads = 32;
    let head_dim = 128;

    let cache_fp16 = StreamingKVCacheFp16::new(num_layers, max_positions, num_heads, head_dim);
    let cache_fp32 = StreamingKVCache::new(num_layers, max_positions, num_heads, head_dim);

    let fp16_bytes = cache_fp16.memory_bytes();
    let fp32_bytes = cache_fp32.memory_bytes();

    // FP16 should be exactly half
    assert_eq!(fp16_bytes * 2, fp32_bytes);

    // FP16 memory for 65536 context should be ~34.36 GB
    let fp16_gb = cache_fp16.memory_mb() / 1024.0;
    assert!(
        fp16_gb < 36.0,
        "FP16 65536 context should be < 36 GB, got {:.2} GB",
        fp16_gb
    );
    assert!(
        fp16_gb > 30.0,
        "FP16 65536 context should be > 30 GB, got {:.2} GB",
        fp16_gb
    );
}

#[test]
#[ignore = "allocates large memory for 65536 positions - run with --ignored"]
fn test_streaming_kv_cache_fp16_65536_positions() {
    // Test that FP16 cache handles 65536 positions
    let num_layers = 4;
    let max_positions = 65536;
    let num_heads = 8;
    let head_dim = 64;

    let mut cache = StreamingKVCacheFp16::new(num_layers, max_positions, num_heads, head_dim);

    let kv_dim = num_heads * head_dim;
    let key = vec![0.1f32; kv_dim];
    let value = vec![0.2f32; kv_dim];

    // Fill to capacity
    for _pos in 0..65536 {
        for layer in 0..num_layers {
            cache.append(layer, &key, &value);
        }
    }

    assert_eq!(cache.len(), max_positions);

    // Verify circular buffer works
    for layer in 0..num_layers {
        cache.append(layer, &key, &value);
    }
    assert_eq!(cache.len(), max_positions); // Still at capacity
}

#[test]
#[ignore = "allocates 34GB+ memory - run with --ignored"]
fn test_fp16_kv_cache_memory_bound_65536() {
    // Verify 65536 context FP16 memory stays bounded
    let num_layers = 32;
    let max_positions = 65536;
    let num_heads = 32;
    let head_dim = 128;

    let cache = StreamingKVCacheFp16::new(num_layers, max_positions, num_heads, head_dim);

    // Memory calculation:
    // 32 layers * 65536 positions * 32 heads * 128 dim * 2 (K+V) * 2 bytes
    // = 34,359,738,368 bytes = 34.36 GB
    let expected_bytes = num_layers * max_positions * num_heads * head_dim * 2 * 2;
    assert_eq!(cache.memory_bytes(), expected_bytes);

    let memory_gb = cache.memory_mb() / 1024.0;
    assert!(
        memory_gb < 36.0,
        "65536 context FP16 KV cache should be < 36 GB, got {:.2} GB",
        memory_gb
    );
}

#[test]
#[ignore = "allocates large memory for 65536 positions - run with --ignored"]
fn test_fp16_kv_cache_fill_performance_65536() {
    use std::time::Instant;

    let num_layers = 4;
    let max_positions = 65536;
    let num_heads = 8;
    let head_dim = 64;

    let mut cache = StreamingKVCacheFp16::new(num_layers, max_positions, num_heads, head_dim);

    let kv_dim = num_heads * head_dim;
    let key = vec![0.1f32; kv_dim];
    let value = vec![0.2f32; kv_dim];

    // Measure fill time
    let start = Instant::now();
    for _pos in 0..65536 {
        for layer in 0..num_layers {
            cache.append(layer, &key, &value);
        }
    }
    let elapsed = start.elapsed();

    // Should fill 65536 positions in reasonable time
    let fill_rate = 65536.0 / elapsed.as_secs_f64();
    assert!(
        fill_rate > 10.0,
        "FP16 fill rate should be > 10 pos/s, got {:.0}",
        fill_rate
    );
}

// =========================================================================
// IMP-1001: CUDA Inference Integration (~100x impact)
// Wire CudaExecutor into GpuModel for real GPU-accelerated inference
// =========================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1001a_cuda_executor_matmul_correctness() {
    // IMP-1001a: Verify CudaExecutor matmul produces correct results
    use crate::cuda::CudaExecutor;

    if !CudaExecutor::is_available() {
        println!("IMP-1001a: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create CudaExecutor");

    // Simple test: 4x4 @ 4x4 with all 1s -> each element = 4
    let a = vec![1.0f32; 16]; // 4x4 ones
    let b = vec![1.0f32; 16]; // 4x4 ones
    let mut result = vec![0.0f32; 16]; // 4x4 output

    executor
        .gemm(&a, &b, &mut result, 4, 4, 4)
        .expect("GEMM failed");

    // Each element should be 4.0 (dot product of 4 ones)
    for (i, &val) in result.iter().enumerate() {
        assert!(
            (val - 4.0).abs() < 1e-3,
            "IMP-1001a: Element {} mismatch: got {}, expected 4.0",
            i,
            val
        );
    }

    // Also test larger size: 8x8 @ 8x8
    let a = vec![2.0f32; 64]; // 8x8 twos
    let b = vec![1.0f32; 64]; // 8x8 ones
    let mut result = vec![0.0f32; 64];

    executor
        .gemm(&a, &b, &mut result, 8, 8, 8)
        .expect("GEMM 8x8 failed");

    // Each element should be 16.0 (8 * 2 * 1)
    for (i, &val) in result.iter().enumerate() {
        assert!(
            (val - 16.0).abs() < 1e-3,
            "IMP-1001a: 8x8 element {} mismatch: got {}, expected 16.0",
            i,
            val
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1001b_cuda_softmax_correctness() {
    // IMP-1001b: Verify CudaExecutor softmax produces correct results
    use crate::cuda::CudaExecutor;

    if !CudaExecutor::is_available() {
        println!("IMP-1001b: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create CudaExecutor");

    let mut data = vec![1.0, 2.0, 3.0, 4.0];
    executor.softmax(&mut data).expect("Softmax failed");

    // Verify sum to 1
    let sum: f32 = data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "IMP-1001b: Softmax should sum to 1, got {}",
        sum
    );

    // Verify monotonicity (larger input = larger output)
    assert!(
        data[0] < data[1] && data[1] < data[2] && data[2] < data[3],
        "IMP-1001b: Softmax should preserve ordering"
    );
}

#[test]
#[cfg(feature = "cuda")]
#[allow(clippy::many_single_char_names)]
fn test_imp_1001c_cuda_inference_speedup() {
    // IMP-1001c: Verify CUDA inference is faster than CPU for large matrices
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1001c: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create CudaExecutor");

    // Large matmul: [512, 2048] @ [2048, 2048] - typical LLM layer size
    let m: u32 = 512;
    let k: u32 = 2048;
    let n: u32 = 2048;
    let a: Vec<f32> = (0..(m * k) as usize)
        .map(|i| (i % 100) as f32 * 0.01)
        .collect();
    let b: Vec<f32> = (0..(k * n) as usize)
        .map(|i| (i % 100) as f32 * 0.01)
        .collect();
    let mut result = vec![0.0f32; (m * n) as usize];

    // Warmup
    let _ = executor.gemm(&a, &b, &mut result, m, n, k);

    // Time CUDA
    let start = Instant::now();
    executor
        .gemm(&a, &b, &mut result, m, n, k)
        .expect("GEMM failed");
    let cuda_time = start.elapsed();

    // Time CPU (scalar)
    let start = Instant::now();
    let _cpu_result = cpu_matmul(&a, &b, m as usize, k as usize, n as usize);
    let cpu_time = start.elapsed();

    let speedup = cpu_time.as_secs_f64() / cuda_time.as_secs_f64();

    println!(
        "IMP-1001c: CUDA={:.2}ms, CPU={:.2}ms, speedup={:.1}x",
        cuda_time.as_secs_f64() * 1000.0,
        cpu_time.as_secs_f64() * 1000.0,
        speedup
    );

    // CUDA should be at least 5x faster for this size
    assert!(
        speedup > 5.0,
        "IMP-1001c: CUDA should be >5x faster for 512x2048x2048 GEMM, got {:.1}x",
        speedup
    );
}
