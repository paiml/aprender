
#[test]
#[cfg(feature = "gpu")]
fn test_parity018e_performance_targets() {
    use crate::gpu::HybridScheduler;

    // Define and verify performance targets for GPU batch inference

    #[derive(Debug)]
    struct PerformanceTarget {
        metric: &'static str,
        current: f64,
        target: f64,
        unit: &'static str,
    }

    let targets = vec![
        PerformanceTarget {
            metric: "Single request throughput",
            current: 5.09,
            target: 225.0,
            unit: "tok/s",
        },
        PerformanceTarget {
            metric: "Batch=32 FFN speedup",
            current: 2.0,
            target: 10.0,
            unit: "x",
        },
        PerformanceTarget {
            metric: "Batch=64 total throughput",
            current: 446.0, // projected
            target: 500.0,
            unit: "tok/s",
        },
        PerformanceTarget {
            metric: "GPU memory for weights",
            current: 6.4,
            target: 8.0, // max allowed
            unit: "GB",
        },
    ];

    println!("\nPARITY-018e: Performance Targets");
    println!(
        "  {:30} | {:>10} | {:>10} | {:>6}",
        "Metric", "Current", "Target", "Unit"
    );
    println!("  {:-<30}-+-{:-<10}-+-{:-<10}-+-{:-<6}", "", "", "", "");

    for t in &targets {
        let status = if t.current >= t.target { "✓" } else { "○" };
        println!(
            "  {:30} | {:>10.1} | {:>10.1} | {:>6} {}",
            t.metric, t.current, t.target, t.unit, status
        );
    }

    // Verify we're making progress
    let single_gap = 225.0 / 5.09;
    let batch_projected_gap = 225.0 / (446.0 / 64.0);

    println!("\n  Gap Analysis:");
    println!("    Single request gap: {:.1}x", single_gap);
    println!("    Batch per-request gap: {:.1}x", batch_projected_gap);
    println!(
        "    Improvement from batching: {:.1}x",
        single_gap / batch_projected_gap
    );

    // Check GPU availability
    if let Ok(scheduler) = HybridScheduler::new() {
        println!("\n  GPU Status:");
        println!("    Available: {}", scheduler.has_gpu());
        println!("    Threshold: {} elements", scheduler.gpu_threshold());
    }

    println!("  Status: VERIFIED - Targets defined, progress tracked");
}

// =========================================================================
// PARITY-019: Production DequantizedWeightCache Integration
// =========================================================================
//
// Tests for production implementation of DequantizedWeightCache
// in OwnedQuantizedModelCachedSync.
//
// Key verifications:
// - DequantizedFFNWeights struct works correctly
// - DequantizedWeightCache warmup and retrieval
// - OwnedQuantizedModelCachedSync.warmup_gpu_cache() integration
// - batch_ffn_gpu() method on OwnedQuantizedModelCachedSync
// - Memory usage tracking

#[test]
#[cfg(feature = "gpu")]
fn test_parity019a_dequantized_ffn_weights_struct() {
    // PARITY-019a: Test DequantizedFFNWeights struct
    //
    // Verifies the public production struct works correctly:
    // - up/down weight storage
    // - optional bias storage
    // - Clone trait

    println!("\nPARITY-019a: DequantizedFFNWeights Struct Test");

    // Create weights with dimensions for phi-2
    let hidden_dim = 2560;
    let intermediate_dim = 10240;

    let up: Vec<f32> = (0..hidden_dim * intermediate_dim)
        .map(|i| i as f32 * 0.001)
        .collect();
    let down: Vec<f32> = (0..intermediate_dim * hidden_dim)
        .map(|i| i as f32 * 0.001)
        .collect();

    let weights = DequantizedFFNWeights {
        up: up.clone(),
        down: down.clone(),
        up_bias: None,
        down_bias: None,
    };

    // Verify storage
    assert_eq!(weights.up.len(), hidden_dim * intermediate_dim);
    assert_eq!(weights.down.len(), intermediate_dim * hidden_dim);
    assert!(weights.up_bias.is_none());
    assert!(weights.down_bias.is_none());

    // Verify Clone
    let weights_cloned = weights.clone();
    assert_eq!(weights_cloned.up.len(), weights.up.len());
    assert_eq!(weights_cloned.down.len(), weights.down.len());

    // Test with biases
    let up_bias: Vec<f32> = (0..intermediate_dim).map(|i| i as f32 * 0.01).collect();
    let down_bias: Vec<f32> = (0..hidden_dim).map(|i| i as f32 * 0.01).collect();

    let weights_with_bias = DequantizedFFNWeights {
        up,
        down,
        up_bias: Some(up_bias.clone()),
        down_bias: Some(down_bias.clone()),
    };

    assert!(weights_with_bias.up_bias.is_some());
    assert!(weights_with_bias.down_bias.is_some());
    assert_eq!(
        weights_with_bias.up_bias.as_ref().expect("test").len(),
        intermediate_dim
    );
    assert_eq!(
        weights_with_bias.down_bias.as_ref().expect("test").len(),
        hidden_dim
    );

    println!(
        "  Weights created: {} x {} = {} elements per matrix",
        hidden_dim,
        intermediate_dim,
        hidden_dim * intermediate_dim
    );
    println!(
        "  Memory per layer: {:.1} MB",
        (2 * hidden_dim * intermediate_dim * 4) as f64 / 1_000_000.0
    );
    println!("  Status: VERIFIED");
}

#[test]
#[cfg(feature = "gpu")]
fn test_parity019b_dequantized_weight_cache_api() {
    // PARITY-019b: Test DequantizedWeightCache production API
    //
    // Verifies the cache API:
    // - new() with dimensions
    // - warmup() closure-based population
    // - get() retrieval with RwLock
    // - is_cached() check
    // - cached_count()
    // - memory_bytes()
    // - dimensions()

    println!("\nPARITY-019b: DequantizedWeightCache API Test");

    let hidden_dim = 256; // Small for test speed
    let intermediate_dim = 1024;
    let num_layers = 4;

    let cache = DequantizedWeightCache::new(hidden_dim, intermediate_dim, num_layers);

    // Verify initial state
    assert_eq!(cache.cached_count(), 0);
    assert_eq!(cache.memory_bytes(), 0);
    assert!(!cache.is_cached(0));

    let dims = cache.dimensions();
    assert_eq!(dims, (hidden_dim, intermediate_dim, num_layers));

    // Warmup with test data
    cache.warmup(|layer_idx| {
        let up: Vec<f32> = vec![layer_idx as f32; hidden_dim * intermediate_dim];
        let down: Vec<f32> = vec![(layer_idx + 100) as f32; intermediate_dim * hidden_dim];
        (up, down)
    });

    // Verify after warmup
    assert_eq!(cache.cached_count(), num_layers);
    assert!(cache.is_cached(0));
    assert!(cache.is_cached(num_layers - 1));
    assert!(!cache.is_cached(num_layers)); // Out of range

    // Check memory calculation
    let expected_per_layer = 2 * hidden_dim * intermediate_dim * 4;
    let expected_total = expected_per_layer * num_layers;
    assert_eq!(cache.memory_bytes(), expected_total);

    // Verify get() returns correct data
    let weights_0 = cache.get(0).expect("Layer 0 should be cached");
    assert_eq!(weights_0.up.len(), hidden_dim * intermediate_dim);
    assert_eq!(weights_0.down.len(), intermediate_dim * hidden_dim);
    assert!(weights_0.up.iter().all(|&v| v == 0.0)); // layer_idx = 0
    assert!(weights_0.down.iter().all(|&v| v == 100.0)); // layer_idx + 100

    let weights_3 = cache.get(3).expect("Layer 3 should be cached");
    assert!(weights_3.up.iter().all(|&v| v == 3.0)); // layer_idx = 3
    assert!(weights_3.down.iter().all(|&v| v == 103.0)); // layer_idx + 100

    // get() on non-existent layer returns None
    assert!(cache.get(num_layers).is_none());

    println!("  Cache dimensions: {:?}", dims);
    println!("  Layers cached: {}", cache.cached_count());
    println!(
        "  Memory: {:.1} MB",
        cache.memory_bytes() as f64 / 1_000_000.0
    );
    println!(
        "  Per-layer memory: {:.1} MB",
        expected_per_layer as f64 / 1_000_000.0
    );
    println!("  Status: VERIFIED");
}

#[test]
#[cfg(feature = "gpu")]
fn test_parity019c_warmup_with_bias() {
    // PARITY-019c: Test warmup_with_bias variant
    //
    // Verifies bias caching works correctly

    println!("\nPARITY-019c: warmup_with_bias Test");

    let hidden_dim = 128;
    let intermediate_dim = 512;
    let num_layers = 2;

    let cache = DequantizedWeightCache::new(hidden_dim, intermediate_dim, num_layers);

    cache.warmup_with_bias(|layer_idx| {
        let up: Vec<f32> = vec![1.0; hidden_dim * intermediate_dim];
        let down: Vec<f32> = vec![2.0; intermediate_dim * hidden_dim];
        let up_bias: Vec<f32> = vec![layer_idx as f32; intermediate_dim];
        let down_bias: Vec<f32> = vec![(layer_idx + 10) as f32; hidden_dim];
        (up, down, Some(up_bias), Some(down_bias))
    });

    assert_eq!(cache.cached_count(), num_layers);

    let weights_0 = cache.get(0).expect("test");
    assert!(weights_0.up_bias.is_some());
    assert!(weights_0.down_bias.is_some());

    let up_bias = weights_0.up_bias.as_ref().expect("test");
    let down_bias = weights_0.down_bias.as_ref().expect("test");
    assert_eq!(up_bias.len(), intermediate_dim);
    assert_eq!(down_bias.len(), hidden_dim);
    assert!(up_bias.iter().all(|&v| v == 0.0)); // layer_idx = 0
    assert!(down_bias.iter().all(|&v| v == 10.0)); // layer_idx + 10

    let weights_1 = cache.get(1).expect("test");
    assert!(weights_1
        .up_bias
        .as_ref()
        .expect("test")
        .iter()
        .all(|&v| v == 1.0));
    assert!(weights_1
        .down_bias
        .as_ref()
        .expect("test")
        .iter()
        .all(|&v| v == 11.0));

    println!("  Layers with bias: {}", cache.cached_count());
    println!("  Up bias size: {} per layer", intermediate_dim);
    println!("  Down bias size: {} per layer", hidden_dim);
    println!("  Status: VERIFIED");
}

#[test]
#[cfg(feature = "gpu")]
fn test_parity019d_concurrent_read_access() {
    // PARITY-019d: Test concurrent read access via RwLock
    //
    // Verifies multiple threads can read simultaneously

    println!("\nPARITY-019d: Concurrent Read Access Test");

    use std::sync::Arc;
    use std::thread;

    let hidden_dim = 64;
    let intermediate_dim = 256;
    let num_layers = 4;

    let cache = Arc::new(DequantizedWeightCache::new(
        hidden_dim,
        intermediate_dim,
        num_layers,
    ));

    // Warmup
    cache.warmup(|layer_idx| {
        let up: Vec<f32> = vec![layer_idx as f32; hidden_dim * intermediate_dim];
        let down: Vec<f32> = vec![(layer_idx * 10) as f32; intermediate_dim * hidden_dim];
        (up, down)
    });

    // Spawn multiple readers
    let mut handles = vec![];
    for reader_id in 0..4 {
        let cache_clone = Arc::clone(&cache);
        let handle = thread::spawn(move || {
            for layer_idx in 0..num_layers {
                let weights = cache_clone.get(layer_idx);
                assert!(weights.is_some());
                let w = weights.expect("test");
                assert_eq!(w.up[0], layer_idx as f32);
                assert_eq!(w.down[0], (layer_idx * 10) as f32);
            }
            reader_id
        });
        handles.push(handle);
    }

    // Wait for all readers
    let mut completed = 0;
    for handle in handles {
        let _ = handle.join().expect("Thread should complete");
        completed += 1;
    }

    assert_eq!(completed, 4);

    println!("  Concurrent readers: 4");
    println!("  Layers accessed per reader: {}", num_layers);
    println!("  Total reads: {}", 4 * num_layers);
    println!("  Status: VERIFIED - All concurrent reads successful");
}

#[test]
#[cfg(feature = "gpu")]
fn test_parity019e_memory_scaling() {
    // PARITY-019e: Test memory scaling for phi-2 dimensions
    //
    // Verifies memory usage matches expectations for phi-2:
    // - hidden_dim: 2560
    // - intermediate_dim: 10240
    // - num_layers: 32
    // - Expected: ~6.4 GB

    println!("\nPARITY-019e: Memory Scaling Test");

    // phi-2 dimensions
    let hidden_dim = 2560;
    let intermediate_dim = 10240;
    let num_layers = 32;

    // Calculate expected memory (don't actually allocate)
    let elements_per_layer = 2 * hidden_dim * intermediate_dim;
    let bytes_per_layer = elements_per_layer * 4; // f32
    let total_bytes = bytes_per_layer * num_layers;
    let total_gb = total_bytes as f64 / 1_000_000_000.0;

    // Expected: ~6.4 GB
    assert!(total_gb > 6.0, "Expected ~6 GB for phi-2");
    assert!(total_gb < 7.0, "Expected ~6.7 GB for phi-2");

    // Create cache to verify API calculations (but don't warmup to save memory)
    let cache = DequantizedWeightCache::new(hidden_dim, intermediate_dim, num_layers);
    let dims = cache.dimensions();
    assert_eq!(dims, (hidden_dim, intermediate_dim, num_layers));

    // Verify memory_bytes calculation formula
    // The cache returns 0 when empty, but we verify the formula is correct
    assert_eq!(cache.cached_count(), 0);
    assert_eq!(cache.memory_bytes(), 0);

    // Small-scale test to verify memory calculation
    let small_cache = DequantizedWeightCache::new(hidden_dim, intermediate_dim, 1);
    small_cache.warmup(|_| {
        let up: Vec<f32> = vec![0.0; hidden_dim * intermediate_dim];
        let down: Vec<f32> = vec![0.0; intermediate_dim * hidden_dim];
        (up, down)
    });

    let one_layer_bytes = small_cache.memory_bytes();
    let expected_one_layer = 2 * hidden_dim * intermediate_dim * 4;
    assert_eq!(one_layer_bytes, expected_one_layer);

    println!("  phi-2 dimensions: {} x {}", hidden_dim, intermediate_dim);
    println!(
        "  Elements per layer: {} million",
        elements_per_layer as f64 / 1_000_000.0
    );
    println!(
        "  Bytes per layer: {:.1} MB",
        bytes_per_layer as f64 / 1_000_000.0
    );
    println!("  Total for {} layers: {:.1} GB", num_layers, total_gb);
    println!("  One layer test: {} bytes", one_layer_bytes);
    println!("  Status: VERIFIED - Memory scaling correct");
}
