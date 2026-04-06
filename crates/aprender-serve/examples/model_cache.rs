//! Model caching example for realizar
//!
//! Demonstrates how to use ModelCache for efficient model reuse:
//! - Cache creation with capacity limits
//! - Model loading with cache hits/misses
//! - Metrics tracking (hit rate, evictions)
//! - LRU eviction behavior
//!
//! Run with: cargo run --example model_cache

use realizar::{
    cache::{CacheKey, ModelCache},
    layers::{Model, ModelConfig},
    tokenizer::BPETokenizer,
};

fn create_demo_model(vocab_size: usize) -> (Model, BPETokenizer) {
    let config = ModelConfig {
        vocab_size,
        hidden_dim: 16,
        num_heads: 1,
        num_layers: 1,
        intermediate_dim: 32,
        eps: 1e-5,
    };

    let model = Model::new(config).expect("Failed to create model");

    let vocab: Vec<String> = (0..vocab_size)
        .map(|i| {
            if i == 0 {
                "<unk>".to_string()
            } else {
                format!("token{i}")
            }
        })
        .collect();
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("Failed to create tokenizer");

    (model, tokenizer)
}

fn main() {
    println!("=== Model Cache Example ===\n");

    // Create a cache with capacity for 3 models
    let cache = ModelCache::new(3);
    println!("Created cache with capacity: 3\n");

    // Example 1: First access (cache miss)
    println!("--- Example 1: First Access (Cache Miss) ---");
    let key1 = CacheKey::new("model_50".to_string());
    let result = cache.get_or_load(&key1, || {
        println!("  Loading model_50 (vocab=50)...");
        Ok(create_demo_model(50))
    });
    assert!(result.is_ok());

    let metrics = cache.metrics();
    println!("  ✓ Model loaded");
    println!("  Cache metrics:");
    println!("    Hits: {}", metrics.hits);
    println!("    Misses: {}", metrics.misses);
    println!("    Hit rate: {:.1}%", metrics.hit_rate());
    println!("    Cache size: {}\n", metrics.size);

    // Example 2: Second access (cache hit)
    println!("--- Example 2: Second Access (Cache Hit) ---");
    let result = cache.get_or_load(&key1, || {
        println!("  This should not print - model already cached!");
        Ok(create_demo_model(50))
    });
    assert!(result.is_ok());

    let metrics = cache.metrics();
    println!("  ✓ Model retrieved from cache");
    println!("  Cache metrics:");
    println!("    Hits: {}", metrics.hits);
    println!("    Misses: {}", metrics.misses);
    println!("    Hit rate: {:.1}%", metrics.hit_rate());
    println!("    Cache size: {}\n", metrics.size);

    // Example 3: Loading multiple models
    println!("--- Example 3: Loading Multiple Models ---");
    let key2 = CacheKey::new("model_100".to_string());
    let key3 = CacheKey::new("model_150".to_string());

    cache
        .get_or_load(&key2, || {
            println!("  Loading model_100 (vocab=100)...");
            Ok(create_demo_model(100))
        })
        .expect("Failed to load model_100");

    cache
        .get_or_load(&key3, || {
            println!("  Loading model_150 (vocab=150)...");
            Ok(create_demo_model(150))
        })
        .expect("Failed to load model_150");

    let metrics = cache.metrics();
    println!("  ✓ Loaded 2 more models");
    println!("  Cache metrics:");
    println!("    Total models cached: {}", metrics.size);
    println!("    Hit rate: {:.1}%\n", metrics.hit_rate());

    // Example 4: LRU eviction (cache at capacity)
    println!("--- Example 4: LRU Eviction ---");
    println!("  Cache capacity: 3 (full)");
    println!("  Loading 4th model will evict least recently used...");

    let key4 = CacheKey::new("model_200".to_string());
    cache
        .get_or_load(&key4, || {
            println!("  Loading model_200 (vocab=200)...");
            Ok(create_demo_model(200))
        })
        .expect("Failed to load model_200");

    let metrics = cache.metrics();
    println!("  ✓ LRU model evicted");
    println!("  Cache metrics:");
    println!("    Evictions: {}", metrics.evictions);
    println!("    Current size: {}", metrics.size);
    println!("    Hit rate: {:.1}%\n", metrics.hit_rate());

    // Example 5: Cache key from config
    println!("--- Example 5: Cache Key from Config ---");
    let config = ModelConfig {
        vocab_size: 300,
        hidden_dim: 32,
        num_heads: 2,
        num_layers: 4,
        intermediate_dim: 64,
        eps: 1e-5,
    };
    let key_from_config = CacheKey::from_config(&config);
    println!(
        "  Config: vocab={}, hidden={}, heads={}, layers={}, intermediate={}",
        config.vocab_size,
        config.hidden_dim,
        config.num_heads,
        config.num_layers,
        config.intermediate_dim
    );
    println!("  Generated key: '{}'", key_from_config.id);

    cache
        .get_or_load(&key_from_config, || Ok(create_demo_model(300)))
        .expect("Failed to load model from config");

    let final_metrics = cache.metrics();
    println!("  ✓ Model loaded with config-based key\n");

    // Final summary
    println!("--- Final Summary ---");
    println!(
        "  Total accesses: {}",
        final_metrics.hits + final_metrics.misses
    );
    println!("  Cache hits: {}", final_metrics.hits);
    println!("  Cache misses: {}", final_metrics.misses);
    println!("  Evictions: {}", final_metrics.evictions);
    println!("  Final cache size: {}", final_metrics.size);
    println!("  Overall hit rate: {:.1}%", final_metrics.hit_rate());
    println!();

    println!("=== Cache Example Complete ===");
}
