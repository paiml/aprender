//! Benchmark suite for model cache operations
//!
//! Measures cache performance including:
//! - Cache hit latency
//! - Cache miss + load latency
//! - LRU eviction overhead
//! - Concurrent access throughput

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use realizar::{
    cache::{CacheKey, ModelCache},
    error::RealizarError,
    layers::{Model, ModelConfig},
    tokenizer::BPETokenizer,
};

type ModelResult = Result<(Model, BPETokenizer), RealizarError>;

fn create_test_model(vocab_size: usize) -> ModelResult {
    let config = ModelConfig {
        vocab_size,
        hidden_dim: 32,
        num_heads: 1,
        num_layers: 2,
        intermediate_dim: 64,
        eps: 1e-5,
    };
    let model = Model::new(config)?;

    let vocab: Vec<String> = (0..vocab_size)
        .map(|i| {
            if i == 0 {
                "<unk>".to_string()
            } else {
                format!("token{i}")
            }
        })
        .collect();
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;

    Ok((model, tokenizer))
}

fn benchmark_cache_hit(c: &mut Criterion) {
    let cache = ModelCache::new(10);
    let key = CacheKey::new("test".to_string());

    // Pre-populate cache
    cache
        .get_or_load(&key, || create_test_model(100))
        .expect("test");

    c.bench_function("cache_hit", |b| {
        b.iter(|| {
            let result = cache.get_or_load(black_box(&key), || create_test_model(100));
            black_box(result)
        });
    });
}

fn benchmark_cache_miss(c: &mut Criterion) {
    c.bench_function("cache_miss_with_load", |b| {
        let cache = ModelCache::new(10);
        let mut counter = 0;

        b.iter(|| {
            let key = CacheKey::new(format!("model{counter}"));
            let result = cache.get_or_load(black_box(&key), || create_test_model(100));
            counter += 1;
            black_box(result)
        });
    });
}

fn benchmark_cache_eviction(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_eviction");

    for capacity in &[2, 5, 10] {
        group.bench_with_input(
            BenchmarkId::from_parameter(capacity),
            capacity,
            |b, &capacity| {
                let cache = ModelCache::new(capacity);

                b.iter(|| {
                    // Fill cache beyond capacity to trigger eviction
                    for i in 0..capacity + 5 {
                        let key = CacheKey::new(format!("model{i}"));
                        let result = cache
                            .get_or_load(black_box(&key), || create_test_model(100))
                            .expect("test");
                        black_box(result);
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_cache_concurrent(c: &mut Criterion) {
    use std::{sync::Arc, thread};

    c.bench_function("cache_concurrent_access", |b| {
        let cache = Arc::new(ModelCache::new(10));
        let key = CacheKey::new("shared".to_string());

        // Pre-populate
        cache
            .get_or_load(&key, || create_test_model(100))
            .expect("test");

        b.iter(|| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let cache = cache.clone();
                    let key = key.clone();
                    thread::spawn(move || {
                        cache
                            .get_or_load(black_box(&key), || create_test_model(100))
                            .expect("test")
                    })
                })
                .collect();

            for handle in handles {
                black_box(handle.join().expect("test"));
            }
        });
    });
}

fn benchmark_cache_metrics(c: &mut Criterion) {
    let cache = ModelCache::new(10);
    let key = CacheKey::new("test".to_string());

    // Create some activity
    for _ in 0..10 {
        cache
            .get_or_load(&key, || create_test_model(100))
            .expect("test");
    }

    c.bench_function("cache_metrics_access", |b| {
        b.iter(|| {
            let metrics = cache.metrics();
            black_box(metrics)
        });
    });
}

fn benchmark_cache_key_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_key");

    group.bench_function("from_string", |b| {
        b.iter(|| {
            let key = CacheKey::new(black_box("model_name".to_string()));
            black_box(key)
        });
    });

    group.bench_function("from_config", |b| {
        let config = ModelConfig {
            vocab_size: 100,
            hidden_dim: 32,
            num_heads: 2,
            num_layers: 4,
            intermediate_dim: 64,
            eps: 1e-5,
        };

        b.iter(|| {
            let key = CacheKey::from_config(black_box(&config));
            black_box(key)
        });
    });

    group.finish();
}

fn benchmark_cache_varying_capacity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_capacity");

    for capacity in &[1, 5, 10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(capacity),
            capacity,
            |b, &capacity| {
                b.iter(|| {
                    let cache = ModelCache::new(black_box(capacity));

                    // Fill to half capacity
                    for i in 0..capacity / 2 {
                        let key = CacheKey::new(format!("model{i}"));
                        cache
                            .get_or_load(&key, || create_test_model(50))
                            .expect("test");
                    }

                    black_box(&cache);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_hit_rate_calculation(c: &mut Criterion) {
    use realizar::cache::CacheMetrics;

    c.bench_function("hit_rate_calculation", |b| {
        let metrics = CacheMetrics {
            hits: 850,
            misses: 150,
            ..Default::default()
        };

        b.iter(|| {
            let rate = metrics.hit_rate();
            black_box(rate)
        });
    });
}

criterion_group!(
    benches,
    benchmark_cache_hit,
    benchmark_cache_miss,
    benchmark_cache_eviction,
    benchmark_cache_concurrent,
    benchmark_cache_metrics,
    benchmark_cache_key_creation,
    benchmark_cache_varying_capacity,
    benchmark_hit_rate_calculation,
);
criterion_main!(benches);
