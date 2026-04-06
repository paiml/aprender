//! Benchmark suite for inference operations
//!
//! Measures end-to-end inference latency for model generation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use realizar::{
    generate::GenerationConfig,
    layers::{Model, ModelConfig},
};

fn create_test_model() -> Model {
    let config = ModelConfig {
        vocab_size: 100,
        hidden_dim: 32,
        num_heads: 1,
        num_layers: 2,
        intermediate_dim: 64,
        eps: 1e-5,
    };
    Model::new(config).expect("test")
}

fn benchmark_model_forward(c: &mut Criterion) {
    let model = create_test_model();
    let mut group = c.benchmark_group("model_forward");

    for seq_len in &[1, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            seq_len,
            |b, &seq_len| {
                let tokens: Vec<usize> = (0..seq_len).map(|i| i % 100).collect();
                b.iter(|| {
                    let logits = model.forward(black_box(&tokens)).expect("test");
                    black_box(logits)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_generation_greedy(c: &mut Criterion) {
    let model = create_test_model();
    let config = GenerationConfig::greedy().with_max_tokens(5);

    c.bench_function("generation_greedy_5_tokens", |b| {
        let prompt = vec![1, 5, 10];
        b.iter(|| {
            let result = model
                .generate(black_box(&prompt), black_box(&config))
                .expect("test");
            black_box(result)
        });
    });
}

fn benchmark_generation_top_k(c: &mut Criterion) {
    let model = create_test_model();
    let config = GenerationConfig::top_k(5).with_max_tokens(5).with_seed(42);

    c.bench_function("generation_top_k_5_tokens", |b| {
        let prompt = vec![1, 5, 10];
        b.iter(|| {
            let result = model
                .generate(black_box(&prompt), black_box(&config))
                .expect("test");
            black_box(result)
        });
    });
}

fn benchmark_generation_top_p(c: &mut Criterion) {
    let model = create_test_model();
    let config = GenerationConfig::top_p(0.9)
        .with_max_tokens(5)
        .with_seed(42);

    c.bench_function("generation_top_p_5_tokens", |b| {
        let prompt = vec![1, 5, 10];
        b.iter(|| {
            let result = model
                .generate(black_box(&prompt), black_box(&config))
                .expect("test");
            black_box(result)
        });
    });
}

fn benchmark_generation_varying_length(c: &mut Criterion) {
    let model = create_test_model();
    let mut group = c.benchmark_group("generation_varying_length");

    for max_tokens in &[1, 5, 10, 20] {
        let config = GenerationConfig::greedy().with_max_tokens(*max_tokens);
        group.bench_with_input(
            BenchmarkId::from_parameter(max_tokens),
            max_tokens,
            |b, _| {
                let prompt = vec![1, 5, 10];
                b.iter(|| {
                    let result = model
                        .generate(black_box(&prompt), black_box(&config))
                        .expect("test");
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_sampling_strategies(c: &mut Criterion) {
    let model = create_test_model();
    let mut group = c.benchmark_group("sampling_strategies");

    let strategies = vec![
        ("greedy", GenerationConfig::greedy()),
        ("top_k_5", GenerationConfig::top_k(5).with_seed(42)),
        ("top_k_50", GenerationConfig::top_k(50).with_seed(42)),
        ("top_p_0.9", GenerationConfig::top_p(0.9).with_seed(42)),
        ("top_p_0.95", GenerationConfig::top_p(0.95).with_seed(42)),
    ];

    for (name, mut config) in strategies {
        config = config.with_max_tokens(10);
        group.bench_with_input(BenchmarkId::from_parameter(name), &config, |b, config| {
            let prompt = vec![1, 5, 10];
            b.iter(|| {
                let result = model
                    .generate(black_box(&prompt), black_box(config))
                    .expect("test");
                black_box(result)
            });
        });
    }

    group.finish();
}

fn benchmark_batch_generation(c: &mut Criterion) {
    let model = create_test_model();
    let config = GenerationConfig::greedy().with_max_tokens(5);
    let mut group = c.benchmark_group("batch_generation");

    // Benchmark different batch sizes
    for batch_size in &[1, 2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                // Create batch of prompts
                let prompts: Vec<Vec<usize>> = (0..batch_size)
                    .map(|i| vec![1 + i % 10, 5 + i % 10, 10 + i % 10])
                    .collect();

                b.iter(|| {
                    // Process batch sequentially (Phase 2 baseline)
                    let results: Vec<_> = prompts
                        .iter()
                        .map(|prompt| {
                            model
                                .generate(black_box(prompt), black_box(&config))
                                .expect("test")
                        })
                        .collect();
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_batch_vs_single(c: &mut Criterion) {
    let model = create_test_model();
    let config = GenerationConfig::greedy().with_max_tokens(5);
    let mut group = c.benchmark_group("batch_vs_single");

    let batch_size = 4;
    let prompts: Vec<Vec<usize>> = (0..batch_size)
        .map(|i| vec![1 + i % 10, 5 + i % 10, 10 + i % 10])
        .collect();

    // Single sequential processing
    group.bench_function("sequential_4_prompts", |b| {
        b.iter(|| {
            let results: Vec<_> = prompts
                .iter()
                .map(|prompt| {
                    model
                        .generate(black_box(prompt), black_box(&config))
                        .expect("test")
                })
                .collect();
            black_box(results)
        });
    });

    // Batch processing (currently same as sequential, but prepared for future optimization)
    group.bench_function("batch_4_prompts", |b| {
        b.iter(|| {
            let results: Vec<_> = prompts
                .iter()
                .map(|prompt| {
                    model
                        .generate(black_box(prompt), black_box(&config))
                        .expect("test")
                })
                .collect();
            black_box(results)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_model_forward,
    benchmark_generation_greedy,
    benchmark_generation_top_k,
    benchmark_generation_top_p,
    benchmark_generation_varying_length,
    benchmark_sampling_strategies,
    benchmark_batch_generation,
    benchmark_batch_vs_single,
);
criterion_main!(benches);
