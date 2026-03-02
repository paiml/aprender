//! Batch encode benchmarks: aprender vs HuggingFace tokenizers v0.22.
//!
//! Simulates training pre-tokenization (entrenar path) with bulk encode
//! of 10/100/1000 prompts of varying lengths (50–500 chars).
//!
//! ## Performance targets
//!
//! | Batch size | aprender Target | HF v0.22 Ref |
//! |------------|-----------------|--------------|
//! | 100 prompts | < 8ms | ~10ms |

use aprender_bench_tokenizer::payloads;
use aprender_bench_tokenizer::{load_aprender_tokenizer, load_hf_tokenizer, skip_no_tokenizer};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_batch_aprender(c: &mut Criterion) {
    let Some(tokenizer) = load_aprender_tokenizer() else {
        skip_no_tokenizer!("batch_aprender");
        return;
    };

    let mut group = c.benchmark_group("batch_aprender");

    for &count in &[10, 100, 1000] {
        let batch = payloads::generate_batch(count);
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::new("prompts", count),
            &batch,
            |b, prompts| {
                b.iter(|| {
                    for prompt in black_box(prompts) {
                        let _ = tokenizer.encode(prompt);
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_batch_hf(c: &mut Criterion) {
    let Some(tokenizer) = load_hf_tokenizer() else {
        skip_no_tokenizer!("batch_hf");
        return;
    };

    let mut group = c.benchmark_group("batch_hf");

    for &count in &[10, 100, 1000] {
        let batch = payloads::generate_batch(count);
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::new("prompts", count),
            &batch,
            |b, prompts| {
                b.iter(|| {
                    for prompt in black_box(prompts) {
                        tokenizer.encode(prompt.as_str(), false).unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_batch_aprender, bench_batch_hf);
criterion_main!(benches);
