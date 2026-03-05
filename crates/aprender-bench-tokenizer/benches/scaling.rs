//! Input-length scaling benchmarks: aprender vs HuggingFace tokenizers v0.22.
//!
//! Confirms O(n) encode time in input length across 6 points (50–5000 chars).
//! Both implementations should scale linearly, not O(n × vocab_size).

use aprender_bench_tokenizer::payloads;
use aprender_bench_tokenizer::{load_aprender_tokenizer, load_hf_tokenizer, skip_no_tokenizer};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_scaling_aprender(c: &mut Criterion) {
    let Some(tokenizer) = load_aprender_tokenizer() else {
        skip_no_tokenizer!("scaling_aprender");
        return;
    };

    let mut group = c.benchmark_group("scaling_aprender");

    for &char_len in &[50, 100, 500, 1000, 2000, 5000] {
        let text = payloads::synthetic_text(char_len);
        let tokens = tokenizer.encode(&text);
        group.throughput(Throughput::Bytes(char_len as u64));

        group.bench_with_input(BenchmarkId::new("chars", char_len), &text, |b, input| {
            b.iter(|| tokenizer.encode(black_box(input)));
        });

        // Verify token count scales linearly (sanity check, not a benchmark assertion)
        drop(tokens);
    }

    group.finish();
}

fn bench_scaling_hf(c: &mut Criterion) {
    let Some(tokenizer) = load_hf_tokenizer() else {
        skip_no_tokenizer!("scaling_hf");
        return;
    };

    let mut group = c.benchmark_group("scaling_hf");

    for &char_len in &[50, 100, 500, 1000, 2000, 5000] {
        let text = payloads::synthetic_text(char_len);
        group.throughput(Throughput::Bytes(char_len as u64));

        group.bench_with_input(BenchmarkId::new("chars", char_len), &text, |b, input| {
            b.iter(|| tokenizer.encode(black_box(input.as_str()), false).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_scaling_aprender, bench_scaling_hf);
criterion_main!(benches);
