//! Decode benchmarks: aprender vs HuggingFace tokenizers v0.22.
//!
//! Measures decode throughput at 4 token counts (10/50/200/500).
//!
//! ## Performance targets
//!
//! | Token count | aprender Target | HF v0.22 Ref |
//! |-------------|-----------------|--------------|
//! | 200 tokens | < 30us | ~40us |

use aprender_bench_tokenizer::payloads;
use aprender_bench_tokenizer::{load_aprender_tokenizer, load_hf_tokenizer, skip_no_tokenizer};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_decode_aprender(c: &mut Criterion) {
    let Some(tokenizer) = load_aprender_tokenizer() else {
        skip_no_tokenizer!("decode_aprender");
        return;
    };

    let mut group = c.benchmark_group("decode_aprender");

    // Encode a long payload to get token IDs, then slice to various lengths
    let all_ids = tokenizer.encode(payloads::VERY_LONG_HUMANEVAL);

    for &n_tokens in &[10, 50, 200, 500] {
        let ids: Vec<u32> = all_ids.iter().copied().cycle().take(n_tokens).collect();
        group.throughput(Throughput::Elements(n_tokens as u64));

        group.bench_with_input(BenchmarkId::new("tokens", n_tokens), &ids, |b, input| {
            b.iter(|| tokenizer.decode(black_box(input)));
        });
    }

    group.finish();
}

fn bench_decode_hf(c: &mut Criterion) {
    let Some(tokenizer) = load_hf_tokenizer() else {
        skip_no_tokenizer!("decode_hf");
        return;
    };

    let mut group = c.benchmark_group("decode_hf");

    let encoding = tokenizer
        .encode(payloads::VERY_LONG_HUMANEVAL, false)
        .unwrap();
    let all_ids = encoding.get_ids();

    for &n_tokens in &[10, 50, 200, 500] {
        let ids: Vec<u32> = all_ids.iter().copied().cycle().take(n_tokens).collect();
        group.throughput(Throughput::Elements(n_tokens as u64));

        group.bench_with_input(BenchmarkId::new("tokens", n_tokens), &ids, |b, input| {
            b.iter(|| tokenizer.decode(black_box(input), false).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_decode_aprender, bench_decode_hf);
criterion_main!(benches);
