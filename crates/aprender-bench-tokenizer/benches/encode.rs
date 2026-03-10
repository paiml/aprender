//! Encode benchmarks: aprender vs HuggingFace tokenizers v0.22.
//!
//! Measures encode throughput across 4 canonical payloads and 6 input-length
//! scaling points with head-to-head comparison.
//!
//! ## Performance targets (GH-378)
//!
//! | Payload | aprender Target | HF v0.22 Ref |
//! |---------|-----------------|--------------|
//! | 636-char code | < 80us | ~104us |
//! | 2000-char HumanEval | < 200us | ~280us |
//!
//! Target: beat HF v0.22 by >= 1.3x on all encode paths.

use aprender_bench_tokenizer::payloads;
use aprender_bench_tokenizer::{load_aprender_tokenizer, load_hf_tokenizer, skip_no_tokenizer};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_encode_aprender(c: &mut Criterion) {
    let Some(tokenizer) = load_aprender_tokenizer() else {
        skip_no_tokenizer!("encode_aprender");
        return;
    };

    let mut group = c.benchmark_group("encode_aprender");

    let payloads: &[(&str, &str)] = &[
        ("short_nl_30c", payloads::SHORT_NL),
        ("medium_code_200c", payloads::MEDIUM_CODE),
        ("long_code_636c", payloads::LONG_CODE_MERGE_SORT),
        ("vlong_humaneval_2000c", payloads::VERY_LONG_HUMANEVAL),
    ];

    for (name, payload) in payloads {
        let tokens = tokenizer.encode(payload);
        group.throughput(Throughput::Elements(tokens.len() as u64));

        group.bench_with_input(BenchmarkId::new("payload", name), payload, |b, input| {
            b.iter(|| tokenizer.encode(black_box(input)));
        });
    }

    group.finish();
}

fn bench_encode_hf(c: &mut Criterion) {
    let Some(tokenizer) = load_hf_tokenizer() else {
        skip_no_tokenizer!("encode_hf");
        return;
    };

    let mut group = c.benchmark_group("encode_hf");

    let payloads: &[(&str, &str)] = &[
        ("short_nl_30c", payloads::SHORT_NL),
        ("medium_code_200c", payloads::MEDIUM_CODE),
        ("long_code_636c", payloads::LONG_CODE_MERGE_SORT),
        ("vlong_humaneval_2000c", payloads::VERY_LONG_HUMANEVAL),
    ];

    for (name, payload) in payloads {
        let encoding = tokenizer.encode(*payload, false).unwrap();
        group.throughput(Throughput::Elements(encoding.get_ids().len() as u64));

        group.bench_with_input(BenchmarkId::new("payload", name), payload, |b, input| {
            b.iter(|| tokenizer.encode(black_box(*input), false).unwrap());
        });
    }

    group.finish();
}

fn bench_encode_scaling(c: &mut Criterion) {
    let apr_tok = load_aprender_tokenizer();
    let hf_tok = load_hf_tokenizer();

    if apr_tok.is_none() && hf_tok.is_none() {
        skip_no_tokenizer!("encode_scaling");
        return;
    }

    let mut group = c.benchmark_group("encode_scaling");

    for &char_len in &[10, 50, 200, 500, 1000, 2000] {
        let text = payloads::synthetic_text(char_len);

        if let Some(ref tok) = apr_tok {
            let tokens = tok.encode(&text);
            group.throughput(Throughput::Elements(tokens.len() as u64));
            group.bench_with_input(BenchmarkId::new("aprender", char_len), &text, |b, input| {
                b.iter(|| tok.encode(black_box(input)));
            });
        }

        if let Some(ref tok) = hf_tok {
            let encoding = tok.encode(text.as_str(), false).unwrap();
            group.throughput(Throughput::Elements(encoding.get_ids().len() as u64));
            group.bench_with_input(BenchmarkId::new("hf", char_len), &text, |b, input| {
                b.iter(|| tok.encode(black_box(input.as_str()), false).unwrap());
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_encode_aprender,
    bench_encode_hf,
    bench_encode_scaling,
);
criterion_main!(benches);
