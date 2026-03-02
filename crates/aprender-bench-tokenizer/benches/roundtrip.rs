//! Roundtrip (encode → decode) benchmarks: aprender vs HuggingFace tokenizers v0.22.
//!
//! ## Performance targets
//!
//! | Payload | aprender Target | HF v0.22 Ref |
//! |---------|-----------------|--------------|
//! | 636-char code | < 110us | ~145us |

use aprender_bench_tokenizer::payloads;
use aprender_bench_tokenizer::{load_aprender_tokenizer, load_hf_tokenizer, skip_no_tokenizer};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_roundtrip_aprender(c: &mut Criterion) {
    let Some(tokenizer) = load_aprender_tokenizer() else {
        skip_no_tokenizer!("roundtrip_aprender");
        return;
    };

    let mut group = c.benchmark_group("roundtrip_aprender");

    let payloads: &[(&str, &str)] = &[
        ("short_nl_30c", payloads::SHORT_NL),
        ("medium_code_200c", payloads::MEDIUM_CODE),
        ("long_code_636c", payloads::LONG_CODE_MERGE_SORT),
    ];

    for (name, payload) in payloads {
        group.bench_with_input(BenchmarkId::new("payload", name), payload, |b, input| {
            b.iter(|| {
                let ids = tokenizer.encode(black_box(input));
                tokenizer.decode(black_box(&ids))
            });
        });
    }

    group.finish();
}

fn bench_roundtrip_hf(c: &mut Criterion) {
    let Some(tokenizer) = load_hf_tokenizer() else {
        skip_no_tokenizer!("roundtrip_hf");
        return;
    };

    let mut group = c.benchmark_group("roundtrip_hf");

    let payloads: &[(&str, &str)] = &[
        ("short_nl_30c", payloads::SHORT_NL),
        ("medium_code_200c", payloads::MEDIUM_CODE),
        ("long_code_636c", payloads::LONG_CODE_MERGE_SORT),
    ];

    for (name, payload) in payloads {
        group.bench_with_input(BenchmarkId::new("payload", name), payload, |b, input| {
            b.iter(|| {
                let encoding = tokenizer.encode(black_box(*input), false).unwrap();
                tokenizer.decode(encoding.get_ids(), false).unwrap()
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_roundtrip_aprender, bench_roundtrip_hf);
criterion_main!(benches);
