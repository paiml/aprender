//! Special token handling benchmarks: aprender vs HuggingFace tokenizers v0.22.
//!
//! Exercises ChatML 3-turn conversation with `<|im_start|>`, `<|im_end|>` markers.
//! Tests the `split_on_special_tokens` path (PMAT-114).
//!
//! ## Performance targets
//!
//! | Payload | aprender Target | HF v0.22 Ref |
//! |---------|-----------------|--------------|
//! | 3-turn ChatML | < 150us | ~200us |

use aprender_bench_tokenizer::payloads;
use aprender_bench_tokenizer::{load_aprender_tokenizer, load_hf_tokenizer, skip_no_tokenizer};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_special_tokens_aprender(c: &mut Criterion) {
    let Some(tokenizer) = load_aprender_tokenizer() else {
        skip_no_tokenizer!("special_tokens_aprender");
        return;
    };

    let mut group = c.benchmark_group("special_tokens_aprender");

    group.bench_function("chatml_3turn", |b| {
        b.iter(|| tokenizer.encode(black_box(payloads::CHATML_CONVERSATION)));
    });

    group.finish();
}

fn bench_special_tokens_hf(c: &mut Criterion) {
    let Some(tokenizer) = load_hf_tokenizer() else {
        skip_no_tokenizer!("special_tokens_hf");
        return;
    };

    let mut group = c.benchmark_group("special_tokens_hf");

    group.bench_function("chatml_3turn", |b| {
        b.iter(|| {
            tokenizer
                .encode(black_box(payloads::CHATML_CONVERSATION), false)
                .unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_special_tokens_aprender,
    bench_special_tokens_hf,
);
criterion_main!(benches);
