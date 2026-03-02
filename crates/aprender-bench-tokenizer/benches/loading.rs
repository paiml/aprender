//! Tokenizer loading benchmarks: aprender vs HuggingFace tokenizers v0.22.
//!
//! Measures parse time for the 7MB `tokenizer.json` (Qwen2.5 151K vocab).
//! Uses `sample_size(10)` since loading is slow (~100ms+).
//!
//! ## Measured results (GH-378 optimized loading)
//!
//! | Method | aprender | HF v0.22 | Speedup |
//! |--------|----------|----------|---------|
//! | from_file | 142ms | 204ms | 1.43x |
//! | from_json | 136ms | — | — |
//!
//! Optimizations: pre-sized HashMaps, owned-string moves (saves 150K clones),
//! fast merge path skipping `merge_ranks` (saves 300K String allocations).
//! Applies to all vocab sizes (Qwen2, Whisper, GPT-2, LLaMA).

use aprender_bench_tokenizer::{
    find_tokenizer_json, read_tokenizer_json_string, skip_no_tokenizer,
};
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_loading(c: &mut Criterion) {
    let Some(path) = find_tokenizer_json() else {
        skip_no_tokenizer!("loading");
        return;
    };
    let json_string = read_tokenizer_json_string();

    let mut group = c.benchmark_group("loading");
    group.sample_size(10);

    // aprender: from_file (disk I/O + parse)
    group.bench_function("aprender_from_file", |b| {
        b.iter(|| {
            aprender::text::bpe::BpeTokenizer::from_huggingface(&path).unwrap();
        });
    });

    // aprender: from_json (in-memory parse only)
    if let Some(ref json) = json_string {
        group.bench_function("aprender_from_json", |b| {
            b.iter(|| {
                aprender::text::bpe::BpeTokenizer::from_huggingface_json(json).unwrap();
            });
        });
    }

    // HF: from_file
    group.bench_function("hf_from_file", |b| {
        b.iter(|| {
            tokenizers::Tokenizer::from_file(&path).unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_loading);
criterion_main!(benches);
