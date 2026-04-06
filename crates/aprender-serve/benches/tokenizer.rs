//! Benchmark suite for tokenizer operations
//!
//! Measures tokenization performance including:
//! - Basic tokenizer encode/decode
//! - BPE tokenizer with merges
//! - SentencePiece tokenizer
//! - Different text lengths
//! - Vocabulary lookups

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use realizar::tokenizer::{BPETokenizer, SentencePieceTokenizer, Tokenizer, Vocabulary};

fn create_basic_tokenizer(vocab_size: usize) -> Tokenizer {
    let vocab: Vec<String> = (0..vocab_size)
        .map(|i| {
            if i == 0 {
                "<unk>".to_string()
            } else {
                format!("token{i}")
            }
        })
        .collect();
    let vocab = Vocabulary::from_tokens(vocab).expect("test");
    Tokenizer::new(vocab, "<unk>").expect("test")
}

fn create_bpe_tokenizer(vocab_size: usize) -> BPETokenizer {
    let vocab: Vec<String> = (0..vocab_size)
        .map(|i| {
            if i == 0 {
                "<unk>".to_string()
            } else {
                format!("token{i}")
            }
        })
        .collect();

    // Create some example merges
    let merges = vec![
        ("t".to_string(), "o".to_string()),
        ("k".to_string(), "e".to_string()),
        ("n".to_string(), "s".to_string()),
    ];

    BPETokenizer::new(vocab, merges, "<unk>").expect("test")
}

fn create_sentencepiece_tokenizer(vocab_size: usize) -> SentencePieceTokenizer {
    let vocab: Vec<(String, f32)> = (0..vocab_size)
        .map(|i| {
            let token = if i == 0 {
                "<unk>".to_string()
            } else {
                format!("token{i}")
            };
            let score = -(i as f32); // Higher frequency = higher score
            (token, score)
        })
        .collect();

    SentencePieceTokenizer::new(vocab, "<unk>").expect("test")
}

// Benchmark: Basic tokenizer encode
fn benchmark_basic_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_encode");

    let texts = [
        ("short", "token1 token2"),
        ("medium", "token1 token2 token3 token4 token5"),
        (
            "long",
            "token1 token2 token3 token4 token5 token6 token7 token8 token9 token10",
        ),
    ];

    for (name, text) in &texts {
        group.bench_with_input(BenchmarkId::from_parameter(name), text, |b, &text| {
            let tokenizer = create_basic_tokenizer(100);
            b.iter(|| {
                let tokens = tokenizer.encode(black_box(text));
                black_box(tokens)
            });
        });
    }

    group.finish();
}

// Benchmark: Basic tokenizer decode
fn benchmark_basic_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_decode");

    let token_sequences = [
        ("short", vec![1, 2]),
        ("medium", vec![1, 2, 3, 4, 5]),
        ("long", vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ];

    for (name, tokens) in &token_sequences {
        group.bench_with_input(BenchmarkId::from_parameter(name), tokens, |b, tokens| {
            let tokenizer = create_basic_tokenizer(100);
            b.iter(|| {
                let text = tokenizer.decode(black_box(tokens)).expect("test");
                black_box(text)
            });
        });
    }

    group.finish();
}

// Benchmark: BPE tokenizer encode
fn benchmark_bpe_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("bpe_encode");

    let texts = [
        ("short", "tokens"),
        ("medium", "tokens tokens tokens"),
        ("long", "tokens tokens tokens tokens tokens tokens"),
    ];

    for (name, text) in &texts {
        group.bench_with_input(BenchmarkId::from_parameter(name), text, |b, &text| {
            let tokenizer = create_bpe_tokenizer(100);
            b.iter(|| {
                let tokens = tokenizer.encode(black_box(text));
                black_box(tokens)
            });
        });
    }

    group.finish();
}

// Benchmark: BPE tokenizer decode
fn benchmark_bpe_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("bpe_decode");

    let token_sequences = [
        ("short", vec![1, 2]),
        ("medium", vec![1, 2, 3, 4, 5]),
        ("long", vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ];

    for (name, tokens) in &token_sequences {
        group.bench_with_input(BenchmarkId::from_parameter(name), tokens, |b, tokens| {
            let tokenizer = create_bpe_tokenizer(100);
            b.iter(|| {
                let text = tokenizer.decode(black_box(tokens)).expect("test");
                black_box(text)
            });
        });
    }

    group.finish();
}

// Benchmark: SentencePiece tokenizer encode
fn benchmark_sentencepiece_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("sentencepiece_encode");

    let texts = [
        ("short", "hello"),
        ("medium", "hello world test"),
        ("long", "hello world test example sentence piece"),
    ];

    for (name, text) in &texts {
        group.bench_with_input(BenchmarkId::from_parameter(name), text, |b, &text| {
            let tokenizer = create_sentencepiece_tokenizer(100);
            b.iter(|| {
                let tokens = tokenizer.encode(black_box(text));
                black_box(tokens)
            });
        });
    }

    group.finish();
}

// Benchmark: SentencePiece tokenizer decode
fn benchmark_sentencepiece_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("sentencepiece_decode");

    let token_sequences = [
        ("short", vec![1, 2]),
        ("medium", vec![1, 2, 3, 4, 5]),
        ("long", vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ];

    for (name, tokens) in &token_sequences {
        group.bench_with_input(BenchmarkId::from_parameter(name), tokens, |b, tokens| {
            let tokenizer = create_sentencepiece_tokenizer(100);
            b.iter(|| {
                let text = tokenizer.decode(black_box(tokens)).expect("test");
                black_box(text)
            });
        });
    }

    group.finish();
}

// Benchmark: Vocabulary lookup
fn benchmark_vocab_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("vocab_lookup");

    for vocab_size in &[100, 1000, 10000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(vocab_size),
            vocab_size,
            |b, &size| {
                let tokenizer = create_basic_tokenizer(size);
                let test_token = format!("token{}", size / 2); // Middle of vocab

                b.iter(|| {
                    let tokens = tokenizer.encode(black_box(&test_token));
                    black_box(tokens)
                });
            },
        );
    }

    group.finish();
}

// Benchmark: Roundtrip encoding/decoding
fn benchmark_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");

    let texts = [
        ("short", "token1 token2"),
        ("medium", "token1 token2 token3 token4 token5"),
        (
            "long",
            "token1 token2 token3 token4 token5 token6 token7 token8 token9 token10",
        ),
    ];

    for (name, text) in &texts {
        group.bench_with_input(BenchmarkId::from_parameter(name), text, |b, &text| {
            let tokenizer = create_basic_tokenizer(100);
            b.iter(|| {
                let tokens = tokenizer.encode(black_box(text));
                let decoded = tokenizer.decode(&tokens).expect("test");
                black_box(decoded)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_basic_encode,
    benchmark_basic_decode,
    benchmark_bpe_encode,
    benchmark_bpe_decode,
    benchmark_sentencepiece_encode,
    benchmark_sentencepiece_decode,
    benchmark_vocab_lookup,
    benchmark_roundtrip
);
criterion_main!(benches);
