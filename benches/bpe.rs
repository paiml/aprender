//! Benchmarks for BPE tokenizer (GH-378).
//!
//! **DEPRECATED**: Use `cargo bench -p aprender-bench-tokenizer` instead.
//! The `aprender-bench-tokenizer` crate provides comprehensive head-to-head
//! benchmarks against HuggingFace tokenizers v0.22 with canonical payloads,
//! scaling analysis, batch encoding, and special token handling.
//!
//! Measures encode/decode throughput across word lengths and real-world payloads.
//! Uses the root `tokenizer.json` (Qwen2.5 151K vocab) shipped in the repo.
//! Skips gracefully if the tokenizer file is absent.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Merge-sort payload from `examples/bench_bpe.rs` — 636 chars, representative code.
const QWEN_PAYLOAD: &str = r#"
def merge_sort(arr: list[int]) -> list[int]:
    """Recursively sort a list using merge sort algorithm."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)

def _merge(left: list[int], right: list[int]) -> list[int]:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
"#;

/// Try to load the repo-root tokenizer.json. Returns None if absent.
fn load_tokenizer() -> Option<aprender::text::bpe::BpeTokenizer> {
    // Try repo-root tokenizer.json first, then models/qwen3-4b/
    let paths = ["tokenizer.json", "models/qwen3-4b/tokenizer.json"];
    for path in &paths {
        if let Ok(tok) = aprender::text::bpe::BpeTokenizer::from_huggingface(path) {
            return Some(tok);
        }
    }
    None
}

/// Generate a synthetic word of given char length for scaling benchmarks.
fn synthetic_word(len: usize) -> String {
    // Mix of ASCII letters and common code tokens to exercise BPE merges
    let base = "def merge_sort(arr: list[int]) -> result.append(left[i]) ";
    base.chars().cycle().take(len).collect()
}

fn bench_bpe_encode(c: &mut Criterion) {
    let Some(tokenizer) = load_tokenizer() else {
        eprintln!("SKIP: tokenizer.json not found — skipping BPE encode benchmarks");
        return;
    };

    let mut group = c.benchmark_group("bpe_encode");

    for &char_len in &[10, 50, 200, 500] {
        let word = synthetic_word(char_len);
        let tokens = tokenizer.encode(&word);
        group.throughput(Throughput::Elements(tokens.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("chars", char_len),
            &word,
            |b, input| {
                b.iter(|| tokenizer.encode(black_box(input)));
            },
        );
    }

    group.finish();
}

fn bench_bpe_encode_qwen_payload(c: &mut Criterion) {
    let Some(tokenizer) = load_tokenizer() else {
        eprintln!("SKIP: tokenizer.json not found — skipping Qwen payload benchmark");
        return;
    };

    let mut group = c.benchmark_group("bpe_encode_qwen_payload");

    let tokens = tokenizer.encode(QWEN_PAYLOAD);
    group.throughput(Throughput::Elements(tokens.len() as u64));

    group.bench_function("merge_sort_636chars", |b| {
        b.iter(|| tokenizer.encode(black_box(QWEN_PAYLOAD)));
    });

    group.finish();
}

fn bench_bpe_decode(c: &mut Criterion) {
    let Some(tokenizer) = load_tokenizer() else {
        eprintln!("SKIP: tokenizer.json not found — skipping BPE decode benchmarks");
        return;
    };

    let mut group = c.benchmark_group("bpe_decode");

    // Encode the Qwen payload, then benchmark decoding
    let token_ids = tokenizer.encode(QWEN_PAYLOAD);
    group.throughput(Throughput::Elements(token_ids.len() as u64));

    group.bench_function("merge_sort_payload", |b| {
        b.iter(|| tokenizer.decode(black_box(&token_ids)));
    });

    // Also bench decode at various token counts
    for &n_tokens in &[10, 50, 200] {
        let ids: Vec<u32> = token_ids.iter().copied().cycle().take(n_tokens).collect();
        group.throughput(Throughput::Elements(n_tokens as u64));

        group.bench_with_input(
            BenchmarkId::new("tokens", n_tokens),
            &ids,
            |b, input| {
                b.iter(|| tokenizer.decode(black_box(input)));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_bpe_encode,
    bench_bpe_encode_qwen_payload,
    bench_bpe_decode,
);
criterion_main!(benches);
