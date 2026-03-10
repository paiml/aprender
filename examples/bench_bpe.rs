//! BPE tokenizer benchmark.
//!
//! Loads a HuggingFace tokenizer.json and measures encode throughput.
//!
//! Usage:
//!   cargo run --release --example bench_bpe [-- /path/to/tokenizer.json]
//!
//! Defaults to ./tokenizer.json (Qwen2.5 shipped in repo root).

use aprender::text::bpe::BpeTokenizer;
use std::time::{Duration, Instant};

/// Sample Python function (~500 chars) used as the encoding payload.
const SAMPLE_CODE: &str = r#"
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

const ITERATIONS: usize = 1_000;
const WARMUP: usize = 50;

fn main() {
    // Resolve tokenizer path from CLI arg or default
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tokenizer.json".to_string());

    eprintln!("=== BPE Tokenizer Benchmark ===");
    eprintln!("Tokenizer : {path}");
    eprintln!(
        "Payload   : {} chars, {} bytes",
        SAMPLE_CODE.len(),
        SAMPLE_CODE.len()
    );
    eprintln!("Iterations: {ITERATIONS} (warmup: {WARMUP})");
    eprintln!();

    // ---- Load ----
    let t0 = Instant::now();
    let tokenizer = BpeTokenizer::from_huggingface(&path)
        .unwrap_or_else(|e| panic!("Failed to load tokenizer from '{path}': {e}"));
    let load_time = t0.elapsed();

    eprintln!("Vocab size: {}", tokenizer.vocab_size());
    eprintln!("Load time : {load_time:.2?}");
    eprintln!();

    // ---- Single encode to inspect output ----
    let tokens = tokenizer.encode(SAMPLE_CODE);
    eprintln!("Tokens per encode: {}", tokens.len());
    eprintln!("First 20 token IDs: {:?}", &tokens[..tokens.len().min(20)]);
    eprintln!();

    // ---- Warmup ----
    for _ in 0..WARMUP {
        let _ = tokenizer.encode(SAMPLE_CODE);
    }

    // ---- Timed run ----
    let mut encode_times: Vec<Duration> = Vec::with_capacity(ITERATIONS);
    let wall_start = Instant::now();

    for _ in 0..ITERATIONS {
        let t = Instant::now();
        let _ = tokenizer.encode(SAMPLE_CODE);
        encode_times.push(t.elapsed());
    }

    let wall_elapsed = wall_start.elapsed();

    // ---- Stats ----
    let total_tokens = tokens.len() as u64 * ITERATIONS as u64;
    let tokens_per_sec = total_tokens as f64 / wall_elapsed.as_secs_f64();

    encode_times.sort();
    let sum: Duration = encode_times.iter().sum();
    let mean = sum / ITERATIONS as u32;
    let p50 = encode_times[ITERATIONS / 2];
    let p95 = encode_times[ITERATIONS * 95 / 100];
    let p99 = encode_times[ITERATIONS * 99 / 100];
    let min = encode_times[0];
    let max = encode_times[ITERATIONS - 1];

    eprintln!("--- Results ({ITERATIONS} iterations) ---");
    eprintln!("Wall time   : {wall_elapsed:.2?}");
    eprintln!("Tokens/sec  : {tokens_per_sec:.0}");
    eprintln!(
        "Encodes/sec : {:.0}",
        ITERATIONS as f64 / wall_elapsed.as_secs_f64()
    );
    eprintln!();
    eprintln!("Per-encode latency:");
    eprintln!("  mean : {mean:.2?}");
    eprintln!("  min  : {min:.2?}");
    eprintln!("  p50  : {p50:.2?}");
    eprintln!("  p95  : {p95:.2?}");
    eprintln!("  p99  : {p99:.2?}");
    eprintln!("  max  : {max:.2?}");
    eprintln!();

    // ---- Decode benchmark ----
    eprintln!("--- Decode Benchmark ({ITERATIONS} iterations) ---");

    // Warmup decode
    for _ in 0..WARMUP {
        let _ = tokenizer.decode(&tokens);
    }

    let mut decode_times: Vec<Duration> = Vec::with_capacity(ITERATIONS);
    let decode_wall_start = Instant::now();

    for _ in 0..ITERATIONS {
        let t = Instant::now();
        let _ = tokenizer.decode(&tokens);
        decode_times.push(t.elapsed());
    }

    let decode_wall = decode_wall_start.elapsed();
    decode_times.sort();

    let decode_sum: Duration = decode_times.iter().sum();
    let decode_mean = decode_sum / ITERATIONS as u32;
    let decode_p50 = decode_times[ITERATIONS / 2];
    let decode_p99 = decode_times[ITERATIONS * 99 / 100];

    let decode_tokens_per_sec = total_tokens as f64 / decode_wall.as_secs_f64();

    eprintln!("Wall time   : {decode_wall:.2?}");
    eprintln!("Tokens/sec  : {decode_tokens_per_sec:.0}");
    eprintln!("Per-decode latency:");
    eprintln!("  mean : {decode_mean:.2?}");
    eprintln!("  p50  : {decode_p50:.2?}");
    eprintln!("  p99  : {decode_p99:.2?}");
}
