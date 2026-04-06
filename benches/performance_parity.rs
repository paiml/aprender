#![allow(clippy::manual_div_ceil)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::redundant_clone)]
#![allow(clippy::excessive_precision)]

//! Performance Parity Benchmarks (Refs PERF-PARITY-001)
//!
//! Implements the benchmark specification v1.1 for comparing realizar with:
//! - llama.cpp (GGUF inference)
//! - Ollama (production deployment)
//!
//! ## Toyota Way Principles
//!
//! - **Genchi Genbutsu**: Go and see - measure actual inference, not test ops
//! - **Jidoka**: Stop on anomaly - fail-fast on thermal throttling or variance
//! - **Kaizen**: Continuous improvement - track regressions over time
//!
//! ## Benchmark Categories
//!
//! | Category      | Target           | Metric                    |
//! |---------------|------------------|---------------------------|
//! | TTFT (Cold)   | ≤1.0× llama.cpp  | Time to first token (ms)  |
//! | TTFT (Warm)   | ≤1.0× llama.cpp  | Time to first token (ms)  |
//! | TPS           | ≥80% llama.cpp   | Tokens per second         |
//! | Memory        | ≤1.1× llama.cpp  | Peak RSS (MB)             |
//!
//! ## Usage
//!
//! ```bash
//! # Run all performance parity benchmarks
//! cargo bench --bench performance_parity
//!
//! # Run specific benchmark group
//! cargo bench --bench performance_parity -- ttft
//! cargo bench --bench performance_parity -- tps
//! ```
//!
//! ## References
//!
//! - [17] Hoefler & Belli, "Scientific Benchmarking of Parallel Computing Systems", SC'15
//! - [11] Dean & Barroso, "The Tail at Scale", CACM 2013

#![allow(clippy::cast_precision_loss)]
#![allow(dead_code)] // Benchmark constants may not all be used

use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};

// Import realizar types
use realizar::layers::{softmax, Attention, FusedQKVAttention, LayerNorm};
use realizar::quantize::{dequantize_q4_k_simd, fused_q4k_dot_simd, fused_q4k_parallel_matvec};
use realizar::Tensor;

/// Fixed prompt for reproducible benchmarks (DO NOT CHANGE)
#[allow(dead_code)]
const BENCHMARK_PROMPT: &str = "The quick brown fox jumps over the lazy dog.";

/// Target sequence lengths for scaling analysis
const SEQ_LENGTHS: &[usize] = &[128, 256, 512, 1024];

/// Hidden dimension for benchmark models
const HIDDEN_DIM: usize = 4096;

/// Number of attention heads
const NUM_HEADS: usize = 32;

/// Head dimension
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS;

// ============================================================================
// Shared Benchmark Helpers (Kaizen: DRY data transformation patterns)
// ============================================================================

/// Q4_K super-block size constants
const Q4K_SUPERBLOCK_SIZE: usize = 256; // values per super-block
const Q4K_SUPERBLOCK_BYTES: usize = 144; // bytes per super-block

/// Create raw Q4_K super-block data for benchmarking.
///
/// Each super-block is 144 bytes encoding 256 values.
/// Uses d=1.0 (f16), dmin=0.0, and a deterministic fill pattern.
fn make_q4k_data(num_values: usize) -> Vec<u8> {
    make_q4k_data_scaled(num_values, 0x3C00) // d=1.0 in f16
}

/// Create raw Q4_K data with a configurable scale factor (d) in f16 encoding.
fn make_q4k_data_scaled(num_values: usize, d_f16: u16) -> Vec<u8> {
    let sb_count = num_values.div_ceil(Q4K_SUPERBLOCK_SIZE);
    let data_size = sb_count * Q4K_SUPERBLOCK_BYTES;
    let mut data = vec![0u8; data_size];

    for sb_idx in 0..sb_count {
        let offset = sb_idx * Q4K_SUPERBLOCK_BYTES;
        data[offset..offset + 2].copy_from_slice(&d_f16.to_le_bytes());
        data[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes()); // dmin=0
        for i in 4..Q4K_SUPERBLOCK_BYTES {
            data[offset + i] = ((sb_idx + i) % 16) as u8;
        }
    }
    data
}

/// Create Q4_K weight matrix data (out_dim rows of in_dim values each).
fn make_q4k_weights(in_dim: usize, out_dim: usize) -> Vec<u8> {
    make_q4k_data(in_dim * out_dim)
}

/// Create an `OwnedQuantizedTensor` with Q4_K benchmark data.
fn make_q4k_tensor(in_dim: usize, out_dim: usize) -> realizar::gguf::OwnedQuantizedTensor {
    let super_blocks_per_row = in_dim.div_ceil(Q4K_SUPERBLOCK_SIZE);
    let bytes_per_row = super_blocks_per_row * Q4K_SUPERBLOCK_BYTES;
    let data_size = out_dim * bytes_per_row;
    let mut data = vec![0u8; data_size];

    for row in 0..out_dim {
        for sb in 0..super_blocks_per_row {
            let offset = row * bytes_per_row + sb * Q4K_SUPERBLOCK_BYTES;
            data[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // d=1.0
            data[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes()); // dmin=0
            for i in 4..Q4K_SUPERBLOCK_BYTES {
                data[offset + i] = ((row + sb + i) % 16) as u8;
            }
        }
    }

    realizar::gguf::OwnedQuantizedTensor {
        data,
        in_dim,
        out_dim,
        qtype: 12, // Q4_K (GGUF_TYPE_Q4_K)
    }
}

/// Create a `GGUFConfig` for benchmarking with sensible defaults.
///
/// Only the most commonly varied parameters are exposed; the rest use
/// standard benchmark defaults (context_length=2048, rope_theta=10000, etc.).
fn make_bench_config(
    hidden_dim: usize,
    intermediate_dim: usize,
    num_layers: usize,
    num_heads: usize,
    vocab_size: usize,
) -> realizar::gguf::GGUFConfig {
    realizar::gguf::GGUFConfig {
        architecture: "test".to_string(),
        constraints: realizar::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim,
        intermediate_dim,
        num_layers,
        num_heads,
        num_kv_heads: num_heads,
        vocab_size,
        context_length: 2048,
        eps: 1e-5,
        rope_theta: 10000.0,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    }
}

/// Create Q, K, V test vectors of shape [seq_len, dim] with deterministic patterns.
fn make_qkv_vecs(seq_len: usize, dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q: Vec<f32> = (0..seq_len * dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    (q, k, v)
}

/// Create sinusoidal activation vector for benchmark input.
fn make_activations(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32 * 0.001).sin()).collect()
}

// ============================================================================
// IMP-001 + IMP-002: Unified Q4_K SIMD Benchmarks (dequant + dot + matvec)
// ============================================================================

/// Table-driven Q4K benchmark entry describing a single benchmark variant.
struct Q4kBenchEntry {
    group_name: &'static str,
    sample_size: usize,
    cases: Q4kBenchCases,
}

/// The different Q4K benchmark case types, unified under one dispatch.
enum Q4kBenchCases {
    /// Dequantize raw Q4_K super-blocks to f32.
    Dequant(&'static [usize]),
    /// Fused Q4_K dot product (single row).
    FusedDot(&'static [usize]),
    /// Fused Q4_K parallel matvec with optional dequant-then-matvec baseline.
    Matvec {
        dimensions: &'static [(usize, usize, &'static str)],
        include_dequant_baseline: bool,
    },
    /// Single-row fused dot benchmarks (fixed sizes).
    SingleRowDot(&'static [usize]),
}

/// Benchmark Q4K fused matvec over a set of dimension configurations.
///
/// For each (in_dim, out_dim, name) tuple, runs the `fused_q4k_parallel_matvec`
/// benchmark and optionally the dequant-then-matvec baseline via
/// `bench_dequant_then_matvec`.
fn bench_q4k_matvec(
    group: &mut BenchmarkGroup<WallTime>,
    dimensions: &[(usize, usize, &str)],
    include_dequant_baseline: bool,
) {
    for &(in_dim, out_dim, name) in dimensions {
        let weights = make_q4k_weights(in_dim, out_dim);
        let activations = make_activations(in_dim);
        let weight_values = in_dim * out_dim;
        let flops = 2 * in_dim * out_dim;

        // Pick throughput metric: weight_values for IMP-100c, flops for IMP-103
        let throughput = if include_dequant_baseline {
            weight_values
        } else {
            flops
        };
        group.throughput(Throughput::Elements(throughput as u64));

        group.bench_with_input(
            BenchmarkId::new("fused_q4k_matvec", format!("{name}_{in_dim}x{out_dim}")),
            &(&weights, &activations, in_dim, out_dim),
            |b, (w, a, in_d, out_d)| {
                b.iter(|| {
                    let result =
                        fused_q4k_parallel_matvec(black_box(*w), black_box(*a), *in_d, *out_d)
                            .expect("test");
                    black_box(result)
                });
            },
        );

        if include_dequant_baseline {
            bench_dequant_then_matvec(group, &weights, &activations, in_dim, out_dim, name);
        }
    }
}

/// Benchmark the dequant-then-matvec baseline for a single dimension pair.
///
/// Dequantizes the full weight matrix then performs a scalar matvec.
/// Used as the "separate operations" baseline for IMP-100c comparisons.
fn bench_dequant_then_matvec(
    group: &mut BenchmarkGroup<WallTime>,
    weights: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    name: &str,
) {
    group.bench_with_input(
        BenchmarkId::new("dequant_then_matvec", format!("{name}_{in_dim}x{out_dim}")),
        &(weights, activations, in_dim, out_dim),
        |b, (w, a, in_d, out_d)| {
            b.iter(|| {
                let dequantized = dequantize_q4_k_simd(black_box(*w)).expect("test");
                let mut output = vec![0.0f32; *out_d];
                for row in 0..*out_d {
                    let row_start = row * *in_d;
                    let row_end = row_start + *in_d;
                    if row_end <= dequantized.len() {
                        let row_slice = &dequantized[row_start..row_end];
                        output[row] = row_slice.iter().zip(a.iter()).map(|(w, x)| w * x).sum();
                    }
                }
                black_box(output)
            });
        },
    );
}

/// Benchmark Q4K SIMD dequantization over a set of super-block counts.
fn bench_q4k_dequant(group: &mut BenchmarkGroup<WallTime>, sb_counts: &[usize]) {
    for &sb_count in sb_counts {
        let num_values = sb_count * Q4K_SUPERBLOCK_SIZE;
        let data = make_q4k_data(num_values);

        group.throughput(Throughput::Elements(num_values as u64));
        group.bench_with_input(
            BenchmarkId::new("simd", format!("{num_values}_values")),
            &data,
            |b, d| {
                b.iter(|| {
                    let result = dequantize_q4_k_simd(black_box(d)).expect("test");
                    black_box(result)
                });
            },
        );
    }
}

/// Run all Q4K-family benchmarks via a single table-driven dispatcher.
fn run_q4k_bench_entry(c: &mut Criterion, entry: &Q4kBenchEntry) {
    let mut group = c.benchmark_group(entry.group_name);
    group.sample_size(entry.sample_size);

    match &entry.cases {
        Q4kBenchCases::Dequant(sb_counts) => {
            bench_q4k_dequant(&mut group, sb_counts);
        },
        Q4kBenchCases::FusedDot(sizes) => {
            for &size in *sizes {
                let weights = make_q4k_data(size);
                let activations = make_activations(size);

                group.throughput(Throughput::Elements(size as u64));
                group.bench_with_input(
                    BenchmarkId::new("fused_simd", format!("{size}_dim")),
                    &(&weights, &activations),
                    |b, (w, a)| {
                        b.iter(|| {
                            let result =
                                fused_q4k_dot_simd(black_box(*w), black_box(*a)).expect("test");
                            black_box(result)
                        });
                    },
                );
            }
        },
        Q4kBenchCases::Matvec {
            dimensions,
            include_dequant_baseline,
        } => {
            bench_q4k_matvec(&mut group, dimensions, *include_dequant_baseline);
        },
        Q4kBenchCases::SingleRowDot(sizes) => {
            for &size in *sizes {
                let row_weights = make_q4k_data(size);
                let activations = make_activations(size);

                group.bench_function(format!("single_row_dot_{size}"), |b| {
                    b.iter(|| {
                        let result =
                            fused_q4k_dot_simd(black_box(&row_weights), black_box(&activations))
                                .expect("test");
                        black_box(result)
                    });
                });
            }
        },
    }

    group.finish();
}

// Q4K benchmark suite table: all Q4K entries driven by a single function
static Q4K_BENCH_SUITE: &[Q4kBenchEntry] = &[
    Q4kBenchEntry {
        group_name: "q4k_dequant_simd",
        sample_size: 100,
        cases: Q4kBenchCases::Dequant(&[1, 4, 16, 64, 256]),
    },
    Q4kBenchEntry {
        group_name: "fused_q4k_dot",
        sample_size: 100,
        cases: Q4kBenchCases::FusedDot(&[256, 1024, 4096, 16384]),
    },
    Q4kBenchEntry {
        group_name: "imp_100c_quantized_vs_dequantized",
        sample_size: 100,
        cases: Q4kBenchCases::Matvec {
            dimensions: &[
                (1024, 1024, "square"),
                (1536, 4096, "qwen_up"),
                (4096, 1536, "qwen_down"),
                (2560, 10240, "phi2_up"),
            ],
            include_dequant_baseline: true,
        },
    },
    Q4kBenchEntry {
        group_name: "imp_103_q4k_matvec_optimization",
        sample_size: 50,
        cases: Q4kBenchCases::Matvec {
            dimensions: &[
                (512, 512, "output_proj"),
                (512, 1024, "ffn_up"),
                (1024, 512, "ffn_down"),
                (512, 1536, "qkv_proj"),
                (512, 2000, "lm_head"),
                (1024, 4096, "large_ffn"),
            ],
            include_dequant_baseline: false,
        },
    },
    Q4kBenchEntry {
        group_name: "imp_103_q4k_single_row_dot",
        sample_size: 50,
        cases: Q4kBenchCases::SingleRowDot(&[512, 4096]),
    },
];

/// Run all Q4K benchmarks from the suite table (Kaizen: single entry point).
fn benchmark_q4k_suite(c: &mut Criterion) {
    for entry in Q4K_BENCH_SUITE {
        run_q4k_bench_entry(c, entry);
    }
}

// ============================================================================
// Unified Layer Forward Benchmarks (Kaizen: DRY layer-forward patterns)
// ============================================================================

/// Describes a parameterized layer-forward benchmark variant.
/// Each entry maps to a criterion benchmark group, reducing the repeated
/// "create layer -> create tensor -> bench forward" DataTransformation pattern.
enum LayerBenchOp {
    /// FusedQKVAttention::forward over sequence lengths
    FusedAttention {
        hidden_dim: usize,
        head_dim: usize,
        seq_lengths: &'static [usize],
    },
    /// LayerNorm::forward over sequence lengths
    LayerNorm {
        hidden_dim: usize,
        seq_lengths: &'static [usize],
    },
    /// softmax over sequence lengths
    Softmax { seq_lengths: &'static [usize] },
    /// Linear::forward over (in_dim, out_dim) pairs
    Linear {
        dimensions: &'static [(usize, usize)],
    },
    /// FusedQKVAttention::forward over batch sizes (memory efficiency)
    FusedAttentionBatch {
        hidden_dim: usize,
        head_dim: usize,
        seq_len: usize,
        batch_sizes: &'static [usize],
    },
    /// Fused vs Separate Attention comparison (IMP-003)
    AttentionComparison {
        hidden_dim: usize,
        head_dim: usize,
        seq_len: usize,
    },
    /// TTFT Prefill simulation (multi-layer forward pass)
    TtftPrefill {
        hidden_dim: usize,
        head_dim: usize,
        num_layers: usize,
        prompt_len: usize,
    },
}

/// Run a layer-forward benchmark from a `LayerBenchOp` descriptor.
fn run_layer_bench(c: &mut Criterion, group_name: &str, sample_size: usize, op: &LayerBenchOp) {
    let mut group = c.benchmark_group(group_name);
    group.sample_size(sample_size);

    match op {
        LayerBenchOp::FusedAttention {
            hidden_dim,
            head_dim,
            seq_lengths,
        } => {
            for &seq_len in *seq_lengths {
                let fused = FusedQKVAttention::new(*head_dim, *hidden_dim).expect("test");
                let input = Tensor::from_vec(
                    vec![seq_len, *hidden_dim],
                    (0..(seq_len * *hidden_dim))
                        .map(|i| (i as f32 * 0.001).sin())
                        .collect(),
                )
                .expect("test");

                group.throughput(Throughput::Elements(seq_len as u64));
                group.bench_with_input(
                    BenchmarkId::new("forward", format!("seq{seq_len}")),
                    &(&fused, &input),
                    |b, (f, i)| {
                        b.iter(|| {
                            let output = f.forward(black_box(*i)).expect("test");
                            black_box(output)
                        });
                    },
                );
            }
        },
        LayerBenchOp::LayerNorm {
            hidden_dim,
            seq_lengths,
        } => {
            for &seq_len in *seq_lengths {
                let layer_norm = realizar::layers::LayerNorm::new(*hidden_dim, 1e-5).expect("test");
                let input = Tensor::from_vec(
                    vec![seq_len, *hidden_dim],
                    (0..(seq_len * *hidden_dim))
                        .map(|i| (i as f32 * 0.01).sin())
                        .collect(),
                )
                .expect("test");

                group.throughput(Throughput::Elements((seq_len * *hidden_dim) as u64));
                group.bench_with_input(
                    BenchmarkId::new("forward", format!("seq{seq_len}")),
                    &(&layer_norm, &input),
                    |b, (ln, i)| {
                        b.iter(|| {
                            let output = ln.forward(black_box(*i)).expect("test");
                            black_box(output)
                        });
                    },
                );
            }
        },
        LayerBenchOp::Softmax { seq_lengths } => {
            for &seq_len in *seq_lengths {
                let input = Tensor::from_vec(
                    vec![seq_len],
                    (0..seq_len).map(|i| (i as f32 * 0.1).sin()).collect(),
                )
                .expect("test");

                group.throughput(Throughput::Elements(seq_len as u64));
                group.bench_with_input(
                    BenchmarkId::new("forward", format!("len{seq_len}")),
                    &input,
                    |b, i| {
                        b.iter(|| {
                            let output = softmax(black_box(i)).expect("test");
                            black_box(output)
                        });
                    },
                );
            }
        },
        LayerBenchOp::Linear { dimensions } => {
            for &(in_dim, out_dim) in *dimensions {
                let linear = realizar::layers::Linear::new(in_dim, out_dim).expect("test");
                let input = Tensor::from_vec(
                    vec![1, in_dim],
                    (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect(),
                )
                .expect("test");

                let ops = (in_dim * out_dim) as u64;
                group.throughput(Throughput::Elements(ops));
                group.bench_with_input(
                    BenchmarkId::new("forward", format!("{in_dim}x{out_dim}")),
                    &(&linear, &input),
                    |b, (l, i)| {
                        b.iter(|| {
                            let output = l.forward(black_box(*i)).expect("test");
                            black_box(output)
                        });
                    },
                );
            }
        },
        LayerBenchOp::FusedAttentionBatch {
            hidden_dim,
            head_dim,
            seq_len,
            batch_sizes,
        } => {
            for &batch_size in *batch_sizes {
                let fused = FusedQKVAttention::new(*head_dim, *hidden_dim).expect("test");
                let input = Tensor::from_vec(
                    vec![batch_size * *seq_len, *hidden_dim],
                    vec![0.1; batch_size * *seq_len * *hidden_dim],
                )
                .expect("test");

                group.throughput(Throughput::Elements((batch_size * *seq_len) as u64));
                group.bench_with_input(
                    BenchmarkId::new("fused_attn_batch", format!("batch{batch_size}")),
                    &(&fused, &input),
                    |b, (f, i)| {
                        b.iter(|| {
                            let output = f.forward(black_box(*i)).expect("test");
                            black_box(output)
                        });
                    },
                );
            }
        },
        LayerBenchOp::AttentionComparison {
            hidden_dim,
            head_dim,
            seq_len,
        } => {
            // Fused attention
            let fused = FusedQKVAttention::new(*head_dim, *hidden_dim).expect("test");
            let input = Tensor::from_vec(
                vec![*seq_len, *hidden_dim],
                (0..(*seq_len * *hidden_dim))
                    .map(|i| (i as f32 * 0.001).sin())
                    .collect(),
            )
            .expect("test");

            group.bench_function("fused_qkv_attention", |b| {
                b.iter(|| {
                    let output = fused.forward(black_box(&input)).expect("test");
                    black_box(output)
                });
            });

            // Separate attention (baseline)
            let attention = Attention::new(*head_dim).expect("test");
            let q = Tensor::from_vec(vec![*seq_len, *head_dim], vec![0.1; *seq_len * *head_dim])
                .expect("test");
            let k = q.clone();
            let v = q.clone();

            group.bench_function("separate_attention", |b| {
                b.iter(|| {
                    let output = attention
                        .forward(black_box(&q), black_box(&k), black_box(&v))
                        .expect("test");
                    black_box(output)
                });
            });
        },
        LayerBenchOp::TtftPrefill {
            hidden_dim,
            head_dim,
            num_layers,
            prompt_len,
        } => {
            let fused_attns: Vec<_> = (0..*num_layers)
                .map(|_| FusedQKVAttention::new(*head_dim, *hidden_dim).expect("test"))
                .collect();
            let layer_norms: Vec<_> = (0..*num_layers)
                .map(|_| LayerNorm::new(*hidden_dim, 1e-5).expect("test"))
                .collect();

            let input = Tensor::from_vec(
                vec![*prompt_len, *hidden_dim],
                (0..(*prompt_len * *hidden_dim))
                    .map(|i| (i as f32 * 0.001).sin())
                    .collect(),
            )
            .expect("test");

            group.throughput(Throughput::Elements(*prompt_len as u64));
            group.bench_function(format!("prefill_{num_layers}_layers"), |b| {
                b.iter(|| {
                    let mut hidden = input.clone();
                    for layer_idx in 0..*num_layers {
                        hidden = layer_norms[layer_idx].forward(&hidden).expect("test");
                        hidden = fused_attns[layer_idx].forward(&hidden).expect("test");
                    }
                    black_box(hidden)
                });
            });
        },
    }

    group.finish();
}

/// Table-driven layer benchmark entry.
struct LayerBenchEntry {
    group_name: &'static str,
    sample_size: usize,
    op: LayerBenchOp,
}

/// Static dimension table for linear benchmarks.
static LINEAR_DIMS: &[(usize, usize)] = &[(256, 256), (256, 1024), (1024, 256), (1024, 1024)];

/// Static batch sizes for memory-efficiency benchmarks.
static MEM_EFF_BATCH_SIZES: &[usize] = &[1, 4, 8];

/// Run all layer benchmarks from the suite table (Kaizen: single entry point).
fn benchmark_layer_suite(c: &mut Criterion) {
    let suite: &[LayerBenchEntry] = &[
        LayerBenchEntry {
            group_name: "fused_attention",
            sample_size: 50,
            op: LayerBenchOp::FusedAttention {
                hidden_dim: 256,
                head_dim: 32,
                seq_lengths: SEQ_LENGTHS,
            },
        },
        LayerBenchEntry {
            group_name: "layer_norm",
            sample_size: 10,
            op: LayerBenchOp::LayerNorm {
                hidden_dim: 256,
                seq_lengths: SEQ_LENGTHS,
            },
        },
        LayerBenchEntry {
            group_name: "softmax",
            sample_size: 10,
            op: LayerBenchOp::Softmax {
                seq_lengths: SEQ_LENGTHS,
            },
        },
        LayerBenchEntry {
            group_name: "linear",
            sample_size: 50,
            op: LayerBenchOp::Linear {
                dimensions: LINEAR_DIMS,
            },
        },
        LayerBenchEntry {
            group_name: "memory_efficiency",
            sample_size: 20,
            op: LayerBenchOp::FusedAttentionBatch {
                hidden_dim: 256,
                head_dim: 32,
                seq_len: 64,
                batch_sizes: MEM_EFF_BATCH_SIZES,
            },
        },
        LayerBenchEntry {
            group_name: "attention_comparison",
            sample_size: 50,
            op: LayerBenchOp::AttentionComparison {
                hidden_dim: 128,
                head_dim: 32,
                seq_len: 128,
            },
        },
        LayerBenchEntry {
            group_name: "ttft_simulation",
            sample_size: 20,
            op: LayerBenchOp::TtftPrefill {
                hidden_dim: 256,
                head_dim: 32,
                num_layers: 4,
                prompt_len: 32,
            },
        },
    ];

    for entry in suite {
        run_layer_bench(c, entry.group_name, entry.sample_size, &entry.op);
    }
}

// Layer benchmarks (fused_attention, layer_norm, softmax, linear, memory_efficiency,
// attention_comparison, ttft_simulation) are all driven by benchmark_layer_suite above.

// ============================================================================
// Tests for benchmark infrastructure
// ============================================================================

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::{dequantize_q4_k_simd, FusedQKVAttention, Tensor, SEQ_LENGTHS};
    use super::{BENCHMARK_PROMPT, HEAD_DIM, HIDDEN_DIM, NUM_HEADS};

    #[test]
    fn test_q4k_benchmark_data_valid() {
        // Verify benchmark data is valid Q4_K format
        let sb_count = 4;
        let data_size = sb_count * 144;
        let mut data = vec![0u8; data_size];

        for sb_idx in 0..sb_count {
            let offset = sb_idx * 144;
            data[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
            data[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes());
        }

        let result = dequantize_q4_k_simd(&data);
        assert!(result.is_ok());
        assert_eq!(result.expect("test").len(), sb_count * 256);
    }

    #[test]
    fn test_fused_attention_benchmark_sizes() {
        // Verify benchmark sizes are valid
        for &seq_len in SEQ_LENGTHS {
            let hidden_dim = 256;
            let head_dim = 32;

            let fused = FusedQKVAttention::new(head_dim, hidden_dim).expect("test");
            let input =
                Tensor::from_vec(vec![seq_len, hidden_dim], vec![0.1; seq_len * hidden_dim])
                    .expect("test");

            let output = fused.forward(&input).expect("test");
            assert_eq!(output.shape(), &[seq_len, hidden_dim]);
        }
    }

    #[test]
    fn test_benchmark_constants_valid() {
        // Verify benchmark constants are sensible
        assert!(HIDDEN_DIM > 0);
        assert!(NUM_HEADS > 0);
        assert_eq!(HIDDEN_DIM % NUM_HEADS, 0);
        assert_eq!(HEAD_DIM, HIDDEN_DIM / NUM_HEADS);
        assert!(!BENCHMARK_PROMPT.is_empty());
    }
}

// IMP-100c: Quantized vs Dequantized Throughput — included in Q4K_BENCH_SUITE

// ============================================================================
// IMP-101d + IMP-106: Unified Cache Strategy Benchmarks
// ============================================================================

/// Cache benchmark variant descriptor for table-driven dispatch.
enum CacheBenchOp {
    /// KV cache attention vs full recompute (IMP-101d)
    KvCacheVsRecompute {
        hidden_dim: usize,
        num_heads: usize,
        num_layers: usize,
        seq_lengths: &'static [usize],
    },
    /// Batch prefill vs sequential prefill (IMP-106c)
    BatchPrefill { prompt_lengths: &'static [usize] },
}

/// Table-driven cache benchmark entry.
struct CacheBenchEntry {
    group_name: &'static str,
    sample_size: usize,
    op: CacheBenchOp,
}

/// Run a cache-strategy benchmark from a `CacheBenchOp` descriptor.
fn run_cache_bench(c: &mut Criterion, entry: &CacheBenchEntry) {
    use realizar::gguf::OwnedQuantizedKVCache;

    let mut group = c.benchmark_group(entry.group_name);
    group.sample_size(entry.sample_size);

    match &entry.op {
        CacheBenchOp::KvCacheVsRecompute {
            hidden_dim,
            num_heads,
            num_layers,
            seq_lengths,
        } => {
            let head_dim = *hidden_dim / *num_heads;

            for &seq_len in *seq_lengths {
                let mut cache = OwnedQuantizedKVCache::new(*num_layers, *hidden_dim, seq_len + 64);
                for pos in 0..(seq_len - 1) {
                    for layer in 0..*num_layers {
                        let k: Vec<f32> = (0..*hidden_dim)
                            .map(|i| ((pos * *hidden_dim + i) as f32 * 0.001).sin())
                            .collect();
                        let v: Vec<f32> = (0..*hidden_dim)
                            .map(|i| ((pos * *hidden_dim + i) as f32 * 0.002).cos())
                            .collect();
                        cache.append(layer, &k, &v);
                    }
                    cache.advance();
                }

                let q: Vec<f32> = (0..*hidden_dim).map(|i| (i as f32 * 0.003).sin()).collect();
                let current_k: Vec<f32> =
                    (0..*hidden_dim).map(|i| (i as f32 * 0.004).cos()).collect();
                let current_v: Vec<f32> =
                    (0..*hidden_dim).map(|i| (i as f32 * 0.005).sin()).collect();

                let hd = *hidden_dim;
                group.throughput(Throughput::Elements(seq_len as u64));
                group.bench_with_input(
                    BenchmarkId::new("kv_cache_attention", format!("seq{seq_len}")),
                    &(&q, &cache, &current_k, &current_v, *num_heads, head_dim),
                    |b, (q, cache, cur_k, cur_v, n_heads, h_dim)| {
                        b.iter(|| {
                            let k_cache = cache.get_k(0);
                            let v_cache = cache.get_v(0);
                            let cache_len = k_cache.len() / hd;
                            let output = bench_cached_attention(
                                q, k_cache, v_cache, cur_k, cur_v, *n_heads, *h_dim, hd, cache_len,
                            );
                            black_box(output)
                        });
                    },
                );

                // Full recompute (O(n²) per token) - baseline comparison
                let all_k: Vec<f32> = (0..(seq_len * *hidden_dim))
                    .map(|i| (i as f32 * 0.001).sin())
                    .collect();
                let all_v: Vec<f32> = (0..(seq_len * *hidden_dim))
                    .map(|i| (i as f32 * 0.002).cos())
                    .collect();
                let all_q: Vec<f32> = (0..(seq_len * *hidden_dim))
                    .map(|i| (i as f32 * 0.003).sin())
                    .collect();

                group.bench_with_input(
                    BenchmarkId::new("full_recompute", format!("seq{seq_len}")),
                    &(&all_q, &all_k, &all_v, *num_heads, head_dim, seq_len),
                    |b, (q, k, v, n_heads, h_dim, s_len)| {
                        b.iter(|| {
                            let output = bench_full_recompute_attention(
                                q, k, v, *n_heads, *h_dim, hd, *s_len,
                            );
                            black_box(output)
                        });
                    },
                );
            }
        },
        CacheBenchOp::BatchPrefill { prompt_lengths } => {
            for &prompt_len in *prompt_lengths {
                let config = make_bench_config(256, 512, 2, 8, 1000);
                let model = create_benchmark_model(&config);
                let prompt: Vec<u32> = (0..prompt_len as u32).collect();

                group.bench_function(BenchmarkId::new("sequential", prompt_len), |b| {
                    b.iter(|| {
                        let mut cache = OwnedQuantizedKVCache::from_config(&config, 128);
                        for (pos, &token_id) in prompt.iter().enumerate() {
                            let _ = model.forward_single_with_cache(
                                black_box(token_id),
                                &mut cache,
                                pos,
                            );
                        }
                        black_box(cache.len())
                    });
                });

                group.bench_function(BenchmarkId::new("batch", prompt_len), |b| {
                    b.iter(|| {
                        let mut cache = OwnedQuantizedKVCache::from_config(&config, 128);
                        let _ = model.prefill_batch(black_box(&prompt), &mut cache);
                        black_box(cache.len())
                    });
                });
            }
        },
    }

    group.finish();
}

/// Static sequence lengths for KV cache benchmark.
static KV_CACHE_SEQ_LENGTHS: &[usize] = &[32, 64, 128, 256];

/// Static prompt lengths for batch prefill benchmark.
static BATCH_PREFILL_PROMPT_LENGTHS: &[usize] = &[4, 8, 16, 32];

/// Cache benchmark suite table: all cache-related entries driven by a single function.
static CACHE_BENCH_SUITE: &[CacheBenchEntry] = &[
    CacheBenchEntry {
        group_name: "imp_101d_kv_cache_attention",
        sample_size: 50,
        op: CacheBenchOp::KvCacheVsRecompute {
            hidden_dim: 256,
            num_heads: 4,
            num_layers: 4,
            seq_lengths: KV_CACHE_SEQ_LENGTHS,
        },
    },
    CacheBenchEntry {
        group_name: "imp_106_batch_prefill",
        sample_size: 50,
        op: CacheBenchOp::BatchPrefill {
            prompt_lengths: BATCH_PREFILL_PROMPT_LENGTHS,
        },
    },
];

/// Run all cache-strategy benchmarks from the suite table (Kaizen: single entry point).
fn benchmark_cache_suite(c: &mut Criterion) {
    for entry in CACHE_BENCH_SUITE {
        run_cache_bench(c, entry);
    }
}

/// Cached attention benchmark helper — single-token decode with KV cache
#[allow(clippy::too_many_arguments)]
fn bench_cached_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    cur_k: &[f32],
    cur_v: &[f32],
    num_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    cache_len: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; hidden_dim];

    for head in 0..num_heads {
        let head_offset = head * head_dim;
        let q_head = &q[head_offset..head_offset + head_dim];

        let mut scores = Vec::with_capacity(cache_len + 1);
        for pos in 0..cache_len {
            let k_start = pos * hidden_dim + head_offset;
            let k_head = &k_cache[k_start..k_start + head_dim];
            let score: f32 = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum();
            scores.push(score * scale);
        }
        let cur_k_head = &cur_k[head_offset..head_offset + head_dim];
        let cur_score: f32 = q_head.iter().zip(cur_k_head).map(|(a, b)| a * b).sum();
        scores.push(cur_score * scale);

        bench_softmax_inplace(&mut scores);

        let out_head = &mut output[head_offset..head_offset + head_dim];
        for (pos, &weight) in scores.iter().enumerate().take(cache_len) {
            let v_start = pos * hidden_dim + head_offset;
            let v_head = &v_cache[v_start..v_start + head_dim];
            for (i, &val) in v_head.iter().enumerate() {
                out_head[i] += weight * val;
            }
        }
        let cur_v_head = &cur_v[head_offset..head_offset + head_dim];
        let cur_weight = scores[cache_len];
        for (i, &val) in cur_v_head.iter().enumerate() {
            out_head[i] += cur_weight * val;
        }
    }
    output
}

/// Full recompute attention benchmark helper — O(n²) without KV cache
fn bench_full_recompute_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * hidden_dim];

    for head in 0..num_heads {
        let head_offset = head * head_dim;
        for i in 0..seq_len {
            let q_start = i * hidden_dim + head_offset;
            let q_head = &q[q_start..q_start + head_dim];

            let mut scores = Vec::with_capacity(i + 1);
            for j in 0..=i {
                let k_start = j * hidden_dim + head_offset;
                let k_head = &k[k_start..k_start + head_dim];
                let score: f32 = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum();
                scores.push(score * scale);
            }
            bench_softmax_inplace(&mut scores);

            let out_start = i * hidden_dim + head_offset;
            let out_head = &mut output[out_start..out_start + head_dim];
            for (j, &weight) in scores.iter().enumerate() {
                let v_start = j * hidden_dim + head_offset;
                let v_head = &v[v_start..v_start + head_dim];
                for (d, &val) in v_head.iter().enumerate() {
                    out_head[d] += weight * val;
                }
            }
        }
    }
    output
}

/// In-place softmax for benchmark attention helpers
fn bench_softmax_inplace(scores: &mut [f32]) {
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_score).exp();
        exp_sum += *s;
    }
    for s in scores.iter_mut() {
        *s /= exp_sum;
    }
}

// ============================================================================
// IMP-102a: End-to-End Generation Benchmark (generate vs generate_with_cache)
// ============================================================================

/// Benchmark comparing full generation with and without KV cache
///
/// This tests the complete inference path including:
/// - Token embedding lookup
/// - Transformer layer processing
/// - LM head projection
/// - Token sampling
fn benchmark_e2e_generation(c: &mut Criterion) {
    use realizar::gguf::{
        OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel, QuantizedGenerateConfig,
    };

    let mut group = c.benchmark_group("imp_102a_e2e_generation");
    group.sample_size(20); // Fewer samples for expensive operations

    // Create a minimal but realistic model configuration
    let hidden_dim = 256;
    let intermediate_dim = 512;
    let num_heads = 4;
    let num_layers = 2;
    let vocab_size = 1000;
    let _head_dim = hidden_dim / num_heads;

    let config = make_bench_config(
        hidden_dim,
        intermediate_dim,
        num_layers,
        num_heads,
        vocab_size,
    );

    // Use shared Q4_K tensor factory (uses d=1.0; stable enough for benchmarking)
    let create_q4k_tensor = make_q4k_tensor;

    // Create token embeddings (vocab_size x hidden_dim)
    let token_embedding: Vec<f32> = (0..(vocab_size * hidden_dim))
        .map(|i| (i as f32 * 0.001).sin() * 0.1)
        .collect();

    // Create transformer layers
    let layers: Vec<OwnedQuantizedLayer> = (0..num_layers)
        .map(|_| OwnedQuantizedLayer {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: OwnedQKVWeights::Fused(create_q4k_tensor(hidden_dim, hidden_dim * 3)),
            qkv_bias: None,
            attn_output_weight: create_q4k_tensor(hidden_dim, hidden_dim),
            attn_output_bias: None,
            ffn_up_weight: create_q4k_tensor(hidden_dim, intermediate_dim),
            ffn_up_bias: None,
            ffn_down_weight: create_q4k_tensor(intermediate_dim, hidden_dim),
            ffn_down_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        })
        .collect();

    let model = OwnedQuantizedModel::new_for_test(
        config.clone(),
        token_embedding,
        layers,
        vec![1.0; hidden_dim],
        None,
        create_q4k_tensor(hidden_dim, vocab_size),
        None,
    );

    // Test different generation lengths
    let gen_configs = [
        (4, 4),   // Short: 4 prompt + 4 generate
        (8, 8),   // Medium: 8 prompt + 8 generate
        (16, 16), // Longer: 16 prompt + 16 generate
    ];

    for (prompt_len, gen_len) in gen_configs {
        let prompt: Vec<u32> = (0..prompt_len).map(|i| (i % vocab_size) as u32).collect();
        let config = QuantizedGenerateConfig::deterministic(gen_len);

        // Benchmark: generate() - O(n²) per token (full recompute)
        group.throughput(Throughput::Elements(gen_len as u64));
        group.bench_with_input(
            BenchmarkId::new("generate_no_cache", format!("p{prompt_len}_g{gen_len}")),
            &(&model, &prompt, &config),
            |b, input: &(&OwnedQuantizedModel, &Vec<u32>, &QuantizedGenerateConfig)| {
                let (m, p, cfg) = input;
                b.iter(|| {
                    let result = m.generate(black_box(*p), black_box(*cfg)).expect("test");
                    black_box(result)
                });
            },
        );

        // Benchmark: generate_with_cache() - O(n) per token (KV cache)
        group.bench_with_input(
            BenchmarkId::new("generate_with_cache", format!("p{prompt_len}_g{gen_len}")),
            &(&model, &prompt, &config),
            |b, input: &(&OwnedQuantizedModel, &Vec<u32>, &QuantizedGenerateConfig)| {
                let (m, p, cfg) = input;
                b.iter(|| {
                    let result = m
                        .generate_with_cache(black_box(*p), black_box(*cfg))
                        .expect("test");
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// IMP-102c: Component-Level Profiling Benchmark
// ============================================================================

/// Benchmark breaking down time spent in each component of generate_with_cache
///
/// This identifies the next bottleneck after KV cache integration by measuring:
/// - Token embedding lookup
/// - Layer norm operations
/// - Fused Q4_K matvec (QKV, output, FFN up, FFN down, LM head)
/// - Attention with cache (scores, softmax, value aggregation)
/// - GELU activation
///
/// Per Toyota Way: Genchi Genbutsu - measure actual component times, not assumptions
fn benchmark_component_profiling(c: &mut Criterion) {
    use std::time::Instant;

    let mut group = c.benchmark_group("imp_102c_component_profiling");
    group.sample_size(50);

    // Model configuration (realistic but small for fast benchmarking)
    let hidden_dim = 512;
    let intermediate_dim = 1024;
    let num_heads = 8;
    let head_dim = hidden_dim / num_heads;
    let vocab_size = 2000;
    let seq_len = 64; // Simulate 64 tokens in KV cache

    // Use shared Q4_K data factory (equivalent to old inline create_q4k_data)
    let create_q4k_data = make_q4k_weights;

    // Component 1: Token embedding lookup
    let embeddings: Vec<f32> = (0..(vocab_size * hidden_dim))
        .map(|i| (i as f32 * 0.001).sin())
        .collect();

    group.bench_function("1_embedding_lookup", |b| {
        let token_id = 42u32;
        b.iter(|| {
            let start = (token_id as usize) * hidden_dim;
            let end = start + hidden_dim;
            let hidden: Vec<f32> = embeddings[start..end].to_vec();
            black_box(hidden)
        });
    });

    // Component 2: Layer norm
    let hidden_vec: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let norm_weight: Vec<f32> = vec![1.0; hidden_dim];
    let eps = 1e-5f32;

    group.bench_function("2_layer_norm", |b| {
        b.iter(|| {
            let input = black_box(&hidden_vec);
            let weight = black_box(&norm_weight);

            // Compute mean
            let mean: f32 = input.iter().sum::<f32>() / input.len() as f32;

            // Compute variance
            let var: f32 =
                input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / input.len() as f32;
            let std = (var + eps).sqrt();

            // Normalize
            let output: Vec<f32> = input
                .iter()
                .zip(weight.iter())
                .map(|(x, w)| ((x - mean) / std) * w)
                .collect();
            black_box(output)
        });
    });

    // Component 3: Fused Q4_K matvec (critical path)
    let qkv_weights = create_q4k_data(hidden_dim, hidden_dim * 3);
    let input_vec: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.001).sin()).collect();

    group.bench_function("3_fused_q4k_qkv_projection", |b| {
        b.iter(|| {
            let result = fused_q4k_parallel_matvec(
                black_box(&qkv_weights),
                black_box(&input_vec),
                hidden_dim,
                hidden_dim * 3,
            )
            .expect("test");
            black_box(result)
        });
    });

    // Component 4: RoPE (rotary position embedding)
    let qk_vec: Vec<f32> = (0..(hidden_dim * 2))
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let position = 32usize;

    group.bench_function("4_rope_apply", |b| {
        b.iter(|| {
            let q = &qk_vec[..hidden_dim];
            let k = &qk_vec[hidden_dim..];
            let pos = black_box(position);

            let mut q_rot = q.to_vec();
            let mut k_rot = k.to_vec();

            // Apply RoPE per head
            for head in 0..num_heads {
                let head_offset = head * head_dim;
                for i in 0..(head_dim / 2) {
                    let theta = 10000.0f32.powf(-2.0 * (i as f32) / (head_dim as f32));
                    let angle = (pos as f32) * theta;
                    let cos_val = angle.cos();
                    let sin_val = angle.sin();

                    let j = head_offset + i;
                    let j2 = head_offset + head_dim / 2 + i;

                    // Rotate Q
                    let q0 = q_rot[j];
                    let q1 = q_rot[j2];
                    q_rot[j] = q0 * cos_val - q1 * sin_val;
                    q_rot[j2] = q0 * sin_val + q1 * cos_val;

                    // Rotate K
                    let k0 = k_rot[j];
                    let k1 = k_rot[j2];
                    k_rot[j] = k0 * cos_val - k1 * sin_val;
                    k_rot[j2] = k0 * sin_val + k1 * cos_val;
                }
            }
            black_box((q_rot, k_rot))
        });
    });

    // Component 5: Attention with cache (main bottleneck candidate)
    let q: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let k_cache: Vec<f32> = (0..(seq_len * hidden_dim))
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let v_cache: Vec<f32> = (0..(seq_len * hidden_dim))
        .map(|i| (i as f32 * 0.002).sin())
        .collect();
    let cur_k: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.003).cos()).collect();
    let cur_v: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.004).sin()).collect();

    group.bench_function("5_attention_with_cache", |b| {
        b.iter(|| {
            let output = bench_cached_attention(
                &q, &k_cache, &v_cache, &cur_k, &cur_v, num_heads, head_dim, hidden_dim, seq_len,
            );
            black_box(output)
        });
    });

    // Component 6: Output projection (fused Q4_K matvec)
    let output_weights = create_q4k_data(hidden_dim, hidden_dim);

    group.bench_function("6_fused_q4k_output_proj", |b| {
        b.iter(|| {
            let result = fused_q4k_parallel_matvec(
                black_box(&output_weights),
                black_box(&input_vec),
                hidden_dim,
                hidden_dim,
            )
            .expect("test");
            black_box(result)
        });
    });

    // Component 7: FFN up projection (fused Q4_K matvec - largest)
    let ffn_up_weights = create_q4k_data(hidden_dim, intermediate_dim);

    group.bench_function("7_fused_q4k_ffn_up", |b| {
        b.iter(|| {
            let result = fused_q4k_parallel_matvec(
                black_box(&ffn_up_weights),
                black_box(&input_vec),
                hidden_dim,
                intermediate_dim,
            )
            .expect("test");
            black_box(result)
        });
    });

    // Component 8: GELU activation
    let ffn_hidden: Vec<f32> = (0..intermediate_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    group.bench_function("8_gelu_activation", |b| {
        b.iter(|| {
            let input = black_box(&ffn_hidden);
            let output: Vec<f32> = input
                .iter()
                .map(|&x| {
                    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    let sqrt_2_pi = 0.7978845608f32;
                    let coeff = 0.044715f32;
                    x * 0.5 * (1.0 + (sqrt_2_pi * (x + coeff * x * x * x)).tanh())
                })
                .collect();
            black_box(output)
        });
    });

    // Component 9: FFN down projection
    let ffn_down_weights = create_q4k_data(intermediate_dim, hidden_dim);
    let ffn_intermediate: Vec<f32> = (0..intermediate_dim)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();

    group.bench_function("9_fused_q4k_ffn_down", |b| {
        b.iter(|| {
            let result = fused_q4k_parallel_matvec(
                black_box(&ffn_down_weights),
                black_box(&ffn_intermediate),
                intermediate_dim,
                hidden_dim,
            )
            .expect("test");
            black_box(result)
        });
    });

    // Component 10: LM head projection (vocab projection - potentially large)
    let lm_head_weights = create_q4k_data(hidden_dim, vocab_size);

    group.bench_function("10_fused_q4k_lm_head", |b| {
        b.iter(|| {
            let result = fused_q4k_parallel_matvec(
                black_box(&lm_head_weights),
                black_box(&input_vec),
                hidden_dim,
                vocab_size,
            )
            .expect("test");
            black_box(result)
        });
    });

    // Summary: Full single-token forward pass (all components)
    group.bench_function("TOTAL_single_token_forward", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                // Embedding lookup
                let token_id = 42usize;
                let emb_start = token_id * hidden_dim;
                let _hidden: Vec<f32> = embeddings[emb_start..emb_start + hidden_dim].to_vec();

                // Layer norm
                let mean: f32 = input_vec.iter().sum::<f32>() / hidden_dim as f32;
                let var: f32 =
                    input_vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden_dim as f32;
                let _normed: Vec<f32> = input_vec
                    .iter()
                    .zip(norm_weight.iter())
                    .map(|(x, w)| ((x - mean) / (var + eps).sqrt()) * w)
                    .collect();

                // QKV projection
                let _qkv =
                    fused_q4k_parallel_matvec(&qkv_weights, &input_vec, hidden_dim, hidden_dim * 3)
                        .expect("test");

                // Attention (simplified)
                let _output = bench_cached_attention(
                    &q, &k_cache, &v_cache, &cur_k, &cur_v, num_heads, head_dim, hidden_dim,
                    seq_len,
                );

                // Output projection
                let _out =
                    fused_q4k_parallel_matvec(&output_weights, &input_vec, hidden_dim, hidden_dim)
                        .expect("test");

                // FFN up
                let _ffn_up = fused_q4k_parallel_matvec(
                    &ffn_up_weights,
                    &input_vec,
                    hidden_dim,
                    intermediate_dim,
                )
                .expect("test");

                // FFN down
                let _ffn_down = fused_q4k_parallel_matvec(
                    &ffn_down_weights,
                    &ffn_intermediate,
                    intermediate_dim,
                    hidden_dim,
                )
                .expect("test");

                // LM head
                let _logits =
                    fused_q4k_parallel_matvec(&lm_head_weights, &input_vec, hidden_dim, vocab_size)
                        .expect("test");

                black_box(());
            }
            start.elapsed()
        });
    });

    group.finish();
}

// IMP-103a: SIMD-Optimized Q4_K Matvec — included in Q4K_BENCH_SUITE

/// Create a benchmark model with proper Q4_K weights
/// NOTE: Disabled - new_for_benchmark method doesn't exist
#[allow(dead_code)]
fn create_benchmark_model(
    _config: &realizar::gguf::GGUFConfig,
) -> realizar::gguf::OwnedQuantizedModel {
    unimplemented!("TODO: Update to use OwnedQuantizedModel struct constructor")
}

/// Create Q4_K benchmark data (delegates to shared make_q4k_tensor helper)
fn create_bench_q4k_data(in_dim: usize, out_dim: usize) -> realizar::gguf::OwnedQuantizedTensor {
    make_q4k_tensor(in_dim, out_dim)
}

// ============================================================================
// IMP-107..120: Unified GPU Benchmark Suite (Kaizen: DRY DataTransformation)
// ============================================================================

/// CPU reference matmul for benchmark comparison
#[cfg(feature = "gpu")]
#[allow(clippy::many_single_char_names)]
fn cpu_matmul_bench(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// CPU reference implementation of causal attention for benchmarking
#[cfg(feature = "gpu")]
fn causal_attention_cpu_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
) -> Vec<f32> {
    let head_dim = hidden_dim / num_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * hidden_dim];

    for head in 0..num_heads {
        let head_offset = head * head_dim;

        for i in 0..seq_len {
            let mut scores = Vec::with_capacity(i + 1);
            let q_start = i * hidden_dim + head_offset;

            for j in 0..=i {
                let k_start = j * hidden_dim + head_offset;
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[q_start + d] * k[k_start + d];
                }
                scores.push(score * scale);
            }

            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                exp_sum += *s;
            }
            for s in &mut scores {
                *s /= exp_sum;
            }

            // Weighted sum
            let out_start = i * hidden_dim + head_offset;
            for (j, &weight) in scores.iter().enumerate() {
                let v_start = j * hidden_dim + head_offset;
                for d in 0..head_dim {
                    output[out_start + d] += weight * v[v_start + d];
                }
            }
        }
    }

    output
}

/// Create benchmark model with config
/// NOTE: Disabled - new_for_benchmark method doesn't exist
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn create_bench_model_with_config(
    _config: &realizar::gguf::GGUFConfig,
) -> realizar::gguf::OwnedQuantizedModel {
    unimplemented!("TODO: Update to use OwnedQuantizedModel struct constructor")
}

/// GPU benchmark variant descriptor for table-driven dispatch.
///
/// Each variant captures a distinct GPU benchmark pattern (IMP-107 through IMP-120).
/// The dispatcher `run_gpu_bench` eliminates the repeated DataTransformation pattern:
/// create data -> configure group -> run benchmark -> assert results.
#[cfg(feature = "gpu")]
enum GpuBenchOp {
    /// IMP-107: GPU vs CPU matmul crossover (cpu vs hybrid scheduler).
    /// Configs: (batch_size/m, k, n).
    BatchMatmul {
        cases: &'static [(usize, usize, usize)],
    },
    /// IMP-108: Batched causal attention vs sequential CPU reference.
    /// Configs: seq_lengths to iterate; uses fixed hidden_dim=256, num_heads=8.
    BatchedCausalAttention { seq_lengths: &'static [usize] },
    /// IMP-109: Fused dequant+matmul vs separate operations.
    /// Configs: (batch_size, hidden_dim, intermediate_dim, label).
    FusedBatchMatmul {
        cases: &'static [(usize, usize, usize, &'static str)],
    },
    /// IMP-110: Parallel vs sequential multi-head attention.
    /// Configs: (seq_len, hidden_dim, num_heads, label).
    ParallelMultiheadAttention {
        cases: &'static [(usize, usize, usize, &'static str)],
    },
    /// IMP-111: Tiled vs standard single-head attention.
    /// Configs: (seq_len, hidden_dim, num_heads, tile_size, label).
    TiledAttention {
        cases: &'static [(usize, usize, usize, usize, &'static str)],
    },
    /// IMP-112: Scheduler caching vs uncached forward pass.
    SchedulerCaching,
    /// IMP-113: Single-dispatch vs multi-dispatch multi-head attention.
    /// Configs: (seq_len, hidden_dim, num_heads, label).
    SingleDispatchAttention {
        cases: &'static [(usize, usize, usize, &'static str)],
    },
    /// IMP-114: Flattened vs loop-based batched GEMM.
    /// Configs: (batch_size, m, k, n, label).
    FlattenedBatchedGemm {
        cases: &'static [(usize, usize, usize, usize, &'static str)],
    },
    /// IMP-115: Fused kernel attention vs separate matmul+softmax+matmul.
    /// Configs: (num_heads, seq_len, head_dim, label).
    FusedKernelAttention {
        cases: &'static [(usize, usize, usize, &'static str)],
    },
    /// IMP-120: GPU vs CPU fused attention crossover point.
    /// Configs: seq_lengths to iterate; fixed head_dim=64, num_heads=8.
    GpuCpuCrossover { seq_lengths: &'static [usize] },
}

/// Table-driven GPU benchmark entry.
#[cfg(feature = "gpu")]
struct GpuBenchEntry {
    group_name: &'static str,
    sample_size: usize,
    op: GpuBenchOp,
}

/// Benchmark tiled vs standard single-head attention (IMP-111).
///
/// For each (seq_len, hidden_dim, num_heads, tile_size, label) case, runs three
/// variants: standard, tiled, and tiled_causal attention.
#[cfg(feature = "gpu")]
fn bench_tiled_attention(
    group: &mut BenchmarkGroup<WallTime>,
    cases: &[(usize, usize, usize, usize, &str)],
) {
    for &(seq_len, hidden_dim, num_heads, tile_size, label) in cases {
        let config = make_bench_config(hidden_dim, hidden_dim * 2, 1, num_heads, 100);
        let model = create_bench_model_with_config(&config);
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (q, k, v) = make_qkv_vecs(seq_len, head_dim);

        let ops = seq_len * seq_len * head_dim * 2;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("standard", label),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = model
                        .standard_single_head_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                            head_dim,
                            scale,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tiled", label),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = model
                        .tiled_single_head_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                            head_dim,
                            scale,
                            tile_size,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tiled_causal", label),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = model
                        .tiled_causal_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                            head_dim,
                            scale,
                            tile_size,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );
    }
}

/// Benchmark scheduler caching vs uncached forward pass (IMP-112).
///
/// Compares uncached, cached, and 5x-cached forward passes using a small model.
#[cfg(feature = "gpu")]
fn bench_scheduler_caching(group: &mut BenchmarkGroup<WallTime>) {
    use realizar::gguf::OwnedQuantizedModelCached;

    let config = make_bench_config(64, 128, 1, 4, 100);
    let model = create_bench_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model.clone());
    let tokens = vec![1u32, 5, 10, 20];

    group.throughput(Throughput::Elements(
        (tokens.len() * config.vocab_size) as u64,
    ));

    // Warm up cached model (initialize scheduler)
    let _ = cached_model
        .forward_batch_gpu_cached(&tokens)
        .expect("test");

    group.bench_function("uncached_forward", |bencher| {
        bencher.iter(|| {
            let result = model.forward_batch_gpu(black_box(&tokens)).expect("test");
            black_box(result)
        });
    });

    group.bench_function("cached_forward", |bencher| {
        bencher.iter(|| {
            let result = cached_model
                .forward_batch_gpu_cached(black_box(&tokens))
                .expect("test");
            black_box(result)
        });
    });

    group.bench_function("cached_5x_forward", |bencher| {
        bencher.iter(|| {
            for _ in 0..5 {
                let result = cached_model
                    .forward_batch_gpu_cached(black_box(&tokens))
                    .expect("test");
                black_box(&result);
            }
        });
    });
}

/// Benchmark GPU vs CPU fused attention crossover (IMP-120).
///
/// For each sequence length, benchmarks cpu_fused, gpu_fused, and adaptive
/// attention to find the crossover point.
#[cfg(feature = "gpu")]
fn bench_gpu_cpu_crossover(group: &mut BenchmarkGroup<WallTime>, seq_lengths: &[usize]) {
    use realizar::gguf::OwnedQuantizedModelCached;

    let head_dim = 64;
    let num_heads = 8;
    let hidden_dim = num_heads * head_dim;

    let config = make_bench_config(hidden_dim, hidden_dim * 4, 1, num_heads, 100);
    let model = create_bench_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    for &seq_len in seq_lengths {
        let q: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let k: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
            .collect();
        let v: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
            .collect();

        let scale = 1.0 / (head_dim as f32).sqrt();
        let ops = seq_len * seq_len * head_dim * 3 + seq_len * seq_len;
        group.throughput(Throughput::Elements(ops as u64));

        // Warm up both paths
        let _ = cached_model.fused_causal_attention(&q, &k, &v, seq_len, head_dim, scale);
        let _ = cached_model.gpu_fused_causal_attention(&q, &k, &v, seq_len, head_dim, scale);

        group.bench_with_input(
            BenchmarkId::new("cpu_fused", format!("seq{seq_len}")),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = cached_model
                        .fused_causal_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                            head_dim,
                            scale,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("gpu_fused", format!("seq{seq_len}")),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = cached_model
                        .gpu_fused_causal_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                            head_dim,
                            scale,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("adaptive", format!("seq{seq_len}")),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = cached_model
                        .adaptive_fused_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                            head_dim,
                            scale,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );
    }
}

/// Benchmark fused dequant+matmul vs separate operations (IMP-109).
///
/// For each (batch_size, hidden_dim, intermediate_dim, label) case, runs fused,
/// separate_predequant, and separate_redequant benchmarks.
#[cfg(feature = "gpu")]
fn bench_fused_batch_matmul(
    group: &mut BenchmarkGroup<WallTime>,
    cases: &[(usize, usize, usize, &str)],
) {
    use realizar::gpu::HybridScheduler;
    use realizar::quantize::{dequantize_q4_k_simd as dequant_simd, QK_K};

    for &(batch_size, hidden_dim, intermediate_dim, label) in cases {
        let config = make_bench_config(hidden_dim, intermediate_dim, 1, 4, 100);
        let model = create_bench_model_with_config(&config);

        let activations: Vec<f32> = (0..batch_size * hidden_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();

        let weight = &model.layers()[0].ffn_up_weight;

        // Pre-dequantize weight for "separate" benchmark
        let weight_f32 = {
            let in_dim = weight.in_dim;
            let out_dim = weight.out_dim;
            let super_blocks_per_row = in_dim.div_ceil(QK_K);
            let mut output = Vec::with_capacity(in_dim * out_dim);
            for row in 0..out_dim {
                let row_start = row * super_blocks_per_row * 144;
                let row_end = row_start + super_blocks_per_row * 144;
                let row_data = &weight.data[row_start..row_end];
                let row_dequant = dequant_simd(row_data).expect("test");
                output.extend_from_slice(&row_dequant[..in_dim.min(row_dequant.len())]);
            }
            output
        };

        group.throughput(Throughput::Elements(
            (batch_size * hidden_dim * intermediate_dim) as u64,
        ));

        group.bench_with_input(
            BenchmarkId::new("fused", label),
            &(&activations, weight),
            |bencher, (act, w)| {
                bencher.iter(|| {
                    let result = model
                        .fused_batch_matmul_gpu(black_box(act), black_box(w), batch_size)
                        .expect("test");
                    black_box(result)
                });
            },
        );

        let weight_clone = weight_f32.clone();
        group.bench_with_input(
            BenchmarkId::new("separate_predequant", label),
            &(&activations, &weight_clone),
            |bencher, (act, w)| {
                bencher.iter(|| {
                    let mut scheduler = HybridScheduler::with_threshold(1000).expect("test");
                    let result = scheduler
                        .matmul(
                            black_box(act),
                            black_box(w),
                            batch_size,
                            hidden_dim,
                            intermediate_dim,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("separate_redequant", label),
            &(&activations, weight),
            |bencher, (act, w)| {
                bencher.iter(|| {
                    let in_dim = w.in_dim;
                    let out_dim = w.out_dim;
                    let super_blocks_per_row = in_dim.div_ceil(QK_K);
                    let mut w_f32 = Vec::with_capacity(in_dim * out_dim);
                    for row in 0..out_dim {
                        let row_start = row * super_blocks_per_row * 144;
                        let row_end = row_start + super_blocks_per_row * 144;
                        let row_data = &w.data[row_start..row_end];
                        let row_dequant = dequant_simd(row_data).expect("test");
                        w_f32.extend_from_slice(&row_dequant[..in_dim.min(row_dequant.len())]);
                    }

                    let mut scheduler = HybridScheduler::with_threshold(1000).expect("test");
                    let result = scheduler
                        .matmul(black_box(act), &w_f32, batch_size, in_dim, out_dim)
                        .expect("test");
                    black_box(result)
                });
            },
        );
    }
}

/// Benchmark single-dispatch vs multi-dispatch attention (IMP-113).
///
/// For each (seq_len, hidden_dim, num_heads, label) case, runs multi_dispatch
/// and single_dispatch benchmarks with a warmed-up cached model.
#[cfg(feature = "gpu")]
fn bench_single_dispatch_attention(
    group: &mut BenchmarkGroup<WallTime>,
    cases: &[(usize, usize, usize, &str)],
) {
    use realizar::gguf::OwnedQuantizedModelCached;

    for &(seq_len, hidden_dim, num_heads, label) in cases {
        let config = make_bench_config(hidden_dim, hidden_dim * 2, 1, num_heads, 100);
        let model = create_bench_model_with_config(&config);
        let cached_model = OwnedQuantizedModelCached::new(model.clone());
        let (q, k, v) = make_qkv_vecs(seq_len, hidden_dim);

        let head_dim = hidden_dim / num_heads;
        let ops = num_heads * seq_len * seq_len * head_dim * 2;
        group.throughput(Throughput::Elements(ops as u64));

        // Warm up cached model
        let _ = cached_model.parallel_multihead_attention_gpu_cached(&q, &k, &v, seq_len);

        group.bench_with_input(
            BenchmarkId::new("multi_dispatch", label),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = cached_model
                        .parallel_multihead_attention_gpu_cached(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("single_dispatch", label),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = cached_model
                        .single_dispatch_multihead_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );
    }
}

/// Benchmark fused kernel attention vs separate operations (IMP-115).
///
/// For each (num_heads, seq_len, head_dim, label) case, runs separate_ops
/// and fused_kernel benchmarks with a warmed-up cached model.
#[cfg(feature = "gpu")]
fn bench_fused_kernel_attention(
    group: &mut BenchmarkGroup<WallTime>,
    cases: &[(usize, usize, usize, &str)],
) {
    use realizar::gguf::OwnedQuantizedModelCached;

    for &(num_heads, seq_len, head_dim, label) in cases {
        let hidden_dim = num_heads * head_dim;
        let config = make_bench_config(hidden_dim, hidden_dim * 4, 1, num_heads, 100);
        let model = create_bench_model_with_config(&config);
        let cached_model = OwnedQuantizedModelCached::new(model);
        let (q, k, v) = make_qkv_vecs(seq_len, hidden_dim);

        let ops = num_heads * seq_len * seq_len * head_dim * 4;
        group.throughput(Throughput::Elements(ops as u64));

        // Warm up
        let _ = cached_model.flattened_multihead_attention(&q, &k, &v, seq_len);

        group.bench_with_input(
            BenchmarkId::new("separate_ops", label),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = cached_model
                        .flattened_multihead_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fused_kernel", label),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = cached_model
                        .fused_multihead_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );
    }
}

/// Run a GPU benchmark from a `GpuBenchOp` descriptor.
///
/// This single dispatcher replaces 11 separate `benchmark_*` functions,
/// eliminating the repeated DataTransformation pattern
/// (create data -> configure group -> run benchmark -> assert results).
#[cfg(feature = "gpu")]
fn run_gpu_bench(c: &mut Criterion, entry: &GpuBenchEntry) {
    use realizar::gguf::OwnedQuantizedModelCached;
    use realizar::gpu::HybridScheduler;

    let mut group = c.benchmark_group(entry.group_name);
    group.sample_size(entry.sample_size);

    match &entry.op {
        // IMP-107: GPU vs CPU matmul crossover
        GpuBenchOp::BatchMatmul { cases } => {
            for &(batch_size, k, n) in *cases {
                let m = batch_size;
                let ops = m * k * n;
                let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
                let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();

                group.throughput(Throughput::Elements(ops as u64));

                group.bench_with_input(
                    BenchmarkId::new("cpu", format!("{m}x{k}x{n}")),
                    &(&a, &b),
                    |bencher, (a, b)| {
                        bencher.iter(|| {
                            let result = cpu_matmul_bench(black_box(a), black_box(b), m, k, n);
                            black_box(result)
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("hybrid", format!("{m}x{k}x{n}")),
                    &(&a, &b),
                    |bencher, (a, b)| {
                        let mut scheduler = HybridScheduler::with_threshold(1000).expect("test");
                        bencher.iter(|| {
                            let result = scheduler
                                .matmul(black_box(a), black_box(b), m, k, n)
                                .expect("test");
                            black_box(result)
                        });
                    },
                );
            }
        },

        // IMP-108: Batched causal attention vs sequential
        GpuBenchOp::BatchedCausalAttention { seq_lengths } => {
            for &seq_len in *seq_lengths {
                let config = make_bench_config(256, 512, 1, 8, 1000);
                let model = create_bench_model_with_config(&config);
                let hidden_dim = config.hidden_dim;
                let num_heads = config.num_heads;
                let (q, k, v) = make_qkv_vecs(seq_len, hidden_dim);

                group.throughput(Throughput::Elements(
                    (seq_len * seq_len * hidden_dim) as u64,
                ));

                group.bench_with_input(
                    BenchmarkId::new("cpu_sequential", seq_len),
                    &(&q, &k, &v),
                    |bencher, (q, k, v)| {
                        bencher.iter(|| {
                            let result = causal_attention_cpu_ref(
                                black_box(q),
                                black_box(k),
                                black_box(v),
                                seq_len,
                                hidden_dim,
                                num_heads,
                            );
                            black_box(result)
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("batched_gpu", seq_len),
                    &(&q, &k, &v),
                    |bencher, (q, k, v)| {
                        bencher.iter(|| {
                            let result = model
                                .batched_causal_attention_gpu(
                                    black_box(q),
                                    black_box(k),
                                    black_box(v),
                                    seq_len,
                                )
                                .expect("test");
                            black_box(result)
                        });
                    },
                );
            }
        },

        // IMP-109: Fused dequant+matmul vs separate
        GpuBenchOp::FusedBatchMatmul { cases } => {
            bench_fused_batch_matmul(&mut group, cases);
        },

        // IMP-110: Parallel vs sequential multi-head attention
        GpuBenchOp::ParallelMultiheadAttention { cases } => {
            for &(seq_len, hidden_dim, num_heads, label) in *cases {
                let config = make_bench_config(hidden_dim, hidden_dim * 2, 1, num_heads, 100);
                let model = create_bench_model_with_config(&config);
                let (q, k, v) = make_qkv_vecs(seq_len, hidden_dim);

                let head_dim = hidden_dim / num_heads;
                let ops = num_heads * seq_len * seq_len * head_dim * 2;
                group.throughput(Throughput::Elements(ops as u64));

                group.bench_with_input(
                    BenchmarkId::new("sequential", label),
                    &(&q, &k, &v),
                    |bencher, (q, k, v)| {
                        bencher.iter(|| {
                            let result = model
                                .batched_causal_attention_gpu(
                                    black_box(q),
                                    black_box(k),
                                    black_box(v),
                                    seq_len,
                                )
                                .expect("test");
                            black_box(result)
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("parallel", label),
                    &(&q, &k, &v),
                    |bencher, (q, k, v)| {
                        bencher.iter(|| {
                            let result = model
                                .parallel_multihead_attention_gpu(
                                    black_box(q),
                                    black_box(k),
                                    black_box(v),
                                    seq_len,
                                )
                                .expect("test");
                            black_box(result)
                        });
                    },
                );
            }
        },

        // IMP-111: Tiled vs standard single-head attention
        GpuBenchOp::TiledAttention { cases } => {
            bench_tiled_attention(&mut group, cases);
        },

        // IMP-112: Scheduler caching vs uncached
        GpuBenchOp::SchedulerCaching => {
            bench_scheduler_caching(&mut group);
        },

        // IMP-113: Single-dispatch vs multi-dispatch attention
        GpuBenchOp::SingleDispatchAttention { cases } => {
            bench_single_dispatch_attention(&mut group, cases);
        },

        // IMP-114: Flattened vs loop-based batched GEMM
        GpuBenchOp::FlattenedBatchedGemm { cases } => {
            for &(batch_size, m, k, n, label) in *cases {
                let config = make_bench_config(64, 128, 1, 4, 100);
                let model = create_bench_model_with_config(&config);
                let cached_model = OwnedQuantizedModelCached::new(model);

                let batched_a: Vec<f32> = (0..batch_size * m * k)
                    .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
                    .collect();
                let batched_b: Vec<f32> = (0..batch_size * k * n)
                    .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
                    .collect();

                let ops = batch_size * m * k * n * 2;
                group.throughput(Throughput::Elements(ops as u64));

                // Warm up
                let _ = cached_model
                    .batched_gemm_single_dispatch(&batched_a, &batched_b, batch_size, m, k, n);

                group.bench_with_input(
                    BenchmarkId::new("loop_based", label),
                    &(&batched_a, &batched_b),
                    |bencher, (a, b)| {
                        bencher.iter(|| {
                            let result = cached_model
                                .batched_gemm_single_dispatch(
                                    black_box(a),
                                    black_box(b),
                                    batch_size,
                                    m,
                                    k,
                                    n,
                                )
                                .expect("test");
                            black_box(result)
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("flattened", label),
                    &(&batched_a, &batched_b),
                    |bencher, (a, b)| {
                        bencher.iter(|| {
                            let result = cached_model
                                .flattened_batched_gemm(
                                    black_box(a),
                                    black_box(b),
                                    batch_size,
                                    m,
                                    k,
                                    n,
                                )
                                .expect("test");
                            black_box(result)
                        });
                    },
                );
            }
        },

        // IMP-115: Fused kernel attention vs separate operations
        GpuBenchOp::FusedKernelAttention { cases } => {
            bench_fused_kernel_attention(&mut group, cases);
        },

        // IMP-120: GPU vs CPU fused attention crossover
        GpuBenchOp::GpuCpuCrossover { seq_lengths } => {
            bench_gpu_cpu_crossover(&mut group, seq_lengths);
        },
    }

    group.finish();
}

// Static config tables for GPU benchmark suite entries.
#[cfg(feature = "gpu")]
static GPU_BATCH_MATMUL_CASES: &[(usize, usize, usize)] = &[
    (1, 256, 256),
    (1, 512, 512),
    (4, 256, 256),
    (8, 256, 512),
    (16, 512, 512),
    (32, 512, 1024),
];

#[cfg(feature = "gpu")]
static GPU_BATCHED_CAUSAL_SEQ_LENGTHS: &[usize] = &[4, 8, 16, 32, 64];

#[cfg(feature = "gpu")]
static GPU_FUSED_BATCH_MATMUL_CASES: &[(usize, usize, usize, &str)] = &[
    (4, 256, 512, "small_4x256x512"),
    (8, 256, 512, "small_8x256x512"),
    (4, 512, 1024, "medium_4x512x1024"),
    (8, 512, 1024, "medium_8x512x1024"),
    (16, 256, 512, "batch_16x256x512"),
    (32, 256, 512, "batch_32x256x512"),
];

#[cfg(feature = "gpu")]
static GPU_PARALLEL_MHA_CASES: &[(usize, usize, usize, &str)] = &[
    (4, 64, 4, "seq4_h4"),
    (8, 64, 4, "seq8_h4"),
    (16, 64, 4, "seq16_h4"),
    (4, 128, 8, "seq4_h8"),
    (8, 128, 8, "seq8_h8"),
    (16, 128, 8, "seq16_h8"),
    (32, 256, 8, "seq32_h8"),
];

#[cfg(feature = "gpu")]
static GPU_TILED_ATTENTION_CASES: &[(usize, usize, usize, usize, &str)] = &[
    (32, 64, 4, 8, "seq32_tile8"),
    (64, 64, 4, 8, "seq64_tile8"),
    (64, 64, 4, 16, "seq64_tile16"),
    (128, 64, 4, 16, "seq128_tile16"),
    (128, 64, 4, 32, "seq128_tile32"),
    (256, 64, 4, 32, "seq256_tile32"),
];

#[cfg(feature = "gpu")]
static GPU_SINGLE_DISPATCH_CASES: &[(usize, usize, usize, &str)] = &[
    (8, 64, 4, "seq8_h4"),
    (16, 64, 4, "seq16_h4"),
    (16, 128, 8, "seq16_h8"),
    (32, 128, 8, "seq32_h8"),
    (32, 256, 8, "seq32_h8_hd256"),
];

#[cfg(feature = "gpu")]
static GPU_FLATTENED_GEMM_CASES: &[(usize, usize, usize, usize, &str)] = &[
    (4, 8, 16, 8, "b4_m8_k16_n8"),
    (8, 16, 8, 16, "b8_m16_k8_n16"),
    (8, 32, 16, 32, "b8_m32_k16_n32"),
    (16, 16, 8, 16, "b16_m16_k8_n16"),
    (16, 8, 8, 8, "b16_m8_k8_n8"),
];

#[cfg(feature = "gpu")]
static GPU_FUSED_KERNEL_ATTENTION_CASES: &[(usize, usize, usize, &str)] = &[
    (4, 8, 16, "h4_seq8_d16"),
    (8, 8, 16, "h8_seq8_d16"),
    (8, 16, 16, "h8_seq16_d16"),
    (8, 32, 16, "h8_seq32_d16"),
];

#[cfg(feature = "gpu")]
static GPU_CPU_CROSSOVER_SEQ_LENGTHS: &[usize] = &[8, 16, 32, 64, 128, 256];

/// GPU benchmark suite table: all GPU entries driven by `run_gpu_bench`.
#[cfg(feature = "gpu")]
static GPU_BENCH_SUITE: &[GpuBenchEntry] = &[
    GpuBenchEntry {
        group_name: "gpu_batch_matmul",
        sample_size: 50,
        op: GpuBenchOp::BatchMatmul {
            cases: GPU_BATCH_MATMUL_CASES,
        },
    },
    GpuBenchEntry {
        group_name: "batched_causal_attention",
        sample_size: 30,
        op: GpuBenchOp::BatchedCausalAttention {
            seq_lengths: GPU_BATCHED_CAUSAL_SEQ_LENGTHS,
        },
    },
    GpuBenchEntry {
        group_name: "fused_batch_matmul_imp109",
        sample_size: 30,
        op: GpuBenchOp::FusedBatchMatmul {
            cases: GPU_FUSED_BATCH_MATMUL_CASES,
        },
    },
    GpuBenchEntry {
        group_name: "parallel_multihead_attention_imp110",
        sample_size: 30,
        op: GpuBenchOp::ParallelMultiheadAttention {
            cases: GPU_PARALLEL_MHA_CASES,
        },
    },
    GpuBenchEntry {
        group_name: "tiled_attention_imp111",
        sample_size: 30,
        op: GpuBenchOp::TiledAttention {
            cases: GPU_TILED_ATTENTION_CASES,
        },
    },
    GpuBenchEntry {
        group_name: "scheduler_caching_imp112",
        sample_size: 20,
        op: GpuBenchOp::SchedulerCaching,
    },
    GpuBenchEntry {
        group_name: "single_dispatch_attention_imp113",
        sample_size: 30,
        op: GpuBenchOp::SingleDispatchAttention {
            cases: GPU_SINGLE_DISPATCH_CASES,
        },
    },
    GpuBenchEntry {
        group_name: "flattened_batched_gemm_imp114",
        sample_size: 30,
        op: GpuBenchOp::FlattenedBatchedGemm {
            cases: GPU_FLATTENED_GEMM_CASES,
        },
    },
    GpuBenchEntry {
        group_name: "fused_kernel_attention_imp115",
        sample_size: 30,
        op: GpuBenchOp::FusedKernelAttention {
            cases: GPU_FUSED_KERNEL_ATTENTION_CASES,
        },
    },
    GpuBenchEntry {
        group_name: "gpu_cpu_crossover_imp120",
        sample_size: 20,
        op: GpuBenchOp::GpuCpuCrossover {
            seq_lengths: GPU_CPU_CROSSOVER_SEQ_LENGTHS,
        },
    },
];

/// Run all GPU benchmarks from the suite table (Kaizen: single entry point).
#[cfg(feature = "gpu")]
fn benchmark_gpu_suite(c: &mut Criterion) {
    for entry in GPU_BENCH_SUITE {
        run_gpu_bench(c, entry);
    }
}

// ============================================================================
// Criterion Groups
// ============================================================================

#[cfg(feature = "gpu")]
criterion_group!(
    benches,
    benchmark_q4k_suite,
    benchmark_layer_suite,
    benchmark_cache_suite,
    benchmark_e2e_generation,
    benchmark_component_profiling,
    benchmark_gpu_suite,
);

#[cfg(not(feature = "gpu"))]
criterion_group!(
    benches,
    benchmark_q4k_suite,
    benchmark_layer_suite,
    benchmark_cache_suite,
    benchmark_e2e_generation,
    benchmark_component_profiling,
);

criterion_main!(benches);
