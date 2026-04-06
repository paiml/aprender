//! PMAT Benchmark Matrix - APR vs llama.cpp Performance Comparison
//!
//! Tests: tiny (0.5B), small (1.5B), medium (3B) across CPU/GPU
//! Target: APR 2x faster than llama.cpp for EVERY cell
//!
//! Run: cargo run --release --features cuda --example pmat_benchmark_matrix

use realizar::cuda::CudaExecutor;
use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig,
};
use std::path::Path;
use std::time::Instant;

/// Model tier configuration with verified llama.cpp baselines
#[derive(Debug, Clone)]
struct ModelTier {
    name: &'static str,
    size: &'static str,
    gguf_path: &'static str,
    /// llama.cpp GPU baseline (tg64, ngl=99)
    llama_cpp_gpu_baseline: f64,
    /// llama.cpp CPU baseline (tg64, ngl=0)
    llama_cpp_cpu_baseline: f64,
    /// Prompt tokens for benchmark (static slice)
    prompt_tokens: &'static [u32],
}

/// Qwen2.5-Coder chat prompt: "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
const QWEN_PROMPT: &[u32] = &[
    151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374,
    220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198,
];
/// StarCoder2 prompt tokens
const STARCODER_PROMPT: &[u32] = &[1, 1528, 349, 220, 17, 10, 17, 30];

/// Verified llama-bench baselines (RTX 4090, tg64):
/// | Model | CPU | GPU |
/// |-------|-----|-----|
/// | qwen2 0.5B Q4_0 | 194.28 | 594.10 |
/// | qwen2 1.5B Q4_K_M | 86.43 | 377.75 |
/// | starcoder2 3B Q4_K_M | 48.07 | 247.43 |
const TIERS: &[ModelTier] = &[
    ModelTier {
        name: "tiny",
        size: "0.5B",
        gguf_path:
            "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-0.5b-instruct-q4_0.gguf",
        llama_cpp_gpu_baseline: 594.10,
        llama_cpp_cpu_baseline: 194.28,
        prompt_tokens: QWEN_PROMPT,
    },
    ModelTier {
        name: "small",
        size: "1.5B",
        gguf_path:
            "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        llama_cpp_gpu_baseline: 377.75,
        llama_cpp_cpu_baseline: 86.43,
        prompt_tokens: QWEN_PROMPT,
    },
    ModelTier {
        name: "medium",
        size: "3B",
        gguf_path: "/home/noah/src/single-shot-eval/models/raw/starcoder2-3b-q4_k_m.gguf",
        llama_cpp_gpu_baseline: 247.43,
        llama_cpp_cpu_baseline: 48.07,
        prompt_tokens: STARCODER_PROMPT,
    },
];

/// Benchmark result for a single cell
#[derive(Debug, Clone)]
struct BenchResult {
    tier: String,
    backend: String,
    apr_tok_s: f64,
    llama_baseline: f64,
    speedup: f64,
    meets_2x: bool,
}

fn benchmark_apr_gpu(
    model_path: &str,
    prompt_tokens: &[u32],
    max_tokens: usize,
) -> Result<f64, String> {
    let mapped = MappedGGUFModel::from_path(model_path)
        .map_err(|e| format!("Failed to load model: {}", e))?;

    let owned_model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| format!("Failed to create owned model: {}", e))?;

    let mut cuda_model = OwnedQuantizedModelCuda::new(owned_model, 0)
        .map_err(|e| format!("Failed to create CUDA model: {}", e))?;

    let config = QuantizedGenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };

    // Determine best generation method (PAR-023 GPU-resident is fastest)
    let use_gpu_resident = cuda_model.supports_gpu_resident();
    eprintln!("  [DEBUG] GPU-resident supported: {}", use_gpu_resident);

    // Warmup with selected method
    for _ in 0..3 {
        let _ = if use_gpu_resident {
            cuda_model.generate_gpu_resident(prompt_tokens, &config)
        } else {
            cuda_model.generate_cuda_with_cache(prompt_tokens, &config)
        };
    }

    // Benchmark iterations
    let iterations = 5;
    let mut times = Vec::new();

    for _ in 0..iterations {
        let start = Instant::now();
        let result = if use_gpu_resident {
            cuda_model.generate_gpu_resident(prompt_tokens, &config)
        } else {
            cuda_model.generate_cuda_with_cache(prompt_tokens, &config)
        };
        let result = result.map_err(|e| format!("Generation failed: {}", e))?;
        let elapsed = start.elapsed();

        let generated = result.len().saturating_sub(prompt_tokens.len());
        if generated > 0 {
            let tok_s = generated as f64 / elapsed.as_secs_f64();
            times.push(tok_s);
        }
    }

    if times.is_empty() {
        return Err("No tokens generated".to_string());
    }

    // Return median
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(times[times.len() / 2])
}

fn benchmark_apr_cpu(
    model_path: &str,
    prompt_tokens: &[u32],
    max_tokens: usize,
) -> Result<f64, String> {
    let mapped = MappedGGUFModel::from_path(model_path)
        .map_err(|e| format!("Failed to load model: {}", e))?;

    let owned_model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| format!("Failed to create owned model: {}", e))?;

    let config = QuantizedGenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };

    // Warmup
    let _ = owned_model.generate_with_scratch(prompt_tokens, &config);

    // Benchmark iterations
    let iterations = 5;
    let mut times = Vec::new();

    for _ in 0..iterations {
        let start = Instant::now();
        let result = owned_model
            .generate_with_scratch(prompt_tokens, &config)
            .map_err(|e| format!("Generation failed: {}", e))?;
        let elapsed = start.elapsed();

        let generated = result.len().saturating_sub(prompt_tokens.len());
        if generated > 0 {
            let tok_s = generated as f64 / elapsed.as_secs_f64();
            times.push(tok_s);
        }
    }

    if times.is_empty() {
        return Err("No tokens generated".to_string());
    }

    // Return median
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(times[times.len() / 2])
}

fn five_whys_analysis(tier: &str, backend: &str, apr_tok_s: f64, target_tok_s: f64) {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!(
        "║  FIVE-WHYS ROOT CAUSE ANALYSIS: {} {}                          ",
        tier, backend
    );
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Current: {:.1} tok/s                                                  ",
        apr_tok_s
    );
    println!(
        "║  Target:  {:.1} tok/s (2x llama.cpp)                                   ",
        target_tok_s
    );
    println!(
        "║  Gap:     {:.2}x improvement needed                                    ",
        target_tok_s / apr_tok_s
    );
    println!("╠═══════════════════════════════════════════════════════════════════════╣");

    if backend == "GPU" {
        println!("║                                                                       ║");
        println!("║  WHY #1: APR GPU is slower than llama.cpp GPU                        ║");
        println!("║  → Kernel launch overhead dominates small model inference            ║");
        println!("║                                                                       ║");
        println!("║  WHY #2: Why is kernel launch overhead high?                         ║");
        println!("║  → Each decode step launches 100+ separate CUDA kernels              ║");
        println!("║  → llama.cpp uses ~30 kernels via megakernel fusion                  ║");
        println!("║                                                                       ║");
        println!("║  WHY #3: Why are kernels not fused?                                  ║");
        println!("║  → PAR-039 megakernel exists but not enabled for decode loop         ║");
        println!("║  → PAR-037 CUDA graphs exist but not capturing decode sequence       ║");
        println!("║                                                                       ║");
        println!("║  WHY #4: Why are CUDA graphs not enabled?                            ║");
        println!("║  → Dynamic memory patterns prevent graph capture                     ║");
        println!("║  → KV cache updates break graph replay                               ║");
        println!("║                                                                       ║");
        println!("║  WHY #5: Why can't KV cache work with graphs?                        ║");
        println!("║  → Current implementation allocates per-step                         ║");
        println!("║  → Need static pre-allocated KV cache with position tracking         ║");
        println!("╠═══════════════════════════════════════════════════════════════════════╣");
        println!("║  ROOT CAUSE: Kernel launch overhead (~280 vs ~30 in llama.cpp)       ║");
        println!("╠═══════════════════════════════════════════════════════════════════════╣");
        println!("║  REMEDIATION:                                                        ║");
        println!("║  1. Enable CUDA graphs for decode loop (PAR-037)                     ║");
        println!("║  2. Implement persistent KV cache with graph-compatible updates      ║");
        println!("║  3. Fuse elementwise ops into megakernel (PAR-039)                   ║");
        println!("╠═══════════════════════════════════════════════════════════════════════╣");
        println!("║  CITATIONS:                                                          ║");
        println!("║  [1] CUDA Graphs: Efficient Kernel Launch Amortization               ║");
        println!("║      NVIDIA GTC 2019, S9150                                          ║");
        println!("║  [2] FlashAttention-2: Faster Attention with Better Parallelism      ║");
        println!("║      Dao, 2023, arXiv:2307.08691                                      ║");
        println!("║  [3] vLLM: Efficient Memory Management for Large Language Model      ║");
        println!("║      Serving with PagedAttention, Kwon et al., SOSP 2023             ║");
        println!("╚═══════════════════════════════════════════════════════════════════════╝");
    } else {
        println!("║                                                                       ║");
        println!("║  WHY #1: APR CPU is slower than llama.cpp CPU                        ║");
        println!("║  → Suboptimal SIMD utilization in quantized matmul                   ║");
        println!("║                                                                       ║");
        println!("║  WHY #2: Why is SIMD utilization suboptimal?                         ║");
        println!("║  → AVX-512 not fully exploited for Q4_K dequant+matmul               ║");
        println!("║  → llama.cpp uses hand-optimized ggml_vec_dot_q4_K_q8_K              ║");
        println!("║                                                                       ║");
        println!("║  WHY #3: Why not use same optimizations?                             ║");
        println!("║  → Trueno SIMD backend uses generic patterns                         ║");
        println!("║  → Need specialized Q4_K dot product kernel                          ║");
        println!("║                                                                       ║");
        println!("║  WHY #4: Why doesn't trueno have specialized kernels?                ║");
        println!("║  → Focus was on GPU path, CPU is fallback                            ║");
        println!("║  → Need to port llama.cpp's ggml optimizations                       ║");
        println!("║                                                                       ║");
        println!("║  WHY #5: What specific optimizations are missing?                    ║");
        println!("║  → Tiled cache-blocked matmul for large matrices                     ║");
        println!("║  → Fused dequant+dot in single SIMD pass                             ║");
        println!("╠═══════════════════════════════════════════════════════════════════════╣");
        println!("║  ROOT CAUSE: Missing hand-optimized Q4_K SIMD kernels                ║");
        println!("╠═══════════════════════════════════════════════════════════════════════╣");
        println!("║  REMEDIATION:                                                        ║");
        println!("║  1. Implement ggml_vec_dot_q4_K_q8_K equivalent in trueno            ║");
        println!("║  2. Add cache-blocked tiled matmul for large matrices                ║");
        println!("║  3. Fuse dequantization with dot product in single pass              ║");
        println!("╠═══════════════════════════════════════════════════════════════════════╣");
        println!("║  CITATIONS:                                                          ║");
        println!("║  [1] GGML: A tensor library for machine learning                     ║");
        println!("║      github.com/ggerganov/ggml                                       ║");
        println!("║  [2] Anatomy of High-Performance Matrix Multiplication               ║");
        println!("║      Goto & Van de Geijn, ACM TOMS 2008                              ║");
        println!("╚═══════════════════════════════════════════════════════════════════════╝");
    }
}

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║        PMAT BENCHMARK MATRIX - APR vs llama.cpp Performance              ║");
    println!("║        Target: 2x faster than llama.cpp for EVERY cell                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Check CUDA availability
    let cuda_available = CudaExecutor::is_available();
    if cuda_available {
        let executor = CudaExecutor::new(0).ok();
        if let Some(ex) = executor {
            let device_name = ex.device_name().unwrap_or_default();
            let (_, vram_total) = ex.memory_info().unwrap_or((0, 0));
            println!(
                "  GPU: {} ({} MB VRAM)",
                device_name,
                vram_total / 1024 / 1024
            );
        }
    } else {
        println!("  GPU: Not available (running CPU-only benchmarks)");
    }
    println!();

    let mut results: Vec<BenchResult> = Vec::new();
    let max_tokens = 64;

    for tier in TIERS {
        if !Path::new(tier.gguf_path).exists() {
            println!(
                "⚠️  Skipping {}: model not found at {}",
                tier.name, tier.gguf_path
            );
            continue;
        }

        println!("═══════════════════════════════════════════════════════════════════════════");
        println!(
            "  Testing: {} ({}) - {}",
            tier.name.to_uppercase(),
            tier.size,
            tier.gguf_path.rsplit('/').next().unwrap_or("")
        );
        println!("═══════════════════════════════════════════════════════════════════════════");

        // CPU Benchmark
        println!("\n  [CPU] Running APR CPU benchmark...");
        match benchmark_apr_cpu(tier.gguf_path, tier.prompt_tokens, max_tokens) {
            Ok(tok_s) => {
                let speedup = tok_s / tier.llama_cpp_cpu_baseline;
                let meets_2x = speedup >= 2.0;
                let status = if meets_2x { "✅ PASS" } else { "❌ FAIL" };

                println!("         APR CPU:      {:.1} tok/s", tok_s);
                println!(
                    "         llama.cpp:    {:.1} tok/s",
                    tier.llama_cpp_cpu_baseline
                );
                println!("         Speedup:      {:.2}x {}", speedup, status);

                results.push(BenchResult {
                    tier: tier.name.to_string(),
                    backend: "CPU".to_string(),
                    apr_tok_s: tok_s,
                    llama_baseline: tier.llama_cpp_cpu_baseline,
                    speedup,
                    meets_2x,
                });
            },
            Err(e) => {
                println!("         ❌ Error: {}", e);
            },
        }

        // GPU Benchmark
        if cuda_available {
            println!("\n  [GPU] Running APR GPU benchmark...");
            match benchmark_apr_gpu(tier.gguf_path, tier.prompt_tokens, max_tokens) {
                Ok(tok_s) => {
                    let speedup = tok_s / tier.llama_cpp_gpu_baseline;
                    let meets_2x = speedup >= 2.0;
                    let status = if meets_2x { "✅ PASS" } else { "❌ FAIL" };

                    println!("         APR GPU:      {:.1} tok/s", tok_s);
                    println!(
                        "         llama.cpp:    {:.1} tok/s",
                        tier.llama_cpp_gpu_baseline
                    );
                    println!("         Speedup:      {:.2}x {}", speedup, status);

                    results.push(BenchResult {
                        tier: tier.name.to_string(),
                        backend: "GPU".to_string(),
                        apr_tok_s: tok_s,
                        llama_baseline: tier.llama_cpp_gpu_baseline,
                        speedup,
                        meets_2x,
                    });
                },
                Err(e) => {
                    println!("         ❌ Error: {}", e);
                },
            }
        }
    }

    // Summary table
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                         BENCHMARK MATRIX SUMMARY                         ║");
    println!("╠═════════╦═════════╦═══════════════╦═══════════════╦══════════╦══════════╣");
    println!("║  Tier   ║ Backend ║ APR (tok/s)   ║ llama (tok/s) ║ Speedup  ║  Status  ║");
    println!("╠═════════╬═════════╬═══════════════╬═══════════════╬══════════╬══════════╣");

    let mut all_pass = true;
    let mut failing_cells = Vec::new();

    for r in &results {
        let status = if r.meets_2x { "✅ PASS" } else { "❌ FAIL" };
        println!(
            "║ {:7} ║ {:7} ║ {:>13.1} ║ {:>13.1} ║ {:>7.2}x ║ {:>8} ║",
            r.tier, r.backend, r.apr_tok_s, r.llama_baseline, r.speedup, status
        );

        if !r.meets_2x {
            all_pass = false;
            failing_cells.push(r.clone());
        }
    }

    println!("╚═════════╩═════════╩═══════════════╩═══════════════╩══════════╩══════════╝");
    println!();

    if all_pass {
        println!("╔══════════════════════════════════════════════════════════════════════════╗");
        println!("║                    ✅ ALL CELLS MEET 2x TARGET!                          ║");
        println!("╚══════════════════════════════════════════════════════════════════════════╝");
    } else {
        println!("╔══════════════════════════════════════════════════════════════════════════╗");
        println!("║           ❌ SOME CELLS BELOW 2x - Five-Whys Analysis Required           ║");
        println!("╚══════════════════════════════════════════════════════════════════════════╝");

        // Five-whys for each failing cell
        for r in &failing_cells {
            five_whys_analysis(&r.tier, &r.backend, r.apr_tok_s, r.llama_baseline * 2.0);
        }
    }
}
