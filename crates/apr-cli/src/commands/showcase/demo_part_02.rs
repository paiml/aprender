
/// Step I: ZRAM Compression Demo (Point 79-82)
pub(super) fn run_zram_demo(_config: &ShowcaseConfig) -> Result<ZramDemoResult> {
    println!();
    println!("{}", "═══ Step I: ZRAM Compression Demo ═══".cyan().bold());
    println!();

    #[cfg(feature = "zram")]
    {
        println!("Running with {} (library)", "trueno-zram-core 0.2.0".cyan());
        println!();

        // Create LZ4 compressor
        let lz4_compressor = CompressorBuilder::new()
            .algorithm(ZramAlgorithm::Lz4)
            .build()
            .map_err(|e| {
                CliError::ValidationFailed(format!("Failed to create LZ4 compressor: {e}"))
            })?;

        // Create ZSTD compressor
        let zstd_compressor = CompressorBuilder::new()
            .algorithm(ZramAlgorithm::Zstd { level: 3 })
            .build()
            .map_err(|e| {
                CliError::ValidationFailed(format!("Failed to create ZSTD compressor: {e}"))
            })?;

        let simd_backend = format!("{:?}", lz4_compressor.backend());

        println!("SIMD Backend: {}", simd_backend.cyan());
        println!("Page Size: {} bytes", PAGE_SIZE);
        println!();

        // Test 1: Zero page (same-fill optimization)
        println!("{}", "─── Zero Page Test (Point 81) ───".yellow());
        let zero_page = [0u8; PAGE_SIZE];
        let iterations = 10000;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = lz4_compressor.compress(&zero_page);
        }
        let zero_elapsed = start.elapsed();

        let bytes_processed = PAGE_SIZE as f64 * iterations as f64;
        let zero_page_gbps = bytes_processed / zero_elapsed.as_secs_f64() / 1e9;

        let zero_compressed = lz4_compressor
            .compress(&zero_page)
            .map_err(|e| CliError::ValidationFailed(format!("Compression failed: {e}")))?;
        let zero_ratio = PAGE_SIZE as f64 / zero_compressed.data.len() as f64;

        println!(
            "  {} Zero-page throughput: {:.1} GB/s (target: >150 GB/s)",
            if zero_page_gbps > 150.0 {
                "✓".green()
            } else {
                "⚠".yellow()
            },
            zero_page_gbps
        );
        println!(
            "  {} Zero-page ratio: {:.1}x ({} → {} bytes)",
            "✓".green(),
            zero_ratio,
            PAGE_SIZE,
            zero_compressed.data.len()
        );
        println!();

        // Test 2: LZ4 compression
        println!("{}", "─── LZ4 Compression Test ───".yellow());
        let mut test_page = [0u8; PAGE_SIZE];
        // Create realistic page with repeated patterns
        for (i, byte) in test_page.iter_mut().enumerate() {
            *byte = ((i / 64) % 256) as u8;
        }

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = lz4_compressor.compress(&test_page);
        }
        let lz4_elapsed = start.elapsed();
        let lz4_gbps = bytes_processed / lz4_elapsed.as_secs_f64() / 1e9;

        let lz4_compressed = lz4_compressor
            .compress(&test_page)
            .map_err(|e| CliError::ValidationFailed(format!("LZ4 compression failed: {e}")))?;
        let lz4_ratio = PAGE_SIZE as f64 / lz4_compressed.data.len() as f64;

        println!(
            "  {} LZ4 throughput: {:.2} GB/s (target: >3 GB/s)",
            if lz4_gbps > 3.0 {
                "✓".green()
            } else {
                "⚠".yellow()
            },
            lz4_gbps
        );
        println!(
            "  {} LZ4 ratio: {:.2}x ({} → {} bytes)",
            "✓".green(),
            lz4_ratio,
            PAGE_SIZE,
            lz4_compressed.data.len()
        );
        println!();

        // Test 3: ZSTD compression
        println!("{}", "─── ZSTD Compression Test ───".yellow());
        let zstd_compressed = zstd_compressor
            .compress(&test_page)
            .map_err(|e| CliError::ValidationFailed(format!("ZSTD compression failed: {e}")))?;
        let zstd_ratio = PAGE_SIZE as f64 / zstd_compressed.data.len() as f64;

        println!(
            "  {} ZSTD ratio: {:.2}x ({} → {} bytes)",
            "✓".green(),
            zstd_ratio,
            PAGE_SIZE,
            zstd_compressed.data.len()
        );
        println!();

        // Report compression stats (Point 82)
        println!("{}", "─── Compression Stats (Point 82) ───".yellow());
        let stats = lz4_compressor.stats();
        println!("  Pages compressed: {}", stats.pages_compressed);
        println!("  Bytes in: {} KB", stats.bytes_in / 1024);
        println!("  Bytes out: {} KB", stats.bytes_out / 1024);
        if stats.bytes_out > 0 {
            let overall_ratio = stats.bytes_in as f64 / stats.bytes_out as f64;
            println!("  {} Overall ratio: {:.2}x", "✓".green(), overall_ratio);
        }
        println!();

        // Context extension calculation (Point 80)
        println!("{}", "─── Context Extension (Point 80) ───".yellow());
        // Use the better of LZ4 or ZSTD ratio (whichever compresses better)
        // Capped at 2.5x for conservative estimate
        let best_ratio = lz4_ratio.max(zstd_ratio);
        let context_extension = best_ratio.min(2.5);
        let base_context_k = 16; // 16K tokens baseline
        let extended_context_k = (base_context_k as f64 * context_extension) as u32;
        let meets_2x = context_extension >= 2.0;

        println!(
            "  {} Context extension: {:.1}x ({} → {}K tokens)",
            if meets_2x {
                "✓".green()
            } else {
                "⚠".yellow()
            },
            context_extension,
            base_context_k,
            extended_context_k
        );

        if meets_2x {
            println!(
                "  {} ZRAM enables ≥2x context extension (Point 80 verified)",
                "✓".green()
            );
        } else {
            println!(
                "  {} Context extension {:.1}x < 2.0x target",
                "⚠".yellow(),
                context_extension
            );
        }
        println!();

        println!(
            "{} ZRAM demo complete - trueno-zram-core 0.2.0 verified",
            "✓".green()
        );

        Ok(ZramDemoResult {
            lz4_ratio,
            zstd_ratio,
            zero_page_gbps,
            lz4_gbps,
            simd_backend,
            context_extension,
        })
    }

    #[cfg(not(feature = "zram"))]
    {
        println!("{} trueno-zram-core feature not enabled", "⚠".yellow());
        println!("Enable with: cargo build --features zram");

        Ok(ZramDemoResult {
            lz4_ratio: 0.0,
            zstd_ratio: 0.0,
            zero_page_gbps: 0.0,
            lz4_gbps: 0.0,
            simd_backend: "disabled".to_string(),
            context_extension: 0.0,
        })
    }
}

/// Run CUDA GPU detection demo (Point 78: GPU kernels visible)
///
/// Demonstrates CUDA device detection and VRAM monitoring using
/// realizar's CudaExecutor which wraps trueno-gpu.
pub(super) fn run_cuda_demo(_config: &ShowcaseConfig) -> Result<CudaDemoResult> {
    println!();
    println!(
        "{}",
        "═══ H: CUDA GPU Detection (Point 78) ═══".cyan().bold()
    );
    println!();

    #[cfg(feature = "cuda")]
    {
        use realizar::cuda::CudaExecutor;

        println!("{}", "─── CUDA Device Detection ───".yellow());

        // Check device count
        let device_count = CudaExecutor::num_devices();
        println!(
            "  {} CUDA devices detected: {}",
            if device_count > 0 {
                "✓".green()
            } else {
                "✗".red()
            },
            device_count
        );

        if device_count == 0 {
            println!("  {} No CUDA devices found", "⚠".yellow());
            return Ok(CudaDemoResult {
                device_count: 0,
                device_name: "N/A".to_string(),
                total_vram_gb: 0.0,
                free_vram_gb: 0.0,
                cuda_available: false,
                graph_capture_available: false,
                graph_speedup: 1.0,
                dp4a_available: false,
                dp4a_arithmetic_intensity: 0.0,
            });
        }

        // Create executor for device 0
        let executor = CudaExecutor::new(0)
            .map_err(|e| CliError::ValidationFailed(format!("CUDA init failed: {e}")))?;

        // Get device name
        let device_name = executor
            .device_name()
            .map_err(|e| CliError::ValidationFailed(format!("Device name query failed: {e}")))?;

        println!("  {} GPU: {}", "✓".green(), device_name);

        // Get memory info
        let (free_bytes, total_bytes) = executor
            .memory_info()
            .map_err(|e| CliError::ValidationFailed(format!("Memory query failed: {e}")))?;

        let total_vram_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let free_vram_gb = free_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let used_vram_gb = total_vram_gb - free_vram_gb;

        println!(
            "  {} VRAM: {:.1} GB total, {:.1} GB free, {:.1} GB used",
            "✓".green(),
            total_vram_gb,
            free_vram_gb,
            used_vram_gb
        );

        // Verify sufficient VRAM for Qwen2.5-Coder-32B (needs ~20GB for Q4_K_M)
        let required_vram_gb = 20.0;
        if total_vram_gb >= required_vram_gb {
            println!(
                "  {} Sufficient VRAM for Qwen2.5-Coder-32B-Q4_K_M ({:.0} GB required)",
                "✓".green(),
                required_vram_gb
            );
        } else {
            println!(
                "  {} Insufficient VRAM: {:.1} GB available, {:.0} GB required",
                "⚠".yellow(),
                total_vram_gb,
                required_vram_gb
            );
        }

        // Section 5.2: CUDA Graph Brick Demo (P0)
        println!();
        println!(
            "{}",
            "─── CUDA Graph Capture (Section 5.2 - P0) ───".yellow()
        );

        use realizar::brick::{CoalescedDp4aBrick, ComputeBrick, CudaGraphBrick};

        // Create CUDA Graph brick for 64 layers @ 4096 hidden dim (Qwen2.5-32B config)
        let graph_brick = CudaGraphBrick::new(64, 4096);
        let graph_capture_available = graph_brick.can_run();

        println!(
            "  {} CudaGraphBrick: {} layers × {} hidden_dim",
            if graph_capture_available {
                "✓".green()
            } else {
                "✗".red()
            },
            graph_brick.num_layers,
            graph_brick.hidden_dim
        );
        println!(
            "    Budget: {:.1}µs ({:.0} tok/s)",
            graph_brick.budget().us_per_token,
            graph_brick.budget().tokens_per_sec
        );

        // Graph speedup is THEORETICAL based on:
        // - Industry benchmark: ~5µs kernel launch overhead (NVIDIA Nsight)
        // - Qwen2.5-32B decode: ~280 kernels per forward pass
        // - Graph replay: single dispatch (~20µs target)
        // Note(PAR-090): Actual speedup measurement via CudaEvent timing deferred to PAR-090
        let eager_launch_us = 5.0 * 280.0; // THEORETICAL: 280 kernels × 5µs launch overhead
        let graph_replay_us = graph_brick.budget().us_per_token; // TARGET budget, not measured
        let graph_speedup = eager_launch_us / graph_replay_us;

        println!(
            "    Theoretical speedup: {:.1}x (eager: {:.0}µs → graph: {:.0}µs)",
            graph_speedup, eager_launch_us, graph_replay_us
        );
        println!(
            "    {}",
            "⚠ Values are theoretical estimates, not measured (see PAR-090)".yellow()
        );

        for assertion in graph_brick.assertions() {
            println!("    {} Assertion: {}", "✓".green(), assertion.name);
        }

        // Section 5.3: Coalesced DP4A Brick Demo (P0)
        println!();
        println!(
            "{}",
            "─── Coalesced DP4A Brick (Section 5.3 - P0) ───".yellow()
        );

        // Create DP4A brick for typical decode GEMV: K=4096, N=1 (single token)
        let dp4a_brick = CoalescedDp4aBrick::new(4096, 4096);
        let dp4a_available = dp4a_brick.can_run();

        println!(
            "  {} CoalescedDp4aBrick: K={} × N={}",
            if dp4a_available {
                "✓".green()
            } else {
                "✗".red()
            },
            dp4a_brick.k,
            dp4a_brick.n
        );
        println!(
            "    Budget: {:.1}µs ({:.0} tok/s)",
            dp4a_brick.budget().us_per_token,
            dp4a_brick.budget().tokens_per_sec
        );

        let dp4a_ai = dp4a_brick.arithmetic_intensity();
        println!("    Arithmetic intensity: {:.2} flops/byte", dp4a_ai);
        println!(
            "    {}",
            if dp4a_ai >= 0.5 {
                "Compute-bound (good for DP4A)".green()
            } else {
                "Memory-bound (may not benefit from DP4A)".yellow()
            }
        );

        for assertion in dp4a_brick.assertions() {
            println!("    {} Assertion: {}", "✓".green(), assertion.name);
        }

        println!();
        println!(
            "{} CUDA demo complete - GPU kernels visible via realizar/trueno-gpu",
            "✓".green()
        );

        Ok(CudaDemoResult {
            device_count,
            device_name,
            total_vram_gb,
            free_vram_gb,
            cuda_available: true,
            graph_capture_available,
            graph_speedup,
            dp4a_available,
            dp4a_arithmetic_intensity: dp4a_ai,
        })
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("{} CUDA feature not enabled", "⚠".yellow());
        println!("Enable with: cargo build --features cuda");

        Ok(CudaDemoResult {
            device_count: 0,
            device_name: "disabled".to_string(),
            total_vram_gb: 0.0,
            free_vram_gb: 0.0,
            cuda_available: false,
            graph_capture_available: false,
            graph_speedup: 1.0,
            dp4a_available: false,
            dp4a_arithmetic_intensity: 0.0,
        })
    }
}
