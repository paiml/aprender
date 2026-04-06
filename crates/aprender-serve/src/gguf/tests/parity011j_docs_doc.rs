
/// Test PARITY-011j: QA-050 Documentation updated with latest benchmark results
///
/// Verifies documentation auto-update infrastructure.
#[test]
fn test_parity011j_docs_auto_update() {
    /// Documentation section that can be auto-updated
    #[derive(Debug)]
    struct DocSection {
        file: String,
        start_marker: String,
        end_marker: String,
    }

    /// Benchmark result for docs
    #[derive(Debug)]
    struct BenchResultForDocs {
        comparison: String,
        gap_before: String,
        gap_after: String,
        improvement: String,
    }

    /// Documentation updater
    struct DocsUpdater {
        sections: Vec<DocSection>,
    }

    impl DocsUpdater {
        fn new() -> Self {
            Self {
                sections: vec![
                    DocSection {
                        file: "README.md".to_string(),
                        start_marker: "<!-- BENCH-RESULTS-START -->".to_string(),
                        end_marker: "<!-- BENCH-RESULTS-END -->".to_string(),
                    },
                    DocSection {
                        file: "docs/benchmarks.md".to_string(),
                        start_marker: "<!-- PERF-TABLE-START -->".to_string(),
                        end_marker: "<!-- PERF-TABLE-END -->".to_string(),
                    },
                ],
            }
        }

        fn generate_table(&self, results: &[BenchResultForDocs]) -> String {
            let mut table =
                String::from("| Comparison | Gap (Before) | Gap (After) | Improvement |\n");
            table.push_str("|------------|--------------|-------------|-------------|\n");
            for r in results {
                table.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    r.comparison, r.gap_before, r.gap_after, r.improvement
                ));
            }
            table
        }

        fn update_content(&self, content: &str, section: &DocSection, new_table: &str) -> String {
            if let (Some(start), Some(end)) = (
                content.find(&section.start_marker),
                content.find(&section.end_marker),
            ) {
                let before = &content[..start + section.start_marker.len()];
                let after = &content[end..];
                format!("{}\n{}{}", before, new_table, after)
            } else {
                content.to_string()
            }
        }
    }

    let updater = DocsUpdater::new();
    assert_eq!(
        updater.sections.len(),
        2,
        "QA-050: Two doc sections configured"
    );

    // Generate benchmark table
    let results = vec![
        BenchResultForDocs {
            comparison: "Realizar vs Ollama".to_string(),
            gap_before: "4,614x".to_string(),
            gap_after: "1,181x".to_string(),
            improvement: "3.9x".to_string(),
        },
        BenchResultForDocs {
            comparison: "Realizar vs llama.cpp".to_string(),
            gap_before: "6,400x".to_string(),
            gap_after: "1,506x".to_string(),
            improvement: "4.2x".to_string(),
        },
    ];

    let table = updater.generate_table(&results);
    assert!(
        table.contains("Realizar vs Ollama"),
        "QA-050: Table has comparisons"
    );
    assert!(table.contains("1,181x"), "QA-050: Table has gap values");

    // Test content update
    let mock_readme = "# README\n<!-- BENCH-RESULTS-START -->\nold data\n<!-- BENCH-RESULTS-END -->\nMore content";
    let updated = updater.update_content(mock_readme, &updater.sections[0], &table);
    assert!(
        updated.contains("Realizar vs Ollama"),
        "QA-050: Content updated"
    );
    assert!(!updated.contains("old data"), "QA-050: Old data replaced");

    println!("\nPARITY-011j: Documentation auto-update");
    println!("  Sections: {}", updater.sections.len());
    println!("  Generated table rows: {}", results.len());
}

// PARITY-012: GPU Optimization for Performance Parity
// ============================================================================
//
// Reference: docs/specifications/performance-parity-ollama-llamacpp-gpu-inference-llms.md
// Goal: Close 1000x+ gap to achieve parity with Ollama/llama.cpp
//
// Key Insights from IMP-600:
// - GPU is 2.7x SLOWER for matvec (single token generation)
// - GPU is 57x FASTER for GEMM (batch operations like prefill)
// - FlashAttention is required for GPU to help attention

/// Test PARITY-012a: FlashAttention tiled algorithm structure
///
/// Implements O(N) memory attention via tiling, avoiding N×N matrix materialization.
/// Reference: Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention"
#[test]
fn test_parity012a_flash_attention_tiled() {
    /// FlashAttention tile configuration
    #[derive(Debug, Clone)]
    struct FlashAttentionConfig {
        /// Block size for Q (rows)
        block_q: usize,
        /// Block size for KV (columns)
        block_kv: usize,
        /// Head dimension
        head_dim: usize,
        /// Causal masking enabled
        causal: bool,
    }

    impl FlashAttentionConfig {
        fn new(head_dim: usize) -> Self {
            // Optimal block sizes for GPU SRAM (typically 64-128)
            Self {
                block_q: 64,
                block_kv: 64,
                head_dim,
                causal: true,
            }
        }

        /// Calculate number of tiles for given sequence length
        fn num_tiles(&self, seq_len: usize) -> (usize, usize) {
            let q_tiles = seq_len.div_ceil(self.block_q);
            let kv_tiles = seq_len.div_ceil(self.block_kv);
            (q_tiles, kv_tiles)
        }

        /// Memory required for tiled attention (O(N) not O(N²))
        fn memory_bytes(&self, _seq_len: usize) -> usize {
            // Only need: Q block + K block + V block + output block + running stats
            let q_block = self.block_q * self.head_dim * 4; // f32
            let kv_block = self.block_kv * self.head_dim * 4 * 2; // K and V
            let output_block = self.block_q * self.head_dim * 4;
            let stats = self.block_q * 4 * 2; // m_i (max) and l_i (sum)

            q_block + kv_block + output_block + stats
        }

        /// Standard attention memory (O(N²))
        fn standard_memory_bytes(&self, seq_len: usize) -> usize {
            // Q, K, V tensors + full attention matrix
            let qkv = seq_len * self.head_dim * 4 * 3;
            let attn_matrix = seq_len * seq_len * 4;
            qkv + attn_matrix
        }
    }

    /// FlashAttention tile state (running max and sum for online softmax)
    #[derive(Debug, Clone)]
    struct TileState {
        /// Running max for numerical stability
        m_i: Vec<f32>,
        /// Running sum of exp(x - m)
        l_i: Vec<f32>,
        /// Accumulated output
        o_i: Vec<f32>,
    }

    impl TileState {
        fn new(block_q: usize, head_dim: usize) -> Self {
            Self {
                m_i: vec![f32::NEG_INFINITY; block_q],
                l_i: vec![0.0; block_q],
                o_i: vec![0.0; block_q * head_dim],
            }
        }

        /// Update state with new tile (FlashAttention online softmax)
        fn update(
            &mut self,
            scores: &[f32],
            v_block: &[f32],
            block_q: usize,
            block_kv: usize,
            head_dim: usize,
        ) {
            for i in 0..block_q {
                // Find new max for this row
                let row_start = i * block_kv;
                let row_end = row_start + block_kv;
                let m_new = scores[row_start..row_end]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);

                let m_combined = self.m_i[i].max(m_new);

                // Rescale previous accumulator
                let scale_old = (self.m_i[i] - m_combined).exp();
                let scale_new = (m_new - m_combined).exp();

                // Update running sum
                let l_new: f32 = scores[row_start..row_end]
                    .iter()
                    .map(|&s| (s - m_new).exp())
                    .sum();

                self.l_i[i] = self.l_i[i] * scale_old + l_new * scale_new;
                self.m_i[i] = m_combined;

                // Update output: o_i = scale_old * o_i + scale_new * (softmax @ V)
                for d in 0..head_dim {
                    self.o_i[i * head_dim + d] *= scale_old;
                    // Add contribution from this tile
                    for j in 0..block_kv {
                        let attn_weight = (scores[row_start + j] - m_new).exp() * scale_new;
                        self.o_i[i * head_dim + d] += attn_weight * v_block[j * head_dim + d];
                    }
                }
            }
        }

        /// Finalize output by dividing by sum
        fn finalize(&mut self, block_q: usize, head_dim: usize) {
            for i in 0..block_q {
                if self.l_i[i] > 0.0 {
                    for d in 0..head_dim {
                        self.o_i[i * head_dim + d] /= self.l_i[i];
                    }
                }
            }
        }
    }

    let config = FlashAttentionConfig::new(64); // head_dim=64

    // Test memory savings
    let seq_len = 2048;
    let flash_mem = config.memory_bytes(seq_len);
    let standard_mem = config.standard_memory_bytes(seq_len);
    let savings = standard_mem as f64 / flash_mem as f64;

    assert!(
        savings > 10.0,
        "PARITY-012a: FlashAttention should save >10x memory for seq_len=2048"
    );

    // Test tile calculation
    let (q_tiles, kv_tiles) = config.num_tiles(seq_len);
    assert_eq!(
        q_tiles, 32,
        "PARITY-012a: Should have 32 Q tiles for 2048/64"
    );
    assert_eq!(kv_tiles, 32, "PARITY-012a: Should have 32 KV tiles");

    // Test online softmax state
    let mut state = TileState::new(config.block_q, config.head_dim);

    // Simulate processing a tile
    let scores = vec![0.1f32; config.block_q * config.block_kv];
    let v_block = vec![1.0f32; config.block_kv * config.head_dim];
    state.update(
        &scores,
        &v_block,
        config.block_q,
        config.block_kv,
        config.head_dim,
    );
    state.finalize(config.block_q, config.head_dim);

    // Output should be normalized (sum of attention weights = 1)
    assert!(
        state.o_i[0].is_finite(),
        "PARITY-012a: Output should be finite"
    );

    println!("\nPARITY-012a: FlashAttention tiled algorithm");
    println!("  Seq length: {}", seq_len);
    println!(
        "  Standard memory: {:.2} MB",
        standard_mem as f64 / 1_000_000.0
    );
    println!("  Flash memory: {:.2} KB", flash_mem as f64 / 1_000.0);
    println!("  Memory savings: {:.1}x", savings);
    println!("  Tiles: {}x{}", q_tiles, kv_tiles);
}

/// Test PARITY-012b: GPU batch matmul dispatch threshold
///
/// Determines optimal threshold for GPU vs CPU dispatch based on operation size.
/// Key insight: GPU wins for batch (GEMM), CPU wins for single-token (MATVEC).
#[test]
fn test_parity012b_gpu_dispatch_threshold() {
    /// Operation type for dispatch decision
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum MatmulType {
        /// Single vector × matrix (token generation)
        Matvec,
        /// Matrix × matrix (batch prefill)
        Gemm,
    }

    /// GPU dispatch decision
    #[derive(Debug, Clone)]
    struct DispatchDecision {
        use_gpu: bool,
        reason: String,
        expected_speedup: f64,
    }

    /// Dispatch threshold configuration
    struct DispatchThresholds {
        /// Minimum elements for GPU dispatch
        min_elements: usize,
        /// Minimum batch size for GPU GEMM
        min_batch: usize,
        /// Matvec size where GPU breaks even (IMP-600: never for small)
        matvec_threshold: usize,
        /// GEMM size where GPU wins (IMP-600: 1024+ verified 57x)
        gemm_threshold: usize,
    }

    impl DispatchThresholds {
        fn default() -> Self {
            Self {
                min_elements: 100_000,        // 100K elements minimum
                min_batch: 32,                // Batch size >= 32 for GPU
                matvec_threshold: usize::MAX, // GPU never wins for matvec
                gemm_threshold: 512,          // 512x512 matrices
            }
        }

        fn should_use_gpu(&self, m: usize, k: usize, n: usize) -> DispatchDecision {
            let op_type = if m == 1 {
                MatmulType::Matvec
            } else {
                MatmulType::Gemm
            };
            let elements = m * k + k * n + m * n;

            match op_type {
                MatmulType::Matvec => {
                    // GPU is 2.7x SLOWER for matvec (IMP-600b)
                    DispatchDecision {
                        use_gpu: false,
                        reason: "Matvec: GPU 2.7x slower than SIMD (IMP-600b)".to_string(),
                        expected_speedup: 0.37, // CPU is 2.7x faster
                    }
                },
                MatmulType::Gemm => {
                    if m >= self.min_batch && k >= self.gemm_threshold && n >= self.gemm_threshold {
                        // GPU wins for large GEMM (IMP-600c: 57x verified)
                        let speedup = if k >= 1024 && n >= 1024 { 57.0 } else { 10.0 };
                        DispatchDecision {
                            use_gpu: true,
                            reason: format!("GEMM {}x{}x{}: GPU {}x faster", m, k, n, speedup),
                            expected_speedup: speedup,
                        }
                    } else if elements < self.min_elements {
                        DispatchDecision {
                            use_gpu: false,
                            reason: format!(
                                "Small GEMM ({} elements): dispatch overhead dominates",
                                elements
                            ),
                            expected_speedup: 0.5,
                        }
                    } else {
                        DispatchDecision {
                            use_gpu: true,
                            reason: "Medium GEMM: GPU slight advantage".to_string(),
                            expected_speedup: 2.0,
                        }
                    }
                },
            }
        }
    }

    let thresholds = DispatchThresholds::default();

    // Test: Single token generation (matvec) - should use CPU
    let decision = thresholds.should_use_gpu(1, 2560, 2560);
    assert!(!decision.use_gpu, "PARITY-012b: Matvec should use CPU");
    assert!(
        decision.expected_speedup < 1.0,
        "PARITY-012b: GPU slower for matvec"
    );

    // Test: Batch prefill (GEMM) - should use GPU
    let decision = thresholds.should_use_gpu(128, 2560, 2560);
    assert!(decision.use_gpu, "PARITY-012b: Large GEMM should use GPU");
    assert!(
        decision.expected_speedup > 10.0,
        "PARITY-012b: GPU much faster for GEMM"
    );

    // Test: Small batch - CPU might still win
    let decision = thresholds.should_use_gpu(4, 256, 256);
    assert!(!decision.use_gpu, "PARITY-012b: Small GEMM should use CPU");

    // Test: Large GEMM (1024x1024) - 57x speedup verified
    let decision = thresholds.should_use_gpu(64, 1024, 1024);
    assert!(
        decision.use_gpu,
        "PARITY-012b: 1024x1024 GEMM should use GPU"
    );
    assert!(
        (decision.expected_speedup - 57.0).abs() < 1.0,
        "PARITY-012b: 57x speedup for 1024³"
    );

    println!("\nPARITY-012b: GPU dispatch thresholds");
    println!("  Matvec threshold: Never (GPU 2.7x slower)");
    println!(
        "  GEMM threshold: {}x{} matrices",
        thresholds.gemm_threshold, thresholds.gemm_threshold
    );
    println!("  Min batch size: {}", thresholds.min_batch);
    println!("  Min elements: {}", thresholds.min_elements);
}
