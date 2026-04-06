
impl BenchmarkGrid {

    // ========================================================================
    // Profiling Log for Chat Paste
    // ========================================================================

    /// Generate profiling log suitable for chat paste
    pub fn render_profiling_log(&self) -> String {
        let mut out = String::new();

        writeln!(out, "```").expect("failed to write benchmark output");
        writeln!(
            out,
            "═══════════════════════════════════════════════════════════════════════"
        )
        .expect("failed to write benchmark output");
        writeln!(out, "INFERENCE PROFILING REPORT").expect("failed to write benchmark output");
        writeln!(
            out,
            "═══════════════════════════════════════════════════════════════════════"
        )
        .expect("failed to write benchmark output");
        writeln!(out).expect("failed to write benchmark output");

        // Model & Hardware
        writeln!(out, "MODEL: {} ({})", self.model_name, self.model_params)
            .expect("failed to write benchmark output");
        writeln!(out, "QUANT: {}", self.quantization).expect("failed to write benchmark output");
        writeln!(
            out,
            "GPU:   {} ({:.1}GB VRAM)",
            self.gpu_name, self.gpu_vram_gb
        )
        .expect("failed to write benchmark output");
        writeln!(out).expect("failed to write benchmark output");

        // Performance Summary
        writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        )
        .expect("failed to write benchmark output");
        writeln!(out, "THROUGHPUT COMPARISON (tok/s)").expect("failed to write benchmark output");
        writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        )
        .expect("failed to write benchmark output");

        if let Some(ref m) = self.gguf_apr {
            writeln!(
                out,
                "APR GGUF:      {:>8.1} tok/s  (TTFT: {:>6.1}ms)",
                m.tokens_per_sec, m.ttft_ms
            )
            .expect("failed to write benchmark output");
        }
        if let Some(ref m) = self.apr_native {
            writeln!(
                out,
                "APR .apr:      {:>8.1} tok/s  (TTFT: {:>6.1}ms)",
                m.tokens_per_sec, m.ttft_ms
            )
            .expect("failed to write benchmark output");
        }
        if let Some(ref m) = self.gguf_ollama {
            writeln!(
                out,
                "Ollama:        {:>8.1} tok/s  (TTFT: {:>6.1}ms)",
                m.tokens_per_sec, m.ttft_ms
            )
            .expect("failed to write benchmark output");
        }
        if let Some(ref m) = self.gguf_llamacpp {
            writeln!(
                out,
                "llama.cpp:     {:>8.1} tok/s  (TTFT: {:>6.1}ms)",
                m.tokens_per_sec, m.ttft_ms
            )
            .expect("failed to write benchmark output");
        }
        writeln!(out).expect("failed to write benchmark output");

        // Speedup Analysis
        writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        )
        .expect("failed to write benchmark output");
        writeln!(out, "SPEEDUP ANALYSIS").expect("failed to write benchmark output");
        writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        )
        .expect("failed to write benchmark output");

        let ollama_tps = self
            .gguf_ollama
            .as_ref()
            .map_or(318.0, |m| m.tokens_per_sec);
        let llamacpp_tps = self
            .gguf_llamacpp
            .as_ref()
            .map_or(200.0, |m| m.tokens_per_sec);

        if let Some(ref m) = self.gguf_apr {
            let vs_ollama = m.tokens_per_sec / ollama_tps;
            let vs_llamacpp = m.tokens_per_sec / llamacpp_tps;
            writeln!(
                out,
                "APR GGUF vs Ollama:     {:>5.2}x  {}",
                vs_ollama,
                if vs_ollama >= 1.0 { "✓" } else { "⚠" }
            )
            .expect("failed to write benchmark output");
            writeln!(
                out,
                "APR GGUF vs llama.cpp:  {:>5.2}x  {}",
                vs_llamacpp,
                if vs_llamacpp >= 1.25 {
                    "✓ Point 41 PASS"
                } else {
                    "⚠ Point 41 FAIL"
                }
            )
            .expect("failed to write benchmark output");
        }

        if let Some(ref m) = self.apr_native {
            let vs_ollama = m.tokens_per_sec / ollama_tps;
            writeln!(
                out,
                "APR .apr vs Ollama:     {:>5.2}x  {}",
                vs_ollama,
                if vs_ollama >= 2.0 {
                    "✓ 2x target"
                } else {
                    ""
                }
            )
            .expect("failed to write benchmark output");
        }
        writeln!(out).expect("failed to write benchmark output");

        // Profiling Hotspots
        if !self.hotspots.is_empty() {
            writeln!(
                out,
                "───────────────────────────────────────────────────────────────────────"
            )
            .expect("failed to write benchmark output");
            writeln!(out, "PROFILING HOTSPOTS (>5% of execution time)")
                .expect("failed to write benchmark output");
            writeln!(
                out,
                "───────────────────────────────────────────────────────────────────────"
            )
            .expect("failed to write benchmark output");

            for hotspot in &self.hotspots {
                writeln!(out, "{}", hotspot.to_line()).expect("failed to write benchmark output");
                if !hotspot.explanation.is_empty() {
                    writeln!(out, "   └─ {}", hotspot.explanation)
                        .expect("failed to write benchmark output");
                }
            }
            writeln!(out).expect("failed to write benchmark output");
        }

        // GPU Metrics
        writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        )
        .expect("failed to write benchmark output");
        writeln!(out, "GPU METRICS").expect("failed to write benchmark output");
        writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        )
        .expect("failed to write benchmark output");

        if let Some(ref m) = self.gguf_apr {
            if let (Some(util), Some(mem)) = (m.gpu_util, m.gpu_mem_mb) {
                writeln!(
                    out,
                    "APR GGUF:   GPU Util: {:>5.1}%  VRAM: {:>6.0}MB",
                    util, mem
                )
                .expect("failed to write benchmark output");
            }
        }
        if let Some(ref m) = self.apr_native {
            if let (Some(util), Some(mem)) = (m.gpu_util, m.gpu_mem_mb) {
                writeln!(
                    out,
                    "APR .apr:   GPU Util: {:>5.1}%  VRAM: {:>6.0}MB",
                    util, mem
                )
                .expect("failed to write benchmark output");
            }
        }
        writeln!(out).expect("failed to write benchmark output");

        // Recommendations
        writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        )
        .expect("failed to write benchmark output");
        writeln!(out, "OPTIMIZATION RECOMMENDATIONS").expect("failed to write benchmark output");
        writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        )
        .expect("failed to write benchmark output");

        let unexpected: Vec<_> = self.hotspots.iter().filter(|h| !h.is_expected).collect();
        if unexpected.is_empty() {
            writeln!(out, "✓ No unexpected hotspots detected")
                .expect("failed to write benchmark output");
        } else {
            for h in unexpected {
                writeln!(out, "⚠ {}: {}", h.component, h.explanation)
                    .expect("failed to write benchmark output");
            }
        }

        // Phase 2 status
        let apr_tps = self.gguf_apr.as_ref().map_or(0.0, |m| m.tokens_per_sec);
        if apr_tps < 500.0 {
            writeln!(out).expect("failed to write benchmark output");
            writeln!(out, "Phase 2 Optimizations (projected 3.28x improvement):")
                .expect("failed to write benchmark output");
            writeln!(out, "  PAR-036: Persistent threads      (1.3x)")
                .expect("failed to write benchmark output");
            writeln!(out, "  PAR-037: CUDA graph capture      (1.5x)")
                .expect("failed to write benchmark output");
            writeln!(out, "  PAR-038: Multi-stream pipeline   (1.2x)")
                .expect("failed to write benchmark output");
            writeln!(out, "  PAR-039: Megakernel fusion       (1.4x)")
                .expect("failed to write benchmark output");
            writeln!(
                out,
                "  Projected: {:.1} × 3.28 = {:.1} tok/s",
                apr_tps,
                apr_tps * 3.28
            )
            .expect("failed to write benchmark output");
        }

        writeln!(
            out,
            "═══════════════════════════════════════════════════════════════════════"
        )
        .expect("failed to write benchmark output");
        writeln!(out, "```").expect("failed to write benchmark output");

        out
    }

    /// Generate compact one-liner for quick comparison
    pub fn render_compact(&self) -> String {
        let apr_tps = self.gguf_apr.as_ref().map_or(0.0, |m| m.tokens_per_sec);
        let ollama_tps = self.gguf_ollama.as_ref().map_or(0.0, |m| m.tokens_per_sec);
        let llamacpp_tps = self
            .gguf_llamacpp
            .as_ref()
            .map_or(0.0, |m| m.tokens_per_sec);

        format!(
            "APR:{:.0} Ollama:{:.0} llama.cpp:{:.0} tok/s | APR vs Ollama:{:.2}x vs llama.cpp:{:.2}x",
            apr_tps, ollama_tps, llamacpp_tps,
            apr_tps / ollama_tps.max(1.0),
            apr_tps / llamacpp_tps.max(1.0)
        )
    }
}

include!("bench_viz_model_gpu_benchmark.rs");
