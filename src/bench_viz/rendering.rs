impl BenchmarkGrid {

    /// Generate profiling log suitable for chat paste
    pub fn render_profiling_log(&self) -> String {
        // Note: Profiling log uses plain text (no colors) for chat paste compatibility
        let mut out = String::new();

        let _ = writeln!(out, "```");
        let _ = writeln!(
            out,
            "═══════════════════════════════════════════════════════════════════════"
        );
        let _ = writeln!(out, "INFERENCE PROFILING REPORT");
        let _ = writeln!(
            out,
            "═══════════════════════════════════════════════════════════════════════"
        );
        let _ = writeln!(out);

        // Model & Hardware
        let _ = writeln!(out, "MODEL: {} ({})", self.model_name, self.model_params);
        let _ = writeln!(out, "QUANT: {}", self.quantization);
        let _ = writeln!(
            out,
            "GPU:   {} ({:.1}GB VRAM)",
            self.gpu_name, self.gpu_vram_gb
        );
        let _ = writeln!(out);

        // Statistical Summary
        let _ = writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        );
        let _ = writeln!(
            out,
            "THROUGHPUT COMPARISON (tok/s) - {} iterations",
            self.config.iterations
        );
        let _ = writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        );

        if let Some(ref m) = self.gguf_apr {
            let tps = m.mean_throughput();
            let ttft = m.mean_ttft();
            let ci = m.throughput_stats.as_ref().map_or_else(String::new, |s| {
                format!(" [CI: {:.1}-{:.1}]", s.ci_95.0, s.ci_95.1)
            });
            let _ = writeln!(
                out,
                "APR GGUF:      {:>8.1} tok/s{} (TTFT: {:>6.1}ms)",
                tps, ci, ttft
            );
        }
        if let Some(ref m) = self.apr_native {
            let tps = m.mean_throughput();
            let ttft = m.mean_ttft();
            let ci = m.throughput_stats.as_ref().map_or_else(String::new, |s| {
                format!(" [CI: {:.1}-{:.1}]", s.ci_95.0, s.ci_95.1)
            });
            let _ = writeln!(
                out,
                "APR .apr:      {:>8.1} tok/s{} (TTFT: {:>6.1}ms)",
                tps, ci, ttft
            );
        }
        if let Some(ref m) = self.gguf_ollama {
            let tps = m.mean_throughput();
            let ttft = m.mean_ttft();
            let _ = writeln!(
                out,
                "Ollama:        {:>8.1} tok/s  (TTFT: {:>6.1}ms)",
                tps, ttft
            );
        }
        if let Some(ref m) = self.gguf_llamacpp {
            let tps = m.mean_throughput();
            let ttft = m.mean_ttft();
            let _ = writeln!(
                out,
                "llama.cpp:     {:>8.1} tok/s  (TTFT: {:>6.1}ms)",
                tps, ttft
            );
        }
        let _ = writeln!(out);

        // Speedup Analysis
        let _ = writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        );
        let _ = writeln!(out, "SPEEDUP ANALYSIS");
        let _ = writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        );

        let ollama_tps = self
            .gguf_ollama
            .as_ref()
            .map_or(318.0, BenchMeasurement::mean_throughput);
        let llamacpp_tps = self
            .gguf_llamacpp
            .as_ref()
            .map_or(200.0, BenchMeasurement::mean_throughput);

        if let Some(ref m) = self.gguf_apr {
            let vs_ollama = m.mean_throughput() / ollama_tps;
            let vs_llamacpp = m.mean_throughput() / llamacpp_tps;
            let _ = writeln!(
                out,
                "APR GGUF vs Ollama:     {:>5.2}x  {}",
                vs_ollama,
                if vs_ollama >= 1.0 { "✓" } else { "⚠" }
            );
            let _ = writeln!(
                out,
                "APR GGUF vs llama.cpp:  {:>5.2}x  {}",
                vs_llamacpp,
                if vs_llamacpp >= 1.25 {
                    "✓ Point 41 PASS"
                } else {
                    "⚠ Point 41 FAIL"
                }
            );
        }

        if let Some(ref m) = self.apr_native {
            let vs_ollama = m.mean_throughput() / ollama_tps;
            let _ = writeln!(
                out,
                "APR .apr vs Ollama:     {:>5.2}x  {}",
                vs_ollama,
                if vs_ollama >= 2.0 {
                    "✓ 2x target"
                } else {
                    ""
                }
            );
        }
        let _ = writeln!(out);

        // Profiling Hotspots
        if !self.hotspots.is_empty() {
            let _ = writeln!(
                out,
                "───────────────────────────────────────────────────────────────────────"
            );
            let _ = writeln!(out, "PROFILING HOTSPOTS (>5% of execution time)");
            let _ = writeln!(
                out,
                "───────────────────────────────────────────────────────────────────────"
            );

            for hotspot in &self.hotspots {
                let _ = writeln!(out, "{}", hotspot.to_line(false));
                if !hotspot.explanation.is_empty() {
                    let _ = writeln!(out, "   └─ {}", hotspot.explanation);
                }
            }
            let _ = writeln!(out);
        }

        // GPU Metrics
        let _ = writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        );
        let _ = writeln!(out, "GPU METRICS");
        let _ = writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        );

        if let Some(ref m) = self.gguf_apr {
            if let (Some(util), Some(mem)) = (m.gpu_util, m.gpu_mem_mb) {
                let _ = writeln!(
                    out,
                    "APR GGUF:   GPU Util: {:>5.1}%  VRAM: {:>6.0}MB",
                    util, mem
                );
            }
        }
        if let Some(ref m) = self.apr_native {
            if let (Some(util), Some(mem)) = (m.gpu_util, m.gpu_mem_mb) {
                let _ = writeln!(
                    out,
                    "APR .apr:   GPU Util: {:>5.1}%  VRAM: {:>6.0}MB",
                    util, mem
                );
            }
        }
        let _ = writeln!(out);

        // Recommendations
        let _ = writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        );
        let _ = writeln!(out, "OPTIMIZATION RECOMMENDATIONS");
        let _ = writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        );

        let unexpected: Vec<_> = self.hotspots.iter().filter(|h| !h.is_expected).collect();
        if unexpected.is_empty() {
            let _ = writeln!(out, "✓ No unexpected hotspots detected");
        } else {
            for h in unexpected {
                let _ = writeln!(out, "⚠ {}: {}", h.component, h.explanation);
            }
        }

        let _ = writeln!(
            out,
            "═══════════════════════════════════════════════════════════════════════"
        );
        let _ = writeln!(out, "```");

        out
    }

    /// Generate compact one-liner for quick comparison
    pub fn render_compact(&self) -> String {
        let apr_tps = self
            .gguf_apr
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);
        let ollama_tps = self
            .gguf_ollama
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);
        let llamacpp_tps = self
            .gguf_llamacpp
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);

        format!(
            "APR:{:.0} Ollama:{:.0} llama.cpp:{:.0} tok/s | APR vs Ollama:{:.2}x vs llama.cpp:{:.2}x",
            apr_tps,
            ollama_tps,
            llamacpp_tps,
            apr_tps / ollama_tps.max(1.0),
            apr_tps / llamacpp_tps.max(1.0)
        )
    }
}
