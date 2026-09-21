
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

        self.write_log_speedups(&mut out);
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

    /// The speedup lines of [`Self::render_profiling_log`].
    ///
    /// #3773: an absent comparator used to default to hard-coded Ollama /
    /// llama.cpp constants. A missing measurement is now UNMEASURED and yields
    /// no ratio.
    fn write_log_speedups(&self, out: &mut String) {
        let ollama_tps = self.gguf_ollama.as_ref().map(|m| m.tokens_per_sec);
        let llamacpp_tps = self.gguf_llamacpp.as_ref().map(|m| m.tokens_per_sec);
        if let Some(ref m) = self.gguf_apr {
            writeln!(
                out,
                "APR GGUF vs Ollama:     {}",
                ratio_verdict(m.tokens_per_sec, ollama_tps, 1.0, ("✓", "⚠"), "UNMEASURED")
            )
            .expect("failed to write benchmark output");
            writeln!(
                out,
                "APR GGUF vs llama.cpp:  {}",
                ratio_verdict(
                    m.tokens_per_sec,
                    llamacpp_tps,
                    1.25,
                    ("✓ Point 41 PASS", "⚠ Point 41 FAIL"),
                    "UNMEASURED (Point 41 not judged)"
                )
            )
            .expect("failed to write benchmark output");
        }
        if let Some(ref m) = self.apr_native {
            writeln!(
                out,
                "APR .apr vs Ollama:     {}",
                ratio_verdict(
                    m.tokens_per_sec,
                    ollama_tps,
                    2.0,
                    ("✓ 2x target", ""),
                    "UNMEASURED"
                )
            )
            .expect("failed to write benchmark output");
        }
    }

    /// Generate compact one-liner for quick comparison
    ///
    /// #3773: an absent comparator used to read as 0 tok/s and divide as 1.0,
    /// printing the APR figure itself as the "speedup". It is now UNMEASURED.
    pub fn render_compact(&self) -> String {
        let apr_tps = self.gguf_apr.as_ref().map_or(0.0, |m| m.tokens_per_sec);
        let ollama_tps = self.gguf_ollama.as_ref().map(|m| m.tokens_per_sec);
        let llamacpp_tps = self.gguf_llamacpp.as_ref().map(|m| m.tokens_per_sec);

        format!(
            "APR:{:.0} Ollama:{} llama.cpp:{} tok/s | APR vs Ollama:{} vs llama.cpp:{}",
            apr_tps,
            compact_tps(ollama_tps),
            compact_tps(llamacpp_tps),
            compact_ratio(apr_tps, ollama_tps),
            compact_ratio(apr_tps, llamacpp_tps)
        )
    }
}

/// `"<ratio>x  <mark>"` against a measured comparator — the pass mark at or
/// above `pass_at`, else the fail mark — or `unmeasured` when there is none.
fn ratio_verdict(
    apr_tps: f64,
    other: Option<f64>,
    pass_at: f64,
    (pass, fail): (&str, &str),
    unmeasured: &str,
) -> String {
    match other {
        Some(o) => {
            let r = apr_tps / o;
            format!("{:>5.2}x  {}", r, if r >= pass_at { pass } else { fail })
        }
        None => unmeasured.to_string(),
    }
}

/// A comparator's throughput, or UNMEASURED when none was recorded.
fn compact_tps(tps: Option<f64>) -> String {
    tps.map_or_else(|| "UNMEASURED".to_string(), |t| format!("{t:.0}"))
}

/// APR over a comparator, or UNMEASURED when there is no positive comparator.
fn compact_ratio(apr_tps: f64, other: Option<f64>) -> String {
    match other {
        Some(o) if o > 0.0 => format!("{:.2}x", apr_tps / o),
        _ => "UNMEASURED".to_string(),
    }
}

include!("bench_viz_model_gpu_benchmark.rs");
