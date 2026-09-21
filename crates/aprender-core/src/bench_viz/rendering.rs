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

        self.write_log_speedups(&mut out);
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

    /// The speedup lines of [`Self::render_profiling_log`].
    ///
    /// #3773: an absent comparator used to default to hard-coded Ollama /
    /// llama.cpp constants "from spec". A missing measurement is now UNMEASURED.
    fn write_log_speedups(&self, out: &mut String) {
        let ollama_tps = self
            .gguf_ollama
            .as_ref()
            .map(BenchMeasurement::mean_throughput);
        let llamacpp_tps = self
            .gguf_llamacpp
            .as_ref()
            .map(BenchMeasurement::mean_throughput);
        if let Some(ref m) = self.gguf_apr {
            let tps = m.mean_throughput();
            let _ = writeln!(
                out,
                "APR GGUF vs Ollama:     {}",
                ratio_verdict(tps, ollama_tps, 1.0, ("✓", "⚠"), "UNMEASURED")
            );
            let _ = writeln!(
                out,
                "APR GGUF vs llama.cpp:  {}",
                ratio_verdict(
                    tps,
                    llamacpp_tps,
                    1.25,
                    ("✓ Point 41 PASS", "⚠ Point 41 FAIL"),
                    "UNMEASURED (Point 41 not judged)"
                )
            );
        }
        if let Some(ref m) = self.apr_native {
            let _ = writeln!(
                out,
                "APR .apr vs Ollama:     {}",
                ratio_verdict(
                    m.mean_throughput(),
                    ollama_tps,
                    2.0,
                    ("✓ 2x target", ""),
                    "UNMEASURED"
                )
            );
        }
    }

    /// Generate compact one-liner for quick comparison
    ///
    /// #3773: an absent comparator used to read as 0 tok/s and divide as 1.0,
    /// printing the APR figure itself as the "speedup". It is now UNMEASURED.
    pub fn render_compact(&self) -> String {
        let apr_tps = self
            .gguf_apr
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);
        let ollama_tps = self
            .gguf_ollama
            .as_ref()
            .map(BenchMeasurement::mean_throughput);
        let llamacpp_tps = self
            .gguf_llamacpp
            .as_ref()
            .map(BenchMeasurement::mean_throughput);

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
