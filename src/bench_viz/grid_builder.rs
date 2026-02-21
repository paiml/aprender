impl BenchmarkGrid {
    /// Create new benchmark grid
    pub fn new() -> Self {
        Self {
            config: BenchConfig::default(),
            ..Default::default()
        }
    }

    /// Set configuration
    #[must_use]
    pub fn with_config(mut self, config: BenchConfig) -> Self {
        self.config = config;
        self
    }

    /// Set model info
    #[must_use]
    pub fn with_model(mut self, name: &str, params: &str, quant: &str) -> Self {
        self.model_name = name.to_string();
        self.model_params = params.to_string();
        self.quantization = quant.to_string();
        self
    }

    /// Set GPU info
    #[must_use]
    pub fn with_gpu(mut self, name: &str, vram_gb: f64) -> Self {
        self.gpu_name = name.to_string();
        self.gpu_vram_gb = vram_gb;
        self
    }

    /// Add GGUF row measurements
    pub fn set_gguf_row(
        &mut self,
        mut apr: BenchMeasurement,
        mut ollama: BenchMeasurement,
        mut llamacpp: BenchMeasurement,
    ) {
        apr.compute_stats(self.config.outlier_threshold);
        ollama.compute_stats(self.config.outlier_threshold);
        llamacpp.compute_stats(self.config.outlier_threshold);
        self.gguf_apr = Some(apr);
        self.gguf_ollama = Some(ollama);
        self.gguf_llamacpp = Some(llamacpp);
    }

    /// Add APR row measurements
    pub fn set_apr_row(
        &mut self,
        mut native: BenchMeasurement,
        mut gguf: BenchMeasurement,
        mut baseline: BenchMeasurement,
    ) {
        native.compute_stats(self.config.outlier_threshold);
        gguf.compute_stats(self.config.outlier_threshold);
        baseline.compute_stats(self.config.outlier_threshold);
        self.apr_native = Some(native);
        self.apr_gguf = Some(gguf);
        self.apr_baseline = Some(baseline);
    }

    /// Add profiling hotspot
    pub fn add_hotspot(&mut self, hotspot: ProfilingHotspot) {
        self.hotspots.push(hotspot);
    }

    // ========================================================================
    // Colored Terminal Visualization
    // ========================================================================

    /// Render as rich colored ASCII grid for terminal
    pub fn render(&self) -> String {
        let use_colors = self.config.colors;
        let (bold, reset, cyan, green, yellow, dim) = if use_colors {
            (
                colors::BOLD,
                colors::RESET,
                colors::CYAN,
                colors::GREEN,
                colors::YELLOW,
                colors::DIM,
            )
        } else {
            ("", "", "", "", "", "")
        };

        let mut out = String::new();

        // Header with colors
        let _ = writeln!(out, "{cyan}╔═══════════════════════════════════════════════════════════════════════╗{reset}");
        let _ = writeln!(out, "{cyan}║{reset} {bold}          INFERENCE BENCHMARK COMPARISON (tok/s GPU){reset}                  {cyan}║{reset}");
        let _ = writeln!(
            out,
            "{cyan}║{reset}  Model: {bold}{:30}{reset} Quant: {bold}{:10}{reset}         {cyan}║{reset}",
            truncate(&self.model_name, 30),
            truncate(&self.quantization, 10)
        );
        let _ = writeln!(
            out,
            "{cyan}║{reset}  GPU: {:35} VRAM: {:5.1}GB              {cyan}║{reset}",
            truncate(&self.gpu_name, 35),
            self.gpu_vram_gb
        );

        // Iteration count
        let iterations = self
            .gguf_apr
            .as_ref()
            .map_or(1, |m| m.throughput_samples.len());
        let _ = writeln!(
            out,
            "{cyan}║{reset}  {dim}Iterations: {} (warmup: {}){reset}                                        {cyan}║{reset}",
            iterations, self.config.warmup_iterations
        );
        let _ = writeln!(out, "{cyan}╠═══════════════════════════════════════════════════════════════════════╣{reset}");

        // Row 1: GGUF comparison
        let _ = writeln!(out, "{cyan}║{reset} {bold}                   GGUF Format Inference{reset}                              {cyan}║{reset}");
        let _ = writeln!(out, "{cyan}╠═══════════════════════╦═══════════════════════╦═══════════════════════╣{reset}");
        let _ = writeln!(out, "{cyan}║{reset}  {green}APR serve GGUF{reset}       {cyan}║{reset}       Ollama          {cyan}║{reset}      llama.cpp        {cyan}║{reset}");
        let _ = writeln!(out, "{cyan}╠═══════════════════════╬═══════════════════════╬═══════════════════════╣{reset}");

        let gguf_apr_tps = self
            .gguf_apr
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);
        let gguf_ollama_tps = self
            .gguf_ollama
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);
        let gguf_llamacpp_tps = self
            .gguf_llamacpp
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);

        // Values with colors based on performance
        let apr_color = if gguf_apr_tps > gguf_ollama_tps {
            green
        } else {
            yellow
        };
        let _ = writeln!(
            out,
            "{cyan}║{reset}  {apr_color}{:>8.1}{reset} tok/s      {cyan}║{reset}  {:>8.1} tok/s      {cyan}║{reset}  {:>8.1} tok/s      {cyan}║{reset}",
            gguf_apr_tps, gguf_ollama_tps, gguf_llamacpp_tps
        );

        // Confidence intervals
        if let Some(ref m) = self.gguf_apr {
            if let Some(ref stats) = m.throughput_stats {
                let _ = writeln!(
                    out,
                    "{cyan}║{reset}  {dim}[{:.1} - {:.1}]{reset}       {cyan}║{reset}  {dim}{}{reset}      {cyan}║{reset}  {dim}{}{reset}      {cyan}║{reset}",
                    stats.ci_95.0,
                    stats.ci_95.1,
                    self.gguf_ollama
                        .as_ref()
                        .and_then(|m| m.throughput_stats.as_ref())
                        .map_or_else(String::new, |s| format!("[{:.1} - {:.1}]", s.ci_95.0, s.ci_95.1)),
                    self.gguf_llamacpp
                        .as_ref()
                        .and_then(|m| m.throughput_stats.as_ref())
                        .map_or_else(String::new, |s| format!("[{:.1} - {:.1}]", s.ci_95.0, s.ci_95.1))
                );
            }
        }

        // Bar visualization with colors
        let max_tps = [gguf_apr_tps, gguf_ollama_tps, gguf_llamacpp_tps]
            .iter()
            .copied()
            .fold(1.0, f64::max);

        let _ = writeln!(
            out,
            "{cyan}║{reset}  {}  {cyan}║{reset}  {}  {cyan}║{reset}  {}  {cyan}║{reset}",
            render_bar_colored(gguf_apr_tps, max_tps, 17, use_colors, true),
            render_bar_colored(gguf_ollama_tps, max_tps, 17, use_colors, false),
            render_bar_colored(gguf_llamacpp_tps, max_tps, 17, use_colors, false)
        );

        // TTFT
        let gguf_apr_ttft = self
            .gguf_apr
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_ttft);
        let gguf_ollama_ttft = self
            .gguf_ollama
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_ttft);
        let gguf_llamacpp_ttft = self
            .gguf_llamacpp
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_ttft);

        let ttft_apr_color = if gguf_apr_ttft < gguf_ollama_ttft {
            green
        } else {
            yellow
        };
        let _ = writeln!(
            out,
            "{cyan}║{reset}  TTFT: {ttft_apr_color}{:>6.1}ms{reset}      {cyan}║{reset}  TTFT: {:>6.1}ms      {cyan}║{reset}  TTFT: {:>6.1}ms      {cyan}║{reset}",
            gguf_apr_ttft, gguf_ollama_ttft, gguf_llamacpp_ttft
        );

        // Row 2: APR server comparison
        let _ = writeln!(out, "{cyan}╠═══════════════════════╩═══════════════════════╩═══════════════════════╣{reset}");
        let _ = writeln!(out, "{cyan}║{reset} {bold}                  APR Server Format Comparison{reset}                        {cyan}║{reset}");
        let _ = writeln!(out, "{cyan}╠═══════════════════════╦═══════════════════════╦═══════════════════════╣{reset}");
        let _ = writeln!(out, "{cyan}║{reset}  {green}APR serve .apr{reset}       {cyan}║{reset}  APR serve GGUF       {cyan}║{reset}  Ollama (baseline)    {cyan}║{reset}");
        let _ = writeln!(out, "{cyan}╠═══════════════════════╬═══════════════════════╬═══════════════════════╣{reset}");

        let apr_native_tps = self
            .apr_native
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);
        let apr_gguf_tps = self
            .apr_gguf
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);
        let apr_baseline_tps = self
            .apr_baseline
            .as_ref()
            .map_or(0.0, BenchMeasurement::mean_throughput);

        let native_color = if apr_native_tps > apr_gguf_tps {
            green
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "{cyan}║{reset}  {native_color}{:>8.1}{reset} tok/s      {cyan}║{reset}  {:>8.1} tok/s      {cyan}║{reset}  {:>8.1} tok/s      {cyan}║{reset}",
            apr_native_tps, apr_gguf_tps, apr_baseline_tps
        );

        let max_tps2 = [apr_native_tps, apr_gguf_tps, apr_baseline_tps]
            .iter()
            .copied()
            .fold(1.0, f64::max);

        let _ = writeln!(
            out,
            "{cyan}║{reset}  {}  {cyan}║{reset}  {}  {cyan}║{reset}  {}  {cyan}║{reset}",
            render_bar_colored(apr_native_tps, max_tps2, 17, use_colors, true),
            render_bar_colored(apr_gguf_tps, max_tps2, 17, use_colors, false),
            render_bar_colored(apr_baseline_tps, max_tps2, 17, use_colors, false)
        );

        // Speedup vs baseline with color-coded pass/fail
        let speedup_native = if apr_baseline_tps > 0.0 {
            apr_native_tps / apr_baseline_tps
        } else {
            0.0
        };
        let speedup_gguf = if apr_baseline_tps > 0.0 {
            apr_gguf_tps / apr_baseline_tps
        } else {
            0.0
        };

        let speedup_color = if speedup_native >= 2.0 {
            green
        } else if speedup_native >= 1.5 {
            yellow
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "{cyan}║{reset}  vs Ollama: {speedup_color}{:>5.2}x{reset}   {cyan}║{reset}  vs Ollama: {:>5.2}x   {cyan}║{reset}  (baseline)           {cyan}║{reset}",
            speedup_native, speedup_gguf
        );

        let _ = writeln!(out, "{cyan}╚═══════════════════════╩═══════════════════════╩═══════════════════════╝{reset}");

        out
    }

    /// Render scientific-style benchmark report
    pub fn render_scientific(&self) -> String {
        let use_colors = self.config.colors;
        let (bold, reset, cyan, green, dim) = if use_colors {
            (
                colors::BOLD,
                colors::RESET,
                colors::CYAN,
                colors::GREEN,
                colors::DIM,
            )
        } else {
            ("", "", "", "", "")
        };

        let mut out = String::new();

        let _ = writeln!(out, "\n{bold}Benchmark Results (criterion-style){reset}");
        let _ = writeln!(out, "{}", "─".repeat(72));
        let _ = writeln!(
            out,
            "{dim}Model: {} | Quant: {} | GPU: {}{reset}",
            self.model_name, self.quantization, self.gpu_name
        );
        let _ = writeln!(
            out,
            "{dim}Iterations: {} (warmup: {}) | Outlier threshold: {:.1}σ{reset}\n",
            self.config.iterations, self.config.warmup_iterations, self.config.outlier_threshold
        );

        // Throughput results
        let _ = writeln!(out, "{cyan}Throughput (tok/s):{reset}");
        if let Some(ref m) = self.gguf_apr {
            if let Some(ref stats) = m.throughput_stats {
                let _ = writeln!(
                    out,
                    "  {}",
                    stats.format_criterion("APR GGUF", "tok/s", use_colors)
                );
            }
        }
        if let Some(ref m) = self.apr_native {
            if let Some(ref stats) = m.throughput_stats {
                let _ = writeln!(
                    out,
                    "  {}",
                    stats.format_criterion("APR .apr", "tok/s", use_colors)
                );
            }
        }
        if let Some(ref m) = self.gguf_ollama {
            if let Some(ref stats) = m.throughput_stats {
                let _ = writeln!(
                    out,
                    "  {}",
                    stats.format_criterion("Ollama", "tok/s", use_colors)
                );
            }
        }
        if let Some(ref m) = self.gguf_llamacpp {
            if let Some(ref stats) = m.throughput_stats {
                let _ = writeln!(
                    out,
                    "  {}",
                    stats.format_criterion("llama.cpp", "tok/s", use_colors)
                );
            }
        }

        // TTFT results
        let _ = writeln!(out, "\n{cyan}Time to First Token (ms):{reset}");
        if let Some(ref m) = self.gguf_apr {
            if let Some(ref stats) = m.ttft_stats {
                let _ = writeln!(
                    out,
                    "  {}",
                    stats.format_criterion("APR GGUF", "ms", use_colors)
                );
            }
        }
        if let Some(ref m) = self.apr_native {
            if let Some(ref stats) = m.ttft_stats {
                let _ = writeln!(
                    out,
                    "  {}",
                    stats.format_criterion("APR .apr", "ms", use_colors)
                );
            }
        }
        if let Some(ref m) = self.gguf_ollama {
            if let Some(ref stats) = m.ttft_stats {
                let _ = writeln!(
                    out,
                    "  {}",
                    stats.format_criterion("Ollama", "ms", use_colors)
                );
            }
        }

        // Speedup analysis
        let _ = writeln!(out, "\n{cyan}Speedup Analysis:{reset}");
        let ollama_tps = self
            .gguf_ollama
            .as_ref()
            .map_or(318.0, BenchMeasurement::mean_throughput);
        let llamacpp_tps = self
            .gguf_llamacpp
            .as_ref()
            .map_or(200.0, BenchMeasurement::mean_throughput);

        if let Some(ref m) = self.gguf_apr {
            let tps = m.mean_throughput();
            let vs_ollama = tps / ollama_tps;
            let vs_llamacpp = tps / llamacpp_tps;
            let pass_color = if vs_llamacpp >= 1.25 {
                green
            } else {
                colors::RED
            };
            let _ = writeln!(out, "  APR GGUF vs Ollama:     {:.2}x", vs_ollama);
            let _ = writeln!(
                out,
                "  APR GGUF vs llama.cpp:  {pass_color}{:.2}x{reset} {}",
                vs_llamacpp,
                if vs_llamacpp >= 1.25 {
                    "✓ Point 41 PASS"
                } else {
                    "✗ Point 41 FAIL"
                }
            );
        }

        let _ = writeln!(out, "{}", "─".repeat(72));

        out
    }
}
