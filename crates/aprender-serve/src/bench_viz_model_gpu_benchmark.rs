impl BenchmarkGrid {
    /// Create new benchmark grid
    pub fn new() -> Self {
        Self::default()
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
        apr: BenchMeasurement,
        ollama: BenchMeasurement,
        llamacpp: BenchMeasurement,
    ) {
        self.gguf_apr = Some(apr);
        self.gguf_ollama = Some(ollama);
        self.gguf_llamacpp = Some(llamacpp);
    }

    /// Add APR row measurements
    pub fn set_apr_row(
        &mut self,
        native: BenchMeasurement,
        gguf: BenchMeasurement,
        baseline: BenchMeasurement,
    ) {
        self.apr_native = Some(native);
        self.apr_gguf = Some(gguf);
        self.apr_baseline = Some(baseline);
    }

    /// Add profiling hotspot
    pub fn add_hotspot(&mut self, hotspot: ProfilingHotspot) {
        self.hotspots.push(hotspot);
    }

    // ========================================================================
    // Terminal Visualization (ASCII)
    // ========================================================================

    /// Render as ASCII grid for terminal
    pub fn render_ascii(&self) -> String {
        let mut out = String::new();

        // Header
        writeln!(
            out,
            "╔═══════════════════════════════════════════════════════════════════════╗"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "║           INFERENCE BENCHMARK COMPARISON (tok/s GPU)                  ║"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "║  Model: {:30} Quant: {:10}         ║",
            truncate(&self.model_name, 30),
            truncate(&self.quantization, 10)
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "║  GPU: {:35} VRAM: {:5.1}GB              ║",
            truncate(&self.gpu_name, 35),
            self.gpu_vram_gb
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "╠═══════════════════════════════════════════════════════════════════════╣"
        )
        .expect("failed to write benchmark output");

        // Row 1: GGUF comparison
        writeln!(
            out,
            "║                    GGUF Format Inference                              ║"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "╠═══════════════════════╦═══════════════════════╦═══════════════════════╣"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "║    APR serve GGUF     ║       Ollama          ║      llama.cpp        ║"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "╠═══════════════════════╬═══════════════════════╬═══════════════════════╣"
        )
        .expect("failed to write benchmark output");

        let gguf_apr_tps = self.gguf_apr.as_ref().map_or(0.0, |m| m.tokens_per_sec);
        let gguf_ollama_tps = self.gguf_ollama.as_ref().map_or(0.0, |m| m.tokens_per_sec);
        let gguf_llamacpp_tps = self
            .gguf_llamacpp
            .as_ref()
            .map_or(0.0, |m| m.tokens_per_sec);

        writeln!(
            out,
            "║  {:>8.1} tok/s      ║  {:>8.1} tok/s      ║  {:>8.1} tok/s      ║",
            gguf_apr_tps, gguf_ollama_tps, gguf_llamacpp_tps
        )
        .expect("failed to write benchmark output");

        // Bar visualization
        let max_tps = [gguf_apr_tps, gguf_ollama_tps, gguf_llamacpp_tps]
            .iter()
            .cloned()
            .fold(1.0, f64::max);

        writeln!(
            out,
            "║  {}  ║  {}  ║  {}  ║",
            render_bar(gguf_apr_tps, max_tps, 17),
            render_bar(gguf_ollama_tps, max_tps, 17),
            render_bar(gguf_llamacpp_tps, max_tps, 17)
        )
        .expect("failed to write benchmark output");

        // TTFT
        let gguf_apr_ttft = self.gguf_apr.as_ref().map_or(0.0, |m| m.ttft_ms);
        let gguf_ollama_ttft = self.gguf_ollama.as_ref().map_or(0.0, |m| m.ttft_ms);
        let gguf_llamacpp_ttft = self.gguf_llamacpp.as_ref().map_or(0.0, |m| m.ttft_ms);

        writeln!(
            out,
            "║  TTFT: {:>6.1}ms      ║  TTFT: {:>6.1}ms      ║  TTFT: {:>6.1}ms      ║",
            gguf_apr_ttft, gguf_ollama_ttft, gguf_llamacpp_ttft
        )
        .expect("failed to write benchmark output");

        // Row 2: APR server comparison
        writeln!(
            out,
            "╠═══════════════════════╩═══════════════════════╩═══════════════════════╣"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "║                   APR Server Format Comparison                        ║"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "╠═══════════════════════╦═══════════════════════╦═══════════════════════╣"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "║   APR serve .apr      ║   APR serve GGUF      ║  Ollama (baseline)    ║"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "╠═══════════════════════╬═══════════════════════╬═══════════════════════╣"
        )
        .expect("failed to write benchmark output");

        let apr_native_tps = self.apr_native.as_ref().map_or(0.0, |m| m.tokens_per_sec);
        let apr_gguf_tps = self.apr_gguf.as_ref().map_or(0.0, |m| m.tokens_per_sec);
        let apr_baseline_tps = self.apr_baseline.as_ref().map_or(0.0, |m| m.tokens_per_sec);

        writeln!(
            out,
            "║  {:>8.1} tok/s      ║  {:>8.1} tok/s      ║  {:>8.1} tok/s      ║",
            apr_native_tps, apr_gguf_tps, apr_baseline_tps
        )
        .expect("failed to write benchmark output");

        let max_tps2 = [apr_native_tps, apr_gguf_tps, apr_baseline_tps]
            .iter()
            .cloned()
            .fold(1.0, f64::max);

        writeln!(
            out,
            "║  {}  ║  {}  ║  {}  ║",
            render_bar(apr_native_tps, max_tps2, 17),
            render_bar(apr_gguf_tps, max_tps2, 17),
            render_bar(apr_baseline_tps, max_tps2, 17)
        )
        .expect("failed to write benchmark output");

        // Speedup vs baseline
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

        writeln!(
            out,
            "║  vs Ollama: {:>5.2}x   ║  vs Ollama: {:>5.2}x   ║  (baseline)           ║",
            speedup_native, speedup_gguf
        )
        .expect("failed to write benchmark output");

        writeln!(
            out,
            "╚═══════════════════════╩═══════════════════════╩═══════════════════════╝"
        )
        .expect("failed to write benchmark output");

        out
    }
}
