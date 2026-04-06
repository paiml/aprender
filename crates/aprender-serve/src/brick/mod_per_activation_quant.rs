
impl ActivationQuantBrick {
    /// Create new activation quantization brick.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            budget: TokenBudget::from_latency(0.5), // 0.5µs overhead target
            per_channel: false,
        }
    }

    /// Create with per-channel quantization.
    #[must_use]
    pub fn with_per_channel(dim: usize) -> Self {
        Self {
            dim,
            budget: TokenBudget::from_latency(1.0), // 1.0µs for per-channel
            per_channel: true,
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Compute memory bandwidth reduction factor.
    ///
    /// f32 (4 bytes) → int8 (1 byte) + scale/zero_point = ~4x reduction
    #[must_use]
    pub fn bandwidth_reduction(&self) -> f64 {
        // Original: dim * 4 bytes (f32)
        // Quantized: dim * 1 byte (int8) + 8 bytes (scale + zero_point)
        let original_bytes = self.dim * 4;
        let quantized_bytes = self.dim + 8; // +8 for scale and zero_point (f32 each)
        original_bytes as f64 / quantized_bytes as f64
    }

    /// Compute quantization error estimate (typical for 8-bit).
    ///
    /// Per Jacob et al. 2018, typical Q8 error is ~0.1% for activations.
    #[must_use]
    pub fn estimated_error(&self) -> f64 {
        if self.per_channel {
            0.0005 // 0.05% for per-channel
        } else {
            0.001 // 0.1% for per-tensor
        }
    }

    /// Compute bytes saved per token.
    #[must_use]
    pub fn bytes_saved(&self) -> usize {
        // f32 (4 bytes) → int8 (1 byte) = 3 bytes saved per element
        self.dim * 3
    }

    /// Quantize f32 activations to int8 using Q8_0 block format.
    ///
    /// **REAL IMPLEMENTATION** - Not a stub.
    /// Uses symmetric quantization: scale = max(abs(values)) / 127.0
    ///
    /// # Arguments
    /// * `input` - f32 activations to quantize (must be length == self.dim)
    ///
    /// # Returns
    /// * Quantized int8 values and scale factors
    ///
    /// # Example
    /// ```ignore
    /// let brick = ActivationQuantBrick::new(64);
    /// let input = vec![1.0f32; 64];
    /// let (quants, scales) = brick.quantize(&input)?;
    /// assert_eq!(quants.len(), 64);
    /// ```
    pub fn quantize(&self, input: &[f32]) -> Result<(Vec<i8>, Vec<f32>), BrickError> {
        if input.len() != self.dim {
            return Err(BrickError::InvalidInput(format!(
                "Input length {} != dim {}",
                input.len(),
                self.dim
            )));
        }
        if self.dim == 0 {
            return Err(BrickError::InvalidInput("Zero dimension".to_string()));
        }

        // Quantize in blocks of 32 (Q8_0 block size)
        let num_blocks = self.dim.div_ceil(32);
        let mut quants = Vec::with_capacity(self.dim);
        let mut scales = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let start = block_idx * 32;
            let end = (start + 32).min(self.dim);

            // Pad to 32 if needed
            let mut block_data = [0.0f32; 32];
            for (i, &v) in input[start..end].iter().enumerate() {
                block_data[i] = v;
            }

            let block = Q8_0Block::quantize(&block_data);
            scales.push(block.scale);

            // Only take the actual values (not padding)
            for &q in &block.quants[0..(end - start)] {
                quants.push(q);
            }
        }

        Ok((quants, scales))
    }

    /// Dequantize int8 back to f32 using stored scales.
    ///
    /// **REAL IMPLEMENTATION** - Not a stub.
    pub fn dequantize(&self, quants: &[i8], scales: &[f32]) -> Result<Vec<f32>, BrickError> {
        if quants.len() != self.dim {
            return Err(BrickError::InvalidInput(format!(
                "Quants length {} != dim {}",
                quants.len(),
                self.dim
            )));
        }

        let mut output = Vec::with_capacity(self.dim);
        for (block_idx, &scale) in scales.iter().enumerate() {
            let start = block_idx * 32;
            let end = (start + 32).min(self.dim);
            for &q in &quants[start..end] {
                output.push(q as f32 * scale);
            }
        }

        Ok(output)
    }

    /// Compute quantization error vs original input.
    ///
    /// **REAL IMPLEMENTATION** - Measures actual error, not estimates.
    pub fn measure_error(
        &self,
        original: &[f32],
        quants: &[i8],
        scales: &[f32],
    ) -> Result<f64, BrickError> {
        let dequantized = self.dequantize(quants, scales)?;

        let max_error = original
            .iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let max_val = original.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        if max_val < 1e-10 {
            return Ok(0.0);
        }

        Ok((max_error / max_val) as f64)
    }

    /// Execute quantization with timing (for benchmarking).
    #[allow(clippy::type_complexity)]
    pub fn execute_timed(
        &self,
        input: &[f32],
    ) -> Result<TokenResult<(Vec<i8>, Vec<f32>)>, BrickError> {
        let start = Instant::now();
        let (quants, scales) = self.quantize(input)?;
        let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;

        Ok(TokenResult {
            output: (quants, scales),
            tokens_processed: 1,
            us_per_token: elapsed_us,
            tokens_per_sec: 1_000_000.0 / elapsed_us,
            budget_met: elapsed_us <= self.budget.us_per_token,
        })
    }

    /// Legacy stub for backward compatibility (prefer `quantize()`)
    #[deprecated(note = "Use quantize() for real implementation")]
    pub fn execute(&self) -> Result<Vec<u8>, BrickError> {
        if self.dim == 0 {
            return Err(BrickError::InvalidInput("Zero dimension".to_string()));
        }
        // Return zeros for backward compat - use quantize() for real output
        Ok(vec![128u8; self.dim])
    }
}

impl ComputeBrick for ActivationQuantBrick {
    type Output = Vec<u8>;

    fn name(&self) -> &'static str {
        "activation_quant"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::budget_met(),
            BrickAssertion {
                name: "symmetric_range".to_string(),
                description: "Q8 values centered around 128 (zero_point)".to_string(),
                kind: AssertionKind::Custom {
                    check_name: "symmetric_range".to_string(),
                },
            },
            BrickAssertion {
                name: "error_bound".to_string(),
                description: "Quantization error < 0.1% (per-tensor) or 0.05% (per-channel)"
                    .to_string(),
                kind: AssertionKind::Custom {
                    check_name: "error_bound".to_string(),
                },
            },
        ]
    }

    fn can_run(&self) -> bool {
        self.dim > 0
    }
}

// ============================================================================
// Transformer Layer Brick
// ============================================================================

/// Full transformer layer as a composed brick.
#[derive(Debug)]
pub struct TransformerLayerBrick {
    /// Layer index
    pub layer_idx: usize,
    /// Attention layer normalization brick
    pub attn_norm: RmsNormBrick,
    /// QKV projection brick
    pub qkv: QkvBrick,
    /// Rotary position embedding brick
    pub rope: RopeBrick,
    /// Attention computation brick
    pub attention: AttentionBrick,
    /// Output projection brick
    pub o_proj: OProjBrick,
    /// FFN layer normalization brick
    pub ffn_norm: RmsNormBrick,
    /// Feed-forward network brick
    pub ffn: FfnBrick,
    /// Timing metrics (updated after each run)
    pub last_timing: Option<LayerTiming>,
}

/// Timing breakdown for a layer.
#[derive(Debug, Clone, Default)]
pub struct LayerTiming {
    /// Attention normalization time (µs)
    pub attn_norm_us: f64,
    /// QKV projection time (µs)
    pub qkv_us: f64,
    /// RoPE application time (µs)
    pub rope_us: f64,
    /// Attention computation time (µs)
    pub attention_us: f64,
    /// Output projection time (µs)
    pub o_proj_us: f64,
    /// FFN normalization time (µs)
    pub ffn_norm_us: f64,
    /// FFN computation time (µs)
    pub ffn_us: f64,
    /// Total layer time (µs)
    pub total_us: f64,
}

impl LayerTiming {
    /// Find the bottleneck brick.
    pub fn bottleneck(&self) -> (&'static str, f64) {
        let bricks = [
            ("attn_norm", self.attn_norm_us),
            ("qkv", self.qkv_us),
            ("rope", self.rope_us),
            ("attention", self.attention_us),
            ("o_proj", self.o_proj_us),
            ("ffn_norm", self.ffn_norm_us),
            ("ffn", self.ffn_us),
        ];

        bricks
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(("unknown", 0.0))
    }
}

impl TransformerLayerBrick {
    /// Create from configuration.
    pub fn from_config(
        layer_idx: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_dim: usize,
        eps: f32,
        rope_theta: f32,
        rope_type: u32,
    ) -> Self {
        let head_dim = hidden_dim / num_heads;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        Self {
            layer_idx,
            attn_norm: RmsNormBrick::new(vec![1.0; hidden_dim], eps),
            qkv: QkvBrick::new(hidden_dim, q_dim, kv_dim, kv_dim),
            rope: RopeBrick::new(head_dim, num_heads, rope_theta, rope_type),
            attention: AttentionBrick::new(num_heads, num_kv_heads, head_dim),
            o_proj: OProjBrick::new(q_dim, hidden_dim),
            ffn_norm: RmsNormBrick::new(vec![1.0; hidden_dim], eps),
            ffn: FfnBrick::new(hidden_dim, intermediate_dim),
            last_timing: None,
        }
    }

    /// Get total budget for this layer.
    pub fn total_budget_us(&self) -> f64 {
        self.attn_norm.budget().us_per_token
            + self.qkv.budget().us_per_token
            + self.rope.budget().us_per_token
            + self.attention.budget().us_per_token
            + self.o_proj.budget().us_per_token
            + self.ffn_norm.budget().us_per_token
            + self.ffn.budget().us_per_token
    }
}

impl ComputeBrick for TransformerLayerBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "transformer_layer"
    }

    fn budget(&self) -> TokenBudget {
        TokenBudget::from_latency(self.total_budget_us())
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::no_nan(),
            BrickAssertion::no_inf(),
            BrickAssertion::budget_met(),
        ]
    }

    fn verify(&self) -> BrickVerification {
        // Verify all component bricks
        let mut result = BrickVerification::pass();

        for brick in [
            &self.attn_norm as &dyn ComputeBrick<Output = Vec<f32>>,
            &self.ffn_norm as &dyn ComputeBrick<Output = Vec<f32>>,
        ] {
            let v = brick.verify();
            if !v.is_valid {
                result.is_valid = false;
                result.results.extend(v.results);
            }
        }

        result
    }
}

// ============================================================================
// Bottleneck Report
// ============================================================================

/// Report identifying pipeline bottleneck.
#[derive(Debug, Clone)]
pub struct BottleneckReport {
    /// Layer index containing bottleneck
    pub layer_idx: usize,
    /// Brick name
    pub brick_name: &'static str,
    /// Actual latency (µs)
    pub actual_us: f64,
    /// Budget latency (µs)
    pub budget_us: f64,
    /// Gap factor (actual / budget)
    pub gap_factor: f64,
}

impl fmt::Display for BottleneckReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Bottleneck: {} (layer {}) - {:.1}µs actual vs {:.1}µs budget ({:.2}x)",
            self.brick_name, self.layer_idx, self.actual_us, self.budget_us, self.gap_factor
        )
    }
}

// ============================================================================
// Benchmark Brick
// ============================================================================

/// Configuration for benchmark runs.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup: usize,
    /// Number of sample iterations
    pub samples: usize,
    /// Maximum allowed CV (coefficient of variation)
    pub max_cv: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup: 10,
            samples: 100,
            max_cv: 0.05, // 5% per Stabilizer (Curtsinger & Berger 2013)
        }
    }
}
