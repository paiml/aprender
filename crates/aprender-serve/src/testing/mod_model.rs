
impl ModelTestCase {
    /// Create a simple test case
    pub fn new(
        desc: impl Into<String>,
        config: ModelConfig,
        format: ModelFormat,
        device: Device,
    ) -> Self {
        Self {
            desc: desc.into(),
            constructor: ConstructorInput::new(config),
            forward: ForwardInput::new(vec![1, 2, 3, 4]),
            expected_output_norm: None,
            source_format: format,
            target_format: None,
            device,
        }
    }

    /// Create a conversion test case
    pub fn conversion(
        desc: impl Into<String>,
        config: ModelConfig,
        source: ModelFormat,
        target: ModelFormat,
        device: Device,
    ) -> Self {
        Self {
            desc: desc.into(),
            constructor: ConstructorInput::new(config),
            forward: ForwardInput::new(vec![1, 2, 3, 4]),
            expected_output_norm: None,
            source_format: source,
            target_format: Some(target),
            device,
        }
    }

    /// Set quantization type
    #[must_use]
    pub fn with_quant(mut self, quant: QuantType) -> Self {
        self.constructor.quantization = Some(quant);
        self
    }

    /// Set forward input tokens
    #[must_use]
    pub fn with_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.forward.tokens = tokens;
        self
    }

    /// Set expected output norm for validation
    #[must_use]
    pub fn with_expected_norm(mut self, norm: f32) -> Self {
        self.expected_output_norm = Some(norm);
        self
    }

    /// Check if this is a conversion test
    pub fn is_conversion_test(&self) -> bool {
        self.target_format.is_some()
    }
}

/// Result of running a model fixture test
#[derive(Debug)]
pub struct TestResult {
    /// Test case that was run
    pub test_case: String,
    /// Whether the test passed
    pub passed: bool,
    /// Output logits (if successful)
    pub output: Option<Vec<f32>>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Execution time in microseconds
    pub duration_us: u64,
    /// Memory usage in bytes
    pub memory_bytes: Option<usize>,
}

impl TestResult {
    /// Create a successful result
    pub fn success(test_case: &str, output: Vec<f32>, duration_us: u64) -> Self {
        Self {
            test_case: test_case.to_string(),
            passed: true,
            output: Some(output),
            error: None,
            duration_us,
            memory_bytes: None,
        }
    }

    /// Create a failure result
    pub fn failure(test_case: &str, error: impl Into<String>, duration_us: u64) -> Self {
        Self {
            test_case: test_case.to_string(),
            passed: false,
            output: None,
            error: Some(error.into()),
            duration_us,
            memory_bytes: None,
        }
    }

    /// Compute L2 norm of output
    pub fn output_l2_norm(&self) -> Option<f32> {
        self.output
            .as_ref()
            .map(|o| o.iter().map(|x| x * x).sum::<f32>().sqrt())
    }
}

/// Tolerance thresholds for numerical comparisons
#[derive(Debug, Clone, Copy)]
pub struct Tolerances {
    /// Absolute tolerance for F32 comparisons
    pub f32_abs: f32,
    /// Relative tolerance for F32 comparisons
    pub f32_rel: f32,
    /// Tolerance for quantized weight comparisons (% L2)
    pub quant_l2_pct: f32,
    /// Tolerance for CPU vs CUDA parity
    pub device_parity: f32,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            f32_abs: 1e-5,
            f32_rel: 1e-4,
            quant_l2_pct: 5.0,
            device_parity: 1e-3,
        }
    }
}

impl Tolerances {
    /// Strict tolerances for exact comparisons
    pub fn strict() -> Self {
        Self {
            f32_abs: 1e-6,
            f32_rel: 1e-5,
            quant_l2_pct: 1.0,
            device_parity: 1e-4,
        }
    }

    /// Relaxed tolerances for quantized models
    pub fn quantized() -> Self {
        Self {
            f32_abs: 1e-3,
            f32_rel: 1e-2,
            quant_l2_pct: 10.0,
            device_parity: 5e-2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_tiny() {
        let config = ModelConfig::tiny();
        assert_eq!(config.hidden_dim, 64);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.head_dim(), 16);
        assert_eq!(config.gqa_group_size(), 2);
        assert!(config.is_gqa());
        assert!(!config.is_mqa());
    }

    #[test]
    fn test_model_config_qwen() {
        let config = ModelConfig::qwen_1_5b();
        assert_eq!(config.hidden_dim, 1536);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.gqa_group_size(), 6);
        assert!(config.is_gqa());
    }

    #[test]
    fn test_model_config_dimensions() {
        let config = ModelConfig::small();
        assert_eq!(config.q_dim(), 256); // 8 heads * 32 head_dim
        assert_eq!(config.k_dim(), 64); // 2 kv_heads * 32 head_dim
        assert_eq!(config.v_dim(), 64);
    }

    #[test]
    fn test_quant_type_bits() {
        assert_eq!(QuantType::F32.bits_per_weight(), 32.0);
        assert_eq!(QuantType::Q4_K.bits_per_weight(), 4.5);
    }

    #[test]
    fn test_quant_type_format_support() {
        assert!(QuantType::F32.supported_by(ModelFormat::Safetensors));
        assert!(!QuantType::Q4_K.supported_by(ModelFormat::Safetensors));
        assert!(QuantType::Q4_K.supported_by(ModelFormat::GGUF));
        assert!(QuantType::Q4_K.supported_by(ModelFormat::APR));
    }

    #[test]
    fn test_model_test_case_creation() {
        let tc = ModelTestCase::new(
            "tiny CPU test",
            ModelConfig::tiny(),
            ModelFormat::GGUF,
            Device::Cpu,
        );
        assert_eq!(tc.desc, "tiny CPU test");
        assert!(!tc.is_conversion_test());
    }

    #[test]
    fn test_model_test_case_conversion() {
        let tc = ModelTestCase::conversion(
            "GGUF to APR",
            ModelConfig::tiny(),
            ModelFormat::GGUF,
            ModelFormat::APR,
            Device::Cpu,
        );
        assert!(tc.is_conversion_test());
        assert_eq!(tc.source_format, ModelFormat::GGUF);
        assert_eq!(tc.target_format, Some(ModelFormat::APR));
    }

    #[test]
    fn test_device_display() {
        assert_eq!(format!("{}", Device::Cpu), "CPU");
        assert_eq!(format!("{}", Device::Cuda(0)), "CUDA:0");
    }

    #[test]
    fn test_format_display() {
        assert_eq!(format!("{}", ModelFormat::GGUF), "GGUF");
        assert_eq!(format!("{}", ModelFormat::APR), "APR");
    }

    #[test]
    fn test_param_count() {
        let config = ModelConfig::tiny();
        let params = config.param_count();
        // Rough estimate: should be in reasonable range for tiny model
        assert!(params > 10_000, "params={}", params);
        assert!(params < 1_000_000, "params={}", params);
    }

    #[test]
    fn test_test_result_l2_norm() {
        let result = TestResult::success("test", vec![3.0, 4.0], 100);
        let norm = result.output_l2_norm().expect("norm");
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_device_is_cuda() {
        assert!(!Device::Cpu.is_cuda());
        assert!(Device::Cuda(0).is_cuda());
        assert_eq!(Device::Cpu.cuda_id(), None);
        assert_eq!(Device::Cuda(7).cuda_id(), Some(7));
    }

    #[test]
    fn test_model_config_tinyllama() {
        let config = ModelConfig::tinyllama();
        assert_eq!(config.hidden_dim, 2048);
        assert_eq!(config.num_layers, 22);
    }

    #[test]
    fn test_model_config_default() {
        let default_config = ModelConfig::default();
        let tiny_config = ModelConfig::tiny();
        assert_eq!(default_config.hidden_dim, tiny_config.hidden_dim);
    }

    #[test]
    fn test_quant_type_all_bits() {
        assert_eq!(QuantType::F16.bits_per_weight(), 16.0);
        assert_eq!(QuantType::BF16.bits_per_weight(), 16.0);
        assert_eq!(QuantType::Q8_0.bits_per_weight(), 8.5);
        assert_eq!(QuantType::Q4_0.bits_per_weight(), 4.5);
        assert_eq!(QuantType::Q5_K.bits_per_weight(), 5.5);
        assert_eq!(QuantType::Q6_K.bits_per_weight(), 6.5);
    }

    #[test]
    fn test_quant_type_unsupported() {
        assert!(!QuantType::Q8_0.supported_by(ModelFormat::PyTorch));
        assert!(!QuantType::Q8_0.supported_by(ModelFormat::Safetensors));
    }

    #[test]
    fn test_forward_input_seq_len() {
        let input = ForwardInput::new(vec![1, 2, 3]);
        assert_eq!(input.seq_len(), 3);
        assert_eq!(input.position, 0);

        let input_pos = ForwardInput::at_position(vec![4, 5], 10);
        assert_eq!(input_pos.seq_len(), 2);
        assert_eq!(input_pos.position, 10);
    }

    #[test]
    fn test_model_test_case_builder() {
        let tc = ModelTestCase::new("test", ModelConfig::tiny(), ModelFormat::APR, Device::Cpu)
            .with_quant(QuantType::Q4_K)
            .with_tokens(vec![1, 2])
            .with_expected_norm(10.0);

        assert_eq!(tc.constructor.quantization, Some(QuantType::Q4_K));
        assert_eq!(tc.forward.tokens, vec![1, 2]);
        assert_eq!(tc.expected_output_norm, Some(10.0));
    }

    #[test]
    fn test_test_result_failure() {
        let result = TestResult::failure("fail test", "error message", 50);
        assert!(!result.passed);
        assert_eq!(result.test_case, "fail test");
        assert_eq!(result.error.expect("error"), "error message");
        assert!(result.output.is_none());
    }

    #[test]
    fn test_tolerances() {
        let default = Tolerances::default();
        let strict = Tolerances::strict();
        let quantized = Tolerances::quantized();

        assert!(strict.f32_abs < default.f32_abs);
        assert!(quantized.f32_abs > default.f32_abs);
    }

    #[test]
    fn test_format_display_pytorch_safetensors() {
        assert_eq!(format!("{}", ModelFormat::PyTorch), "PyTorch");
        assert_eq!(format!("{}", ModelFormat::Safetensors), "Safetensors");
    }

    #[test]
    fn test_constructor_input_with_quant() {
        let config = ModelConfig::tiny();
        let ci = ConstructorInput::with_quant(config, QuantType::Q8_0, 123);
        assert_eq!(ci.quantization, Some(QuantType::Q8_0));
        assert_eq!(ci.weights_seed, 123);
    }
}
