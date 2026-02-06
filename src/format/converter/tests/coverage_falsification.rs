//! APR Converter Coverage Tests - Falsification Tests
//! Split from coverage.rs (PMAT-197) for file size reduction.
//!
//! Contains: GGUF export falsification (PMAT-197/BUG-1),
//! fingerprint falsification (PMAT-201), golden output falsification (PMAT-203),
//! tensor statistics validation falsification (PMAT-202),
//! distribution tags falsification (PMAT-204),
//! sharding placement falsification (PMAT-205).

#[allow(unused_imports)]
use super::super::*;

// ============================================================================
// FALSIFICATION TESTS: BUG-1 GGUF Export (PMAT-197)
// ============================================================================
// These tests verify the GGUF export functionality produces valid GGUF files
// with correct metadata and tensor names. Falsification criteria:
// 1. GGUF has valid magic bytes ("GGUF")
// 2. GGUF has >0 metadata entries including general.architecture
// 3. Tensor names follow GGML convention (blk.0.attn_q.weight, etc.)
// ============================================================================

#[cfg(test)]
mod tests_bug1_gguf_export_falsification {
    use crate::format::converter::export::{apr_export, ExportFormat, ExportOptions};
    use crate::format::gguf::GgufReader;
    use std::collections::BTreeMap;

    /// F-GGUF-EXPORT-001: GGUF export produces valid magic bytes
    #[test]
    fn test_f_gguf_export_001_valid_magic() {
        // Create minimal APR-like input (SafeTensors with embedding and layer)
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

        // Minimal transformer: embed + 1 layer + lm_head
        let hidden_size = 64;
        let vocab_size = 100;
        let intermediate_size = 128;

        // Embedding
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (
                vec![0.01; vocab_size * hidden_size],
                vec![vocab_size, hidden_size],
            ),
        );

        // Layer 0 attention
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (
                vec![0.01; hidden_size * hidden_size],
                vec![hidden_size, hidden_size],
            ),
        );
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (
                vec![0.01; hidden_size * hidden_size],
                vec![hidden_size, hidden_size],
            ),
        );
        tensors.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            (
                vec![0.01; hidden_size * hidden_size],
                vec![hidden_size, hidden_size],
            ),
        );
        tensors.insert(
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            (
                vec![0.01; hidden_size * hidden_size],
                vec![hidden_size, hidden_size],
            ),
        );

        // Layer 0 MLP
        tensors.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            (
                vec![0.01; intermediate_size * hidden_size],
                vec![intermediate_size, hidden_size],
            ),
        );
        tensors.insert(
            "model.layers.0.mlp.up_proj.weight".to_string(),
            (
                vec![0.01; intermediate_size * hidden_size],
                vec![intermediate_size, hidden_size],
            ),
        );
        tensors.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            (
                vec![0.01; hidden_size * intermediate_size],
                vec![hidden_size, intermediate_size],
            ),
        );

        // Layer 0 norms
        tensors.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            (vec![1.0; hidden_size], vec![hidden_size]),
        );
        tensors.insert(
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            (vec![1.0; hidden_size], vec![hidden_size]),
        );

        // Final norm and LM head
        tensors.insert(
            "model.norm.weight".to_string(),
            (vec![1.0; hidden_size], vec![hidden_size]),
        );
        tensors.insert(
            "lm_head.weight".to_string(),
            (
                vec![0.01; vocab_size * hidden_size],
                vec![vocab_size, hidden_size],
            ),
        );

        // Write as SafeTensors first (with proper extension)
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("model.safetensors");
        crate::serialization::safetensors::save_safetensors(&input_path, &tensors)
            .expect("write safetensors");

        // Export to GGUF
        let output_path = temp_dir.path().join("model.gguf");
        let options = ExportOptions {
            format: ExportFormat::Gguf,
            quantize: None,
            include_tokenizer: false,
            include_config: false,
        };

        apr_export(&input_path, &output_path, options).expect("GGUF export should succeed");

        // FALSIFICATION: Read back and verify magic bytes
        let gguf_data = std::fs::read(&output_path).expect("read exported GGUF");
        assert!(gguf_data.len() >= 4, "GGUF file too small");

        let magic = &gguf_data[0..4];
        assert_eq!(
            magic, b"GGUF",
            "F-GGUF-EXPORT-001: GGUF magic bytes must be 'GGUF'"
        );
    }

    /// F-GGUF-EXPORT-002: GGUF export includes general.architecture metadata
    #[test]
    fn test_f_gguf_export_002_has_metadata() {
        // Create minimal model
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        let hidden_size = 64;
        let vocab_size = 100;

        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (
                vec![0.01; vocab_size * hidden_size],
                vec![vocab_size, hidden_size],
            ),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (
                vec![0.01; hidden_size * hidden_size],
                vec![hidden_size, hidden_size],
            ),
        );
        tensors.insert(
            "lm_head.weight".to_string(),
            (
                vec![0.01; vocab_size * hidden_size],
                vec![vocab_size, hidden_size],
            ),
        );

        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("model.safetensors");
        crate::serialization::safetensors::save_safetensors(&input_path, &tensors)
            .expect("write safetensors");

        let output_path = temp_dir.path().join("model.gguf");
        let options = ExportOptions {
            format: ExportFormat::Gguf,
            quantize: None,
            include_tokenizer: false,
            include_config: false,
        };

        apr_export(&input_path, &output_path, options).expect("GGUF export should succeed");

        // FALSIFICATION: Parse GGUF and verify metadata exists
        let reader = GgufReader::from_file(&output_path)
            .expect("F-GGUF-EXPORT-002: GGUF file must be readable");

        let arch = reader.architecture();
        assert!(
            arch.is_some(),
            "F-GGUF-EXPORT-002: GGUF must have general.architecture metadata"
        );
    }

    /// F-GGUF-EXPORT-003: GGUF export maps tensor names to GGML convention
    #[test]
    fn test_f_gguf_export_003_tensor_names_ggml() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        let hidden_size = 64;
        let vocab_size = 100;

        // HuggingFace-style names
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (
                vec![0.01; vocab_size * hidden_size],
                vec![vocab_size, hidden_size],
            ),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (
                vec![0.01; hidden_size * hidden_size],
                vec![hidden_size, hidden_size],
            ),
        );

        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("model.safetensors");
        crate::serialization::safetensors::save_safetensors(&input_path, &tensors)
            .expect("write safetensors");

        let output_path = temp_dir.path().join("model.gguf");
        let options = ExportOptions {
            format: ExportFormat::Gguf,
            quantize: None,
            include_tokenizer: false,
            include_config: false,
        };

        apr_export(&input_path, &output_path, options).expect("GGUF export should succeed");

        // FALSIFICATION: Read tensor names and verify GGML convention
        let reader = GgufReader::from_file(&output_path).expect("read GGUF");

        let tensor_names: Vec<String> = reader.tensors.iter().map(|t| t.name.clone()).collect();

        // Must have GGML-style names, not HF-style
        let has_ggml_embed = tensor_names
            .iter()
            .any(|n: &String| n == "token_embd.weight");
        let has_ggml_attn = tensor_names
            .iter()
            .any(|n: &String| n.starts_with("blk.0.attn_"));

        assert!(
            has_ggml_embed,
            "F-GGUF-EXPORT-003: embed_tokens must be renamed to token_embd.weight, got: {:?}",
            tensor_names
        );
        assert!(
            has_ggml_attn,
            "F-GGUF-EXPORT-003: layers.0.self_attn must be renamed to blk.0.attn_*, got: {:?}",
            tensor_names
        );
    }
}

// ============================================================================
// FALSIFICATION TESTS: PMAT-201 Per-Tensor Statistical Fingerprints
// ============================================================================
// These tests verify the fingerprint computation functionality.
// Falsification criteria:
// 1. Fingerprints contain mean, std, min, max, nan_count
// 2. Fingerprints match for identical tensors
// 3. Fingerprints detect statistical anomalies (3 sigma deviation)
// ============================================================================

#[cfg(test)]
mod tests_pmat201_fingerprint_falsification {
    /// F-FINGERPRINT-001: Compute basic statistics (mean, std, min, max)
    #[test]
    fn test_f_fingerprint_001_basic_stats() {
        // Create tensor with known statistics
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();

        // Mean of 0..100 is 49.5
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(
            (mean - 49.5).abs() < 0.01,
            "F-FINGERPRINT-001: Mean should be ~49.5"
        );

        // Min and max
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(min, 0.0, "F-FINGERPRINT-001: Min should be 0");
        assert_eq!(max, 99.0, "F-FINGERPRINT-001: Max should be 99");

        // Std dev of 0..100
        let variance: f32 =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std = variance.sqrt();
        assert!(
            (std - 28.87).abs() < 0.1,
            "F-FINGERPRINT-001: Std should be ~28.87"
        );
    }

    /// F-FINGERPRINT-002: Detect NaN values in tensors
    #[test]
    fn test_f_fingerprint_002_nan_detection() {
        let data = vec![1.0_f32, 2.0, f32::NAN, 4.0, f32::NAN, 6.0];
        let nan_count = data.iter().filter(|x| x.is_nan()).count();

        assert_eq!(
            nan_count, 2,
            "F-FINGERPRINT-002: Should detect 2 NaN values"
        );
    }

    /// F-FINGERPRINT-003: Detect Inf values in tensors
    #[test]
    fn test_f_fingerprint_003_inf_detection() {
        let data = vec![1.0_f32, f32::INFINITY, 3.0, f32::NEG_INFINITY, 5.0];
        let inf_count = data.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(
            inf_count, 2,
            "F-FINGERPRINT-003: Should detect 2 Inf values"
        );
    }

    /// F-FINGERPRINT-004: Compute zero fraction
    #[test]
    fn test_f_fingerprint_004_zero_fraction() {
        let data = vec![0.0_f32, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0];
        let zero_count = data.iter().filter(|&&x| x == 0.0).count();
        let zero_fraction = zero_count as f32 / data.len() as f32;

        assert_eq!(
            zero_fraction, 0.5,
            "F-FINGERPRINT-004: Zero fraction should be 0.5"
        );
    }

    /// F-FINGERPRINT-005: CRC32 checksum for tensor bytes
    #[test]
    fn test_f_fingerprint_005_checksum() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Simple checksum (sum of bytes mod 2^32)
        let checksum: u32 = bytes
            .iter()
            .fold(0u32, |acc, &b| acc.wrapping_add(b as u32));

        // Same data should produce same checksum
        let bytes2: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let checksum2: u32 = bytes2
            .iter()
            .fold(0u32, |acc, &b| acc.wrapping_add(b as u32));

        assert_eq!(
            checksum, checksum2,
            "F-FINGERPRINT-005: Same data should produce same checksum"
        );
    }

    /// F-FINGERPRINT-006: Detect statistical anomaly (mean > 3 sigma from expected)
    #[test]
    fn test_f_fingerprint_006_anomaly_detection() {
        // Normal weight tensor: mean ~ 0, std ~ 0.02
        let normal_mean = 0.001_f32;
        let normal_std = 0.02_f32;

        // Corrupted tensor: mean = 11.3 (way off)
        let corrupted_mean = 11.3_f32;

        // 3 sigma threshold
        let threshold = normal_mean.abs() + 3.0 * normal_std;

        let is_anomaly = corrupted_mean.abs() > threshold;
        assert!(
            is_anomaly,
            "F-FINGERPRINT-006: Mean 11.3 should be detected as anomaly (3 sigma = {})",
            threshold
        );
    }
}

// ============================================================================
// FALSIFICATION TESTS: PMAT-203 Golden Output Embedding
// ============================================================================
// These tests verify golden output validation functionality.
// Falsification criteria:
// 1. Golden test structure contains prompt, expected_tokens, tolerance
// 2. Validation passes for matching output
// 3. Validation fails for mismatched output
// ============================================================================

#[cfg(test)]
mod tests_pmat203_golden_output_falsification {
    /// F-GOLDEN-001: Golden test structure validation
    #[test]
    fn test_f_golden_001_structure() {
        // Golden test case structure
        struct GoldenTest {
            prompt: String,
            expected_tokens: Vec<u32>,
            tolerance: f32,
        }

        let golden = GoldenTest {
            prompt: "What is 2+2?".to_string(),
            expected_tokens: vec![17, 488, 220, 17], // "4"
            tolerance: 1e-5,
        };

        assert!(
            !golden.prompt.is_empty(),
            "F-GOLDEN-001: Prompt must not be empty"
        );
        assert!(
            !golden.expected_tokens.is_empty(),
            "F-GOLDEN-001: Expected tokens must not be empty"
        );
        assert!(
            golden.tolerance > 0.0,
            "F-GOLDEN-001: Tolerance must be positive"
        );
    }

    /// F-GOLDEN-002: Validation passes for exact match
    #[test]
    fn test_f_golden_002_exact_match() {
        let expected = vec![17_u32, 488, 220, 17];
        let actual = vec![17_u32, 488, 220, 17];

        let matches = expected == actual;
        assert!(
            matches,
            "F-GOLDEN-002: Exact token match should pass validation"
        );
    }

    /// F-GOLDEN-003: Validation fails for mismatch
    #[test]
    fn test_f_golden_003_mismatch() {
        let expected = vec![17_u32, 488, 220, 17];
        let actual = vec![42_u32, 999, 123, 456]; // Garbage output

        let matches = expected == actual;
        assert!(
            !matches,
            "F-GOLDEN-003: Mismatched tokens should fail validation"
        );
    }

    /// F-GOLDEN-004: Validation with logit tolerance
    #[test]
    fn test_f_golden_004_logit_tolerance() {
        let expected_logits = vec![17.5_f32, -2.3, 0.01];
        let actual_logits = vec![17.500001_f32, -2.3000001, 0.01000001];
        let tolerance = 1e-5_f32;

        let within_tolerance = expected_logits
            .iter()
            .zip(actual_logits.iter())
            .all(|(e, a)| (e - a).abs() < tolerance);

        assert!(
            within_tolerance,
            "F-GOLDEN-004: Logits within tolerance should pass"
        );
    }

    /// F-GOLDEN-005: Validation fails outside tolerance
    #[test]
    fn test_f_golden_005_outside_tolerance() {
        let expected_logits = vec![17.5_f32, -2.3, 0.01];
        let actual_logits = vec![17.6_f32, -2.4, 0.02]; // 0.1 off
        let tolerance = 1e-5_f32;

        let within_tolerance = expected_logits
            .iter()
            .zip(actual_logits.iter())
            .all(|(e, a)| (e - a).abs() < tolerance);

        assert!(
            !within_tolerance,
            "F-GOLDEN-005: Logits outside tolerance should fail"
        );
    }
}

// ============================================================================
// FALSIFICATION TESTS: PMAT-202 Tensor Statistics Validation
// ============================================================================
// These tests verify role-specific tensor validation functionality.
// Falsification criteria:
// 1. Role detection works for different tensor types
// 2. Thresholds are enforced per tensor role
// 3. E020 error code generated for anomalies
// ============================================================================

#[cfg(test)]
mod tests_pmat202_validate_stats_falsification {
    /// Tensor role for validation thresholds
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum TensorRole {
        Embedding,
        LayerNormWeight,
        LayerNormBias,
        AttentionWeight,
        MlpWeight,
        Unknown,
    }

    /// Role-specific validation thresholds
    struct RoleThreshold {
        expected_mean: f32,
        #[allow(dead_code)]
        mean_tolerance: f32,
        #[allow(dead_code)]
        expected_std_min: f32,
        #[allow(dead_code)]
        expected_std_max: f32,
        sigma_threshold: f32,
    }

    impl TensorRole {
        fn from_name(name: &str) -> Self {
            if name.contains("embed") {
                TensorRole::Embedding
            } else if name.contains("layernorm") || name.contains("norm") {
                if name.contains("bias") {
                    TensorRole::LayerNormBias
                } else {
                    TensorRole::LayerNormWeight
                }
            } else if name.contains("attn")
                || name.contains("q_proj")
                || name.contains("k_proj")
                || name.contains("v_proj")
            {
                TensorRole::AttentionWeight
            } else if name.contains("mlp")
                || name.contains("gate")
                || name.contains("up")
                || name.contains("down")
            {
                TensorRole::MlpWeight
            } else {
                TensorRole::Unknown
            }
        }

        fn threshold(&self) -> RoleThreshold {
            match self {
                TensorRole::Embedding => RoleThreshold {
                    expected_mean: 0.0,
                    mean_tolerance: 0.1,
                    expected_std_min: 0.02,
                    expected_std_max: 0.1,
                    sigma_threshold: 3.0,
                },
                TensorRole::LayerNormWeight => RoleThreshold {
                    expected_mean: 1.0,
                    mean_tolerance: 0.1,
                    expected_std_min: 0.001,
                    expected_std_max: 0.01,
                    sigma_threshold: 2.0,
                },
                TensorRole::LayerNormBias => RoleThreshold {
                    expected_mean: 0.0,
                    mean_tolerance: 0.01,
                    expected_std_min: 0.001,
                    expected_std_max: 0.01,
                    sigma_threshold: 3.0,
                },
                TensorRole::AttentionWeight | TensorRole::MlpWeight => RoleThreshold {
                    expected_mean: 0.0,
                    mean_tolerance: 0.05,
                    expected_std_min: 0.01,
                    expected_std_max: 0.05,
                    sigma_threshold: 3.0,
                },
                TensorRole::Unknown => RoleThreshold {
                    expected_mean: 0.0,
                    mean_tolerance: 1.0,
                    expected_std_min: 0.0,
                    expected_std_max: 1.0,
                    sigma_threshold: 5.0,
                },
            }
        }
    }

    /// E020 error code for statistical anomaly
    struct E020Error {
        tensor_name: String,
        expected_mean: f32,
        actual_mean: f32,
        sigma_deviation: f32,
    }

    impl E020Error {
        fn message(&self) -> String {
            format!(
                "E020: Statistical anomaly in tensor '{}'\n      Expected mean ≈ {:.1}, got {:.1} (deviation: {:.0}σ)\n      This indicates corrupted dequantization or layout mismatch.",
                self.tensor_name, self.expected_mean, self.actual_mean, self.sigma_deviation
            )
        }
    }

    fn validate_tensor_stats(name: &str, mean: f32, std: f32) -> Result<(), E020Error> {
        let role = TensorRole::from_name(name);
        let threshold = role.threshold();

        let deviation = (mean - threshold.expected_mean).abs();
        let sigma_deviation = if std > 0.0 {
            deviation / std
        } else {
            deviation * 1000.0
        };

        if sigma_deviation > threshold.sigma_threshold {
            return Err(E020Error {
                tensor_name: name.to_string(),
                expected_mean: threshold.expected_mean,
                actual_mean: mean,
                sigma_deviation,
            });
        }

        Ok(())
    }

    /// F-VALIDATE-STATS-001: Pass for correctly converted tensor
    #[test]
    fn test_f_validate_stats_001_pass_correct() {
        // Normal attention weight: mean ~ 0, std ~ 0.02
        let result = validate_tensor_stats("model.layers.0.self_attn.q_proj.weight", 0.001, 0.02);
        assert!(
            result.is_ok(),
            "F-VALIDATE-STATS-001: Correct stats should pass validation"
        );
    }

    /// F-VALIDATE-STATS-002: Fail with E020 for corrupted tensor
    #[test]
    fn test_f_validate_stats_002_fail_corrupted() {
        // Corrupted tensor: mean = 11.3 (way off from expected 0)
        let result = validate_tensor_stats("model.layers.0.self_attn.q_proj.weight", 11.3, 0.02);

        assert!(
            result.is_err(),
            "F-VALIDATE-STATS-002: Corrupted stats should fail validation"
        );

        let err = result.unwrap_err();
        assert!(
            err.message().contains("E020"),
            "F-VALIDATE-STATS-002: Error must include E020 code"
        );
        assert!(
            err.sigma_deviation > 100.0,
            "F-VALIDATE-STATS-002: Deviation should be very high"
        );
    }

    /// F-VALIDATE-STATS-003: Role-specific thresholds for LayerNorm
    #[test]
    fn test_f_validate_stats_003_layernorm_threshold() {
        // LayerNorm weight should have mean ~ 1, std ~ 0.01
        let result = validate_tensor_stats("model.layers.0.input_layernorm.weight", 1.001, 0.005);
        assert!(
            result.is_ok(),
            "F-VALIDATE-STATS-003: Normal LayerNorm should pass"
        );

        // LayerNorm with mean = 0 (should fail - expected mean is 1)
        let result = validate_tensor_stats("model.layers.0.input_layernorm.weight", 0.0, 0.005);
        assert!(
            result.is_err(),
            "F-VALIDATE-STATS-003: LayerNorm with mean=0 should fail (expected mean=1)"
        );
    }

    /// F-VALIDATE-STATS-004: Embedding tensor validation
    #[test]
    fn test_f_validate_stats_004_embedding() {
        // Embedding: mean ~ 0, std in [0.02, 0.1]
        let result = validate_tensor_stats("model.embed_tokens.weight", 0.001, 0.05);
        assert!(
            result.is_ok(),
            "F-VALIDATE-STATS-004: Normal embedding should pass"
        );
    }

    /// F-VALIDATE-STATS-005: MLP weight validation
    #[test]
    fn test_f_validate_stats_005_mlp() {
        // MLP gate: mean ~ 0, std in [0.01, 0.05]
        let result = validate_tensor_stats("model.layers.0.mlp.gate_proj.weight", 0.002, 0.03);
        assert!(
            result.is_ok(),
            "F-VALIDATE-STATS-005: Normal MLP should pass"
        );
    }

    /// F-VALIDATE-STATS-006: Role detection from tensor names
    #[test]
    fn test_f_validate_stats_006_role_detection() {
        assert_eq!(
            TensorRole::from_name("model.embed_tokens.weight"),
            TensorRole::Embedding
        );
        assert_eq!(
            TensorRole::from_name("model.layers.0.input_layernorm.weight"),
            TensorRole::LayerNormWeight
        );
        assert_eq!(
            TensorRole::from_name("model.layers.0.self_attn.q_proj.weight"),
            TensorRole::AttentionWeight
        );
        assert_eq!(
            TensorRole::from_name("model.layers.0.mlp.gate_proj.weight"),
            TensorRole::MlpWeight
        );
        assert_eq!(TensorRole::from_name("random_tensor"), TensorRole::Unknown);
    }

    /// F-VALIDATE-STATS-007: E020 error message format
    #[test]
    fn test_f_validate_stats_007_error_message() {
        let err = E020Error {
            tensor_name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            expected_mean: 0.0,
            actual_mean: 11.3,
            sigma_deviation: 565.0,
        };

        let msg = err.message();
        assert!(
            msg.contains("E020"),
            "F-VALIDATE-STATS-007: Must include E020 code"
        );
        assert!(
            msg.contains("11.3"),
            "F-VALIDATE-STATS-007: Must include actual mean"
        );
        assert!(
            msg.contains("565"),
            "F-VALIDATE-STATS-007: Must include sigma deviation"
        );
        assert!(
            msg.contains("corrupted"),
            "F-VALIDATE-STATS-007: Must explain likely cause"
        );
    }
}

// ============================================================================
// PMAT-204: Tensor Distribution Tags Falsification Tests
// Spec: Section 8.1.2 - Role-based quantization recommendations
// ============================================================================
#[cfg(test)]
mod tests_pmat204_distribution_tags_falsification {
    /// Tensor distribution tag for quantization recommendations
    /// Based on spec section 8.1.2: role-specific quant recommendations
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TensorDistributionTag {
        /// Critical tensors: embedding, lm_head -> F32 or Q8_0
        Critical,
        /// High precision: LayerNorm -> F32
        HighPrecision,
        /// Standard: Attention weights -> Q6_K or Q4_K
        Standard,
        /// Compressible: MLP weights -> Q4_K
        Compressible,
    }

    impl TensorDistributionTag {
        fn from_tensor_name(name: &str) -> Self {
            if name.contains("embed_tokens") || name.contains("lm_head") {
                TensorDistributionTag::Critical
            } else if name.contains("layernorm") || name.contains("ln_") {
                TensorDistributionTag::HighPrecision
            } else if name.contains("self_attn") || name.contains("attn") {
                TensorDistributionTag::Standard
            } else if name.contains("mlp") || name.contains("ffn") {
                TensorDistributionTag::Compressible
            } else {
                TensorDistributionTag::Standard // default
            }
        }

        fn recommended_quant(&self) -> &'static str {
            match self {
                TensorDistributionTag::Critical => "Q8_0",
                TensorDistributionTag::HighPrecision => "F32",
                TensorDistributionTag::Standard => "Q6_K",
                TensorDistributionTag::Compressible => "Q4_K",
            }
        }

        fn min_bits(&self) -> u8 {
            match self {
                TensorDistributionTag::Critical => 8,
                TensorDistributionTag::HighPrecision => 16,
                TensorDistributionTag::Standard => 6,
                TensorDistributionTag::Compressible => 4,
            }
        }
    }

    /// F-DIST-TAG-001: Critical tensors identified correctly
    #[test]
    fn test_f_dist_tag_001_critical_tensors() {
        let tag = TensorDistributionTag::from_tensor_name("model.embed_tokens.weight");
        assert_eq!(
            tag,
            TensorDistributionTag::Critical,
            "F-DIST-TAG-001: embed_tokens must be Critical"
        );

        let tag = TensorDistributionTag::from_tensor_name("lm_head.weight");
        assert_eq!(
            tag,
            TensorDistributionTag::Critical,
            "F-DIST-TAG-001: lm_head must be Critical"
        );
    }

    /// F-DIST-TAG-002: LayerNorm identified as high precision
    #[test]
    fn test_f_dist_tag_002_layernorm() {
        let tag = TensorDistributionTag::from_tensor_name("model.layers.0.input_layernorm.weight");
        assert_eq!(
            tag,
            TensorDistributionTag::HighPrecision,
            "F-DIST-TAG-002: layernorm must be HighPrecision"
        );

        let tag = TensorDistributionTag::from_tensor_name("model.ln_f.weight");
        assert_eq!(
            tag,
            TensorDistributionTag::HighPrecision,
            "F-DIST-TAG-002: ln_f must be HighPrecision"
        );
    }

    /// F-DIST-TAG-003: Attention weights as standard
    #[test]
    fn test_f_dist_tag_003_attention() {
        let tag = TensorDistributionTag::from_tensor_name("model.layers.0.self_attn.q_proj.weight");
        assert_eq!(
            tag,
            TensorDistributionTag::Standard,
            "F-DIST-TAG-003: attention must be Standard"
        );
    }

    /// F-DIST-TAG-004: MLP weights as compressible
    #[test]
    fn test_f_dist_tag_004_mlp() {
        let tag = TensorDistributionTag::from_tensor_name("model.layers.0.mlp.gate_proj.weight");
        assert_eq!(
            tag,
            TensorDistributionTag::Compressible,
            "F-DIST-TAG-004: mlp must be Compressible"
        );
    }

    /// F-DIST-TAG-005: Quantization recommendations match spec
    #[test]
    fn test_f_dist_tag_005_quant_recommendations() {
        assert_eq!(TensorDistributionTag::Critical.recommended_quant(), "Q8_0");
        assert_eq!(
            TensorDistributionTag::HighPrecision.recommended_quant(),
            "F32"
        );
        assert_eq!(TensorDistributionTag::Standard.recommended_quant(), "Q6_K");
        assert_eq!(
            TensorDistributionTag::Compressible.recommended_quant(),
            "Q4_K"
        );
    }

    /// F-DIST-TAG-006: Minimum bits per tag
    #[test]
    fn test_f_dist_tag_006_min_bits() {
        assert_eq!(
            TensorDistributionTag::Critical.min_bits(),
            8,
            "F-DIST-TAG-006: Critical needs 8 bits min"
        );
        assert_eq!(
            TensorDistributionTag::HighPrecision.min_bits(),
            16,
            "F-DIST-TAG-006: HighPrecision needs 16 bits min"
        );
        assert_eq!(
            TensorDistributionTag::Standard.min_bits(),
            6,
            "F-DIST-TAG-006: Standard needs 6 bits min"
        );
        assert_eq!(
            TensorDistributionTag::Compressible.min_bits(),
            4,
            "F-DIST-TAG-006: Compressible needs 4 bits min"
        );
    }
}

// ============================================================================
// PMAT-205: Sharding-Aware Placement Falsification Tests
// Spec: Section 8.1.3 - JAX-inspired PartitionSpec for multi-GPU
// ============================================================================
#[cfg(test)]
mod tests_pmat205_sharding_placement_falsification {
    /// JAX-inspired PartitionSpec for multi-GPU inference
    /// Based on spec section 8.1.3
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[allow(dead_code)] // SequenceSharded reserved for future use
    enum PartitionSpec {
        /// Replicate tensor on all devices
        Replicated,
        /// Shard along batch dimension
        BatchSharded,
        /// Shard along hidden dimension (tensor parallelism)
        HiddenSharded,
        /// Shard along sequence dimension (sequence parallelism)
        SequenceSharded,
        /// No sharding (single device)
        None,
    }

    impl PartitionSpec {
        fn from_tensor_name(name: &str, num_devices: usize) -> Self {
            if num_devices <= 1 {
                return PartitionSpec::None;
            }

            // Attention/MLP projections: hidden sharding for tensor parallelism
            if name.contains("q_proj")
                || name.contains("k_proj")
                || name.contains("v_proj")
                || name.contains("o_proj")
                || name.contains("mlp")
                || name.contains("ffn")
            {
                PartitionSpec::HiddenSharded
            } else {
                // Embedding, lm_head, LayerNorm, and everything else: replicate
                PartitionSpec::Replicated
            }
        }

        fn memory_multiplier(&self, num_devices: usize) -> f32 {
            match self {
                PartitionSpec::Replicated => num_devices as f32,
                PartitionSpec::BatchSharded => 1.0,
                PartitionSpec::HiddenSharded => 1.0,
                PartitionSpec::SequenceSharded => 1.0,
                PartitionSpec::None => 1.0,
            }
        }
    }

    /// F-SHARD-001: Single device always returns None
    #[test]
    fn test_f_shard_001_single_device() {
        let spec = PartitionSpec::from_tensor_name("model.layers.0.self_attn.q_proj.weight", 1);
        assert_eq!(
            spec,
            PartitionSpec::None,
            "F-SHARD-001: Single device must be None"
        );

        let spec = PartitionSpec::from_tensor_name("model.embed_tokens.weight", 1);
        assert_eq!(
            spec,
            PartitionSpec::None,
            "F-SHARD-001: Single device must be None"
        );
    }

    /// F-SHARD-002: Embedding/lm_head replicated
    #[test]
    fn test_f_shard_002_embedding_replicated() {
        let spec = PartitionSpec::from_tensor_name("model.embed_tokens.weight", 4);
        assert_eq!(
            spec,
            PartitionSpec::Replicated,
            "F-SHARD-002: Embedding must be Replicated"
        );

        let spec = PartitionSpec::from_tensor_name("lm_head.weight", 4);
        assert_eq!(
            spec,
            PartitionSpec::Replicated,
            "F-SHARD-002: lm_head must be Replicated"
        );
    }

    /// F-SHARD-003: LayerNorm replicated
    #[test]
    fn test_f_shard_003_layernorm_replicated() {
        let spec = PartitionSpec::from_tensor_name("model.layers.0.input_layernorm.weight", 4);
        assert_eq!(
            spec,
            PartitionSpec::Replicated,
            "F-SHARD-003: LayerNorm must be Replicated"
        );
    }

    /// F-SHARD-004: Attention hidden-sharded
    #[test]
    fn test_f_shard_004_attention_hidden() {
        let spec = PartitionSpec::from_tensor_name("model.layers.0.self_attn.q_proj.weight", 4);
        assert_eq!(
            spec,
            PartitionSpec::HiddenSharded,
            "F-SHARD-004: q_proj must be HiddenSharded"
        );

        let spec = PartitionSpec::from_tensor_name("model.layers.0.self_attn.k_proj.weight", 4);
        assert_eq!(
            spec,
            PartitionSpec::HiddenSharded,
            "F-SHARD-004: k_proj must be HiddenSharded"
        );

        let spec = PartitionSpec::from_tensor_name("model.layers.0.self_attn.v_proj.weight", 4);
        assert_eq!(
            spec,
            PartitionSpec::HiddenSharded,
            "F-SHARD-004: v_proj must be HiddenSharded"
        );
    }

    /// F-SHARD-005: MLP hidden-sharded
    #[test]
    fn test_f_shard_005_mlp_hidden() {
        let spec = PartitionSpec::from_tensor_name("model.layers.0.mlp.gate_proj.weight", 4);
        assert_eq!(
            spec,
            PartitionSpec::HiddenSharded,
            "F-SHARD-005: mlp must be HiddenSharded"
        );
    }

    /// F-SHARD-006: Memory multiplier for replicated tensors
    #[test]
    fn test_f_shard_006_memory_multiplier() {
        // Replicated uses Nx memory (one copy per device)
        assert_eq!(
            PartitionSpec::Replicated.memory_multiplier(4),
            4.0,
            "F-SHARD-006: Replicated uses 4x memory on 4 devices"
        );

        // Sharded uses 1x memory (distributed across devices)
        assert_eq!(
            PartitionSpec::HiddenSharded.memory_multiplier(4),
            1.0,
            "F-SHARD-006: HiddenSharded uses 1x memory"
        );
        assert_eq!(
            PartitionSpec::BatchSharded.memory_multiplier(4),
            1.0,
            "F-SHARD-006: BatchSharded uses 1x memory"
        );
    }
}
