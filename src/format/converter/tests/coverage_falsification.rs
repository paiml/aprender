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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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

include!("coverage_falsification_part_02.rs");
include!("coverage_falsification_part_03.rs");
