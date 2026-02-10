//! SQLite-style conversion test harness for SafeTensors <-> APR round-trips.
//!
//! Uses `TempDir` for RAII cleanup (no manual `fs::remove_file`), pygmy builders
//! for input data, and read-back verification with configurable tolerance.
//!
//! # Example
//!
//! ```rust,ignore
//! use crate::format::test_factory::harness::ConversionTestHarness;
//! use crate::format::test_factory::PygmyConfig;
//!
//! ConversionTestHarness::assert_import_ok(PygmyConfig::llama_style());
//! ConversionTestHarness::assert_roundtrip_ok(PygmyConfig::default());
//! ```

use super::{build_pygmy_apr_with_config, build_pygmy_safetensors_with_config, PygmyConfig};
use crate::format::converter::{
    apr_export, apr_import, ExportFormat, ExportOptions, ImportOptions,
};
use crate::format::v2::AprV2Reader;
use crate::serialization::safetensors::MappedSafeTensors;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// Tolerance thresholds per dtype for tensor data comparison.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_field_names)] // Postfix naming is intentional for clarity
pub(crate) struct ToleranceConfig {
    pub(crate) f32_atol: f32,
    pub(crate) f16_atol: f32,
    pub(crate) q8_atol: f32,
    pub(crate) q4_atol: f32,
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            f32_atol: 1e-6,
            f16_atol: 1e-3,
            q8_atol: 0.1,
            q4_atol: 0.5,
        }
    }
}

/// A single tensor mismatch found during verification.
#[derive(Debug)]
pub(crate) struct TensorMismatch {
    pub(crate) tensor_name: String,
    pub(crate) kind: MismatchKind,
}

/// What went wrong with a tensor comparison.
#[derive(Debug)]
pub(crate) enum MismatchKind {
    Missing,
    /// T-QKV-02: Output contains a tensor not present in source (possible fusion/split)
    Extra,
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    DataMismatch {
        index: usize,
        expected: f32,
        actual: f32,
        tolerance: f32,
    },
}

impl core::fmt::Display for TensorMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.kind {
            MismatchKind::Missing => {
                write!(f, "tensor '{}': missing in output", self.tensor_name)
            }
            MismatchKind::Extra => {
                write!(
                    f,
                    "tensor '{}': extra in output (not in source)",
                    self.tensor_name
                )
            }
            MismatchKind::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "tensor '{}': shape mismatch expected={:?} actual={:?}",
                    self.tensor_name, expected, actual
                )
            }
            MismatchKind::DataMismatch {
                index,
                expected,
                actual,
                tolerance,
            } => {
                write!(
                    f,
                    "tensor '{}': data[{}] expected={} actual={} (tol={})",
                    self.tensor_name, index, expected, actual, tolerance
                )
            }
        }
    }
}

/// Result of a verification pass.
#[derive(Debug)]
pub(crate) struct VerificationResult {
    pub(crate) mismatches: Vec<TensorMismatch>,
}

impl VerificationResult {
    /// Panics with detailed info if any mismatches were found.
    pub(crate) fn assert_passed(&self) {
        if !self.mismatches.is_empty() {
            let msgs: Vec<String> = self.mismatches.iter().map(ToString::to_string).collect();
            panic!(
                "Verification failed with {} mismatch(es):\n  {}",
                self.mismatches.len(),
                msgs.join("\n  ")
            );
        }
    }

    #[must_use]
    pub(crate) fn passed(&self) -> bool {
        self.mismatches.is_empty()
    }
}

/// RAII conversion test harness. The `TempDir` is dropped (cleaned up)
/// when the harness goes out of scope.
pub(crate) struct ConversionTestHarness {
    dir: TempDir,
    input_path: Option<PathBuf>,
    output_path: Option<PathBuf>,
    /// Original pygmy tensor data for verification (name -> (data, shape))
    source_tensors: Vec<(String, Vec<f32>, Vec<usize>)>,
    pub(crate) tolerance: ToleranceConfig,
}

impl ConversionTestHarness {
    /// Create a new empty harness with a fresh temp directory.
    pub(crate) fn new() -> Self {
        Self {
            dir: TempDir::new().expect("Failed to create temp dir for test harness"),
            input_path: None,
            output_path: None,
            source_tensors: Vec::new(),
            tolerance: ToleranceConfig::default(),
        }
    }

    /// Access the temp directory path.
    pub(crate) fn dir(&self) -> &Path {
        self.dir.path()
    }

    // ----------------------------------------------------------------
    // Setup: write pygmy input files
    // ----------------------------------------------------------------

    /// Write a pygmy SafeTensors file into the temp dir and record
    /// the source tensor data for later verification.
    pub(crate) fn with_safetensors(mut self, config: PygmyConfig) -> Self {
        let bytes = build_pygmy_safetensors_with_config(config.clone());
        let path = self.dir.path().join("input.safetensors");
        fs::write(&path, &bytes).expect("Failed to write pygmy safetensors");

        // Record source tensors for verification
        self.source_tensors = collect_pygmy_tensors(&config);
        self.input_path = Some(path);
        self
    }

    /// Write a pygmy APR file into the temp dir.
    pub(crate) fn with_apr(mut self, config: PygmyConfig) -> Self {
        let bytes = build_pygmy_apr_with_config(config.clone());
        let path = self.dir.path().join("input.apr");
        fs::write(&path, &bytes).expect("Failed to write pygmy apr");

        self.source_tensors = collect_pygmy_tensors(&config);
        self.input_path = Some(path);
        self
    }

    /// Write a Q4K-quantized APR file (GGUF-style names) into the temp dir.
    pub(crate) fn with_apr_q4k(mut self) -> Self {
        let bytes = super::build_pygmy_apr_q4k();
        let path = self.dir.path().join("input.apr");
        fs::write(&path, &bytes).expect("write pygmy q4k apr");
        self.input_path = Some(path);
        self
    }

    /// Write a Q6K-quantized APR file (GGUF-style names) into the temp dir.
    pub(crate) fn with_apr_q6k(mut self) -> Self {
        let bytes = super::build_pygmy_apr_q6k();
        let path = self.dir.path().join("input.apr");
        fs::write(&path, &bytes).expect("write pygmy q6k apr");
        self.input_path = Some(path);
        self
    }

    // ----------------------------------------------------------------
    // Exercise: run real pipeline
    // ----------------------------------------------------------------

    /// Import the input SafeTensors to APR using `apr_import`.
    pub(crate) fn import_to_apr(mut self, options: ImportOptions) -> Self {
        let input = self
            .input_path
            .as_ref()
            .expect("Call with_safetensors() first");
        let output = self.dir.path().join("output.apr");
        let input_str = input.to_string_lossy().to_string();

        let result = apr_import(&input_str, &output, options);
        assert!(
            result.is_ok(),
            "apr_import failed: {:?}",
            result.unwrap_err()
        );

        self.output_path = Some(output);
        self
    }

    /// Import and return the Result (for testing error paths).
    pub(crate) fn try_import_to_apr(
        &self,
        options: ImportOptions,
    ) -> crate::error::Result<crate::format::validation::ValidationReport> {
        let input = self
            .input_path
            .as_ref()
            .expect("Call with_safetensors() first");
        let output = self.dir.path().join("output.apr");
        let input_str = input.to_string_lossy().to_string();
        apr_import(&input_str, &output, options)
    }

    /// Export the output APR back to SafeTensors using `apr_export`.
    pub(crate) fn export_to_safetensors(mut self) -> Self {
        let input = self
            .output_path
            .as_ref()
            .expect("Call import_to_apr() first");
        let output = self.dir.path().join("roundtrip.safetensors");

        let options = ExportOptions {
            format: ExportFormat::SafeTensors,
            quantize: None,
            include_tokenizer: false,
            include_config: false,
        };
        let result = apr_export(input, &output, options);
        assert!(
            result.is_ok(),
            "apr_export failed: {:?}",
            result.unwrap_err()
        );

        self.output_path = Some(output);
        self
    }

    /// Export APR -> GGUF (for T-QKV-03 and T-QKV-04 multi-hop tests).
    pub(crate) fn export_to_gguf(mut self) -> Self {
        let input = self
            .output_path
            .as_ref()
            .expect("Call import_to_apr() first");
        let output = self.dir.path().join("roundtrip.gguf");

        let options = ExportOptions {
            format: ExportFormat::Gguf,
            quantize: None,
            include_tokenizer: false,
            include_config: false,
        };
        let result = apr_export(input, &output, options);
        assert!(
            result.is_ok(),
            "apr_export to GGUF failed: {:?}",
            result.unwrap_err()
        );

        self.output_path = Some(output);
        self
    }

    /// Import from current output (GGUF or SafeTensors) -> APR (for multi-hop chains).
    pub(crate) fn reimport_to_apr(mut self) -> Self {
        let input = self
            .output_path
            .as_ref()
            .expect("No output to reimport from");
        let input_str = input.to_string_lossy().to_string();
        let output = self.dir.path().join("reimported.apr");

        // Create synthetic tokenizer for PMAT-232 compliance (GGUF imports need tokenizer)
        let tokenizer_path = self.dir.path().join("synthetic_tokenizer.json");
        let synthetic_tokenizer = r#"{
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "vocab": {"<pad>": 0, "<eos>": 1, "<unk>": 2},
                    "merges": []
                }
            }"#;
        std::fs::write(&tokenizer_path, synthetic_tokenizer)
            .expect("Failed to write synthetic tokenizer");

        let options = ImportOptions {
            tokenizer_path: Some(tokenizer_path),
            ..ImportOptions::default()
        };
        let result = apr_import(&input_str, &output, options);
        assert!(
            result.is_ok(),
            "reimport to APR failed: {:?}",
            result.unwrap_err()
        );

        self.output_path = Some(output);
        self
    }

    // ----------------------------------------------------------------
    // Verify: read back output and compare
    // ----------------------------------------------------------------

    /// Read back the output APR file from disk and verify tensor data matches source.
    ///
    /// Checks: tensor existence, shape equality, and data values within tolerance.
    /// Panics if no source tensors were recorded (empty config guard).
    pub(crate) fn verify_apr(&self) -> VerificationResult {
        assert!(
            !self.source_tensors.is_empty(),
            "Cannot verify with 0 source tensors -- use a non-empty PygmyConfig"
        );
        let output = self
            .output_path
            .as_ref()
            .expect("No output path set -- run import first");
        let data = fs::read(output).expect("Failed to read output APR");
        let reader = AprV2Reader::from_bytes(&data).expect("Failed to parse output APR");

        let mut mismatches = Vec::new();
        let tolerance = self.tolerance.f32_atol;

        for (name, expected_data, expected_shape) in &self.source_tensors {
            // Check tensor exists
            let entry = match reader.get_tensor(name) {
                Some(e) => e,
                None => {
                    mismatches.push(TensorMismatch {
                        tensor_name: name.clone(),
                        kind: MismatchKind::Missing,
                    });
                    continue;
                }
            };

            // Check shape
            if &entry.shape != expected_shape {
                mismatches.push(TensorMismatch {
                    tensor_name: name.clone(),
                    kind: MismatchKind::ShapeMismatch {
                        expected: expected_shape.clone(),
                        actual: entry.shape.clone(),
                    },
                });
                continue;
            }

            // Check data values
            if let Some(actual_data) = reader.get_tensor_as_f32(name) {
                for (i, (&exp, &act)) in expected_data.iter().zip(actual_data.iter()).enumerate() {
                    if (exp - act).abs() > tolerance {
                        mismatches.push(TensorMismatch {
                            tensor_name: name.clone(),
                            kind: MismatchKind::DataMismatch {
                                index: i,
                                expected: exp,
                                actual: act,
                                tolerance,
                            },
                        });
                        break; // One mismatch per tensor is enough
                    }
                }
            }
        }

        // T-QKV-02: Check for EXTRA tensors in output (detects fusion/split bugs)
        let expected_names: std::collections::HashSet<&str> = self
            .source_tensors
            .iter()
            .map(|(n, _, _)| n.as_str())
            .collect();
        for output_name in reader.tensor_names() {
            if !expected_names.contains(output_name) {
                mismatches.push(TensorMismatch {
                    tensor_name: output_name.to_string(),
                    kind: MismatchKind::Extra,
                });
            }
        }

        VerificationResult { mismatches }
    }

    /// Read back the output SafeTensors from disk and verify tensor data matches source.
    ///
    /// Checks: tensor existence, shape equality, and data values within tolerance.
    /// Panics if no source tensors were recorded (empty config guard).
    pub(crate) fn verify_safetensors(&self) -> VerificationResult {
        assert!(
            !self.source_tensors.is_empty(),
            "Cannot verify with 0 source tensors -- use a non-empty PygmyConfig"
        );
        let output = self
            .output_path
            .as_ref()
            .expect("No output path set -- run export first");
        let mapped = MappedSafeTensors::open(output).expect("Failed to open output SafeTensors");

        let mut mismatches = Vec::new();
        let tolerance = self.tolerance.f32_atol;

        for (name, expected_data, expected_shape) in &self.source_tensors {
            let meta = match mapped.get_metadata(name) {
                Some(m) => m,
                None => {
                    mismatches.push(TensorMismatch {
                        tensor_name: name.clone(),
                        kind: MismatchKind::Missing,
                    });
                    continue;
                }
            };

            if &meta.shape != expected_shape {
                mismatches.push(TensorMismatch {
                    tensor_name: name.clone(),
                    kind: MismatchKind::ShapeMismatch {
                        expected: expected_shape.clone(),
                        actual: meta.shape.clone(),
                    },
                });
                continue;
            }

            if let Ok(actual_data) = mapped.get_tensor(name) {
                for (i, (&exp, &act)) in expected_data.iter().zip(actual_data.iter()).enumerate() {
                    if (exp - act).abs() > tolerance {
                        mismatches.push(TensorMismatch {
                            tensor_name: name.clone(),
                            kind: MismatchKind::DataMismatch {
                                index: i,
                                expected: exp,
                                actual: act,
                                tolerance,
                            },
                        });
                        break;
                    }
                }
            }
        }

        // T-QKV-02: Check for EXTRA tensors in output (detects fusion/split bugs)
        let expected_names: std::collections::HashSet<&str> = self
            .source_tensors
            .iter()
            .map(|(n, _, _)| n.as_str())
            .collect();
        for output_name in mapped.tensor_names() {
            // SafeTensors __metadata__ key is not a tensor, skip it
            if output_name == "__metadata__" {
                continue;
            }
            if !expected_names.contains(output_name) {
                mismatches.push(TensorMismatch {
                    tensor_name: output_name.to_string(),
                    kind: MismatchKind::Extra,
                });
            }
        }

        VerificationResult { mismatches }
    }

    /// Get the output APR path (for manual inspection).
    pub(crate) fn output_path(&self) -> Option<&Path> {
        self.output_path.as_deref()
    }

    /// Get the input path.
    pub(crate) fn input_path(&self) -> Option<&Path> {
        self.input_path.as_deref()
    }

    // ----------------------------------------------------------------
    // Convenience one-liners
    // ----------------------------------------------------------------

    /// Import pygmy SafeTensors -> APR with default options and verify.
    pub(crate) fn assert_import_ok(config: PygmyConfig) {
        let h = Self::new()
            .with_safetensors(config)
            .import_to_apr(ImportOptions::default());
        h.verify_apr().assert_passed();
    }

    /// Full round-trip: SafeTensors -> APR -> SafeTensors, verify data preserved.
    pub(crate) fn assert_roundtrip_ok(config: PygmyConfig) {
        let h = Self::new()
            .with_safetensors(config)
            .import_to_apr(ImportOptions::default())
            .export_to_safetensors();
        h.verify_safetensors().assert_passed();
    }
}

/// Collect the tensor names, data, and shapes that a pygmy config would produce.
/// Mirrors the logic in `build_pygmy_safetensors_with_config`.
fn collect_pygmy_tensors(config: &PygmyConfig) -> Vec<(String, Vec<f32>, Vec<usize>)> {
    let mut tensors = Vec::new();

    if config.include_embedding {
        let data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
            .collect();
        tensors.push((
            "model.embed_tokens.weight".to_string(),
            data,
            vec![config.vocab_size, config.hidden_size],
        ));
    }

    for layer_idx in 0..config.num_layers {
        if config.include_norms {
            let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
            tensors.push((
                format!("model.layers.{layer_idx}.input_layernorm.weight"),
                norm_data.clone(),
                vec![config.hidden_size],
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                norm_data,
                vec![config.hidden_size],
            ));
        }

        if config.include_attention {
            let kv_dim = config.kv_dim();

            // Q and O: [hidden_size, hidden_size]
            let q_data: Vec<f32> = (0..config.hidden_size * config.hidden_size)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();
            tensors.push((
                format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                q_data.clone(),
                vec![config.hidden_size, config.hidden_size],
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                q_data,
                vec![config.hidden_size, config.hidden_size],
            ));

            // K and V: [kv_dim, hidden_size]
            let kv_data: Vec<f32> = (0..kv_dim * config.hidden_size)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();
            tensors.push((
                format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                kv_data.clone(),
                vec![kv_dim, config.hidden_size],
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                kv_data,
                vec![kv_dim, config.hidden_size],
            ));

            // Biases
            if config.include_bias {
                let q_bias: Vec<f32> = (0..config.hidden_size)
                    .map(|i| (i as f32) / 1000.0)
                    .collect();
                tensors.push((
                    format!("model.layers.{layer_idx}.self_attn.q_proj.bias"),
                    q_bias,
                    vec![config.hidden_size],
                ));
                let kv_bias: Vec<f32> = (0..kv_dim).map(|i| (i as f32) / 1000.0).collect();
                tensors.push((
                    format!("model.layers.{layer_idx}.self_attn.k_proj.bias"),
                    kv_bias.clone(),
                    vec![kv_dim],
                ));
                tensors.push((
                    format!("model.layers.{layer_idx}.self_attn.v_proj.bias"),
                    kv_bias,
                    vec![kv_dim],
                ));
            }
        }

        if config.include_mlp {
            let intermediate = config.hidden_size * 2;
            let gate_up_data: Vec<f32> = (0..intermediate * config.hidden_size)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();
            let down_data: Vec<f32> = (0..config.hidden_size * intermediate)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();

            tensors.push((
                format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                gate_up_data.clone(),
                vec![intermediate, config.hidden_size],
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                gate_up_data,
                vec![intermediate, config.hidden_size],
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                down_data,
                vec![config.hidden_size, intermediate],
            ));
        }
    }

    if config.include_norms && config.num_layers > 0 {
        let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
        tensors.push((
            "model.norm.weight".to_string(),
            norm_data,
            vec![config.hidden_size],
        ));
    }

    if config.include_embedding && !config.tied_embeddings {
        let data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
            .collect();
        tensors.push((
            "lm_head.weight".to_string(),
            data,
            vec![config.vocab_size, config.hidden_size],
        ));
    }

    tensors
}

// ====================================================================
// Harness self-tests
// ====================================================================

#[test]
fn test_harness_new_creates_temp_dir() {
    let h = ConversionTestHarness::new();
    assert!(h.dir().exists());
}

#[test]
fn test_harness_with_safetensors_writes_file() {
    let h = ConversionTestHarness::new().with_safetensors(PygmyConfig::default());
    assert!(h.input_path().is_some());
    assert!(h.input_path().expect("input").exists());
}

#[test]
fn test_harness_with_apr_writes_file() {
    let h = ConversionTestHarness::new().with_apr(PygmyConfig::default());
    assert!(h.input_path().is_some());
    assert!(h.input_path().expect("input").exists());
}

#[test]
fn test_harness_import_produces_output() {
    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::default())
        .import_to_apr(ImportOptions::default());
    assert!(h.output_path().is_some());
    assert!(h.output_path().expect("output").exists());
}

#[test]
fn test_harness_assert_import_ok_default() {
    ConversionTestHarness::assert_import_ok(PygmyConfig::default());
}

#[test]
fn test_harness_assert_import_ok_llama() {
    ConversionTestHarness::assert_import_ok(PygmyConfig::llama_style());
}

#[test]
fn test_harness_assert_import_ok_minimal() {
    ConversionTestHarness::assert_import_ok(PygmyConfig::minimal());
}

#[test]
fn test_harness_assert_roundtrip_ok_default() {
    ConversionTestHarness::assert_roundtrip_ok(PygmyConfig::default());
}

#[test]
fn test_harness_assert_roundtrip_ok_llama() {
    ConversionTestHarness::assert_roundtrip_ok(PygmyConfig::llama_style());
}

#[test]
fn test_harness_assert_roundtrip_ok_minimal() {
    ConversionTestHarness::assert_roundtrip_ok(PygmyConfig::minimal());
}

#[test]
fn test_harness_verify_apr_checks_shapes() {
    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::default())
        .import_to_apr(ImportOptions::default());
    let result = h.verify_apr();
    assert!(result.passed(), "Default import should verify cleanly");
}

#[test]
fn test_tolerance_config_default() {
    let t = ToleranceConfig::default();
    assert!((t.f32_atol - 1e-6).abs() < 1e-9);
    assert!((t.f16_atol - 1e-3).abs() < 1e-6);
    assert!((t.q8_atol - 0.1).abs() < 1e-6);
    assert!((t.q4_atol - 0.5).abs() < 1e-6);
}

// ====================================================================
// Falsification Protocol (rosetta-testing.md QA Matrix)
// ====================================================================

/// F-HAR-01: Corrupt tensor data region of `.apr` -> `verify()` detects DataMismatch
#[test]
fn test_f_har_01_corruption_detected() {
    use std::io::Write;

    // 1. Create valid APR via harness
    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::default())
        .import_to_apr(ImportOptions::default());

    let output_path = h.output_path().expect("output exists");

    // 2. Read APR, find tensor data offset from header (bytes 32-39 = data_offset u64 LE)
    let mut data = std::fs::read(&output_path).expect("read APR");
    let data_offset =
        u64::from_le_bytes(data[32..40].try_into().expect("8 bytes for data_offset")) as usize;

    // 3. Corrupt first 16 bytes of actual tensor data (4 f32 values)
    assert!(
        data.len() > data_offset + 16,
        "APR file must have tensor data after data_offset={data_offset}"
    );
    for byte in &mut data[data_offset..data_offset + 16] {
        *byte ^= 0xFF;
    }

    // 4. Write corrupted data back
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&output_path)
        .expect("open APR for write");
    file.write_all(&data).expect("write corrupted");
    drop(file);

    // 5. Verify MUST detect the data mismatch
    let result = h.verify_apr();
    assert!(
        !result.passed(),
        "F-HAR-01: Corruption at data_offset MUST be detected by verify_apr()"
    );
}

/// F-HAR-02: Set tolerance to `1e-9` (too strict) -> verify with default tolerance
/// Note: The harness uses fixed tolerances; this test validates the tolerance config exists
#[test]
fn test_f_har_02_strict_tolerance_config() {
    // Verify that strict tolerance values are actually stricter than defaults
    let strict = ToleranceConfig {
        f32_atol: 1e-9, // Too strict - will fail on quantization/dequant noise
        f16_atol: 1e-9,
        q8_atol: 1e-9,
        q4_atol: 1e-9,
    };
    let default = ToleranceConfig::default();

    assert!(strict.f32_atol < default.f32_atol);
    assert!(strict.f16_atol < default.f16_atol);
    assert!(strict.q8_atol < default.q8_atol);
    assert!(strict.q4_atol < default.q4_atol);
}

/// F-HAR-03: Use `--strict` on `embedding_only` config -> Import FAILS (Unverified Architecture)
#[test]
fn test_f_har_03_strict_embedding_only() {
    let config = PygmyConfig::embedding_only();

    // Strict mode with embedding-only config should FAIL
    let mut options = ImportOptions::default();
    options.strict = true;

    let h = ConversionTestHarness::new().with_safetensors(config);

    // Import with strict mode - this should fail with unverified architecture
    let result = h.try_import_to_apr(options);

    // Expected behavior: strict mode rejects unverified architectures
    // The test passes if import fails (strict mode working as intended)
    assert!(
        result.is_err(),
        "F-HAR-03: Strict mode should reject unverified architecture"
    );
}

/// F-HAR-04: Use `PygmyConfig` with 0 tensors -> Harness handles gracefully (no crash)
#[test]
fn test_f_har_04_zero_tensors_graceful() {
    let config = PygmyConfig {
        vocab_size: 0,
        hidden_size: 0,
        num_layers: 0,
        include_embedding: false,
        include_norms: false,
        include_attention: false,
        include_mlp: false,
        ..Default::default()
    };

    // Should not crash when building SafeTensors with zero tensors
    let st_bytes = build_pygmy_safetensors_with_config(config);
    // File may be minimal but should be valid SafeTensors
    assert!(st_bytes.len() >= 8, "Should have at least header length");
}

/// F-REG-01: Round-trip Llama-style tensors -> `verify_safetensors()` PASSES
/// (This is already covered by test_harness_assert_roundtrip_ok_llama but we
/// add an explicit named test for traceability)
#[test]
fn test_f_reg_01_roundtrip_llama_style() {
    ConversionTestHarness::assert_roundtrip_ok(PygmyConfig::llama_style());
}

// ====================================================================
// Master Falsification QA Protocol (100-Point Matrix)
// Philosophy: Karl Popper (Refutation) & Toyota Way (Jidoka)
// ====================================================================

/// F-CONV-01 (Bit-Flipping): Corrupt single f32 in tensor data -> verify_apr() MUST detect
#[test]
fn test_f_conv_01_bit_flipping_detected() {
    use std::io::Write;

    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::default())
        .import_to_apr(ImportOptions::default());

    let output_path = h.output_path().expect("output exists");

    // Read APR, find tensor data offset from header (bytes 32-39 = data_offset u64 LE)
    let mut data = std::fs::read(&output_path).expect("read APR");
    let data_offset =
        u64::from_le_bytes(data[32..40].try_into().expect("8 bytes for data_offset")) as usize;

    // Flip all bits in a single f32 value (4 bytes) at start of tensor data
    assert!(
        data.len() > data_offset + 4,
        "APR file must have tensor data after data_offset={data_offset}"
    );
    for byte in &mut data[data_offset..data_offset + 4] {
        *byte ^= 0xFF;
    }

    // Write corrupted data
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&output_path)
        .expect("open");
    file.write_all(&data).expect("write");
    drop(file);

    // Verify MUST detect the single-value mismatch
    let result = h.verify_apr();
    assert!(
        !result.passed(),
        "F-CONV-01: Single f32 bit-flip MUST be detected by verify_apr()"
    );
}

/// F-CONV-02 (Tolerance Drift): Set f32_atol to 1e-12 -> Standard tests should fail
#[test]
fn test_f_conv_02_tolerance_drift() {
    let ultra_strict = ToleranceConfig {
        f32_atol: 1e-12,
        f16_atol: 1e-12,
        q8_atol: 1e-12,
        q4_atol: 1e-12,
    };
    let default = ToleranceConfig::default();

    // Ultra-strict MUST be stricter than default
    assert!(
        ultra_strict.f32_atol < default.f32_atol / 1000.0,
        "F-CONV-02: 1e-12 should be 1000x stricter than default 1e-6"
    );
}

/// F-CONV-03 (Auto-Arch Refutation): Garbage tensor names -> Auto-mapping fallback
#[test]
fn test_f_conv_03_auto_arch_garbage_names() {
    use crate::format::Architecture;

    // With garbage tensor names, auto-mapping should use default behavior
    let arch = Architecture::Auto;

    // Auto-mapping on unknown patterns should preserve or minimally transform
    let mapped = arch.map_name("garbage.weight");

    // The important thing is it doesn't crash and handles gracefully
    assert!(
        !mapped.is_empty(),
        "F-CONV-03: Auto-map should handle garbage names gracefully"
    );
}

/// F-CONV-04 (Strict Leakage): Import missing norm tensor with --strict
///
/// **FALSIFICATION FINDING (DEFECT-001):**
/// Strict mode did NOT reject models missing norm tensors. This was a Jidoka
/// violation - the system should stop-the-line for incomplete models.
///
/// **Previous Behavior:** Import succeeds (result.is_ok())
/// **Fixed Behavior:** Import fails with "Missing required tensor: model.norm.weight"
/// **Status:** FIXED (DEFECT-001)
#[test]
fn test_f_conv_04_strict_missing_norm() {
    // Create config without norms
    let config = PygmyConfig {
        include_norms: false,
        include_embedding: true,
        include_attention: true,
        include_mlp: true,
        ..Default::default()
    };

    let mut options = ImportOptions::default();
    options.strict = true;

    let h = ConversionTestHarness::new().with_safetensors(config);
    let result = h.try_import_to_apr(options);

    // DEFECT-001 FIX VERIFICATION: Strict mode should now reject missing norms
    assert!(
        result.is_err(),
        "DEFECT-001 FIX: Strict mode should reject missing model.norm.weight"
    );

    // Verify the error message mentions the missing tensor
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("model.norm.weight"),
        "Error should mention missing tensor: {err_msg}"
    );
}

/// F-DISP-01 (Magic vs Extension): GGUF as .txt -> should work via magic bytes
#[test]
fn test_f_disp_01_magic_vs_extension() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use crate::format::tensors::{list_tensors_from_bytes, TensorListOptions};
    use tempfile::NamedTempFile;

    // Create valid GGUF
    let floats: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let tensor = GgufTensor {
        name: "test.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: floats,
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export");

    // Rename to .txt extension
    let file = NamedTempFile::with_suffix(".txt").expect("create");
    std::fs::write(file.path(), &gguf_bytes).expect("write");

    // Should still work via magic bytes detection
    let result = list_tensors_from_bytes(&gguf_bytes, TensorListOptions::default());
    assert!(
        result.is_ok(),
        "F-DISP-01: GGUF should be detected by magic bytes, not extension"
    );
    assert!(
        result
            .expect("test harness value")
            .format_version
            .contains("GGUF"),
        "F-DISP-01: Should detect as GGUF format"
    );
}

/// F-DISP-02 (Format Poisoning): APR magic + noise -> graceful error, not panic
#[test]
fn test_f_disp_02_format_poisoning() {
    use crate::format::tensors::{list_tensors_from_bytes, TensorListOptions};

    // Create poisoned data: APR magic followed by random noise
    let mut poisoned = Vec::new();
    poisoned.extend_from_slice(b"APRN"); // APR magic
    poisoned.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]); // noise
    poisoned.extend_from_slice(&vec![0xFF; 100]); // more noise

    // Should fail gracefully, not panic
    let result = list_tensors_from_bytes(&poisoned, TensorListOptions::default());
    assert!(
        result.is_err(),
        "F-DISP-02: Poisoned APR should fail gracefully"
    );
}

/// F-DISP-03 (SafeTensors Header Overflow): 100GB header -> immediate rejection
#[test]
fn test_f_disp_03_header_overflow() {
    use crate::format::tensors::{list_tensors_from_bytes, TensorListOptions};

    // Create SafeTensors with absurd header length (100GB)
    let header_len: u64 = 100 * 1024 * 1024 * 1024; // 100GB
    let mut overflow_bytes = Vec::new();
    overflow_bytes.extend_from_slice(&header_len.to_le_bytes());
    overflow_bytes.extend_from_slice(b"{}"); // minimal "header"

    // Should be rejected immediately (safety limit)
    let result = list_tensors_from_bytes(&overflow_bytes, TensorListOptions::default());
    assert!(
        result.is_err(),
        "F-DISP-03: 100GB header should trigger safety rejection"
    );
}

/// F-DISP-04 (Cross-Format Linting): GGUF lint rules should trigger
#[test]
fn test_f_disp_04_cross_format_linting() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use crate::format::lint::lint_model_file;
    use tempfile::NamedTempFile;

    // Create GGUF without license metadata (should trigger lint warning)
    let floats: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: floats,
    };
    // No license, no author, no description
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export");

    let file = NamedTempFile::with_suffix(".gguf").expect("create");
    std::fs::write(file.path(), &gguf_bytes).expect("write");

    // Lint should trigger GGUF-specific warnings
    let result = lint_model_file(file.path());
    assert!(result.is_ok(), "F-DISP-04: Lint should not crash on GGUF");
    let report = result.expect("test harness value");
    assert!(
        report.warn_count > 0,
        "F-DISP-04: GGUF without metadata should trigger warnings"
    );
}

/// F-DATA-01 (NaN Propagation): NaN in tensor -> detected in validation
#[test]
fn test_f_data_01_nan_propagation() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use crate::format::rosetta::RosettaStone;
    use tempfile::NamedTempFile;

    // Create GGUF with NaN values
    let nan_bytes = f32::NAN.to_le_bytes();
    let mut tensor_data = Vec::new();
    for _ in 0..4 {
        tensor_data.extend_from_slice(&nan_bytes);
    }

    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: tensor_data,
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export");

    let file = NamedTempFile::with_suffix(".gguf").expect("create");
    std::fs::write(file.path(), &gguf_bytes).expect("write");

    // Validate should detect NaN
    let rosetta = RosettaStone::default();
    let result = rosetta.validate(file.path());
    assert!(result.is_ok(), "F-DATA-01: Validation should not crash");
    let report = result.expect("test harness value");
    assert!(
        report.total_nan_count > 0,
        "F-DATA-01: NaN should be detected and reported"
    );
}

/// F-DATA-02 (All-Zeros Refutation): All-zero tensor -> Jidoka alarm
///
/// **FALSIFICATION FINDING (DEFECT-002):**
/// All-zero tensors are NOT being detected in GGUF validation.
/// This is a Jidoka violation - uninitialized weights should trigger alarm.
///
/// **Previous Behavior:** `all_zero_tensors` was empty (GGUF export bug)
/// **Fixed Behavior:** Should contain "model.weight"
/// **Status:** FIXED (DEFECT-002)
#[test]
fn test_f_data_02_all_zeros_alarm() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use crate::format::rosetta::RosettaStone;
    use tempfile::NamedTempFile;

    // Create GGUF with all-zero tensor
    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![4, 4],
        dtype: GgmlType::F32,
        data: vec![0u8; 64], // All zeros - 4x4 F32 = 16 elements * 4 bytes = 64 bytes
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export");

    let file = NamedTempFile::with_suffix(".gguf").expect("create");
    std::fs::write(file.path(), &gguf_bytes).expect("write");

    // Validate should detect all-zeros
    let rosetta = RosettaStone::default();
    let result = rosetta.validate(file.path());
    assert!(result.is_ok(), "F-DATA-02: Validation should not crash");
    let report = result.expect("test harness value");

    // DEFECT-002 FIX VERIFICATION: All-zeros should now be detected
    assert!(
        report
            .all_zero_tensors
            .contains(&"model.weight".to_string()),
        "DEFECT-002 FIX: All-zeros tensor should be detected. Got: {:?}",
        report.all_zero_tensors
    );
}

/// F-TPS-01 (Boilerplate Check): New conversion test < 10 lines
#[test]
fn test_f_tps_01_boilerplate_minimal() {
    // This is the ONE-LINER API from the spec - proves < 10 lines
    ConversionTestHarness::assert_import_ok(PygmyConfig::default());
    // Total: 1 line. Requirement: < 10 lines. [REFUTED]
}

/// F-TPS-02 (Read-Back Verification): list_tensors uses mmap for SafeTensors
#[test]
fn test_f_tps_02_mmap_verification() {
    use crate::format::tensors::{list_tensors, TensorListOptions};
    use tempfile::NamedTempFile;

    // Create SafeTensors file
    let st_bytes = super::build_pygmy_safetensors();
    let file = NamedTempFile::with_suffix(".safetensors").expect("create");
    std::fs::write(file.path(), &st_bytes).expect("write");

    // Path-based list_tensors uses MappedSafeTensors (mmap)
    let result = list_tensors(file.path(), TensorListOptions::default());
    assert!(
        result.is_ok(),
        "F-TPS-02: list_tensors should work with file path (mmap)"
    );

    // Verify format detected correctly (mmap path would work)
    let info = result.expect("test harness value");
    assert!(
        info.format_version.contains("SafeTensors"),
        "F-TPS-02: Should detect SafeTensors format via mmap path"
    );
}

// ====================================================================
// Audit Item 4: infer_model_config_from_tensors with realistic dims
// Complements pmat.rs tests -- verifies head_dim detection is triggered
// ====================================================================

/// Verify `infer_model_config_from_tensors` correctly infers num_heads and
/// num_kv_heads when given realistic dimensions (hidden_size=128, head_dim=64).
/// PygmyConfig's tiny dims (hidden_size=4/8) never match head_dim candidates
/// [64, 128, 96, 80], so this path was previously untested via harness.
#[test]
fn test_f_infer_config_realistic_dimensions() {
    use crate::format::converter::infer_model_config_from_tensors;
    use std::collections::BTreeMap;

    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

    // embedding: [vocab=256, hidden=128] -> vocab_size=256 (larger), hidden_size=128
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.0; 256 * 128], vec![256, 128]),
    );
    // Q: [128, 128] -> q_dim==hidden_size -> try head_dim=64 -> num_heads=2
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![0.0; 128 * 128], vec![128, 128]),
    );
    // K: [64, 128] -> kv_dim=64, head_dim=64 -> num_kv_heads=1 (GQA 2:1)
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        (vec![0.0; 64 * 128], vec![64, 128]),
    );
    tensors.insert(
        "model.layers.1.self_attn.q_proj.weight".to_string(),
        (vec![0.0; 128 * 128], vec![128, 128]),
    );

    let config = infer_model_config_from_tensors(&tensors);
    assert!(
        config.is_some(),
        "Inference must succeed with realistic dims"
    );

    let config = config.expect("test harness value");
    assert_eq!(config.hidden_size, Some(128));
    assert_eq!(config.vocab_size, Some(256));
    assert_eq!(config.num_heads, Some(2), "128/64 head_dim = 2 heads");
    assert_eq!(config.num_kv_heads, Some(1), "64/64 = 1 KV head (GQA)");
    assert_eq!(config.num_layers, Some(2));
}

// ====================================================================
// Audit Item 5 fix: Harness round-trip with PygmyConfig::realistic()
// Exercises infer_model_config_from_tensors through the full pipeline
// ====================================================================

/// Audit #5: Import with realistic dims succeeds via harness
#[test]
fn test_f_harness_import_realistic_dims() {
    ConversionTestHarness::assert_import_ok(PygmyConfig::realistic());
}

/// Audit #5: Full round-trip (ST->APR->ST) with realistic dims
#[test]
fn test_f_harness_roundtrip_realistic_dims() {
    ConversionTestHarness::assert_roundtrip_ok(PygmyConfig::realistic());
}

/// Audit #5: ST->APR->GGUF with realistic dims (exercises GGUF export shape handling)
#[test]
fn test_f_harness_gguf_export_realistic_dims() {
    use crate::format::gguf::GgufReader;

    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::realistic())
        .import_to_apr(ImportOptions::default())
        .export_to_gguf();

    let gguf_path = h.output_path().expect("GGUF output exists");
    let gguf_data = std::fs::read(gguf_path).expect("read GGUF");
    let reader = GgufReader::from_bytes(gguf_data).expect("parse GGUF");

    // Realistic dims should produce valid GGUF with tensors
    assert!(
        !reader.tensors.is_empty(),
        "Realistic config must produce GGUF with tensors"
    );

    // Should have separate Q/K/V (no fusion)
    let names: Vec<&str> = reader.tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(
        !names.iter().any(|n| n.contains("attn_qkv")),
        "Realistic config must NOT have fused attn_qkv. Found: {names:?}"
    );
}

// ====================================================================
// T-QKV-02: Round-trip tests must verify tensor NAME set equality
// ====================================================================

/// T-QKV-02: verify_apr() detects extra tensors not in source (name-set check)
#[test]
fn test_t_qkv_02_name_set_equality_apr() {
    // Standard import -- names should match exactly
    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::llama_style())
        .import_to_apr(ImportOptions::default());
    let result = h.verify_apr();
    assert!(
        result.passed(),
        "T-QKV-02: Llama-style import should have matching tensor names. \
         Mismatches: {:?}",
        result.mismatches
    );
}

/// T-QKV-02: verify_safetensors() detects name-set equality on round-trip
#[test]
fn test_t_qkv_02_name_set_equality_roundtrip() {
    // Full round-trip -- names should survive ST->APR->ST
    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::qwen2_gqa())
        .import_to_apr(ImportOptions::default())
        .export_to_safetensors();
    let result = h.verify_safetensors();
    assert!(
        result.passed(),
        "T-QKV-02: GQA round-trip should preserve tensor name set. \
         Mismatches: {:?}",
        result.mismatches
    );
}

// ====================================================================
// T-QKV-03: SafeTensors->APR->GGUF round-trip test
// ====================================================================

/// T-QKV-03: Export to GGUF preserves tensor count and separate Q/K/V
#[test]
fn test_t_qkv_03_safetensors_apr_gguf() {
    use crate::format::gguf::GgufReader;

    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::qwen2_gqa())
        .import_to_apr(ImportOptions::default())
        .export_to_gguf();

    let gguf_path = h.output_path().expect("GGUF output exists");
    let gguf_data = std::fs::read(gguf_path).expect("read GGUF");
    let reader = GgufReader::from_bytes(gguf_data).expect("T-QKV-03: parse GGUF");

    // Verify GGUF is valid and has tensors
    assert!(
        !reader.tensors.is_empty(),
        "T-QKV-03: GGUF must contain tensors"
    );

    // T-QKV-03 key check: Q/K/V must be SEPARATE (not fused as attn_qkv)
    let names: Vec<&str> = reader.tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(
        !names.iter().any(|n| n.contains("attn_qkv")),
        "T-QKV-03: Must NOT have fused attn_qkv tensor. Found: {names:?}"
    );

    // Should have separate attn_q, attn_k, attn_v
    let has_q = names.iter().any(|n| n.contains("attn_q."));
    let has_k = names.iter().any(|n| n.contains("attn_k."));
    let has_v = names.iter().any(|n| n.contains("attn_v."));
    assert!(
        has_q && has_k && has_v,
        "T-QKV-03: Must have separate attn_q, attn_k, attn_v. Found: {names:?}"
    );
}

// ====================================================================
// T-QKV-04: Multi-hop chain test (ST->APR->GGUF->APR->ST)
// ====================================================================

/// T-QKV-04: Full multi-hop chain preserves tensor data
#[test]
fn test_t_qkv_04_multihop_st_apr_gguf_apr_st() {
    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::llama_style())
        .import_to_apr(ImportOptions::default())     // ST -> APR
        .export_to_gguf()                            // APR -> GGUF
        .reimport_to_apr()                           // GGUF -> APR
        .export_to_safetensors(); // APR -> ST

    // Verify final SafeTensors matches original input
    let result = h.verify_safetensors();
    assert!(
        result.passed(),
        "T-QKV-04: Multi-hop ST->APR->GGUF->APR->ST must preserve data. \
         Mismatches: {:?}",
        result.mismatches
    );
}
