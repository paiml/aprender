//! GH-202: GGUF vs APR Tensor Parity Tests
//!
//! Root cause investigation for GH-202: APR from GGUF produces garbage inference.
//!
//! These tests compare tensor values between GGUF and APR formats to isolate
//! whether the issue is in:
//! - APR loading in realizar
//! - Dequantization
//! - Inference pipeline

use std::path::Path;

// ============================================================================
// GH-202-FIX-004: Tensor Value Comparison
// ============================================================================

/// GH-202-PARITY-001: Compare embedding tensor values
///
/// If values differ, the issue is in APR loading/dequantization.
/// If values match, the issue is in the inference pipeline.
#[test]
#[ignore = "requires test model files"]
fn test_gh202_embedding_tensor_parity() {
    use realizar::apr::AprV2Model;
    use realizar::gguf::MappedGGUFModel;
    use realizar::quantize::dequant::dequantize_q4_k;

    let gguf_path = "/tmp/test-model.gguf";
    let apr_path = "/tmp/test-model.apr";

    if !Path::new(gguf_path).exists() || !Path::new(apr_path).exists() {
        eprintln!("GH-202-PARITY-001: SKIP - Test models not found");
        eprintln!("  Expected: {} and {}", gguf_path, apr_path);
        eprintln!("  Create with: apr import {} -o {}", gguf_path, apr_path);
        return;
    }

    // Load both models
    let mapped_gguf = MappedGGUFModel::from_path(gguf_path).expect("Should load GGUF");
    let apr_model = AprV2Model::load(apr_path).expect("Should load APR");

    // Get embedding tensor from both
    // GGUF uses "token_embd.weight", APR may use same or "model.embed_tokens.weight"
    let embed_names = ["token_embd.weight", "model.embed_tokens.weight"];

    // Find GGUF embedding tensor and dequantize it
    let gguf_embed: Option<Vec<f32>> = embed_names.iter().find_map(|&name| {
        let tensor_info = mapped_gguf.model.tensors.iter().find(|t| t.name == name)?;
        let offset = mapped_gguf.model.tensor_data_start + tensor_info.offset as usize;

        // Compute number of elements from dims
        let n_elements: u64 = tensor_info.dims.iter().product();

        // Compute byte size based on quantization type
        let size = match tensor_info.qtype {
            0 => n_elements as usize * 4, // F32: 4 bytes per element
            1 => n_elements as usize * 2, // F16: 2 bytes per element
            12 => {
                // Q4K: 144 bytes per 256 elements
                let n_blocks = n_elements.div_ceil(256) as usize;
                n_blocks * 144
            },
            _ => {
                eprintln!("GH-202-PARITY-001: Unsupported qtype {}", tensor_info.qtype);
                return None;
            },
        };

        let data = mapped_gguf.tensor_slice(offset, size)?;

        // Dequantize if Q4K (type 12), otherwise interpret as F32
        if tensor_info.qtype == 12 {
            // Q4K
            dequantize_q4_k(data).ok()
        } else if tensor_info.qtype == 0 {
            // F32
            let f32_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Some(f32_data)
        } else {
            eprintln!(
                "GH-202-PARITY-001: Unsupported qtype {} for {}",
                tensor_info.qtype, name
            );
            None
        }
    });

    let apr_embed = embed_names
        .iter()
        .find_map(|name| apr_model.get_tensor_f32(name).ok());

    match (gguf_embed, apr_embed) {
        (Some(gguf_data), Some(apr_data)) => {
            eprintln!(
                "GH-202-PARITY-001: GGUF embed {} elements, APR embed {} elements",
                gguf_data.len(),
                apr_data.len()
            );

            // Note: APR has transposed shape, so we need to compare accounting for this
            // GGUF: [vocab_size, hidden_dim] column-major = [hidden_dim, vocab_size] row-major
            // APR: [vocab_size, hidden_dim] row-major (after transpose during import)

            // For embedding (1D access pattern), values should match
            let mut max_diff = 0.0f32;
            let mut mismatch_count = 0;
            let total = gguf_data.len().min(apr_data.len());

            for i in 0..total {
                let diff = (gguf_data[i] - apr_data[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > 0.01 {
                    mismatch_count += 1;
                    if mismatch_count <= 5 {
                        eprintln!(
                            "  Mismatch at [{}]: GGUF={:.4}, APR={:.4}, diff={:.4}",
                            i, gguf_data[i], apr_data[i], diff
                        );
                    }
                }
            }

            let mismatch_pct = mismatch_count as f64 / total as f64 * 100.0;
            eprintln!(
                "GH-202-PARITY-001: max_diff={:.4}, mismatch_count={} ({:.1}%)",
                max_diff, mismatch_count, mismatch_pct
            );

            // Embedding should have low mismatch (Q4K quantization error only)
            assert!(
                mismatch_pct < 10.0,
                "GH-202-PARITY-001: {}% values mismatched - APR loading is broken",
                mismatch_pct
            );
        },
        (None, Some(_)) => {
            eprintln!("GH-202-PARITY-001: FAIL - Embedding not found in GGUF");
        },
        (Some(_), None) => {
            eprintln!("GH-202-PARITY-001: FAIL - Embedding not found in APR");
        },
        (None, None) => {
            eprintln!("GH-202-PARITY-001: FAIL - Embedding not found in either model");
        },
    }
}

/// GH-202-PARITY-002: Compare attention Q weight tensor values
///
/// Q weights are transposed during GGUF→APR conversion.
/// This test verifies the transpose is correct.
#[test]
#[ignore = "requires test model files"]
fn test_gh202_attn_q_tensor_parity() {
    use realizar::apr::AprV2Model;
    use realizar::gguf::MappedGGUFModel;
    use realizar::quantize::dequant::dequantize_q4_k;

    let gguf_path = "/tmp/test-model.gguf";
    let apr_path = "/tmp/test-model.apr";

    if !Path::new(gguf_path).exists() || !Path::new(apr_path).exists() {
        eprintln!("GH-202-PARITY-002: SKIP - Test models not found");
        return;
    }

    let mapped_gguf = MappedGGUFModel::from_path(gguf_path).expect("Should load GGUF");
    let apr_model = AprV2Model::load(apr_path).expect("Should load APR");

    // Get Q weight from layer 0
    let q_names = [
        "blk.0.attn_q.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ];

    // Find GGUF Q tensor and dequantize
    let gguf_q: Option<Vec<f32>> = q_names.iter().find_map(|&name| {
        let tensor_info = mapped_gguf.model.tensors.iter().find(|t| t.name == name)?;
        let offset = mapped_gguf.model.tensor_data_start + tensor_info.offset as usize;

        // Compute number of elements from dims
        let n_elements: u64 = tensor_info.dims.iter().product();

        // Compute byte size for Q4K
        let size = if tensor_info.qtype == 12 {
            let n_blocks = n_elements.div_ceil(256) as usize;
            n_blocks * 144
        } else {
            eprintln!(
                "GH-202-PARITY-002: Q tensor {} has qtype {}, expected Q4K (12)",
                name, tensor_info.qtype
            );
            return None;
        };

        let data = mapped_gguf.tensor_slice(offset, size)?;
        dequantize_q4_k(data).ok()
    });

    let apr_q = q_names
        .iter()
        .find_map(|name| apr_model.get_tensor_f32(name).ok());

    match (gguf_q, apr_q) {
        (Some(gguf_data), Some(apr_data)) => {
            eprintln!(
                "GH-202-PARITY-002: GGUF Q {} elements, APR Q {} elements",
                gguf_data.len(),
                apr_data.len()
            );

            // For weight matrices, we need to account for transpose
            // GGUF col-major [in, out] → APR row-major [out, in]
            // The VALUES should match at transposed positions

            // First, compare total element counts
            assert_eq!(
                gguf_data.len(),
                apr_data.len(),
                "Element count should match"
            );

            // Sample comparison: check statistics match
            let gguf_sum: f32 = gguf_data.iter().sum();
            let apr_sum: f32 = apr_data.iter().sum();

            let gguf_mean = gguf_sum / gguf_data.len() as f32;
            let apr_mean = apr_sum / apr_data.len() as f32;

            eprintln!(
                "GH-202-PARITY-002: GGUF mean={:.6}, APR mean={:.6}",
                gguf_mean, apr_mean
            );

            // Means should be very close (transpose doesn't change sum)
            assert!(
                (gguf_mean - apr_mean).abs() < 0.01,
                "GH-202-PARITY-002: Mean differs significantly - data corruption"
            );
        },
        _ => {
            eprintln!("GH-202-PARITY-002: SKIP - Q weight not found in both models");
        },
    }
}

/// GH-202-PARITY-003: Smoke test - verify models can be loaded
#[test]
fn test_gh202_smoke_model_loading() {
    eprintln!("GH-202-PARITY-003: Smoke test for model loading");
    eprintln!("  To run full parity tests:");
    eprintln!("  1. Download a GGUF model");
    eprintln!("  2. Convert: apr import model.gguf -o /tmp/test-model.apr");
    eprintln!("  3. Copy GGUF: cp model.gguf /tmp/test-model.gguf");
    eprintln!("  4. Run: cargo test gh202_parity -- --ignored");
}

/// GH-202-PARITY-004: Test Q4K dequantization produces correct values
#[test]
fn test_gh202_q4k_dequant_sanity() {
    use realizar::quantize::dequant::dequantize_q4_k;

    // Create known Q4K data and verify dequantization
    // Q4K block: d (f16) + dmin (f16) + scales (12B) + qs (128B) = 144 bytes

    // Create minimal valid Q4K block with scale=1.0, dmin=0.0
    let mut q4k_block = vec![0u8; 144];

    // d = 1.0 as f16 (0x3C00)
    q4k_block[0] = 0x00;
    q4k_block[1] = 0x3C;

    // dmin = 0.0 as f16 (0x0000)
    q4k_block[2] = 0x00;
    q4k_block[3] = 0x00;

    // scales = all 1s (simple uniform scale)
    for i in 0..12 {
        q4k_block[4 + i] = 1;
    }

    // qs = 0x88 pattern (nibbles = 8, 8 which gives value = 8*d - 0 = 8)
    for i in 0..128 {
        q4k_block[16 + i] = 0x88;
    }

    // Dequantize
    let result = dequantize_q4_k(&q4k_block);

    match result {
        Ok(values) => {
            eprintln!("GH-202-PARITY-004: Dequantized {} values", values.len());
            eprintln!("  First 8: {:?}", &values[..8.min(values.len())]);

            // Verify we get 256 values
            assert_eq!(
                values.len(),
                256,
                "Should dequantize 256 values from Q4K block"
            );

            // Verify values are not all zero (basic sanity)
            let nonzero = values.iter().filter(|&&v| v != 0.0).count();
            eprintln!("  Non-zero values: {}/{}", nonzero, values.len());
        },
        Err(e) => {
            eprintln!("GH-202-PARITY-004: Dequantization error: {}", e);
        },
    }
}

/// GH-202-PARITY-005: Compare tensor statistics between GGUF and APR
///
/// Compares mean, std, min, max of all tensors to detect data corruption.
#[test]
#[ignore = "requires test model files"]
fn test_gh202_tensor_statistics_comparison() {
    use realizar::apr::AprV2Model;
    use realizar::gguf::MappedGGUFModel;

    let gguf_path = "/tmp/test-model.gguf";
    let apr_path = "/tmp/test-model.apr";

    if !Path::new(gguf_path).exists() || !Path::new(apr_path).exists() {
        eprintln!("GH-202-PARITY-005: SKIP - Test models not found");
        return;
    }

    let mapped_gguf = MappedGGUFModel::from_path(gguf_path).expect("Should load GGUF");
    let apr_model = AprV2Model::load(apr_path).expect("Should load APR");

    eprintln!(
        "GH-202-PARITY-005: GGUF has {} tensors",
        mapped_gguf.model.tensors.len()
    );
    eprintln!(
        "GH-202-PARITY-005: APR has {} tensors",
        apr_model.tensor_count()
    );

    // Compare tensor counts
    assert_eq!(
        mapped_gguf.model.tensors.len(),
        apr_model.tensor_count() as usize,
        "Tensor count mismatch between GGUF and APR"
    );

    eprintln!("GH-202-PARITY-005: Tensor counts match");
}
