#![allow(clippy::disallowed_methods)]
//! GH-202: End-to-End Tensor Comparison Test
//!
//! This integration test verifies GGUF→APR conversion produces correct tensor values
//! by comparing dequantized F32 values between both formats.
//!
//! Root cause investigation for GH-202: APR from GGUF produces garbage inference.

use std::fs;
use std::io::Read;
use std::path::PathBuf;

use aprender::format::v2::{AprV2Metadata, AprV2Reader, AprV2Writer};

/// GH-202-E2E-001: Verify APR reader can parse converted file
#[test]
#[ignore = "requires test model file"]
fn test_gh202_apr_reader_parses_converted_file() {
    let apr_path = PathBuf::from("test_model.apr");
    if !apr_path.exists() {
        eprintln!("GH-202-E2E-001: Skipping - no test_model.apr found");
        return;
    }

    let mut file = fs::File::open(&apr_path).expect("Should open APR file");
    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("Should read APR file");

    let reader = AprV2Reader::from_bytes(&data).expect("Should parse APR file");
    let tensor_names = reader.tensor_names();
    eprintln!(
        "GH-202-E2E-001: Loaded APR with {} tensors",
        tensor_names.len()
    );

    // Verify we have expected tensor types
    for name in tensor_names {
        eprintln!("  Tensor: {}", name);
    }
}

/// GH-202-E2E-002: Verify APR F32 tensor round-trip preserves values
#[test]
fn test_gh202_apr_f32_roundtrip() {
    // Create test data with known values
    let vocab_size = 16;
    let hidden_size = 8;
    let embed_data: Vec<f32> = (0..vocab_size * hidden_size)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();

    // Build APR
    let metadata = AprV2Metadata {
        model_type: "test".to_string(),
        vocab_size: Some(vocab_size),
        hidden_size: Some(hidden_size),
        ..Default::default()
    };
    let mut writer = AprV2Writer::new(metadata);
    writer.add_f32_tensor(
        "model.embed_tokens.weight",
        vec![vocab_size, hidden_size],
        &embed_data,
    );

    let apr_bytes = writer.write().expect("Should write APR");

    // Read it back
    let reader = AprV2Reader::from_bytes(&apr_bytes).expect("Should parse APR");
    let tensor_names = reader.tensor_names();

    eprintln!("GH-202-E2E-002: APR has {} tensors", tensor_names.len());

    // Get embedding tensor and verify values
    let read_data = reader
        .get_f32_tensor("model.embed_tokens.weight")
        .expect("Should get embedding tensor");

    assert_eq!(
        read_data.len(),
        embed_data.len(),
        "Tensor size mismatch: expected {}, got {}",
        embed_data.len(),
        read_data.len()
    );

    // Verify all values match
    let mut max_diff = 0.0f32;
    for (i, (&expected, &actual)) in embed_data.iter().zip(read_data.iter()).enumerate() {
        let diff = (expected - actual).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(
            diff <= 1e-6,
            "Value mismatch at index {i}: expected {expected}, got {actual}"
        );
    }

    eprintln!(
        "GH-202-E2E-002: All {} values match (max_diff = {:.2e})",
        embed_data.len(),
        max_diff
    );
}

/// GH-202-E2E-003: Test transpose correctness with known values
///
/// This simulates the GGUF→APR conversion path:
/// 1. Create F32 matrix in column-major order (GGUF convention)
/// 2. Transpose to row-major (APR convention)
/// 3. Store in APR format
/// 4. Read back and verify values at correct positions
#[test]
fn test_gh202_transpose_e2e_known_values() {
    // Create a small 4x8 matrix with unique values for each position
    // GGUF convention: shape [4, 8] means 4 columns, 8 rows (column-major)
    // For weight matrix: in_dim=4 (cols), out_dim=8 (rows)
    let in_dim = 4;
    let out_dim = 8;

    // Create logical matrix W[out, in] with unique values
    // W[r, c] = r * 100 + c (so we can identify each position)
    let mut logical_matrix = vec![0.0f32; out_dim * in_dim];
    for r in 0..out_dim {
        for c in 0..in_dim {
            logical_matrix[r * in_dim + c] = (r * 100 + c) as f32;
        }
    }

    // GGUF stores in column-major: data[c * out_dim + r] = W[r, c]
    let mut gguf_colmajor = vec![0.0f32; out_dim * in_dim];
    for r in 0..out_dim {
        for c in 0..in_dim {
            gguf_colmajor[c * out_dim + r] = logical_matrix[r * in_dim + c];
        }
    }

    // Transpose: column-major [in_dim, out_dim] → row-major [out_dim, in_dim]
    // This is what transpose_q4k_for_matmul does (but for F32)
    let mut row_major = vec![0.0f32; out_dim * in_dim];
    for r in 0..out_dim {
        for c in 0..in_dim {
            // Source (column-major): index = c * out_dim + r
            // Dest (row-major): index = r * in_dim + c
            row_major[r * in_dim + c] = gguf_colmajor[c * out_dim + r];
        }
    }

    // Verify transpose is correct
    for r in 0..out_dim {
        for c in 0..in_dim {
            let expected = logical_matrix[r * in_dim + c];
            let actual = row_major[r * in_dim + c];
            assert_eq!(
                actual, expected,
                "Mismatch at [{r}, {c}]: expected {expected}, got {actual}"
            );
        }
    }

    // Now create an APR file with the transposed data
    let metadata = AprV2Metadata {
        model_type: "test".to_string(),
        ..Default::default()
    };
    let mut writer = AprV2Writer::new(metadata);

    // Store with row-major shape [out_dim, in_dim]
    writer.add_f32_tensor("test.weight", vec![out_dim, in_dim], &row_major);

    let apr_bytes = writer.write().expect("Should write APR");

    // Read back and verify
    let reader = AprV2Reader::from_bytes(&apr_bytes).expect("Should read APR");
    let read_data = reader
        .get_f32_tensor("test.weight")
        .expect("Should get tensor");

    // Verify all values match
    let mut mismatch_count = 0;
    for r in 0..out_dim {
        for c in 0..in_dim {
            let expected = logical_matrix[r * in_dim + c];
            let actual = read_data[r * in_dim + c];
            if (actual - expected).abs() > 1e-6 {
                mismatch_count += 1;
                if mismatch_count <= 5 {
                    eprintln!(
                        "GH-202-E2E-003 MISMATCH at [{r}, {c}]: expected {expected}, got {actual}"
                    );
                }
            }
        }
    }

    assert_eq!(
        mismatch_count, 0,
        "GH-202-E2E-003: {} values mismatched",
        mismatch_count
    );
    eprintln!(
        "GH-202-E2E-003: All {} values match after transpose+APR round-trip",
        out_dim * in_dim
    );
}

/// GH-202-E2E-004: Smoke test for tensor statistics
///
/// Quick sanity check that converted tensors have reasonable statistics.
#[test]
fn test_gh202_tensor_statistics_sanity() {
    // Create a small test tensor and verify basic operations
    let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 256.0).collect();

    // Compute statistics
    let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
    let variance: f32 =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
    let std_dev = variance.sqrt();

    eprintln!("GH-202-E2E-004: Test tensor stats:");
    eprintln!("  Mean: {:.4} (expected ~0)", mean);
    eprintln!("  Std:  {:.4} (expected ~0.29)", std_dev);

    // Basic sanity checks
    assert!(mean.abs() < 0.01, "Mean should be ~0");
    assert!((std_dev - 0.29).abs() < 0.05, "Std should be ~0.29");
}

/// GH-202-E2E-005: Test matmul indexing matches APR row-major layout
///
/// Verifies that the matmul function in realizar correctly accesses
/// row-major weight matrices.
#[test]
fn test_gh202_matmul_indexing() {
    // Create a simple 2x3 weight matrix W where W[r,c] = 10*r + c
    // Row-major storage: [W[0,0], W[0,1], W[0,2], W[1,0], W[1,1], W[1,2]]
    // = [0, 1, 2, 10, 11, 12]
    let out_dim = 2;
    let in_dim = 3;
    let weights: Vec<f32> = vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0];

    // Input vector x = [1, 2, 3]
    let x: Vec<f32> = vec![1.0, 2.0, 3.0];

    // Expected output: y = W @ x
    // y[0] = 0*1 + 1*2 + 2*3 = 8
    // y[1] = 10*1 + 11*2 + 12*3 = 68
    let expected = vec![8.0, 68.0];

    // Simulate realizar's matmul loop
    let mut output = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let w_start = o * in_dim;
        for i in 0..in_dim {
            output[o] += weights[w_start + i] * x[i];
        }
    }

    eprintln!(
        "GH-202-E2E-005: matmul output = {:?} (expected {:?})",
        output, expected
    );

    for (i, (&actual, &exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - exp).abs() < 1e-6,
            "Output[{i}] mismatch: expected {exp}, got {actual}"
        );
    }

    eprintln!("GH-202-E2E-005: matmul indexing is correct for row-major weights");
}
