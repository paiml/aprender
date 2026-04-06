//! SafeTensors loading example for realizar
//!
//! Demonstrates how to:
//! - Load SafeTensors files (compatible with aprender models)
//! - Extract tensor data using the helper API
//! - Inspect model structure and metadata
//!
//! This example shows interoperability with aprender-trained models.

use realizar::safetensors::SafetensorsModel;

fn main() {
    println!("=== SafeTensors Loading Example ===\n");

    // Example 1: Create a simple SafeTensors file (simulating aprender output)
    println!("--- Creating Example SafeTensors File ---");
    let safetensors_data = create_example_linear_regression_model();
    println!("Created SafeTensors file: {} bytes", safetensors_data.len());
    println!();

    // Example 2: Load and parse SafeTensors
    println!("--- Loading SafeTensors Model ---");
    let model =
        SafetensorsModel::from_bytes(&safetensors_data).expect("Failed to load SafeTensors model");

    println!("Successfully loaded model");
    println!("  - Number of tensors: {}", model.tensors.len());
    println!("  - Total data size: {} bytes", model.data.len());
    println!();

    // Example 3: Inspect metadata
    println!("--- Tensor Metadata ---");
    for (name, info) in &model.tensors {
        println!("Tensor: {}", name);
        println!("  - dtype: {:?}", info.dtype);
        println!("  - shape: {:?}", info.shape);
        println!("  - data_offsets: {:?}", info.data_offsets);
        let size = info.data_offsets[1] - info.data_offsets[0];
        println!("  - size: {} bytes", size);
        println!();
    }

    // Example 4: Extract tensor data (new helper API)
    println!("--- Extracting Tensor Data ---");

    // Extract coefficients
    let coefficients = model
        .get_tensor_f32("coefficients")
        .expect("Failed to extract coefficients");
    println!("Coefficients: {:?}", coefficients);
    println!("  - length: {}", coefficients.len());
    println!("  - values: {:?}", coefficients);
    println!();

    // Extract intercept
    let intercept = model
        .get_tensor_f32("intercept")
        .expect("Failed to extract intercept");
    println!("Intercept: {:?}", intercept);
    println!("  - length: {}", intercept.len());
    println!("  - value: {}", intercept[0]);
    println!();

    // Example 5: Use extracted data for inference
    println!("--- Linear Regression Inference ---");
    // Model: y = 2.0*x1 + 3.0*x2 + 1.5*x3 + 0.5
    let features = vec![1.0, 2.0, 3.0];
    let prediction = linear_regression_predict(&coefficients, intercept[0], &features);
    println!("Input features: {:?}", features);
    println!("Model equation: y = 2.0*x1 + 3.0*x2 + 1.5*x3 + 0.5");
    println!("Prediction: {:.2}", prediction);

    // Verify calculation: 2.0*1.0 + 3.0*2.0 + 1.5*3.0 + 0.5 = 2.0 + 6.0 + 4.5 + 0.5 = 13.0
    let expected = 2.0 * 1.0 + 3.0 * 2.0 + 1.5 * 3.0 + 0.5;
    println!("Expected: {:.2}", expected);
    println!("Match: {}", (prediction - expected).abs() < 1e-6);
    println!();

    println!("=== SafeTensors Loading Complete ===");
}

/// Create an example SafeTensors file representing a LinearRegression model
/// This simulates what aprender would produce when saving a trained model
fn create_example_linear_regression_model() -> Vec<u8> {
    use std::collections::BTreeMap;

    // Model: y = 2.0*x1 + 3.0*x2 + 1.5*x3 + 0.5
    // coefficients: [2.0, 3.0, 1.5]
    // intercept: [0.5]

    // Create metadata (JSON)
    let mut metadata = BTreeMap::new();

    // Coefficients tensor
    metadata.insert(
        "coefficients".to_string(),
        serde_json::json!({
            "dtype": "F32",
            "shape": [3],
            "data_offsets": [0, 12]  // 3 floats * 4 bytes = 12
        }),
    );

    // Intercept tensor
    metadata.insert(
        "intercept".to_string(),
        serde_json::json!({
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [12, 16]  // 1 float * 4 bytes = 4, starts at 12
        }),
    );

    let metadata_json = serde_json::to_string(&metadata).expect("Failed to serialize metadata");
    let metadata_bytes = metadata_json.as_bytes();

    let mut data = Vec::new();

    // Header: 8-byte metadata length (little-endian)
    data.extend_from_slice(&(metadata_bytes.len() as u64).to_le_bytes());

    // Metadata: JSON
    data.extend_from_slice(metadata_bytes);

    // Tensor data: F32 values (little-endian)
    // coefficients: [2.0, 3.0, 1.5]
    data.extend_from_slice(&2.0f32.to_le_bytes());
    data.extend_from_slice(&3.0f32.to_le_bytes());
    data.extend_from_slice(&1.5f32.to_le_bytes());

    // intercept: [0.5]
    data.extend_from_slice(&0.5f32.to_le_bytes());

    data
}

/// Simple linear regression prediction
fn linear_regression_predict(coefficients: &[f32], intercept: f32, features: &[f32]) -> f32 {
    assert_eq!(
        coefficients.len(),
        features.len(),
        "Coefficient and feature lengths must match"
    );

    let dot_product: f32 = coefficients
        .iter()
        .zip(features.iter())
        .map(|(c, f)| c * f)
        .sum();

    dot_product + intercept
}
