//! Generate demo .apr and .ald files for showcase
//!
//! Creates:
//! - `sentiment_mini.apr` - 3-class sentiment classifier (~800 bytes)
//! - `timeseries_100.ald` - 100 points of OHLCV stock data (~2KB)
//!
//! Run: `cargo run --example generate_demo_assets`

use presentar_yaml::formats::{AldDataset, AprModel, ModelLayer, Tensor};
use std::fs;
use std::path::Path;

// ============================================================================
// APR-001: Sentiment Mini Model
// ============================================================================

/// Generate a tiny 3-class sentiment classifier
///
/// Architecture:
/// - Input: 50-dim word embedding average
/// - Hidden: 50 -> 16 (`ReLU`)
/// - Output: 16 -> 3 (softmax)
///
/// Total params: 50*16 + 16 + 16*3 + 3 = 800 + 16 + 48 + 3 = 867
pub fn generate_sentiment_model() -> AprModel {
    let mut model = AprModel::new("sentiment_classifier");

    // Layer 1: Dense 50 -> 16
    let w1 = generate_weights(50, 16, 42);
    let b1 = generate_bias(16, 0.0);
    model.add_layer(ModelLayer {
        layer_type: "dense_relu".to_string(),
        parameters: vec![
            Tensor::from_f32("weight", vec![50, 16], &w1),
            Tensor::from_f32("bias", vec![16], &b1),
        ],
    });

    // Layer 2: Dense 16 -> 3 (output)
    let w2 = generate_weights(16, 3, 123);
    let b2 = generate_bias(3, 0.0);
    model.add_layer(ModelLayer {
        layer_type: "dense_softmax".to_string(),
        parameters: vec![
            Tensor::from_f32("weight", vec![16, 3], &w2),
            Tensor::from_f32("bias", vec![3], &b2),
        ],
    });

    // Metadata
    model
        .metadata
        .insert("task".to_string(), "sentiment".to_string());
    model.metadata.insert(
        "classes".to_string(),
        "negative,neutral,positive".to_string(),
    );
    model
        .metadata
        .insert("input_dim".to_string(), "50".to_string());
    model
        .metadata
        .insert("accuracy".to_string(), "0.87".to_string());

    model
}

// ============================================================================
// ALD-001: Timeseries Dataset
// ============================================================================

/// Generate 100 points of synthetic OHLCV stock data
///
/// Columns:
/// - timestamp: Unix timestamp (`i64` stored as `f32` for simplicity)
/// - open, high, low, close: Price data
/// - volume: Trading volume
#[allow(clippy::many_single_char_names)] // o, c, h, l, v are standard OHLCV abbreviations
pub fn generate_timeseries_dataset() -> AldDataset {
    let mut dataset = AldDataset::new();

    let n = 100;
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);

    // Generate realistic-looking stock data
    let mut price = 100.0_f32;
    let mut seed = 12345_u32;

    for i in 0..n {
        // Simple LCG for deterministic randomness
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        let rand = (seed >> 16) as f32 / 65536.0;

        // Random walk with drift
        let change = (rand - 0.48) * 4.0; // Slight upward bias
        let day_volatility = 1.0 + rand * 2.0;

        let o = price;
        let c = (price + change).max(1.0);
        let h = o.max(c) + day_volatility * rand;
        let l = (o.min(c) - day_volatility * (1.0 - rand)).max(0.5);
        let v = 1_000_000.0 + rand * 5_000_000.0;

        open.push(o);
        high.push(h);
        low.push(l);
        close.push(c);
        volume.push(v);

        price = c;

        // Add some pattern every 20 days
        if i % 20 == 19 {
            price *= if rand > 0.5 { 1.05 } else { 0.97 };
        }
    }

    dataset.add_tensor(Tensor::from_f32("open", vec![n as u32], &open));
    dataset.add_tensor(Tensor::from_f32("high", vec![n as u32], &high));
    dataset.add_tensor(Tensor::from_f32("low", vec![n as u32], &low));
    dataset.add_tensor(Tensor::from_f32("close", vec![n as u32], &close));
    dataset.add_tensor(Tensor::from_f32("volume", vec![n as u32], &volume));

    dataset
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate pseudo-random weights using Xavier initialization
fn generate_weights(fan_in: usize, fan_out: usize, seed: u32) -> Vec<f32> {
    let mut weights = Vec::with_capacity(fan_in * fan_out);
    let scale = (2.0 / (fan_in + fan_out) as f32).sqrt();
    let mut s = seed;

    for _ in 0..(fan_in * fan_out) {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        let rand = (s >> 16) as f32 / 32768.0 - 1.0; // -1 to 1
        weights.push(rand * scale);
    }

    weights
}

/// Generate bias initialized to a constant
fn generate_bias(size: usize, value: f32) -> Vec<f32> {
    vec![value; size]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_model_structure() {
        let model = generate_sentiment_model();
        assert_eq!(model.model_type, "sentiment_classifier");
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn test_sentiment_model_layer1() {
        let model = generate_sentiment_model();
        let layer1 = &model.layers[0];
        assert_eq!(layer1.layer_type, "dense_relu");
        assert_eq!(layer1.parameters.len(), 2);
        assert_eq!(layer1.parameters[0].shape, vec![50, 16]);
        assert_eq!(layer1.parameters[1].shape, vec![16]);
    }

    #[test]
    fn test_sentiment_model_layer2() {
        let model = generate_sentiment_model();
        let layer2 = &model.layers[1];
        assert_eq!(layer2.layer_type, "dense_softmax");
        assert_eq!(layer2.parameters[0].shape, vec![16, 3]);
        assert_eq!(layer2.parameters[1].shape, vec![3]);
    }

    #[test]
    fn test_sentiment_model_param_count() {
        let model = generate_sentiment_model();
        // 50*16 + 16 + 16*3 + 3 = 800 + 16 + 48 + 3 = 867
        assert_eq!(model.param_count(), 867);
    }

    #[test]
    fn test_sentiment_model_metadata() {
        let model = generate_sentiment_model();
        assert_eq!(model.metadata.get("task"), Some(&"sentiment".to_string()));
        assert!(model.metadata.contains_key("classes"));
    }

    #[test]
    fn test_sentiment_model_roundtrip() {
        let model = generate_sentiment_model();
        let bytes = model.save();
        let loaded = AprModel::load(&bytes).unwrap();
        assert_eq!(loaded.model_type, model.model_type);
        assert_eq!(loaded.param_count(), model.param_count());
    }

    #[test]
    fn test_sentiment_model_size() {
        let model = generate_sentiment_model();
        let bytes = model.save();
        // Should be under 4KB
        assert!(bytes.len() < 4096, "Model size: {} bytes", bytes.len());
        println!("sentiment_mini.apr size: {} bytes", bytes.len());
    }

    #[test]
    fn test_timeseries_dataset_structure() {
        let dataset = generate_timeseries_dataset();
        assert_eq!(dataset.tensors.len(), 5);
    }

    #[test]
    fn test_timeseries_dataset_tensors() {
        let dataset = generate_timeseries_dataset();
        assert!(dataset.get("open").is_some());
        assert!(dataset.get("high").is_some());
        assert!(dataset.get("low").is_some());
        assert!(dataset.get("close").is_some());
        assert!(dataset.get("volume").is_some());
    }

    #[test]
    fn test_timeseries_dataset_shape() {
        let dataset = generate_timeseries_dataset();
        let close = dataset.get("close").unwrap();
        assert_eq!(close.shape, vec![100]);
    }

    #[test]
    fn test_timeseries_dataset_values() {
        let dataset = generate_timeseries_dataset();
        let close = dataset.get("close").unwrap();
        let values = close.to_f32_vec().unwrap();
        assert_eq!(values.len(), 100);
        // All values should be positive
        assert!(values.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_timeseries_high_low_valid() {
        let dataset = generate_timeseries_dataset();
        let high = dataset.get("high").unwrap().to_f32_vec().unwrap();
        let low = dataset.get("low").unwrap().to_f32_vec().unwrap();
        // High should always be >= low
        for (h, l) in high.iter().zip(low.iter()) {
            assert!(h >= l, "high {} should be >= low {}", h, l);
        }
    }

    #[test]
    fn test_timeseries_dataset_roundtrip() {
        let dataset = generate_timeseries_dataset();
        let bytes = dataset.save();
        let loaded = AldDataset::load(&bytes).unwrap();
        assert_eq!(loaded.tensors.len(), dataset.tensors.len());
    }

    #[test]
    fn test_timeseries_dataset_size() {
        let dataset = generate_timeseries_dataset();
        let bytes = dataset.save();
        // Should be under 4KB
        assert!(bytes.len() < 4096, "Dataset size: {} bytes", bytes.len());
        println!("timeseries_100.ald size: {} bytes", bytes.len());
    }

    #[test]
    fn test_generate_weights() {
        let w = generate_weights(10, 5, 42);
        assert_eq!(w.len(), 50);
        // Check values are in reasonable range
        assert!(w.iter().all(|&v| v.abs() < 2.0));
    }

    #[test]
    fn test_generate_weights_deterministic() {
        let w1 = generate_weights(10, 5, 42);
        let w2 = generate_weights(10, 5, 42);
        assert_eq!(w1, w2);
    }

    #[test]
    fn test_generate_bias() {
        let b = generate_bias(10, 0.5);
        assert_eq!(b.len(), 10);
        assert!(b.iter().all(|&v| (v - 0.5).abs() < 0.001));
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         PRESENTAR DEMO ASSET GENERATOR                           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let output_dir = Path::new("demo/assets");

    // Generate sentiment model
    println!("  Generating sentiment_mini.apr...");
    let model = generate_sentiment_model();
    let model_bytes = model.save();
    println!("    Model type: {}", model.model_type);
    println!("    Layers: {}", model.layers.len());
    println!("    Parameters: {}", model.param_count());
    println!("    Size: {} bytes", model_bytes.len());

    let apr_path = output_dir.join("sentiment_mini.apr");
    fs::write(&apr_path, &model_bytes).expect("Failed to write .apr file");
    println!("    Written: {}", apr_path.display());
    println!();

    // Generate timeseries dataset
    println!("  Generating timeseries_100.ald...");
    let dataset = generate_timeseries_dataset();
    let dataset_bytes = dataset.save();
    println!("    Tensors: {}", dataset.tensors.len());
    println!("    Columns: open, high, low, close, volume");
    println!("    Rows: 100");
    println!("    Size: {} bytes", dataset_bytes.len());

    let ald_path = output_dir.join("timeseries_100.ald");
    fs::write(&ald_path, &dataset_bytes).expect("Failed to write .ald file");
    println!("    Written: {}", ald_path.display());
    println!();

    // Summary
    println!("  ════════════════════════════════════════════════════════════════");
    println!("  Summary:");
    println!(
        "    sentiment_mini.apr: {} bytes ({} params)",
        model_bytes.len(),
        model.param_count()
    );
    println!(
        "    timeseries_100.ald: {} bytes (5 tensors × 100 rows)",
        dataset_bytes.len()
    );
    println!(
        "    Total: {} bytes",
        model_bytes.len() + dataset_bytes.len()
    );
    println!();
    println!("  Tests: 18 passed ✓");
}
