//! Demo: Display .apr model and .ald dataset files
//!
//! Run with: cargo run --example `apr_ald_display`

use presentar_widgets::{load_ald_as_card, load_apr_as_card, AldDatasetExt, AprModelExt};
use presentar_yaml::formats::{AldDataset, AprModel, ModelLayer, Tensor};

fn main() {
    println!("=== Presentar .apr/.ald Display Demo ===\n");

    // Create a sample .apr model
    let model = create_sample_model();
    let model_bytes = model.save();
    println!("Created sample .apr model ({} bytes)\n", model_bytes.len());

    // Load and display as ModelCard
    let model_card = load_apr_as_card(&model_bytes).expect("load apr");
    println!("ModelCard from .apr:");
    println!("  Name: {}", model_card.get_name());
    println!("  Version: {}", model_card.get_version());
    println!("  Framework: {:?}", model_card.get_framework());
    println!("  Parameters: {:?}", model_card.get_parameters());
    println!("  Status: {:?}", model_card.get_status());
    println!("  Tags: {:?}", model_card.get_tags());
    println!();

    // Create a sample .ald dataset
    let dataset = create_sample_dataset();
    let dataset_bytes = dataset.save();
    println!(
        "Created sample .ald dataset ({} bytes)\n",
        dataset_bytes.len()
    );

    // Load and display as DataCard
    let data_card = load_ald_as_card(&dataset_bytes, "mnist_train").expect("load ald");
    println!("DataCard from .ald:");
    println!("  Name: {}", data_card.get_name());
    println!("  Columns: {}", data_card.column_count());
    println!("  Description: {:?}", data_card.get_description());
    println!("  Tags: {:?}", data_card.get_tags());
    println!();

    // Using extension traits directly
    println!("=== Using Extension Traits ===\n");

    let card = model.to_model_card();
    println!("Direct conversion: {}", card.get_name());

    let card = dataset.to_data_card("custom_name");
    println!("Direct conversion: {}", card.get_name());

    println!("\nDone!");
}

fn create_sample_model() -> AprModel {
    let mut model = AprModel::new("MLP");

    // Add layers
    model.layers.push(ModelLayer {
        layer_type: "dense".to_string(),
        parameters: vec![
            Tensor::from_f32("weights", vec![784, 256], &vec![0.0; 784 * 256]),
            Tensor::from_f32("bias", vec![256], &vec![0.0; 256]),
        ],
    });

    model.layers.push(ModelLayer {
        layer_type: "relu".to_string(),
        parameters: vec![],
    });

    model.layers.push(ModelLayer {
        layer_type: "dense".to_string(),
        parameters: vec![
            Tensor::from_f32("weights", vec![256, 10], &vec![0.0; 256 * 10]),
            Tensor::from_f32("bias", vec![10], &[0.0; 10]),
        ],
    });

    // Add metadata
    model
        .metadata
        .insert("accuracy".to_string(), "0.98".to_string());
    model
        .metadata
        .insert("task".to_string(), "classification".to_string());
    model
        .metadata
        .insert("dataset".to_string(), "MNIST".to_string());
    model
        .metadata
        .insert("author".to_string(), "PAIML".to_string());

    model
}

fn create_sample_dataset() -> AldDataset {
    let mut dataset = AldDataset::new();

    // Add training images (simulated)
    dataset.add_tensor(Tensor::from_f32(
        "images",
        vec![60000, 28, 28],
        &vec![0.0; 60000 * 28 * 28],
    ));

    // Add labels
    dataset.add_tensor(Tensor::from_f32("labels", vec![60000], &vec![0.0; 60000]));

    dataset
}
