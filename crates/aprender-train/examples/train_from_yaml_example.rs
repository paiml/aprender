//! Example: Declarative Training from YAML Configuration
//!
//! Demonstrates the Ludwig-style declarative API for model training.

use entrenar::config::train_from_yaml;
use std::fs;

fn main() {
    println!("=== Train from YAML Example ===\n");

    // Create output directory
    fs::create_dir_all("./output").expect("Failed to create output directory");

    // Run training from YAML config
    println!("Loading configuration from examples/config.yaml...\n");

    match train_from_yaml("examples/config.yaml") {
        Ok(()) => {
            println!("=== Training Successful ===");
            println!("\nTrained model saved to: ./output/final_model.json");
        }
        Err(e) => {
            eprintln!("Training failed: {e}");
            std::process::exit(1);
        }
    }
}
