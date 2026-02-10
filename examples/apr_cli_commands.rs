//! APR CLI Commands Demo
//!
//! Demonstrates creating test models and using the apr-cli commands.
//! This example creates model files that work with all 24 apr-cli commands.
//!
//! Toyota Way Alignment:
//! - **Genchi Genbutsu**: Go and see - inspect actual model data
//! - **Jidoka**: Built-in quality - validate models automatically
//! - **Visualization**: Make problems visible with trace and debug
//!
//! Run with: `cargo run --example apr_cli_commands`
//!
//! After running, use the apr CLI on the generated files:
//! ```bash
//! cargo build -p apr-cli
//! ./target/debug/apr inspect /tmp/apr_cli_demo/demo_model.apr
//! ./target/debug/apr validate /tmp/apr_cli_demo/demo_model.apr --quality
//! ./target/debug/apr debug /tmp/apr_cli_demo/demo_model.apr --drama
//! ./target/debug/apr tensors /tmp/apr_cli_demo/demo_model.apr --stats
//! ./target/debug/apr trace /tmp/apr_cli_demo/demo_model.apr --verbose
//! ./target/debug/apr diff /tmp/apr_cli_demo/demo_model.apr /tmp/apr_cli_demo/demo_model_v2.apr
//! ./target/debug/apr probar /tmp/apr_cli_demo/demo_model.apr -o /tmp/apr_cli_demo/probar
//! ./target/debug/apr explain E002
//!
//! # Inference commands (requires --features inference):
//! cargo build -p apr-cli --features inference
//! ./target/debug/apr run /tmp/apr_cli_demo/demo_model.apr --input "[1.0, 2.0]"
//! ./target/debug/apr serve /tmp/apr_cli_demo/demo_model.apr --port 8080
//! ```

use aprender::serialization::apr::AprWriter;
use serde_json::json;
use std::fs;
use std::path::Path;

fn main() -> Result<(), String> {
    println!("=== APR CLI Commands Demo ===\n");

    // Create output directory
    let demo_dir = Path::new("/tmp/apr_cli_demo");
    fs::create_dir_all(demo_dir).map_err(|e| e.to_string())?;

    // Part 1: Create a demo model
    println!("--- Part 1: Creating Demo Model ---\n");
    let model_path = create_demo_model(demo_dir)?;
    println!("Created: {}\n", model_path.display());

    // Part 2: Create a second model for diff comparison
    println!("--- Part 2: Creating Second Model (for diff) ---\n");
    let model_v2_path = create_demo_model_v2(demo_dir)?;
    println!("Created: {}\n", model_v2_path.display());

    // Part 3: Show CLI commands
    println!("--- Part 3: CLI Commands Reference ---\n");
    print_cli_commands(&model_path, &model_v2_path);

    println!("\n=== Demo Complete! ===");
    println!("\nModel files created in: {}", demo_dir.display());
    println!("Build the CLI with: cargo build -p apr-cli");
    println!("Then run the commands shown above.");

    Ok(())
}

fn create_demo_model(dir: &Path) -> Result<std::path::PathBuf, String> {
    let mut writer = AprWriter::new();

    // Add model metadata
    writer.set_metadata("model_type", json!("linear_regression"));
    writer.set_metadata("model_name", json!("Demo Linear Regression"));
    writer.set_metadata("description", json!("A demo model for CLI testing"));
    writer.set_metadata("n_features", json!(2));
    writer.set_metadata("n_outputs", json!(1));
    writer.set_metadata("framework", json!("aprender"));
    writer.set_metadata("framework_version", json!(env!("CARGO_PKG_VERSION")));

    // Add hyperparameters
    writer.set_metadata(
        "hyperparameters",
        json!({
            "n_layer": 4,
            "n_embd": 128,
            "learning_rate": 0.01
        }),
    );

    // Add training info
    writer.set_metadata(
        "training",
        json!({
            "dataset": "synthetic",
            "n_samples": 1000,
            "n_epochs": 100,
            "final_loss": 0.0234
        }),
    );

    // Add tensors (simulating a small model)
    println!("  Adding tensors...");

    // Weights tensor
    let weights: Vec<f32> = vec![1.5, 0.8];
    writer.add_tensor_f32("weights", vec![2, 1], &weights);

    // Bias tensor
    let bias: Vec<f32> = vec![0.5];
    writer.add_tensor_f32("bias", vec![1], &bias);

    // Embedding layer (to make it more interesting for trace)
    let embedding: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    writer.add_tensor_f32("embedding", vec![128], &embedding);

    // Layer norm weights
    let ln_weight: Vec<f32> = vec![1.0; 128];
    writer.add_tensor_f32("layer_norm.weight", vec![128], &ln_weight);

    // Write to file
    let path = dir.join("demo_model.apr");
    let bytes = writer.to_bytes()?;
    fs::write(&path, &bytes).map_err(|e| e.to_string())?;

    println!("  Model type: Linear Regression");
    println!("  Tensors: 4");
    println!("  Size: {} bytes", bytes.len());

    Ok(path)
}

fn create_demo_model_v2(dir: &Path) -> Result<std::path::PathBuf, String> {
    let mut writer = AprWriter::new();

    // Slightly different metadata
    writer.set_metadata("model_type", json!("linear_regression"));
    writer.set_metadata("model_name", json!("Demo Linear Regression v2"));
    writer.set_metadata("description", json!("Updated model with more training"));
    writer.set_metadata("n_features", json!(2));
    writer.set_metadata("n_outputs", json!(1));
    writer.set_metadata("framework", json!("aprender"));
    writer.set_metadata("framework_version", json!(env!("CARGO_PKG_VERSION")));

    // Different hyperparameters
    writer.set_metadata(
        "hyperparameters",
        json!({
            "n_layer": 4,
            "n_embd": 128,
            "learning_rate": 0.005  // Changed
        }),
    );

    // More training
    writer.set_metadata(
        "training",
        json!({
            "dataset": "synthetic_extended",  // Changed
            "n_samples": 2000,                // Changed
            "n_epochs": 200,                  // Changed
            "final_loss": 0.0156              // Improved
        }),
    );

    // Slightly different weights (simulating retraining)
    let weights: Vec<f32> = vec![1.52, 0.79]; // Slightly different
    writer.add_tensor_f32("weights", vec![2, 1], &weights);

    let bias: Vec<f32> = vec![0.48]; // Slightly different
    writer.add_tensor_f32("bias", vec![1], &bias);

    let embedding: Vec<f32> = (0..128).map(|i| (i as f32) * 0.0101).collect();
    writer.add_tensor_f32("embedding", vec![128], &embedding);

    let ln_weight: Vec<f32> = vec![1.0; 128];
    writer.add_tensor_f32("layer_norm.weight", vec![128], &ln_weight);

    let path = dir.join("demo_model_v2.apr");
    let bytes = writer.to_bytes()?;
    fs::write(&path, &bytes).map_err(|e| e.to_string())?;

    println!("  Model type: Linear Regression v2");
    println!("  Tensors: 4");
    println!("  Size: {} bytes", bytes.len());

    Ok(path)
}

fn print_cli_commands(model_path: &Path, model_v2_path: &Path) {
    let model = model_path.display();
    let model_v2 = model_v2_path.display();
    let demo_dir = model_path
        .parent()
        .expect("model path should have a parent directory")
        .display();

    println!("Build the CLI first:");
    println!("  cargo build -p apr-cli\n");
    println!("For inference commands (run, serve):");
    println!("  cargo build -p apr-cli --features inference\n");

    println!("=== 24 APR CLI Commands ===\n");

    println!("--- Model Inspection ---\n");

    println!("1. INSPECT - View model metadata:");
    println!("   ./target/debug/apr inspect {model}");
    println!("   ./target/debug/apr inspect {model} --json");
    println!("   ./target/debug/apr inspect {model} --weights\n");

    println!("2. TENSORS - List tensor info:");
    println!("   ./target/debug/apr tensors {model}");
    println!("   ./target/debug/apr tensors {model} --stats");
    println!("   ./target/debug/apr tensors {model} --json\n");

    println!("3. TRACE - Layer-by-layer analysis:");
    println!("   ./target/debug/apr trace {model}");
    println!("   ./target/debug/apr trace {model} --verbose");
    println!("   ./target/debug/apr trace {model} --json\n");

    println!("4. DEBUG - Debug output:");
    println!("   ./target/debug/apr debug {model}");
    println!("   ./target/debug/apr debug {model} --drama");
    println!("   ./target/debug/apr debug {model} --hex --limit 64\n");

    println!("--- Quality & Validation ---\n");

    println!("5. VALIDATE - Check model integrity (100-point QA):");
    println!("   ./target/debug/apr validate {model}");
    println!("   ./target/debug/apr validate {model} --quality");
    println!("   ./target/debug/apr validate {model} --strict\n");

    println!("6. LINT - Best practices check:");
    println!("   ./target/debug/apr lint {model}\n");

    println!("7. DIFF - Compare two models:");
    println!("   ./target/debug/apr diff {model} {model_v2}");
    println!("   ./target/debug/apr diff {model} {model_v2} --json\n");

    println!("--- Model Transformation ---\n");

    println!("8. CONVERT - Quantization/optimization:");
    println!("   ./target/debug/apr convert {model} --quantize int8 -o {demo_dir}/model-int8.apr");
    println!(
        "   ./target/debug/apr convert {model} --quantize fp16 -o {demo_dir}/model-fp16.apr\n"
    );

    println!("9. EXPORT - Export to other formats:");
    println!(
        "   ./target/debug/apr export {model} --format safetensors -o {demo_dir}/model.safetensors"
    );
    println!("   ./target/debug/apr export {model} --format gguf -o {demo_dir}/model.gguf\n");

    println!("10. MERGE - Merge models:");
    println!("    ./target/debug/apr merge {model} {model_v2} --strategy average -o {demo_dir}/merged.apr");
    println!("    ./target/debug/apr merge {model} {model_v2} --strategy weighted -o {demo_dir}/merged.apr\n");

    println!("--- Import & Interop ---\n");

    println!("11. IMPORT - Import external models:");
    println!("    ./target/debug/apr import ./external.safetensors -o imported.apr");
    println!("    ./target/debug/apr import hf://org/repo -o model.apr --arch whisper\n");

    println!("--- Testing & Regression ---\n");

    println!("12. CANARY - Regression testing:");
    println!("    ./target/debug/apr canary create {model} --input ref.wav --output {demo_dir}/canary.json");
    println!("    ./target/debug/apr canary check {model_v2} --canary {demo_dir}/canary.json\n");

    println!("13. PROBAR - Visual regression testing export:");
    println!("    ./target/debug/apr probar {model} -o {demo_dir}/probar_output");
    println!("    ./target/debug/apr probar {model} -o {demo_dir}/probar_output --format json\n");

    println!("--- Help & Documentation ---\n");

    println!("14. EXPLAIN - Get explanations:");
    println!("    ./target/debug/apr explain E002");
    println!("    ./target/debug/apr explain --tensor encoder.conv1.weight");
    println!("    ./target/debug/apr explain --file {model}\n");

    println!("--- Interactive ---\n");

    println!("15. TUI - Interactive terminal UI:");
    println!("    ./target/debug/apr tui {model}");
    println!("    Tabs: Overview [1], Tensors [2], Stats [3], Help [?]");
    println!("    Navigation: j/k or arrows, Tab to switch, q to quit\n");

    println!("--- Inference (requires --features inference) ---\n");

    println!("16. RUN - Run inference on a model:");
    println!("    ./target/debug/apr run {model} --input \"[1.0, 2.0]\"");
    println!("    ./target/debug/apr run {model} --input \"1.0,2.0\"");
    println!("    ./target/debug/apr run {model} --input \"[1.0, 2.0]\" --json\n");

    println!("17. SERVE - Start inference server:");
    println!("    ./target/debug/apr serve {model} --port 8080");
    println!("    ./target/debug/apr serve {model} --host 0.0.0.0 --port 3000");
    println!("    # Then: curl http://localhost:8080/health");
    println!(
        "    # Then: curl -X POST http://localhost:8080/predict -d '{{\"input\": [1.0, 2.0]}}'\n"
    );

    println!("18. CHAT - Interactive chat (LLM models):");
    println!("    ./target/debug/apr chat model.gguf");
    println!("    ./target/debug/apr chat model.gguf --system \"You are a helpful assistant\"\n");

    println!("--- HuggingFace Hub ---\n");

    println!("19. PUBLISH - Push model to HuggingFace Hub:");
    println!("    ./target/debug/apr publish {demo_dir}/ org/model-name");
    println!("    ./target/debug/apr publish {demo_dir}/ org/model-name --dry-run");
    println!(
        "    ./target/debug/apr publish {demo_dir}/ org/model-name --license mit --tags rust,ml\n"
    );

    println!("20. PULL - Download model from HuggingFace Hub:");
    println!("    ./target/debug/apr pull hf://org/repo-name -o ./models/");
    println!(
        "    ./target/debug/apr pull hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF -o ./models/\n"
    );

    println!("--- Benchmarking & QA ---\n");

    println!("21. QA - Run falsifiable QA checklist:");
    println!("    ./target/debug/apr qa model.gguf");
    println!("    ./target/debug/apr qa model.gguf --assert-tps 100");
    println!("    ./target/debug/apr qa model.gguf --json\n");

    println!("22. SHOWCASE - Performance benchmark demo:");
    println!("    ./target/debug/apr showcase model.gguf");
    println!("    ./target/debug/apr showcase model.gguf --warmup 3 --iterations 10\n");

    println!("23. PROFILE - Deep performance profiling:");
    println!("    ./target/debug/apr profile model.gguf");
    println!("    ./target/debug/apr profile model.gguf --roofline\n");

    println!("24. BENCH - Run benchmarks:");
    println!("    ./target/debug/apr bench model.gguf");
    println!("    ./target/debug/apr bench model.gguf --iterations 100\n");
}
