//! Example: Extract kernel equations from `PyTorch` source
//!
//! Demonstrates the full pipeline:
//! `PyTorch` source → YAML contract → Lean theorem → Rust kernel
//!
//! Run with: `cargo run --example extract_pytorch`

fn main() {
    println!("pv extract-pytorch — PyTorch → Provable Contract Pipeline");
    println!("=========================================================\n");

    let pytorch_path = "/home/noah/src/pytorch/torch/nn/functional.py";
    if !std::path::Path::new(pytorch_path).exists() {
        println!("PyTorch source not found. Showing example output:\n");
        show_example();
        return;
    }

    let target = format!("{pytorch_path}::softmax");
    match provable_contracts::extract::extract_from_pytorch(&target) {
        Ok(kernel) => {
            println!("Extracted: {}", kernel.function_name);
            println!("Arguments: {}", kernel.arguments.len());
            println!("Equations: {}\n", kernel.equations.len());

            for eq in &kernel.equations {
                println!("  {}: {}", eq.name, eq.formula);
                println!("    pre:  {:?}", eq.preconditions);
                println!("    post: {:?}", eq.postconditions);
            }

            let yaml = provable_contracts::extract::kernel_to_yaml(&kernel);
            println!("\n=== Generated YAML (first 20 lines) ===\n");
            for line in yaml.lines().take(20) {
                println!("{line}");
            }
        }
        Err(e) => eprintln!("Error: {e}"),
    }

    println!("\n=== Pipeline ===");
    println!("1. extract-pytorch  → YAML contract");
    println!("2. lean-gen         → Lean theorem stub");
    println!("3. prove            → no sorry");
    println!("4. scaffold         → Rust trait + tests");
    println!("5. implement        → kernel code");
    println!("6. pv lint (5 gates)→ all pass");
    println!("7. pmat comply      → COMPLIANT");
}

fn show_example() {
    println!("Extracted: softmax");
    println!("  Formula: σ(x)_i = exp(x_i) / Σ_j exp(x_j)");
    println!("  Pre:  [\"!input.is_empty()\", \"dim < input.ndim()\"]");
    println!("  Post: [\"ret in [0,1]\", \"sum(ret) ≈ 1.0\"]");
}
