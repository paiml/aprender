//! `pv extract-pytorch` — extract kernels from `PyTorch` source.

use provable_contracts::extract;
use std::path::Path;

pub fn run(target: &str, output: Option<&Path>) -> Result<(), Box<dyn std::error::Error>> {
    println!("pv extract-pytorch");
    println!("==================\n");
    println!("Source: {target}\n");

    let kernel = extract::extract_from_pytorch(target)?;

    println!("Function: {}", kernel.function_name);
    println!("Arguments: {}", kernel.arguments.len());
    println!("Equations: {}", kernel.equations.len());
    for eq in &kernel.equations {
        println!("  {}: {}", eq.name, eq.formula);
        println!("    pre:  {:?}", eq.preconditions);
        println!("    post: {:?}", eq.postconditions);
    }

    let yaml = extract::kernel_to_yaml(&kernel);

    if let Some(out) = output {
        std::fs::write(out, &yaml)?;
        println!("\nContract written to: {}", out.display());
    } else {
        let default_path = format!(
            "contracts/{}-v1.yaml",
            kernel.function_name.replace('_', "-")
        );
        let p = Path::new(&default_path);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&default_path, &yaml)?;
        println!("\nContract written to: {default_path}");
    }

    Ok(())
}
