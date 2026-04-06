use std::path::Path;

use provable_contracts::scaffold::{
    generate_contract_tests, generate_standalone_trait, generate_trait,
};
use provable_contracts::schema::parse_contract;

pub fn run(
    path: &Path,
    standalone_trait: bool,
    output: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;

    if standalone_trait {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let code = generate_standalone_trait(&contract, stem);

        if let Some(out_path) = output {
            std::fs::write(out_path, &code)?;
            println!("Generated trait: {}", out_path.display());
        } else {
            print!("{code}");
        }
        return Ok(());
    }

    println!("// === Trait Definition ===\n");
    print!("{}", generate_trait(&contract));
    println!("\n// === Contract Tests ===\n");
    print!("{}", generate_contract_tests(&contract));

    Ok(())
}
