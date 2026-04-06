use std::path::Path;

use provable_contracts::binding::parse_binding;
use provable_contracts::probar_gen::{generate_probar_tests, generate_wired_probar_tests};
use provable_contracts::schema::parse_contract;

pub fn run(path: &Path, binding_path: Option<&Path>) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;

    if let Some(bp) = binding_path {
        let binding = parse_binding(bp)?;
        let contract_file = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        print!(
            "{}",
            generate_wired_probar_tests(&contract, contract_file, &binding)
        );
    } else {
        print!("{}", generate_probar_tests(&contract));
    }

    Ok(())
}
