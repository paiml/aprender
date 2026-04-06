// build.rs — provable-contracts binding enforcement (L1)
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct BindingFile {
    #[allow(dead_code)]
    version: String,
    #[allow(dead_code)]
    target_crate: String,
    bindings: Vec<Binding>,
}

#[derive(Deserialize)]
struct Binding {
    contract: String,
    equation: String,
    status: String,
}

fn main() {
    // From crates/presentar-cli/ -> ../../.. -> src/ -> provable-contracts/
    let binding_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("provable-contracts")
        .join("contracts")
        .join("presentar")
        .join("binding.yaml");

    println!("cargo:rerun-if-changed={}", binding_path.display());

    if !binding_path.exists() {
        println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
        return;
    }

    let Ok(yaml) = std::fs::read_to_string(&binding_path) else {
        println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
        return;
    };

    let Ok(bindings) = serde_yaml_ng::from_str::<BindingFile>(&yaml) else {
        println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
        return;
    };

    let mut implemented = 0u32;
    let total = bindings.bindings.len() as u32;

    for b in &bindings.bindings {
        let stem = b
            .contract
            .trim_end_matches(".yaml")
            .to_uppercase()
            .replace('-', "_");
        let eq = b.equation.to_uppercase().replace('-', "_");
        let var = format!("CONTRACT_{stem}_{eq}");
        println!("cargo:rustc-env={var}={}", b.status);
        if b.status == "implemented" {
            implemented += 1;
        }
    }

    println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=binding.yaml");
    println!("cargo:rustc-env=CONTRACT_TOTAL={total}");
    println!("cargo:rustc-env=CONTRACT_IMPLEMENTED={implemented}");
}
