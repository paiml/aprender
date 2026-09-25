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
    // #4219: stamp APR_GIT_SHA for `--version` before anything can return early.
    build_sha::emit();

    // From crates/presentar-cli/ -> ../../.. -> src/ -> provable-contracts/
    let binding_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts/presentar/binding.yaml");

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
        let var = provable_contracts::build_helper::env_key(
            b.contract.trim_end_matches(".yaml"),
            &b.equation,
        );
        println!("cargo:rustc-env={var}={}", b.status);
        if b.status == "implemented" {
            implemented += 1;
        }
    }

    println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=binding.yaml");
    println!("cargo:rustc-env=CONTRACT_TOTAL={total}");
    println!("cargo:rustc-env=CONTRACT_IMPLEMENTED={implemented}");
}
