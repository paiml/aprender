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
    let binding_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("provable-contracts")
        .join("contracts")
        .join("batuta")
        .join("binding.yaml");

    println!("cargo:rerun-if-changed={}", binding_path.display());

    if !binding_path.exists() {
        println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
        return;
    }

    let yaml = match std::fs::read_to_string(&binding_path) {
        Ok(s) => s,
        Err(_) => {
            println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
            return;
        }
    };

    let bindings: BindingFile = match serde_yaml_ng::from_str(&yaml) {
        Ok(b) => b,
        Err(_) => {
            println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
            return;
        }
    };

    let mut implemented = 0u32;
    let total = bindings.bindings.len() as u32;

    for b in &bindings.bindings {
        let stem = b.contract.trim_end_matches(".yaml").to_uppercase().replace('-', "_");
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

    // Phase 2: contract PRE/POST env vars
    {
        let cdir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts");
        if let Ok(es) = std::fs::read_dir(&cdir) {
            #[derive(serde::Deserialize, Default)]
            struct CY {
                #[serde(default)]
                equations: std::collections::BTreeMap<String, EY>,
            }
            #[derive(serde::Deserialize, Default)]
            struct EY {
                #[serde(default)]
                preconditions: Vec<String>,
                #[serde(default)]
                postconditions: Vec<String>,
            }
            let (mut tp, mut tq) = (0, 0);
            for e in es.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) != Some("yaml") {
                    continue;
                }
                if p.file_name().is_some_and(|n| n.to_string_lossy().contains("binding")) {
                    continue;
                }
                println!("cargo:rerun-if-changed={}", p.display());
                let s = p
                    .file_stem()
                    .and_then(|x| x.to_str())
                    .unwrap_or("x")
                    .to_uppercase()
                    .replace('-', "_");
                if let Ok(c) = std::fs::read_to_string(&p) {
                    if let Ok(y) = serde_yaml_ng::from_str::<CY>(&c) {
                        for (n, eq) in &y.equations {
                            let k =
                                format!("CONTRACT_{}_{}", s, n.to_uppercase().replace('-', "_"));
                            if !eq.preconditions.is_empty() {
                                println!(
                                    "cargo:rustc-env={k}_PRE_COUNT={}",
                                    eq.preconditions.len()
                                );
                                for (i, v) in eq.preconditions.iter().enumerate() {
                                    println!("cargo:rustc-env={k}_PRE_{i}={v}");
                                }
                                tp += eq.preconditions.len();
                            }
                            if !eq.postconditions.is_empty() {
                                println!(
                                    "cargo:rustc-env={k}_POST_COUNT={}",
                                    eq.postconditions.len()
                                );
                                for (i, v) in eq.postconditions.iter().enumerate() {
                                    println!("cargo:rustc-env={k}_POST_{i}={v}");
                                }
                                tq += eq.postconditions.len();
                            }
                        }
                    }
                }
            }
            println!("cargo:warning=[contract] Assertions: {tp} preconditions, {tq} postconditions from YAML");
        }
    }
}
