// build.rs — provable-contracts binding + PRE/POST enforcement (L1)
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

#[allow(clippy::too_many_lines)]
fn main() {
    // Phase 1: binding env vars
    let contracts_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("contracts");

    // Scan all binding.yaml files in subdirectories
    let mut total_bindings = 0u32;
    let mut total_implemented = 0u32;
    let mut found_any = false;

    if let Ok(entries) = std::fs::read_dir(&contracts_dir) {
        for entry in entries.flatten() {
            let binding_path = entry.path().join("binding.yaml");
            if !binding_path.exists() {
                continue;
            }
            println!("cargo:rerun-if-changed={}", binding_path.display());

            let Ok(yaml) = std::fs::read_to_string(&binding_path) else {
                continue;
            };

            let bindings: BindingFile = match serde_yaml::from_str(&yaml) {
                Ok(b) => b,
                Err(_) => continue,
            };

            found_any = true;
            for b in &bindings.bindings {
                let stem = b
                    .contract
                    .trim_end_matches(".yaml")
                    .to_uppercase()
                    .replace('-', "_");
                let eq = b.equation.to_uppercase().replace('-', "_");
                let var = format!("CONTRACT_{stem}_{eq}");
                println!("cargo:rustc-env={var}={}", b.status);
                total_bindings += 1;
                if b.status == "implemented" {
                    total_implemented += 1;
                }
            }
        }
    }

    if found_any {
        println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=binding.yaml");
        println!("cargo:rustc-env=CONTRACT_TOTAL={total_bindings}");
        println!("cargo:rustc-env=CONTRACT_IMPLEMENTED={total_implemented}");
    } else {
        println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
    }

    // Phase 2: contract PRE/POST env vars
    {
        let cdir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("contracts");
        if let Ok(es) = std::fs::read_dir(&cdir) {
            #[derive(Deserialize, Default)]
            struct CY {
                #[serde(default)]
                equations: std::collections::BTreeMap<String, EY>,
            }
            #[derive(Deserialize, Default)]
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
                if p.file_name()
                    .is_some_and(|n| n.to_string_lossy().contains("binding"))
                {
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
                    if let Ok(y) = serde_yaml::from_str::<CY>(&c) {
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
            println!(
                "cargo:warning=[contract] Assertions: {tp} preconditions, {tq} postconditions from YAML"
            );
        }
    }
}
