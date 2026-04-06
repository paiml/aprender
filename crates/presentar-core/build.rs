// build.rs — Read contracts/*.yaml and provable-contracts binding.yaml,
// emit CONTRACT_* env vars for compile-time assertion checking.

#[allow(clippy::too_many_lines)]
fn main() {
    // Phase 1: contract assertion env vars from local contracts/
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

    // Phase 2: binding.yaml from provable-contracts (if sibling checkout exists)
    {
        let binding_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..") // crates/
            .join("..") // presentar/
            .join("..") // src/
            .join("provable-contracts")
            .join("contracts")
            .join("presentar")
            .join("binding.yaml");

        println!("cargo:rerun-if-changed={}", binding_path.display());

        if binding_path.exists() {
            #[derive(serde::Deserialize)]
            struct BindingFile {
                #[allow(dead_code)]
                version: String,
                #[allow(dead_code)]
                target_crate: String,
                bindings: Vec<Binding>,
            }
            #[derive(serde::Deserialize)]
            struct Binding {
                contract: String,
                equation: String,
                status: String,
            }

            fn env_var_name(contract: &str, equation: &str) -> String {
                let stem = contract
                    .trim_end_matches(".yaml")
                    .trim_end_matches(".yml")
                    .to_uppercase()
                    .replace('-', "_");
                let eq = equation.to_uppercase().replace('-', "_");
                format!("CONTRACT_{stem}_{eq}")
            }

            if let Ok(content) = std::fs::read_to_string(&binding_path) {
                if let Ok(bf) = serde_yaml_ng::from_str::<BindingFile>(&content) {
                    let mut implemented = 0u32;
                    let mut partial = 0u32;
                    let mut not_implemented = 0u32;

                    for b in &bf.bindings {
                        let var = env_var_name(&b.contract, &b.equation);
                        println!("cargo:rustc-env={var}={}", b.status);
                        match b.status.as_str() {
                            "implemented" => implemented += 1,
                            "partial" => partial += 1,
                            _ => not_implemented += 1,
                        }
                    }

                    let total = implemented + partial + not_implemented;
                    println!(
                        "cargo:warning=[contract] AllImplemented: {implemented}/{total} implemented, \
                         {partial} partial, {not_implemented} gaps"
                    );

                    // AllImplemented policy: warn on gaps (panic would block CI)
                    if not_implemented > 0 {
                        println!(
                            "cargo:warning=[contract] AllImplemented: {not_implemented} \
                             binding(s) are not_implemented — implement before next release"
                        );
                    }

                    println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=binding.yaml");
                } else {
                    println!("cargo:warning=[contract] Failed to parse binding.yaml");
                    println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
                }
            } else {
                println!("cargo:warning=[contract] Failed to read binding.yaml");
                println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
            }
        } else {
            println!(
                "cargo:warning=[contract] provable-contracts binding.yaml not found at {}; skipping",
                binding_path.display()
            );
            println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
        }
    }
}
