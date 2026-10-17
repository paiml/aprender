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

/// The #5205 S4 reader options (same as `src/yaml.rs`; build scripts cannot use the crate).
fn parse_yaml<T: serde::de::DeserializeOwned>(s: &str) -> Result<T, serde_saphyr::Error> {
    serde_saphyr::from_str_with_options(
        s,
        serde_saphyr::options! {
            strict_booleans: true,
            merge_keys: serde_saphyr::MergeKeyPolicy::AsOrdinary,
            emit_comments: false,
            reject_non_finite_typeless_float: false,
        },
    )
}

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

fn main() {
    // #4219: stamp APR_GIT_SHA for `--version` before anything can return early.
    build_sha::emit();

    let binding_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/batuta/binding.yaml");

    println!("cargo:rerun-if-changed={}", binding_path.display());

    let Some(bindings) = load_bindings(&binding_path) else {
        println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
        return;
    };

    emit_bindings(&bindings);

    // Phase 2: contract PRE/POST env vars
    emit_contract_assertions(&Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts"));
}

/// Read and parse the binding file; `None` when it is absent, unreadable or invalid.
fn load_bindings(binding_path: &Path) -> Option<BindingFile> {
    if !binding_path.exists() {
        return None;
    }
    let yaml = std::fs::read_to_string(binding_path).ok()?;
    parse_yaml(&yaml).ok()
}

/// Emit one `CONTRACT_*` env var per binding plus the totals.
fn emit_bindings(bindings: &BindingFile) {
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

/// Emit `_PRE_*`/`_POST_*` env vars for every contract YAML in `cdir`.
fn emit_contract_assertions(cdir: &Path) {
    let Ok(es) = std::fs::read_dir(cdir) else {
        return;
    };
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
        let s =
            p.file_stem().and_then(|x| x.to_str()).unwrap_or("x").to_uppercase().replace('-', "_");
        let Some(y) = std::fs::read_to_string(&p).ok().and_then(|c| parse_yaml::<CY>(&c).ok())
        else {
            continue;
        };
        for (n, eq) in &y.equations {
            let k = provable_contracts::build_helper::env_key(&s, n);
            tp += emit_conditions(&k, "PRE", &eq.preconditions);
            tq += emit_conditions(&k, "POST", &eq.postconditions);
        }
    }
    println!(
        "cargo:warning=[contract] Assertions: {tp} preconditions, {tq} postconditions from YAML"
    );
}

/// Emit `{k}_{kind}_COUNT` and `{k}_{kind}_{i}` for a non-empty list; returns its length.
fn emit_conditions(k: &str, kind: &str, conds: &[String]) -> usize {
    if conds.is_empty() {
        return 0;
    }
    println!("cargo:rustc-env={k}_{kind}_COUNT={}", conds.len());
    for (i, v) in conds.iter().enumerate() {
        println!("cargo:rustc-env={k}_{kind}_{i}={v}");
    }
    conds.len()
}
