// Contract: apr-version-traceability-v1 F-VERSION-001..004 (paiml/aprender#597, #1862).
// Resolve APR_GIT_SHA with a multi-source fallback hierarchy:
//   1. APR_GIT_SHA_OVERRIDE env var (CI/release hook)
//   2. .cargo_vcs_info.json `git.sha1` (crates.io / packaged builds, #4110)
//   3. git rev-parse --short HEAD, retried with the checkout marked safe.directory (#4110)
//   4. committed .git-sha file
//   5. "v{CARGO_PKG_VERSION}+no-git" (informative fallback — never bare "unknown")
//
// The resolution and its rerun-if-changed triggers live in the shared
// `aprender-build-sha` crate (#4219) so every workspace [[bin]] stamps the same SHA.
//
// #4370: this is also apr-cli's `#[contract]` producer. It reads the binding
// registry `contracts/aprender/binding.yaml` (whose `apr_cli::` rows are this
// crate's), emits `CONTRACT_<C>_<E>` for every row whose equation exists in
// `contracts/<C>.yaml`, the pre/postconditions of every registered contract,
// and the `CONTRACT_BINDING_SOURCE=binding.yaml` sentinel. With the sentinel
// set, a site naming a contract or equation the registry does not vouch for
// is a compile error (#4368).
use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

#[derive(Deserialize)]
struct BindingFile {
    bindings: Vec<Binding>,
}

#[derive(Deserialize)]
struct Binding {
    contract: String,
    equation: String,
    status: String,
}

#[derive(Deserialize, Default)]
struct ContractYaml {
    #[serde(default)]
    equations: BTreeMap<String, EquationYaml>,
}

#[derive(Deserialize, Default)]
struct EquationYaml {
    #[serde(default)]
    preconditions: Vec<String>,
    #[serde(default)]
    postconditions: Vec<String>,
}

fn main() {
    build_sha::emit();
    emit_contract_registry();
}

fn emit_contract_registry() {
    let contracts = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
    let registry = contracts.join("aprender/binding.yaml");
    println!("cargo:rerun-if-changed={}", registry.display());
    let parsed = std::fs::read_to_string(&registry)
        .map_err(|e| e.to_string())
        .and_then(|y| serde_yaml::from_str::<BindingFile>(&y).map_err(|e| e.to_string()));
    let bindings = match parsed {
        Ok(b) => b.bindings,
        Err(e) => {
            // A packaged (crates.io) build has no contracts/ tree. Without the
            // sentinel the sites expand to nothing, as before this producer.
            println!(
                "cargo:warning=[contract] no binding registry at {} ({e}); #[contract] sites are unchecked",
                registry.display()
            );
            println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
            return;
        }
    };

    let mut loaded: BTreeMap<String, Option<ContractYaml>> = BTreeMap::new();
    let mut dropped = Vec::new();
    for b in &bindings {
        let stem = b.contract.trim_end_matches(".yaml");
        let contract = loaded
            .entry(stem.to_string())
            .or_insert_with(|| load_contract(&contracts, stem));
        // A row whose equation is not in the tree must not vouch for a site.
        if contract
            .as_ref()
            .is_some_and(|c| c.equations.contains_key(&b.equation))
        {
            let key = provable_contracts::build_helper::env_key(stem, &b.equation);
            println!("cargo:rustc-env={key}={}", b.status);
        } else {
            dropped.push(format!("{stem}/{}", b.equation));
        }
    }
    if !dropped.is_empty() {
        println!(
            "cargo:warning=[contract] {} registry row(s) name no equation in contracts/ and bind nothing: {}",
            dropped.len(),
            dropped.join(", ")
        );
    }

    let (mut pre, mut post) = (0usize, 0usize);
    for (stem, contract) in &loaded {
        let Some(contract) = contract else { continue };
        for (eq, e) in &contract.equations {
            let key = provable_contracts::build_helper::env_key(stem, eq);
            emit_conditions(&key, "PRE", &e.preconditions);
            emit_conditions(&key, "POST", &e.postconditions);
            pre += e.preconditions.len();
            post += e.postconditions.len();
        }
    }
    println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=binding.yaml");
    println!(
        "cargo:warning=[contract] apr-cli registry: {} rows, {pre} preconditions, {post} postconditions",
        bindings.len() - dropped.len()
    );
}

fn load_contract(contracts: &Path, stem: &str) -> Option<ContractYaml> {
    let path = contracts.join(format!("{stem}.yaml"));
    println!("cargo:rerun-if-changed={}", path.display());
    let yaml = std::fs::read_to_string(&path).ok()?;
    serde_yaml::from_str(&yaml).ok()
}

fn emit_conditions(key: &str, kind: &str, conditions: &[String]) {
    if conditions.is_empty() {
        return;
    }
    println!("cargo:rustc-env={key}_{kind}_COUNT={}", conditions.len());
    for (i, c) in conditions.iter().enumerate() {
        println!("cargo:rustc-env={key}_{kind}_{i}={c}");
    }
}
