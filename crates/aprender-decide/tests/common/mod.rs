//! Shared plumbing for the integration targets: the verify policy read from the contracts
//! through the library's one mapping, exactly as `examples/pack_laya.rs` reads it (never a literal).
//!
//! Each target compiles this module on its own and uses a subset of it.
#![allow(dead_code)]

use aprender_decide::verify::{GateContractView, ParityContractView, VerifyPolicy};
use std::path::{Path, PathBuf};

/// The workspace root (from this crate's manifest dir, never the environment).
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `contracts/<name>` parsed.
pub fn contract(name: &str) -> serde_yaml::Value {
    let path = workspace_root().join("contracts").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));
    serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("parse {name}: {e}"))
}

fn at<'a>(v: &'a serde_yaml::Value, keys: &[&str]) -> &'a serde_yaml::Value {
    keys.iter().fold(v, |v, k| &v[*k])
}

/// A number at `keys`.
pub fn f64_at(v: &serde_yaml::Value, keys: &[&str]) -> f64 {
    at(v, keys)
        .as_f64()
        .unwrap_or_else(|| panic!("contract value {}", keys.join(".")))
}

/// A string at `keys`.
pub fn str_at(v: &serde_yaml::Value, keys: &[&str]) -> String {
    at(v, keys)
        .as_str()
        .unwrap_or_else(|| panic!("contract value {}", keys.join(".")))
        .to_string()
}

/// A list of integers at `keys`.
pub fn seeds_at(v: &serde_yaml::Value, keys: &[&str]) -> Vec<i64> {
    at(v, keys)
        .as_sequence()
        .unwrap_or_else(|| panic!("contract value {}", keys.join(".")))
        .iter()
        .map(|s| {
            s.as_i64()
                .unwrap_or_else(|| panic!("{}: not an integer", keys.join(".")))
        })
        .collect()
}

/// `contracts/<name>` parsed into a typed view (the library's `GateContractView` /
/// `ParityContractView`).
pub fn contract_view<T: serde::de::DeserializeOwned>(name: &str) -> T {
    let path = workspace_root().join("contracts").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));
    serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("parse {name} into its view: {e}"))
}

/// The contract policy, through the library's ONE mapping (`VerifyPolicy::from_contract_views`)
/// over the same typed views `examples/pack_laya.rs` reads — so no test can pass on a mapping
/// the CLI does not use.
pub fn policy() -> VerifyPolicy {
    VerifyPolicy::from_contract_views(
        &contract_view::<GateContractView>("laya-finetune-gate-v1.yaml"),
        &contract_view::<ParityContractView>("laya-parity-v1.yaml"),
    )
}
