//! PRA-001 T14: `trace-datacard-v1` — RED stub.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

pub use crate::corpus::sha256_hex;

pub const SCHEME: &str = "trace-datacard-v1";
pub const INDEX_DIR: &str = "index/agent-trace-v1";
pub const DATACARD_DIR: &str = "datacard";

#[derive(Debug, Default)]
pub struct CardMeta;

#[derive(Debug)]
pub struct Snapshot {
    pub id: String,
    pub rows: u64,
    pub public_eligible: u64,
}

impl Snapshot {
    pub fn read(_root: &Path) -> Result<Self, String> {
        Err("unimplemented".into())
    }
    pub fn manifest(&self) -> String {
        String::new()
    }
    pub fn count(&self, _col: &str, _val: &str) -> u64 {
        0
    }
}

pub fn croissant(_s: &Snapshot, _m: &CardMeta) -> Value {
    json!({})
}

pub fn validate(_card: &Value) -> Vec<String> {
    Vec::new()
}

pub fn render(card: &Value) -> String {
    card.to_string()
}

pub fn datasheet(_s: &Snapshot, _m: &CardMeta) -> String {
    String::new()
}

pub fn write_snapshot(_root: &Path, _m: &CardMeta) -> Result<PathBuf, String> {
    Err("unimplemented".into())
}

#[cfg(test)]
#[path = "datacard_tests.rs"]
mod tests;
