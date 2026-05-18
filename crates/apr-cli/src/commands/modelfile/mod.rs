//! CRUX-K-11: Modelfile DSL parser.
//!
//! Parses Ollama-style `Modelfile` syntax into a stable apr config dict.
//!
//! Reference: <https://github.com/ollama/ollama/blob/main/docs/modelfile.md>
//!
//! # Grammar
//!
//! ```text
//! stmt      := directive value
//! directive := FROM | PARAMETER | TEMPLATE | SYSTEM | LICENSE | MESSAGE | ADAPTER
//!              (case-insensitive)
//! value     := single-line | triple-quoted-block
//! ```
//!
//! # Output shape
//!
//! ```json
//! {
//!   "from": "tinyllama",
//!   "parameters": {"temperature": 0.7, "top_p": 0.9},
//!   "template": "...",
//!   "system": "You are helpful.",
//!   "license": null,
//!   "messages": [{"role": "user", "content": "..."}],
//!   "adapter": null
//! }
//! ```

use crate::error::CliError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

pub mod parser;

#[cfg(test)]
mod tests;

pub use parser::{parse_modelfile, parse_modelfile_str, ModelfileError};

/// Parsed Modelfile config.
///
/// Field types mirror Ollama's Modelfile semantics; missing optional
/// directives serialize as JSON `null` (or empty containers) so consumers
/// can use a uniform shape across Modelfiles.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelfileConfig {
    /// FROM directive — REQUIRED. The base model reference (e.g. `tinyllama`).
    pub from: String,
    /// PARAMETER directives — collected into a flat dict (`name → typed value`).
    /// Numeric values are parsed as `f64`; other values remain strings.
    #[serde(default)]
    pub parameters: BTreeMap<String, serde_json::Value>,
    /// TEMPLATE directive — the Jinja-style prompt template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    /// SYSTEM directive — the system prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// LICENSE directive — SPDX or free-form license text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// MESSAGE directives — ordered list of (role, content) pairs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<MessageEntry>,
    /// ADAPTER directive — LoRA / adapter reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
}

/// A single MESSAGE directive entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageEntry {
    /// Speaker role (e.g. `system`, `user`, `assistant`).
    pub role: String,
    /// Message body.
    pub content: String,
}

/// Run `apr modelfile parse <FILE> [--format json]`.
///
/// # Errors
///
/// Returns a `CliError::ValidationFailed` when the Modelfile fails to parse
/// (missing FROM, unknown directive, unterminated triple-quoted block, etc.).
pub fn run_parse(file: &Path, format: &str) -> Result<(), CliError> {
    let raw = std::fs::read_to_string(file).map_err(|e| {
        CliError::ValidationFailed(format!("failed to read {}: {e}", file.display()))
    })?;
    let config = parse_modelfile_str(&raw, file).map_err(|e| {
        // CRUX-K-11 invariant: unknown directive → exit != 0 with file:line:col.
        // Emit the parser's location-prefixed message verbatim so test
        // FALSIFY-CRUX-K-11-003 can grep for `:LINE:`.
        CliError::ValidationFailed(e.to_string())
    })?;

    match format {
        "json" => {
            let pretty = serde_json::to_string_pretty(&config)
                .map_err(|e| CliError::ValidationFailed(format!("JSON serialize failed: {e}")))?;
            println!("{pretty}");
        }
        "human" | "" => {
            println!("FROM: {}", config.from);
            for (k, v) in &config.parameters {
                println!("PARAMETER {k}: {v}");
            }
            if let Some(t) = &config.template {
                println!("TEMPLATE:\n{t}");
            }
            if let Some(s) = &config.system {
                println!("SYSTEM:\n{s}");
            }
            if let Some(l) = &config.license {
                println!("LICENSE: {l}");
            }
            for m in &config.messages {
                println!("MESSAGE {} {}", m.role, m.content);
            }
            if let Some(a) = &config.adapter {
                println!("ADAPTER: {a}");
            }
        }
        other => {
            return Err(CliError::ValidationFailed(format!(
                "unknown --format {other}; expected `json` or `human`"
            )));
        }
    }

    Ok(())
}
