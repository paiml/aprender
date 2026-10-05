//! Declarative fine-tune / distill / merge recipe (PMAT-4838, contracts/apr-recipe-v1.yaml).
//!
//! Not to be confused with `recipe.rs`, the Modelfile-style `apr create` recipe.
//!
//! One YAML file names the whole run: base model, data, method, teacher,
//! merge inputs, eval and output. A recipe is loaded, defaults are filled in,
//! and then it is validated against every rule in the contract before any
//! run starts. Validation reports every violation at once, not the first one.
//!
//! The recipe hash is the SHA-256 of the canonical JSON form of the loaded
//! recipe (sorted keys, defaults filled in). Comments, key order, quoting and
//! omitted defaults do not change it; any changed value does. Receipts carry
//! this hash so two runs can be shown to have used the same recipe.
//!
//! ```yaml
//! recipe: apr.recipe/v1
//! kind: finetune
//! name: qwen35-lora-alpaca
//! base: { model: Qwen/Qwen3.5-0.8B }
//! method: lora
//! lora: { rank: 16, alpha: 32 }
//! data: { train: data/train.jsonl, eval: data/eval.jsonl }
//! output: { path: out/adapter.apr }
//! ```

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::path::Path;

/// The only recipe version this loader accepts.
pub const RECIPE_VERSION: &str = "apr.recipe/v1";

/// What the recipe runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RecipeKind {
    Finetune,
    Distill,
    Merge,
}

/// Fine-tuning method. Same names as `apr finetune --method`, minus `auto`:
/// a recipe states its method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    Full,
    Lora,
    Qlora,
}

/// Merge strategy. Same names as `apr merge --strategy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MergeStrategy {
    Average,
    Weighted,
    Slerp,
    Ties,
    Dare,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRef {
    /// Local path or hub id.
    pub model: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoraSpec {
    pub rank: usize,
    pub alpha: f64,
    #[serde(default)]
    pub dropout: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataSpec {
    pub train: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingSpec {
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

impl Default for TrainingSpec {
    fn default() -> Self {
        Self {
            epochs: default_epochs(),
            batch_size: default_batch_size(),
            learning_rate: default_learning_rate(),
            max_seq_len: default_max_seq_len(),
            seed: default_seed(),
        }
    }
}

fn default_epochs() -> usize {
    3
}
fn default_batch_size() -> usize {
    16
}
fn default_learning_rate() -> f64 {
    0.0002
}
fn default_max_seq_len() -> usize {
    512
}
fn default_seed() -> u64 {
    42
}

/// Knowledge-distillation loss settings. Defaults match `apr distill --config`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistillSpec {
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_kd_alpha")]
    pub alpha: f64,
}

impl Default for DistillSpec {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            alpha: default_kd_alpha(),
        }
    }
}

fn default_temperature() -> f64 {
    4.0
}
fn default_kd_alpha() -> f64 {
    0.7
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MergeSpec {
    pub strategy: MergeStrategy,
    pub models: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalSpec {
    pub metric: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSpec {
    pub path: String,
}

/// A loaded recipe. `base` is the model fine-tuned, or the student when
/// `kind` is `distill`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub recipe: String,
    pub kind: RecipeKind,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<ModelRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher: Option<ModelRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<Method>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<LoraSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<DataSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training: Option<TrainingSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distillation: Option<DistillSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge: Option<MergeSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval: Option<EvalSpec>,
    pub output: OutputSpec,
}

/// One broken rule. `rule` is the contract rule id (`RCP-00N`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    pub rule: &'static str,
    pub field: String,
    pub message: String,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}: {}", self.rule, self.field, self.message)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RecipeError {
    #[error("cannot read recipe {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    /// RCP-002: malformed YAML, a wrong type, or an unknown key.
    #[error("RCP-002 recipe does not parse: {0}")]
    Parse(String),
    #[error("recipe is invalid ({} violation(s)):\n{}", .0.len(), join_violations(.0))]
    Invalid(Vec<Violation>),
}

fn join_violations(v: &[Violation]) -> String {
    v.iter()
        .map(|x| format!("  {x}"))
        .collect::<Vec<_>>()
        .join("\n")
}

impl Recipe {
    /// Parse, fill defaults and validate. Every violation is returned.
    pub fn from_yaml(yaml: &str) -> Result<Self, RecipeError> {
        let mut recipe: Self =
            serde_yaml::from_str(yaml).map_err(|e| RecipeError::Parse(e.to_string()))?;
        recipe.fill_defaults();
        let violations = recipe.validate();
        if violations.is_empty() {
            Ok(recipe)
        } else {
            Err(RecipeError::Invalid(violations))
        }
    }

    pub fn load(path: &Path) -> Result<Self, RecipeError> {
        let text = std::fs::read_to_string(path).map_err(|source| RecipeError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_yaml(&text)
    }

    /// Kinds that train get explicit training (and distillation) sections, so
    /// an omitted default and a written default hash the same.
    fn fill_defaults(&mut self) {
        if matches!(self.kind, RecipeKind::Finetune | RecipeKind::Distill)
            && self.training.is_none()
        {
            self.training = Some(TrainingSpec::default());
        }
        if self.kind == RecipeKind::Distill && self.distillation.is_none() {
            self.distillation = Some(DistillSpec::default());
        }
    }

    /// Check every contract rule. Empty means valid.
    #[must_use]
    pub fn validate(&self) -> Vec<Violation> {
        let mut v = Vec::new();
        self.check_version(&mut v);
        self.check_sections(&mut v);
        self.check_lora(&mut v);
        self.check_training(&mut v);
        self.check_distillation(&mut v);
        self.check_merge(&mut v);
        self.check_non_empty(&mut v);
        v
    }

    /// SHA-256 (hex) of the canonical JSON form: sorted keys, no whitespace.
    #[must_use]
    pub fn hash(&self) -> String {
        let value = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        let mut canonical = String::new();
        write_canonical(&value, &mut canonical);
        let digest = Sha256::digest(canonical.as_bytes());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn check_version(&self, v: &mut Vec<Violation>) {
        if self.recipe != RECIPE_VERSION {
            push(
                v,
                "RCP-001",
                "recipe",
                format!("must be {RECIPE_VERSION:?}, found {:?}", self.recipe),
            );
        }
    }

    /// RCP-003: each kind has required sections and forbidden ones.
    fn check_sections(&self, v: &mut Vec<Violation>) {
        let present = [
            ("base", self.base.is_some()),
            ("teacher", self.teacher.is_some()),
            ("method", self.method.is_some()),
            ("lora", self.lora.is_some()),
            ("data", self.data.is_some()),
            ("training", self.training.is_some()),
            ("distillation", self.distillation.is_some()),
            ("merge", self.merge.is_some()),
        ];
        let (required, forbidden): (&[&str], &[&str]) = match self.kind {
            RecipeKind::Finetune => (
                &["base", "method", "data"],
                &["teacher", "distillation", "merge"],
            ),
            RecipeKind::Distill => (&["base", "teacher", "data"], &["merge"]),
            RecipeKind::Merge => (
                &["merge"],
                &[
                    "base",
                    "teacher",
                    "method",
                    "lora",
                    "data",
                    "training",
                    "distillation",
                ],
            ),
        };
        let kind = kind_name(self.kind);
        for (name, is_present) in present {
            if required.contains(&name) && !is_present {
                push(v, "RCP-003", name, format!("required when kind is {kind}"));
            }
            if forbidden.contains(&name) && is_present {
                push(
                    v,
                    "RCP-003",
                    name,
                    format!("not allowed when kind is {kind}"),
                );
            }
        }
    }

    /// RCP-004: a `lora` section exactly when the method is lora or qlora.
    fn check_lora(&self, v: &mut Vec<Violation>) {
        if self.kind == RecipeKind::Merge {
            return;
        }
        let wants_lora = matches!(self.method, Some(Method::Lora | Method::Qlora));
        match (&self.lora, wants_lora) {
            (None, true) => push(
                v,
                "RCP-004",
                "lora",
                "required when method is lora or qlora",
            ),
            (Some(_), false) => push(
                v,
                "RCP-004",
                "lora",
                "allowed only when method is lora or qlora",
            ),
            _ => {}
        }
        if let Some(l) = &self.lora {
            if l.rank == 0 {
                push(v, "RCP-004", "lora.rank", "must be at least 1");
            }
            if !(l.alpha.is_finite() && l.alpha > 0.0) {
                push(v, "RCP-004", "lora.alpha", "must be finite and > 0");
            }
            if !(l.dropout.is_finite() && (0.0..1.0).contains(&l.dropout)) {
                push(v, "RCP-004", "lora.dropout", "must be in [0, 1)");
            }
        }
    }

    /// RCP-005: training numbers are in range.
    fn check_training(&self, v: &mut Vec<Violation>) {
        let Some(t) = &self.training else { return };
        if t.epochs == 0 {
            push(v, "RCP-005", "training.epochs", "must be at least 1");
        }
        if t.batch_size == 0 {
            push(v, "RCP-005", "training.batch_size", "must be at least 1");
        }
        if t.max_seq_len == 0 {
            push(v, "RCP-005", "training.max_seq_len", "must be at least 1");
        }
        if !(t.learning_rate.is_finite() && t.learning_rate > 0.0) {
            push(
                v,
                "RCP-005",
                "training.learning_rate",
                "must be finite and > 0",
            );
        }
    }

    /// RCP-006: distillation loss settings are in range.
    fn check_distillation(&self, v: &mut Vec<Violation>) {
        let Some(d) = &self.distillation else { return };
        if !(d.temperature.is_finite() && d.temperature > 0.0) {
            push(
                v,
                "RCP-006",
                "distillation.temperature",
                "must be finite and > 0",
            );
        }
        if !(d.alpha.is_finite() && (0.0..=1.0).contains(&d.alpha)) {
            push(v, "RCP-006", "distillation.alpha", "must be in [0, 1]");
        }
    }

    /// RCP-007: merge inputs and weights agree with the strategy.
    fn check_merge(&self, v: &mut Vec<Violation>) {
        let Some(m) = &self.merge else { return };
        if m.models.len() < 2 {
            push(v, "RCP-007", "merge.models", "needs at least 2 models");
        }
        match (m.strategy, &m.weights) {
            (MergeStrategy::Weighted, None) => {
                push(
                    v,
                    "RCP-007",
                    "merge.weights",
                    "required for strategy weighted",
                );
            }
            (MergeStrategy::Average | MergeStrategy::Ties, Some(_)) => push(
                v,
                "RCP-007",
                "merge.weights",
                "not used by this strategy; remove it",
            ),
            _ => {}
        }
        if let Some(w) = &m.weights {
            if w.len() != m.models.len() {
                push(
                    v,
                    "RCP-007",
                    "merge.weights",
                    format!("has {} entries for {} models", w.len(), m.models.len()),
                );
            }
            if w.iter().any(|x| !(x.is_finite() && *x >= 0.0)) {
                push(
                    v,
                    "RCP-007",
                    "merge.weights",
                    "every weight must be finite and >= 0",
                );
            }
        }
    }

    /// RCP-008: names, model refs and paths are not blank.
    fn check_non_empty(&self, v: &mut Vec<Violation>) {
        let mut fields: Vec<(String, &str)> = vec![
            ("name".into(), self.name.as_str()),
            ("output.path".into(), self.output.path.as_str()),
        ];
        if let Some(b) = &self.base {
            fields.push(("base.model".into(), b.model.as_str()));
        }
        if let Some(t) = &self.teacher {
            fields.push(("teacher.model".into(), t.model.as_str()));
        }
        if let Some(d) = &self.data {
            fields.push(("data.train".into(), d.train.as_str()));
            if let Some(e) = &d.eval {
                fields.push(("data.eval".into(), e.as_str()));
            }
        }
        if let Some(e) = &self.eval {
            fields.push(("eval.metric".into(), e.metric.as_str()));
        }
        if let Some(m) = &self.merge {
            for (i, model) in m.models.iter().enumerate() {
                fields.push((format!("merge.models[{i}]"), model.as_str()));
            }
        }
        for (field, value) in fields {
            if value.trim().is_empty() {
                push(v, "RCP-008", &field, "must not be empty");
            }
        }
    }
}

fn kind_name(kind: RecipeKind) -> &'static str {
    match kind {
        RecipeKind::Finetune => "finetune",
        RecipeKind::Distill => "distill",
        RecipeKind::Merge => "merge",
    }
}

fn push(v: &mut Vec<Violation>, rule: &'static str, field: &str, message: impl Into<String>) {
    v.push(Violation {
        rule,
        field: field.to_string(),
        message: message.into(),
    });
}

/// JSON with object keys sorted at every level and no whitespace, so the
/// hash does not depend on serde_json's map ordering feature.
fn write_canonical(value: &serde_json::Value, out: &mut String) {
    use serde_json::Value;
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String((*k).clone()).to_string());
                out.push(':');
                if let Some(child) = map.get(*k) {
                    write_canonical(child, out);
                }
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}

#[cfg(test)]
#[path = "train_recipe_tests.rs"]
mod tests;
