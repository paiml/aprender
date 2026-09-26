//! One declarative recipe for fine-tune and distill (E8 #4002 row R-1,
//! `contracts/apr-recipe-v1.yaml`).
//!
//! A recipe fully determines a run together with the binary sha: base model,
//! data (by hash), method, eval, and seed. [`Recipe::parse`] refuses an invalid
//! recipe before anything is loaded, and every refusal names the recipe field
//! it is about ([`RecipeError::field`]), so a caller can tell a recipe defect
//! from a CLI flag error.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The only recipe schema version this binary accepts.
pub const RECIPE_VERSION: u32 = 1;

/// A recipe refusal. `field` is the dotted path of the offending key.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("recipe field `{field}`: {reason}")]
pub struct RecipeError {
    pub field: String,
    pub reason: String,
}

fn err(field: &str, reason: impl Into<String>) -> RecipeError {
    RecipeError { field: field.to_string(), reason: reason.into() }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub recipe_version: u32,
    pub base: Base,
    pub data: DataRef,
    pub method: Method,
    pub eval: Eval,
    pub training: Training,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Base {
    /// Model id or path of the model being tuned (the student, for distill).
    pub model: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataRef {
    pub train: String,
    /// sha256 of the train file: the data is part of the recipe identity.
    pub sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MethodKind {
    Lora,
    Qlora,
    Full,
    Distill,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Method {
    pub kind: MethodKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f64>,
    /// Teacher model: required for `distill`, refused otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Eval {
    pub held_out: String,
    pub sha256: String,
    pub metric: EvalMetric,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvalMetric {
    Loss,
    Accuracy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Training {
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f64,
    /// Required, never defaulted: a defaulted seed makes two recipes that
    /// differ only by an omitted key look identical.
    pub seed: u64,
}

const TOP_LEVEL: [&str; 6] = ["recipe_version", "base", "data", "method", "eval", "training"];

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl Recipe {
    /// Parse and validate a recipe. Nothing is loaded: this runs before any
    /// model or data access.
    pub fn parse(yaml: &str) -> Result<Self, RecipeError> {
        let value: serde_yaml::Value =
            serde_yaml::from_str(yaml).map_err(|e| err("<root>", format!("not YAML: {e}")))?;
        let map = value.as_mapping().ok_or_else(|| err("<root>", "a recipe is a YAML mapping"))?;
        // Name a missing top-level section by path before serde does, so the
        // refusal is the same shape for every section.
        for key in TOP_LEVEL {
            if !map.contains_key(key) {
                return Err(err(key, "missing"));
            }
        }
        for k in map.keys() {
            let name = k.as_str().unwrap_or("<non-string key>");
            if !TOP_LEVEL.contains(&name) {
                return Err(err(name, "unknown key"));
            }
        }
        for key in TOP_LEVEL {
            // Deserialize each section alone so a nested error carries its section.
            let section = &map[key];
            let check = match key {
                "recipe_version" => serde_yaml::from_value::<u32>(section.clone()).map(|_| ()),
                "base" => serde_yaml::from_value::<Base>(section.clone()).map(|_| ()),
                "data" => serde_yaml::from_value::<DataRef>(section.clone()).map(|_| ()),
                "method" => serde_yaml::from_value::<Method>(section.clone()).map(|_| ()),
                "eval" => serde_yaml::from_value::<Eval>(section.clone()).map(|_| ()),
                _ => serde_yaml::from_value::<Training>(section.clone()).map(|_| ()),
            };
            check.map_err(|e| err(key, e.to_string()))?;
        }
        let recipe: Recipe =
            serde_yaml::from_value(value).map_err(|e| err("<root>", e.to_string()))?;
        recipe.validate()?;
        Ok(recipe)
    }

    /// Semantic rules serde cannot express.
    pub fn validate(&self) -> Result<(), RecipeError> {
        if self.recipe_version != RECIPE_VERSION {
            return Err(err(
                "recipe_version",
                format!(
                    "{} is not supported (this binary reads {RECIPE_VERSION})",
                    self.recipe_version
                ),
            ));
        }
        if self.base.model.trim().is_empty() {
            return Err(err("base.model", "empty"));
        }
        if self.data.train.trim().is_empty() {
            return Err(err("data.train", "empty"));
        }
        if !is_sha256(&self.data.sha256) {
            return Err(err("data.sha256", "not 64 lowercase hex"));
        }
        if self.eval.held_out.trim().is_empty() {
            return Err(err("eval.held_out", "empty"));
        }
        if !is_sha256(&self.eval.sha256) {
            return Err(err("eval.sha256", "not 64 lowercase hex"));
        }
        if self.eval.sha256 == self.data.sha256 {
            return Err(err("eval.sha256", "equals data.sha256: the eval split is the train data"));
        }
        let m = &self.method;
        match m.kind {
            MethodKind::Lora | MethodKind::Qlora => {
                if m.rank.unwrap_or(0) == 0 {
                    return Err(err("method.rank", "required and > 0 for lora/qlora"));
                }
            }
            MethodKind::Full => {
                if m.rank.is_some() || m.alpha.is_some() {
                    return Err(err("method.rank", "full fine-tune takes no adapter rank/alpha"));
                }
            }
            MethodKind::Distill => {
                let teacher = m.teacher.as_deref().unwrap_or("");
                if teacher.trim().is_empty() {
                    return Err(err("method.teacher", "required for distill"));
                }
                if teacher == self.base.model {
                    return Err(err(
                        "method.teacher",
                        "equals base.model: a model cannot teach itself",
                    ));
                }
            }
        }
        if m.kind != MethodKind::Distill && (m.teacher.is_some() || m.temperature.is_some()) {
            return Err(err("method.teacher", "only distill takes a teacher/temperature"));
        }
        if let Some(t) = m.temperature {
            if !(t.is_finite() && t > 0.0) {
                return Err(err("method.temperature", "must be finite and > 0"));
            }
        }
        if let Some(a) = m.alpha {
            if !(a.is_finite() && a > 0.0) {
                return Err(err("method.alpha", "must be finite and > 0"));
            }
        }
        let t = &self.training;
        if t.epochs == 0 {
            return Err(err("training.epochs", "must be > 0"));
        }
        if t.batch_size == 0 {
            return Err(err("training.batch_size", "must be > 0"));
        }
        if !(t.learning_rate.is_finite() && t.learning_rate > 0.0) {
            return Err(err("training.learning_rate", "must be finite and > 0"));
        }
        Ok(())
    }

    /// sha256 of the canonical JSON form: key order and YAML formatting do not
    /// change the hash, any value change does.
    pub fn hash(&self) -> String {
        let canonical = serde_json::to_value(self).map(|v| v.to_string()).unwrap_or_default();
        let digest = Sha256::digest(canonical.as_bytes());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[cfg(test)]
#[path = "recipe_tests.rs"]
mod tests;
