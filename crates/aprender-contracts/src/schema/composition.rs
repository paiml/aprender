use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Typed shape contract for compositional verification across contract boundaries.
///
/// Used in `Equation.assumes` and `Equation.guarantees` to create
/// mechanically verifiable edges in the dependency graph. The COMPOSITION-001
/// lint gate unifies `guarantees.shapes` with downstream `assumes.shapes`
/// to prove end-to-end pipeline shape consistency.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShapeContract {
    /// Named shape bindings, e.g. `{"output": {"dims": ["batch", "seq", "config.hidden_size"]}}`.
    #[serde(default)]
    pub shapes: BTreeMap<String, ShapeExpr>,
    /// Constraints that must hold, e.g. `["config.hidden_size % config.num_heads == 0"]`.
    #[serde(default)]
    pub constraints: Vec<String>,
    /// Which upstream contract provides these shapes (assumes only).
    #[serde(default)]
    pub from_contract: Option<String>,
    /// Which upstream equation provides these shapes (assumes only).
    #[serde(default)]
    pub from_equation: Option<String>,
}

/// A tensor shape expression with dimension list and optional dtype.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeExpr {
    /// Dimension expressions, e.g. `["batch", "seq", "config.num_heads * config.head_dim"]`.
    pub dims: Vec<String>,
    /// Optional dtype constraint, e.g. `"config.compute_dtype"` or `"f32"`.
    #[serde(default)]
    pub dtype: Option<String>,
}
