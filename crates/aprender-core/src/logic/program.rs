//! `TensorProgram`: Executable Tensor Logic Programs
//!
//! A `TensorProgram` consists of:
//! - **Facts**: Ground truth tensors (e.g., Parent relation)
//! - **Rules**: Equations that derive new tensors (e.g., Grandparent = Parent @ Parent)
//! - **Mode**: Boolean or Continuous execution
//!
//! # Example
//!
//! ```rust
//! use aprender::logic::{ProgramBuilder, Equation, LogicMode};
//!
//! let mut program = ProgramBuilder::new(LogicMode::Boolean)
//!     .add_fact("parent", vec![
//!         vec![0.0, 1.0, 0.0],
//!         vec![0.0, 0.0, 1.0],
//!         vec![0.0, 0.0, 0.0],
//!     ])
//!     .add_rule("grandparent", Equation::Join("parent".into(), "parent".into()))
//!     .build();
//!
//! let results = program.forward();
//! let grandparent = results.get("grandparent").expect("grandparent rule was defined");
//! assert_eq!(grandparent[0][2], 1.0); // Alice is grandparent of Charlie
//! ```

use super::ops::{logical_join, logical_negation, logical_project, logical_union, LogicMode};
use std::collections::HashMap;

/// An equation defines how to compute a derived tensor
#[derive(Debug, Clone)]
pub enum Equation {
    /// Join two tensors (matrix multiplication)
    Join(String, String),
    /// Join multiple tensors sequentially
    JoinMultiple(Vec<String>),
    /// Project tensor over dimension
    Project(String, usize),
    /// Union of two tensors
    Union(String, String),
    /// Negation of a tensor
    Negation(String),
    /// Copy from another tensor
    Copy(String),
}

/// A `TensorProgram` is an executable collection of facts and rules
#[derive(Debug)]
pub struct TensorProgram {
    /// Execution mode (Boolean or Continuous)
    mode: LogicMode,
    /// Ground truth tensors (facts)
    facts: HashMap<String, Vec<Vec<f64>>>,
    /// Derived tensors (computed via rules)
    derived: HashMap<String, Vec<Vec<f64>>>,
    /// Rules in execution order
    rules: Vec<(String, Equation)>,
}

impl TensorProgram {
    /// Create a new empty program
    #[must_use]
    pub fn new(mode: LogicMode) -> Self {
        Self {
            mode,
            facts: HashMap::new(),
            derived: HashMap::new(),
            rules: Vec::new(),
        }
    }

    /// Add a fact (ground truth tensor)
    pub fn add_fact(&mut self, name: &str, tensor: Vec<Vec<f64>>) {
        self.facts.insert(name.to_string(), tensor);
    }

    /// Add a rule (derived tensor)
    pub fn add_rule(&mut self, name: &str, equation: Equation) {
        self.rules.push((name.to_string(), equation));
    }

    /// Execute forward chaining: compute all derived tensors
    pub fn forward(&mut self) -> &HashMap<String, Vec<Vec<f64>>> {
        self.derived.clear();

        for (name, equation) in &self.rules.clone() {
            let result = self.evaluate_equation(equation);
            self.derived.insert(name.clone(), result);
        }

        &self.derived
    }

    /// Query for a specific tensor (backward chaining style)
    pub fn query(&mut self, name: &str) -> Option<Vec<Vec<f64>>> {
        // First check if it's a fact
        if let Some(tensor) = self.facts.get(name) {
            return Some(tensor.clone());
        }

        // Check if already derived
        if let Some(tensor) = self.derived.get(name) {
            return Some(tensor.clone());
        }

        // Find and evaluate the rule
        for (rule_name, equation) in &self.rules.clone() {
            if rule_name == name {
                let result = self.evaluate_equation(equation);
                self.derived.insert(name.to_string(), result.clone());
                return Some(result);
            }
        }

        None
    }

    /// Get a tensor by name (fact or derived)
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Vec<Vec<f64>>> {
        self.facts.get(name).or_else(|| self.derived.get(name))
    }

    /// Get all computed results
    #[must_use]
    pub fn results(&self) -> HashMap<String, Vec<Vec<f64>>> {
        let mut all = self.facts.clone();
        all.extend(self.derived.clone());
        all
    }

    fn evaluate_equation(&self, equation: &Equation) -> Vec<Vec<f64>> {
        match equation {
            Equation::Join(a, b) => {
                let t1 = self.get_tensor(a);
                let t2 = self.get_tensor(b);
                logical_join(&t1, &t2, self.mode)
            }
            Equation::JoinMultiple(tensors) => {
                if tensors.is_empty() {
                    return vec![];
                }
                let mut result = self.get_tensor(&tensors[0]);
                for name in tensors.iter().skip(1) {
                    let t = self.get_tensor(name);
                    result = logical_join(&result, &t, self.mode);
                }
                result
            }
            Equation::Project(name, dim) => {
                let t = self.get_tensor(name);
                let projected = logical_project(&t, *dim, self.mode);
                // Convert 1D to 2D (row vector)
                vec![projected]
            }
            Equation::Union(a, b) => {
                let t1 = self.get_tensor(a);
                let t2 = self.get_tensor(b);
                logical_union(&t1, &t2, self.mode)
            }
            Equation::Negation(name) => {
                let t = self.get_tensor(name);
                logical_negation(&t, self.mode)
            }
            Equation::Copy(name) => self.get_tensor(name),
        }
    }

    fn get_tensor(&self, name: &str) -> Vec<Vec<f64>> {
        self.facts
            .get(name)
            .or_else(|| self.derived.get(name))
            .map_or_else(Vec::new, Clone::clone)
    }
}

/// Builder for constructing `TensorPrograms` fluently
#[derive(Debug)]
pub struct ProgramBuilder {
    program: TensorProgram,
}

impl ProgramBuilder {
    /// Create a new builder with the specified mode
    #[must_use]
    pub fn new(mode: LogicMode) -> Self {
        Self {
            program: TensorProgram::new(mode),
        }
    }

    /// Add a fact to the program
    #[must_use]
    pub fn add_fact(mut self, name: &str, tensor: Vec<Vec<f64>>) -> Self {
        self.program.add_fact(name, tensor);
        self
    }

    /// Add a rule to the program
    #[must_use]
    pub fn add_rule(mut self, name: &str, equation: Equation) -> Self {
        self.program.add_rule(name, equation);
        self
    }

    /// Build the final program
    #[must_use]
    pub fn build(self) -> TensorProgram {
        self.program
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_family_tree_reasoning() {
        // Alice(0) -> Bob(1) -> Charlie(2)
        let parent = vec![
            vec![0.0, 1.0, 0.0], // Alice is parent of Bob
            vec![0.0, 0.0, 1.0], // Bob is parent of Charlie
            vec![0.0, 0.0, 0.0], // Charlie has no children
        ];

        let mut program = ProgramBuilder::new(LogicMode::Boolean)
            .add_fact("parent", parent)
            .add_rule(
                "grandparent",
                Equation::Join("parent".into(), "parent".into()),
            )
            .add_rule("has_child", Equation::Project("parent".into(), 1))
            .build();

        let results = program.forward();

        // Verify grandparent relation
        let grandparent = results.get("grandparent").unwrap();
        assert_eq!(grandparent[0][2], 1.0, "Alice is grandparent of Charlie");
        assert_eq!(grandparent[0][0], 0.0, "Alice is not her own grandparent");
        assert_eq!(grandparent[0][1], 0.0, "Alice is not grandparent of Bob");
        assert_eq!(grandparent[1][2], 0.0, "Bob is not grandparent of Charlie");

        // Verify has_child relation
        let has_child = results.get("has_child").unwrap();
        assert_eq!(has_child[0][0], 1.0, "Alice has child");
        assert_eq!(has_child[0][1], 1.0, "Bob has child");
        assert_eq!(has_child[0][2], 0.0, "Charlie has no child");
    }

    #[test]
    fn test_query_backward_chaining() {
        let parent = vec![vec![0.0, 1.0], vec![0.0, 0.0]];

        let mut program = ProgramBuilder::new(LogicMode::Boolean)
            .add_fact("parent", parent)
            .add_rule(
                "grandparent",
                Equation::Join("parent".into(), "parent".into()),
            )
            .build();

        // Query should trigger computation
        let result = program.query("grandparent");
        assert!(result.is_some());
    }

    #[test]
    fn test_union_rule() {
        let a = vec![vec![1.0, 0.0]];
        let b = vec![vec![0.0, 1.0]];

        let mut program = ProgramBuilder::new(LogicMode::Boolean)
            .add_fact("a", a)
            .add_fact("b", b)
            .add_rule("a_or_b", Equation::Union("a".into(), "b".into()))
            .build();

        let results = program.forward();
        let union = results.get("a_or_b").unwrap();

        assert_eq!(union[0][0], 1.0);
        assert_eq!(union[0][1], 1.0);
    }

    #[test]
    fn test_negation_rule() {
        let a = vec![vec![1.0, 0.0]];

        let mut program = ProgramBuilder::new(LogicMode::Boolean)
            .add_fact("a", a)
            .add_rule("not_a", Equation::Negation("a".into()))
            .build();

        let results = program.forward();
        let negated = results.get("not_a").unwrap();

        assert_eq!(negated[0][0], 0.0);
        assert_eq!(negated[0][1], 1.0);
    }

    #[test]
    fn test_continuous_mode() {
        let parent = vec![
            vec![0.0, 0.8, 0.0],
            vec![0.0, 0.0, 0.7],
            vec![0.0, 0.0, 0.0],
        ];

        let mut program = ProgramBuilder::new(LogicMode::Continuous)
            .add_fact("parent", parent)
            .add_rule(
                "grandparent",
                Equation::Join("parent".into(), "parent".into()),
            )
            .build();

        let results = program.forward();
        let grandparent = results.get("grandparent").unwrap();

        // 0.8 * 0.7 = 0.56
        assert!((grandparent[0][2] - 0.56).abs() < 1e-6);
    }

    #[test]
    fn test_copy_equation() {
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut program = ProgramBuilder::new(LogicMode::Boolean)
            .add_fact("original", data.clone())
            .add_rule("copy", Equation::Copy("original".into()))
            .build();

        let results = program.forward();
        let copied = results.get("copy").unwrap();
        assert_eq!(copied[0][0], 1.0);
        assert_eq!(copied[0][1], 0.0);
        assert_eq!(copied[1][0], 0.0);
        assert_eq!(copied[1][1], 1.0);
    }

    #[test]
    fn test_join_multiple_empty() {
        let mut program = ProgramBuilder::new(LogicMode::Boolean)
            .add_rule("empty_join", Equation::JoinMultiple(vec![]))
            .build();

        let results = program.forward();
        let empty = results.get("empty_join").unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_join_multiple_single() {
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut program = ProgramBuilder::new(LogicMode::Boolean)
            .add_fact("a", data)
            .add_rule("single", Equation::JoinMultiple(vec!["a".into()]))
            .build();

        let results = program.forward();
        let result = results.get("single").unwrap();
        assert_eq!(result[0][0], 1.0);
        assert_eq!(result[1][1], 1.0);
    }

    #[test]
    fn test_join_multiple_three() {
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut program = ProgramBuilder::new(LogicMode::Boolean)
            .add_fact("id", identity)
            .add_rule(
                "triple",
                Equation::JoinMultiple(vec!["id".into(), "id".into(), "id".into()]),
            )
            .build();

        let results = program.forward();
        let result = results.get("triple").unwrap();
        // Identity * Identity * Identity = Identity
        assert_eq!(result[0][0], 1.0);
        assert_eq!(result[1][1], 1.0);
    }

    #[test]
    fn test_query_returns_fact() {
        let data = vec![vec![1.0, 0.0]];
        let mut program = ProgramBuilder::new(LogicMode::Boolean)
            .add_fact("my_fact", data.clone())
            .build();

        let result = program.query("my_fact");
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0][0], 1.0);
    }

    #[test]
    fn test_query_returns_already_derived() {
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut program = ProgramBuilder::new(LogicMode::Boolean)
            .add_fact("a", data)
            .add_rule("b", Equation::Copy("a".into()))
            .build();

        // First call derives and caches
        let _ = program.forward();
        // Second query hits the derived cache
        let result = program.query("b");
        assert!(result.is_some());
    }

    #[test]
    fn test_query_returns_none() {
        let mut program = ProgramBuilder::new(LogicMode::Boolean).build();
        let result = program.query("nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_get_fact_and_derived() {
        let data = vec![vec![1.0, 0.0]];
        let mut program = ProgramBuilder::new(LogicMode::Boolean)
            .add_fact("fact1", data)
            .add_rule("derived1", Equation::Copy("fact1".into()))
            .build();

        // Before forward, derived is not available
        assert!(program.get("fact1").is_some());
        assert!(program.get("derived1").is_none());

        // After forward, both are available
        program.forward();
        assert!(program.get("fact1").is_some());
        assert!(program.get("derived1").is_some());

        // Missing key returns None
        assert!(program.get("missing").is_none());
    }

    #[test]
    fn test_results_method() {
        let data = vec![vec![1.0]];
        let mut program = ProgramBuilder::new(LogicMode::Boolean)
            .add_fact("f", data)
            .add_rule("d", Equation::Copy("f".into()))
            .build();

        program.forward();
        let all = program.results();
        assert!(all.contains_key("f"));
        assert!(all.contains_key("d"));
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_get_tensor_missing_returns_empty() {
        let mut program = ProgramBuilder::new(LogicMode::Boolean)
            .add_rule(
                "bad",
                Equation::Join("nonexistent_a".into(), "nonexistent_b".into()),
            )
            .build();

        // forward will call get_tensor for missing names, returning empty vecs
        let results = program.forward();
        let bad = results.get("bad").unwrap();
        assert!(bad.is_empty());
    }

    #[test]
    fn test_equation_debug_clone() {
        let eq = Equation::Join("a".into(), "b".into());
        let debug_str = format!("{:?}", eq);
        assert!(debug_str.contains("Join"));

        let cloned = eq.clone();
        let cloned_debug = format!("{:?}", cloned);
        assert_eq!(debug_str, cloned_debug);
    }

    #[test]
    fn test_program_builder_debug() {
        let builder = ProgramBuilder::new(LogicMode::Boolean);
        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("ProgramBuilder"));
    }

    #[test]
    fn test_tensor_program_debug() {
        let program = TensorProgram::new(LogicMode::Continuous);
        let debug_str = format!("{:?}", program);
        assert!(debug_str.contains("TensorProgram"));
    }
}
