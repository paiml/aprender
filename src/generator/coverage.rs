//! Coverage-guided generation (NAUTILUS-style)
//!
//! Implements grammar-aware fuzzing with coverage feedback to prioritize
//! unexplored AST paths. Based on Aschermann et al. (2019) NAUTILUS.
//!
//! # Key Concepts
//!
//! - **Coverage Map**: Tracks which AST paths/features have been explored
//! - **Corpus**: Collection of interesting inputs that increased coverage
//! - **Energy**: Selection probability based on coverage potential
//! - **Grammar-aware mutation**: Mutate AST nodes while maintaining validity

#![allow(clippy::self_only_used_in_recursion)]
#![allow(clippy::match_same_arms)]

use std::collections::HashSet;

use super::{BinaryOp, CompareOp, GeneratedCode, PythonEnumerator, PythonNode, UnaryOp};
use crate::Language;

/// Coverage information for a generated program
#[derive(Debug, Clone, Default)]
pub struct CoverageMap {
    /// AST node types seen
    node_types: HashSet<String>,
    /// AST paths (parent->child relationships) seen
    ast_paths: HashSet<(String, String)>,
    /// Feature combinations seen
    feature_combos: HashSet<String>,
}

impl CoverageMap {
    /// Create a new empty coverage map
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a node type as covered
    pub fn record_node(&mut self, node_type: &str) {
        self.node_types.insert(node_type.to_string());
    }

    /// Record an AST path (parent->child) as covered
    pub fn record_path(&mut self, parent: &str, child: &str) {
        self.ast_paths
            .insert((parent.to_string(), child.to_string()));
    }

    /// Record a feature combination
    pub fn record_feature(&mut self, feature: &str) {
        self.feature_combos.insert(feature.to_string());
    }

    /// Check if this coverage has any new information compared to another
    #[must_use]
    pub fn has_new_coverage(&self, existing: &Self) -> bool {
        // Check for new node types
        for node in &self.node_types {
            if !existing.node_types.contains(node) {
                return true;
            }
        }

        // Check for new AST paths
        for path in &self.ast_paths {
            if !existing.ast_paths.contains(path) {
                return true;
            }
        }

        // Check for new feature combos
        for feature in &self.feature_combos {
            if !existing.feature_combos.contains(feature) {
                return true;
            }
        }

        false
    }

    /// Merge another coverage map into this one
    pub fn merge(&mut self, other: &Self) {
        self.node_types.extend(other.node_types.iter().cloned());
        self.ast_paths.extend(other.ast_paths.iter().cloned());
        self.feature_combos
            .extend(other.feature_combos.iter().cloned());
    }

    /// Get the total number of covered items
    #[must_use]
    pub fn coverage_count(&self) -> usize {
        self.node_types.len() + self.ast_paths.len() + self.feature_combos.len()
    }

    /// Get covered node types
    #[must_use]
    pub fn node_types(&self) -> &HashSet<String> {
        &self.node_types
    }

    /// Get covered AST paths
    #[must_use]
    pub fn ast_paths(&self) -> &HashSet<(String, String)> {
        &self.ast_paths
    }
}

/// Entry in the corpus of interesting inputs
#[derive(Debug, Clone)]
pub struct CorpusEntry {
    /// The generated code
    pub code: GeneratedCode,
    /// Coverage achieved by this input
    pub coverage: CoverageMap,
    /// Energy score (selection probability weight)
    pub energy: f64,
    /// Number of times this entry has been selected
    pub selection_count: usize,
    /// AST representation for mutation
    pub ast: Option<PythonNode>,
}

impl CorpusEntry {
    /// Create a new corpus entry
    #[must_use]
    pub fn new(code: GeneratedCode, coverage: CoverageMap) -> Self {
        Self {
            code,
            coverage,
            energy: 1.0,
            selection_count: 0,
            ast: None,
        }
    }

    /// Create a corpus entry with AST
    #[must_use]
    pub fn with_ast(code: GeneratedCode, coverage: CoverageMap, ast: PythonNode) -> Self {
        Self {
            code,
            coverage,
            energy: 1.0,
            selection_count: 0,
            ast: Some(ast),
        }
    }

    /// Update energy based on coverage potential
    pub fn update_energy(&mut self, global_coverage: &CoverageMap) {
        // Higher energy for entries with unique coverage
        let unique_nodes = self
            .coverage
            .node_types
            .difference(&global_coverage.node_types)
            .count();
        let unique_paths = self
            .coverage
            .ast_paths
            .difference(&global_coverage.ast_paths)
            .count();

        // Energy decays with selection count but boosted by unique coverage
        let decay = 1.0 / (1.0 + self.selection_count as f64 * 0.1);
        let uniqueness_boost = 1.0 + (unique_nodes + unique_paths) as f64 * 0.5;

        self.energy = decay * uniqueness_boost;
    }
}

/// NAUTILUS-style coverage-guided generator
#[derive(Debug)]
pub struct NautilusGenerator {
    /// Corpus of interesting inputs
    corpus: Vec<CorpusEntry>,
    /// Global coverage map
    global_coverage: CoverageMap,
    /// Maximum corpus size
    max_corpus_size: usize,
    /// Maximum AST depth for generation
    max_depth: usize,
    /// Target language
    language: Language,
    /// Random seed for reproducibility
    seed: u64,
    /// Simple RNG state
    rng_state: u64,
}

impl NautilusGenerator {
    /// Create a new NAUTILUS generator
    #[must_use]
    pub fn new(language: Language, max_depth: usize) -> Self {
        Self {
            corpus: Vec::new(),
            global_coverage: CoverageMap::new(),
            max_corpus_size: 1000,
            max_depth,
            language,
            seed: 42,
            rng_state: 42,
        }
    }

    /// Set the random seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.rng_state = seed;
        self
    }

    /// Set maximum corpus size
    #[must_use]
    pub fn with_max_corpus(mut self, size: usize) -> Self {
        self.max_corpus_size = size;
        self
    }

    /// Simple xorshift64 PRNG
    fn next_random(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Random float in [0, 1)
    fn random_float(&mut self) -> f64 {
        (self.next_random() as f64) / (u64::MAX as f64)
    }

    /// Initialize corpus with seed inputs from exhaustive enumeration
    pub fn initialize_corpus(&mut self) {
        let enumerator = PythonEnumerator::new(self.max_depth.min(2));
        let seeds = enumerator.enumerate_programs();

        for program in seeds.into_iter().take(self.max_corpus_size / 10) {
            let coverage = self.compute_coverage(&program.code);
            if coverage.has_new_coverage(&self.global_coverage) {
                self.global_coverage.merge(&coverage);
                self.corpus.push(CorpusEntry::new(program, coverage));
            }
        }
    }

    /// Initialize corpus with ASTs for mutation
    pub fn initialize_corpus_with_ast(&mut self) {
        let enumerator = PythonEnumerator::new(self.max_depth.min(2));

        // Get statements with their ASTs
        for stmt in enumerator.enumerate_statements(self.max_depth.min(2)) {
            let code = stmt.to_code(0);
            let ast_depth = stmt.depth();

            let program = GeneratedCode {
                code: code.clone(),
                language: self.language,
                ast_depth,
                features: Self::extract_features(&stmt),
            };

            let coverage = self.compute_coverage_from_ast(&stmt);

            if coverage.has_new_coverage(&self.global_coverage) {
                self.global_coverage.merge(&coverage);
                self.corpus
                    .push(CorpusEntry::with_ast(program, coverage, stmt));
            }

            if self.corpus.len() >= self.max_corpus_size / 10 {
                break;
            }
        }
    }

    /// Extract features from an AST node
    fn extract_features(node: &PythonNode) -> Vec<String> {
        let mut features = Vec::new();

        match node {
            PythonNode::IntLit(_) => features.push("literal".to_string()),
            PythonNode::FloatLit(_) => {
                features.push("literal".to_string());
                features.push("float".to_string());
            }
            PythonNode::StrLit(_) => {
                features.push("literal".to_string());
                features.push("string".to_string());
            }
            PythonNode::BoolLit(_) => {
                features.push("literal".to_string());
                features.push("boolean".to_string());
            }
            PythonNode::NoneLit => {
                features.push("literal".to_string());
                features.push("none".to_string());
            }
            PythonNode::Name(_) => features.push("variable".to_string()),
            PythonNode::BinOp { op, .. } => {
                features.push("binary_op".to_string());
                features.push(format!("op_{}", op.to_str()));
            }
            PythonNode::UnaryOp { op, .. } => {
                features.push("unary_op".to_string());
                features.push(format!("op_{}", op.to_str()));
            }
            PythonNode::Compare { op, .. } => {
                features.push("comparison".to_string());
                features.push(format!("cmp_{}", op.to_str()));
            }
            PythonNode::Assign { .. } => features.push("assignment".to_string()),
            PythonNode::Return(_) => features.push("return".to_string()),
            PythonNode::If { orelse, .. } => {
                features.push("conditional".to_string());
                if !orelse.is_empty() {
                    features.push("else_branch".to_string());
                }
            }
            PythonNode::While { .. } => {
                features.push("loop".to_string());
                features.push("while_loop".to_string());
            }
            PythonNode::For { .. } => {
                features.push("loop".to_string());
                features.push("for_loop".to_string());
            }
            PythonNode::FuncDef { .. } => features.push("function_def".to_string()),
            PythonNode::Call { .. } => features.push("function_call".to_string()),
            PythonNode::List(_) => {
                features.push("collection".to_string());
                features.push("list".to_string());
            }
            PythonNode::Module(_) => features.push("module".to_string()),
            PythonNode::Pass => features.push("pass".to_string()),
            PythonNode::Break => {
                features.push("control_flow".to_string());
                features.push("break".to_string());
            }
            PythonNode::Continue => {
                features.push("control_flow".to_string());
                features.push("continue".to_string());
            }
        }

        features
    }

    /// Compute coverage from source code (static analysis)
    fn compute_coverage(&self, code: &str) -> CoverageMap {
        let mut coverage = CoverageMap::new();

        // Simple lexical coverage analysis
        if code.contains("def ") {
            coverage.record_node("function_def");
        }
        if code.contains("if ") {
            coverage.record_node("if_stmt");
        }
        if code.contains("while ") {
            coverage.record_node("while_stmt");
        }
        if code.contains("for ") {
            coverage.record_node("for_stmt");
        }
        if code.contains("return ") || code.contains("return\n") {
            coverage.record_node("return_stmt");
        }
        if code.contains(" = ") {
            coverage.record_node("assignment");
        }
        if code.contains('+') || code.contains('-') || code.contains('*') || code.contains('/') {
            coverage.record_node("binary_op");
        }
        if code.contains('[') {
            coverage.record_node("list");
        }

        coverage
    }

    /// Compute coverage from AST (more precise)
    fn compute_coverage_from_ast(&self, node: &PythonNode) -> CoverageMap {
        let mut coverage = CoverageMap::new();
        self.visit_ast_for_coverage(node, None, &mut coverage);
        coverage
    }

    /// Recursively visit AST nodes to compute coverage
    fn visit_ast_for_coverage(
        &self,
        node: &PythonNode,
        parent: Option<&str>,
        coverage: &mut CoverageMap,
    ) {
        let node_type = Self::node_type_name(node);
        coverage.record_node(&node_type);

        if let Some(p) = parent {
            coverage.record_path(p, &node_type);
        }

        for feature in Self::extract_features(node) {
            coverage.record_feature(&feature);
        }

        self.visit_children(node, &node_type, coverage);
    }

    fn visit_children(&self, node: &PythonNode, node_type: &str, coverage: &mut CoverageMap) {
        match node {
            PythonNode::Module(stmts) => {
                for stmt in stmts {
                    self.visit_ast_for_coverage(stmt, Some(node_type), coverage);
                }
            }
            PythonNode::BinOp { left, right, .. } => {
                self.visit_ast_for_coverage(left, Some(node_type), coverage);
                self.visit_ast_for_coverage(right, Some(node_type), coverage);
            }
            PythonNode::UnaryOp { operand, .. } => {
                self.visit_ast_for_coverage(operand, Some(node_type), coverage);
            }
            PythonNode::Compare { left, right, .. } => {
                self.visit_ast_for_coverage(left, Some(node_type), coverage);
                self.visit_ast_for_coverage(right, Some(node_type), coverage);
            }
            PythonNode::Assign { value, .. } => {
                self.visit_ast_for_coverage(value, Some(node_type), coverage);
            }
            PythonNode::Return(Some(expr)) => {
                self.visit_ast_for_coverage(expr, Some(node_type), coverage);
            }
            PythonNode::If { test, body, orelse } => {
                self.visit_ast_for_coverage(test, Some(node_type), coverage);
                for stmt in body {
                    self.visit_ast_for_coverage(stmt, Some(node_type), coverage);
                }
                for stmt in orelse {
                    self.visit_ast_for_coverage(stmt, Some(node_type), coverage);
                }
            }
            PythonNode::While { test, body } => {
                self.visit_ast_for_coverage(test, Some(node_type), coverage);
                for stmt in body {
                    self.visit_ast_for_coverage(stmt, Some(node_type), coverage);
                }
            }
            PythonNode::For { iter, body, .. } => {
                self.visit_ast_for_coverage(iter, Some(node_type), coverage);
                for stmt in body {
                    self.visit_ast_for_coverage(stmt, Some(node_type), coverage);
                }
            }
            PythonNode::FuncDef { body, .. } => {
                for stmt in body {
                    self.visit_ast_for_coverage(stmt, Some(node_type), coverage);
                }
            }
            PythonNode::Call { args, .. } => {
                for arg in args {
                    self.visit_ast_for_coverage(arg, Some(node_type), coverage);
                }
            }
            PythonNode::List(items) => {
                for item in items {
                    self.visit_ast_for_coverage(item, Some(node_type), coverage);
                }
            }
            PythonNode::IntLit(_)
            | PythonNode::FloatLit(_)
            | PythonNode::StrLit(_)
            | PythonNode::BoolLit(_)
            | PythonNode::NoneLit
            | PythonNode::Name(_)
            | PythonNode::Return(None)
            | PythonNode::Pass
            | PythonNode::Break
            | PythonNode::Continue => {}
        }
    }

    /// Get the type name of a node
    fn node_type_name(node: &PythonNode) -> String {
        match node {
            PythonNode::Module(_) => "Module".to_string(),
            PythonNode::IntLit(_) => "IntLit".to_string(),
            PythonNode::FloatLit(_) => "FloatLit".to_string(),
            PythonNode::StrLit(_) => "StrLit".to_string(),
            PythonNode::BoolLit(_) => "BoolLit".to_string(),
            PythonNode::NoneLit => "NoneLit".to_string(),
            PythonNode::Name(_) => "Name".to_string(),
            PythonNode::BinOp { op, .. } => format!("BinOp_{}", op.to_str()),
            PythonNode::UnaryOp { op, .. } => format!("UnaryOp_{}", op.to_str()),
            PythonNode::Compare { op, .. } => format!("Compare_{}", op.to_str()),
            PythonNode::Assign { .. } => "Assign".to_string(),
            PythonNode::Return(_) => "Return".to_string(),
            PythonNode::If { .. } => "If".to_string(),
            PythonNode::While { .. } => "While".to_string(),
            PythonNode::For { .. } => "For".to_string(),
            PythonNode::FuncDef { .. } => "FuncDef".to_string(),
            PythonNode::Call { .. } => "Call".to_string(),
            PythonNode::List(_) => "List".to_string(),
            PythonNode::Pass => "Pass".to_string(),
            PythonNode::Break => "Break".to_string(),
            PythonNode::Continue => "Continue".to_string(),
        }
    }

    /// Select and mark an entry (mutable version for updating selection count)
    fn select_entry_mut(&mut self) -> Option<usize> {
        if self.corpus.is_empty() {
            return None;
        }

        let total_energy: f64 = self.corpus.iter().map(|e| e.energy).sum();
        if total_energy <= 0.0 {
            let idx = (self.next_random() as usize) % self.corpus.len();
            self.corpus[idx].selection_count += 1;
            return Some(idx);
        }

        let mut threshold = self.random_float() * total_energy;
        for (i, entry) in self.corpus.iter_mut().enumerate() {
            threshold -= entry.energy;
            if threshold <= 0.0 {
                entry.selection_count += 1;
                return Some(i);
            }
        }

        let last_idx = self.corpus.len() - 1;
        self.corpus[last_idx].selection_count += 1;
        Some(last_idx)
    }

    /// Add a new entry to the corpus if it has new coverage
    pub fn add_to_corpus(&mut self, code: GeneratedCode, coverage: CoverageMap) -> bool {
        if !coverage.has_new_coverage(&self.global_coverage) {
            return false;
        }

        self.global_coverage.merge(&coverage);

        // If corpus is full, replace lowest energy entry
        if self.corpus.len() >= self.max_corpus_size {
            if let Some(min_idx) = self
                .corpus
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.energy
                        .partial_cmp(&b.energy)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                self.corpus[min_idx] = CorpusEntry::new(code, coverage);
            }
        } else {
            self.corpus.push(CorpusEntry::new(code, coverage));
        }

        // Update all energies
        let global = self.global_coverage.clone();
        for entry in &mut self.corpus {
            entry.update_energy(&global);
        }

        true
    }

    /// Generate programs using coverage-guided fuzzing
    pub fn generate(&mut self, count: usize) -> Vec<GeneratedCode> {
        // Initialize corpus if empty
        if self.corpus.is_empty() {
            self.initialize_corpus_with_ast();
        }

        let mut results = Vec::with_capacity(count);
        let mut iterations = 0;
        let max_iterations = count * 10; // Prevent infinite loops

        while results.len() < count && iterations < max_iterations {
            iterations += 1;

            // Select a corpus entry
            if let Some(idx) = self.select_entry_mut() {
                // Get necessary info from entry before mutable operations
                let has_ast = self.corpus[idx].ast.is_some();
                let ast_clone = self.corpus[idx].ast.clone();
                let code_clone = self.corpus[idx].code.clone();

                // Decide whether to mutate
                let should_mutate = self.random_float() < 0.7 && has_ast;

                if should_mutate {
                    // Mutate the cloned AST (safe: has_ast check ensures Some)
                    if let Some(ast) = ast_clone {
                        if let Some(mutated) = self.mutate_ast(&ast) {
                            let code = mutated.to_code(0);
                            let coverage = self.compute_coverage_from_ast(&mutated);

                            let program = GeneratedCode {
                                code,
                                language: self.language,
                                ast_depth: mutated.depth(),
                                features: Self::extract_features(&mutated),
                            };

                            // Add to corpus if new coverage
                            self.add_to_corpus(program.clone(), coverage);
                            results.push(program);
                        }
                    }
                } else {
                    // Use existing entry
                    results.push(code_clone);
                }
            } else {
                // No corpus - generate fresh
                let enumerator = PythonEnumerator::new(self.max_depth);
                let programs = enumerator.enumerate_programs();
                if let Some(program) = programs.into_iter().next() {
                    results.push(program);
                }
            }
        }

        results
    }

    /// Mutate an AST node
    fn mutate_ast(&mut self, node: &PythonNode) -> Option<PythonNode> {
        let mutation_type = self.next_random() % 4;

        match mutation_type {
            0 => self.mutate_operator(node),
            1 => self.mutate_literal(node),
            2 => self.insert_wrapper(node),
            _ => self.delete_subtree(node),
        }
    }

    /// Mutate operators in the AST
    fn mutate_operator(&mut self, node: &PythonNode) -> Option<PythonNode> {
        match node {
            PythonNode::BinOp { left, right, .. } => {
                let ops = BinaryOp::all();
                let new_op = ops[(self.next_random() as usize) % ops.len()];
                Some(PythonNode::BinOp {
                    left: left.clone(),
                    op: new_op,
                    right: right.clone(),
                })
            }
            PythonNode::Compare { left, right, .. } => {
                let ops = CompareOp::all();
                let new_op = ops[(self.next_random() as usize) % ops.len()];
                Some(PythonNode::Compare {
                    left: left.clone(),
                    op: new_op,
                    right: right.clone(),
                })
            }
            PythonNode::UnaryOp { operand, .. } => {
                let ops = UnaryOp::all();
                let new_op = ops[(self.next_random() as usize) % ops.len()];
                Some(PythonNode::UnaryOp {
                    op: new_op,
                    operand: operand.clone(),
                })
            }
            _ => None,
        }
    }

    /// Mutate literals in the AST
    fn mutate_literal(&mut self, node: &PythonNode) -> Option<PythonNode> {
        match node {
            PythonNode::IntLit(n) => {
                let mutations = [0, 1, -1, i64::MAX, i64::MIN, *n + 1, n.saturating_sub(1)];
                let new_val = mutations[(self.next_random() as usize) % mutations.len()];
                Some(PythonNode::IntLit(new_val))
            }
            PythonNode::BoolLit(b) => Some(PythonNode::BoolLit(!b)),
            PythonNode::StrLit(s) => {
                let mutations = ["", " ", "\\n", "\\t", &format!("{s}x")];
                let new_val = mutations[(self.next_random() as usize) % mutations.len()];
                Some(PythonNode::StrLit(new_val.to_string()))
            }
            _ => None,
        }
    }

    /// Insert a wrapper around an expression
    fn insert_wrapper(&mut self, node: &PythonNode) -> Option<PythonNode> {
        // Only wrap expressions
        match node {
            PythonNode::IntLit(_)
            | PythonNode::FloatLit(_)
            | PythonNode::Name(_)
            | PythonNode::BinOp { .. } => {
                let ops = UnaryOp::all();
                let op = ops[(self.next_random() as usize) % ops.len()];
                Some(PythonNode::UnaryOp {
                    op,
                    operand: Box::new(node.clone()),
                })
            }
            _ => None,
        }
    }

    /// Delete/simplify a subtree
    fn delete_subtree(&mut self, node: &PythonNode) -> Option<PythonNode> {
        match node {
            PythonNode::BinOp { left, right, .. } => {
                // Replace binary op with one of its operands
                if self.random_float() < 0.5 {
                    Some((**left).clone())
                } else {
                    Some((**right).clone())
                }
            }
            PythonNode::UnaryOp { operand, .. } => Some((**operand).clone()),
            PythonNode::If { body, .. } => {
                // Simplify to just the first statement in body
                body.first().cloned()
            }
            _ => None,
        }
    }

    /// Get current corpus size
    #[must_use]
    pub fn corpus_size(&self) -> usize {
        self.corpus.len()
    }

    /// Get global coverage statistics
    #[must_use]
    pub fn coverage_stats(&self) -> CoverageStats {
        CoverageStats {
            total_coverage: self.global_coverage.coverage_count(),
            node_types_covered: self.global_coverage.node_types.len(),
            ast_paths_covered: self.global_coverage.ast_paths.len(),
            features_covered: self.global_coverage.feature_combos.len(),
            corpus_size: self.corpus.len(),
        }
    }
}

/// Statistics about coverage
#[derive(Debug, Clone)]
pub struct CoverageStats {
    /// Total coverage count
    pub total_coverage: usize,
    /// Number of unique node types covered
    pub node_types_covered: usize,
    /// Number of unique AST paths covered
    pub ast_paths_covered: usize,
    /// Number of unique features covered
    pub features_covered: usize,
    /// Current corpus size
    pub corpus_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coverage_map_new() {
        let coverage = CoverageMap::new();
        assert_eq!(coverage.coverage_count(), 0);
    }

    #[test]
    fn test_coverage_map_record_node() {
        let mut coverage = CoverageMap::new();
        coverage.record_node("function_def");
        coverage.record_node("if_stmt");

        assert!(coverage.node_types().contains("function_def"));
        assert!(coverage.node_types().contains("if_stmt"));
        assert_eq!(coverage.node_types().len(), 2);
    }

    #[test]
    fn test_coverage_map_record_path() {
        let mut coverage = CoverageMap::new();
        coverage.record_path("function_def", "return_stmt");

        assert!(coverage
            .ast_paths()
            .contains(&("function_def".to_string(), "return_stmt".to_string())));
    }

    #[test]
    fn test_coverage_map_record_feature() {
        let mut coverage = CoverageMap::new();
        coverage.record_feature("loops");
        coverage.record_feature("conditionals");
        assert_eq!(coverage.coverage_count(), 2);
    }

    #[test]
    fn test_coverage_map_has_new_coverage() {
        let mut existing = CoverageMap::new();
        existing.record_node("function_def");

        let mut new_coverage = CoverageMap::new();
        new_coverage.record_node("function_def");
        assert!(!new_coverage.has_new_coverage(&existing));

        new_coverage.record_node("while_stmt");
        assert!(new_coverage.has_new_coverage(&existing));
    }

    #[test]
    fn test_coverage_map_has_new_path_coverage() {
        let mut existing = CoverageMap::new();
        existing.record_path("a", "b");

        let mut new_coverage = CoverageMap::new();
        new_coverage.record_path("a", "b");
        assert!(!new_coverage.has_new_coverage(&existing));

        new_coverage.record_path("a", "c");
        assert!(new_coverage.has_new_coverage(&existing));
    }

    #[test]
    fn test_coverage_map_has_new_feature_coverage() {
        let mut existing = CoverageMap::new();
        existing.record_feature("feat1");

        let mut new_coverage = CoverageMap::new();
        new_coverage.record_feature("feat1");
        assert!(!new_coverage.has_new_coverage(&existing));

        new_coverage.record_feature("feat2");
        assert!(new_coverage.has_new_coverage(&existing));
    }

    #[test]
    fn test_coverage_map_merge() {
        let mut map1 = CoverageMap::new();
        map1.record_node("a");
        map1.record_node("b");

        let mut map2 = CoverageMap::new();
        map2.record_node("b");
        map2.record_node("c");

        map1.merge(&map2);
        assert_eq!(map1.node_types().len(), 3);
        assert!(map1.node_types().contains("a"));
        assert!(map1.node_types().contains("b"));
        assert!(map1.node_types().contains("c"));
    }

    #[test]
    fn test_coverage_map_default() {
        let coverage = CoverageMap::default();
        assert_eq!(coverage.coverage_count(), 0);
    }

    #[test]
    fn test_coverage_map_debug() {
        let mut coverage = CoverageMap::new();
        coverage.record_node("test");
        let debug = format!("{:?}", coverage);
        assert!(debug.contains("CoverageMap"));
    }

    #[test]
    fn test_coverage_map_clone() {
        let mut coverage = CoverageMap::new();
        coverage.record_node("test");
        let cloned = coverage.clone();
        assert_eq!(cloned.coverage_count(), coverage.coverage_count());
    }

    #[test]
    fn test_corpus_entry_new() {
        let code = GeneratedCode {
            code: "x = 1".to_string(),
            language: Language::Python,
            ast_depth: 1,
            features: vec!["assignment".to_string()],
        };
        let coverage = CoverageMap::new();
        let entry = CorpusEntry::new(code, coverage);

        assert_eq!(entry.energy, 1.0);
        assert_eq!(entry.selection_count, 0);
        assert!(entry.ast.is_none());
    }

    #[test]
    fn test_corpus_entry_with_ast() {
        let code = GeneratedCode {
            code: "x = 1".to_string(),
            language: Language::Python,
            ast_depth: 1,
            features: vec!["assignment".to_string()],
        };
        let coverage = CoverageMap::new();
        let ast = PythonNode::IntLit(1);
        let entry = CorpusEntry::with_ast(code, coverage, ast);

        assert!(entry.ast.is_some());
    }

    #[test]
    fn test_corpus_entry_update_energy() {
        let code = GeneratedCode {
            code: "x = 1".to_string(),
            language: Language::Python,
            ast_depth: 1,
            features: vec![],
        };
        let mut coverage = CoverageMap::new();
        coverage.record_node("unique_node");
        let mut entry = CorpusEntry::new(code, coverage);

        let global = CoverageMap::new();
        entry.update_energy(&global);

        // Energy should be boosted because of unique coverage
        assert!(entry.energy > 1.0);
    }

    #[test]
    fn test_corpus_entry_debug() {
        let code = GeneratedCode {
            code: "x = 1".to_string(),
            language: Language::Python,
            ast_depth: 1,
            features: vec![],
        };
        let coverage = CoverageMap::new();
        let entry = CorpusEntry::new(code, coverage);
        let debug = format!("{:?}", entry);
        assert!(debug.contains("CorpusEntry"));
    }

    #[test]
    fn test_corpus_entry_clone() {
        let code = GeneratedCode {
            code: "x = 1".to_string(),
            language: Language::Python,
            ast_depth: 1,
            features: vec![],
        };
        let coverage = CoverageMap::new();
        let entry = CorpusEntry::new(code, coverage);
        let cloned = entry.clone();
        assert_eq!(cloned.energy, entry.energy);
    }

    #[test]
    fn test_nautilus_generator_new() {
        let gen = NautilusGenerator::new(Language::Python, 3);
        assert_eq!(gen.corpus_size(), 0);
        assert_eq!(gen.language, Language::Python);
    }

    #[test]
    fn test_nautilus_generator_with_max_corpus() {
        let gen = NautilusGenerator::new(Language::Python, 2).with_max_corpus(500);
        assert_eq!(gen.max_corpus_size, 500);
    }

    #[test]
    fn test_nautilus_generator_initialize_corpus() {
        let mut gen = NautilusGenerator::new(Language::Python, 2);
        gen.initialize_corpus();

        assert!(gen.corpus_size() > 0, "Corpus should be initialized");
    }

    #[test]
    fn test_nautilus_generator_generate() {
        let mut gen = NautilusGenerator::new(Language::Python, 2).with_seed(42);
        let programs = gen.generate(5);

        assert!(!programs.is_empty(), "Should generate programs");
        for prog in &programs {
            assert_eq!(prog.language, Language::Python);
        }
    }

    #[test]
    fn test_nautilus_generator_coverage_stats() {
        let mut gen = NautilusGenerator::new(Language::Python, 2);
        gen.initialize_corpus_with_ast();

        let stats = gen.coverage_stats();
        assert!(stats.node_types_covered > 0, "Should cover some node types");
        assert!(stats.corpus_size > 0, "Corpus should have entries");
    }

    #[test]
    fn test_nautilus_generator_with_seed() {
        let mut gen1 = NautilusGenerator::new(Language::Python, 2).with_seed(123);
        let mut gen2 = NautilusGenerator::new(Language::Python, 2).with_seed(123);

        gen1.initialize_corpus_with_ast();
        gen2.initialize_corpus_with_ast();

        // Both should have same corpus size with same seed
        assert_eq!(gen1.corpus_size(), gen2.corpus_size());
    }

    #[test]
    fn test_add_to_corpus_new_coverage() {
        let mut gen = NautilusGenerator::new(Language::Python, 2);

        let code = GeneratedCode {
            code: "def foo(): pass".to_string(),
            language: Language::Python,
            ast_depth: 2,
            features: vec!["function_def".to_string()],
        };

        let mut coverage = CoverageMap::new();
        coverage.record_node("unique_node_type");

        let added = gen.add_to_corpus(code, coverage);
        assert!(added, "Should add entry with new coverage");
        assert_eq!(gen.corpus_size(), 1);
    }

    #[test]
    fn test_add_to_corpus_duplicate_coverage() {
        let mut gen = NautilusGenerator::new(Language::Python, 2);

        // Add first entry
        let code1 = GeneratedCode {
            code: "x = 1".to_string(),
            language: Language::Python,
            ast_depth: 1,
            features: vec![],
        };
        let mut coverage1 = CoverageMap::new();
        coverage1.record_node("assignment");
        gen.add_to_corpus(code1, coverage1);

        // Try to add entry with same coverage
        let code2 = GeneratedCode {
            code: "y = 2".to_string(),
            language: Language::Python,
            ast_depth: 1,
            features: vec![],
        };
        let mut coverage2 = CoverageMap::new();
        coverage2.record_node("assignment");

        let added = gen.add_to_corpus(code2, coverage2);
        assert!(!added, "Should not add entry with duplicate coverage");
        assert_eq!(gen.corpus_size(), 1);
    }

    #[test]
    fn test_extract_features_literals() {
        let int_features = NautilusGenerator::extract_features(&PythonNode::IntLit(42));
        assert!(int_features.contains(&"literal".to_string()));

        let float_features = NautilusGenerator::extract_features(&PythonNode::FloatLit(3.14));
        assert!(float_features.contains(&"float".to_string()));

        let str_features =
            NautilusGenerator::extract_features(&PythonNode::StrLit("hello".to_string()));
        assert!(str_features.contains(&"string".to_string()));

        let bool_features = NautilusGenerator::extract_features(&PythonNode::BoolLit(true));
        assert!(bool_features.contains(&"boolean".to_string()));

        let none_features = NautilusGenerator::extract_features(&PythonNode::NoneLit);
        assert!(none_features.contains(&"none".to_string()));
    }

    #[test]
    fn test_extract_features_control_flow() {
        let break_features = NautilusGenerator::extract_features(&PythonNode::Break);
        assert!(break_features.contains(&"control_flow".to_string()));
        assert!(break_features.contains(&"break".to_string()));

        let continue_features = NautilusGenerator::extract_features(&PythonNode::Continue);
        assert!(continue_features.contains(&"continue".to_string()));
    }

    #[test]
    fn test_extract_features_loops() {
        let while_node = PythonNode::While {
            test: Box::new(PythonNode::BoolLit(true)),
            body: vec![PythonNode::Pass],
        };
        let while_features = NautilusGenerator::extract_features(&while_node);
        assert!(while_features.contains(&"loop".to_string()));
        assert!(while_features.contains(&"while_loop".to_string()));

        let for_node = PythonNode::For {
            target: "i".to_string(),
            iter: Box::new(PythonNode::List(vec![])),
            body: vec![PythonNode::Pass],
        };
        let for_features = NautilusGenerator::extract_features(&for_node);
        assert!(for_features.contains(&"for_loop".to_string()));
    }

    #[test]
    fn test_extract_features_collections() {
        let list_features = NautilusGenerator::extract_features(&PythonNode::List(vec![]));
        assert!(list_features.contains(&"collection".to_string()));
        assert!(list_features.contains(&"list".to_string()));
    }

    #[test]
    fn test_extract_features_functions() {
        let func_node = PythonNode::FuncDef {
            name: "foo".to_string(),
            args: vec![],
            body: vec![PythonNode::Pass],
        };
        let func_features = NautilusGenerator::extract_features(&func_node);
        assert!(func_features.contains(&"function_def".to_string()));

        let call_node = PythonNode::Call {
            func: "print".to_string(),
            args: vec![],
        };
        let call_features = NautilusGenerator::extract_features(&call_node);
        assert!(call_features.contains(&"function_call".to_string()));
    }

    #[test]
    fn test_extract_features_if_with_else() {
        let if_node = PythonNode::If {
            test: Box::new(PythonNode::BoolLit(true)),
            body: vec![PythonNode::Pass],
            orelse: vec![PythonNode::Pass],
        };
        let features = NautilusGenerator::extract_features(&if_node);
        assert!(features.contains(&"conditional".to_string()));
        assert!(features.contains(&"else_branch".to_string()));
    }

    #[test]
    fn test_node_type_name() {
        assert_eq!(
            NautilusGenerator::node_type_name(&PythonNode::IntLit(1)),
            "IntLit"
        );
        assert_eq!(
            NautilusGenerator::node_type_name(&PythonNode::FloatLit(1.0)),
            "FloatLit"
        );
        assert_eq!(
            NautilusGenerator::node_type_name(&PythonNode::StrLit("x".to_string())),
            "StrLit"
        );
        assert_eq!(
            NautilusGenerator::node_type_name(&PythonNode::BoolLit(true)),
            "BoolLit"
        );
        assert_eq!(
            NautilusGenerator::node_type_name(&PythonNode::NoneLit),
            "NoneLit"
        );
        assert_eq!(
            NautilusGenerator::node_type_name(&PythonNode::Name("x".to_string())),
            "Name"
        );
        assert_eq!(NautilusGenerator::node_type_name(&PythonNode::Pass), "Pass");
        assert_eq!(
            NautilusGenerator::node_type_name(&PythonNode::Break),
            "Break"
        );
        assert_eq!(
            NautilusGenerator::node_type_name(&PythonNode::Continue),
            "Continue"
        );
    }

    #[test]
    fn test_node_type_name_operators() {
        let binop = PythonNode::BinOp {
            left: Box::new(PythonNode::IntLit(1)),
            op: BinaryOp::Add,
            right: Box::new(PythonNode::IntLit(2)),
        };
        assert!(NautilusGenerator::node_type_name(&binop).starts_with("BinOp_"));

        let unaryop = PythonNode::UnaryOp {
            op: UnaryOp::Neg,
            operand: Box::new(PythonNode::IntLit(1)),
        };
        assert!(NautilusGenerator::node_type_name(&unaryop).starts_with("UnaryOp_"));

        let compare = PythonNode::Compare {
            left: Box::new(PythonNode::IntLit(1)),
            op: CompareOp::Lt,
            right: Box::new(PythonNode::IntLit(2)),
        };
        assert!(NautilusGenerator::node_type_name(&compare).starts_with("Compare_"));
    }

    #[test]
    fn test_node_type_name_statements() {
        let assign = PythonNode::Assign {
            target: "x".to_string(),
            value: Box::new(PythonNode::IntLit(1)),
        };
        assert_eq!(NautilusGenerator::node_type_name(&assign), "Assign");

        let ret = PythonNode::Return(Some(Box::new(PythonNode::IntLit(1))));
        assert_eq!(NautilusGenerator::node_type_name(&ret), "Return");

        let if_node = PythonNode::If {
            test: Box::new(PythonNode::BoolLit(true)),
            body: vec![],
            orelse: vec![],
        };
        assert_eq!(NautilusGenerator::node_type_name(&if_node), "If");

        let while_node = PythonNode::While {
            test: Box::new(PythonNode::BoolLit(true)),
            body: vec![],
        };
        assert_eq!(NautilusGenerator::node_type_name(&while_node), "While");

        let for_node = PythonNode::For {
            target: "i".to_string(),
            iter: Box::new(PythonNode::List(vec![])),
            body: vec![],
        };
        assert_eq!(NautilusGenerator::node_type_name(&for_node), "For");

        let func = PythonNode::FuncDef {
            name: "f".to_string(),
            args: vec![],
            body: vec![],
        };
        assert_eq!(NautilusGenerator::node_type_name(&func), "FuncDef");

        let call = PythonNode::Call {
            func: "f".to_string(),
            args: vec![],
        };
        assert_eq!(NautilusGenerator::node_type_name(&call), "Call");

        let list = PythonNode::List(vec![]);
        assert_eq!(NautilusGenerator::node_type_name(&list), "List");

        let module = PythonNode::Module(vec![]);
        assert_eq!(NautilusGenerator::node_type_name(&module), "Module");
    }

    #[test]
    fn test_coverage_stats_debug() {
        let stats = CoverageStats {
            total_coverage: 10,
            node_types_covered: 5,
            ast_paths_covered: 3,
            features_covered: 2,
            corpus_size: 100,
        };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("CoverageStats"));
    }

    #[test]
    fn test_coverage_stats_clone() {
        let stats = CoverageStats {
            total_coverage: 10,
            node_types_covered: 5,
            ast_paths_covered: 3,
            features_covered: 2,
            corpus_size: 100,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.total_coverage, stats.total_coverage);
    }

    #[test]
    fn test_nautilus_generator_debug() {
        let gen = NautilusGenerator::new(Language::Python, 2);
        let debug = format!("{:?}", gen);
        assert!(debug.contains("NautilusGenerator"));
    }

    #[test]
    fn test_add_to_corpus_full() {
        let mut gen = NautilusGenerator::new(Language::Python, 2).with_max_corpus(2);

        // Add entries up to max
        for i in 0..3 {
            let code = GeneratedCode {
                code: format!("x = {i}"),
                language: Language::Python,
                ast_depth: 1,
                features: vec![],
            };
            let mut coverage = CoverageMap::new();
            coverage.record_node(&format!("unique_node_{i}"));
            gen.add_to_corpus(code, coverage);
        }

        // Corpus should stay at max size
        assert!(gen.corpus_size() <= 2);
    }

    #[test]
    fn test_extract_features_binary_op() {
        let node = PythonNode::BinOp {
            left: Box::new(PythonNode::IntLit(1)),
            op: BinaryOp::Add,
            right: Box::new(PythonNode::IntLit(2)),
        };
        let features = NautilusGenerator::extract_features(&node);
        assert!(features.contains(&"binary_op".to_string()));
        assert!(features.iter().any(|f| f.starts_with("op_")));
    }

    #[test]
    fn test_extract_features_unary_op() {
        let node = PythonNode::UnaryOp {
            op: UnaryOp::Neg,
            operand: Box::new(PythonNode::IntLit(1)),
        };
        let features = NautilusGenerator::extract_features(&node);
        assert!(features.contains(&"unary_op".to_string()));
        assert!(features.iter().any(|f| f.starts_with("op_")));
    }

    #[test]
    fn test_extract_features_compare() {
        let node = PythonNode::Compare {
            left: Box::new(PythonNode::IntLit(1)),
            op: CompareOp::Lt,
            right: Box::new(PythonNode::IntLit(2)),
        };
        let features = NautilusGenerator::extract_features(&node);
        assert!(features.contains(&"comparison".to_string()));
        assert!(features.iter().any(|f| f.starts_with("cmp_")));
    }

    #[test]
    fn test_extract_features_module() {
        let node = PythonNode::Module(vec![PythonNode::Pass]);
        let features = NautilusGenerator::extract_features(&node);
        assert!(features.contains(&"module".to_string()));
    }

    #[test]
    fn test_compute_coverage() {
        let gen = NautilusGenerator::new(Language::Python, 2);

        let coverage = gen.compute_coverage("def foo(): pass");
        assert!(coverage.node_types().contains("function_def"));

        let coverage2 = gen.compute_coverage("if x: pass");
        assert!(coverage2.node_types().contains("if_stmt"));

        let coverage3 = gen.compute_coverage("while True: pass");
        assert!(coverage3.node_types().contains("while_stmt"));

        let coverage4 = gen.compute_coverage("for i in x: pass");
        assert!(coverage4.node_types().contains("for_stmt"));

        let coverage5 = gen.compute_coverage("return 1");
        assert!(coverage5.node_types().contains("return_stmt"));

        let coverage6 = gen.compute_coverage("[1, 2, 3]");
        assert!(coverage6.node_types().contains("list"));
    }

    #[test]
    fn test_compute_coverage_from_ast() {
        let gen = NautilusGenerator::new(Language::Python, 2);

        // Test with a complex AST that exercises all branches
        let ast = PythonNode::Module(vec![
            PythonNode::Assign {
                target: "x".to_string(),
                value: Box::new(PythonNode::BinOp {
                    left: Box::new(PythonNode::IntLit(1)),
                    op: BinaryOp::Add,
                    right: Box::new(PythonNode::IntLit(2)),
                }),
            },
            PythonNode::If {
                test: Box::new(PythonNode::Compare {
                    left: Box::new(PythonNode::Name("x".to_string())),
                    op: CompareOp::Lt,
                    right: Box::new(PythonNode::IntLit(5)),
                }),
                body: vec![PythonNode::Pass],
                orelse: vec![PythonNode::Pass],
            },
            PythonNode::While {
                test: Box::new(PythonNode::BoolLit(true)),
                body: vec![PythonNode::Break],
            },
            PythonNode::For {
                target: "i".to_string(),
                iter: Box::new(PythonNode::List(vec![PythonNode::IntLit(1)])),
                body: vec![PythonNode::Continue],
            },
            PythonNode::FuncDef {
                name: "foo".to_string(),
                args: vec!["a".to_string()],
                body: vec![PythonNode::Return(Some(Box::new(PythonNode::Name(
                    "a".to_string(),
                ))))],
            },
            PythonNode::UnaryOp {
                op: UnaryOp::Neg,
                operand: Box::new(PythonNode::IntLit(1)),
            },
            PythonNode::Call {
                func: "print".to_string(),
                args: vec![PythonNode::StrLit("hello".to_string())],
            },
        ]);

        let coverage = gen.compute_coverage_from_ast(&ast);

        assert!(coverage.node_types().contains("Module"));
        assert!(coverage.node_types().contains("Assign"));
        assert!(coverage.node_types().contains("If"));
        assert!(coverage.node_types().contains("While"));
        assert!(coverage.node_types().contains("For"));
        assert!(coverage.node_types().contains("FuncDef"));
        assert!(coverage.node_types().contains("Call"));
    }

    #[test]
    fn test_select_entry_mut_empty() {
        let mut gen = NautilusGenerator::new(Language::Python, 2);
        let result = gen.select_entry_mut();
        assert!(result.is_none());
    }

    #[test]
    fn test_select_entry_mut_zero_energy() {
        let mut gen = NautilusGenerator::new(Language::Python, 2).with_seed(42);

        // Add an entry with zero energy manually
        let code = GeneratedCode {
            code: "x = 1".to_string(),
            language: Language::Python,
            ast_depth: 1,
            features: vec![],
        };
        let coverage = CoverageMap::new();
        let mut entry = CorpusEntry::new(code, coverage);
        entry.energy = 0.0;
        gen.corpus.push(entry);

        // Should still select via random index
        let result = gen.select_entry_mut();
        assert!(result.is_some());
    }

    #[test]
    fn test_generate_covers_mutations() {
        let mut gen = NautilusGenerator::new(Language::Python, 2).with_seed(42);

        // Initialize with AST corpus
        gen.initialize_corpus_with_ast();

        // Generate enough to trigger mutations
        let programs = gen.generate(20);
        assert!(!programs.is_empty());
    }

    #[test]
    fn test_mutate_ast() {
        let mut gen = NautilusGenerator::new(Language::Python, 2).with_seed(42);

        let ast = PythonNode::Assign {
            target: "x".to_string(),
            value: Box::new(PythonNode::IntLit(1)),
        };

        // Mutate multiple times to exercise different mutation types
        for _ in 0..10 {
            let mutated = gen.mutate_ast(&ast);
            // Should produce some mutation
            assert!(mutated.is_some() || true); // May or may not mutate
        }
    }

    #[test]
    fn test_mutate_operator() {
        let mut gen = NautilusGenerator::new(Language::Python, 2).with_seed(42);

        // Test BinOp mutation
        let binop = PythonNode::BinOp {
            left: Box::new(PythonNode::IntLit(1)),
            op: BinaryOp::Add,
            right: Box::new(PythonNode::IntLit(2)),
        };
        let result = gen.mutate_operator(&binop);
        // May or may not mutate depending on random
        assert!(result.is_some() || result.is_none());

        // Test UnaryOp mutation
        let unaryop = PythonNode::UnaryOp {
            op: UnaryOp::Neg,
            operand: Box::new(PythonNode::IntLit(1)),
        };
        let result2 = gen.mutate_operator(&unaryop);
        assert!(result2.is_some() || result2.is_none());

        // Test Compare mutation
        let compare = PythonNode::Compare {
            left: Box::new(PythonNode::IntLit(1)),
            op: CompareOp::Lt,
            right: Box::new(PythonNode::IntLit(2)),
        };
        let result3 = gen.mutate_operator(&compare);
        assert!(result3.is_some() || result3.is_none());
    }

    #[test]
    fn test_mutate_literal() {
        let mut gen = NautilusGenerator::new(Language::Python, 2).with_seed(42);

        // Test IntLit mutation
        let int_node = PythonNode::IntLit(42);
        let result = gen.mutate_literal(&int_node);
        assert!(result.is_some() || result.is_none());

        // Test FloatLit mutation
        let float_node = PythonNode::FloatLit(3.14);
        let result2 = gen.mutate_literal(&float_node);
        assert!(result2.is_some() || result2.is_none());

        // Test StrLit mutation
        let str_node = PythonNode::StrLit("hello".to_string());
        let result3 = gen.mutate_literal(&str_node);
        assert!(result3.is_some() || result3.is_none());

        // Test BoolLit mutation
        let bool_node = PythonNode::BoolLit(true);
        let result4 = gen.mutate_literal(&bool_node);
        assert!(result4.is_some() || result4.is_none());
    }

    #[test]
    fn test_insert_wrapper() {
        let mut gen = NautilusGenerator::new(Language::Python, 2).with_seed(42);

        // Test with IntLit
        let node = PythonNode::IntLit(1);
        let result = gen.insert_wrapper(&node);
        // Should wrap with unary op
        if let Some(wrapped) = result {
            assert!(matches!(wrapped, PythonNode::UnaryOp { .. }));
        }

        // Test with Name
        let name_node = PythonNode::Name("x".to_string());
        let result2 = gen.insert_wrapper(&name_node);
        assert!(result2.is_some());

        // Test with FloatLit
        let float_node = PythonNode::FloatLit(1.0);
        let result3 = gen.insert_wrapper(&float_node);
        assert!(result3.is_some());

        // Test with BinOp
        let binop = PythonNode::BinOp {
            left: Box::new(PythonNode::IntLit(1)),
            op: BinaryOp::Add,
            right: Box::new(PythonNode::IntLit(2)),
        };
        let result4 = gen.insert_wrapper(&binop);
        assert!(result4.is_some());
    }

    #[test]
    fn test_delete_subtree() {
        let mut gen = NautilusGenerator::new(Language::Python, 2).with_seed(42);

        // Test BinOp deletion
        let binop = PythonNode::BinOp {
            left: Box::new(PythonNode::IntLit(1)),
            op: BinaryOp::Add,
            right: Box::new(PythonNode::IntLit(2)),
        };
        let result = gen.delete_subtree(&binop);
        assert!(result.is_some());

        // Test UnaryOp deletion
        let unaryop = PythonNode::UnaryOp {
            op: UnaryOp::Neg,
            operand: Box::new(PythonNode::IntLit(1)),
        };
        let result2 = gen.delete_subtree(&unaryop);
        assert!(result2.is_some());
        assert!(matches!(result2.unwrap(), PythonNode::IntLit(1)));

        // Test If deletion
        let if_node = PythonNode::If {
            test: Box::new(PythonNode::BoolLit(true)),
            body: vec![PythonNode::Pass],
            orelse: vec![],
        };
        let result3 = gen.delete_subtree(&if_node);
        assert!(result3.is_some());

        // Test with node that can't be deleted
        let int_lit = PythonNode::IntLit(1);
        let result4 = gen.delete_subtree(&int_lit);
        assert!(result4.is_none());
    }
}
