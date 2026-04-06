//! AST-based mutation operators
//!
//! Provides proper AST-level mutations using the `PythonNode` representation.
//! Unlike string-based mutations, these guarantee syntactic validity.

// Allow match arms with same bodies - intentional for different AST node types
// Allow self_only_used_in_recursion - the methods need &self for trait consistency
#![allow(clippy::match_same_arms)]
#![allow(clippy::self_only_used_in_recursion)]

use crate::generator::{BinaryOp, CompareOp, PythonNode, UnaryOp};

use super::MutationOperator;

/// A mutation applied to an AST node
#[derive(Debug, Clone)]
pub struct AstMutation {
    /// The operator used
    pub operator: MutationOperator,
    /// Path to the mutated node (indices in children)
    pub path: Vec<usize>,
    /// Description of the mutation
    pub description: String,
    /// The mutated AST
    pub mutated_ast: PythonNode,
}

/// AST-based mutator for Python code
#[derive(Debug, Default)]
pub struct AstMutator {
    /// Enabled mutation operators
    enabled_operators: Vec<MutationOperator>,
}

impl AstMutator {
    /// Create a new AST mutator with all operators enabled
    #[must_use]
    pub fn new() -> Self {
        Self {
            enabled_operators: MutationOperator::all(),
        }
    }

    /// Create a mutator with specific operators enabled
    #[must_use]
    pub fn with_operators(operators: Vec<MutationOperator>) -> Self {
        Self {
            enabled_operators: operators,
        }
    }

    /// Generate all possible mutations for the given AST
    #[must_use]
    pub fn mutate(&self, ast: &PythonNode) -> Vec<AstMutation> {
        let mut mutations = Vec::new();

        for operator in &self.enabled_operators {
            let op_mutations = match operator {
                MutationOperator::Aor => self.apply_aor(ast, vec![]),
                MutationOperator::Ror => self.apply_ror(ast, vec![]),
                MutationOperator::Lor => self.apply_lor(ast, vec![]),
                MutationOperator::Uoi => self.apply_uoi(ast, vec![]),
                MutationOperator::Abs => self.apply_abs(ast, vec![]),
                MutationOperator::Sdl => self.apply_sdl(ast, vec![]),
                MutationOperator::Svr => self.apply_svr(ast, vec![]),
                MutationOperator::Bsr => self.apply_bsr(ast, vec![]),
            };
            mutations.extend(op_mutations);
        }

        mutations
    }

    /// AOR: Arithmetic Operator Replacement
    /// `a + b` → `a - b`, `a * b`, `a / b`, etc.
    fn apply_aor(&self, ast: &PythonNode, path: Vec<usize>) -> Vec<AstMutation> {
        let mut mutations = Vec::new();

        match ast {
            PythonNode::BinOp { left, op, right } => {
                // Generate mutations for this BinOp
                let replacements = arithmetic_replacements(*op);
                for new_op in replacements {
                    let mutated = PythonNode::BinOp {
                        left: left.clone(),
                        op: new_op,
                        right: right.clone(),
                    };
                    mutations.push(AstMutation {
                        operator: MutationOperator::Aor,
                        path: path.clone(),
                        description: format!("Replace {} with {}", op.to_str(), new_op.to_str()),
                        mutated_ast: mutated,
                    });
                }

                // Recurse into children
                let mut left_path = path.clone();
                left_path.push(0);
                mutations.extend(self.apply_aor(left, left_path));

                let mut right_path = path;
                right_path.push(1);
                mutations.extend(self.apply_aor(right, right_path));
            }
            PythonNode::Module(stmts) => {
                for (i, stmt) in stmts.iter().enumerate() {
                    let mut child_path = path.clone();
                    child_path.push(i);
                    mutations.extend(self.apply_aor(stmt, child_path));
                }
            }
            PythonNode::Assign { value, .. } => {
                let mut child_path = path;
                child_path.push(0);
                mutations.extend(self.apply_aor(value, child_path));
            }
            PythonNode::If { test, body, orelse } => {
                let mut test_path = path.clone();
                test_path.push(0);
                mutations.extend(self.apply_aor(test, test_path));

                for (i, stmt) in body.iter().enumerate() {
                    let mut child_path = path.clone();
                    child_path.push(1 + i);
                    mutations.extend(self.apply_aor(stmt, child_path));
                }

                for (i, stmt) in orelse.iter().enumerate() {
                    let mut child_path = path.clone();
                    child_path.push(1 + body.len() + i);
                    mutations.extend(self.apply_aor(stmt, child_path));
                }
            }
            PythonNode::Return(Some(value)) => {
                let mut child_path = path;
                child_path.push(0);
                mutations.extend(self.apply_aor(value, child_path));
            }
            PythonNode::UnaryOp { operand, .. } => {
                let mut child_path = path;
                child_path.push(0);
                mutations.extend(self.apply_aor(operand, child_path));
            }
            PythonNode::Compare { left, right, .. } => {
                let mut left_path = path.clone();
                left_path.push(0);
                mutations.extend(self.apply_aor(left, left_path));

                let mut right_path = path;
                right_path.push(1);
                mutations.extend(self.apply_aor(right, right_path));
            }
            _ => {}
        }

        mutations
    }

    /// ROR: Relational Operator Replacement
    /// `a < b` → `a <= b`, `a > b`, `a >= b`, `a == b`, `a != b`
    fn apply_ror(&self, ast: &PythonNode, path: Vec<usize>) -> Vec<AstMutation> {
        let mut mutations = Vec::new();

        match ast {
            PythonNode::Compare { left, op, right } => {
                // Generate mutations for this Compare
                let replacements = relational_replacements(*op);
                for new_op in replacements {
                    let mutated = PythonNode::Compare {
                        left: left.clone(),
                        op: new_op,
                        right: right.clone(),
                    };
                    mutations.push(AstMutation {
                        operator: MutationOperator::Ror,
                        path: path.clone(),
                        description: format!("Replace {} with {}", op.to_str(), new_op.to_str()),
                        mutated_ast: mutated,
                    });
                }

                // Recurse
                let mut left_path = path.clone();
                left_path.push(0);
                mutations.extend(self.apply_ror(left, left_path));

                let mut right_path = path;
                right_path.push(1);
                mutations.extend(self.apply_ror(right, right_path));
            }
            PythonNode::Module(stmts) => {
                for (i, stmt) in stmts.iter().enumerate() {
                    let mut child_path = path.clone();
                    child_path.push(i);
                    mutations.extend(self.apply_ror(stmt, child_path));
                }
            }
            PythonNode::If { test, body, orelse } => {
                let mut test_path = path.clone();
                test_path.push(0);
                mutations.extend(self.apply_ror(test, test_path));

                for (i, stmt) in body.iter().enumerate() {
                    let mut child_path = path.clone();
                    child_path.push(1 + i);
                    mutations.extend(self.apply_ror(stmt, child_path));
                }

                for (i, stmt) in orelse.iter().enumerate() {
                    let mut child_path = path.clone();
                    child_path.push(1 + body.len() + i);
                    mutations.extend(self.apply_ror(stmt, child_path));
                }
            }
            PythonNode::While { test, body } => {
                let mut test_path = path.clone();
                test_path.push(0);
                mutations.extend(self.apply_ror(test, test_path));

                for (i, stmt) in body.iter().enumerate() {
                    let mut child_path = path.clone();
                    child_path.push(1 + i);
                    mutations.extend(self.apply_ror(stmt, child_path));
                }
            }
            _ => {}
        }

        mutations
    }

    /// LOR: Logical Operator Replacement
    /// `a and b` → `a or b`
    fn apply_lor(&self, ast: &PythonNode, path: Vec<usize>) -> Vec<AstMutation> {
        let mut mutations = Vec::new();

        match ast {
            PythonNode::BinOp { left, op, right } => {
                // Only mutate logical operators
                let replacements = logical_replacements(*op);
                for new_op in replacements {
                    let mutated = PythonNode::BinOp {
                        left: left.clone(),
                        op: new_op,
                        right: right.clone(),
                    };
                    mutations.push(AstMutation {
                        operator: MutationOperator::Lor,
                        path: path.clone(),
                        description: format!("Replace {} with {}", op.to_str(), new_op.to_str()),
                        mutated_ast: mutated,
                    });
                }

                // Recurse
                let mut left_path = path.clone();
                left_path.push(0);
                mutations.extend(self.apply_lor(left, left_path));

                let mut right_path = path;
                right_path.push(1);
                mutations.extend(self.apply_lor(right, right_path));
            }
            PythonNode::Module(stmts) => {
                for (i, stmt) in stmts.iter().enumerate() {
                    let mut child_path = path.clone();
                    child_path.push(i);
                    mutations.extend(self.apply_lor(stmt, child_path));
                }
            }
            PythonNode::If { test, body, orelse } => {
                let mut test_path = path.clone();
                test_path.push(0);
                mutations.extend(self.apply_lor(test, test_path));

                for (i, stmt) in body.iter().enumerate() {
                    let mut child_path = path.clone();
                    child_path.push(1 + i);
                    mutations.extend(self.apply_lor(stmt, child_path));
                }

                for (i, stmt) in orelse.iter().enumerate() {
                    let mut child_path = path.clone();
                    child_path.push(1 + body.len() + i);
                    mutations.extend(self.apply_lor(stmt, child_path));
                }
            }
            _ => {}
        }

        mutations
    }

    /// UOI: Unary Operator Insertion
    /// `x` → `-x`, `not x`
    fn apply_uoi(&self, ast: &PythonNode, path: Vec<usize>) -> Vec<AstMutation> {
        let mut mutations = Vec::new();

        match ast {
            PythonNode::Name(name) => {
                // Insert negation
                let neg_mutated = PythonNode::UnaryOp {
                    op: UnaryOp::Neg,
                    operand: Box::new(ast.clone()),
                };
                mutations.push(AstMutation {
                    operator: MutationOperator::Uoi,
                    path: path.clone(),
                    description: format!("Insert negation: {name} → -{name}"),
                    mutated_ast: neg_mutated,
                });

                // Insert not
                let not_mutated = PythonNode::UnaryOp {
                    op: UnaryOp::Not,
                    operand: Box::new(ast.clone()),
                };
                mutations.push(AstMutation {
                    operator: MutationOperator::Uoi,
                    path,
                    description: format!("Insert not: {name} → not {name}"),
                    mutated_ast: not_mutated,
                });
            }
            PythonNode::IntLit(n) => {
                // Insert negation for integers
                let neg_mutated = PythonNode::UnaryOp {
                    op: UnaryOp::Neg,
                    operand: Box::new(ast.clone()),
                };
                mutations.push(AstMutation {
                    operator: MutationOperator::Uoi,
                    path,
                    description: format!("Insert negation: {n} → -{n}"),
                    mutated_ast: neg_mutated,
                });
            }
            PythonNode::Module(stmts) => {
                for (i, stmt) in stmts.iter().enumerate() {
                    let mut child_path = path.clone();
                    child_path.push(i);
                    mutations.extend(self.apply_uoi(stmt, child_path));
                }
            }
            PythonNode::Assign { value, .. } => {
                let mut child_path = path;
                child_path.push(0);
                mutations.extend(self.apply_uoi(value, child_path));
            }
            PythonNode::BinOp { left, right, .. } => {
                let mut left_path = path.clone();
                left_path.push(0);
                mutations.extend(self.apply_uoi(left, left_path));

                let mut right_path = path;
                right_path.push(1);
                mutations.extend(self.apply_uoi(right, right_path));
            }
            _ => {}
        }

        mutations
    }

    /// ABS: Absolute Value Insertion
    /// `x` → `abs(x)`
    fn apply_abs(&self, ast: &PythonNode, path: Vec<usize>) -> Vec<AstMutation> {
        let mut mutations = Vec::new();

        match ast {
            PythonNode::Name(name) => {
                let abs_mutated = PythonNode::Call {
                    func: "abs".to_string(),
                    args: vec![ast.clone()],
                };
                mutations.push(AstMutation {
                    operator: MutationOperator::Abs,
                    path,
                    description: format!("Insert abs: {name} → abs({name})"),
                    mutated_ast: abs_mutated,
                });
            }
            PythonNode::IntLit(n) => {
                let abs_mutated = PythonNode::Call {
                    func: "abs".to_string(),
                    args: vec![ast.clone()],
                };
                mutations.push(AstMutation {
                    operator: MutationOperator::Abs,
                    path,
                    description: format!("Insert abs: {n} → abs({n})"),
                    mutated_ast: abs_mutated,
                });
            }
            PythonNode::Module(stmts) => {
                for (i, stmt) in stmts.iter().enumerate() {
                    let mut child_path = path.clone();
                    child_path.push(i);
                    mutations.extend(self.apply_abs(stmt, child_path));
                }
            }
            PythonNode::Assign { value, .. } => {
                let mut child_path = path;
                child_path.push(0);
                mutations.extend(self.apply_abs(value, child_path));
            }
            _ => {}
        }

        mutations
    }

    /// SDL: Statement Deletion
    /// Delete a statement from the program
    fn apply_sdl(&self, ast: &PythonNode, path: Vec<usize>) -> Vec<AstMutation> {
        let mut mutations = Vec::new();

        if let PythonNode::Module(stmts) = ast {
            // For each statement, create a mutation that deletes it
            for i in 0..stmts.len() {
                if stmts.len() > 1 {
                    // Can only delete if more than one statement
                    let mut new_stmts = stmts.clone();
                    new_stmts.remove(i);

                    let mutated = PythonNode::Module(new_stmts);
                    let mut del_path = path.clone();
                    del_path.push(i);

                    mutations.push(AstMutation {
                        operator: MutationOperator::Sdl,
                        path: del_path,
                        description: format!("Delete statement {}", i + 1),
                        mutated_ast: mutated,
                    });
                }
            }

            // Also check for deletable statements within compound statements
            for (i, stmt) in stmts.iter().enumerate() {
                let mut child_path = path.clone();
                child_path.push(i);
                mutations.extend(self.apply_sdl_compound(stmt, child_path));
            }
        }

        mutations
    }

    /// Helper for SDL in compound statements
    fn apply_sdl_compound(&self, ast: &PythonNode, path: Vec<usize>) -> Vec<AstMutation> {
        let mut mutations = Vec::new();

        match ast {
            PythonNode::If { test, body, orelse } => {
                // Delete statements from body
                if body.len() > 1 {
                    for i in 0..body.len() {
                        let mut new_body = body.clone();
                        new_body.remove(i);

                        let mutated = PythonNode::If {
                            test: test.clone(),
                            body: new_body,
                            orelse: orelse.clone(),
                        };

                        let mut del_path = path.clone();
                        del_path.push(i);

                        mutations.push(AstMutation {
                            operator: MutationOperator::Sdl,
                            path: del_path,
                            description: format!("Delete if-body statement {}", i + 1),
                            mutated_ast: mutated,
                        });
                    }
                }

                // Delete statements from orelse
                if orelse.len() > 1 {
                    for i in 0..orelse.len() {
                        let mut new_orelse = orelse.clone();
                        new_orelse.remove(i);

                        let mutated = PythonNode::If {
                            test: test.clone(),
                            body: body.clone(),
                            orelse: new_orelse,
                        };

                        let mut del_path = path.clone();
                        del_path.push(body.len() + i);

                        mutations.push(AstMutation {
                            operator: MutationOperator::Sdl,
                            path: del_path,
                            description: format!("Delete else-body statement {}", i + 1),
                            mutated_ast: mutated,
                        });
                    }
                }
            }
            PythonNode::FuncDef { name, args, body } => {
                if body.len() > 1 {
                    for i in 0..body.len() {
                        let mut new_body = body.clone();
                        new_body.remove(i);

                        let mutated = PythonNode::FuncDef {
                            name: name.clone(),
                            args: args.clone(),
                            body: new_body,
                        };

                        let mut del_path = path.clone();
                        del_path.push(i);

                        mutations.push(AstMutation {
                            operator: MutationOperator::Sdl,
                            path: del_path,
                            description: format!("Delete function body statement {}", i + 1),
                            mutated_ast: mutated,
                        });
                    }
                }
            }
            _ => {}
        }

        mutations
    }

    /// SVR: Scalar Variable Replacement
    /// `x` → `y` (replace one variable with another in scope)
    fn apply_svr(&self, ast: &PythonNode, path: Vec<usize>) -> Vec<AstMutation> {
        let mut mutations = Vec::new();

        // First, collect all variable names in the AST
        let var_names = collect_variable_names(ast);

        if var_names.len() < 2 {
            return mutations; // Need at least 2 variables for SVR
        }

        self.apply_svr_recursive(ast, path, &var_names, &mut mutations);
        mutations
    }

    fn apply_svr_recursive(
        &self,
        ast: &PythonNode,
        path: Vec<usize>,
        var_names: &[String],
        mutations: &mut Vec<AstMutation>,
    ) {
        match ast {
            PythonNode::Name(name) => {
                // Replace with each other variable
                for other_var in var_names {
                    if other_var != name {
                        let mutated = PythonNode::Name(other_var.clone());
                        mutations.push(AstMutation {
                            operator: MutationOperator::Svr,
                            path: path.clone(),
                            description: format!("Replace {name} with {other_var}"),
                            mutated_ast: mutated,
                        });
                    }
                }
            }
            PythonNode::Module(stmts) => {
                for (i, stmt) in stmts.iter().enumerate() {
                    let mut child_path = path.clone();
                    child_path.push(i);
                    self.apply_svr_recursive(stmt, child_path, var_names, mutations);
                }
            }
            PythonNode::Assign { value, .. } => {
                let mut child_path = path;
                child_path.push(0);
                self.apply_svr_recursive(value, child_path, var_names, mutations);
            }
            PythonNode::BinOp { left, right, .. } => {
                let mut left_path = path.clone();
                left_path.push(0);
                self.apply_svr_recursive(left, left_path, var_names, mutations);

                let mut right_path = path;
                right_path.push(1);
                self.apply_svr_recursive(right, right_path, var_names, mutations);
            }
            _ => {}
        }
    }

    /// BSR: Boundary Substitution Replacement
    /// `0` → `-1`, `1` → `0`, `""` → `" "`, etc.
    fn apply_bsr(&self, ast: &PythonNode, path: Vec<usize>) -> Vec<AstMutation> {
        let mut mutations = Vec::new();

        match ast {
            PythonNode::IntLit(n) => {
                let boundaries = boundary_int_values(*n);
                for new_val in boundaries {
                    let mutated = PythonNode::IntLit(new_val);
                    mutations.push(AstMutation {
                        operator: MutationOperator::Bsr,
                        path: path.clone(),
                        description: format!("Boundary: {n} → {new_val}"),
                        mutated_ast: mutated,
                    });
                }
            }
            PythonNode::StrLit(s) => {
                let boundaries = boundary_str_values(s);
                for new_val in boundaries {
                    let mutated = PythonNode::StrLit(new_val.clone());
                    mutations.push(AstMutation {
                        operator: MutationOperator::Bsr,
                        path: path.clone(),
                        description: format!("Boundary: \"{s}\" → \"{new_val}\""),
                        mutated_ast: mutated,
                    });
                }
            }
            PythonNode::BoolLit(b) => {
                let mutated = PythonNode::BoolLit(!b);
                mutations.push(AstMutation {
                    operator: MutationOperator::Bsr,
                    path,
                    description: format!("Boundary: {b} → {}", !b),
                    mutated_ast: mutated,
                });
            }
            PythonNode::Module(stmts) => {
                for (i, stmt) in stmts.iter().enumerate() {
                    let mut child_path = path.clone();
                    child_path.push(i);
                    mutations.extend(self.apply_bsr(stmt, child_path));
                }
            }
            PythonNode::Assign { value, .. } => {
                let mut child_path = path;
                child_path.push(0);
                mutations.extend(self.apply_bsr(value, child_path));
            }
            PythonNode::BinOp { left, right, .. } => {
                let mut left_path = path.clone();
                left_path.push(0);
                mutations.extend(self.apply_bsr(left, left_path));

                let mut right_path = path;
                right_path.push(1);
                mutations.extend(self.apply_bsr(right, right_path));
            }
            _ => {}
        }

        mutations
    }
}

/// Get arithmetic operator replacements
fn arithmetic_replacements(op: BinaryOp) -> Vec<BinaryOp> {
    match op {
        BinaryOp::Add => vec![BinaryOp::Sub, BinaryOp::Mult],
        BinaryOp::Sub => vec![BinaryOp::Add, BinaryOp::Mult],
        BinaryOp::Mult => vec![BinaryOp::Add, BinaryOp::Div],
        BinaryOp::Div => vec![BinaryOp::Mult, BinaryOp::FloorDiv],
        BinaryOp::Mod => vec![BinaryOp::Div, BinaryOp::FloorDiv],
        BinaryOp::FloorDiv => vec![BinaryOp::Div, BinaryOp::Mod],
        BinaryOp::Pow => vec![BinaryOp::Mult],
        BinaryOp::And | BinaryOp::Or => vec![], // Not arithmetic
    }
}

/// Get relational operator replacements
fn relational_replacements(op: CompareOp) -> Vec<CompareOp> {
    match op {
        CompareOp::Eq => vec![CompareOp::NotEq],
        CompareOp::NotEq => vec![CompareOp::Eq],
        CompareOp::Lt => vec![CompareOp::LtE, CompareOp::Gt],
        CompareOp::LtE => vec![CompareOp::Lt, CompareOp::GtE],
        CompareOp::Gt => vec![CompareOp::GtE, CompareOp::Lt],
        CompareOp::GtE => vec![CompareOp::Gt, CompareOp::LtE],
    }
}

/// Get logical operator replacements
fn logical_replacements(op: BinaryOp) -> Vec<BinaryOp> {
    match op {
        BinaryOp::And => vec![BinaryOp::Or],
        BinaryOp::Or => vec![BinaryOp::And],
        _ => vec![], // Not logical
    }
}

/// Collect all variable names from an AST
fn collect_variable_names(ast: &PythonNode) -> Vec<String> {
    let mut names = Vec::new();
    collect_names_recursive(ast, &mut names);
    names.sort();
    names.dedup();
    names
}

fn collect_names_recursive(ast: &PythonNode, names: &mut Vec<String>) {
    match ast {
        PythonNode::Name(name) => names.push(name.clone()),
        PythonNode::Assign { target, value } => {
            names.push(target.clone());
            collect_names_recursive(value, names);
        }
        PythonNode::Module(stmts) => {
            for stmt in stmts {
                collect_names_recursive(stmt, names);
            }
        }
        PythonNode::BinOp { left, right, .. } | PythonNode::Compare { left, right, .. } => {
            collect_names_recursive(left, names);
            collect_names_recursive(right, names);
        }
        PythonNode::UnaryOp { operand, .. } => {
            collect_names_recursive(operand, names);
        }
        PythonNode::If { test, body, orelse } => {
            collect_names_recursive(test, names);
            for stmt in body {
                collect_names_recursive(stmt, names);
            }
            for stmt in orelse {
                collect_names_recursive(stmt, names);
            }
        }
        PythonNode::FuncDef { args, body, .. } => {
            for arg in args {
                names.push(arg.clone());
            }
            for stmt in body {
                collect_names_recursive(stmt, names);
            }
        }
        PythonNode::Return(Some(value)) => collect_names_recursive(value, names),
        _ => {}
    }
}

/// Get boundary integer values
fn boundary_int_values(n: i64) -> Vec<i64> {
    match n {
        0 => vec![-1, 1],
        1 => vec![0, 2],
        -1 => vec![0, -2],
        _ if n > 0 => vec![n - 1, n + 1, 0],
        _ => vec![n - 1, n + 1, 0],
    }
}

/// Get boundary string values
fn boundary_str_values(s: &str) -> Vec<String> {
    if s.is_empty() {
        vec![" ".to_string(), "a".to_string()]
    } else {
        vec![String::new(), format!("{s} ")]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aor_basic() {
        let ast = PythonNode::BinOp {
            left: Box::new(PythonNode::Name("a".to_string())),
            op: BinaryOp::Add,
            right: Box::new(PythonNode::Name("b".to_string())),
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Aor]);
        let mutations = mutator.mutate(&ast);

        assert!(!mutations.is_empty());
        // Should have mutations replacing + with - and *
        assert!(mutations
            .iter()
            .any(|m| m.description.contains("Replace + with -")));
    }

    #[test]
    fn test_ror_basic() {
        let ast = PythonNode::Compare {
            left: Box::new(PythonNode::Name("x".to_string())),
            op: CompareOp::Lt,
            right: Box::new(PythonNode::IntLit(10)),
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Ror]);
        let mutations = mutator.mutate(&ast);

        assert!(!mutations.is_empty());
        assert!(mutations
            .iter()
            .any(|m| m.description.contains("Replace < with <=")));
    }

    #[test]
    fn test_lor_basic() {
        let ast = PythonNode::BinOp {
            left: Box::new(PythonNode::Name("a".to_string())),
            op: BinaryOp::And,
            right: Box::new(PythonNode::Name("b".to_string())),
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Lor]);
        let mutations = mutator.mutate(&ast);

        assert!(!mutations.is_empty());
        assert!(mutations
            .iter()
            .any(|m| m.description.contains("Replace and with or")));
    }

    #[test]
    fn test_uoi_basic() {
        let ast = PythonNode::Assign {
            target: "x".to_string(),
            value: Box::new(PythonNode::Name("y".to_string())),
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Uoi]);
        let mutations = mutator.mutate(&ast);

        assert!(!mutations.is_empty());
        assert!(mutations
            .iter()
            .any(|m| m.description.contains("Insert negation")));
    }

    #[test]
    fn test_abs_basic() {
        let ast = PythonNode::Assign {
            target: "x".to_string(),
            value: Box::new(PythonNode::Name("y".to_string())),
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Abs]);
        let mutations = mutator.mutate(&ast);

        assert!(!mutations.is_empty());
        assert!(mutations
            .iter()
            .any(|m| m.description.contains("Insert abs")));
    }

    #[test]
    fn test_sdl_basic() {
        let ast = PythonNode::Module(vec![
            PythonNode::Assign {
                target: "x".to_string(),
                value: Box::new(PythonNode::IntLit(1)),
            },
            PythonNode::Assign {
                target: "y".to_string(),
                value: Box::new(PythonNode::IntLit(2)),
            },
        ]);

        let mutator = AstMutator::with_operators(vec![MutationOperator::Sdl]);
        let mutations = mutator.mutate(&ast);

        assert_eq!(mutations.len(), 2); // Can delete either statement
    }

    #[test]
    fn test_svr_basic() {
        let ast = PythonNode::Module(vec![
            PythonNode::Assign {
                target: "x".to_string(),
                value: Box::new(PythonNode::IntLit(1)),
            },
            PythonNode::Assign {
                target: "y".to_string(),
                value: Box::new(PythonNode::Name("x".to_string())),
            },
        ]);

        let mutator = AstMutator::with_operators(vec![MutationOperator::Svr]);
        let mutations = mutator.mutate(&ast);

        assert!(!mutations.is_empty());
        // Should have mutation replacing x with y
        assert!(mutations
            .iter()
            .any(|m| m.description.contains("Replace x with y")));
    }

    #[test]
    fn test_bsr_basic() {
        let ast = PythonNode::Assign {
            target: "x".to_string(),
            value: Box::new(PythonNode::IntLit(0)),
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Bsr]);
        let mutations = mutator.mutate(&ast);

        assert!(!mutations.is_empty());
        // Should have mutations for 0 → -1 and 0 → 1
        assert!(mutations.iter().any(|m| m.description.contains("0 → -1")));
        assert!(mutations.iter().any(|m| m.description.contains("0 → 1")));
    }

    #[test]
    fn test_all_operators() {
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
                    right: Box::new(PythonNode::IntLit(10)),
                }),
                body: vec![PythonNode::Pass],
                orelse: vec![],
            },
        ]);

        let mutator = AstMutator::new();
        let mutations = mutator.mutate(&ast);

        // Should have mutations from multiple operators
        assert!(mutations.len() > 5);

        // Check we have mutations from different operators
        let operators: std::collections::HashSet<_> =
            mutations.iter().map(|m| m.operator).collect();
        assert!(operators.len() >= 3);
    }

    #[test]
    fn test_mutation_produces_valid_ast() {
        let ast = PythonNode::BinOp {
            left: Box::new(PythonNode::IntLit(1)),
            op: BinaryOp::Add,
            right: Box::new(PythonNode::IntLit(2)),
        };

        let mutator = AstMutator::new();
        let mutations = mutator.mutate(&ast);

        // All mutations should produce valid Python code
        for mutation in &mutations {
            let code = mutation.mutated_ast.to_code(0);
            assert!(!code.is_empty());
        }
    }

    #[test]
    fn test_ast_mutator_default() {
        let mutator = AstMutator::default();
        assert!(mutator.enabled_operators.is_empty());
    }

    #[test]
    fn test_ast_mutator_debug() {
        let mutator = AstMutator::new();
        let debug = format!("{:?}", mutator);
        assert!(debug.contains("AstMutator"));
    }

    #[test]
    fn test_ast_mutation_debug() {
        let mutation = AstMutation {
            operator: MutationOperator::Aor,
            path: vec![0, 1],
            description: "Test mutation".to_string(),
            mutated_ast: PythonNode::IntLit(1),
        };
        let debug = format!("{:?}", mutation);
        assert!(debug.contains("AstMutation"));
    }

    #[test]
    fn test_ast_mutation_clone() {
        let mutation = AstMutation {
            operator: MutationOperator::Aor,
            path: vec![0, 1],
            description: "Test mutation".to_string(),
            mutated_ast: PythonNode::IntLit(1),
        };
        let cloned = mutation.clone();
        assert_eq!(cloned.operator, mutation.operator);
        assert_eq!(cloned.path, mutation.path);
    }

    #[test]
    fn test_aor_in_if_body() {
        let ast = PythonNode::If {
            test: Box::new(PythonNode::BoolLit(true)),
            body: vec![PythonNode::Assign {
                target: "x".to_string(),
                value: Box::new(PythonNode::BinOp {
                    left: Box::new(PythonNode::IntLit(1)),
                    op: BinaryOp::Add,
                    right: Box::new(PythonNode::IntLit(2)),
                }),
            }],
            orelse: vec![PythonNode::Assign {
                target: "y".to_string(),
                value: Box::new(PythonNode::BinOp {
                    left: Box::new(PythonNode::IntLit(3)),
                    op: BinaryOp::Sub,
                    right: Box::new(PythonNode::IntLit(4)),
                }),
            }],
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Aor]);
        let mutations = mutator.mutate(&ast);
        assert!(!mutations.is_empty());
    }

    #[test]
    fn test_aor_in_return() {
        let ast = PythonNode::Return(Some(Box::new(PythonNode::BinOp {
            left: Box::new(PythonNode::IntLit(1)),
            op: BinaryOp::Mult,
            right: Box::new(PythonNode::IntLit(2)),
        })));

        let mutator = AstMutator::with_operators(vec![MutationOperator::Aor]);
        let mutations = mutator.mutate(&ast);
        assert!(!mutations.is_empty());
    }

    #[test]
    fn test_aor_in_unary_op() {
        let ast = PythonNode::UnaryOp {
            op: UnaryOp::Neg,
            operand: Box::new(PythonNode::BinOp {
                left: Box::new(PythonNode::IntLit(1)),
                op: BinaryOp::Add,
                right: Box::new(PythonNode::IntLit(2)),
            }),
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Aor]);
        let mutations = mutator.mutate(&ast);
        assert!(!mutations.is_empty());
    }

    #[test]
    fn test_aor_in_compare() {
        let ast = PythonNode::Compare {
            left: Box::new(PythonNode::BinOp {
                left: Box::new(PythonNode::IntLit(1)),
                op: BinaryOp::Add,
                right: Box::new(PythonNode::IntLit(2)),
            }),
            op: CompareOp::Lt,
            right: Box::new(PythonNode::IntLit(10)),
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Aor]);
        let mutations = mutator.mutate(&ast);
        assert!(!mutations.is_empty());
    }

    #[test]
    fn test_ror_in_while() {
        let ast = PythonNode::While {
            test: Box::new(PythonNode::Compare {
                left: Box::new(PythonNode::Name("x".to_string())),
                op: CompareOp::Lt,
                right: Box::new(PythonNode::IntLit(10)),
            }),
            body: vec![PythonNode::Assign {
                target: "x".to_string(),
                value: Box::new(PythonNode::BinOp {
                    left: Box::new(PythonNode::Name("x".to_string())),
                    op: BinaryOp::Add,
                    right: Box::new(PythonNode::IntLit(1)),
                }),
            }],
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Ror]);
        let mutations = mutator.mutate(&ast);
        assert!(!mutations.is_empty());
    }

    #[test]
    fn test_ror_in_if_orelse() {
        let ast = PythonNode::If {
            test: Box::new(PythonNode::Compare {
                left: Box::new(PythonNode::Name("x".to_string())),
                op: CompareOp::Gt,
                right: Box::new(PythonNode::IntLit(0)),
            }),
            body: vec![PythonNode::Pass],
            orelse: vec![PythonNode::If {
                test: Box::new(PythonNode::Compare {
                    left: Box::new(PythonNode::Name("x".to_string())),
                    op: CompareOp::Eq,
                    right: Box::new(PythonNode::IntLit(0)),
                }),
                body: vec![PythonNode::Pass],
                orelse: vec![],
            }],
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Ror]);
        let mutations = mutator.mutate(&ast);
        assert!(!mutations.is_empty());
    }

    #[test]
    fn test_lor_in_if() {
        let ast = PythonNode::If {
            test: Box::new(PythonNode::BinOp {
                left: Box::new(PythonNode::Name("a".to_string())),
                op: BinaryOp::And,
                right: Box::new(PythonNode::Name("b".to_string())),
            }),
            body: vec![PythonNode::Pass],
            orelse: vec![PythonNode::Pass],
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Lor]);
        let mutations = mutator.mutate(&ast);
        assert!(!mutations.is_empty());
    }

    #[test]
    fn test_bsr_in_binop() {
        let ast = PythonNode::BinOp {
            left: Box::new(PythonNode::IntLit(0)),
            op: BinaryOp::Add,
            right: Box::new(PythonNode::IntLit(1)),
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Bsr]);
        let mutations = mutator.mutate(&ast);
        assert!(!mutations.is_empty());
    }

    #[test]
    fn test_bsr_string() {
        let ast = PythonNode::Assign {
            target: "s".to_string(),
            value: Box::new(PythonNode::StrLit("hello".to_string())),
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Bsr]);
        let mutations = mutator.mutate(&ast);
        assert!(!mutations.is_empty());
    }

    #[test]
    fn test_bsr_empty_string() {
        let ast = PythonNode::Assign {
            target: "s".to_string(),
            value: Box::new(PythonNode::StrLit(String::new())),
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Bsr]);
        let mutations = mutator.mutate(&ast);
        assert!(!mutations.is_empty());
    }

    #[test]
    fn test_arithmetic_replacements() {
        assert!(!arithmetic_replacements(BinaryOp::Add).is_empty());
        assert!(!arithmetic_replacements(BinaryOp::Sub).is_empty());
        assert!(!arithmetic_replacements(BinaryOp::Mult).is_empty());
        assert!(!arithmetic_replacements(BinaryOp::Div).is_empty());
        assert!(!arithmetic_replacements(BinaryOp::Mod).is_empty());
        assert!(!arithmetic_replacements(BinaryOp::FloorDiv).is_empty());
        assert!(!arithmetic_replacements(BinaryOp::Pow).is_empty());
        assert!(arithmetic_replacements(BinaryOp::And).is_empty()); // Not arithmetic
    }

    #[test]
    fn test_relational_replacements() {
        assert!(!relational_replacements(CompareOp::Eq).is_empty());
        assert!(!relational_replacements(CompareOp::NotEq).is_empty());
        assert!(!relational_replacements(CompareOp::Lt).is_empty());
        assert!(!relational_replacements(CompareOp::LtE).is_empty());
        assert!(!relational_replacements(CompareOp::Gt).is_empty());
        assert!(!relational_replacements(CompareOp::GtE).is_empty());
    }

    #[test]
    fn test_logical_replacements() {
        assert!(!logical_replacements(BinaryOp::And).is_empty());
        assert!(!logical_replacements(BinaryOp::Or).is_empty());
        assert!(logical_replacements(BinaryOp::Add).is_empty()); // Not logical
    }

    #[test]
    fn test_collect_variable_names() {
        let ast = PythonNode::Module(vec![
            PythonNode::Assign {
                target: "x".to_string(),
                value: Box::new(PythonNode::IntLit(1)),
            },
            PythonNode::Assign {
                target: "y".to_string(),
                value: Box::new(PythonNode::Name("x".to_string())),
            },
        ]);

        let names = collect_variable_names(&ast);
        assert!(names.contains(&"x".to_string()));
        assert!(names.contains(&"y".to_string()));
    }

    #[test]
    fn test_collect_variable_names_in_binop() {
        let ast = PythonNode::BinOp {
            left: Box::new(PythonNode::Name("a".to_string())),
            op: BinaryOp::Add,
            right: Box::new(PythonNode::Name("b".to_string())),
        };

        let names = collect_variable_names(&ast);
        assert!(names.contains(&"a".to_string()));
        assert!(names.contains(&"b".to_string()));
    }

    #[test]
    fn test_collect_variable_names_in_unaryop() {
        let ast = PythonNode::UnaryOp {
            op: UnaryOp::Neg,
            operand: Box::new(PythonNode::Name("x".to_string())),
        };

        let names = collect_variable_names(&ast);
        assert!(names.contains(&"x".to_string()));
    }

    #[test]
    fn test_collect_variable_names_in_if() {
        let ast = PythonNode::If {
            test: Box::new(PythonNode::Name("cond".to_string())),
            body: vec![PythonNode::Assign {
                target: "a".to_string(),
                value: Box::new(PythonNode::IntLit(1)),
            }],
            orelse: vec![PythonNode::Assign {
                target: "b".to_string(),
                value: Box::new(PythonNode::IntLit(2)),
            }],
        };

        let names = collect_variable_names(&ast);
        assert!(names.contains(&"cond".to_string()));
        assert!(names.contains(&"a".to_string()));
        assert!(names.contains(&"b".to_string()));
    }

    #[test]
    fn test_collect_variable_names_in_funcdef() {
        let ast = PythonNode::FuncDef {
            name: "foo".to_string(),
            args: vec!["x".to_string(), "y".to_string()],
            body: vec![PythonNode::Return(Some(Box::new(PythonNode::Name(
                "x".to_string(),
            ))))],
        };

        let names = collect_variable_names(&ast);
        assert!(names.contains(&"x".to_string()));
        assert!(names.contains(&"y".to_string()));
    }

    #[test]
    fn test_boundary_int_values() {
        let vals_0 = boundary_int_values(0);
        assert!(vals_0.contains(&-1));
        assert!(vals_0.contains(&1));

        let vals_1 = boundary_int_values(1);
        assert!(vals_1.contains(&0));
        assert!(vals_1.contains(&2));

        let vals_neg1 = boundary_int_values(-1);
        assert!(vals_neg1.contains(&0));
        assert!(vals_neg1.contains(&-2));

        let vals_5 = boundary_int_values(5);
        assert!(vals_5.contains(&4));
        assert!(vals_5.contains(&6));
        assert!(vals_5.contains(&0));

        let vals_neg5 = boundary_int_values(-5);
        assert!(vals_neg5.contains(&-6));
        assert!(vals_neg5.contains(&-4));
    }

    #[test]
    fn test_boundary_str_values() {
        let vals_empty = boundary_str_values("");
        assert!(vals_empty.contains(&" ".to_string()));
        assert!(vals_empty.contains(&"a".to_string()));

        let vals_nonempty = boundary_str_values("hello");
        assert!(vals_nonempty.contains(&String::new()));
        assert!(vals_nonempty.contains(&"hello ".to_string()));
    }

    #[test]
    fn test_abs_with_intlit() {
        // Test ABS mutation on IntLit directly
        let ast = PythonNode::Assign {
            target: "x".to_string(),
            value: Box::new(PythonNode::IntLit(42)),
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Abs]);
        let mutations = mutator.mutate(&ast);

        assert!(!mutations.is_empty());
        assert!(mutations.iter().any(|m| m.description.contains("abs(42)")));
    }

    #[test]
    fn test_sdl_if_with_multiple_body_statements() {
        // If with >1 statements in body - tests SDL compound for If body
        let ast = PythonNode::Module(vec![PythonNode::If {
            test: Box::new(PythonNode::BoolLit(true)),
            body: vec![
                PythonNode::Assign {
                    target: "x".to_string(),
                    value: Box::new(PythonNode::IntLit(1)),
                },
                PythonNode::Assign {
                    target: "y".to_string(),
                    value: Box::new(PythonNode::IntLit(2)),
                },
            ],
            orelse: vec![],
        }]);

        let mutator = AstMutator::with_operators(vec![MutationOperator::Sdl]);
        let mutations = mutator.mutate(&ast);

        assert!(!mutations.is_empty());
        assert!(mutations.iter().any(|m| m.description.contains("if-body")));
    }

    #[test]
    fn test_sdl_if_with_multiple_orelse_statements() {
        // If with >1 statements in orelse - tests SDL compound for If orelse
        let ast = PythonNode::Module(vec![PythonNode::If {
            test: Box::new(PythonNode::BoolLit(true)),
            body: vec![PythonNode::Pass],
            orelse: vec![
                PythonNode::Assign {
                    target: "x".to_string(),
                    value: Box::new(PythonNode::IntLit(1)),
                },
                PythonNode::Assign {
                    target: "y".to_string(),
                    value: Box::new(PythonNode::IntLit(2)),
                },
            ],
        }]);

        let mutator = AstMutator::with_operators(vec![MutationOperator::Sdl]);
        let mutations = mutator.mutate(&ast);

        assert!(!mutations.is_empty());
        assert!(mutations
            .iter()
            .any(|m| m.description.contains("else-body")));
    }

    #[test]
    fn test_sdl_funcdef_with_multiple_body_statements() {
        // FuncDef with >1 statements in body - tests SDL compound for FuncDef
        let ast = PythonNode::Module(vec![PythonNode::FuncDef {
            name: "foo".to_string(),
            args: vec![],
            body: vec![
                PythonNode::Assign {
                    target: "x".to_string(),
                    value: Box::new(PythonNode::IntLit(1)),
                },
                PythonNode::Return(Some(Box::new(PythonNode::Name("x".to_string())))),
            ],
        }]);

        let mutator = AstMutator::with_operators(vec![MutationOperator::Sdl]);
        let mutations = mutator.mutate(&ast);

        assert!(!mutations.is_empty());
        assert!(mutations
            .iter()
            .any(|m| m.description.contains("function body")));
    }

    #[test]
    fn test_svr_in_binop() {
        // SVR with variable replacement inside BinOp - tests apply_svr_recursive for BinOp
        let ast = PythonNode::Module(vec![
            PythonNode::Assign {
                target: "x".to_string(),
                value: Box::new(PythonNode::IntLit(1)),
            },
            PythonNode::Assign {
                target: "y".to_string(),
                value: Box::new(PythonNode::IntLit(2)),
            },
            PythonNode::Assign {
                target: "z".to_string(),
                value: Box::new(PythonNode::BinOp {
                    left: Box::new(PythonNode::Name("x".to_string())),
                    op: BinaryOp::Add,
                    right: Box::new(PythonNode::Name("y".to_string())),
                }),
            },
        ]);

        let mutator = AstMutator::with_operators(vec![MutationOperator::Svr]);
        let mutations = mutator.mutate(&ast);

        assert!(!mutations.is_empty());
        // Should have replacements for x and y in the binop
        assert!(mutations.iter().any(|m| m.description.contains("Replace")));
    }

    #[test]
    fn test_bsr_boolean() {
        // BSR mutation on BoolLit - tests apply_bsr for BoolLit
        let ast = PythonNode::Assign {
            target: "flag".to_string(),
            value: Box::new(PythonNode::BoolLit(true)),
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Bsr]);
        let mutations = mutator.mutate(&ast);

        assert!(!mutations.is_empty());
        assert!(mutations
            .iter()
            .any(|m| m.description.contains("true → false")));
    }

    #[test]
    fn test_bsr_boolean_false() {
        // BSR mutation on BoolLit false
        let ast = PythonNode::Assign {
            target: "flag".to_string(),
            value: Box::new(PythonNode::BoolLit(false)),
        };

        let mutator = AstMutator::with_operators(vec![MutationOperator::Bsr]);
        let mutations = mutator.mutate(&ast);

        assert!(!mutations.is_empty());
        assert!(mutations
            .iter()
            .any(|m| m.description.contains("false → true")));
    }
}
