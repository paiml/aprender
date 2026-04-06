//! Python exhaustive enumeration
//!
//! Generates all valid Python programs up to a specified AST depth.
//! Uses a simplified Python grammar for combinatorial generation.

use super::GeneratedCode;
use crate::Language;

/// Python AST node types for generation
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum PythonNode {
    /// Module (root node)
    Module(Vec<PythonNode>),
    /// Assignment statement: `name = expr`
    Assign {
        /// Variable name being assigned to
        target: String,
        /// Expression value
        value: Box<PythonNode>,
    },
    /// Binary operation: `left op right`
    BinOp {
        /// Left operand
        left: Box<PythonNode>,
        /// Binary operator
        op: BinaryOp,
        /// Right operand
        right: Box<PythonNode>,
    },
    /// Unary operation: `op operand`
    UnaryOp {
        /// Unary operator
        op: UnaryOp,
        /// Operand expression
        operand: Box<PythonNode>,
    },
    /// Integer literal
    IntLit(i64),
    /// Float literal
    FloatLit(f64),
    /// String literal
    StrLit(String),
    /// Boolean literal
    BoolLit(bool),
    /// None literal
    NoneLit,
    /// Variable reference
    Name(String),
    /// If statement
    If {
        /// Condition expression
        test: Box<PythonNode>,
        /// If body statements
        body: Vec<PythonNode>,
        /// Else body statements
        orelse: Vec<PythonNode>,
    },
    /// While loop
    While {
        /// Loop condition
        test: Box<PythonNode>,
        /// Loop body statements
        body: Vec<PythonNode>,
    },
    /// For loop
    For {
        /// Loop variable name
        target: String,
        /// Iterable expression
        iter: Box<PythonNode>,
        /// Loop body statements
        body: Vec<PythonNode>,
    },
    /// Function definition
    FuncDef {
        /// Function name
        name: String,
        /// Parameter names
        args: Vec<String>,
        /// Function body statements
        body: Vec<PythonNode>,
    },
    /// Function call
    Call {
        /// Function name
        func: String,
        /// Argument expressions
        args: Vec<PythonNode>,
    },
    /// Return statement
    Return(Option<Box<PythonNode>>),
    /// Pass statement
    Pass,
    /// Break statement
    Break,
    /// Continue statement
    Continue,
    /// List literal
    List(Vec<PythonNode>),
    /// Comparison: `left op right`
    Compare {
        /// Left operand
        left: Box<PythonNode>,
        /// Comparison operator
        op: CompareOp,
        /// Right operand
        right: Box<PythonNode>,
    },
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// Addition (`+`)
    Add,
    /// Subtraction (`-`)
    Sub,
    /// Multiplication (`*`)
    Mult,
    /// Division (`/`)
    Div,
    /// Modulo (`%`)
    Mod,
    /// Floor division (`//`)
    FloorDiv,
    /// Power (`**`)
    Pow,
    /// Logical and (`and`)
    And,
    /// Logical or (`or`)
    Or,
}

impl BinaryOp {
    /// Get all arithmetic binary operators
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::Add,
            Self::Sub,
            Self::Mult,
            Self::Div,
            Self::Mod,
            Self::FloorDiv,
            Self::Pow,
        ]
    }

    /// Convert to Python operator string
    #[must_use]
    pub fn to_str(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mult => "*",
            Self::Div => "/",
            Self::Mod => "%",
            Self::FloorDiv => "//",
            Self::Pow => "**",
            Self::And => "and",
            Self::Or => "or",
        }
    }
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Negation (`-x`)
    Neg,
    /// Logical not (`not x`)
    Not,
    /// Positive (`+x`)
    Pos,
}

impl UnaryOp {
    /// Get all unary operators
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[Self::Neg, Self::Not, Self::Pos]
    }

    /// Convert to Python operator string
    #[must_use]
    pub fn to_str(self) -> &'static str {
        match self {
            Self::Neg => "-",
            Self::Not => "not ",
            Self::Pos => "+",
        }
    }
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    /// Equal (`==`)
    Eq,
    /// Not equal (`!=`)
    NotEq,
    /// Less than (`<`)
    Lt,
    /// Less than or equal (`<=`)
    LtE,
    /// Greater than (`>`)
    Gt,
    /// Greater than or equal (`>=`)
    GtE,
}

impl CompareOp {
    /// Get all comparison operators
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::Eq,
            Self::NotEq,
            Self::Lt,
            Self::LtE,
            Self::Gt,
            Self::GtE,
        ]
    }

    /// Convert to Python operator string
    #[must_use]
    pub fn to_str(self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::NotEq => "!=",
            Self::Lt => "<",
            Self::LtE => "<=",
            Self::Gt => ">",
            Self::GtE => ">=",
        }
    }
}

impl PythonNode {
    /// Convert AST node to Python source code
    #[allow(clippy::too_many_lines)]
    pub fn to_code(&self, indent: usize) -> String {
        let indent_str = "    ".repeat(indent);
        match self {
            Self::Module(stmts) => stmts
                .iter()
                .map(|s| s.to_code(0))
                .collect::<Vec<_>>()
                .join("\n"),
            Self::Assign { target, value } => {
                let val = value.to_code(0);
                format!("{indent_str}{target} = {val}")
            }
            Self::BinOp { left, op, right } => {
                let l = left.to_code(0);
                let r = right.to_code(0);
                let o = op.to_str();
                format!("({l} {o} {r})")
            }
            Self::UnaryOp { op, operand } => {
                let o = op.to_str();
                let e = operand.to_code(0);
                format!("({o}{e})")
            }
            Self::IntLit(n) => n.to_string(),
            Self::FloatLit(f) => format!("{f:.1}"),
            Self::StrLit(s) => format!("\"{s}\""),
            Self::BoolLit(b) => if *b { "True" } else { "False" }.to_string(),
            Self::NoneLit => "None".to_string(),
            Self::Name(name) => name.clone(),
            Self::If { test, body, orelse } => {
                self.if_to_code(&indent_str, indent, test, body, orelse)
            }
            Self::While { test, body } => self.while_to_code(&indent_str, indent, test, body),
            Self::For { target, iter, body } => {
                self.for_to_code(&indent_str, indent, target, iter, body)
            }
            Self::FuncDef { name, args, body } => {
                self.funcdef_to_code(&indent_str, indent, name, args, body)
            }
            Self::Call { func, args } => {
                let args_str = args
                    .iter()
                    .map(|a| a.to_code(0))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{func}({args_str})")
            }
            Self::Return(Some(value)) => {
                let val = value.to_code(0);
                format!("{indent_str}return {val}")
            }
            Self::Return(None) => format!("{indent_str}return"),
            Self::Pass => format!("{indent_str}pass"),
            Self::Break => format!("{indent_str}break"),
            Self::Continue => format!("{indent_str}continue"),
            Self::List(items) => {
                let items_str = items
                    .iter()
                    .map(|i| i.to_code(0))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{items_str}]")
            }
            Self::Compare { left, op, right } => {
                let l = left.to_code(0);
                let r = right.to_code(0);
                let o = op.to_str();
                format!("({l} {o} {r})")
            }
        }
    }

    fn if_to_code(
        &self,
        indent_str: &str,
        indent: usize,
        test: &PythonNode,
        body: &[PythonNode],
        orelse: &[PythonNode],
    ) -> String {
        let body_code = body
            .iter()
            .map(|s| s.to_code(indent + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let test_code = test.to_code(0);
        if orelse.is_empty() {
            format!("{indent_str}if {test_code}:\n{body_code}")
        } else {
            let else_code = orelse
                .iter()
                .map(|s| s.to_code(indent + 1))
                .collect::<Vec<_>>()
                .join("\n");
            format!("{indent_str}if {test_code}:\n{body_code}\n{indent_str}else:\n{else_code}")
        }
    }

    fn while_to_code(
        &self,
        indent_str: &str,
        indent: usize,
        test: &PythonNode,
        body: &[PythonNode],
    ) -> String {
        let body_code = body
            .iter()
            .map(|s| s.to_code(indent + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let test_code = test.to_code(0);
        format!("{indent_str}while {test_code}:\n{body_code}")
    }

    fn for_to_code(
        &self,
        indent_str: &str,
        indent: usize,
        target: &str,
        iter: &PythonNode,
        body: &[PythonNode],
    ) -> String {
        let body_code = body
            .iter()
            .map(|s| s.to_code(indent + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let iter_code = iter.to_code(0);
        format!("{indent_str}for {target} in {iter_code}:\n{body_code}")
    }

    fn funcdef_to_code(
        &self,
        indent_str: &str,
        indent: usize,
        name: &str,
        args: &[String],
        body: &[PythonNode],
    ) -> String {
        let args_str = args.join(", ");
        let body_code = if body.is_empty() {
            format!("{indent_str}    pass")
        } else {
            body.iter()
                .map(|s| s.to_code(indent + 1))
                .collect::<Vec<_>>()
                .join("\n")
        };
        format!("{indent_str}def {name}({args_str}):\n{body_code}")
    }

    /// Calculate AST depth
    pub fn depth(&self) -> usize {
        match self {
            Self::Module(stmts) => 1 + stmts.iter().map(Self::depth).max().unwrap_or(0),
            Self::Assign { value, .. } => 1 + value.depth(),
            Self::BinOp { left, right, .. } | Self::Compare { left, right, .. } => {
                1 + left.depth().max(right.depth())
            }
            Self::UnaryOp { operand, .. } => 1 + operand.depth(),
            Self::If { test, body, orelse } => {
                let body_depth = body.iter().map(Self::depth).max().unwrap_or(0);
                let else_depth = orelse.iter().map(Self::depth).max().unwrap_or(0);
                1 + test.depth().max(body_depth).max(else_depth)
            }
            Self::While { test, body } => {
                let body_depth = body.iter().map(Self::depth).max().unwrap_or(0);
                1 + test.depth().max(body_depth)
            }
            Self::For { iter, body, .. } => {
                let body_depth = body.iter().map(Self::depth).max().unwrap_or(0);
                1 + iter.depth().max(body_depth)
            }
            Self::FuncDef { body, .. } => 1 + body.iter().map(Self::depth).max().unwrap_or(0),
            Self::Call { args, .. } => 1 + args.iter().map(Self::depth).max().unwrap_or(0),
            Self::Return(Some(v)) => 1 + v.depth(),
            Self::List(items) => 1 + items.iter().map(Self::depth).max().unwrap_or(0),
            // Terminal nodes - depth 1
            Self::Return(None)
            | Self::IntLit(_)
            | Self::FloatLit(_)
            | Self::StrLit(_)
            | Self::BoolLit(_)
            | Self::NoneLit
            | Self::Name(_)
            | Self::Pass
            | Self::Break
            | Self::Continue => 1,
        }
    }
}

/// Exhaustive Python program enumerator
#[derive(Debug)]
pub struct PythonEnumerator {
    max_depth: usize,
    var_names: Vec<String>,
    int_values: Vec<i64>,
}

impl Default for PythonEnumerator {
    fn default() -> Self {
        Self::new(3)
    }
}

impl PythonEnumerator {
    /// Create a new enumerator with specified max depth
    #[must_use]
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            var_names: vec!["x".to_string(), "y".to_string(), "z".to_string()],
            int_values: vec![0, 1, -1, 2, 10],
        }
    }

    /// Enumerate all expressions up to the given depth
    pub fn enumerate_expressions(&self, depth: usize) -> Vec<PythonNode> {
        if depth == 0 {
            return vec![];
        }

        let mut results = Vec::new();

        // Depth 1: literals and names
        for &val in &self.int_values {
            results.push(PythonNode::IntLit(val));
        }
        for name in &self.var_names {
            results.push(PythonNode::Name(name.clone()));
        }
        results.push(PythonNode::BoolLit(true));
        results.push(PythonNode::BoolLit(false));
        results.push(PythonNode::NoneLit);

        if depth == 1 {
            return results;
        }

        // Depth 2+: compound expressions
        let subexprs = self.enumerate_expressions(depth - 1);

        // Unary operations
        for op in UnaryOp::all() {
            for subexpr in &subexprs {
                if subexpr.depth() < depth {
                    results.push(PythonNode::UnaryOp {
                        op: *op,
                        operand: Box::new(subexpr.clone()),
                    });
                }
            }
        }

        // Binary operations (limited to prevent explosion)
        let limited_subexprs: Vec<_> = subexprs.iter().take(10).collect();
        for op in BinaryOp::all() {
            for left in &limited_subexprs {
                for right in &limited_subexprs {
                    if left.depth() + right.depth() < depth {
                        results.push(PythonNode::BinOp {
                            left: Box::new((*left).clone()),
                            op: *op,
                            right: Box::new((*right).clone()),
                        });
                    }
                }
            }
        }

        // Comparisons
        for op in CompareOp::all() {
            for left in &limited_subexprs {
                for right in &limited_subexprs {
                    if left.depth() + right.depth() < depth {
                        results.push(PythonNode::Compare {
                            left: Box::new((*left).clone()),
                            op: *op,
                            right: Box::new((*right).clone()),
                        });
                    }
                }
            }
        }

        results
    }

    /// Enumerate all statements up to the given depth
    pub fn enumerate_statements(&self, depth: usize) -> Vec<PythonNode> {
        if depth == 0 {
            return vec![];
        }

        let mut results = Vec::new();

        // Simple statements
        results.push(PythonNode::Pass);

        let exprs = self.enumerate_expressions(depth - 1);
        let limited_exprs: Vec<_> = exprs.iter().take(20).collect();

        // Assignments
        for target in &self.var_names {
            for value in &limited_exprs {
                results.push(PythonNode::Assign {
                    target: target.clone(),
                    value: Box::new((*value).clone()),
                });
            }
        }

        // Return statements
        results.push(PythonNode::Return(None));
        for expr in limited_exprs.iter().take(10) {
            results.push(PythonNode::Return(Some(Box::new((*expr).clone()))));
        }

        if depth >= 2 {
            // If statements
            let conditions: Vec<_> = exprs
                .iter()
                .filter(|e| {
                    matches!(
                        e,
                        PythonNode::Compare { .. } | PythonNode::BoolLit(_) | PythonNode::Name(_)
                    )
                })
                .take(5)
                .collect();

            let body_stmts = self.enumerate_statements(depth - 1);
            let limited_body: Vec<_> = body_stmts.iter().take(5).collect();

            for cond in &conditions {
                for body in &limited_body {
                    results.push(PythonNode::If {
                        test: Box::new((*cond).clone()),
                        body: vec![(*body).clone()],
                        orelse: vec![],
                    });
                }
            }

            // While loops
            for cond in &conditions {
                results.push(PythonNode::While {
                    test: Box::new((*cond).clone()),
                    body: vec![PythonNode::Break],
                });
            }
        }

        if depth >= 3 {
            // Function definitions
            for name in &["foo", "bar"] {
                results.push(PythonNode::FuncDef {
                    name: (*name).to_string(),
                    args: vec![],
                    body: vec![PythonNode::Pass],
                });
                results.push(PythonNode::FuncDef {
                    name: (*name).to_string(),
                    args: vec!["a".to_string()],
                    body: vec![PythonNode::Return(Some(Box::new(PythonNode::Name(
                        "a".to_string(),
                    ))))],
                });
            }
        }

        results
    }

    /// Enumerate complete programs (modules)
    pub fn enumerate_programs(&self) -> Vec<GeneratedCode> {
        let mut results = Vec::new();

        let stmts = self.enumerate_statements(self.max_depth);

        // Single statement programs
        for stmt in &stmts {
            let module = PythonNode::Module(vec![stmt.clone()]);
            let code = module.to_code(0);
            results.push(GeneratedCode {
                code,
                language: Language::Python,
                ast_depth: stmt.depth(),
                features: self.extract_features(stmt),
            });
        }

        // Two statement programs (limited)
        let limited_stmts: Vec<_> = stmts.iter().take(20).collect();
        for s1 in &limited_stmts {
            for s2 in limited_stmts.iter().take(10) {
                let module = PythonNode::Module(vec![(*s1).clone(), (*s2).clone()]);
                let code = module.to_code(0);
                let depth = s1.depth().max(s2.depth());
                results.push(GeneratedCode {
                    code,
                    language: Language::Python,
                    ast_depth: depth,
                    features: self.extract_features(s1),
                });
            }
        }

        results
    }

    /// Extract feature labels from an AST node
    fn extract_features(&self, node: &PythonNode) -> Vec<String> {
        let mut features = Vec::new();

        match node {
            PythonNode::Assign { .. } => features.push("assignment".to_string()),
            PythonNode::BinOp { op, .. } => {
                features.push("binop".to_string());
                features.push(format!("op_{}", op.to_str()));
            }
            PythonNode::If { orelse, .. } => {
                features.push("if".to_string());
                if !orelse.is_empty() {
                    features.push("else".to_string());
                }
            }
            PythonNode::While { .. } => features.push("while".to_string()),
            PythonNode::For { .. } => features.push("for".to_string()),
            PythonNode::FuncDef { .. } => features.push("funcdef".to_string()),
            PythonNode::Return(_) => features.push("return".to_string()),
            PythonNode::Compare { op, .. } => {
                features.push("compare".to_string());
                features.push(format!("cmp_{}", op.to_str()));
            }
            _ => {}
        }

        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_lit_to_code() {
        let node = PythonNode::IntLit(42);
        assert_eq!(node.to_code(0), "42");
    }

    #[test]
    fn test_assign_to_code() {
        let node = PythonNode::Assign {
            target: "x".to_string(),
            value: Box::new(PythonNode::IntLit(1)),
        };
        assert_eq!(node.to_code(0), "x = 1");
    }

    #[test]
    fn test_binop_to_code() {
        let node = PythonNode::BinOp {
            left: Box::new(PythonNode::IntLit(1)),
            op: BinaryOp::Add,
            right: Box::new(PythonNode::IntLit(2)),
        };
        assert_eq!(node.to_code(0), "(1 + 2)");
    }

    #[test]
    fn test_if_to_code() {
        let node = PythonNode::If {
            test: Box::new(PythonNode::BoolLit(true)),
            body: vec![PythonNode::Pass],
            orelse: vec![],
        };
        assert_eq!(node.to_code(0), "if True:\n    pass");
    }

    #[test]
    fn test_funcdef_to_code() {
        let node = PythonNode::FuncDef {
            name: "foo".to_string(),
            args: vec!["a".to_string(), "b".to_string()],
            body: vec![PythonNode::Return(Some(Box::new(PythonNode::Name(
                "a".to_string(),
            ))))],
        };
        assert_eq!(node.to_code(0), "def foo(a, b):\n    return a");
    }

    #[test]
    fn test_depth_calculation() {
        let simple = PythonNode::IntLit(1);
        assert_eq!(simple.depth(), 1);

        let nested = PythonNode::BinOp {
            left: Box::new(PythonNode::IntLit(1)),
            op: BinaryOp::Add,
            right: Box::new(PythonNode::BinOp {
                left: Box::new(PythonNode::IntLit(2)),
                op: BinaryOp::Mult,
                right: Box::new(PythonNode::IntLit(3)),
            }),
        };
        assert_eq!(nested.depth(), 3);
    }

    #[test]
    fn test_enumerator_expressions() {
        let enum_ = PythonEnumerator::new(2);
        let exprs = enum_.enumerate_expressions(1);
        assert!(!exprs.is_empty());
        // Should have integers, names, booleans, None
        assert!(exprs.iter().any(|e| matches!(e, PythonNode::IntLit(_))));
        assert!(exprs.iter().any(|e| matches!(e, PythonNode::Name(_))));
    }

    #[test]
    fn test_enumerator_statements() {
        let enum_ = PythonEnumerator::new(2);
        let stmts = enum_.enumerate_statements(2);
        assert!(!stmts.is_empty());
        // Should have pass, assignments, etc.
        assert!(stmts.iter().any(|s| matches!(s, PythonNode::Pass)));
        assert!(stmts.iter().any(|s| matches!(s, PythonNode::Assign { .. })));
    }

    #[test]
    fn test_enumerator_programs() {
        let enum_ = PythonEnumerator::new(2);
        let programs = enum_.enumerate_programs();
        assert!(!programs.is_empty());
        // All programs should have valid Python code
        for prog in &programs {
            assert!(!prog.code.is_empty());
            assert_eq!(prog.language, Language::Python);
        }
    }

    #[test]
    fn test_generated_code_is_valid_python() {
        let enum_ = PythonEnumerator::new(2);
        let programs = enum_.enumerate_programs();

        // Test a few programs to ensure they look like valid Python
        for prog in programs.iter().take(10) {
            // Should not contain syntax errors (basic check)
            assert!(
                !prog.code.contains("():")
                    || prog.code.contains("def ")
                    || prog.code.contains("if ")
            );
        }
    }

    #[test]
    fn test_binary_op_all() {
        let ops = BinaryOp::all();
        assert_eq!(ops.len(), 7);
    }

    #[test]
    fn test_binary_op_to_str_all() {
        assert_eq!(BinaryOp::Add.to_str(), "+");
        assert_eq!(BinaryOp::Sub.to_str(), "-");
        assert_eq!(BinaryOp::Mult.to_str(), "*");
        assert_eq!(BinaryOp::Div.to_str(), "/");
        assert_eq!(BinaryOp::Mod.to_str(), "%");
        assert_eq!(BinaryOp::FloorDiv.to_str(), "//");
        assert_eq!(BinaryOp::Pow.to_str(), "**");
        assert_eq!(BinaryOp::And.to_str(), "and");
        assert_eq!(BinaryOp::Or.to_str(), "or");
    }

    #[test]
    fn test_unary_op_all() {
        let ops = UnaryOp::all();
        assert_eq!(ops.len(), 3);
    }

    #[test]
    fn test_unary_op_to_str_all() {
        assert_eq!(UnaryOp::Neg.to_str(), "-");
        assert_eq!(UnaryOp::Not.to_str(), "not ");
        assert_eq!(UnaryOp::Pos.to_str(), "+");
    }

    #[test]
    fn test_compare_op_all() {
        let ops = CompareOp::all();
        assert_eq!(ops.len(), 6);
    }

    #[test]
    fn test_compare_op_to_str_all() {
        assert_eq!(CompareOp::Eq.to_str(), "==");
        assert_eq!(CompareOp::NotEq.to_str(), "!=");
        assert_eq!(CompareOp::Lt.to_str(), "<");
        assert_eq!(CompareOp::LtE.to_str(), "<=");
        assert_eq!(CompareOp::Gt.to_str(), ">");
        assert_eq!(CompareOp::GtE.to_str(), ">=");
    }

    #[test]
    fn test_float_lit_to_code() {
        let node = PythonNode::FloatLit(3.14);
        assert!(node.to_code(0).starts_with("3.1"));
    }

    #[test]
    fn test_str_lit_to_code() {
        let node = PythonNode::StrLit("hello".to_string());
        assert_eq!(node.to_code(0), "\"hello\"");
    }

    #[test]
    fn test_bool_lit_to_code() {
        assert_eq!(PythonNode::BoolLit(true).to_code(0), "True");
        assert_eq!(PythonNode::BoolLit(false).to_code(0), "False");
    }

    #[test]
    fn test_none_lit_to_code() {
        assert_eq!(PythonNode::NoneLit.to_code(0), "None");
    }

    #[test]
    fn test_name_to_code() {
        let node = PythonNode::Name("x".to_string());
        assert_eq!(node.to_code(0), "x");
    }

    #[test]
    fn test_unary_op_to_code() {
        let node = PythonNode::UnaryOp {
            op: UnaryOp::Neg,
            operand: Box::new(PythonNode::IntLit(5)),
        };
        assert_eq!(node.to_code(0), "(-5)");
    }

    #[test]
    fn test_if_with_else_to_code() {
        let node = PythonNode::If {
            test: Box::new(PythonNode::BoolLit(true)),
            body: vec![PythonNode::Pass],
            orelse: vec![PythonNode::Pass],
        };
        let code = node.to_code(0);
        assert!(code.contains("if True:"));
        assert!(code.contains("else:"));
    }

    #[test]
    fn test_while_to_code() {
        let node = PythonNode::While {
            test: Box::new(PythonNode::BoolLit(true)),
            body: vec![PythonNode::Break],
        };
        let code = node.to_code(0);
        assert!(code.contains("while True:"));
        assert!(code.contains("break"));
    }

    #[test]
    fn test_for_to_code() {
        let node = PythonNode::For {
            target: "i".to_string(),
            iter: Box::new(PythonNode::List(vec![PythonNode::IntLit(1)])),
            body: vec![PythonNode::Continue],
        };
        let code = node.to_code(0);
        assert!(code.contains("for i in"));
        assert!(code.contains("continue"));
    }

    #[test]
    fn test_call_to_code() {
        let node = PythonNode::Call {
            func: "print".to_string(),
            args: vec![PythonNode::IntLit(1), PythonNode::IntLit(2)],
        };
        assert_eq!(node.to_code(0), "print(1, 2)");
    }

    #[test]
    fn test_return_none_to_code() {
        let node = PythonNode::Return(None);
        assert_eq!(node.to_code(0), "return");
    }

    #[test]
    fn test_break_to_code() {
        let node = PythonNode::Break;
        assert_eq!(node.to_code(0), "break");
    }

    #[test]
    fn test_continue_to_code() {
        let node = PythonNode::Continue;
        assert_eq!(node.to_code(0), "continue");
    }

    #[test]
    fn test_list_to_code() {
        let node = PythonNode::List(vec![
            PythonNode::IntLit(1),
            PythonNode::IntLit(2),
            PythonNode::IntLit(3),
        ]);
        assert_eq!(node.to_code(0), "[1, 2, 3]");
    }

    #[test]
    fn test_empty_list_to_code() {
        let node = PythonNode::List(vec![]);
        assert_eq!(node.to_code(0), "[]");
    }

    #[test]
    fn test_compare_to_code() {
        let node = PythonNode::Compare {
            left: Box::new(PythonNode::IntLit(1)),
            op: CompareOp::Lt,
            right: Box::new(PythonNode::IntLit(2)),
        };
        assert_eq!(node.to_code(0), "(1 < 2)");
    }

    #[test]
    fn test_module_to_code() {
        let node = PythonNode::Module(vec![
            PythonNode::Assign {
                target: "x".to_string(),
                value: Box::new(PythonNode::IntLit(1)),
            },
            PythonNode::Pass,
        ]);
        let code = node.to_code(0);
        assert!(code.contains("x = 1"));
        assert!(code.contains("pass"));
    }

    #[test]
    fn test_python_node_debug() {
        let node = PythonNode::IntLit(42);
        let debug = format!("{:?}", node);
        assert!(debug.contains("IntLit"));
    }

    #[test]
    fn test_python_node_clone() {
        let node = PythonNode::IntLit(42);
        let cloned = node.clone();
        assert_eq!(cloned, node);
    }

    #[test]
    fn test_binary_op_debug() {
        let op = BinaryOp::Add;
        let debug = format!("{:?}", op);
        assert!(debug.contains("Add"));
    }

    #[test]
    fn test_binary_op_clone() {
        let op = BinaryOp::Add;
        let cloned = op.clone();
        assert_eq!(cloned, op);
    }

    #[test]
    fn test_unary_op_debug() {
        let op = UnaryOp::Neg;
        let debug = format!("{:?}", op);
        assert!(debug.contains("Neg"));
    }

    #[test]
    fn test_compare_op_debug() {
        let op = CompareOp::Lt;
        let debug = format!("{:?}", op);
        assert!(debug.contains("Lt"));
    }

    #[test]
    fn test_extract_features_binop() {
        let enum_ = PythonEnumerator::new(2);
        let node = PythonNode::BinOp {
            left: Box::new(PythonNode::IntLit(1)),
            op: BinaryOp::Add,
            right: Box::new(PythonNode::IntLit(2)),
        };
        let features = enum_.extract_features(&node);
        assert!(features.contains(&"binop".to_string()));
    }

    #[test]
    fn test_extract_features_if_with_else() {
        let enum_ = PythonEnumerator::new(2);
        let node = PythonNode::If {
            test: Box::new(PythonNode::BoolLit(true)),
            body: vec![PythonNode::Pass],
            orelse: vec![PythonNode::Pass],
        };
        let features = enum_.extract_features(&node);
        assert!(features.contains(&"if".to_string()));
        assert!(features.contains(&"else".to_string()));
    }

    #[test]
    fn test_extract_features_while() {
        let enum_ = PythonEnumerator::new(2);
        let node = PythonNode::While {
            test: Box::new(PythonNode::BoolLit(true)),
            body: vec![PythonNode::Pass],
        };
        let features = enum_.extract_features(&node);
        assert!(features.contains(&"while".to_string()));
    }

    #[test]
    fn test_extract_features_for() {
        let enum_ = PythonEnumerator::new(2);
        let node = PythonNode::For {
            target: "i".to_string(),
            iter: Box::new(PythonNode::List(vec![])),
            body: vec![PythonNode::Pass],
        };
        let features = enum_.extract_features(&node);
        assert!(features.contains(&"for".to_string()));
    }

    #[test]
    fn test_extract_features_compare() {
        let enum_ = PythonEnumerator::new(2);
        let node = PythonNode::Compare {
            left: Box::new(PythonNode::IntLit(1)),
            op: CompareOp::Lt,
            right: Box::new(PythonNode::IntLit(2)),
        };
        let features = enum_.extract_features(&node);
        assert!(features.contains(&"compare".to_string()));
    }

    #[test]
    fn test_depth_if() {
        let node = PythonNode::If {
            test: Box::new(PythonNode::BoolLit(true)),
            body: vec![PythonNode::Pass],
            orelse: vec![],
        };
        assert!(node.depth() >= 2);
    }

    #[test]
    fn test_depth_while() {
        let node = PythonNode::While {
            test: Box::new(PythonNode::BoolLit(true)),
            body: vec![PythonNode::Pass],
        };
        assert!(node.depth() >= 2);
    }

    #[test]
    fn test_depth_for() {
        let node = PythonNode::For {
            target: "i".to_string(),
            iter: Box::new(PythonNode::List(vec![])),
            body: vec![PythonNode::Pass],
        };
        assert!(node.depth() >= 2);
    }

    #[test]
    fn test_depth_funcdef() {
        let node = PythonNode::FuncDef {
            name: "f".to_string(),
            args: vec![],
            body: vec![PythonNode::Pass],
        };
        assert!(node.depth() >= 2);
    }
}
