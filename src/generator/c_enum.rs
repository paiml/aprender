//! C exhaustive enumeration
//!
//! Generates all valid C programs up to a specified AST depth.
//! Uses a simplified C grammar for combinatorial generation.

use super::GeneratedCode;
use crate::Language;

/// C AST node types for generation
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum CNode {
    /// Translation unit (root node)
    TranslationUnit(Vec<CNode>),
    /// Function definition
    FuncDef {
        /// Return type
        return_type: CType,
        /// Function name
        name: String,
        /// Parameters
        params: Vec<(CType, String)>,
        /// Function body
        body: Vec<CNode>,
    },
    /// Variable declaration
    VarDecl {
        /// Variable type
        var_type: CType,
        /// Variable name
        name: String,
        /// Optional initializer
        init: Option<Box<CNode>>,
    },
    /// Assignment: `lhs = rhs`
    Assign {
        /// Left-hand side
        lhs: Box<CNode>,
        /// Right-hand side
        rhs: Box<CNode>,
    },
    /// Binary operation
    BinOp {
        /// Left operand
        left: Box<CNode>,
        /// Operator
        op: CBinaryOp,
        /// Right operand
        right: Box<CNode>,
    },
    /// Unary operation
    UnaryOp {
        /// Operator
        op: CUnaryOp,
        /// Operand
        operand: Box<CNode>,
    },
    /// Integer literal
    IntLit(i64),
    /// Float literal
    FloatLit(f64),
    /// Character literal
    CharLit(char),
    /// String literal
    StrLit(String),
    /// Variable reference
    Ident(String),
    /// If statement
    If {
        /// Condition
        cond: Box<CNode>,
        /// Then body
        then_body: Vec<CNode>,
        /// Else body
        else_body: Vec<CNode>,
    },
    /// While loop
    While {
        /// Condition
        cond: Box<CNode>,
        /// Loop body
        body: Vec<CNode>,
    },
    /// For loop
    For {
        /// Initialization
        init: Option<Box<CNode>>,
        /// Condition
        cond: Option<Box<CNode>>,
        /// Increment
        incr: Option<Box<CNode>>,
        /// Loop body
        body: Vec<CNode>,
    },
    /// Return statement
    Return(Option<Box<CNode>>),
    /// Break statement
    Break,
    /// Continue statement
    Continue,
    /// Function call
    Call {
        /// Function name
        func: String,
        /// Arguments
        args: Vec<CNode>,
    },
    /// Array access
    ArrayAccess {
        /// Array expression
        array: Box<CNode>,
        /// Index expression
        index: Box<CNode>,
    },
    /// Comparison operation
    Compare {
        /// Left operand
        left: Box<CNode>,
        /// Operator
        op: CCompareOp,
        /// Right operand
        right: Box<CNode>,
    },
    /// Expression statement
    ExprStmt(Box<CNode>),
    /// Compound statement (block)
    Block(Vec<CNode>),
    /// Ternary operator: `cond ? then : else`
    Ternary {
        /// Condition
        cond: Box<CNode>,
        /// Then expression
        then_expr: Box<CNode>,
        /// Else expression
        else_expr: Box<CNode>,
    },
    /// Sizeof expression
    Sizeof(CType),
    /// Cast expression
    Cast {
        /// Target type
        target_type: CType,
        /// Expression
        expr: Box<CNode>,
    },
    /// Pointer dereference
    Deref(Box<CNode>),
    /// Address-of
    AddrOf(Box<CNode>),
    /// Struct access: `expr.field`
    StructAccess {
        /// Expression
        expr: Box<CNode>,
        /// Field name
        field: String,
    },
    /// Pointer struct access: `expr->field`
    PtrAccess {
        /// Expression
        expr: Box<CNode>,
        /// Field name
        field: String,
    },
    /// Increment: `++x` or `x++`
    Increment {
        /// Operand
        operand: Box<CNode>,
        /// Pre-increment (++x) or post-increment (x++)
        pre: bool,
    },
    /// Decrement: `--x` or `x--`
    Decrement {
        /// Operand
        operand: Box<CNode>,
        /// Pre-decrement (--x) or post-decrement (x--)
        pre: bool,
    },
}

/// C types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CType {
    /// void
    Void,
    /// int
    Int,
    /// char
    Char,
    /// float
    Float,
    /// double
    Double,
    /// long
    Long,
    /// unsigned int
    UInt,
    /// int pointer
    IntPtr,
    /// char pointer
    CharPtr,
    /// void pointer
    VoidPtr,
}

impl CType {
    /// Get all basic types
    #[must_use]
    pub fn all_basic() -> &'static [Self] {
        &[Self::Int, Self::Char, Self::Float, Self::Double, Self::Long]
    }

    /// Convert to C type string
    #[must_use]
    pub fn to_str(self) -> &'static str {
        match self {
            Self::Void => "void",
            Self::Int => "int",
            Self::Char => "char",
            Self::Float => "float",
            Self::Double => "double",
            Self::Long => "long",
            Self::UInt => "unsigned int",
            Self::IntPtr => "int*",
            Self::CharPtr => "char*",
            Self::VoidPtr => "void*",
        }
    }
}

/// C binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CBinaryOp {
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Modulo
    Mod,
    /// Bitwise AND
    BitAnd,
    /// Bitwise OR
    BitOr,
    /// Bitwise XOR
    BitXor,
    /// Left shift
    Shl,
    /// Right shift
    Shr,
    /// Logical AND
    LogAnd,
    /// Logical OR
    LogOr,
}

impl CBinaryOp {
    /// Get all arithmetic operators
    #[must_use]
    pub fn arithmetic() -> &'static [Self] {
        &[Self::Add, Self::Sub, Self::Mul, Self::Div, Self::Mod]
    }

    /// Get all bitwise operators
    #[must_use]
    pub fn bitwise() -> &'static [Self] {
        &[
            Self::BitAnd,
            Self::BitOr,
            Self::BitXor,
            Self::Shl,
            Self::Shr,
        ]
    }

    /// Convert to C operator string
    #[must_use]
    pub fn to_str(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Mod => "%",
            Self::BitAnd => "&",
            Self::BitOr => "|",
            Self::BitXor => "^",
            Self::Shl => "<<",
            Self::Shr => ">>",
            Self::LogAnd => "&&",
            Self::LogOr => "||",
        }
    }
}

/// C unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CUnaryOp {
    /// Negation
    Neg,
    /// Logical NOT
    Not,
    /// Bitwise NOT
    BitNot,
}

impl CUnaryOp {
    /// Get all unary operators
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[Self::Neg, Self::Not, Self::BitNot]
    }

    /// Convert to C operator string
    #[must_use]
    pub fn to_str(self) -> &'static str {
        match self {
            Self::Neg => "-",
            Self::Not => "!",
            Self::BitNot => "~",
        }
    }
}

/// C comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CCompareOp {
    /// Equal
    Eq,
    /// Not equal
    Ne,
    /// Less than
    Lt,
    /// Greater than
    Gt,
    /// Less than or equal
    Le,
    /// Greater than or equal
    Ge,
}

impl CCompareOp {
    /// Get all comparison operators
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[Self::Eq, Self::Ne, Self::Lt, Self::Gt, Self::Le, Self::Ge]
    }

    /// Convert to C operator string
    #[must_use]
    pub fn to_str(self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Gt => ">",
            Self::Le => "<=",
            Self::Ge => ">=",
        }
    }
}

impl CNode {
    /// Convert AST node to C code string
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn to_code(&self, indent: usize) -> String {
        let indent_str = "    ".repeat(indent);
        match self {
            Self::TranslationUnit(items) => items
                .iter()
                .map(|item| item.to_code(0))
                .collect::<Vec<_>>()
                .join("\n\n"),
            Self::FuncDef {
                return_type,
                name,
                params,
                body,
            } => {
                let params_str = if params.is_empty() {
                    "void".to_string()
                } else {
                    params
                        .iter()
                        .map(|(t, n)| format!("{} {}", t.to_str(), n))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let body_str = body
                    .iter()
                    .map(|s| s.to_code(indent + 1))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "{}{} {}({}) {{\n{}\n{}}}",
                    indent_str,
                    return_type.to_str(),
                    name,
                    params_str,
                    body_str,
                    indent_str
                )
            }
            Self::VarDecl {
                var_type,
                name,
                init,
            } => {
                if let Some(init_expr) = init {
                    format!(
                        "{}{} {} = {};",
                        indent_str,
                        var_type.to_str(),
                        name,
                        init_expr.to_code(0)
                    )
                } else {
                    format!("{}{} {};", indent_str, var_type.to_str(), name)
                }
            }
            Self::Assign { lhs, rhs } => {
                format!("{}{} = {};", indent_str, lhs.to_code(0), rhs.to_code(0))
            }
            Self::BinOp { left, op, right } => {
                format!("({} {} {})", left.to_code(0), op.to_str(), right.to_code(0))
            }
            Self::UnaryOp { op, operand } => {
                format!("({}{})", op.to_str(), operand.to_code(0))
            }
            Self::IntLit(n) => n.to_string(),
            Self::FloatLit(f) => format!("{f:.1}"),
            Self::CharLit(c) => format!("'{c}'"),
            Self::StrLit(s) => format!("\"{s}\""),
            Self::Ident(name) => name.clone(),
            Self::If {
                cond,
                then_body,
                else_body,
            } => {
                let then_str = then_body
                    .iter()
                    .map(|s| s.to_code(indent + 1))
                    .collect::<Vec<_>>()
                    .join("\n");
                if else_body.is_empty() {
                    format!(
                        "{}if ({}) {{\n{}\n{}}}",
                        indent_str,
                        cond.to_code(0),
                        then_str,
                        indent_str
                    )
                } else {
                    let else_str = else_body
                        .iter()
                        .map(|s| s.to_code(indent + 1))
                        .collect::<Vec<_>>()
                        .join("\n");
                    format!(
                        "{}if ({}) {{\n{}\n{}}} else {{\n{}\n{}}}",
                        indent_str,
                        cond.to_code(0),
                        then_str,
                        indent_str,
                        else_str,
                        indent_str
                    )
                }
            }
            Self::While { cond, body } => {
                let body_str = body
                    .iter()
                    .map(|s| s.to_code(indent + 1))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "{}while ({}) {{\n{}\n{}}}",
                    indent_str,
                    cond.to_code(0),
                    body_str,
                    indent_str
                )
            }
            Self::For {
                init,
                cond,
                incr,
                body,
            } => {
                let init_str = init.as_ref().map_or(String::new(), |i| i.to_code(0));
                let cond_str = cond.as_ref().map_or(String::new(), |c| c.to_code(0));
                let incr_str = incr.as_ref().map_or(String::new(), |i| i.to_code(0));
                let body_str = body
                    .iter()
                    .map(|s| s.to_code(indent + 1))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "{indent_str}for ({init_str}; {cond_str}; {incr_str}) {{\n{body_str}\n{indent_str}}}"
                )
            }
            Self::Return(expr) => {
                if let Some(e) = expr {
                    format!("{}return {};", indent_str, e.to_code(0))
                } else {
                    format!("{indent_str}return;")
                }
            }
            Self::Break => format!("{indent_str}break;"),
            Self::Continue => format!("{indent_str}continue;"),
            Self::Call { func, args } => {
                let args_str = args
                    .iter()
                    .map(|a| a.to_code(0))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{func}({args_str})")
            }
            Self::ArrayAccess { array, index } => {
                format!("{}[{}]", array.to_code(0), index.to_code(0))
            }
            Self::Compare { left, op, right } => {
                format!("({} {} {})", left.to_code(0), op.to_str(), right.to_code(0))
            }
            Self::ExprStmt(expr) => format!("{}{};", indent_str, expr.to_code(0)),
            Self::Block(stmts) => {
                let stmts_str = stmts
                    .iter()
                    .map(|s| s.to_code(indent + 1))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("{indent_str}{{\n{stmts_str}\n{indent_str}}}")
            }
            Self::Ternary {
                cond,
                then_expr,
                else_expr,
            } => {
                format!(
                    "({} ? {} : {})",
                    cond.to_code(0),
                    then_expr.to_code(0),
                    else_expr.to_code(0)
                )
            }
            Self::Sizeof(t) => format!("sizeof({})", t.to_str()),
            Self::Cast { target_type, expr } => {
                format!("(({}){})", target_type.to_str(), expr.to_code(0))
            }
            Self::Deref(expr) => format!("(*{})", expr.to_code(0)),
            Self::AddrOf(expr) => format!("(&{})", expr.to_code(0)),
            Self::StructAccess { expr, field } => format!("{}.{}", expr.to_code(0), field),
            Self::PtrAccess { expr, field } => format!("{}->{}", expr.to_code(0), field),
            Self::Increment { operand, pre } => {
                if *pre {
                    format!("++{}", operand.to_code(0))
                } else {
                    format!("{}++", operand.to_code(0))
                }
            }
            Self::Decrement { operand, pre } => {
                if *pre {
                    format!("--{}", operand.to_code(0))
                } else {
                    format!("{}--", operand.to_code(0))
                }
            }
        }
    }

    /// Calculate AST depth
    #[must_use]
    pub fn depth(&self) -> usize {
        match self {
            Self::TranslationUnit(items) => 1 + items.iter().map(Self::depth).max().unwrap_or(0),
            Self::FuncDef { body, .. } => 1 + body.iter().map(Self::depth).max().unwrap_or(0),
            Self::VarDecl { init, .. } => 1 + init.as_ref().map_or(0, |i| i.depth()),
            Self::Assign { lhs, rhs } => 1 + lhs.depth().max(rhs.depth()),
            Self::BinOp { left, right, .. } | Self::Compare { left, right, .. } => {
                1 + left.depth().max(right.depth())
            }
            Self::UnaryOp { operand, .. } => 1 + operand.depth(),
            Self::If {
                cond,
                then_body,
                else_body,
            } => {
                let then_depth = then_body.iter().map(Self::depth).max().unwrap_or(0);
                let else_depth = else_body.iter().map(Self::depth).max().unwrap_or(0);
                1 + cond.depth().max(then_depth).max(else_depth)
            }
            Self::While { cond, body }
            | Self::For {
                cond: Some(cond),
                body,
                ..
            } => {
                let body_depth = body.iter().map(Self::depth).max().unwrap_or(0);
                1 + cond.depth().max(body_depth)
            }
            Self::For { body, .. } => 1 + body.iter().map(Self::depth).max().unwrap_or(0),
            Self::Return(Some(e)) => 1 + e.depth(),
            Self::Call { args, .. } => 1 + args.iter().map(Self::depth).max().unwrap_or(0),
            Self::ArrayAccess { array, index } => 1 + array.depth().max(index.depth()),
            Self::ExprStmt(e) => 1 + e.depth(),
            Self::Block(stmts) => 1 + stmts.iter().map(Self::depth).max().unwrap_or(0),
            Self::Ternary {
                cond,
                then_expr,
                else_expr,
            } => 1 + cond.depth().max(then_expr.depth()).max(else_expr.depth()),
            Self::Cast { expr, .. } | Self::Deref(expr) | Self::AddrOf(expr) => 1 + expr.depth(),
            Self::StructAccess { expr, .. } | Self::PtrAccess { expr, .. } => 1 + expr.depth(),
            Self::Increment { operand, .. } | Self::Decrement { operand, .. } => {
                1 + operand.depth()
            }
            // Terminal nodes
            Self::Return(None)
            | Self::Break
            | Self::Continue
            | Self::IntLit(_)
            | Self::FloatLit(_)
            | Self::CharLit(_)
            | Self::StrLit(_)
            | Self::Ident(_)
            | Self::Sizeof(_) => 1,
        }
    }
}

/// Exhaustive C program enumerator
#[derive(Debug)]
pub struct CEnumerator {
    max_depth: usize,
    var_names: Vec<String>,
    int_values: Vec<i64>,
}

impl Default for CEnumerator {
    fn default() -> Self {
        Self::new(3)
    }
}

impl CEnumerator {
    /// Create a new enumerator with specified max depth
    #[must_use]
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            var_names: vec!["x".to_string(), "y".to_string(), "n".to_string()],
            int_values: vec![0, 1, -1, 2, 10, 42],
        }
    }

    /// Enumerate all expressions up to the given depth
    pub fn enumerate_expressions(&self, depth: usize) -> Vec<CNode> {
        if depth == 0 {
            return vec![];
        }

        let mut results = Vec::new();

        // Depth 1: literals and names
        for val in &self.int_values {
            results.push(CNode::IntLit(*val));
        }
        for name in &self.var_names {
            results.push(CNode::Ident(name.clone()));
        }

        if depth >= 2 {
            // Binary operations
            let sub_exprs = self.enumerate_expressions(depth - 1);
            let limited: Vec<_> = sub_exprs.iter().take(15).collect();

            for left in &limited {
                for right in limited.iter().take(10) {
                    for op in CBinaryOp::arithmetic() {
                        results.push(CNode::BinOp {
                            left: Box::new((*left).clone()),
                            op: *op,
                            right: Box::new((*right).clone()),
                        });
                    }
                }
            }

            // Comparisons
            for left in &limited {
                for right in limited.iter().take(10) {
                    for op in CCompareOp::all() {
                        results.push(CNode::Compare {
                            left: Box::new((*left).clone()),
                            op: *op,
                            right: Box::new((*right).clone()),
                        });
                    }
                }
            }

            // Unary operations
            for operand in limited.iter().take(8) {
                for op in CUnaryOp::all() {
                    results.push(CNode::UnaryOp {
                        op: *op,
                        operand: Box::new((*operand).clone()),
                    });
                }
            }

            // Increment/Decrement
            for name in &self.var_names {
                results.push(CNode::Increment {
                    operand: Box::new(CNode::Ident(name.clone())),
                    pre: true,
                });
                results.push(CNode::Increment {
                    operand: Box::new(CNode::Ident(name.clone())),
                    pre: false,
                });
            }
        }

        results
    }

    /// Enumerate all statements up to the given depth
    pub fn enumerate_statements(&self, depth: usize) -> Vec<CNode> {
        if depth == 0 {
            return vec![];
        }

        let mut results = Vec::new();

        let exprs = self.enumerate_expressions(depth - 1);
        let limited_exprs: Vec<_> = exprs.iter().take(20).collect();

        // Variable declarations
        for var_type in CType::all_basic() {
            for name in &self.var_names {
                results.push(CNode::VarDecl {
                    var_type: *var_type,
                    name: name.clone(),
                    init: None,
                });
                // With initialization
                for val in self.int_values.iter().take(3) {
                    results.push(CNode::VarDecl {
                        var_type: *var_type,
                        name: name.clone(),
                        init: Some(Box::new(CNode::IntLit(*val))),
                    });
                }
            }
        }

        // Assignments
        for name in &self.var_names {
            for value in &limited_exprs {
                results.push(CNode::Assign {
                    lhs: Box::new(CNode::Ident(name.clone())),
                    rhs: Box::new((*value).clone()),
                });
            }
        }

        // Return statements
        results.push(CNode::Return(None));
        for expr in limited_exprs.iter().take(10) {
            results.push(CNode::Return(Some(Box::new((*expr).clone()))));
        }

        // Break and continue
        results.push(CNode::Break);
        results.push(CNode::Continue);

        if depth >= 2 {
            // If statements
            let conditions: Vec<_> = exprs
                .iter()
                .filter(|e| matches!(e, CNode::Compare { .. } | CNode::Ident(_)))
                .take(5)
                .collect();

            let body_stmts = self.enumerate_statements(depth - 1);
            let limited_body: Vec<_> = body_stmts.iter().take(5).collect();

            for cond in &conditions {
                for body in &limited_body {
                    results.push(CNode::If {
                        cond: Box::new((*cond).clone()),
                        then_body: vec![(*body).clone()],
                        else_body: vec![],
                    });
                }
            }

            // While loops
            for cond in &conditions {
                results.push(CNode::While {
                    cond: Box::new((*cond).clone()),
                    body: vec![CNode::Break],
                });
            }

            // For loops
            for name in self.var_names.iter().take(2) {
                results.push(CNode::For {
                    init: Some(Box::new(CNode::VarDecl {
                        var_type: CType::Int,
                        name: name.clone(),
                        init: Some(Box::new(CNode::IntLit(0))),
                    })),
                    cond: Some(Box::new(CNode::Compare {
                        left: Box::new(CNode::Ident(name.clone())),
                        op: CCompareOp::Lt,
                        right: Box::new(CNode::IntLit(10)),
                    })),
                    incr: Some(Box::new(CNode::Increment {
                        operand: Box::new(CNode::Ident(name.clone())),
                        pre: false,
                    })),
                    body: vec![CNode::Break],
                });
            }
        }

        results
    }

    /// Enumerate complete programs
    #[must_use]
    pub fn enumerate_programs(&self) -> Vec<GeneratedCode> {
        let mut results = Vec::new();

        let stmts = self.enumerate_statements(self.max_depth);

        // Simple main functions with single statement
        for stmt in stmts.iter().take(50) {
            let func = CNode::FuncDef {
                return_type: CType::Int,
                name: "main".to_string(),
                params: vec![],
                body: vec![
                    stmt.clone(),
                    CNode::Return(Some(Box::new(CNode::IntLit(0)))),
                ],
            };
            let unit = CNode::TranslationUnit(vec![func]);
            let code = unit.to_code(0);
            results.push(GeneratedCode {
                code,
                language: Language::C,
                ast_depth: stmt.depth() + 2,
                features: self.extract_features(stmt),
            });
        }

        // Functions with parameters
        for stmt in stmts.iter().take(20) {
            let func = CNode::FuncDef {
                return_type: CType::Int,
                name: "compute".to_string(),
                params: vec![(CType::Int, "a".to_string()), (CType::Int, "b".to_string())],
                body: vec![stmt.clone()],
            };
            let unit = CNode::TranslationUnit(vec![func]);
            let code = unit.to_code(0);
            results.push(GeneratedCode {
                code,
                language: Language::C,
                ast_depth: stmt.depth() + 2,
                features: self.extract_features(stmt),
            });
        }

        results
    }

    /// Extract feature labels from an AST node
    fn extract_features(&self, node: &CNode) -> Vec<String> {
        let mut features = Vec::new();

        match node {
            CNode::VarDecl { .. } => features.push("var_decl".to_string()),
            CNode::Assign { .. } => features.push("assignment".to_string()),
            CNode::BinOp { op, .. } => {
                features.push("binop".to_string());
                features.push(format!("op_{}", op.to_str()));
            }
            CNode::If { else_body, .. } => {
                features.push("if".to_string());
                if !else_body.is_empty() {
                    features.push("else".to_string());
                }
            }
            CNode::While { .. } => features.push("while".to_string()),
            CNode::For { .. } => features.push("for".to_string()),
            CNode::Return(_) => features.push("return".to_string()),
            CNode::Compare { op, .. } => {
                features.push("compare".to_string());
                features.push(format!("cmp_{}", op.to_str()));
            }
            CNode::Increment { .. } => features.push("increment".to_string()),
            CNode::Decrement { .. } => features.push("decrement".to_string()),
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
        let node = CNode::IntLit(42);
        assert_eq!(node.to_code(0), "42");
    }

    #[test]
    fn test_var_decl_to_code() {
        let node = CNode::VarDecl {
            var_type: CType::Int,
            name: "x".to_string(),
            init: Some(Box::new(CNode::IntLit(0))),
        };
        assert_eq!(node.to_code(0), "int x = 0;");
    }

    #[test]
    fn test_binop_to_code() {
        let node = CNode::BinOp {
            left: Box::new(CNode::Ident("x".to_string())),
            op: CBinaryOp::Add,
            right: Box::new(CNode::IntLit(1)),
        };
        assert_eq!(node.to_code(0), "(x + 1)");
    }

    #[test]
    fn test_func_def_to_code() {
        let node = CNode::FuncDef {
            return_type: CType::Int,
            name: "main".to_string(),
            params: vec![],
            body: vec![CNode::Return(Some(Box::new(CNode::IntLit(0))))],
        };
        let code = node.to_code(0);
        assert!(code.contains("int main(void)"));
        assert!(code.contains("return 0;"));
    }

    #[test]
    fn test_if_to_code() {
        let node = CNode::If {
            cond: Box::new(CNode::Ident("x".to_string())),
            then_body: vec![CNode::Return(Some(Box::new(CNode::IntLit(1))))],
            else_body: vec![],
        };
        let code = node.to_code(0);
        assert!(code.contains("if (x)"));
        assert!(code.contains("return 1;"));
    }

    #[test]
    fn test_for_to_code() {
        let node = CNode::For {
            init: Some(Box::new(CNode::VarDecl {
                var_type: CType::Int,
                name: "i".to_string(),
                init: Some(Box::new(CNode::IntLit(0))),
            })),
            cond: Some(Box::new(CNode::Compare {
                left: Box::new(CNode::Ident("i".to_string())),
                op: CCompareOp::Lt,
                right: Box::new(CNode::IntLit(10)),
            })),
            incr: Some(Box::new(CNode::Increment {
                operand: Box::new(CNode::Ident("i".to_string())),
                pre: false,
            })),
            body: vec![CNode::Break],
        };
        let code = node.to_code(0);
        assert!(code.contains("for (int i = 0;"));
        assert!(code.contains("(i < 10)"));
        assert!(code.contains("i++"));
    }

    #[test]
    fn test_enumerator_creates_programs() {
        let enumerator = CEnumerator::new(2);
        let programs = enumerator.enumerate_programs();
        assert!(!programs.is_empty(), "Should generate programs");
    }

    #[test]
    fn test_programs_are_c() {
        let enumerator = CEnumerator::new(2);
        let programs = enumerator.enumerate_programs();
        for prog in &programs {
            assert_eq!(prog.language, Language::C);
        }
    }

    #[test]
    fn test_depth_calculation() {
        let node = CNode::BinOp {
            left: Box::new(CNode::IntLit(1)),
            op: CBinaryOp::Add,
            right: Box::new(CNode::BinOp {
                left: Box::new(CNode::IntLit(2)),
                op: CBinaryOp::Mul,
                right: Box::new(CNode::IntLit(3)),
            }),
        };
        assert_eq!(node.depth(), 3);
    }

    #[test]
    fn test_compare_to_code() {
        let node = CNode::Compare {
            left: Box::new(CNode::Ident("x".to_string())),
            op: CCompareOp::Lt,
            right: Box::new(CNode::IntLit(10)),
        };
        assert_eq!(node.to_code(0), "(x < 10)");
    }

    #[test]
    fn test_increment_to_code() {
        let pre = CNode::Increment {
            operand: Box::new(CNode::Ident("x".to_string())),
            pre: true,
        };
        let post = CNode::Increment {
            operand: Box::new(CNode::Ident("x".to_string())),
            pre: false,
        };
        assert_eq!(pre.to_code(0), "++x");
        assert_eq!(post.to_code(0), "x++");
    }

    #[test]
    fn test_c_type_to_str() {
        assert_eq!(CType::Int.to_str(), "int");
        assert_eq!(CType::Void.to_str(), "void");
        assert_eq!(CType::IntPtr.to_str(), "int*");
    }

    #[test]
    fn test_extract_features() {
        let enumerator = CEnumerator::new(2);
        let node = CNode::If {
            cond: Box::new(CNode::Ident("x".to_string())),
            then_body: vec![CNode::Break],
            else_body: vec![CNode::Continue],
        };
        let features = enumerator.extract_features(&node);
        assert!(features.contains(&"if".to_string()));
        assert!(features.contains(&"else".to_string()));
    }
}
