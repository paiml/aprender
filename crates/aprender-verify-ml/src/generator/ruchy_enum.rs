//! Ruchy exhaustive enumeration
//!
//! Generates all valid Ruchy programs up to a specified AST depth.
//! Ruchy is a Rust-like language with actor model and effect system.

use super::GeneratedCode;
use crate::Language;

/// Ruchy AST node types for generation
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum RuchyNode {
    /// Module (root node)
    Module(Vec<RuchyNode>),
    /// Function definition
    FnDef {
        /// Function name
        name: String,
        /// Parameters (name, type)
        params: Vec<(String, RuchyType)>,
        /// Return type
        return_type: Option<RuchyType>,
        /// Function body
        body: Vec<RuchyNode>,
    },
    /// Let binding
    Let {
        /// Variable name
        name: String,
        /// Optional type annotation
        ty: Option<RuchyType>,
        /// Value expression
        value: Box<RuchyNode>,
        /// Is mutable
        mutable: bool,
    },
    /// Assignment
    Assign {
        /// Target
        target: Box<RuchyNode>,
        /// Value
        value: Box<RuchyNode>,
    },
    /// Binary operation
    BinOp {
        /// Left operand
        left: Box<RuchyNode>,
        /// Operator
        op: RuchyBinaryOp,
        /// Right operand
        right: Box<RuchyNode>,
    },
    /// Unary operation
    UnaryOp {
        /// Operator
        op: RuchyUnaryOp,
        /// Operand
        operand: Box<RuchyNode>,
    },
    /// Integer literal
    IntLit(i64),
    /// Float literal
    FloatLit(f64),
    /// String literal
    StrLit(String),
    /// Boolean literal
    BoolLit(bool),
    /// Identifier
    Ident(String),
    /// If expression
    If {
        /// Condition
        cond: Box<RuchyNode>,
        /// Then branch
        then_body: Vec<RuchyNode>,
        /// Else branch
        else_body: Vec<RuchyNode>,
    },
    /// Match expression
    Match {
        /// Value being matched
        value: Box<RuchyNode>,
        /// Match arms (pattern, body)
        arms: Vec<(RuchyNode, Vec<RuchyNode>)>,
    },
    /// While loop
    While {
        /// Condition
        cond: Box<RuchyNode>,
        /// Body
        body: Vec<RuchyNode>,
    },
    /// For loop (for x in iter)
    For {
        /// Variable name
        var: String,
        /// Iterator expression
        iter: Box<RuchyNode>,
        /// Body
        body: Vec<RuchyNode>,
    },
    /// Return expression
    Return(Option<Box<RuchyNode>>),
    /// Break expression
    Break,
    /// Continue expression
    Continue,
    /// Function call
    Call {
        /// Function expression
        func: Box<RuchyNode>,
        /// Arguments
        args: Vec<RuchyNode>,
    },
    /// Method call
    MethodCall {
        /// Receiver
        receiver: Box<RuchyNode>,
        /// Method name
        method: String,
        /// Arguments
        args: Vec<RuchyNode>,
    },
    /// Struct definition
    StructDef {
        /// Struct name
        name: String,
        /// Fields (name, type)
        fields: Vec<(String, RuchyType)>,
    },
    /// Struct instantiation
    StructInit {
        /// Struct name
        name: String,
        /// Field initializers
        fields: Vec<(String, RuchyNode)>,
    },
    /// Field access
    FieldAccess {
        /// Receiver
        receiver: Box<RuchyNode>,
        /// Field name
        field: String,
    },
    /// Optional chaining (?.)
    OptionalChain {
        /// Receiver
        receiver: Box<RuchyNode>,
        /// Field name
        field: String,
    },
    /// Pipeline operator (|>)
    Pipeline {
        /// Left expression
        left: Box<RuchyNode>,
        /// Right function
        right: Box<RuchyNode>,
    },
    /// Array literal
    Array(Vec<RuchyNode>),
    /// Range (start..end)
    Range {
        /// Start
        start: Box<RuchyNode>,
        /// End
        end: Box<RuchyNode>,
        /// Inclusive
        inclusive: bool,
    },
    /// Closure
    Closure {
        /// Parameters
        params: Vec<String>,
        /// Body
        body: Box<RuchyNode>,
    },
    /// Block expression
    Block(Vec<RuchyNode>),
    /// Actor spawn
    Spawn(Box<RuchyNode>),
    /// Send message
    Send {
        /// Target actor
        target: Box<RuchyNode>,
        /// Message
        message: Box<RuchyNode>,
    },
    /// Comparison
    Compare {
        /// Left operand
        left: Box<RuchyNode>,
        /// Operator
        op: RuchyCompareOp,
        /// Right operand
        right: Box<RuchyNode>,
    },
    /// Null coalescing (??)
    NullCoalesce {
        /// Value
        value: Box<RuchyNode>,
        /// Default
        default: Box<RuchyNode>,
    },
    /// Pattern for match
    Pattern(RuchyPattern),
}

/// Ruchy types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuchyType {
    /// i32
    I32,
    /// i64
    I64,
    /// f64
    F64,
    /// bool
    Bool,
    /// String
    String,
    /// Unit ()
    Unit,
    /// Option<T>
    Option(Box<RuchyType>),
    /// Vec<T>
    Vec(Box<RuchyType>),
    /// Custom type
    Custom(String),
}

impl RuchyType {
    /// Get all basic types
    #[must_use]
    pub fn all_basic() -> &'static [Self] {
        &[Self::I32, Self::I64, Self::F64, Self::Bool, Self::String]
    }

    /// Convert to Ruchy type string
    #[must_use]
    pub fn to_str(&self) -> String {
        match self {
            Self::I32 => "i32".to_string(),
            Self::I64 => "i64".to_string(),
            Self::F64 => "f64".to_string(),
            Self::Bool => "bool".to_string(),
            Self::String => "String".to_string(),
            Self::Unit => "()".to_string(),
            Self::Option(inner) => format!("Option<{}>", inner.to_str()),
            Self::Vec(inner) => format!("Vec<{}>", inner.to_str()),
            Self::Custom(name) => name.clone(),
        }
    }
}

/// Ruchy binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuchyBinaryOp {
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
    /// Logical AND
    And,
    /// Logical OR
    Or,
    /// Bitwise AND
    BitAnd,
    /// Bitwise OR
    BitOr,
    /// Bitwise XOR
    BitXor,
}

impl RuchyBinaryOp {
    /// Get arithmetic operators
    #[must_use]
    pub fn arithmetic() -> &'static [Self] {
        &[Self::Add, Self::Sub, Self::Mul, Self::Div, Self::Mod]
    }

    /// Convert to Ruchy operator string
    #[must_use]
    pub fn to_str(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Mod => "%",
            Self::And => "&&",
            Self::Or => "||",
            Self::BitAnd => "&",
            Self::BitOr => "|",
            Self::BitXor => "^",
        }
    }
}

/// Ruchy unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuchyUnaryOp {
    /// Negation
    Neg,
    /// Logical NOT
    Not,
    /// Reference
    Ref,
    /// Mutable reference
    RefMut,
    /// Dereference
    Deref,
}

impl RuchyUnaryOp {
    /// Get all unary operators
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[Self::Neg, Self::Not]
    }

    /// Convert to Ruchy operator string
    #[must_use]
    pub fn to_str(self) -> &'static str {
        match self {
            Self::Neg => "-",
            Self::Not => "!",
            Self::Ref => "&",
            Self::RefMut => "&mut ",
            Self::Deref => "*",
        }
    }
}

/// Ruchy comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuchyCompareOp {
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

impl RuchyCompareOp {
    /// Get all comparison operators
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[Self::Eq, Self::Ne, Self::Lt, Self::Gt, Self::Le, Self::Ge]
    }

    /// Convert to Ruchy operator string
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

/// Ruchy patterns for match
#[derive(Debug, Clone, PartialEq)]
pub enum RuchyPattern {
    /// Wildcard _
    Wildcard,
    /// Variable binding
    Binding(String),
    /// Literal
    Literal(Box<RuchyNode>),
    /// Tuple pattern
    Tuple(Vec<RuchyPattern>),
}

impl RuchyNode {
    /// Convert AST node to Ruchy code string
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn to_code(&self, indent: usize) -> String {
        let indent_str = "    ".repeat(indent);
        match self {
            Self::Module(items) => items
                .iter()
                .map(|item| item.to_code(0))
                .collect::<Vec<_>>()
                .join("\n\n"),
            Self::FnDef {
                name,
                params,
                return_type,
                body,
            } => {
                let params_str = params
                    .iter()
                    .map(|(n, t)| format!("{}: {}", n, t.to_str()))
                    .collect::<Vec<_>>()
                    .join(", ");
                let ret_str = return_type
                    .as_ref()
                    .map_or(String::new(), |t| format!(" -> {}", t.to_str()));
                let body_str = body
                    .iter()
                    .map(|s| s.to_code(indent + 1))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "{indent_str}fn {name}({params_str}){ret_str} {{\n{body_str}\n{indent_str}}}"
                )
            }
            Self::Let {
                name,
                ty,
                value,
                mutable,
            } => {
                let mut_str = if *mutable { "mut " } else { "" };
                let ty_str = ty
                    .as_ref()
                    .map_or(String::new(), |t| format!(": {}", t.to_str()));
                format!(
                    "{}let {}{}{} = {};",
                    indent_str,
                    mut_str,
                    name,
                    ty_str,
                    value.to_code(0)
                )
            }
            Self::Assign { target, value } => {
                format!(
                    "{}{} = {};",
                    indent_str,
                    target.to_code(0),
                    value.to_code(0)
                )
            }
            Self::BinOp { left, op, right } => {
                format!("({} {} {})", left.to_code(0), op.to_str(), right.to_code(0))
            }
            Self::UnaryOp { op, operand } => {
                format!("({}{})", op.to_str(), operand.to_code(0))
            }
            Self::IntLit(n) => n.to_string(),
            Self::FloatLit(f) => format!("{f:.1}"),
            Self::StrLit(s) => format!("\"{s}\""),
            Self::BoolLit(b) => b.to_string(),
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
                        "{}if {} {{\n{}\n{}}}",
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
                        "{}if {} {{\n{}\n{}}} else {{\n{}\n{}}}",
                        indent_str,
                        cond.to_code(0),
                        then_str,
                        indent_str,
                        else_str,
                        indent_str
                    )
                }
            }
            Self::Match { value, arms } => {
                let arms_str = arms
                    .iter()
                    .map(|(pat, body)| {
                        let body_str = body
                            .iter()
                            .map(|s| s.to_code(0))
                            .collect::<Vec<_>>()
                            .join("; ");
                        format!("{}    {} => {{ {} }}", indent_str, pat.to_code(0), body_str)
                    })
                    .collect::<Vec<_>>()
                    .join(",\n");
                format!(
                    "{}match {} {{\n{}\n{}}}",
                    indent_str,
                    value.to_code(0),
                    arms_str,
                    indent_str
                )
            }
            Self::While { cond, body } => {
                let body_str = body
                    .iter()
                    .map(|s| s.to_code(indent + 1))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "{}while {} {{\n{}\n{}}}",
                    indent_str,
                    cond.to_code(0),
                    body_str,
                    indent_str
                )
            }
            Self::For { var, iter, body } => {
                let body_str = body
                    .iter()
                    .map(|s| s.to_code(indent + 1))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "{}for {} in {} {{\n{}\n{}}}",
                    indent_str,
                    var,
                    iter.to_code(0),
                    body_str,
                    indent_str
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
                format!("{}({})", func.to_code(0), args_str)
            }
            Self::MethodCall {
                receiver,
                method,
                args,
            } => {
                let args_str = args
                    .iter()
                    .map(|a| a.to_code(0))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}.{}({})", receiver.to_code(0), method, args_str)
            }
            Self::StructDef { name, fields } => {
                let fields_str = fields
                    .iter()
                    .map(|(n, t)| format!("{}    {}: {},", indent_str, n, t.to_str()))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("{indent_str}struct {name} {{\n{fields_str}\n{indent_str}}}")
            }
            Self::StructInit { name, fields } => {
                let fields_str = fields
                    .iter()
                    .map(|(n, v)| format!("{}: {}", n, v.to_code(0)))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{name} {{ {fields_str} }}")
            }
            Self::FieldAccess { receiver, field } => {
                format!("{}.{}", receiver.to_code(0), field)
            }
            Self::OptionalChain { receiver, field } => {
                format!("{}?.{}", receiver.to_code(0), field)
            }
            Self::Pipeline { left, right } => {
                format!("{} |> {}", left.to_code(0), right.to_code(0))
            }
            Self::Array(items) => {
                let items_str = items
                    .iter()
                    .map(|i| i.to_code(0))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{items_str}]")
            }
            Self::Range {
                start,
                end,
                inclusive,
            } => {
                let op = if *inclusive { "..=" } else { ".." };
                format!("{}{}{}", start.to_code(0), op, end.to_code(0))
            }
            Self::Closure { params, body } => {
                let params_str = params.join(", ");
                format!("|{}| {}", params_str, body.to_code(0))
            }
            Self::Block(stmts) => {
                let stmts_str = stmts
                    .iter()
                    .map(|s| s.to_code(indent + 1))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("{indent_str}{{\n{stmts_str}\n{indent_str}}}")
            }
            Self::Spawn(expr) => format!("spawn {}", expr.to_code(0)),
            Self::Send { target, message } => {
                format!("{} <- {}", target.to_code(0), message.to_code(0))
            }
            Self::Compare { left, op, right } => {
                format!("({} {} {})", left.to_code(0), op.to_str(), right.to_code(0))
            }
            Self::NullCoalesce { value, default } => {
                format!("{} ?? {}", value.to_code(0), default.to_code(0))
            }
            Self::Pattern(pat) => pat.to_code(),
        }
    }

    /// Calculate AST depth
    #[must_use]
    pub fn depth(&self) -> usize {
        match self {
            Self::Module(items) => 1 + items.iter().map(Self::depth).max().unwrap_or(0),
            Self::FnDef { body, .. } => 1 + body.iter().map(Self::depth).max().unwrap_or(0),
            Self::Let { value, .. } => 1 + value.depth(),
            Self::Assign { target, value } => 1 + target.depth().max(value.depth()),
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
            Self::Match { value, arms } => {
                let arms_depth = arms
                    .iter()
                    .flat_map(|(_, body)| body.iter().map(Self::depth))
                    .max()
                    .unwrap_or(0);
                1 + value.depth().max(arms_depth)
            }
            Self::While { cond, body } => {
                let body_depth = body.iter().map(Self::depth).max().unwrap_or(0);
                1 + cond.depth().max(body_depth)
            }
            Self::For { iter, body, .. } => {
                let body_depth = body.iter().map(Self::depth).max().unwrap_or(0);
                1 + iter.depth().max(body_depth)
            }
            Self::Return(Some(e)) => 1 + e.depth(),
            Self::Call { func, args } => {
                let args_depth = args.iter().map(Self::depth).max().unwrap_or(0);
                1 + func.depth().max(args_depth)
            }
            Self::MethodCall { receiver, args, .. } => {
                let args_depth = args.iter().map(Self::depth).max().unwrap_or(0);
                1 + receiver.depth().max(args_depth)
            }
            Self::FieldAccess { receiver, .. } | Self::OptionalChain { receiver, .. } => {
                1 + receiver.depth()
            }
            Self::Pipeline { left, right }
            | Self::NullCoalesce {
                value: left,
                default: right,
            } => 1 + left.depth().max(right.depth()),
            Self::Array(items) => 1 + items.iter().map(Self::depth).max().unwrap_or(0),
            Self::Range { start, end, .. } => 1 + start.depth().max(end.depth()),
            Self::Closure { body, .. } => 1 + body.depth(),
            Self::Block(stmts) => 1 + stmts.iter().map(Self::depth).max().unwrap_or(0),
            Self::Spawn(e) => 1 + e.depth(),
            Self::Send { target, message } => 1 + target.depth().max(message.depth()),
            Self::StructInit { fields, .. } => {
                1 + fields.iter().map(|(_, v)| v.depth()).max().unwrap_or(0)
            }
            // Terminal nodes
            Self::Return(None)
            | Self::Break
            | Self::Continue
            | Self::IntLit(_)
            | Self::FloatLit(_)
            | Self::StrLit(_)
            | Self::BoolLit(_)
            | Self::Ident(_)
            | Self::StructDef { .. }
            | Self::Pattern(_) => 1,
        }
    }
}

impl RuchyPattern {
    /// Convert pattern to code
    #[must_use]
    pub fn to_code(&self) -> String {
        match self {
            Self::Wildcard => "_".to_string(),
            Self::Binding(name) => name.clone(),
            Self::Literal(lit) => lit.to_code(0),
            Self::Tuple(pats) => {
                let pats_str = pats
                    .iter()
                    .map(Self::to_code)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({pats_str})")
            }
        }
    }
}

/// Exhaustive Ruchy program enumerator
#[derive(Debug)]
pub struct RuchyEnumerator {
    max_depth: usize,
    var_names: Vec<String>,
    int_values: Vec<i64>,
}

impl Default for RuchyEnumerator {
    fn default() -> Self {
        Self::new(3)
    }
}

impl RuchyEnumerator {
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
    pub fn enumerate_expressions(&self, depth: usize) -> Vec<RuchyNode> {
        if depth == 0 {
            return vec![];
        }

        let mut results = Vec::new();

        // Depth 1: literals and names
        for val in &self.int_values {
            results.push(RuchyNode::IntLit(*val));
        }
        results.push(RuchyNode::BoolLit(true));
        results.push(RuchyNode::BoolLit(false));
        for name in &self.var_names {
            results.push(RuchyNode::Ident(name.clone()));
        }

        if depth >= 2 {
            // Binary operations
            let sub_exprs = self.enumerate_expressions(depth - 1);
            let limited: Vec<_> = sub_exprs.iter().take(15).collect();

            for left in &limited {
                for right in limited.iter().take(10) {
                    for op in RuchyBinaryOp::arithmetic() {
                        results.push(RuchyNode::BinOp {
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
                    for op in RuchyCompareOp::all() {
                        results.push(RuchyNode::Compare {
                            left: Box::new((*left).clone()),
                            op: *op,
                            right: Box::new((*right).clone()),
                        });
                    }
                }
            }

            // Unary operations
            for operand in limited.iter().take(8) {
                for op in RuchyUnaryOp::all() {
                    results.push(RuchyNode::UnaryOp {
                        op: *op,
                        operand: Box::new((*operand).clone()),
                    });
                }
            }

            // Ranges
            for start in self.int_values.iter().take(3) {
                for end in self.int_values.iter().take(3) {
                    results.push(RuchyNode::Range {
                        start: Box::new(RuchyNode::IntLit(*start)),
                        end: Box::new(RuchyNode::IntLit(*end)),
                        inclusive: false,
                    });
                }
            }

            // Pipeline
            for left in limited.iter().take(5) {
                results.push(RuchyNode::Pipeline {
                    left: Box::new((*left).clone()),
                    right: Box::new(RuchyNode::Ident("f".to_string())),
                });
            }
        }

        results
    }

    /// Enumerate all statements up to the given depth
    pub fn enumerate_statements(&self, depth: usize) -> Vec<RuchyNode> {
        if depth == 0 {
            return vec![];
        }

        let mut results = Vec::new();

        let exprs = self.enumerate_expressions(depth - 1);
        let limited_exprs: Vec<_> = exprs.iter().take(20).collect();

        // Let bindings
        for name in &self.var_names {
            for value in &limited_exprs {
                results.push(RuchyNode::Let {
                    name: name.clone(),
                    ty: None,
                    value: Box::new((*value).clone()),
                    mutable: false,
                });
                results.push(RuchyNode::Let {
                    name: name.clone(),
                    ty: None,
                    value: Box::new((*value).clone()),
                    mutable: true,
                });
            }
        }

        // Return statements
        results.push(RuchyNode::Return(None));
        for expr in limited_exprs.iter().take(10) {
            results.push(RuchyNode::Return(Some(Box::new((*expr).clone()))));
        }

        // Break and continue
        results.push(RuchyNode::Break);
        results.push(RuchyNode::Continue);

        if depth >= 2 {
            // If expressions
            let conditions: Vec<_> = exprs
                .iter()
                .filter(|e| matches!(e, RuchyNode::Compare { .. } | RuchyNode::BoolLit(_)))
                .take(5)
                .collect();

            let body_stmts = self.enumerate_statements(depth - 1);
            let limited_body: Vec<_> = body_stmts.iter().take(5).collect();

            for cond in &conditions {
                for body in &limited_body {
                    results.push(RuchyNode::If {
                        cond: Box::new((*cond).clone()),
                        then_body: vec![(*body).clone()],
                        else_body: vec![],
                    });
                }
            }

            // While loops
            for cond in &conditions {
                results.push(RuchyNode::While {
                    cond: Box::new((*cond).clone()),
                    body: vec![RuchyNode::Break],
                });
            }

            // For loops
            for name in self.var_names.iter().take(2) {
                results.push(RuchyNode::For {
                    var: name.clone(),
                    iter: Box::new(RuchyNode::Range {
                        start: Box::new(RuchyNode::IntLit(0)),
                        end: Box::new(RuchyNode::IntLit(10)),
                        inclusive: false,
                    }),
                    body: vec![RuchyNode::Break],
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
            let func = RuchyNode::FnDef {
                name: "main".to_string(),
                params: vec![],
                return_type: None,
                body: vec![stmt.clone()],
            };
            let module = RuchyNode::Module(vec![func]);
            let code = module.to_code(0);
            results.push(GeneratedCode {
                code,
                language: Language::Ruchy,
                ast_depth: stmt.depth() + 2,
                features: self.extract_features(stmt),
            });
        }

        // Functions with parameters
        for stmt in stmts.iter().take(20) {
            let func = RuchyNode::FnDef {
                name: "compute".to_string(),
                params: vec![
                    ("a".to_string(), RuchyType::I32),
                    ("b".to_string(), RuchyType::I32),
                ],
                return_type: Some(RuchyType::I32),
                body: vec![stmt.clone()],
            };
            let module = RuchyNode::Module(vec![func]);
            let code = module.to_code(0);
            results.push(GeneratedCode {
                code,
                language: Language::Ruchy,
                ast_depth: stmt.depth() + 2,
                features: self.extract_features(stmt),
            });
        }

        results
    }

    /// Extract feature labels from an AST node
    fn extract_features(&self, node: &RuchyNode) -> Vec<String> {
        let mut features = Vec::new();

        match node {
            RuchyNode::Let { mutable, .. } => {
                features.push("let".to_string());
                if *mutable {
                    features.push("mut".to_string());
                }
            }
            RuchyNode::BinOp { op, .. } => {
                features.push("binop".to_string());
                features.push(format!("op_{}", op.to_str()));
            }
            RuchyNode::If { else_body, .. } => {
                features.push("if".to_string());
                if !else_body.is_empty() {
                    features.push("else".to_string());
                }
            }
            RuchyNode::While { .. } => features.push("while".to_string()),
            RuchyNode::For { .. } => features.push("for".to_string()),
            RuchyNode::Return(_) => features.push("return".to_string()),
            RuchyNode::Compare { op, .. } => {
                features.push("compare".to_string());
                features.push(format!("cmp_{}", op.to_str()));
            }
            RuchyNode::Pipeline { .. } => features.push("pipeline".to_string()),
            RuchyNode::Range { inclusive, .. } => {
                features.push("range".to_string());
                if *inclusive {
                    features.push("inclusive".to_string());
                }
            }
            RuchyNode::Match { .. } => features.push("match".to_string()),
            RuchyNode::Spawn(_) => features.push("spawn".to_string()),
            RuchyNode::Send { .. } => features.push("send".to_string()),
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
        let node = RuchyNode::IntLit(42);
        assert_eq!(node.to_code(0), "42");
    }

    #[test]
    fn test_let_to_code() {
        let node = RuchyNode::Let {
            name: "x".to_string(),
            ty: None,
            value: Box::new(RuchyNode::IntLit(0)),
            mutable: false,
        };
        assert_eq!(node.to_code(0), "let x = 0;");
    }

    #[test]
    fn test_let_mut_to_code() {
        let node = RuchyNode::Let {
            name: "x".to_string(),
            ty: Some(RuchyType::I32),
            value: Box::new(RuchyNode::IntLit(0)),
            mutable: true,
        };
        assert_eq!(node.to_code(0), "let mut x: i32 = 0;");
    }

    #[test]
    fn test_fn_def_to_code() {
        let node = RuchyNode::FnDef {
            name: "main".to_string(),
            params: vec![],
            return_type: None,
            body: vec![RuchyNode::Return(None)],
        };
        let code = node.to_code(0);
        assert!(code.contains("fn main()"));
        assert!(code.contains("return;"));
    }

    #[test]
    fn test_pipeline_to_code() {
        let node = RuchyNode::Pipeline {
            left: Box::new(RuchyNode::IntLit(1)),
            right: Box::new(RuchyNode::Ident("f".to_string())),
        };
        assert_eq!(node.to_code(0), "1 |> f");
    }

    #[test]
    fn test_range_to_code() {
        let node = RuchyNode::Range {
            start: Box::new(RuchyNode::IntLit(0)),
            end: Box::new(RuchyNode::IntLit(10)),
            inclusive: false,
        };
        assert_eq!(node.to_code(0), "0..10");

        let inclusive = RuchyNode::Range {
            start: Box::new(RuchyNode::IntLit(0)),
            end: Box::new(RuchyNode::IntLit(10)),
            inclusive: true,
        };
        assert_eq!(inclusive.to_code(0), "0..=10");
    }

    #[test]
    fn test_for_to_code() {
        let node = RuchyNode::For {
            var: "i".to_string(),
            iter: Box::new(RuchyNode::Range {
                start: Box::new(RuchyNode::IntLit(0)),
                end: Box::new(RuchyNode::IntLit(10)),
                inclusive: false,
            }),
            body: vec![RuchyNode::Break],
        };
        let code = node.to_code(0);
        assert!(code.contains("for i in 0..10"));
        assert!(code.contains("break;"));
    }

    #[test]
    fn test_enumerator_creates_programs() {
        let enumerator = RuchyEnumerator::new(2);
        let programs = enumerator.enumerate_programs();
        assert!(!programs.is_empty(), "Should generate programs");
    }

    #[test]
    fn test_programs_are_ruchy() {
        let enumerator = RuchyEnumerator::new(2);
        let programs = enumerator.enumerate_programs();
        for prog in &programs {
            assert_eq!(prog.language, Language::Ruchy);
        }
    }

    #[test]
    fn test_depth_calculation() {
        let node = RuchyNode::BinOp {
            left: Box::new(RuchyNode::IntLit(1)),
            op: RuchyBinaryOp::Add,
            right: Box::new(RuchyNode::BinOp {
                left: Box::new(RuchyNode::IntLit(2)),
                op: RuchyBinaryOp::Mul,
                right: Box::new(RuchyNode::IntLit(3)),
            }),
        };
        assert_eq!(node.depth(), 3);
    }

    #[test]
    fn test_type_to_str() {
        assert_eq!(RuchyType::I32.to_str(), "i32");
        assert_eq!(
            RuchyType::Option(Box::new(RuchyType::I32)).to_str(),
            "Option<i32>"
        );
    }

    #[test]
    fn test_extract_features() {
        let enumerator = RuchyEnumerator::new(2);
        let node = RuchyNode::Let {
            name: "x".to_string(),
            ty: None,
            value: Box::new(RuchyNode::IntLit(0)),
            mutable: true,
        };
        let features = enumerator.extract_features(&node);
        assert!(features.contains(&"let".to_string()));
        assert!(features.contains(&"mut".to_string()));
    }
}
