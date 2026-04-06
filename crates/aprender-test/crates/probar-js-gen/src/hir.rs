//! High-level Intermediate Representation for JavaScript.
//!
//! # Design Principles
//!
//! 1. **Type Safety**: All JS constructs have typed Rust equivalents
//! 2. **Validation**: Invalid JS is unrepresentable in the type system
//! 3. **Determinism**: Same HIR always produces same JS output
//!
//! # References
//! - Maffeis et al. (2008) "An Operational Semantics for JavaScript"
//! - Guha et al. (2010) "The Essence of JavaScript"
//! - ECMA-262 (ES2022) Specification

use serde::{Deserialize, Serialize};

/// A complete JavaScript module.
///
/// Modules are the top-level unit of code generation.
/// They contain statements and track generation metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsModule {
    /// Module-level statements
    pub statements: Vec<Stmt>,
    /// Generation metadata (tool version, hash, etc.)
    pub metadata: Option<GenerationMetadata>,
}

impl Default for JsModule {
    fn default() -> Self {
        Self::new()
    }
}

impl JsModule {
    /// Create a new empty module.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            statements: Vec::new(),
            metadata: None,
        }
    }
}

/// Metadata about code generation for immutability enforcement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationMetadata {
    /// Tool that generated this code
    pub tool: String,
    /// Tool version
    pub version: String,
    /// Blake3 hash of input specification
    pub input_hash: String,
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// Command to regenerate
    pub regenerate_cmd: String,
}

/// JavaScript statement.
///
/// Each variant maps to a specific JS statement type.
/// Invalid combinations are prevented at compile time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Stmt {
    /// Variable declaration: `let name = expr;`
    Let {
        /// Variable name
        name: Identifier,
        /// Initial value
        value: Expr,
    },
    /// Constant declaration: `const name = expr;`
    Const {
        /// Constant name
        name: Identifier,
        /// Value
        value: Expr,
    },
    /// Assignment: `name = expr;`
    Assign {
        /// Target name
        name: Identifier,
        /// New value
        value: Expr,
    },
    /// Member assignment: `obj.member = value;`
    MemberAssign {
        /// Object expression
        object: Expr,
        /// Member name
        member: Identifier,
        /// New value
        value: Expr,
    },
    /// Compound assignment: `target += value;`
    AddAssign {
        /// Target expression
        target: Expr,
        /// Value to add
        value: Expr,
    },
    /// Post-increment: `expr++;`
    PostIncrement(Expr),
    /// Expression statement: `expr;`
    Expr(Expr),
    /// Return statement: `return expr;` or `return;`
    Return(Option<Expr>),
    /// If statement: `if (cond) { then } else { else }`
    If {
        /// Condition
        condition: Expr,
        /// Then branch
        then_branch: Vec<Stmt>,
        /// Optional else branch
        else_branch: Option<Vec<Stmt>>,
    },
    /// For loop: `for (let i = start; i < end; i++) { body }`
    For {
        /// Loop variable
        var: Identifier,
        /// Start value
        start: Expr,
        /// End value (exclusive)
        end: Expr,
        /// Loop body
        body: Vec<Stmt>,
    },
    /// While loop: `while (cond) { body }`
    While {
        /// Condition
        condition: Expr,
        /// Loop body
        body: Vec<Stmt>,
    },
    /// Try-catch: `try { body } catch (e) { handler }`
    TryCatch {
        /// Try body
        body: Vec<Stmt>,
        /// Catch variable name
        catch_var: Identifier,
        /// Catch handler
        handler: Vec<Stmt>,
    },
    /// Block: `{ stmts }`
    Block(Vec<Stmt>),
    /// Comment: `// text`
    Comment(String),
    /// Class definition
    Class(JsClass),
    /// Switch statement
    Switch(JsSwitch),
    /// `self.onmessage = async function(e) { body }`
    OnMessage(Vec<Stmt>),
    /// `registerProcessor(name, class)`
    RegisterProcessor {
        /// Processor name
        name: String,
        /// Class name
        class: Identifier,
    },
}

/// JavaScript expression.
///
/// All expression types that can appear in JS code.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// Null literal
    Null,
    /// Boolean literal
    Bool(bool),
    /// Number literal
    Num(f64),
    /// String literal (will be properly escaped)
    Str(String),
    /// Identifier reference
    Ident(Identifier),
    /// `this` keyword
    This,
    /// Member access: `obj.prop`
    Member {
        /// Object
        object: Box<Expr>,
        /// Property name
        property: Identifier,
    },
    /// Computed member: `obj[expr]`
    Index {
        /// Object
        object: Box<Expr>,
        /// Index expression
        index: Box<Expr>,
    },
    /// Function call: `func(args)`
    Call {
        /// Function expression
        callee: Box<Expr>,
        /// Arguments
        args: Vec<Expr>,
    },
    /// `new Constructor(args)`
    New {
        /// Constructor
        constructor: Box<Expr>,
        /// Arguments
        args: Vec<Expr>,
    },
    /// `await expr`
    Await(Box<Expr>),
    /// `import(path)` - dynamic import
    Import(Box<Expr>),
    /// Binary operation: `left op right`
    Binary {
        /// Left operand
        left: Box<Expr>,
        /// Operator
        op: BinOp,
        /// Right operand
        right: Box<Expr>,
    },
    /// Unary operation: `op expr`
    Unary {
        /// Operator
        op: UnaryOp,
        /// Operand
        operand: Box<Expr>,
    },
    /// Ternary: `cond ? then : else`
    Ternary {
        /// Condition
        condition: Box<Expr>,
        /// Then expression
        then_expr: Box<Expr>,
        /// Else expression
        else_expr: Box<Expr>,
    },
    /// Object literal: `{ key: value, ... }`
    Object(Vec<(String, Expr)>),
    /// Array literal: `[expr, ...]`
    Array(Vec<Expr>),
    /// Arrow function: `(params) => expr`
    Arrow {
        /// Parameters
        params: Vec<Identifier>,
        /// Body expression
        body: Box<Expr>,
    },
    /// Arrow function with block: `(params) => { stmts }`
    ArrowBlock {
        /// Parameters
        params: Vec<Identifier>,
        /// Body statements
        body: Vec<Stmt>,
    },
    /// Assignment expression: `left = right`
    Assign {
        /// Target
        target: Box<Expr>,
        /// Value
        value: Box<Expr>,
    },
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinOp {
    /// Addition: `+`
    Add,
    /// Subtraction: `-`
    Sub,
    /// Multiplication: `*`
    Mul,
    /// Division: `/`
    Div,
    /// Modulo: `%`
    Mod,
    /// Equality: `==`
    Eq,
    /// Strict equality: `===`
    EqStrict,
    /// Inequality: `!=`
    Ne,
    /// Strict inequality: `!==`
    NeStrict,
    /// Less than: `<`
    Lt,
    /// Less than or equal: `<=`
    Le,
    /// Greater than: `>`
    Gt,
    /// Greater than or equal: `>=`
    Ge,
    /// Logical and: `&&`
    And,
    /// Logical or: `||`
    Or,
    /// Bitwise and: `&`
    BitAnd,
    /// Bitwise or: `|`
    BitOr,
}

impl BinOp {
    /// Get the JavaScript operator string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Mod => "%",
            Self::Eq => "==",
            Self::EqStrict => "===",
            Self::Ne => "!=",
            Self::NeStrict => "!==",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
            Self::And => "&&",
            Self::Or => "||",
            Self::BitAnd => "&",
            Self::BitOr => "|",
        }
    }
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    /// Logical not: `!`
    Not,
    /// Negation: `-`
    Neg,
    /// Type of: `typeof`
    TypeOf,
}

impl UnaryOp {
    /// Get the JavaScript operator string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Not => "!",
            Self::Neg => "-",
            Self::TypeOf => "typeof ",
        }
    }
}

/// A validated JavaScript identifier.
///
/// Identifiers are validated at construction time to ensure they:
/// - Are not reserved words
/// - Contain only valid characters
/// - Don't start with a digit
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Identifier(String);

impl Identifier {
    /// JavaScript reserved words that cannot be used as identifiers.
    pub const RESERVED_WORDS: &'static [&'static str] = &[
        "break",
        "case",
        "catch",
        "continue",
        "debugger",
        "default",
        "delete",
        "do",
        "else",
        "finally",
        "for",
        "function",
        "if",
        "in",
        "instanceof",
        "new",
        "return",
        "switch",
        "this",
        "throw",
        "try",
        "typeof",
        "var",
        "void",
        "while",
        "with",
        "class",
        "const",
        "enum",
        "export",
        "extends",
        "import",
        "super",
        "implements",
        "interface",
        "let",
        "package",
        "private",
        "protected",
        "public",
        "static",
        "yield",
        "await",
        "null",
        "true",
        "false",
    ];

    /// Create a new identifier, validating it.
    ///
    /// # Errors
    ///
    /// Returns an error if the identifier is:
    /// - Empty
    /// - A reserved word
    /// - Contains invalid characters
    /// - Starts with a digit
    pub fn new(name: impl Into<String>) -> crate::Result<Self> {
        let name = name.into();

        if name.is_empty() {
            return Err(crate::JsGenError::InvalidIdentifier {
                name,
                reason: "identifier cannot be empty".to_string(),
            });
        }

        // Check first character
        let first = name.chars().next().unwrap_or(' ');
        if first.is_ascii_digit() {
            return Err(crate::JsGenError::InvalidIdentifier {
                name,
                reason: "identifier cannot start with a digit".to_string(),
            });
        }

        // Check all characters
        for c in name.chars() {
            if !c.is_ascii_alphanumeric() && c != '_' && c != '$' {
                return Err(crate::JsGenError::InvalidIdentifier {
                    name,
                    reason: format!("invalid character '{c}'"),
                });
            }
        }

        // Check reserved words
        if Self::RESERVED_WORDS.contains(&name.as_str()) {
            return Err(crate::JsGenError::InvalidIdentifier {
                name,
                reason: "reserved word".to_string(),
            });
        }

        Ok(Self(name))
    }

    /// Create an identifier without validation (for trusted input).
    ///
    /// # Safety
    ///
    /// This is not unsafe in the memory sense, but it bypasses validation.
    /// Only use for identifiers known to be valid at compile time.
    #[must_use]
    pub fn new_unchecked(name: &'static str) -> Self {
        Self(name.to_string())
    }

    /// Get the identifier string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// JavaScript class definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsClass {
    /// Class name
    pub name: Identifier,
    /// Parent class (extends)
    pub extends: Option<Identifier>,
    /// Constructor body (super() is added automatically if extends is set)
    pub constructor: Option<Vec<Stmt>>,
    /// Methods
    pub methods: Vec<JsMethod>,
}

/// JavaScript class method.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsMethod {
    /// Method name
    pub name: Identifier,
    /// Parameters
    pub params: Vec<Identifier>,
    /// Method body
    pub body: Vec<Stmt>,
}

/// JavaScript switch statement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsSwitch {
    /// Expression to switch on
    pub expr: Expr,
    /// Cases
    pub cases: Vec<(Expr, Vec<Stmt>)>,
    /// Default case
    pub default: Option<Vec<Stmt>>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn identifier_valid() {
        assert!(Identifier::new("foo").is_ok());
        assert!(Identifier::new("_bar").is_ok());
        assert!(Identifier::new("$baz").is_ok());
        assert!(Identifier::new("foo123").is_ok());
        assert!(Identifier::new("camelCase").is_ok());
    }

    #[test]
    fn identifier_invalid_reserved() {
        let err = Identifier::new("class").unwrap_err();
        assert!(err.to_string().contains("reserved word"));
    }

    #[test]
    fn identifier_invalid_starts_digit() {
        let err = Identifier::new("123foo").unwrap_err();
        assert!(err.to_string().contains("cannot start with a digit"));
    }

    #[test]
    fn identifier_invalid_empty() {
        let err = Identifier::new("").unwrap_err();
        assert!(err.to_string().contains("cannot be empty"));
    }

    #[test]
    fn identifier_invalid_chars() {
        let err = Identifier::new("foo-bar").unwrap_err();
        assert!(err.to_string().contains("invalid character"));
    }

    #[test]
    fn binop_as_str() {
        assert_eq!(BinOp::Add.as_str(), "+");
        assert_eq!(BinOp::EqStrict.as_str(), "===");
        assert_eq!(BinOp::And.as_str(), "&&");
    }

    #[test]
    fn unaryop_as_str() {
        assert_eq!(UnaryOp::Not.as_str(), "!");
        assert_eq!(UnaryOp::TypeOf.as_str(), "typeof ");
    }
}
