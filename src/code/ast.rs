//! Abstract Syntax Tree representation for code analysis
//!
//! Provides lightweight AST node types for code2vec path extraction.
//! This is not a full parser - it's designed to work with pre-parsed AST data.

use std::fmt;

/// Types of AST nodes for code analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AstNodeType {
    /// Function or method definition
    Function,
    /// Function/method parameter
    Parameter,
    /// Return statement or expression
    Return,
    /// Variable declaration
    Variable,
    /// Assignment expression
    Assignment,
    /// Binary operation (e.g., +, -, *, /)
    BinaryOp,
    /// Unary operation (e.g., !, -)
    UnaryOp,
    /// If/else conditional
    Conditional,
    /// Loop construct (for, while, loop)
    Loop,
    /// Function call expression
    Call,
    /// Literal value (number, string, bool)
    Literal,
    /// Array/vector access
    Index,
    /// Field access (e.g., obj.field)
    FieldAccess,
    /// Block of statements
    Block,
    /// Type annotation
    TypeAnnotation,
    /// Generic type parameter
    Generic,
    /// Match/switch expression
    Match,
    /// Match arm
    MatchArm,
    /// Struct definition
    Struct,
    /// Enum definition
    Enum,
    /// Trait/interface definition
    Trait,
    /// Implementation block
    Impl,
    /// Module declaration
    Module,
    /// Import/use statement
    Import,
}

impl fmt::Display for AstNodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Function => "Func",
            Self::Parameter => "Param",
            Self::Return => "Ret",
            Self::Variable => "Var",
            Self::Assignment => "Assign",
            Self::BinaryOp => "BinOp",
            Self::UnaryOp => "UnOp",
            Self::Conditional => "Cond",
            Self::Loop => "Loop",
            Self::Call => "Call",
            Self::Literal => "Lit",
            Self::Index => "Idx",
            Self::FieldAccess => "Field",
            Self::Block => "Block",
            Self::TypeAnnotation => "Type",
            Self::Generic => "Gen",
            Self::Match => "Match",
            Self::MatchArm => "Arm",
            Self::Struct => "Struct",
            Self::Enum => "Enum",
            Self::Trait => "Trait",
            Self::Impl => "Impl",
            Self::Module => "Mod",
            Self::Import => "Import",
        };
        write!(f, "{s}")
    }
}

/// Types of tokens (terminal nodes in the AST)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenType {
    /// Identifier (variable name, function name, etc.)
    Identifier,
    /// Numeric literal
    Number,
    /// String literal
    String,
    /// Boolean literal
    Boolean,
    /// Keyword (if, else, fn, let, etc.)
    Keyword,
    /// Operator (+, -, *, /, etc.)
    Operator,
    /// Punctuation (parentheses, braces, etc.)
    Punctuation,
    /// Type name
    TypeName,
    /// Comment
    Comment,
}

impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Identifier => "Id",
            Self::Number => "Num",
            Self::String => "Str",
            Self::Boolean => "Bool",
            Self::Keyword => "Kw",
            Self::Operator => "Op",
            Self::Punctuation => "Punct",
            Self::TypeName => "Type",
            Self::Comment => "Comment",
        };
        write!(f, "{s}")
    }
}

/// A token (terminal node) in the AST
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Token {
    /// Type of token
    token_type: TokenType,
    /// Token value/content
    value: String,
}

impl Token {
    /// Create a new token
    #[must_use]
    pub fn new(token_type: TokenType, value: impl Into<String>) -> Self {
        Self {
            token_type,
            value: value.into(),
        }
    }

    /// Get the token type
    #[must_use]
    pub fn token_type(&self) -> TokenType {
        self.token_type
    }

    /// Get the token value
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.token_type, self.value)
    }
}

/// A node in the Abstract Syntax Tree
#[derive(Debug, Clone)]
pub struct AstNode {
    /// Type of AST node
    node_type: AstNodeType,
    /// Node value (e.g., function name, variable name)
    value: String,
    /// Child nodes
    children: Vec<AstNode>,
    /// Optional token for terminal nodes
    token: Option<Token>,
}

impl AstNode {
    /// Create a new AST node
    #[must_use]
    pub fn new(node_type: AstNodeType, value: impl Into<String>) -> Self {
        Self {
            node_type,
            value: value.into(),
            children: Vec::new(),
            token: None,
        }
    }

    /// Create a terminal node with a token
    #[must_use]
    pub fn terminal(token: Token) -> Self {
        Self {
            node_type: AstNodeType::Literal,
            value: token.value().to_string(),
            children: Vec::new(),
            token: Some(token),
        }
    }

    /// Get the node type
    #[must_use]
    pub fn node_type(&self) -> AstNodeType {
        self.node_type
    }

    /// Get the node value
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Get the children of this node
    #[must_use]
    pub fn children(&self) -> &[AstNode] {
        &self.children
    }

    /// Get mutable access to children
    pub fn children_mut(&mut self) -> &mut Vec<AstNode> {
        &mut self.children
    }

    /// Add a child node
    pub fn add_child(&mut self, child: AstNode) {
        self.children.push(child);
    }

    /// Check if this is a terminal node (leaf)
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.children.is_empty()
    }

    /// Get the token if this is a terminal node
    #[must_use]
    pub fn token(&self) -> Option<&Token> {
        self.token.as_ref()
    }

    /// Count all nodes in the subtree (including self)
    #[must_use]
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(AstNode::node_count).sum::<usize>()
    }

    /// Get the depth of the tree
    #[must_use]
    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(AstNode::depth).max().unwrap_or(0)
        }
    }

    /// Collect all terminal nodes (leaves)
    #[must_use]
    pub fn terminals(&self) -> Vec<&AstNode> {
        if self.is_terminal() {
            vec![self]
        } else {
            self.children.iter().flat_map(AstNode::terminals).collect()
        }
    }
}

impl fmt::Display for AstNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.node_type, self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_node_type_display() {
        assert_eq!(AstNodeType::Function.to_string(), "Func");
        assert_eq!(AstNodeType::Parameter.to_string(), "Param");
        assert_eq!(AstNodeType::Return.to_string(), "Ret");
    }

    #[test]
    fn test_token_type_display() {
        assert_eq!(TokenType::Identifier.to_string(), "Id");
        assert_eq!(TokenType::Number.to_string(), "Num");
        assert_eq!(TokenType::String.to_string(), "Str");
    }

    #[test]
    fn test_token_creation() {
        let token = Token::new(TokenType::Identifier, "my_var");
        assert_eq!(token.token_type(), TokenType::Identifier);
        assert_eq!(token.value(), "my_var");
        assert_eq!(token.to_string(), "Id:my_var");
    }

    #[test]
    fn test_ast_node_creation() {
        let node = AstNode::new(AstNodeType::Function, "calculate");
        assert_eq!(node.node_type(), AstNodeType::Function);
        assert_eq!(node.value(), "calculate");
        assert!(node.children().is_empty());
        assert!(node.is_terminal());
    }

    #[test]
    fn test_ast_node_with_children() {
        let mut func = AstNode::new(AstNodeType::Function, "add");
        func.add_child(AstNode::new(AstNodeType::Parameter, "x"));
        func.add_child(AstNode::new(AstNodeType::Parameter, "y"));
        func.add_child(AstNode::new(AstNodeType::Return, "result"));

        assert_eq!(func.children().len(), 3);
        assert!(!func.is_terminal());
        assert_eq!(func.node_count(), 4);
        assert_eq!(func.depth(), 2);
    }

    #[test]
    fn test_terminal_node() {
        let token = Token::new(TokenType::Number, "42");
        let node = AstNode::terminal(token);

        assert!(node.is_terminal());
        assert!(node.token().is_some());
        assert_eq!(node.token().map(Token::value), Some("42"));
    }

    #[test]
    fn test_collect_terminals() {
        let mut func = AstNode::new(AstNodeType::Function, "test");
        func.add_child(AstNode::new(AstNodeType::Parameter, "a"));
        func.add_child(AstNode::new(AstNodeType::Parameter, "b"));

        let terminals = func.terminals();
        assert_eq!(terminals.len(), 2);
    }

    #[test]
    fn test_deep_tree() {
        let mut root = AstNode::new(AstNodeType::Function, "deep");
        let mut level1 = AstNode::new(AstNodeType::Block, "body");
        let mut level2 = AstNode::new(AstNodeType::Conditional, "if");
        level2.add_child(AstNode::new(AstNodeType::Return, "early"));
        level1.add_child(level2);
        root.add_child(level1);

        assert_eq!(root.depth(), 4);
        assert_eq!(root.node_count(), 4);
    }
}
