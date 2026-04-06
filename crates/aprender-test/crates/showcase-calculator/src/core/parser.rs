//! Expression parser with 100% test coverage
//!
//! Probar: Error prevention - Type-safe tokens prevent invalid syntax

use crate::core::{CalcError, CalcResult, Operation};

/// Token types from lexical analysis
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// Numeric literal
    Number(f64),
    /// Binary operator
    Operator(Operation),
    /// Left parenthesis
    LeftParen,
    /// Right parenthesis
    RightParen,
}

impl Token {
    /// Returns true if this token is an operator
    #[must_use]
    pub const fn is_operator(&self) -> bool {
        matches!(self, Self::Operator(_))
    }

    /// Returns true if this token is a number
    #[must_use]
    pub const fn is_number(&self) -> bool {
        matches!(self, Self::Number(_))
    }

    /// Returns true if this token is a left parenthesis
    #[must_use]
    pub const fn is_left_paren(&self) -> bool {
        matches!(self, Self::LeftParen)
    }

    /// Returns true if this token is a right parenthesis
    #[must_use]
    pub const fn is_right_paren(&self) -> bool {
        matches!(self, Self::RightParen)
    }
}

/// Abstract Syntax Tree node
#[derive(Debug, Clone, PartialEq)]
pub enum AstNode {
    /// Numeric literal
    Number(f64),
    /// Binary operation
    BinaryOp {
        /// Left operand
        left: Box<AstNode>,
        /// Operator
        op: Operation,
        /// Right operand
        right: Box<AstNode>,
    },
    /// Unary negation
    Negate(Box<AstNode>),
}

impl AstNode {
    /// Creates a new number node
    #[must_use]
    pub fn number(value: f64) -> Self {
        Self::Number(value)
    }

    /// Creates a new binary operation node
    #[must_use]
    pub fn binary(left: AstNode, op: Operation, right: AstNode) -> Self {
        Self::BinaryOp {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    /// Creates a new negation node
    #[must_use]
    pub fn negate(inner: AstNode) -> Self {
        Self::Negate(Box::new(inner))
    }
}

/// Tokenizer for converting expression strings to tokens
#[derive(Debug)]
pub struct Tokenizer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Tokenizer<'a> {
    /// Creates a new tokenizer for the given input
    #[must_use]
    pub fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    /// Returns the remaining input
    #[must_use]
    pub fn remaining(&self) -> &str {
        &self.input[self.pos..]
    }

    /// Tokenizes the entire input
    pub fn tokenize(&mut self) -> CalcResult<Vec<Token>> {
        let mut tokens = Vec::new();
        while let Some(token) = self.next_token()? {
            tokens.push(token);
        }
        Ok(tokens)
    }

    /// Returns the next token, or None if at end of input
    pub fn next_token(&mut self) -> CalcResult<Option<Token>> {
        self.skip_whitespace();

        if self.pos >= self.input.len() {
            return Ok(None);
        }

        let ch = self
            .current_char()
            .ok_or_else(|| CalcError::ParseError("Unexpected end of input".into()))?;

        let token = match ch {
            '0'..='9' | '.' => self.read_number()?,
            '+' => {
                self.advance();
                Token::Operator(Operation::Add)
            }
            '-' => {
                self.advance();
                Token::Operator(Operation::Subtract)
            }
            '*' => {
                self.advance();
                Token::Operator(Operation::Multiply)
            }
            '/' => {
                self.advance();
                Token::Operator(Operation::Divide)
            }
            '%' => {
                self.advance();
                Token::Operator(Operation::Modulo)
            }
            '^' => {
                self.advance();
                Token::Operator(Operation::Power)
            }
            '(' => {
                self.advance();
                Token::LeftParen
            }
            ')' => {
                self.advance();
                Token::RightParen
            }
            _ => {
                return Err(CalcError::ParseError(format!(
                    "Unexpected character: '{ch}'"
                )));
            }
        };

        Ok(Some(token))
    }

    fn current_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn advance(&mut self) {
        if let Some(ch) = self.current_char() {
            self.pos += ch.len_utf8();
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_number(&mut self) -> CalcResult<Token> {
        let start = self.pos;
        let mut has_dot = false;

        while let Some(ch) = self.current_char() {
            if ch.is_ascii_digit() {
                self.advance();
            } else if ch == '.' && !has_dot {
                has_dot = true;
                self.advance();
            } else {
                break;
            }
        }

        let num_str = &self.input[start..self.pos];
        let value: f64 = num_str
            .parse()
            .map_err(|_| CalcError::ParseError(format!("Invalid number: '{num_str}'")))?;

        Ok(Token::Number(value))
    }
}

/// Recursive descent parser for expressions
///
/// Grammar:
/// ```text
/// expression ::= term (('+' | '-') term)*
/// term       ::= factor (('*' | '/' | '%') factor)*
/// factor     ::= base ('^' factor)?    // Right associative
/// base       ::= '-' base | primary
/// primary    ::= NUMBER | '(' expression ')'
/// ```
#[derive(Debug)]
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    /// Creates a new parser from tokens
    #[must_use]
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Parses a string expression into an AST
    pub fn parse_str(input: &str) -> CalcResult<AstNode> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(CalcError::EmptyExpression);
        }

        let mut tokenizer = Tokenizer::new(trimmed);
        let tokens = tokenizer.tokenize()?;

        if tokens.is_empty() {
            return Err(CalcError::EmptyExpression);
        }

        let mut parser = Self::new(tokens);
        let ast = parser.parse_expression()?;

        // Ensure all tokens consumed
        if parser.pos < parser.tokens.len() {
            return Err(CalcError::ParseError(format!(
                "Unexpected token at position {}",
                parser.pos
            )));
        }

        Ok(ast)
    }

    /// Parses tokens into an AST
    pub fn parse(&mut self) -> CalcResult<AstNode> {
        if self.tokens.is_empty() {
            return Err(CalcError::EmptyExpression);
        }
        self.parse_expression()
    }

    fn current(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&Token> {
        let token = self.tokens.get(self.pos);
        if token.is_some() {
            self.pos += 1;
        }
        token
    }

    fn parse_expression(&mut self) -> CalcResult<AstNode> {
        let mut left = self.parse_term()?;

        while let Some(token) = self.current() {
            let op = match token {
                Token::Operator(Operation::Add) => Operation::Add,
                Token::Operator(Operation::Subtract) => Operation::Subtract,
                _ => break,
            };
            self.advance();
            let right = self.parse_term()?;
            left = AstNode::binary(left, op, right);
        }

        Ok(left)
    }

    fn parse_term(&mut self) -> CalcResult<AstNode> {
        let mut left = self.parse_factor()?;

        while let Some(token) = self.current() {
            let op = match token {
                Token::Operator(Operation::Multiply) => Operation::Multiply,
                Token::Operator(Operation::Divide) => Operation::Divide,
                Token::Operator(Operation::Modulo) => Operation::Modulo,
                _ => break,
            };
            self.advance();
            let right = self.parse_factor()?;
            left = AstNode::binary(left, op, right);
        }

        Ok(left)
    }

    fn parse_factor(&mut self) -> CalcResult<AstNode> {
        let base = self.parse_base()?;

        // Power is right-associative
        if matches!(self.current(), Some(Token::Operator(Operation::Power))) {
            self.advance();
            let exponent = self.parse_factor()?; // Recurse for right associativity
            return Ok(AstNode::binary(base, Operation::Power, exponent));
        }

        Ok(base)
    }

    fn parse_base(&mut self) -> CalcResult<AstNode> {
        // Handle unary minus
        if matches!(self.current(), Some(Token::Operator(Operation::Subtract))) {
            self.advance();
            let inner = self.parse_base()?;
            return Ok(AstNode::negate(inner));
        }

        self.parse_primary()
    }

    fn parse_primary(&mut self) -> CalcResult<AstNode> {
        let token = self
            .advance()
            .ok_or_else(|| CalcError::ParseError("Unexpected end of expression".into()))?;

        match token {
            Token::Number(n) => Ok(AstNode::number(*n)),
            Token::LeftParen => {
                let expr = self.parse_expression()?;
                match self.advance() {
                    Some(Token::RightParen) => Ok(expr),
                    Some(t) => Err(CalcError::ParseError(format!(
                        "Expected ')' but found {t:?}"
                    ))),
                    None => Err(CalcError::ParseError("Unclosed parenthesis".into())),
                }
            }
            _ => Err(CalcError::ParseError(format!(
                "Unexpected token: {token:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Token tests =====

    #[test]
    fn test_token_is_operator() {
        assert!(Token::Operator(Operation::Add).is_operator());
        assert!(!Token::Number(5.0).is_operator());
        assert!(!Token::LeftParen.is_operator());
    }

    #[test]
    fn test_token_is_number() {
        assert!(Token::Number(5.0).is_number());
        assert!(!Token::Operator(Operation::Add).is_number());
    }

    #[test]
    fn test_token_is_left_paren() {
        assert!(Token::LeftParen.is_left_paren());
        assert!(!Token::RightParen.is_left_paren());
    }

    #[test]
    fn test_token_is_right_paren() {
        assert!(Token::RightParen.is_right_paren());
        assert!(!Token::LeftParen.is_right_paren());
    }

    // ===== AstNode tests =====

    #[test]
    fn test_ast_node_number() {
        let node = AstNode::number(42.0);
        assert_eq!(node, AstNode::Number(42.0));
    }

    #[test]
    fn test_ast_node_binary() {
        let node = AstNode::binary(AstNode::number(1.0), Operation::Add, AstNode::number(2.0));
        match node {
            AstNode::BinaryOp { left, op, right } => {
                assert_eq!(*left, AstNode::Number(1.0));
                assert_eq!(op, Operation::Add);
                assert_eq!(*right, AstNode::Number(2.0));
            }
            _ => panic!("Expected BinaryOp"),
        }
    }

    #[test]
    fn test_ast_node_negate() {
        let node = AstNode::negate(AstNode::number(5.0));
        match node {
            AstNode::Negate(inner) => {
                assert_eq!(*inner, AstNode::Number(5.0));
            }
            _ => panic!("Expected Negate"),
        }
    }

    // ===== Tokenizer tests =====

    #[test]
    fn test_tokenize_single_number() {
        let mut t = Tokenizer::new("42");
        let tokens = t.tokenize().unwrap();
        assert_eq!(tokens, vec![Token::Number(42.0)]);
    }

    #[test]
    fn test_tokenize_decimal_number() {
        let mut t = Tokenizer::new("3.14");
        let tokens = t.tokenize().unwrap();
        assert_eq!(tokens, vec![Token::Number(3.14)]);
    }

    #[test]
    fn test_tokenize_operators() {
        let mut t = Tokenizer::new("+ - * / % ^");
        let tokens = t.tokenize().unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Operator(Operation::Add),
                Token::Operator(Operation::Subtract),
                Token::Operator(Operation::Multiply),
                Token::Operator(Operation::Divide),
                Token::Operator(Operation::Modulo),
                Token::Operator(Operation::Power),
            ]
        );
    }

    #[test]
    fn test_tokenize_parentheses() {
        let mut t = Tokenizer::new("()");
        let tokens = t.tokenize().unwrap();
        assert_eq!(tokens, vec![Token::LeftParen, Token::RightParen]);
    }

    #[test]
    fn test_tokenize_expression() {
        let mut t = Tokenizer::new("2 + 3 * 4");
        let tokens = t.tokenize().unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Number(2.0),
                Token::Operator(Operation::Add),
                Token::Number(3.0),
                Token::Operator(Operation::Multiply),
                Token::Number(4.0),
            ]
        );
    }

    #[test]
    fn test_tokenize_with_parens() {
        let mut t = Tokenizer::new("(2 + 3) * 4");
        let tokens = t.tokenize().unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::LeftParen,
                Token::Number(2.0),
                Token::Operator(Operation::Add),
                Token::Number(3.0),
                Token::RightParen,
                Token::Operator(Operation::Multiply),
                Token::Number(4.0),
            ]
        );
    }

    #[test]
    fn test_tokenize_no_spaces() {
        let mut t = Tokenizer::new("1+2*3");
        let tokens = t.tokenize().unwrap();
        assert_eq!(tokens.len(), 5);
    }

    #[test]
    fn test_tokenize_invalid_char() {
        let mut t = Tokenizer::new("2 @ 3");
        let result = t.tokenize();
        assert!(matches!(result, Err(CalcError::ParseError(_))));
    }

    #[test]
    fn test_tokenize_empty() {
        let mut t = Tokenizer::new("");
        let tokens = t.tokenize().unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_whitespace_only() {
        let mut t = Tokenizer::new("   ");
        let tokens = t.tokenize().unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenizer_remaining() {
        let mut t = Tokenizer::new("1 + 2");
        t.next_token().unwrap();
        assert_eq!(t.remaining(), " + 2");
    }

    #[test]
    fn test_tokenize_leading_decimal() {
        let mut t = Tokenizer::new(".5");
        let tokens = t.tokenize().unwrap();
        assert_eq!(tokens, vec![Token::Number(0.5)]);
    }

    // ===== Parser tests =====

    #[test]
    fn test_parse_single_number() {
        let ast = Parser::parse_str("42").unwrap();
        assert_eq!(ast, AstNode::Number(42.0));
    }

    #[test]
    fn test_parse_decimal() {
        let ast = Parser::parse_str("3.14").unwrap();
        assert_eq!(ast, AstNode::Number(3.14));
    }

    #[test]
    fn test_parse_simple_addition() {
        let ast = Parser::parse_str("2 + 3").unwrap();
        assert_eq!(
            ast,
            AstNode::binary(AstNode::number(2.0), Operation::Add, AstNode::number(3.0))
        );
    }

    #[test]
    fn test_parse_simple_subtraction() {
        let ast = Parser::parse_str("5 - 2").unwrap();
        assert_eq!(
            ast,
            AstNode::binary(
                AstNode::number(5.0),
                Operation::Subtract,
                AstNode::number(2.0)
            )
        );
    }

    #[test]
    fn test_parse_simple_multiplication() {
        let ast = Parser::parse_str("3 * 4").unwrap();
        assert_eq!(
            ast,
            AstNode::binary(
                AstNode::number(3.0),
                Operation::Multiply,
                AstNode::number(4.0)
            )
        );
    }

    #[test]
    fn test_parse_simple_division() {
        let ast = Parser::parse_str("8 / 2").unwrap();
        assert_eq!(
            ast,
            AstNode::binary(
                AstNode::number(8.0),
                Operation::Divide,
                AstNode::number(2.0)
            )
        );
    }

    #[test]
    fn test_parse_simple_modulo() {
        let ast = Parser::parse_str("7 % 3").unwrap();
        assert_eq!(
            ast,
            AstNode::binary(
                AstNode::number(7.0),
                Operation::Modulo,
                AstNode::number(3.0)
            )
        );
    }

    #[test]
    fn test_parse_simple_power() {
        let ast = Parser::parse_str("2 ^ 3").unwrap();
        assert_eq!(
            ast,
            AstNode::binary(AstNode::number(2.0), Operation::Power, AstNode::number(3.0))
        );
    }

    #[test]
    fn test_parse_precedence_mul_over_add() {
        // 2 + 3 * 4 = 2 + (3 * 4) = 14
        let ast = Parser::parse_str("2 + 3 * 4").unwrap();
        // Should be: Add(2, Mul(3, 4))
        match ast {
            AstNode::BinaryOp {
                op: Operation::Add, ..
            } => {}
            _ => panic!("Expected Add at top level"),
        }
    }

    #[test]
    fn test_parse_precedence_power_highest() {
        // 2 * 3 ^ 2 = 2 * (3 ^ 2) = 18
        let ast = Parser::parse_str("2 * 3 ^ 2").unwrap();
        match ast {
            AstNode::BinaryOp {
                op: Operation::Multiply,
                ..
            } => {}
            _ => panic!("Expected Multiply at top level"),
        }
    }

    #[test]
    fn test_parse_power_right_associative() {
        // 2 ^ 3 ^ 2 = 2 ^ (3 ^ 2) = 512
        let ast = Parser::parse_str("2 ^ 3 ^ 2").unwrap();
        match ast {
            AstNode::BinaryOp {
                left,
                op: Operation::Power,
                right,
            } => {
                assert_eq!(*left, AstNode::Number(2.0));
                match *right {
                    AstNode::BinaryOp {
                        op: Operation::Power,
                        ..
                    } => {}
                    _ => panic!("Expected Power on right"),
                }
            }
            _ => panic!("Expected Power at top level"),
        }
    }

    #[test]
    fn test_parse_parentheses() {
        // (2 + 3) * 4 = 20
        let ast = Parser::parse_str("(2 + 3) * 4").unwrap();
        match ast {
            AstNode::BinaryOp {
                op: Operation::Multiply,
                left,
                ..
            } => match *left {
                AstNode::BinaryOp {
                    op: Operation::Add, ..
                } => {}
                _ => panic!("Expected Add inside parens"),
            },
            _ => panic!("Expected Multiply at top level"),
        }
    }

    #[test]
    fn test_parse_nested_parentheses() {
        let ast = Parser::parse_str("((2 + 3))").unwrap();
        match ast {
            AstNode::BinaryOp {
                op: Operation::Add, ..
            } => {}
            _ => panic!("Expected Add"),
        }
    }

    #[test]
    fn test_parse_unary_minus() {
        let ast = Parser::parse_str("-5").unwrap();
        match ast {
            AstNode::Negate(inner) => {
                assert_eq!(*inner, AstNode::Number(5.0));
            }
            _ => panic!("Expected Negate"),
        }
    }

    #[test]
    fn test_parse_unary_minus_in_expression() {
        let ast = Parser::parse_str("3 + -2").unwrap();
        match ast {
            AstNode::BinaryOp {
                op: Operation::Add,
                right,
                ..
            } => match *right {
                AstNode::Negate(_) => {}
                _ => panic!("Expected Negate on right"),
            },
            _ => panic!("Expected Add"),
        }
    }

    #[test]
    fn test_parse_double_negative() {
        let ast = Parser::parse_str("--5").unwrap();
        match ast {
            AstNode::Negate(inner) => match *inner {
                AstNode::Negate(_) => {}
                _ => panic!("Expected nested Negate"),
            },
            _ => panic!("Expected Negate"),
        }
    }

    #[test]
    fn test_parse_empty_expression() {
        let result = Parser::parse_str("");
        assert!(matches!(result, Err(CalcError::EmptyExpression)));
    }

    #[test]
    fn test_parse_whitespace_only() {
        let result = Parser::parse_str("   ");
        assert!(matches!(result, Err(CalcError::EmptyExpression)));
    }

    #[test]
    fn test_parse_unclosed_paren() {
        let result = Parser::parse_str("(2 + 3");
        assert!(matches!(result, Err(CalcError::ParseError(_))));
    }

    #[test]
    fn test_parse_extra_close_paren() {
        let result = Parser::parse_str("2 + 3)");
        assert!(matches!(result, Err(CalcError::ParseError(_))));
    }

    #[test]
    fn test_parse_missing_operand() {
        let result = Parser::parse_str("2 +");
        assert!(matches!(result, Err(CalcError::ParseError(_))));
    }

    #[test]
    fn test_parse_consecutive_operators() {
        // "2 + * 3" - should fail (+ followed by *)
        let result = Parser::parse_str("2 + * 3");
        assert!(matches!(result, Err(CalcError::ParseError(_))));
    }

    #[test]
    fn test_parse_complex_expression() {
        // 42 * (3 + 7)
        let ast = Parser::parse_str("42 * (3 + 7)").unwrap();
        match ast {
            AstNode::BinaryOp {
                op: Operation::Multiply,
                left,
                right,
            } => {
                assert_eq!(*left, AstNode::Number(42.0));
                match *right {
                    AstNode::BinaryOp {
                        op: Operation::Add, ..
                    } => {}
                    _ => panic!("Expected Add in parens"),
                }
            }
            _ => panic!("Expected Multiply"),
        }
    }

    #[test]
    fn test_parser_new() {
        let tokens = vec![Token::Number(5.0)];
        let parser = Parser::new(tokens);
        assert_eq!(parser.pos, 0);
    }

    #[test]
    fn test_parser_parse_method() {
        let tokens = vec![Token::Number(42.0)];
        let mut parser = Parser::new(tokens);
        let ast = parser.parse().unwrap();
        assert_eq!(ast, AstNode::Number(42.0));
    }

    #[test]
    fn test_parser_parse_empty_tokens() {
        let mut parser = Parser::new(vec![]);
        let result = parser.parse();
        assert!(matches!(result, Err(CalcError::EmptyExpression)));
    }
}
