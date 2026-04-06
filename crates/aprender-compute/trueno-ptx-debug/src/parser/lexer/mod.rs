//! PTX Lexer - Tokenization of PTX source code

use super::ast::SourceLocation;
use super::error::ParseError;

/// Token kinds for PTX lexing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    /// End of file
    Eof,
    /// `.version`, `.target`, `.address_size`, etc.
    Directive,
    /// .entry keyword
    Entry,
    /// .func keyword
    Func,
    /// .reg declaration
    Reg,
    /// .shared declaration
    Shared,
    /// .local declaration
    Local,
    /// .global declaration
    Global,
    /// .param declaration
    Param,
    /// Identifier (register name, label, etc.)
    Identifier,
    /// Integer literal
    Integer,
    /// Float literal
    Float,
    /// Instruction (ld, st, mov, add, etc.)
    Instruction,
    /// Label (name:)
    Label,
    /// Comment (// or /* */)
    Comment,
    /// Opening brace {
    LBrace,
    /// Closing brace }
    RBrace,
    /// Opening parenthesis (
    LParen,
    /// Closing parenthesis )
    RParen,
    /// Opening bracket [
    LBracket,
    /// Closing bracket ]
    RBracket,
    /// Comma ,
    Comma,
    /// Semicolon ;
    Semicolon,
    /// Colon :
    Colon,
    /// Unknown token
    Unknown,
}

/// A token in the PTX source
#[derive(Debug, Clone)]
pub struct Token {
    /// The kind of token
    pub kind: TokenKind,
    /// The text of the token
    pub text: String,
    /// Source location
    pub location: SourceLocation,
}

impl Default for Token {
    fn default() -> Self {
        Self { kind: TokenKind::Eof, text: String::new(), location: SourceLocation::default() }
    }
}

/// PTX Lexer
pub struct Lexer<'a> {
    source: &'a str,
    pos: usize,
    line: usize,
    column: usize,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer for the given source
    pub fn new(source: &'a str) -> Self {
        Self { source, pos: 0, line: 1, column: 1 }
    }

    fn peek(&self) -> Option<char> {
        self.source[self.pos..].chars().next()
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.source[self.pos..].chars().nth(offset)
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(c)
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.advance();
            } else if c == '/' {
                if !self.skip_comment() {
                    break;
                }
            } else {
                break;
            }
        }
    }

    /// Skip a comment starting at current position. Returns true if a comment was skipped.
    fn skip_comment(&mut self) -> bool {
        if self.peek_at(1) == Some('/') {
            self.skip_line_comment();
            true
        } else if self.peek_at(1) == Some('*') {
            self.skip_block_comment();
            true
        } else {
            false
        }
    }

    fn skip_line_comment(&mut self) {
        while let Some(c) = self.peek() {
            self.advance();
            if c == '\n' {
                break;
            }
        }
    }

    fn skip_block_comment(&mut self) {
        self.advance(); // skip /
        self.advance(); // skip *
        while let Some(c) = self.peek() {
            self.advance();
            if c == '*' && self.peek() == Some('/') {
                self.advance();
                break;
            }
        }
    }

    /// Get the next token from the source
    pub fn next_token(&mut self) -> Result<Token, ParseError> {
        self.skip_whitespace();

        let location = SourceLocation { line: self.line, column: self.column, file: None };

        let Some(c) = self.peek() else {
            return Ok(Token { kind: TokenKind::Eof, text: String::new(), location });
        };

        match c {
            '{' => {
                self.advance();
                Ok(Token { kind: TokenKind::LBrace, text: "{".into(), location })
            }
            '}' => {
                self.advance();
                Ok(Token { kind: TokenKind::RBrace, text: "}".into(), location })
            }
            '(' => {
                self.advance();
                Ok(Token { kind: TokenKind::LParen, text: "(".into(), location })
            }
            ')' => {
                self.advance();
                Ok(Token { kind: TokenKind::RParen, text: ")".into(), location })
            }
            '[' => {
                self.advance();
                Ok(Token { kind: TokenKind::LBracket, text: "[".into(), location })
            }
            ']' => {
                self.advance();
                Ok(Token { kind: TokenKind::RBracket, text: "]".into(), location })
            }
            ',' => {
                self.advance();
                Ok(Token { kind: TokenKind::Comma, text: ",".into(), location })
            }
            ';' => {
                self.advance();
                Ok(Token { kind: TokenKind::Semicolon, text: ";".into(), location })
            }
            '.' => self.read_directive(location),
            '%' => self.read_register(location),
            '@' => self.read_predicate(location),
            '0'..='9' | '-' => self.read_number(location),
            _ if c.is_alphabetic() || c == '_' => self.read_identifier_or_instruction(location),
            _ => {
                self.advance();
                Ok(Token { kind: TokenKind::Unknown, text: c.to_string(), location })
            }
        }
    }

    fn read_directive(&mut self, location: SourceLocation) -> Result<Token, ParseError> {
        let start = self.pos;
        self.advance(); // skip '.'

        // Read directive name
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let directive_name = &self.source[start..self.pos];

        // For version, target, address_size - read the value too
        let text = if directive_name.starts_with(".version")
            || directive_name.starts_with(".target")
            || directive_name.starts_with(".address_size")
        {
            self.skip_whitespace();
            let value_start = self.pos;
            while let Some(c) = self.peek() {
                if c == '\n' || c == ';' || c == '{' || c == '(' {
                    break;
                }
                self.advance();
            }
            format!("{} {}", directive_name, self.source[value_start..self.pos].trim())
        } else {
            directive_name.to_string()
        };

        let kind = self.classify_directive(&text);
        Ok(Token { kind, text, location })
    }

    fn classify_directive(&self, text: &str) -> TokenKind {
        const DIRECTIVE_MAP: &[(&str, TokenKind)] = &[
            (".entry", TokenKind::Entry),
            (".func", TokenKind::Func),
            (".reg", TokenKind::Reg),
            (".shared", TokenKind::Shared),
            (".local", TokenKind::Local),
            (".global", TokenKind::Global),
            (".param", TokenKind::Param),
        ];

        DIRECTIVE_MAP
            .iter()
            .find(|(prefix, _)| text.starts_with(prefix))
            .map_or(TokenKind::Directive, |(_, kind)| kind.clone())
    }

    fn read_register(&mut self, location: SourceLocation) -> Result<Token, ParseError> {
        let start = self.pos;
        self.advance(); // skip '%'

        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        Ok(Token {
            kind: TokenKind::Identifier,
            text: self.source[start..self.pos].to_string(),
            location,
        })
    }

    fn read_predicate(&mut self, location: SourceLocation) -> Result<Token, ParseError> {
        let start = self.pos;
        self.advance(); // skip '@'

        // May have '!' for negation
        if self.peek() == Some('!') {
            self.advance();
        }

        // Read predicate register name
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' || c == '%' {
                self.advance();
            } else {
                break;
            }
        }

        Ok(Token {
            kind: TokenKind::Identifier,
            text: self.source[start..self.pos].to_string(),
            location,
        })
    }

    fn read_number(&mut self, location: SourceLocation) -> Result<Token, ParseError> {
        let start = self.pos;

        if self.peek() == Some('-') {
            self.advance();
        }

        if let Some(tok) = self.try_read_hex(start, &location) {
            return Ok(tok);
        }

        self.advance_while(|c| c.is_ascii_digit());

        let is_float = self.read_fractional_part() | self.read_exponent_part();

        Ok(Token {
            kind: if is_float { TokenKind::Float } else { TokenKind::Integer },
            text: self.source[start..self.pos].to_string(),
            location,
        })
    }

    fn try_read_hex(&mut self, start: usize, location: &SourceLocation) -> Option<Token> {
        if self.peek() != Some('0') {
            return None;
        }
        self.advance();
        if !matches!(self.peek(), Some('x' | 'X')) {
            return None;
        }
        self.advance();
        self.advance_while(|c| c.is_ascii_hexdigit());
        Some(Token {
            kind: TokenKind::Integer,
            text: self.source[start..self.pos].to_string(),
            location: location.clone(),
        })
    }

    fn read_fractional_part(&mut self) -> bool {
        if self.peek() != Some('.') {
            return false;
        }
        self.advance();
        self.advance_while(|c| c.is_ascii_digit());
        true
    }

    fn read_exponent_part(&mut self) -> bool {
        if !matches!(self.peek(), Some('e' | 'E')) {
            return false;
        }
        self.advance();
        if matches!(self.peek(), Some('+' | '-')) {
            self.advance();
        }
        self.advance_while(|c| c.is_ascii_digit());
        true
    }

    fn advance_while(&mut self, predicate: impl Fn(char) -> bool) {
        while let Some(c) = self.peek() {
            if predicate(c) {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_identifier_or_instruction(
        &mut self,
        location: SourceLocation,
    ) -> Result<Token, ParseError> {
        let start = self.pos;

        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let text = &self.source[start..self.pos];

        // Check if it's a label (followed by :)
        if self.peek() == Some(':') {
            self.advance();
            return Ok(Token {
                kind: TokenKind::Label,
                text: self.source[start..self.pos].to_string(),
                location,
            });
        }

        // Check if it's an instruction
        if self.is_instruction(text) {
            // For instructions, read modifiers and operands
            let instr_end = self.pos;
            self.skip_whitespace();

            // Read the rest of the line for operands
            let operand_start = self.pos;
            while let Some(c) = self.peek() {
                if c == '\n' || c == ';' || c == '{' || c == '}' {
                    break;
                }
                self.advance();
            }

            let full_text = if operand_start < self.pos {
                format!(
                    "{} {}",
                    &self.source[start..instr_end],
                    self.source[operand_start..self.pos].trim()
                )
            } else {
                self.source[start..instr_end].to_string()
            };

            return Ok(Token { kind: TokenKind::Instruction, text: full_text, location });
        }

        Ok(Token { kind: TokenKind::Identifier, text: text.to_string(), location })
    }

    fn is_instruction(&self, text: &str) -> bool {
        matches!(
            text,
            "ld" | "st"
                | "mov"
                | "add"
                | "sub"
                | "mul"
                | "div"
                | "rem"
                | "mad"
                | "fma"
                | "neg"
                | "abs"
                | "min"
                | "max"
                | "and"
                | "or"
                | "xor"
                | "not"
                | "shl"
                | "shr"
                | "setp"
                | "selp"
                | "cvt"
                | "cvta"
                | "bra"
                | "call"
                | "ret"
                | "exit"
                | "bar"
                | "membar"
                | "atom"
                | "red"
                | "tex"
                | "tld4"
                | "suld"
                | "sust"
                | "shfl"
                | "vote"
                | "match"
                | "mma"
                | "wmma"
                | "ldmatrix"
                | "cp"
                | "prefetch"
                | "prefetchu"
        )
    }
}

#[cfg(test)]
mod tests;
