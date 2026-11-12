//! Parser for the ggplot2 text syntax: `ggplot(df, aes(...)) + geom_point() + ...`.
//!
//! PMAT-6918 (contract `present-plot-spec-v1`). Pure Rust, no rendering.
//!
//! Only the calls in [`CALLS`] exist in 0.72.6. Everything else is a *named*
//! [`SpecError`] carrying the 1-based line and column of the offending call,
//! argument or value; nothing is ignored silently and no input panics.
//!
//! Setting the same thing twice follows ggplot2: the later call wins
//! (`labs(x = "a") + xlab("b")` gives `x = "b"`).

use std::fmt;

/// Deepest `name(...)` nesting accepted inside a value (stack-overflow guard).
const MAX_DEPTH: usize = 64;

/// Why `geom_smooth` is restricted; appended to every smooth error.
const SMOOTH_NOTE: &str = "geom_smooth supports only method = \"lm\" with se = FALSE in 0.72.6; \
                           loess and the confidence band are not supported";

// ───────────────────────────── positions and errors ─────────────────────────────

/// A 1-based source position (columns count characters).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pos {
    pub line: usize,
    pub col: usize,
}

impl fmt::Display for Pos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}, column {}", self.line, self.col)
    }
}

/// A rejected plot spec. Every variant names its cause and carries a [`Pos`].
#[derive(Debug, Clone, PartialEq)]
pub enum SpecError {
    /// The text is not valid ggplot2 call syntax.
    Syntax { msg: String, at: Pos },
    /// A call that is not in the 0.72.6 table.
    UnknownCall { name: String, at: Pos },
    /// A named argument the call does not take.
    UnknownArg { call: String, arg: String, at: Pos },
    /// A positional argument the call does not take.
    UnexpectedPositional { call: String, at: Pos },
    /// The same argument given twice.
    DuplicateArg { call: String, arg: String, at: Pos },
    /// A required argument is absent.
    MissingArg { call: String, arg: String, at: Pos },
    /// An argument value of the wrong kind or outside the supported set.
    BadValue {
        call: String,
        arg: String,
        expected: String,
        at: Pos,
    },
    /// A known call used in a way 0.72.6 does not support (loess, bar stat).
    Unsupported {
        call: String,
        detail: String,
        at: Pos,
    },
}

impl SpecError {
    /// Where in the source the problem is.
    pub fn at(&self) -> Pos {
        match self {
            Self::Syntax { at, .. }
            | Self::UnknownCall { at, .. }
            | Self::UnknownArg { at, .. }
            | Self::UnexpectedPositional { at, .. }
            | Self::DuplicateArg { at, .. }
            | Self::MissingArg { at, .. }
            | Self::BadValue { at, .. }
            | Self::Unsupported { at, .. } => *at,
        }
    }
}

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax { msg, at } => write!(f, "{at}: syntax error: {msg}"),
            Self::UnknownCall { name, at } => write!(
                f,
                "{at}: unsupported call `{name}` (supported in 0.72.6: {})",
                supported_calls().join(", ")
            ),
            Self::UnknownArg { call, arg, at } => {
                write!(f, "{at}: `{call}` does not take an argument named `{arg}`")
            }
            Self::UnexpectedPositional { call, at } => {
                write!(f, "{at}: `{call}` does not take this positional argument")
            }
            Self::DuplicateArg { call, arg, at } => {
                write!(f, "{at}: `{call}` got argument `{arg}` more than once")
            }
            Self::MissingArg { call, arg, at } => {
                write!(f, "{at}: `{call}` requires argument `{arg}`")
            }
            Self::BadValue {
                call,
                arg,
                expected,
                at,
            } => write!(f, "{at}: `{call}` argument `{arg}` must be {expected}"),
            Self::Unsupported { call, detail, at } => {
                write!(f, "{at}: `{call}`: {detail}")
            }
        }
    }
}

impl std::error::Error for SpecError {}

fn syntax(at: Pos, msg: impl Into<String>) -> SpecError {
    SpecError::Syntax {
        msg: msg.into(),
        at,
    }
}

// ───────────────────────────── the typed spec ─────────────────────────────

/// Column mappings (`aes(x = ..., y = ..., colour = ..., fill = ...)`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Aes {
    pub x: Option<String>,
    pub y: Option<String>,
    pub colour: Option<String>,
    pub fill: Option<String>,
}

impl Aes {
    fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// `position = ...` of `geom_col` / `geom_bar`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    Dodge,
    Stack,
}

impl Position {
    fn name(self) -> &'static str {
        match self {
            Self::Dodge => "dodge",
            Self::Stack => "stack",
        }
    }

    fn from_name(s: &str) -> Option<Self> {
        [Self::Dodge, Self::Stack]
            .into_iter()
            .find(|p| p.name() == s)
    }
}

/// One `geom_*` layer.
#[derive(Debug, Clone, PartialEq)]
pub enum Layer {
    Point {
        aes: Aes,
        size: Option<f64>,
    },
    Line {
        aes: Aes,
        size: Option<f64>,
    },
    Step {
        aes: Aes,
        size: Option<f64>,
    },
    Col {
        aes: Aes,
        position: Option<Position>,
    },
    /// Always `stat = "identity"`; any other stat is rejected at parse time.
    Bar {
        aes: Aes,
        position: Option<Position>,
    },
    /// Always `method = "lm", se = FALSE`; anything else is rejected at parse time.
    Smooth {
        aes: Aes,
    },
    Abline {
        slope: f64,
        intercept: f64,
    },
    Hline {
        yintercept: f64,
    },
    Vline {
        xintercept: f64,
    },
}

/// `facet_wrap(~ var, ncol, nrow)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Facet {
    pub var: String,
    pub ncol: Option<u32>,
    pub nrow: Option<u32>,
}

/// An x/y pair of `(low, high)` limits.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Limits {
    pub x: Option<(f64, f64)>,
    pub y: Option<(f64, f64)>,
}

/// `labs()`, `xlab()`, `ylab()` and `ggtitle()` merged.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Labs {
    pub title: Option<String>,
    pub subtitle: Option<String>,
    pub caption: Option<String>,
    pub x: Option<String>,
    pub y: Option<String>,
    pub colour: Option<String>,
    pub fill: Option<String>,
}

/// `theme_minimal(base_size = ...)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Theme {
    pub base_size: Option<f64>,
}

/// A whole parsed plot.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PlotSpec {
    /// First positional argument of `ggplot`, kept verbatim.
    pub data: Option<String>,
    pub aes: Aes,
    pub layers: Vec<Layer>,
    pub facet: Option<Facet>,
    pub flip: bool,
    /// `coord_cartesian(xlim, ylim)`.
    pub coord: Limits,
    /// `scale_x_continuous` / `scale_y_continuous` `limits`.
    pub scale: Limits,
    pub colour_values: Option<Vec<String>>,
    pub fill_values: Option<Vec<String>>,
    pub labs: Labs,
    pub theme: Option<Theme>,
}

// ───────────────────────────── lexer ─────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Num(f64),
    Str(String),
    LParen,
    RParen,
    Comma,
    Plus,
    Eq,
    Tilde,
    Eof,
}

impl Tok {
    fn describe(&self) -> String {
        match self {
            Self::Ident(s) => format!("`{s}`"),
            Self::Num(n) => format!("number {n}"),
            Self::Str(_) => "a string".to_string(),
            Self::LParen => "`(`".to_string(),
            Self::RParen => "`)`".to_string(),
            Self::Comma => "`,`".to_string(),
            Self::Plus => "`+`".to_string(),
            Self::Eq => "`=`".to_string(),
            Self::Tilde => "`~`".to_string(),
            Self::Eof => "end of input".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
struct Token {
    tok: Tok,
    at: Pos,
}

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_' || c == '.'
}

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '.'
}

struct Lexer {
    cs: Vec<char>,
    i: usize,
    line: usize,
    col: usize,
}

impl Lexer {
    fn new(src: &str) -> Self {
        Self {
            cs: src.chars().collect(),
            i: 0,
            line: 1,
            col: 1,
        }
    }

    fn pos(&self) -> Pos {
        Pos {
            line: self.line,
            col: self.col,
        }
    }

    fn peek_at(&self, n: usize) -> Option<char> {
        self.cs.get(self.i + n).copied()
    }

    fn digit_at(&self, n: usize) -> bool {
        self.peek_at(n).is_some_and(|c| c.is_ascii_digit())
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek_at(0)?;
        self.i += 1;
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn skip_trivia(&mut self) {
        while let Some(c) = self.peek_at(0) {
            if c == '#' {
                while self.peek_at(0).is_some_and(|c| c != '\n') {
                    self.bump();
                }
            } else if c.is_whitespace() {
                self.bump();
            } else {
                break;
            }
        }
    }

    fn next_token(&mut self) -> Result<Token, SpecError> {
        self.skip_trivia();
        let at = self.pos();
        let tok = match self.peek_at(0) {
            None => Tok::Eof,
            Some(c) => self.token_from(c, at)?,
        };
        Ok(Token { tok, at })
    }

    fn token_from(&mut self, c: char, at: Pos) -> Result<Tok, SpecError> {
        let simple = match c {
            '(' => Some(Tok::LParen),
            ')' => Some(Tok::RParen),
            ',' => Some(Tok::Comma),
            '+' => Some(Tok::Plus),
            '=' => Some(Tok::Eq),
            '~' => Some(Tok::Tilde),
            _ => None,
        };
        if let Some(t) = simple {
            self.bump();
            return Ok(t);
        }
        if c == '"' || c == '\'' {
            return self.string(c, at);
        }
        if self.starts_number() {
            return self.number(at);
        }
        if is_ident_start(c) {
            return Ok(self.ident());
        }
        Err(syntax(at, format!("unexpected character {c:?}")))
    }

    fn starts_number(&self) -> bool {
        match self.peek_at(0).unwrap_or('\0') {
            '-' => self.digit_at(1) || (self.peek_at(1) == Some('.') && self.digit_at(2)),
            '.' => self.digit_at(1),
            c => c.is_ascii_digit(),
        }
    }

    fn digits(&mut self, out: &mut String) {
        while let Some(c) = self.peek_at(0).filter(char::is_ascii_digit) {
            out.push(c);
            self.bump();
        }
    }

    fn number(&mut self, at: Pos) -> Result<Tok, SpecError> {
        let mut s = String::new();
        if self.peek_at(0) == Some('-') {
            s.push('-');
            self.bump();
        }
        self.digits(&mut s);
        if self.peek_at(0) == Some('.') && self.digit_at(1) {
            s.push('.');
            self.bump();
            self.digits(&mut s);
        }
        if matches!(self.peek_at(0), Some('e' | 'E')) {
            let sign = self.peek_at(1).filter(|c| *c == '+' || *c == '-');
            if self.digit_at(1 + usize::from(sign.is_some())) {
                s.push('e');
                self.bump();
                if let Some(sg) = sign {
                    s.push(sg);
                    self.bump();
                }
                self.digits(&mut s);
            }
        }
        match s.parse::<f64>() {
            Ok(v) if v.is_finite() => Ok(Tok::Num(v)),
            _ => Err(syntax(at, format!("number `{s}` is not a finite number"))),
        }
    }

    fn string(&mut self, quote: char, at: Pos) -> Result<Tok, SpecError> {
        self.bump();
        let mut s = String::new();
        loop {
            let here = self.pos();
            match self.bump() {
                None => return Err(syntax(at, "unterminated string")),
                Some(c) if c == quote => return Ok(Tok::Str(s)),
                Some('\\') => match self.bump() {
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some('r') => s.push('\r'),
                    Some(c @ ('\\' | '"' | '\'')) => s.push(c),
                    Some(c) => return Err(syntax(here, format!("unknown escape `\\{c}`"))),
                    None => return Err(syntax(at, "unterminated string")),
                },
                Some(c) => s.push(c),
            }
        }
    }

    fn ident(&mut self) -> Tok {
        let mut s = String::new();
        while let Some(c) = self.peek_at(0).filter(|c| is_ident_char(*c)) {
            s.push(c);
            self.bump();
        }
        Tok::Ident(s)
    }
}

fn lex(src: &str) -> Result<Vec<Token>, SpecError> {
    let mut lx = Lexer::new(src);
    let mut out = Vec::new();
    loop {
        let t = lx.next_token()?;
        let done = t.tok == Tok::Eof;
        out.push(t);
        if done {
            return Ok(out);
        }
    }
}

// ───────────────────────────── parse tree ─────────────────────────────

#[derive(Debug, Clone)]
struct Call {
    name: String,
    at: Pos,
    args: Vec<Arg>,
}

#[derive(Debug, Clone)]
struct Arg {
    /// `color` is folded into `colour` here, once.
    name: Option<String>,
    at: Pos,
    val: Val,
}

#[derive(Debug, Clone)]
struct Val {
    kind: Vk,
    at: Pos,
}

#[derive(Debug, Clone)]
enum Vk {
    Num(f64),
    Str(String),
    Bool(bool),
    Ident(String),
    Vec(Vec<Val>),
    Formula(String),
    Call(Call),
}

fn is_aes_call(v: &Val) -> bool {
    matches!(&v.kind, Vk::Call(c) if c.name == "aes")
}

struct Parser {
    /// Non-empty, always ending in `Eof` (guaranteed by [`lex`]).
    toks: Vec<Token>,
    i: usize,
}

impl Parser {
    fn idx(&self, n: usize) -> usize {
        (self.i + n).min(self.toks.len() - 1)
    }

    fn peek(&self) -> &Tok {
        &self.toks[self.idx(0)].tok
    }

    fn peek2(&self) -> &Tok {
        &self.toks[self.idx(1)].tok
    }

    fn at(&self) -> Pos {
        self.toks[self.idx(0)].at
    }

    fn bump(&mut self) -> Tok {
        let t = self.peek().clone();
        self.i = self.idx(1);
        t
    }

    fn unexpected(&self, what: &str) -> SpecError {
        syntax(
            self.at(),
            format!("expected {what}, found {}", self.peek().describe()),
        )
    }

    fn calls(&mut self) -> Result<Vec<Call>, SpecError> {
        let mut out = vec![self.call(0)?];
        while *self.peek() == Tok::Plus {
            self.bump();
            out.push(self.call(0)?);
        }
        if *self.peek() != Tok::Eof {
            return Err(self.unexpected("`+` or end of input"));
        }
        Ok(out)
    }

    fn call(&mut self, depth: usize) -> Result<Call, SpecError> {
        let at = self.at();
        let Tok::Ident(name) = self.peek().clone() else {
            return Err(self.unexpected("a call such as `ggplot(...)`"));
        };
        self.bump();
        if *self.peek() != Tok::LParen {
            return Err(self.unexpected("`(` after the call name"));
        }
        self.bump();
        let args = self.args(depth)?;
        Ok(Call { name, at, args })
    }

    fn args(&mut self, depth: usize) -> Result<Vec<Arg>, SpecError> {
        let mut out = Vec::new();
        loop {
            if *self.peek() == Tok::RParen {
                self.bump();
                return Ok(out);
            }
            out.push(self.arg(depth)?);
            match self.peek() {
                Tok::Comma => {
                    self.bump();
                }
                Tok::RParen => {}
                _ => return Err(self.unexpected("`,` or `)`")),
            }
        }
    }

    fn arg(&mut self, depth: usize) -> Result<Arg, SpecError> {
        let at = self.at();
        let name = match (self.peek(), self.peek2()) {
            (Tok::Ident(n), Tok::Eq) => Some(n.clone()),
            _ => None,
        };
        if name.is_some() {
            self.bump();
            self.bump();
        }
        let name = name.map(|n| {
            if n == "color" {
                "colour".to_string()
            } else {
                n
            }
        });
        let val = self.value(depth)?;
        Ok(Arg { name, at, val })
    }

    fn value(&mut self, depth: usize) -> Result<Val, SpecError> {
        let at = self.at();
        if depth >= MAX_DEPTH {
            return Err(syntax(at, "calls are nested too deeply"));
        }
        let kind = match self.peek().clone() {
            Tok::Num(n) => {
                self.bump();
                Vk::Num(n)
            }
            Tok::Str(s) => {
                self.bump();
                Vk::Str(s)
            }
            Tok::Tilde => {
                self.bump();
                match self.peek().clone() {
                    Tok::Ident(n) | Tok::Str(n) => {
                        self.bump();
                        Vk::Formula(n)
                    }
                    _ => return Err(self.unexpected("a variable name after `~`")),
                }
            }
            Tok::Ident(_) if *self.peek2() == Tok::LParen => vector_or_call(self.call(depth + 1)?)?,
            Tok::Ident(n) => {
                self.bump();
                match n.as_str() {
                    "TRUE" => Vk::Bool(true),
                    "FALSE" => Vk::Bool(false),
                    _ => Vk::Ident(n),
                }
            }
            _ => return Err(self.unexpected("a value")),
        };
        Ok(Val { kind, at })
    }
}

/// `c(...)` is a vector value; every other call stays a call.
fn vector_or_call(c: Call) -> Result<Vk, SpecError> {
    if c.name != "c" {
        return Ok(Vk::Call(c));
    }
    let mut items = Vec::new();
    for a in c.args {
        if a.name.is_some() {
            return Err(syntax(
                a.at,
                "named elements inside c(...) are not supported",
            ));
        }
        items.push(a.val);
    }
    Ok(Vk::Vec(items))
}

// ───────────────────────────── argument binding ─────────────────────────────

/// A call's arguments checked against its allowed names and positional slots.
struct Bound<'a> {
    call: &'a Call,
    named: Vec<(String, &'a Arg)>,
}

/// `allowed` are the argument names; `slots` name the free positional arguments in order.
/// A positional `aes(...)` always lands in `mapping` when `mapping` is allowed.
fn bind<'a>(call: &'a Call, allowed: &[&str], slots: &[&str]) -> Result<Bound<'a>, SpecError> {
    let mut named: Vec<(String, &Arg)> = Vec::new();
    let mut free = slots.iter();
    for a in &call.args {
        let key = match &a.name {
            Some(n) if allowed.contains(&n.as_str()) => n.clone(),
            Some(n) => {
                return Err(SpecError::UnknownArg {
                    call: call.name.clone(),
                    arg: n.clone(),
                    at: a.at,
                })
            }
            None if allowed.contains(&"mapping") && is_aes_call(&a.val) => "mapping".to_string(),
            None => match free.next() {
                Some(s) => String::from(*s),
                None => {
                    return Err(SpecError::UnexpectedPositional {
                        call: call.name.clone(),
                        at: a.at,
                    })
                }
            },
        };
        if named.iter().any(|(k, _)| *k == key) {
            return Err(SpecError::DuplicateArg {
                call: call.name.clone(),
                arg: key,
                at: a.at,
            });
        }
        named.push((key, a));
    }
    Ok(Bound { call, named })
}

impl<'a> Bound<'a> {
    fn get(&self, key: &str) -> Option<&'a Arg> {
        self.named.iter().find(|(k, _)| k == key).map(|(_, a)| *a)
    }

    fn get_with<T>(
        &self,
        key: &str,
        expected: &str,
        f: impl Fn(&Vk) -> Option<T>,
    ) -> Result<Option<T>, SpecError> {
        let Some(a) = self.get(key) else {
            return Ok(None);
        };
        f(&a.val.kind).map(Some).ok_or_else(|| SpecError::BadValue {
            call: self.call.name.clone(),
            arg: key.to_string(),
            expected: expected.to_string(),
            at: a.val.at,
        })
    }

    fn need<T>(&self, key: &str, v: Option<T>) -> Result<T, SpecError> {
        v.ok_or_else(|| SpecError::MissingArg {
            call: self.call.name.clone(),
            arg: key.to_string(),
            at: self.call.at,
        })
    }

    fn num(&self, key: &str) -> Result<Option<f64>, SpecError> {
        self.get_with(key, "a number", |v| match v {
            Vk::Num(n) => Some(*n),
            _ => None,
        })
    }

    fn text(&self, key: &str) -> Result<Option<String>, SpecError> {
        self.get_with(key, "a string", |v| match v {
            Vk::Str(s) => Some(s.clone()),
            _ => None,
        })
    }

    /// A column or data name: a bare identifier or a string.
    fn name(&self, key: &str) -> Result<Option<String>, SpecError> {
        self.get_with(key, "a name or a string", |v| match v {
            Vk::Ident(s) | Vk::Str(s) => Some(s.clone()),
            _ => None,
        })
    }

    fn flag(&self, key: &str) -> Result<Option<bool>, SpecError> {
        self.get_with(key, "TRUE or FALSE", |v| match v {
            Vk::Bool(b) => Some(*b),
            _ => None,
        })
    }

    fn pair(&self, key: &str) -> Result<Option<(f64, f64)>, SpecError> {
        self.get_with(key, "a pair of numbers like c(0, 100)", |v| match v {
            Vk::Vec(items) => match items.as_slice() {
                [Val {
                    kind: Vk::Num(a), ..
                }, Val {
                    kind: Vk::Num(b), ..
                }] => Some((*a, *b)),
                _ => None,
            },
            _ => None,
        })
    }

    fn strs(&self, key: &str) -> Result<Option<Vec<String>>, SpecError> {
        self.get_with(
            key,
            "a vector of strings like c(\"a\", \"b\")",
            |v| match v {
                Vk::Vec(items) => items
                    .iter()
                    .map(|i| match &i.kind {
                        Vk::Str(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect(),
                _ => None,
            },
        )
    }

    fn count(&self, key: &str) -> Result<Option<u32>, SpecError> {
        self.get_with(key, "a whole number of at least 1", |v| match v {
            Vk::Num(n) if n.fract() == 0.0 => u32::try_from(*n as i64).ok().filter(|u| *u >= 1),
            _ => None,
        })
    }

    fn position(&self) -> Result<Option<Position>, SpecError> {
        self.get_with("position", "\"dodge\" or \"stack\"", |v| match v {
            Vk::Str(s) => Position::from_name(s),
            _ => None,
        })
    }

    fn formula(&self, key: &str) -> Result<Option<String>, SpecError> {
        self.get_with(key, "a one-sided formula like ~ var", |v| match v {
            Vk::Formula(s) => Some(s.clone()),
            _ => None,
        })
    }

    fn aes(&self) -> Result<Aes, SpecError> {
        let Some(a) = self.get("mapping") else {
            return Ok(Aes::default());
        };
        match &a.val.kind {
            Vk::Call(c) if c.name == "aes" => aes_of(c),
            _ => Err(SpecError::BadValue {
                call: self.call.name.clone(),
                arg: "mapping".to_string(),
                expected: "aes(...)".to_string(),
                at: a.val.at,
            }),
        }
    }
}

fn aes_of(call: &Call) -> Result<Aes, SpecError> {
    let b = bind(call, &["x", "y", "colour", "fill"], &["x", "y"])?;
    Ok(Aes {
        x: b.name("x")?,
        y: b.name("y")?,
        colour: b.name("colour")?,
        fill: b.name("fill")?,
    })
}

// ───────────────────────────── the supported-call table ─────────────────────────────

type Handler = fn(&mut PlotSpec, &Call) -> Result<(), SpecError>;

/// THE table of calls accepted in 0.72.6. Anything not listed is `UnknownCall`.
const CALLS: &[(&str, Handler)] = &[
    ("ggplot", h_ggplot),
    ("geom_point", h_point),
    ("geom_line", h_line),
    ("geom_step", h_step),
    ("geom_col", h_col),
    ("geom_bar", h_bar),
    ("geom_smooth", h_smooth),
    ("geom_abline", h_abline),
    ("geom_hline", h_hline),
    ("geom_vline", h_vline),
    ("facet_wrap", h_facet_wrap),
    ("coord_cartesian", h_coord_cartesian),
    ("coord_flip", h_coord_flip),
    ("labs", h_labs),
    ("xlab", h_xlab),
    ("ylab", h_ylab),
    ("ggtitle", h_ggtitle),
    ("theme_minimal", h_theme_minimal),
    ("scale_colour_manual", h_colour_manual),
    ("scale_color_manual", h_colour_manual),
    ("scale_fill_manual", h_fill_manual),
    ("scale_x_continuous", h_scale_x),
    ("scale_y_continuous", h_scale_y),
];

/// Names of every call [`parse`] accepts, in table order.
pub fn supported_calls() -> Vec<&'static str> {
    CALLS.iter().map(|(n, _)| *n).collect()
}

type Done = Result<(), SpecError>;

fn h_ggplot(s: &mut PlotSpec, c: &Call) -> Done {
    let b = bind(c, &["data", "mapping"], &["data"])?;
    s.data = b.name("data")?;
    s.aes = b.aes()?;
    Ok(())
}

fn geom_sized(s: &mut PlotSpec, c: &Call, make: fn(Aes, Option<f64>) -> Layer) -> Done {
    let b = bind(c, &["mapping", "size"], &[])?;
    s.layers.push(make(b.aes()?, b.num("size")?));
    Ok(())
}

fn h_point(s: &mut PlotSpec, c: &Call) -> Done {
    geom_sized(s, c, |aes, size| Layer::Point { aes, size })
}

fn h_line(s: &mut PlotSpec, c: &Call) -> Done {
    geom_sized(s, c, |aes, size| Layer::Line { aes, size })
}

fn h_step(s: &mut PlotSpec, c: &Call) -> Done {
    geom_sized(s, c, |aes, size| Layer::Step { aes, size })
}

fn h_col(s: &mut PlotSpec, c: &Call) -> Done {
    let b = bind(c, &["mapping", "position"], &[])?;
    s.layers.push(Layer::Col {
        aes: b.aes()?,
        position: b.position()?,
    });
    Ok(())
}

fn h_bar(s: &mut PlotSpec, c: &Call) -> Done {
    let b = bind(c, &["mapping", "position", "stat"], &[])?;
    if b.text("stat")?.as_deref() != Some("identity") {
        return Err(SpecError::Unsupported {
            call: c.name.clone(),
            detail: "only stat = \"identity\" is supported (the default stat = \"count\" is not); \
                     use geom_col or pass stat = \"identity\""
                .to_string(),
            at: c.at,
        });
    }
    s.layers.push(Layer::Bar {
        aes: b.aes()?,
        position: b.position()?,
    });
    Ok(())
}

fn h_smooth(s: &mut PlotSpec, c: &Call) -> Done {
    let b = bind(c, &["mapping", "method", "se"], &[])?;
    let reason = match (b.text("method")?.as_deref(), b.flag("se")?) {
        (Some("lm"), Some(false)) => {
            s.layers.push(Layer::Smooth { aes: b.aes()? });
            return Ok(());
        }
        (Some("lm"), _) => "se must be FALSE".to_string(),
        (Some(m), _) => format!("method = \"{m}\" is not supported"),
        (None, _) => "method is required (the default smoother is loess)".to_string(),
    };
    Err(SpecError::Unsupported {
        call: c.name.clone(),
        detail: format!("{reason}; {SMOOTH_NOTE}"),
        at: c.at,
    })
}

fn h_abline(s: &mut PlotSpec, c: &Call) -> Done {
    let b = bind(c, &["slope", "intercept"], &["slope", "intercept"])?;
    let slope = b.need("slope", b.num("slope")?)?;
    let intercept = b.need("intercept", b.num("intercept")?)?;
    s.layers.push(Layer::Abline { slope, intercept });
    Ok(())
}

fn h_hline(s: &mut PlotSpec, c: &Call) -> Done {
    let b = bind(c, &["yintercept"], &["yintercept"])?;
    let yintercept = b.need("yintercept", b.num("yintercept")?)?;
    s.layers.push(Layer::Hline { yintercept });
    Ok(())
}

fn h_vline(s: &mut PlotSpec, c: &Call) -> Done {
    let b = bind(c, &["xintercept"], &["xintercept"])?;
    let xintercept = b.need("xintercept", b.num("xintercept")?)?;
    s.layers.push(Layer::Vline { xintercept });
    Ok(())
}

fn h_facet_wrap(s: &mut PlotSpec, c: &Call) -> Done {
    let b = bind(c, &["facets", "ncol", "nrow"], &["facets"])?;
    let var = b.need("facets", b.formula("facets")?)?;
    s.facet = Some(Facet {
        var,
        ncol: b.count("ncol")?,
        nrow: b.count("nrow")?,
    });
    Ok(())
}

fn h_coord_cartesian(s: &mut PlotSpec, c: &Call) -> Done {
    let b = bind(c, &["xlim", "ylim"], &[])?;
    s.coord.x = b.pair("xlim")?.or(s.coord.x);
    s.coord.y = b.pair("ylim")?.or(s.coord.y);
    Ok(())
}

fn h_coord_flip(s: &mut PlotSpec, c: &Call) -> Done {
    bind(c, &[], &[])?;
    s.flip = true;
    Ok(())
}

fn h_labs(s: &mut PlotSpec, c: &Call) -> Done {
    let b = bind(
        c,
        &["title", "subtitle", "caption", "x", "y", "colour", "fill"],
        &[],
    )?;
    let labs = &mut s.labs;
    let slots = [
        (&mut labs.title, "title"),
        (&mut labs.subtitle, "subtitle"),
        (&mut labs.caption, "caption"),
        (&mut labs.x, "x"),
        (&mut labs.y, "y"),
        (&mut labs.colour, "colour"),
        (&mut labs.fill, "fill"),
    ];
    for (slot, key) in slots {
        if let Some(text) = b.text(key)? {
            *slot = Some(text);
        }
    }
    Ok(())
}

fn h_xlab(s: &mut PlotSpec, c: &Call) -> Done {
    let b = bind(c, &["label"], &["label"])?;
    s.labs.x = Some(b.need("label", b.text("label")?)?);
    Ok(())
}

fn h_ylab(s: &mut PlotSpec, c: &Call) -> Done {
    let b = bind(c, &["label"], &["label"])?;
    s.labs.y = Some(b.need("label", b.text("label")?)?);
    Ok(())
}

fn h_ggtitle(s: &mut PlotSpec, c: &Call) -> Done {
    let b = bind(c, &["label", "subtitle"], &["label", "subtitle"])?;
    s.labs.title = Some(b.need("label", b.text("label")?)?);
    if let Some(sub) = b.text("subtitle")? {
        s.labs.subtitle = Some(sub);
    }
    Ok(())
}

fn h_theme_minimal(s: &mut PlotSpec, c: &Call) -> Done {
    let b = bind(c, &["base_size"], &[])?;
    s.theme = Some(Theme {
        base_size: b.num("base_size")?,
    });
    Ok(())
}

fn manual_values(c: &Call) -> Result<Vec<String>, SpecError> {
    let b = bind(c, &["values"], &[])?;
    b.need("values", b.strs("values")?)
}

fn h_colour_manual(s: &mut PlotSpec, c: &Call) -> Done {
    s.colour_values = Some(manual_values(c)?);
    Ok(())
}

fn h_fill_manual(s: &mut PlotSpec, c: &Call) -> Done {
    s.fill_values = Some(manual_values(c)?);
    Ok(())
}

fn h_scale_x(s: &mut PlotSpec, c: &Call) -> Done {
    let b = bind(c, &["limits"], &[])?;
    s.scale.x = b.pair("limits")?.or(s.scale.x);
    Ok(())
}

fn h_scale_y(s: &mut PlotSpec, c: &Call) -> Done {
    let b = bind(c, &["limits"], &[])?;
    s.scale.y = b.pair("limits")?.or(s.scale.y);
    Ok(())
}

// ───────────────────────────── entry point ─────────────────────────────

/// Parse ggplot2 text syntax into a [`PlotSpec`].
///
/// Total: every input returns `Ok` or a [`SpecError`]; none panics.
pub fn parse(src: &str) -> Result<PlotSpec, SpecError> {
    let calls = Parser {
        toks: lex(src)?,
        i: 0,
    }
    .calls()?;
    let mut spec = PlotSpec::default();
    for (i, call) in calls.iter().enumerate() {
        let Some((_, handler)) = CALLS.iter().find(|(n, _)| *n == call.name) else {
            return Err(SpecError::UnknownCall {
                name: call.name.clone(),
                at: call.at,
            });
        };
        if (i == 0) != (call.name == "ggplot") {
            return Err(syntax(
                call.at,
                "ggplot(...) must appear exactly once, as the first call",
            ));
        }
        handler(&mut spec, call)?;
    }
    Ok(spec)
}

// ───────────────────────────── canonical text ─────────────────────────────

fn quote(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '\\' => o.push_str("\\\\"),
            '"' => o.push_str("\\\""),
            '\n' => o.push_str("\\n"),
            '\t' => o.push_str("\\t"),
            '\r' => o.push_str("\\r"),
            _ => o.push(c),
        }
    }
    o.push('"');
    o
}

/// A bare identifier when `s` lexes back as exactly one, else a quoted string.
fn name_text(s: &str) -> String {
    let bare = s
        .chars()
        .next()
        .is_some_and(|c| c.is_alphabetic() || c == '_')
        && s.chars().all(is_ident_char)
        && s != "TRUE"
        && s != "FALSE";
    if bare {
        s.to_string()
    } else {
        quote(s)
    }
}

fn pair_text((a, b): (f64, f64)) -> String {
    format!("c({a:?}, {b:?})")
}

fn call_text(name: &str, args: &[String]) -> String {
    format!("{name}({})", args.join(", "))
}

fn aes_text(aes: &Aes) -> String {
    let fields = [
        ("x", &aes.x),
        ("y", &aes.y),
        ("colour", &aes.colour),
        ("fill", &aes.fill),
    ];
    let args: Vec<String> = fields
        .iter()
        .filter_map(|(k, v)| v.as_ref().map(|v| format!("{k} = {}", name_text(v))))
        .collect();
    call_text("aes", &args)
}

fn aes_args(aes: &Aes) -> Vec<String> {
    if aes.is_empty() {
        Vec::new()
    } else {
        vec![aes_text(aes)]
    }
}

fn sized_text(name: &str, aes: &Aes, size: Option<f64>) -> String {
    let mut args = aes_args(aes);
    args.extend(size.map(|v| format!("size = {v:?}")));
    call_text(name, &args)
}

fn positioned_text(name: &str, aes: &Aes, position: Option<Position>, bar: bool) -> String {
    let mut args = aes_args(aes);
    if bar {
        args.push("stat = \"identity\"".to_string());
    }
    args.extend(position.map(|p| format!("position = {}", quote(p.name()))));
    call_text(name, &args)
}

fn layer_text(layer: &Layer) -> String {
    match layer {
        Layer::Point { aes, size } => sized_text("geom_point", aes, *size),
        Layer::Line { aes, size } => sized_text("geom_line", aes, *size),
        Layer::Step { aes, size } => sized_text("geom_step", aes, *size),
        Layer::Col { aes, position } => positioned_text("geom_col", aes, *position, false),
        Layer::Bar { aes, position } => positioned_text("geom_bar", aes, *position, true),
        Layer::Smooth { aes } => {
            let mut args = aes_args(aes);
            args.push("method = \"lm\"".to_string());
            args.push("se = FALSE".to_string());
            call_text("geom_smooth", &args)
        }
        Layer::Abline { slope, intercept } => {
            format!("geom_abline(slope = {slope:?}, intercept = {intercept:?})")
        }
        Layer::Hline { yintercept } => format!("geom_hline(yintercept = {yintercept:?})"),
        Layer::Vline { xintercept } => format!("geom_vline(xintercept = {xintercept:?})"),
    }
}

fn coord_text(lim: Limits) -> Option<String> {
    let args: Vec<String> = [("xlim", lim.x), ("ylim", lim.y)]
        .into_iter()
        .filter_map(|(k, v)| v.map(|p| format!("{k} = {}", pair_text(p))))
        .collect();
    (!args.is_empty()).then(|| call_text("coord_cartesian", &args))
}

fn scale_text(name: &str, limits: Option<(f64, f64)>) -> Option<String> {
    limits.map(|p| format!("{name}(limits = {})", pair_text(p)))
}

fn values_text(name: &str, values: Option<&[String]>) -> Option<String> {
    values.map(|vs| {
        let items: Vec<String> = vs.iter().map(|v| quote(v)).collect();
        format!("{name}(values = c({}))", items.join(", "))
    })
}

fn labs_text(l: &Labs) -> Option<String> {
    let fields = [
        ("title", &l.title),
        ("subtitle", &l.subtitle),
        ("caption", &l.caption),
        ("x", &l.x),
        ("y", &l.y),
        ("colour", &l.colour),
        ("fill", &l.fill),
    ];
    let args: Vec<String> = fields
        .iter()
        .filter_map(|(k, v)| v.as_ref().map(|v| format!("{k} = {}", quote(v))))
        .collect();
    (!args.is_empty()).then(|| call_text("labs", &args))
}

fn facet_text(f: &Facet) -> String {
    let mut args = vec![format!("~ {}", name_text(&f.var))];
    args.extend(f.ncol.map(|n| format!("ncol = {n}")));
    args.extend(f.nrow.map(|n| format!("nrow = {n}")));
    call_text("facet_wrap", &args)
}

impl PlotSpec {
    fn parts(&self) -> Vec<String> {
        let mut head: Vec<String> = self.data.iter().map(|d| name_text(d)).collect();
        head.extend(aes_args(&self.aes));
        let mut parts = vec![call_text("ggplot", &head)];
        parts.extend(self.layers.iter().map(layer_text));
        parts.extend(self.facet.iter().map(facet_text));
        parts.extend(self.flip.then(|| "coord_flip()".to_string()));
        parts.extend(coord_text(self.coord));
        parts.extend(scale_text("scale_x_continuous", self.scale.x));
        parts.extend(scale_text("scale_y_continuous", self.scale.y));
        parts.extend(values_text(
            "scale_colour_manual",
            self.colour_values.as_deref(),
        ));
        parts.extend(values_text(
            "scale_fill_manual",
            self.fill_values.as_deref(),
        ));
        parts.extend(labs_text(&self.labs));
        parts.extend(self.theme.map(|t| {
            let size = t.base_size.map(|b| format!("base_size = {b:?}"));
            call_text("theme_minimal", &size.into_iter().collect::<Vec<_>>())
        }));
        parts
    }
}

/// Canonical text: `parse(spec.to_string()) == spec` for every spec [`parse`] can produce.
impl fmt::Display for PlotSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.parts().join(" +\n  "))
    }
}

// ───────────────────────────── tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const FULL: &str = "# comment lines and trailing comments are allowed
ggplot(df, aes(x = week, y = score, colour = team)) +   # trailing
  geom_point(size = 2) +
  geom_smooth(method = \"lm\", se = FALSE) +
  facet_wrap(~ cohort, ncol = 2) +
  coord_cartesian(ylim = c(0, 100)) +
  scale_colour_manual(values = c(\"#1b9e77\", \"#d95f02\")) +
  labs(title = \"Scores\", x = \"Week\", y = \"Score\", colour = \"Team\") +
  theme_minimal(base_size = 18)
";

    fn ok(src: &str) -> PlotSpec {
        parse(src).expect("the spec parses")
    }

    fn err(src: &str) -> SpecError {
        parse(src).expect_err("the spec is rejected")
    }

    fn at(line: usize, col: usize) -> Pos {
        Pos { line, col }
    }

    fn s(v: &str) -> Option<String> {
        Some(v.to_string())
    }

    // ── (1) the worked example and its round trip ──

    #[test]
    fn full_example_parses_to_the_typed_spec() {
        let p = ok(FULL);
        assert_eq!(p.data, s("df"));
        assert_eq!(
            p.aes,
            Aes {
                x: s("week"),
                y: s("score"),
                colour: s("team"),
                fill: None
            }
        );
        assert_eq!(
            p.layers,
            vec![
                Layer::Point {
                    aes: Aes::default(),
                    size: Some(2.0)
                },
                Layer::Smooth {
                    aes: Aes::default()
                },
            ]
        );
        assert_eq!(
            p.facet,
            Some(Facet {
                var: "cohort".into(),
                ncol: Some(2),
                nrow: None
            })
        );
        assert_eq!(
            p.coord,
            Limits {
                x: None,
                y: Some((0.0, 100.0))
            }
        );
        assert_eq!(p.scale, Limits::default());
        assert_eq!(
            p.colour_values,
            Some(vec!["#1b9e77".to_string(), "#d95f02".to_string()])
        );
        assert_eq!(p.fill_values, None);
        assert_eq!(p.labs.title, s("Scores"));
        assert_eq!(p.labs.x, s("Week"));
        assert_eq!(p.labs.y, s("Score"));
        assert_eq!(p.labs.colour, s("Team"));
        assert_eq!(
            p.theme,
            Some(Theme {
                base_size: Some(18.0)
            })
        );
        assert!(!p.flip);
    }

    #[test]
    fn full_example_round_trips_through_display() {
        let p = ok(FULL);
        let text = p.to_string();
        assert_eq!(ok(&text), p);
        assert_eq!(ok(&text).to_string(), text, "Display is canonical");
    }

    #[test]
    fn display_is_the_exact_canonical_text() {
        let p =
            ok("ggplot(df, aes(x = a, y = \"b c\")) + geom_col(position = 'dodge') + coord_flip()");
        assert_eq!(
            p.to_string(),
            "ggplot(df, aes(x = a, y = \"b c\")) +\n  geom_col(position = \"dodge\") +\n  coord_flip()"
        );
    }

    #[test]
    fn every_supported_call_round_trips() {
        let src = "ggplot(\"my data\", aes(x = a, y = b, colour = c, fill = d)) +
            geom_point(aes(x = TRUE2), size = 1.5) + geom_line(size = -2) + geom_step() +
            geom_col(aes(fill = g), position = \"stack\") + geom_bar(stat = \"identity\") +
            geom_smooth(aes(colour = k), method = \"lm\", se = FALSE) +
            geom_abline(slope = 1, intercept = -0.5) + geom_hline(yintercept = 3) + geom_vline(xintercept = 1e3) +
            facet_wrap(~ g, ncol = 2, nrow = 3) + coord_flip() +
            coord_cartesian(xlim = c(0, 1), ylim = c(-1, 1)) +
            scale_x_continuous(limits = c(0, 9)) + scale_y_continuous(limits = c(1, 2)) +
            scale_colour_manual(values = c(\"a\", \"b\")) + scale_fill_manual(values = c()) +
            labs(title = \"t\", subtitle = \"s\", caption = \"c\", x = \"x\", y = \"y\", colour = \"c\", fill = \"f\") +
            theme_minimal()";
        let p = ok(src);
        assert_eq!(p.layers.len(), 9);
        assert_eq!(ok(&p.to_string()), p);
    }

    #[test]
    fn display_quotes_names_that_would_not_lex_back() {
        let p = PlotSpec {
            data: s("TRUE"),
            aes: Aes {
                x: s(""),
                y: s("1abc"),
                colour: s("a b"),
                fill: s("FALSE"),
            },
            labs: Labs {
                title: s("line\n\t\r\"\\ 'q'"),
                ..Labs::default()
            },
            facet: Some(Facet {
                var: "9 lives".into(),
                ncol: None,
                nrow: None,
            }),
            ..PlotSpec::default()
        };
        assert_eq!(ok(&p.to_string()), p);
    }

    // ── values and lexing ──

    #[test]
    fn numbers_cover_sign_decimal_and_exponent_forms() {
        let n = |lit: &str, want: f64| {
            let p = ok(&format!("ggplot() + geom_hline(yintercept = {lit})"));
            assert_eq!(p.layers, vec![Layer::Hline { yintercept: want }], "{lit}");
        };
        n("7", 7.0);
        n("-7", -7.0);
        n("2.5", 2.5);
        n("-2.5", -2.5);
        n(".5", 0.5);
        n("-.5", -0.5);
        n("1e3", 1000.0);
        n("1E+2", 100.0);
        n("25e-1", 2.5);
    }

    #[test]
    fn a_dangling_exponent_is_not_part_of_the_number() {
        let e = err("ggplot() + geom_hline(yintercept = 1e)");
        assert!(matches!(e, SpecError::Syntax { .. }), "{e}");
        let e = err("ggplot() + geom_hline(yintercept = 1e+)");
        assert!(matches!(e, SpecError::Syntax { .. }), "{e}");
    }

    #[test]
    fn strings_take_both_quotes_and_the_escape_set() {
        let p = ok(r#"ggplot() + labs(title = 'it\'s', x = "a\"b", y = "t\tn\nr\rb\\")"#);
        assert_eq!(p.labs.title, s("it's"));
        assert_eq!(p.labs.x, s("a\"b"));
        assert_eq!(p.labs.y, s("t\tn\nr\rb\\"));
    }

    #[test]
    fn a_string_may_contain_a_newline_and_a_hash() {
        let p = ok("ggplot() + labs(title = \"a # not a comment\nsecond line\")");
        assert_eq!(p.labs.title, s("a # not a comment\nsecond line"));
    }

    #[test]
    fn color_and_colour_are_the_same_key() {
        let a = ok("ggplot(df, aes(color = g)) + labs(color = \"G\") + scale_color_manual(values = c(\"r\"))");
        let b = ok("ggplot(df, aes(colour = g)) + labs(colour = \"G\") + scale_colour_manual(values = c(\"r\"))");
        assert_eq!(a, b);
        assert_eq!(a.aes.colour, s("g"));
    }

    #[test]
    fn ggplot_argument_forms() {
        let want = Aes {
            x: s("a"),
            y: s("b"),
            ..Aes::default()
        };
        for src in [
            "ggplot(d, aes(x = a, y = b))",
            "ggplot(data = d, mapping = aes(x = a, y = b))",
            "ggplot(\"d\", aes(a, b))",
            "ggplot(data = d, aes(a, b))",
        ] {
            let p = ok(src);
            assert_eq!(p.data, s("d"), "{src}");
            assert_eq!(p.aes, want, "{src}");
        }
        assert_eq!(ok("ggplot(aes(a, b))").data, None);
        assert_eq!(ok("ggplot(aes(a, b))").aes, want);
        assert_eq!(ok("ggplot()"), PlotSpec::default());
        assert_eq!(ok("ggplot(d,)").data, s("d"), "trailing comma");
    }

    #[test]
    fn aes_may_sit_inside_a_geom() {
        let p =
            ok("ggplot(d) + geom_point(aes(x = a)) + geom_line(mapping = aes(y = b), size = 3)");
        assert_eq!(
            p.layers,
            vec![
                Layer::Point {
                    aes: Aes {
                        x: s("a"),
                        ..Aes::default()
                    },
                    size: None
                },
                Layer::Line {
                    aes: Aes {
                        y: s("b"),
                        ..Aes::default()
                    },
                    size: Some(3.0)
                },
            ]
        );
    }

    #[test]
    fn later_calls_win_like_ggplot2() {
        let p = ok("ggplot() + labs(x = \"a\", title = \"T\") + xlab(\"b\") + ylab(\"y\") + ggtitle(\"T2\", subtitle = \"S\") + facet_wrap(~a) + facet_wrap(~b, nrow = 1) + theme_minimal() + theme_minimal(base_size = 9) + coord_cartesian(xlim = c(0, 1)) + coord_cartesian(ylim = c(2, 3)) + scale_x_continuous(limits = c(0, 1)) + scale_x_continuous() + scale_y_continuous(limits = c(5, 6))");
        assert_eq!(p.labs.x, s("b"));
        assert_eq!(p.labs.y, s("y"));
        assert_eq!(p.labs.title, s("T2"));
        assert_eq!(p.labs.subtitle, s("S"));
        assert_eq!(p.facet.as_ref().map(|f| f.var.as_str()), Some("b"));
        assert_eq!(
            p.theme,
            Some(Theme {
                base_size: Some(9.0)
            })
        );
        assert_eq!(
            p.coord,
            Limits {
                x: Some((0.0, 1.0)),
                y: Some((2.0, 3.0))
            }
        );
        assert_eq!(
            p.scale,
            Limits {
                x: Some((0.0, 1.0)),
                y: Some((5.0, 6.0))
            }
        );
        let q = ok("ggplot() + ggtitle(label = \"only\")");
        assert_eq!((q.labs.title, q.labs.subtitle), (s("only"), None));
    }

    #[test]
    fn facet_counts_and_formula_forms() {
        let f = ok("ggplot() + facet_wrap(facets = ~ \"my col\", ncol = 3, nrow = 4)").facet;
        assert_eq!(
            f,
            Some(Facet {
                var: "my col".into(),
                ncol: Some(3),
                nrow: Some(4)
            })
        );
    }

    #[test]
    fn positional_intercepts_and_labels() {
        let p = ok("ggplot() + geom_abline(2, 1) + geom_hline(5) + geom_vline(6) + xlab(\"X\") + ylab(label = \"Y\")");
        assert_eq!(
            p.layers,
            vec![
                Layer::Abline {
                    slope: 2.0,
                    intercept: 1.0
                },
                Layer::Hline { yintercept: 5.0 },
                Layer::Vline { xintercept: 6.0 },
            ]
        );
        assert_eq!((p.labs.x, p.labs.y), (s("X"), s("Y")));
    }

    #[test]
    fn newlines_and_comments_are_whitespace() {
        let p = ok("ggplot(\n  d\n  # c\n  ,\n  aes(x = a)\n)\n+ coord_flip() # end");
        assert!(p.flip);
    }

    #[test]
    fn the_supported_table_is_the_documented_set() {
        let calls = supported_calls();
        assert_eq!(calls.len(), 23);
        for want in [
            "ggplot",
            "geom_point",
            "geom_smooth",
            "facet_wrap",
            "scale_fill_manual",
            "theme_minimal",
        ] {
            assert!(calls.contains(&want), "{want}");
        }
        let mut sorted = calls.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), calls.len(), "no duplicate call names");
    }

    // ── (2) unsupported = named error with position ──

    #[test]
    fn unsupported_calls_are_named_with_line_and_column() {
        for name in [
            "geom_histogram",
            "geom_boxplot",
            "scale_x_log10",
            "theme_bw",
            "geom_density",
        ] {
            let e = err(&format!("ggplot(d) +\n    {name}()"));
            assert_eq!(
                e,
                SpecError::UnknownCall {
                    name: name.to_string(),
                    at: at(2, 5)
                }
            );
            let msg = e.to_string();
            assert!(msg.contains(name), "{msg}");
            assert!(msg.contains("line 2, column 5"), "{msg}");
            assert!(msg.contains("geom_point"), "lists what is supported: {msg}");
        }
    }

    #[test]
    fn an_unknown_first_call_is_named_before_the_ggplot_rule() {
        let e = err("geom_histogram()");
        assert!(
            matches!(e, SpecError::UnknownCall { ref name, .. } if name == "geom_histogram"),
            "{e}"
        );
        let e = err("aes(x = a)");
        assert!(
            matches!(e, SpecError::UnknownCall { ref name, .. } if name == "aes"),
            "{e}"
        );
    }

    #[test]
    fn unknown_arguments_name_call_argument_and_position() {
        let cases = [
            (
                "ggplot(d) + geom_point(alpha = 0.5)",
                "geom_point",
                "alpha",
                at(1, 24),
            ),
            (
                "ggplot(d) + geom_line(linetype = 2)",
                "geom_line",
                "linetype",
                at(1, 23),
            ),
            ("ggplot(d, aes(shape = g))", "aes", "shape", at(1, 15)),
            (
                "ggplot(d) + theme_minimal(base_family = 1)",
                "theme_minimal",
                "base_family",
                at(1, 27),
            ),
            (
                "ggplot(d) + geom_col(stat = \"identity\")",
                "geom_col",
                "stat",
                at(1, 22),
            ),
            (
                "ggplot(d) + facet_wrap(~a, scales = 1)",
                "facet_wrap",
                "scales",
                at(1, 28),
            ),
            ("ggplot(d) + labs(tag = 1)", "labs", "tag", at(1, 18)),
            (
                "ggplot(d) + coord_flip(xlim = 1)",
                "coord_flip",
                "xlim",
                at(1, 24),
            ),
        ];
        for (src, call, arg, pos) in cases {
            let e = err(src);
            assert_eq!(
                e,
                SpecError::UnknownArg {
                    call: call.to_string(),
                    arg: arg.to_string(),
                    at: pos
                },
                "{src}"
            );
            assert!(
                e.to_string().contains(call) && e.to_string().contains(arg),
                "{e}"
            );
        }
    }

    #[test]
    fn positional_and_duplicate_and_missing_arguments_are_named() {
        let e = err("ggplot(d) + geom_point(3)");
        assert_eq!(
            e,
            SpecError::UnexpectedPositional {
                call: "geom_point".into(),
                at: at(1, 24)
            }
        );
        assert!(e.to_string().contains("geom_point"));
        let e = err("ggplot(d) + geom_point(size = 1, size = 2)");
        assert_eq!(
            e,
            SpecError::DuplicateArg {
                call: "geom_point".into(),
                arg: "size".into(),
                at: at(1, 34)
            }
        );
        assert!(e.to_string().contains("size"));
        let e = err("ggplot(d) + geom_point(aes(x = a), aes(y = b))");
        assert!(
            matches!(e, SpecError::DuplicateArg { ref arg, .. } if arg == "mapping"),
            "{e}"
        );
        let e = err("ggplot(d) + geom_hline()");
        assert_eq!(
            e,
            SpecError::MissingArg {
                call: "geom_hline".into(),
                arg: "yintercept".into(),
                at: at(1, 13)
            }
        );
        assert!(e.to_string().contains("yintercept"));
        for (src, arg) in [
            ("ggplot() + geom_abline(slope = 1)", "intercept"),
            ("ggplot() + geom_abline(intercept = 1)", "slope"),
            ("ggplot() + geom_vline()", "xintercept"),
            ("ggplot() + facet_wrap(ncol = 2)", "facets"),
            ("ggplot() + xlab()", "label"),
            ("ggplot() + ylab()", "label"),
            ("ggplot() + ggtitle()", "label"),
            ("ggplot() + scale_colour_manual()", "values"),
            ("ggplot() + scale_fill_manual()", "values"),
        ] {
            let e = err(src);
            assert!(
                matches!(e, SpecError::MissingArg { arg: ref a, .. } if a == arg),
                "{src}: {e}"
            );
        }
    }

    #[test]
    fn bad_values_name_the_argument_and_the_expectation() {
        let cases = [
            (
                "ggplot(d) + geom_point(size = \"big\")",
                "geom_point",
                "size",
                at(1, 31),
            ),
            ("ggplot(d, aes(x = 1))", "aes", "x", at(1, 19)),
            (
                "ggplot(d) + geom_point(mapping = 3)",
                "geom_point",
                "mapping",
                at(1, 34),
            ),
            (
                "ggplot(d) + geom_smooth(method = lm)",
                "geom_smooth",
                "method",
                at(1, 34),
            ),
            (
                "ggplot(d) + geom_smooth(method = \"lm\", se = 0)",
                "geom_smooth",
                "se",
                at(1, 45),
            ),
            (
                "ggplot(d) + geom_col(position = \"fill\")",
                "geom_col",
                "position",
                at(1, 33),
            ),
            (
                "ggplot(d) + geom_col(position = 1)",
                "geom_col",
                "position",
                at(1, 33),
            ),
            (
                "ggplot(d) + facet_wrap(a)",
                "facet_wrap",
                "facets",
                at(1, 24),
            ),
            (
                "ggplot(d) + facet_wrap(~a, ncol = 0)",
                "facet_wrap",
                "ncol",
                at(1, 35),
            ),
            (
                "ggplot(d) + facet_wrap(~a, ncol = 1.5)",
                "facet_wrap",
                "ncol",
                at(1, 35),
            ),
            (
                "ggplot(d) + facet_wrap(~a, nrow = -2)",
                "facet_wrap",
                "nrow",
                at(1, 35),
            ),
            (
                "ggplot(d) + facet_wrap(~a, nrow = 99999999999)",
                "facet_wrap",
                "nrow",
                at(1, 35),
            ),
            (
                "ggplot(d) + coord_cartesian(ylim = 5)",
                "coord_cartesian",
                "ylim",
                at(1, 36),
            ),
            (
                "ggplot(d) + coord_cartesian(ylim = c(1))",
                "coord_cartesian",
                "ylim",
                at(1, 36),
            ),
            (
                "ggplot(d) + coord_cartesian(ylim = c(1, \"a\"))",
                "coord_cartesian",
                "ylim",
                at(1, 36),
            ),
            (
                "ggplot(d) + scale_x_continuous(limits = c(1, 2, 3))",
                "scale_x_continuous",
                "limits",
                at(1, 41),
            ),
            (
                "ggplot(d) + scale_colour_manual(values = c(1, 2))",
                "scale_colour_manual",
                "values",
                at(1, 42),
            ),
            (
                "ggplot(d) + scale_fill_manual(values = \"a\")",
                "scale_fill_manual",
                "values",
                at(1, 40),
            ),
            ("ggplot(d) + labs(title = 5)", "labs", "title", at(1, 26)),
            (
                "ggplot(d) + geom_hline(yintercept = \"a\")",
                "geom_hline",
                "yintercept",
                at(1, 37),
            ),
            ("ggplot(foo(1))", "ggplot", "data", at(1, 8)),
            ("ggplot(TRUE)", "ggplot", "data", at(1, 8)),
        ];
        for (src, call, arg, pos) in cases {
            let e = err(src);
            assert!(
                matches!(&e, SpecError::BadValue { call: c, arg: a, expected, at: p }
                    if (c.as_str(), a.as_str(), *p) == (call, arg, pos) && !expected.is_empty()),
                "{src}: {e}"
            );
            assert!(
                e.to_string().contains(call) && e.to_string().contains(arg),
                "{e}"
            );
        }
    }

    #[test]
    fn geom_bar_accepts_only_identity() {
        for src in [
            "ggplot(d) + geom_bar()",
            "ggplot(d) + geom_bar(stat = \"count\")",
            "ggplot(d) + geom_bar(stat = \"bin\")",
            "ggplot(d) + geom_bar(position = \"dodge\")",
        ] {
            let e = err(src);
            assert!(
                matches!(&e, SpecError::Unsupported { call, at: p, .. } if call == "geom_bar" && *p == at(1, 13)),
                "{src}: {e}"
            );
            assert!(e.to_string().contains("identity"), "{e}");
        }
        let p = ok(
            "ggplot(d) + geom_bar(stat = \"identity\", position = \"stack\", mapping = aes(y = v))",
        );
        assert_eq!(
            p.layers,
            vec![Layer::Bar {
                aes: Aes {
                    y: s("v"),
                    ..Aes::default()
                },
                position: Some(Position::Stack)
            }]
        );
    }

    // ── (3) loess / se is an explicit error ──

    #[test]
    fn geom_smooth_other_than_lm_with_se_false_is_rejected_in_words() {
        for src in [
            "ggplot(d) + geom_smooth()",
            "ggplot(d) + geom_smooth(se = FALSE)",
            "ggplot(d) + geom_smooth(method = \"loess\")",
            "ggplot(d) + geom_smooth(method = \"loess\", se = FALSE)",
            "ggplot(d) + geom_smooth(method = \"gam\", se = FALSE)",
            "ggplot(d) + geom_smooth(method = \"lm\")",
            "ggplot(d) + geom_smooth(method = \"lm\", se = TRUE)",
        ] {
            let e = err(src);
            assert!(
                matches!(&e, SpecError::Unsupported { call, at: p, .. } if call == "geom_smooth" && *p == at(1, 13)),
                "{src}: {e}"
            );
            let msg = e.to_string();
            assert!(msg.contains("loess"), "{src}: {msg}");
            assert!(msg.contains("confidence band"), "{src}: {msg}");
            assert!(msg.contains("0.72.6"), "{src}: {msg}");
            assert!(msg.contains("geom_smooth"), "{src}: {msg}");
        }
        assert!(err("ggplot(d) + geom_smooth(method = \"gam\")")
            .to_string()
            .contains("\"gam\""));
        assert!(err("ggplot(d) + geom_smooth(method = \"lm\")")
            .to_string()
            .contains("se must be FALSE"));
        assert!(err("ggplot(d) + geom_smooth()")
            .to_string()
            .contains("method is required"));
    }

    // ── syntax errors ──

    #[test]
    fn syntax_errors_carry_the_cause_and_position() {
        let cases: &[(&str, Pos, &str)] = &[
            ("", at(1, 1), "end of input"),
            ("   # only a comment\n", at(2, 1), "end of input"),
            ("ggplot", at(1, 7), "`(`"),
            ("ggplot(", at(1, 8), "a value"),
            ("ggplot(d", at(1, 9), "`,` or `)`"),
            ("ggplot(d) +", at(1, 12), "a call"),
            ("ggplot(d) geom_point()", at(1, 11), "`+`"),
            ("ggplot(d) + 5", at(1, 13), "a call"),
            ("ggplot(d) + geom_point", at(1, 23), "`(`"),
            ("ggplot(d) @", at(1, 11), "unexpected character '@'"),
            (
                "ggplot(d) - geom_point()",
                at(1, 11),
                "unexpected character '-'",
            ),
            ("ggplot(\"abc", at(1, 8), "unterminated string"),
            ("ggplot('abc\\", at(1, 8), "unterminated string"),
            ("ggplot(\"a\\qb\")", at(1, 10), "unknown escape `\\q`"),
            ("ggplot(d, x = 1e999)", at(1, 15), "not a finite number"),
            (
                "ggplot(d) + facet_wrap(~)",
                at(1, 25),
                "variable name after `~`",
            ),
            (
                "ggplot(d) + scale_colour_manual(values = c(a = \"x\"))",
                at(1, 44),
                "named elements",
            ),
            ("ggplot(= 1)", at(1, 8), "a value"),
            ("ggplot(d)) ", at(1, 10), "`+` or end of input"),
            (
                "geom_point() + ggplot(d)",
                at(1, 1),
                "ggplot(...) must appear exactly once",
            ),
            (
                "ggplot(d) + ggplot(d)",
                at(1, 13),
                "ggplot(...) must appear exactly once",
            ),
        ];
        for &(src, pos, needle) in cases {
            let e = err(src);
            assert!(matches!(e, SpecError::Syntax { .. }), "{src}: {e}");
            assert_eq!(e.at(), pos, "{src}: {e}");
            assert!(e.to_string().contains(needle), "{src}: {e}");
            assert!(
                e.to_string()
                    .contains(&format!("line {}, column {}", pos.line, pos.col)),
                "{e}"
            );
        }
    }

    #[test]
    fn deep_nesting_is_an_error_not_a_stack_overflow() {
        let deep = format!("ggplot(d, x = {}1{})", "c(".repeat(200), ")".repeat(200));
        let e = err(&deep);
        assert!(
            matches!(&e, SpecError::Syntax { msg, .. } if msg.contains("nested too deeply")),
            "{e}"
        );
        let fine = format!("ggplot(d, x = {}1{})", "c(".repeat(40), ")".repeat(40));
        assert!(matches!(
            err(&fine),
            SpecError::UnexpectedPositional { .. } | SpecError::UnknownArg { .. }
        ));
    }

    #[test]
    fn at_reads_the_position_of_every_variant() {
        let p = at(3, 4);
        let all = [
            SpecError::Syntax {
                msg: String::new(),
                at: p,
            },
            SpecError::UnknownCall {
                name: String::new(),
                at: p,
            },
            SpecError::UnknownArg {
                call: String::new(),
                arg: String::new(),
                at: p,
            },
            SpecError::UnexpectedPositional {
                call: String::new(),
                at: p,
            },
            SpecError::DuplicateArg {
                call: String::new(),
                arg: String::new(),
                at: p,
            },
            SpecError::MissingArg {
                call: String::new(),
                arg: String::new(),
                at: p,
            },
            SpecError::BadValue {
                call: String::new(),
                arg: String::new(),
                expected: String::new(),
                at: p,
            },
            SpecError::Unsupported {
                call: String::new(),
                detail: String::new(),
                at: p,
            },
        ];
        for e in all {
            assert_eq!(e.at(), p);
            assert!(e.to_string().contains("line 3, column 4"), "{e}");
            let _: &dyn std::error::Error = &e;
        }
    }

    #[test]
    fn token_descriptions_name_what_was_found() {
        let e = err("ggplot(d) + 1.5");
        assert!(e.to_string().contains("number 1.5"), "{e}");
        let e = err("ggplot(d) + \"s\"");
        assert!(e.to_string().contains("a string"), "{e}");
        for (src, found) in [
            ("ggplot(d) + )", "`)`"),
            ("ggplot(d) + ,", "`,`"),
            ("ggplot(d) + +", "`+`"),
            ("ggplot(d) + =", "`=`"),
            ("ggplot(d) + ~", "`~`"),
            ("ggplot(d) + (", "`(`"),
        ] {
            let e = err(src);
            assert!(e.to_string().contains(found), "{src}: {e}");
        }
    }

    // ── (5) no panic on any input ──

    fn arb_name() -> impl Strategy<Value = Option<String>> {
        proptest::option::of(any::<String>())
    }

    fn arb_f64() -> impl Strategy<Value = f64> + Clone {
        use proptest::num::f64 as pf;
        pf::NORMAL | pf::SUBNORMAL | pf::ZERO
    }

    fn arb_aes() -> impl Strategy<Value = Aes> {
        (arb_name(), arb_name(), arb_name(), arb_name()).prop_map(|(x, y, colour, fill)| Aes {
            x,
            y,
            colour,
            fill,
        })
    }

    fn arb_position() -> impl Strategy<Value = Option<Position>> {
        proptest::option::of(prop_oneof![Just(Position::Dodge), Just(Position::Stack)])
    }

    fn arb_layer() -> impl Strategy<Value = Layer> {
        let size = proptest::option::of(arb_f64());
        prop_oneof![
            (arb_aes(), size.clone()).prop_map(|(aes, size)| Layer::Point { aes, size }),
            (arb_aes(), size.clone()).prop_map(|(aes, size)| Layer::Line { aes, size }),
            (arb_aes(), size).prop_map(|(aes, size)| Layer::Step { aes, size }),
            (arb_aes(), arb_position()).prop_map(|(aes, position)| Layer::Col { aes, position }),
            (arb_aes(), arb_position()).prop_map(|(aes, position)| Layer::Bar { aes, position }),
            arb_aes().prop_map(|aes| Layer::Smooth { aes }),
            (arb_f64(), arb_f64())
                .prop_map(|(slope, intercept)| Layer::Abline { slope, intercept }),
            arb_f64().prop_map(|yintercept| Layer::Hline { yintercept }),
            arb_f64().prop_map(|xintercept| Layer::Vline { xintercept }),
        ]
    }

    fn arb_limits() -> impl Strategy<Value = Limits> {
        let pair = proptest::option::of((arb_f64(), arb_f64()));
        (pair.clone(), pair).prop_map(|(x, y)| Limits { x, y })
    }

    fn arb_labs() -> impl Strategy<Value = Labs> {
        (
            (arb_name(), arb_name(), arb_name(), arb_name()),
            (arb_name(), arb_name(), arb_name()),
        )
            .prop_map(|((title, subtitle, caption, x), (y, colour, fill))| Labs {
                title,
                subtitle,
                caption,
                x,
                y,
                colour,
                fill,
            })
    }

    fn arb_facet() -> impl Strategy<Value = Option<Facet>> {
        let n = proptest::option::of(1u32..50);
        proptest::option::of(
            (any::<String>(), n.clone(), n).prop_map(|(var, ncol, nrow)| Facet { var, ncol, nrow }),
        )
    }

    fn arb_spec() -> impl Strategy<Value = PlotSpec> {
        let values = proptest::option::of(proptest::collection::vec(any::<String>(), 0..4));
        let head = (
            arb_name(),
            arb_aes(),
            proptest::collection::vec(arb_layer(), 0..5),
        );
        let mid = (arb_facet(), any::<bool>(), arb_limits(), arb_limits());
        let tail = (
            values.clone(),
            values,
            arb_labs(),
            proptest::option::of(proptest::option::of(arb_f64())),
        );
        (head, mid, tail).prop_map(
            |(
                (data, aes, layers),
                (facet, flip, coord, scale),
                (colour_values, fill_values, labs, theme),
            )| PlotSpec {
                data,
                aes,
                layers,
                facet,
                flip,
                coord,
                scale,
                colour_values,
                fill_values,
                labs,
                theme: theme.map(|base_size| Theme { base_size }),
            },
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn round_trip_parse_of_display_is_identity(spec in arb_spec()) {
            let text = spec.to_string();
            let back = parse(&text);
            prop_assert_eq!(back.as_ref(), Ok(&spec), "text was:\n{}", text);
        }

        #[test]
        fn parse_never_panics_on_arbitrary_strings(src in any::<String>()) {
            let _ = parse(&src);
        }

        #[test]
        fn parse_never_panics_on_arbitrary_bytes(bytes in proptest::collection::vec(any::<u8>(), 0..200)) {
            let _ = parse(&String::from_utf8_lossy(&bytes));
        }

        #[test]
        fn parse_never_panics_on_token_soup(src in "[a-z_().,=~+\"' 0-9\\\\#\n-]{0,120}") {
            if let Err(e) = parse(&src) {
                let p = e.at();
                prop_assert!(p.line >= 1 && p.col >= 1);
                prop_assert!(!e.to_string().is_empty());
            }
        }

        #[test]
        fn mutated_valid_specs_never_panic(cut in 0usize..FULL.len(), junk in "[ -~]{0,6}") {
            let mut s = FULL.to_string();
            let cut = (0..=cut).rev().find(|i| s.is_char_boundary(*i)).unwrap_or(0);
            s.insert_str(cut, &junk);
            let _ = parse(&s);
            let _ = parse(&FULL[..cut]);
        }
    }
}
