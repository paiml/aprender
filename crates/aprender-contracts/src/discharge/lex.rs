//! PVL-001 EV-6a (#4139): Lean source read at TOKEN level. Comments and string/char literals are blanked first, so
//! `sorry-free` in a doc comment or `"axiom"` in a string is never an escape, and an `import` in a doc comment is
//! never an import. Newlines survive blanking, so every token keeps its line.

/// `src` with every comment (`--` to end of line, `/- … -/` nested) and string/char literal replaced by spaces.
#[must_use]
pub fn blank(src: &str) -> String {
    let c: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < c.len() {
        let skip = literal_or_comment_len(&c, i, out.chars().next_back());
        if skip == 0 {
            out.push(c[i]);
            i += 1;
            continue;
        }
        for &ch in &c[i..(i + skip).min(c.len())] {
            out.push(if ch == '\n' { '\n' } else { ' ' });
        }
        i += skip;
    }
    out
}

/// How many chars from `i` belong to a comment or literal that starts there (0 when none does).
fn literal_or_comment_len(c: &[char], i: usize, prev: Option<char>) -> usize {
    let at = |k: usize| c.get(k).copied();
    match (c[i], at(i + 1)) {
        ('-', Some('-')) => (i..c.len()).find(|&k| c[k] == '\n').unwrap_or(c.len()) - i,
        ('/', Some('-')) => block_comment_len(c, i),
        ('"', _) => string_len(c, i),
        ('\'', _) if !prev.is_some_and(is_ident_char) => char_literal_len(c, i),
        _ => 0,
    }
}

fn block_comment_len(c: &[char], i: usize) -> usize {
    let (mut depth, mut k) = (0usize, i);
    while k + 1 < c.len() {
        if c[k] == '/' && c[k + 1] == '-' {
            depth += 1;
            k += 2;
        } else if c[k] == '-' && c[k + 1] == '/' {
            depth -= 1;
            k += 2;
            if depth == 0 {
                return k - i;
            }
        } else {
            k += 1;
        }
    }
    c.len() - i
}

fn string_len(c: &[char], i: usize) -> usize {
    let mut k = i + 1;
    while k < c.len() && c[k] != '"' {
        k += if c[k] == '\\' { 2 } else { 1 };
    }
    (k + 1).min(c.len()) - i
}

/// `'a'`, `'\n'`, `'\''` — only when the quote closes within the literal's width; otherwise the `'` is not a literal.
fn char_literal_len(c: &[char], i: usize) -> usize {
    let width = if c.get(i + 1) == Some(&'\\') { 3 } else { 2 };
    if c.get(i + width) == Some(&'\'') {
        width + 1
    } else {
        0
    }
}

/// A char that continues a Lean identifier (`h₁`, `x'`, `Nat.succ`, `foo?`).
#[must_use]
pub fn is_ident_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '_' | '\'' | '!' | '?' | '.')
}

fn is_ident_start(ch: char) -> bool {
    (ch.is_alphabetic() || ch == '_') && !ch.is_numeric()
}

/// One identifier-or-keyword token of blanked source, with its 1-based line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub text: String,
    pub line: usize,
}

/// Every identifier/keyword token of `blanked` (output of [`blank`]), in order. Symbols are not tokens.
#[must_use]
pub fn tokens(blanked: &str) -> Vec<Token> {
    let mut out = Vec::new();
    for (n, line) in blanked.lines().enumerate() {
        let mut cur = String::new();
        for ch in line.chars().chain(std::iter::once(' ')) {
            if (cur.is_empty() && is_ident_start(ch)) || (!cur.is_empty() && is_ident_char(ch)) {
                cur.push(ch);
            } else if !cur.is_empty() {
                out.push(Token {
                    text: std::mem::take(&mut cur).trim_end_matches('.').to_string(),
                    line: n + 1,
                });
            }
        }
    }
    out
}

/// The modules a file's header imports (`import A B`, `public import A`, `import all A`), from blanked source.
#[must_use]
pub fn imports(blanked: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in blanked.lines() {
        let mut words = line
            .split_whitespace()
            .skip_while(|w| matches!(*w, "public" | "private" | "meta"));
        if words.next() != Some("import") {
            continue;
        }
        out.extend(words.filter(|w| *w != "all").map(str::to_string));
    }
    out
}

/// Declaration keywords whose next identifier is the declaration's name.
pub const NAMED_DECLS: &[&str] = &[
    "theorem",
    "lemma",
    "def",
    "axiom",
    "abbrev",
    "opaque",
    "structure",
    "inductive",
    "class",
];

/// One declaration: the token index of its keyword, its keyword, its fully qualified name and its privacy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decl {
    pub at: usize,
    pub keyword: String,
    pub fqn: String,
    pub name: String,
    pub private: bool,
    pub line: usize,
}

enum Scope {
    Namespace(Vec<String>),
    Other,
}

/// The declarations of a token stream, each named in its enclosing `namespace`s (`section`/`mutual` add nothing;
/// `end` closes the innermost scope; `_root_.x` escapes every namespace).
#[must_use]
pub fn decls(toks: &[Token]) -> Vec<Decl> {
    let mut scopes: Vec<Scope> = Vec::new();
    let mut out = Vec::new();
    for (i, t) in toks.iter().enumerate() {
        match t.text.as_str() {
            "namespace" => scopes.push(Scope::Namespace(
                toks.get(i + 1)
                    .map(|n| n.text.split('.').map(str::to_string).collect())
                    .unwrap_or_default(),
            )),
            "section" | "mutual" => scopes.push(Scope::Other),
            "end" => {
                scopes.pop();
            }
            "instance" | "example" => out.push(unnamed_decl(toks, i)),
            k if NAMED_DECLS.contains(&k) => {
                if let Some(d) = named_decl(toks, i, &scopes) {
                    out.push(d);
                }
            }
            _ => {}
        }
    }
    out
}

fn unnamed_decl(toks: &[Token], i: usize) -> Decl {
    Decl {
        at: i,
        keyword: toks[i].text.clone(),
        fqn: format!("<{}>", toks[i].text),
        name: format!("<{}>", toks[i].text),
        private: i > 0 && toks[i - 1].text == "private",
        line: toks[i].line,
    }
}

fn named_decl(toks: &[Token], i: usize, scopes: &[Scope]) -> Option<Decl> {
    let name = toks.get(i + 1)?.text.clone();
    let fqn = match name.strip_prefix("_root_.") {
        Some(rooted) => rooted.to_string(),
        None => {
            let mut parts: Vec<String> = scopes
                .iter()
                .filter_map(|s| match s {
                    Scope::Namespace(p) => Some(p.clone()),
                    Scope::Other => None,
                })
                .flatten()
                .collect();
            parts.push(name.clone());
            parts.join(".")
        }
    };
    Some(Decl {
        at: i,
        keyword: toks[i].text.clone(),
        fqn,
        name: name.rsplit('.').next().unwrap_or(&name).to_string(),
        private: i > 0 && toks[i - 1].text == "private",
        line: toks[i].line,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(src: &str) -> Vec<String> {
        tokens(&blank(src)).into_iter().map(|t| t.text).collect()
    }

    #[test]
    fn comments_and_strings_are_not_tokens() {
        let src =
            "-- sorry here\n/- axiom /- nested sorry -/ still -/ def a := \"sorry\" -- admit\n";
        assert_eq!(texts(src), vec!["def", "a"]);
    }

    #[test]
    fn a_line_comment_holding_slash_dash_opens_no_block() {
        assert_eq!(
            texts("-- see /- note\ntheorem t : True := trivial\n"),
            vec!["theorem", "t", "True", "trivial"]
        );
    }

    #[test]
    fn primes_are_identifiers_and_char_literals_are_blanked() {
        assert_eq!(
            texts("def x' := 'a'\ndef y := '\"'\n"),
            vec!["def", "x'", "def", "y"]
        );
    }

    #[test]
    fn lines_survive_blanking() {
        let t = tokens(&blank("/- a\nb\n-/\nsorry\n"));
        assert_eq!((t[0].text.as_str(), t[0].line), ("sorry", 4));
    }

    #[test]
    fn imports_skip_doc_comments() {
        let b = blank("/-!\nimport X.C\n-/\nimport X.A X.B\npublic import X.D\n");
        assert_eq!(imports(&b), vec!["X.A", "X.B", "X.D"]);
    }

    #[test]
    fn namespaces_qualify_and_end_closes() {
        let src = "namespace A.B\ntheorem t : True := trivial\nsection S\nlemma u : True := trivial\nend S\nend A.B\n\
                   theorem v : True := trivial\nnamespace C\ntheorem _root_.w : True := trivial\nprivate theorem p : True := trivial\nend C\n";
        let d = decls(&tokens(&blank(src)));
        let f: Vec<(&str, bool)> = d.iter().map(|d| (d.fqn.as_str(), d.private)).collect();
        assert_eq!(
            f,
            vec![
                ("A.B.t", false),
                ("A.B.u", false),
                ("v", false),
                ("w", false),
                ("C.p", true)
            ]
        );
    }
}
