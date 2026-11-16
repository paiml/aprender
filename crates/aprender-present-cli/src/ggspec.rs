//! Parser for the ggplot2 spec text the `plot` verb shares with R (PLOT-GG-001, #6918):
//! `ggplot(d, aes(x, y)) + geom_point() + facet_wrap(~v) + labs(title = "t") + theme_bw()`.
//! Every unsupported call or argument is an error naming it and its 1-based column; nothing is
//! ignored. `parse(print(ast)) == ast` holds for every AST the parser can produce.

/// One argument value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Num(f64),
    Str(String),
    Bool(bool),
    Sym(String),
    Formula(String),
    Vec(Vec<Value>),
    Aes(Vec<Arg>),
}

/// A positional (`name == None`) or named argument.
#[derive(Debug, Clone, PartialEq)]
pub struct Arg {
    pub name: Option<String>,
    pub value: Value,
}

/// One call in the `+` chain.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub name: String,
    pub args: Vec<Arg>,
}

/// A parse or validation error at a 1-based column.
#[derive(Debug, Clone, PartialEq)]
pub struct SpecError {
    pub col: usize,
    pub msg: String,
}

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "column {}: {}", self.col, self.msg)
    }
}

const GEOMS: &[&str] = &[
    "geom_point",
    "geom_line",
    "geom_path",
    "geom_step",
    "geom_hline",
    "geom_vline",
    "geom_abline",
    "geom_smooth",
];
const THEMES: &[&str] = &[
    "theme_bw",
    "theme_minimal",
    "theme_gray",
    "theme_grey",
    "theme_classic",
    "theme_light",
    "theme_dark",
    "theme_void",
    "theme_linedraw",
];
const GEOM_ARGS: &[&str] = &[
    "mapping",
    "data",
    "colour",
    "color",
    "size",
    "alpha",
    "linetype",
    "linewidth",
    "direction",
    "yintercept",
    "xintercept",
    "slope",
    "intercept",
    "method",
    "se",
];

fn allowed(call: &str, name: &str) -> bool {
    match call {
        "ggplot" => matches!(name, "data" | "mapping"),
        "facet_wrap" => matches!(name, "facets" | "nrow" | "ncol"),
        "labs" => matches!(name, "title" | "subtitle" | "caption" | "x" | "y"),
        "aes" => true,
        c if THEMES.contains(&c) => name == "base_size",
        _ => GEOM_ARGS.contains(&name),
    }
}

struct P<'a> {
    s: &'a [char],
    i: usize,
}

type R<T> = Result<T, SpecError>;

impl P<'_> {
    fn err<T>(&self, col: usize, msg: String) -> R<T> {
        Err(SpecError { col, msg })
    }
    fn ws(&mut self) {
        while self.s.get(self.i).is_some_and(|c| c.is_whitespace()) {
            self.i += 1;
        }
    }
    fn eat(&mut self, c: char) -> bool {
        self.ws();
        let hit = self.s.get(self.i) == Some(&c);
        self.i += usize::from(hit);
        hit
    }
    fn expect(&mut self, c: char) -> R<()> {
        if self.eat(c) {
            return Ok(());
        }
        let found = self
            .s
            .get(self.i)
            .map_or("end of input".into(), |c| format!("'{c}'"));
        self.err(self.i + 1, format!("expected '{c}', found {found}"))
    }
    fn ident(&mut self) -> Option<String> {
        self.ws();
        let st = self.i;
        let ok = |c: &char, first: bool| {
            c.is_alphabetic() || *c == '_' || (!first && (c.is_ascii_digit() || *c == '.'))
        };
        while self.s.get(self.i).is_some_and(|c| ok(c, self.i == st)) {
            self.i += 1;
        }
        (self.i > st).then(|| self.s[st..self.i].iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &[char]) -> P<'_> {
        P { s, i: 0 }
    }

    #[test]
    fn ast_values_compare_and_error_displays() {
        let arg = |name: Option<&str>, value| Arg {
            name: name.map(Into::into),
            value,
        };
        let aes = Value::Aes(vec![arg(Some("x"), Value::Sym("a".into()))]);
        let vs = [Value::Num(1.5), Value::Str("s".into()), Value::Bool(true)];
        let vs = [vs.to_vec(), vec![Value::Formula("v".into()), aes]].concat();
        let args = vec![arg(None, Value::Vec(vs))];
        let call = Call {
            name: "c".into(),
            args,
        };
        assert_eq!(call.clone(), call);
        let msg = "boom".into();
        assert_eq!(SpecError { col: 3, msg }.to_string(), "column 3: boom");
    }

    #[test]
    fn allow_lists_gate_arguments() {
        assert!(GEOMS.contains(&"geom_point") && THEMES.contains(&"theme_bw"));
        assert!(allowed("ggplot", "mapping") && !allowed("ggplot", "size"));
        assert!(allowed("facet_wrap", "nrow") && !allowed("labs", "size"));
        assert!(allowed("aes", "anything") && allowed("theme_bw", "base_size"));
        assert!(allowed("geom_point", "size") && !allowed("geom_point", "bogus"));
    }

    #[test]
    fn helpers_scan_and_report_columns() {
        let cs: Vec<char> = "  ab_1.x ( ".chars().collect();
        let mut q = p(&cs);
        assert_eq!(q.ident().as_deref(), Some("ab_1.x"));
        assert!(q.eat('(') && !q.eat('('));
        let e = q.expect(')').expect_err("end of input");
        assert_eq!(
            (e.col, e.msg.as_str()),
            (12, "expected ')', found end of input")
        );
        let cs: Vec<char> = "9".chars().collect();
        assert_eq!(p(&cs).ident(), None);
        assert_eq!(p(&cs).expect('x').expect_err("x").col, 1);
    }
}
