//! `pv check-symbolic <contract>` (PMAT-6399): test the stated invariants of each equation against
//! that equation's own `formula`, at domain edge points and fixed-seed pseudo-random points.
//!
//! Subset (deliberately small): the formula is `;`-separated definitions `F(pattern, ..) = expr`
//! (patterns: a variable, a number, `k*i`, `ki`, `k*i+c`) over `+ - * / ^`, `sin cos exp sqrt abs`.
//! An invariant is `a = b` or a chain `a <= F(..) <= b` (`<`, `<=`, `>`, `>=`, `=`), joined by `and`,
//! after a trailing `(comment)` and a `for all ...` suffix are dropped. It must call at least one
//! function the formula defines.
//!
//! Verdicts: `holds` (>= 1 point evaluated, none failed), `violated` (with a counterexample) and
//! `unparsed` (with the reason). `unparsed` is a third state: it is reported and never counted as a pass.
//!
//! Arithmetic is `f64` with a relative tolerance of 1e-9, NOT exact rational: `sin`/`cos` are not
//! rational, so the report names its arithmetic. A point where a definition has no matching clause or
//! the value is not finite is skipped; an invariant with no evaluated point is `unparsed`, not `holds`.

use std::collections::BTreeMap;
use std::path::Path;

use provable_contracts::schema::parse_contract;

const TOL: f64 = 1e-9;

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Id(String),
    Op(char),
}

#[derive(Debug, Clone)]
enum Expr {
    Num(f64),
    Var(String),
    Neg(Box<Expr>),
    Bin(char, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
}

fn take_run(c: &[char], i: &mut usize, ok: impl Fn(char) -> bool) -> String {
    let st = *i;
    while *i < c.len() && ok(c[*i]) {
        *i += 1;
    }
    c[st..*i].iter().collect()
}

fn lex(s: &str) -> Result<Vec<Tok>, String> {
    let c: Vec<char> = s.chars().collect();
    let (mut i, mut out) = (0, Vec::new());
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
        } else if ch.is_ascii_digit() {
            let t = take_run(&c, &mut i, |x| x.is_ascii_digit() || x == '.');
            out.push(Tok::Num(
                t.parse().map_err(|_| format!("bad number '{t}'"))?,
            ));
        } else if ch.is_ascii_alphabetic() || ch == '_' {
            out.push(Tok::Id(take_run(&c, &mut i, |x| {
                x.is_ascii_alphanumeric() || x == '_'
            })));
        } else if "+-*/^(),".contains(ch) {
            out.push(Tok::Op(ch));
            i += 1;
        } else {
            return Err(format!("unsupported character '{ch}'"));
        }
    }
    Ok(out)
}

struct Parser {
    t: Vec<Tok>,
    p: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.p)
    }
    fn op(&mut self, o: char) -> bool {
        if self.peek() == Some(&Tok::Op(o)) {
            self.p += 1;
            true
        } else {
            false
        }
    }
    fn expr(&mut self) -> Result<Expr, String> {
        let mut l = self.term()?;
        loop {
            let o = if self.op('+') {
                '+'
            } else if self.op('-') {
                '-'
            } else {
                return Ok(l);
            };
            l = Expr::Bin(o, Box::new(l), Box::new(self.term()?));
        }
    }
    fn term(&mut self) -> Result<Expr, String> {
        let mut l = self.unary()?;
        loop {
            let o = if self.op('*') {
                '*'
            } else if self.op('/') {
                '/'
            } else if matches!(l, Expr::Num(_)) && matches!(self.peek(), Some(Tok::Id(_))) {
                '*' // implicit multiplication: `2i`
            } else {
                return Ok(l);
            };
            l = Expr::Bin(o, Box::new(l), Box::new(self.unary()?));
        }
    }
    fn unary(&mut self) -> Result<Expr, String> {
        if self.op('-') {
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        let b = self.primary()?;
        if self.op('^') {
            return Ok(Expr::Bin('^', Box::new(b), Box::new(self.unary()?)));
        }
        Ok(b)
    }
    fn call_args(&mut self) -> Result<Vec<Expr>, String> {
        let mut args = Vec::new();
        if self.op(')') {
            return Ok(args);
        }
        loop {
            args.push(self.expr()?);
            if self.op(')') {
                return Ok(args);
            }
            if !self.op(',') {
                return Err("expected ',' or ')'".into());
            }
        }
    }
    fn primary(&mut self) -> Result<Expr, String> {
        match self.t.get(self.p).cloned() {
            Some(Tok::Num(n)) => {
                self.p += 1;
                Ok(Expr::Num(n))
            }
            Some(Tok::Id(name)) => {
                self.p += 1;
                if !self.op('(') {
                    return Ok(Expr::Var(name));
                }
                Ok(Expr::Call(name, self.call_args()?))
            }
            Some(Tok::Op('(')) => {
                self.p += 1;
                let e = self.expr()?;
                if self.op(')') {
                    Ok(e)
                } else {
                    Err("expected ')'".into())
                }
            }
            other => Err(format!("unexpected {other:?}")),
        }
    }
}

fn parse_expr(s: &str) -> Result<Expr, String> {
    let mut p = Parser { t: lex(s)?, p: 0 };
    let e = p.expr()?;
    match p.peek() {
        None => Ok(e),
        Some(t) => Err(format!("unparsed trailing token {t:?}")),
    }
}

fn collect_vars(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::Var(v) => {
            if !out.contains(v) {
                out.push(v.clone());
            }
        }
        Expr::Neg(a) => collect_vars(a, out),
        Expr::Bin(_, a, b) => {
            collect_vars(a, out);
            collect_vars(b, out);
        }
        Expr::Call(_, args) => args.iter().for_each(|a| collect_vars(a, out)),
        Expr::Num(_) => {}
    }
}

/// Variables used as `k*i` / `ki` index multipliers: quantified over integers, never reals.
fn collect_index_vars(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::Bin('*', a, b) => {
            if let (Expr::Num(_), Expr::Var(v)) = (&**a, &**b) {
                out.push(v.clone());
            }
            collect_index_vars(a, out);
            collect_index_vars(b, out);
        }
        Expr::Neg(a) => collect_index_vars(a, out),
        Expr::Bin(_, a, b) => {
            collect_index_vars(a, out);
            collect_index_vars(b, out);
        }
        Expr::Call(_, args) => args.iter().for_each(|a| collect_index_vars(a, out)),
        _ => {}
    }
}

fn collect_calls(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::Call(n, args) => {
            out.push(n.clone());
            for a in args.iter() {
                collect_calls(a, out);
            }
        }
        Expr::Neg(a) => collect_calls(a, out),
        Expr::Bin(_, a, b) => {
            collect_calls(a, out);
            collect_calls(b, out);
        }
        _ => {}
    }
}

/// One definition `name(patterns) = body`.
struct Clause {
    name: String,
    pats: Vec<Expr>,
    body: Expr,
}

type Env = BTreeMap<String, f64>;

fn near_int(x: f64) -> Option<f64> {
    ((x - x.round()).abs() < TOL).then(|| x.round())
}

/// Bind `pat` against the concrete `v`; `None` = no match (or an unsupported pattern).
fn match_pat(pat: &Expr, v: f64, env: &mut Env) -> Option<()> {
    match pat {
        Expr::Num(c) => ((v - c).abs() < TOL).then_some(()),
        Expr::Var(n) => {
            env.insert(n.clone(), v);
            Some(())
        }
        Expr::Bin('*', a, b) => match (&**a, &**b) {
            (Expr::Num(k), Expr::Var(n)) if *k != 0.0 => {
                env.insert(n.clone(), near_int(v / k)?);
                Some(())
            }
            _ => None,
        },
        Expr::Bin('+', a, b) => match &**b {
            Expr::Num(c) => match_pat(a, v - c, env),
            _ => None,
        },
        _ => None,
    }
}

fn eval(e: &Expr, env: &Env, defs: &[Clause]) -> Option<f64> {
    let r = match e {
        Expr::Num(n) => *n,
        Expr::Var(v) => *env.get(v)?,
        Expr::Neg(a) => -eval(a, env, defs)?,
        Expr::Bin(o, a, b) => {
            let (x, y) = (eval(a, env, defs)?, eval(b, env, defs)?);
            match o {
                '+' => x + y,
                '-' => x - y,
                '*' => x * y,
                '/' => x / y,
                _ => x.powf(y),
            }
        }
        Expr::Call(n, args) => {
            let v: Vec<f64> = args
                .iter()
                .map(|a| eval(a, env, defs))
                .collect::<Option<_>>()?;
            match (n.as_str(), v.as_slice()) {
                ("sin", [x]) => x.sin(),
                ("cos", [x]) => x.cos(),
                ("exp", [x]) => x.exp(),
                ("sqrt", [x]) => x.sqrt(),
                ("abs", [x]) => x.abs(),
                _ => {
                    let amb: Env = env.clone();
                    defs.iter()
                        .filter(|c| c.name == *n && c.pats.len() == v.len())
                        .find_map(|c| {
                            let mut benv = amb.clone();
                            c.pats
                                .iter()
                                .zip(&v)
                                .try_for_each(|(p, x)| match_pat(p, *x, &mut benv))?;
                            eval(&c.body, &benv, defs)
                        })?
                }
            }
        }
    };
    r.is_finite().then_some(r)
}

fn parse_defs(formula: &str) -> Vec<Clause> {
    formula
        .split(';')
        .filter_map(|c| {
            let (l, r) = c.split_once('=')?;
            let (Expr::Call(name, pats), body) = (parse_expr(l).ok()?, parse_expr(r).ok()?) else {
                return None;
            };
            Some(Clause { name, pats, body })
        })
        .collect()
}

/// Drop a trailing ` (comment)` and a ` for all ...` / ` for ...` suffix.
fn strip_prose(s: &str) -> String {
    let mut s = s.trim().to_string();
    if s.ends_with(')') {
        if let Some(i) = s.rfind(" (") {
            if s[i + 2..].matches('(').count() < s[i + 2..].matches(')').count() {
                s.truncate(i);
            }
        }
    }
    if let Some(i) = s.find(" for ") {
        s.truncate(i);
    }
    s.trim().to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cmp {
    Eq,
    Le,
    Lt,
    Ge,
    Gt,
}

fn holds_cmp(c: Cmp, a: f64, b: f64) -> bool {
    let eps = TOL * (1.0 + a.abs() + b.abs());
    match c {
        Cmp::Eq => (a - b).abs() <= eps,
        Cmp::Le => a <= b + eps,
        Cmp::Ge => a + eps >= b,
        Cmp::Lt => a < b,
        Cmp::Gt => a > b,
    }
}

/// Split `a <= b <= c` at depth 0 into operands and comparators.
fn split_chain(s: &str) -> Result<(Vec<String>, Vec<Cmp>), String> {
    let c: Vec<char> = s.chars().collect();
    let (mut depth, mut i, mut cur) = (0i32, 0, String::new());
    let (mut parts, mut cmps) = (Vec::new(), Vec::new());
    while i < c.len() {
        match c[i] {
            '(' => depth += 1,
            ')' => depth -= 1,
            _ => {}
        }
        let two = c.get(i + 1) == Some(&'=');
        let cmp = match (depth == 0, c[i]) {
            (true, '<') if two => Some((Cmp::Le, 2)),
            (true, '>') if two => Some((Cmp::Ge, 2)),
            (true, '<') => Some((Cmp::Lt, 1)),
            (true, '>') => Some((Cmp::Gt, 1)),
            (true, '=') => Some((Cmp::Eq, if two { 2 } else { 1 })),
            _ => None,
        };
        if let Some((k, n)) = cmp {
            parts.push(std::mem::take(&mut cur));
            cmps.push(k);
            i += n;
        } else {
            cur.push(c[i]);
            i += 1;
        }
    }
    parts.push(cur);
    if cmps.is_empty() {
        return Err("no comparison (=, <, <=, >, >=)".into());
    }
    Ok((parts, cmps))
}

/// Fixed-seed LCG so the sampled points are reproducible.
fn pool(seed: u64) -> Vec<f64> {
    let mut v = vec![0.0, 1.0, -1.0, 2.0, 3.0, 7.0, 0.5, 64.0];
    let mut s = seed;
    for _ in 0..6 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push(((s >> 33) % 2001) as f64 / 10.0 - 100.0);
    }
    for _ in 0..4 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push(((s >> 33) % 200) as f64);
    }
    v
}

/// Verdict for one invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Holds { points: usize },
    Violated { counterexample: String },
    Unparsed { reason: String },
}

fn parse_all(parts: &[String]) -> Result<Vec<Expr>, String> {
    parts
        .iter()
        .map(|p| parse_expr(p).map_err(|e| format!("'{}': {e}", p.trim())))
        .collect()
}

/// Free in a definition body, bound by none of its patterns (e.g. `d`), and not already an invariant variable.
fn ambient_vars(defs: &[Clause], vars: &[String]) -> Vec<String> {
    let mut amb = Vec::new();
    for d in defs {
        let (mut bound, mut free) = (Vec::new(), Vec::new());
        for p in &d.pats {
            collect_vars(p, &mut bound);
        }
        collect_vars(&d.body, &mut free);
        amb.extend(
            free.into_iter()
                .filter(|v| !bound.contains(v) && !vars.contains(v)),
        );
    }
    amb.sort();
    amb.dedup();
    amb
}

/// One sample axis per variable; index multipliers only take non-negative integers.
fn sample_axes(vars: &[String], idxv: &[String], amb: &[String]) -> Vec<(String, Vec<f64>)> {
    let axis = |(k, v): (usize, &String)| {
        let p = pool(k as u64 + 1);
        let p = if idxv.contains(v) {
            p.into_iter()
                .filter(|x| *x >= 0.0 && x.fract() == 0.0)
                .collect()
        } else {
            p
        };
        (v.clone(), p)
    };
    vars.iter()
        .enumerate()
        .map(axis)
        .chain(amb.iter().map(|v| (v.clone(), vec![2.0, 4.0, 8.0, 64.0])))
        .collect()
}

/// Advance the odometer; false once every combination has been visited.
fn advance(idx: &mut [usize], names: &[(String, Vec<f64>)]) -> bool {
    for (i, slot) in idx.iter_mut().enumerate() {
        *slot += 1;
        if *slot < names[i].1.len() {
            return true;
        }
        *slot = 0;
    }
    false
}

fn counterexample(env: &Env, parts: &[String], cmps: &[Cmp], v: &[f64]) -> Option<String> {
    let k = (0..cmps.len()).find(|&k| !holds_cmp(cmps[k], v[k], v[k + 1]))?;
    let at: Vec<String> = env.iter().map(|(n, x)| format!("{n}={x}")).collect();
    Some(format!(
        "at {}: '{}' = {} vs '{}' = {} fails {:?}",
        at.join(", "),
        parts[k].trim(),
        v[k],
        parts[k + 1].trim(),
        v[k + 1],
        cmps[k]
    ))
}

/// Evaluate every combination; Err carries the first counterexample, Ok the number of in-domain points.
fn sweep(
    names: &[(String, Vec<f64>)],
    exprs: &[Expr],
    defs: &[Clause],
    parts: &[String],
    cmps: &[Cmp],
) -> Result<usize, String> {
    let (mut idx, mut points) = (vec![0usize; names.len()], 0usize);
    loop {
        let env: Env = names
            .iter()
            .zip(&idx)
            .map(|((n, p), &i)| (n.clone(), p[i]))
            .collect();
        let vals: Option<Vec<f64>> = exprs.iter().map(|e| eval(e, &env, defs)).collect();
        if let Some(v) = vals {
            points += 1;
            if let Some(c) = counterexample(&env, parts, cmps, &v) {
                return Err(c);
            }
        }
        if !advance(&mut idx, names) {
            return Ok(points);
        }
    }
}

fn check_conjunct(text: &str, defs: &[Clause]) -> Verdict {
    let unparsed = |r: String| Verdict::Unparsed { reason: r };
    let (parts, cmps) = match split_chain(text) {
        Ok(x) => x,
        Err(e) => return unparsed(e),
    };
    let exprs = match parse_all(&parts) {
        Ok(x) => x,
        Err(e) => return unparsed(e),
    };
    let (mut calls, mut vars, mut idxv) = (Vec::new(), Vec::new(), Vec::new());
    for e in &exprs {
        collect_calls(e, &mut calls);
        collect_vars(e, &mut vars);
        collect_index_vars(e, &mut idxv);
    }
    if !calls.iter().any(|c| defs.iter().any(|d| d.name == *c)) {
        return unparsed("calls no function defined by the equation's formula".into());
    }
    let amb = ambient_vars(defs, &vars);
    let names = sample_axes(&vars, &idxv, &amb);
    let points = match sweep(&names, &exprs, defs, &parts, &cmps) {
        Ok(n) => n,
        Err(c) => return Verdict::Violated { counterexample: c },
    };
    if points == 0 {
        unparsed("no sampled point fell in the definition's domain".into())
    } else {
        Verdict::Holds { points }
    }
}

/// Check one invariant against one equation's formula.
pub fn check_invariant(formula: &str, invariant: &str) -> Verdict {
    let defs = parse_defs(formula);
    if defs.is_empty() {
        return Verdict::Unparsed {
            reason: "formula has no parseable `F(..) = expr` definition".into(),
        };
    }
    let mut pts = 0;
    for conj in strip_prose(invariant).split(" and ") {
        match check_conjunct(conj, &defs) {
            Verdict::Holds { points } => pts += points,
            other => return other,
        }
    }
    Verdict::Holds { points: pts }
}

/// Run the command: print every invariant's verdict; error if any is violated.
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to an unwrap
pub fn run(path: &Path, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    let mut rows = Vec::new();
    for (id, eq) in &contract.equations {
        for inv in &eq.invariants {
            rows.push((id.clone(), inv.clone(), check_invariant(&eq.formula, inv)));
        }
    }
    let n = |f: fn(&Verdict) -> bool| rows.iter().filter(|r| f(&r.2)).count();
    let (h, v, u) = (
        n(|x| matches!(x, Verdict::Holds { .. })),
        n(|x| matches!(x, Verdict::Violated { .. })),
        n(|x| matches!(x, Verdict::Unparsed { .. })),
    );
    if json {
        let items: Vec<serde_json::Value> = rows
            .iter()
            .map(|(e, i, vd)| {
                let (s, d) = match vd {
                    Verdict::Holds { points } => ("holds", format!("{points} points")),
                    Verdict::Violated { counterexample } => ("violated", counterexample.clone()),
                    Verdict::Unparsed { reason } => ("unparsed", reason.clone()),
                };
                serde_json::json!({"equation": e, "invariant": i, "verdict": s, "detail": d})
            })
            .collect();
        let out = serde_json::json!({"arithmetic": "f64, tol 1e-9", "holds": h, "violated": v, "unparsed": u, "invariants": items});
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!(
            "check-symbolic {} (f64, tol 1e-9; unparsed is NOT a pass)",
            path.display()
        );
        for (e, i, vd) in &rows {
            match vd {
                Verdict::Holds { points } => println!("  holds     [{e}] {i}  ({points} points)"),
                Verdict::Violated { counterexample } => {
                    println!("  VIOLATED  [{e}] {i}\n            {counterexample}")
                }
                Verdict::Unparsed { reason } => println!("  unparsed  [{e}] {i}  ({reason})"),
            }
        }
        println!("holds {h}, violated {v}, unparsed {u}");
    }
    if v > 0 {
        return Err(format!("{v} invariant(s) violated").into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIN: &str =
        "PE(pos, 2i) = sin(pos / 10000^(2i/d)); PE(pos, 2i+1) = cos(pos / 10000^(2i/d))";

    fn contract_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/absolute-position-v1.yaml")
    }

    #[test]
    fn zero_position_holds_and_bound_holds() {
        assert!(matches!(
            check_invariant(SIN, "PE(0, 2i) = 0 and PE(0, 2i+1) = 1"),
            Verdict::Holds { .. }
        ));
        let b = check_invariant(SIN, "-1 <= PE(pos, j) <= 1 for all pos, j (sin/cos range)");
        assert!(
            matches!(b, Verdict::Holds { points } if points > 100),
            "{b:?}"
        );
    }

    #[test]
    fn sin_to_cos_is_violated_with_counterexample() {
        let bad = SIN.replace("sin(", "cos(");
        match check_invariant(&bad, "PE(0, 2i) = 0 and PE(0, 2i+1) = 1") {
            Verdict::Violated { counterexample } => {
                assert!(counterexample.contains("i="), "{counterexample}")
            }
            v => panic!("expected violated, got {v:?}"),
        }
    }

    #[test]
    fn wrong_bound_is_violated() {
        assert!(matches!(
            check_invariant(SIN, "PE(pos, j) <= 0.5"),
            Verdict::Violated { .. }
        ));
    }

    #[test]
    fn prose_and_foreign_invariants_are_unparsed_never_holds() {
        for s in [
            "PE(pos+k) is a linear rotation of PE(pos) by angle k*omega(i) (angle addition)",
            "output.shape = token_embed.shape (shape preservation)",
            "t < max_position for all valid positions",
            "output[t] is finite for finite inputs",
        ] {
            assert!(
                matches!(check_invariant(SIN, s), Verdict::Unparsed { .. }),
                "{s}"
            );
        }
    }

    #[test]
    fn point_outside_every_clause_domain_is_unparsed() {
        // F is only defined at 2i; `F(1) = 0` never matches a clause, so zero points -> unparsed, not holds
        assert!(matches!(
            check_invariant("F(2i) = 0", "F(1) = 0"),
            Verdict::Unparsed { .. }
        ));
    }

    #[test]
    fn real_contract_cos_mutation_flips_holds_to_violated() {
        let text = std::fs::read_to_string(contract_path()).expect("contract");
        let dir = std::env::temp_dir().join(format!("pv-6399-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let ok = dir.join("ok.yaml");
        let bad = dir.join("bad.yaml");
        std::fs::write(&ok, &text).expect("w");
        std::fs::write(&bad, text.replace("sin(", "cos(")).expect("w");
        assert!(
            run(&ok, false).is_ok(),
            "unmodified contract must have no violation"
        );
        assert!(run(&bad, false).is_err(), "sin->cos must report violated");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
