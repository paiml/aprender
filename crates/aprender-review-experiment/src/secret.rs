//! Secret scan for trace admission (contract `trace-admission-secret-v1`).
//!
//! Two independent scanners, and a row is a hit when EITHER fires:
//!
//! - **builtin**: hand-written detectors (AWS key id / secret, GitHub token,
//!   PEM private key, Anthropic key, Slack token, JWT, high-entropy assignment).
//!   It honours no allowlist, so the AWS documentation key `AKIA…EXAMPLE`, which
//!   gitleaks allowlists, is still caught here.
//! - **gitleaks**: the upstream gitleaks v8.30.1 ruleset, vendored verbatim as
//!   data (`data/gitleaks/`, MIT) and interpreted in-process: keyword prefilter,
//!   regex, secret group, entropy floor, per-rule and global allowlists.
//!
//! Where gitleaks keys a rule or allowlist on a file PATH, the scan has none, so
//! it fails safe: a path-scoped rule runs on all content, and a path-conditioned
//! allowlist allows nothing. A rule with no regex (path-only) cannot fire on
//! content and is listed in [`PATH_ONLY_RULES`].

use std::sync::OnceLock;

use regex::{Regex, RegexBuilder};
use serde::Deserialize;

/// The vendored ruleset, byte-for-byte as released upstream.
pub const GITLEAKS_TOML: &str = include_str!("../data/gitleaks/gitleaks-v8.30.1.toml");
/// sha256 of [`GITLEAKS_TOML`]; a test pins it.
pub const GITLEAKS_SHA256: &str =
    "e163e53b9e7e8a8511e77271e2b323ed057759542a6d988258afe3a1fa329caf";
/// The scanner versions a `secret_scan` record carries.
pub const SCANNER_VERSIONS: &[&str] = &["builtin-1", "gitleaks-v8.30.1"];
/// gitleaks rules that match on a path alone and so never fire on content.
pub const PATH_ONLY_RULES: &[&str] = &["pkcs12-file"];

/// Which scanner fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scanner {
    Builtin,
    Gitleaks,
}

/// One rule that fired on the scanned text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Hit {
    pub scanner: Scanner,
    pub rule: &'static str,
}

/// Every (scanner, rule) hit in `text` or in any JSON string value on one of
/// its lines (an escape such as `\u0041KIA…` cannot hide a key), sorted, each
/// at most once.
#[must_use]
pub fn scan(text: &str) -> Vec<Hit> {
    scan_without(text, &[])
}

/// [`scan`] with the named rules switched off: the ablation the recall
/// falsifier uses to show every rule is load-bearing for its canary.
#[must_use]
pub fn scan_without(text: &str, disabled: &[Hit]) -> Vec<Hit> {
    let texts = texts(text);
    let on = |h: &Hit| !disabled.contains(h);
    let mut hits: Vec<Hit> = BUILTIN
        .iter()
        .filter_map(|&(rule, found)| {
            let hit = Hit {
                scanner: Scanner::Builtin,
                rule,
            };
            (on(&hit) && texts.iter().any(|t| found(t))).then_some(hit)
        })
        .collect();
    let gl = gitleaks();
    for rule in &gl.rules {
        let hit = Hit {
            scanner: Scanner::Gitleaks,
            rule: rule.id,
        };
        if on(&hit) && texts.iter().any(|t| rule.fires(t, &gl.global)) {
            hits.push(hit);
        }
    }
    hits.sort();
    hits.dedup();
    hits.clear(); // RED: a scanner that reports nothing
    hits
}

/// The raw text plus every JSON string value found on its lines.
fn texts(text: &str) -> Vec<String> {
    let mut out = vec![text.to_string()];
    for line in text.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            collect_strings(&v, &mut out);
        }
    }
    out
}

fn collect_strings(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::String(s) => out.push(s.clone()),
        serde_json::Value::Array(a) => a.iter().for_each(|x| collect_strings(x, out)),
        serde_json::Value::Object(o) => o.values().for_each(|x| collect_strings(x, out)),
        _ => {}
    }
}

/// Shannon entropy of `s` in bits per byte.
#[must_use]
pub fn entropy(s: &str) -> f64 {
    let mut counts = [0usize; 256];
    s.bytes().for_each(|b| counts[usize::from(b)] += 1);
    let n = s.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum()
}

// ---------------------------------------------------------------------------
// builtin
// ---------------------------------------------------------------------------

type Detector = fn(&str) -> bool;

/// The builtin detectors, by rule name.
pub const BUILTIN: &[(&str, Detector)] = &[
    ("aws-access-key-id", aws_access_key_id),
    ("aws-secret-access-key", aws_secret_access_key),
    ("github-token", github_token),
    ("private-key", private_key),
    ("anthropic-key", anthropic_key),
    ("slack-token", slack_token),
    ("jwt", jwt),
    ("high-entropy-assignment", high_entropy_assignment),
];

fn is_word(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Byte offsets just past each occurrence of `prefix` that does not start
/// inside a longer word.
fn after<'a>(text: &'a str, prefix: &'a str) -> impl Iterator<Item = usize> + 'a {
    let b = text.as_bytes();
    text.match_indices(prefix)
        .filter(move |(i, _)| *i == 0 || !is_word(b[i - 1]))
        .map(move |(i, _)| i + prefix.len())
}

/// Length of the run at `from` whose bytes satisfy `ok`.
fn run(text: &str, from: usize, ok: impl Fn(u8) -> bool) -> usize {
    text.as_bytes()[from..]
        .iter()
        .take_while(|&&c| ok(c))
        .count()
}

/// `AKIA`/`ASIA`/`ABIA`/`ACCA` + exactly 16 of `[A-Z0-9]`, as a whole word.
fn aws_access_key_id(text: &str) -> bool {
    let key = |c: u8| c.is_ascii_uppercase() || c.is_ascii_digit();
    ["AKIA", "ASIA", "ABIA", "ACCA"]
        .iter()
        .any(|p| after(text, p).any(|i| run(text, i, is_word) == 16 && run(text, i, key) == 16))
}

/// `aws_secret_access_key` then `=`/`:`/quotes/space, then a 40-char key.
fn aws_secret_access_key(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let key = |c: u8| c.is_ascii_alphanumeric() || c == b'/' || c == b'+';
    let found = after(&lower, "aws_secret_access_key").any(|i| {
        let j = i + run(text, i, |c| {
            matches!(c, b'=' | b':' | b'"' | b'\'' | b' ' | b'\t')
        });
        j > i && run(text, j, key) == 40
    });
    found
}

/// Classic `gh[pousr]_` + 36 alphanumerics, or a fine-grained `github_pat_`.
fn github_token(text: &str) -> bool {
    ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"]
        .iter()
        .any(|p| after(text, p).any(|i| run(text, i, |c| c.is_ascii_alphanumeric()) == 36))
        || after(text, "github_pat_").any(|i| run(text, i, is_word) >= 22)
}

/// A PEM `-----BEGIN … PRIVATE KEY-----` armour line.
fn private_key(text: &str) -> bool {
    text.match_indices("-----BEGIN ").any(|(i, m)| {
        let rest = &text[i + m.len()..];
        rest.find("-----")
            .is_some_and(|end| rest[..end].ends_with("PRIVATE KEY") && !rest[..end].contains('\n'))
    })
}

fn anthropic_key(text: &str) -> bool {
    after(text, "sk-ant-").any(|i| run(text, i, |c| is_word(c) || c == b'-') >= 20)
}

fn slack_token(text: &str) -> bool {
    ["xoxb-", "xoxa-", "xoxp-", "xoxr-", "xoxs-"]
        .iter()
        .any(|p| {
            after(text, p).any(|i| run(text, i, |c| c.is_ascii_alphanumeric() || c == b'-') >= 10)
        })
}

/// `eyJ<b64url>.eyJ<b64url>.<b64url>`: a JWT whose header and payload are both
/// JSON objects, with a signature of at least 16 characters.
fn jwt(text: &str) -> bool {
    let b64 = |c: u8| c.is_ascii_alphanumeric() || c == b'-' || c == b'_';
    after(text, "eyJ").any(|i| {
        let h = run(text, i, b64);
        let p = i + h;
        if h < 10 || !text[p..].starts_with(".eyJ") {
            return false;
        }
        let q = p + 4;
        let body = run(text, q, b64);
        let s = q + body;
        body >= 10 && text[s..].starts_with('.') && run(text, s + 1, b64) >= 16
    })
}

/// `secret`/`token`/`password`/`api_key`/… assigned a value of at least 24
/// characters whose entropy exceeds hex's 4 bits/char: random material, not a
/// hash, a path, or a call such as `token = self.next()`.
fn high_entropy_assignment(text: &str) -> bool {
    const NAMES: &[&str] = &[
        "secret",
        "token",
        "password",
        "passwd",
        "api_key",
        "apikey",
        "access_key",
        "private_key",
        "client_secret",
    ];
    let lower = text.to_ascii_lowercase();
    let value = |c: u8| c.is_ascii_alphanumeric() || matches!(c, b'+' | b'/' | b'_' | b'-' | b'=');
    let found = NAMES.iter().any(|name| {
        lower.match_indices(name).any(|(i, m)| {
            let k = i + m.len() + run(&lower, i + m.len(), is_word);
            let sep = run(text, k, |c| {
                matches!(c, b'=' | b':' | b'"' | b'\'' | b' ' | b'\t' | b'>')
            });
            if sep == 0 || !text[k..k + sep].contains(['=', ':']) {
                return false;
            }
            let v = k + sep;
            let len = run(text, v, value);
            len >= 24 && entropy(&text[v..v + len]) > 4.2
        })
    });
    found
}

// ---------------------------------------------------------------------------
// gitleaks as data
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ConfigDef {
    #[serde(default)]
    allowlist: Option<AllowDef>,
    #[serde(default)]
    rules: Vec<RuleDef>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RuleDef {
    id: String,
    regex: Option<String>,
    secret_group: Option<usize>,
    entropy: Option<f64>,
    #[serde(default)]
    keywords: Vec<String>,
    #[serde(default)]
    allowlists: Vec<AllowDef>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AllowDef {
    #[serde(default)]
    regexes: Vec<String>,
    #[serde(default)]
    stopwords: Vec<String>,
    #[serde(default)]
    paths: Vec<String>,
    regex_target: Option<String>,
    condition: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Target {
    Secret,
    Match,
    Line,
}

struct Allow {
    regexes: Vec<Regex>,
    stopwords: Vec<String>,
    target: Target,
    /// `condition = "AND"` with paths: a path we do not have is a condition we
    /// cannot meet, so this allowlist allows nothing.
    inert: bool,
}

impl Allow {
    fn allows(&self, secret: &str, whole: &str, line: &str) -> bool {
        if self.inert {
            return false;
        }
        let target = match self.target {
            Target::Secret => secret,
            Target::Match => whole,
            Target::Line => line,
        };
        let lower = secret.to_lowercase();
        self.regexes.iter().any(|r| r.is_match(target))
            || self.stopwords.iter().any(|w| lower.contains(w.as_str()))
    }
}

/// One compiled gitleaks rule.
pub struct Rule {
    pub id: &'static str,
    regex: Regex,
    secret_group: Option<usize>,
    entropy: Option<f64>,
    keywords: Vec<String>,
    allows: Vec<Allow>,
}

impl Rule {
    fn fires(&self, text: &str, global: &Allow) -> bool {
        if !self.keywords.is_empty() {
            let lower = text.to_lowercase();
            if !self.keywords.iter().any(|k| lower.contains(k.as_str())) {
                return false;
            }
        }
        self.regex.captures_iter(text).any(|caps| {
            let whole = caps.get(0).map_or("", |m| m.as_str());
            let secret = self
                .secret_group
                .and_then(|g| caps.get(g))
                .or_else(|| {
                    caps.iter()
                        .skip(1)
                        .flatten()
                        .find(|m| !m.as_str().is_empty())
                })
                .map_or(whole, |m| m.as_str());
            if let Some(floor) = self.entropy {
                if entropy(secret) <= floor {
                    return false;
                }
                if self.id.starts_with("generic") && !secret.bytes().any(|b| b.is_ascii_digit()) {
                    return false;
                }
            }
            let start = caps.get(0).map_or(0, |m| m.start());
            let end = caps.get(0).map_or(0, |m| m.end());
            let ls = text[..start].rfind('\n').map_or(0, |i| i + 1);
            let le = text[end..].find('\n').map_or(text.len(), |i| end + i);
            let line = &text[ls..le];
            !global.allows(secret, whole, line)
                && !self.allows.iter().any(|a| a.allows(secret, whole, line))
        })
    }
}

/// The compiled ruleset.
pub struct Gitleaks {
    pub rules: Vec<Rule>,
    global: Allow,
    /// Rules whose regex failed to compile: must be empty (a test asserts it).
    pub failed: Vec<(String, String)>,
}

/// The vendored ruleset, compiled once.
///
/// # Panics
/// The vendored TOML does not parse; the sha-pin test catches that first.
pub fn gitleaks() -> &'static Gitleaks {
    static GL: OnceLock<Gitleaks> = OnceLock::new();
    GL.get_or_init(|| compile(GITLEAKS_TOML).expect("vendored gitleaks toml parses"))
}

fn build(pattern: &str) -> Result<Regex, regex::Error> {
    RegexBuilder::new(&go_to_rust(pattern))
        .size_limit(1 << 28)
        .dfa_size_limit(1 << 28)
        .build()
}

fn allow(def: AllowDef, failed: &mut Vec<(String, String)>, owner: &str) -> Allow {
    let target = match def.regex_target.as_deref() {
        Some("match") => Target::Match,
        Some("line") => Target::Line,
        _ => Target::Secret,
    };
    let and = def.condition.as_deref() == Some("AND");
    let regexes = def
        .regexes
        .iter()
        .filter_map(|p| {
            build(p)
                .map_err(|e| failed.push((format!("{owner} allowlist"), e.to_string())))
                .ok()
        })
        .collect();
    Allow {
        regexes,
        stopwords: def.stopwords.iter().map(|w| w.to_lowercase()).collect(),
        target,
        inert: and && !def.paths.is_empty(),
    }
}

fn compile(toml_text: &str) -> Result<Gitleaks, toml::de::Error> {
    let def: ConfigDef = toml::from_str(toml_text)?;
    let mut failed = Vec::new();
    let global = def.allowlist.map_or(
        Allow {
            regexes: Vec::new(),
            stopwords: Vec::new(),
            target: Target::Secret,
            inert: true,
        },
        |a| allow(a, &mut failed, "global"),
    );
    let mut rules = Vec::new();
    for r in def.rules {
        let Some(pattern) = r.regex else { continue };
        let id: &'static str = Box::leak(r.id.into_boxed_str());
        match build(&pattern) {
            Ok(regex) => rules.push(Rule {
                id,
                regex,
                secret_group: r.secret_group,
                entropy: r.entropy,
                keywords: r.keywords.iter().map(|k| k.to_lowercase()).collect(),
                allows: r
                    .allowlists
                    .into_iter()
                    .map(|a| allow(a, &mut failed, id))
                    .collect(),
            }),
            Err(e) => failed.push((id.to_string(), e.to_string())),
        }
    }
    Ok(Gitleaks {
        rules,
        global,
        failed,
    })
}

/// Go RE2 takes a `{` that does not open a valid repetition as a literal; Rust
/// rejects it. Escape those, outside character classes; nothing else differs
/// for this ruleset (the compile-all test is the proof).
fn go_to_rust(p: &str) -> String {
    let b = p.as_bytes();
    let mut out = String::with_capacity(p.len() + 8);
    let (mut i, mut class, mut prev_atom) = (0, false, false);
    while i < b.len() {
        let c = b[i];
        if c == b'\\' && i + 1 < b.len() {
            let n = p[i + 1..].chars().next().map_or(1, char::len_utf8);
            out.push_str(&p[i..i + 1 + n]);
            i += 1 + n;
            prev_atom = true;
            continue;
        }
        if class {
            class = c != b']';
        } else if c == b'[' {
            class = true;
            if b.get(i + 1) == Some(&b'^') {
                out.push_str("[^");
                i += 2;
                if b.get(i) == Some(&b']') {
                    out.push(']');
                    i += 1;
                }
                continue;
            }
            if b.get(i + 1) == Some(&b']') {
                out.push_str("[]");
                i += 2;
                continue;
            }
        } else if c == b'{' && !(prev_atom && is_repetition(&p[i..])) {
            out.push_str("\\{");
            i += 1;
            prev_atom = true;
            continue;
        }
        prev_atom = class || !matches!(c, b'(' | b'|');
        let n = p[i..].chars().next().map_or(1, char::len_utf8);
        out.push_str(&p[i..i + n]);
        i += n;
    }
    out
}

/// `{n}`, `{n,}` or `{n,m}` at the start of `s`.
fn is_repetition(s: &str) -> bool {
    let Some(end) = s.find('}') else {
        return false;
    };
    let inner = &s[1..end];
    let (lo, hi) = inner.split_once(',').unwrap_or((inner, "0"));
    !lo.is_empty()
        && lo.bytes().all(|c| c.is_ascii_digit())
        && hi.bytes().all(|c| c.is_ascii_digit())
}

#[cfg(test)]
#[path = "secret_tests.rs"]
mod tests;
