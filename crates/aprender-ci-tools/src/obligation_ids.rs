//! `obligation-ids`: deterministic `<PREFIX>-<TYPE>-<NNN>` ids for anonymous proof
//! obligations (was `scripts/lib/obligation_ids.py`). Same arguments, same stdout, same exit
//! codes, same bytes written.
//!
//! The original parsed YAML with PyYAML (YAML 1.1); this parses it with `serde_yaml` (YAML
//! 1.2). The two agree on every file the contracts tree holds, but not on every YAML file.
//! Where they could disagree about something this tool reads, the port refuses (exit 2,
//! nothing written) instead of guessing:
//! - a file that mentions `proof_obligations` and carries a tag (`!x`, `!!x`, `%` directive),
//!   one of the YAML 1.1 indicator scalars (`<<` merge, `=`, `!`, `&`, `*`) unquoted, a scalar
//!   PyYAML cannot construct (an impossible timestamp, `0b_`), a character PyYAML's reader
//!   rejects, or text `serde_yaml` cannot parse. A file that never mentions the key cannot be
//!   a candidate under either parser, so it is skipped as the original skipped it;
//! - an `id:`, `type:` or `kani_harnesses:` value the two versions type differently (`yes`,
//!   `0o7`, `2024-01-01`), or a number there;
//! - a non-ASCII contract stem or `type:` (Python's Unicode case mapping), a non-UTF-8 name.
//!
//! Where the original crashed (a truthy non-list `kani_harnesses:`, an unhashable `id:`, a
//! candidate with no `\nproof_obligations:\n` line, a failed write) the port exits 1 with
//! nothing on stdout, having written exactly the files the original had written by then.
//! `--help` prints the help argparse prints at its default 80 columns; the original rewraps it
//! to `COLUMNS` or the terminal width, the port does not.

use crate::pystr::{py_path_str, py_strip};
use regex::Regex;
use serde::de::{Deserialize, Deserializer, EnumAccess, MapAccess, SeqAccess, VariantAccess};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::path::Path;
use std::sync::LazyLock;

/// What the run printed and how it exits.
pub struct Outcome {
    pub stdout: String,
    pub stderr: String,
    pub code: u8,
}

impl Outcome {
    fn ok(stdout: String, code: u8) -> Self {
        Self {
            stdout,
            stderr: String::new(),
            code,
        }
    }

    fn stop(stop: Stop) -> Self {
        let (code, stderr) = match stop {
            Stop::Refuse(why) => (2, format!("obligation-ids: refusing: {why}")),
            Stop::Crash(why) => (1, format!("obligation-ids: {why}")),
        };
        Self {
            stdout: String::new(),
            stderr,
            code,
        }
    }
}

/// Why a run ends early: an input the port will not guess at (exit 2), or a point where
/// the original raised (exit 1).
#[derive(Debug, PartialEq)]
pub enum Stop {
    Refuse(String),
    Crash(String),
}

const USAGE: &str = "usage: obligation_ids.py [-h] [--check] [--selftest] [root]";
/// argparse's help at its default 80 columns (no `COLUMNS`, stdout not a terminal).
const HELP: &str = r#"usage: obligation_ids.py [-h] [--check] [--selftest] [root]

Deterministic ids for anonymous proof obligations. An obligation with no `id:`
cannot be cited — not by a kani harness, not by a test, not by a receipt, not
by a commit. 3,622 of them were anonymous, and `pv validate` reported `0
error(s)` for every one (#3314). THE RULE <PREFIX>-<TYPE>-<NNN> NNN the
obligation's 1-based position in proof_obligations TYPE from the obligation's
own `type:` field PREFIX the prefix this file's kani harnesses ALREADY cite,
if any; otherwise the initials of the contract's name parts. The initials rule
is not invented: it reproduces both prefixes a human chose in this tree
(gated-delta-net-v1 -> GDN, qwen35-hybrid-forward-v1 -> QHF). COLLISIONS. 111
initials-prefixes are claimed by more than one contract. That is cosmetic only
while nothing cites across files — and it stops being cosmetic the first time
something does. So the id space is made global on day one: when a base prefix
is claimed by more than one file, every claimant that is not already committed
to it by its own kani harnesses takes a 4-hex suffix derived from its own
path. Keying the suffix to the file's OWN path (not to the set of colliders)
is what makes it stable: adding a contract later can never renumber one that
already exists. NEVER RENAMES. A file whose obligations all carry ids is
skipped entirely — its convention is its own (52 such files use REG-OB-001,
PO-HEH-001, OBLIG-DATA-QUALITY-007-..., none of which this rule would
reproduce). Within a partially-named file, existing ids are untouched and a
computed id that would collide with one is bumped. IDEMPOTENT. Running twice
is a no-op: the second pass sees every file fully named and skips it.
`--check` asserts that and exits non-zero if not.

positional arguments:
  root

options:
  -h, --help  show this help message and exit
  --check     exit 1 if any anonymous obligation remains (idempotence gate)
  --selftest
"#;

const TYPE_CODE: &[(&str, &str)] = &[
    ("invariant", "INV"),
    ("equivalence", "EQ"),
    ("bound", "BND"),
    ("bounds", "BND"),
    ("postcondition", "POST"),
    ("precondition", "PRE"),
    ("monotonicity", "MON"),
    ("soundness", "SND"),
    ("completeness", "CMP"),
    ("determinism", "DET"),
    ("roundtrip", "RT"),
    ("conservation", "CON"),
    ("safety", "SAFE"),
    ("idempotency", "IDEM"),
    ("idempotence", "IDEM"),
    ("classification", "CLS"),
    ("ordering", "ORD"),
    ("frame", "FRM"),
    ("state_machine", "SM"),
    ("termination", "TERM"),
    ("liveness", "LIVE"),
    ("linearity", "LIN"),
    ("loop_invariant", "LINV"),
    ("loop_variant", "LVAR"),
    ("associativity", "ASSOC"),
    ("independence", "IND"),
    ("old_state", "OLD"),
    ("subcontract", "SUB"),
    ("symmetry", "SYM"),
    ("equality", "EQ"),
    ("purity", "PURE"),
    ("totality", "TOT"),
];
const UNTYPED: &str = "OBL";

/// Owned by another change; the original never touched them, but their ids still claim.
const EXCLUDE: [&str; 2] = [
    "contracts/qwen35-hybrid-forward-v1.yaml",
    "contracts/qwen35-e2e-verification-v1.yaml",
];

static ID_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Z][A-Z0-9]*-[A-Z]+-\d{3}$").expect("ID_RE is a valid regex"));
static ID_PREFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^([A-Z][A-Z0-9]*)-").expect("ID_PREFIX is a valid regex"));
static VERSION_PART: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^v[0-9]+$").expect("VERSION_PART is a valid regex"));
static NEXT_KEY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\n[A-Za-z_][A-Za-z0-9_]*:").expect("NEXT_KEY is a valid regex"));
static FIRST_ITEM: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^([ \t]*)- ").expect("FIRST_ITEM is a valid regex"));
static NEGATIVE_NUMBER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^-\d+$|^-\d*\.\d+$").expect("NEGATIVE_NUMBER is a valid regex"));
/// A tag or a directive in the raw text. `serde_yaml` reads `!!binary`, `!!set` and the
/// rest as untagged where PyYAML constructs (or rejects) them.
static TAG_TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)(?:^|[ \t\[{,])(?:!!|!<)|^%").expect("TAG_TOKEN is a valid regex")
});

// PyYAML 6's implicit resolvers (YAML 1.1), each anchored as `Resolver` anchors them.
static Y11_BOOL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new("^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)$")
        .expect("Y11_BOOL is a valid regex")
});
static Y11_NULL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("^(?:~|null|Null|NULL|)$").expect("Y11_NULL is a valid regex"));
static Y11_FLOAT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?",
        r"|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?",
        r"|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*",
        r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$",
    ))
    .expect("Y11_FLOAT is a valid regex")
});
static Y11_INT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"^(?:[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)",
        r"|[-+]?0x[0-9a-fA-F_]+|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$",
    ))
    .expect("Y11_INT is a valid regex")
});
static Y11_BAD_INT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[-+]?0[bx]_+$").expect("Y11_BAD_INT is a valid regex"));
static Y11_TIMESTAMP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"^(?:[0-9]{4}-[0-9]{2}-[0-9]{2}",
        r"|[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}(?:[Tt]|[ \t]+)[0-9]{1,2}:[0-9]{2}:[0-9]{2}",
        r"(?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9]{1,2}(?::[0-9]{2})?))?)$",
    ))
    .expect("Y11_TIMESTAMP is a valid regex")
});
static Y11_TIMESTAMP_PARTS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"^(?P<y>[0-9]{4})-(?P<mo>[0-9]{1,2})-(?P<d>[0-9]{1,2})",
        r"(?:(?:[Tt]|[ \t]+)(?P<h>[0-9]{1,2}):(?P<mi>[0-9]{2}):(?P<s>[0-9]{2})",
        r"(?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+](?P<th>[0-9]{1,2})(?::(?P<tm>[0-9]{2}))?))?)?$",
    ))
    .expect("Y11_TIMESTAMP_PARTS is a valid regex")
});

/// The YAML 1.1 indicator scalars PyYAML resolves to a merge, `value` or `yaml` tag and
/// `safe_load` then merges or rejects, each with the pattern of its UNQUOTED appearance.
static SPECIAL_SCALARS: LazyLock<Vec<(&'static str, Regex)>> = LazyLock::new(|| {
    ["<<", "=", "!", "&", "*"]
        .into_iter()
        .map(|tok| {
            let re = format!(
                r"(?m)(?:^|[ \t\[{{,?:-]){}(?:$|[ \t,\]}}:#])",
                regex::escape(tok)
            );
            (tok, Regex::new(&re).expect("special-scalar regex is valid"))
        })
        .collect()
});

/// A parsed YAML node, kept only as finely as this tool reads it.
#[derive(Clone, Debug, PartialEq)]
pub enum Y {
    Null,
    Bool(bool),
    Num,
    Str(String),
    Seq(Vec<Y>),
    Map(Vec<(Y, Y)>),
    /// A locally tagged node (`!x`); PyYAML's `safe_load` rejects it.
    Tagged,
}

impl Y {
    /// Python truthiness.
    fn truthy(&self) -> bool {
        match self {
            Y::Null => false,
            Y::Bool(b) => *b,
            Y::Num | Y::Tagged => true,
            Y::Str(s) => !s.is_empty(),
            Y::Seq(v) => !v.is_empty(),
            Y::Map(m) => !m.is_empty(),
        }
    }

    /// `dict.get(key)`: the last duplicate wins, as Python's mapping construction does.
    fn get(&self, key: &str) -> Option<&Y> {
        let Y::Map(entries) = self else {
            return None;
        };
        entries
            .iter()
            .rev()
            .find(|(k, _)| matches!(k, Y::Str(s) if s == key))
            .map(|(_, v)| v)
    }
}

struct YVisitor;

impl<'de> serde::de::Visitor<'de> for YVisitor {
    type Value = Y;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("any YAML node")
    }
    fn visit_bool<E>(self, b: bool) -> Result<Y, E> {
        Ok(Y::Bool(b))
    }
    fn visit_i64<E>(self, _: i64) -> Result<Y, E> {
        Ok(Y::Num)
    }
    fn visit_u64<E>(self, _: u64) -> Result<Y, E> {
        Ok(Y::Num)
    }
    fn visit_i128<E>(self, _: i128) -> Result<Y, E> {
        Ok(Y::Num)
    }
    fn visit_u128<E>(self, _: u128) -> Result<Y, E> {
        Ok(Y::Num)
    }
    fn visit_f64<E>(self, _: f64) -> Result<Y, E> {
        Ok(Y::Num)
    }
    fn visit_str<E>(self, s: &str) -> Result<Y, E> {
        Ok(Y::Str(s.to_owned()))
    }
    fn visit_string<E>(self, s: String) -> Result<Y, E> {
        Ok(Y::Str(s))
    }
    fn visit_unit<E>(self) -> Result<Y, E> {
        Ok(Y::Null)
    }
    fn visit_none<E>(self) -> Result<Y, E> {
        Ok(Y::Null)
    }
    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Y, D::Error> {
        Y::deserialize(d)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Y, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element::<Y>()? {
            items.push(item);
        }
        Ok(Y::Seq(items))
    }
    // Entry by entry, so a duplicate key is kept (and the last one read) rather than an error.
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Y, A::Error> {
        let mut entries = Vec::new();
        while let Some(entry) = map.next_entry::<Y, Y>()? {
            entries.push(entry);
        }
        Ok(Y::Map(entries))
    }
    fn visit_enum<A: EnumAccess<'de>>(self, data: A) -> Result<Y, A::Error> {
        let (_tag, node): (String, _) = data.variant()?;
        node.newtype_variant::<Y>()?;
        Ok(Y::Tagged)
    }
}

impl<'de> Deserialize<'de> for Y {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(YVisitor)
    }
}

/// How PyYAML would construct this text as a PLAIN scalar.
#[derive(Debug, PartialEq)]
enum Plain11 {
    Str,
    /// bool, null, int, float or a timestamp
    NotStr,
    /// matched a resolver whose constructor raises
    Invalid,
}

fn plain11(s: &str) -> Plain11 {
    if Y11_BOOL.is_match(s) || Y11_NULL.is_match(s) || Y11_FLOAT.is_match(s) {
        Plain11::NotStr
    } else if Y11_INT.is_match(s) {
        if Y11_BAD_INT.is_match(s) {
            Plain11::Invalid
        } else {
            Plain11::NotStr
        }
    } else if Y11_TIMESTAMP.is_match(s) {
        if timestamp_constructs(s) {
            Plain11::NotStr
        } else {
            Plain11::Invalid
        }
    } else {
        Plain11::Str
    }
}

/// Whether `datetime.date`/`datetime.datetime`/`datetime.timezone` accept the fields.
fn timestamp_constructs(s: &str) -> bool {
    let Some(c) = Y11_TIMESTAMP_PARTS.captures(s) else {
        return false;
    };
    let num = |name: &str| -> u32 {
        c.name(name)
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0)
    };
    let (year, month, day) = (num("y"), num("mo"), num("d"));
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    year >= 1
        && (1..=days).contains(&day)
        && num("h") <= 23
        && num("mi") <= 59
        && num("s") <= 59
        && num("th") * 60 + num("tm") < 1440
}

/// A character PyYAML's `Reader` rejects, or one of the line breaks and the BOM that the
/// two parsers' scanners do not provably treat alike.
fn reader_doubt(c: char) -> bool {
    let printable = matches!(c, '\t' | '\n' | '\r' | ' '..='~' | '\u{85}')
        || ('\u{a0}'..='\u{d7ff}').contains(&c)
        || ('\u{e000}'..='\u{fffd}').contains(&c)
        || c >= '\u{10000}';
    !printable || matches!(c, '\u{85}' | '\u{2028}' | '\u{2029}' | '\u{feff}')
}

#[derive(Default)]
struct Scan {
    tagged: bool,
    unhashable_key: bool,
    invalid_scalar: bool,
    special: BTreeSet<&'static str>,
}

fn scan(y: &Y, out: &mut Scan) {
    match y {
        Y::Tagged => out.tagged = true,
        Y::Str(s) => {
            if plain11(s) == Plain11::Invalid {
                out.invalid_scalar = true;
            }
            if let Some((tok, _)) = SPECIAL_SCALARS.iter().find(|(tok, _)| s == tok) {
                out.special.insert(tok);
            }
        }
        Y::Seq(items) => items.iter().for_each(|i| scan(i, out)),
        Y::Map(entries) => {
            for (k, v) in entries {
                if matches!(k, Y::Seq(_) | Y::Map(_)) {
                    out.unhashable_key = true;
                }
                scan(k, out);
                scan(v, out);
            }
        }
        Y::Null | Y::Bool(_) | Y::Num => {}
    }
}

/// Whether `tok` appears unquoted, once its quoted spellings are taken out.
fn appears_plain(text: &str, tok: &str, re: &Regex) -> bool {
    let unquoted = text
        .replace(&format!("'{tok}'"), "")
        .replace(&format!("\"{tok}\""), "");
    re.is_match(&unquoted)
}

/// A file as the original's `load()` saw it.
pub struct Loaded {
    /// The raw bytes, for the byte-identical-copy rule.
    pub raw: Vec<u8>,
    /// The text-mode read (universal newlines), which a rewrite edits.
    pub text: String,
    pub doc: Y,
}

impl Loaded {
    fn obligations(&self) -> &[Y] {
        match self.doc.get("proof_obligations") {
            Some(Y::Seq(obs)) => obs,
            _ => &[],
        }
    }
}

/// `load(path)`: `Ok(None)` wherever the original returned None.
pub fn load(path: &str) -> Result<Option<Loaded>, Stop> {
    let Ok(raw) = std::fs::read(path) else {
        return Ok(None);
    };
    let Ok(text) = std::str::from_utf8(&raw) else {
        return Ok(None);
    };
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let doubt = |why: &str| {
        if text.contains("proof_obligations") {
            Err(Stop::Refuse(format!(
                "{path}: {why}; PyYAML may read it differently"
            )))
        } else {
            Ok(None)
        }
    };
    if text.chars().any(reader_doubt) {
        return doubt("a control character, BOM or Unicode line break");
    }
    if TAG_TOKEN.is_match(&text) {
        return doubt("a tag or directive");
    }
    let Ok(doc) = serde_yaml::from_str::<Y>(&text) else {
        return doubt("serde_yaml cannot parse it");
    };
    let mut s = Scan::default();
    scan(&doc, &mut s);
    if s.tagged {
        return doubt("a tagged node");
    }
    if s.invalid_scalar {
        return doubt("a scalar YAML 1.1 cannot construct");
    }
    for (tok, re) in SPECIAL_SCALARS.iter() {
        if s.special.contains(tok) && appears_plain(&text, tok, re) {
            return doubt(&format!("an unquoted `{tok}`"));
        }
    }
    if s.unhashable_key {
        return Ok(None);
    }
    let shaped = matches!(&doc, Y::Map(_))
        && matches!(doc.get("proof_obligations"),
            Some(Y::Seq(obs)) if !obs.is_empty() && obs.iter().all(|o| matches!(o, Y::Map(_))));
    Ok(shaped.then(|| Loaded {
        raw: raw.clone(),
        text,
        doc,
    }))
}

/// `Path.stem`.
fn stem(rel: &str) -> &str {
    let name = rel.rsplit('/').next().unwrap_or(rel);
    match name.rfind('.') {
        Some(i) if i > 0 && i + 1 < name.len() => &name[..i],
        _ => name,
    }
}

/// `type_code(raw)`.
pub fn type_code(raw: Option<&Y>) -> Result<String, Stop> {
    let t = match raw {
        None | Some(Y::Null) => return Ok(UNTYPED.to_owned()),
        Some(Y::Bool(b)) => if *b { "true" } else { "false" }.to_owned(),
        Some(Y::Str(s)) if s.is_ascii() && (s.is_empty() || plain11(s) == Plain11::Str) => {
            py_strip(s).to_ascii_lowercase()
        }
        Some(other) => {
            return Err(Stop::Refuse(format!(
                "a `type:` the original may have read differently: {other:?}"
            )))
        }
    };
    if t.is_empty() {
        return Ok(UNTYPED.to_owned());
    }
    if let Some((_, code)) = TYPE_CODE.iter().find(|(name, _)| *name == t) {
        return Ok((*code).to_owned());
    }
    let letters: String = t
        .to_ascii_uppercase()
        .chars()
        .filter(char::is_ascii_uppercase)
        .take(4)
        .collect();
    Ok(if letters.is_empty() {
        UNTYPED.to_owned()
    } else {
        letters
    })
}

/// `initials(stem)`, for an ASCII stem.
pub fn initials(stem: &str) -> String {
    let out: String = stem
        .split(['-', '_', '.'])
        .filter(|p| !p.is_empty() && !VERSION_PART.is_match(p))
        .filter_map(|p| p.chars().next())
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_uppercase())
        .collect();
    let out = out.trim_start_matches(|c: char| !c.is_ascii_uppercase());
    if out.is_empty() {
        "OBL".to_owned()
    } else {
        out.to_owned()
    }
}

/// `cited_prefix(doc)`: the prefix this file's own kani harnesses already cite.
fn cited_prefix(doc: &Y) -> Result<Option<String>, Stop> {
    let harnesses = match doc.get("kani_harnesses") {
        None | Some(Y::Null | Y::Bool(false) | Y::Map(_)) => return Ok(None),
        Some(Y::Bool(true)) => {
            return Err(Stop::Crash(
                "TypeError: 'bool' object is not iterable (kani_harnesses: true)".to_owned(),
            ))
        }
        Some(Y::Str(s)) if s.is_empty() || plain11(s) == Plain11::Str => return Ok(None),
        Some(Y::Seq(items)) => items,
        Some(other) => {
            return Err(Stop::Refuse(format!(
                "a `kani_harnesses:` the original may have read differently: {other:?}"
            )))
        }
    };
    let mut seen: Vec<(String, usize)> = Vec::new();
    for k in harnesses {
        let Some(Y::Str(ob)) = k.get("obligation") else {
            continue;
        };
        let ob = py_strip(ob);
        if !ID_RE.is_match(ob) {
            continue;
        }
        let pre = ob.split('-').next().unwrap_or(ob);
        match seen.iter_mut().find(|(p, _)| p == pre) {
            Some((_, n)) => *n += 1,
            None => seen.push((pre.to_owned(), 1)),
        }
    }
    // `Counter.most_common(1)`: the highest count, the first seen on a tie.
    let best = seen.iter().map(|(_, n)| *n).max();
    Ok(best.and_then(|best| seen.into_iter().find(|(_, n)| *n == best).map(|(p, _)| p)))
}

/// `_id_prefix(value)`.
fn id_prefix(value: Option<&Y>) -> Option<String> {
    let Some(Y::Str(s)) = value else {
        return None;
    };
    ID_PREFIX.captures(py_strip(s)).map(|c| c[1].to_owned())
}

/// A file with at least one anonymous obligation.
pub struct Cand {
    pub loaded: Loaded,
    pub cited: Option<String>,
}

/// One walk of the tree, loaded as the original's `collect` and `_collect_claims` load it.
pub struct Corpus {
    /// Every candidate, by relative path.
    pub cands: BTreeMap<String, Cand>,
    /// prefix -> the files whose EXISTING ids claim it (non-excluded files only).
    fixed: BTreeMap<String, BTreeSet<String>>,
    /// The excluded paths the walk found, loaded only when prefixes are assigned.
    excluded: Vec<String>,
}

/// `root.rglob("*.yaml")` in `sorted()` order: no symlinked directory is entered.
pub fn walk(root: &str) -> Result<Vec<String>, Stop> {
    fn visit(dir: &str, prefix: &str, out: &mut Vec<String>) -> Result<(), Stop> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Ok(());
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                return Err(Stop::Refuse(format!(
                    "a non-UTF-8 name under {dir}: {}",
                    entry.path().display()
                )));
            };
            let path = format!("{prefix}{name}");
            let is_dir = entry.file_type().is_ok_and(|t| t.is_dir());
            if name.ends_with(".yaml") {
                out.push(path.clone());
            }
            if is_dir {
                visit(&path, &format!("{path}/"), out)?;
            }
        }
        Ok(())
    }
    let prefix = if root == "." {
        String::new()
    } else if root.ends_with('/') {
        root.to_owned()
    } else {
        format!("{root}/")
    };
    let mut out = Vec::new();
    visit(root, &prefix, &mut out)?;
    out.sort_by(|a, b| a.split('/').cmp(b.split('/')));
    Ok(out)
}

/// An `id:` YAML 1.1 types differently from YAML 1.2 (or a number either way).
fn refuse_ambiguous_id(rel: &str, id: Option<&Y>) -> Result<(), Stop> {
    let ambiguous = match id {
        Some(Y::Num) => true,
        Some(Y::Str(s)) => !s.is_empty() && plain11(s) != Plain11::Str,
        _ => false,
    };
    match id {
        Some(y) if ambiguous => Err(Stop::Refuse(format!(
            "{rel}: an `id:` the original may have read differently: {y:?}"
        ))),
        _ => Ok(()),
    }
}

/// `collect(root)`, plus the existing-id claims `_collect_claims` makes from the same files.
pub fn collect(root: &str) -> Result<Corpus, Stop> {
    let mut corpus = Corpus {
        cands: BTreeMap::new(),
        fixed: BTreeMap::new(),
        excluded: Vec::new(),
    };
    for rel in walk(root)? {
        if EXCLUDE.contains(&rel.as_str()) {
            corpus.excluded.push(rel);
            continue;
        }
        let Some(loaded) = load(&rel)? else {
            continue;
        };
        let mut named = true;
        for o in loaded.obligations() {
            let id = o.get("id");
            refuse_ambiguous_id(&rel, id)?;
            named &= id.is_some_and(Y::truthy);
            if let Some(pre) = id_prefix(id) {
                corpus.fixed.entry(pre).or_default().insert(rel.clone());
            }
        }
        if named {
            continue;
        }
        let cited = cited_prefix(&loaded.doc)?;
        corpus.cands.insert(rel, Cand { loaded, cited });
    }
    Ok(corpus)
}

fn sha_suffix(me: &str) -> String {
    let hex: String = Sha256::digest(me.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    hex[..4].to_ascii_uppercase()
}

/// `assign_prefixes(cands, root)`: relpath -> final prefix.
pub fn assign_prefixes(corpus: &Corpus) -> Result<BTreeMap<String, String>, Stop> {
    let mut fixed = corpus.fixed.clone();
    for rel in &corpus.excluded {
        if let Some(loaded) = load(rel)? {
            for o in loaded.obligations() {
                if let Some(pre) = id_prefix(o.get("id")) {
                    fixed.entry(pre).or_default().insert(rel.clone());
                }
            }
        }
    }
    let mut base = BTreeMap::new();
    let mut movable: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (rel, c) in &corpus.cands {
        let b = match &c.cited {
            Some(p) => p.clone(),
            None if stem(rel).is_ascii() => initials(stem(rel)),
            None => return Err(Stop::Refuse(format!("{rel}: a non-ASCII contract stem"))),
        };
        movable.entry(b.clone()).or_default().insert(rel.clone());
        base.insert(rel.clone(), b);
    }
    let mut by_bytes: HashMap<&[u8], Vec<&str>> = HashMap::new();
    for (rel, c) in &corpus.cands {
        by_bytes.entry(&c.loaded.raw).or_default().push(rel);
    }
    let mut canon: HashMap<&str, &str> = HashMap::new();
    for rels in by_bytes.values() {
        let first = rels.iter().min().copied().unwrap_or_default();
        for r in rels {
            canon.insert(r, first);
        }
    }
    let canon_of = |r: &str| canon.get(r).map_or(r.to_owned(), |c| (*c).to_owned());
    let mut out = BTreeMap::new();
    for (rel, b) in base {
        let pool = fixed
            .get(&b)
            .filter(|s| !s.is_empty())
            .or_else(|| movable.get(&b));
        let owner = pool.and_then(|p| p.iter().map(|r| canon_of(r)).min());
        let me = canon_of(&rel);
        let prefix = if owner.as_deref() == Some(me.as_str()) {
            b
        } else {
            format!("{b}{}", sha_suffix(&me))
        };
        out.insert(rel, prefix);
    }
    Ok(out)
}

/// `plan_file(obs, prefix)`: (1-based position, id) for the anonymous rows.
pub fn plan_file(obs: &[Y], prefix: &str) -> Result<Vec<(usize, String)>, Stop> {
    let mut taken: BTreeSet<String> = BTreeSet::new();
    for o in obs {
        match o.get("id") {
            Some(Y::Str(s)) if !s.is_empty() => {
                taken.insert(s.clone());
            }
            Some(y @ (Y::Seq(_) | Y::Map(_))) if y.truthy() => {
                return Err(Stop::Crash(
                    "TypeError: unhashable type (a list or mapping `id:`)".to_owned(),
                ))
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    for (i, o) in obs.iter().enumerate() {
        let i = i + 1;
        if o.get("id").is_some_and(Y::truthy) {
            continue;
        }
        let code = type_code(o.get("type"))?;
        let mut n = i;
        let mut cand = format!("{prefix}-{code}-{n:03}");
        while taken.contains(&cand) {
            n += 1;
            cand = format!("{prefix}-{code}-{n:03}");
        }
        taken.insert(cand.clone());
        out.push((i, cand));
    }
    Ok(out)
}

/// `str.splitlines(keepends=True)`.
fn splitlines_keepends(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        let end = match c {
            '\r' if it.peek().is_some_and(|&(_, n)| n == '\n') => {
                it.next();
                i + 2
            }
            '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}'
            | '\u{2029}' => i + c.len_utf8(),
            _ => continue,
        };
        out.push(&s[start..end]);
        start = end;
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// `rewrite(text, plan)`: `- id: X` inserted as the first key of each planned item.
pub fn rewrite(text: &str, plan: &[(usize, String)]) -> Result<String, Stop> {
    if plan.is_empty() {
        return Ok(text.to_owned());
    }
    let start = text.find("\nproof_obligations:\n").ok_or_else(|| {
        Stop::Crash("ValueError: substring not found (no `\\nproof_obligations:\\n`)".to_owned())
    })? + 1;
    let body = start + "proof_obligations:\n".len();
    let end = NEXT_KEY
        .find(&text[body..])
        .map_or(text.len(), |m| body + m.start() + 1);
    let (block, tail) = (&text[body..end], &text[end..]);
    let Some(im) = FIRST_ITEM.captures(block) else {
        return Ok(text.to_owned());
    };
    let indent = im.get(1).map_or("", |m| m.as_str());
    let item = format!("{indent}- ");
    let mut items: Vec<Vec<String>> = Vec::new();
    for line in splitlines_keepends(block) {
        match items.last_mut() {
            Some(group) if !line.starts_with(&item) => group.push(line.to_owned()),
            _ => items.push(vec![line.to_owned()]),
        }
    }
    let mut pos = 0;
    for it in &mut items {
        if !it[0].starts_with(&item) {
            continue;
        }
        pos += 1;
        if let Some((_, id)) = plan.iter().find(|(p, _)| *p == pos) {
            it[0] = format!("{indent}- id: {id}\n{indent}  {}", &it[0][item.len()..]);
        }
    }
    let mut out = text[..body].to_owned();
    items.iter().flatten().for_each(|l| out.push_str(l));
    out.push_str(tail);
    Ok(out)
}

fn check(corpus: &Corpus) -> Outcome {
    if corpus.cands.is_empty() {
        return Outcome::ok("OK: every proof obligation carries an id\n".to_owned(), 0);
    }
    let mut out = format!(
        "FAIL: {} file(s) still carry an anonymous obligation\n",
        corpus.cands.len()
    );
    for rel in corpus.cands.keys().take(10) {
        out.push_str(&format!("  {rel}\n"));
    }
    Outcome::ok(out, 1)
}

/// Write mode. Every edit is computed first, in the original's order, so a refusal writes
/// nothing; a crash still writes the files the original wrote before it raised.
fn name_all(corpus: &Corpus) -> Result<String, Stop> {
    let prefixes = assign_prefixes(corpus)?;
    let mut edits = Vec::new();
    let mut crash = None;
    for (rel, c) in &corpus.cands {
        let step = plan_file(c.loaded.obligations(), &prefixes[rel]).and_then(|plan| {
            let text = rewrite(&c.loaded.text, &plan)?;
            Ok((plan.len(), text))
        });
        match step {
            Ok((n, text)) => edits.push((rel, n, text)),
            Err(Stop::Crash(why)) => {
                crash = Some(why);
                break;
            }
            Err(refuse) => return Err(refuse),
        }
    }
    let (mut files, mut named) = (0, 0);
    for (rel, n, text) in edits {
        if n == 0 {
            continue;
        }
        std::fs::write(rel, text).map_err(|e| Stop::Crash(format!("{rel}: {e}")))?;
        files += 1;
        named += n;
    }
    if let Some(why) = crash {
        return Err(Stop::Crash(why));
    }
    Ok(format!(
        "named {named} obligation(s) across {files} file(s)\n"
    ))
}

/// The parsed command line.
#[derive(Debug, PartialEq)]
pub struct Args {
    pub root: String,
    pub check: bool,
    pub selftest: bool,
}

/// How argparse reads one argument.
enum Token {
    Positional,
    Unknown,
    Dashes,
    Check,
    Selftest,
}

fn usage_error(msg: &str) -> (u8, String) {
    (2, format!("{USAGE}\nobligation_ids.py: error: {msg}"))
}

fn help() -> (u8, String) {
    (0, HELP.to_owned())
}

/// One `--long[=value]` argument: an exact name, or the unique option it abbreviates.
fn long_option(a: &str) -> Result<Token, (u8, String)> {
    const LONG: [&str; 3] = ["--check", "--selftest", "--help"];
    let (name, explicit) = match a.split_once('=') {
        Some((n, v)) => (n, Some(v)),
        None => (a, None),
    };
    let hits: Vec<&str> = if LONG.contains(&name) {
        vec![name]
    } else {
        LONG.into_iter().filter(|o| o.starts_with(name)).collect()
    };
    match (hits.as_slice(), explicit) {
        ([opt], Some(v)) => Err(usage_error(&format!(
            "argument {}: ignored explicit argument '{v}'",
            if *opt == "--help" { "-h/--help" } else { opt }
        ))),
        (["--check"], None) => Ok(Token::Check),
        (["--selftest"], None) => Ok(Token::Selftest),
        ([_], None) => Err(help()),
        ([], _) if a.contains(' ') => Ok(Token::Positional),
        ([], _) => Ok(Token::Unknown),
        (many, _) => Err(usage_error(&format!(
            "ambiguous option: {name} could match {}",
            many.join(", ")
        ))),
    }
}

/// One argument before any `--`.
fn token(a: &str) -> Result<Token, (u8, String)> {
    if !a.starts_with('-') || a == "-" {
        Ok(Token::Positional)
    } else if a == "--" {
        Ok(Token::Dashes)
    } else if let Some(v) = a.strip_prefix("-h=") {
        Err(usage_error(&format!(
            "argument -h/--help: ignored explicit argument '{v}'"
        )))
    } else if a.starts_with("-h") {
        Err(help())
    } else if a.starts_with("--") {
        long_option(a)
    } else if NEGATIVE_NUMBER.is_match(a) || a.contains(' ') {
        Ok(Token::Positional)
    } else {
        Ok(Token::Unknown)
    }
}

/// argparse's reading of `[root] [--check] [--selftest]`: `Err((code, text))` for help
/// (0, stdout) or a usage error (2, stderr).
pub fn parse_args(argv: &[String]) -> Result<Args, (u8, String)> {
    let (mut check, mut selftest) = (false, false);
    let mut positional: Vec<&String> = Vec::new();
    let mut unknown: Vec<&String> = Vec::new();
    let mut after_dashes = false;
    for a in argv {
        let t = if after_dashes {
            Token::Positional
        } else {
            token(a)?
        };
        match t {
            Token::Positional => positional.push(a),
            Token::Unknown => unknown.push(a),
            Token::Dashes => after_dashes = true,
            Token::Check => check = true,
            Token::Selftest => selftest = true,
        }
    }
    let extra: Vec<&str> = positional
        .iter()
        .skip(1)
        .chain(unknown.iter())
        .map(|s| s.as_str())
        .collect();
    if !extra.is_empty() {
        return Err(usage_error(&format!(
            "unrecognized arguments: {}",
            extra.join(" ")
        )));
    }
    let root = positional
        .first()
        .map_or_else(|| "contracts".to_owned(), |r| (*r).clone());
    Ok(Args {
        root,
        check,
        selftest,
    })
}

/// The subcommand: `argv` is everything after `obligation-ids`.
pub fn run(argv: &[String]) -> Outcome {
    let args = match parse_args(argv) {
        Ok(args) => args,
        Err((0, help)) => return Outcome::ok(help, 0),
        Err((code, msg)) => {
            return Outcome {
                stdout: String::new(),
                stderr: msg,
                code,
            }
        }
    };
    if args.selftest {
        return selftest::run();
    }
    let root = py_path_str(Path::new(&args.root));
    let corpus = match collect(&root) {
        Ok(c) => c,
        Err(stop) => return Outcome::stop(stop),
    };
    if args.check {
        return check(&corpus);
    }
    match name_all(&corpus) {
        Ok(out) => Outcome::ok(out, 0),
        Err(stop) => Outcome::stop(stop),
    }
}

/// The original's `--selftest`, case for case and label for label.
mod selftest {
    use super::{assign_prefixes, collect, plan_file, rewrite, Outcome, Stop};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    type Case = (&'static str, &'static str, &'static [&'static str], bool);

    const CASES: [Case; 6] = [
        (
            "initials + type codes",
            "proof_obligations:\n- type: invariant\n  property: one\n- type: bound\n  property: two\n",
            &["ABC-INV-001", "ABC-BND-002"],
            false,
        ),
        (
            "untyped obligation",
            "proof_obligations:\n- property: no type here\n",
            &["ABC-OBL-001"],
            false,
        ),
        (
            "kani-cited prefix wins over initials",
            "proof_obligations:\n- type: invariant\n  property: one\nkani_harnesses:\n- id: K1\n  obligation: ZZZ-INV-001\n",
            &["ZZZ-INV-001"],
            false,
        ),
        (
            "existing id is never shadowed",
            "proof_obligations:\n- id: ABC-INV-001\n  type: invariant\n  property: kept\n- type: invariant\n  property: anonymous\n",
            &["ABC-INV-002"],
            false,
        ),
        (
            "fully named file is skipped",
            "proof_obligations:\n- id: REG-OB-001\n  type: invariant\n  property: kept\n",
            &[],
            true,
        ),
        (
            "INDENTED list items are named too",
            "proof_obligations:\n  - type: invariant\n    property: one\n  - type: bound\n    property: two\n",
            &["ABC-INV-001", "ABC-BND-002"],
            false,
        ),
    ];

    const FLUSH: &str =
        "kind: K\nproof_obligations:\n- type: invariant\n  property: one\nnext_key: v\n";
    const INDENTED: &str =
        "kind: K\nproof_obligations:\n  - type: invariant\n    property: one\nnext_key: v\n";
    const NAMED: &str =
        "kind: K\nproof_obligations:\n- id: ABC-INV-001\n  type: invariant\n  property: one\nnext_key: v\n";

    fn report(out: &mut String, ok: bool, label: &str, detail: &str) -> u8 {
        if ok {
            out.push_str(&format!("PASS {label}\n"));
            0
        } else {
            out.push_str(&format!("FAIL {label}: {detail}\n"));
            1
        }
    }

    /// A fresh directory, removed again when dropped.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Result<Self, Stop> {
            static N: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "obligation-ids-selftest-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir)
                .map_err(|e| Stop::Crash(format!("{}: {e}", dir.display())))?;
            Ok(Self(dir))
        }

        fn write(&self, rel: &str, body: &str) -> Result<(), Stop> {
            let path = self.0.join(rel);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| Stop::Crash(format!("{}: {e}", parent.display())))?;
            }
            std::fs::write(&path, body).map_err(|e| Stop::Crash(format!("{}: {e}", path.display())))
        }

        fn root(&self) -> String {
            self.0.to_string_lossy().into_owned()
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn name_of(rel: &str) -> &str {
        Path::new(rel)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(rel)
    }

    fn run_case(body: &str, want: &[&str], skip: bool) -> Result<(bool, String), Stop> {
        let d = TempDir::new()?;
        d.write("a-b-c-v1.yaml", body)?;
        let corpus = collect(&d.root())?;
        if skip {
            return Ok(if corpus.cands.is_empty() {
                (true, String::new())
            } else {
                (false, "expected the file to be SKIPPED".to_owned())
            });
        }
        let Some((rel, cand)) = corpus.cands.iter().next() else {
            return Ok((
                false,
                "file was skipped but should have been named".to_owned(),
            ));
        };
        let prefixes = assign_prefixes(&corpus)?;
        let got: Vec<String> = plan_file(cand.loaded.obligations(), &prefixes[rel])?
            .into_iter()
            .map(|(_, id)| id)
            .collect();
        Ok((got == want, format!("got {got:?} want {want:?}")))
    }

    fn cases(out: &mut String) -> Result<u8, Stop> {
        let mut rc = 0;
        for (name, body, want, skip) in CASES {
            let (ok, why) = run_case(body, want, skip)?;
            rc |= report(out, ok, name, &why);
        }
        Ok(rc)
    }

    fn collisions(out: &mut String) -> Result<u8, Stop> {
        let mut rc = 0;
        let d = TempDir::new()?;
        d.write(
            "alpha-beta-v1.yaml",
            "proof_obligations:\n- type: invariant\n  property: one\n",
        )?;
        d.write(
            "apple-banana-v1.yaml",
            "proof_obligations:\n- type: invariant\n  property: two\n",
        )?;
        d.write(
            "zeta-zulu-v1.yaml",
            "proof_obligations:\n- id: ZZ-INV-001\n  type: invariant\n  property: owned\n",
        )?;
        d.write(
            "zebra-zoo-v1.yaml",
            "proof_obligations:\n- type: invariant\n  property: three\n",
        )?;
        let corpus = collect(&d.root())?;
        let pre = assign_prefixes(&corpus)?;
        let by_name = |n: &str| {
            pre.iter()
                .find(|(k, _)| name_of(k) == n)
                .map(|(_, v)| v.clone())
        };
        let ab: Vec<String> = ["alpha-beta-v1.yaml", "apple-banana-v1.yaml"]
            .iter()
            .filter_map(|n| by_name(n))
            .collect();
        let distinct = ab.len() == 2 && ab[0] != ab[1];
        rc |= report(
            out,
            distinct,
            "collision/two candidates get distinct prefixes",
            &format!("got {ab:?}"),
        );
        let zz = by_name("zebra-zoo-v1.yaml").unwrap_or_default();
        rc |= report(
            out,
            zz != "ZZ",
            "collision/a skipped file still owns its prefix",
            &format!("candidate took {zz:?}, which the named file owns"),
        );

        let same = "proof_obligations:\n- type: invariant\n  property: twin\n";
        d.write("twin-v1.yaml", same)?;
        d.write("sub/twin-v1.yaml", same)?;
        let corpus2 = collect(&d.root())?;
        let pre2 = assign_prefixes(&corpus2)?;
        let twins: Vec<&String> = pre2
            .iter()
            .filter(|(k, _)| k.ends_with("twin-v1.yaml"))
            .map(|(_, v)| v)
            .collect();
        let one = !twins.is_empty() && twins.iter().all(|v| *v == twins[0]);
        rc |= report(
            out,
            one,
            "collision/byte-identical duplicate-stem copies share one prefix",
            &format!("got {twins:?}"),
        );

        let mut ids = Vec::new();
        for (rel, c) in &corpus.cands {
            ids.extend(
                plan_file(c.loaded.obligations(), &pre[rel])?
                    .into_iter()
                    .map(|(_, id)| id),
            );
        }
        let unique: std::collections::BTreeSet<&String> = ids.iter().collect();
        rc |= report(
            out,
            unique.len() == ids.len(),
            "collision/no id is minted twice",
            &format!("got {ids:?}"),
        );
        Ok(rc)
    }

    fn rewrites(out: &mut String) -> Result<u8, Stop> {
        let one = vec![(1, "ABC-INV-001".to_owned())];
        let table: [(&str, &str, Vec<(usize, String)>, String); 3] = [
            (
                "flush list",
                FLUSH,
                one.clone(),
                FLUSH.replace("- type:", "- id: ABC-INV-001\n  type:"),
            ),
            (
                "indented list",
                INDENTED,
                one,
                INDENTED.replace("  - type:", "  - id: ABC-INV-001\n    type:"),
            ),
            (
                "no-op when nothing to name",
                NAMED,
                Vec::new(),
                NAMED.to_owned(),
            ),
        ];
        let mut rc = 0;
        for (name, src, plan, want) in table {
            let got = rewrite(src, &plan)?;
            rc |= report(
                out,
                got == want,
                &format!("rewrite/{name}"),
                &format!("{got:?}"),
            );
        }
        Ok(rc)
    }

    pub(super) fn run() -> Outcome {
        let mut out = String::new();
        let rc = cases(&mut out).and_then(|a| {
            let b = collisions(&mut out)?;
            let c = rewrites(&mut out)?;
            Ok(a | b | c)
        });
        match rc {
            Ok(rc) => {
                out.push_str(if rc == 0 {
                    "selftest: PASS\n"
                } else {
                    "selftest: FAIL\n"
                });
                Outcome::ok(out, rc)
            }
            Err(stop) => Outcome::stop(stop),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| (*s).to_owned()).collect()
    }

    fn parsed(a: &[&str]) -> Result<(String, bool, bool), u8> {
        parse_args(&argv(a))
            .map(|x| (x.root, x.check, x.selftest))
            .map_err(|(code, _)| code)
    }

    #[test]
    fn argparse_table_matches_python_313() {
        let ok = |r: &str, c, s| Ok((r.to_owned(), c, s));
        assert_eq!(parsed(&[]), ok("contracts", false, false));
        assert_eq!(parsed(&["--check", "a"]), ok("a", true, false));
        assert_eq!(parsed(&["a", "--check"]), ok("a", true, false));
        assert_eq!(parsed(&["a", "b"]), Err(2));
        assert_eq!(parsed(&["--", "--check"]), ok("--check", false, false));
        assert_eq!(parsed(&["--", "a", "--"]), Err(2));
        assert_eq!(parsed(&["-5"]), ok("-5", false, false));
        assert_eq!(parsed(&["-1.5"]), ok("-1.5", false, false));
        assert_eq!(parsed(&["-a b"]), ok("-a b", false, false));
        assert_eq!(parsed(&["--ch"]), ok("contracts", true, false));
        assert_eq!(parsed(&["--c=1"]), Err(2));
        assert_eq!(parsed(&["-hx"]), Err(0));
        assert_eq!(parsed(&["-h=x"]), Err(2));
        assert_eq!(parsed(&["--help=1"]), Err(2));
        assert_eq!(
            parse_args(&argv(&["-h"])).map_err(|(_, t)| t.len()),
            Err(1962)
        );
        assert_eq!(parsed(&["--bogus", "-h"]), Err(0));
        assert_eq!(parsed(&["--check=1", "-h"]), Err(2));
        assert_eq!(parsed(&["-"]), ok("-", false, false));
        assert_eq!(parsed(&[""]), ok("", false, false));
        assert_eq!(parsed(&["-x"]), Err(2));
        assert_eq!(parsed(&["--he"]), Err(0));
        assert_eq!(
            parsed(&["--selftest", "--check"]),
            ok("contracts", true, true)
        );
    }

    #[test]
    fn plain11_types_as_pyyaml_does() {
        for s in [
            "yes", "Off", "~", "", "1.5", ".inf", "07", "1_000", "0x1F", "1:30",
        ] {
            assert_ne!(plain11(s), Plain11::Str, "{s}");
        }
        assert_eq!(plain11("0o7"), Plain11::Str);
        assert_eq!(plain11("0o7x"), Plain11::Str);
        assert_eq!(plain11("1e5"), Plain11::Str);
        assert_eq!(plain11("ABC-INV-001"), Plain11::Str);
        assert_eq!(plain11("2024-02-29"), Plain11::NotStr);
        assert_eq!(plain11("2023-02-29"), Plain11::Invalid);
        assert_eq!(plain11("2024-1-1 24:00:00"), Plain11::Invalid);
        assert_eq!(plain11("2024-1-1 23:00:00 +24"), Plain11::Invalid);
        assert_eq!(plain11("0b_"), Plain11::Invalid);
    }

    #[test]
    fn type_codes_and_initials() {
        let s = |v: &str| Some(Y::Str(v.to_owned()));
        assert_eq!(type_code(None), Ok("OBL".to_owned()));
        assert_eq!(type_code(s(" Invariant ").as_ref()), Ok("INV".to_owned()));
        assert_eq!(type_code(s("weird-kind 9").as_ref()), Ok("WEIR".to_owned()));
        assert_eq!(type_code(Some(&Y::Bool(false))), Ok("FALS".to_owned()));
        assert!(matches!(type_code(s("yes").as_ref()), Err(Stop::Refuse(_))));
        assert_eq!(initials("gated-delta-net-v1"), "GDN");
        assert_eq!(initials("qwen35-hybrid-forward-v1"), "QHF");
        assert_eq!(initials("9-lives"), "L");
        assert_eq!(initials("v1"), "OBL");
    }

    #[test]
    fn rewrite_needs_the_key_on_its_own_line() {
        let plan = vec![(1, "A-INV-001".to_owned())];
        assert!(matches!(
            rewrite("proof_obligations:\n- type: x\n", &plan),
            Err(Stop::Crash(_))
        ));
        assert_eq!(
            rewrite("k: 1\nproof_obligations:\n- a: 1\n\x0c- b\n", &plan),
            Ok("k: 1\nproof_obligations:\n- id: A-INV-001\n  a: 1\n\x0c- b\n".to_owned())
        );
    }

    #[test]
    fn splitlines_keepends_matches_python() {
        assert_eq!(
            splitlines_keepends("a\r\nb\rc\x0bd\u{2028}e"),
            ["a\r\n", "b\r", "c\x0b", "d\u{2028}", "e"]
        );
        assert!(splitlines_keepends("").is_empty());
    }

    #[test]
    fn stem_is_pathlib_stem() {
        assert_eq!(stem("x/a.yaml"), "a");
        assert_eq!(stem(".yaml"), ".yaml");
        assert_eq!(stem("x..yaml"), "x.");
    }

    #[test]
    fn selftest_passes() {
        let o = selftest::run();
        assert_eq!(o.code, 0, "{}{}", o.stdout, o.stderr);
        assert!(o.stdout.ends_with("selftest: PASS\n"));
    }
}
