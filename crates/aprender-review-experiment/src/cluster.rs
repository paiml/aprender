//! Near-duplicate cluster check for sealed test items (PRA-001 T7, gate G-CON;
//! contract `review-corpus-contamination-v1`, FALSIFY-RCC-004..006).
//!
//! The exact checks in [`crate::contamination`] (diff sha, hunk sha) miss a
//! sealed diff that was re-indented or had an identifier renamed. This module
//! catches it with a bottom-k MinHash sketch over normalised token shingles:
//!
//! - **whitespace**: tokens are taken with all whitespace dropped, and blank
//!   lines produce no tokens, so re-indenting or re-spacing changes nothing;
//! - **renames**: every identifier becomes `$` (the MOSS normalisation), so any
//!   rename, even a partial one, yields the same token stream; keywords and
//!   literals are kept, so `x += 1` and `x += 2` still differ.
//!
//! A sealed item's sketch is the `SKETCH` smallest shingle hashes. A text leaks
//! the item when at least `CONTAINMENT` of that sketch is among the text's own
//! shingle hashes: an unbiased estimate of how much of the item the text holds,
//! so a sealed diff embedded in a longer prompt is still seen.

use std::collections::HashSet;

/// Tokens per shingle.
pub const SHINGLE: usize = 12;
/// Hashes kept per sealed item (bottom-k).
pub const SKETCH: usize = 64;
/// Items with fewer distinct shingles are too generic to cluster on.
pub const MIN_SHINGLES: usize = 32;
/// Fraction of a sealed sketch that must appear in a text for a hit.
pub const CONTAINMENT: f64 = 0.7;
/// Header line of a rendered sketch file.
pub const SKETCH_SCHEME: &str = "review-corpus-v1 cluster-sketch shingle=12 k=64";

const HEADERS: &[&str] = &["diff --git", "index ", "--- ", "+++ ", "@@"];

const KEYWORDS: &[&str] = &[
    "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern",
    "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub",
    "ref", "return", "self", "Self", "static", "struct", "super", "trait", "true", "type",
    "unsafe", "use", "where", "while",
];

/// Normalised tokens of one hunk body (or any text): whitespace dropped, a
/// `+`/`-` line marker kept as a token, identifiers replaced by `$`.
#[must_use]
pub fn tokens(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in body.lines() {
        let (marker, rest) = match line.chars().next() {
            Some(c @ ('+' | '-')) => (Some(c), &line[1..]),
            Some(' ') => (None, &line[1..]),
            _ => (None, line),
        };
        if rest.trim().is_empty() {
            continue;
        }
        out.extend(marker.map(String::from));
        let mut chars = rest.chars().peekable();
        while let Some(c) = chars.next() {
            if c.is_whitespace() {
                continue;
            }
            if c.is_alphanumeric() || c == '_' {
                let mut w = String::from(c);
                while let Some(&n) = chars.peek() {
                    if n.is_alphanumeric() || n == '_' {
                        w.push(n);
                        chars.next();
                    } else {
                        break;
                    }
                }
                out.push(ident(w));
            } else {
                out.push(c.to_string());
            }
        }
    }
    out
}

fn ident(w: String) -> String {
    let lead = w.chars().next().unwrap_or('0');
    if lead.is_ascii_digit() || KEYWORDS.contains(&w.as_str()) {
        w
    } else {
        "$".into()
    }
}

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn hash(shingle: &[String]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for t in shingle {
        for b in t.bytes().chain([0xff]) {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    mix(h)
}

/// Distinct shingle hashes of a text. Diff header lines (`diff --git`, `index`,
/// `---`, `+++`, `@@`) are skipped, so a re-based or re-pathed copy matches;
/// every other line is tokenised, so a copy whose line markers were mangled (a
/// context space eaten by re-indenting) or that sits inside a prompt is seen.
#[must_use]
pub fn shingles(text: &str) -> HashSet<u64> {
    let body: Vec<&str> = text
        .lines()
        .filter(|l| !HEADERS.iter().any(|h| l.starts_with(h)))
        .collect();
    tokens(&body.join("\n"))
        .windows(SHINGLE)
        .map(hash)
        .collect()
}

/// Bottom-k sketch of a sealed diff, ascending; `None` below `MIN_SHINGLES`.
#[must_use]
pub fn sketch(diff: &str) -> Option<Vec<u64>> {
    let mut s: Vec<u64> = shingles(diff).into_iter().collect();
    if s.len() < MIN_SHINGLES {
        return None;
    }
    s.sort_unstable();
    s.truncate(SKETCH);
    Some(s)
}

/// Fraction of `sketch` present in `set` (0 for an empty sketch).
#[must_use]
pub fn containment(sketch: &[u64], set: &HashSet<u64>) -> f64 {
    if sketch.is_empty() {
        return 0.0;
    }
    let n = sketch.iter().filter(|h| set.contains(h)).count();
    n as f64 / sketch.len() as f64
}

/// Render sealed sketches: the scheme line, then `id hex,hex,…` sorted by id.
#[must_use]
pub fn render_sketches(items: &[(String, Vec<u64>)]) -> String {
    let mut rows: Vec<&(String, Vec<u64>)> = items.iter().collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = format!("{SKETCH_SCHEME}\n");
    for (id, s) in rows {
        let hex: Vec<String> = s.iter().map(|h| format!("{h:016x}")).collect();
        out.push_str(&format!("{id} {}\n", hex.join(",")));
    }
    out
}

/// Parse a sketch file. `None` if the scheme line is missing or any row is
/// malformed: a sketch file that cannot be read clusters nothing.
#[must_use]
pub fn parse_sketches(text: &str) -> Option<Vec<(String, Vec<u64>)>> {
    let mut lines = text.lines();
    if lines.next()? != SKETCH_SCHEME {
        return None;
    }
    lines
        .filter(|l| !l.is_empty())
        .map(|l| {
            let (id, hex) = l.split_once(' ')?;
            let s = hex
                .split(',')
                .map(|h| u64::from_str_radix(h, 16).ok())
                .collect::<Option<Vec<u64>>>()?;
            Some((id.to_string(), s))
        })
        .collect()
}

#[cfg(test)]
#[path = "cluster_tests.rs"]
mod tests;
