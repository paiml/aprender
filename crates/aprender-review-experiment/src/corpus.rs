//! REX-02 review corpus (contract `review-corpus-v1`, spec §2.2).
//!
//! Pure functions only: classifying paths, sizing and stratifying diffs,
//! sanitising planted-mutation diffs, locating the defect lines, assigning the
//! seeded dev/test split and rendering the sealed test manifest. The builder
//! that talks to git and gh lives in `examples/rex.rs`.

use crate::stats::SplitMix64;
use serde::{Deserialize, Serialize};

/// Corpus scheme id; also the first line of the sealed manifest.
pub const SCHEME: &str = "review-corpus-v1";
/// Share of each (class, stratum) cell that goes to `dev` (§2.2).
pub const DEV_SHARE: f64 = 0.3;
/// Token proxy `[A]`: one token per 4 diff bytes, rounded up. Strata are
/// decided on this proxy, never on a model tokenizer, so they do not move when
/// the model changes.
pub const BYTES_PER_TOKEN: u64 = 4;

/// Item class (§2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Class {
    /// Planted: a mutation fixture applied as a diff.
    P,
    /// Real: a merged fix, reverse-applied.
    R,
    /// Good: a merged PR that stayed clean.
    G,
}

impl Class {
    /// P and R carry a defect; G does not.
    #[must_use]
    pub fn is_defect(self) -> bool {
        !matches!(self, Self::G)
    }
}

/// Diff-size stratum on the token proxy: S < 2k, M 2k–8k, L > 8k.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Stratum {
    S,
    M,
    L,
}

/// Split (§2.2): `dev` may be read by prompt authors; `test` is sealed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Split {
    Dev,
    Test,
}

/// A defect location: the file and the new-side line of the diff.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Loc {
    pub file: String,
    pub line: u32,
}

/// One corpus item. `defect` is empty exactly for class G.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Item {
    pub id: String,
    pub class: Class,
    pub stratum: Stratum,
    pub split: Split,
    pub diff_sha256: String,
    pub bytes: u64,
    pub approx_tokens: u64,
    /// Where the item came from (PR number + commit, or mutant identity).
    pub source: String,
    pub defect: Vec<Loc>,
}

impl Item {
    /// Build an item from its diff text. The split starts as `Test` and is set
    /// by [`assign_splits`].
    #[must_use]
    pub fn new(id: String, class: Class, source: String, diff: &str) -> Self {
        let bytes = diff.len() as u64;
        let approx_tokens = bytes.div_ceil(BYTES_PER_TOKEN);
        let defect = if class.is_defect() {
            defect_locs(diff)
        } else {
            Vec::new()
        };
        Self {
            id,
            class,
            stratum: stratum(approx_tokens),
            split: Split::Test,
            diff_sha256: sha256_hex(diff.as_bytes()),
            bytes,
            approx_tokens,
            source,
            defect,
        }
    }
}

/// Lowercase hex sha256.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    crate::prereg::sha256_hex(bytes)
}

/// Stratum of a token count.
#[must_use]
pub fn stratum(tokens: u64) -> Stratum {
    match tokens {
        0..2000 => Stratum::S,
        2000..=8000 => Stratum::M,
        _ => Stratum::L,
    }
}

/// A file a reviewer is shown: Rust or shell source, not a test, bench or
/// example. Test hunks are excluded so that R items reintroduce the defect
/// without also deleting the test that names it.
#[must_use]
pub fn is_review_path(path: &str) -> bool {
    let code = path.ends_with(".rs") || path.ends_with(".sh");
    let test_like = path.starts_with("tests/")
        || ["/tests/", "/test/", "/benches/", "/examples/"]
            .iter()
            .any(|d| path.contains(d))
        || path.ends_with("_tests.rs")
        || path.ends_with("_test.rs")
        || path.ends_with("/tests.rs");
    code && !test_like
}

/// Strip everything in a cargo-mutants diff that names the mutation: the
/// `+++ replace … with …` header and the `/* ~ changed by cargo-mutants ~ */`
/// marker. The result is a plain `a/`/`b/` unified diff of `file`.
#[must_use]
pub fn sanitize_mutant_diff(diff: &str, file: &str) -> String {
    let mut out = format!("--- a/{file}\n+++ b/{file}\n");
    for line in diff.lines().skip_while(|l| !l.starts_with("@@")) {
        out.push_str(&line.replace(" /* ~ changed by cargo-mutants ~ */", ""));
        out.push('\n');
    }
    out
}

/// True when `text` still names the mutation (a leak into the prompt).
#[must_use]
pub fn names_the_mutation(text: &str) -> bool {
    text.contains("cargo-mutants") || text.contains("+++ replace ")
}

fn hunk_new_start(header: &str) -> Option<u32> {
    let plus = header.split_whitespace().find(|t| t.starts_with('+'))?;
    plus[1..].split(',').next()?.parse().ok()
}

/// Defect locations: every added line (new-side line number); for a hunk that
/// only deletes, the new-side line where the deletion sits.
#[must_use]
pub fn defect_locs(diff: &str) -> Vec<Loc> {
    let mut locs = Vec::new();
    let mut file = String::new();
    let mut line = 0u32;
    let mut hunk_added = false;
    for l in diff.lines() {
        if let Some(p) = l.strip_prefix("+++ ") {
            file = p.strip_prefix("b/").unwrap_or(p).to_string();
        } else if l.starts_with("@@") {
            line = hunk_new_start(l).unwrap_or(0);
            hunk_added = false;
        } else if l.starts_with('+') {
            locs.push(Loc {
                file: file.clone(),
                line,
            });
            line += 1;
            hunk_added = true;
        } else if l.starts_with('-') {
            if !hunk_added && !l.starts_with("---") {
                locs.push(Loc {
                    file: file.clone(),
                    line,
                });
                hunk_added = true;
            }
        } else {
            line += 1;
        }
    }
    locs.sort();
    locs.dedup();
    locs
}

fn is_comment(path: &str, changed: &str) -> bool {
    let t = changed.trim();
    t.is_empty() || t.starts_with("//") || (path.ends_with(".sh") && t.starts_with('#'))
}

/// Keep only the hunks that change code. A fix PR's comment-only hunks are not
/// the fix, and reverse-applying them labels a comment as the defect. Files
/// left with no hunk are dropped whole.
#[must_use]
pub fn drop_comment_only_hunks(diff: &str) -> String {
    let mut out = String::new();
    let mut header = String::new();
    let mut hunk = String::new();
    let mut path = String::new();
    let mut code = false;
    let mut flush = |header: &mut String, hunk: &mut String, code: bool| {
        if code {
            out.push_str(header);
            header.clear();
            out.push_str(hunk);
        }
        hunk.clear();
    };
    for l in diff.lines() {
        if l.starts_with("diff --git") {
            flush(&mut header, &mut hunk, code);
            header.clear();
            code = false;
            header.push_str(l);
            header.push('\n');
        } else if l.starts_with("@@") {
            flush(&mut header, &mut hunk, code);
            code = false;
            hunk.push_str(l);
            hunk.push('\n');
        } else if hunk.is_empty() {
            if let Some(p) = l.strip_prefix("+++ ") {
                path = p.to_string();
            }
            header.push_str(l);
            header.push('\n');
        } else {
            let changed = l.strip_prefix(['+', '-']);
            code |= changed.is_some_and(|c| !is_comment(&path, c));
            hunk.push_str(l);
            hunk.push('\n');
        }
    }
    flush(&mut header, &mut hunk, code);
    out
}

/// Seeded, stratified split: within each (class, stratum) cell, items are
/// ordered by SplitMix64 keys drawn in id order, and the first
/// `round(DEV_SHARE × n)` become `dev`.
pub fn assign_splits(items: &mut [Item], seed: u64) {
    items.sort_by(|a, b| a.id.cmp(&b.id));
    let mut rng = SplitMix64::new(seed);
    let keys: Vec<u64> = items.iter().map(|_| rng.next_u64()).collect();
    let mut cells: Vec<(Class, Stratum)> = items.iter().map(|i| (i.class, i.stratum)).collect();
    cells.sort();
    cells.dedup();
    for cell in cells {
        let mut idx: Vec<usize> = (0..items.len())
            .filter(|&i| (items[i].class, items[i].stratum) == cell)
            .collect();
        idx.sort_by_key(|&i| keys[i]);
        let dev = dev_count(idx.len());
        for (rank, &i) in idx.iter().enumerate() {
            items[i].split = if rank < dev { Split::Dev } else { Split::Test };
        }
    }
}

/// `round(DEV_SHARE × n)`.
#[must_use]
pub fn dev_count(n: usize) -> usize {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let d = (DEV_SHARE * n as f64).round() as usize;
    d
}

/// Normalised hunk bodies: each `@@` hunk's lines without its header, so a
/// diff re-based to other line numbers still matches.
#[must_use]
pub fn hunks(diff: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur: Option<String> = None;
    for l in diff.lines() {
        if l.starts_with("@@") || l.starts_with("--- ") || l.starts_with("diff --git") {
            out.extend(cur.take());
            if l.starts_with("@@") {
                cur = Some(String::new());
            }
        } else if let Some(c) = cur.as_mut() {
            if l.starts_with([' ', '+', '-']) {
                c.push_str(l.trim_end());
                c.push('\n');
            }
        }
    }
    out.extend(cur);
    out
}

/// Minimum hunk length (lines) that is fingerprinted. Shorter hunks (`-}`/`+}`)
/// are too generic to prove a leak.
pub const MIN_HUNK_LINES: usize = 4;

/// sha256 of every fingerprintable hunk of a diff, sorted and deduplicated.
#[must_use]
pub fn hunk_fingerprints(diff: &str) -> Vec<String> {
    let mut f: Vec<String> = hunks(diff)
        .iter()
        .filter(|h| h.lines().count() >= MIN_HUNK_LINES && h.lines().any(|l| !l.starts_with(' ')))
        .map(|h| sha256_hex(h.as_bytes()))
        .collect();
    f.sort();
    f.dedup();
    f
}

/// One sealed test item: id, diff sha, hunk fingerprints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sealed {
    pub id: String,
    pub diff_sha256: String,
    pub hunks: Vec<String>,
}

/// Render the sealed manifest: the scheme line, then one line per test item
/// (`id diff_sha hunk,hunk,…`), sorted by id.
#[must_use]
pub fn render_manifest(sealed: &[Sealed]) -> String {
    let mut rows: Vec<&Sealed> = sealed.iter().collect();
    rows.sort_by(|a, b| a.id.cmp(&b.id));
    let mut out = format!("{SCHEME} test-manifest\n");
    for s in rows {
        out.push_str(&format!(
            "{} {} {}\n",
            s.id,
            s.diff_sha256,
            s.hunks.join(",")
        ));
    }
    out
}

/// Parse a sealed manifest. `None` if the scheme line is missing or a row is
/// malformed — a manifest that cannot be read seals nothing.
#[must_use]
pub fn parse_manifest(text: &str) -> Option<Vec<Sealed>> {
    let mut lines = text.lines();
    if lines.next()? != format!("{SCHEME} test-manifest") {
        return None;
    }
    lines
        .filter(|l| !l.is_empty())
        .map(|l| {
            let mut f = l.split(' ');
            let id = f.next()?.to_string();
            let diff_sha256 = f.next()?.to_string();
            let hunks = f
                .next()
                .unwrap_or("")
                .split(',')
                .filter(|h| !h.is_empty())
                .map(str::to_string)
                .collect();
            (diff_sha256.len() == 64).then_some(Sealed {
                id,
                diff_sha256,
                hunks,
            })
        })
        .collect()
}

/// Corpus version: sha256 of the sealed manifest bytes.
#[must_use]
pub fn corpus_version(manifest: &str) -> String {
    format!("{SCHEME}@{}", &sha256_hex(manifest.as_bytes())[..16])
}

/// Deterministic pick of up to `n` items, balanced across strata where the
/// population allows: candidates are shuffled per stratum with `seed`, then
/// taken round-robin S, M, L until `n` are chosen or all are exhausted.
#[must_use]
pub fn pick_balanced(mut cands: Vec<Item>, n: usize, seed: u64) -> Vec<Item> {
    cands.sort_by(|a, b| a.id.cmp(&b.id));
    let mut rng = SplitMix64::new(seed);
    let keys: Vec<u64> = cands.iter().map(|_| rng.next_u64()).collect();
    let mut order: Vec<usize> = (0..cands.len()).collect();
    order.sort_by_key(|&i| keys[i]);
    let mut per: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for i in order {
        per[cands[i].stratum as usize].push(i);
    }
    let mut chosen = Vec::new();
    let mut round = 0;
    while chosen.len() < n && per.iter().any(|p| p.len() > round) {
        for p in &per {
            if chosen.len() < n && round < p.len() {
                chosen.push(p[round]);
            }
        }
        round += 1;
    }
    chosen.sort_unstable();
    chosen.into_iter().map(|i| cands[i].clone()).collect()
}

#[cfg(test)]
#[path = "corpus_tests.rs"]
mod tests;
