//! REX-11 B1 challengers (§5.3): prompt versions (B1a) and retrieval
//! few-shot (B1b), built so that the sealed test split cannot leak (R-2).
//!
//! - The retrieval pool is gold specimens outside the test split. A test
//!   item offered to [`select`] is an error, not a skipped row: the caller
//!   read a sealed diff, and that is the violation.
//! - The query never retrieves itself (same id or same diff sha), so a dev
//!   evaluation is not an open-book test.
//! - Similarity is the Jaccard index of changed-line token bigrams:
//!   deterministic, no model, ties broken by id.
//! - Every prompt a challenger sends, B1a file or rendered B1b prompt, passes
//!   [`guard`]: any sealed diff sha or hunk inside it is a hard FAIL
//!   (`review-corpus-contamination-v1`).

use std::collections::BTreeSet;

use crate::contamination::Index;
use crate::corpus::{Item, Split};

/// Token bigrams of the changed lines (`+`/`-`, not the file headers).
fn shingles(diff: &str) -> BTreeSet<(String, String)> {
    let toks: Vec<String> = diff
        .lines()
        .filter(|l| {
            (l.starts_with('+') || l.starts_with('-'))
                && !l.starts_with("+++")
                && !l.starts_with("---")
        })
        .flat_map(|l| {
            l[1..]
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .filter(|t| !t.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect();
    toks.windows(2)
        .map(|w| (w[0].clone(), w[1].clone()))
        .collect()
}

/// Jaccard index of two diffs' changed-line bigrams; 0 when either is empty.
#[must_use]
pub fn similarity(a: &str, b: &str) -> f64 {
    let (x, y) = (shingles(a), shingles(b));
    let union = x.union(&y).count();
    if union == 0 {
        return 0.0;
    }
    x.intersection(&y).count() as f64 / union as f64
}

/// The `k` pool items most similar to the query, most similar first.
pub fn select<'a>(
    query: &Item,
    query_diff: &str,
    pool: &'a [(Item, String)],
    k: usize,
) -> Result<Vec<&'a (Item, String)>, String> {
    if let Some((i, _)) = pool.iter().find(|(i, _)| i.split == Split::Test) {
        return Err(format!(
            "{}: a sealed test item in the retrieval pool (R-2)",
            i.id
        ));
    }
    let mut ranked: Vec<(f64, &(Item, String))> = pool
        .iter()
        .filter(|(i, _)| i.id != query.id && i.diff_sha256 != query.diff_sha256)
        .map(|p| (similarity(query_diff, &p.1), p))
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1 .0.id.cmp(&b.1 .0.id)));
    Ok(ranked.into_iter().take(k).map(|(_, p)| p).collect())
}

/// The gold answer shown for an example: the verdict line, then the defect
/// locations for a defect.
fn gold(item: &Item) -> String {
    if item.class.is_defect() {
        let locs: Vec<String> = item
            .defect
            .iter()
            .map(|l| format!("- defect at {}:{}", l.file, l.line))
            .collect();
        format!("VERDICT: FAIL\n{}", locs.join("\n"))
    } else {
        "VERDICT: PASS".into()
    }
}

/// `base`, then the examples with their gold answers. The item under review
/// is appended by `harness::compose` as before.
#[must_use]
pub fn render(base: &str, examples: &[&(Item, String)]) -> String {
    let mut s = base.trim_end().to_string();
    if examples.is_empty() {
        return s + "\n";
    }
    s.push_str("\n\nReviewed examples:\n");
    for (n, (item, diff)) in examples.iter().enumerate() {
        let nl = if diff.ends_with('\n') { "" } else { "\n" };
        s.push_str(&format!(
            "\nExample {}:\n```diff\n{diff}{nl}```\n{}\n",
            n + 1,
            gold(item)
        ));
    }
    s.push_str("\nNow review this diff.\n");
    s
}

/// The bodies of the prompt's fenced blocks. Text after a fence (a gold
/// `- defect at …` line, prose starting with `-`) would otherwise join the
/// last hunk and change its fingerprint, so each block is scanned alone.
fn fenced(prompt: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur: Option<String> = None;
    for l in prompt.lines() {
        if l.trim_start().starts_with("```") {
            match cur.take() {
                Some(b) => out.push(b),
                None => cur = Some(String::new()),
            }
        } else if let Some(b) = cur.as_mut() {
            b.push_str(l);
            b.push('\n');
        }
    }
    out.extend(cur);
    out
}

/// Refuse a prompt that carries any sealed item, whether in running text or
/// inside a fenced block. An empty index proves nothing, so it is refused too.
pub fn guard(prompt: &str, sealed: &Index, what: &str) -> Result<(), String> {
    if sealed.is_empty() {
        return Err("no sealed manifest indexed: the guard would prove nothing".into());
    }
    let mut hits = sealed.scan(prompt, what);
    for block in fenced(prompt) {
        hits.extend(sealed.scan(&block, what));
    }
    hits.sort();
    hits.dedup();
    if hits.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{what}: {} sealed test leak(s): {hits:?}",
            hits.len()
        ))
    }
}

#[cfg(test)]
#[path = "fewshot_tests.rs"]
mod tests;
