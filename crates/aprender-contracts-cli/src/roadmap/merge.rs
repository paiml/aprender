//! `roadmap merge` — a git merge driver for roadmap.yaml that merges by ENTRY ID, not by line
//! (PMAT-3118; spec 6.4), ported from `scripts/lib/roadmap_merge.py`.
//!
//! A squash-merge from the queue makes `pmat work` re-serialise the whole roadmap, so every
//! stacked branch goes DIRTY on that one file. An id-keyed merge reconciles it: an entry is a
//! BYTE-EXACT block and only conflicts with ITSELF.
//!
//! ```text
//! ours-only change  -> take ours        theirs-only change -> take theirs
//! identical change  -> take either      both changed differently -> CONFLICT
//! one deleted, other untouched -> delete   deleted vs edited -> CONFLICT
//! ```
//!
//! ORDER IS THE CHECKER'S ORDER (`scripts/check_roadmap_sorted.sh`): within each id-prefix
//! numerically ascending; prefixes keep first-appearance order; legacy ids stay where they sit.

use std::collections::{HashMap, HashSet};

use super::{cmp_numeral, parse_id, split_entries, Entry};

/// A conflict: the message names what conflicted (an id, or the header).
#[derive(Debug, PartialEq, Eq)]
pub struct Conflict(pub String);

fn resolve_header(base: &str, ours: &str, theirs: &str) -> Result<String, Conflict> {
    if ours == base {
        return Ok(theirs.to_string());
    }
    if theirs == base || ours == theirs {
        return Ok(ours.to_string());
    }
    Err(Conflict(
        "header (everything before the first `- id:`)".to_string(),
    ))
}

fn resolve_existing<'a>(
    eid: &str,
    b: &'a str,
    o: Option<&'a str>,
    t: Option<&'a str>,
) -> Result<Option<&'a str>, Conflict> {
    let clash = || Conflict(eid.to_string());
    match (o, t) {
        (Some(o), Some(t)) => {
            if o == b {
                Ok(Some(t))
            } else if t == b || o == t {
                Ok(Some(o))
            } else {
                Err(clash())
            }
        }
        // One side deleted it: safe only if the other left it byte-identical to base.
        (survivor_o, survivor_t) => match survivor_o.or(survivor_t) {
            Some(s) if s != b => Err(clash()),
            _ => Ok(None),
        },
    }
}

fn resolve_entry<'a>(
    eid: &str,
    b: Option<&'a str>,
    o: Option<&'a str>,
    t: Option<&'a str>,
) -> Result<Option<&'a str>, Conflict> {
    if let Some(b) = b {
        return resolve_existing(eid, b, o, t);
    }
    if let (Some(x), Some(y)) = (o, t) {
        if x != y {
            return Err(Conflict(eid.to_string()));
        }
    }
    Ok(o.or(t))
}

/// Permute each id-prefix's entries within the slots that prefix already occupies in `reference`:
/// ascending numeral inside the prefix, every other position left exactly as given.
#[must_use]
pub fn enforce_prefix_order(reference: &[String], mutate: bool) -> Vec<String> {
    let mut slots: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, eid) in reference.iter().enumerate() {
        let Some((prefix, _)) = parse_id(eid) else {
            continue;
        };
        match slots.iter_mut().find(|(p, _)| *p == prefix) {
            Some((_, v)) => v.push(i),
            None => slots.push((prefix, vec![i])),
        }
    }
    let mut out = reference.to_vec();
    for (_, positions) in &slots {
        let mut ordered: Vec<&String> = positions.iter().map(|&i| &reference[i]).collect();
        let numeral = |e: &str| parse_id(e).map(|p| p.1).unwrap_or_default();
        ordered.sort_by(|a, b| cmp_numeral(&numeral(a), &numeral(b), mutate));
        for (pos, eid) in positions.iter().zip(ordered) {
            out[*pos] = eid.clone();
        }
    }
    out
}

/// Reference layout = base order, then ids only ours added, then ids only theirs added (each in
/// its own file order); then the checker's order is enforced over that layout.
fn sort_entries(live: &HashSet<&str>, seqs: [&[Entry]; 3], mutate: bool) -> Vec<String> {
    let mut reference = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    for seq in seqs {
        for (eid, _) in seq {
            if live.contains(eid.as_str()) && seen.insert(eid) {
                reference.push(eid.clone());
            }
        }
    }
    enforce_prefix_order(&reference, mutate)
}

fn dict(entries: &[Entry]) -> HashMap<&str, &str> {
    entries
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect()
}

/// Three-way merge by entry id.
///
/// # Errors
///
/// [`Conflict`] naming the id (or the header) both sides changed differently.
pub fn merge(b_txt: &str, o_txt: &str, t_txt: &str, mutate: bool) -> Result<String, Conflict> {
    let (b_pre, b_ent) = split_entries(b_txt);
    let (o_pre, o_ent) = split_entries(o_txt);
    let (t_pre, t_ent) = split_entries(t_txt);
    let f_pre = resolve_header(&b_pre, &o_pre, &t_pre)?;
    let (b, o, t) = (dict(&b_ent), dict(&o_ent), dict(&t_ent));
    let mut ids: Vec<&str> = b.keys().chain(o.keys()).chain(t.keys()).copied().collect();
    ids.sort_unstable();
    ids.dedup();
    let mut f_ent: HashMap<&str, &str> = HashMap::new();
    for eid in ids {
        let hit = resolve_entry(
            eid,
            b.get(eid).copied(),
            o.get(eid).copied(),
            t.get(eid).copied(),
        )?;
        if let Some(v) = hit {
            f_ent.insert(eid, v);
        }
    }
    let live: HashSet<&str> = f_ent.keys().copied().collect();
    let order = sort_entries(&live, [&b_ent, &o_ent, &t_ent], mutate);
    let mut out = f_pre;
    for k in &order {
        out.push_str(f_ent[k.as_str()]);
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
#[path = "merge_tests.rs"]
mod tests;
