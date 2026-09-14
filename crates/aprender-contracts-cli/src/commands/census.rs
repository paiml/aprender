//! `pv census` — ONT-001 v4.3 ONT-1: one cardinality over the contract corpus,
//! reported `by_anchoring` and `by_entity_type`.
//!
//! WHY THIS READS RAW YAML AND NOT `Contract`.
//!
//! `Contract` is not `#[serde(deny_unknown_fields)]`, so serde silently DROPS an
//! `entity:` block it does not model. Measured 2026-09-14: prepending
//! `entity: kernel` to any contract and running `pv validate` prints
//! "0 error(s), 0 warning(s)  Contract is valid." — the key is accepted by being
//! ignored. A census built on the typed struct would therefore report
//! `unanchored` for every contract in the corpus FOREVER, including after the
//! corpus was fully anchored, and the ONT-001 §11.2 ratchet reading it would
//! record 0 while the work was done. The raw document is the only surface on
//! which the anchor is visible today.
//!
//! ONT-001 §0.0 anchoring levels, verbatim:
//!   absent            -> unanchored  (a law, pattern, policy or process)
//!   `{type}` only     -> class-level (a shape over every entity of that type)
//!   `{type, ref}`     -> instance-level (one thing)
//!
//! R-2 applies to the instrument: a corpus this cannot read is a DECLINE (exit 2),
//! never a census of zero. Zero contracts found and "no contracts directory" are
//! the same output, and one of them is a broken tool.

use std::collections::BTreeMap;
use std::path::Path;

/// How a contract names the thing it is a contract for (ONT-001 §0.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchoring {
    /// No `entity:` block. Not a defect: R-5 makes `entity:` optional by design.
    Unanchored,
    /// `entity: { type }` — a shape over every entity of that type.
    Class,
    /// `entity: { type, ref }` — one named thing.
    Instance,
}

/// The counts ONT-1 owes. Cardinality first, then the two breakdowns.
#[derive(Debug, Default)]
pub struct Census {
    pub total: usize,
    pub unanchored: usize,
    pub class: usize,
    pub instance: usize,
    pub by_entity_type: BTreeMap<String, usize>,
}

/// Classify ONE raw YAML document. Pure, so the case table can pin every row
/// without touching a filesystem.
#[must_use]
pub fn classify(yaml: &str) -> (Anchoring, Option<String>) {
    let Some(block) = entity_block(yaml) else {
        return (Anchoring::Unanchored, None);
    };
    let ty = scalar_field(&block, "type");
    let has_ref = scalar_field(&block, "ref").is_some();
    match (ty, has_ref) {
        (Some(t), true) => (Anchoring::Instance, Some(t)),
        (Some(t), false) => (Anchoring::Class, Some(t)),
        // `entity:` present but naming no type is not an anchor. Reporting it as
        // one would let a malformed block inflate the ratchet.
        (None, _) => (Anchoring::Unanchored, None),
    }
}

/// The text of the top-level `entity:` block, inline or nested, or None.
///
/// Split from its nested-block reader because the combined form measured
/// cognitive 26 against this repo's ceiling of 25 — the pre-commit gate refused
/// the commit, which is the gate doing its job on a first draft.
fn entity_block(yaml: &str) -> Option<String> {
    let mut lines = yaml.lines();
    while let Some(line) = lines.next() {
        let Some(rest) = line.strip_prefix("entity:") else {
            continue;
        };
        let rest = rest.trim();
        if rest.is_empty() {
            // nested: `entity:` alone, the block is the indented lines below it
            return Some(indented_block(&mut lines));
        }
        // inline: `entity: { type: code, ref: src/x.rs }`
        return Some(rest.to_string());
    }
    None
}

/// The run of more-indented lines that follow a bare `key:`, flattened.
fn indented_block<'a>(lines: &mut impl Iterator<Item = &'a str>) -> String {
    let mut block = String::new();
    for next in lines {
        if next.trim().is_empty() {
            continue;
        }
        if !next.starts_with([' ', '\t']) {
            break;
        }
        block.push_str(next.trim());
        block.push('\n');
    }
    block
}

/// `type: code` / `type = code` / `{ type: code }` -> "code".
fn scalar_field(block: &str, key: &str) -> Option<String> {
    let needle = format!("{key}:");
    let idx = block.find(&needle)?;
    let after = &block[idx + needle.len()..];
    let val: String = after
        .trim_start()
        .chars()
        .take_while(|c| !matches!(c, ',' | '}' | '\n' | '#'))
        .collect();
    let val = val.trim().trim_matches(['"', '\'']).to_string();
    if val.is_empty() {
        None
    } else {
        Some(val)
    }
}

/// Walk a contract directory and count. Returns the census, or an error when the
/// corpus cannot be read — a DECLINE, never a census of zero.
pub fn census_of(dir: &Path) -> Result<Census, Box<dyn std::error::Error>> {
    if !dir.is_dir() {
        return Err(format!(
            "no contract directory at {} — a census of zero and an unreadable \
             corpus are the same number, and one of them is a broken tool (ONT R-2)",
            dir.display()
        )
        .into());
    }
    let mut c = Census::default();
    walk(dir, &mut c)?;
    if c.total == 0 {
        return Err(format!(
            "0 contracts under {} — refusing to report a census nothing was read for",
            dir.display()
        )
        .into());
    }
    Ok(c)
}

fn walk(dir: &Path, c: &mut Census) -> Result<(), Box<dyn std::error::Error>> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            walk(&path, c)?;
            continue;
        }
        if path.extension().is_none_or(|e| e != "yaml") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        c.total += 1;
        let (anchoring, ty) = classify(&text);
        match anchoring {
            Anchoring::Unanchored => c.unanchored += 1,
            Anchoring::Class => c.class += 1,
            Anchoring::Instance => c.instance += 1,
        }
        if let Some(t) = ty {
            *c.by_entity_type.entry(t).or_insert(0) += 1;
        }
    }
    Ok(())
}

pub fn run(contract_dir: &Path, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let c = census_of(contract_dir)?;
    if json {
        print!("{{\"total\":{},", c.total);
        print!(
            "\"by_anchoring\":{{\"unanchored\":{},\"class\":{},\"instance\":{}}},",
            c.unanchored, c.class, c.instance
        );
        print!("\"by_entity_type\":{{");
        let mut first = true;
        for (k, v) in &c.by_entity_type {
            if !first {
                print!(",");
            }
            first = false;
            print!("\"{k}\":{v}");
        }
        println!("}}}}");
        return Ok(());
    }
    println!("== pv census (ONT-001 ONT-1) ==");
    println!("contracts: {}", c.total);
    println!();
    println!("by_anchoring");
    println!(
        "  unanchored {:>6}   (no entity: — a law, pattern or policy; optional by R-5)",
        c.unanchored
    );
    println!(
        "  class      {:>6}   (entity: {{type}} — a shape over every entity of that type)",
        c.class
    );
    println!(
        "  instance   {:>6}   (entity: {{type, ref}} — one named thing)",
        c.instance
    );
    println!();
    println!("by_entity_type");
    if c.by_entity_type.is_empty() {
        println!("  (none — no contract in this corpus names an entity type)");
    } else {
        for (k, v) in &c.by_entity_type {
            println!("  {k:<18} {v:>6}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_entity_is_unanchored() {
        assert_eq!(classify("id: X\nkind: Kernel\n").0, Anchoring::Unanchored);
    }

    #[test]
    fn inline_type_only_is_class_level() {
        let (a, t) = classify("entity: { type: readme }\n");
        assert_eq!(a, Anchoring::Class);
        assert_eq!(t.as_deref(), Some("readme"));
    }

    #[test]
    fn inline_type_and_ref_is_instance_level() {
        let (a, t) = classify("entity: { type: readme, ref: README.md }\n");
        assert_eq!(a, Anchoring::Instance);
        assert_eq!(t.as_deref(), Some("readme"));
    }

    #[test]
    fn nested_block_is_read_too() {
        let (a, t) = classify("id: X\nentity:\n  type: gguf\n  ref: m.gguf\nshape: {}\n");
        assert_eq!(a, Anchoring::Instance);
        assert_eq!(t.as_deref(), Some("gguf"));
    }

    #[test]
    fn nested_type_only_is_class_level() {
        let (a, t) = classify("entity:\n  type: csv\nshape: {}\n");
        assert_eq!(a, Anchoring::Class);
        assert_eq!(t.as_deref(), Some("csv"));
    }

    /// An `entity:` block naming no type is NOT an anchor. Counting it would let
    /// a malformed block inflate the §11.2 ratchet.
    #[test]
    fn entity_without_a_type_is_not_an_anchor() {
        assert_eq!(
            classify("entity: { ref: README.md }\n").0,
            Anchoring::Unanchored
        );
    }

    /// `entity:` must be top-level. An indented one belongs to another key.
    #[test]
    fn indented_entity_is_not_the_top_level_block() {
        assert_eq!(
            classify("metadata:\n  entity: { type: code }\n").0,
            Anchoring::Unanchored
        );
    }

    #[test]
    fn quoted_values_are_unquoted() {
        let (_, t) = classify("entity: { type: \"apr-model\", ref: 'm.apr' }\n");
        assert_eq!(t.as_deref(), Some("apr-model"));
    }

    #[test]
    fn a_trailing_comment_is_not_part_of_the_type() {
        let (_, t) = classify("entity:\n  type: sqlite  # the db\n");
        assert_eq!(t.as_deref(), Some("sqlite"));
    }

    /// R-2 on the instrument: an unreadable corpus declines, never censuses zero.
    #[test]
    fn a_missing_directory_is_an_error_not_a_zero_census() {
        assert!(census_of(Path::new("/nonexistent/contracts")).is_err());
    }
}
