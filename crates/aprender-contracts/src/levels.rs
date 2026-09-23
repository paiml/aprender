//! PVL-001 EV-3 (PVL-3): ONE definition of the proof levels.
//!
//! [`ProofLevel`] is authoritative. Before this, three places disagreed:
//! the enum (L3 Kani, L4 Lean, L5 Lean + bindings), the README table from
//! `readme_gen` (L4 "Kani BMC", L5 "Lean 4 theorem", L1 "Type system"), and
//! three copies of the ladder doc (L4 Kani, L5 Lean, plus an L0 the enum does
//! not have). A reader of the README was told an L4 contract was Kani-checked
//! when `pv` had computed that it was Lean-proved.
//!
//! Now the README's method column comes from [`ProofLevel::method`], and each
//! ladder doc carries [`ladder_block`] verbatim under [`MARKER`]. The test below
//! fails if either drifts from the enum.

use crate::proof_status::ProofLevel;

/// The first line of the generated block in every ladder doc copy.
pub const MARKER: &str = "<!-- generated from ProofLevel; do not edit -->";
/// The last line of the generated block.
pub const END_MARKER: &str = "<!-- end generated from ProofLevel -->";

impl ProofLevel {
    /// Every level, highest first (the order the README and docs print).
    pub const ALL_DESCENDING: [ProofLevel; 5] = [
        ProofLevel::L5,
        ProofLevel::L4,
        ProofLevel::L3,
        ProofLevel::L2,
        ProofLevel::L1,
    ];

    /// What reaching this level means: the one definition every surface prints. Each
    /// string states what `compute_proof_level_with_grounding` actually checks (a quorum
    /// lane on PR #4092 found the first wording claimed less for L3 and the wrong thing
    /// for L4).
    #[must_use]
    pub fn method(self) -> &'static str {
        match self {
            ProofLevel::L1 => "Contract YAML with equations",
            ProofLevel::L2 => "Falsification tests cover every obligation",
            ProofLevel::L3 => "L2 + at least one Kani bounded-model-check harness",
            ProofLevel::L4 => {
                "Every obligation has a sorry-free in-tree Lean 4 theorem (or is not applicable)"
            }
            ProofLevel::L5 => "L4 + every binding implemented",
        }
    }
}

/// The ladder table every doc copy carries, from [`MARKER`] to [`END_MARKER`].
#[must_use]
pub fn ladder_block() -> String {
    let mut out = String::new();
    out.push_str(MARKER);
    out.push('\n');
    out.push_str("| Level | Method |\n|-------|--------|\n");
    for level in ProofLevel::ALL_DESCENDING {
        out.push_str(&format!("| {level} | {} |\n", level.method()));
    }
    out.push_str(
        "\nL4 and L5 are grounded only textually until PVL-001 EV-8b lands: a claimed Lean \
         proof counts when a sorry-free Lean theorem in this tree matches it (a claim with \
         none is reported self-declared and excluded from L4), but no checked lake \
         discharge summary is read yet.\n",
    );
    out.push_str(END_MARKER);
    out
}

/// Lines of `doc` OUTSIDE the generated block that pair a level with the wrong tool:
/// L4/L5 followed on the same line by Kani (Kani is L3), or L5 followed by Lean with no
/// mention of bindings (Lean alone is L4). The rest of the line is searched, not a
/// fixed window: column-aligned tables put the tool far from the level.
/// The generated block alone is not enough: a quorum lane (PR #4092, lane 2) found
/// prose a few lines below it still teaching "Level 4 (Kani)".
#[cfg(test)]
fn stale_level_pairings(doc: &str) -> Vec<String> {
    let outside: String = match (doc.find(MARKER), doc.find(END_MARKER)) {
        (Some(a), Some(b)) if a < b => format!("{}{}", &doc[..a], &doc[b + END_MARKER.len()..]),
        _ => doc.to_string(),
    };
    let pairs: [(&str, &str); 6] = [
        ("l4", "kani"),
        ("level 4", "kani"),
        ("l5", "kani"),
        ("level 5", "kani"),
        ("l5", "lean"),
        ("level 5", "lean"),
    ];
    let mut stale = Vec::new();
    for line in outside.lines() {
        let low = line.to_lowercase();
        let hit = pairs.iter().any(|(tok, tool)| {
            low.match_indices(tok).any(|(i, _)| {
                let starts_word = i == 0 || !low.as_bytes()[i - 1].is_ascii_alphanumeric();
                let after = &low[i + tok.len()..];
                let ends_word = !after.starts_with(|c: char| c.is_ascii_alphanumeric());
                starts_word
                    && ends_word
                    && after.contains(tool)
                    && !(*tool == "lean" && low.contains("binding"))
            })
        });
        if hit {
            stale.push(line.trim().to_string());
        }
    }
    stale
}

// The tests live DIRECTLY in `levels` (not in a `tests` submodule): PVL-001 EV-3's accept is
// `cargo test -p aprender-contracts --lib -- levels::readme_and_ladder_docs_match_enum`, and
// libtest's filter is a substring of the full path, so under `levels::tests::` that accept
// command ran ZERO tests and passed.

/// The three copies of the ladder doc (PVL-001 EV-3 names exactly these).
#[cfg(test)]
const LADDER_COPIES: [(&str, &str); 3] = [
    (
        "crates/aprender-contracts-staging/docs/specifications/sub/verification-ladder.md",
        include_str!(
            "../../aprender-contracts-staging/docs/specifications/sub/verification-ladder.md"
        ),
    ),
    (
        "crates/aprender-contracts-staging/book/src/verification-ladder.md",
        include_str!("../../aprender-contracts-staging/book/src/verification-ladder.md"),
    ),
    (
        "docs/specifications/aprender-contracts-staging/sub/verification-ladder.md",
        include_str!(
            "../../../docs/specifications/aprender-contracts-staging/sub/verification-ladder.md"
        ),
    ),
];

/// PVL-001 EV-3's accept test: `levels::readme_and_ladder_docs_match_enum`.
#[test]
fn readme_and_ladder_docs_match_enum() {
    // 1. The README's verification table prints, for each level, the enum's method.
    let readme = crate::readme_gen::verification_ladder_table(&[0, 0, 0, 0, 0]);
    for level in ProofLevel::ALL_DESCENDING {
        let row = format!("| {level} | 0 | {} |", level.method());
        assert!(
            readme.lines().any(|l| l == row),
            "README ladder row for {level} is not the enum's definition.\nwant: {row}\nREADME table:\n{readme}"
        );
    }
    // 2. Every ladder doc copy carries the generated block, byte for byte.
    let block = ladder_block();
    for (path, text) in LADDER_COPIES {
        let stale = stale_level_pairings(text);
        assert!(
            stale.is_empty(),
            "{path} still pairs a level with the wrong tool outside the generated block \
             (Kani is L3, Lean alone is L4):\n{}",
            stale.join("\n")
        );
        assert!(
            text.contains(&block),
            "{path} does not carry the block generated from ProofLevel. Replace its \
             proof-level table with:\n{block}"
        );
    }
}

/// The block itself names every level exactly once, in order: a regression in
/// `ladder_block` cannot pass by printing nothing.
#[test]
fn the_ladder_block_names_each_level_once_highest_first() {
    let block = ladder_block();
    let is_level_row =
        |l: &&str| l.len() > 4 && l.starts_with("| L") && l.as_bytes()[3].is_ascii_digit();
    let rows: Vec<&str> = block.lines().filter(is_level_row).collect();
    assert_eq!(rows.len(), 5, "{block}");
    for (row, level) in rows.iter().zip(ProofLevel::ALL_DESCENDING) {
        assert!(row.starts_with(&format!("| {level} |")), "{row}");
    }
    assert!(block.starts_with(MARKER) && block.ends_with(END_MARKER));
}

/// The stale-pairing detector's case table: the old doc lines must be flagged, the
/// corrected ones must not, so neither "flag nothing" nor "flag every level" passes.
#[cfg(test)]
#[test]
fn stale_level_pairings_case_table() {
    let must_flag = [
        "| Obligation Type | Level 1 (Types) | Level 3 (probar) | Level 4 (Kani) | Level 5 (Lean) |",
        "| Obligation Type | L1 (Types) | L3 (probar) | L4 (Kani) | L5 (Lean) |",
        "3. **L4:** Kani exhaustively verified for ALL inputs within the kernel's",
        "3. **Level 4:** Kani has exhaustively verified the property for ALL inputs up",
        "  L5    Theorem proving         Lean 4          True for ALL inputs. Period.",
    ];
    for line in must_flag {
        assert_eq!(stale_level_pairings(line).len(), 1, "must flag: {line}");
    }
    let must_not_flag = [
        "| Obligation Type | Types (rustc) | L2 (probar/proptest) | L3 (Kani) | L4 (Lean) |",
        "3. **L3:** Kani exhaustively verified it for ALL inputs within the kernel's",
        "4. **L4:** a Lean 4 theorem proves it unbounded; **L5** additionally requires",
        "## How Kani (L3) and Lean (L4) Compose",
        "| L5 | L4 + every binding implemented |",
        "E4 and E5 are defined in YAML but not yet run in CI.",
        "| **E4** | Logic bugs, overflows | Kani `#[kani::proof]` BMC |",
    ];
    for line in must_not_flag {
        assert!(
            stale_level_pairings(line).is_empty(),
            "must not flag: {line}"
        );
    }
    // The generated block itself is excluded even though it names L5 and Lean.
    assert!(stale_level_pairings(&ladder_block()).is_empty());
}
