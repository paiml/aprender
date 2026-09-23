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

    /// What reaching this level means: the one definition every surface prints.
    #[must_use]
    pub fn method(self) -> &'static str {
        match self {
            ProofLevel::L1 => "Contract YAML with equations",
            ProofLevel::L2 => "Falsification tests cover the obligations",
            ProofLevel::L3 => "Kani bounded model check",
            ProofLevel::L4 => "Lean 4 theorem proved",
            ProofLevel::L5 => "Lean 4 theorem proved + every binding verified implemented",
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
        "\nL4 and L5 are self-declared until PVL-001 EV-8b lands: the level is computed \
         from the contract's own YAML, not from a checked Lean discharge summary.\n",
    );
    out.push_str(END_MARKER);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three copies of the ladder doc (PVL-001 EV-3 names exactly these).
    const LADDER_COPIES: [(&str, &str); 3] = [
        (
            "crates/aprender-contracts-staging/docs/specifications/sub/verification-ladder.md",
            include_str!("../../aprender-contracts-staging/docs/specifications/sub/verification-ladder.md"),
        ),
        (
            "crates/aprender-contracts-staging/book/src/verification-ladder.md",
            include_str!("../../aprender-contracts-staging/book/src/verification-ladder.md"),
        ),
        (
            "docs/specifications/aprender-contracts-staging/sub/verification-ladder.md",
            include_str!("../../../docs/specifications/aprender-contracts-staging/sub/verification-ladder.md"),
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
}
