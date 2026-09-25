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
                "Every obligation has a sorry-free in-tree Lean 4 theorem or is not applicable, with at least one proved"
            }
            ProofLevel::L5 => "L4 + at least one binding, every binding implemented",
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
         none is reported self-declared and excluded from L4), a not-applicable count is \
         taken from the contract's own verification summary, and no checked lake \
         discharge summary is read yet.\n",
    );
    out.push_str(END_MARKER);
    out
}

/// Lines of `doc` OUTSIDE the generated block that pair a level with the wrong tool.
///
/// Each tool mention (`kani`, `lean`) is paired with the NEAREST level token on its line
/// (`L1`..`L5`, `Level 1`..`Level 5`), in either direction, and flagged when that level
/// is wrong for the tool: Kani paired with L4/L5 (Kani is L3), or Lean paired with L5
/// on a line that says nothing about bindings (Lean alone is L4). Nearest-in-either-
/// direction catches "Level 4 (Kani)" and "Kani is used at Level 4" alike, and leaves
/// "How Kani (L3) and Lean (L4) Compose" alone. Quorum lanes on PR #4092 found both
/// gaps: prose below the block still teaching "Level 4 (Kani)" (round 1), and a
/// detector that only looked forward from the level token (round 3).
#[cfg(test)]
fn stale_level_pairings(doc: &str) -> Vec<String> {
    let outside: String = match (doc.find(MARKER), doc.find(END_MARKER)) {
        (Some(a), Some(b)) if a < b => format!("{}{}", &doc[..a], &doc[b + END_MARKER.len()..]),
        _ => doc.to_string(),
    };
    let mut stale = Vec::new();
    for line in outside.lines() {
        let low = line.to_lowercase();
        let levels = level_tokens(&low);
        let wrong = ["kani", "lean"].iter().any(|tool| {
            word_starts(&low, tool).any(|t| {
                let t_end = t + tool.len();
                let nearest = levels.iter().min_by_key(|(start, end, _)| {
                    if *end <= t {
                        t - end
                    } else if *start >= t_end {
                        start - t_end
                    } else {
                        0
                    }
                });
                match (*tool, nearest) {
                    ("kani", Some((_, _, n))) => *n >= 4,
                    ("lean", Some((_, _, n))) => *n == 5 && !low.contains("binding"),
                    _ => false,
                }
            })
        });
        if wrong {
            stale.push(line.trim().to_string());
        }
    }
    stale
}

/// Byte offsets where `needle` starts a word in `hay`.
#[cfg(test)]
fn word_starts<'a>(hay: &'a str, needle: &'a str) -> impl Iterator<Item = usize> + 'a {
    hay.match_indices(needle)
        .map(|(i, _)| i)
        .filter(move |&i| i == 0 || !hay.as_bytes()[i - 1].is_ascii_alphanumeric())
}

/// `(start, end, n)` for every whole-word level token `lN` / `level N` (N in 1..=5).
#[cfg(test)]
fn level_tokens(low: &str) -> Vec<(usize, usize, u8)> {
    let mut out = Vec::new();
    for n in 1u8..=5 {
        for tok in [format!("l{n}"), format!("level {n}")] {
            for i in word_starts(low, &tok) {
                let end = i + tok.len();
                if !low[end..].starts_with(|c: char| c.is_ascii_alphanumeric()) {
                    out.push((i, end, n));
                }
            }
        }
    }
    out
}

// The tests live DIRECTLY in `levels` (not in a `tests` submodule): PVL-001 EV-3's accept is
// `cargo test -p aprender-contracts --lib -- levels::readme_and_ladder_docs_match_enum`, and
// libtest's filter is a substring of the full path, so under `levels::tests::` that accept
// command ran ZERO tests and passed.

/// The three copies of the ladder doc (PVL-001 EV-3 names exactly these), relative to the
/// workspace root. Read from disk, not `include_str!`: two live outside this crate, and an
/// `include!` target outside the crate fails check_package_includes.sh even under `cfg(test)`.
#[cfg(test)]
const LADDER_COPIES: [&str; 3] = [
    "crates/aprender-contracts-staging/docs/specifications/sub/verification-ladder.md",
    "crates/aprender-contracts-staging/book/src/verification-ladder.md",
    "docs/specifications/aprender-contracts-staging/sub/verification-ladder.md",
];

/// A ladder copy's text. A copy that cannot be read fails the test: "could not check" is never "matches".
#[cfg(test)]
fn ladder_copy(path: &str) -> String {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::fs::read_to_string(root.join(path))
        .unwrap_or_else(|e| panic!("{path}: {e} (PVL-001 EV-3 reads all three copies)"))
}

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
    // The two guards a vacuous reading would drop (quorum round 4, measured): L4 needs at
    // least one GROUNDED obligation (ONT-2a: all-not-applicable is not L4), and L5 needs at
    // least one binding (is_fully_bound: zero bindings is not "all bound"). The code side
    // is pinned by proof_status_tests::level_all_not_applicable_is_not_l4 and
    // ::level_l5_needs_at_least_one_binding.
    assert!(ProofLevel::L4.method().contains("at least one proved"));
    assert!(ProofLevel::L5.method().contains("at least one binding"));
    // 2. Every ladder doc copy carries the generated block, byte for byte.
    let block = ladder_block();
    for path in LADDER_COPIES {
        let text = ladder_copy(path);
        let text = text.as_str();
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

/// Docs outside the three ladder copies that name a proof level next to Kani or Lean
/// (#4106). They carry no generated block; they only must not pair a level with the
/// wrong tool. `legacy/` and PP-066 are records of the old numbering and stay out.
#[cfg(test)]
const LADDER_CITING_DOCS: [&str; 23] = [
    "crates/aprender-contracts-staging/docs/specifications/sub/eiffel-dbc-explain.md",
    "docs/specifications/aprender-contracts-staging/sub/eiffel-dbc-explain.md",
    "crates/aprender-contracts-staging/docs/specifications/sub/lean-kani-composition.md",
    "docs/specifications/aprender-contracts-staging/sub/lean-kani-composition.md",
    "crates/aprender-contracts-staging/book/src/lean-kani-composition.md",
    "crates/aprender-contracts-staging/docs/specifications/sub/eiffel-dbc-domains-2.md",
    "docs/specifications/aprender-contracts-staging/sub/eiffel-dbc-domains-2.md",
    "crates/aprender-contracts-staging/docs/specifications/pv-spec.md",
    "docs/specifications/aprender-contracts-staging/pv-spec.md",
    "crates/aprender-contracts-staging/book/src/examples.md",
    "crates/aprender-contracts-staging/book/src/integration.md",
    "docs/specifications/components/cli-silent-failure-enforcement.md",
    "crates/aprender-contracts-staging/book/src/expression-languages.md",
    "crates/aprender-contracts-staging/docs/specifications/sub/eiffel-dbc.md",
    "docs/specifications/aprender-contracts-staging/sub/eiffel-dbc.md",
    "crates/aprender-contracts-staging/docs/specifications/sub/eiffel-dbc-type-invariants.md",
    "docs/specifications/aprender-contracts-staging/sub/eiffel-dbc-type-invariants.md",
    "crates/aprender-contracts-staging/docs/specifications/sub/escape-proof-enforcement.md",
    "docs/specifications/aprender-contracts-staging/sub/escape-proof-enforcement.md",
    "crates/aprender-contracts-staging/docs/specifications/sub/two-tier-architecture.md",
    "docs/specifications/aprender-contracts-staging/sub/two-tier-architecture.md",
    "crates/aprender-contracts-staging/docs/specifications/sub/lint-2.md",
    "docs/specifications/aprender-contracts-staging/sub/lint-2.md",
];

#[test]
fn ladder_citing_docs_pair_levels_with_the_right_tool() {
    let stale: Vec<String> = LADDER_CITING_DOCS
        .iter()
        .flat_map(|path| {
            stale_level_pairings(&ladder_copy(path))
                .into_iter()
                .map(move |line| format!("{path}: {line}"))
        })
        .collect();
    assert!(
        stale.is_empty(),
        "a doc pairs a level with the wrong tool (Kani is L3, Lean alone is L4; \
         enforcement layers are E0-E5):\n{}",
        stale.join("\n")
    );
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
        // reversed order: the tool first, the level after it (round 3's gap)
        "Kani is the tool used at Level 4 for bounded checks.",
        "The Lean prover is what L5 means here.",
        // two levels on the line: Kani's nearest is the later L4, not the earlier L2,
        // so a detector that only looks backward from the tool misses it
        "Falsification is L2 then Kani at L4.",
    ];
    for line in must_flag {
        assert_eq!(stale_level_pairings(line).len(), 1, "must flag: {line}");
    }
    let must_not_flag = [
        "| Obligation Type | Types (rustc) | L2 (probar/proptest) | L3 (Kani) | L4 (Lean) |",
        "3. **L3:** Kani exhaustively verified it for ALL inputs within the kernel's",
        "4. **L4:** a Lean 4 theorem proves it unbounded; **L5** additionally requires",
        "## How Kani (L3) and Lean (L4) Compose",
        "| L5 | L4 + at least one binding, every binding implemented |",
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
