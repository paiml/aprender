//! PRM-00b pre-registration lock (contract `rex-prereg-v1`, scheme `rex-prereg-v2`, rule R-1).
//!
//! `rex-prereg-v2` locks PRM-001 v3 (`docs/specifications/PRM-001-prometheus.md`). The v1
//! lock (`docs/audits/rex-001/prereg.lock`, prereg_sha `ef51087d…`) is SUPERSEDED, never
//! deleted or edited (operator, 2026-09-25); its spec and plan stay byte-frozen in-tree and
//! `falsify_rex_prereg_004` keeps them verifying.
//!
//! The prereg sha is a sha256 over the sha256 of four frozen components, in a
//! fixed order:
//!
//! 1. `spec_s2_s5`  — spec bytes from the line `## §2 ` up to (not including)
//!    the line `## §6 ` (design, hypotheses, decision rule, improvement loop);
//! 2. `stats_rs`    — the analysis code;
//! 3. `analysis_plan` — `docs/audits/prm-001/analysis-plan.md` (plan v2);
//! 4. `prompt_v1`   — the fixed review prompt.
//!
//! The sealed test-item manifest is NOT a prereg component: it is filled once by
//! REX-02 (placeholder → sealed) and carried separately as the corpus version.
//! Everything outside §2–§5 (header, §7 tickets, §9 schema) may be edited
//! without a new spec version.

use sha2::{Digest, Sha256};

/// The spec as committed in-tree.
pub const SPEC: &str = include_str!("../../../docs/specifications/PRM-001-prometheus.md");
/// The analysis code.
pub const STATS_RS: &str = include_str!("stats.rs");
/// The analysis plan.
pub const ANALYSIS_PLAN: &str = include_str!("../../../docs/audits/prm-001/analysis-plan.md");
/// Prompt v1 (the G1 prompt header; the diff is appended after it).
pub const PROMPT_V1: &str = include_str!("../../../docs/audits/review-corpus/prompts/v1.txt");
/// The committed lock.
pub const LOCK: &str = include_str!("../../../docs/audits/prm-001/prereg.lock");

/// The superseded v1 lock and the v1 spec and plan it froze (never edited).
pub const LOCK_V1: &str = include_str!("../../../docs/audits/rex-001/prereg.lock");
/// The v1 spec (`rex-prereg-v1`), byte-frozen in §2–§5.
pub const SPEC_V1: &str =
    include_str!("../../../docs/specifications/review-experiment-protocol.md");
/// The v1 analysis plan, byte-frozen.
pub const ANALYSIS_PLAN_V1: &str = include_str!("../../../docs/audits/rex-001/analysis-plan.md");

/// Contract id stamped into the digest so a lock can never match another scheme.
pub const SCHEME: &str = "rex-prereg-v2";

/// Lowercase hex sha256.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Spec §2–§5 exactly. `None` if either heading is missing or out of order —
/// a spec that lost its §2 or §6 heading cannot be pre-registered.
#[must_use]
pub fn spec_sections_2_to_5(spec: &str) -> Option<&str> {
    let start = line_start(spec, "## §2 ")?;
    let end = line_start(&spec[start..], "## §6 ")? + start;
    Some(&spec[start..end])
}

fn line_start(text: &str, prefix: &str) -> Option<usize> {
    if text.starts_with(prefix) {
        return Some(0);
    }
    text.find(&format!("\n{prefix}")).map(|i| i + 1)
}

/// Component digests in lock order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Components {
    pub spec_s2_s5: String,
    pub stats_rs: String,
    pub analysis_plan: String,
    pub prompt_v1: String,
}

impl Components {
    /// Digest the given sources. `None` when the spec has no §2..§6 span.
    #[must_use]
    pub fn of(spec: &str, stats: &str, plan: &str, prompt: &str) -> Option<Self> {
        Some(Self {
            spec_s2_s5: sha256_hex(spec_sections_2_to_5(spec)?.as_bytes()),
            stats_rs: sha256_hex(stats.as_bytes()),
            analysis_plan: sha256_hex(plan.as_bytes()),
            prompt_v1: sha256_hex(prompt.as_bytes()),
        })
    }

    /// The components as compiled into this crate.
    #[must_use]
    pub fn in_tree() -> Option<Self> {
        Self::of(SPEC, STATS_RS, ANALYSIS_PLAN, PROMPT_V1)
    }

    /// The prereg sha.
    #[must_use]
    pub fn prereg_sha(&self) -> String {
        sha256_hex(self.canonical().as_bytes())
    }

    fn canonical(&self) -> String {
        format!(
            "{SCHEME}\nspec_s2_s5 {}\nstats_rs {}\nanalysis_plan {}\nprompt_v1 {}\n",
            self.spec_s2_s5, self.stats_rs, self.analysis_plan, self.prompt_v1
        )
    }

    /// Render a lock file (used once, by REX-00, to write `prereg.lock`).
    #[must_use]
    pub fn render_lock(&self) -> String {
        format!(
            "# PRM-001 pre-registration lock (rex-prereg-v2) over spec PRM-001 v3. Written by PRM-00b; v1 had 0 data rows.\n\
             # A mismatch against the tree is a new spec version (R-1), never an edit here.\n\
             spec_s2_s5={}\nstats_rs={}\nanalysis_plan={}\nprompt_v1={}\nprereg_sha={}\n",
            self.spec_s2_s5,
            self.stats_rs,
            self.analysis_plan,
            self.prompt_v1,
            self.prereg_sha()
        )
    }
}

/// Look up `key=value` in a lock body.
#[must_use]
pub fn lock_value<'a>(lock: &'a str, key: &str) -> Option<&'a str> {
    lock.lines()
        .filter(|l| !l.starts_with('#'))
        .find_map(|l| l.strip_prefix(key)?.strip_prefix('='))
        .map(str::trim)
}

/// Every mismatch between a lock and the given components (empty = verified).
#[must_use]
pub fn verify(lock: &str, c: &Components) -> Vec<String> {
    let want = [
        ("spec_s2_s5", c.spec_s2_s5.clone()),
        ("stats_rs", c.stats_rs.clone()),
        ("analysis_plan", c.analysis_plan.clone()),
        ("prompt_v1", c.prompt_v1.clone()),
        ("prereg_sha", c.prereg_sha()),
    ];
    want.iter()
        .filter_map(|(k, v)| match lock_value(lock, k) {
            Some(got) if got == v => None,
            Some(got) => Some(format!("{k}: lock {got} != tree {v}")),
            None => Some(format!("{k}: missing from lock")),
        })
        .collect()
}

/// The prereg sha recorded in the committed lock (what every receipt carries).
#[must_use]
pub fn locked_prereg_sha() -> Option<&'static str> {
    lock_value(LOCK, "prereg_sha")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FALSIFY-REX-PREREG-001: the committed lock matches the tree. Any edit to
    /// §2–§5, the analysis code, the plan or prompt v1 turns this RED.
    #[test]
    fn falsify_rex_prereg_001_lock_matches_tree() {
        let c = Components::in_tree().expect("spec has §2 and §6 headings");
        let bad = verify(LOCK, &c);
        assert!(
            bad.is_empty(),
            "REX-001 prereg drift (R-1: this is a new spec version, not a lock edit):\n{}\n\
             expected lock:\n{}",
            bad.join("\n"),
            c.render_lock()
        );
    }

    /// FALSIFY-REX-PREREG-002: a planted edit to §3 changes the prereg sha.
    #[test]
    fn falsify_rex_prereg_002_planted_section3_edit_is_caught() {
        let planted = SPEC.replacen("McNemar exact, one-sided", "t-test", 1);
        assert_ne!(planted, SPEC, "the §3 anchor text must exist");
        let c = Components::of(&planted, STATS_RS, ANALYSIS_PLAN, PROMPT_V1).expect("span");
        assert!(!verify(LOCK, &c).is_empty());
    }

    /// FALSIFY-REX-PREREG-003: edits OUTSIDE §2–§5 do not move the prereg sha.
    #[test]
    fn falsify_rex_prereg_003_status_line_edit_is_not_a_new_version() {
        let edited = SPEC.replacen(
            "**Runner:** the aprender traffic cop",
            "**Runner:** the fleet",
            1,
        );
        assert_ne!(edited, SPEC);
        let c = Components::of(&edited, STATS_RS, ANALYSIS_PLAN, PROMPT_V1).expect("span");
        assert!(verify(LOCK, &c).is_empty());
    }

    /// FALSIFY-REX-PREREG-004: the superseded v1 lock is untouched and the v1
    /// spec §2–§5, plan and prompt it froze still verify against it. (v1
    /// `stats_rs` is verified against git `5f0ae10ac`; the code moved on.)
    #[test]
    fn falsify_rex_prereg_004_v1_lock_is_frozen_and_verifies() {
        assert_eq!(
            lock_value(LOCK_V1, "prereg_sha"),
            Some("ef51087dc79bab0ad160e8a14f5b13e2ea43986b30c05dafc63c84c2dc21cdc0")
        );
        assert!(LOCK_V1.starts_with("# REX-001 pre-registration lock (rex-prereg-v1)."));
        let span = spec_sections_2_to_5(SPEC_V1).expect("v1 §2..§6");
        assert_eq!(
            lock_value(LOCK_V1, "spec_s2_s5"),
            Some(sha256_hex(span.as_bytes()).as_str())
        );
        assert_eq!(
            lock_value(LOCK_V1, "analysis_plan"),
            Some(sha256_hex(ANALYSIS_PLAN_V1.as_bytes()).as_str())
        );
        assert_eq!(
            lock_value(LOCK_V1, "prompt_v1"),
            Some(sha256_hex(PROMPT_V1.as_bytes()).as_str())
        );
        assert_ne!(
            lock_value(LOCK_V1, "prereg_sha"),
            locked_prereg_sha(),
            "v2 is a new lock"
        );
    }
    #[test]
    fn span_requires_both_headings() {
        assert!(spec_sections_2_to_5("## §2 a\nb\n## §6 c\n").is_some());
        assert_eq!(spec_sections_2_to_5("## §2 a\nb\n"), None);
        assert_eq!(spec_sections_2_to_5("x\n## §6 c\n## §2 a\n"), None);
        assert_eq!(
            spec_sections_2_to_5("pre\n## §2 a\nb\n## §6 c\n"),
            Some("## §2 a\nb\n")
        );
    }

    #[test]
    fn prompt_v1_is_the_g1_prompt() {
        assert!(PROMPT_V1.starts_with("You are a code reviewer. Review this diff."));
        assert!(PROMPT_V1.ends_with("Be concrete.\n\n"));
        assert_eq!(locked_prereg_sha().map(str::len), Some(64));
    }
}
