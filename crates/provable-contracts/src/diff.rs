//! Contract diff — detect drift between contract versions.
//!
//! Compares two [`Contract`] values and produces a [`ContractDiff`]
//! listing added, removed, and changed sections. Suggests a semver
//! bump (major, minor, patch) based on the nature of the changes.

use std::collections::BTreeSet;

use crate::schema::Contract;

/// The result of diffing two contracts.
#[derive(Debug, Clone)]
pub struct ContractDiff {
    /// Version in the "old" contract.
    pub old_version: String,
    /// Version in the "new" contract.
    pub new_version: String,
    /// Per-section change summaries.
    pub sections: Vec<SectionDiff>,
    /// Suggested semver bump based on the changes.
    pub suggested_bump: SemverBump,
}

/// A change summary for one section of the contract.
#[derive(Debug, Clone)]
pub struct SectionDiff {
    /// Section name (e.g. "equations", "`proof_obligations`").
    pub section: String,
    /// Items added in the new contract.
    pub added: Vec<String>,
    /// Items removed from the old contract.
    pub removed: Vec<String>,
    /// Items present in both but changed.
    pub changed: Vec<String>,
}

/// Methods for querying a section diff
impl SectionDiff {
    /// True if this section has no changes.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }
}

/// Suggested semantic version bump.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemverBump {
    /// No changes detected.
    None,
    /// Only additive or cosmetic changes.
    Patch,
    /// New equations, obligations, or tests added.
    Minor,
    /// Equations or obligations removed or semantically changed.
    Major,
}

/// Display the semver bump level as a lowercase string
impl std::fmt::Display for SemverBump {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Patch => write!(f, "patch"),
            Self::Minor => write!(f, "minor"),
            Self::Major => write!(f, "major"),
        }
    }
}

/// Diff two contracts and produce a structured change report.
pub fn diff_contracts(old: &Contract, new: &Contract) -> ContractDiff {
    let mut sections = Vec::new();
    let mut max_bump = SemverBump::None;

    // Equations
    let eq_diff = diff_keys(
        "equations",
        &old.equations.keys().cloned().collect(),
        &new.equations.keys().cloned().collect(),
    );
    let eq_changed = diff_equation_changes(old, new);
    let eq_section = SectionDiff {
        section: "equations".to_string(),
        added: eq_diff.0,
        removed: eq_diff.1,
        changed: eq_changed,
    };
    max_bump = bump_max(
        max_bump,
        section_bump(&eq_section, SemverBump::Major, SemverBump::Minor),
    );
    sections.push(eq_section);

    // Proof obligations
    let old_obs: BTreeSet<String> = old
        .proof_obligations
        .iter()
        .map(|o| format!("{}:{}", o.obligation_type, o.property))
        .collect();
    let new_obs: BTreeSet<String> = new
        .proof_obligations
        .iter()
        .map(|o| format!("{}:{}", o.obligation_type, o.property))
        .collect();
    let ob_diff = diff_sets("proof_obligations", &old_obs, &new_obs);
    max_bump = bump_max(
        max_bump,
        section_bump(&ob_diff, SemverBump::Major, SemverBump::Minor),
    );
    sections.push(ob_diff);

    // Falsification tests
    let old_ft: BTreeSet<String> = old
        .falsification_tests
        .iter()
        .map(|t| t.id.clone())
        .collect();
    let new_ft: BTreeSet<String> = new
        .falsification_tests
        .iter()
        .map(|t| t.id.clone())
        .collect();
    let ft_diff = diff_sets("falsification_tests", &old_ft, &new_ft);
    max_bump = bump_max(
        max_bump,
        section_bump(&ft_diff, SemverBump::Minor, SemverBump::Minor),
    );
    sections.push(ft_diff);

    // Kani harnesses
    let old_kh: BTreeSet<String> = old.kani_harnesses.iter().map(|h| h.id.clone()).collect();
    let new_kh: BTreeSet<String> = new.kani_harnesses.iter().map(|h| h.id.clone()).collect();
    let kh_diff = diff_sets("kani_harnesses", &old_kh, &new_kh);
    max_bump = bump_max(
        max_bump,
        section_bump(&kh_diff, SemverBump::Minor, SemverBump::Minor),
    );
    sections.push(kh_diff);

    // Enforcement rules
    let old_enf: BTreeSet<String> = old.enforcement.keys().cloned().collect();
    let new_enf: BTreeSet<String> = new.enforcement.keys().cloned().collect();
    let enf_diff = diff_sets("enforcement", &old_enf, &new_enf);
    max_bump = bump_max(
        max_bump,
        section_bump(&enf_diff, SemverBump::Minor, SemverBump::Patch),
    );
    sections.push(enf_diff);

    // Metadata version change
    if old.metadata.version != new.metadata.version {
        max_bump = bump_max(max_bump, SemverBump::Patch);
    }

    ContractDiff {
        old_version: old.metadata.version.clone(),
        new_version: new.metadata.version.clone(),
        sections,
        suggested_bump: max_bump,
    }
}

/// Compute added and removed keys between two sorted sets
fn diff_keys(
    _section: &str,
    old_keys: &BTreeSet<String>,
    new_keys: &BTreeSet<String>,
) -> (Vec<String>, Vec<String>) {
    let added: Vec<String> = new_keys.difference(old_keys).cloned().collect();
    let removed: Vec<String> = old_keys.difference(new_keys).cloned().collect();
    (added, removed)
}

/// Build a `SectionDiff` from the symmetric difference of two string sets
fn diff_sets(section: &str, old: &BTreeSet<String>, new: &BTreeSet<String>) -> SectionDiff {
    let added: Vec<String> = new.difference(old).cloned().collect();
    let removed: Vec<String> = old.difference(new).cloned().collect();
    SectionDiff {
        section: section.to_string(),
        added,
        removed,
        changed: Vec::new(),
    }
}

/// Detect field-level changes in equations present in both old and new contracts
fn diff_equation_changes(old: &Contract, new: &Contract) -> Vec<String> {
    let mut changed = Vec::new();
    for (name, old_eq) in &old.equations {
        if let Some(new_eq) = new.equations.get(name) {
            if old_eq.formula != new_eq.formula {
                changed.push(format!("{name}: formula changed"));
            }
            if old_eq.domain != new_eq.domain {
                changed.push(format!("{name}: domain changed"));
            }
            if old_eq.codomain != new_eq.codomain {
                changed.push(format!("{name}: codomain changed"));
            }
            if old_eq.invariants != new_eq.invariants {
                changed.push(format!("{name}: invariants changed"));
            }
        }
    }
    changed
}

/// Determine bump for a section: removals/changes → `on_break`, additions → `on_add`.
fn section_bump(section: &SectionDiff, on_break: SemverBump, on_add: SemverBump) -> SemverBump {
    if !section.removed.is_empty() || !section.changed.is_empty() {
        on_break
    } else if !section.added.is_empty() {
        on_add
    } else {
        SemverBump::None
    }
}

/// Return the higher-priority semver bump between current and candidate
fn bump_max(current: SemverBump, candidate: SemverBump) -> SemverBump {
    let rank = |b: SemverBump| match b {
        SemverBump::None => 0,
        SemverBump::Patch => 1,
        SemverBump::Minor => 2,
        SemverBump::Major => 3,
    };
    if rank(candidate) > rank(current) {
        candidate
    } else {
        current
    }
}

/// True if the diff contains no changes at all.
pub fn is_identical(diff: &ContractDiff) -> bool {
    diff.sections.iter().all(SectionDiff::is_empty) && diff.suggested_bump == SemverBump::None
}

#[cfg(test)]
mod tests {
    include!("diff_tests.rs");
}
