//! ONT-001 §3.4 — the one verdict lattice (ONT-6, PMAT-3451).
//!
//! ```text
//!    Pass       ← top; the only arming element
//!     |
//!   Unknown     ← one Reason (§3.4 order)
//!     |
//!    Fail       ← bottom; absorbing
//! ```
//!
//! `meet = min` is Kleene's strong conjunction (K3). Exit mapping: Pass→0, Fail→1, Unknown→2 with a
//! `decline: <reason>` line. Proof obligations, all discharged by the `#[cfg(kani)]` harnesses below and
//! exhaustively by the unit tests over the 17 elements:
//!
//! - KANI-ONT-6-1 — meet laws: commutative, associative, idempotent, Pass is the identity, Fail absorbs,
//!   and the meet is below both operands.
//! - KANI-ONT-6-2 — `arm(v) ⇒ v == Pass`.
//! - KANI-ONT-6-3 — every fleet label maps to exactly one element (no label shadows another).

use std::fmt;

/// Why a verdict is Unknown. Declaration order IS the lattice order among Unknowns (§3.4 order), so
/// `meet(Unknown(a), Unknown(b)) = Unknown(min(a, b))` is commutative by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reason {
    NotRun,
    Skip,
    Report,
    Warn,
    ToolAbsent,
    NotArmed,
    NoCheckable,
    WitnessStale,
    PositiveControlFailed,
    NoShapes,
    NoFocus,
    Differential,
    Advisory,
    ExtractorMissing,
    Prose,
}

impl Reason {
    /// Every reason, in lattice order.
    pub const ALL: [Self; 15] = [
        Self::NotRun,
        Self::Skip,
        Self::Report,
        Self::Warn,
        Self::ToolAbsent,
        Self::NotArmed,
        Self::NoCheckable,
        Self::WitnessStale,
        Self::PositiveControlFailed,
        Self::NoShapes,
        Self::NoFocus,
        Self::Differential,
        Self::Advisory,
        Self::ExtractorMissing,
        Self::Prose,
    ];
}

impl fmt::Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// A gate's verdict. Variant order is the lattice order (Fail < Unknown(_) < Pass), so the derived `Ord`
/// is the lattice and `meet` is `min`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Verdict {
    Fail,
    Unknown(Reason),
    Pass,
}

impl Verdict {
    /// The 17 elements: Fail, the 15 Unknowns in order, Pass.
    #[must_use]
    pub fn all() -> Vec<Self> {
        let mut v = vec![Self::Fail];
        v.extend(Reason::ALL.iter().map(|r| Self::Unknown(*r)));
        v.push(Self::Pass);
        v
    }

    /// K3 strong conjunction.
    #[must_use]
    pub fn meet(self, other: Self) -> Self {
        // RED (PMAT-3451): deliberately wrong until GREEN — returns the left operand.
        let _ = other;
        self
    }

    /// Only Pass arms a merge.
    #[must_use]
    pub fn arm(self) -> bool {
        // RED (PMAT-3451): deliberately wrong until GREEN — anything but Fail arms.
        self != Self::Fail
    }

    /// Process exit code: Pass→0, Fail→1, Unknown→2.
    #[must_use]
    pub fn exit_code(self) -> i32 {
        // RED (PMAT-3451): deliberately wrong until GREEN.
        0
    }

    /// The stderr line an Unknown prints (`decline: <reason>`); None for Pass and Fail.
    #[must_use]
    pub fn decline_line(self) -> Option<String> {
        // RED (PMAT-3451): deliberately wrong until GREEN.
        None
    }

    /// A `pv lint` gate's (passed, skipped) pair.
    #[must_use]
    pub fn from_gate(passed: bool, skipped: bool) -> Self {
        // RED (PMAT-3451): deliberately wrong until GREEN.
        let _ = (passed, skipped);
        Self::Pass
    }

    /// A SHACL shapes report (§3.4). Zero shapes or zero focus nodes is a decline, never an accept (R-2).
    #[must_use]
    pub fn from_shapes_report(
        violations: usize,
        warnings: usize,
        shapes_n: usize,
        focus_n: usize,
    ) -> Self {
        // RED (PMAT-3451): deliberately wrong until GREEN.
        let _ = (violations, warnings, shapes_n, focus_n);
        Self::Pass
    }

    /// Map one fleet label to its element (operator ruling 2026-09-17: "Existing reasons"). An unlisted
    /// spelling is refused (None), never guessed.
    #[must_use]
    pub fn from_label(label: &str) -> Option<Self> {
        // RED (PMAT-3451): deliberately wrong until GREEN.
        let _ = label;
        None
    }
}

/// The closed fleet vocabulary: `dogfood.sh mark`, lane-reduce, and pv's own verdict words.
pub const FLEET_LABELS: &[(&str, Verdict)] = &[
    ("PASS", Verdict::Pass),
    ("FAIL", Verdict::Fail),
    ("SKIP", Verdict::Unknown(Reason::Skip)),
    ("REPORT", Verdict::Unknown(Reason::Report)),
    ("WARN", Verdict::Unknown(Reason::Warn)),
    ("MANUAL", Verdict::Unknown(Reason::NotRun)),
    ("DEFER", Verdict::Unknown(Reason::NotRun)),
    ("NO-VERDICT", Verdict::Unknown(Reason::NotRun)),
    ("do-not-implement-as-written", Verdict::Fail),
    ("accept", Verdict::Pass),
    ("reject", Verdict::Fail),
    ("decline", Verdict::Unknown(Reason::NotRun)),
    ("error", Verdict::Unknown(Reason::ToolAbsent)),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seventeen_elements_in_lattice_order() {
        let all = Verdict::all();
        assert_eq!(all.len(), 17);
        assert!(
            all.windows(2).all(|w| w[0] < w[1]),
            "Fail < Unknown(NotRun) < … < Unknown(Prose) < Pass"
        );
    }

    #[test]
    fn meet_laws_exhaustive() {
        let all = Verdict::all();
        for &a in &all {
            assert_eq!(a.meet(a), a, "idempotent {a:?}");
            assert_eq!(a.meet(Verdict::Pass), a, "Pass is the identity {a:?}");
            assert_eq!(a.meet(Verdict::Fail), Verdict::Fail, "Fail absorbs {a:?}");
            for &b in &all {
                assert_eq!(a.meet(b), b.meet(a), "commutative {a:?} {b:?}");
                assert!(a.meet(b) <= a && a.meet(b) <= b, "below both {a:?} {b:?}");
                for &c in &all {
                    assert_eq!(
                        a.meet(b).meet(c),
                        a.meet(b.meet(c)),
                        "associative {a:?} {b:?} {c:?}"
                    );
                }
            }
        }
        assert_eq!(
            Verdict::Unknown(Reason::Warn).meet(Verdict::Unknown(Reason::Skip)),
            Verdict::Unknown(Reason::Skip),
            "two Unknowns meet at the lower reason"
        );
    }

    #[test]
    fn only_pass_arms() {
        for v in Verdict::all() {
            assert_eq!(v.arm(), v == Verdict::Pass, "{v:?}");
        }
    }

    #[test]
    fn exit_mapping_and_decline_line() {
        assert_eq!(Verdict::Pass.exit_code(), 0);
        assert_eq!(Verdict::Fail.exit_code(), 1);
        assert_eq!(Verdict::Pass.decline_line(), None);
        assert_eq!(Verdict::Fail.decline_line(), None);
        for r in Reason::ALL {
            assert_eq!(Verdict::Unknown(r).exit_code(), 2, "{r:?}");
            assert_eq!(
                Verdict::Unknown(r).decline_line(),
                Some(format!("decline: {r}"))
            );
        }
        assert_eq!(
            Verdict::Unknown(Reason::NotArmed).decline_line().as_deref(),
            Some("decline: NotArmed")
        );
    }

    #[test]
    fn gate_mapping() {
        assert_eq!(Verdict::from_gate(true, false), Verdict::Pass);
        assert_eq!(Verdict::from_gate(false, false), Verdict::Fail);
        assert_eq!(
            Verdict::from_gate(false, true),
            Verdict::Unknown(Reason::Skip)
        );
        assert_eq!(
            Verdict::from_gate(true, true),
            Verdict::Unknown(Reason::Skip),
            "skipped wins"
        );
    }

    #[test]
    fn shapes_report_mapping() {
        assert_eq!(
            Verdict::from_shapes_report(0, 0, 3, 5),
            Verdict::Pass,
            "conforms"
        );
        assert_eq!(
            Verdict::from_shapes_report(1, 0, 3, 5),
            Verdict::Fail,
            "any violation"
        );
        assert_eq!(
            Verdict::from_shapes_report(2, 4, 3, 5),
            Verdict::Fail,
            "violations beat warnings"
        );
        assert_eq!(
            Verdict::from_shapes_report(0, 1, 3, 5),
            Verdict::Unknown(Reason::Warn),
            "warnings only"
        );
        assert_eq!(
            Verdict::from_shapes_report(0, 0, 0, 5),
            Verdict::Unknown(Reason::NoShapes),
            "no shapes"
        );
        assert_eq!(
            Verdict::from_shapes_report(0, 0, 3, 0),
            Verdict::Unknown(Reason::NoFocus),
            "no focus"
        );
        assert_eq!(
            Verdict::from_shapes_report(1, 0, 0, 0),
            Verdict::Unknown(Reason::NoShapes),
            "zero first (R-2)"
        );
    }

    #[test]
    fn every_fleet_label_maps_to_exactly_one_element() {
        for (i, (label, v)) in FLEET_LABELS.iter().enumerate() {
            assert_eq!(Verdict::from_label(label), Some(*v), "{label}");
            assert!(
                FLEET_LABELS[..i].iter().all(|(l, _)| l != label),
                "duplicate label {label}"
            );
        }
        for unlisted in ["pass", "Pass", "NOT-RUN", "not_run", "", "BLIND"] {
            assert_eq!(
                Verdict::from_label(unlisted),
                None,
                "unlisted spelling {unlisted:?} is refused"
            );
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    fn any_verdict() -> Verdict {
        let all = Verdict::all();
        let i: usize = kani::any();
        kani::assume(i < all.len());
        all[i]
    }

    /// KANI-ONT-6-1: meet laws.
    #[kani::proof]
    #[kani::unwind(20)]
    fn kani_ont_6_1() {
        let (a, b, c) = (any_verdict(), any_verdict(), any_verdict());
        assert_eq!(a.meet(b), b.meet(a));
        assert_eq!(a.meet(b).meet(c), a.meet(b.meet(c)));
        assert_eq!(a.meet(a), a);
        assert_eq!(a.meet(Verdict::Pass), a);
        assert_eq!(a.meet(Verdict::Fail), Verdict::Fail);
        assert!(a.meet(b) <= a && a.meet(b) <= b);
    }

    /// KANI-ONT-6-2: arm(v) ⇒ v == Pass.
    #[kani::proof]
    #[kani::unwind(20)]
    fn kani_ont_6_2() {
        let v = any_verdict();
        if v.arm() {
            assert_eq!(v, Verdict::Pass);
        }
    }

    /// KANI-ONT-6-3: every fleet label maps to exactly one element.
    #[kani::proof]
    #[kani::unwind(16)]
    fn kani_ont_6_3() {
        let i: usize = kani::any();
        kani::assume(i < FLEET_LABELS.len());
        let (label, v) = FLEET_LABELS[i];
        assert_eq!(Verdict::from_label(label), Some(v));
    }
}
