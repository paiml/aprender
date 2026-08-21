//! The competitive-parity LEDGER (`metadata.kind: competitive-parity`).
//!
//! # Why the ledger is shaped like this
//!
//! The operator's mandate — "every entry point must be proven equal or better"
//! — cannot be enforced as literally worded, because a rule that admits only
//! wins makes **deleting a losing comparison the cheapest compliant action**.
//! This repository has already done exactly that: the StandardScaler speed beat
//! measured `1.443` (apr 0.69×, a LOSS) on the canonical Intel CI runner and was
//! deleted the same day under PMAT-733 (commit `d7e08043b`, squashed as
//! `edfa106d0`, PR #2040) — test, contract and both nightly workflow steps, 395
//! deletions. `git log --all --diff-filter=D` over `crates/*/tests/beat_*` and
//! `contracts/beat-*` returns that ONE commit, and it removed the ONLY two
//! losing rows in the history. Deletion is not a hypothetical failure mode here.
//!
//! So the gate is INVERTED. It checks that a FRESH, DATED verdict EXISTS, drawn
//! from a closed vocabulary. It never checks that the verdict says `BETTER`.
//!
//! * [`Verdict::Worse`] and [`Verdict::Unmeasured`] are FIRST-CLASS and
//!   recordable. Recording a loss is compliant; deleting it is not.
//! * Staleness blocks. The verdict VALUE does not.
//! * Deleting a row lowers `__MEASURED__`, which the ratchet
//!   (`scripts/check_competitive_parity.sh`) refuses.
//!
//! # Freshness is evaluated at CHECK TIME, for every verdict class
//!
//! The first design bounded only `UNMEASURED` rows. That is backwards:
//! `MEASURED` is exactly where both withdrawn claims lived. `apr` published
//! "1.371× faster decode than ollama" for eight weeks after the measurement
//! stopped reproducing, and
//! `contracts/beat-ollama-decode-throughput-speed-v1.yaml` re-pinned
//! `baseline_value` from 1.3710 to 1.0150 out of a 2026-07-31 run while leaving
//! `baseline_sourced_date: "2026-06-15"` untouched — the one contract in the
//! repo whose baseline was actually refreshed has a freshness field that lies
//! about it. Every one of the 24 dated beat contracts is 48–71 days old and not
//! one carries an expiry.
//!
//! Hence [`ParityRow::valid_until`] is REQUIRED on every row regardless of
//! verdict, and [`ParityRow::effective_verdict`] degrades an expired row to
//! [`Verdict::Unmeasured`] — which lowers `__MEASURED__`, which fails the
//! ratchet. That is the mechanism that would have caught 1.371×.
//!
//! # EVERY DATE IS ANCHORED TO TODAY — the class, not the three instances
//!
//! Three defects were found and fixed here before this one, and every fix
//! introduced a NEW author-supplied field that nothing bounded against today:
//!
//! | fix | the new field | how it was bypassed |
//! |---|---|---|
//! | set ratchet | the emitted `__ROW__` key | `.trim()` on the way out, and a newline in a key injected extra key lines |
//! | `valid_until` cap | `measured_on` | PARITY-011 bounds a DIFFERENCE; both ends were author-supplied, so `2099-01-01`/`2099-06-01` sat inside it |
//! | downgrade record | `recheck_by` | validated against `recorded_on`, never against today, so the debt never came due |
//!
//! Patching a fourth pairwise ceiling would have produced a fourth. The rule
//! adopted instead is a single invariant, applied uniformly:
//!
//! > **No date anywhere in this document may be more than
//! > [`MAX_FUTURE_DAYS`] days after TODAY.** Not after another date the same
//! > author wrote — after TODAY, the one anchor an author does not control.
//!
//! It is enforced in two places, deliberately overlapping:
//!
//! * [`LedgerDate`] refuses an out-of-horizon value at **parse** time, so the
//!   four fields that exist today cannot HOLD one. This is poka-yoke: a field
//!   declared `LedgerDate` is bounded by existing, the way [`Verdict`] is
//!   closed by being an enum rather than by a lint that checks the spelling.
//! * `PARITY-016` sweeps the whole **serialized document** — every scalar, at
//!   any depth, under any key — because a newtype only helps a field that USES
//!   it, and the next author writing `certified_on: String` is the next
//!   exemption. The sweep is keyed on the VALUE, so a field that does not
//!   exist yet is bounded before it is written. `#[serde(flatten)] extra` on
//!   each struct captures keys serde does not know instead of dropping them,
//!   so a date added to the YAML alone is swept too.
//!
//! `PARITY-017` then pins the dates that record a PAST event (`measured_on`,
//! `recorded_on`) to the past, and [`Downgrade::is_overdue`] makes the escape
//! valve come due. The three compose: no row can be fresh for more than 180
//! days from today, and no downgrade can excuse one for longer, whatever the
//! file says.
//!
//! The horizon is MONOTONE in time — a date only ever becomes less future — so
//! a document that parses today parses forever after.

use std::collections::{BTreeSet, HashSet};

use serde::{Deserialize, Serialize};

/// The FUTURE HORIZON, in days after **today**, that any date in this ledger
/// may reach.
///
/// # Why this is a horizon and not another pairwise ceiling
///
/// Three defects were fixed in this file before this one, and every fix
/// introduced a NEW author-supplied date bounded only against ANOTHER
/// author-supplied date:
///
/// * `valid_until` was capped at [`MAX_VALIDITY_DAYS`] from `measured_on` —
///   but `measured_on` is author-supplied and was itself unbounded, so
///   `measured_on: 2099-01-01` with `valid_until: 2099-06-01` sat inside the
///   ceiling and 2099 was STILL the exemption;
/// * `recheck_by` was capped from `recorded_on` — same shape, same hole, so
///   the escape valve never came due;
/// * and the next field of that shape would have been the next exemption.
///
/// A difference between two numbers the same author writes bounds NOTHING.
/// The only anchor an author does not control is **today**, so that is the
/// anchor: no date in this ledger may be more than `MAX_FUTURE_DAYS` days
/// after the UTC date on which the document is READ. [`LedgerDate`] enforces
/// it at PARSE time, so an out-of-horizon value cannot be constructed, and
/// `PARITY-016` sweeps the whole serialized document, so a field added LATER —
/// by anyone, under any name, of any type — is bounded before it is written.
///
/// The number is [`MAX_VALIDITY_DAYS`] because the bounds then COMPOSE into
/// the property that actually matters: with `measured_on <= today`
/// (PARITY-017), `valid_until - measured_on <= 180` (PARITY-011) and every
/// date inside this horizon, no row can be fresh for more than 180 days from
/// today whatever the file says.
///
/// The bound is MONOTONE in time — a date only ever becomes less future — so a
/// document that parses today parses forever after. It can never turn a
/// previously accepted file red by the clock moving on.
pub const MAX_FUTURE_DAYS: i64 = MAX_VALIDITY_DAYS;

/// An ISO `YYYY-MM-DD` date that **cannot hold a far-future value**.
///
/// This is the class control, and it is a TYPE rather than a validator rule on
/// purpose: a rule is something the next author must remember to write for the
/// next field, and the record here is that they will not. A field declared
/// `LedgerDate` is bounded by existing, and `serde` refuses the document
/// otherwise — the same mechanism that makes [`Verdict`] a closed vocabulary
/// rather than a lint.
///
/// # What it does NOT reject
///
/// A string that is not a well-formed ISO date is ACCEPTED here and reported
/// by `PARITY-007` / `PARITY-013`, which name the field and quote the value.
/// That is deliberate: rejecting it at parse would replace several precise
/// diagnostics with one YAML error, and an unparseable date is not a futurity
/// exemption — [`ParityRow::is_expired`] fails CLOSED on it, so it expires the
/// row rather than blessing it. The only thing this type refuses is the one
/// thing no downstream rule can recover from: a syntactically perfect date far
/// enough ahead to outlive review.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LedgerDate(String);

impl LedgerDate {
    /// The raw string as written.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The string with surrounding whitespace removed — what every rule
    /// compares.
    #[must_use]
    pub fn trim(&self) -> &str {
        self.0.trim()
    }

    /// True when nothing was written at all (the `#[serde(default)]` case).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.trim().is_empty()
    }

    /// Construct WITHOUT the horizon check. Test-only, and named so a reader
    /// of a diff can see it: production values arrive through `Deserialize`,
    /// which cannot skip the bound.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn unchecked(s: &str) -> Self {
        Self(s.to_string())
    }

    /// How many days past the horizon `s` is, or `None` when it is inside it
    /// (or is not an ISO date at all, or `today` is unusable).
    ///
    /// Shared by [`LedgerDate`]'s `Deserialize` and by `PARITY-016`, so the
    /// type bound and the document sweep cannot disagree about where the line
    /// is.
    #[must_use]
    pub fn overshoot(s: &str, today: &str) -> Option<i64> {
        let s = s.trim();
        parse_iso_date(s)?;
        let ahead = days_between(today, s)?;
        (ahead > MAX_FUTURE_DAYS).then_some(ahead - MAX_FUTURE_DAYS)
    }
}

impl std::fmt::Debug for LedgerDate {
    /// Prints as the bare string, so `{:?}` in a diagnostic quotes what the
    /// author wrote rather than `LedgerDate("…")`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

impl std::fmt::Display for LedgerDate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for LedgerDate {
    /// Transparent: the serialized document must contain the bare date string,
    /// because `PARITY-016` sweeps that document and a wrapper object would
    /// hide the scalar from the sweep.
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for LedgerDate {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        let raw = String::deserialize(d)?;
        // Fail CLOSED on a broken clock. "Cannot tell how far ahead this is"
        // must never resolve to "accept it": that is the shape of every gate
        // in this repo that reported success while measuring nothing.
        let today = today_utc().ok_or_else(|| {
            D::Error::custom(
                "system clock is before the Unix epoch, so no date in this ledger can be \
                 bounded against today - refusing to parse rather than accept an unbounded \
                 date",
            )
        })?;
        if let Some(over) = Self::overshoot(&raw, &today) {
            return Err(D::Error::custom(format!(
                "date {:?} is {over} day(s) past the {MAX_FUTURE_DAYS}-day horizon from today \
                 ({today}). Every date in a competitive-parity ledger is bounded against TODAY, \
                 not against another date the same author wrote: bounding a DIFFERENCE leaves \
                 both ends free, which is how `measured_on: 2099-01-01` would keep a row \
                 permanently fresh inside a 180-day window. Re-date the row, or record it \
                 UNMEASURED",
                raw.trim(),
            )));
        }
        Ok(Self(raw))
    }
}

/// The CLOSED verdict vocabulary. A value outside this set is a parse error,
/// not a lint — `serde` rejects the document, so no ledger can invent a verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Verdict {
    /// apr is measurably ahead of the competitor on this dimension.
    Better,
    /// The two are indistinguishable within the measurement's own noise.
    Parity,
    /// The competitor is ahead. **Recordable, and compliant.**
    Worse,
    /// The competitor has no counterpart command/property, so no ratio is
    /// meaningful (e.g. `apr qa` — nothing in llama.cpp or ollama runs
    /// falsifiable QA gates on a model artifact).
    NotComparable,
    /// We assert nothing. This is what an expired row degrades to, and it is
    /// also the honest home of a published claim with no receipt — e.g.
    /// "llama.cpp ~1.55× faster (431 vs 277 tok/s)", whose numbers were BORN in
    /// `docs/BEATS.md` (`git log -S"431 vs 277"` returns only the docs commit
    /// that introduced them).
    Unmeasured,
}

impl Verdict {
    /// Whether this verdict counts toward `__MEASURED__`.
    ///
    /// Everything except [`Verdict::Unmeasured`] counts — including
    /// [`Verdict::Worse`]. That is the whole inversion: a recorded loss is a
    /// measurement, and removing it costs you a point on the ratchet.
    #[must_use]
    pub fn is_measured(self) -> bool {
        self != Self::Unmeasured
    }

    /// The canonical spelling, as it must appear in YAML.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Better => "BETTER",
            Self::Parity => "PARITY",
            Self::Worse => "WORSE",
            Self::NotComparable => "NOT_COMPARABLE",
            Self::Unmeasured => "UNMEASURED",
        }
    }
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The ledger block: one row per in-scope entry point.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParityLedger {
    /// How the in-scope universe was derived, so a shrinking `__IN_SCOPE__` can
    /// be argued against something.
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub rows: Vec<ParityRow>,
    /// RECORDED DOWNGRADES — the escape valve that keeps honesty affordable.
    ///
    /// A pure shrink-never floor on `__MEASURED__` mechanically FORBIDS the
    /// honest correction: the `apr code` row cites
    /// `evidence/phase-5/arena-scores.json`, which does not exist in this
    /// repository, and its own `note:` says it should be `UNMEASURED` — yet
    /// filing it as `UNMEASURED` would breach the floor. A ratchet that
    /// punishes increasing honesty produces dishonest ledgers, which is the
    /// same failure as PMAT-733 approached from the opposite side.
    ///
    /// So the two properties are SEPARATED:
    ///
    /// * the SET of rows that exist may never shrink (delete a row and the
    ///   ratchet names the missing key, whatever the totals say);
    /// * the SET of rows whose verdict is MEASURED may shrink, but ONLY with a
    ///   [`Downgrade`] naming the row, a reason from a CLOSED vocabulary, an
    ///   owner, and a date by which it must be re-measured.
    ///
    /// The reason is a serde enum, not prose: an unknown reason is a PARSE
    /// error, so "recorded a reason" cannot be discharged by writing a
    /// sentence. And [`Downgrade::entry_point`] must match a row that is still
    /// present (PARITY-012), so deleting a row can never be laundered as a
    /// downgrade.
    #[serde(default)]
    pub downgrades: Vec<Downgrade>,
    /// RECORDED REMOVALS — the receipt an entry point's disappearance costs.
    ///
    /// See [`Removal`]. The short version: the ratchet excused a deleted row
    /// whenever the entry point was absent from the live enumeration, and the
    /// live enumeration is produced by a binary built FROM THE BRANCH — so the
    /// author wrote the excuse. Deleting the subcommand, its scope line and
    /// its row in one commit removed a losing comparison at rc=0 with nothing
    /// recorded. Retirement stays possible; it stops being free.
    ///
    /// `Vec` with `#[serde(default)]`, never `Option`, for the reason recorded
    /// on [`ParityLedger::coverage`]: an older ledger on protected `main` has
    /// no `removals:` key and MUST keep parsing, or the comparand collapses to
    /// "no prior state", which reads as BOOTSTRAP — the strongest possible
    /// pass — produced by a schema edit.
    #[serde(default)]
    pub removals: Vec<Removal>,
    /// The COVERAGE RATCHET — see [`CoverageRatchet`].
    ///
    /// `Option`, and required by `PARITY-021` rather than by the type, ON
    /// PURPOSE. The comparand for every other rule here is the ledger as it
    /// exists on protected `main`, and that ledger has to keep PARSING under a
    /// newer schema or the comparison silently becomes "no prior state" —
    /// which is the vacuous-comparand failure this whole design exists to
    /// close. A newly-required field expressed as a non-`Option` type would
    /// make every older `main` ledger a PARSE error, and a parse error emits
    /// no sets at all. So new blocks arrive as `Option` + a validator rule:
    /// the CURRENT tree is refused loudly, and the PRIOR tree is still
    /// readable as a comparand.
    #[serde(default)]
    pub coverage: Option<CoverageRatchet>,
    /// Every key in the YAML that this struct does not name — see
    /// [`Downgrade::extra`].
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, serde_yaml::Value>,
}

/// The COVERAGE RATCHET: a dated, author-visible schedule of how much of the
/// in-scope surface must carry a ledger row.
///
/// # The question this answers, and why it could not be deferred
///
/// Five rows over 41 scope entries over 111 live subcommands. "Competitive
/// parity" was being asserted over ~4.5% of the CLI surface, forever, with
/// every gate green — because nothing anywhere required an in-scope entry
/// point to HAVE a row. Every rule in this file makes the rows that exist
/// honest; none of them makes rows exist. A ledger of five impeccable rows
/// that never grows is the same failure as a ledger of five dishonest ones,
/// arrived at by omission instead of by fabrication.
///
/// # Why a SCHEDULE and not a ratio
///
/// A ratio (`covered / in_scope >= x`) tracks scope growth automatically, and
/// that is exactly its problem: the denominator is a quantity the author also
/// controls, so a ratio is payable by shrinking scope. Scope shrinking is
/// separately guarded, but a floor whose numerator AND denominator are both
/// author-influenced is a floor with two levers on it. An absolute count has
/// one lever, is readable without arithmetic, and — being an integer in a
/// reviewed file — cannot drift by a hundredth of a point per release.
///
/// The dissent is recorded rather than argued away: an absolute count does NOT
/// track scope growth, so a release that adds 20 subcommands dilutes the
/// achieved ratio while the gate stays green. See the `dissent:` field, which
/// the contract carries in prose so a reader meets it at the same time as the
/// rule.
///
/// # What makes it RATCHET rather than sit still
///
/// `steps` is a strictly increasing schedule of `(by, covered_min)` pairs.
/// `PARITY-023` requires at least one step dated in the FUTURE, so the ledger
/// always owes an increase; and because every date in this document is bounded
/// at [`MAX_FUTURE_DAYS`] from today, that future step can never be more than
/// half a year out. The schedule therefore has to be renewed — visibly, in a
/// reviewed diff, against a protected-`main` comparand that refuses to let a
/// step be deleted or dated later.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoverageRatchet {
    /// Why this shape was chosen. Prose; read by humans, required by
    /// `PARITY-021` so the decision cannot arrive unexplained.
    #[serde(default)]
    pub rationale: Option<String>,
    /// The case AGAINST this shape, in the same file as the rule.
    ///
    /// Required (`PARITY-021`) because a floor with no recorded objection
    /// reads as unanimous, and the next author has no way to tell whether the
    /// obvious alternative was rejected or never considered.
    #[serde(default)]
    pub dissent: Option<String>,
    /// The schedule, strictly increasing in both `by` and `covered_min`.
    #[serde(default)]
    pub steps: Vec<CoverageStep>,
    /// Keys this struct does not name — see [`Downgrade::extra`]. Present so
    /// `PARITY-016` sweeps dates written into fields the schema has never
    /// heard of.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, serde_yaml::Value>,
}

/// One step of the [`CoverageRatchet`]: from `by` onward, at least
/// `covered_min` distinct in-scope entry points must carry a row, and at
/// least `scope_min` entry points must BE in scope.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoverageStep {
    /// The date this step COMES DUE. A [`LedgerDate`], so it is bounded
    /// against today at parse time like every other date here.
    #[serde(default)]
    pub by: LedgerDate,
    /// Distinct covered entry points required from `by` onward.
    #[serde(default)]
    pub covered_min: usize,
    /// Entry points required to BE IN SCOPE from `by` onward — the SECOND
    /// joint.
    ///
    /// # The joint this closes
    ///
    /// `covered_min` bounds rows against SCOPE. Nothing bounded scope against
    /// the WORLD, and the measured shape was 5 rows over 41 scope entries over
    /// 111 live subcommands: bounding only the first joint leaves the whole
    /// claim payable by never widening the audited surface. A release that
    /// adds twenty subcommands dilutes the real ratio while `covered_min` is
    /// still met exactly — the first objection recorded in this ledger's own
    /// `dissent:`, now answered rather than only noted.
    ///
    /// # Why an ABSOLUTE COUNT again, and not `scope / live_universe`
    ///
    /// The live universe is enumerated from a binary built FROM THE BRANCH, so
    /// the author writes it: a ratio against it is payable by DELETING a
    /// subcommand, which is PMAT-733 with the arithmetic done from the far
    /// end. An absolute integer has no denominator to shrink. Deletion is
    /// separately made expensive (a `removals:` record plus the entry point
    /// genuinely being gone), and the two controls compose: a deletion that
    /// drops scope below this floor is refused by THIS rule even when the
    /// record is present and honest.
    ///
    /// `Option`, and required by `PARITY-022` rather than by the type, for the
    /// reason recorded on [`ParityLedger::coverage`]: a newly-required field
    /// expressed in the TYPE would make every older `main` ledger a parse
    /// error, and a parse error emits no sets at all — which reads as
    /// BOOTSTRAP, the strongest possible pass.
    #[serde(default)]
    pub scope_min: Option<usize>,
    /// Keys this struct does not name — see [`Downgrade::extra`].
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, serde_yaml::Value>,
}

impl CoverageRatchet {
    /// The highest `covered_min` whose step has COME DUE as of `today`.
    ///
    /// Zero when no step has come due yet, which is what makes the mechanism
    /// landable on the day it is introduced: the first step may be dated
    /// today at the coverage the ledger already has.
    ///
    /// An unparseable `by` counts as DUE (fail closed) rather than as
    /// not-yet-due: a date nobody can read must not buy a deferral.
    #[must_use]
    pub fn floor_as_of(&self, today: &str) -> usize {
        self.steps
            .iter()
            .filter(|s| !s.is_future(today))
            .map(|s| s.covered_min)
            .max()
            .unwrap_or(0)
    }

    /// The highest `scope_min` whose step has COME DUE as of `today`.
    ///
    /// Zero when no step that has come due declares one, which is what keeps
    /// the field landable as an `Option` on a schema that older `main` ledgers
    /// must keep parsing. `PARITY-022` refuses a schedule whose steps omit it,
    /// so the zero is reachable only from a comparand, never from a tree under
    /// test.
    ///
    /// An unparseable `by` counts as DUE (fail closed), exactly as in
    /// [`CoverageRatchet::floor_as_of`]: a date nobody can read must not buy a
    /// deferral.
    #[must_use]
    pub fn scope_floor_as_of(&self, today: &str) -> usize {
        self.steps
            .iter()
            .filter(|s| !s.is_future(today))
            .filter_map(|s| s.scope_min)
            .max()
            .unwrap_or(0)
    }

    /// The nearest step still in the FUTURE as of `today`, if any.
    ///
    /// `PARITY-023` refuses a schedule with none: a ratchet that owes nothing
    /// has stopped ratcheting, and the failure mode of a coverage floor is
    /// precisely that it is set once at the achievable value and never moved.
    #[must_use]
    pub fn next_step(&self, today: &str) -> Option<&CoverageStep> {
        self.steps
            .iter()
            .filter(|s| s.is_future(today))
            .min_by(|a, b| a.by.trim().cmp(b.by.trim()))
    }

    /// The highest `covered_min` anywhere in the schedule.
    #[must_use]
    pub fn ceiling(&self) -> usize {
        self.steps.iter().map(|s| s.covered_min).max().unwrap_or(0)
    }
}

impl CoverageStep {
    /// Is this step still in the future as of `today`?
    ///
    /// Fails CLOSED: a `by` that is not an ISO date is NOT in the future, so
    /// it counts toward the floor immediately rather than deferring it.
    #[must_use]
    pub fn is_future(&self, today: &str) -> bool {
        days_between(today, self.by.trim()).is_some_and(|d| d > 0)
    }
}

/// Why a row's VERDICT changed, from a CLOSED vocabulary. Prose is not
/// accepted; an unknown value fails to parse.
///
/// # Why this vocabulary covers more than downgrades now
///
/// The first version of this enum named only the ways a measurement can STOP
/// being one, because the only guarded transition was `MEASURED ->
/// UNMEASURED`. That left every OTHER relabelling free, and the free ones were
/// the cheap ones: re-declaring both `WORSE` rows as `NOT_COMPARABLE` left
/// `__ROWS__`, `__MEASURED__` and `__NON_WINS__` bit-for-bit identical while
/// the StandardScaler 0.69x loss and the Lasso ~19x loss both became "no
/// counterpart exists". Every total held; the DIRECTION of the result left the
/// tree. That is PMAT-733 paid for in a currency the ratchet did not count.
///
/// So the record generalises from "downgrade" to "verdict transition", and the
/// vocabulary has to be able to name an honest UPWARD move too — otherwise the
/// only way to record a genuine re-measurement would be to lie about the
/// reason, and a ratchet that punishes honesty produces dishonest ledgers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DowngradeReason {
    /// The cited receipt does not exist where it is cited, so the number is a
    /// remembered one. This is the `apr code` case exactly.
    ReceiptMissing,
    /// The harness that produced the number is no longer in the tree, so the
    /// measurement cannot be reproduced without recovering it.
    HarnessDeleted,
    /// No date is attached to the measurement anywhere, so its freshness is
    /// unknowable rather than merely old.
    MeasurementUndated,
    /// The competitor build that produced the number was never captured, so
    /// the comparison has no fixed oracle.
    CompetitorUnpinnable,
    /// The host class the measurement requires is not available to re-run it.
    HostUnavailable,
    /// The comparison has been replaced by a different, better-specified row.
    Superseded,
    /// The comparison was RE-RUN and the number moved. The only reason that
    /// can justify a transition INTO a better verdict, and the one that has to
    /// exist for the honest fix to be recordable at all.
    Remeasured,
    /// The competitor turned out to have no counterpart for this dimension, so
    /// no ratio is meaningful. This is the honest route to `NOT_COMPARABLE` —
    /// and naming it is exactly what makes the dishonest route expensive:
    /// re-declaring a measured LOSS as `NOT_COMPARABLE` now has to be written
    /// down, owned, and given a date on which it is re-examined.
    NoCounterpart,
    /// The comparison itself was re-specified (different invocation, host
    /// class, or dimension), so the old verdict is about a different question.
    ComparisonRespecified,
}

impl DowngradeReason {
    /// The canonical spelling, as it must appear in YAML.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReceiptMissing => "RECEIPT_MISSING",
            Self::HarnessDeleted => "HARNESS_DELETED",
            Self::MeasurementUndated => "MEASUREMENT_UNDATED",
            Self::CompetitorUnpinnable => "COMPETITOR_UNPINNABLE",
            Self::HostUnavailable => "HOST_UNAVAILABLE",
            Self::Superseded => "SUPERSEDED",
            Self::Remeasured => "REMEASURED",
            Self::NoCounterpart => "NO_COUNTERPART",
            Self::ComparisonRespecified => "COMPARISON_RESPECIFIED",
        }
    }

    /// Every reason, for diagnostics that must list the closed vocabulary.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::ReceiptMissing,
            Self::HarnessDeleted,
            Self::MeasurementUndated,
            Self::CompetitorUnpinnable,
            Self::HostUnavailable,
            Self::Superseded,
            Self::Remeasured,
            Self::NoCounterpart,
            Self::ComparisonRespecified,
        ]
    }

    /// The vocabulary rendered as ` / `-separated canonical spellings.
    #[must_use]
    pub fn vocabulary() -> String {
        Self::all()
            .iter()
            .map(|r| r.as_str())
            .collect::<Vec<_>>()
            .join(" / ")
    }
}

impl std::fmt::Display for DowngradeReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A recorded, dated, owned VERDICT TRANSITION for one row.
///
/// # The generalisation, and why it is a generalisation rather than a sibling
///
/// This started life as a `MEASURED -> UNMEASURED` downgrade record, because
/// that was the only transition anything guarded. It was also the only
/// EXPENSIVE one. Every other relabelling was free, and the cheapest of them
/// undid the whole point of the ledger: declare both `WORSE` rows
/// `NOT_COMPARABLE` and `__ROWS__`, `__MEASURED__` and `__NON_WINS__` are all
/// unchanged — 5, 3 and 5 before and after — while the StandardScaler 0.69x
/// and the Lasso ~19x losses have become "the competitor has no counterpart".
/// The totals are conserved and the DIRECTION of the result is gone, which is
/// what PMAT-733 was actually about.
///
/// The fix is NOT to gate on the verdict's VALUE. The gate deliberately never
/// checks that a verdict says `BETTER`, because a rule admitting only wins
/// makes deleting a losing comparison the cheapest compliant action — that is
/// the inversion this whole mechanism rests on, and it is load-bearing. What
/// is gated is the TRANSITION: the value stays unconstrained, and CHANGING it
/// stops being free.
///
/// So one record type covers both, because they are the same act — a row's
/// declared verdict moved and somebody must own that:
///
/// * `to_verdict` absent ⇒ the legacy shape: a downgrade to `UNMEASURED`,
///   which is what `PARITY-014` has always required.
/// * `to_verdict` present ⇒ it must equal the row's declared verdict, and
///   `from_verdict` must name where it moved from. The shell ratchet
///   cross-checks that pair against the verdict recorded in the COMMITTED
///   baseline, so a record cannot excuse a transition it does not describe.
///
/// A second mechanism beside this one would have meant two vocabularies, two
/// expiry rules and two sets of paperwork, of which exactly one would have
/// been kept current.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Downgrade {
    /// Must EXACTLY match a [`ParityRow::entry_point`] still present in the
    /// ledger (PARITY-012). Deleting the row dangles the downgrade, so a
    /// deletion can never be dressed up as an honest correction.
    #[serde(default)]
    pub entry_point: String,
    /// `None` ⇒ PARITY-013; an unknown STRING is a parse error.
    #[serde(default)]
    pub reason: Option<DowngradeReason>,
    /// The verdict this row USED to declare. Required whenever
    /// [`Downgrade::to_verdict`] is present (PARITY-019), and cross-checked by
    /// the shell ratchet against the verdict in the committed baseline: a
    /// record claiming `WORSE -> NOT_COMPARABLE` does not excuse a
    /// `PARITY -> BETTER` move.
    #[serde(default)]
    pub from_verdict: Option<Verdict>,
    /// The verdict this row declares NOW. Absent ⇒ the legacy downgrade shape,
    /// which `PARITY-014` reads as "the row must declare `UNMEASURED`".
    #[serde(default)]
    pub to_verdict: Option<Verdict>,
    /// Who owes the re-measurement.
    #[serde(default)]
    pub owner: String,
    /// ISO `YYYY-MM-DD` the downgrade was recorded. Bounded against TODAY by
    /// its TYPE, and additionally forbidden from being in the future at all
    /// (PARITY-017) — a record cannot have been made tomorrow.
    #[serde(default)]
    pub recorded_on: LedgerDate,
    /// ISO `YYYY-MM-DD` by which the row must be re-measured or the downgrade
    /// re-argued.
    ///
    /// This was the third bypass: it was bounded only against `recorded_on`,
    /// which is author-supplied, so the escape valve never came due. It is now
    /// a [`LedgerDate`] — bounded against TODAY — as well as being capped from
    /// `recorded_on`, and PARITY-018 makes an OVERDUE recheck degrade the row
    /// the same way an expired `valid_until` does.
    #[serde(default)]
    pub recheck_by: LedgerDate,
    /// Free-text elaboration. NOT the machine-checkable part.
    #[serde(default)]
    pub detail: Option<String>,
    /// Every key in the YAML that this struct does not name.
    ///
    /// Captured rather than DROPPED so that `PARITY-016` sweeps it: serde
    /// silently discards unknown fields, so a `recheck_by_v2: "2099-01-01"`
    /// added to the file alone would be invisible to a sweep of the typed
    /// struct. It is not invisible to a sweep of this map.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, serde_yaml::Value>,
}

/// Why an entry point LEFT the world, from a CLOSED vocabulary. Prose is not
/// accepted; an unknown value fails to parse.
///
/// # The hole this vocabulary prices
///
/// The ratchet excused a ROW deletion and a SCOPE deletion whenever the entry
/// point was absent from the live enumeration — and the live enumeration comes
/// from a binary built FROM THE BRANCH. The author writes the CLI, so the
/// author wrote the excuse: deleting `apr qa` from the clap tree, from
/// `scripts/competitive_parity_scope.txt` and from `rows:` in one commit
/// removed a losing comparison at rc=0, with nothing recorded anywhere. That
/// is PMAT-733 executed one level down — instead of deleting the measurement,
/// delete the thing measured.
///
/// Removal is still ALLOWED, because entry points genuinely retire and a
/// ratchet that forbids retirement is a ratchet that gets deleted. It is no
/// longer FREE and no longer SILENT: it costs exactly what every other
/// irreversible move in this ledger costs — an owned, dated record naming the
/// exact thing, with a reason from a vocabulary `serde` enforces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RemovalReason {
    /// The entry point was retired outright: no successor exists and the
    /// capability is gone from the product.
    Retired,
    /// The entry point still exists under a different name. `replacement` is
    /// REQUIRED (PARITY-027) and must itself be live and in scope, so a
    /// "rename" cannot point at nothing.
    Renamed,
    /// The entry point was folded into another one, which now carries the
    /// capability. `replacement` is REQUIRED for the same reason.
    MergedInto,
}

impl RemovalReason {
    /// The string form, matching the serde representation.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Retired => "RETIRED",
            Self::Renamed => "RENAMED",
            Self::MergedInto => "MERGED_INTO",
        }
    }

    /// Every variant, so a diagnostic can print the vocabulary.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[Self::Retired, Self::Renamed, Self::MergedInto]
    }

    /// Does this reason REQUIRE a `replacement`?
    #[must_use]
    pub fn requires_replacement(self) -> bool {
        matches!(self, Self::Renamed | Self::MergedInto)
    }

    /// The vocabulary as a comma-separated list, for diagnostics.
    #[must_use]
    pub fn vocabulary() -> String {
        Self::all()
            .iter()
            .map(|r| r.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl std::fmt::Display for RemovalReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A RECORDED REMOVAL: the receipt an entry point's disappearance costs.
///
/// # Why this is a separate block from `downgrades:`
///
/// The standing doctrine here is that a second mechanism beside an existing
/// one means two vocabularies, two expiry rules and two sets of paperwork, of
/// which exactly one stays current — so the default is to GENERALISE
/// [`Downgrade`] rather than add a sibling. It cannot be generalised here, and
/// the reason is structural rather than stylistic: `PARITY-012` requires a
/// downgrade's `entry_point` to match a row that is STILL PRESENT, and that
/// requirement is load-bearing (it is what stops a deletion being dressed up
/// as an honest correction). A removal names something that is precisely NOT
/// present. One block cannot hold both rules, and weakening PARITY-012 to make
/// room would reopen the hole it closes. So: two blocks, one shared date type,
/// one shared canonicality rule, and `PARITY-026` keeps their entry-point sets
/// DISJOINT so a record can never be spent on both sides.
///
/// # What a removal does NOT buy
///
/// It does not lower the coverage floor. A removal that drops
/// [`ParityLedger::covered_count`] below the step that has come due is refused
/// by `PARITY-024` with the record present, in date and entirely honest —
/// which is the composition that matters: the record makes the deletion
/// VISIBLE and OWNED, and the coverage ratchet makes it PAID FOR, by a
/// replacement row somewhere else in the surface.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Removal {
    /// The entry point that left, EXACTLY as it was written in the ledger row
    /// or the scope file it is being spent against.
    ///
    /// Exact, never a prefix and never a scope key standing in for its rows:
    /// `apr run` must not discharge the deletion of `apr run --gpu
    /// (concurrency=1 single-request decode)`. Those are different comparison
    /// surfaces — the whole reason a row may qualify its entry point — so one
    /// record erasing several of them would restore the count-currency defect
    /// the set ratchet exists to remove.
    #[serde(default)]
    pub entry_point: String,
    /// `None` ⇒ PARITY-025; an unknown STRING is a parse error.
    #[serde(default)]
    pub reason: Option<RemovalReason>,
    /// Who is accountable for the deletion.
    #[serde(default)]
    pub owner: String,
    /// ISO `YYYY-MM-DD` the removal was recorded. Bounded against TODAY by its
    /// TYPE, and additionally forbidden from being in the future at all
    /// (PARITY-025) — a deletion cannot have been recorded tomorrow.
    #[serde(default)]
    pub recorded_on: LedgerDate,
    /// The entry point that carries the capability now. REQUIRED when
    /// [`RemovalReason::requires_replacement`] (PARITY-027), and checked by
    /// the shell ratchet to be LIVE and IN SCOPE — so `RENAMED` cannot be used
    /// to point a deletion at nothing.
    #[serde(default)]
    pub replacement: Option<String>,
    /// Free-text elaboration. NOT the machine-checkable part.
    #[serde(default)]
    pub detail: Option<String>,
    /// Every key in the YAML that this struct does not name — see
    /// [`Downgrade::extra`]. Present so `PARITY-016` sweeps dates written into
    /// fields the schema has never heard of.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, serde_yaml::Value>,
}

/// One competitive comparison, dated at both ends.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParityRow {
    /// The apr entry point under comparison (`apr run --gpu`, `pv`, …).
    #[serde(default)]
    pub entry_point: String,
    /// The competing tool.
    #[serde(default)]
    pub competitor: String,
    /// The competitor's EXACT version. Prose such as "latest" or "unpinned"
    /// is rejected (PARITY-004): exactly one comparison in the whole repo
    /// pins a version today (`bitsandbytes==0.49.2`), and every `uv run --with
    /// scikit-learn` beat silently re-resolves its oracle night to night.
    #[serde(default)]
    pub competitor_version: String,
    /// The exact apr-side invocation.
    #[serde(default)]
    pub invocation_apr: String,
    /// The exact competitor-side invocation.
    #[serde(default)]
    pub invocation_competitor: String,
    /// What is being compared (`decode_tok_s`, `wall_clock_ratio`, …).
    #[serde(default)]
    pub dimension: String,
    /// The verdict. `None` ⇒ PARITY-006; an unknown STRING is a parse error.
    #[serde(default)]
    pub verdict: Option<Verdict>,
    /// ISO `YYYY-MM-DD` the measurement was taken.
    ///
    /// This was the SECOND bypass's real anchor: PARITY-011 bounded
    /// `valid_until - measured_on`, and `measured_on` itself was unbounded, so
    /// `2099-01-01` / `2099-06-01` sat comfortably inside the ceiling. It is
    /// now a [`LedgerDate`] (bounded against TODAY at parse time) and
    /// PARITY-017 additionally refuses any date in the future: a measurement
    /// cannot have been taken tomorrow.
    #[serde(default)]
    pub measured_on: LedgerDate,
    /// ISO `YYYY-MM-DD` after which the row is stale. Required for EVERY
    /// verdict class — see the module docs.
    #[serde(default)]
    pub valid_until: LedgerDate,
    /// Who re-measures it when it expires.
    #[serde(default)]
    pub owner: String,
    /// Pointer to the receipt: a path, a commit, or a contract id.
    #[serde(default)]
    pub evidence: String,
    /// Free-text qualification (host, n=, known gaps).
    #[serde(default)]
    pub note: Option<String>,
    /// Every key in the YAML that this struct does not name — see
    /// [`Downgrade::extra`]. Captured so `PARITY-016` can bound a date field
    /// that does not exist yet.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, serde_yaml::Value>,
}

/// The first byte of `key` that disqualifies it as a RATCHET KEY, if any.
///
/// # Why a key needs a character class at all
///
/// The ratchet's set membership is carried over a text channel: `pv
/// parity-ledger` prints `__ROW__=<key>` and a shell reads the lines back. A
/// key that may contain a NEWLINE is therefore a line-injection primitive —
/// an `entry_point` written as the block scalar
///
/// ```text
/// apr qa
/// __ROW__=lib:aprender-core::StandardScaler::fit_transform
/// __MEASURED_ROW__=lib:aprender-core::StandardScaler::fit_transform
/// ```
///
/// prints three well-formed key lines from ONE fabricated row, so the deleted
/// StandardScaler row's baseline keys are satisfied by a row that is not it.
/// That defeats the set ratchet completely, at constant totals, which is the
/// exact defect the set ratchet was built to close.
///
/// Two independent controls close it, and BOTH are required because either
/// alone is one edit from useless: this one refuses the character at the
/// SOURCE (PARITY-002 / PARITY-012), and the emitter length-prefixes every key
/// so a consumer can prove the line it read is the whole key.
///
/// The admitted class is printable ASCII (`0x20..=0x7E`) with no leading or
/// trailing space. Entry points are command lines and Rust paths; nothing in
/// the live enumeration is outside it, and restricting it means the byte
/// length the emitter prints is also the character length a shell counts,
/// under any locale.
#[must_use]
pub fn bad_key_byte(key: &str) -> Option<(usize, char)> {
    key.chars()
        .enumerate()
        .find(|&(_, c)| !(' '..='~').contains(&c))
}

/// Is `key` already in canonical form — printable ASCII, no surrounding
/// whitespace, non-empty?
///
/// Canonicality is REQUIRED of the author rather than imposed by the reader.
/// The previous emitter called `.trim()` on its way out, which meant the
/// emitted key and the authored key could differ: a key could be perturbed in
/// the file and still match the baseline. Requiring the authored bytes to be
/// canonical makes the emitted key and the baseline key byte-identical by
/// CONSTRUCTION rather than by a normalisation that some future caller forgets
/// to apply.
#[must_use]
pub fn is_canonical_key(key: &str) -> bool {
    !key.is_empty() && key.trim() == key && bad_key_byte(key).is_none()
}

impl ParityRow {
    /// True when `today` is strictly after `valid_until`.
    ///
    /// An unparseable or empty `valid_until` counts as EXPIRED — fail closed.
    /// The alternative (treat a missing bound as "never expires") is precisely
    /// the hole the 1.371× claim lived in for eight weeks.
    #[must_use]
    pub fn is_expired(&self, today: &str) -> bool {
        match (
            parse_iso_date(self.valid_until.trim()),
            parse_iso_date(today),
        ) {
            (Some(until), Some(now)) => now > until,
            _ => true,
        }
    }

    /// The verdict AS OF `today`: an expired row is [`Verdict::Unmeasured`]
    /// whatever it claims, and a row with no verdict at all is too.
    #[must_use]
    pub fn effective_verdict(&self, today: &str) -> Verdict {
        if self.is_expired(today) {
            return Verdict::Unmeasured;
        }
        self.verdict.unwrap_or(Verdict::Unmeasured)
    }
}

impl Downgrade {
    /// True when `today` is strictly after `recheck_by` — the debt has come
    /// due and the justification has lapsed.
    ///
    /// An unparseable or empty `recheck_by` counts as OVERDUE — fail closed,
    /// for the same reason [`ParityRow::is_expired`] does.
    ///
    /// # Why this exists (the third bypass, in full)
    ///
    /// `recheck_by` was validated only against `recorded_on` and NEVER against
    /// today. Nothing anywhere asked whether the recheck had come due, so a
    /// downgrade — the one move that lets `__MEASURED__` fall — was permanent
    /// the moment it was written. Both adversarial reviewers found this
    /// independently, and the aggravating half is that the old
    /// `MEASURED_MIN` floor was deleted outright when the record was
    /// introduced, so a series of downgrades could drain the ledger to zero
    /// measured rows while every gate stayed green.
    ///
    /// An overdue downgrade stops being emitted as `__DOWNGRADE__=`, which
    /// makes the drop it was paying for UNJUSTIFIED again, which is RED — and
    /// `pv parity-ledger` fails outright, exactly as an expired row does.
    /// Staleness blocks; the paperwork does not exempt it.
    #[must_use]
    pub fn is_overdue(&self, today: &str) -> bool {
        match (
            parse_iso_date(self.recheck_by.trim()),
            parse_iso_date(today),
        ) {
            (Some(due), Some(now)) => now > due,
            _ => true,
        }
    }

    /// The verdict this record says the row now declares.
    ///
    /// Absent `to_verdict` means the legacy downgrade shape, whose destination
    /// was always [`Verdict::Unmeasured`] — so it is not a special case, it is
    /// a default.
    #[must_use]
    pub fn destination(&self) -> Verdict {
        self.to_verdict.unwrap_or(Verdict::Unmeasured)
    }

    /// Does this record excuse the row's ABSENCE from `__MEASURED_ROW__`?
    ///
    /// Only a record whose destination is `UNMEASURED` does. A record for
    /// `WORSE -> NOT_COMPARABLE` describes a row that is still MEASURED, and
    /// letting it also pay for a later, unrelated drop out of the measured set
    /// would make the second move free — which is the whole class of defect
    /// this file keeps closing.
    #[must_use]
    pub fn excuses_unmeasured(&self) -> bool {
        self.destination() == Verdict::Unmeasured
    }
}

impl ParityLedger {
    /// Downgrades whose `recheck_by` has passed as of `today`, in file order.
    #[must_use]
    pub fn overdue_downgrades(&self, today: &str) -> Vec<&Downgrade> {
        self.downgrades
            .iter()
            .filter(|d| d.is_overdue(today))
            .collect()
    }

    /// Rows whose effective verdict counts as measured, as of `today`.
    #[must_use]
    pub fn measured_count(&self, today: &str) -> usize {
        self.rows
            .iter()
            .filter(|r| r.effective_verdict(today).is_measured())
            .count()
    }

    /// Rows recording a non-win (`WORSE` / `UNMEASURED` / `NOT_COMPARABLE`) as
    /// of `today`. A ledger whose rows are ALL wins is untested in the
    /// direction that matters, so the ratchet requires this to be non-zero.
    #[must_use]
    pub fn non_win_count(&self, today: &str) -> usize {
        self.rows
            .iter()
            .filter(|r| r.effective_verdict(today) != Verdict::Better)
            .count()
    }

    /// Rows past their `valid_until` as of `today`, in file order.
    #[must_use]
    pub fn expired_rows(&self, today: &str) -> Vec<&ParityRow> {
        self.rows.iter().filter(|r| r.is_expired(today)).collect()
    }

    /// The recorded downgrade for `entry_point`, if any.
    ///
    /// Compared on the TRIMMED string, because YAML block scalars leave
    /// trailing whitespace and a downgrade that silently fails to match its row
    /// would read as "recorded" while justifying nothing.
    #[must_use]
    pub fn downgrade_for(&self, entry_point: &str) -> Option<&Downgrade> {
        let key = entry_point.trim();
        self.downgrades.iter().find(|d| d.entry_point.trim() == key)
    }

    /// Records that are still IN DATE as of `today`, in file order.
    ///
    /// An overdue record excuses nothing — that is what makes the escape valve
    /// a debt with a due date rather than a retirement — so every rule that
    /// spends an excuse spends one of THESE.
    #[must_use]
    pub fn in_date_records(&self, today: &str) -> Vec<&Downgrade> {
        self.downgrades
            .iter()
            .filter(|d| !d.is_overdue(today))
            .collect()
    }

    /// Rows whose DECLARED verdict is a measurement, regardless of the clock.
    ///
    /// # Why the comparand uses DECLARED and the current tree uses EFFECTIVE
    ///
    /// The prior state is read from the ledger on protected `main`, and it is
    /// read TODAY — days or weeks after it landed. If the prior "was measured"
    /// set were computed from EFFECTIVE verdicts, the mere passage of time
    /// would erase entries from the set the current tree has to satisfy, and
    /// the bar would fall on a day nobody touched either file. A ratchet whose
    /// bar decays on the calendar is not a ratchet.
    ///
    /// So: prior = DECLARED (time-stable), current = EFFECTIVE (the clock is
    /// allowed to make the current tree owe MORE, never less). A row that
    /// expired since `main` must therefore be re-measured or carry a recorded
    /// transition — which is the same debt an expired row already owes to
    /// `block_on_staleness`, arriving through a second, independent path.
    #[must_use]
    pub fn declared_measured_rows(&self) -> Vec<&ParityRow> {
        self.rows
            .iter()
            .filter(|r| r.verdict.is_some_and(Verdict::is_measured))
            .collect()
    }

    /// The scope KEY of an entry point: what `scripts/competitive_parity_scope.txt`
    /// lists, which a row may QUALIFY.
    ///
    /// `apr run --gpu` and `apr run --gpu (concurrency=1 single-request
    /// decode)` are genuinely different comparison surfaces and both are
    /// legitimate rows, but the SCOPE is a list of entry points and both key
    /// back to `apr run`. Without this collapse, two rows on one subcommand
    /// would read as two covered entry points and the coverage ratchet would
    /// be payable by splitting a row in half.
    ///
    /// `lib:` and `bin:` entries are already exact and pass through unchanged.
    #[must_use]
    pub fn scope_key(entry_point: &str) -> String {
        let e = entry_point.trim();
        match e.strip_prefix("apr ") {
            Some(rest) => {
                let sub = rest.split_whitespace().next().unwrap_or("");
                format!("apr {sub}")
            }
            None => e.to_string(),
        }
    }

    /// The DISTINCT scope keys this ledger covers.
    #[must_use]
    pub fn covered_entry_points(&self) -> BTreeSet<String> {
        self.rows
            .iter()
            .map(|r| Self::scope_key(r.entry_point.trim()))
            .collect()
    }

    /// How many distinct in-scope entry points carry at least one row.
    #[must_use]
    pub fn covered_count(&self) -> usize {
        self.covered_entry_points().len()
    }

    /// The coverage floor that has COME DUE as of `today`, and the coverage
    /// actually achieved, as `(achieved, floor)`.
    ///
    /// A ledger with no `coverage:` block has floor 0 here; `PARITY-021`
    /// refuses the absent block separately, so this never silently blesses it.
    #[must_use]
    pub fn coverage_status(&self, today: &str) -> (usize, usize) {
        let floor = self.coverage.as_ref().map_or(0, |c| c.floor_as_of(today));
        (self.covered_count(), floor)
    }

    /// In-date records describing a move TO `BETTER`, counted per unique entry
    /// point.
    ///
    /// This is the GIVE in the non-win floor. `NON_WINS_MIN=5` over 5 rows was
    /// SATURATED — the gate mechanically forbade recording an honest win,
    /// which is the fabrication failure arrived at from the other side. The
    /// floor is now `prior_non_wins - recorded_upgrades`, so turning a
    /// measured loss into a measured win costs exactly the same owned, dated,
    /// closed-vocabulary record every other verdict change costs, and nothing
    /// more.
    #[must_use]
    pub fn recorded_upgrades(&self, today: &str) -> usize {
        let e: HashSet<&str> = self
            .in_date_records(today)
            .iter()
            .filter(|d| d.to_verdict == Some(Verdict::Better))
            .map(|d| d.entry_point.trim())
            .collect();
        e.len()
    }

    /// The EXCUSE BUDGET, as `(spent, allowed)`.
    ///
    /// # The third lever, and the one nothing bounded
    ///
    /// [`Downgrade::is_overdue`] bounds how LONG a single excuse lasts.
    /// Nothing bounded how MANY may be outstanding at once, and the two are
    /// independent: a ledger can hold a fresh, in-date, correctly-owned,
    /// closed-vocabulary record for EVERY row, at which point `__MEASURED__`
    /// is zero, every baseline `MEASURED_ROW` drop is justified, `pv
    /// parity-ledger` exits 0 and the gate is green over a ledger that
    /// measures nothing. Each record is individually impeccable; the aggregate
    /// is a ledger that has quietly stopped making claims.
    ///
    /// The bound has to have GIVE in it, because a floor with no give forbids
    /// the honest correction and produces dishonest ledgers — that lesson is
    /// already paid for. So the budget is not a constant: **a ledger may not
    /// owe more re-measurements than it holds measurements.** Paying a debt
    /// buys the capacity to take on another, which is exactly the incentive
    /// wanted, and the ledger can never be more than half excused.
    ///
    /// Counted per unique `entry_point`, so duplicate records cannot inflate
    /// the numerator (`PARITY-012` refuses duplicates anyway; this does not
    /// depend on that rule holding).
    #[must_use]
    pub fn excuse_budget(&self, today: &str) -> (usize, usize) {
        let spent: HashSet<&str> = self
            .in_date_records(today)
            .iter()
            .map(|d| d.entry_point.trim())
            .collect();
        (spent.len(), self.measured_count(today))
    }
}

/// The furthest a `valid_until` (or a `recheck_by`) may be set beyond the date
/// it is anchored to, in days.
///
/// WHY A BOUND EXISTS AT ALL. Check-time freshness is only as strong as the
/// dates it reads. Rewriting all five `valid_until` values to `2099-12-31`
/// satisfied every rule in the first design: the row is not expired, so it
/// counts as MEASURED forever, and "staleness blocks" becomes voluntary. An
/// unbounded expiry field is an exemption with a date on it.
///
/// WHY 180. The five seeded rows span 55–109 days between `measured_on` and
/// `valid_until`, so a two-quarter ceiling admits every honest row already
/// written with ~65% headroom while refusing anything that is trying to outlive
/// review. It is deliberately anchored to `measured_on`, not to today: a row
/// recording an OLD measurement honestly must stay writable, and it is
/// `is_expired` — not this bound — that then makes it degrade.
pub const MAX_VALIDITY_DAYS: i64 = 180;

/// Days from `from` to `to` (negative when `to` precedes `from`), or `None`
/// when either side is not a strict ISO calendar date.
#[must_use]
pub fn days_between(from: &str, to: &str) -> Option<i64> {
    let (fy, fm, fd) = parse_iso_date(from)?;
    let (ty, tm, td) = parse_iso_date(to)?;
    Some(
        days_from_civil(i64::from(ty), u32::from(tm), u32::from(td))
            - days_from_civil(i64::from(fy), u32::from(fm), u32::from(fd)),
    )
}

/// Convert a proleptic Gregorian `(y, m, d)` to days-since-1970-01-01.
///
/// Hinnant, "chrono-Compatible Low-Level Date Algorithms", `days_from_civil` —
/// the exact inverse of [`civil_from_days`], which this module already carries.
#[must_use]
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = i64::from(if m > 2 { m - 3 } else { m + 9 }); // [0, 11]
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Parse a strict ISO `YYYY-MM-DD` calendar date, returning `(y, m, d)`.
///
/// Deliberately strict and leap-year aware: `2026-02-30` and `2027-02-29` are
/// rejected, `2028-02-29` is accepted. Returning `None` makes the caller fail
/// closed, so a typo'd expiry expires the row instead of blessing it forever.
#[must_use]
pub fn parse_iso_date(s: &str) -> Option<(u16, u8, u8)> {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let digits = |from: usize, to: usize| -> Option<u32> {
        let mut v: u32 = 0;
        for &c in &b[from..to] {
            if !c.is_ascii_digit() {
                return None;
            }
            v = v * 10 + u32::from(c - b'0');
        }
        Some(v)
    };
    let y = digits(0, 4)?;
    let m = digits(5, 7)?;
    let d = digits(8, 10)?;
    if !(1..=12).contains(&m) || d < 1 {
        return None;
    }
    if d > u32::from(days_in_month(y, m)) {
        return None;
    }
    Some((
        u16::try_from(y).ok()?,
        u8::try_from(m).ok()?,
        u8::try_from(d).ok()?,
    ))
}

/// Days in `month` of `year` (proleptic Gregorian).
#[must_use]
pub fn days_in_month(year: u32, month: u32) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Today's UTC date as `YYYY-MM-DD`, for check-time freshness evaluation.
///
/// Hand-rolled (Howard Hinnant's `civil_from_days`) rather than pulling
/// `chrono` into a crate that has four dependencies, none of them a date
/// library. `None` when the system clock is before the epoch.
#[must_use]
pub fn today_utc() -> Option<String> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    let days = i64::try_from(secs / 86_400).ok()?;
    let (y, m, d) = civil_from_days(days);
    Some(format!("{y:04}-{m:02}-{d:02}"))
}

/// Convert days-since-1970-01-01 to a proleptic Gregorian `(y, m, d)`.
///
/// Hinnant, "chrono-Compatible Low-Level Date Algorithms", `civil_from_days`.
#[must_use]
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    #[allow(clippy::cast_sign_loss)]
    (y, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    include!("parity_tests.rs");
}
