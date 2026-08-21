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

use serde::{Deserialize, Serialize};

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
}

/// Why a row that used to be MEASURED is now `UNMEASURED`, from a CLOSED
/// vocabulary. Prose is not accepted; an unknown value fails to parse.
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
        }
    }
}

impl std::fmt::Display for DowngradeReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A recorded, dated, owned downgrade of one row to `UNMEASURED`.
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
    /// Who owes the re-measurement.
    #[serde(default)]
    pub owner: String,
    /// ISO `YYYY-MM-DD` the downgrade was recorded.
    #[serde(default)]
    pub recorded_on: String,
    /// ISO `YYYY-MM-DD` by which the row must be re-measured or the downgrade
    /// re-argued. Bounded like `valid_until`, so a downgrade cannot be made
    /// permanent by dating it to 2099.
    #[serde(default)]
    pub recheck_by: String,
    /// Free-text elaboration. NOT the machine-checkable part.
    #[serde(default)]
    pub detail: Option<String>,
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
    #[serde(default)]
    pub measured_on: String,
    /// ISO `YYYY-MM-DD` after which the row is stale. Required for EVERY
    /// verdict class — see the module docs.
    #[serde(default)]
    pub valid_until: String,
    /// Who re-measures it when it expires.
    #[serde(default)]
    pub owner: String,
    /// Pointer to the receipt: a path, a commit, or a contract id.
    #[serde(default)]
    pub evidence: String,
    /// Free-text qualification (host, n=, known gaps).
    #[serde(default)]
    pub note: Option<String>,
}

impl ParityRow {
    /// True when `today` is strictly after `valid_until`.
    ///
    /// An unparseable or empty `valid_until` counts as EXPIRED — fail closed.
    /// The alternative (treat a missing bound as "never expires") is precisely
    /// the hole the 1.371× claim lived in for eight weeks.
    #[must_use]
    pub fn is_expired(&self, today: &str) -> bool {
        match (parse_iso_date(&self.valid_until), parse_iso_date(today)) {
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

impl ParityLedger {
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
