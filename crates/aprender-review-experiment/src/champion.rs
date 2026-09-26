//! REX-11 champion/challenger promotion (§5.4), `review-champion-challenger-v1`.
//!
//! A challenger (any §5.3 candidate: a prompt version, retrieval few-shot, a
//! B2 checkpoint) replaces the champion only if all five §5.4 gates hold on
//! the current sealed test version:
//! 1. per-item correctness improves: McNemar exact, paired, one-sided, α;
//! 2. the precision point estimate is not lower than the champion's;
//! 3. H1 (invariance) and H2 (determinism) hold on the primary cell;
//! 4. the contamination contract is green: 0 test hashes in its training data;
//! 5. its p95 on the primary cell meets the §4 queue budget.
//!
//! Every gate fails closed: an unmeasured gate is a failed gate. Each decision
//! consumes one of the test version's [`EVALUATIONS_PER_VERSION`] evaluations
//! (§2.2); once they are spent the next challenger is **refused**, consumes
//! nothing, and the test set must rotate. A challenger that never ran is
//! refused too (R-6): it is not an evaluation. Demotion (§5.4): two
//! consecutive gold-labelled escapes where the champion said PASS and a voter
//! said FAIL trigger an HRQ and re-evaluation.

use serde::{Deserialize, Serialize};

use crate::score::{correctness_pairs, score, Scored};
use crate::stats::{discordant, mcnemar_exact_one_sided, ALPHA};

pub const SCHEME: &str = "review-champion-challenger-v1";
/// §2.2 `[A]`: promotion decisions a sealed test version supports.
pub const EVALUATIONS_PER_VERSION: usize = 20;

/// What serves the lane: promotion is a forjar pin bump of these shas.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pin {
    pub weights_sha256: String,
    pub prompt_sha256: String,
    pub adapter_sha256: Option<String>,
}

/// Everything §5.4 reads about one challenger. `None` is "not measured".
#[derive(Debug, Clone)]
pub struct Evidence<'a> {
    pub id: &'a str,
    /// The §5.3 tier (B1a, B1b, B2a…).
    pub tier: &'a str,
    pub pin: Pin,
    pub test_version: &'a str,
    pub challenger: &'a [Scored],
    pub champion: &'a [Scored],
    pub h1_holds: Option<bool>,
    pub h2_holds: Option<bool>,
    pub contamination_hits: Option<u64>,
    pub p95_s: Option<f64>,
    /// §4 step 5: the slowest voting lane's measured p95 on the same items.
    pub queue_budget_p95_s: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Promoted,
    Rejected,
    /// Not an evaluation: nothing consumed.
    Refused,
}

/// One ledger row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Decision {
    pub schema: String,
    pub test_version: String,
    pub challenger: String,
    pub tier: String,
    pub outcome: Outcome,
    /// The test version's evaluations used after this row.
    pub evaluations_used: usize,
    pub champion_before: Pin,
    pub pin: Pin,
    pub mcnemar_b: u64,
    pub mcnemar_c: u64,
    pub mcnemar_p: Option<f64>,
    pub precision_champion: Option<f64>,
    pub precision_challenger: Option<f64>,
    pub precision_delta: Option<f64>,
    pub reasons: Vec<String>,
}

/// The serving champion: the pin of the last promotion, else the initial one.
#[must_use]
pub fn champion<'a>(ledger: &'a [Decision], initial: &'a Pin) -> &'a Pin {
    ledger
        .iter()
        .rev()
        .find(|d| d.outcome == Outcome::Promoted)
        .map_or(initial, |d| &d.pin)
}

/// Evaluations the ledger has spent on `test_version`.
#[must_use]
pub fn evaluations_used(ledger: &[Decision], test_version: &str) -> usize {
    ledger
        .iter()
        .filter(|d| d.test_version == test_version && d.outcome != Outcome::Refused)
        .count()
}

fn gate(reasons: &mut Vec<String>, ok: bool, why: impl FnOnce() -> String) {
    if !ok {
        reasons.push(why());
    }
}

/// Decide one challenger against the ledger's champion. Pure: the caller
/// appends the returned row.
#[must_use]
pub fn decide(ledger: &[Decision], initial: &Pin, e: &Evidence<'_>) -> Decision {
    let used = evaluations_used(ledger, e.test_version);
    let champ = champion(ledger, initial).clone();
    let (sc, sh) = (score(e.challenger), score(e.champion));
    let (b, c) = discordant(&correctness_pairs(e.challenger, e.champion));
    let mut d = Decision {
        schema: SCHEME.into(),
        test_version: e.test_version.into(),
        challenger: e.id.into(),
        tier: e.tier.into(),
        outcome: Outcome::Refused,
        evaluations_used: used,
        champion_before: champ.clone(),
        pin: e.pin.clone(),
        mcnemar_b: b,
        mcnemar_c: c,
        mcnemar_p: None,
        precision_champion: sh.precision.value(),
        precision_challenger: sc.precision.value(),
        precision_delta: sc
            .precision
            .value()
            .zip(sh.precision.value())
            .map(|(a, b)| a - b),
        reasons: Vec::new(),
    };
    let ran = e.challenger.iter().any(|s| s.verdict.executed());
    let refusals = [
        (used >= EVALUATIONS_PER_VERSION).then(|| {
            format!("test version {} has spent its {EVALUATIONS_PER_VERSION} evaluations: rotate it (§2.2)", e.test_version)
        }),
        (!ran).then(|| "the challenger has no executed row on the test split (R-6)".to_string()),
        e.champion.is_empty().then(|| "no champion rows on this test version".to_string()),
        (e.pin == champ).then(|| "the challenger is the champion".to_string()),
    ];
    d.reasons = refusals.into_iter().flatten().collect();
    if !d.reasons.is_empty() {
        return d;
    }
    let p = mcnemar_exact_one_sided(b, c);
    d.mcnemar_p = Some(p);
    d.evaluations_used = used + 1;
    let r = &mut d.reasons;
    gate(r, p < ALPHA, || {
        format!("1: McNemar one-sided p = {p:.4} (b={b}, c={c}) is not < {ALPHA}")
    });
    gate(r, d.precision_delta.is_some_and(|x| x >= 0.0), || {
        format!(
            "2: precision {:?} is lower than the champion's {:?}, or unmeasured",
            d.precision_challenger, d.precision_champion
        )
    });
    gate(r, e.h1_holds == Some(true), || {
        format!("3: H1 invariance {:?}", e.h1_holds)
    });
    gate(r, e.h2_holds == Some(true), || {
        format!("3: H2 determinism {:?}", e.h2_holds)
    });
    gate(r, e.contamination_hits == Some(0), || {
        format!(
            "4: contamination hits {:?} (must be 0)",
            e.contamination_hits
        )
    });
    let fits = matches!((e.p95_s, e.queue_budget_p95_s),
        (Some(p), Some(q)) if p.is_finite() && q.is_finite() && p <= q);
    gate(r, fits, || {
        format!(
            "5: p95 {:?} s vs queue budget {:?} s",
            e.p95_s, e.queue_budget_p95_s
        )
    });
    d.outcome = if d.reasons.is_empty() {
        Outcome::Promoted
    } else {
        Outcome::Rejected
    };
    d
}

/// A gold-labelled escape joined back to the PR that let it through (§5.2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Escape {
    pub pr: String,
    pub gold: bool,
    pub champion_passed: bool,
    pub a_voter_failed: bool,
}

/// §5.4 demotion: the first pair of consecutive gold escapes where the
/// champion said PASS and a voter said FAIL, if any. Silver rows are not
/// evidence and neither break nor extend a run.
#[must_use]
pub fn demotion(escapes: &[Escape]) -> Option<(String, String)> {
    let gold: Vec<&Escape> = escapes.iter().filter(|e| e.gold).collect();
    gold.windows(2)
        .find(|w| w.iter().all(|e| e.champion_passed && e.a_voter_failed))
        .map(|w| (w[0].pr.clone(), w[1].pr.clone()))
}

#[cfg(test)]
#[path = "champion_tests.rs"]
mod tests;
