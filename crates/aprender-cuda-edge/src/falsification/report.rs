//! Falsification report aggregation.
//!
//! Collects claim statuses across all frameworks and computes coverage
//! metrics.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::checklist::{all_claims, ClaimStatus, FalsificationClaim, Framework};

/// Aggregated falsification report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationReport {
    /// Status for each claim by ID.
    statuses: HashMap<String, ClaimStatus>,
    /// Total number of claims.
    total_claims: usize,
}

impl FalsificationReport {
    /// Create a new report with all claims in Pending status.
    #[must_use]
    pub fn new() -> Self {
        let claims = all_claims();
        let total_claims = claims.len();
        let statuses =
            claims.into_iter().map(|c| (c.id.to_string(), ClaimStatus::Pending)).collect();
        Self { statuses, total_claims }
    }

    /// Mark a claim as verified (property held).
    pub fn mark_verified(&mut self, claim_id: &str) {
        if let Some(status) = self.statuses.get_mut(claim_id) {
            *status = ClaimStatus::Verified;
        }
    }

    /// Mark a claim as violated (property falsified — bug found).
    pub fn mark_violated(&mut self, claim_id: &str) {
        if let Some(status) = self.statuses.get_mut(claim_id) {
            *status = ClaimStatus::Violated;
        }
    }

    /// Mark a claim as skipped.
    pub fn mark_skipped(&mut self, claim_id: &str) {
        if let Some(status) = self.statuses.get_mut(claim_id) {
            *status = ClaimStatus::Skipped;
        }
    }

    /// Mark a claim as in progress.
    pub fn mark_in_progress(&mut self, claim_id: &str) {
        if let Some(status) = self.statuses.get_mut(claim_id) {
            *status = ClaimStatus::InProgress;
        }
    }

    /// Get the status of a specific claim.
    #[must_use]
    pub fn status(&self, claim_id: &str) -> Option<ClaimStatus> {
        self.statuses.get(claim_id).copied()
    }

    /// Calculate coverage: (verified + violated) / (total - skipped).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn coverage(&self) -> f64 {
        let verified = self.count_by_status(ClaimStatus::Verified);
        let violated = self.count_by_status(ClaimStatus::Violated);
        let skipped = self.count_by_status(ClaimStatus::Skipped);

        let applicable = self.total_claims - skipped;
        if applicable == 0 {
            return 1.0;
        }
        (verified + violated) as f64 / applicable as f64
    }

    /// Returns true if all applicable claims are either verified or violated.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        let pending = self.count_by_status(ClaimStatus::Pending);
        let in_progress = self.count_by_status(ClaimStatus::InProgress);
        pending == 0 && in_progress == 0
    }

    /// Count claims with a specific status.
    #[must_use]
    pub fn count_by_status(&self, status: ClaimStatus) -> usize {
        self.statuses.values().filter(|s| **s == status).count()
    }

    /// Returns all violated claim IDs.
    #[must_use]
    pub fn violated_claims(&self) -> Vec<String> {
        self.statuses
            .iter()
            .filter(|(_, s)| **s == ClaimStatus::Violated)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Returns claims grouped by framework with their statuses.
    #[must_use]
    pub fn by_framework(&self) -> HashMap<Framework, Vec<(FalsificationClaim, ClaimStatus)>> {
        let claims = all_claims();
        let mut result: HashMap<Framework, Vec<_>> = HashMap::new();

        for claim in claims {
            let status = self.statuses.get(claim.id).copied().unwrap_or(ClaimStatus::Pending);
            result.entry(claim.framework).or_default().push((claim, status));
        }

        result
    }
}

impl Default for FalsificationReport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods, clippy::redundant_closure_for_method_calls)]
mod tests {
    use super::*;

    #[test]
    fn new_report_has_all_pending() {
        let report = FalsificationReport::new();
        assert_eq!(report.count_by_status(ClaimStatus::Pending), 50);
    }

    #[test]
    fn mark_verified_updates_status() {
        let mut report = FalsificationReport::new();
        report.mark_verified("NF-001");
        assert_eq!(report.status("NF-001"), Some(ClaimStatus::Verified));
    }

    #[test]
    fn mark_violated_updates_status() {
        let mut report = FalsificationReport::new();
        report.mark_violated("NF-001");
        assert_eq!(report.status("NF-001"), Some(ClaimStatus::Violated));
    }

    #[test]
    fn coverage_all_pending_is_zero() {
        let report = FalsificationReport::new();
        assert!((report.coverage() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn coverage_all_verified() {
        let mut report = FalsificationReport::new();
        for claim in all_claims() {
            report.mark_verified(claim.id);
        }
        assert!((report.coverage() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn is_complete_when_all_done() {
        let mut report = FalsificationReport::new();
        for claim in all_claims() {
            report.mark_verified(claim.id);
        }
        assert!(report.is_complete());
    }

    #[test]
    fn is_not_complete_with_pending() {
        let report = FalsificationReport::new();
        assert!(!report.is_complete());
    }

    #[test]
    fn violated_claims_returns_ids() {
        let mut report = FalsificationReport::new();
        report.mark_violated("NF-001");
        report.mark_violated("SP-002");
        let violated = report.violated_claims();
        assert_eq!(violated.len(), 2);
        assert!(violated.contains(&"NF-001".to_string()));
        assert!(violated.contains(&"SP-002".to_string()));
    }

    #[test]
    fn report_default() {
        let report = FalsificationReport::default();
        assert_eq!(report.count_by_status(ClaimStatus::Pending), 50);
    }

    #[test]
    fn coverage_all_skipped() {
        let mut report = FalsificationReport::new();
        for claim in all_claims() {
            report.mark_skipped(claim.id);
        }
        // All skipped means applicable = 0, returns 1.0
        assert!((report.coverage() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mark_in_progress_updates_status() {
        let mut report = FalsificationReport::new();
        report.mark_in_progress("NF-001");
        assert_eq!(report.status("NF-001"), Some(ClaimStatus::InProgress));
    }

    #[test]
    fn by_framework_groups_all_claims() {
        let report = FalsificationReport::new();
        let grouped = report.by_framework();
        let total: usize = grouped.values().map(|v| v.len()).sum();
        assert_eq!(total, 50);
    }
}
