//! Double-Blind Verification Framework (PMAT-020)
//!
//! Implements the Double-Blind Verification protocol per §36.2 of cbtop spec.
//! Separation of Dev (implementation) and QA (verification) roles with
//! black-box falsification attempts.
//!
//! # Protocol
//!
//! 1. **Group A (Dev)**: Implements feature and claims "Falsification Passed"
//! 2. **Group B (QA)**: Receives only binary + F-criteria (no source)
//! 3. **Blind Test**: Group B attempts to falsify the binary black-box
//! 4. **Confirmation**: Only if Group B fails to falsify is release approved
//!
//! # Citations
//!
//! - [Rosenthal & Fode 1963] "Psychology of the Scientist: Experimenter Bias" Psychological Bulletin
//! - [Holman et al. 2015] "A Systematic Review of Double-Blind Experiments in SE" IEEE TSE

mod types;
pub use types::*;

/// Double-blind verification session
#[derive(Debug)]
pub struct VerificationSession {
    /// Session ID
    pub id: String,
    /// Claims submitted by Dev
    claims: Vec<FalsificationClaim>,
    /// Black-box artifacts generated
    artifacts: Vec<BlackBoxArtifact>,
    /// Verification attempts by QA
    attempts: Vec<VerificationAttempt>,
    /// Scorecard
    pub scorecard: ScorecardV2,
    /// Audit trail
    audit_trail: Vec<AuditEntry>,
    /// Current state
    state: SessionState,
}

impl Default for VerificationSession {
    fn default() -> Self {
        Self::new("session-1")
    }
}

impl VerificationSession {
    /// Create a new session
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            claims: Vec::new(),
            artifacts: Vec::new(),
            attempts: Vec::new(),
            scorecard: ScorecardV2::new(),
            audit_trail: Vec::new(),
            state: SessionState::AwaitingClaims,
        }
    }

    /// Get current state
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Submit a claim (Dev role only)
    pub fn submit_claim(
        &mut self,
        role: Role,
        claim: FalsificationClaim,
    ) -> Result<(), &'static str> {
        if !role.can_claim() {
            return Err("Only Dev role can submit claims");
        }
        if !claim.is_valid() {
            return Err("Invalid claim structure");
        }

        // Record audit
        let entry = AuditEntry::new(
            &format!("audit-{}", self.audit_trail.len()),
            role,
            &claim.claimant,
            &format!("Submitted claim for {}", claim.feature),
        )
        .with_artifact(&claim.id);
        self.audit_trail.push(entry);

        self.claims.push(claim);
        self.state = SessionState::AwaitingVerification;
        Ok(())
    }

    /// Generate black-box artifact from claim
    pub fn generate_artifact(
        &mut self,
        claim_id: &str,
        binary_hash: &str,
    ) -> Option<BlackBoxArtifact> {
        let claim = self.claims.iter().find(|c| c.id == claim_id)?;
        let artifact = BlackBoxArtifact::from_claim(claim, binary_hash);

        let entry = AuditEntry::new(
            &format!("audit-{}", self.audit_trail.len()),
            Role::System,
            "System",
            &format!("Generated black-box artifact from claim {}", claim_id),
        )
        .with_artifact(&artifact.id);
        self.audit_trail.push(entry);

        self.artifacts.push(artifact.clone());
        Some(artifact)
    }

    /// Submit verification attempt (QA role only)
    pub fn submit_attempt(
        &mut self,
        role: Role,
        attempt: VerificationAttempt,
    ) -> Result<(), &'static str> {
        if !role.can_verify() {
            return Err("Only QA role can submit verification attempts");
        }

        // Record audit
        let entry = AuditEntry::new(
            &format!("audit-{}", self.audit_trail.len()),
            role,
            &attempt.verifier,
            &format!("Submitted verification attempt: {:?}", attempt.result),
        )
        .with_artifact(&attempt.artifact_id);
        self.audit_trail.push(entry);

        self.attempts.push(attempt);
        Ok(())
    }

    /// Get all attempts for an artifact
    pub fn get_attempts(&self, artifact_id: &str) -> Vec<&VerificationAttempt> {
        self.attempts
            .iter()
            .filter(|a| a.artifact_id == artifact_id)
            .collect()
    }

    /// Make release decision (System role only)
    pub fn make_decision(&mut self, role: Role) -> Result<ReleaseDecision, &'static str> {
        if !role.can_approve() {
            return Err("Only System role can make release decisions");
        }

        if self.attempts.is_empty() {
            return Ok(ReleaseDecision::Pending {
                reason: "No verification attempts yet".to_string(),
            });
        }

        // Check if any attempt successfully falsified
        let falsified = self
            .attempts
            .iter()
            .any(|a| a.result == VerificationResult::Falsified);

        // Check if all attempts are inconclusive
        let all_inconclusive = self
            .attempts
            .iter()
            .all(|a| a.result == VerificationResult::Inconclusive);

        let decision = if falsified {
            ReleaseDecision::Rejected {
                reason: "QA successfully falsified the claim".to_string(),
            }
        } else if all_inconclusive {
            ReleaseDecision::Pending {
                reason: "All verification attempts were inconclusive".to_string(),
            }
        } else {
            // Check scorecard
            if self.scorecard.passes() {
                ReleaseDecision::Approved {
                    reason: format!(
                        "QA failed to falsify and scorecard passes ({:.1}/100, grade {})",
                        self.scorecard.total_score(),
                        self.scorecard.grade()
                    ),
                }
            } else {
                ReleaseDecision::Rejected {
                    reason: format!(
                        "Scorecard fails ({:.1}/100 < 70, grade {})",
                        self.scorecard.total_score(),
                        self.scorecard.grade()
                    ),
                }
            }
        };

        // Record audit
        let entry = AuditEntry::new(
            &format!("audit-{}", self.audit_trail.len()),
            role,
            "System",
            &format!("Release decision: {:?}", decision),
        );
        self.audit_trail.push(entry);

        self.state = SessionState::Completed;
        Ok(decision)
    }

    /// Get audit trail
    pub fn audit_trail(&self) -> &[AuditEntry] {
        &self.audit_trail
    }

    /// Get claim count
    pub fn claim_count(&self) -> usize {
        self.claims.len()
    }

    /// Get attempt count
    pub fn attempt_count(&self) -> usize {
        self.attempts.len()
    }

    /// Generate verification report
    pub fn generate_report(&self) -> VerificationReport {
        VerificationReport {
            session_id: self.id.clone(),
            total_claims: self.claims.len(),
            total_artifacts: self.artifacts.len(),
            total_attempts: self.attempts.len(),
            falsified_count: self
                .attempts
                .iter()
                .filter(|a| a.result == VerificationResult::Falsified)
                .count(),
            unfalsified_count: self
                .attempts
                .iter()
                .filter(|a| a.result == VerificationResult::Unfalsified)
                .count(),
            inconclusive_count: self
                .attempts
                .iter()
                .filter(|a| a.result == VerificationResult::Inconclusive)
                .count(),
            scorecard_total: self.scorecard.total_score(),
            scorecard_grade: self.scorecard.grade().to_string(),
            audit_entries: self.audit_trail.len(),
        }
    }
}

#[cfg(test)]
mod tests;
