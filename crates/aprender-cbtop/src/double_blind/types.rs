//! Types, enums, and data models for the double-blind verification framework.

use std::collections::HashMap;
use std::time::SystemTime;

/// Role in the double-blind verification process
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// Developer - implements features, makes claims
    Dev,
    /// Quality Assurance - verifies claims black-box
    Qa,
    /// System - makes release decisions
    System,
}

impl Role {
    /// Get role name
    pub fn name(&self) -> &'static str {
        match self {
            Role::Dev => "Developer",
            Role::Qa => "QA",
            Role::System => "System",
        }
    }

    /// Check if role can make claims
    pub fn can_claim(&self) -> bool {
        matches!(self, Role::Dev)
    }

    /// Check if role can verify
    pub fn can_verify(&self) -> bool {
        matches!(self, Role::Qa)
    }

    /// Check if role can approve releases
    pub fn can_approve(&self) -> bool {
        matches!(self, Role::System)
    }
}

/// Verification result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationResult {
    /// Claim was successfully falsified (QA found a bug)
    Falsified,
    /// Unable to falsify the claim (QA failed to find bugs)
    Unfalsified,
    /// Verification was inconclusive
    Inconclusive,
}

impl VerificationResult {
    /// Check if release should be approved based on this result
    pub fn should_approve(&self) -> bool {
        matches!(self, VerificationResult::Unfalsified)
    }
}

/// A falsification criterion (F-criteria)
#[derive(Debug, Clone)]
pub struct FalsificationCriterion {
    /// Criterion ID (e.g., "F1001")
    pub id: String,
    /// Description of what to test
    pub description: String,
    /// Pass condition
    pub pass_condition: String,
}

impl FalsificationCriterion {
    /// Create a new criterion
    pub fn new(id: &str, description: &str, pass_condition: &str) -> Self {
        Self {
            id: id.to_string(),
            description: description.to_string(),
            pass_condition: pass_condition.to_string(),
        }
    }

    /// Compute hash of criterion for integrity verification
    pub fn hash(&self) -> u64 {
        // Simple hash combining id, description, and pass_condition
        let mut hash: u64 = 0;
        for byte in self.id.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(u64::from(byte));
        }
        for byte in self.description.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(u64::from(byte));
        }
        for byte in self.pass_condition.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(u64::from(byte));
        }
        hash
    }
}

/// Developer's claim that falsification passed
#[derive(Debug, Clone)]
pub struct FalsificationClaim {
    /// Claim ID
    pub id: String,
    /// Feature or component being claimed
    pub feature: String,
    /// F-criteria that were tested
    pub criteria: Vec<FalsificationCriterion>,
    /// Hash of all criteria for integrity
    pub criteria_hash: u64,
    /// Timestamp of claim
    pub timestamp: SystemTime,
    /// Developer making the claim
    pub claimant: String,
    /// Version of the software
    pub version: String,
    /// Evidence supporting the claim (test results, logs)
    pub evidence: Vec<String>,
}

impl FalsificationClaim {
    /// Create a new claim
    pub fn new(id: &str, feature: &str, claimant: &str, version: &str) -> Self {
        Self {
            id: id.to_string(),
            feature: feature.to_string(),
            criteria: Vec::new(),
            criteria_hash: 0,
            timestamp: SystemTime::now(),
            claimant: claimant.to_string(),
            version: version.to_string(),
            evidence: Vec::new(),
        }
    }

    /// Add a criterion to the claim
    pub fn add_criterion(&mut self, criterion: FalsificationCriterion) {
        self.criteria.push(criterion);
        self.update_hash();
    }

    /// Add evidence to the claim
    pub fn add_evidence(&mut self, evidence: &str) {
        self.evidence.push(evidence.to_string());
    }

    /// Update the criteria hash
    fn update_hash(&mut self) {
        let mut hash: u64 = 0;
        for criterion in &self.criteria {
            hash = hash.wrapping_add(criterion.hash());
        }
        self.criteria_hash = hash;
    }

    /// Verify the criteria hash matches
    pub fn verify_hash(&self) -> bool {
        let mut expected: u64 = 0;
        for criterion in &self.criteria {
            expected = expected.wrapping_add(criterion.hash());
        }
        expected == self.criteria_hash
    }

    /// Check if claim is valid (has required fields)
    pub fn is_valid(&self) -> bool {
        !self.id.is_empty()
            && !self.feature.is_empty()
            && !self.claimant.is_empty()
            && !self.version.is_empty()
            && !self.criteria.is_empty()
            && self.verify_hash()
    }
}

/// Black-box artifact for QA (no source code)
#[derive(Debug, Clone)]
pub struct BlackBoxArtifact {
    /// Artifact ID
    pub id: String,
    /// Binary hash (SHA256 hex)
    pub binary_hash: String,
    /// F-criteria to test against
    pub criteria: Vec<FalsificationCriterion>,
    /// Criteria hash for integrity
    pub criteria_hash: u64,
    /// Version being tested
    pub version: String,
    /// Deadline for verification
    pub deadline: Option<SystemTime>,
}

impl BlackBoxArtifact {
    /// Create from a claim (strips source-related info)
    pub fn from_claim(claim: &FalsificationClaim, binary_hash: &str) -> Self {
        Self {
            id: format!("BB-{}", claim.id),
            binary_hash: binary_hash.to_string(),
            criteria: claim.criteria.clone(),
            criteria_hash: claim.criteria_hash,
            version: claim.version.clone(),
            deadline: None,
        }
    }

    /// Set verification deadline
    pub fn with_deadline(mut self, deadline: SystemTime) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Check if deadline has passed
    pub fn is_expired(&self) -> bool {
        if let Some(deadline) = self.deadline {
            SystemTime::now() > deadline
        } else {
            false
        }
    }

    /// Verify criteria hash matches claim
    pub fn verify_criteria_integrity(&self, claim: &FalsificationClaim) -> bool {
        self.criteria_hash == claim.criteria_hash
    }
}

/// A single verification attempt by QA
#[derive(Debug, Clone)]
pub struct VerificationAttempt {
    /// Attempt ID
    pub id: String,
    /// Artifact being tested
    pub artifact_id: String,
    /// QA engineer making the attempt
    pub verifier: String,
    /// Result of verification
    pub result: VerificationResult,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Evidence collected (logs, traces)
    pub evidence: Vec<String>,
    /// Individual criterion results
    pub criterion_results: HashMap<String, bool>,
}

impl VerificationAttempt {
    /// Create a new attempt
    pub fn new(id: &str, artifact_id: &str, verifier: &str) -> Self {
        Self {
            id: id.to_string(),
            artifact_id: artifact_id.to_string(),
            verifier: verifier.to_string(),
            result: VerificationResult::Inconclusive,
            timestamp: SystemTime::now(),
            evidence: Vec::new(),
            criterion_results: HashMap::new(),
        }
    }

    /// Record result for a criterion
    pub fn record_criterion(&mut self, criterion_id: &str, passed: bool) {
        self.criterion_results.insert(criterion_id.to_string(), passed);
    }

    /// Add evidence
    pub fn add_evidence(&mut self, evidence: &str) {
        self.evidence.push(evidence.to_string());
    }

    /// Finalize the attempt with result
    pub fn finalize(&mut self, result: VerificationResult) {
        self.result = result;
        self.timestamp = SystemTime::now();
    }

    /// Check if any criterion was falsified
    pub fn has_falsification(&self) -> bool {
        self.criterion_results.values().any(|&passed| !passed)
    }

    /// Get count of passed criteria
    pub fn passed_count(&self) -> usize {
        self.criterion_results.values().filter(|&&p| p).count()
    }

    /// Get count of failed criteria
    pub fn failed_count(&self) -> usize {
        self.criterion_results.values().filter(|&&p| !p).count()
    }
}

/// Scorecard component with weight
#[derive(Debug, Clone)]
pub struct ScorecardComponent {
    /// Component name
    pub name: String,
    /// Weight (0.0 to 1.0, must sum to 1.0 across all components)
    pub weight: f64,
    /// Score (0 to 100)
    pub score: u32,
}

impl ScorecardComponent {
    /// Create a new component
    pub fn new(name: &str, weight: f64, score: u32) -> Self {
        Self { name: name.to_string(), weight, score: score.min(100) }
    }

    /// Calculate weighted score
    pub fn weighted_score(&self) -> f64 {
        self.weight * f64::from(self.score)
    }
}

/// Falsification Scorecard v2 per section 36.3
#[derive(Debug, Clone)]
pub struct ScorecardV2 {
    /// Components with weights
    pub components: Vec<ScorecardComponent>,
    /// Version (v1 or v2)
    pub version: u8,
}

impl Default for ScorecardV2 {
    fn default() -> Self {
        Self::new()
    }
}

impl ScorecardV2 {
    /// Create a new v2 scorecard with default components
    pub fn new() -> Self {
        Self {
            components: vec![
                ScorecardComponent::new("Core Correctness", 0.30, 0),
                ScorecardComponent::new("Performance", 0.30, 0),
                ScorecardComponent::new("Resilience", 0.20, 0),
                ScorecardComponent::new("Usability", 0.20, 0),
            ],
            version: 2,
        }
    }

    /// Set score for a component by name
    pub fn set_score(&mut self, name: &str, score: u32) -> bool {
        for component in &mut self.components {
            if component.name == name {
                component.score = score.min(100);
                return true;
            }
        }
        false
    }

    /// Calculate total weighted score
    pub fn total_score(&self) -> f64 {
        self.components.iter().map(|c| c.weighted_score()).sum()
    }

    /// Check if weights sum to 1.0
    pub fn weights_valid(&self) -> bool {
        let sum: f64 = self.components.iter().map(|c| c.weight).sum();
        (sum - 1.0).abs() < 1e-10
    }

    /// Check if scorecard passes (>= 70)
    pub fn passes(&self) -> bool {
        self.total_score() >= 70.0
    }

    /// Get grade based on score
    pub fn grade(&self) -> &'static str {
        let score = self.total_score();
        if score >= 90.0 {
            "A"
        } else if score >= 80.0 {
            "B"
        } else if score >= 70.0 {
            "C"
        } else if score >= 60.0 {
            "D"
        } else {
            "F"
        }
    }
}

/// Release decision
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleaseDecision {
    /// Approved for release
    Approved { reason: String },
    /// Rejected
    Rejected { reason: String },
    /// Pending more verification
    Pending { reason: String },
}

impl ReleaseDecision {
    /// Check if approved
    pub fn is_approved(&self) -> bool {
        matches!(self, ReleaseDecision::Approved { .. })
    }

    /// Get reason
    pub fn reason(&self) -> &str {
        match self {
            ReleaseDecision::Approved { reason }
            | ReleaseDecision::Rejected { reason }
            | ReleaseDecision::Pending { reason } => reason,
        }
    }
}

/// Audit trail entry
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Entry ID
    pub id: String,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Role performing action
    pub role: Role,
    /// Actor name
    pub actor: String,
    /// Action description
    pub action: String,
    /// Related artifact IDs
    pub artifacts: Vec<String>,
}

impl AuditEntry {
    /// Create a new audit entry
    pub fn new(id: &str, role: Role, actor: &str, action: &str) -> Self {
        Self {
            id: id.to_string(),
            timestamp: SystemTime::now(),
            role,
            actor: actor.to_string(),
            action: action.to_string(),
            artifacts: Vec::new(),
        }
    }

    /// Add related artifact
    pub fn with_artifact(mut self, artifact_id: &str) -> Self {
        self.artifacts.push(artifact_id.to_string());
        self
    }
}

/// Session state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Awaiting claims from Dev
    AwaitingClaims,
    /// Claims received, awaiting verification
    AwaitingVerification,
    /// Verification complete, awaiting decision
    AwaitingDecision,
    /// Session completed
    Completed,
}

/// Summary report of verification session
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// Session ID
    pub session_id: String,
    /// Total claims submitted
    pub total_claims: usize,
    /// Total artifacts generated
    pub total_artifacts: usize,
    /// Total verification attempts
    pub total_attempts: usize,
    /// Number of falsified claims
    pub falsified_count: usize,
    /// Number of unfalsified claims
    pub unfalsified_count: usize,
    /// Number of inconclusive attempts
    pub inconclusive_count: usize,
    /// Scorecard total
    pub scorecard_total: f64,
    /// Scorecard grade
    pub scorecard_grade: String,
    /// Audit trail entries
    pub audit_entries: usize,
}

impl VerificationReport {
    /// Check if report indicates success
    pub fn is_success(&self) -> bool {
        self.falsified_count == 0 && self.scorecard_total >= 70.0
    }
}
