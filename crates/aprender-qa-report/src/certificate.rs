//! Certificate Generation (QA-ART-03)
//!
//! Generates CERTIFICATE.md files for certified models.

use crate::mqs::MqsScore;
use crate::popperian::PopperianScore;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Certificate data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    /// Model identifier
    pub model_id: String,
    /// Model version
    pub version: String,
    /// Certification status
    pub status: CertificationStatus,
    /// MQS score (0-1000)
    pub mqs_score: u16,
    /// Score out of 170 (Verification Matrix)
    pub verification_score: u16,
    /// Maximum possible score
    pub max_score: u16,
    /// Grade (A+, A, B+, B, C, F)
    pub grade: String,
    /// Black swan events caught
    pub black_swans_caught: usize,
    /// Certification timestamp
    pub certified_at: DateTime<Utc>,
    /// Expiry date (certifications expire after 90 days)
    pub expires_at: DateTime<Utc>,
    /// Auditor/system that performed certification
    pub auditor: String,
    /// Hash of evidence.json for traceability
    pub evidence_hash: String,
}

/// Certification status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CertificationStatus {
    /// Score >= 95% AND zero P0 failures
    Certified,
    /// Score >= 80% AND zero P0 failures
    Provisional,
    /// Score < 80% OR any P0 failure
    Rejected,
}

/// Status determination and badge rendering for certification levels
impl CertificationStatus {
    /// Get status from score percentage and P0 failure count
    #[must_use]
    pub fn from_score(score_percent: f64, p0_failures: usize) -> Self {
        if p0_failures > 0 {
            return Self::Rejected;
        }
        if score_percent >= 95.0 {
            Self::Certified
        } else if score_percent >= 80.0 {
            Self::Provisional
        } else {
            Self::Rejected
        }
    }

    /// Get badge string for status
    #[must_use]
    pub const fn badge(&self) -> &'static str {
        match self {
            Self::Certified => "![CERTIFIED](https://img.shields.io/badge/CERTIFIED-brightgreen)",
            Self::Provisional => "![PROVISIONAL](https://img.shields.io/badge/PROVISIONAL-yellow)",
            Self::Rejected => "![REJECTED](https://img.shields.io/badge/REJECTED-red)",
        }
    }
}

impl std::fmt::Display for CertificationStatus {
    /// Format certification status as a human-readable uppercase string
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Certified => write!(f, "CERTIFIED"),
            Self::Provisional => write!(f, "PROVISIONAL"),
            Self::Rejected => write!(f, "REJECTED"),
        }
    }
}

/// Certificate generator
#[derive(Debug, Default)]
pub struct CertificateGenerator {
    /// Auditor name
    auditor: String,
}

/// Certificate creation and markdown rendering from MQS and Popperian scores
impl CertificateGenerator {
    /// Create a new certificate generator
    #[must_use]
    pub fn new(auditor: impl Into<String>) -> Self {
        Self {
            auditor: auditor.into(),
        }
    }

    /// Generate certificate from MQS and Popperian scores
    #[must_use]
    pub fn generate(
        &self,
        model_id: &str,
        version: &str,
        mqs: &MqsScore,
        popperian: &PopperianScore,
        evidence_hash: &str,
    ) -> Certificate {
        let now = Utc::now();
        let expires = now + chrono::Duration::days(90);

        // Calculate verification matrix score (scale to 170)
        let verification_score = ((mqs.raw_score as f64 / 1000.0) * 170.0) as u16;
        let score_percent = (verification_score as f64 / 170.0) * 100.0;

        // Count P0 failures from gateway failures
        let p0_failures = usize::from(!mqs.gateways_passed);

        let status = CertificationStatus::from_score(score_percent, p0_failures);

        Certificate {
            model_id: model_id.to_string(),
            version: version.to_string(),
            status,
            mqs_score: mqs.raw_score as u16,
            verification_score,
            max_score: 170,
            grade: mqs.grade.clone(),
            black_swans_caught: popperian.black_swan_count,
            certified_at: now,
            expires_at: expires,
            auditor: self.auditor.clone(),
            evidence_hash: evidence_hash.to_string(),
        }
    }

    /// Generate CERTIFICATE.md content
    #[must_use]
    pub fn to_markdown(&self, cert: &Certificate) -> String {
        format!(
            r#"# Model Certification Certificate

{badge}

## Model Information

| Field | Value |
|-------|-------|
| **Model ID** | `{model_id}` |
| **Version** | `{version}` |
| **Status** | **{status}** |

## Verification Matrix Score

| Metric | Value |
|--------|-------|
| **Score** | {score}/{max_score} ({percent:.1}%) |
| **MQS Score** | {mqs}/1000 |
| **Grade** | **{grade}** |
| **Black Swans Caught** | {black_swans} |

## Certification Details

| Field | Value |
|-------|-------|
| **Certified At** | {certified_at} |
| **Expires At** | {expires_at} |
| **Auditor** | {auditor} |
| **Evidence Hash** | `{evidence_hash}` |

## Popperian Falsification Statement

> This certificate attests that the model `{model_id}` has **survived** {score}/{max_score}
> rigorous falsification attempts without critical failure.
>
> In accordance with Popperian epistemology, this does NOT prove the model is correct.
> It demonstrates that we attempted to break it in {score} specific ways, and failed.

## Validity

This certificate is valid for **90 days** from the certification date.
Re-certification is required after any model update or configuration change.

---

*Generated by APR Model QA Playbook - Popperian Falsification Framework*
*Philosophy: Karl Popper (1959) + Nassim Taleb (2007) + Toyota Production System (1988)*
"#,
            badge = cert.status.badge(),
            model_id = cert.model_id,
            version = cert.version,
            status = cert.status,
            score = cert.verification_score,
            max_score = cert.max_score,
            percent = (cert.verification_score as f64 / cert.max_score as f64) * 100.0,
            mqs = cert.mqs_score,
            grade = cert.grade,
            black_swans = cert.black_swans_caught,
            certified_at = cert.certified_at.format("%Y-%m-%d %H:%M:%S UTC"),
            expires_at = cert.expires_at.format("%Y-%m-%d %H:%M:%S UTC"),
            auditor = cert.auditor,
            evidence_hash = cert.evidence_hash,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mqs::{CategoryScores, GatewayResult};

    /// Create a sample MqsScore with 920 raw score for testing
    fn test_mqs_score() -> MqsScore {
        MqsScore {
            model_id: "test/model".to_string(),
            raw_score: 920,
            normalized_score: 92.0,
            grade: "A".to_string(),
            gateways: vec![
                GatewayResult::passed("G1", "Model loads"),
                GatewayResult::passed("G2", "Basic inference"),
                GatewayResult::passed("G3", "Stability"),
                GatewayResult::passed("G4", "Output quality"),
            ],
            gateways_passed: true,
            categories: CategoryScores::default(),
            total_tests: 100,
            tests_passed: 95,
            tests_failed: 5,
            penalties: vec![],
            total_penalty: 0,
            proof_bonus: None,
        }
    }

    /// Create a sample PopperianScore with 95% corroboration for testing
    fn test_popperian_score() -> PopperianScore {
        PopperianScore {
            model_id: "test/model".to_string(),
            hypotheses_tested: 100,
            corroborated: 95,
            falsified: 5,
            inconclusive: 0,
            corroboration_ratio: 0.95,
            severity_weighted_score: 0.92,
            confidence_level: 0.9,
            reproducibility_index: 1.0,
            black_swan_count: 0,
            falsifications: vec![],
        }
    }

    /// Verify scores >= 95% with zero P0 failures yield Certified status
    #[test]
    fn test_certification_status_certified() {
        assert_eq!(
            CertificationStatus::from_score(95.0, 0),
            CertificationStatus::Certified
        );
        assert_eq!(
            CertificationStatus::from_score(100.0, 0),
            CertificationStatus::Certified
        );
    }

    /// Verify scores between 80% and 95% yield Provisional status
    #[test]
    fn test_certification_status_provisional() {
        assert_eq!(
            CertificationStatus::from_score(80.0, 0),
            CertificationStatus::Provisional
        );
        assert_eq!(
            CertificationStatus::from_score(94.9, 0),
            CertificationStatus::Provisional
        );
    }

    /// Verify scores below 80% or any P0 failures yield Rejected status
    #[test]
    fn test_certification_status_rejected() {
        assert_eq!(
            CertificationStatus::from_score(79.9, 0),
            CertificationStatus::Rejected
        );
        // Any P0 failure = rejected
        assert_eq!(
            CertificationStatus::from_score(100.0, 1),
            CertificationStatus::Rejected
        );
    }

    /// Verify CertificationStatus Display outputs uppercase status names
    #[test]
    fn test_certification_status_display() {
        assert_eq!(format!("{}", CertificationStatus::Certified), "CERTIFIED");
        assert_eq!(
            format!("{}", CertificationStatus::Provisional),
            "PROVISIONAL"
        );
        assert_eq!(format!("{}", CertificationStatus::Rejected), "REJECTED");
    }

    /// Verify badge strings contain correct color indicators
    #[test]
    fn test_certification_status_badge() {
        assert!(CertificationStatus::Certified
            .badge()
            .contains("brightgreen"));
        assert!(CertificationStatus::Provisional.badge().contains("yellow"));
        assert!(CertificationStatus::Rejected.badge().contains("red"));
    }

    /// Verify certificate generation populates all expected fields
    #[test]
    fn test_certificate_generation() {
        let generator = CertificateGenerator::new("APR QA System");
        let mqs = test_mqs_score();
        let popperian = test_popperian_score();

        let cert = generator.generate("test/model", "1.0.0", &mqs, &popperian, "abc123");

        assert_eq!(cert.model_id, "test/model");
        assert_eq!(cert.version, "1.0.0");
        assert_eq!(cert.mqs_score, 920);
        assert_eq!(cert.grade, "A");
        assert_eq!(cert.black_swans_caught, 0);
        assert_eq!(cert.evidence_hash, "abc123");
        assert!(cert.expires_at > cert.certified_at);
    }

    /// Verify markdown output contains all certificate sections
    #[test]
    fn test_certificate_markdown() {
        let generator = CertificateGenerator::new("APR QA System");
        let mqs = test_mqs_score();
        let popperian = test_popperian_score();

        let cert = generator.generate("test/model", "1.0.0", &mqs, &popperian, "abc123");
        let markdown = generator.to_markdown(&cert);

        assert!(markdown.contains("# Model Certification Certificate"));
        assert!(markdown.contains("test/model"));
        assert!(markdown.contains("1.0.0"));
        assert!(markdown.contains("abc123"));
        assert!(markdown.contains("Popperian Falsification"));
        assert!(markdown.contains("Karl Popper"));
    }

    /// Verify gateway failure produces Rejected certification status
    #[test]
    fn test_certificate_with_gateway_failure() {
        let generator = CertificateGenerator::new("APR QA System");
        let mut mqs = test_mqs_score();
        mqs.gateways_passed = false; // Gateway failure
        let popperian = test_popperian_score();

        let cert = generator.generate("test/model", "1.0.0", &mqs, &popperian, "abc123");

        assert_eq!(cert.status, CertificationStatus::Rejected);
    }

    /// Verify default CertificateGenerator has empty auditor
    #[test]
    fn test_certificate_generator_default() {
        let generator = CertificateGenerator::default();
        assert!(generator.auditor.is_empty());
    }

    /// Verify Certificate clone preserves all fields
    #[test]
    fn test_certificate_clone() {
        let generator = CertificateGenerator::new("Test");
        let mqs = test_mqs_score();
        let popperian = test_popperian_score();

        let cert = generator.generate("test/model", "1.0.0", &mqs, &popperian, "abc123");
        let cloned = cert.clone();

        assert_eq!(cloned.model_id, cert.model_id);
        assert_eq!(cloned.status, cert.status);
    }

    /// Verify perfect MQS of 1000 scales to maximum 170 verification score
    #[test]
    fn test_verification_score_scaling() {
        let generator = CertificateGenerator::new("Test");
        let mut mqs = test_mqs_score();
        mqs.raw_score = 1000; // Perfect MQS
        let popperian = test_popperian_score();

        let cert = generator.generate("test/model", "1.0.0", &mqs, &popperian, "abc123");

        assert_eq!(cert.verification_score, 170);
        assert_eq!(cert.max_score, 170);
    }
}
