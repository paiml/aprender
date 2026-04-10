/// Certification tier for tier-aware scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CertificationTier {
    /// MVP tier: Tests 18 combinations (3 formats × 2 backends × 3 modalities).
    /// Pass = B grade (800 score), PROVISIONAL status.
    #[default]
    Mvp,
    /// Full tier: Complete 170-point verification matrix.
    /// Pass = A+ grade (950+ score), CERTIFIED status.
    Full,
}

/// MVP tier pass threshold (90% pass rate).
pub const MVP_PASS_THRESHOLD: f64 = 0.90;

/// MVP tier pass score (B grade).
pub const MVP_PASS_SCORE: u32 = 800;

/// Full tier pass threshold (95% on verification matrix).
pub const FULL_PASS_THRESHOLD: f64 = 0.95;

/// Full tier pass score (A+ grade).
pub const FULL_PASS_SCORE: u32 = 950;

/// Calculate certification status from MQS score.
///
/// Threshold: `MVP_PASS_SCORE` (800) for Certified, 700 for Provisional.
/// Must stay aligned with `CertificationRow::derive_status()`.
#[must_use]
pub const fn status_from_score(mqs_score: u32, has_p0_failure: bool) -> CertificationStatus {
    if has_p0_failure {
        CertificationStatus::Blocked
    } else if mqs_score >= MVP_PASS_SCORE {
        CertificationStatus::Certified
    } else if mqs_score >= 700 {
        CertificationStatus::Provisional
    } else {
        CertificationStatus::Blocked
    }
}

/// Calculate certification status for a specific tier.
///
/// # Arguments
/// * `tier` - The certification tier (MVP or Full)
/// * `pass_rate` - The pass rate from test execution (0.0 to 1.0)
/// * `has_p0_failure` - Whether any P0 (critical) test failed
#[must_use]
pub fn status_from_tier(
    tier: CertificationTier,
    pass_rate: f64,
    has_p0_failure: bool,
) -> CertificationStatus {
    if has_p0_failure {
        return CertificationStatus::Blocked;
    }

    match tier {
        CertificationTier::Mvp => {
            if pass_rate >= MVP_PASS_THRESHOLD {
                CertificationStatus::Provisional
            } else {
                CertificationStatus::Blocked
            }
        }
        CertificationTier::Full => {
            if pass_rate >= FULL_PASS_THRESHOLD {
                CertificationStatus::Certified
            } else if pass_rate >= MVP_PASS_THRESHOLD {
                CertificationStatus::Provisional
            } else {
                CertificationStatus::Blocked
            }
        }
    }
}

/// Convert pass rate to a scaled score, clamping to valid range.
#[inline]
fn scale_to_f_grade(pass_rate: f64) -> u32 {
    // Clamp pass_rate to [0.0, 1.0] and scale to [0, 699]
    let clamped = pass_rate.clamp(0.0, 1.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let score = (clamped * 699.0) as u32;
    score.min(699)
}

/// Calculate MQS score for a specific tier.
///
/// # Arguments
/// * `tier` - The certification tier (MVP or Full)
/// * `pass_rate` - The pass rate from test execution (0.0 to 1.0)
/// * `has_p0_failure` - Whether any P0 (critical) test failed
#[must_use]
pub fn score_from_tier(tier: CertificationTier, pass_rate: f64, has_p0_failure: bool) -> u32 {
    if has_p0_failure {
        // Scale score based on pass rate, max 699 (F grade)
        return scale_to_f_grade(pass_rate);
    }

    match tier {
        CertificationTier::Mvp => {
            if pass_rate >= MVP_PASS_THRESHOLD {
                // MVP pass: B grade (800)
                MVP_PASS_SCORE
            } else {
                // Scale score based on pass rate, max 699 (F grade)
                scale_to_f_grade(pass_rate)
            }
        }
        CertificationTier::Full => {
            if pass_rate >= FULL_PASS_THRESHOLD {
                // Full pass: A+ grade (950+)
                let bonus = ((pass_rate - FULL_PASS_THRESHOLD) * 1000.0).clamp(0.0, 50.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let bonus_u32 = bonus as u32;
                FULL_PASS_SCORE + bonus_u32
            } else if pass_rate >= MVP_PASS_THRESHOLD {
                // Between MVP and Full threshold: B to B+ range (800-899)
                let ratio =
                    (pass_rate - MVP_PASS_THRESHOLD) / (FULL_PASS_THRESHOLD - MVP_PASS_THRESHOLD);
                let bonus = (ratio * 99.0).clamp(0.0, 99.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let bonus_u32 = bonus as u32;
                800 + bonus_u32
            } else {
                // Below MVP threshold: F grade
                scale_to_f_grade(pass_rate)
            }
        }
    }
}

/// Calculate letter grade from MQS score.
#[must_use]
pub const fn grade_from_score(mqs_score: u32) -> &'static str {
    match mqs_score {
        950..=1000 => "A+",
        900..=949 => "A",
        850..=899 => "B+",
        800..=849 => "B",
        700..=799 => "C",
        _ => "F",
    }
}

/// Calculate grade for a specific tier.
///
/// # Arguments
/// * `tier` - The certification tier (MVP or Full)
/// * `pass_rate` - The pass rate from test execution (0.0 to 1.0)
/// * `has_p0_failure` - Whether any P0 (critical) test failed
#[must_use]
pub fn grade_from_tier(
    tier: CertificationTier,
    pass_rate: f64,
    has_p0_failure: bool,
) -> &'static str {
    let score = score_from_tier(tier, pass_rate, has_p0_failure);
    grade_from_score(score)
}

/// Update README content with new certification table.
///
/// # Errors
///
/// Returns `CertifyError::MarkerNotFound` if the markers are not found.
pub fn update_readme(readme: &str, table_content: &str) -> Result<String> {
    let start_idx = readme
        .find(START_MARKER)
        .ok_or_else(|| CertifyError::MarkerNotFound(START_MARKER.to_string()))?;
    let end_idx = readme
        .find(END_MARKER)
        .ok_or_else(|| CertifyError::MarkerNotFound(END_MARKER.to_string()))?;

    if start_idx >= end_idx {
        return Err(CertifyError::MarkerNotFound(format!(
            "START marker (pos {start_idx}) must appear before END marker (pos {end_idx})"
        )));
    }

    let before = &readme[..start_idx + START_MARKER.len()];
    let after = &readme[end_idx..];

    Ok(format!("{before}\n{table_content}\n{after}"))
}
