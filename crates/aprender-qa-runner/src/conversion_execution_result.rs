
/// Result of conversion test execution
#[derive(Debug)]
pub struct ConversionExecutionResult {
    /// Total tests run
    pub total: usize,
    /// Tests passed
    pub passed: usize,
    /// Tests failed
    pub failed: usize,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Individual results
    pub results: Vec<ConversionResult>,
    /// Evidence collected
    pub evidence: Vec<Evidence>,
}

impl ConversionExecutionResult {
    /// Check if all conversion tests passed
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }

    /// Get pass rate as percentage
    ///
    /// Returns 0.0 for 0 total tests (Popperian: untested ≠ passed).
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.passed as f64 / self.total as f64) * 100.0
        }
    }
}

/// Convert ConversionResult to Evidence
impl From<ConversionResult> for Evidence {
    fn from(result: ConversionResult) -> Self {
        match result {
            ConversionResult::Corroborated {
                source_format,
                target_format,
                backend,
                max_diff,
            } => {
                let scenario = QaScenario::new(
                    ModelId::new("conversion", "test"),
                    Modality::Run,
                    backend,
                    target_format,
                    format!("Convert {source_format:?} to {target_format:?}"),
                    0,
                );
                Evidence::corroborated(
                    &format!("F-CONV-{source_format:?}-{target_format:?}"),
                    scenario,
                    &format!("Conversion successful, max_diff: {max_diff:.2e}"),
                    0,
                )
            }
            ConversionResult::Falsified {
                gate_id,
                reason,
                evidence,
            } => {
                let scenario = QaScenario::new(
                    ModelId::new("conversion", "test"),
                    Modality::Run,
                    evidence.backend,
                    evidence.target_format,
                    format!(
                        "Convert {:?} to {:?}",
                        evidence.source_format, evidence.target_format
                    ),
                    0,
                );
                Evidence::falsified(&gate_id, scenario, reason, &evidence.converted_hash, 0)
            }
        }
    }
}
