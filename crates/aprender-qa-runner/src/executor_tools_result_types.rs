
/// Result of a tool test
#[derive(Debug, Clone)]
pub struct ToolTestResult {
    /// Tool name
    pub tool: String,
    /// Whether test passed
    pub passed: bool,
    /// Exit code
    pub exit_code: i32,
    /// Stdout output
    pub stdout: String,
    /// Stderr output
    pub stderr: String,
    /// Duration in ms
    pub duration_ms: u64,
    /// Gate ID for this test
    pub gate_id: String,
}

impl ToolTestResult {
    /// Convert to Evidence
    #[must_use]
    pub fn to_evidence(&self, model_id: &ModelId) -> Evidence {
        let scenario = QaScenario::new(
            model_id.clone(),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            format!("apr {} test", self.tool),
            0,
        );

        if self.passed {
            Evidence::corroborated(&self.gate_id, scenario, &self.stdout, self.duration_ms)
        } else {
            Evidence::falsified(
                &self.gate_id,
                scenario,
                format!("Exit code: {}, stderr: {}", self.exit_code, self.stderr),
                &self.stdout,
                self.duration_ms,
            )
        }
    }
}

/// Result of playbook execution
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Playbook name
    pub playbook_name: String,
    /// Total scenarios
    pub total_scenarios: usize,
    /// Passed scenarios
    pub passed: usize,
    /// Failed scenarios
    pub failed: usize,
    /// Skipped scenarios
    pub skipped: usize,
    /// Total duration in milliseconds
    pub duration_ms: u64,
    /// Gateway failure (if any)
    pub gateway_failed: Option<String>,
    /// Collected evidence
    pub evidence: EvidenceCollector,
}

impl ExecutionResult {
    /// Check if execution was successful
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.gateway_failed.is_none() && self.failed == 0
    }

    /// Get pass rate as percentage
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        if self.total_scenarios == 0 {
            return 0.0;
        }
        (self.passed as f64 / self.total_scenarios as f64) * 100.0
    }
}

