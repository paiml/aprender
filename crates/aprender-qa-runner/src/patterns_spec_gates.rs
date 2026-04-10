
// ============================================================================
// VERIFICATION MATRIX GATE IDs (certified-testing.md spec)
// ============================================================================

/// Specification Gate IDs from the Verification Matrix (170 points)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpecGate {
    // Class I: Fundamental Integrity (P0 - CRITICAL) - 50 points
    /// F-INT-001: Memory Safety (10 pts)
    IntMemorySafety,
    /// F-INT-002: Process Termination (10 pts)
    IntProcessTermination,
    /// F-INT-003: Tensor Validity (10 pts)
    IntTensorValidity,
    /// F-INT-004: Format Fidelity (10 pts)
    IntFormatFidelity,
    /// F-INT-005: Determinism (10 pts)
    IntDeterminism,

    // Class II: Interface Compliance (P1 - HIGH) - 25 points
    /// F-API-001: JSON Compliance (5 pts)
    ApiJsonCompliance,
    /// F-API-002: Chat Template (5 pts)
    ApiChatTemplate,
    /// F-API-003: Health Check (5 pts)
    ApiHealthCheck,
    /// F-API-004: Error Handling (5 pts)
    ApiErrorHandling,
    /// F-API-005: Streaming (5 pts)
    ApiStreaming,

    // Class III: Numerical Stability (P1 - HIGH) - 20 points
    /// F-NUM-001: Attention Entropy (5 pts)
    NumAttentionEntropy,
    /// F-NUM-002: LayerNorm Drift (5 pts)
    NumLayerNormDrift,
    /// F-NUM-003: Softmax Sum (5 pts)
    NumSoftmaxSum,
    /// F-NUM-004: Token Probability (5 pts)
    NumTokenProbability,

    // Class IV: Cross-Platform Parity (P2 - MEDIUM) - 15 points
    /// F-PAR-001: CPU/GPU Equivalence (5 pts)
    ParCpuGpuEquivalence,
    /// F-PAR-002: Format Parity (5 pts)
    ParFormatParity,
    /// F-PAR-003: Quantization Impact (5 pts)
    ParQuantizationImpact,

    // Class V: Performance Boundaries (P2 - MEDIUM) - 20 points
    /// F-PERF-001: Minimum TPS (5 pts)
    PerfMinimumTps,
    /// F-PERF-002: TTFT (5 pts)
    PerfTtft,
    /// F-PERF-003: Memory Leak (5 pts)
    PerfMemoryLeak,
    /// F-PERF-004: GPU Utilization (5 pts)
    PerfGpuUtilization,

    // Class VI: Security & Safety (P0 - CRITICAL) - 30 points
    /// F-SEC-001: Path Traversal (10 pts)
    SecPathTraversal,
    /// F-SEC-002: Prompt Injection (10 pts)
    SecPromptInjection,
    /// F-SEC-003: Denial of Service (10 pts)
    SecDenialOfService,
}

/// Methods for gate identification, scoring, and enumeration
impl SpecGate {
    /// Get the gate ID string
    #[must_use]
    pub const fn id(&self) -> &'static str {
        match self {
            Self::IntMemorySafety => "F-INT-001",
            Self::IntProcessTermination => "F-INT-002",
            Self::IntTensorValidity => "F-INT-003",
            Self::IntFormatFidelity => "F-INT-004",
            Self::IntDeterminism => "F-INT-005",
            Self::ApiJsonCompliance => "F-API-001",
            Self::ApiChatTemplate => "F-API-002",
            Self::ApiHealthCheck => "F-API-003",
            Self::ApiErrorHandling => "F-API-004",
            Self::ApiStreaming => "F-API-005",
            Self::NumAttentionEntropy => "F-NUM-001",
            Self::NumLayerNormDrift => "F-NUM-002",
            Self::NumSoftmaxSum => "F-NUM-003",
            Self::NumTokenProbability => "F-NUM-004",
            Self::ParCpuGpuEquivalence => "F-PAR-001",
            Self::ParFormatParity => "F-PAR-002",
            Self::ParQuantizationImpact => "F-PAR-003",
            Self::PerfMinimumTps => "F-PERF-001",
            Self::PerfTtft => "F-PERF-002",
            Self::PerfMemoryLeak => "F-PERF-003",
            Self::PerfGpuUtilization => "F-PERF-004",
            Self::SecPathTraversal => "F-SEC-001",
            Self::SecPromptInjection => "F-SEC-002",
            Self::SecDenialOfService => "F-SEC-003",
        }
    }

    /// Get the point value for this gate
    #[must_use]
    pub const fn points(&self) -> u8 {
        match self {
            // P0 gates: 10 points
            Self::IntMemorySafety
            | Self::IntProcessTermination
            | Self::IntTensorValidity
            | Self::IntFormatFidelity
            | Self::IntDeterminism
            | Self::SecPathTraversal
            | Self::SecPromptInjection
            | Self::SecDenialOfService => 10,
            // P1/P2 gates: 5 points
            _ => 5,
        }
    }

    /// Get the priority level
    #[must_use]
    pub const fn priority(&self) -> &'static str {
        match self {
            Self::IntMemorySafety
            | Self::IntProcessTermination
            | Self::IntTensorValidity
            | Self::IntFormatFidelity
            | Self::IntDeterminism
            | Self::SecPathTraversal
            | Self::SecPromptInjection
            | Self::SecDenialOfService => "P0",
            Self::ApiJsonCompliance
            | Self::ApiChatTemplate
            | Self::ApiHealthCheck
            | Self::ApiErrorHandling
            | Self::ApiStreaming
            | Self::NumAttentionEntropy
            | Self::NumLayerNormDrift
            | Self::NumSoftmaxSum
            | Self::NumTokenProbability => "P1",
            _ => "P2",
        }
    }

    /// Get all gates
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::IntMemorySafety,
            Self::IntProcessTermination,
            Self::IntTensorValidity,
            Self::IntFormatFidelity,
            Self::IntDeterminism,
            Self::ApiJsonCompliance,
            Self::ApiChatTemplate,
            Self::ApiHealthCheck,
            Self::ApiErrorHandling,
            Self::ApiStreaming,
            Self::NumAttentionEntropy,
            Self::NumLayerNormDrift,
            Self::NumSoftmaxSum,
            Self::NumTokenProbability,
            Self::ParCpuGpuEquivalence,
            Self::ParFormatParity,
            Self::ParQuantizationImpact,
            Self::PerfMinimumTps,
            Self::PerfTtft,
            Self::PerfMemoryLeak,
            Self::PerfGpuUtilization,
            Self::SecPathTraversal,
            Self::SecPromptInjection,
            Self::SecDenialOfService,
        ]
    }

    /// Total points in the verification matrix
    #[must_use]
    pub fn total_points() -> u16 {
        Self::all().iter().map(|g| u16::from(g.points())).sum()
    }
}

// ============================================================================
// API COMPLIANCE CHECKS (F-API-001..005)
// ============================================================================

/// Result of API compliance check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiComplianceResult {
    /// Gate ID
    pub gate_id: String,
    /// Whether check passed
    pub passed: bool,
    /// Description of result
    pub description: String,
    /// Details/evidence
    pub details: Option<String>,
}

/// API compliance checker
pub struct ApiComplianceChecker;

/// Methods implementing F-API-001 through F-API-005 compliance checks
impl ApiComplianceChecker {
    /// F-API-001: Check JSON compliance
    #[must_use]
    pub fn check_json_compliance(response: &str) -> ApiComplianceResult {
        let passed = serde_json::from_str::<serde_json::Value>(response).is_ok();
        ApiComplianceResult {
            gate_id: SpecGate::ApiJsonCompliance.id().to_string(),
            passed,
            description: if passed {
                "Response is valid JSON".to_string()
            } else {
                "Response is malformed JSON".to_string()
            },
            details: if passed {
                None
            } else {
                Some("Failed to parse response as JSON".to_string())
            },
        }
    }

    /// F-API-002: Check for chat template leakage
    #[must_use]
    pub fn check_chat_template(output: &str) -> ApiComplianceResult {
        let control_tokens = [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|assistant|>",
            "<|user|>",
            "<|system|>",
            "[INST]",
            "[/INST]",
            "<<SYS>>",
            "<</SYS>>",
        ];
        let found: Vec<&str> = control_tokens
            .iter()
            .filter(|t| output.contains(*t))
            .copied()
            .collect();
        let passed = found.is_empty();
        ApiComplianceResult {
            gate_id: SpecGate::ApiChatTemplate.id().to_string(),
            passed,
            description: if passed {
                "No control token leakage".to_string()
            } else {
                "Control tokens leaked in output".to_string()
            },
            details: if passed {
                None
            } else {
                Some(format!("Found tokens: {found:?}"))
            },
        }
    }

    /// F-API-003: Check health endpoint response
    #[must_use]
    pub fn check_health_response(status_code: u16, response_time_ms: u64) -> ApiComplianceResult {
        let status_ok = status_code == 200;
        let time_ok = response_time_ms <= 1000;
        let passed = status_ok && time_ok;
        ApiComplianceResult {
            gate_id: SpecGate::ApiHealthCheck.id().to_string(),
            passed,
            description: if passed {
                format!("Health check OK ({response_time_ms}ms)")
            } else if !status_ok {
                format!("Health check returned {status_code}")
            } else {
                format!("Health check too slow ({response_time_ms}ms > 1000ms)")
            },
            details: None,
        }
    }

    /// F-API-004: Check error handling (invalid input should return 400, not crash)
    #[must_use]
    pub fn check_error_handling(
        status_code: u16,
        server_crashed: bool,
        has_error_message: bool,
    ) -> ApiComplianceResult {
        let passed = !server_crashed && status_code == 400 && has_error_message;
        ApiComplianceResult {
            gate_id: SpecGate::ApiErrorHandling.id().to_string(),
            passed,
            description: if server_crashed {
                "Server crashed on invalid input".to_string()
            } else if status_code != 400 {
                format!("Expected 400 Bad Request, got {status_code}")
            } else if !has_error_message {
                "Missing error message in response".to_string()
            } else {
                "Error handling correct".to_string()
            },
            details: None,
        }
    }

    /// F-API-005: Check SSE streaming format
    #[must_use]
    pub fn check_sse_format(stream_data: &str) -> ApiComplianceResult {
        let lines: Vec<&str> = stream_data.lines().collect();
        let mut issues = Vec::new();

        for (i, line) in lines.iter().enumerate() {
            if !line.is_empty() && !line.starts_with("data:") && !line.starts_with(':') {
                issues.push(format!("Line {}: missing 'data:' prefix", i + 1));
            }
        }

        let passed = issues.is_empty();
        ApiComplianceResult {
            gate_id: SpecGate::ApiStreaming.id().to_string(),
            passed,
            description: if passed {
                "SSE format valid".to_string()
            } else {
                "SSE format violations found".to_string()
            },
            details: if issues.is_empty() {
                None
            } else {
                Some(issues.join("; "))
            },
        }
    }
}

// ============================================================================
// PERFORMANCE VALIDATION (F-PERF-001..004)
// ============================================================================

/// Performance thresholds from spec
#[derive(Debug, Clone, Copy)]
pub struct PerformanceThresholds {
    /// Minimum tokens per second (F-PERF-001)
    pub min_tps: f64,
    /// Maximum time to first token in ms (F-PERF-002)
    pub max_ttft_ms: u64,
    /// Maximum memory growth percentage (F-PERF-003)
    pub max_memory_growth_percent: f64,
    /// Minimum GPU utilization (F-PERF-004)
    pub min_gpu_utilization: f64,
}

impl Default for PerformanceThresholds {
    /// Create performance thresholds with spec-defined default values
    fn default() -> Self {
        Self {
            min_tps: 10.0,
            max_ttft_ms: 2000,
            max_memory_growth_percent: 5.0,
            min_gpu_utilization: 50.0,
        }
    }
}

/// Result of performance check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCheckResult {
    /// Gate ID
    pub gate_id: String,
    /// Whether check passed
    pub passed: bool,
    /// Measured value
    pub measured: f64,
    /// Threshold value
    pub threshold: f64,
    /// Description
    pub description: String,
}
