use crate::http_client::*;
// ===========================================

/// Per spec QA-022: Recovery from GPU timeout without crash
#[derive(Debug, Clone)]
pub struct GpuTimeoutResult {
    /// Whether timeout occurred
    pub timeout_occurred: bool,
    /// Timeout duration (ms)
    pub timeout_ms: u64,
    /// Whether recovery was successful
    pub recovery_successful: bool,
    /// Whether GPU is still functional after recovery
    pub gpu_functional: bool,
    /// Meets QA-022 requirements
    pub meets_qa022: bool,
}

impl GpuTimeoutResult {
    pub fn no_timeout() -> Self {
        Self {
            timeout_occurred: false,
            timeout_ms: 0,
            recovery_successful: true,
            gpu_functional: true,
            meets_qa022: true,
        }
    }

    pub fn timeout_recovered(timeout_ms: u64) -> Self {
        Self {
            timeout_occurred: true,
            timeout_ms,
            recovery_successful: true,
            gpu_functional: true,
            meets_qa022: true,
        }
    }

    pub fn timeout_failed(timeout_ms: u64) -> Self {
        Self {
            timeout_occurred: true,
            timeout_ms,
            recovery_successful: false,
            gpu_functional: false,
            meets_qa022: false,
        }
    }
}

/// GPU health check
#[derive(Debug, Clone)]
pub struct GpuHealthCheck {
    /// GPU is responsive
    pub responsive: bool,
    /// GPU memory available
    pub memory_available_mb: f64,
    /// GPU compute available
    pub compute_available: bool,
    /// Last kernel execution time (ms)
    pub last_kernel_ms: f64,
}

impl GpuHealthCheck {
    pub fn healthy(memory_mb: f64) -> Self {
        Self {
            responsive: true,
            memory_available_mb: memory_mb,
            compute_available: true,
            last_kernel_ms: 0.0,
        }
    }

    pub fn degraded(memory_mb: f64, kernel_ms: f64) -> Self {
        Self {
            responsive: true,
            memory_available_mb: memory_mb,
            compute_available: true,
            last_kernel_ms: kernel_ms,
        }
    }

    pub fn unresponsive() -> Self {
        Self {
            responsive: false,
            memory_available_mb: 0.0,
            compute_available: false,
            last_kernel_ms: f64::INFINITY,
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.responsive && self.compute_available && self.last_kernel_ms < 1000.0
    }
}

/// IMP-175a: Test GPU timeout result types
#[test]
fn test_imp_175a_gpu_timeout_result() {
    let no_timeout = GpuTimeoutResult::no_timeout();
    assert!(
        no_timeout.meets_qa022,
        "IMP-175a: No timeout should meet QA-022"
    );
    assert!(
        !no_timeout.timeout_occurred,
        "IMP-175a: No timeout should not have timeout"
    );

    let recovered = GpuTimeoutResult::timeout_recovered(5000);
    assert!(
        recovered.meets_qa022,
        "IMP-175a: Recovered timeout should meet QA-022"
    );
    assert!(
        recovered.timeout_occurred,
        "IMP-175a: Recovered should have timeout"
    );
    assert!(
        recovered.gpu_functional,
        "IMP-175a: Recovered should have functional GPU"
    );

    let failed = GpuTimeoutResult::timeout_failed(30000);
    assert!(
        !failed.meets_qa022,
        "IMP-175a: Failed recovery should NOT meet QA-022"
    );
    assert!(
        !failed.gpu_functional,
        "IMP-175a: Failed should have non-functional GPU"
    );

    println!("\nIMP-175a: GPU Timeout Results:");
    println!("  No timeout: meets_qa022={}", no_timeout.meets_qa022);
    println!(
        "  Recovered: meets_qa022={}, timeout={}ms",
        recovered.meets_qa022, recovered.timeout_ms
    );
    println!(
        "  Failed: meets_qa022={}, gpu_functional={}",
        failed.meets_qa022, failed.gpu_functional
    );
}

/// IMP-175b: Test GPU health check
#[test]
fn test_imp_175b_gpu_health_check() {
    let healthy = GpuHealthCheck::healthy(8000.0);
    assert!(
        healthy.is_healthy(),
        "IMP-175b: Healthy GPU should be healthy"
    );
    assert!(
        healthy.responsive,
        "IMP-175b: Healthy GPU should be responsive"
    );

    let degraded = GpuHealthCheck::degraded(4000.0, 500.0);
    assert!(
        degraded.is_healthy(),
        "IMP-175b: Degraded but responsive should be healthy"
    );

    let unresponsive = GpuHealthCheck::unresponsive();
    assert!(
        !unresponsive.is_healthy(),
        "IMP-175b: Unresponsive GPU should not be healthy"
    );

    println!("\nIMP-175b: GPU Health Check:");
    println!(
        "  Healthy: responsive={}, memory={:.0}MB",
        healthy.responsive, healthy.memory_available_mb
    );
    println!(
        "  Degraded: responsive={}, kernel={:.0}ms",
        degraded.responsive, degraded.last_kernel_ms
    );
    println!("  Unresponsive: responsive={}", unresponsive.responsive);
}

/// Timeout recovery strategy
#[derive(Debug, Clone)]
pub struct TimeoutRecoveryPlan {
    /// Retry count
    pub max_retries: usize,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Initial timeout (ms)
    pub initial_timeout_ms: u64,
    /// Whether to reset GPU state
    pub reset_gpu_state: bool,
}

impl TimeoutRecoveryPlan {
    pub fn default_plan() -> Self {
        Self {
            max_retries: 3,
            backoff_multiplier: 2.0,
            initial_timeout_ms: 5000,
            reset_gpu_state: true,
        }
    }

    pub fn aggressive() -> Self {
        Self {
            max_retries: 5,
            backoff_multiplier: 1.5,
            initial_timeout_ms: 2000,
            reset_gpu_state: true,
        }
    }

    pub fn timeout_at_retry(&self, retry: usize) -> u64 {
        (self.initial_timeout_ms as f64 * self.backoff_multiplier.powi(retry as i32)) as u64
    }
}

/// IMP-175c: Test timeout recovery planning
#[test]
fn test_imp_175c_recovery_planning() {
    let default_plan = TimeoutRecoveryPlan::default_plan();
    assert_eq!(
        default_plan.max_retries, 3,
        "IMP-175c: Default should have 3 retries"
    );
    assert_eq!(
        default_plan.timeout_at_retry(0),
        5000,
        "IMP-175c: First retry timeout"
    );
    assert_eq!(
        default_plan.timeout_at_retry(1),
        10000,
        "IMP-175c: Second retry timeout (2x)"
    );
    assert_eq!(
        default_plan.timeout_at_retry(2),
        20000,
        "IMP-175c: Third retry timeout (4x)"
    );

    let aggressive = TimeoutRecoveryPlan::aggressive();
    assert_eq!(
        aggressive.max_retries, 5,
        "IMP-175c: Aggressive should have 5 retries"
    );
    assert!(
        aggressive.initial_timeout_ms < default_plan.initial_timeout_ms,
        "IMP-175c: Aggressive should have shorter initial timeout"
    );

    println!("\nIMP-175c: Timeout Recovery Planning:");
    println!(
        "  Default: retries={}, initial={}ms, backoff={:.1}x",
        default_plan.max_retries, default_plan.initial_timeout_ms, default_plan.backoff_multiplier
    );
    println!(
        "  Aggressive: retries={}, initial={}ms, backoff={:.1}x",
        aggressive.max_retries, aggressive.initial_timeout_ms, aggressive.backoff_multiplier
    );
}

/// IMP-175d: Real-world GPU timeout recovery
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_175d_realworld_timeout_recovery() {
    // Use very short timeout to trigger timeout behavior
    let client = ModelHttpClient::with_timeout(1); // 1 second timeout

    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Write a very long story about ".to_string(),
        max_tokens: 500, // Long generation to trigger timeout
        temperature: Some(0.7),
        stream: false,
    };

    let start = std::time::Instant::now();
    let result = client.llamacpp_completion("http://127.0.0.1:8082", &request);
    let elapsed = start.elapsed().as_millis() as u64;

    let timeout_result = match result {
        Ok(_) => GpuTimeoutResult::no_timeout(),
        Err(_) => {
            // Try a simple request to verify GPU still works
            let simple_request = CompletionRequest {
                model: "default".to_string(),
                prompt: "Hi".to_string(),
                max_tokens: 5,
                temperature: Some(0.0),
                stream: false,
            };

            let recovery_client = ModelHttpClient::with_timeout(30);
            match recovery_client.llamacpp_completion("http://127.0.0.1:8082", &simple_request) {
                Ok(_) => GpuTimeoutResult::timeout_recovered(elapsed),
                Err(_) => GpuTimeoutResult::timeout_failed(elapsed),
            }
        },
    };

    println!("\nIMP-175d: Real-World GPU Timeout Recovery:");
    println!("  Timeout occurred: {}", timeout_result.timeout_occurred);
    println!(
        "  Recovery successful: {}",
        timeout_result.recovery_successful
    );
    println!("  GPU functional: {}", timeout_result.gpu_functional);
    println!(
        "  QA-022: {}",
        if timeout_result.meets_qa022 {
            "PASS"
        } else {
            "FAIL"
        }
    );
}

// ===========================================
// IMP-176: Malformed GGUF Handling (QA-023)
// ===========================================

/// Per spec QA-023: Correct behavior on malformed GGUF files
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValidationError {
    /// Invalid magic number
    InvalidMagic,
    /// Unsupported version
    UnsupportedVersion,
    /// Corrupted header
    CorruptedHeader,
    /// Invalid tensor metadata
    InvalidTensorMeta,
    /// Checksum mismatch
    ChecksumMismatch,
    /// Truncated data
    TruncatedData,
}

/// GGUF validation result
#[derive(Debug, Clone)]
pub struct GgufValidationResult {
    /// File path
    pub file_path: String,
    /// Whether file is valid
    pub is_valid: bool,
    /// Validation errors found
    pub errors: Vec<GgufValidationError>,
    /// Whether error was handled gracefully
    pub graceful_handling: bool,
    /// Meets QA-023 requirements
    pub meets_qa023: bool,
}

impl GgufValidationResult {
    pub fn valid(path: &str) -> Self {
        Self {
            file_path: path.to_string(),
            is_valid: true,
            errors: Vec::new(),
            graceful_handling: true,
            meets_qa023: true,
        }
    }

    pub fn invalid_graceful(path: &str, errors: Vec<GgufValidationError>) -> Self {
        Self {
            file_path: path.to_string(),
            is_valid: false,
            errors,
            graceful_handling: true,
            meets_qa023: true, // Graceful handling meets QA-023
        }
    }

    pub fn invalid_crash(path: &str, errors: Vec<GgufValidationError>) -> Self {
        Self {
            file_path: path.to_string(),
            is_valid: false,
            errors,
            graceful_handling: false,
            meets_qa023: false, // Crash does NOT meet QA-023
        }
    }
}

/// IMP-176a: Test GGUF validation error types
#[test]
fn test_imp_176a_gguf_validation_errors() {
    let valid = GgufValidationResult::valid("model.gguf");
    assert!(valid.is_valid, "IMP-176a: Valid file should be valid");
    assert!(valid.meets_qa023, "IMP-176a: Valid file should meet QA-023");

    let invalid_magic =
        GgufValidationResult::invalid_graceful("bad.gguf", vec![GgufValidationError::InvalidMagic]);
    assert!(
        !invalid_magic.is_valid,
        "IMP-176a: Invalid magic should be invalid"
    );
    assert!(
        invalid_magic.meets_qa023,
        "IMP-176a: Graceful handling should meet QA-023"
    );

    let crash = GgufValidationResult::invalid_crash(
        "crash.gguf",
        vec![GgufValidationError::CorruptedHeader],
    );
    assert!(!crash.meets_qa023, "IMP-176a: Crash should NOT meet QA-023");

    println!("\nIMP-176a: GGUF Validation Errors:");
    println!(
        "  Valid: is_valid={}, meets_qa023={}",
        valid.is_valid, valid.meets_qa023
    );
    println!(
        "  Invalid (graceful): errors={:?}, meets_qa023={}",
        invalid_magic.errors, invalid_magic.meets_qa023
    );
    println!(
        "  Crash: graceful={}, meets_qa023={}",
        crash.graceful_handling, crash.meets_qa023
    );
}

/// GGUF magic number validator
#[derive(Debug)]
pub struct GgufMagicValidator;

include!("validate_bytes_gguf.rs");
include!("handled_gracefully_panicked.rs");
include!("imp_179d.rs");
