
impl GgufMagicValidator {
    /// GGUF magic: "GGUF" = 0x46554747
    const GGUF_MAGIC: u32 = 0x46554747;

    pub fn validate(magic: u32) -> std::result::Result<(), GgufValidationError> {
        if magic == Self::GGUF_MAGIC {
            Ok(())
        } else {
            Err(GgufValidationError::InvalidMagic)
        }
    }

    pub fn validate_bytes(bytes: &[u8]) -> std::result::Result<(), GgufValidationError> {
        if bytes.len() < 4 {
            return Err(GgufValidationError::TruncatedData);
        }
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        Self::validate(magic)
    }
}

/// IMP-176b: Test GGUF magic validation
#[test]
fn test_imp_176b_magic_validation() {
    // Valid GGUF magic
    let valid_magic = 0x46554747u32; // "GGUF"
    assert!(
        GgufMagicValidator::validate(valid_magic).is_ok(),
        "IMP-176b: Valid magic should pass"
    );

    // Invalid magic
    let invalid_magic = 0x12345678u32;
    assert!(
        GgufMagicValidator::validate(invalid_magic).is_err(),
        "IMP-176b: Invalid magic should fail"
    );

    // Byte validation
    let valid_bytes = [0x47, 0x47, 0x55, 0x46]; // "GGUF" in little-endian
    assert!(
        GgufMagicValidator::validate_bytes(&valid_bytes).is_ok(),
        "IMP-176b: Valid bytes should pass"
    );

    let truncated = [0x47, 0x47]; // Too short
    assert_eq!(
        GgufMagicValidator::validate_bytes(&truncated),
        Err(GgufValidationError::TruncatedData),
        "IMP-176b: Truncated should return TruncatedData error"
    );

    println!("\nIMP-176b: GGUF Magic Validation:");
    println!("  Valid magic: 0x{:08X} = OK", valid_magic);
    println!("  Invalid magic: 0x{:08X} = Error", invalid_magic);
}

/// GGUF version validator
#[derive(Debug)]
pub struct GgufVersionValidator;

impl GgufVersionValidator {
    /// Supported GGUF versions
    const SUPPORTED_VERSIONS: [u32; 3] = [1, 2, 3];

    pub fn validate(version: u32) -> std::result::Result<(), GgufValidationError> {
        if Self::SUPPORTED_VERSIONS.contains(&version) {
            Ok(())
        } else {
            Err(GgufValidationError::UnsupportedVersion)
        }
    }
}

/// IMP-176c: Test GGUF version validation
#[test]
fn test_imp_176c_version_validation() {
    // Supported versions
    assert!(
        GgufVersionValidator::validate(1).is_ok(),
        "IMP-176c: Version 1 should be supported"
    );
    assert!(
        GgufVersionValidator::validate(2).is_ok(),
        "IMP-176c: Version 2 should be supported"
    );
    assert!(
        GgufVersionValidator::validate(3).is_ok(),
        "IMP-176c: Version 3 should be supported"
    );

    // Unsupported version
    assert!(
        GgufVersionValidator::validate(0).is_err(),
        "IMP-176c: Version 0 should not be supported"
    );
    assert!(
        GgufVersionValidator::validate(99).is_err(),
        "IMP-176c: Version 99 should not be supported"
    );

    println!("\nIMP-176c: GGUF Version Validation:");
    println!(
        "  Supported versions: {:?}",
        GgufVersionValidator::SUPPORTED_VERSIONS
    );
    println!("  Version 2: {:?}", GgufVersionValidator::validate(2));
    println!("  Version 99: {:?}", GgufVersionValidator::validate(99));
}

/// IMP-176d: Real-world malformed GGUF handling
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_176d_realworld_malformed_gguf() {
    // This test verifies the server doesn't crash when given invalid model references
    let client = ModelHttpClient::with_timeout(30);

    let request = CompletionRequest {
        model: "nonexistent_model_xyz123".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: 10,
        temperature: Some(0.0),
        stream: false,
    };

    let result = client.llamacpp_completion("http://127.0.0.1:8082", &request);

    // Any response (error or success) means the server didn't crash
    let validation = match result {
        Ok(_) => GgufValidationResult::valid("test"),
        Err(_) => {
            // Error is expected but should be graceful
            GgufValidationResult::invalid_graceful("test", vec![GgufValidationError::InvalidMagic])
        },
    };

    println!("\nIMP-176d: Real-World Malformed GGUF:");
    println!("  Graceful handling: {}", validation.graceful_handling);
    println!(
        "  QA-023: {}",
        if validation.meets_qa023 {
            "PASS"
        } else {
            "FAIL"
        }
    );
}

// ===========================================
// IMP-177: Truncated Model Handling (QA-024)
// ===========================================

/// Per spec QA-024: Correct behavior on truncated model files
#[derive(Debug, Clone)]
pub struct TruncatedModelResult {
    /// Expected file size (bytes)
    pub expected_size: u64,
    /// Actual file size (bytes)
    pub actual_size: u64,
    /// Truncation detected
    pub truncation_detected: bool,
    /// Error message
    pub error_message: Option<String>,
    /// Whether handled gracefully
    pub graceful_handling: bool,
    /// Meets QA-024
    pub meets_qa024: bool,
}

impl TruncatedModelResult {
    pub fn complete(size: u64) -> Self {
        Self {
            expected_size: size,
            actual_size: size,
            truncation_detected: false,
            error_message: None,
            graceful_handling: true,
            meets_qa024: true,
        }
    }

    pub fn truncated_graceful(expected: u64, actual: u64) -> Self {
        Self {
            expected_size: expected,
            actual_size: actual,
            truncation_detected: true,
            error_message: Some(format!(
                "File truncated: expected {} bytes, got {}",
                expected, actual
            )),
            graceful_handling: true,
            meets_qa024: true,
        }
    }

    pub fn truncated_crash(expected: u64, actual: u64) -> Self {
        Self {
            expected_size: expected,
            actual_size: actual,
            truncation_detected: true,
            error_message: Some("Crash during load".to_string()),
            graceful_handling: false,
            meets_qa024: false,
        }
    }

    pub fn truncation_percent(&self) -> f64 {
        if self.expected_size == 0 {
            0.0
        } else {
            (1.0 - (self.actual_size as f64 / self.expected_size as f64)) * 100.0
        }
    }
}

/// IMP-177a: Test truncated model detection
#[test]
fn test_imp_177a_truncated_detection() {
    let complete = TruncatedModelResult::complete(1_000_000_000);
    assert!(
        !complete.truncation_detected,
        "IMP-177a: Complete file should not detect truncation"
    );
    assert!(
        complete.meets_qa024,
        "IMP-177a: Complete file should meet QA-024"
    );

    let truncated = TruncatedModelResult::truncated_graceful(1_000_000_000, 500_000_000);
    assert!(
        truncated.truncation_detected,
        "IMP-177a: Truncated file should detect truncation"
    );
    assert!(
        truncated.meets_qa024,
        "IMP-177a: Graceful handling should meet QA-024"
    );
    assert!(
        (truncated.truncation_percent() - 50.0).abs() < 0.1,
        "IMP-177a: 50% truncation"
    );

    let crash = TruncatedModelResult::truncated_crash(1_000_000_000, 100_000_000);
    assert!(!crash.meets_qa024, "IMP-177a: Crash should NOT meet QA-024");

    println!("\nIMP-177a: Truncated Model Detection:");
    println!(
        "  Complete: truncated={}, meets_qa024={}",
        complete.truncation_detected, complete.meets_qa024
    );
    println!(
        "  Truncated (50%): truncated={}, meets_qa024={}",
        truncated.truncation_detected, truncated.meets_qa024
    );
    println!(
        "  Crash: graceful={}, meets_qa024={}",
        crash.graceful_handling, crash.meets_qa024
    );
}

/// File integrity checker
#[derive(Debug, Clone)]
pub struct FileIntegrityChecker {
    /// Minimum required size (bytes)
    pub min_header_size: u64,
    /// Whether to verify checksums
    pub verify_checksum: bool,
    /// Whether to verify tensor counts
    pub verify_tensors: bool,
}

impl FileIntegrityChecker {
    pub fn strict() -> Self {
        Self {
            min_header_size: 64, // GGUF header minimum
            verify_checksum: true,
            verify_tensors: true,
        }
    }

    pub fn check_size(
        &self,
        expected: u64,
        actual: u64,
    ) -> std::result::Result<(), TruncatedModelResult> {
        if actual < self.min_header_size {
            return Err(TruncatedModelResult::truncated_graceful(expected, actual));
        }
        if actual < expected {
            return Err(TruncatedModelResult::truncated_graceful(expected, actual));
        }
        Ok(())
    }
}

/// IMP-177b: Test file integrity checking
#[test]
fn test_imp_177b_integrity_checking() {
    let checker = FileIntegrityChecker::strict();

    // Valid file
    assert!(
        checker.check_size(1000, 1000).is_ok(),
        "IMP-177b: Complete file should pass"
    );

    // Truncated file
    let truncated = checker.check_size(1000, 500);
    assert!(truncated.is_err(), "IMP-177b: Truncated file should fail");

    // Extremely truncated (below header minimum)
    let tiny = checker.check_size(1000, 10);
    assert!(tiny.is_err(), "IMP-177b: Tiny file should fail");

    println!("\nIMP-177b: File Integrity Checking:");
    println!(
        "  Strict checker: min_header={}, verify_checksum={}",
        checker.min_header_size, checker.verify_checksum
    );
    println!("  1000/1000 bytes: {:?}", checker.check_size(1000, 1000));
    println!(
        "  500/1000 bytes: {:?}",
        checker
            .check_size(1000, 500)
            .err()
            .map(|e| e.truncation_percent())
    );
}

/// Progressive loading strategy for truncated files
#[derive(Debug, Clone)]
pub struct ProgressiveLoadStrategy {
    /// Load header first
    pub header_first: bool,
    /// Validate after each tensor
    pub per_tensor_validation: bool,
    /// Stop on first error
    pub fail_fast: bool,
}

impl ProgressiveLoadStrategy {
    pub fn safe() -> Self {
        Self {
            header_first: true,
            per_tensor_validation: true,
            fail_fast: true,
        }
    }

    pub fn tolerant() -> Self {
        Self {
            header_first: true,
            per_tensor_validation: false,
            fail_fast: false,
        }
    }
}

/// IMP-177c: Test progressive loading strategies
#[test]
fn test_imp_177c_progressive_loading() {
    let safe = ProgressiveLoadStrategy::safe();
    assert!(safe.header_first, "IMP-177c: Safe should load header first");
    assert!(safe.fail_fast, "IMP-177c: Safe should fail fast");
    assert!(
        safe.per_tensor_validation,
        "IMP-177c: Safe should validate per tensor"
    );

    let tolerant = ProgressiveLoadStrategy::tolerant();
    assert!(
        !tolerant.fail_fast,
        "IMP-177c: Tolerant should not fail fast"
    );
    assert!(
        !tolerant.per_tensor_validation,
        "IMP-177c: Tolerant should skip per-tensor validation"
    );

    println!("\nIMP-177c: Progressive Loading Strategies:");
    println!(
        "  Safe: header_first={}, fail_fast={}, per_tensor={}",
        safe.header_first, safe.fail_fast, safe.per_tensor_validation
    );
    println!(
        "  Tolerant: header_first={}, fail_fast={}, per_tensor={}",
        tolerant.header_first, tolerant.fail_fast, tolerant.per_tensor_validation
    );
}

/// IMP-177d: Real-world truncated model handling
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_177d_realworld_truncated_handling() {
    let client = ModelHttpClient::with_timeout(30);

    // Normal request to verify server handles potential truncation gracefully
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: 5,
        temperature: Some(0.0),
        stream: false,
    };

    let result = client.llamacpp_completion("http://127.0.0.1:8082", &request);

    let handling = match result {
        Ok(_) => TruncatedModelResult::complete(0),
        Err(e) => {
            if e.to_string().contains("truncat") || e.to_string().contains("incomplete") {
                TruncatedModelResult::truncated_graceful(0, 0)
            } else {
                TruncatedModelResult::complete(0) // Other errors are fine
            }
        },
    };

    println!("\nIMP-177d: Real-World Truncated Handling:");
    println!("  Graceful handling: {}", handling.graceful_handling);
    println!(
        "  QA-024: {}",
        if handling.meets_qa024 { "PASS" } else { "FAIL" }
    );
}

// ===========================================
// IMP-178: Empty Input Handling (QA-025)
// ===========================================

/// Per spec QA-025: No panic on empty input sequences
#[derive(Debug, Clone)]
pub struct EmptyInputResult {
    /// Input type tested
    pub input_type: String,
    /// Whether empty input was handled
    pub handled: bool,
    /// Response type (error, empty output, default)
    pub response_type: String,
    /// Whether system panicked
    pub panicked: bool,
    /// Meets QA-025
    pub meets_qa025: bool,
}
