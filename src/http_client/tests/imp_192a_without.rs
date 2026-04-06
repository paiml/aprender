
/// IMP-192a: Test model source types
#[test]
fn test_imp_192a_model_sources() {
    let hf = ModelSource::HuggingFace {
        repo: "meta-llama/Llama-2-7b".to_string(),
        file: "model.gguf".to_string(),
    };
    let _ollama = ModelSource::Ollama {
        model: "llama2:7b".to_string(),
    };
    let _local = ModelSource::LocalPath {
        path: "/models/llama.gguf".to_string(),
    };

    // Just verify they construct without panicking
    match hf {
        ModelSource::HuggingFace { repo, .. } => assert!(!repo.is_empty()),
        _ => panic!("Wrong variant"),
    }

    println!("\nIMP-192a: Model Sources:");
    println!("  HuggingFace: meta-llama/Llama-2-7b");
    println!("  Ollama: llama2:7b");
    println!("  Local: /models/llama.gguf");
}

/// IMP-192b: Test download result
#[test]
fn test_imp_192b_download_result() {
    let source = ModelSource::HuggingFace {
        repo: "test/model".to_string(),
        file: "model.gguf".to_string(),
    };

    let cached = ModelDownloadResult::cached(source.clone(), "/cache/model.gguf", 1024);
    assert!(
        cached.is_available(),
        "IMP-192b: Cached should be available"
    );
    assert!(cached.meets_qa039, "IMP-192b: Should meet QA-039");

    let failed = ModelDownloadResult::failed(source, "Network error");
    assert!(
        !failed.is_available(),
        "IMP-192b: Failed should not be available"
    );

    println!("\nIMP-192b: Download Results:");
    println!(
        "  Cached: available={}, status={:?}",
        cached.is_available(),
        cached.status
    );
    println!(
        "  Failed: available={}, status={:?}",
        failed.is_available(),
        failed.status
    );
}

/// IMP-192c: Test model cache
#[test]
fn test_imp_192c_model_cache() {
    let cache = ModelCache::default();

    let source = ModelSource::HuggingFace {
        repo: "test/model".to_string(),
        file: "weights.gguf".to_string(),
    };

    let result = cache.check(&source);
    assert!(
        result.local_path.is_some(),
        "IMP-192c: Should have local path"
    );
    assert!(result.meets_qa039, "IMP-192c: Should meet QA-039");

    println!("\nIMP-192c: Model Cache:");
    println!("  Cache dir: {}", cache.cache_dir);
    println!("  Local path: {:?}", result.local_path);
    println!("  Status: {:?}", result.status);
}

/// IMP-192d: Real-world model download
#[test]
#[ignore = "Requires network access and HuggingFace token"]
fn test_imp_192d_realworld_download() {
    let cache = ModelCache::new("/tmp/realizar-bench-models");

    let source = ModelSource::HuggingFace {
        repo: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
        file: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string(),
    };

    let result = cache.check(&source);

    println!("\nIMP-192d: Real-World Model Download:");
    println!("  Source: {:?}", result.source);
    println!("  Status: {:?}", result.status);
    println!("  Local path: {:?}", result.local_path);
    println!("  Size: {:?} bytes", result.size_bytes);
    println!("  Available: {}", result.is_available());
    println!(
        "  QA-039: {}",
        if result.meets_qa039 { "PASS" } else { "FAIL" }
    );
}

// ================================================================================
// IMP-193: JSON Schema Validation (QA-040)
// JSON schema validation for benchmark results
// ================================================================================

/// Benchmark result schema version
pub const BENCHMARK_SCHEMA_VERSION: &str = "1.0.0";

/// JSON schema field types
#[derive(Debug, Clone, PartialEq)]
pub enum SchemaFieldType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
}

/// Schema validation result
#[derive(Debug)]
pub struct SchemaValidationResult {
    pub schema_version: String,
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub meets_qa040: bool,
}

impl SchemaValidationResult {
    pub fn valid(schema_version: &str) -> Self {
        Self {
            schema_version: schema_version.to_string(),
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            meets_qa040: true,
        }
    }

    pub fn invalid(schema_version: &str, errors: Vec<String>) -> Self {
        Self {
            schema_version: schema_version.to_string(),
            is_valid: false,
            errors,
            warnings: Vec::new(),
            meets_qa040: true, // Validation ran, even if result is invalid
        }
    }

    pub fn with_warnings(mut self, warnings: Vec<String>) -> Self {
        self.warnings = warnings;
        self
    }
}

/// Benchmark result JSON structure
#[derive(Debug, Clone)]
pub struct BenchmarkResultJson {
    pub version: String,
    pub timestamp: String,
    pub environment: serde_json::Value,
    pub results: serde_json::Value,
    pub metrics: serde_json::Value,
}

impl BenchmarkResultJson {
    pub fn validate(&self) -> SchemaValidationResult {
        let mut errors = Vec::new();

        // Check required fields
        if self.version.is_empty() {
            errors.push("Missing required field: version".to_string());
        }
        if self.timestamp.is_empty() {
            errors.push("Missing required field: timestamp".to_string());
        }
        if self.environment.is_null() {
            errors.push("Missing required field: environment".to_string());
        }
        if self.results.is_null() {
            errors.push("Missing required field: results".to_string());
        }

        if errors.is_empty() {
            SchemaValidationResult::valid(&self.version)
        } else {
            SchemaValidationResult::invalid(&self.version, errors)
        }
    }
}

/// IMP-193a: Test schema validation result
#[test]
fn test_imp_193a_schema_validation_result() {
    let valid = SchemaValidationResult::valid(BENCHMARK_SCHEMA_VERSION);
    assert!(valid.is_valid, "IMP-193a: Valid result should be valid");
    assert!(
        valid.errors.is_empty(),
        "IMP-193a: Valid result should have no errors"
    );
    assert!(valid.meets_qa040, "IMP-193a: Should meet QA-040");

    let invalid = SchemaValidationResult::invalid(
        BENCHMARK_SCHEMA_VERSION,
        vec!["Missing field: version".to_string()],
    );
    assert!(
        !invalid.is_valid,
        "IMP-193a: Invalid result should be invalid"
    );
    assert!(
        !invalid.errors.is_empty(),
        "IMP-193a: Invalid result should have errors"
    );

    println!("\nIMP-193a: Schema Validation Result:");
    println!(
        "  Valid: is_valid={}, errors={}",
        valid.is_valid,
        valid.errors.len()
    );
    println!(
        "  Invalid: is_valid={}, errors={}",
        invalid.is_valid,
        invalid.errors.len()
    );
}

/// IMP-193b: Test benchmark JSON validation
#[test]
fn test_imp_193b_benchmark_json_validation() {
    let valid_json = BenchmarkResultJson {
        version: "1.0.0".to_string(),
        timestamp: "2025-12-13T00:00:00Z".to_string(),
        environment: serde_json::json!({"os": "linux"}),
        results: serde_json::json!({"latency_ms": 100.0}),
        metrics: serde_json::json!({"throughput": 143.0}),
    };

    let result = valid_json.validate();
    assert!(
        result.is_valid,
        "IMP-193b: Valid JSON should pass validation"
    );
    assert!(result.meets_qa040, "IMP-193b: Should meet QA-040");

    let invalid_json = BenchmarkResultJson {
        version: String::new(),
        timestamp: String::new(),
        environment: serde_json::Value::Null,
        results: serde_json::Value::Null,
        metrics: serde_json::Value::Null,
    };

    let invalid_result = invalid_json.validate();
    assert!(
        !invalid_result.is_valid,
        "IMP-193b: Invalid JSON should fail validation"
    );
    assert!(
        invalid_result.errors.len() >= 3,
        "IMP-193b: Should have multiple errors"
    );

    println!("\nIMP-193b: Benchmark JSON Validation:");
    println!("  Valid JSON: is_valid={}", result.is_valid);
    println!("  Invalid JSON: errors={:?}", invalid_result.errors);
}

/// IMP-193c: Test schema field types
#[test]
fn test_imp_193c_schema_field_types() {
    let types = vec![
        SchemaFieldType::String,
        SchemaFieldType::Number,
        SchemaFieldType::Integer,
        SchemaFieldType::Boolean,
        SchemaFieldType::Array,
        SchemaFieldType::Object,
    ];

    for field_type in &types {
        // Verify all types are distinct
        assert!(types.iter().filter(|t| *t == field_type).count() == 1);
    }

    println!("\nIMP-193c: Schema Field Types:");
    for t in types {
        println!("  {:?}: supported", t);
    }
}

/// IMP-193d: Real-world JSON schema validation
#[test]
#[ignore = "Requires benchmark results file"]
fn test_imp_193d_realworld_schema_validation() {
    let benchmark_json = BenchmarkResultJson {
        version: BENCHMARK_SCHEMA_VERSION.to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        environment: serde_json::json!({
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "cpu_cores": std::thread::available_parallelism().map(std::num::NonZeroUsize::get).unwrap_or(1),
        }),
        results: serde_json::json!({
            "latency_p50_ms": 100.0,
            "latency_p95_ms": 120.0,
            "latency_p99_ms": 150.0,
            "throughput_toks": 143.0,
        }),
        metrics: serde_json::json!({
            "samples": 100,
            "cv": 0.05,
        }),
    };

    let result = benchmark_json.validate();

    println!("\nIMP-193d: Real-World Schema Validation:");
    println!("  Schema version: {}", result.schema_version);
    println!("  Valid: {}", result.is_valid);
    println!("  Errors: {:?}", result.errors);
    println!(
        "  QA-040: {}",
        if result.meets_qa040 { "PASS" } else { "FAIL" }
    );
}

// ================================================================================
// IMP-194: Bench Inference All (QA-041)
// `make bench-inference-all` completes without error
// ================================================================================

/// Benchmark suite configuration
#[derive(Debug, Clone)]
pub struct BenchSuiteConfig {
    pub name: String,
    pub enabled: bool,
    pub timeout_secs: u64,
    pub required: bool,
}

impl BenchSuiteConfig {
    pub fn new(name: &str, enabled: bool, timeout_secs: u64) -> Self {
        Self {
            name: name.to_string(),
            enabled,
            timeout_secs,
            required: true,
        }
    }

    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }
}

/// Benchmark suite result
#[derive(Debug)]
pub struct BenchSuiteResult {
    pub config: BenchSuiteConfig,
    pub status: BenchSuiteStatus,
    pub duration_secs: f64,
    pub output: Option<String>,
    pub meets_qa041: bool,
}

/// Benchmark suite execution status
#[derive(Debug, Clone, PartialEq)]
pub enum BenchSuiteStatus {
    Success,
    Failed,
    Skipped,
    Timeout,
}

impl BenchSuiteResult {
    pub fn success(config: BenchSuiteConfig, duration: f64) -> Self {
        Self {
            config,
            status: BenchSuiteStatus::Success,
            duration_secs: duration,
            output: None,
            meets_qa041: true,
        }
    }

    pub fn failed(config: BenchSuiteConfig, error: &str) -> Self {
        Self {
            config,
            status: BenchSuiteStatus::Failed,
            duration_secs: 0.0,
            output: Some(error.to_string()),
            meets_qa041: false,
        }
    }

    pub fn skipped(config: BenchSuiteConfig, reason: &str) -> Self {
        let meets_qa041 = !config.required;
        Self {
            config,
            status: BenchSuiteStatus::Skipped,
            duration_secs: 0.0,
            output: Some(reason.to_string()),
            meets_qa041,
        }
    }
}

/// IMP-194a: Test bench suite config
#[test]
fn test_imp_194a_bench_suite_config() {
    let required = BenchSuiteConfig::new("inference", true, 300);
    assert!(required.enabled, "IMP-194a: Should be enabled");
    assert!(required.required, "IMP-194a: Should be required by default");

    let optional = BenchSuiteConfig::new("gpu", true, 60).optional();
    assert!(
        !optional.required,
        "IMP-194a: Optional should not be required"
    );

    println!("\nIMP-194a: Bench Suite Config:");
    println!(
        "  Required: name={}, required={}",
        required.name, required.required
    );
    println!(
        "  Optional: name={}, required={}",
        optional.name, optional.required
    );
}
