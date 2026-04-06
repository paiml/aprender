use super::*;

// ========================================================================
// SERVE-LAM-001: Configuration Tests
// ========================================================================

#[test]
fn test_SERVE_LAM_001_default_config() {
    let config = LambdaConfig::default();
    assert_eq!(config.memory_mb, 3008);
    assert_eq!(config.timeout_secs, 60);
    assert_eq!(config.region, "us-east-1");
    assert_eq!(config.architecture, LambdaArchitecture::Arm64);
}

#[test]
fn test_SERVE_LAM_001_builder_pattern() {
    let config = LambdaConfig::new("my-inference")
        .with_model("pacha://llama3:8b-q4")
        .with_memory(4096)
        .with_timeout(120)
        .with_region("eu-west-1")
        .with_env("LOG_LEVEL", "debug")
        .with_provisioned_concurrency(5);

    assert_eq!(config.function_name, "my-inference");
    assert_eq!(config.model_uri, "pacha://llama3:8b-q4");
    assert_eq!(config.memory_mb, 4096);
    assert_eq!(config.timeout_secs, 120);
    assert_eq!(config.region, "eu-west-1");
    assert_eq!(config.environment.get("LOG_LEVEL"), Some(&"debug".to_string()));
    assert_eq!(config.provisioned_concurrency, 5);
}

#[test]
fn test_SERVE_LAM_001_memory_clamping() {
    let config = LambdaConfig::new("test").with_memory(50);
    assert_eq!(config.memory_mb, 128);

    let config = LambdaConfig::new("test").with_memory(50000);
    assert_eq!(config.memory_mb, 10240);
}

#[test]
fn test_SERVE_LAM_001_timeout_clamping() {
    let config = LambdaConfig::new("test").with_timeout(0);
    assert_eq!(config.timeout_secs, 1);

    let config = LambdaConfig::new("test").with_timeout(2000);
    assert_eq!(config.timeout_secs, 900);
}

#[test]
fn test_SERVE_LAM_001_validation() {
    let config = LambdaConfig::new("test").with_model("pacha://model:1.0");
    assert!(config.validate().is_ok());

    let config = LambdaConfig::default();
    assert!(matches!(config.validate(), Err(ConfigError::MissingField("function_name"))));

    let config = LambdaConfig::new("test");
    assert!(matches!(config.validate(), Err(ConfigError::MissingField("model_uri"))));
}

#[test]
fn test_SERVE_LAM_001_cost_estimation() {
    let config = LambdaConfig::new("test").with_model("pacha://model:1.0").with_memory(3008);

    let cost_1k = config.estimate_cost(1000, 500);
    let cost_1m = config.estimate_cost(1_000_000, 500);

    assert!(cost_1k > 0.0);
    assert!(cost_1m > cost_1k);
    assert!(cost_1m < 100.0); // Sanity check
}

#[test]
fn test_SERVE_LAM_001_provisioned_cost() {
    let config_no_prov =
        LambdaConfig::new("test").with_model("pacha://model:1.0").with_provisioned_concurrency(0);

    let config_with_prov =
        LambdaConfig::new("test").with_model("pacha://model:1.0").with_provisioned_concurrency(10);

    let cost_no_prov = config_no_prov.estimate_cost(1000, 500);
    let cost_with_prov = config_with_prov.estimate_cost(1000, 500);

    assert!(cost_with_prov > cost_no_prov);
}

// ========================================================================
// SERVE-LAM-002: Runtime Tests
// ========================================================================

#[test]
fn test_SERVE_LAM_002_runtime_identifiers() {
    assert_eq!(LambdaRuntime::Provided.identifier(), "provided.al2023");
    assert_eq!(LambdaRuntime::Python312.identifier(), "python3.12");
    assert_eq!(LambdaRuntime::Container.identifier(), "container");
}

#[test]
fn test_SERVE_LAM_002_architecture_identifiers() {
    assert_eq!(LambdaArchitecture::Arm64.identifier(), "arm64");
    assert_eq!(LambdaArchitecture::X86_64.identifier(), "x86_64");
}

// ========================================================================
// SERVE-LAM-003: Deployer Tests
// ========================================================================

#[test]
fn test_SERVE_LAM_003_deployer_creation() {
    let config = LambdaConfig::new("test").with_model("pacha://model:1.0");
    let deployer = LambdaDeployer::new(config);

    assert_eq!(deployer.status(), DeploymentStatus::NotDeployed);
    assert!(deployer.function_arn().is_none());
}

#[test]
fn test_SERVE_LAM_003_deployer_validation() {
    let config = LambdaConfig::new("test").with_model("pacha://model:1.0");
    let deployer = LambdaDeployer::new(config);
    assert!(deployer.validate().is_ok());

    let config = LambdaConfig::default();
    let deployer = LambdaDeployer::new(config);
    assert!(deployer.validate().is_err());
}

#[test]
fn test_SERVE_LAM_003_deployer_estimate() {
    let config = LambdaConfig::new("test").with_model("pacha://model:1.0").with_memory(3008);
    let deployer = LambdaDeployer::new(config);

    let estimate = deployer.estimate();
    assert!(estimate.package_size_mb > 0);
    assert!(estimate.estimated_cold_start_ms > 0);
    assert!(estimate.monthly_cost_1m_req > estimate.monthly_cost_1k_req);
}

#[test]
fn test_SERVE_LAM_003_deployer_iac() {
    let config = LambdaConfig::new("my-inference")
        .with_model("pacha://llama3:8b")
        .with_memory(4096)
        .with_timeout(120);
    let deployer = LambdaDeployer::new(config);

    let iac = deployer.generate_iac();
    assert!(iac.contains("my-inference"));
    assert!(iac.contains("4096"));
    assert!(iac.contains("120"));
    assert!(iac.contains("pacha://llama3:8b"));
}

#[test]
fn test_SERVE_LAM_003_deployer_status_tracking() {
    let config = LambdaConfig::new("test").with_model("pacha://model:1.0");
    let mut deployer = LambdaDeployer::new(config);

    assert_eq!(deployer.status(), DeploymentStatus::NotDeployed);

    deployer.set_status(DeploymentStatus::Packaging);
    assert_eq!(deployer.status(), DeploymentStatus::Packaging);

    deployer.set_function_arn("arn:aws:lambda:us-east-1:123:function:test".to_string());
    assert_eq!(deployer.status(), DeploymentStatus::Active);
    assert_eq!(deployer.function_arn(), Some("arn:aws:lambda:us-east-1:123:function:test"));
}

// ========================================================================
// SERVE-LAM-004: Client Tests
// ========================================================================

#[test]
fn test_SERVE_LAM_004_client_creation() {
    let client = LambdaClient::new("arn:aws:lambda:us-east-1:123:function:test", "us-east-1");

    assert_eq!(client.function_arn(), "arn:aws:lambda:us-east-1:123:function:test");
    assert_eq!(client.region(), "us-east-1");
    assert_eq!(client.timeout(), Duration::from_secs(60));
}

#[test]
fn test_SERVE_LAM_004_client_with_timeout() {
    let client = LambdaClient::new("test", "us-east-1").with_timeout(Duration::from_secs(120));

    assert_eq!(client.timeout(), Duration::from_secs(120));
}

// ========================================================================
// SERVE-LAM-005: Request/Response Tests
// ========================================================================

#[test]
fn test_SERVE_LAM_005_inference_request() {
    let request = InferenceRequest::new("Hello, world!").with_max_tokens(100).with_temperature(0.7);

    assert_eq!(request.input, "Hello, world!");
    assert_eq!(request.max_tokens, Some(100));
    assert!((request.temperature.unwrap() - 0.7).abs() < f32::EPSILON);
}

#[test]
fn test_SERVE_LAM_005_request_serialization() {
    let request = InferenceRequest::new("test").with_max_tokens(50).with_temperature(0.5);

    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("test"));
    assert!(json.contains("50"));

    let deserialized: InferenceRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.input, "test");
    assert_eq!(deserialized.max_tokens, Some(50));
}

// ========================================================================
// SERVE-LAM-006: VPC Config Tests
// ========================================================================

#[test]
fn test_SERVE_LAM_006_vpc_config() {
    let vpc = VpcConfig::new(
        vec!["subnet-123".to_string(), "subnet-456".to_string()],
        vec!["sg-789".to_string()],
    );

    assert_eq!(vpc.subnet_ids.len(), 2);
    assert_eq!(vpc.security_group_ids.len(), 1);
}

#[test]
fn test_SERVE_LAM_006_config_with_vpc() {
    let vpc = VpcConfig::new(vec!["subnet-123".to_string()], vec!["sg-789".to_string()]);
    let config = LambdaConfig::new("test").with_model("pacha://model:1.0").with_vpc(vpc);

    assert!(config.vpc_config.is_some());
}

#[test]
fn test_SERVE_LAM_006_iac_with_vpc() {
    let vpc = VpcConfig::new(vec!["subnet-123".to_string()], vec!["sg-456".to_string()]);
    let config = LambdaConfig::new("test").with_model("pacha://model:1.0").with_vpc(vpc);
    let deployer = LambdaDeployer::new(config);

    let iac = deployer.generate_iac();
    assert!(iac.contains("VpcConfig"));
    assert!(iac.contains("subnet-123"));
    assert!(iac.contains("sg-456"));
}

// ========================================================================
// SERVE-LAM-007: Error Tests
// ========================================================================

#[test]
fn test_SERVE_LAM_007_config_error_display() {
    let err = ConfigError::MissingField("model_uri");
    assert!(err.to_string().contains("model_uri"));

    let err = ConfigError::InvalidMemory(50);
    assert!(err.to_string().contains("50"));
}

#[test]
fn test_SERVE_LAM_007_deployment_error_display() {
    let err = DeploymentError::ModelNotFound("pacha://missing:1.0".to_string());
    assert!(err.to_string().contains("missing"));

    let err = DeploymentError::PackageTooLarge { size_mb: 500, max_mb: 250 };
    assert!(err.to_string().contains("500"));
}

// ========================================================================
// Additional coverage tests
// ========================================================================

#[test]
fn test_with_storage_clamping() {
    let config = LambdaConfig::new("test").with_storage(100);
    assert_eq!(config.ephemeral_storage_mb, 512); // min clamped

    let config = LambdaConfig::new("test").with_storage(50000);
    assert_eq!(config.ephemeral_storage_mb, 10240); // max clamped

    let config = LambdaConfig::new("test").with_storage(2048);
    assert_eq!(config.ephemeral_storage_mb, 2048); // within range
}

#[test]
fn test_with_architecture() {
    let config = LambdaConfig::new("test").with_architecture(LambdaArchitecture::X86_64);
    assert_eq!(config.architecture, LambdaArchitecture::X86_64);
}

#[test]
fn test_with_runtime() {
    let config = LambdaConfig::new("test").with_runtime(LambdaRuntime::Container);
    assert_eq!(config.runtime, LambdaRuntime::Container);

    let config = LambdaConfig::new("test").with_runtime(LambdaRuntime::Python312);
    assert_eq!(config.runtime, LambdaRuntime::Python312);
}

#[test]
fn test_deployer_config_accessor() {
    let config = LambdaConfig::new("my-function").with_model("pacha://model:1.0").with_memory(5120);
    let deployer = LambdaDeployer::new(config);

    assert_eq!(deployer.config().function_name, "my-function");
    assert_eq!(deployer.config().memory_mb, 5120);
}

#[test]
fn test_estimate_cold_start_python_runtime() {
    let config = LambdaConfig::new("test")
        .with_model("pacha://model:1.0")
        .with_runtime(LambdaRuntime::Python312)
        .with_memory(3008);
    let deployer = LambdaDeployer::new(config);
    let estimate = deployer.estimate();
    // Python has higher base cold start
    assert!(estimate.estimated_cold_start_ms >= 2000);
}

#[test]
fn test_estimate_cold_start_container_runtime() {
    let config = LambdaConfig::new("test")
        .with_model("pacha://model:1.0")
        .with_runtime(LambdaRuntime::Container)
        .with_memory(3008);
    let deployer = LambdaDeployer::new(config);
    let estimate = deployer.estimate();
    // Container has highest base cold start
    assert!(estimate.estimated_cold_start_ms >= 2500);
}

#[test]
fn test_estimate_cold_start_low_memory() {
    let config = LambdaConfig::new("test").with_model("pacha://model:1.0").with_memory(1024);
    let deployer = LambdaDeployer::new(config);
    let estimate = deployer.estimate();
    // Low memory = slower cold start
    assert!(estimate.estimated_cold_start_ms > 2000);
}

#[test]
fn test_deployment_status_default() {
    let status = DeploymentStatus::default();
    assert_eq!(status, DeploymentStatus::NotDeployed);
}

#[test]
fn test_lambda_runtime_default() {
    let runtime = LambdaRuntime::default();
    assert_eq!(runtime, LambdaRuntime::Provided);
}

#[test]
fn test_lambda_architecture_default() {
    let arch = LambdaArchitecture::default();
    assert_eq!(arch, LambdaArchitecture::Arm64);
}

#[test]
fn test_config_error_equality() {
    let err1 = ConfigError::MissingField("model_uri");
    let err2 = ConfigError::MissingField("model_uri");
    assert_eq!(err1, err2);

    let err3 = ConfigError::InvalidMemory(50);
    let err4 = ConfigError::InvalidMemory(50);
    assert_eq!(err3, err4);

    let err5 = ConfigError::InvalidTimeout(0);
    assert_ne!(err3, err5);
}

#[test]
fn test_config_error_timeout_display() {
    let err = ConfigError::InvalidTimeout(0);
    let display = err.to_string();
    assert!(display.contains("0"));
    assert!(display.contains("1-900"));
}

#[test]
fn test_deployment_error_aws_error() {
    let err = DeploymentError::AwsError("AccessDenied".to_string());
    let display = err.to_string();
    assert!(display.contains("AWS error"));
    assert!(display.contains("AccessDenied"));
}

#[test]
fn test_deployment_error_config() {
    let config_err = ConfigError::MissingField("function_name");
    let err = DeploymentError::Config(config_err);
    let display = err.to_string();
    assert!(display.contains("Configuration error"));
    assert!(display.contains("function_name"));
}

#[test]
fn test_inference_response_fields() {
    let response = InferenceResponse {
        output: "Hello!".to_string(),
        tokens_generated: 10,
        latency_ms: 150,
        cold_start: true,
    };
    assert_eq!(response.output, "Hello!");
    assert_eq!(response.tokens_generated, 10);
    assert_eq!(response.latency_ms, 150);
    assert!(response.cold_start);
}

#[test]
fn test_deployment_estimate_fields() {
    let estimate = DeploymentEstimate {
        package_size_mb: 500,
        estimated_cold_start_ms: 2000,
        monthly_cost_1k_req: 0.05,
        monthly_cost_100k_req: 5.0,
        monthly_cost_1m_req: 50.0,
    };
    assert_eq!(estimate.package_size_mb, 500);
    assert_eq!(estimate.estimated_cold_start_ms, 2000);
    assert!(estimate.monthly_cost_1m_req > estimate.monthly_cost_100k_req);
}

#[test]
fn test_inference_request_default_fields() {
    let request = InferenceRequest::new("prompt");
    assert_eq!(request.input, "prompt");
    assert!(request.max_tokens.is_none());
    assert!(request.temperature.is_none());
    assert!(request.parameters.is_empty());
}

#[test]
fn test_deployment_status_variants() {
    assert_eq!(DeploymentStatus::Packaging, DeploymentStatus::Packaging);
    assert_eq!(DeploymentStatus::Uploading, DeploymentStatus::Uploading);
    assert_eq!(DeploymentStatus::Deploying, DeploymentStatus::Deploying);
    assert_eq!(DeploymentStatus::Active, DeploymentStatus::Active);
    assert_eq!(DeploymentStatus::Failed, DeploymentStatus::Failed);
}

#[test]
fn test_validation_invalid_memory_low() {
    // Use raw struct construction to bypass clamping
    let config = LambdaConfig {
        function_name: "test".to_string(),
        model_uri: "pacha://model:1.0".to_string(),
        memory_mb: 50, // Below minimum
        ..Default::default()
    };
    assert!(matches!(config.validate(), Err(ConfigError::InvalidMemory(50))));
}

#[test]
fn test_validation_invalid_memory_high() {
    let config = LambdaConfig {
        function_name: "test".to_string(),
        model_uri: "pacha://model:1.0".to_string(),
        memory_mb: 99999,
        ..Default::default()
    };
    assert!(matches!(config.validate(), Err(ConfigError::InvalidMemory(99999))));
}

#[test]
fn test_validation_invalid_timeout_zero() {
    let config = LambdaConfig {
        function_name: "test".to_string(),
        model_uri: "pacha://model:1.0".to_string(),
        timeout_secs: 0,
        ..Default::default()
    };
    assert!(matches!(config.validate(), Err(ConfigError::InvalidTimeout(0))));
}

#[test]
fn test_validation_invalid_timeout_high() {
    let config = LambdaConfig {
        function_name: "test".to_string(),
        model_uri: "pacha://model:1.0".to_string(),
        timeout_secs: 1000,
        ..Default::default()
    };
    assert!(matches!(config.validate(), Err(ConfigError::InvalidTimeout(1000))));
}
