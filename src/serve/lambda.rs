//! AWS Lambda Inference Deployment
//!
//! Deploy and manage ML inference on AWS Lambda for serverless, pay-per-use inference.
//!
//! ## Features
//!
//! - Model packaging with Docker/OCI containers
//! - Cold start optimization with provisioned concurrency
//! - Automatic scaling with Lambda's built-in capabilities
//! - Integration with Pacha registry for model artifacts
//!
//! ## Toyota Way Principles
//!
//! - Muda Elimination: Pay only for actual inference compute
//! - Heijunka: Automatic scaling levels inference load
//! - Jidoka: Built-in error handling and retry logic

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// ============================================================================
// SERVE-LAM-001: Lambda Configuration
// ============================================================================

/// Lambda function configuration for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaConfig {
    /// Function name (unique identifier)
    pub function_name: String,
    /// AWS region
    pub region: String,
    /// Memory size in MB (128-10240)
    pub memory_mb: u32,
    /// Timeout in seconds (max 900 for Lambda)
    pub timeout_secs: u32,
    /// Runtime environment
    pub runtime: LambdaRuntime,
    /// Model reference (Pacha URI)
    pub model_uri: String,
    /// Environment variables
    pub environment: HashMap<String, String>,
    /// Provisioned concurrency (0 = on-demand only)
    pub provisioned_concurrency: u32,
    /// VPC configuration (for Private tier)
    pub vpc_config: Option<VpcConfig>,
    /// Ephemeral storage in MB (512-10240)
    pub ephemeral_storage_mb: u32,
    /// Architecture
    pub architecture: LambdaArchitecture,
}

impl Default for LambdaConfig {
    fn default() -> Self {
        Self {
            function_name: String::new(),
            region: "us-east-1".to_string(),
            memory_mb: 3008, // Good for inference
            timeout_secs: 60,
            runtime: LambdaRuntime::Provided,
            model_uri: String::new(),
            environment: HashMap::new(),
            provisioned_concurrency: 0,
            vpc_config: None,
            ephemeral_storage_mb: 512,
            architecture: LambdaArchitecture::Arm64, // Better price/perf
        }
    }
}

impl LambdaConfig {
    /// Create a new Lambda config with function name
    #[must_use]
    pub fn new(function_name: impl Into<String>) -> Self {
        Self { function_name: function_name.into(), ..Default::default() }
    }

    /// Set the model URI
    #[must_use]
    pub fn with_model(mut self, model_uri: impl Into<String>) -> Self {
        self.model_uri = model_uri.into();
        self
    }

    /// Set memory size
    #[must_use]
    pub fn with_memory(mut self, mb: u32) -> Self {
        self.memory_mb = mb.clamp(128, 10240);
        self
    }

    /// Set timeout
    #[must_use]
    pub fn with_timeout(mut self, secs: u32) -> Self {
        self.timeout_secs = secs.clamp(1, 900);
        self
    }

    /// Set region
    #[must_use]
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }

    /// Set runtime
    #[must_use]
    pub fn with_runtime(mut self, runtime: LambdaRuntime) -> Self {
        self.runtime = runtime;
        self
    }

    /// Add environment variable
    #[must_use]
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.environment.insert(key.into(), value.into());
        self
    }

    /// Set provisioned concurrency
    #[must_use]
    pub fn with_provisioned_concurrency(mut self, count: u32) -> Self {
        self.provisioned_concurrency = count;
        self
    }

    /// Set VPC configuration
    #[must_use]
    pub fn with_vpc(mut self, vpc: VpcConfig) -> Self {
        self.vpc_config = Some(vpc);
        self
    }

    /// Set ephemeral storage
    #[must_use]
    pub fn with_storage(mut self, mb: u32) -> Self {
        self.ephemeral_storage_mb = mb.clamp(512, 10240);
        self
    }

    /// Set architecture
    #[must_use]
    pub fn with_architecture(mut self, arch: LambdaArchitecture) -> Self {
        self.architecture = arch;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.function_name.is_empty() {
            return Err(ConfigError::MissingField("function_name"));
        }
        if self.model_uri.is_empty() {
            return Err(ConfigError::MissingField("model_uri"));
        }
        if self.memory_mb < 128 || self.memory_mb > 10240 {
            return Err(ConfigError::InvalidMemory(self.memory_mb));
        }
        if self.timeout_secs == 0 || self.timeout_secs > 900 {
            return Err(ConfigError::InvalidTimeout(self.timeout_secs));
        }
        Ok(())
    }

    /// Estimate monthly cost for given invocations
    #[must_use]
    pub fn estimate_cost(&self, invocations_per_month: u64, avg_duration_ms: u64) -> f64 {
        // Lambda pricing (approximate, us-east-1, ARM):
        // $0.0000133334 per GB-second
        // $0.20 per 1M requests
        // Provisioned: $0.0000041667 per GB-second provisioned

        let gb_seconds = (self.memory_mb as f64 / 1024.0)
            * (avg_duration_ms as f64 / 1000.0)
            * invocations_per_month as f64;

        let compute_cost = gb_seconds * 0.0000133334;
        let request_cost = (invocations_per_month as f64 / 1_000_000.0) * 0.20;

        // Add provisioned concurrency cost (per hour)
        let provisioned_cost = if self.provisioned_concurrency > 0 {
            let gb_provisioned =
                (self.memory_mb as f64 / 1024.0) * self.provisioned_concurrency as f64;
            gb_provisioned * 0.0000041667 * 3600.0 * 24.0 * 30.0 // Monthly
        } else {
            0.0
        };

        compute_cost + request_cost + provisioned_cost
    }
}

/// Lambda runtime environment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LambdaRuntime {
    /// Custom runtime (provided.al2023)
    #[default]
    Provided,
    /// Python 3.12
    Python312,
    /// Container image (Docker)
    Container,
}

impl LambdaRuntime {
    /// Get the AWS runtime identifier
    #[must_use]
    pub const fn identifier(&self) -> &'static str {
        match self {
            Self::Provided => "provided.al2023",
            Self::Python312 => "python3.12",
            Self::Container => "container",
        }
    }
}

/// Lambda architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LambdaArchitecture {
    /// ARM64 (Graviton2) - better price/performance
    #[default]
    Arm64,
    /// x86_64
    X86_64,
}

impl LambdaArchitecture {
    /// Get the AWS architecture identifier
    #[must_use]
    pub const fn identifier(&self) -> &'static str {
        match self {
            Self::Arm64 => "arm64",
            Self::X86_64 => "x86_64",
        }
    }
}

/// VPC configuration for Lambda
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpcConfig {
    /// Subnet IDs
    pub subnet_ids: Vec<String>,
    /// Security group IDs
    pub security_group_ids: Vec<String>,
}

impl VpcConfig {
    /// Create new VPC config
    #[must_use]
    pub fn new(subnet_ids: Vec<String>, security_group_ids: Vec<String>) -> Self {
        Self { subnet_ids, security_group_ids }
    }
}

// ============================================================================
// SERVE-LAM-002: Lambda Deployer
// ============================================================================

/// Lambda deployment manager
#[derive(Debug, Clone)]
pub struct LambdaDeployer {
    /// Deployment configuration
    config: LambdaConfig,
    /// Deployment status
    status: DeploymentStatus,
    /// Function ARN (after deployment)
    function_arn: Option<String>,
}

/// Deployment status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DeploymentStatus {
    /// Not yet deployed
    #[default]
    NotDeployed,
    /// Packaging model
    Packaging,
    /// Uploading to S3
    Uploading,
    /// Creating/updating function
    Deploying,
    /// Active and ready
    Active,
    /// Deployment failed
    Failed,
}

impl LambdaDeployer {
    /// Create a new deployer
    #[must_use]
    pub fn new(config: LambdaConfig) -> Self {
        Self { config, status: DeploymentStatus::NotDeployed, function_arn: None }
    }

    /// Get current deployment status
    #[must_use]
    pub fn status(&self) -> DeploymentStatus {
        self.status
    }

    /// Get function ARN (if deployed)
    #[must_use]
    pub fn function_arn(&self) -> Option<&str> {
        self.function_arn.as_deref()
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &LambdaConfig {
        &self.config
    }

    /// Validate deployment prerequisites
    pub fn validate(&self) -> Result<(), DeploymentError> {
        self.config.validate().map_err(DeploymentError::Config)?;
        Ok(())
    }

    /// Estimate deployment (dry run)
    #[must_use]
    pub fn estimate(&self) -> DeploymentEstimate {
        let model_size_mb = 1024; // Placeholder - would be fetched from registry
        let package_size_mb = model_size_mb + 50; // Model + runtime

        DeploymentEstimate {
            package_size_mb,
            estimated_cold_start_ms: estimate_cold_start(&self.config),
            monthly_cost_1k_req: self.config.estimate_cost(1000, 500),
            monthly_cost_100k_req: self.config.estimate_cost(100_000, 500),
            monthly_cost_1m_req: self.config.estimate_cost(1_000_000, 500),
        }
    }

    /// Generate infrastructure-as-code (CloudFormation/SAM template)
    #[must_use]
    pub fn generate_iac(&self) -> String {
        let vpc_config = if let Some(ref vpc) = self.config.vpc_config {
            format!(
                r"
      VpcConfig:
        SubnetIds:
          {}
        SecurityGroupIds:
          {}",
                vpc.subnet_ids
                    .iter()
                    .map(|s| format!("- {s}"))
                    .collect::<Vec<_>>()
                    .join("\n          "),
                vpc.security_group_ids
                    .iter()
                    .map(|s| format!("- {s}"))
                    .collect::<Vec<_>>()
                    .join("\n          ")
            )
        } else {
            String::new()
        };

        let _provisioned = if self.config.provisioned_concurrency > 0 {
            format!(
                r"
  {}Concurrency:
    Type: AWS::Lambda::Version
    Properties:
      FunctionName: !Ref {}Function
      ProvisionedConcurrencyConfig:
        ProvisionedConcurrentExecutions: {}",
                self.config.function_name,
                self.config.function_name,
                self.config.provisioned_concurrency
            )
        } else {
            String::new()
        };

        format!(
            r"AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: ML Inference Lambda - {}

Resources:
  {}Function:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: {}
      Runtime: {}
      Handler: bootstrap
      CodeUri: ./deployment-package.zip
      MemorySize: {}
      Timeout: {}
      Architectures:
        - {}
      Environment:
        Variables:
          MODEL_URI: {}
          RUST_LOG: info{}{}
      EphemeralStorage:
        Size: {}

Outputs:
  FunctionArn:
    Description: Lambda Function ARN
    Value: !GetAtt {}Function.Arn
  FunctionUrl:
    Description: Lambda Function URL
    Value: !GetAtt {}FunctionUrl.FunctionUrl",
            self.config.function_name,
            self.config.function_name,
            self.config.function_name,
            self.config.runtime.identifier(),
            self.config.memory_mb,
            self.config.timeout_secs,
            self.config.architecture.identifier(),
            self.config.model_uri,
            self.config
                .environment
                .iter()
                .map(|(k, v)| format!("\n          {k}: {v}"))
                .collect::<String>(),
            vpc_config,
            self.config.ephemeral_storage_mb,
            self.config.function_name,
            self.config.function_name,
        )
    }

    /// Set deployment status (for tracking)
    pub fn set_status(&mut self, status: DeploymentStatus) {
        self.status = status;
    }

    /// Set function ARN after successful deployment
    pub fn set_function_arn(&mut self, arn: String) {
        self.function_arn = Some(arn);
        self.status = DeploymentStatus::Active;
    }
}

/// Deployment estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentEstimate {
    /// Package size in MB
    pub package_size_mb: u64,
    /// Estimated cold start in ms
    pub estimated_cold_start_ms: u64,
    /// Monthly cost for 1K requests
    pub monthly_cost_1k_req: f64,
    /// Monthly cost for 100K requests
    pub monthly_cost_100k_req: f64,
    /// Monthly cost for 1M requests
    pub monthly_cost_1m_req: f64,
}

/// Estimate cold start time based on config
fn estimate_cold_start(config: &LambdaConfig) -> u64 {
    // Base cold start for provided runtime
    let base_ms: u64 = match config.runtime {
        LambdaRuntime::Provided => 100,
        LambdaRuntime::Python312 => 200,
        LambdaRuntime::Container => 500,
    };

    // Memory affects cold start (more memory = faster init)
    let memory_factor =
        if config.memory_mb >= 3008 { 1.0 } else { 1.5 - (config.memory_mb as f64 / 6016.0) };

    // Model loading estimate (rough)
    let model_load_ms: u64 = 2000; // 2 seconds for model loading

    ((base_ms as f64 * memory_factor) as u64) + model_load_ms
}

// ============================================================================
// SERVE-LAM-003: Inference Client
// ============================================================================

/// Lambda inference request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Input text/prompt
    pub input: String,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Temperature (0.0-2.0)
    pub temperature: Option<f32>,
    /// Additional parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

impl InferenceRequest {
    /// Create a new inference request
    #[must_use]
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            max_tokens: None,
            temperature: None,
            parameters: HashMap::new(),
        }
    }

    /// Set max tokens
    #[must_use]
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set temperature
    #[must_use]
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }
}

/// Lambda inference response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Generated output
    pub output: String,
    /// Number of tokens generated
    pub tokens_generated: u32,
    /// Inference latency in ms
    pub latency_ms: u64,
    /// Whether this was a cold start
    pub cold_start: bool,
}

/// Lambda inference client
#[derive(Debug, Clone)]
pub struct LambdaClient {
    /// Function ARN or name
    function_arn: String,
    /// AWS region
    region: String,
    /// Invocation timeout
    timeout: Duration,
}

impl LambdaClient {
    /// Create a new Lambda client
    #[must_use]
    pub fn new(function_arn: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            function_arn: function_arn.into(),
            region: region.into(),
            timeout: Duration::from_secs(60),
        }
    }

    /// Set invocation timeout
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Get function ARN
    #[must_use]
    pub fn function_arn(&self) -> &str {
        &self.function_arn
    }

    /// Get region
    #[must_use]
    pub fn region(&self) -> &str {
        &self.region
    }

    /// Get timeout
    #[must_use]
    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

// ============================================================================
// SERVE-LAM-004: Error Types
// ============================================================================

/// Configuration error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// Missing required field
    MissingField(&'static str),
    /// Invalid memory size
    InvalidMemory(u32),
    /// Invalid timeout
    InvalidTimeout(u32),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField(field) => write!(f, "Missing required field: {field}"),
            Self::InvalidMemory(mb) => write!(f, "Invalid memory size: {mb}MB (must be 128-10240)"),
            Self::InvalidTimeout(secs) => write!(f, "Invalid timeout: {secs}s (must be 1-900)"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Deployment error
#[derive(Debug)]
pub enum DeploymentError {
    /// Configuration error
    Config(ConfigError),
    /// AWS API error
    AwsError(String),
    /// Model not found
    ModelNotFound(String),
    /// Package too large
    PackageTooLarge { size_mb: u64, max_mb: u64 },
}

impl std::fmt::Display for DeploymentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(e) => write!(f, "Configuration error: {e}"),
            Self::AwsError(e) => write!(f, "AWS error: {e}"),
            Self::ModelNotFound(uri) => write!(f, "Model not found: {uri}"),
            Self::PackageTooLarge { size_mb, max_mb } => {
                write!(f, "Package too large: {size_mb}MB (max {max_mb}MB)")
            }
        }
    }
}

impl std::error::Error for DeploymentError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
#[path = "lambda_tests.rs"]
mod tests;
