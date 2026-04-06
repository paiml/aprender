//! Serverless Function Execution
//!
//! Lambda-compatible serverless functions with sub-100ms cold start.
//! Built on pepita `MicroVM` for hardware-level isolation.
//!
//! ## Example
//!
//! ```rust,ignore
//! use repartir::serverless::{Function, Runtime, Trigger, FunctionService};
//!
//! // Define a function
//! let func = Function::builder()
//!     .name("process-data")
//!     .runtime(Runtime::RustNative { binary: "./processor".into() })
//!     .memory_mib(256)
//!     .timeout(Duration::from_secs(30))
//!     .build()?;
//!
//! // Create service and register
//! let service = FunctionService::new()?;
//! service.register(func)?;
//!
//! // Invoke
//! let result = service.invoke("process-data", b"{\"key\": \"value\"}").await?;
//! ```

use crate::error::{RepartirError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use uuid::Uuid;

// ============================================================================
// RUNTIME
// ============================================================================

/// Function runtime environment
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Runtime {
    /// Native Rust binary (fastest cold start)
    RustNative {
        /// Path to compiled binary
        binary: PathBuf,
    },
    /// OCI container image
    Container {
        /// Image reference (e.g., "myregistry/myimage:tag")
        image: String,
    },
    /// WebAssembly module
    Wasm {
        /// Path to .wasm file
        module: PathBuf,
    },
}

impl Default for Runtime {
    fn default() -> Self {
        Self::RustNative { binary: PathBuf::new() }
    }
}

// ============================================================================
// TRIGGER
// ============================================================================

/// Event trigger for function invocation
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum Trigger {
    /// HTTP/REST trigger
    Http {
        /// URL path pattern
        path: String,
        /// Allowed HTTP methods
        methods: Vec<HttpMethod>,
    },
    /// Scheduled trigger (cron)
    Schedule {
        /// Cron expression
        cron: String,
    },
    /// Message queue trigger
    Queue {
        /// Queue name
        queue_name: String,
    },
    /// File system event trigger
    FileSystem {
        /// Watch path
        path: PathBuf,
        /// Event types to watch
        events: Vec<FsEvent>,
    },
    /// Manual invocation only
    #[default]
    Manual,
}

/// HTTP methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum HttpMethod {
    /// GET request
    Get,
    /// POST request
    Post,
    /// PUT request
    Put,
    /// DELETE request
    Delete,
    /// PATCH request
    Patch,
}

/// File system events
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FsEvent {
    /// File created
    Create,
    /// File modified
    Modify,
    /// File deleted
    Delete,
    /// File renamed
    Rename,
}

// ============================================================================
// FUNCTION
// ============================================================================

/// Serverless function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    /// Unique function ID
    pub id: Uuid,
    /// Human-readable name
    pub name: String,
    /// Execution runtime
    pub runtime: Runtime,
    /// Memory allocation in MiB
    pub memory_mib: u32,
    /// Execution timeout
    pub timeout: Duration,
    /// Environment variables
    pub env: HashMap<String, String>,
    /// Event triggers
    pub triggers: Vec<Trigger>,
    /// Handler entry point (for Wasm/Container)
    pub handler: Option<String>,
}

impl Function {
    /// Create a new function builder
    #[must_use]
    pub fn builder() -> FunctionBuilder {
        FunctionBuilder::default()
    }

    /// Get function ID
    #[must_use]
    pub const fn id(&self) -> Uuid {
        self.id
    }

    /// Get function name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get memory allocation
    #[must_use]
    pub const fn memory_mib(&self) -> u32 {
        self.memory_mib
    }

    /// Get timeout
    #[must_use]
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Builder for Function
#[derive(Debug, Default)]
pub struct FunctionBuilder {
    name: Option<String>,
    runtime: Option<Runtime>,
    memory_mib: Option<u32>,
    timeout: Option<Duration>,
    env: HashMap<String, String>,
    triggers: Vec<Trigger>,
    handler: Option<String>,
}

impl FunctionBuilder {
    /// Set function name
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set runtime
    #[must_use]
    pub fn runtime(mut self, runtime: Runtime) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// Set memory allocation in MiB
    #[must_use]
    pub const fn memory_mib(mut self, mib: u32) -> Self {
        self.memory_mib = Some(mib);
        self
    }

    /// Set execution timeout
    #[must_use]
    pub const fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Add environment variable
    #[must_use]
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Add trigger
    #[must_use]
    pub fn trigger(mut self, trigger: Trigger) -> Self {
        self.triggers.push(trigger);
        self
    }

    /// Set handler entry point
    #[must_use]
    pub fn handler(mut self, handler: impl Into<String>) -> Self {
        self.handler = Some(handler.into());
        self
    }

    /// Build the function
    ///
    /// # Errors
    ///
    /// Returns error if name or runtime is not set
    pub fn build(self) -> Result<Function> {
        let name = self.name.ok_or_else(|| RepartirError::InvalidTask {
            reason: "Function name is required".to_string(),
        })?;

        let runtime = self.runtime.ok_or_else(|| RepartirError::InvalidTask {
            reason: "Function runtime is required".to_string(),
        })?;

        Ok(Function {
            id: Uuid::new_v4(),
            name,
            runtime,
            memory_mib: self.memory_mib.unwrap_or(128),
            timeout: self.timeout.unwrap_or(Duration::from_secs(30)),
            env: self.env,
            triggers: self.triggers,
            handler: self.handler,
        })
    }
}

// ============================================================================
// INVOCATION
// ============================================================================

/// Function invocation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvocationRequest {
    /// Request ID
    pub request_id: Uuid,
    /// Function name or ID
    pub function: String,
    /// Input payload
    pub payload: Vec<u8>,
    /// Invocation type
    pub invocation_type: InvocationType,
}

impl InvocationRequest {
    /// Create a new invocation request
    #[must_use]
    pub fn new(function: impl Into<String>, payload: impl Into<Vec<u8>>) -> Self {
        Self {
            request_id: Uuid::new_v4(),
            function: function.into(),
            payload: payload.into(),
            invocation_type: InvocationType::RequestResponse,
        }
    }

    /// Set as async invocation
    #[must_use]
    pub const fn async_invoke(mut self) -> Self {
        self.invocation_type = InvocationType::Event;
        self
    }
}

/// Invocation type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum InvocationType {
    /// Synchronous request-response
    #[default]
    RequestResponse,
    /// Asynchronous event (fire and forget)
    Event,
    /// Dry run (validate only)
    DryRun,
}

/// Function invocation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvocationResponse {
    /// Request ID (matches request)
    pub request_id: Uuid,
    /// Status code (0 = success)
    pub status_code: i32,
    /// Output payload
    pub payload: Vec<u8>,
    /// Execution duration
    pub duration: Duration,
    /// Error message (if any)
    pub error: Option<String>,
    /// Billed duration in ms
    pub billed_duration_ms: u64,
    /// Memory used in MiB
    pub memory_used_mib: u32,
}

impl InvocationResponse {
    /// Create success response
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn success(request_id: Uuid, payload: Vec<u8>, duration: Duration) -> Self {
        Self {
            request_id,
            status_code: 0,
            payload,
            duration,
            error: None,
            billed_duration_ms: duration.as_millis() as u64,
            memory_used_mib: 0,
        }
    }

    /// Create error response
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn error(request_id: Uuid, error: impl Into<String>, duration: Duration) -> Self {
        Self {
            request_id,
            status_code: 1,
            payload: Vec::new(),
            duration,
            error: Some(error.into()),
            billed_duration_ms: duration.as_millis() as u64,
            memory_used_mib: 0,
        }
    }

    /// Check if invocation succeeded
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.status_code == 0
    }

    /// Get payload as string
    ///
    /// # Errors
    ///
    /// Returns error if payload is not valid UTF-8
    pub fn payload_str(&self) -> Result<&str> {
        std::str::from_utf8(&self.payload).map_err(|e| RepartirError::InvalidTask {
            reason: format!("Payload is not valid UTF-8: {e}"),
        })
    }
}

// ============================================================================
// FUNCTION SERVICE
// ============================================================================

/// Serverless function service
#[derive(Debug)]
pub struct FunctionService {
    /// Registered functions
    functions: HashMap<String, Function>,
    /// Function instances (warm pool)
    instances: HashMap<Uuid, FunctionInstance>,
    /// Service configuration
    config: ServiceConfig,
}

/// Function instance (for warm pool)
#[derive(Debug)]
struct FunctionInstance {
    /// Instance ID
    #[allow(dead_code)]
    id: Uuid,
    /// Function ID
    #[allow(dead_code)]
    function_id: Uuid,
    /// Is instance busy
    busy: bool,
    /// Last used timestamp
    #[allow(dead_code)]
    last_used: std::time::Instant,
}

/// Service configuration
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    /// Maximum concurrent invocations
    pub max_concurrency: usize,
    /// Warm pool size per function
    pub warm_pool_size: usize,
    /// Instance idle timeout
    pub idle_timeout: Duration,
    /// Enable metrics
    pub metrics_enabled: bool,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 1000,
            warm_pool_size: 3,
            idle_timeout: Duration::from_secs(300),
            metrics_enabled: true,
        }
    }
}

impl FunctionService {
    /// Create a new function service
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(ServiceConfig::default())
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(config: ServiceConfig) -> Self {
        Self { functions: HashMap::new(), instances: HashMap::new(), config }
    }

    /// Register a function
    ///
    /// # Errors
    ///
    /// Returns error if function with same name already exists
    pub fn register(&mut self, function: Function) -> Result<()> {
        if self.functions.contains_key(&function.name) {
            return Err(RepartirError::InvalidTask {
                reason: format!("Function '{}' already registered", function.name),
            });
        }
        self.functions.insert(function.name.clone(), function);
        Ok(())
    }

    /// Unregister a function
    pub fn unregister(&mut self, name: &str) -> Option<Function> {
        self.functions.remove(name)
    }

    /// Get a function by name
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Function> {
        self.functions.get(name)
    }

    /// List all registered functions
    #[must_use]
    pub fn list(&self) -> Vec<&Function> {
        self.functions.values().collect()
    }

    /// Invoke a function (mock implementation)
    ///
    /// # Errors
    ///
    /// Returns error if function not found or execution fails
    #[allow(clippy::needless_pass_by_value)]
    pub fn invoke(&mut self, request: InvocationRequest) -> Result<InvocationResponse> {
        let start = std::time::Instant::now();

        // Find function
        let function =
            self.functions.get(&request.function).ok_or_else(|| RepartirError::InvalidTask {
                reason: format!("Function '{}' not found", request.function),
            })?;

        // Dry run just validates
        if request.invocation_type == InvocationType::DryRun {
            return Ok(InvocationResponse::success(
                request.request_id,
                Vec::new(),
                start.elapsed(),
            ));
        }

        // Mock execution: echo the input with function info
        let output = format!(
            "{{\"function\":\"{}\",\"input_size\":{},\"memory_mib\":{}}}",
            function.name,
            request.payload.len(),
            function.memory_mib
        );

        Ok(InvocationResponse::success(request.request_id, output.into_bytes(), start.elapsed()))
    }

    /// Get service configuration
    #[must_use]
    pub const fn config(&self) -> &ServiceConfig {
        &self.config
    }

    /// Get number of registered functions
    #[must_use]
    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    /// Get available instances count
    #[must_use]
    pub fn available_instances(&self) -> usize {
        self.instances.values().filter(|i| !i.busy).count()
    }
}

impl Default for FunctionService {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// COLD START METRICS
// ============================================================================

/// Cold start metrics
#[derive(Debug, Clone, Default)]
pub struct ColdStartMetrics {
    /// Total cold starts
    pub cold_starts: u64,
    /// Total warm starts
    pub warm_starts: u64,
    /// Average cold start latency in ms
    pub avg_cold_start_ms: f64,
    /// Average warm start latency in ms
    pub avg_warm_start_ms: f64,
    /// P99 cold start latency in ms
    pub p99_cold_start_ms: f64,
}

impl ColdStartMetrics {
    /// Calculate warm start ratio
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn warm_start_ratio(&self) -> f64 {
        let total = self.cold_starts + self.warm_starts;
        if total == 0 {
            return 0.0;
        }
        self.warm_starts as f64 / total as f64
    }
}

// ============================================================================
// TESTS (EXTREME TDD)
// ============================================================================

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::unchecked_time_subtraction,
    clippy::manual_range_contains,
    clippy::panic,
    clippy::redundant_clone
)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // Runtime Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_runtime_rust_native() {
        let runtime = Runtime::RustNative { binary: PathBuf::from("/usr/bin/myapp") };
        if let Runtime::RustNative { binary } = runtime {
            assert_eq!(binary, PathBuf::from("/usr/bin/myapp"));
        } else {
            panic!("Expected RustNative runtime");
        }
    }

    #[test]
    fn test_runtime_container() {
        let runtime = Runtime::Container { image: "myregistry/myimage:latest".to_string() };
        if let Runtime::Container { image } = runtime {
            assert_eq!(image, "myregistry/myimage:latest");
        } else {
            panic!("Expected Container runtime");
        }
    }

    #[test]
    fn test_runtime_wasm() {
        let runtime = Runtime::Wasm { module: PathBuf::from("./func.wasm") };
        if let Runtime::Wasm { module } = runtime {
            assert_eq!(module, PathBuf::from("./func.wasm"));
        } else {
            panic!("Expected Wasm runtime");
        }
    }

    #[test]
    fn test_runtime_default() {
        let runtime = Runtime::default();
        assert!(matches!(runtime, Runtime::RustNative { .. }));
    }

    #[test]
    fn test_runtime_serialize_deserialize() {
        let runtime = Runtime::Container { image: "test:latest".to_string() };
        let json = serde_json::to_string(&runtime).unwrap();
        let deserialized: Runtime = serde_json::from_str(&json).unwrap();
        assert_eq!(runtime, deserialized);
    }

    // ------------------------------------------------------------------------
    // Trigger Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_trigger_http() {
        let trigger = Trigger::Http {
            path: "/api/v1/process".to_string(),
            methods: vec![HttpMethod::Get, HttpMethod::Post],
        };
        if let Trigger::Http { path, methods } = trigger {
            assert_eq!(path, "/api/v1/process");
            assert_eq!(methods.len(), 2);
        } else {
            panic!("Expected Http trigger");
        }
    }

    #[test]
    fn test_trigger_schedule() {
        let trigger = Trigger::Schedule { cron: "0 * * * *".to_string() };
        if let Trigger::Schedule { cron } = trigger {
            assert_eq!(cron, "0 * * * *");
        } else {
            panic!("Expected Schedule trigger");
        }
    }

    #[test]
    fn test_trigger_queue() {
        let trigger = Trigger::Queue { queue_name: "my-queue".to_string() };
        if let Trigger::Queue { queue_name } = trigger {
            assert_eq!(queue_name, "my-queue");
        } else {
            panic!("Expected Queue trigger");
        }
    }

    #[test]
    fn test_trigger_filesystem() {
        let trigger = Trigger::FileSystem {
            path: PathBuf::from("/data/uploads"),
            events: vec![FsEvent::Create, FsEvent::Modify],
        };
        if let Trigger::FileSystem { path, events } = trigger {
            assert_eq!(path, PathBuf::from("/data/uploads"));
            assert_eq!(events.len(), 2);
        } else {
            panic!("Expected FileSystem trigger");
        }
    }

    #[test]
    fn test_trigger_default() {
        let trigger = Trigger::default();
        assert!(matches!(trigger, Trigger::Manual));
    }

    // ------------------------------------------------------------------------
    // Function Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_function_builder_minimal() {
        let func = Function::builder()
            .name("test-func")
            .runtime(Runtime::RustNative { binary: PathBuf::from("./test") })
            .build()
            .unwrap();

        assert_eq!(func.name(), "test-func");
        assert_eq!(func.memory_mib(), 128); // default
        assert_eq!(func.timeout(), Duration::from_secs(30)); // default
    }

    #[test]
    fn test_function_builder_full() {
        let func = Function::builder()
            .name("complex-func")
            .runtime(Runtime::Container { image: "myimage:v1".to_string() })
            .memory_mib(512)
            .timeout(Duration::from_secs(60))
            .env("API_KEY", "secret")
            .env("DEBUG", "true")
            .trigger(Trigger::Http {
                path: "/process".to_string(),
                methods: vec![HttpMethod::Post],
            })
            .handler("handler.process")
            .build()
            .unwrap();

        assert_eq!(func.name(), "complex-func");
        assert_eq!(func.memory_mib(), 512);
        assert_eq!(func.timeout(), Duration::from_secs(60));
        assert_eq!(func.env.len(), 2);
        assert_eq!(func.triggers.len(), 1);
        assert_eq!(func.handler, Some("handler.process".to_string()));
    }

    #[test]
    fn test_function_builder_no_name_error() {
        let result = Function::builder()
            .runtime(Runtime::RustNative { binary: PathBuf::from("./test") })
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_function_builder_no_runtime_error() {
        let result = Function::builder().name("test").build();

        assert!(result.is_err());
    }

    #[test]
    fn test_function_id_unique() {
        let func1 = Function::builder().name("func1").runtime(Runtime::default()).build().unwrap();

        let func2 = Function::builder().name("func2").runtime(Runtime::default()).build().unwrap();

        assert_ne!(func1.id(), func2.id());
    }

    // ------------------------------------------------------------------------
    // InvocationRequest Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_invocation_request_new() {
        let request = InvocationRequest::new("my-func", b"hello".to_vec());

        assert_eq!(request.function, "my-func");
        assert_eq!(request.payload, b"hello");
        assert_eq!(request.invocation_type, InvocationType::RequestResponse);
    }

    #[test]
    fn test_invocation_request_async() {
        let request = InvocationRequest::new("my-func", Vec::new()).async_invoke();

        assert_eq!(request.invocation_type, InvocationType::Event);
    }

    // ------------------------------------------------------------------------
    // InvocationResponse Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_invocation_response_success() {
        let request_id = Uuid::new_v4();
        let response =
            InvocationResponse::success(request_id, b"result".to_vec(), Duration::from_millis(100));

        assert!(response.is_success());
        assert_eq!(response.request_id, request_id);
        assert_eq!(response.payload, b"result");
        assert!(response.error.is_none());
    }

    #[test]
    fn test_invocation_response_error() {
        let request_id = Uuid::new_v4();
        let response =
            InvocationResponse::error(request_id, "Something failed", Duration::from_millis(50));

        assert!(!response.is_success());
        assert_eq!(response.status_code, 1);
        assert_eq!(response.error, Some("Something failed".to_string()));
    }

    #[test]
    fn test_invocation_response_payload_str() {
        let response = InvocationResponse::success(
            Uuid::new_v4(),
            b"hello world".to_vec(),
            Duration::from_millis(10),
        );

        assert_eq!(response.payload_str().unwrap(), "hello world");
    }

    #[test]
    fn test_invocation_response_payload_str_invalid_utf8() {
        let response = InvocationResponse::success(
            Uuid::new_v4(),
            vec![0xFF, 0xFE],
            Duration::from_millis(10),
        );

        assert!(response.payload_str().is_err());
    }

    // ------------------------------------------------------------------------
    // FunctionService Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_function_service_new() {
        let service = FunctionService::new();
        assert_eq!(service.function_count(), 0);
        assert_eq!(service.config().max_concurrency, 1000);
    }

    #[test]
    fn test_function_service_with_config() {
        let config =
            ServiceConfig { max_concurrency: 500, warm_pool_size: 5, ..Default::default() };
        let service = FunctionService::with_config(config);
        assert_eq!(service.config().max_concurrency, 500);
        assert_eq!(service.config().warm_pool_size, 5);
    }

    #[test]
    fn test_function_service_register() {
        let mut service = FunctionService::new();

        let func = Function::builder().name("my-func").runtime(Runtime::default()).build().unwrap();

        service.register(func).unwrap();
        assert_eq!(service.function_count(), 1);
    }

    #[test]
    fn test_function_service_register_duplicate_error() {
        let mut service = FunctionService::new();

        let func1 =
            Function::builder().name("my-func").runtime(Runtime::default()).build().unwrap();

        let func2 =
            Function::builder().name("my-func").runtime(Runtime::default()).build().unwrap();

        service.register(func1).unwrap();
        let result = service.register(func2);
        assert!(result.is_err());
    }

    #[test]
    fn test_function_service_unregister() {
        let mut service = FunctionService::new();

        let func = Function::builder().name("my-func").runtime(Runtime::default()).build().unwrap();

        service.register(func).unwrap();
        assert_eq!(service.function_count(), 1);

        let removed = service.unregister("my-func");
        assert!(removed.is_some());
        assert_eq!(service.function_count(), 0);
    }

    #[test]
    fn test_function_service_get() {
        let mut service = FunctionService::new();

        let func = Function::builder()
            .name("my-func")
            .runtime(Runtime::default())
            .memory_mib(256)
            .build()
            .unwrap();

        service.register(func).unwrap();

        let retrieved = service.get("my-func");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().memory_mib(), 256);
    }

    #[test]
    fn test_function_service_get_not_found() {
        let service = FunctionService::new();
        assert!(service.get("nonexistent").is_none());
    }

    #[test]
    fn test_function_service_list() {
        let mut service = FunctionService::new();

        for i in 0..3 {
            let func = Function::builder()
                .name(format!("func-{i}"))
                .runtime(Runtime::default())
                .build()
                .unwrap();
            service.register(func).unwrap();
        }

        let list = service.list();
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn test_function_service_invoke() {
        let mut service = FunctionService::new();

        let func = Function::builder().name("echo").runtime(Runtime::default()).build().unwrap();

        service.register(func).unwrap();

        let request = InvocationRequest::new("echo", b"test input".to_vec());
        let response = service.invoke(request).unwrap();

        assert!(response.is_success());
        assert!(!response.payload.is_empty());
    }

    #[test]
    fn test_function_service_invoke_not_found() {
        let mut service = FunctionService::new();

        let request = InvocationRequest::new("nonexistent", Vec::new());
        let result = service.invoke(request);

        assert!(result.is_err());
    }

    #[test]
    fn test_function_service_invoke_dry_run() {
        let mut service = FunctionService::new();

        let func = Function::builder().name("test").runtime(Runtime::default()).build().unwrap();

        service.register(func).unwrap();

        let mut request = InvocationRequest::new("test", Vec::new());
        request.invocation_type = InvocationType::DryRun;

        let response = service.invoke(request).unwrap();
        assert!(response.is_success());
        assert!(response.payload.is_empty()); // Dry run returns empty payload
    }

    // ------------------------------------------------------------------------
    // ColdStartMetrics Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_cold_start_metrics_default() {
        let metrics = ColdStartMetrics::default();
        assert_eq!(metrics.cold_starts, 0);
        assert_eq!(metrics.warm_starts, 0);
    }

    #[test]
    fn test_cold_start_metrics_warm_ratio_zero() {
        let metrics = ColdStartMetrics::default();
        assert_eq!(metrics.warm_start_ratio(), 0.0);
    }

    #[test]
    fn test_cold_start_metrics_warm_ratio() {
        let metrics = ColdStartMetrics { cold_starts: 20, warm_starts: 80, ..Default::default() };
        assert!((metrics.warm_start_ratio() - 0.8).abs() < 0.001);
    }

    // ------------------------------------------------------------------------
    // ServiceConfig Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_service_config_default() {
        let config = ServiceConfig::default();
        assert_eq!(config.max_concurrency, 1000);
        assert_eq!(config.warm_pool_size, 3);
        assert_eq!(config.idle_timeout, Duration::from_secs(300));
        assert!(config.metrics_enabled);
    }

    // ------------------------------------------------------------------------
    // Integration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_end_to_end_function_lifecycle() {
        // Create service
        let mut service = FunctionService::new();
        assert_eq!(service.function_count(), 0);

        // Register function
        let func = Function::builder()
            .name("processor")
            .runtime(Runtime::RustNative { binary: PathBuf::from("./processor") })
            .memory_mib(256)
            .timeout(Duration::from_secs(60))
            .trigger(Trigger::Http {
                path: "/process".to_string(),
                methods: vec![HttpMethod::Post],
            })
            .build()
            .unwrap();

        let func_id = func.id();
        service.register(func).unwrap();
        assert_eq!(service.function_count(), 1);

        // Verify registration
        let retrieved = service.get("processor").unwrap();
        assert_eq!(retrieved.id(), func_id);

        // Invoke
        let request = InvocationRequest::new("processor", b"{\"data\": \"test\"}".to_vec());
        let response = service.invoke(request).unwrap();
        assert!(response.is_success());

        // Unregister
        service.unregister("processor");
        assert_eq!(service.function_count(), 0);
    }
}
