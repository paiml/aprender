//! Task definition and execution results.
//!
//! Core abstractions for distributed work in repartir.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use uuid::Uuid;

/// Unique identifier for a task.
///
/// Per Iron Lotus case study (Section 12.3), we use UUIDs instead of
/// indices to prevent invalidation when workers disconnect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(Uuid);

impl TaskId {
    /// Creates a new random task ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Returns the inner UUID.
    #[must_use]
    pub const fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Execution backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    /// Execute on local CPU.
    Cpu,
    /// Execute on GPU (requires `gpu` feature).
    #[cfg(feature = "gpu")]
    Gpu,
    /// Execute in `MicroVM` (requires `microvm` feature).
    #[cfg(feature = "microvm")]
    MicroVm,
    /// Execute with SIMD acceleration (requires `simd` feature).
    #[cfg(feature = "simd")]
    Simd,
    /// Execute on remote worker (requires `remote` feature).
    #[cfg(feature = "remote")]
    Remote,
}

/// Priority level for task scheduling.
///
/// Higher priority tasks are scheduled before lower priority tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum Priority {
    /// Low priority (background tasks).
    Low = 0,
    /// Normal priority (default).
    #[default]
    Normal = 1,
    /// High priority (latency-sensitive).
    High = 2,
}

/// A task to be executed.
///
/// Tasks are the fundamental unit of work in repartir. Each task
/// represents a Rust binary to execute with specific arguments and environment.
///
/// # Example
///
/// ```
/// use repartir::task::{Task, Backend};
///
/// let task = Task::builder()
///     .binary("./target/release/worker")
///     .args(vec!["--input".to_string(), "data.bin".to_string()])
///     .backend(Backend::Cpu)
///     .build();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier.
    id: TaskId,
    /// Path to the binary to execute (CPU/Remote) or shader file (GPU).
    binary: PathBuf,
    /// Command-line arguments (CPU/Remote only).
    args: Vec<String>,
    /// Environment variables (CPU/Remote only).
    env: HashMap<String, String>,
    /// Execution backend.
    backend: Backend,
    /// Priority level.
    priority: Priority,
    /// Optional timeout.
    timeout: Option<Duration>,
    /// GPU compute shader code (WGSL source as UTF-8 bytes, GPU only).
    #[cfg(feature = "gpu")]
    shader_code: Option<Vec<u8>>,
    /// GPU input buffer data (GPU only).
    #[cfg(feature = "gpu")]
    input_buffers: Vec<Vec<u8>>,
    /// GPU output buffer sizes (GPU only).
    #[cfg(feature = "gpu")]
    output_buffer_sizes: Vec<u64>,
    /// GPU workgroup dimensions (x, y, z) - GPU only.
    #[cfg(feature = "gpu")]
    workgroup_size: Option<(u32, u32, u32)>,
    /// Whether checkpointing is enabled (v2.0).
    #[cfg(feature = "checkpoint")]
    checkpoint_enabled: bool,
    /// Checkpoint interval (v2.0).
    #[cfg(feature = "checkpoint")]
    checkpoint_interval: Option<Duration>,
    /// Data dependencies for locality-aware scheduling (v2.0).
    data_dependencies: Vec<String>,
}

impl Task {
    /// Creates a new task builder.
    #[must_use]
    pub fn builder() -> TaskBuilder {
        contract_pre_configuration!();
        let result = TaskBuilder::default();
        contract_post_configuration!(&"ok");
        result
    }

    /// Returns the task ID.
    #[must_use]
    pub const fn id(&self) -> TaskId {
        self.id
    }

    /// Returns the binary path.
    #[must_use]
    pub const fn binary(&self) -> &PathBuf {
        &self.binary
    }

    /// Returns the arguments.
    #[must_use]
    pub fn args(&self) -> &[String] {
        &self.args
    }

    /// Returns the environment variables.
    #[must_use]
    pub const fn env(&self) -> &HashMap<String, String> {
        &self.env
    }

    /// Returns the backend.
    #[must_use]
    pub const fn backend(&self) -> Backend {
        self.backend
    }

    /// Returns the priority.
    #[must_use]
    pub const fn priority(&self) -> Priority {
        self.priority
    }

    /// Returns the timeout.
    #[must_use]
    pub const fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// Returns the GPU shader code (WGSL source as UTF-8 bytes).
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn shader_code(&self) -> Option<&[u8]> {
        self.shader_code.as_deref()
    }

    /// Returns the GPU input buffers.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn input_buffers(&self) -> &[Vec<u8>] {
        &self.input_buffers
    }

    /// Returns the GPU output buffer sizes.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn output_buffer_sizes(&self) -> &[u64] {
        &self.output_buffer_sizes
    }

    /// Returns the GPU workgroup size (x, y, z).
    #[cfg(feature = "gpu")]
    #[must_use]
    pub const fn workgroup_size(&self) -> Option<(u32, u32, u32)> {
        self.workgroup_size
    }

    /// Returns whether checkpointing is enabled (v2.0).
    #[cfg(feature = "checkpoint")]
    #[must_use]
    pub const fn checkpoint_enabled(&self) -> bool {
        self.checkpoint_enabled
    }

    /// Returns the checkpoint interval (v2.0).
    #[cfg(feature = "checkpoint")]
    #[must_use]
    pub const fn checkpoint_interval(&self) -> Option<Duration> {
        self.checkpoint_interval
    }

    /// Returns the data dependencies (v2.0).
    #[must_use]
    pub fn data_dependencies(&self) -> &[String] {
        &self.data_dependencies
    }
}

/// Builder for `Task`.
#[derive(Debug)]
pub struct TaskBuilder {
    binary: Option<PathBuf>,
    args: Vec<String>,
    env: HashMap<String, String>,
    backend: Backend,
    priority: Priority,
    timeout: Option<Duration>,
    #[cfg(feature = "gpu")]
    shader_code: Option<Vec<u8>>,
    #[cfg(feature = "gpu")]
    input_buffers: Vec<Vec<u8>>,
    #[cfg(feature = "gpu")]
    output_buffer_sizes: Vec<u64>,
    #[cfg(feature = "gpu")]
    workgroup_size: Option<(u32, u32, u32)>,
    #[cfg(feature = "checkpoint")]
    checkpoint_enabled: bool,
    #[cfg(feature = "checkpoint")]
    checkpoint_interval: Option<Duration>,
    data_dependencies: Vec<String>,
}

impl Default for TaskBuilder {
    fn default() -> Self {
        Self {
            binary: None,
            args: Vec::new(),
            env: HashMap::new(),
            backend: Backend::Cpu,
            priority: Priority::default(),
            timeout: None,
            #[cfg(feature = "gpu")]
            shader_code: None,
            #[cfg(feature = "gpu")]
            input_buffers: Vec::new(),
            #[cfg(feature = "gpu")]
            output_buffer_sizes: Vec::new(),
            #[cfg(feature = "gpu")]
            workgroup_size: None,
            #[cfg(feature = "checkpoint")]
            checkpoint_enabled: false,
            #[cfg(feature = "checkpoint")]
            checkpoint_interval: None,
            data_dependencies: Vec::new(),
        }
    }
}

impl TaskBuilder {
    /// Sets the binary path.
    #[must_use]
    pub fn binary<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.binary = Some(path.into());
        self
    }

    /// Sets the arguments.
    #[must_use]
    pub fn args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    /// Adds a single argument.
    #[must_use]
    pub fn arg<S: Into<String>>(mut self, arg: S) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Sets the environment variables.
    #[must_use]
    pub fn env(mut self, env: HashMap<String, String>) -> Self {
        self.env = env;
        self
    }

    /// Adds a single environment variable.
    #[must_use]
    pub fn env_var<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Sets the backend.
    #[must_use]
    pub const fn backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Sets the priority.
    #[must_use]
    pub const fn priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Sets the timeout.
    #[must_use]
    pub const fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the GPU shader code (WGSL source).
    ///
    /// For GPU compute tasks, provide the WGSL shader source code as UTF-8 bytes.
    /// This is mutually exclusive with binary execution.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use repartir::task::{Task, Backend};
    /// let shader = r#"
    ///     @group(0) @binding(0) var<storage, read> input: array<f32>;
    ///     @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    ///
    ///     @compute @workgroup_size(64)
    ///     fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    ///         output[global_id.x] = input[global_id.x] * 2.0;
    ///     }
    /// "#;
    ///
    /// let task = Task::builder()
    ///     .shader_code(shader.as_bytes().to_vec())
    ///     .backend(Backend::Gpu)
    ///     .build();
    /// ```
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn shader_code(mut self, code: Vec<u8>) -> Self {
        self.shader_code = Some(code);
        self
    }

    /// Adds an input buffer for GPU compute.
    ///
    /// Input buffers are uploaded to the GPU before shader execution.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn input_buffer(mut self, data: Vec<u8>) -> Self {
        self.input_buffers.push(data);
        self
    }

    /// Sets all input buffers at once for GPU compute.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn input_buffers(mut self, buffers: Vec<Vec<u8>>) -> Self {
        self.input_buffers = buffers;
        self
    }

    /// Adds an output buffer size for GPU compute.
    ///
    /// Output buffer sizes (in bytes) must be specified before execution.
    /// Results will be read back from GPU into buffers of these sizes.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn output_buffer_size(mut self, size: u64) -> Self {
        self.output_buffer_sizes.push(size);
        self
    }

    /// Sets all output buffer sizes at once for GPU compute.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn output_buffer_sizes(mut self, sizes: Vec<u64>) -> Self {
        self.output_buffer_sizes = sizes;
        self
    }

    /// Sets the GPU workgroup dimensions (x, y, z).
    ///
    /// This determines how many workgroups are dispatched.
    /// Default is (1, 1, 1) if not set.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub const fn workgroup_size(mut self, x: u32, y: u32, z: u32) -> Self {
        self.workgroup_size = Some((x, y, z));
        self
    }

    /// Enables checkpointing for this task (v2.0).
    ///
    /// When enabled, the task can be checkpointed periodically
    /// and resumed from the last checkpoint on failure.
    ///
    /// Requires the `checkpoint` feature.
    #[cfg(feature = "checkpoint")]
    #[must_use]
    pub const fn enable_checkpointing(mut self, enabled: bool) -> Self {
        self.checkpoint_enabled = enabled;
        self
    }

    /// Sets the checkpoint interval (v2.0).
    ///
    /// The task will be checkpointed at this interval.
    /// Only effective when checkpointing is enabled.
    ///
    /// Requires the `checkpoint` feature.
    #[cfg(feature = "checkpoint")]
    #[must_use]
    pub const fn checkpoint_interval(mut self, interval: Duration) -> Self {
        self.checkpoint_interval = Some(interval);
        self
    }

    /// Sets data dependencies for locality-aware scheduling (v2.0).
    ///
    /// The scheduler will prefer workers that have these data items cached.
    ///
    /// # Arguments
    ///
    /// * `deps` - List of data keys (tensor IDs, checkpoint IDs, file paths)
    #[must_use]
    pub fn data_dependencies(mut self, deps: Vec<String>) -> Self {
        self.data_dependencies = deps;
        self
    }

    /// Adds a single data dependency (v2.0).
    #[must_use]
    pub fn data_dependency<S: Into<String>>(mut self, dep: S) -> Self {
        self.data_dependencies.push(dep.into());
        self
    }

    /// Builds the task.
    ///
    /// # Errors
    ///
    /// Returns an error if the binary path is not set.
    pub fn build(self) -> Result<Task> {
        contract_pre_configuration!();
        // For GPU tasks with shader code, binary is optional
        #[cfg(feature = "gpu")]
        let binary_required = self.shader_code.is_none();
        #[cfg(not(feature = "gpu"))]
        let binary_required = true;

        let binary = if binary_required {
            self.binary.ok_or_else(|| crate::error::RepartirError::InvalidTask {
                reason: "Binary path not set".to_string(),
            })?
        } else {
            self.binary.unwrap_or_else(|| PathBuf::from(""))
        };

        Ok(Task {
            id: TaskId::new(),
            binary,
            args: self.args,
            env: self.env,
            backend: self.backend,
            priority: self.priority,
            timeout: self.timeout,
            #[cfg(feature = "gpu")]
            shader_code: self.shader_code,
            #[cfg(feature = "gpu")]
            input_buffers: self.input_buffers,
            #[cfg(feature = "gpu")]
            output_buffer_sizes: self.output_buffer_sizes,
            #[cfg(feature = "gpu")]
            workgroup_size: self.workgroup_size,
            #[cfg(feature = "checkpoint")]
            checkpoint_enabled: self.checkpoint_enabled,
            #[cfg(feature = "checkpoint")]
            checkpoint_interval: self.checkpoint_interval,
            data_dependencies: self.data_dependencies,
        })
    }
}

/// Result of task execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Task ID that was executed.
    task_id: TaskId,
    /// Exit code from the process (0 for success, non-zero for error).
    /// For GPU tasks, 0 indicates successful execution.
    exit_code: i32,
    /// Standard output (CPU/Remote binary execution only).
    stdout: Vec<u8>,
    /// Standard error (CPU/Remote binary execution only).
    stderr: Vec<u8>,
    /// Execution duration.
    duration: Duration,
    /// GPU output buffers (GPU execution only).
    #[cfg(feature = "gpu")]
    gpu_output_buffers: Vec<Vec<u8>>,
}

impl ExecutionResult {
    /// Creates a new execution result for binary tasks.
    #[must_use]
    pub const fn new(
        task_id: TaskId,
        exit_code: i32,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
        duration: Duration,
    ) -> Self {
        Self {
            task_id,
            exit_code,
            stdout,
            stderr,
            duration,
            #[cfg(feature = "gpu")]
            gpu_output_buffers: Vec::new(),
        }
    }

    /// Creates a new execution result for GPU compute tasks.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub const fn new_gpu(
        task_id: TaskId,
        exit_code: i32,
        output_buffers: Vec<Vec<u8>>,
        duration: Duration,
    ) -> Self {
        Self {
            task_id,
            exit_code,
            stdout: Vec::new(),
            stderr: Vec::new(),
            duration,
            gpu_output_buffers: output_buffers,
        }
    }

    /// Returns the task ID.
    #[must_use]
    pub const fn task_id(&self) -> TaskId {
        self.task_id
    }

    /// Returns the exit code.
    #[must_use]
    pub const fn exit_code(&self) -> i32 {
        self.exit_code
    }

    /// Returns whether the task succeeded (exit code 0).
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.exit_code == 0
    }

    /// Returns the standard output.
    #[must_use]
    pub fn stdout(&self) -> &[u8] {
        &self.stdout
    }

    /// Returns the standard error.
    #[must_use]
    pub fn stderr(&self) -> &[u8] {
        &self.stderr
    }

    /// Returns the execution duration.
    #[must_use]
    pub const fn duration(&self) -> Duration {
        self.duration
    }

    /// Converts stdout to a UTF-8 string.
    ///
    /// # Errors
    ///
    /// Returns an error if stdout is not valid UTF-8.
    pub fn stdout_str(&self) -> Result<&str> {
        std::str::from_utf8(&self.stdout).map_err(|_| crate::error::RepartirError::InvalidTask {
            reason: "Stdout is not valid UTF-8".to_string(),
        })
    }

    /// Converts stderr to a UTF-8 string.
    ///
    /// # Errors
    ///
    /// Returns an error if stderr is not valid UTF-8.
    pub fn stderr_str(&self) -> Result<&str> {
        std::str::from_utf8(&self.stderr).map_err(|_| crate::error::RepartirError::InvalidTask {
            reason: "Stderr is not valid UTF-8".to_string(),
        })
    }

    /// Returns the GPU output buffers (GPU tasks only).
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn gpu_output_buffers(&self) -> &[Vec<u8>] {
        &self.gpu_output_buffers
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn test_task_id_creation() {
        let id1 = TaskId::new();
        let id2 = TaskId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_task_builder() {
        let result = Task::builder()
            .binary("/bin/echo")
            .arg("hello")
            .arg("world")
            .backend(Backend::Cpu)
            .priority(Priority::High)
            .build();

        assert!(result.is_ok());
        let task = result.unwrap();
        assert_eq!(task.args(), &["hello", "world"]);
        assert_eq!(task.backend(), Backend::Cpu);
        assert_eq!(task.priority(), Priority::High);
    }

    #[test]
    fn test_task_builder_missing_binary() {
        let result = Task::builder().arg("test").build();
        assert!(result.is_err());
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_execution_result() {
        let task_id = TaskId::new();
        let result = ExecutionResult::new(
            task_id,
            0,
            b"output".to_vec(),
            b"".to_vec(),
            Duration::from_secs(1),
        );

        assert!(result.is_success());
        assert_eq!(result.exit_code(), 0);
        assert_eq!(result.stdout_str().unwrap(), "output");
    }

    #[test]
    fn test_task_id_default() {
        let id = TaskId::default();
        assert!(id.as_uuid().get_version_num() == 4);
    }

    #[test]
    fn test_task_id_display() {
        let id = TaskId::new();
        let displayed = format!("{id}");
        assert!(!displayed.is_empty());
        assert_eq!(displayed, id.as_uuid().to_string());
    }

    #[test]
    fn test_task_id_as_uuid() {
        let id = TaskId::new();
        let uuid = id.as_uuid();
        assert!(uuid.get_version_num() == 4);
    }

    #[test]
    fn test_task_builder_full() {
        let mut env = HashMap::new();
        env.insert("KEY1".to_string(), "VALUE1".to_string());

        let result = Task::builder()
            .binary("/bin/test")
            .args(vec!["arg1".to_string(), "arg2".to_string()])
            .env(env.clone())
            .env_var("KEY2", "VALUE2")
            .backend(Backend::Cpu)
            .priority(Priority::Low)
            .timeout(Duration::from_secs(10))
            .build();

        assert!(result.is_ok());
        let task = result.unwrap();
        assert_eq!(task.binary().to_str().unwrap(), "/bin/test");
        assert_eq!(task.args(), &["arg1", "arg2"]);
        assert_eq!(task.env().get("KEY1").unwrap(), "VALUE1");
        assert_eq!(task.env().get("KEY2").unwrap(), "VALUE2");
        assert_eq!(task.backend(), Backend::Cpu);
        assert_eq!(task.priority(), Priority::Low);
        assert_eq!(task.timeout(), Some(Duration::from_secs(10)));
    }

    #[test]
    fn test_execution_result_failure() {
        let task_id = TaskId::new();
        let result = ExecutionResult::new(
            task_id,
            1,
            b"".to_vec(),
            b"error message".to_vec(),
            Duration::from_millis(500),
        );

        assert!(!result.is_success());
        assert_eq!(result.exit_code(), 1);
        assert_eq!(result.task_id(), task_id);
        assert_eq!(result.stdout(), b"");
        assert_eq!(result.stderr(), b"error message");
        assert_eq!(result.stderr_str().unwrap(), "error message");
        assert_eq!(result.duration(), Duration::from_millis(500));
    }

    #[test]
    fn test_execution_result_invalid_utf8() {
        let task_id = TaskId::new();
        let invalid_utf8 = vec![0xFF, 0xFE, 0xFD];
        let result = ExecutionResult::new(
            task_id,
            0,
            invalid_utf8.clone(),
            invalid_utf8,
            Duration::from_secs(1),
        );

        assert!(result.stdout_str().is_err());
        assert!(result.stderr_str().is_err());
    }

    #[test]
    fn test_priority_default() {
        let priority = Priority::default();
        assert_eq!(priority, Priority::Normal);
    }

    #[test]
    fn test_backend_cpu() {
        let task = Task::builder().binary("/bin/echo").backend(Backend::Cpu).build().unwrap();
        assert_eq!(task.backend(), Backend::Cpu);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_backend_gpu() {
        let task = Task::builder().binary("/bin/echo").backend(Backend::Gpu).build().unwrap();
        assert_eq!(task.backend(), Backend::Gpu);
    }

    #[cfg(feature = "remote")]
    #[test]
    fn test_backend_remote() {
        let task = Task::builder().binary("/bin/echo").backend(Backend::Remote).build().unwrap();
        assert_eq!(task.backend(), Backend::Remote);
    }

    #[cfg(feature = "checkpoint")]
    #[test]
    fn test_checkpoint_enabled() {
        let task = Task::builder()
            .binary("/bin/echo")
            .enable_checkpointing(true)
            .checkpoint_interval(Duration::from_secs(60))
            .build()
            .unwrap();

        assert!(task.checkpoint_enabled());
        assert_eq!(task.checkpoint_interval(), Some(Duration::from_secs(60)));
    }

    #[cfg(feature = "checkpoint")]
    #[test]
    fn test_checkpoint_disabled_by_default() {
        let task = Task::builder().binary("/bin/echo").build().unwrap();

        assert!(!task.checkpoint_enabled());
        assert_eq!(task.checkpoint_interval(), None);
    }

    #[test]
    fn test_data_dependencies() {
        let task = Task::builder()
            .binary("/bin/echo")
            .data_dependencies(vec!["tensor_1".to_string(), "checkpoint_42".to_string()])
            .build()
            .unwrap();

        assert_eq!(task.data_dependencies().len(), 2);
        assert_eq!(task.data_dependencies()[0], "tensor_1");
        assert_eq!(task.data_dependencies()[1], "checkpoint_42");
    }

    #[test]
    fn test_data_dependency_single() {
        let task = Task::builder()
            .binary("/bin/echo")
            .data_dependency("tensor_batch_42")
            .data_dependency("checkpoint_123")
            .build()
            .unwrap();

        assert_eq!(task.data_dependencies().len(), 2);
        assert_eq!(task.data_dependencies()[0], "tensor_batch_42");
        assert_eq!(task.data_dependencies()[1], "checkpoint_123");
    }
}
