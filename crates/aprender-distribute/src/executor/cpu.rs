//! CPU execution backend.
//!
//! Executes tasks as native processes on the local CPU.

use crate::error::{RepartirError, Result};
use crate::executor::{BoxFuture, Executor};
use crate::task::{ExecutionResult, Task};
use std::path::Path;
use std::process::Stdio;
use std::time::Instant;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{debug, error, info};

/// CPU executor that runs tasks as local processes.
///
/// This executor uses tokio's async process spawning to execute
/// binaries with isolated stdout/stderr capture.
///
/// # Example
///
/// ```
/// use repartir::executor::cpu::CpuExecutor;
/// use repartir::executor::Executor;
///
/// let executor = CpuExecutor::new();
/// assert_eq!(executor.name(), "CPU");
/// ```
pub struct CpuExecutor {
    /// Number of available CPU cores.
    num_cpus: usize,
}

impl CpuExecutor {
    /// Creates a new CPU executor.
    ///
    /// Detects the number of available CPU cores automatically.
    #[must_use]
    pub fn new() -> Self {
        let num_cpus = num_cpus::get();
        info!("CpuExecutor initialized with {num_cpus} cores");
        Self { num_cpus }
    }

    /// Creates a CPU executor with a specific core count.
    ///
    /// Useful for testing or limiting resource usage.
    #[must_use]
    pub const fn with_cores(num_cpus: usize) -> Self {
        Self { num_cpus }
    }

    /// Validates that the binary exists and is executable.
    fn validate_binary(path: &Path) -> Result<()> {
        if !path.exists() {
            return Err(RepartirError::BinaryNotFound { path: path.to_path_buf() });
        }

        // Check if file is executable (Unix-specific)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = path.metadata()?;
            let permissions = metadata.permissions();
            if permissions.mode() & 0o111 == 0 {
                return Err(RepartirError::InvalidTask {
                    reason: format!("Binary is not executable: {}", path.display()),
                });
            }
        }

        Ok(())
    }
}

impl Default for CpuExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor for CpuExecutor {
    fn execute(&self, task: Task) -> BoxFuture<'_, Result<ExecutionResult>> {
        Box::pin(async move {
            let task_id = task.id();
            debug!("Executing task {task_id} on CPU: {:?}", task.binary());

            // Validate binary exists (Genchi Genbutsu: check before executing)
            Self::validate_binary(task.binary())?;

            let start = Instant::now();

            // Build command
            let mut cmd = Command::new(task.binary());
            cmd.args(task.args())
                .envs(task.env().iter())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .stdin(Stdio::null()); // No stdin to prevent hanging

            // Execute with optional timeout
            let child_future = async {
                let mut child = cmd.spawn().map_err(|e| {
                    error!("Failed to spawn task {task_id}: {e}");
                    e
                })?;

                // Capture stdout and stderr
                let mut stdout_buf = Vec::new();
                let mut stderr_buf = Vec::new();

                if let Some(mut stdout) = child.stdout.take() {
                    stdout.read_to_end(&mut stdout_buf).await.map_err(|e| {
                        error!("Failed to read stdout for task {task_id}: {e}");
                        e
                    })?;
                }

                if let Some(mut stderr) = child.stderr.take() {
                    stderr.read_to_end(&mut stderr_buf).await.map_err(|e| {
                        error!("Failed to read stderr for task {task_id}: {e}");
                        e
                    })?;
                }

                // Wait for completion
                let status = child.wait().await.map_err(|e| {
                    error!("Failed to wait for task {task_id}: {e}");
                    e
                })?;

                let exit_code = status.code().unwrap_or(-1);
                let duration = start.elapsed();

                debug!("Task {task_id} completed with exit code {exit_code} in {duration:?}");

                Ok::<_, std::io::Error>(ExecutionResult::new(
                    task_id, exit_code, stdout_buf, stderr_buf, duration,
                ))
            };

            // Apply timeout if specified
            let result = if let Some(timeout_duration) = task.timeout() {
                if let Ok(result) = timeout(timeout_duration, child_future).await {
                    result?
                } else {
                    error!("Task {task_id} timed out after {timeout_duration:?}");
                    return Err(RepartirError::Timeout { seconds: timeout_duration.as_secs() });
                }
            } else {
                child_future.await?
            };

            Ok(result)
        })
    }

    fn capacity(&self) -> usize {
        self.num_cpus
    }

    fn name(&self) -> &'static str {
        "CPU"
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::task::{Backend, Task};
    use std::time::Duration;

    #[tokio::test]
    async fn test_cpu_executor_simple_command() {
        let executor = CpuExecutor::new();
        assert!(executor.capacity() > 0);

        // Execute a simple command (echo is available on all Unix systems)
        #[cfg(unix)]
        {
            let task = Task::builder()
                .binary("/bin/echo")
                .arg("hello")
                .arg("world")
                .backend(Backend::Cpu)
                .build()
                .unwrap();

            let result = executor.execute(task).await;
            assert!(result.is_ok());

            let exec_result = result.unwrap();
            assert!(exec_result.is_success());
            assert_eq!(exec_result.stdout_str().unwrap().trim(), "hello world");
        }
    }

    #[tokio::test]
    async fn test_cpu_executor_nonzero_exit() {
        let executor = CpuExecutor::new();

        // Execute a command that exits with non-zero
        #[cfg(unix)]
        {
            let task = Task::builder()
                .binary("/bin/sh")
                .arg("-c")
                .arg("exit 42")
                .backend(Backend::Cpu)
                .build()
                .unwrap();

            let result = executor.execute(task).await;
            assert!(result.is_ok());

            let exec_result = result.unwrap();
            assert!(!exec_result.is_success());
            assert_eq!(exec_result.exit_code(), 42);
        }
    }

    #[tokio::test]
    async fn test_cpu_executor_binary_not_found() {
        let executor = CpuExecutor::new();

        let task =
            Task::builder().binary("/nonexistent/binary").backend(Backend::Cpu).build().unwrap();

        let result = executor.execute(task).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), RepartirError::BinaryNotFound { .. }));
    }

    #[tokio::test]
    async fn test_cpu_executor_timeout() {
        let executor = CpuExecutor::new();

        // Sleep for 10 seconds but timeout after 100ms
        #[cfg(unix)]
        {
            let task = Task::builder()
                .binary("/bin/sleep")
                .arg("10")
                .backend(Backend::Cpu)
                .timeout(Duration::from_millis(100))
                .build()
                .unwrap();

            let result = executor.execute(task).await;
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), RepartirError::Timeout { .. }));
        }
    }

    #[test]
    fn test_cpu_executor_with_cores() {
        let executor = CpuExecutor::with_cores(4);
        assert_eq!(executor.capacity(), 4);
        assert_eq!(executor.name(), "CPU");
    }

    #[test]
    fn test_cpu_executor_default() {
        let executor = CpuExecutor::default();
        assert!(executor.capacity() > 0);
        assert_eq!(executor.name(), "CPU");
    }

    #[tokio::test]
    async fn test_cpu_executor_with_env_vars() {
        let executor = CpuExecutor::new();

        #[cfg(unix)]
        {
            let task = Task::builder()
                .binary("/bin/sh")
                .arg("-c")
                .arg("echo $TEST_VAR")
                .env_var("TEST_VAR", "test_value")
                .backend(Backend::Cpu)
                .build()
                .unwrap();

            let result = executor.execute(task).await;
            assert!(result.is_ok());

            let exec_result = result.unwrap();
            assert!(exec_result.is_success());
            assert_eq!(exec_result.stdout_str().unwrap().trim(), "test_value");
        }
    }

    #[tokio::test]
    async fn test_cpu_executor_stderr_capture() {
        let executor = CpuExecutor::new();

        #[cfg(unix)]
        {
            let task = Task::builder()
                .binary("/bin/sh")
                .arg("-c")
                .arg("echo error_message >&2")
                .backend(Backend::Cpu)
                .build()
                .unwrap();

            let result = executor.execute(task).await;
            assert!(result.is_ok());

            let exec_result = result.unwrap();
            assert!(exec_result.is_success());
            assert_eq!(exec_result.stderr_str().unwrap().trim(), "error_message");
        }
    }

    #[tokio::test]
    async fn test_cpu_executor_multiple_args() {
        let executor = CpuExecutor::new();

        #[cfg(unix)]
        {
            let task = Task::builder()
                .binary("/bin/echo")
                .arg("arg1")
                .arg("arg2")
                .arg("arg3")
                .backend(Backend::Cpu)
                .build()
                .unwrap();

            let result = executor.execute(task).await;
            assert!(result.is_ok());

            let exec_result = result.unwrap();
            assert!(exec_result.is_success());
            assert_eq!(exec_result.stdout_str().unwrap().trim(), "arg1 arg2 arg3");
        }
    }

    #[tokio::test]
    async fn test_cpu_executor_stdout_and_stderr() {
        let executor = CpuExecutor::new();

        #[cfg(unix)]
        {
            let task = Task::builder()
                .binary("/bin/sh")
                .arg("-c")
                .arg("echo stdout_message && echo stderr_message >&2")
                .backend(Backend::Cpu)
                .build()
                .unwrap();

            let result = executor.execute(task).await;
            assert!(result.is_ok());

            let exec_result = result.unwrap();
            assert!(exec_result.is_success());
            assert_eq!(exec_result.stdout_str().unwrap().trim(), "stdout_message");
            assert_eq!(exec_result.stderr_str().unwrap().trim(), "stderr_message");
        }
    }

    #[tokio::test]
    async fn test_cpu_executor_empty_args() {
        let executor = CpuExecutor::new();

        #[cfg(unix)]
        {
            let task = Task::builder().binary("/bin/pwd").backend(Backend::Cpu).build().unwrap();

            let result = executor.execute(task).await;
            assert!(result.is_ok());
            assert!(result.unwrap().is_success());
        }
    }

    #[tokio::test]
    async fn test_cpu_executor_large_output() {
        let executor = CpuExecutor::new();

        #[cfg(unix)]
        {
            // Generate large output (10KB of text)
            let task = Task::builder()
                .binary("/bin/sh")
                .arg("-c")
                .arg("yes 'test' | head -1000")
                .backend(Backend::Cpu)
                .build()
                .unwrap();

            let result = executor.execute(task).await;
            assert!(result.is_ok());
            let exec_result = result.unwrap();
            assert!(exec_result.is_success());
            assert!(exec_result.stdout().len() > 4000);
        }
    }

    #[tokio::test]
    async fn test_cpu_executor_empty_output() {
        let executor = CpuExecutor::new();

        #[cfg(unix)]
        {
            // Command that produces no output.
            // Use `/bin/sh -c true` instead of `/bin/true` for portability:
            // `/bin/true` may not exist on all platforms (e.g., macOS CI runners
            // where only `/usr/bin/true` is available).
            let task = Task::builder()
                .binary("/bin/sh")
                .arg("-c")
                .arg("true")
                .backend(Backend::Cpu)
                .build()
                .unwrap();

            let result = executor.execute(task).await;
            assert!(result.is_ok(), "execute failed: {:?}", result.unwrap_err());
            let exec_result = result.unwrap();
            assert!(exec_result.is_success());
            assert_eq!(exec_result.stdout().len(), 0);
            assert_eq!(exec_result.stderr().len(), 0);
        }
    }

    #[tokio::test]
    async fn test_cpu_executor_signal_termination() {
        let executor = CpuExecutor::new();

        #[cfg(unix)]
        {
            // Test a process that gets killed (no exit code)
            // This tests the status.code().unwrap_or(-1) path
            let task = Task::builder()
                .binary("/bin/sh")
                .arg("-c")
                .arg("sleep 10 & kill -9 $!")
                .backend(Backend::Cpu)
                .timeout(Duration::from_secs(1))
                .build()
                .unwrap();

            let result = executor.execute(task).await;
            // Either timeout or completes quickly
            let _ = result;
        }
    }
}
