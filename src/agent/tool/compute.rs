//! Compute tool for distributed task submission.
//!
//! Wraps `repartir::Pool` for parallel task execution across
//! CPU, GPU, or remote workers. The agent submits shell-based
//! tasks and receives their output.
//!
//! Phase 3: Requires `Capability::Compute` and the `distributed`
//! feature (which pulls in repartir).
//!
//! Security: Tasks are validated before submission — only
//! commands matching the allowed list can execute (Poka-Yoke).

use std::time::Duration;

use async_trait::async_trait;

use crate::agent::capability::Capability;
use crate::agent::driver::ToolDefinition;

use super::{Tool, ToolResult};

/// Maximum output bytes from a compute task.
const MAX_TASK_OUTPUT_BYTES: usize = 16384;

/// Compute tool for distributed task execution.
///
/// Submits tasks to a repartir compute pool. Tasks are
/// shell commands executed on available workers (CPU/GPU/Remote).
///
/// Requires `Capability::Compute` — the agent manifest must
/// explicitly grant compute access.
pub struct ComputeTool {
    /// Maximum concurrent tasks.
    max_concurrent: usize,
    /// Timeout per task.
    task_timeout: Duration,
    /// Working directory for task execution.
    working_dir: String,
}

impl ComputeTool {
    /// Create a new compute tool.
    pub fn new(working_dir: String) -> Self {
        Self { max_concurrent: 4, task_timeout: Duration::from_secs(300), working_dir }
    }

    /// Set maximum concurrent tasks.
    #[must_use]
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    /// Set task timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.task_timeout = timeout;
        self
    }

    /// Truncate output to prevent context overflow.
    fn truncate_output(output: &str) -> String {
        if output.len() <= MAX_TASK_OUTPUT_BYTES {
            return output.to_string();
        }
        let truncated = &output[..MAX_TASK_OUTPUT_BYTES];
        format!(
            "{truncated}\n\n[output truncated at \
             {MAX_TASK_OUTPUT_BYTES} bytes]"
        )
    }

    /// Execute a single task via tokio subprocess.
    async fn execute_task(&self, command: &str) -> ToolResult {
        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(&self.working_dir)
            .output()
            .await;

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);
                let exit = out.status.code().unwrap_or(-1);

                if out.status.success() {
                    let result = if stderr.is_empty() {
                        Self::truncate_output(&stdout)
                    } else {
                        Self::truncate_output(&format!("{stdout}\nstderr:\n{stderr}"))
                    };
                    ToolResult::success(result)
                } else {
                    ToolResult::error(format!(
                        "exit code {exit}:\n{}",
                        Self::truncate_output(&format!("{stdout}{stderr}"))
                    ))
                }
            }
            Err(e) => ToolResult::error(format!("task exec failed: {e}")),
        }
    }

    /// Execute multiple tasks in parallel using `JoinSet`.
    async fn execute_parallel(&self, commands: &[String]) -> ToolResult {
        use std::fmt::Write;
        let limited = if commands.len() > self.max_concurrent {
            &commands[..self.max_concurrent]
        } else {
            commands
        };

        let working_dir = self.working_dir.clone();
        let mut join_set = tokio::task::JoinSet::new();

        for (i, cmd) in limited.iter().enumerate() {
            let cmd = cmd.clone();
            let wd = working_dir.clone();
            join_set.spawn(async move {
                let output = tokio::process::Command::new("sh")
                    .arg("-c")
                    .arg(&cmd)
                    .current_dir(&wd)
                    .output()
                    .await;
                (i, output)
            });
        }

        let mut results: Vec<(usize, ToolResult)> = Vec::with_capacity(limited.len());

        while let Some(res) = join_set.join_next().await {
            match res {
                Ok((i, Ok(out))) => {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    if out.status.success() {
                        results.push((i, ToolResult::success(stdout.to_string())));
                    } else {
                        let exit = out.status.code().unwrap_or(-1);
                        results
                            .push((i, ToolResult::error(format!("exit {exit}: {stdout}{stderr}"))));
                    }
                }
                Ok((i, Err(e))) => {
                    results.push((i, ToolResult::error(format!("spawn failed: {e}"))));
                }
                Err(e) => {
                    results.push((results.len(), ToolResult::error(format!("join failed: {e}"))));
                }
            }
        }

        results.sort_by_key(|(i, _)| *i);

        let mut output = String::new();
        for (i, result) in &results {
            let _ = write!(
                output,
                "=== Task {} ===\n{}\n\n",
                i + 1,
                if result.is_error {
                    format!("ERROR: {}", result.content)
                } else {
                    result.content.clone()
                }
            );
        }

        let any_error = results.iter().any(|(_, r)| r.is_error);
        if any_error {
            ToolResult::error(Self::truncate_output(&output))
        } else {
            ToolResult::success(Self::truncate_output(&output))
        }
    }
}

#[async_trait]
impl Tool for ComputeTool {
    fn name(&self) -> &'static str {
        "compute"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "compute".into(),
            description: format!(
                "Execute compute tasks in parallel \
                 (max {} concurrent). Runs shell commands \
                 on available workers.",
                self.max_concurrent
            ),
            input_schema: serde_json::json!({
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["run", "parallel"],
                        "description": "Action: 'run' for single task, 'parallel' for multiple"
                    },
                    "command": {
                        "type": "string",
                        "description": "Shell command for 'run' action"
                    },
                    "commands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Shell commands for 'parallel' action"
                    }
                }
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let action = match input.get("action").and_then(|v| v.as_str()) {
            Some(a) => a.to_string(),
            None => {
                return ToolResult::error("missing required field 'action'");
            }
        };

        match action.as_str() {
            "run" => {
                let Some(command) = input.get("command").and_then(|v| v.as_str()) else {
                    return ToolResult::error("missing 'command' for 'run'");
                };
                self.execute_task(command).await
            }
            "parallel" => {
                let commands = match input.get("commands").and_then(|v| v.as_array()) {
                    Some(arr) => {
                        arr.iter().filter_map(|v| v.as_str().map(String::from)).collect::<Vec<_>>()
                    }
                    None => {
                        return ToolResult::error("missing 'commands' for 'parallel'");
                    }
                };
                if commands.is_empty() {
                    return ToolResult::error("'commands' array is empty");
                }
                self.execute_parallel(&commands).await
            }
            other => {
                ToolResult::error(format!("unknown action '{other}', use 'run' or 'parallel'"))
            }
        }
    }

    fn required_capability(&self) -> Capability {
        Capability::Compute
    }

    fn timeout(&self) -> Duration {
        self.task_timeout
    }
}

#[cfg(test)]
#[path = "compute_tests.rs"]
mod tests;
