//! AprServeDriver — first-class inference via `apr serve` subprocess.
//!
//! Spawns `apr serve run <model>` as a child process with CUDA/GPU support,
//! then connects via OpenAI-compatible HTTP API. This is the **preferred**
//! inference path for `batuta code` / `apr code`:
//!
//! - Full CUDA/GPU acceleration (apr-cli has all features)
//! - APR and GGUF format support (prefers APR)
//! - No feature flag issues (batuta doesn't need `cuda` feature)
//! - Sovereign: localhost only, no data egress
//!
//! PMAT-160: Replaces embedded RealizarDriver as primary inference.
//! RealizarDriver remains as fallback when `apr` binary is not on PATH.

use async_trait::async_trait;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use super::{CompletionRequest, CompletionResponse, LlmDriver, Message, ToolCall};
use crate::agent::result::{AgentError, DriverError, StopReason, TokenUsage};
use crate::serve::backends::PrivacyTier;

/// Driver that uses `apr serve` subprocess for inference.
pub struct AprServeDriver {
    /// Base URL for the local server (e.g., `http://127.0.0.1:19384`)
    base_url: String,
    /// Model name for OpenAI API requests
    model_name: String,
    /// Child process handle (killed on drop)
    _child: Child,
    /// Context window size
    context_window_size: usize,
}

impl Drop for AprServeDriver {
    /// PMAT-166: Graceful shutdown — SIGTERM first, SIGKILL after 2s timeout.
    fn drop(&mut self) {
        let pid = self._child.id();

        // Try graceful shutdown first (SIGTERM on Unix via kill command)
        #[cfg(unix)]
        {
            let _ = Command::new("kill")
                .args(["-TERM", &pid.to_string()])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();

            // Wait up to 2s for graceful exit
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            loop {
                match self._child.try_wait() {
                    Ok(Some(_)) => return, // Exited cleanly
                    Ok(None) if std::time::Instant::now() < deadline => {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                    _ => break, // Timeout or error — force kill
                }
            }
        }

        // Fallback: force kill (always runs on Windows, or after SIGTERM timeout)
        let _ = self._child.kill();
        let _ = self._child.wait();
    }
}

impl AprServeDriver {
    /// Launch `apr serve run` and wait for readiness.
    ///
    /// Picks a random port, spawns the subprocess, polls the health
    /// endpoint until ready (max 30s). Returns error if `apr` not
    /// found or server fails to start.
    pub fn launch(model_path: PathBuf, context_window: Option<usize>) -> Result<Self, AgentError> {
        let apr_path = find_apr_binary()?;

        // Pick a random high port to avoid conflicts
        let port = 19384 + (std::process::id() % 1000) as u16;
        let base_url = format!("http://127.0.0.1:{port}");

        let model_name = model_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "local".to_string());

        // PMAT-181: Enable GPU with serial prefill. The FP8 batched prefill produces
        // wrong output for Qwen3 (Q6K→FP8 requantization bug). Serial prefill uses
        // Q4K/Q6K GEMV kernels which produce correct output. BATCHED_PREFILL=0 disables
        // the FP8 path while keeping CUDA acceleration for decode tokens.
        let child = Command::new(&apr_path)
            .args([
                "serve",
                "run",
                &model_path.to_string_lossy(),
                "--port",
                &port.to_string(),
                "--host",
                "127.0.0.1",
                "--gpu",
            ])
            .env("BATCHED_PREFILL", "0")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                AgentError::Driver(DriverError::InferenceFailed(format!(
                    "failed to spawn apr serve: {e}"
                )))
            })?;

        eprintln!("Launched apr serve on port {port} (pid {})", child.id());

        let mut driver = Self {
            base_url,
            model_name,
            _child: child,
            context_window_size: context_window.unwrap_or(4096),
        };

        // Wait for server to be ready
        driver.wait_for_ready()?;

        Ok(driver)
    }

    /// Poll health endpoint until server is ready (max 30s).
    ///
    /// PMAT-171: Detects subprocess death during startup. On timeout or crash,
    /// reads stderr from the child process for actionable debug output.
    fn wait_for_ready(&mut self) -> Result<(), AgentError> {
        let addr = self.base_url.trim_start_matches("http://").to_string();
        let sock_addr: std::net::SocketAddr =
            addr.parse().unwrap_or_else(|_| std::net::SocketAddr::from(([127, 0, 0, 1], 19384)));

        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(30);

        loop {
            if start.elapsed() > timeout {
                let stderr = self.drain_stderr();
                let mut msg = "apr serve did not become ready within 30s".to_string();
                if !stderr.is_empty() {
                    msg.push_str(&format!("\nsubprocess stderr:\n{stderr}"));
                }
                msg.push_str(&format!(
                    "\nDebug manually: apr serve run <model> --port {} --host 127.0.0.1",
                    addr.rsplit(':').next().unwrap_or("19384")
                ));
                return Err(AgentError::Driver(DriverError::InferenceFailed(msg)));
            }

            // Check if subprocess died
            if let Ok(Some(status)) = self._child.try_wait() {
                let stderr = self.drain_stderr();
                let mut msg = format!("apr serve exited with {status} during startup");
                if !stderr.is_empty() {
                    msg.push_str(&format!("\nsubprocess stderr:\n{stderr}"));
                }
                return Err(AgentError::Driver(DriverError::InferenceFailed(msg)));
            }

            if std::net::TcpStream::connect_timeout(
                &sock_addr,
                std::time::Duration::from_millis(200),
            )
            .is_ok()
            {
                eprintln!("apr serve ready ({:.1}s)", start.elapsed().as_secs_f64());
                return Ok(());
            }

            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }

    /// Read available stderr from the child process (non-blocking, last 2KB).
    fn drain_stderr(&mut self) -> String {
        use std::io::Read;
        let Some(stderr) = self._child.stderr.as_mut() else {
            return String::new();
        };
        let mut buf = vec![0u8; 2048];
        let n = stderr.read(&mut buf).unwrap_or(0);
        let text = String::from_utf8_lossy(&buf[..n]).to_string();
        // Return last few lines for concise output
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() > 10 {
            lines[lines.len() - 10..].join("\n")
        } else {
            text
        }
    }

    /// Build OpenAI-compatible request body.
    ///
    /// PMAT-176: Only strips the verbose `## Available Tools` section injected
    /// by `build_enriched_system()` (full JSON schemas ~2000 tokens). Preserves
    /// the compact `## Tools` table from `CODE_SYSTEM_PROMPT` — that table has
    /// tool names, use cases, and example inputs designed for 1.5B-7B models.
    fn build_openai_body(&self, request: &CompletionRequest) -> serde_json::Value {
        let mut messages = Vec::new();

        if let Some(ref system) = request.system {
            // PMAT-176: Only strip the verbose enriched section (full JSON schemas).
            // Keep the compact "## Tools" table from CODE_SYSTEM_PROMPT — it has
            // descriptions and examples that small models need for tool discovery.
            let compact_system = system
                .find("\n\n## Available Tools")
                .map(|i| &system[..i])
                .unwrap_or(system)
                .to_string();

            messages.push(serde_json::json!({
                "role": "system",
                "content": compact_system
            }));
        }

        for msg in &request.messages {
            match msg {
                Message::User(text) => messages.push(serde_json::json!({
                    "role": "user",
                    "content": text
                })),
                Message::Assistant(text) => messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": text
                })),
                Message::AssistantToolUse(call) => messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": format!("<tool_call>\n{}\n</tool_call>",
                        serde_json::json!({"name": call.name, "input": call.input}))
                })),
                Message::ToolResult(result) => messages.push(serde_json::json!({
                    "role": "user",
                    "content": format!("<tool_result>\n{}\n</tool_result>", result.content)
                })),
                _ => {}
            }
        }

        // PMAT-170: Cap max_tokens for HTTP path. The manifest default (4096)
        // causes very long generation on local models. 1024 accommodates:
        // - Tool call JSON (~100-200 tokens each)
        // - File edit content (multi-line diffs)
        // - Explanation text alongside tool calls
        // Previous 512 cap truncated complex edits mid-output.
        let max_tokens = request.max_tokens.min(1024);

        serde_json::json!({
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": request.temperature,
            "stream": false
        })
    }
}

#[async_trait]
impl LlmDriver for AprServeDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, AgentError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = self.build_openai_body(&request);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| AgentError::Driver(DriverError::Network(format!("http client: {e}"))))?;
        let response = client
            .post(&url)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentError::Driver(DriverError::Network(format!("apr serve: {e}"))))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::Driver(DriverError::Network(format!(
                "apr serve HTTP {status}: {text}"
            ))));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| AgentError::Driver(DriverError::InferenceFailed(format!("parse: {e}"))))?;

        // Extract response from OpenAI format
        let raw_text = json["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();

        // PMAT-180: Strip Qwen3 thinking blocks. The model may emit
        // <think>...</think> or bare </think> tokens. Remove them before
        // parsing tool calls — thinking content is internal reasoning.
        let text = strip_thinking_blocks(&raw_text);

        let usage = json.get("usage").cloned().unwrap_or(serde_json::json!({}));
        let input_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0);
        let output_tokens = usage["completion_tokens"].as_u64().unwrap_or(0);

        // Parse tool calls from text (same parser as RealizarDriver)
        let (clean_text, tool_calls) = super::realizar::parse_tool_calls_pub(&text);

        let stop_reason =
            if tool_calls.is_empty() { StopReason::EndTurn } else { StopReason::ToolUse };

        Ok(CompletionResponse {
            text: clean_text,
            stop_reason,
            tool_calls,
            usage: TokenUsage { input_tokens, output_tokens },
        })
    }

    fn context_window(&self) -> usize {
        self.context_window_size
    }

    fn privacy_tier(&self) -> PrivacyTier {
        // Sovereign: apr serve runs on localhost, zero network egress
        PrivacyTier::Sovereign
    }
}

/// Strip Qwen3 thinking blocks (`<think>...</think>`) and bare `</think>` tags.
fn strip_thinking_blocks(text: &str) -> String {
    let mut result = text.to_string();
    // Strip <think>...</think> blocks (may span multiple lines)
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result[start..].find("</think>") {
            result.replace_range(start..start + end + "</think>".len(), "");
        } else {
            // Unclosed <think> — strip to end
            result.truncate(start);
            break;
        }
    }
    // Strip bare </think> tags (model sometimes emits just closing tags)
    result = result.replace("</think>", "");
    result.trim().to_string()
}

/// Find the `apr` binary on PATH.
fn find_apr_binary() -> Result<PathBuf, AgentError> {
    which::which("apr").map_err(|_| {
        AgentError::Driver(DriverError::InferenceFailed(
            "apr binary not found on PATH. Install: cargo install apr-cli".into(),
        ))
    })
}

#[cfg(test)]
#[path = "apr_serve_tests.rs"]
mod tests;
