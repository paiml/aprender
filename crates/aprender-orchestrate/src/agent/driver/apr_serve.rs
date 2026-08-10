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
    /// Model file size in bytes (used to scale the startup-ready timeout
    /// for large MoE GGUFs). `None` if stat failed at launch time.
    model_size_bytes: Option<u64>,
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
        let mut cmd = Command::new(&apr_path);
        cmd.args([
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
        .stderr(Stdio::piped());

        // Issue #1712: kernel-enforced reaping. Drop on AprServeDriver only fires
        // for graceful Rust exit — if `apr code` is killed by `timeout`, SIGTERM,
        // SIGKILL, or an uncaught panic, the Drop never runs and `apr serve`
        // orphans (~3 GB RSS each). PR_SET_PDEATHSIG asks the kernel to SIGTERM
        // the child the moment its parent dies, independent of cleanup paths.
        configure_parent_death_signal(&mut cmd);

        let child = cmd.spawn().map_err(|e| {
            AgentError::Driver(DriverError::InferenceFailed(format!(
                "failed to spawn apr serve: {e}"
            )))
        })?;

        eprintln!("Launched apr serve on port {port} (pid {})", child.id());

        let model_size_bytes = std::fs::metadata(&model_path).ok().map(|m| m.len());

        let mut driver = Self {
            base_url,
            model_name,
            _child: child,
            context_window_size: context_window.unwrap_or(4096),
            model_size_bytes,
        };

        // Wait for server to be ready
        driver.wait_for_ready()?;

        Ok(driver)
    }

    /// Poll health endpoint until server is ready.
    ///
    /// Default budget is 30 seconds — fine for small models that mmap and
    /// validate in <5s. Large MoE GGUFs (e.g., Qwen3-Coder-30B at 18.5 GB)
    /// can exceed 30s on cold-cache loads, so the budget is overridable
    /// via `APR_SERVE_READY_TIMEOUT_S` (an integer count of seconds).
    /// The default also auto-scales by model file size when the model
    /// is known: +1 second per 500 MB above 2 GB (e.g., an 18 GB model
    /// gets ~62s budget; a 1 GB model gets the 30s baseline).
    ///
    /// PMAT-171: Detects subprocess death during startup. On timeout or crash,
    /// reads stderr from the child process for actionable debug output.
    ///
    /// PR #1781: Configurable timeout + size-aware default.
    fn wait_for_ready(&mut self) -> Result<(), AgentError> {
        let addr = self.base_url.trim_start_matches("http://").to_string();
        let sock_addr: std::net::SocketAddr =
            addr.parse().unwrap_or_else(|_| std::net::SocketAddr::from(([127, 0, 0, 1], 19384)));

        let timeout_secs = self.resolve_ready_timeout_secs();
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(timeout_secs);

        loop {
            if start.elapsed() > timeout {
                let stderr = self.drain_stderr();
                let mut msg = format!(
                    "apr serve did not become ready within {timeout_secs}s (override via APR_SERVE_READY_TIMEOUT_S)"
                );
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

    /// Resolve the startup-ready timeout in seconds.
    ///
    /// Reads `APR_SERVE_READY_TIMEOUT_S` from the env (operator override)
    /// and falls back to a size-aware default. See
    /// [`compute_ready_timeout_secs`] for the resolution rules + unit tests.
    fn resolve_ready_timeout_secs(&self) -> u64 {
        let env_override = std::env::var("APR_SERVE_READY_TIMEOUT_S").ok();
        compute_ready_timeout_secs(self.model_size_bytes, env_override.as_deref())
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
        //
        // aprender#1789 follow-up: env-var override for large MoE models
        // without KV cache. At ~0.5 tok/s (30B-MoE-no-KV), 1024 tokens
        // takes ~34 min — exceeds reasonable per-turn budgets. Allow the
        // operator (or bench harness) to dial down for slow models.
        let max_tokens_cap = std::env::var("APR_AGENT_MAX_TOKENS_CAP")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(1024);
        let max_tokens = request.max_tokens.min(max_tokens_cap);

        // 3-knob toolkit (qwen3-moe-sampling-v1 + qwen3-moe-repetition-penalty-v1):
        // operator env-var overrides for sampling parameters. When set, these
        // flow from apr code → HTTP body → apr serve's try_qwen3_moe_backend
        // → QuantizedGenerateConfig → run_qwen3_moe_generate → sample_from_logits.
        // When UNSET, the request still uses temperature from CompletionRequest
        // (existing behavior); other fields default to the
        // QuantizedGenerateConfig defaults (greedy).
        let temperature = std::env::var("APR_AGENT_TEMPERATURE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(request.temperature);
        let top_k = std::env::var("APR_AGENT_TOP_K").ok().and_then(|v| v.parse::<usize>().ok());
        let top_p = std::env::var("APR_AGENT_TOP_P").ok().and_then(|v| v.parse::<f32>().ok());
        let repeat_penalty =
            std::env::var("APR_AGENT_REPEAT_PENALTY").ok().and_then(|v| v.parse::<f32>().ok());
        let repeat_last_n =
            std::env::var("APR_AGENT_REPEAT_LAST_N").ok().and_then(|v| v.parse::<usize>().ok());
        let seed = std::env::var("APR_AGENT_SEED").ok().and_then(|v| v.parse::<u64>().ok());

        let mut body = serde_json::json!({
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": false,
        });
        if let Some(v) = top_k {
            body["top_k"] = serde_json::json!(v);
        }
        if let Some(v) = top_p {
            body["top_p"] = serde_json::json!(v);
        }
        if let Some(v) = repeat_penalty {
            body["repeat_penalty"] = serde_json::json!(v);
        }
        if let Some(v) = repeat_last_n {
            body["repeat_last_n"] = serde_json::json!(v);
        }
        if let Some(v) = seed {
            body["seed"] = serde_json::json!(v);
        }
        body
    }
}

/// Compute the startup-ready timeout in seconds for `apr serve`.
///
/// Resolution order:
/// 1. If `env_override` parses as a `u64`, use it verbatim (operator
///    override; minimum 1s clamp via `.max(1)`).
/// 2. Otherwise compute a size-aware default: 30s baseline + 1s per
///    500 MB above 2 GB. A 1 GB model gets 30s; a 4 GB model gets ~34s;
///    an 18 GB model gets ~62s; a 30 GB model gets ~86s.
/// 3. If model size is unknown (stat failed at launch), fall back to
///    30s baseline.
///
/// Always returns at least `MIN_TIMEOUT_S = 1` to avoid the pathological
/// 0-second budget case.
///
/// Extracted as a free function so the resolution logic is unit-testable
/// without spawning a subprocess. Called from
/// [`AprServeDriver::resolve_ready_timeout_secs`] with the live env.
#[must_use]
pub fn compute_ready_timeout_secs(
    model_size_bytes: Option<u64>,
    env_override: Option<&str>,
) -> u64 {
    const MIN_TIMEOUT_S: u64 = 1;
    const BASELINE_S: u64 = 30;
    const SIZE_FREE_BYTES: u64 = 2 * 1024 * 1024 * 1024; // 2 GB
    const BYTES_PER_EXTRA_SECOND: u64 = 500 * 1024 * 1024; // 500 MB

    if let Some(raw) = env_override {
        if let Ok(n) = raw.parse::<u64>() {
            return n.max(MIN_TIMEOUT_S);
        }
    }
    let Some(bytes) = model_size_bytes else {
        return BASELINE_S;
    };
    let extra_bytes = bytes.saturating_sub(SIZE_FREE_BYTES);
    let extra_secs = extra_bytes / BYTES_PER_EXTRA_SECOND;
    BASELINE_S.saturating_add(extra_secs)
}

#[async_trait]
impl LlmDriver for AprServeDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, AgentError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = self.build_openai_body(&request);

        // aprender#1789 follow-up: 120s default is too short for large MoE
        // models without KV cache (each token is full-prefill; a 256-token
        // generation at ~30B params can exceed 30 minutes wall). Empirical
        // evidence: paiml/claude-code-parity-apr Phase 6 bench against
        // Qwen3-Coder-30B-A3B saw every fixture die with "error sending
        // request" at exactly the 120s mark. Same root-cause class as
        // aprender#1782 (configurable + size-aware default).
        //
        // Override via `APR_AGENT_HTTP_TIMEOUT_S` env var. Default of 1800s
        // (30 min) matches the bench's per-turn-timeout ceiling + leaves
        // headroom for large MoE inference until M32d KV cache lands.
        let http_timeout_secs = std::env::var("APR_AGENT_HTTP_TIMEOUT_S")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1800);
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(http_timeout_secs))
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

/// Issue #1712: ask the kernel to SIGTERM the child when the parent dies.
///
/// On Linux/Unix this uses `PR_SET_PDEATHSIG` via `pre_exec` so the child
/// receives SIGTERM the instant its parent exits — whether the parent died
/// gracefully, was SIGKILLed by `timeout`, or was terminated by the OOM
/// killer. Without this, `apr serve` orphans hold ~3 GB RSS each.
///
/// A `getppid()==1` check immediately after `prctl` closes the small race
/// where the parent dies between fork and prctl (in which case the death
/// signal has already missed its window).
// PR_SET_PDEATHSIG / this `prctl` form are LINUX-ONLY — gate to `target_os = "linux"`, NOT the
// broader `unix` (which also matches macOS/BSD, where `libc::prctl` and `libc::PR_SET_PDEATHSIG`
// do not exist and the build fails: `cargo install aprender` was broken on macOS, PMAT-840).
#[cfg(target_os = "linux")]
#[allow(unsafe_code)] // pre_exec is unsafe-by-API; body uses only async-signal-safe calls
fn configure_parent_death_signal(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    // SAFETY: `prctl` and `getppid` are async-signal-safe; `pre_exec` runs
    // between fork and exec where only async-signal-safe calls are allowed.
    unsafe {
        cmd.pre_exec(|| {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM, 0, 0, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::getppid() == 1 {
                return Err(std::io::Error::other(
                    "parent died before PR_SET_PDEATHSIG took effect",
                ));
            }
            Ok(())
        });
    }
}

#[cfg(not(target_os = "linux"))]
fn configure_parent_death_signal(_cmd: &mut Command) {
    // Only Linux has PR_SET_PDEATHSIG (via prctl). macOS/BSD/Windows have no portable
    // equivalent — abrupt-parent-death orphan reaping is best-effort there (Drop still runs
    // on graceful exit). The point of this stub is to keep the crate building on those targets.
}

/// Environment variable that overrides `apr` binary resolution.
pub(crate) const APR_BIN_ENV: &str = "APR_BIN";

/// Resolve the `apr` binary that `apr code` launches as its inference backend.
///
/// #2384: this used to be a bare `which::which("apr")`. That made `apr code`
/// serve its answers from whatever `apr` happened to be first on `$PATH` —
/// a freshly installed 0.63.0 silently started a 0.60.0 backend — and it fails
/// outright for anyone who runs the binary by path without installing it on
/// `$PATH`. When the running executable *is* `apr`, launch **itself**.
///
/// `$PATH` is still the last resort, so a host process that is not `apr`
/// (tests, embedders of the `batuta` library) keeps working.
fn find_apr_binary() -> Result<PathBuf, AgentError> {
    resolve_apr_binary(std::env::var_os(APR_BIN_ENV), std::env::current_exe().ok(), || {
        which::which("apr").ok()
    })
    .ok_or_else(|| {
        AgentError::Driver(DriverError::InferenceFailed(
            "apr binary not found: this process is not `apr`, and no `apr` is on PATH. \
             Install: cargo install aprender"
                .into(),
        ))
    })
}

/// Pure core of [`find_apr_binary`], parameterised over the process state it
/// reads so the resolution policy is testable without mutating `$PATH` or the
/// environment of the running test process.
fn resolve_apr_binary<F>(
    override_var: Option<std::ffi::OsString>,
    current_exe: Option<PathBuf>,
    path_lookup: F,
) -> Option<PathBuf>
where
    F: FnOnce() -> Option<PathBuf>,
{
    if let Some(explicit) = override_var {
        if !explicit.is_empty() {
            return Some(PathBuf::from(explicit));
        }
    }
    if let Some(exe) = current_exe {
        // Exact stem match only: `apr-cli` and `batuta-<hash>` test harnesses
        // are not the `apr` CLI, and launching them with `serve run …` would
        // be worse than falling back to `$PATH`.
        if exe.file_stem().is_some_and(|stem| stem == "apr") {
            return Some(exe);
        }
    }
    path_lookup()
}

#[cfg(test)]
#[path = "apr_serve_tests.rs"]
mod tests;
