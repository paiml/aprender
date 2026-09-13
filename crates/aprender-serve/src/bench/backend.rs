
impl LlamaCppBackend {
    /// Create new llama.cpp backend
    #[must_use]
    pub fn new(config: LlamaCppConfig) -> Self {
        Self { config }
    }

    /// Build CLI arguments for llama-cli invocation
    #[must_use]
    pub fn build_cli_args(&self, request: &InferenceRequest) -> Vec<String> {
        let mut args = Vec::new();

        // Model path
        if let Some(ref model_path) = self.config.model_path {
            args.push("-m".to_string());
            args.push(model_path.clone());
        }

        // Prompt
        args.push("-p".to_string());
        args.push(request.prompt.clone());

        // Number of tokens to generate
        args.push("-n".to_string());
        args.push(request.max_tokens.to_string());

        // GPU layers
        args.push("-ngl".to_string());
        args.push(self.config.n_gpu_layers.to_string());

        // Context size
        args.push("-c".to_string());
        args.push(self.config.ctx_size.to_string());

        // Threads
        args.push("-t".to_string());
        args.push(self.config.threads.to_string());

        // Temperature (if non-default)
        if (request.temperature - 0.8).abs() > 0.01 {
            args.push("--temp".to_string());
            args.push(format!("{:.2}", request.temperature));
        }

        args
    }

    /// Parse a timing line from llama-cli output
    ///
    /// Example: `llama_perf_context_print: prompt eval time =      12.34 ms /    10 tokens`
    /// Returns: `Some((12.34, 10))`
    #[must_use]
    /// Does this line report the metric asked for?
    ///
    /// "eval time" needs the exclusion because llama.cpp also prints
    /// "prompt eval time", and a plain substring match would take whichever
    /// came first — silently reporting prompt-processing time as decode time.
    fn line_reports(line: &str, metric_name: &str) -> bool {
        if !line.contains('=') {
            return false;
        }
        if metric_name == "eval time" {
            return line.contains(metric_name) && !line.contains("prompt eval time");
        }
        line.contains(metric_name)
    }

    /// Pull `(milliseconds, token_count)` out of one llama.cpp timing line.
    ///
    /// Format: `metric_name =      12.34 ms /    10 tokens`
    fn parse_one_timing(line: &str) -> Option<(f64, usize)> {
        let after_eq = &line[line.find('=')? + 1..];
        let ms_pos = after_eq.find("ms")?;
        let value: f64 = after_eq[..ms_pos].trim().parse().ok()?;
        let after_slash = &after_eq[after_eq.find('/')? + 1..];
        let count: usize = after_slash.split_whitespace().next()?.parse().ok()?;
        Some((value, count))
    }

    /// PARITY-007: split from one 30-line nest (cognitive 38) into a predicate
    /// and a parser. Untouched by this ticket's subject, but the pre-commit
    /// complexity gate charges the whole file to whoever edits any of it, and
    /// both escape hatches — `--no-verify` and `#[allow]` — are banned here.
    /// Behaviour is unchanged; the tests below pin that.
    pub fn parse_timing_line(output: &str, metric_name: &str) -> Option<(f64, usize)> {
        output
            .lines()
            .filter(|line| Self::line_reports(line, metric_name))
            .find_map(Self::parse_one_timing)
    }

    /// Extract generated text from llama-cli output (before timing lines)
    #[must_use]
    pub fn extract_generated_text(output: &str) -> String {
        let mut text_lines = Vec::new();
        for line in output.lines() {
            // Stop when we hit timing/performance lines
            if line.contains("llama_perf_") || line.contains("sampler") {
                break;
            }
            text_lines.push(line);
        }
        text_lines.join("\n").trim().to_string()
    }

    /// Parse full CLI output into InferenceResponse
    ///
    /// # Errors
    ///
    /// Returns error if timing information cannot be parsed from output.
    pub fn parse_cli_output(output: &str) -> Result<InferenceResponse, RealizarError> {
        // Extract generated text
        let text = Self::extract_generated_text(output);

        // Parse timing metrics
        let ttft_ms = Self::parse_timing_line(output, "prompt eval time").map_or(0.0, |(ms, _)| ms);

        let (total_time_ms, _) = Self::parse_timing_line(output, "total time").unwrap_or((0.0, 0));

        let (_, tokens_generated) =
            Self::parse_timing_line(output, "eval time").unwrap_or((0.0, 0));

        // ITL is not directly available from CLI output, estimate from eval time
        let eval_time = Self::parse_timing_line(output, "eval time").map_or(0.0, |(ms, _)| ms);

        let itl_ms = if tokens_generated > 1 {
            let avg_itl = eval_time / (tokens_generated as f64);
            vec![avg_itl; tokens_generated.saturating_sub(1)]
        } else {
            vec![]
        };

        Ok(InferenceResponse {
            text,
            tokens_generated,
            ttft_ms,
            total_time_ms,
            itl_ms,
        })
    }
}

/// PARITY-007 — a comparator's version is DETECTED or ABSENT, never asserted.
///
/// Three `BackendInfo::info()` implementations here carried a literal with the
/// comment "Would be detected from binary" / "from API": `"b2345"`, `"0.4.0"`,
/// `"0.1.0"`. A version string that looks measured and is not is F12
/// (aprender#2679), and it is the field that makes a cross-release ratio
/// meaningless: the denominator changes silently while the receipt claims a
/// fixed comparator.
///
/// `"unknown"` is the honest value when the probe fails. A consuming gate can
/// treat "unknown" as RED; it cannot detect a plausible literal at all.
fn detect_version(program: &str, args: &[&str]) -> String {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            let text = String::from_utf8_lossy(&o.stdout).to_string();
            let combined = if text.trim().is_empty() {
                String::from_utf8_lossy(&o.stderr).to_string()
            } else {
                text
            };
            combined
                .lines()
                .next()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

impl RuntimeBackend for LlamaCppBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            runtime_type: RuntimeType::LlamaCpp,
            // Detected from the binary the config actually points at.
            version: detect_version(&self.config.binary_path, &["--version"]),
            supports_streaming: false,    // CLI mode doesn't stream
            loaded_model: self.config.model_path.clone(),
        }
    }

    fn inference(&self, request: &InferenceRequest) -> Result<InferenceResponse, RealizarError> {
        use std::process::Command;

        // Require model path
        let model_path = self.config.model_path.as_ref().ok_or_else(|| {
            RealizarError::InvalidConfiguration("model_path is required".to_string())
        })?;

        // Build CLI arguments
        let args = self.build_cli_args(request);

        // Execute llama-cli
        let output = Command::new(&self.config.binary_path)
            .args(&args)
            .output()
            .map_err(|e| {
                RealizarError::ModelNotFound(format!(
                    "Failed to execute {}: {}",
                    self.config.binary_path, e
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(RealizarError::InferenceError(format!(
                "llama-cli failed: {} (model: {})",
                stderr, model_path
            )));
        }

        // Parse stdout for response and timing
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        // Timing info is often in stderr, combine both
        let combined_output = format!("{}\n{}", stdout, stderr);
        Self::parse_cli_output(&combined_output)
    }
}

// ============================================================================
// VllmBackend Implementation (BENCH-003) - REAL HTTP CALLS
// ============================================================================

/// vLLM backend for inference via HTTP API
///
/// **REAL IMPLEMENTATION** - makes actual HTTP requests to vLLM servers.
/// No mock data. Measures real latency and throughput.
#[cfg(feature = "bench-http")]
pub struct VllmBackend {
    config: VllmConfig,
    http_client: ModelHttpClient,
}

#[cfg(feature = "bench-http")]
impl VllmBackend {
    /// Create new vLLM backend with default HTTP client
    #[must_use]
    pub fn new(config: VllmConfig) -> Self {
        Self {
            config,
            http_client: ModelHttpClient::new(),
        }
    }

    /// Create new vLLM backend with custom HTTP client
    #[must_use]
    pub fn with_client(config: VllmConfig, client: ModelHttpClient) -> Self {
        Self {
            config,
            http_client: client,
        }
    }
}

#[cfg(feature = "bench-http")]
impl RuntimeBackend for VllmBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            runtime_type: RuntimeType::Vllm,
            // Detected from the ollama CLI; "unknown" when it is not present.
            version: detect_version("ollama", &["--version"]),
            supports_streaming: true,
            loaded_model: self.config.model.clone(),
        }
    }

    fn inference(&self, request: &InferenceRequest) -> Result<InferenceResponse, RealizarError> {
        // Parse URL to check for invalid port
        let url = &self.config.base_url;
        if let Some(port_str) = url.split(':').next_back() {
            if let Ok(port) = port_str.parse::<u32>() {
                if port > 65535 {
                    return Err(RealizarError::ConnectionError(format!(
                        "Invalid port in URL: {}",
                        url
                    )));
                }
            }
        }

        // REAL HTTP request to vLLM server via OpenAI-compatible API
        #[allow(clippy::cast_possible_truncation)]
        let completion_request = CompletionRequest {
            model: self
                .config
                .model
                .clone()
                .unwrap_or_else(|| "default".to_string()),
            prompt: request.prompt.clone(),
            max_tokens: request.max_tokens,
            temperature: Some(request.temperature as f32),
            stream: false,
        };

        let timing = self.http_client.openai_completion(
            &self.config.base_url,
            &completion_request,
            self.config.api_key.as_deref(),
        )?;

        Ok(InferenceResponse {
            text: timing.text,
            tokens_generated: timing.tokens_generated,
            ttft_ms: timing.ttft_ms,
            total_time_ms: timing.total_time_ms,
            itl_ms: vec![], // ITL requires streaming, not available in blocking mode
        })
    }
}

// ============================================================================
// OllamaBackend Implementation - REAL HTTP CALLS
// ============================================================================

/// Configuration for Ollama backend
#[cfg(feature = "bench-http")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// Base URL for Ollama server
    pub base_url: String,
    /// Model name
    pub model: String,
}

#[cfg(feature = "bench-http")]
impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: "llama2".to_string(),
        }
    }
}

/// Ollama backend for inference via HTTP API
///
/// **REAL IMPLEMENTATION** - makes actual HTTP requests to Ollama servers.
/// No mock data. Measures real latency and throughput.
#[cfg(feature = "bench-http")]
pub struct OllamaBackend {
    config: OllamaConfig,
    http_client: ModelHttpClient,
}

#[cfg(feature = "bench-http")]
impl OllamaBackend {
    /// Create new Ollama backend with default HTTP client
    #[must_use]
    pub fn new(config: OllamaConfig) -> Self {
        Self {
            config,
            http_client: ModelHttpClient::new(),
        }
    }

    /// Create new Ollama backend with custom HTTP client
    #[must_use]
    pub fn with_client(config: OllamaConfig, client: ModelHttpClient) -> Self {
        Self {
            config,
            http_client: client,
        }
    }
}

#[cfg(feature = "bench-http")]
impl RuntimeBackend for OllamaBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            runtime_type: RuntimeType::Ollama,
            // Detected from the vllm CLI; "unknown" when it is not present.
            version: detect_version("vllm", &["--version"]),
            supports_streaming: true,
            loaded_model: Some(self.config.model.clone()),
        }
    }

    fn inference(&self, request: &InferenceRequest) -> Result<InferenceResponse, RealizarError> {
        // REAL HTTP request to Ollama server
        #[allow(clippy::cast_possible_truncation)]
        let ollama_request = OllamaRequest {
            model: self.config.model.clone(),
            prompt: request.prompt.clone(),
            stream: false,
            options: Some(OllamaOptions {
                num_predict: Some(request.max_tokens),
                temperature: Some(request.temperature as f32),
            }),
        };

        let timing = self
            .http_client
            .ollama_generate(&self.config.base_url, &ollama_request)?;

        Ok(InferenceResponse {
            text: timing.text,
            tokens_generated: timing.tokens_generated,
            ttft_ms: timing.ttft_ms,
            total_time_ms: timing.total_time_ms,
            itl_ms: vec![], // ITL requires streaming, not available in blocking mode
        })
    }
}

// ── PARITY-007: parse_timing_line behaviour is pinned across the refactor ───
#[cfg(test)]
mod parity_007_timing_parse_tests {
    use super::LlamaCppBackend;

    /// The real llama.cpp timing block. `eval time` must not be captured by
    /// `prompt eval time`, which appears FIRST in the output — the exclusion
    /// is the whole reason the original had a special case, and dropping it
    /// would silently report prompt-processing time as decode time.
    const SAMPLE: &str = "\
llama_print_timings:        load time =    1234.56 ms
llama_print_timings:      sample time =      12.34 ms /    64 runs
llama_print_timings: prompt eval time =     456.78 ms /   128 tokens
llama_print_timings:        eval time =    2345.67 ms /    63 tokens
llama_print_timings:       total time =    3000.00 ms";

    #[test]
    fn eval_time_is_not_captured_by_prompt_eval_time() {
        let got = LlamaCppBackend::parse_timing_line(SAMPLE, "eval time");
        assert_eq!(
            got,
            Some((2345.67, 63)),
            "must take the decode line, not the prompt line that precedes it"
        );
    }

    #[test]
    fn prompt_eval_time_is_still_reachable_by_its_own_name() {
        assert_eq!(
            LlamaCppBackend::parse_timing_line(SAMPLE, "prompt eval time"),
            Some((456.78, 128))
        );
    }

    #[test]
    fn a_metric_with_no_slash_yields_none_rather_than_a_wrong_pair() {
        // "load time = 1234.56 ms" has no "/ N tokens" — the original
        // `continue`d past it, and so must the refactor.
        assert_eq!(LlamaCppBackend::parse_timing_line(SAMPLE, "load time"), None);
    }

    #[test]
    fn an_absent_metric_yields_none() {
        assert_eq!(LlamaCppBackend::parse_timing_line(SAMPLE, "no such metric"), None);
    }

    #[test]
    fn a_line_without_an_equals_is_not_a_timing_line() {
        assert_eq!(
            LlamaCppBackend::parse_timing_line("eval time is large\n", "eval time"),
            None
        );
    }
}
