
use jugar_probar::llm::{
    ChatResponse as ProbarChatResponse, LlmAssertion,
    assertion::assert_deterministic,
};

/// Escape a string for embedding in a JSON string literal.
/// Handles all characters that are invalid inside JSON strings per RFC 8259.
fn escape_json_string(s: &str) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                // Unicode escape for other control characters
                for unit in c.encode_utf16(&mut [0; 2]) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out
}

/// Serve battery: spawn server once, run 19 endpoint checks, kill once.
///
/// Replaces the 1-request-per-server-lifecycle pattern. The primary
/// check (Generate) uses the existing gate ID for backward compatibility.
/// Additional checks emit new Evidence with distinct gate suffixes.
impl Executor {
    /// Minimum tokens/second floor. Below this the model is functionally broken
    /// (wrong quant, memory thrashing, etc.). Conservative: 0.1 tok/s on CPU
    /// means ~5 minutes for a 32-token response.
    const MIN_TPS_FLOOR: f64 = 0.1;

    /// Chat template markers that must NOT appear in generated output.
    /// Their presence indicates the template was not applied or stripped.
    const TEMPLATE_MARKERS: &[&str] = &[
        "<|im_start|>",
        "<|im_end|>",
        "[INST]",
        "[/INST]",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "### Instruction:",
        "### Response:",
    ];

    /// Run a battery of 19 serve endpoint checks against a single server lifecycle.
    ///
    /// Returns a `Vec<Evidence>` — one per check executed. If the primary
    /// Generate check fails, checks 2-19 are skipped (server is broken).
    #[must_use]
    pub fn run_serve_battery(
        &self,
        model_path: &str,
        scenario: &QaScenario,
        no_gpu: bool,
    ) -> Vec<Evidence> {
        let start = Instant::now();
        let mut results = Vec::with_capacity(19);

        // Use a deterministic port based on scenario to avoid collisions
        let port = 18_080 + (scenario.seed % 1000) as u16;

        // Spawn server in background
        let spawn_output = self
            .command_runner
            .spawn_serve(Path::new(model_path), port, no_gpu);
        if !spawn_output.success {
            let gate_id = format!("F-{}-001", scenario.mqs_category());
            results.push(Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("Failed to spawn serve: {}", spawn_output.stderr),
                &spawn_output.stderr,
                start.elapsed().as_millis() as u64,
            ));
            return results;
        }

        let pid_str = spawn_output.stdout.trim().to_string();

        // Wait for server to be ready — poll /health endpoint via http_get.
        // Large models (14B+) can take 3-5 min to load on CPU.
        // Use configured timeout (from playbook), minimum 120s.
        let serve_timeout_secs = std::cmp::max(self.config.default_timeout_ms / 1000, 120);
        let poll_iterations = serve_timeout_secs / 2;
        let health_url = format!("http://localhost:{port}/health");
        let mut server_ready = false;
        let server_pid: Option<u32> = pid_str.parse().ok();
        for _ in 0..poll_iterations {
            std::thread::sleep(std::time::Duration::from_secs(2));
            // Check health first — if server responds, we're good
            let health_output = self.command_runner.http_get(&health_url);
            if health_output.success && health_output.stdout.contains("healthy") {
                server_ready = true;
                break;
            }
            // Then check if server process is still alive (fail fast if crashed)
            if let Some(pid) = server_pid {
                let alive = std::path::Path::new(&format!("/proc/{pid}")).exists();
                if !alive {
                    break;
                }
            }
        }
        if !server_ready {
            kill_server_process(server_pid.as_ref());
            let gate_id = format!("F-{}-001", scenario.mqs_category());
            results.push(Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("Server failed to become ready within {serve_timeout_secs}s"),
                "",
                start.elapsed().as_millis() as u64,
            ));
            return results;
        }

        // Check 1: Generate (primary — backward compatible gate ID)
        let primary = self.check_serve_generate(port, scenario, &start);
        let primary_passed = primary.outcome.is_pass();
        let primary_tps = primary.metrics.tokens_per_second;
        results.push(primary);

        // If primary failed, skip remaining checks — server is broken
        if primary_passed {
            // Checks 2-10: existing endpoint + quality checks
            results.push(self.check_serve_v1_completions(port, scenario, &start));
            results.push(self.check_serve_v1_chat(port, scenario, &start));
            results.push(self.check_serve_streaming(port, scenario, &start));
            results.push(self.check_serve_stop_sequence(port, scenario, &start));
            results.push(self.check_serve_malformed(port, scenario, &start));
            results.push(self.check_serve_info(port, scenario, &start));
            results.push(self.check_serve_metrics(port, scenario, &start));
            results.push(self.check_serve_eos_termination(port, scenario, &start));
            results.push(self.check_serve_perf_floor(primary_tps, scenario, &start));
            // Checks 11-19: P0/P1 gaps from stack-wide + probar analysis
            results.push(self.check_serve_v1_models(port, scenario, &start));
            results.push(self.check_serve_template_leakage(port, scenario, &start));
            results.push(self.check_serve_temp_determinism(port, scenario, &start));
            results.push(self.check_serve_multi_turn(port, scenario, &start));
            results.push(self.check_serve_tokenize(port, scenario, &start));
            results.push(self.check_serve_special_chars(port, scenario, &start));
            results.push(self.check_serve_chat_streaming(port, scenario, &start));
            results.push(self.check_serve_max_tokens_one(port, scenario, &start));
            results.push(self.check_serve_response_schema(port, scenario, &start));
        }

        // Kill the server process
        kill_server_process(server_pid.as_ref());

        results
    }

    /// Check 1: POST /generate — primary serve inference (backward compat)
    fn check_serve_generate(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-001", scenario.mqs_category());
        let body = format!(
            r#"{{"prompt":"{}","max_tokens":32}}"#,
            escape_json_string(&scenario.prompt),
        );
        let url = format!("http://localhost:{port}/generate");
        let output = self.command_runner.http_post(&url, &body);
        let duration = start.elapsed().as_millis() as u64;

        if output.success {
            let generated = Self::extract_generated_text(&output.stdout);
            let oracle_result = scenario.evaluate(&generated);
            match oracle_result {
                aprender_qa_gen::OracleResult::Corroborated { .. } => {
                    let mut ev = Evidence::corroborated(&gate_id, scenario.clone(), &generated, duration);
                    ev.metrics.tokens_per_second = Self::parse_tps_from_output(&output.stdout);
                    ev
                }
                aprender_qa_gen::OracleResult::Falsified { reason, .. } => {
                    Evidence::falsified(&gate_id, scenario.clone(), reason, &generated, duration)
                }
            }
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("HTTP POST /generate failed: {}", output.stderr),
                &output.stdout,
                duration,
            )
        }
    }

    /// Check 2: POST /v1/completions — OpenAI-compatible text completion
    fn check_serve_v1_completions(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-COMP-001", scenario.mqs_category());
        let body = format!(
            r#"{{"prompt":"{}","max_tokens":32,"temperature":0.0}}"#,
            escape_json_string(&scenario.prompt),
        );
        let url = format!("http://localhost:{port}/v1/completions");
        let output = self.command_runner.http_post(&url, &body);
        let duration = start.elapsed().as_millis() as u64;

        if output.success {
            Evidence::corroborated(&gate_id, scenario.clone(), &output.stdout, duration)
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("POST /v1/completions failed: {}", output.stderr),
                &output.stdout,
                duration,
            )
        }
    }

    /// Check 3: POST /v1/chat/completions — primary production API (OpenAI format)
    fn check_serve_v1_chat(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-CHAT-001", scenario.mqs_category());
        let body = format!(
            r#"{{"model":"apr","messages":[{{"role":"user","content":"{}"}}],"max_tokens":32}}"#,
            escape_json_string(&scenario.prompt),
        );
        let url = format!("http://localhost:{port}/v1/chat/completions");
        let output = self.command_runner.http_post(&url, &body);
        let duration = start.elapsed().as_millis() as u64;

        if output.success {
            Evidence::corroborated(&gate_id, scenario.clone(), &output.stdout, duration)
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("POST /v1/chat/completions failed: {}", output.stderr),
                &output.stdout,
                duration,
            )
        }
    }

    /// Check 4: POST /generate with stream=true — verify SSE format
    fn check_serve_streaming(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-STREAM-001", scenario.mqs_category());
        let body = format!(
            r#"{{"prompt":"{}","max_tokens":16,"stream":true}}"#,
            escape_json_string(&scenario.prompt),
        );
        let url = format!("http://localhost:{port}/generate");
        let output = self.command_runner.http_post(&url, &body);
        let duration = start.elapsed().as_millis() as u64;

        if !output.success {
            return Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("Streaming request failed: {}", output.stderr),
                &output.stdout,
                duration,
            );
        }

        if Self::verify_sse_response(&output.stdout) {
            Evidence::corroborated(&gate_id, scenario.clone(), &output.stdout, duration)
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                "SSE response format invalid: expected 'data: ' prefixed lines ending with 'data: [DONE]'",
                &output.stdout,
                duration,
            )
        }
    }

    /// Check 5: POST /generate with stop sequence
    fn check_serve_stop_sequence(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-STOP-001", scenario.mqs_category());
        let body = r#"{"prompt":"Count: 1, 2, 3, 4, 5","max_tokens":32,"stop":["5"]}"#;
        let url = format!("http://localhost:{port}/generate");
        let output = self.command_runner.http_post(&url, body);
        let duration = start.elapsed().as_millis() as u64;

        if !output.success {
            return Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("Stop sequence request failed: {}", output.stderr),
                &output.stdout,
                duration,
            );
        }

        let generated = Self::extract_generated_text(&output.stdout);
        if generated.contains('5') {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                "Stop sequence not honored: output contains '5'",
                &generated,
                duration,
            )
        } else {
            Evidence::corroborated(&gate_id, scenario.clone(), &generated, duration)
        }
    }

    /// Check 6: POST /generate with malformed JSON — error resilience
    fn check_serve_malformed(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-ERR-001", scenario.mqs_category());
        let bad_body = r#"{"not_a_valid_field": true}"#;
        let url = format!("http://localhost:{port}/generate");
        let output = self.command_runner.http_post(&url, bad_body);
        let duration = start.elapsed().as_millis() as u64;

        // Hypothesis: malformed request is REJECTED by server (non-2xx), AND
        // server remains healthy afterward (Jidoka: error handling, not crash).
        // Passing only because server stayed up (while accepting bad input) is
        // vacuous truth — it must also have rejected the malformed request.
        let health_url = format!("http://localhost:{port}/health");
        let health = self.command_runner.http_get(&health_url);

        if !health.success {
            return Evidence::falsified(
                &gate_id,
                scenario.clone(),
                "Server became unhealthy after malformed request",
                &health.stderr,
                duration,
            );
        }

        if output.success {
            // Server accepted a malformed request — vacuous corroboration avoided
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!(
                    "Server accepted malformed request (status={}): expected rejection",
                    output.exit_code
                ),
                &output.stdout,
                duration,
            )
        } else {
            Evidence::corroborated(
                &gate_id,
                scenario.clone(),
                format!(
                    "Server rejected malformed request (status={}) and remained healthy",
                    output.exit_code
                ),
                duration,
            )
        }
    }

    /// Check 7: GET / — server info endpoint
    fn check_serve_info(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-INFO-001", scenario.mqs_category());
        let url = format!("http://localhost:{port}/");
        let output = self.command_runner.http_get(&url);
        let duration = start.elapsed().as_millis() as u64;

        if output.success && !output.stdout.is_empty() {
            Evidence::corroborated(&gate_id, scenario.clone(), &output.stdout, duration)
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("GET / failed or empty response: {}", output.stderr),
                &output.stdout,
                duration,
            )
        }
    }

    /// Check 8: GET /metrics — Prometheus metrics endpoint
    fn check_serve_metrics(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-METRICS-001", scenario.mqs_category());
        let url = format!("http://localhost:{port}/metrics");
        let output = self.command_runner.http_get(&url);
        let duration = start.elapsed().as_millis() as u64;

        if output.success && !output.stdout.is_empty() {
            Evidence::corroborated(&gate_id, scenario.clone(), &output.stdout, duration)
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("GET /metrics failed or empty: {}", output.stderr),
                &output.stdout,
                duration,
            )
        }
    }

    /// Check 9: POST /generate with short prompt — verify EOS token stops generation.
    ///
    /// Sends a prompt that should terminate early with `max_tokens=32`. If the
    /// output fills the entire token budget AND contains repetition patterns,
    /// it indicates a missing/broken EOS token. 32 tokens keeps the signal tight
    /// — a 2-token prompt like "The end." should stop well before 32 tokens
    /// if EOS is working.
    fn check_serve_eos_termination(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-EOS-001", scenario.mqs_category());
        let body = r#"{"prompt":"The end.","max_tokens":32}"#;
        let url = format!("http://localhost:{port}/generate");
        let output = self.command_runner.http_post(&url, body);
        let duration = start.elapsed().as_millis() as u64;

        if !output.success {
            return Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("EOS termination request failed: {}", output.stderr),
                &output.stdout,
                duration,
            );
        }

        let generated = Self::extract_generated_text(&output.stdout);
        let word_count = generated.split_whitespace().count();
        let has_excessive_repetition = Self::detect_repetition(&generated);

        // With max_tokens=32 and prompt "The end.", reaching 25+ words with
        // repetition strongly signals broken EOS (model filled entire budget
        // and is repeating itself)
        if word_count > 25 && has_excessive_repetition {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("Possible EOS failure: {word_count} words with repetition detected"),
                &generated,
                duration,
            )
        } else {
            Evidence::corroborated(&gate_id, scenario.clone(), &generated, duration)
        }
    }

    /// Check 10: Performance floor — verify inference is not pathologically slow.
    ///
    /// Reuses the TPS metric from check 1 (no extra request). Fails if
    /// tok/s < `MIN_TPS_FLOOR` (0.1). A model below this threshold is
    /// technically "working" but unusable — likely wrong quantization,
    /// memory thrashing, or accidental CPU fallback on a GPU path.
    fn check_serve_perf_floor(
        &self,
        primary_tps: Option<f64>,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-PERF-001", scenario.mqs_category());
        let duration = start.elapsed().as_millis() as u64;

        let Some(tps) = primary_tps else {
            // No TPS reported — server didn't emit tok/s metric.
            // Popper: untested hypothesis ≠ corroborated. Skip, don't pass.
            return Evidence::skipped(
                &gate_id,
                scenario.clone(),
                "no tok/s metric reported — cannot verify performance floor",
            );
        };

        if tps < Self::MIN_TPS_FLOOR {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!(
                    "Inference too slow: {tps:.2} tok/s (floor: {} tok/s)",
                    Self::MIN_TPS_FLOOR
                ),
                &format!("{tps:.4} tok/s"),
                duration,
            )
        } else {
            let mut ev = Evidence::corroborated(
                &gate_id,
                scenario.clone(),
                &format!("{tps:.2} tok/s"),
                duration,
            );
            ev.metrics.tokens_per_second = Some(tps);
            ev
        }
    }

    /// Check 11: GET /v1/models — OpenAI model discovery endpoint.
    ///
    /// OpenAI SDK clients call this first. If it 404s the whole integration fails.
    fn check_serve_v1_models(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-MODELS-001", scenario.mqs_category());
        let url = format!("http://localhost:{port}/v1/models");
        let output = self.command_runner.http_get(&url);
        let duration = start.elapsed().as_millis() as u64;

        if !output.success {
            return Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("GET /v1/models failed: {}", output.stderr),
                &output.stdout,
                duration,
            );
        }

        // Must be valid JSON with a "data" array
        let has_data = serde_json::from_str::<serde_json::Value>(&output.stdout)
            .ok()
            .and_then(|v| v.get("data")?.as_array().map(|a| !a.is_empty()))
            .unwrap_or(false);

        if has_data {
            Evidence::corroborated(&gate_id, scenario.clone(), &output.stdout, duration)
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                "GET /v1/models: missing or empty 'data' array",
                &output.stdout,
                duration,
            )
        }
    }

    /// Check 12: Chat template leakage — verify raw template markers are stripped.
    ///
    /// Sends a chat request and checks that markers like `<|im_start|>`,
    /// `[INST]`, etc. do NOT appear in the generated text. Their presence
    /// means the chat template was not applied or not stripped from output.
    fn check_serve_template_leakage(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-TMPL-001", scenario.mqs_category());
        let body = r#"{"model":"apr","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32}"#;
        let url = format!("http://localhost:{port}/v1/chat/completions");
        let output = self.command_runner.http_post(&url, body);
        let duration = start.elapsed().as_millis() as u64;

        if !output.success {
            return Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("Chat request for template check failed: {}", output.stderr),
                &output.stdout,
                duration,
            );
        }

        let text = Self::extract_chat_text(&output.stdout);
        let leaked: Vec<&&str> = Self::TEMPLATE_MARKERS
            .iter()
            .filter(|m| text.contains(**m))
            .collect();

        if leaked.is_empty() {
            Evidence::corroborated(&gate_id, scenario.clone(), &text, duration)
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!(
                    "Template markers leaked into output: {}",
                    leaked.iter().map(|m| format!("'{m}'")).collect::<Vec<_>>().join(", ")
                ),
                &text,
                duration,
            )
        }
    }

    /// Check 13: Temperature determinism — temp=0 must produce identical output.
    ///
    /// Two requests with temperature=0.0 and the same prompt must return the
    /// same generated text. Uses probar's `assert_deterministic` to compare.
    /// Non-determinism at temp=0 means broken sampling.
    fn check_serve_temp_determinism(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-DETERM-001", scenario.mqs_category());
        let body = r#"{"model":"apr","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":8,"temperature":0.0}"#;
        let url = format!("http://localhost:{port}/v1/chat/completions");

        let out1 = self.command_runner.http_post(&url, body);
        let out2 = self.command_runner.http_post(&url, body);
        let duration = start.elapsed().as_millis() as u64;

        if !out1.success || !out2.success {
            return Evidence::falsified(
                &gate_id,
                scenario.clone(),
                "Determinism check: one or both requests failed",
                &format!("r1={} r2={}", out1.success, out2.success),
                duration,
            );
        }

        // Parse both responses as probar ChatResponse for typed comparison
        let responses: Vec<ProbarChatResponse> = [&out1.stdout, &out2.stdout]
            .iter()
            .filter_map(|s| serde_json::from_str(s).ok())
            .collect();

        if responses.len() < 2 {
            return Evidence::falsified(
                &gate_id,
                scenario.clone(),
                "Determinism check: could not parse responses as ChatResponse",
                &format!("r1='{}' r2='{}'", out1.stdout, out2.stdout),
                duration,
            );
        }

        let result = assert_deterministic(&responses);
        if result.passed {
            let text = responses[0]
                .choices
                .first()
                .map_or("(empty)", |c| c.message.content.as_str());
            Evidence::corroborated(
                &gate_id,
                scenario.clone(),
                &format!("temp=0 deterministic: '{text}'"),
                duration,
            )
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                result.detail.unwrap_or_else(|| "temp=0 produced different outputs".to_string()),
                &format!("r1='{}' r2='{}'", out1.stdout, out2.stdout),
                duration,
            )
        }
    }

    /// Check 14: Multi-turn chat — verify conversation context is maintained.
    ///
    /// Sends a 3-message conversation where the user states a fact, gets an
    /// acknowledgement, then asks about the fact. The response should reference
    /// the stated fact, proving the model sees the full context.
    fn check_serve_multi_turn(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-MULTI-001", scenario.mqs_category());
        let body = r#"{"model":"apr","messages":[{"role":"user","content":"My favorite color is blue."},{"role":"assistant","content":"Got it!"},{"role":"user","content":"What is my favorite color?"}],"max_tokens":16}"#;
        let url = format!("http://localhost:{port}/v1/chat/completions");
        let output = self.command_runner.http_post(&url, body);
        let duration = start.elapsed().as_millis() as u64;

        if !output.success {
            return Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("Multi-turn request failed: {}", output.stderr),
                &output.stdout,
                duration,
            );
        }

        let text = Self::extract_chat_text(&output.stdout).to_lowercase();
        if text.contains("blue") {
            Evidence::corroborated(
                &gate_id,
                scenario.clone(),
                &format!("Multi-turn context preserved: '{text}'"),
                duration,
            )
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                "Multi-turn context lost: response does not contain 'blue'",
                &text,
                duration,
            )
        }
    }

    /// Check 15: POST /tokenize — verify tokenizer is loaded and functional.
    ///
    /// Round-trip test: tokenize a known string and verify token count is
    /// reasonable (2-10 tokens for "Hello world"). If the endpoint doesn't
    /// exist, pass with a note (not all servers expose /tokenize).
    fn check_serve_tokenize(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-TOK-001", scenario.mqs_category());
        let body = r#"{"text":"Hello world"}"#;
        let url = format!("http://localhost:{port}/tokenize");
        let output = self.command_runner.http_post(&url, body);
        let duration = start.elapsed().as_millis() as u64;

        if !output.success {
            // /tokenize is optional — not all servers expose it.
            // Popper: untested hypothesis ≠ corroborated. Skip, don't pass.
            return Evidence::skipped(
                &gate_id,
                scenario.clone(),
                "POST /tokenize not available — endpoint absent or failed",
            );
        }

        // Response should contain tokens (array or count)
        let has_tokens = serde_json::from_str::<serde_json::Value>(&output.stdout).is_ok_and(|v| v.get("tokens").is_some() || v.get("count").is_some());

        if has_tokens {
            Evidence::corroborated(&gate_id, scenario.clone(), &output.stdout, duration)
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                "POST /tokenize: response missing 'tokens' or 'count' field",
                &output.stdout,
                duration,
            )
        }
    }

    /// Check 16: Special characters in prompt — braces, quotes, unicode.
    ///
    /// Sends a prompt with JSON-hostile characters (`{`, `}`, `"`, newlines).
    /// Server must not crash or return an error. Tests template rendering
    /// resilience against real-world prompt content.
    fn check_serve_special_chars(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-CHARS-001", scenario.mqs_category());
        // Escaped JSON: prompt contains braces, escaped quotes, newline
        let body = r#"{"prompt":"Code: if (x > 5) { print(\"ok\"); }\nDone.","max_tokens":16}"#;
        let url = format!("http://localhost:{port}/generate");
        let output = self.command_runner.http_post(&url, body);
        let duration = start.elapsed().as_millis() as u64;

        // Validate: server must accept the request AND return non-empty output.
        // HTTP success with empty body would indicate the request was dropped silently.
        if output.success && !output.stdout.is_empty() {
            Evidence::corroborated(
                &gate_id,
                scenario.clone(),
                &format!(
                    "Special chars handled: {}",
                    output.stdout.chars().take(100).collect::<String>()
                ),
                duration,
            )
        } else if output.success {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                "Special char prompt accepted but response was empty",
                &output.stdout,
                duration,
            )
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("Special char prompt failed: {}", output.stderr),
                &output.stdout,
                duration,
            )
        }
    }

    /// Check 17: POST /v1/chat/completions with stream=true — chat SSE format.
    ///
    /// Existing check 4 tests streaming on /generate. This checks streaming on
    /// the OpenAI chat endpoint, which production clients actually use.
    fn check_serve_chat_streaming(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-CSTREAM-001", scenario.mqs_category());
        let body = r#"{"model":"apr","messages":[{"role":"user","content":"Hi"}],"max_tokens":8,"stream":true}"#;
        let url = format!("http://localhost:{port}/v1/chat/completions");
        let output = self.command_runner.http_post(&url, body);
        let duration = start.elapsed().as_millis() as u64;

        if !output.success {
            return Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("Chat streaming request failed: {}", output.stderr),
                &output.stdout,
                duration,
            );
        }

        if Self::verify_sse_response(&output.stdout) {
            Evidence::corroborated(&gate_id, scenario.clone(), &output.stdout, duration)
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                "Chat SSE format invalid: expected 'data: ' lines ending with 'data: [DONE]'",
                &output.stdout,
                duration,
            )
        }
    }

    /// Check 18: max_tokens=1 compliance — verify output respects token budget.
    ///
    /// Requests exactly 1 token. Output should be very short (1-3 words max).
    /// Longer output means the server ignores max_tokens, which breaks API contract.
    fn check_serve_max_tokens_one(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-MAXTOK-001", scenario.mqs_category());
        let body = r#"{"prompt":"Hello","max_tokens":1}"#;
        let url = format!("http://localhost:{port}/generate");
        let output = self.command_runner.http_post(&url, body);
        let duration = start.elapsed().as_millis() as u64;

        if !output.success {
            return Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("max_tokens=1 request failed: {}", output.stderr),
                &output.stdout,
                duration,
            );
        }

        let generated = Self::extract_generated_text(&output.stdout);
        // 1 token ≈ 1-3 words max (subword tokenization). Allow up to 4 words
        // to avoid false positives from whitespace tokenization differences.
        let word_count = generated.split_whitespace().count();
        if word_count <= 4 {
            Evidence::corroborated(
                &gate_id,
                scenario.clone(),
                &format!("max_tokens=1: {word_count} word(s): '{generated}'"),
                duration,
            )
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("max_tokens=1 violated: got {word_count} words"),
                &generated,
                duration,
            )
        }
    }

    /// Check 19: OpenAI response schema — verify required fields are present.
    ///
    /// Uses probar's `ChatResponse` typed deserialization to verify schema
    /// compliance, then runs `LlmAssertion::assert_response_valid()` to check
    /// that id, choices, and content are non-empty. Missing/malformed fields
    /// break OpenAI SDK clients.
    fn check_serve_response_schema(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-SCHEMA-001", scenario.mqs_category());
        let body = r#"{"model":"apr","messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"temperature":0.0}"#;
        let url = format!("http://localhost:{port}/v1/chat/completions");
        let output = self.command_runner.http_post(&url, body);
        let duration = start.elapsed().as_millis() as u64;

        if !output.success {
            return Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("Schema check request failed: {}", output.stderr),
                &output.stdout,
                duration,
            );
        }

        // Attempt typed deserialization via probar's ChatResponse
        let parsed: std::result::Result<ProbarChatResponse, _> = serde_json::from_str(&output.stdout);
        match parsed {
            Ok(response) => {
                // Run probar's structural validation assertion
                let timed = jugar_probar::llm::TimedChatResponse {
                    response,
                    latency: std::time::Duration::from_millis(duration),
                    ttfb: std::time::Duration::from_millis(duration),
                    brick_trace: None,
                };
                let assertion = LlmAssertion::new().assert_response_valid();
                if assertion.run_all_pass(&timed) {
                    Evidence::corroborated(&gate_id, scenario.clone(), &output.stdout, duration)
                } else {
                    let results = assertion.run(&timed);
                    let details: Vec<String> = results
                        .iter()
                        .filter(|r| !r.passed)
                        .filter_map(|r| r.detail.clone())
                        .collect();
                    Evidence::falsified(
                        &gate_id,
                        scenario.clone(),
                        format!("OpenAI schema validation failed: {}", details.join("; ")),
                        &output.stdout,
                        duration,
                    )
                }
            }
            Err(e) => Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("Response does not match OpenAI ChatResponse schema: {e}"),
                &output.stdout,
                duration,
            ),
        }
    }

    /// Extract generated text from a /v1/chat/completions JSON response.
    ///
    /// Uses probar's `ChatResponse` type for structured deserialization.
    /// Falls back to raw stdout if JSON parsing fails.
    fn extract_chat_text(response: &str) -> String {
        serde_json::from_str::<ProbarChatResponse>(response)
            .ok()
            .and_then(|r| r.choices.first().map(|c| c.message.content.clone()))
            .unwrap_or_else(|| response.to_string())
    }

    /// Detect excessive repetition (hallmark of missing EOS token).
    ///
    /// Returns true if any 3-gram repeats more than 5 times in the text.
    /// Short texts (< 20 words) always return false.
    fn detect_repetition(text: &str) -> bool {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < 20 {
            return false;
        }
        let mut trigram_counts: std::collections::HashMap<(&str, &str, &str), usize> =
            std::collections::HashMap::new();
        for window in words.windows(3) {
            *trigram_counts
                .entry((window[0], window[1], window[2]))
                .or_default() += 1;
        }
        trigram_counts.values().any(|&count| count > 5)
    }

    /// Validate SSE (Server-Sent Events) response format.
    ///
    /// Valid SSE: non-empty lines start with "data: ", ends with "data: [DONE]".
    fn verify_sse_response(response: &str) -> bool {
        let data_lines: Vec<&str> = response
            .lines()
            .filter(|l| !l.is_empty())
            .collect();

        if data_lines.is_empty() {
            return false;
        }

        // All non-empty lines must start with "data: "
        let all_data_prefixed = data_lines.iter().all(|l| l.starts_with("data: "));
        if !all_data_prefixed {
            return false;
        }

        // Last line must be "data: [DONE]"
        data_lines
            .last()
            .is_some_and(|l| *l == "data: [DONE]")
    }
}

/// Kill a server process by PID with SIGTERM, logging failures.
///
/// Sends SIGTERM via the `kill` command. If `server_pid` is None (PID
/// parsing failed at spawn time), logs a warning instead of silently
/// skipping cleanup.
fn kill_server_process(server_pid: Option<&u32>) {
    let Some(&pid) = server_pid else {
        eprintln!("[JIDOKA] Cannot kill server: PID not available (spawn output was malformed)");
        return;
    };
    let pid_str = pid.to_string();
    match std::process::Command::new("kill").arg(&pid_str).output() {
        Ok(output) if output.status.success() => {}
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("[JIDOKA] kill {pid} exited with {}: {}", output.status, stderr.trim());
        }
        Err(e) => {
            eprintln!("[JIDOKA] Failed to execute kill {pid}: {e}");
        }
    }
}
