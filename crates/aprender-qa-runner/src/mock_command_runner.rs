
/// Mock command runner for testing
///
/// This struct uses many boolean flags intentionally - each flag controls
/// an independent success/failure behavior for testing different scenarios.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct MockCommandRunner {
    /// Default response for inference
    pub inference_response: String,
    /// Whether inference should succeed
    pub inference_success: bool,
    /// Default response for convert
    pub convert_success: bool,
    /// Tokens per second to report
    pub tps: f64,
    /// Simulate a crash (negative exit code)
    pub crash: bool,
    /// Custom stderr message for inference
    pub inference_stderr: Option<String>,
    /// Simulate profile_ci feature not available
    pub profile_ci_unavailable: bool,
    /// Custom stderr for profile_ci
    pub profile_ci_stderr: Option<String>,
    /// Whether inspect should fail
    pub inspect_success: bool,
    /// Whether validate should fail
    pub validate_success: bool,
    /// Whether bench should fail
    pub bench_success: bool,
    /// Whether check should fail
    pub check_success: bool,
    /// Whether profile should fail
    pub profile_success: bool,
    /// Whether diff_tensors should fail
    pub diff_tensors_success: bool,
    /// Whether compare_inference should fail
    pub compare_inference_success: bool,
    /// Custom exit code (if Some, overrides normal exit code logic)
    pub custom_exit_code: Option<i32>,
    /// Whether profile_with_flamegraph should fail
    pub profile_flamegraph_success: bool,
    /// Whether profile_with_focus should fail
    pub profile_focus_success: bool,
    /// Whether fingerprint_model should fail
    pub fingerprint_success: bool,
    /// Whether validate_stats should fail
    pub validate_stats_success: bool,
    /// Whether validate_model_strict should fail
    pub validate_strict_success: bool,
    /// Whether pull_model should succeed
    pub pull_success: bool,
    /// Path returned by pull_model on success
    pub pull_model_path: String,
    /// Whether inspect_model_json should succeed
    pub inspect_json_success: bool,
    /// Tensor names returned by inspect_model_json
    pub inspect_tensor_names: Vec<String>,
    /// Whether ollama inference should succeed
    pub ollama_success: bool,
    /// Custom response for ollama inference
    pub ollama_response: String,
    /// Whether ollama pull should succeed
    pub ollama_pull_success: bool,
    /// Whether ollama create should succeed
    pub ollama_create_success: bool,
    /// Whether serve_model should succeed
    pub serve_success: bool,
    /// Whether http_get should succeed
    pub http_get_success: bool,
    /// Custom HTTP response body
    pub http_get_response: String,
    /// Whether profile_memory should succeed
    pub profile_memory_success: bool,
    /// Whether run_chat should succeed
    pub chat_success: bool,
    /// Custom response for chat
    pub chat_response: String,
    /// Whether http_post should succeed
    pub http_post_success: bool,
    /// Custom response for http_post
    pub http_post_response: String,
    /// Whether spawn_serve should succeed
    pub spawn_serve_success: bool,
    /// Custom stdout for check_model (when check_success is true)
    pub check_response: String,
    /// Whether quantize_model should succeed
    pub quantize_success: bool,
    /// Whether import_model should succeed
    pub import_success: bool,
    /// Whether prune_model should succeed
    pub prune_success: bool,
    /// Whether distill_model should succeed
    pub distill_success: bool,
}

impl Default for MockCommandRunner {
    fn default() -> Self {
        Self {
            inference_response: "The answer is 4.".to_string(),
            inference_success: true,
            convert_success: true,
            tps: 25.0,
            crash: false,
            inference_stderr: None,
            profile_ci_unavailable: false,
            profile_ci_stderr: None,
            inspect_success: true,
            validate_success: true,
            bench_success: true,
            check_success: true,
            profile_success: true,
            diff_tensors_success: true,
            compare_inference_success: true,
            custom_exit_code: None,
            profile_flamegraph_success: true,
            profile_focus_success: true,
            fingerprint_success: true,
            validate_stats_success: true,
            validate_strict_success: true,
            pull_success: true,
            pull_model_path: "/mock/model.safetensors".to_string(),
            inspect_json_success: true,
            inspect_tensor_names: vec![
                "model.embed_tokens.weight".to_string(),
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                "model.layers.0.self_attn.k_proj.weight".to_string(),
                "model.layers.0.self_attn.v_proj.weight".to_string(),
                "model.layers.0.self_attn.o_proj.weight".to_string(),
                "model.layers.0.mlp.gate_proj.weight".to_string(),
                "model.layers.0.mlp.up_proj.weight".to_string(),
                "model.layers.0.mlp.down_proj.weight".to_string(),
                "model.norm.weight".to_string(),
                "lm_head.weight".to_string(),
            ],
            ollama_success: true,
            ollama_response: "The answer is 4.".to_string(),
            ollama_pull_success: true,
            ollama_create_success: true,
            serve_success: true,
            http_get_success: true,
            http_get_response: r#"{"models":[]}"#.to_string(),
            profile_memory_success: true,
            chat_success: true,
            chat_response: "The answer is 4.".to_string(),
            http_post_success: true,
            http_post_response: r#"{"choices":[{"text":"The answer is 4."}]}"#.to_string(),
            spawn_serve_success: true,
            check_response: "All checks passed".to_string(),
            quantize_success: true,
            import_success: true,
            prune_success: true,
            distill_success: true,
        }
    }
}

impl MockCommandRunner {
    /// Create a new mock runner with default responses
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the inference response
    #[must_use]
    pub fn with_inference_response(mut self, response: impl Into<String>) -> Self {
        self.inference_response = response.into();
        self
    }

    /// Set whether inference should fail
    #[must_use]
    pub fn with_inference_failure(mut self) -> Self {
        self.inference_success = false;
        self
    }

    /// Set whether convert should fail
    #[must_use]
    pub fn with_convert_failure(mut self) -> Self {
        self.convert_success = false;
        self
    }

    /// Set the TPS to report
    #[must_use]
    pub fn with_tps(mut self, tps: f64) -> Self {
        self.tps = tps;
        self
    }

    /// Simulate a crash (negative exit code)
    #[must_use]
    pub fn with_crash(mut self) -> Self {
        self.crash = true;
        self
    }

    /// Set the inference response with custom stderr
    #[must_use]
    pub fn with_inference_response_and_stderr(
        mut self,
        response: impl Into<String>,
        stderr: impl Into<String>,
    ) -> Self {
        self.inference_response = response.into();
        self.inference_stderr = Some(stderr.into());
        self
    }

    /// Simulate profile_ci feature not available
    #[must_use]
    pub fn with_profile_ci_unavailable(mut self) -> Self {
        self.profile_ci_unavailable = true;
        self
    }

    /// Set custom stderr for profile_ci
    #[must_use]
    pub fn with_profile_ci_stderr(mut self, stderr: impl Into<String>) -> Self {
        self.profile_ci_stderr = Some(stderr.into());
        self
    }

    /// Set whether inspect should fail
    #[must_use]
    pub fn with_inspect_failure(mut self) -> Self {
        self.inspect_success = false;
        self
    }

    /// Set whether validate should fail
    #[must_use]
    pub fn with_validate_failure(mut self) -> Self {
        self.validate_success = false;
        self
    }

    /// Set whether bench should fail
    #[must_use]
    pub fn with_bench_failure(mut self) -> Self {
        self.bench_success = false;
        self
    }

    /// Set whether check should fail
    #[must_use]
    pub fn with_check_failure(mut self) -> Self {
        self.check_success = false;
        self
    }

    /// Set whether profile should fail
    #[must_use]
    pub fn with_profile_failure(mut self) -> Self {
        self.profile_success = false;
        self
    }

    /// Set whether diff_tensors should fail
    #[must_use]
    pub fn with_diff_tensors_failure(mut self) -> Self {
        self.diff_tensors_success = false;
        self
    }

    /// Set whether compare_inference should fail
    #[must_use]
    pub fn with_compare_inference_failure(mut self) -> Self {
        self.compare_inference_success = false;
        self
    }

    /// Set a custom exit code for inference
    #[must_use]
    pub fn with_exit_code(mut self, code: i32) -> Self {
        self.custom_exit_code = Some(code);
        self
    }

    /// Set whether profile_with_flamegraph should fail
    #[must_use]
    pub fn with_profile_flamegraph_failure(mut self) -> Self {
        self.profile_flamegraph_success = false;
        self
    }

    /// Set whether profile_with_focus should fail
    #[must_use]
    pub fn with_profile_focus_failure(mut self) -> Self {
        self.profile_focus_success = false;
        self
    }

    /// Set whether fingerprint_model should fail
    #[must_use]
    pub fn with_fingerprint_failure(mut self) -> Self {
        self.fingerprint_success = false;
        self
    }

    /// Set whether validate_stats should fail
    #[must_use]
    pub fn with_validate_stats_failure(mut self) -> Self {
        self.validate_stats_success = false;
        self
    }

    /// Set whether validate_model_strict should fail
    #[must_use]
    pub fn with_validate_strict_failure(mut self) -> Self {
        self.validate_strict_success = false;
        self
    }

    /// Set whether pull_model should fail
    #[must_use]
    pub fn with_pull_failure(mut self) -> Self {
        self.pull_success = false;
        self
    }

    /// Set the model path returned by pull_model
    #[must_use]
    pub fn with_pull_model_path(mut self, path: impl Into<String>) -> Self {
        self.pull_model_path = path.into();
        self
    }

    /// Set whether inspect_model_json should fail
    #[must_use]
    pub fn with_inspect_json_failure(mut self) -> Self {
        self.inspect_json_success = false;
        self
    }

    /// Set custom tensor names for inspect_model_json
    #[must_use]
    pub fn with_tensor_names(mut self, names: Vec<String>) -> Self {
        self.inspect_tensor_names = names;
        self
    }

    /// Set custom ollama inference response
    #[must_use]
    pub fn with_ollama_response(mut self, response: impl Into<String>) -> Self {
        self.ollama_response = response.into();
        self
    }

    /// Set whether ollama inference should fail
    #[must_use]
    pub fn with_ollama_failure(mut self) -> Self {
        self.ollama_success = false;
        self
    }

    /// Set whether ollama pull should fail
    #[must_use]
    pub fn with_ollama_pull_failure(mut self) -> Self {
        self.ollama_pull_success = false;
        self
    }

    /// Set whether ollama create should fail
    #[must_use]
    pub fn with_ollama_create_failure(mut self) -> Self {
        self.ollama_create_success = false;
        self
    }

    /// Set whether serve_model should fail
    #[must_use]
    pub fn with_serve_failure(mut self) -> Self {
        self.serve_success = false;
        self
    }

    /// Set whether http_get should fail
    #[must_use]
    pub fn with_http_get_failure(mut self) -> Self {
        self.http_get_success = false;
        self
    }

    /// Set custom HTTP response body
    #[must_use]
    pub fn with_http_get_response(mut self, response: impl Into<String>) -> Self {
        self.http_get_response = response.into();
        self
    }

    /// Set whether profile_memory should fail
    #[must_use]
    pub fn with_profile_memory_failure(mut self) -> Self {
        self.profile_memory_success = false;
        self
    }

    /// Set whether run_chat should fail
    #[must_use]
    pub fn with_chat_failure(mut self) -> Self {
        self.chat_success = false;
        self
    }

    /// Set custom chat response
    #[must_use]
    pub fn with_chat_response(mut self, response: impl Into<String>) -> Self {
        self.chat_response = response.into();
        self
    }

    /// Set whether http_post should fail
    #[must_use]
    pub fn with_http_post_failure(mut self) -> Self {
        self.http_post_success = false;
        self
    }

    /// Set custom http_post response
    #[must_use]
    pub fn with_http_post_response(mut self, response: impl Into<String>) -> Self {
        self.http_post_response = response.into();
        self
    }

    /// Set whether spawn_serve should fail
    #[must_use]
    pub fn with_spawn_serve_failure(mut self) -> Self {
        self.spawn_serve_success = false;
        self
    }

    /// Set custom stdout for check_model
    #[must_use]
    pub fn with_check_response(mut self, response: impl Into<String>) -> Self {
        self.check_response = response.into();
        self
    }

    /// Set whether quantize_model should fail
    #[must_use]
    pub fn with_quantize_failure(mut self) -> Self {
        self.quantize_success = false;
        self
    }

    /// Set whether import_model should fail
    #[must_use]
    pub fn with_import_failure(mut self) -> Self {
        self.import_success = false;
        self
    }

    /// Set whether prune_model should fail
    #[must_use]
    pub fn with_prune_failure(mut self) -> Self {
        self.prune_success = false;
        self
    }

    /// Set whether distill_model should fail
    #[must_use]
    pub fn with_distill_failure(mut self) -> Self {
        self.distill_success = false;
        self
    }
}
