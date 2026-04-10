/// Mock implementation of CommandRunner for testing without real model binaries
impl CommandRunner for MockCommandRunner {
    /// Simulate model inference with configurable success/failure and responses
    fn run_inference(
        &self,
        _model_path: &Path,
        prompt: &str,
        _max_tokens: u32,
        _no_gpu: bool,
        _extra_args: &[&str],
    ) -> CommandOutput {
        // Custom exit code takes precedence
        if let Some(exit_code) = self.custom_exit_code {
            return CommandOutput {
                stdout: String::new(),
                stderr: "Custom exit code error".to_string(),
                exit_code,
                success: exit_code == 0,
            };
        }

        // Simulate crash
        if self.crash {
            return CommandOutput {
                stdout: String::new(),
                stderr: "SIGSEGV: Segmentation fault".to_string(),
                exit_code: -11, // SIGSEGV
                success: false,
            };
        }

        if !self.inference_success {
            return CommandOutput::failure(1, "Inference failed");
        }

        // Generate appropriate response based on prompt
        let response = if prompt.contains("2+2") || prompt.contains("2 + 2") {
            "The answer is 4.".to_string()
        } else if prompt.starts_with("def ") || prompt.starts_with("fn ") {
            "    return result".to_string()
        } else if prompt.is_empty() {
            String::new()
        } else {
            self.inference_response.clone()
        };

        let stdout = format!(
            "Output:\n{}\nCompleted in 1.5s\ntok/s: {:.1}",
            response, self.tps
        );

        // Return with stderr if set
        if let Some(ref stderr) = self.inference_stderr {
            CommandOutput::with_output(stdout, stderr.clone(), 0)
        } else {
            CommandOutput::success(stdout)
        }
    }

    /// Simulate model format conversion
    fn convert_model(&self, _source: &Path, _target: &Path) -> CommandOutput {
        if self.convert_success {
            CommandOutput::success("Conversion successful")
        } else {
            CommandOutput::failure(1, "Conversion failed")
        }
    }

    /// Simulate model inspection returning format and tensor info
    fn inspect_model(&self, _model_path: &Path) -> CommandOutput {
        if self.inspect_success {
            CommandOutput::success(r#"{"format":"GGUF","tensors":100,"parameters":"1.5B"}"#)
        } else {
            CommandOutput::failure(1, "Inspect failed: invalid model format")
        }
    }

    /// Simulate basic model validation
    fn validate_model(&self, _model_path: &Path) -> CommandOutput {
        if self.validate_success {
            CommandOutput::success("Model validation passed")
        } else {
            CommandOutput::failure(1, "Validation failed: corrupted tensors")
        }
    }

    /// Simulate strict model validation with detailed tensor checks
    fn validate_model_strict(&self, _model_path: &Path) -> CommandOutput {
        if self.validate_strict_success {
            CommandOutput::success(r#"{"valid":true,"tensors_checked":100,"issues":[]}"#)
        } else {
            CommandOutput::with_output(
                r#"{"valid":false,"tensors_checked":100,"issues":["all-zeros tensor: lm_head.weight (6.7GB F32)","expected BF16 but found F32"]}"#,
                "Validation failed: corrupt model detected",
                1,
            )
        }
    }

    /// Simulate model benchmarking returning throughput and latency metrics
    fn bench_model(&self, _model_path: &Path) -> CommandOutput {
        if self.bench_success {
            let output = format!(
                r#"{{"throughput_tps":{:.1},"latency_p50_ms":78.2,"latency_p99_ms":156.5}}"#,
                self.tps
            );
            CommandOutput::success(output)
        } else {
            CommandOutput::failure(1, "Benchmark failed: model load error")
        }
    }

    /// Simulate model safety check
    fn check_model(&self, _model_path: &Path) -> CommandOutput {
        if self.check_success {
            CommandOutput::success(&self.check_response)
        } else {
            CommandOutput::failure(1, "Check failed: safety issues detected")
        }
    }

    /// Simulate model profiling with warmup and measurement phases
    fn profile_model(&self, _model_path: &Path, _warmup: u32, _measure: u32) -> CommandOutput {
        if self.profile_success {
            let output = format!(
                r#"{{"throughput_tps":{:.1},"latency_p50_ms":78.2,"latency_p99_ms":156.5}}"#,
                self.tps
            );
            CommandOutput::success(output)
        } else {
            CommandOutput::failure(1, "Profile failed: insufficient memory")
        }
    }

    /// Simulate CI profile with threshold assertions
    fn profile_ci(
        &self,
        _model_path: &Path,
        min_throughput: Option<f64>,
        max_p99: Option<f64>,
        _warmup: u32,
        _measure: u32,
        _no_gpu: bool,
    ) -> CommandOutput {
        // Simulate feature not available
        if self.profile_ci_unavailable {
            let stderr = self.profile_ci_stderr.clone().unwrap_or_else(|| {
                "unexpected argument '--ci': apr profile does not support --ci mode".to_string()
            });
            return CommandOutput::with_output("", stderr, 1);
        }

        let throughput_pass = min_throughput.is_none_or(|t| self.tps >= t);
        let p99_pass = max_p99.is_none_or(|p| 156.5 <= p);
        let passed = throughput_pass && p99_pass;

        let output = format!(
            r#"{{"throughput_tps":{:.1},"latency_p50_ms":78.2,"latency_p99_ms":156.5,"passed":{}}}"#,
            self.tps, passed
        );

        if passed {
            CommandOutput::success(output)
        } else {
            CommandOutput::with_output(output, "", 1)
        }
    }

    /// Simulate tensor diff comparison between two models
    fn diff_tensors(&self, _model_a: &Path, _model_b: &Path, json: bool) -> CommandOutput {
        if !self.diff_tensors_success {
            return CommandOutput::failure(1, "Diff tensors failed: incompatible models");
        }
        if json {
            CommandOutput::success(
                r#"{"total_tensors":100,"mismatched_tensors":0,"transposed_tensors":0,"mismatches":[],"passed":true}"#,
            )
        } else {
            CommandOutput::success("All tensors match")
        }
    }

    /// Simulate token-level inference comparison between two models
    fn compare_inference(
        &self,
        _model_a: &Path,
        _model_b: &Path,
        _prompt: &str,
        _max_tokens: u32,
        _tolerance: f64,
    ) -> CommandOutput {
        if self.compare_inference_success {
            CommandOutput::success(
                r#"{"total_tokens":10,"matching_tokens":10,"max_logit_diff":0.0001,"passed":true,"token_comparisons":[]}"#,
            )
        } else {
            CommandOutput::failure(1, "Compare inference failed: output mismatch")
        }
    }

    /// Simulate flamegraph profiling
    fn profile_with_flamegraph(
        &self,
        _model_path: &Path,
        _output_path: &Path,
        _no_gpu: bool,
    ) -> CommandOutput {
        if self.profile_flamegraph_success {
            CommandOutput::success("Profile complete, flamegraph written")
        } else {
            CommandOutput::failure(1, "Profile flamegraph failed: profiler error")
        }
    }

    /// Simulate focused profiling on a specific component
    fn profile_with_focus(&self, _model_path: &Path, _focus: &str, _no_gpu: bool) -> CommandOutput {
        if self.profile_focus_success {
            let output = format!(
                r#"{{"throughput_tps":{:.1},"latency_p50_ms":78.2,"latency_p99_ms":156.5}}"#,
                self.tps
            );
            CommandOutput::success(output)
        } else {
            CommandOutput::failure(1, "Profile focus failed: invalid focus target")
        }
    }

    /// Simulate model fingerprinting with tensor statistics
    fn fingerprint_model(&self, _model_path: &Path, json: bool) -> CommandOutput {
        if self.fingerprint_success {
            if json {
                CommandOutput::success(
                    r#"{"tensors":{"0.q_proj.weight":{"mean":0.001,"std":0.05,"min":-0.2,"max":0.2}}}"#,
                )
            } else {
                CommandOutput::success("Fingerprint: 100 tensors captured")
            }
        } else {
            CommandOutput::failure(1, "Fingerprint failed: model load error")
        }
    }

    /// Simulate statistical validation between two fingerprints
    fn validate_stats(&self, _fp_a: &Path, _fp_b: &Path) -> CommandOutput {
        if self.validate_stats_success {
            CommandOutput::success(
                r#"{"passed":true,"total_tensors":100,"failed_tensors":0,"details":[]}"#,
            )
        } else {
            CommandOutput::failure(1, "Stats validation failed: 3 tensors exceed tolerance")
        }
    }

    /// Simulate pulling a model from HuggingFace registry
    fn pull_model(&self, _hf_repo: &str) -> CommandOutput {
        if self.pull_success {
            CommandOutput::success(format!("Path: {}", self.pull_model_path))
        } else {
            CommandOutput::failure(1, "Pull failed: model not found in registry")
        }
    }

    /// Simulate JSON-formatted model inspection with tensor names
    fn inspect_model_json(&self, _model_path: &Path) -> CommandOutput {
        if self.inspect_json_success {
            let tensor_names_json: String = self
                .inspect_tensor_names
                .iter()
                .map(|s| format!("\"{s}\""))
                .collect::<Vec<_>>()
                .join(", ");
            CommandOutput::success(format!(
                r#"{{"format":"SafeTensors","tensor_count":{},"tensor_names":[{}],"parameters":"1.5B"}}"#,
                self.inspect_tensor_names.len(),
                tensor_names_json
            ))
        } else {
            CommandOutput::failure(1, "Inspect failed: invalid model format")
        }
    }

    /// Simulate Ollama inference with a model tag
    fn run_ollama_inference(
        &self,
        _model_tag: &str,
        _prompt: &str,
        _temperature: f64,
    ) -> CommandOutput {
        if self.ollama_success {
            CommandOutput::success(format!(
                "Output:\n{}\nCompleted in 1.0s",
                self.ollama_response
            ))
        } else {
            CommandOutput::failure(1, "Ollama inference failed: model not found")
        }
    }

    /// Simulate pulling an Ollama model from registry
    fn pull_ollama_model(&self, _model_tag: &str) -> CommandOutput {
        if self.ollama_pull_success {
            CommandOutput::success("pulling manifest... done")
        } else {
            CommandOutput::failure(1, "Ollama pull failed: model not found in registry")
        }
    }

    /// Simulate creating a custom Ollama model from a modelfile
    fn create_ollama_model(&self, _model_tag: &str, _modelfile_path: &Path) -> CommandOutput {
        if self.ollama_create_success {
            CommandOutput::success("creating model... done")
        } else {
            CommandOutput::failure(1, "Ollama create failed: invalid modelfile")
        }
    }

    /// Simulate starting a model serving endpoint
    fn serve_model(&self, _model_path: &Path, _port: u16) -> CommandOutput {
        if self.serve_success {
            CommandOutput::success(r#"{"status":"listening","port":8080}"#)
        } else {
            CommandOutput::failure(1, "Serve failed: port in use")
        }
    }

    /// Simulate an HTTP GET request
    fn http_get(&self, _url: &str) -> CommandOutput {
        if self.http_get_success {
            CommandOutput::success(&self.http_get_response)
        } else {
            CommandOutput::failure(1, "HTTP request failed: connection refused")
        }
    }

    /// Simulate memory profiling returning RSS and cache metrics
    fn profile_memory(&self, _model_path: &Path) -> CommandOutput {
        if self.profile_memory_success {
            CommandOutput::success(r#"{"peak_rss_mb":1024,"model_size_mb":512,"kv_cache_mb":256}"#)
        } else {
            CommandOutput::failure(1, "Profile memory failed: insufficient memory")
        }
    }

    /// Simulate chat-mode inference with a model
    fn run_chat(
        &self,
        _model_path: &Path,
        prompt: &str,
        _no_gpu: bool,
        _extra_args: &[&str],
    ) -> CommandOutput {
        if !self.chat_success {
            return CommandOutput::failure(1, "Chat failed");
        }

        let response = if prompt.contains("2+2") || prompt.contains("2 + 2") {
            "The answer is 4.".to_string()
        } else {
            self.chat_response.clone()
        };

        let stdout = format!(
            "Output:\n{}\nCompleted in 1.5s\ntok/s: {:.1}",
            response, self.tps
        );
        CommandOutput::success(stdout)
    }

    /// Simulate an HTTP POST request with a body
    fn http_post(&self, _url: &str, _body: &str) -> CommandOutput {
        if self.http_post_success {
            CommandOutput::success(&self.http_post_response)
        } else {
            CommandOutput::failure(1, "HTTP POST failed: connection refused")
        }
    }

    /// Simulate spawning a background model server returning a mock PID
    fn spawn_serve(&self, _model_path: &Path, _port: u16, _no_gpu: bool) -> CommandOutput {
        if self.spawn_serve_success {
            CommandOutput::success("12345") // Mock PID
        } else {
            CommandOutput::failure(1, "Spawn serve failed: port in use")
        }
    }

    /// Simulate model quantization
    fn quantize_model(
        &self,
        _model_path: &Path,
        _output_path: &Path,
        scheme: &str,
    ) -> CommandOutput {
        if self.quantize_success {
            CommandOutput::success(format!(
                r#"{{"status":"success","scheme":"{scheme}","output_size_bytes":524288000,"tensor_count":100,"dtype":"{scheme}"}}"#
            ))
        } else {
            CommandOutput::failure(1, "Quantization failed: unsupported scheme")
        }
    }

    /// Simulate model format import
    fn import_model(&self, _source_path: &Path, _output_path: &Path) -> CommandOutput {
        if self.import_success {
            CommandOutput::success(
                r#"{"status":"success","output_size_bytes":1048576000,"tensor_count":100}"#,
            )
        } else {
            CommandOutput::failure(1, "Import failed: unsupported source format")
        }
    }

    /// Simulate model weight pruning
    fn prune_model(
        &self,
        _model_path: &Path,
        _output_path: &Path,
        method: &str,
        target_ratio: f64,
    ) -> CommandOutput {
        if self.prune_success {
            CommandOutput::success(format!(
                r#"{{"status":"success","method":"{method}","target_ratio":{target_ratio},"actual_sparsity":{target_ratio},"output_size_bytes":524288000,"tensor_count":100}}"#
            ))
        } else {
            CommandOutput::failure(1, "Prune failed: invalid method")
        }
    }

    /// Simulate knowledge distillation
    fn distill_model(
        &self,
        _teacher_path: &Path,
        _student_path: &Path,
        _output_path: &Path,
        _data_path: &str,
    ) -> CommandOutput {
        if self.distill_success {
            CommandOutput::success(
                r#"{"status":"success","initial_loss":2.5,"final_loss":1.2,"output_size_bytes":262144000,"teacher_size_bytes":1048576000}"#,
            )
        } else {
            CommandOutput::failure(1, "Distillation failed: data path not found")
        }
    }
}

#[cfg(test)]
#[path = "command_tests.rs"]
mod tests;
