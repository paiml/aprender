impl ToolExecutor {

    /// Execute apr serve lifecycle test (F-INTEG-003)
    ///
    /// Tests the full serve lifecycle:
    /// 1. Start server
    /// 2. Wait for health endpoint
    /// 3. Make inference request
    /// 4. Shutdown cleanly
    #[must_use]
    pub fn execute_serve_lifecycle(&self) -> ToolTestResult {
        use std::io::{BufRead, BufReader};
        use std::process::{Command, Stdio};
        use std::time::Duration;

        let start = std::time::Instant::now();
        let port = 18080; // Use high port to avoid conflicts

        // Start server
        let mut server_cmd = Command::new("apr");
        server_cmd
            .arg("serve")
            .arg(&self.model_path)
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if self.no_gpu {
            server_cmd.arg("--no-gpu");
        }

        let mut server = match server_cmd.spawn() {
            Ok(child) => child,
            Err(e) => {
                return ToolTestResult {
                    tool: "serve-lifecycle".to_string(),
                    passed: false,
                    exit_code: -1,
                    stdout: String::new(),
                    stderr: format!("Failed to start server: {e}"),
                    duration_ms: start.elapsed().as_millis() as u64,
                    gate_id: "F-INTEG-003".to_string(),
                };
            }
        };

        // Wait for server to be ready (check stderr for "Listening on")
        let stderr = server.stderr.take();
        let ready = stderr.map_or_else(
            || {
                // Wait a fixed time if can't read stderr
                std::thread::sleep(Duration::from_secs(3));
                true
            },
            |stderr| {
                let reader = BufReader::new(stderr);
                let mut ready = false;
                for line in reader.lines().take(20).flatten() {
                    if line.contains("Listening") || line.contains("listening") {
                        ready = true;
                        break;
                    }
                }
                ready
            },
        );

        if !ready {
            // Give it more time
            std::thread::sleep(Duration::from_secs(2));
        }

        // Test health endpoint
        let health_result = Command::new("curl")
            .arg("-sf")
            .arg(format!("http://localhost:{port}/health"))
            .arg("--connect-timeout")
            .arg("5")
            .output();

        let health_ok = health_result.map(|o| o.status.success()).unwrap_or(false);

        // Test inference endpoint
        let inference_result = Command::new("curl")
            .arg("-sf")
            .arg("-X")
            .arg("POST")
            .arg(format!("http://localhost:{port}/v1/chat/completions"))
            .arg("-H")
            .arg("Content-Type: application/json")
            .arg("-d")
            .arg(r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":5}"#)
            .arg("--connect-timeout")
            .arg("10")
            .output();

        let inference_ok = inference_result
            .map(|o| o.status.success())
            .unwrap_or(false);

        // Shutdown server
        let _ = server.kill();
        let _ = server.wait();

        let duration_ms = start.elapsed().as_millis() as u64;

        let passed = health_ok && inference_ok;
        let stdout = format!(
            "Health check: {}\nInference: {}",
            if health_ok { "OK" } else { "FAILED" },
            if inference_ok { "OK" } else { "FAILED" }
        );
        let stderr = if passed {
            String::new()
        } else {
            format!("Serve lifecycle incomplete: health={health_ok}, inference={inference_ok}")
        };

        ToolTestResult {
            tool: "serve-lifecycle".to_string(),
            passed,
            exit_code: i32::from(!passed),
            stdout,
            stderr,
            duration_ms,
            gate_id: "F-INTEG-003".to_string(),
        }
    }

    /// Execute all tool tests
    #[must_use]
    pub fn execute_all(&self) -> Vec<ToolTestResult> {
        self.execute_all_with_serve(false)
    }

    /// Execute all tool tests, optionally including serve lifecycle
    #[must_use]
    pub fn execute_all_with_serve(&self, include_serve: bool) -> Vec<ToolTestResult> {
        let mut results = vec![
            // Core tool tests
            self.execute_inspect(),
            self.execute_inspect_verified(), // T-GH192-01: metadata verification
            self.execute_validate(),
            self.execute_check(),
            self.execute_bench(),
        ];

        // Trace level tests
        for level in &["none", "basic", "layer", "payload"] {
            results.push(self.execute_trace(level));
        }

        // Profile tests (F-PROFILE-001 basic, F-PROFILE-006/007/008 CI mode)
        results.push(self.execute_profile());
        results.push(self.execute_profile_ci());
        results.push(self.execute_profile_ci_assertion_failure());
        results.push(self.execute_profile_ci_p99());

        // Serve lifecycle test (F-INTEG-003)
        if include_serve {
            results.push(self.execute_serve_lifecycle());
        }

        results
    }

    fn build_result_from_output(
        &self,
        tool: &str,
        output: crate::command::CommandOutput,
        start: std::time::Instant,
    ) -> ToolTestResult {
        let duration_ms = start.elapsed().as_millis() as u64;

        ToolTestResult {
            tool: tool.to_string(),
            passed: output.success,
            exit_code: output.exit_code,
            stdout: output.stdout,
            stderr: output.stderr,
            duration_ms,
            gate_id: format!("F-{}-001", tool.to_uppercase().replace('-', "_")),
        }
    }
}
