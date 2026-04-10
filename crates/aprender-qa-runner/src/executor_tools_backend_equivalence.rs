/// Backend equivalence and additional tool test methods for ToolExecutor
impl ToolExecutor {
    /// Execute backend equivalence test (F-CONV-BE-001)
    ///
    /// Compares CPU vs GPU output to verify they produce equivalent results.
    /// Skips if GPU is not available.
    #[must_use]
    pub fn execute_backend_equivalence(&self) -> ToolTestResult {
        use std::process::Command;
        let start = std::time::Instant::now();

        let prompt = "What is 2+2?";

        // Run with CPU (--no-gpu)
        let cpu_output = Command::new("apr")
            .arg("run")
            .arg(&self.model_path)
            .arg("-p")
            .arg(prompt)
            .arg("--max-tokens")
            .arg("8")
            .arg("--no-gpu")
            .output();

        let cpu_result = match cpu_output {
            Ok(out) => {
                if out.status.success() {
                    Some(String::from_utf8_lossy(&out.stdout).to_string())
                } else {
                    None
                }
            }
            Err(_) => None,
        };

        // Run with GPU
        let gpu_output = Command::new("apr")
            .arg("run")
            .arg(&self.model_path)
            .arg("-p")
            .arg(prompt)
            .arg("--max-tokens")
            .arg("8")
            .arg("--gpu")
            .output();

        let gpu_result = match gpu_output {
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                // Check if GPU is not available
                if stderr.contains("No GPU") || stderr.contains("CUDA") || !out.status.success() {
                    None // GPU not available
                } else {
                    Some(String::from_utf8_lossy(&out.stdout).to_string())
                }
            }
            Err(_) => None,
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        match (cpu_result, gpu_result) {
            (Some(cpu), Some(gpu)) => {
                // Compare outputs - they should be similar (not necessarily identical due to FP)
                let equivalent = cpu.trim() == gpu.trim();
                ToolTestResult {
                    tool: "backend-equivalence".to_string(),
                    passed: equivalent,
                    exit_code: i32::from(!equivalent),
                    stdout: format!("CPU: {}\nGPU: {}", cpu.trim(), gpu.trim()),
                    stderr: if equivalent {
                        String::new()
                    } else {
                        "CPU and GPU outputs differ".to_string()
                    },
                    duration_ms,
                    gate_id: "F-CONV-BE-001".to_string(),
                }
            }
            (Some(_), None) => ToolTestResult {
                tool: "backend-equivalence".to_string(),
                passed: false,
                exit_code: -2,
                stdout: String::new(),
                stderr: "GPU not available - skipping backend equivalence test".to_string(),
                duration_ms,
                gate_id: "F-CONV-BE-001".to_string(),
            },
            _ => ToolTestResult {
                tool: "backend-equivalence".to_string(),
                passed: false,
                exit_code: -1,
                stdout: String::new(),
                stderr: "Failed to run inference on both backends".to_string(),
                duration_ms,
                gate_id: "F-CONV-BE-001".to_string(),
            },
        }
    }
}

include!("executor_serve_lifecycle.rs");
