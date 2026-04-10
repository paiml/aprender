
impl CommandRunner for RealCommandRunner {
    fn run_inference(
        &self,
        model_path: &Path,
        prompt: &str,
        max_tokens: u32,
        no_gpu: bool,
        extra_args: &[&str],
    ) -> CommandOutput {
        let model_str = model_path.display().to_string();
        let max_tokens_str = max_tokens.to_string();

        let mut args = vec![
            "run",
            &model_str,
            "-p",
            prompt,
            "--max-tokens",
            &max_tokens_str,
        ];

        if no_gpu {
            args.push("--no-gpu");
        }

        args.extend(extra_args.iter());
        self.execute(&args)
    }

    fn convert_model(&self, source: &Path, target: &Path) -> CommandOutput {
        let source_str = source.display().to_string();
        let target_str = target.display().to_string();
        self.execute(&["rosetta", "convert", &source_str, &target_str])
    }

    fn inspect_model(&self, model_path: &Path) -> CommandOutput {
        let path_str = model_path.display().to_string();
        self.execute(&["rosetta", "inspect", &path_str])
    }

    fn validate_model(&self, model_path: &Path) -> CommandOutput {
        let path_str = model_path.display().to_string();
        self.execute(&["validate", &path_str])
    }

    fn validate_model_strict(&self, model_path: &Path) -> CommandOutput {
        let path_str = model_path.display().to_string();
        self.execute(&["validate", "--strict", "--json", &path_str])
    }

    fn bench_model(&self, model_path: &Path) -> CommandOutput {
        let path_str = model_path.display().to_string();
        self.execute(&["bench", &path_str])
    }

    fn check_model(&self, model_path: &Path) -> CommandOutput {
        let path_str = model_path.display().to_string();
        self.execute(&["check", &path_str])
    }

    fn profile_model(&self, model_path: &Path, warmup: u32, measure: u32) -> CommandOutput {
        let path_str = model_path.display().to_string();
        let warmup_str = warmup.to_string();
        let measure_str = measure.to_string();
        self.execute(&[
            "profile",
            &path_str,
            "--warmup",
            &warmup_str,
            "--measure",
            &measure_str,
        ])
    }

    fn profile_ci(
        &self,
        model_path: &Path,
        min_throughput: Option<f64>,
        max_p99: Option<f64>,
        warmup: u32,
        measure: u32,
        no_gpu: bool,
    ) -> CommandOutput {
        let path_str = model_path.display().to_string();
        let warmup_str = warmup.to_string();
        let measure_str = measure.to_string();

        let mut args = vec![
            "profile",
            &path_str,
            "--ci",
            "--warmup",
            &warmup_str,
            "--measure",
            &measure_str,
            "--json",
        ];

        if no_gpu {
            args.push("--no-gpu");
        }

        let throughput_str;
        if let Some(t) = min_throughput {
            throughput_str = t.to_string();
            args.push("--assert-throughput");
            args.push(&throughput_str);
        }

        let p99_str;
        if let Some(p) = max_p99 {
            p99_str = p.to_string();
            args.push("--assert-p99");
            args.push(&p99_str);
        }

        self.execute(&args)
    }

    fn diff_tensors(&self, model_a: &Path, model_b: &Path, json: bool) -> CommandOutput {
        let a_str = model_a.display().to_string();
        let b_str = model_b.display().to_string();

        let mut args = vec!["rosetta", "diff-tensors", &a_str, &b_str];
        if json {
            args.push("--json");
        }
        self.execute(&args)
    }

    fn compare_inference(
        &self,
        model_a: &Path,
        model_b: &Path,
        prompt: &str,
        max_tokens: u32,
        tolerance: f64,
    ) -> CommandOutput {
        let a_str = model_a.display().to_string();
        let b_str = model_b.display().to_string();
        let max_tokens_str = max_tokens.to_string();
        let tolerance_str = tolerance.to_string();

        self.execute(&[
            "rosetta",
            "compare-inference",
            &a_str,
            &b_str,
            "--prompt",
            prompt,
            "--max-tokens",
            &max_tokens_str,
            "--tolerance",
            &tolerance_str,
            "--json",
        ])
    }

    fn profile_with_flamegraph(
        &self,
        model_path: &Path,
        output_path: &Path,
        no_gpu: bool,
    ) -> CommandOutput {
        let model_str = model_path.display().to_string();
        let output_str = output_path.display().to_string();

        let mut args = vec![
            "run",
            &model_str,
            "-p",
            "Hello",
            "--max-tokens",
            "4",
            "--profile",
            "--profile-output",
            &output_str,
        ];

        if no_gpu {
            args.push("--no-gpu");
        }

        self.execute(&args)
    }

    fn profile_with_focus(&self, model_path: &Path, focus: &str, no_gpu: bool) -> CommandOutput {
        let model_str = model_path.display().to_string();

        let mut args = vec![
            "run",
            &model_str,
            "-p",
            "Hello",
            "--max-tokens",
            "4",
            "--profile",
            "--focus",
            focus,
        ];

        if no_gpu {
            args.push("--no-gpu");
        }

        self.execute(&args)
    }

    fn fingerprint_model(&self, model_path: &Path, json: bool) -> CommandOutput {
        let path_str = model_path.display().to_string();
        let mut args = vec!["rosetta", "fingerprint", &path_str];
        if json {
            args.push("--json");
        }
        self.execute(&args)
    }

    fn validate_stats(&self, fp_a: &Path, fp_b: &Path) -> CommandOutput {
        let a_str = fp_a.display().to_string();
        let b_str = fp_b.display().to_string();
        self.execute(&["rosetta", "validate-stats", &a_str, "--reference", &b_str])
    }

    fn pull_model(&self, hf_repo: &str) -> CommandOutput {
        self.execute(&["pull", "--json", hf_repo])
    }

    fn inspect_model_json(&self, model_path: &Path) -> CommandOutput {
        let path_str = model_path.display().to_string();
        self.execute(&["rosetta", "inspect", "--json", &path_str])
    }

    fn run_ollama_inference(
        &self,
        model_tag: &str,
        prompt: &str,
        temperature: f64,
    ) -> CommandOutput {
        use std::process::Command;

        let temp_str = temperature.to_string();
        match Command::new("ollama")
            .args(["run", model_tag, "--temp", &temp_str])
            .arg(prompt)
            .output()
        {
            Ok(output) => CommandOutput {
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                exit_code: output.status.code().unwrap_or(-1),
                success: output.status.success(),
            },
            Err(e) => CommandOutput::failure(-1, format!("Failed to execute ollama: {e}")),
        }
    }

    fn pull_ollama_model(&self, model_tag: &str) -> CommandOutput {
        use std::process::Command;

        match Command::new("ollama").args(["pull", model_tag]).output() {
            Ok(output) => CommandOutput {
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                exit_code: output.status.code().unwrap_or(-1),
                success: output.status.success(),
            },
            Err(e) => CommandOutput::failure(-1, format!("Failed to execute ollama: {e}")),
        }
    }

    fn create_ollama_model(&self, model_tag: &str, modelfile_path: &Path) -> CommandOutput {
        use std::process::Command;

        let path_str = modelfile_path.display().to_string();
        match Command::new("ollama")
            .args(["create", model_tag, "-f", &path_str])
            .output()
        {
            Ok(output) => CommandOutput {
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                exit_code: output.status.code().unwrap_or(-1),
                success: output.status.success(),
            },
            Err(e) => CommandOutput::failure(-1, format!("Failed to execute ollama create: {e}")),
        }
    }

    fn serve_model(&self, model_path: &Path, port: u16) -> CommandOutput {
        let model_str = model_path.display().to_string();
        let port_str = port.to_string();
        self.execute(&["serve", &model_str, "--port", &port_str])
    }

    fn http_get(&self, url: &str) -> CommandOutput {
        use std::process::Command;

        match Command::new("curl").args(["-s", "-m", "10", url]).output() {
            Ok(output) => CommandOutput {
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                exit_code: output.status.code().unwrap_or(-1),
                success: output.status.success(),
            },
            Err(e) => CommandOutput::failure(-1, format!("Failed to execute curl: {e}")),
        }
    }

    fn profile_memory(&self, model_path: &Path) -> CommandOutput {
        // apr profile requires GGUF or APR format — resolve from workspace
        let gguf = model_path.join("gguf").join("model.gguf");
        let apr = model_path.join("apr").join("model.apr");
        let path = if gguf.exists() {
            gguf
        } else if apr.exists() {
            apr
        } else {
            model_path.to_path_buf()
        };
        let path_str = path.display().to_string();
        self.execute(&["profile", &path_str, "--format", "json"])
    }

    fn run_chat(
        &self,
        model_path: &Path,
        prompt: &str,
        no_gpu: bool,
        extra_args: &[&str],
    ) -> CommandOutput {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let model_str = model_path.display().to_string();
        let mut args = vec!["chat", &model_str];
        if no_gpu {
            args.push("--no-gpu");
        }
        args.extend(extra_args.iter());

        match Command::new(&self.apr_binary)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(mut child) => {
                if let Some(mut stdin) = child.stdin.take() {
                    if let Err(e) = stdin.write_all(prompt.as_bytes()) {
                        eprintln!("[JIDOKA] Failed to write prompt to stdin: {e}");
                    }
                    if let Err(e) = stdin.write_all(b"\n") {
                        eprintln!("[JIDOKA] Failed to write newline to stdin: {e}");
                    }
                }
                match child.wait_with_output() {
                    Ok(output) => CommandOutput {
                        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                        exit_code: output.status.code().unwrap_or(-1),
                        success: output.status.success(),
                    },
                    Err(e) => CommandOutput::failure(-1, format!("Failed to wait for chat: {e}")),
                }
            }
            Err(e) => CommandOutput::failure(-1, format!("Failed to execute chat: {e}")),
        }
    }

    fn http_post(&self, url: &str, body: &str) -> CommandOutput {
        use std::process::Command;

        match Command::new("curl")
            .args([
                "-s",
                "-m",
                "120",
                "-X",
                "POST",
                "-H",
                "Content-Type: application/json",
                "-d",
                body,
                url,
            ])
            .output()
        {
            Ok(output) => CommandOutput {
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                exit_code: output.status.code().unwrap_or(-1),
                success: output.status.success(),
            },
            Err(e) => CommandOutput::failure(-1, format!("Failed to execute curl POST: {e}")),
        }
    }

    fn spawn_serve(&self, model_path: &Path, port: u16, no_gpu: bool) -> CommandOutput {
        use std::process::{Command, Stdio};

        let model_str = model_path.display().to_string();
        let port_str = port.to_string();
        let mut args = vec!["serve", &model_str, "--port", &port_str];
        if no_gpu {
            args.push("--no-gpu");
        }

        match Command::new(&self.apr_binary)
            .args(&args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => CommandOutput::success(format!("{}", child.id())),
            Err(e) => CommandOutput::failure(-1, format!("Failed to spawn serve: {e}")),
        }
    }

    fn quantize_model(
        &self,
        model_path: &Path,
        output_path: &Path,
        scheme: &str,
    ) -> CommandOutput {
        let model_str = model_path.display().to_string();
        let output_str = output_path.display().to_string();
        self.execute(&["quantize", &model_str, "--scheme", scheme, "--json", "-o", &output_str])
    }

    fn import_model(&self, source_path: &Path, output_path: &Path) -> CommandOutput {
        let source_str = source_path.display().to_string();
        let output_str = output_path.display().to_string();
        self.execute(&["import", &source_str, "--json", "-o", &output_str])
    }

    fn prune_model(
        &self,
        model_path: &Path,
        output_path: &Path,
        method: &str,
        target_ratio: f64,
    ) -> CommandOutput {
        let model_str = model_path.display().to_string();
        let output_str = output_path.display().to_string();
        let ratio_str = target_ratio.to_string();
        self.execute(&[
            "prune", &model_str,
            "--method", method,
            "--target-ratio", &ratio_str,
            "--json", "-o", &output_str,
        ])
    }

    fn distill_model(
        &self,
        teacher_path: &Path,
        student_path: &Path,
        output_path: &Path,
        data_path: &str,
    ) -> CommandOutput {
        let teacher_str = teacher_path.display().to_string();
        let student_str = student_path.display().to_string();
        let output_str = output_path.display().to_string();
        self.execute(&[
            "distill", &teacher_str,
            "--student", &student_str,
            "--data", data_path,
            "--json", "-o", &output_str,
        ])
    }
}
