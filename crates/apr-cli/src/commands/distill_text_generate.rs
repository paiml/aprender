fn start_teacher_server(apr_bin: &Path, model: &str) -> Result<std::process::Child> {
    use std::process::{Command, Stdio};
    Command::new(apr_bin)
        .args(["serve", "run", model, "--gpu", "--port", "8090"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to start apr serve: {e}")))
}

fn wait_for_server_health(server: &mut std::process::Child, json_output: bool) -> Result<()> {
    let health_url = "http://127.0.0.1:8090/health";
    for attempt in 0..180 {
        std::thread::sleep(std::time::Duration::from_secs(1));
        if let Ok(Some(status)) = server.try_wait() {
            let _ = server.kill();
            return Err(CliError::ValidationFailed(format!(
                "apr serve exited with status {status} during startup"
            )));
        }
        match ureq::get(health_url).call() {
            Ok(resp) if resp.status() == 200 => {
                if !json_output {
                    output::pipeline_stage("Starting teacher server", output::StageStatus::Done);
                    output::kv("  Ready after", format!("{}s", attempt + 1));
                    println!();
                }
                return Ok(());
            }
            _ => continue,
        }
    }
    let _ = server.kill();
    let _ = server.wait();
    Err(CliError::ValidationFailed(
        "Teacher server did not become ready within 180 seconds".into(),
    ))
}

/// Validate that teacher model and prompts files exist.
fn validate_distill_paths(config: &TextDistillConfig) -> Result<()> {
    let teacher_path = std::path::Path::new(&config.teacher.model);
    if !teacher_path.exists() {
        return Err(CliError::FileNotFound(teacher_path.to_path_buf()));
    }
    let prompts_path = std::path::Path::new(&config.synthetic_data.prompts);
    if !prompts_path.exists() {
        return Err(CliError::FileNotFound(prompts_path.to_path_buf()));
    }
    Ok(())
}

/// Print the text-generate header showing config summary.
fn print_generate_header(config: &TextDistillConfig, config_path: &Path) {
    output::header("apr distill apply — Stage: Generate Synthetic Data (GH-455)");
    println!();
    output::kv("  Config", config_path.display().to_string());
    output::kv("  Teacher", &config.teacher.model);
    output::kv("  Prompts", &config.synthetic_data.prompts);
    output::kv("  Output", &config.synthetic_data.output);
    output::kv(
        "  Max tokens/completion",
        config.teacher.max_tokens.to_string(),
    );
    output::kv(
        "  Temperature",
        format!("{:.2}", config.teacher.temperature),
    );
    output::kv(
        "  Target tokens",
        config.synthetic_data.target_tokens.to_string(),
    );
    println!();
}

/// Read prompts from a JSONL file, skipping blank lines.
fn read_prompts_jsonl(path: &Path) -> Result<Vec<serde_json::Value>> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut prompts = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| CliError::ValidationFailed(format!("Invalid prompt JSONL: {e}")))?;
        prompts.push(parsed);
    }
    Ok(prompts)
}

/// State loaded from an existing output file for resume support.
/// Key is `(prompt_text, sample_idx)` so multi-sample generation
/// (samples_per_prompt > 1) correctly resumes without collapsing
/// distinct samples into a single slot.
struct ResumeState {
    existing_samples: std::collections::HashSet<(String, u32)>,
    total_tokens: u64,
    generated_count: u64,
}

/// Load resume state from an existing output JSONL, creating parent dirs as needed.
/// For backwards compatibility with pre-samples_per_prompt outputs, records
/// missing a `sample_idx` field are treated as `sample_idx = 0`.
fn load_resume_state(output_path: &Path) -> Result<ResumeState> {
    use std::io::{BufRead, BufReader};
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut state = ResumeState {
        existing_samples: std::collections::HashSet::new(),
        total_tokens: 0,
        generated_count: 0,
    };
    if output_path.exists() {
        let existing = std::fs::File::open(output_path)?;
        for line in BufReader::new(existing).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&line) {
                if let Some(p) = parsed.get("prompt").and_then(|v| v.as_str()) {
                    let sample_idx = parsed
                        .get("sample_idx")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    state.existing_samples.insert((p.to_string(), sample_idx));
                }
                state.total_tokens += parsed.get("tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                state.generated_count += 1;
            }
        }
    }
    Ok(state)
}

/// POST to /generate with retry (up to 3 attempts). Returns None if all retries exhausted.
fn send_generate_request(
    url: &str,
    request_body: &str,
    prompt_index: usize,
    json_output: bool,
) -> (Option<ureq::Response>, bool) {
    let mut skipped = false;
    for retry in 0..3 {
        match ureq::post(url)
            .set("Content-Type", "application/json")
            .send_string(request_body)
        {
            Ok(r) => return (Some(r), false),
            Err(e) if retry < 2 => {
                if !json_output {
                    eprintln!(
                        "  Retry {}/{} for prompt {}: {e}",
                        retry + 1,
                        3,
                        prompt_index
                    );
                }
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
            Err(e) => {
                if !json_output {
                    eprintln!("  Skipping prompt {} after 3 retries: {e}", prompt_index);
                }
                skipped = true;
            }
        }
    }
    (None, skipped)
}

/// Format and print the final result of text generation.
fn format_generate_result(
    config: &TextDistillConfig,
    prompts_total: usize,
    generated_count: u64,
    skipped_count: u64,
    total_tokens: u64,
    target: u64,
    elapsed: std::time::Duration,
    json_output: bool,
) {
    if json_output {
        let result = serde_json::json!({
            "stage": "generate",
            "status": "completed",
            "prompts_total": prompts_total,
            "completions_generated": generated_count,
            "completions_skipped": skipped_count,
            "total_tokens": total_tokens,
            "target_tokens": target,
            "elapsed_seconds": elapsed.as_secs(),
            "output": config.synthetic_data.output,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&result).unwrap_or_default()
        );
    } else {
        output::pipeline_stage("Generating completions", output::StageStatus::Done);
        println!();
        output::kv("  Completions", generated_count.to_string());
        output::kv("  Skipped", skipped_count.to_string());
        output::kv("  Tokens", total_tokens.to_string());
        output::kv("  Target", target.to_string());
        output::kv("  Elapsed", format!("{:.0}s", elapsed.as_secs_f64()));
        output::kv(
            "  Throughput",
            format!(
                "{:.1} tok/s",
                total_tokens as f64 / elapsed.as_secs_f64().max(0.001)
            ),
        );
        output::kv("  Output", &config.synthetic_data.output);
        println!();
        println!(
            "  {} Synthetic data generated. Tokenize and train next.",
            "DONE".green().bold()
        );
    }
}

fn run_text_generate(
    config: &TextDistillConfig,
    config_path: &Path,
    json_output: bool,
) -> Result<()> {
    use std::io::Write;

    validate_distill_paths(config)?;

    if !json_output {
        print_generate_header(config, config_path);
    }

    let apr_bin = std::env::current_exe().map_err(|e| {
        CliError::ValidationFailed(format!("Cannot determine apr binary path: {e}"))
    })?;

    if !json_output {
        output::pipeline_stage("Starting teacher server", output::StageStatus::Running);
        output::kv("  Binary", apr_bin.display().to_string());
    }

    let mut server = start_teacher_server(&apr_bin, &config.teacher.model)?;
    wait_for_server_health(&mut server, json_output)?;

    let prompts_path = std::path::Path::new(&config.synthetic_data.prompts);
    let prompts = read_prompts_jsonl(prompts_path)?;

    if !json_output {
        output::pipeline_stage("Generating completions", output::StageStatus::Running);
        output::kv("  Loaded prompts", prompts.len().to_string());
    }

    let output_path = std::path::Path::new(&config.synthetic_data.output);
    let mut resume = load_resume_state(output_path)?;
    let mut skipped_count = 0u64;

    if !resume.existing_samples.is_empty() && !json_output {
        println!(
            "  Resuming: {} existing records, {} tokens",
            resume.existing_samples.len(),
            resume.total_tokens
        );
    }

    let output_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)?;
    let mut writer = std::io::BufWriter::new(output_file);

    let generate_url = format!("http://127.0.0.1:8090/generate");
    let target = config.synthetic_data.target_tokens;
    let samples_per_prompt = config.synthetic_data.samples_per_prompt.max(1);
    let start_time = std::time::Instant::now();
    let mut progress_counter: u64 = 0;

    'outer: for sample_idx in 0..samples_per_prompt {
        for (i, prompt_json) in prompts.iter().enumerate() {
            if resume.total_tokens >= target {
                break 'outer;
            }

            let prompt_text = prompt_json
                .get("prompt")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    CliError::ValidationFailed(format!("Prompt {} missing 'prompt' field", i))
                })?;

            if resume
                .existing_samples
                .contains(&(prompt_text.to_string(), sample_idx))
            {
                continue;
            }

            // ALB-111: Skip pathologically long prompts (55K char prompt caused hours-long prefill)
            if prompt_text.len() > config.synthetic_data.max_prompt_chars {
                if !json_output && sample_idx == 0 {
                    eprintln!(
                        "  Skipping prompt {} ({} chars > {} max)",
                        i,
                        prompt_text.len(),
                        config.synthetic_data.max_prompt_chars,
                    );
                }
                skipped_count += 1;
                continue;
            }

            let request_body = serde_json::to_string(&serde_json::json!({
                "prompt": prompt_text,
                "max_tokens": config.teacher.max_tokens,
                "temperature": config.teacher.temperature,
                "strategy": "top_p",
                "top_p": config.teacher.top_p,
            }))
            .expect("JSON serialization cannot fail");

            let (resp, was_skipped) =
                send_generate_request(&generate_url, &request_body, i, json_output);
            if was_skipped {
                skipped_count += 1;
            }
            let Some(resp) = resp else {
                continue;
            };

            let gen_result: serde_json::Value = {
                let body = resp.into_string().map_err(|e| {
                    CliError::NetworkError(format!("Failed to read response body: {e}"))
                })?;
                serde_json::from_str(&body).map_err(|e| {
                    CliError::NetworkError(format!("Invalid generate response: {e}"))
                })?
            };

            let num_tokens = gen_result
                .get("num_generated")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            let text = gen_result
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if num_tokens < u64::from(config.synthetic_data.min_completion_tokens) {
                skipped_count += 1;
                continue;
            }

            // Write output JSONL record
            let record = serde_json::json!({
                "prompt": prompt_text,
                "completion": text,
                "tokens": num_tokens,
                "sample_idx": sample_idx,
                "source": prompt_json.get("source").and_then(|v| v.as_str()).unwrap_or(""),
                "kind": prompt_json.get("kind").and_then(|v| v.as_str()).unwrap_or(""),
            });
            writeln!(
                writer,
                "{}",
                serde_json::to_string(&record).map_err(|e| CliError::ValidationFailed(format!(
                    "JSON serialize error: {e}"
                )))?
            )?;
            writer.flush()?;

            resume.total_tokens += num_tokens;
            resume.generated_count += 1;
            progress_counter += 1;

            if progress_counter % 10 == 0 && !json_output {
                let elapsed = start_time.elapsed().as_secs_f64();
                let tok_per_sec = if elapsed > 0.0 {
                    resume.total_tokens as f64 / elapsed
                } else {
                    0.0
                };
                let total_work = prompts.len() as u64 * u64::from(samples_per_prompt);
                println!(
                    "  [sample {}/{}, prompt {}/{}; {}/{} completions] {} tokens ({:.0} tok/s), {} skipped",
                    sample_idx + 1,
                    samples_per_prompt,
                    i + 1,
                    prompts.len(),
                    resume.generated_count,
                    total_work,
                    resume.total_tokens,
                    tok_per_sec,
                    skipped_count
                );
            }
        }
    }

    writer.flush()?;

    // Shutdown server
    let _ = server.kill();
    let _ = server.wait();

    format_generate_result(
        config,
        prompts.len(),
        resume.generated_count,
        skipped_count,
        resume.total_tokens,
        target,
        start_time.elapsed(),
        json_output,
    );

    Ok(())
}
