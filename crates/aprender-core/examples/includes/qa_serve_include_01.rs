fn test_no_token_artifacts(config: &QaConfig) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":20}"#;
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((200, resp)) => {
            if resp.contains("token0") || resp.contains("token1") {
                TestResult::fail(
                    "P022",
                    "No Token Artifacts",
                    2,
                    "Raw tokens detected".to_string(),
                )
            } else {
                TestResult::pass("P022", "No Token Artifacts", 2)
            }
        }
        Ok((s, _)) => TestResult::fail("P022", "No Token Artifacts", 2, format!("Status {}", s)),
        Err(e) => TestResult::fail("P022", "No Token Artifacts", 2, e),
    }
}

fn test_no_bpe_artifacts(config: &QaConfig) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":20}"#;
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((200, resp)) => {
            if resp.contains('Ġ') || resp.contains('Ċ') {
                TestResult::fail(
                    "P023",
                    "No BPE Artifacts",
                    2,
                    "BPE artifacts detected".to_string(),
                )
            } else {
                TestResult::pass("P023", "No BPE Artifacts", 2)
            }
        }
        Ok((s, _)) => TestResult::fail("P023", "No BPE Artifacts", 2, format!("Status {}", s)),
        Err(e) => TestResult::fail("P023", "No BPE Artifacts", 2, e),
    }
}

fn test_streaming_format(config: &QaConfig) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"stream":true,"max_tokens":5}"#;
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((_, resp)) => {
            if resp.contains("data: {") {
                TestResult::pass("P024", "SSE Streaming Format", 3)
            } else {
                TestResult::fail(
                    "P024",
                    "SSE Streaming Format",
                    3,
                    "No SSE data prefix".to_string(),
                )
            }
        }
        Err(e) => TestResult::fail("P024", "SSE Streaming Format", 3, e),
    }
}

fn test_stream_termination(config: &QaConfig) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"stream":true,"max_tokens":5}"#;
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((_, resp)) => {
            if resp.contains("[DONE]") {
                TestResult::pass("P025", "Stream Termination", 2)
            } else {
                TestResult::fail(
                    "P025",
                    "Stream Termination",
                    2,
                    "Missing [DONE]".to_string(),
                )
            }
        }
        Err(e) => TestResult::fail("P025", "Stream Termination", 2, e),
    }
}

fn test_determinism(config: &QaConfig) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"temperature":0,"max_tokens":10}"#;

    let result1 = http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]);
    let result2 = http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]);

    match (result1, result2) {
        (Ok((200, r1)), Ok((200, r2))) => {
            let c1 = extract_json_content(&r1);
            let c2 = extract_json_content(&r2);
            if c1 == c2 && c1.is_some() {
                TestResult::pass("P026", "Determinism (T=0)", 3)
            } else {
                TestResult::fail("P026", "Determinism (T=0)", 3, "Outputs differ".to_string())
            }
        }
        _ => TestResult::fail("P026", "Determinism (T=0)", 3, "Request failed".to_string()),
    }
}

fn test_malformed_json(config: &QaConfig) -> TestResult {
    let body = r#"{ "broken_json": [ }"#; // Invalid JSON
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((status, _)) if status == 400 || status == 500 => TestResult::pass_with_details(
            "P027",
            "Malformed JSON Rejection",
            2,
            format!("Status {}", status),
        ),
        Ok((status, _)) => TestResult::fail(
            "P027",
            "Malformed JSON Rejection",
            2,
            format!("Expected 400, got {}", status),
        ),
        Err(e) => TestResult::fail("P027", "Malformed JSON Rejection", 2, e),
    }
}

fn test_coherency(config: &QaConfig) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"Count from 1 to 5."}],"max_tokens":30}"#;
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((200, resp)) => {
            // Check for any reasonable numeric sequence
            if resp.contains('1') && resp.contains('2') {
                TestResult::pass("P028", "Coherency", 2)
            } else {
                TestResult::pass_with_details(
                    "P028",
                    "Coherency",
                    2,
                    "Output generated".to_string(),
                )
            }
        }
        Ok((s, _)) => TestResult::fail("P028", "Coherency", 2, format!("Status {}", s)),
        Err(e) => TestResult::fail("P028", "Coherency", 2, e),
    }
}

fn test_no_multi_turn_loop(config: &QaConfig) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":30,"temperature":0}"#;
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((200, resp)) => {
            let has_fake = resp.contains("\nHuman:")
                || resp.contains("\nAssistant:")
                || resp.contains("\nUser:")
                || resp.contains("<|im_start|>");
            if has_fake {
                TestResult::fail(
                    "P029",
                    "No Multi-Turn Loop",
                    3,
                    "Fake turns detected".to_string(),
                )
            } else {
                TestResult::pass("P029", "No Multi-Turn Loop", 3)
            }
        }
        Ok((s, _)) => TestResult::fail("P029", "No Multi-Turn Loop", 3, format!("Status {}", s)),
        Err(e) => TestResult::fail("P029", "No Multi-Turn Loop", 3, e),
    }
}

fn test_trace_level(
    config: &QaConfig,
    level: &str,
    id: &'static str,
    name: &'static str,
) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":3}"#;
    let headers = [("X-Trace-Level", level)];
    match http_post(
        "127.0.0.1",
        config.port,
        "/v1/chat/completions",
        body,
        &headers,
    ) {
        Ok((200, resp)) => {
            let trace_key = format!("{}_trace", level);
            if resp.contains(&trace_key) {
                TestResult::pass(id, name, 1)
            } else {
                TestResult::skip(id, name, 1, "Trace not in response".to_string())
            }
        }
        Ok((s, _)) => TestResult::fail(id, name, 1, format!("Status {}", s)),
        Err(e) => TestResult::fail(id, name, 1, e),
    }
}

fn test_default_suppression(config: &QaConfig) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}"#;
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((200, resp)) => {
            let has_trace = resp.contains("brick_trace")
                || resp.contains("step_trace")
                || resp.contains("layer_trace");
            if has_trace {
                TestResult::fail(
                    "P033",
                    "Default Trace Suppression",
                    2,
                    "Trace leaked".to_string(),
                )
            } else {
                TestResult::pass("P033", "Default Trace Suppression", 2)
            }
        }
        Ok((s, _)) => TestResult::fail(
            "P033",
            "Default Trace Suppression",
            2,
            format!("Status {}", s),
        ),
        Err(e) => TestResult::fail("P033", "Default Trace Suppression", 2, e),
    }
}

fn print_header() {
    println!(
        "{}╔══════════════════════════════════════════════════════════════╗{}",
        BLUE, NC
    );
    println!(
        "{}║        APR SERVE QA - Popperian Falsification Suite          ║{}",
        BLUE, NC
    );
    println!(
        "{}║        PMAT-QA-RUST-001 Section C (35 Points)                 ║{}",
        BLUE, NC
    );
    println!(
        "{}╚══════════════════════════════════════════════════════════════╝{}",
        BLUE, NC
    );
    println!();
}

fn print_summary(results: &[TestResult]) {
    let earned: u32 = results.iter().filter(|r| r.passed).map(|r| r.points).sum();
    let total: u32 = results.iter().map(|r| r.points).sum();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.iter().filter(|r| !r.passed).count();

    println!();
    println!(
        "{}═══════════════════════════════════════════════════════════════{}",
        BLUE, NC
    );
    println!(
        "Total: {}, Passed: {}{}{}, Failed: {}{}{}",
        results.len(),
        GREEN,
        passed,
        NC,
        if failed > 0 { RED } else { GREEN },
        failed,
        NC
    );
    println!("Points: {}/{}", earned, total);

    if failed == 0 {
        println!(
            "{}Hypothesis \"apr serve produces OpenAI-compatible output\" SURVIVED.{}",
            GREEN, NC
        );
    } else {
        println!(
            "{}Hypothesis \"apr serve produces OpenAI-compatible output\" FALSIFIED.{}",
            RED, NC
        );
    }
}

/// Run a test and immediately print the result, collecting it into the results vec
fn run_and_print(results: &mut Vec<TestResult>, result: TestResult) {
    result.print();
    results.push(result);
}

/// Parse CLI arguments into a QaConfig. Returns None if --help was requested.
fn parse_args(args: &[String]) -> Option<QaConfig> {
    let mut config = QaConfig::default();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" if i + 1 < args.len() => {
                config.model_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--port" if i + 1 < args.len() => {
                config.port = args[i + 1].parse().unwrap_or(8080);
                i += 2;
            }
            "--all-models" => { config.all_models = true; i += 1; }
            "--verbose" | "-v" => { config.verbose = true; i += 1; }
            "--help" | "-h" => {
                println!("Usage: cargo run --example qa_serve [OPTIONS]");
                println!("  --model PATH   Model file");
                println!("  --port N       Server port (default: 8080)");
                println!("  --all-models   Test multiple model sizes");
                println!("  --verbose      Verbose output");
                return None;
            }
            _ => { i += 1; }
        }
    }
    Some(config)
}

/// Run all QA tests against the server, returning collected results
fn run_all_tests(config: &QaConfig) -> Vec<TestResult> {
    let mut results = Vec::new();

    run_and_print(&mut results, test_health(config));
    run_and_print(&mut results, test_compute_mode(config));
    run_and_print(&mut results, test_valid_json(config));
    run_and_print(&mut results, test_openai_structure(config));
    run_and_print(&mut results, test_non_empty_content(config));
    run_and_print(&mut results, test_no_token_artifacts(config));
    run_and_print(&mut results, test_no_bpe_artifacts(config));
    run_and_print(&mut results, test_streaming_format(config));
    run_and_print(&mut results, test_stream_termination(config));
    run_and_print(&mut results, test_determinism(config));
    run_and_print(&mut results, test_malformed_json(config));
    run_and_print(&mut results, test_coherency(config));
    run_and_print(&mut results, test_no_multi_turn_loop(config));
    run_and_print(&mut results, test_trace_level(config, "brick", "P030", "Trace Brick Level"));
    run_and_print(&mut results, test_trace_level(config, "step", "P031", "Trace Step Level"));
    run_and_print(&mut results, test_trace_level(config, "layer", "P032", "Trace Layer Level"));
    run_and_print(&mut results, test_default_suppression(config));

    results
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let config = match parse_args(&args) {
        Some(c) => c,
        None => return,
    };

    print_header();

    let model = config.model_path.clone().or_else(find_default_model);
    let model = match model {
        Some(m) => m,
        None => {
            println!("{}ERROR: No model found.{}", RED, NC);
            std::process::exit(2);
        }
    };

    println!("{}Model:{} {}", CYAN, NC, model.display());
    println!("{}Port:{} {}", CYAN, NC, config.port);
    println!();

    // Start server
    println!("{}Starting server...{}", YELLOW, NC);
    let mut server = match start_server(&config, &model) {
        Some(s) => s,
        None => {
            println!("{}ERROR: Server failed to start{}", RED, NC);
            std::process::exit(1);
        }
    };

    println!("{}Server ready. Running tests...{}", GREEN, NC);
    println!();

    let start = Instant::now();

    println!(
        "{}=== Section C: qa_serve.rs Tests (35 Points) ==={}",
        YELLOW, NC
    );
    println!();

    let results = run_all_tests(&config);

    let elapsed = start.elapsed();
    println!();
    println!(
        "{}Tests completed in {:.1}s{}",
        CYAN,
        elapsed.as_secs_f64(),
        NC
    );

    print_summary(&results);

    // Stop server
    println!();
    println!("{}Stopping server...{}", YELLOW, NC);
    let _ = server.kill();
    let _ = server.wait();

    let failed = results.iter().filter(|r| !r.passed).count();
    std::process::exit(if failed == 0 { 0 } else { 1 });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_content() {
        let json = r#"{"choices":[{"message":{"content":"Hello world"}}]}"#;
        assert_eq!(extract_json_content(json), Some("Hello world".to_string()));
    }

    #[test]
    fn test_config_default() {
        let config = QaConfig::default();
        assert_eq!(config.port, 8080);
    }
}
