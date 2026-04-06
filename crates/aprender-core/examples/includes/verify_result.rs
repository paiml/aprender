/// Run test based on modality (PMAT-QA-PROTOCOL-001 §7.4)
fn run_modality_test(
    config: &Config,
    cell: &MatrixCell,
    prompt: &str,
    max_tokens: u32,
) -> Result<String, String> {
    let max_tokens_str = max_tokens.to_string();

    match cell.modality {
        Modality::Run => {
            let mut args: Vec<&str> = vec![
                "run",
                &cell.model_uri,
                "--prompt",
                prompt,
                "--max-tokens",
                &max_tokens_str,
            ];
            if let Some(flag) = cell.backend.flag() {
                args.push(flag);
            }
            if cell.with_trace {
                args.push("--trace");
            }
            run_apr(config, &args)
        }
        Modality::Chat => run_chat_test(
            config,
            &cell.model_uri,
            prompt,
            cell.backend,
            cell.with_trace,
            DEFAULT_TIMEOUT,
        ),
        Modality::Serve => run_serve_test(
            config,
            &cell.model_uri,
            prompt,
            cell.backend,
            cell.with_trace,
            DEFAULT_TIMEOUT,
        ),
    }
}

/// Strip ANSI escape sequences from a string (e.g., \x1b[1;32m → "")
fn strip_ansi(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip ESC [ ... m sequences
            if chars.next() == Some('[') {
                for c2 in chars.by_ref() {
                    if c2.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else {
            result.push(c);
        }
    }
    result
}

/// Extract just the model output from apr run output (between "Output:" and "Completed in")
/// This filters out compilation warnings, paths, and timing info that might contain false positives.
/// Handles ANSI color codes in output (e.g., \x1b[1;32mOutput:\x1b[0m)
fn extract_output(raw: &str) -> String {
    let lines: Vec<&str> = raw.lines().collect();
    let mut in_output = false;
    let mut content = Vec::new();
    for line in lines {
        let clean = strip_ansi(line);
        if clean.starts_with("Output:") {
            in_output = true;
            continue;
        }
        if clean.starts_with("Completed in ") {
            break;
        }
        if in_output {
            content.push(strip_ansi(line));
        }
    }
    content.join("\n").trim().to_string()
}

/// Garbage patterns that indicate model collapse or tokenization failure
/// (PMAT-QA-PROTOCOL-001 §7.5)
const GARBAGE_PATTERNS: &[&str] = &[
    "\u{FFFD}", // Replacement character (encoding error)
    "[UNK]",    // Unknown token marker
    "akunji",   // Known GQA bug garbage
    "olumbia",  // Known layout bug garbage
    "专门窗",   // Known GQA bug CJK garbage
    "token0",   // Raw token ID leak
    "token1",   // Raw token ID leak
    "<0x",      // Byte token leak (e.g., <0x0A>)
];

/// BPE artifacts that indicate incomplete detokenization
const BPE_ARTIFACTS: &[char] = &[
    'Ġ', // GPT-2 style space prefix
    'Ċ', // GPT-2 style newline
    'ĉ', // GPT-2 style tab
];

/// Verification result for output inspection (PMAT-QA-PROTOCOL-001 §7.5)
#[derive(Debug)]
enum VerifyResult {
    /// Captures verified output for potential debugging/logging
    #[allow(dead_code)]
    Pass(String),
    FailEmpty,
    FailGarbage(String),
    FailBpeArtifact(char),
    FailMissingAnswer(String),
}

/// Verify output is correct: not empty, no garbage, contains expected answer
/// (PMAT-QA-PROTOCOL-001 §7.5)
///
/// Order of checks is CRITICAL (fail fast on garbage):
/// 1. Not empty
/// 2. No garbage patterns (BEFORE checking answer)
/// 3. No BPE artifacts
/// 4. Contains expected answer
fn verify_output(output: &str, expected_contains: Option<&str>) -> VerifyResult {
    let trimmed = output.trim();

    // 1. Empty check
    if trimmed.is_empty() {
        return VerifyResult::FailEmpty;
    }

    // 2. Garbage detection (FAIL FAST - before answer check)
    for pattern in GARBAGE_PATTERNS {
        if trimmed.contains(pattern) {
            return VerifyResult::FailGarbage((*pattern).to_string());
        }
    }

    // 3. BPE artifact check
    for &artifact in BPE_ARTIFACTS {
        if trimmed.contains(artifact) {
            return VerifyResult::FailBpeArtifact(artifact);
        }
    }

    // 4. Expected answer check with word boundary validation
    // (Fixed: PMAT-098 Red Team falsification found naive substring matching is brittle)
    if let Some(expected) = expected_contains {
        if !contains_as_word(trimmed, expected) {
            return VerifyResult::FailMissingAnswer(format!(
                "Expected '{}' as standalone word, got: {}",
                expected,
                trimmed.chars().take(50).collect::<String>()
            ));
        }
    }

    VerifyResult::Pass(trimmed.to_string())
}

/// Check if `needle` appears in `haystack` as a standalone word (not embedded in another word/number)
/// This prevents false positives like "14" matching expected "4"
fn contains_as_word(haystack: &str, needle: &str) -> bool {
    // Find all occurrences and check word boundaries
    let mut search_start = 0;
    while let Some(pos) = haystack[search_start..].find(needle) {
        let abs_pos = search_start + pos;
        let end_pos = abs_pos + needle.len();

        // Check left boundary: start of string OR non-alphanumeric
        let left_ok = abs_pos == 0 || {
            let prev_char = haystack[..abs_pos]
                .chars()
                .last()
                .expect("non-empty prefix must have a last char");
            !prev_char.is_alphanumeric()
        };

        // Check right boundary: end of string OR non-alphanumeric
        let right_ok = end_pos >= haystack.len() || {
            let next_char = haystack[end_pos..]
                .chars()
                .next()
                .expect("non-empty suffix must have a next char");
            !next_char.is_alphanumeric()
        };

        if left_ok && right_ok {
            return true;
        }

        // Continue searching after this occurrence
        search_start = abs_pos + 1;
        if search_start >= haystack.len() {
            break;
        }
    }
    false
}

/// Convert VerifyResult to TestResult for a named test
fn verify_to_test(name: &'static str, max_points: u32, result: VerifyResult) -> TestResult {
    match result {
        VerifyResult::Pass(_) => TestResult::pass(name, max_points, "Clean output".to_string()),
        VerifyResult::FailEmpty => TestResult::fail(name, max_points, "Empty output".to_string()),
        VerifyResult::FailGarbage(p) => {
            TestResult::fail(name, max_points, format!("GARBAGE: '{p}'"))
        }
        VerifyResult::FailBpeArtifact(c) => {
            TestResult::fail(name, max_points, format!("BPE artifact: '{c}'"))
        }
        VerifyResult::FailMissingAnswer(msg) => TestResult::fail(name, max_points, msg),
    }
}

/// Run output verification test: run model, extract output, verify quality
fn run_verify_test(
    config: &Config,
    cell: &MatrixCell,
    name: &'static str,
    max_points: u32,
    prompt: &str,
    max_tokens: u32,
    expected: Option<&str>,
) -> TestResult {
    match run_modality_test(config, cell, prompt, max_tokens) {
        Ok(raw) => verify_to_test(
            name,
            max_points,
            verify_output(&extract_output(&raw), expected),
        ),
        Err(e) => TestResult::fail(name, max_points, e),
    }
}

/// Run performance test: measure tok/s against threshold
fn run_perf_test(config: &Config, cell: &MatrixCell) -> TestResult {
    let perf_start = Instant::now();
    match run_modality_test(config, cell, "Count from 1 to 20.", 50) {
        Ok(output) => {
            let elapsed = perf_start.elapsed().as_secs_f64();
            let tokens_est = (output.split_whitespace().count() as f64 * 1.3).max(10.0);
            let tps = tokens_est / elapsed;
            let base_target = match (cell.backend, cell.format) {
                (Backend::Cpu, _) => config.min_cpu_tps,
                (Backend::Gpu, Format::SafeTensors) => config.min_gpu_tps_float32,
                (Backend::Gpu, _) => config.min_gpu_tps,
            };
            let target = match cell.modality {
                Modality::Run => base_target,
                Modality::Chat | Modality::Serve => base_target * 0.5,
            };
            if tps >= target {
                TestResult::pass("Performance", 3, format!("{tps:.1} tok/s >= {target:.1}"))
            } else {
                TestResult::fail("Performance", 3, format!("{tps:.1} tok/s < {target:.1}"))
            }
        }
        Err(e) => TestResult::fail("Performance", 3, e),
    }
}

/// Run all tests for a single matrix cell
/// Dispatches to run_modality_test for Chat/Serve modalities (PMAT-QA-PROTOCOL-001 §7.4)
fn run_cell_tests(config: &Config, cell: &MatrixCell) -> CellResult {
    let start = Instant::now();
    let mut tests = Vec::new();

    if cell.backend == Backend::Gpu && !gpu_available() {
        tests.push(TestResult::skip(
            "All Tests",
            15,
            "No GPU available".to_string(),
        ));
        return CellResult {
            cell: cell.clone(),
            tests,
            total_points: 0,
            max_points: 15,
            elapsed: start.elapsed(),
        };
    }

    // Test 1: Model loads (2 points)
    match run_modality_test(
        config,
        cell,
        "What is 2+2? Answer with just the number.",
        10,
    ) {
        Ok(_) => tests.push(TestResult::pass(
            "Model Load",
            2,
            format!("{} via {:?}", cell.model_uri, cell.modality),
        )),
        Err(e) => {
            tests.push(TestResult::fail("Model Load", 2, e));
            return CellResult {
                cell: cell.clone(),
                tests,
                total_points: 0,
                max_points: 15,
                elapsed: start.elapsed(),
            };
        }
    }

    // Test 2: Correct output (3 points)
    tests.push(run_verify_test(
        config,
        cell,
        "Correct Output",
        3,
        "What is 2+2? Answer with just the number.",
        10,
        Some("4"),
    ));

    // Test 3: No garbage (3 points)
    tests.push(run_verify_test(
        config,
        cell,
        "No Garbage",
        3,
        "Say hello.",
        20,
        None,
    ));

    // Test 4: No BPE artifacts (2 points)
    match run_modality_test(config, cell, "Say hello.", 20) {
        Ok(raw) => {
            let output = extract_output(&raw);
            let has_bpe = BPE_ARTIFACTS.iter().any(|&c| output.contains(c));
            tests.push(if has_bpe {
                TestResult::fail("No BPE Artifacts", 2, "Ġ/Ċ/ĉ detected".to_string())
            } else {
                TestResult::pass("No BPE Artifacts", 2, "Clean tokens".to_string())
            });
        }
        Err(e) => tests.push(TestResult::fail("No BPE Artifacts", 2, e)),
    }

    // Test 5: Trace works (2 points)
    let trace_cell = MatrixCell {
        with_trace: true,
        ..cell.clone()
    };
    match run_modality_test(config, &trace_cell, "Hi", 5) {
        Ok(_) => tests.push(TestResult::pass(
            "Trace Works",
            2,
            format!("{:?} + trace accepted", cell.modality),
        )),
        Err(e) if e.contains("not supported") || e.contains("trace") => {
            tests.push(TestResult::skip(
                "Trace Works",
                2,
                format!("Trace not supported for {:?}", cell.modality),
            ));
        }
        Err(e) => tests.push(TestResult::fail("Trace Works", 2, e)),
    }

    // Test 6: Performance (3 points)
    tests.push(run_perf_test(config, cell));

    let total: u32 = tests.iter().map(|t| t.points).sum();
    let max: u32 = tests.iter().map(|t| t.max_points).sum();
    CellResult {
        cell: cell.clone(),
        tests,
        total_points: total,
        max_points: max,
        elapsed: start.elapsed(),
    }
}

fn print_cell_result(result: &CellResult) {
    let status = if result.passed() {
        format!("{}✓ PASS{}", GREEN, NC)
    } else {
        format!("{}✗ FAIL{}", RED, NC)
    };

    println!();
    println!(
        "{}┌─────────────────────────────────────────────────────────────┐{}",
        BLUE, NC
    );
    println!(
        "{}│{} {} {:<42} {:>8} {}│{}",
        BLUE,
        NC,
        BOLD,
        result.cell.label(),
        status,
        BLUE,
        NC
    );
    println!(
        "{}├─────────────────────────────────────────────────────────────┤{}",
        BLUE, NC
    );

    for test in &result.tests {
        let icon = if test.passed {
            format!("{}✓{}", GREEN, NC)
        } else {
            format!("{}✗{}", RED, NC)
        };
        let points = format!("{}/{}", test.points, test.max_points);
        let detail = test.details.as_deref().unwrap_or("");
        println!(
            "{}│{} {} {:<20} {:>5}  {:<25}{}│{}",
            BLUE,
            NC,
            icon,
            test.name,
            points,
            detail.chars().take(25).collect::<String>(),
            BLUE,
            NC
        );
    }

    println!(
        "{}├─────────────────────────────────────────────────────────────┤{}",
        BLUE, NC
    );
    println!(
        "{}│{} Total: {}/{} points ({:.1}s) {:>24}{}│{}",
        BLUE,
        NC,
        result.total_points,
        result.max_points,
        result.elapsed.as_secs_f64(),
        "",
        BLUE,
        NC
    );
    println!(
        "{}└─────────────────────────────────────────────────────────────┘{}",
        BLUE, NC
    );
}

