/// Gate 1: Metadata Plausibility Validation (Bug 210, GH-222)
///
/// Validates that model hyperparameters (rope_theta, max_position_embeddings, rms_norm_eps)
/// fall within known plausible ranges for the detected architecture family.
///
/// This gate catches the root cause of GH-222: importing SafeTensors without config.json
/// silently stored rope_theta=10000.0 for Qwen2, which should be 1000000.0.
fn run_metadata_plausibility_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!(
            "{}",
            "Running metadata plausibility validation (Bug 210)...".yellow()
        );
    }

    // #3750: the 4-byte magic here, and that format's header below, never the whole model
    let magic = super::model_header::read_prefix(path, 4)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read model: {e}")))?;

    if magic.len() < 4 {
        let duration = start.elapsed();
        return Ok(GateResult::failed(
            "metadata_plausibility",
            "File too small for metadata extraction",
            None,
            None,
            duration,
        ));
    }

    let (architecture, rope_theta, max_pos, rms_norm_eps) = extract_model_metadata(&magic, path)?;

    let mut violations: Vec<String> = Vec::new();
    let mut tally = MetadataTally::default();

    check_rope_theta(
        architecture.as_deref(),
        rope_theta,
        &magic,
        &mut violations,
        &mut tally,
    );
    check_max_position_embeddings(max_pos, &mut violations, &mut tally);
    check_rms_norm_eps(rms_norm_eps, &mut violations, &mut tally);
    check_arch_theta_cross_validation(
        architecture.as_ref(),
        rope_theta,
        &mut violations,
        &mut tally,
    );

    let duration = start.elapsed();

    if violations.is_empty() {
        Ok(GateResult::passed(
            "metadata_plausibility",
            &format!(
                "{} (arch={}, rope_theta={}, max_pos={})",
                tally.summary(),
                architecture.as_deref().unwrap_or("unknown"),
                rope_theta.map_or("none".to_string(), |t| format!("{t}")),
                max_pos.map_or("none".to_string(), |p| format!("{p}")),
            ),
            Some(tally.passed as f64),
            Some(0.0),
            duration,
        ))
    } else {
        Ok(GateResult::failed(
            "metadata_plausibility",
            &format!(
                "{} metadata violation(s): {}",
                violations.len(),
                violations.join("; ")
            ),
            Some(violations.len() as f64),
            Some(0.0),
            duration,
        ))
    }
}

/// #3873: what the plausibility checks MEASURED, kept apart from what they could not
/// measure. An absent field is not a violation, but it is not a pass either: counting it
/// as one let a GGUF with no metadata at all report "4 metadata checks passed".
#[derive(Debug, Default)]
struct MetadataTally {
    passed: usize,
    absent: Vec<&'static str>,
}

impl MetadataTally {
    fn summary(&self) -> String {
        if self.absent.is_empty() {
            format!("{} metadata checks passed", self.passed)
        } else {
            format!(
                "{} metadata checks passed, {} not checked (absent: {})",
                self.passed,
                self.absent.len(),
                self.absent.join(", ")
            )
        }
    }
}

/// Return plausible rope_theta range for an architecture family.
fn rope_theta_range(arch: Option<&str>) -> (f64, f64, &'static str) {
    match arch {
        Some("qwen2" | "qwen2.5" | "qwen") => (
            100_000.0,
            f64::MAX,
            "expected ~1000000.0 (100x too low, will produce garbage)",
        ),
        Some("llama" | "llama2" | "llama3") => (1000.0, 10_000_000.0, "expected 10000-500000"),
        _ => (100.0, 100_000_000.0, "outside plausible range [100, 100M]"),
    }
}

/// Check rope_theta plausibility per architecture family.
fn check_rope_theta(
    arch: Option<&str>,
    rope_theta: Option<f32>,
    data: &[u8],
    violations: &mut Vec<String>,
    tally: &mut MetadataTally,
) {
    let Some(theta) = rope_theta else {
        if data.len() >= 4 && &data[0..4] == b"GGUF" {
            tally.absent.push("rope_theta");
        } else {
            violations.push("rope_theta missing from APR metadata".to_string());
        }
        return;
    };
    let theta_f64 = f64::from(theta);
    let (min, max, msg) = rope_theta_range(arch);
    if theta_f64 >= min && theta_f64 <= max {
        tally.passed += 1;
    } else {
        violations.push(format!(
            "rope_theta={theta} for {} — {msg}",
            arch.unwrap_or("unknown")
        ));
    }
}

/// Check max_position_embeddings is within plausible range.
fn check_max_position_embeddings(
    max_pos: Option<usize>,
    violations: &mut Vec<String>,
    tally: &mut MetadataTally,
) {
    if let Some(val) = max_pos {
        if (128..=1_048_576).contains(&val) {
            tally.passed += 1;
        } else {
            violations.push(format!(
                "max_position_embeddings={val} outside plausible range [128, 1M]"
            ));
        }
    } else {
        tally.absent.push("max_position_embeddings");
    }
}

/// Check rms_norm_eps is within plausible range.
fn check_rms_norm_eps(
    rms_norm_eps: Option<f32>,
    violations: &mut Vec<String>,
    tally: &mut MetadataTally,
) {
    if let Some(eps) = rms_norm_eps {
        let eps_f64 = f64::from(eps);
        if eps_f64 <= 0.0 || eps_f64 > 0.01 {
            violations.push(format!(
                "rms_norm_eps={eps} outside plausible range (0, 0.01]"
            ));
        } else {
            tally.passed += 1;
        }
    } else {
        tally.absent.push("rms_norm_eps");
    }
}

/// Cross-validate architecture against rope_theta (Bug 210 signature detection).
fn check_arch_theta_cross_validation(
    architecture: Option<&String>,
    rope_theta: Option<f32>,
    violations: &mut Vec<String>,
    tally: &mut MetadataTally,
) {
    if let (Some(arch), Some(theta)) = (architecture, rope_theta) {
        let theta_f64 = f64::from(theta);
        let suspicious = matches!(arch.as_str(), "qwen2" | "qwen2.5" | "qwen")
            && (theta_f64 - 10000.0).abs() < 1.0;
        if suspicious {
            violations.push(format!(
                "CRITICAL: {arch} with rope_theta=10000.0 — \
                 likely missing config.json (Bug 210)"
            ));
        } else {
            tally.passed += 1;
        }
    } else {
        tally.absent.push("arch/rope_theta cross-check");
    }
}

/// Metadata extracted from model file for plausibility validation.
type ModelMetadata = (Option<String>, Option<f32>, Option<usize>, Option<f32>);

/// Extract model metadata (GGUF, APR, or SafeTensors format) given the file's magic.
///
/// #3750: each format is read to the end of its header and no further: the GGUF header
/// prefix, the APR header + metadata + tensor index, or SafeTensors' sibling config.json.
fn extract_model_metadata(magic: &[u8], path: &Path) -> Result<ModelMetadata> {
    let magic = &magic[0..4];

    if magic == b"GGUF" {
        // GGUF format: use GgufReader, over the header prefix
        let reader = super::model_header::gguf_header(path)
            .map_err(|e| CliError::ValidationFailed(format!("GGUF parse failed: {e}")))?;
        let arch = reader.architecture();
        let rope_theta = reader.rope_theta();
        let max_pos = reader.context_length();
        let rms_norm_eps = reader.rms_norm_eps();
        Ok((arch, rope_theta, max_pos, rms_norm_eps))
    } else if &magic[0..3] == b"APR" || magic == b"APRN" {
        // APR format: parse v2 header + JSON metadata
        use aprender::format::v2::AprV2Reader;
        let prefix = super::model_header::apr_header_prefix(path)
            .map_err(|e| CliError::ValidationFailed(format!("APR parse failed: {e}")))?;
        let reader = AprV2Reader::from_bytes(&prefix)
            .map_err(|e| CliError::ValidationFailed(format!("APR parse failed: {e}")))?;
        let meta = reader.metadata();
        let _ = path;
        Ok((
            meta.architecture.clone(),
            meta.rope_theta,
            meta.max_position_embeddings,
            meta.rms_norm_eps,
        ))
    } else {
        // SafeTensors or unknown format: try to load config.json from sibling (GAP-UX-002)
        let config_path = {
            #[cfg(feature = "inference")]
            {
                realizar::safetensors::find_sibling_file(path, "config.json")
            }
            #[cfg(not(feature = "inference"))]
            {
                // Without realizar, manually look for sibling config.json
                path.parent()
                    .map(|dir| dir.join("config.json"))
                    .filter(|p| p.exists())
            }
        };
        if let Some(config_path) = config_path {
            // Read architecture and rope_theta from HF config.json
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| CliError::ValidationFailed(format!("config.json read failed: {e}")))?;
            let arch = extract_json_string(&config_str, "model_type");
            let rope_theta = extract_json_f32(&config_str, "rope_theta");
            let max_pos = extract_json_usize(&config_str, "max_position_embeddings");
            let rms_norm_eps = extract_json_f32(&config_str, "rms_norm_eps");
            Ok((arch, rope_theta, max_pos, rms_norm_eps))
        } else {
            // No config.json — return None for all fields (gate will note the gap)
            Ok((None, None, None, None))
        }
    }
}

/// Extract a string value from JSON by key (simple parser, no serde dependency).
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let idx = json.find(&pattern)?;
    let after_key = &json[idx + pattern.len()..];
    // Skip whitespace and colon
    let after_colon = after_key.find(':').map(|i| &after_key[i + 1..])?;
    let trimmed = after_colon.trim_start();
    if trimmed.starts_with('"') {
        let start = 1;
        let end = trimmed[start..].find('"')?;
        Some(trimmed[start..start + end].to_string())
    } else {
        None
    }
}

/// Extract an f32 value from JSON by key.
fn extract_json_f32(json: &str, key: &str) -> Option<f32> {
    let pattern = format!("\"{key}\"");
    let idx = json.find(&pattern)?;
    let after_key = &json[idx + pattern.len()..];
    let after_colon = after_key.find(':').map(|i| &after_key[i + 1..])?;
    let trimmed = after_colon.trim_start();
    // Parse until next comma, brace, or whitespace
    let end = trimmed.find([',', '}', '\n'])?;
    trimmed[..end].trim().parse::<f32>().ok()
}

/// Extract a usize value from JSON by key.
fn extract_json_usize(json: &str, key: &str) -> Option<usize> {
    let pattern = format!("\"{key}\"");
    let idx = json.find(&pattern)?;
    let after_key = &json[idx + pattern.len()..];
    let after_colon = after_key.find(':').map(|i| &after_key[i + 1..])?;
    let trimmed = after_colon.trim_start();
    let end = trimmed.find([',', '}', '\n'])?;
    trimmed[..end].trim().parse::<usize>().ok()
}

/// Output verification result (PMAT-QA-PROTOCOL-001 §7.4)
#[derive(Debug, Clone)]
pub enum OutputVerification {
    /// Output passed all checks
    Pass,
    /// Output failed verification
    Fail {
        /// Reason for failure
        reason: String,
    },
}

/// Statistical gibberish detection — returns Some(reason) if output is junk.
///
/// Three signals; any one triggers rejection:
/// 1. **Non-ASCII saturation**: For ASCII-prompt completions, > 60% non-ASCII
///    bytes is a strong gibberish indicator. English+code answers are ASCII-heavy.
/// 2. **Repeated-fragment detection**: A 4+ byte substring appearing 3+ times
///    consecutively (e.g. "udaÅĤo udaÅĤo udaÅĤo") flags BPE/loop pathologies.
/// 3. **Replacement-character density**: > 1 U+FFFD per 32 chars indicates
///    repeated UTF-8 decode failures.
///
/// All thresholds are conservative — coherent English/code outputs pass cleanly,
/// while the Qwen2-0.5B observed gibberish ("ëĸ» Ãĥ pÃ³Åº zwiÄħzku") is rejected.
///
/// The three signals are one function each and are tried in order, so the whole
/// check stays under the pre-commit cognitive-complexity threshold; the order
/// and the thresholds are unchanged.
fn detect_gibberish(output: &str, test_id: &str) -> Option<String> {
    gibberish_non_ascii_saturation(output, test_id)
        .or_else(|| gibberish_repeated_fragment(output, test_id))
        .or_else(|| gibberish_replacement_density(output, test_id))
        .or_else(|| gibberish_dominant_character(output, test_id))
}

/// Signal 4 (#3782): a degenerate completion — 8+ non-space characters, 90%+ of
/// them the SAME character.
///
/// Signal 2 only examines an output once it is 12 bytes long and only in 4-byte
/// fragments, so everything from 1 to 11 characters of a single repeated
/// character reached the answer check untouched. That matters because two golden
/// patterns are a SINGLE character: the greeting case lists `"!"`, and the
/// arithmetic case's whole expected answer is `"4"`. Substring-any then scores
/// a dead-logit loop as correct:
///
/// * `"!!!!!!!!"` satisfies the greeting case — `!` is token id 0 in the Qwen
///   vocab, which is exactly what a model with dead logits emits, and #3726's
///   non-ASCII-to-id-0 shape lands here too.
/// * `"44444444"` satisfies the ARITHMETIC case, which the issue does not
///   mention and which is the flagship golden test.
///
/// Dropping `"!"` fixes the first and cannot fix the second: `"4"` is the
/// legitimate answer to 2+2 and has to stay. So the check belongs here, ahead of
/// the answer check, where it covers every case including ones added later.
///
/// The threshold is CRUX's (#3774), deliberately, so the two judges cannot
/// disagree about what "degenerate" means on the same completion.
fn gibberish_dominant_character(output: &str, test_id: &str) -> Option<String> {
    let chars: Vec<char> = output.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.len() < 8 {
        return None;
    }
    let mut counts: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    for c in &chars {
        *counts.entry(*c).or_insert(0) += 1;
    }
    let (dominant, hits) = counts.into_iter().max_by_key(|&(_, n)| n)?;
    let ratio = hits as f64 / chars.len() as f64;
    if ratio >= 0.9 {
        return Some(format!(
            "{test_id}: degenerate output ({hits}/{} non-space characters are {dominant:?}, {:.0}% >= 90%)",
            chars.len(),
            ratio * 100.0
        ));
    }
    None
}

/// Signal 1: non-ASCII saturation (> 60% of a 16+ char completion).
fn gibberish_non_ascii_saturation(output: &str, test_id: &str) -> Option<String> {
    let total = output.chars().count();
    if total < 16 {
        return None;
    }
    let non_ascii = output.chars().filter(|c| !c.is_ascii()).count();
    let ratio = non_ascii as f64 / total as f64;
    if ratio > 0.6 {
        return Some(format!(
            "{test_id}: gibberish (non-ASCII ratio {:.1}% > 60%)",
            ratio * 100.0
        ));
    }
    None
}

/// Signal 2: a 4+ byte fragment repeated 3+ times in a row.
fn gibberish_repeated_fragment(output: &str, test_id: &str) -> Option<String> {
    let bytes = output.as_bytes();
    if bytes.len() >= 12 {
        let max_frag = 16.min(bytes.len() / 3);
        for frag_len in 4..=max_frag {
            let mut i = 0;
            while i + frag_len * 3 <= bytes.len() {
                let frag = &bytes[i..i + frag_len];
                if &bytes[i + frag_len..i + 2 * frag_len] == frag
                    && &bytes[i + 2 * frag_len..i + 3 * frag_len] == frag
                {
                    let preview = String::from_utf8_lossy(frag).into_owned();
                    return Some(format!(
                        "{test_id}: gibberish (fragment {preview:?} repeats 3+ times)"
                    ));
                }
                i += 1;
            }
        }
    }

    None
}

/// Signal 3: replacement-character density (> 1 U+FFFD per 32 chars).
fn gibberish_replacement_density(output: &str, test_id: &str) -> Option<String> {
    let total = output.chars().count();
    let fffd_count = output.matches('\u{FFFD}').count();
    if fffd_count > 0 && total >= 32 && fffd_count * 32 > total {
        return Some(format!(
            "{test_id}: gibberish (U+FFFD density {fffd_count}/{total} > 1/32)"
        ));
    }
    None
}

/// Verify output is correct: not empty, no garbage, contains expected answer
/// (PMAT-QA-PROTOCOL-001 §7.4)
///
/// Order of checks is CRITICAL (fail fast on garbage):
/// 1. Not empty
/// 2. No garbage patterns (BEFORE checking answer)
/// 3. No BPE artifacts
/// 4. Contains expected answer

/// Truncate so the reader can TELL (#3904).
///
/// This one cost a diagnosis. The golden gate's failure reason is the string a human
/// reads — hours later, out of a qa receipt, on a machine that cannot re-run the model —
/// and it was cut at 100 characters with nothing said. On #3914 the receipt read
///
///   got: '<s>[INST] What is the capital of France? [/INST]\n\n[S][INST] France is the
///        capital of France.\n\n[S][IN'
///
/// and the `[S]` could not be explained from it, because the explanation was in the
/// characters the gate had already produced and thrown away. Generation had to be
/// reproduced locally to read a string this message once held.
///
/// NOTE FOR ANYONE CHECKING COVERAGE: `check_no_silent_truncation.sh` did NOT catch this
/// and cannot. Its scan matches `[:N]` slice syntax, which is Python; Rust truncates with
/// `.chars().take(N)`. That gap is its own enumeration, not a patch to this fix.
fn loudly_truncated(s: &str, n: usize) -> String {
    let kept: String = s.chars().take(n).collect();
    let dropped = s.chars().count().saturating_sub(n);
    if dropped == 0 {
        kept
    } else {
        format!("{kept} ... and {dropped} more chars")
    }
}
pub fn verify_output(
    output: &str,
    test_id: &str,
    expected_patterns: &[&str],
) -> OutputVerification {
    // Check 1: Not empty
    if output.trim().is_empty() {
        return OutputVerification::Fail {
            reason: format!("{test_id}: Empty output"),
        };
    }

    // Check 2: Garbage patterns (fail fast BEFORE checking answer)
    let garbage_patterns = ["\u{FFFD}", "[UNK]", "akunji", "olumbia"];
    for pattern in &garbage_patterns {
        if output.contains(pattern) {
            return OutputVerification::Fail {
                reason: format!("{test_id}: Garbage detected: '{pattern}'"),
            };
        }
    }

    // Check 2.5: Statistical gibberish detection (Toyota Way root-cause fix).
    // The fixed garbage-pattern list above misses new defect classes — e.g.
    // Qwen2-0.5B-Instruct emits CJK/Polish/diacritic byte-fragments like
    // "udaÅĤo", "ëĸ»", "zwiÄħzku" that no prior pattern catches. We add three
    // statistical signals; ANY positive trip rejects the output.
    if let Some(reason) = detect_gibberish(output, test_id) {
        return OutputVerification::Fail { reason };
    }

    // Check 3: BPE artifacts (null bytes, excessive control chars)
    let null_count = output.bytes().filter(|&b| b == 0).count();
    if null_count > 0 {
        return OutputVerification::Fail {
            reason: format!("{test_id}: {null_count} null bytes detected (BPE artifact)"),
        };
    }

    // Check 4: Contains expected answer
    if !expected_patterns.is_empty() {
        let found = expected_patterns
            .iter()
            .any(|p| output.to_lowercase().contains(&p.to_lowercase()));
        if !found {
            return OutputVerification::Fail {
                reason: format!(
                    "{test_id}: Expected one of {:?}, got: '{}'",
                    expected_patterns,
                    loudly_truncated(output, 100)
                ),
            };
        }
    }

    OutputVerification::Pass
}

/// What the generated text held once complete `<think>...</think>` blocks were
/// removed: an answer, or a block that never closed.
///
/// #3724: the previous shape returned a `String` and, on an unclosed `<think>`,
/// truncated at it — so a model that was still reasoning when the budget ran out
/// produced `""`, which `verify_output` then reported as **"Empty output"**. On
/// lambda that is exactly what qwen3-8b-q4km did: 527 tokens, 15 of prompt plus
/// the entire 512-token budget, opening with `<think>` and still reasoning at the
/// cut. The gate reported an empty answer for a model that had answered nothing
/// *yet*. Those are different failures and the gate must not collapse them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThinkingSplit {
    /// Every `<think>` block closed (or there were none). This is the answer to judge.
    Answer(String),
    /// A `<think>` opened and no `</think>` followed: the budget ended inside the
    /// reasoning. Everything after the tag is chain-of-thought, never an answer,
    /// and the caller must say so by name rather than judge the empty remainder.
    Unclosed,
}

/// GH-279-4 / #3724: split model output into the answer and the reasoning.
///
/// Qwen3 thinking mode generates chain-of-thought inside `<think>` tags before the
/// actual answer. The golden output gate validates the ANSWER, not the reasoning.
///
/// Behavior:
/// - No `<think>` tags → `Answer`, passthrough (no-op for non-thinking models)
/// - Complete `<think>...</think>` → `Answer`, blocks removed
/// - Multiple blocks → `Answer`, all removed
/// - Unclosed `<think>` (budget exhausted during reasoning) → `Unclosed`
#[must_use]
pub fn split_thinking_blocks(output: &str) -> ThinkingSplit {
    let mut result = output.to_string();
    // Strip all complete <think>...</think> blocks
    while let (Some(start), Some(end)) = (result.find("<think>"), result.find("</think>")) {
        if end > start {
            result = format!("{}{}", &result[..start], &result[end + "</think>".len()..]);
        } else {
            break;
        }
    }
    // An opening tag with no closer: the budget ended mid-reasoning (#3724).
    if result.contains("<think>") {
        return ThinkingSplit::Unclosed;
    }
    ThinkingSplit::Answer(result.trim().to_string())
}

/// The reason a gate reports when generation ended inside a `<think>` block.
///
/// #3724 requires the budget to be named: "think block unclosed within N tokens"
/// tells the reader the model was still reasoning, which "Empty output" did not.
///
/// #3907: the closing clause used to read "check that the prompt is the one production
/// sends for this architecture". That was #3724's own suspect and #3724 REMOVED it —
/// `golden_prompt_for()` and `apr serve` both call `format_messages(.., Some(arch))`,
/// one rendering with nothing to diverge from. A reader who followed the hint spent an
/// hour proving a fixed defect stayed fixed. A DIAGNOSTIC THAT OUTLIVES THE DEFECT IT
/// NAMES SENDS EVERY FUTURE READER DOWN A DEAD PATH, so it now names the question that
/// is actually open: whether this model has a measured budget at all.
#[must_use]
pub fn unclosed_think_reason(leg: &str, budget: usize, generated_chars: usize) -> String {
    format!(
        "{leg}: think block unclosed within {budget} tokens \
         (the model was still reasoning at the budget; {generated_chars} chars generated, \
         no answer was reached). This is not an empty answer. Before treating it as a model \
         defect, check whether {budget} is MEASURED for this model in \
         contracts/thinking-budgets-v1.yaml or inherited from `default` — the default's basis \
         is one 8B model (#3907)."
    )
}

/// #3711: what the GPU half of one golden case came to.
///
/// An ERROR is a verdict about the GPU, never a skip. It used to be a skip: CUDA
/// init or generation failing went to `note_gpu_golden_skip`, the case went on to
/// judge only the CPU answer, and the gate said "N golden test cases passed". So
/// on a host where CUDA generation was broken, `golden_output` read GREEN. That
/// was absence scored as conformance, on the one gate that has to prove every
/// Q4_K model works on CUDA. Only a leg that never STARTED is not run, and it
/// says why.
/// #3870: the CPU leg's verdict, computed BEFORE the GPU message is composed.
///
/// It exists so `GpuGoldenLeg::failure_given_cpu` cannot claim anything about
/// the CPU leg that was not measured. The variants mirror the CPU path's own
/// outcomes exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CpuGoldenVerdict {
    /// The CPU answer matched the golden patterns.
    Passed,
    /// The CPU answer is wrong, and why.
    WrongAnswer(String),
    /// Still inside an unclosed `<think>` at the budget, so the CPU leg reached
    /// no verdict. Not a pass and not a wrong answer.
    Unclosed {
        /// The token budget the generation was given.
        budget: usize,
        /// How much text it produced without reaching an answer.
        generated_chars: usize,
    },
}

impl CpuGoldenVerdict {
    /// Judge the CPU leg's text with the same split-then-verify shape
    /// `GpuGoldenLeg::judge` uses for the GPU's, so the two legs cannot drift
    /// apart in how they decide.
    pub(crate) fn judge(
        output_text: &str,
        prompt: &str,
        expected_patterns: &[&str],
        budget: usize,
    ) -> Self {
        // GH-279-4: generate_with_cache returns prompt + generated tokens.
        let generated = output_text.strip_prefix(prompt).unwrap_or(output_text);
        let answer = match split_thinking_blocks(generated) {
            ThinkingSplit::Unclosed => {
                return Self::Unclosed {
                    budget,
                    generated_chars: generated.len(),
                }
            }
            ThinkingSplit::Answer(a) => a,
        };
        match verify_output(&answer, "golden_output", expected_patterns) {
            OutputVerification::Pass => Self::Passed,
            OutputVerification::Fail { reason } => Self::WrongAnswer(reason),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GpuGoldenLeg {
    /// Generated on the device, and the answer matches the golden patterns.
    Passed,
    /// Generated on the device, and the answer is wrong.
    WrongAnswer(String),
    /// CUDA init or generation ERRORED on a cuda build with a device.
    Errored(String),
    /// Generated on the device and was STILL REASONING when the budget ran out:
    /// a `<think>` block that never closed. #3724: this used to be truncated away
    /// and judged as an empty answer, so "the model answered nothing" and "the
    /// model had not finished thinking" were the same report. It is neither a
    /// wrong answer nor an error, and it is certainly not a skip.
    Unclosed {
        /// The token budget the generation was given.
        budget: usize,
        /// How much text it produced without reaching an answer.
        generated_chars: usize,
    },
    /// Never started, and why: no cuda feature, no device, not a GGUF, or judged
    /// by the runtime rung.
    NotRun(&'static str),
}

impl GpuGoldenLeg {
    /// Judge one GPU generation, given its decoded text or the error that stopped it.
    pub(crate) fn judge(
        generated: std::result::Result<String, String>,
        expected_patterns: &[&str],
        budget: usize,
    ) -> Self {
        let text = match generated {
            Err(e) => return Self::Errored(e),
            Ok(text) => text,
        };
        // GH-279-4 / #3724: split before judging. A block that never closed is its
        // own outcome; judging the remainder would report "Empty output" for a
        // model that had not finished reasoning.
        let answer = match split_thinking_blocks(&text) {
            ThinkingSplit::Unclosed => {
                return Self::Unclosed {
                    budget,
                    generated_chars: text.len(),
                }
            }
            ThinkingSplit::Answer(answer) => answer,
        };
        match verify_output(&answer, "golden_output_gpu", expected_patterns) {
            OutputVerification::Pass => Self::Passed,
            OutputVerification::Fail { reason } => Self::WrongAnswer(reason),
        }
    }

    /// `Some(reason)` fails the golden gate: a wrong answer, or an error.
    /// #3870: the GPU leg's failure message, composed WITH the CPU leg's verdict
    /// in hand.
    ///
    /// This used to be `failure(&self)` and it hardcoded "(CPU passed)" into the
    /// `WrongAnswer` arm. Nothing measured that. Worse, `validate_golden_test_case`
    /// returned on a GPU failure BEFORE the CPU pattern check ran, so the claim
    /// was made about a leg that had not been judged at all.
    ///
    /// That inverted a release verdict: `tinyllama-1.1b-chat-v1.0.Q4_K_M` was
    /// reported as a CUDA correctness defect blocking the tag under the
    /// all-Q4_K-on-CUDA rule, when it fails IDENTICALLY on CPU — the same
    /// `[S][INST]` loop, no "Paris", on both backends. One measurement and one
    /// string, read as two measurements.
    ///
    /// Taking the CPU verdict by argument is the point: the claim cannot be
    /// composed without it, so the ordering defect cannot return by someone
    /// reintroducing an early return.
    pub(crate) fn failure_given_cpu(&self, cpu: &CpuGoldenVerdict) -> Option<String> {
        match self {
            Self::Passed | Self::NotRun(_) => None,
            Self::WrongAnswer(reason) => Some(match cpu {
                // The #3477 signature, and now actually measured: GPU wrong,
                // CPU right. That discrimination is the reason this gate exists,
                // so it keeps its exact wording.
                CpuGoldenVerdict::Passed => format!("GPU output failed (CPU passed): {reason}"),
                CpuGoldenVerdict::WrongAnswer(cpu_reason) => format!(
                    "GPU output failed AND CPU failed too, so this is not a GPU-specific \
                     defect — CPU: {cpu_reason} — GPU: {reason}"
                ),
                CpuGoldenVerdict::Unclosed { budget, .. } => format!(
                    "GPU output failed and the CPU leg was still reasoning at its \
                     {budget}-token budget, so there is no CPU verdict to compare \
                     against — GPU: {reason}"
                ),
            }),
            Self::Errored(e) => Some(format!(
                "GPU golden generation ERRORED on a cuda build with a CUDA device: a broken \
                 GPU, not a skip (#3711): {e}"
            )),
            Self::Unclosed {
                budget,
                generated_chars,
            } => Some(unclosed_think_reason(
                "golden_output_gpu",
                *budget,
                *generated_chars,
            )),
        }
    }

    /// What the gate's pass message says about the GPU leg.
    pub(crate) fn describe(&self) -> String {
        match self {
            Self::NotRun(why) => format!("GPU leg SKIPPED: {why}"),
            _ => "GPU leg judged on CUDA device 0".to_string(),
        }
    }
}

/// Why the dense GPU leg will not start, or `None` when it will (#3711). These are
/// the only skips, and each names which.
pub(crate) fn gpu_golden_not_run(
    cuda_feature: bool,
    cuda_device: bool,
    gguf: bool,
) -> Option<&'static str> {
    if !cuda_feature {
        Some("this build has no cuda feature")
    } else if !cuda_device {
        Some("no CUDA device on this host")
    } else if !gguf {
        Some("the dense GPU leg judges GGUF only")
    } else {
        None
    }
}

/// #3711: the backend the runtime rung names is the one the dispatch REPORTED
/// (`InferenceResult::used_gpu`), never the one the build could have used. The
/// pass message used to say "GPU hybrid forward" on every cuda build, including
/// one on a host with no device. Worse, the hybrid's GPU forward falls back to the
/// CPU on any GPU failure, so a broken GPU passed on the CPU's answer.
///
/// `gpu_not_run` is `None` when the GPU should serve the model: a cuda build, a
/// device, and an architecture the GPU runs. `Err` then means the dispatch
/// reported CPU, and it fails the gate.
pub(crate) fn runtime_golden_backend(
    used_gpu: bool,
    gpu_not_run: Option<&'static str>,
) -> std::result::Result<String, String> {
    match (used_gpu, gpu_not_run) {
        (true, _) => Ok("GPU: the dispatch reported used_gpu=true".to_string()),
        (false, Some(why)) => Ok(format!(
            "CPU: the dispatch reported used_gpu=false, and the GPU was not expected ({why})"
        )),
        (false, None) => Err(
            "the GPU should have served this model (a cuda build, a CUDA \
             device, an architecture the GPU runs) and the dispatch reported CPU: the GPU \
             forward failed and fell back, and its reason is on stderr. A broken GPU, not a \
             pass (#3711)"
                .to_string(),
        ),
    }
}

/// JIDOKA: Validate GPU golden output matches expected patterns (PMAT-232 lesson).
///
/// Without this, GPU correctness was NEVER tested — `apr qa` golden output only ran CPU.
/// Called only when `gpu_golden_not_run` said the leg starts (a cuda build, a device,
/// a GGUF), so an init or generation error here is `Errored`, a gate FAIL (#3711).
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
fn validate_gpu_golden_output(
    mapped: &realizar::gguf::MappedGGUFModel,
    prompt_tokens: &[u32],
    gen_config: &realizar::gguf::QuantizedGenerateConfig,
    gguf: &realizar::gguf::GGUFModel,
    expected_patterns: &[&str],
    config: &QaConfig,
) -> Result<GpuGoldenLeg> {
    use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};
    // #3432 / #3477: the Qwen3.5 hybrid never reaches this dense-loader gate —
    // `golden_gate_for` routes it to `run_golden_output_gate_runtime`, which goes
    // through `realizar::run_inference` and therefore through the hybrid's own
    // GPU forward (#3090). Loading it through `OwnedQuantizedModel::from_mapped`
    // here would turn that into a hard `Validation failed` aborting every later
    // gate, so the guard stays as a fail-safe and says where the GPU output IS
    // judged rather than claiming it is not.
    if realizar::gguf::hybrid_forward_handles(mapped.model.architecture().unwrap_or_default()) {
        const JUDGED_BY_RUNTIME: &str = "the Gated DeltaNet hybrid's GPU output is judged by \
             the runtime rung (run_inference → Qwen35CudaModel, #3090), not by the dense loader";
        note_gpu_golden_skip(config, JUDGED_BY_RUNTIME);
        return Ok(GpuGoldenLeg::NotRun(JUDGED_BY_RUNTIME));
    }
    let model = OwnedQuantizedModel::from_mapped(mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Model failed: {e}")))?;
    let generated = match OwnedQuantizedModelCuda::new(model, 0) {
        Ok(cuda_model) => qa_dense_generate(
            &mut qa_dense_cuda(cuda_model),
            prompt_tokens,
            gen_config,
            true,
        )
        .map(|gpu_tokens| gguf.decode(&gpu_tokens))
        .map_err(|e| format!("GPU generation: {e}")),
        Err(e) => Err(format!("CUDA init on device 0: {e}")),
    };
    // #3711 + #3724: ONE typed leg. The budget is passed so the leg can
    // distinguish "still reasoning when the budget ran out" from "answered
    // wrongly" and from "never started" — three outcomes, not two.
    Ok(GpuGoldenLeg::judge(
        generated,
        expected_patterns,
        gen_config.max_tokens,
    ))
}

/// Note, in a verbose human-readable run, that the GPU half of the golden gate
/// did not run and why.
///
/// Extracted so each skip reason is one call: the nested `if !config.json &&
/// config.verbose` blocks were what pushed `validate_gpu_golden_output` past the
/// pre-commit cognitive-complexity threshold when the #3090 early return landed.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
fn note_gpu_golden_skip(config: &QaConfig, message: &str) {
    if !config.json && config.verbose {
        println!("{}", message.yellow());
    }
}

/// #3477: golden generation for a GGUF the dense loader refuses but the runtime
/// serves (Qwen3.5 Gated DeltaNet — CPU #3091, GPU #3090).
///
/// `golden_output_gguf_cpu` builds the model with
/// `OwnedQuantizedModel::from_mapped`; for `qwen35` that call returns the
/// "NEITHER the CPU nor the GPU backend implements ..." error, which `apr qa`
/// surfaced as `Validation failed` and which aborted the entire run. We go
/// through the same public entry point `apr run` uses — `run_inference`, which
/// dispatches `qwen35` to `forward_qwen35` — so the gate certifies the backend
/// that actually serves the model rather than re-implementing the dispatch.
///
/// The GPU is NOT disabled here: since #3090 the runtime routes the hybrid to
/// `Qwen35CudaModel` on a cuda build with a device, so this gate judges the GPU's
/// output — which is the point of #3477. On a CPU-only build or host the same
/// call serves CPU tokens and the gate judges those. Either way the backend
/// under judgement is the backend a user gets.
///
/// The golden prompts are already ChatML, so they are tokenized here and passed
/// via `with_input_tokens` to bypass `prepare_tokens`' chat-template auto-wrap
/// (the same reason `golden_output_apr` does it). Stop tokens come from the
/// model's own EOS, which `run_gguf_inference` merges in.
///
/// Returns the text, the dispatch's own `used_gpu` (#3711) and the generated token count (#3961).
#[cfg(feature = "inference")]
fn golden_output_runtime(
    path: &Path,
    prompt: &str,
    max_tokens: usize,
) -> Result<(String, bool, usize)> {
    use realizar::gguf::MappedGGUFModel;
    use realizar::{run_inference, InferenceConfig};

    let prompt_tokens = {
        let mapped = MappedGGUFModel::from_path(path)
            .map_err(|e| CliError::ValidationFailed(format!("Map failed: {e}")))?;
        mapped.model.encode(prompt).ok_or_else(|| {
            CliError::ValidationFailed(
                "GGUF tokenizer could not encode the golden prompt".to_string(),
            )
        })?
    };

    let infer_config = InferenceConfig::new(path)
        .with_input_tokens(prompt_tokens)
        .with_max_tokens(max_tokens)
        .with_temperature(0.0)
        .with_top_k(1);
    let result = run_inference(&infer_config)
        .map_err(|e| CliError::ValidationFailed(format!("Generation failed: {e}")))?;
    Ok((result.text, result.used_gpu, result.generated_token_count))
}

/// Gate 1 for an architecture the dense loader refuses: the same golden cases,
/// run through the runtime entry point on whichever backend serves the model.
///
/// This gate is the point of #3477: `apr qa` must be able to say PASS about the
/// backend that runs the model — and since #3090 that backend is the GPU.
///
/// `cpu_only`: the GPU backend declines this architecture, so the CPU is the
/// expected backend. Otherwise a cuda build with a device must be served by the
/// GPU, and a CPU fallback fails the gate (#3711).
#[cfg(feature = "inference")]
fn run_golden_output_gate_runtime(
    path: &Path,
    config: &QaConfig,
    cpu_only: bool,
) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!(
            "{}",
            "Running golden output test through the runtime entry point (GPU #3090 / CPU #3091)..."
                .yellow()
        );
    }

    // #3724: this leg serves the hybrid rungs, and it asks them the way production
    // asks them too — same detector, keyed on `general.architecture`. Reading the
    // header is cheap; a file that will not map is left to the generation call
    // below, which reports the mapping error properly.
    // #3990: the header map is kept, because the thinking-ON leg renders the model's own template.
    let mapped_header = realizar::gguf::MappedGGUFModel::from_path(path).ok();
    let architecture = mapped_header
        .as_ref()
        .and_then(|m| m.model.architecture())
        .map(String::from);
    let test_cases = golden_test_cases_for(architecture.as_deref());
    // GH-279-4: thinking models need room for <think>...</think> + the answer.
    let golden_max_tokens = config.max_tokens.max(512);
    let gpu_not_run = if cpu_only {
        Some("the GPU backend declines this architecture")
    } else {
        gpu_golden_not_run(cfg!(feature = "cuda"), cuda_device_present(), true)
    };

    let mut served_by = String::new();
    for (prompt, expected_patterns) in &test_cases {
        let (output_text, used_gpu, _) =
            golden_output_runtime(path, prompt.as_str(), golden_max_tokens)?;
        // #3711: the backend first — a GPU that fell back is a FAIL even when the CPU's answer is right
        match runtime_golden_backend(used_gpu, gpu_not_run) {
            Ok(label) => served_by = label,
            Err(failure) => {
                return Ok(GateResult::failed(
                    "golden_output",
                    &failure,
                    None,
                    None,
                    start.elapsed(),
                ))
            }
        }
        // GH-279-4 / #3724: and then the answer — an unclosed block is reported by
        // name on this leg too. The hybrid rungs run here, and a truncating strip
        // would hide the same defect the dense leg just learned to name.
        let answer_text = match split_thinking_blocks(&output_text) {
            ThinkingSplit::Answer(answer) => answer,
            ThinkingSplit::Unclosed => {
                return Ok(GateResult::failed(
                    "golden_output",
                    &unclosed_think_reason(
                        "golden_output_runtime",
                        golden_max_tokens,
                        output_text.len(),
                    ),
                    None,
                    None,
                    start.elapsed(),
                ));
            }
        };
        if let OutputVerification::Fail { reason } =
            verify_output(&answer_text, "golden_output_runtime", expected_patterns)
        {
            return Ok(GateResult::failed(
                "golden_output",
                &reason,
                None,
                None,
                start.elapsed(),
            ));
        }
    }

    // #3724 done_when 3: the hybrid rungs are thinking-capable too, and this leg
    // is where they are judged.
    let on_leg = match runtime_thinking_on_leg(
        path,
        architecture.as_deref(),
        mapped_header.as_ref().map(|m| &m.model),
        gpu_not_run,
        start,
    )? {
        Ok(on_leg) => on_leg,
        Err(failed) => return Ok(failed),
    };

    Ok(GateResult::passed(
        "golden_output",
        &format!(
            "{} golden test cases passed through the runtime entry point (served by {served_by}){on_leg}",
            test_cases.len(),
        ),
        Some(test_cases.len() as f64),
        Some(test_cases.len() as f64),
        start.elapsed(),
    ))
}

/// #3724 done_when 3: the thinking-ON leg of the runtime golden gate. `Ok(suffix)` is the
/// pass message's ON-leg clause (empty when the model has no thinking-on case), and
/// `Err(failed)` ends the gate. Extracted from `run_golden_output_gate_runtime` unchanged,
/// to keep that function under the complexity ratchet.
fn runtime_thinking_on_leg(
    path: &Path,
    architecture: Option<&str>,
    header: Option<&realizar::gguf::GGUFModel>,
    gpu_not_run: Option<&'static str>,
    start: Instant,
) -> Result<std::result::Result<String, GateResult>> {
    // #3724 done_when 3: the hybrid rungs are thinking-capable too, and this leg
    // is where they are judged.
    let on_case = match thinking_on_case_for_model(architecture, header) {
        Ok(c) => c,
        Err(reason) => {
            return Ok(Err(GateResult::failed(
                "golden_output",
                &format!("golden_output_thinking_on: {reason}"),
                None,
                None,
                start.elapsed(),
            )))
        }
    };
    if let Some((on_prompt, on_patterns)) = on_case {
        // #3907 WIRING (#3907 landed the resolver and reached only the DENSE leg at
        // golden_output.rs:795; this is the HYBRID leg, and it kept passing the raw
        // `THINKING_ON_BUDGET` const to BOTH the generation and the judging). Measured
        // before this change: APR_THINKING_ON_BUDGET=8192, =512 and unset all produced
        // byte-identical output — "within 2048 tokens ... 8901 chars" — and =abc, which
        // the resolver must reject as "is not a token count", changed nothing. The
        // override was compiled in and unreachable. So the one model the budget table
        // was written for was the one model that could not reach it.
        let model_file = path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        let (on_budget, budget_basis) = match thinking_on_budget_for(&model_file) {
            Ok(v) => v,
            Err(reason) => {
                return Ok(Err(GateResult::failed(
                    "golden_output",
                    &format!("golden_output_thinking_on: {reason}"),
                    None,
                    None,
                    start.elapsed(),
                )))
            }
        };
        let (on_text, on_used_gpu, on_tokens) =
            golden_output_runtime(path, on_prompt.as_str(), on_budget)?;
        let generated = on_text.strip_prefix(on_prompt.as_str()).unwrap_or(&on_text);
        let judged = on_leg_judged_text(&on_prompt, generated);
        let generated = judged.as_str();
        if let Some(reason) = thinking_on_leg_failure(
            generated,
            &on_patterns,
            on_budget,
            on_used_gpu,
            gpu_not_run,
            &budget_basis,
        ) {
            return Ok(Err(GateResult::failed(
                "golden_output",
                &reason,
                None,
                None,
                start.elapsed(),
            )));
        }
        // #3961: a pass says what the ON leg did -- its own backend, how much it reasoned
        // and how long it ran. `think_body_chars` is the number that told the 0.8B model
        // that skipped reasoning (0) from the controls that did (445-2149).
        return Ok(Ok(format!(
            "; thinking-ON leg served by {}, think_body_chars={}, generated_tokens={on_tokens}",
            if on_used_gpu { "GPU" } else { "CPU" },
            think_body_chars(generated),
        )));
    }
    Ok(Ok(String::new()))
}

/// #3961: the hybrid thinking-ON leg's whole verdict, pure so a table can drive it.
#[cfg(feature = "inference")]
pub(crate) fn thinking_on_leg_failure(
    generated: &str,
    on_patterns: &[&str],
    on_budget: usize,
    on_used_gpu: bool,
    gpu_not_run: Option<&'static str>,
    budget_basis: &str,
) -> Option<String> {
    // The backend first, as on the OFF cases: a GPU-expected leg the CPU served is a FAIL
    // even when the CPU's answer is right. The ON leg used to skip this ("the backend was
    // already judged on the OFF cases") -- an assumption nothing checked, on the longest
    // generation the gate runs, the one most exposed to a mid-run fallback (#3961).
    if let Err(failure) = runtime_golden_backend(on_used_gpu, gpu_not_run) {
        return Some(format!(
            "golden_output_thinking_on: {failure} [budget basis — {budget_basis}]"
        ));
    }
    judge_thinking_on_output(generated, &on_patterns, on_budget)
        .map(|r| format!("{r} [budget basis — {budget_basis}]"))
}

/// Without `inference` there is no runtime to certify.
#[cfg(not(feature = "inference"))]
fn run_golden_output_gate_runtime(
    path: &Path,
    config: &QaConfig,
    cpu_only: bool,
) -> Result<GateResult> {
    let _ = (path, config, cpu_only);
    Ok(GateResult::skipped(
        "golden_output",
        "Requires 'inference' feature",
    ))
}

/// Run golden output test for APR format models.
///
/// PMAT-CODE-SHIP-006-FIX (2026-05-10): routed through `realizar::run_inference`
/// + `OwnedQuantizedModel::from_apr` (the same working path that SHIP-002 +
/// SHIP-008 LIVE-discharged) instead of the legacy `AprTransformer::from_apr_file`
/// + `generate_with_cache` path. The legacy path produced "\\ns\\ns" degenerate
/// output on canonical 7B teacher (recorded as §61.8 Branch A); the
/// run_inference path produces clean ChatML responses.
///
/// Caller passes a pre-formatted ChatML prompt (e.g.,
/// `"<|im_start|>user\\nWhat is 2+2?<|im_end|>\\n<|im_start|>assistant\\n"`)
/// per `golden_test_cases()` — we tokenize it directly and pass via
/// `InferenceConfig::with_input_tokens` to BYPASS the chat-template auto-wrap
/// in `prepare_tokens_apr` (which would double-wrap a pre-formatted prompt).
#[cfg(feature = "inference")]
fn golden_output_apr(path: &Path, prompt: &str, max_tokens: usize) -> Result<(Vec<u32>, String)> {
    use realizar::apr::AprV2Model;
    use realizar::{run_inference, InferenceConfig};

    // Tokenize the (already-ChatML-formatted) prompt with the embedded BPE
    // tokenizer. This produces the exact prompt token sequence the qa gate
    // intends; passing it via with_input_tokens bypasses prepare_tokens'
    // ChatML auto-wrap (which would otherwise double-wrap pre-formatted prompts).
    let apr_model = AprV2Model::load(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR: {e}")))?;
    let tokenizer = apr_model
        .load_embedded_bpe_tokenizer()
        .ok_or_else(|| CliError::ValidationFailed("APR missing embedded tokenizer".to_string()))?;
    let prompt_tokens = tokenizer.encode(prompt);

    let config = InferenceConfig::new(path)
        .with_input_tokens(prompt_tokens)
        .with_max_tokens(max_tokens)
        .with_temperature(0.0)
        .with_top_k(1);

    let result = run_inference(&config)
        .map_err(|e| CliError::ValidationFailed(format!("Generation failed: {e}")))?;

    Ok((result.tokens, result.text))
}

/// #3782: a degenerate completion is not a correct answer, on ANY golden case.
///
/// The issue names the greeting case's `"!"`. Measured against all three cases
/// first, the hole was wider: the ARITHMETIC case's whole expected answer is the
/// single character `"4"`, so `"44444444"` scored correct on the flagship golden
/// test and nothing in the issue mentions it. `"4"` cannot be dropped the way
/// `"!"` can — it is the right answer — so the guard has to be general.
///
/// Boundary measured on the pre-fix gate: `"!"` repeated 1..=11 all PASSED;
/// 12 and up were caught by `gibberish_repeated_fragment`, whose
/// `bytes.len() >= 12` / 4-byte-fragment shape is exactly the gap.
#[cfg(test)]
mod pmat3782_degenerate_is_not_an_answer {
    use super::{verify_output, OutputVerification};

    fn rejected(output: &str, patterns: &[&str]) -> bool {
        matches!(
            verify_output(output, "PMAT-3782", patterns),
            OutputVerification::Fail { .. }
        )
    }

    /// The three live golden cases, with the degenerate completion that the
    /// substring-any check would otherwise score as correct for each.
    /// `(case, patterns, degenerate output, why it was accepted)`
    const DEGENERATE: &[(&str, &[&str], &str, &str)] = &[
        (
            "arithmetic",
            &["4"],
            "44444444",
            "the expected answer IS a single character, so any run of it matches — \
             the flagship golden case, and not mentioned in #3782",
        ),
        (
            "arithmetic (11, just under the old 12-byte floor)",
            &["4"],
            "44444444444",
            "gibberish_repeated_fragment needs 12 bytes; this is 11",
        ),
        (
            "greeting",
            &["Hello", "Hi", "hey", "hello", "well"],
            "!!!!!!!!",
            "#3782 as filed: `!` is token id 0 in the Qwen vocab, what dead logits emit",
        ),
    ];

    /// Every case at once, so a regression names each golden case it re-opened.
    #[test]
    fn no_golden_case_accepts_a_degenerate_completion() {
        let accepted: Vec<String> = DEGENERATE
            .iter()
            .filter(|(_, pats, out, _)| !rejected(out, pats))
            .map(|(case, _, out, why)| format!("\n  - {case}: {out:?} scored CORRECT. {why}"))
            .collect();
        assert!(
            accepted.is_empty(),
            "#3782 REGRESSION: {} of {} golden cases accept a degenerate completion, \
             so a model emitting a dead-logit loop passes apr qa's golden_output:{}",
            accepted.len(),
            DEGENERATE.len(),
            accepted.join("")
        );
    }

    /// A token-0 loop at the gate's real generation length.
    #[test]
    fn a_token_zero_loop_is_rejected() {
        for n in [8usize, 12, 32, 64] {
            assert!(
                rejected(&"!".repeat(n), &["Hello", "Hi", "hey", "hello", "well"]),
                "#3782: a {n}-token loop of `!` (token id 0) scored correct"
            );
        }
    }

    /// The over-correction: a real answer must still pass. A guard that rejects
    /// everything is not a guard, and the arithmetic case answers with ONE
    /// character, which is the case most at risk from a careless length rule.
    #[test]
    fn real_answers_still_pass() {
        let ok: &[(&str, &[&str])] = &[
            ("4", &["4"]),
            ("2 + 2 = 4", &["4"]),
            ("The answer is 4.", &["4"]),
            ("Hello! How are you doing today?", &["Hello", "Hi"]),
            ("The capital of France is Paris.", &["Paris"]),
            // 90% is a floor, not a ceiling: heavy but legitimate punctuation.
            ("Hello!!!!!!!!", &["Hello"]),
        ];
        let wrongly: Vec<String> = ok
            .iter()
            .filter(|(out, pats)| rejected(out, pats))
            .map(|(out, _)| format!("\n  - {out:?}"))
            .collect();
        assert!(
            wrongly.is_empty(),
            "#3782 OVER-CORRECTION: the degenerate guard rejected {} legitimate \
             answer(s):{}",
            wrongly.len(),
            wrongly.join("")
        );
    }
}

/// #3904: the golden gate's failure reason must say what it dropped.
#[cfg(all(test, feature = "inference"))]
mod thinking_on_leg_3961 {
    use super::thinking_on_leg_failure;

    const REASONED: &str = "<think>two plus two is four</think>2 + 2 = 4.";

    /// #3961 MUST-RED: the ON leg fell back to the CPU where the GPU was expected. The OFF
    /// cases judged THEIR backend; this leg is the longest generation the gate runs and was
    /// judged on its text alone, so a CPU-served answer passed as a GPU cell (#3922 shape).
    #[test]
    fn an_on_leg_that_fell_back_is_red() {
        let got = thinking_on_leg_failure(REASONED, &["4"], 2048, false, None, "row qwen35")
            .expect("a GPU-expected ON leg the CPU served is not a pass");
        assert!(got.contains("golden_output_thinking_on"), "{got}");
        assert!(got.contains("fell back"), "{got}");
    }

    /// Positive controls: GPU-served, and CPU where the GPU was never expected.
    #[test]
    fn an_on_leg_on_its_expected_backend_passes() {
        assert_eq!(
            thinking_on_leg_failure(REASONED, &["4"], 2048, true, None, "b"),
            None
        );
        assert_eq!(
            thinking_on_leg_failure(REASONED, &["4"], 2048, false, Some("no cuda build"), "b"),
            None
        );
    }

    /// #3948 quorum item 4: an ON-leg failure names the budget basis, as the dense leg's
    /// does (golden_output.rs), so "closed EMPTY within 2048" cannot cite the wrong row.
    #[test]
    fn an_on_leg_failure_names_its_budget_basis() {
        let got = thinking_on_leg_failure(
            "<think>two plus two is four</think>It is five.",
            &["4"],
            2048,
            true,
            None,
            "row Qwen3.5-0.8B-Q4_K_M",
        )
        .expect("a wrong answer fails");
        assert!(
            got.contains("[budget basis — row Qwen3.5-0.8B-Q4_K_M]"),
            "{got}"
        );
    }
}

#[cfg(test)]
mod loud_truncation_3904 {
    use super::*;

    #[test]
    fn a_short_reason_is_untouched() {
        assert_eq!(
            loudly_truncated("The capital of France is Paris.", 100),
            "The capital of France is Paris."
        );
    }

    /// Exactly at the bound is NOT truncated, so the marker never appears on a complete
    /// string. An off-by-one here would make every full-length output look decapitated.
    #[test]
    fn exactly_the_bound_says_nothing() {
        let s = "x".repeat(100);
        assert_eq!(loudly_truncated(&s, 100), s);
    }

    /// THE ROW. A bare `.chars().take(100)` passes every assertion above and fails this
    /// one: it produces a decapitated string indistinguishable from a complete short one.
    #[test]
    fn a_cut_reason_says_how_much_it_dropped() {
        let s = "y".repeat(137);
        let got = loudly_truncated(&s, 100);
        assert!(got.starts_with(&"y".repeat(100)), "{got}");
        assert!(
            got.contains("and 37 more chars"),
            "a reader cannot tell a cut string from a complete one: {got}"
        );
    }

    /// Counted in CHARS, not bytes — the reason carries model output, which is not ASCII.
    #[test]
    fn the_count_is_chars_not_bytes() {
        let s = "é".repeat(150);
        let got = loudly_truncated(&s, 100);
        assert!(got.contains("and 50 more chars"), "{got}");
    }

    /// End to end through the message the receipt actually carries.
    #[test]
    fn the_golden_failure_reason_is_loud() {
        // Varied, non-degenerate text: a long run of one character trips the
        // gibberish check first and never reaches the pattern branch, so a fixture
        // built from `"z".repeat(300)` would exercise a different failure entirely.
        let long: String = "<s>[INST] q [/INST] France is a country in western Europe \
            whose largest city and seat of government has been the subject of this \
            question for as long as anyone has been asking models about it at all."
            .to_string();
        assert!(long.chars().count() > 100, "fixture must exceed the bound");
        let v = verify_output(&long, "t", &["Paris"]);
        match v {
            OutputVerification::Fail { reason } => assert!(
                reason.contains("more chars"),
                "the one string a human reads was cut silently: {reason}"
            ),
            other => panic!("expected Fail, got {other:?}"),
        }
    }

    /// #3907 wiring: BOTH hybrid ON-leg sites take the resolver's budget.
    ///
    /// A source read, because the behavioural discriminator needs a GPU: measured on
    /// lambda with `APR_THINKING_ON_BUDGET=256`, both-routed reports "within 256
    /// tokens ... 1105 chars" and a JUDGING-unrouted mutant reports "within 2048
    /// tokens ... 1105 chars" — the same generation, a misreported budget. A single
    /// test touching only one site cannot tell a one-site fix from a two-site one,
    /// and "a fix reaching one of N sites" is the defect this commit repairs.
    #[test]
    fn both_hybrid_on_leg_sites_take_the_resolved_budget() {
        let src = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/commands/output_verification.rs"
        ))
        .expect("own source readable");

        // EVERY assertion scans the code ABOVE the test module, never the whole file.
        // The first draft did not, and all three positive assertions matched their OWN
        // text inside this test — so the guard passed with BOTH sites unrouted. Caught
        // by mutating it rather than by reading it: a source-reading guard that scans
        // itself is satisfied by its own assertion strings, which is the vacuous shape
        // this commit exists to remove one level down.
        let code = src.split("#[cfg(test)]").next().unwrap_or(&src);

        assert!(
            code.contains("golden_output_runtime(path, on_prompt.as_str(), on_budget)"),
            "the hybrid ON leg's GENERATION must take the resolved budget, not a literal (#3907)"
        );
        assert!(
            code.contains("judge_thinking_on_output(generated, &on_patterns, on_budget)"),
            "the hybrid ON leg's JUDGING must take the resolved budget, not a literal (#3907)"
        );
        assert!(
            code.contains("thinking_on_budget_for(&model_file)"),
            "the hybrid ON leg must resolve through thinking_on_budget_for, which carries the \
             per-model row, its refusal path and the APR_THINKING_ON_BUDGET probe (#3907)"
        );
        assert!(
            !code.contains("THINKING_ON_BUDGET)"),
            "no ON-leg site may pass the old THINKING_ON_BUDGET const — it is deleted (#3907)"
        );
    }
}

#[cfg(test)]
mod absent_is_not_a_pass_3873 {
    use super::*;

    fn run_all(
        arch: Option<&str>,
        theta: Option<f32>,
        max_pos: Option<usize>,
        eps: Option<f32>,
    ) -> (Vec<String>, MetadataTally) {
        let mut violations = Vec::new();
        let mut tally = MetadataTally::default();
        let arch_owned = arch.map(str::to_string);
        check_rope_theta(arch, theta, b"GGUF", &mut violations, &mut tally);
        check_max_position_embeddings(max_pos, &mut violations, &mut tally);
        check_rms_norm_eps(eps, &mut violations, &mut tally);
        check_arch_theta_cross_validation(arch_owned.as_ref(), theta, &mut violations, &mut tally);
        (violations, tally)
    }

    #[test]
    fn a_gguf_with_no_metadata_claims_no_passed_check() {
        let (violations, tally) = run_all(None, None, None, None);
        assert!(violations.is_empty(), "absence is not a violation for GGUF");
        assert_eq!(tally.passed, 0, "nothing was measured, so nothing passed");
        assert_eq!(tally.absent.len(), 4);
        let summary = tally.summary();
        assert!(
            summary.starts_with("0 metadata checks passed, 4 not checked"),
            "{summary}"
        );
    }

    #[test]
    fn a_fully_populated_file_reports_four_measured_passes() {
        let (violations, tally) = run_all(Some("qwen2"), Some(1_000_000.0), Some(32_768), Some(1e-6));
        assert!(violations.is_empty(), "{violations:?}");
        assert_eq!(tally.passed, 4);
        assert!(tally.absent.is_empty());
        assert_eq!(tally.summary(), "4 metadata checks passed");
    }

    #[test]
    fn a_partial_file_names_exactly_the_missing_fields() {
        let (_, tally) = run_all(Some("llama"), Some(500_000.0), None, Some(1e-5));
        assert_eq!(tally.passed, 3);
        assert_eq!(tally.absent, vec!["max_position_embeddings"]);
    }

    #[test]
    fn apr_missing_rope_theta_stays_a_violation() {
        let mut violations = Vec::new();
        let mut tally = MetadataTally::default();
        check_rope_theta(Some("qwen2"), None, b"APRN", &mut violations, &mut tally);
        assert_eq!(violations.len(), 1);
        assert_eq!(tally.passed, 0);
    }
}
