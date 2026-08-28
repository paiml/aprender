//! Standardized prompt profiles for LLM benchmarking.
//!
//! Provides deterministic prompt sets with calibrated input/output token counts
//! for reproducible benchmarks. Follows HuggingFace Inference-Benchmarker
//! methodology with fixed-length profiles.

use super::client::{ChatMessage, ChatRequest, Role};
use std::path::Path;

/// Standardized prompt profiles for benchmarking.
///
/// Each profile targets a specific input/output token count to ensure
/// comparable results across backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptProfile {
    /// ~10 input tokens, max_tokens=1. TTFT-only measurement (prefill speed).
    Micro,
    /// ~32 input tokens, max_tokens=32. Quick latency check.
    Short,
    /// ~128 input tokens, max_tokens=128. Standard comparison (default).
    Medium,
    /// ~512 input tokens, max_tokens=256. Sustained decode measurement.
    Long,
}

impl PromptProfile {
    /// Parse a profile name (case-insensitive).
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "micro" => Some(Self::Micro),
            "short" => Some(Self::Short),
            "medium" => Some(Self::Medium),
            "long" => Some(Self::Long),
            _ => None,
        }
    }
}

/// Load the built-in prompts for a given profile.
///
/// All prompts use temperature=0.0 for deterministic output.
pub fn load_profile(profile: PromptProfile) -> Vec<ChatRequest> {
    match profile {
        PromptProfile::Micro => vec![micro_prompt()],
        PromptProfile::Short => vec![short_prompt()],
        PromptProfile::Medium => vec![medium_prompt()],
        PromptProfile::Long => vec![long_prompt()],
    }
}

/// Load a benchmark prompt corpus from a **JSONL** file.
///
/// # The format is JSONL, and this is the file that says so
///
/// One JSON object per line, no enclosing array, no YAML:
///
/// ```jsonl
/// {"_meta":{"corpus":"W1","provenance":"SYNTHETIC ..."}}
/// {"id":0,"prompt":"// w1-0000\nlet mut ...","max_tokens":128,"temperature":0.0,"seed":0,"target_prompt_tokens":512}
/// ```
///
/// `APR-PERF-GATE-001` v2.2 §4.3.1 and §4.3.2 both name a `.jsonl` corpus, and
/// `prompts-w2.jsonl` has been on main in that shape since W2 landed. This
/// loader previously parsed YAML with a top-level `prompts:` key and the CLI
/// help advertised "a JSON array" — three formats, no two matching, and the
/// consequence was that **the only committed corpus in the tree could not be
/// read by the only loader in the tree.** The spec is the authority; the two
/// implementations were the drift, so both moved to JSONL.
///
/// # Records
///
/// `prompt` is the only required field. `id` and `target_prompt_tokens` are
/// carried by the corpus for the receipt and ignored here. Unknown fields are
/// **rejected**, not ignored: a record that says `content:` (the old YAML key)
/// or `max_token:` would otherwise load with a silently defaulted budget, and
/// a benchmark whose generation budget silently defaulted is measuring
/// something other than what its receipt claims.
///
/// # Errors
///
/// Every failure names JSONL and the offending line. In particular an empty
/// corpus is an error, never an empty `Vec` — a load-test that issues zero
/// requests and reports success is the vacuous pass this gate exists to
/// remove.
pub fn load_from_file(path: &Path) -> Result<Vec<ChatRequest>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    parse_jsonl(&content).map_err(|e| format!("{}: {e}", path.display()))
}

/// Parse a JSONL corpus body. Split out from [`load_from_file`] so the format
/// contract is testable without touching the filesystem.
fn parse_jsonl(content: &str) -> Result<Vec<ChatRequest>, String> {
    reject_non_jsonl_shape(content)?;

    let mut requests = Vec::new();
    for (idx, raw) in content.lines().enumerate() {
        let lineno = idx + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if is_meta_line(line) {
            // The corpus's own provenance header. Allowed on line 1 only, so a
            // `_meta` object cannot hide in the middle of a corpus and quietly
            // shrink the workload by one request.
            if lineno == 1 {
                continue;
            }
            return Err(format!(
                "line {lineno}: a `_meta` header record is only allowed on line 1 \
                 of a JSONL corpus"
            ));
        }
        let record: PromptRecord = serde_json::from_str(line).map_err(|e| {
            format!("line {lineno}: expected one JSON object per line (JSONL), got: {e}")
        })?;
        requests.push(record.into_request());
    }

    if requests.is_empty() {
        return Err(
            "contains no prompt records — a benchmark corpus that loads zero prompts \
             would report success having issued no requests"
                .to_string(),
        );
    }
    Ok(requests)
}

/// True when `line` is a JSONL provenance header (`{"_meta": ...}`).
fn is_meta_line(line: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(line)
        .ok()
        .and_then(|v| v.as_object().map(|o| o.contains_key("_meta")))
        .unwrap_or(false)
}

/// Refuse the two formats this loader used to be confused with, naming JSONL.
///
/// Without this, a YAML corpus fails with a serde_json message about line 1
/// and a JSON array fails with one about a trailing bracket — both true, and
/// neither tells the operator that the format they wrote is not the format
/// this file wants.
fn reject_non_jsonl_shape(content: &str) -> Result<(), String> {
    let Some(first) = content.lines().map(str::trim).find(|l| !l.is_empty()) else {
        return Err("is empty — expected JSONL, one JSON object per line".to_string());
    };
    if first.starts_with('[') {
        return Err(
            "looks like a JSON array — expected JSONL: one JSON object per line, \
             with no enclosing `[ ]` and no commas between records"
                .to_string(),
        );
    }
    if !first.starts_with('{') {
        return Err(format!(
            "line 1 does not begin a JSON object (it begins {:?}) — expected JSONL: \
             one JSON object per line. The YAML `prompts:` corpus format this loader \
             used to accept is no longer supported; regenerate as JSONL",
            first.chars().take(16).collect::<String>()
        ));
    }
    Ok(())
}

/// One record of a JSONL prompt corpus.
///
/// `deny_unknown_fields` is deliberate — see [`load_from_file`].
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptRecord {
    /// The prompt text. The only required field.
    prompt: String,
    /// Message role. Defaults to `user`.
    role: Option<String>,
    /// Generation budget for this request.
    max_tokens: Option<u32>,
    /// Sampling temperature. Defaults to `0.0` (greedy).
    temperature: Option<f64>,
    /// Sampling seed, so the run is reproducible (§4.4.4).
    seed: Option<u64>,
    /// Suppress end-of-sequence stopping, so the token count per request is
    /// pinned to `max_tokens` (§4.3.1).
    ignore_eos: Option<bool>,
    /// Recorded by the corpus for the receipt; not used to build the request.
    #[allow(dead_code)]
    id: Option<u64>,
    /// Recorded by the corpus for the receipt; the actual count is measured by
    /// the harness against the model's own tokenizer, never asserted here.
    #[allow(dead_code)]
    target_prompt_tokens: Option<u32>,
}

impl PromptRecord {
    fn into_request(self) -> ChatRequest {
        ChatRequest {
            model: String::new(),
            messages: vec![ChatMessage {
                role: self.role.as_deref().map_or(Role::User, parse_role),
                content: self.prompt,
            }],
            temperature: Some(self.temperature.unwrap_or(0.0)),
            max_tokens: self.max_tokens,
            stream: Some(false),
            seed: self.seed,
            ignore_eos: self.ignore_eos,
        }
    }
}

fn parse_role(s: &str) -> Role {
    match s.to_lowercase().as_str() {
        "system" => Role::System,
        "assistant" => Role::Assistant,
        _ => Role::User,
    }
}

// --- Built-in prompt profiles ---

fn micro_prompt() -> ChatRequest {
    ChatRequest {
        model: String::new(),
        messages: vec![ChatMessage {
            role: Role::User,
            content: "Say hello.".to_string(),
        }],
        temperature: Some(0.0),
        max_tokens: Some(1),
        stream: Some(false),
        seed: None,
        ignore_eos: None,
    }
}

fn short_prompt() -> ChatRequest {
    ChatRequest {
        model: String::new(),
        messages: vec![ChatMessage {
            role: Role::User,
            content: "Explain what a hash table is and why it provides O(1) average lookup time."
                .to_string(),
        }],
        temperature: Some(0.0),
        max_tokens: Some(32),
        stream: Some(false),
        seed: None,
        ignore_eos: None,
    }
}

fn medium_prompt() -> ChatRequest {
    ChatRequest {
        model: String::new(),
        messages: vec![ChatMessage {
            role: Role::User,
            content: "Write a detailed explanation of how binary search works, \
                      including its time complexity, when to use it, and common \
                      pitfalls. Include a step-by-step example with the array \
                      [2, 5, 8, 12, 16, 23, 38, 56, 72, 91] searching for 23. \
                      Explain why the algorithm requires a sorted array and what \
                      happens if the array is unsorted. Discuss the difference \
                      between iterative and recursive implementations and their \
                      respective trade-offs in terms of stack usage and performance."
                .to_string(),
        }],
        temperature: Some(0.0),
        max_tokens: Some(128),
        stream: Some(false),
        seed: None,
        ignore_eos: None,
    }
}

fn long_prompt() -> ChatRequest {
    ChatRequest {
        model: String::new(),
        messages: vec![ChatMessage {
            role: Role::User,
            content: "You are a systems programming expert. Write a comprehensive \
                      guide on implementing a memory allocator in Rust. Cover the \
                      following topics in detail:\n\n\
                      1. The difference between stack and heap allocation, including \
                      how the operating system manages virtual memory pages and how \
                      the brk/mmap system calls work on Linux.\n\n\
                      2. The design of a simple bump allocator, including its \
                      advantages (fast allocation, no fragmentation tracking) and \
                      disadvantages (no individual deallocation, memory waste).\n\n\
                      3. The free list allocator design pattern, explaining how \
                      freed blocks are tracked, how coalescing adjacent free blocks \
                      works, and the trade-offs between first-fit, best-fit, and \
                      worst-fit allocation strategies.\n\n\
                      4. The buddy system allocator, explaining how power-of-two \
                      block sizes enable efficient splitting and merging, and how \
                      this approach reduces external fragmentation at the cost of \
                      internal fragmentation.\n\n\
                      5. How Rust's ownership system and the GlobalAlloc trait \
                      interact with custom allocators. Show how to implement the \
                      GlobalAlloc trait and register a custom allocator using \
                      #[global_allocator].\n\n\
                      6. Thread safety considerations: how to make an allocator \
                      thread-safe using Mutex or atomic operations, and the \
                      performance implications of lock contention in multi-threaded \
                      workloads. Discuss arena-per-thread strategies.\n\n\
                      7. Real-world allocator designs like jemalloc, mimalloc, and \
                      tcmalloc. Explain their key innovations and when you would \
                      choose one over another.\n\n\
                      Include code examples for each allocator type."
                .to_string(),
        }],
        temperature: Some(0.0),
        max_tokens: Some(256),
        stream: Some(false),
        seed: None,
        ignore_eos: None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_from_name() {
        assert_eq!(
            PromptProfile::from_name("micro"),
            Some(PromptProfile::Micro)
        );
        assert_eq!(
            PromptProfile::from_name("short"),
            Some(PromptProfile::Short)
        );
        assert_eq!(
            PromptProfile::from_name("medium"),
            Some(PromptProfile::Medium)
        );
        assert_eq!(PromptProfile::from_name("long"), Some(PromptProfile::Long));
        assert_eq!(
            PromptProfile::from_name("MEDIUM"),
            Some(PromptProfile::Medium)
        );
        assert_eq!(PromptProfile::from_name("unknown"), None);
    }

    #[test]
    fn test_micro_profile() {
        let prompts = load_profile(PromptProfile::Micro);
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].max_tokens, Some(1));
        assert_eq!(prompts[0].temperature, Some(0.0));
    }

    #[test]
    fn test_short_profile() {
        let prompts = load_profile(PromptProfile::Short);
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].max_tokens, Some(32));
    }

    #[test]
    fn test_medium_profile() {
        let prompts = load_profile(PromptProfile::Medium);
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].max_tokens, Some(128));
    }

    #[test]
    fn test_long_profile() {
        let prompts = load_profile(PromptProfile::Long);
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].max_tokens, Some(256));
    }

    // ---------------------------------------------------------------------
    // JSONL format contract (PERF-039).
    //
    // MUTATION TABLE. Every row is a corpus in a format this loader must
    // REFUSE, and each asserts that the refusal message NAMES the expected
    // format. The defect being excluded is not "wrong format accepted" alone —
    // it is a loader that reads zero prompts and lets the caller proceed to
    // report a successful benchmark having issued no requests. The rows marked
    // "was silently ..." describe what the pre-PERF-039 YAML loader did with
    // that same input.
    //
    //  input shape                          | must | message must name
    //  -------------------------------------|------|------------------------
    //  JSON array `[{...}]`                  | ERR  | "JSONL"
    //  YAML `prompts:` (the OLD format)      | ERR  | "JSONL"
    //  empty file                            | ERR  | "JSONL"
    //  whitespace-only file                  | ERR  | "JSONL"
    //  well-formed JSONL, zero records       | ERR  | "zero prompts"
    //  `_meta` header alone, no records      | ERR  | "zero prompts"
    //  `_meta` on a line other than 1        | ERR  | "line 1"
    //  record missing `prompt`               | ERR  | line number
    //  record with `content:` (old YAML key) | ERR  | line number
    //  record with a typo'd field            | ERR  | line number
    //  the committed prompts-w1.jsonl        | OK   | 256 records
    // ---------------------------------------------------------------------

    /// Assert `body` is refused and the message mentions `needle`.
    fn refused(body: &str, needle: &str) -> String {
        let err = parse_jsonl(body)
            .err()
            .unwrap_or_else(|| panic!("must be REFUSED, was accepted: {body:?}"));
        assert!(
            err.to_lowercase().contains(&needle.to_lowercase()),
            "message must name {needle:?}, got: {err}"
        );
        err
    }

    #[test]
    fn jsonl_array_is_refused_naming_jsonl() {
        // Binds to the SHAPE guard, not to the per-line parser: with the guard
        // disarmed a `[` line is still refused, just by a message that does not
        // tell the operator their whole file is the wrong shape. Asserting the
        // generic word "JSONL" alone was green under that mutation.
        let err = refused(r#"[{"prompt":"hi","max_tokens":4}]"#, "JSONL");
        assert!(
            err.contains("JSON array"),
            "must name the shape, got: {err}"
        );
    }

    #[test]
    fn jsonl_yaml_is_refused_naming_jsonl() {
        // The exact shape the pre-PERF-039 loader accepted.
        let err = refused(
            "prompts:\n  - role: user\n    content: \"What is 2+2?\"\n    max_tokens: 16\n",
            "JSONL",
        );
        assert!(
            err.contains("no longer supported"),
            "must say the YAML corpus format is gone, got: {err}"
        );
    }

    #[test]
    fn jsonl_empty_file_is_refused() {
        refused("", "JSONL");
        refused("   \n\n\t\n", "JSONL");
    }

    #[test]
    fn jsonl_zero_records_is_refused_not_empty_vec() {
        // THE vacuous-pass row: a syntactically fine corpus carrying no work.
        // It must be an error, never `Ok(vec![])`.
        let err = refused("{\"_meta\":{\"corpus\":\"W1\"}}\n", "zero prompts");
        assert!(err.contains("no prompt records"), "got: {err}");
    }

    #[test]
    fn jsonl_meta_outside_line_one_is_refused() {
        refused(
            "{\"prompt\":\"a\"}\n{\"_meta\":{\"corpus\":\"W1\"}}\n{\"prompt\":\"b\"}\n",
            "line 1",
        );
    }

    #[test]
    fn jsonl_record_missing_prompt_is_refused_with_line_number() {
        refused("{\"prompt\":\"a\"}\n{\"max_tokens\":4}\n", "line 2");
    }

    #[test]
    fn jsonl_old_yaml_content_key_is_refused_with_line_number() {
        // `content` was the YAML loader's key. Converting a YAML corpus to
        // JSONL naively produces exactly this, and it must not load with a
        // silently defaulted prompt.
        refused("{\"role\":\"user\",\"content\":\"hi\"}\n", "line 1");
    }

    #[test]
    fn jsonl_typoed_field_is_refused_not_defaulted() {
        // `max_token` (singular) would otherwise deserialize to
        // `max_tokens: None` and the benchmark would run an unbudgeted
        // generation while its receipt claimed 128.
        refused("{\"prompt\":\"a\",\"max_token\":128}\n", "line 1");
    }

    #[test]
    fn jsonl_happy_path_carries_every_field() {
        let reqs = parse_jsonl(
            "{\"_meta\":{\"corpus\":\"W1\"}}\n\
             {\"id\":0,\"prompt\":\"hi\",\"max_tokens\":128,\"temperature\":0.0,\
             \"seed\":7,\"ignore_eos\":true,\"target_prompt_tokens\":512}\n\
             {\"prompt\":\"bye\"}\n",
        )
        .expect("valid JSONL");
        assert_eq!(reqs.len(), 2, "_meta must not be counted as a request");
        assert_eq!(reqs[0].max_tokens, Some(128));
        assert_eq!(reqs[0].seed, Some(7));
        assert_eq!(reqs[0].ignore_eos, Some(true));
        assert_eq!(reqs[0].messages[0].role, Role::User);
        assert_eq!(reqs[0].messages[0].content, "hi");
        // Defaults: greedy, no seed, no ignore_eos, no budget.
        assert_eq!(reqs[1].temperature, Some(0.0));
        assert_eq!(reqs[1].seed, None);
        assert_eq!(reqs[1].ignore_eos, None);
        assert_eq!(reqs[1].max_tokens, None);
    }

    #[test]
    fn jsonl_blank_lines_are_skipped_not_counted() {
        let reqs = parse_jsonl("{\"prompt\":\"a\"}\n\n{\"prompt\":\"b\"}\n\n")
            .expect("blank lines are legal in JSONL");
        assert_eq!(reqs.len(), 2);
    }

    #[test]
    fn jsonl_role_is_honored() {
        let reqs = parse_jsonl("{\"prompt\":\"s\",\"role\":\"system\"}\n").expect("valid");
        assert_eq!(reqs[0].messages[0].role, Role::System);
    }

    /// The committed W1 corpus must load through THIS loader.
    ///
    /// Before PERF-039 the only committed corpus in the tree
    /// (`prompts-w2.jsonl`) could not be read by the only loader in the tree,
    /// and nothing noticed because nothing tried. This test is what makes the
    /// corpus and the loader a single artifact rather than two.
    #[test]
    fn committed_w1_corpus_loads() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../aprender-serve/benchmarks/qwen-coder/prompts-w1.jsonl");
        let reqs =
            load_from_file(&path).unwrap_or_else(|e| panic!("committed W1 corpus must load: {e}"));
        assert_eq!(reqs.len(), 256, "§4.4.2 needs 8*16 sampled + 2*16 warmup");
        for r in &reqs {
            assert_eq!(r.max_tokens, Some(128), "§4.3.1 tg128");
            assert_eq!(r.temperature, Some(0.0), "§4.3.1 greedy");
            assert_eq!(r.seed, Some(0), "§4.3.1 seed = 0");
        }
        let distinct: std::collections::HashSet<&str> = reqs
            .iter()
            .map(|r| r.messages[0].content.as_str())
            .collect();
        assert_eq!(
            distinct.len(),
            reqs.len(),
            "identical prompts would let prefix caching drive Arm A"
        );
    }

    /// The committed W2 corpus must load through the SAME loader.
    #[test]
    fn committed_w2_corpus_loads() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../aprender-serve/benchmarks/qwen-coder/prompts-w2.jsonl");
        let reqs =
            load_from_file(&path).unwrap_or_else(|e| panic!("committed W2 corpus must load: {e}"));
        assert_eq!(reqs.len(), 100);
    }

    #[test]
    fn test_load_from_file_missing() {
        let result = load_from_file(Path::new("/nonexistent/prompts.jsonl"));
        assert!(result.is_err());
    }

    #[test]
    fn test_all_profiles_have_deterministic_settings() {
        for profile in [
            PromptProfile::Micro,
            PromptProfile::Short,
            PromptProfile::Medium,
            PromptProfile::Long,
        ] {
            let prompts = load_profile(profile);
            for p in &prompts {
                assert_eq!(
                    p.temperature,
                    Some(0.0),
                    "Profile {profile:?} should be deterministic"
                );
                assert_eq!(p.stream, Some(false));
                assert!(
                    p.max_tokens.is_some(),
                    "Profile {profile:?} should set max_tokens"
                );
            }
        }
    }
}
