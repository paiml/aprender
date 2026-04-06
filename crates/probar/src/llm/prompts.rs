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

/// Load prompts from a YAML file.
///
/// Expected format:
/// ```yaml
/// prompts:
///   - role: user
///     content: "..."
///     max_tokens: 128
///     temperature: 0.0
/// ```
pub fn load_from_file(path: &Path) -> Result<Vec<ChatRequest>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

    let doc: PromptFile =
        serde_yaml_ng::from_str(&content).map_err(|e| format!("Failed to parse YAML: {e}"))?;

    if doc.prompts.is_empty() {
        return Err("Prompt file contains no prompts".to_string());
    }

    let requests = doc
        .prompts
        .into_iter()
        .map(|p| ChatRequest {
            model: String::new(),
            messages: vec![ChatMessage {
                role: parse_role(&p.role),
                content: p.content,
            }],
            temperature: Some(p.temperature.unwrap_or(0.0)),
            max_tokens: p.max_tokens,
            stream: Some(false),
        })
        .collect();

    Ok(requests)
}

/// YAML prompt file structure.
#[derive(Debug, serde::Deserialize)]
struct PromptFile {
    prompts: Vec<PromptEntry>,
}

/// A single prompt entry in a YAML file.
#[derive(Debug, serde::Deserialize)]
struct PromptEntry {
    role: String,
    content: String,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
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

    #[test]
    fn test_load_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prompts.yaml");
        std::fs::write(
            &path,
            r#"
prompts:
  - role: user
    content: "What is 2+2?"
    max_tokens: 16
    temperature: 0.0
  - role: user
    content: "Explain quicksort"
    max_tokens: 128
"#,
        )
        .unwrap();

        let requests = load_from_file(&path).unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].max_tokens, Some(16));
        assert_eq!(requests[0].temperature, Some(0.0));
        assert_eq!(requests[1].max_tokens, Some(128));
        assert_eq!(requests[1].temperature, Some(0.0)); // default
    }

    #[test]
    fn test_load_from_file_missing() {
        let result = load_from_file(Path::new("/nonexistent/prompts.yaml"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_from_file_empty_prompts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.yaml");
        std::fs::write(&path, "prompts: []\n").unwrap();
        let result = load_from_file(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no prompts"));
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
