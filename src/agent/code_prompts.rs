//! System prompts and exit codes for `apr code` / `batuta code`.
//!
//! Extracted from code.rs to keep module under 500-line threshold.

/// Compact system prompt for -p mode (PMAT-197).
/// Minimal context to avoid overwhelming small models (Qwen3 1.7B).
/// The full CODE_SYSTEM_PROMPT causes Qwen3 1.7B to loop on `</think>` tags
/// when combined with 9 tool JSON schemas consuming most of the context window.
pub(super) const COMPACT_SYSTEM_PROMPT: &str = "\
Answer the question. Be direct.\
";

/// System prompt — PMAT-168: optimized for 1.5B-7B with explicit tool table.
pub(super) const CODE_SYSTEM_PROMPT: &str = "\
You are apr code, a sovereign AI coding assistant. All inference runs locally — \
no data ever leaves the machine.

## Tools

You have 9 tools. To use one, emit a <tool_call> block:

<tool_call>
{\"name\": \"tool_name\", \"input\": {\"param\": \"value\"}}
</tool_call>

| Tool | Use for | Example input |
|------|---------|---------------|
| file_read | Read a file | {\"path\": \"src/main.rs\"} |
| file_write | Create/overwrite file | {\"path\": \"new.rs\", \"content\": \"fn main() {}\"} |
| file_edit | Replace text in file | {\"path\": \"src/lib.rs\", \"old\": \"foo\", \"new\": \"bar\"} |
| glob | Find files by pattern | {\"pattern\": \"src/**/*.rs\"} |
| grep | Search file contents | {\"pattern\": \"TODO\", \"path\": \"src/\"} |
| shell | Run a command | {\"command\": \"cargo test --lib\"} |
| memory | Remember/recall facts | {\"action\": \"remember\", \"key\": \"bug\", \"value\": \"off-by-one\"} |
| pmat_query | Search code by intent | {\"query\": \"error handling\", \"limit\": 5} |
| rag | Search project docs | {\"query\": \"authentication flow\"} |

## Guidelines

- Read files before editing — understand first
- Use file_edit for changes, file_write only for new files
- Run tests after changes: shell with cargo test
- Use pmat_query for code search (returns quality-graded functions), glob for files, grep for text
- Be concise
";

/// Exit codes for non-interactive mode (spec §9.1).
pub mod exit_code {
    pub const SUCCESS: i32 = 0;
    pub const AGENT_ERROR: i32 = 1;
    pub const BUDGET_EXHAUSTED: i32 = 2;
    pub const MAX_TURNS: i32 = 3;
    pub const SANDBOX_VIOLATION: i32 = 4;
    pub const NO_MODEL: i32 = 5;
}

/// Mid-size system prompt for 2-7B models (PMAT-198).
/// Includes tool names and format but shorter examples than CODE_SYSTEM_PROMPT.
const MID_SYSTEM_PROMPT: &str = "\
You are apr code, a sovereign AI coding assistant. All inference runs locally.

To use a tool: <tool_call>{\"name\": \"tool\", \"input\": {...}}</tool_call>

Tools: file_read, file_write, file_edit, glob, grep, shell, memory, pmat_query, rag

- Read before editing. Use file_edit for changes, file_write for new files.
- Run tests: shell with cargo test
- Be concise
";

/// Estimate model parameter count (billions) from filename.
///
/// Parses patterns like `Qwen3-1.7B`, `llama-8b`, `phi-3-mini-3.8b`.
/// Returns 0.0 if no size found (caller should assume large model).
pub(super) fn estimate_model_params_from_name(path: &std::path::Path) -> f64 {
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();

    // Match patterns: "1.7b", "8b", "0.6b", "70b", "3.8b"
    // Look for a number followed by 'b' (case-insensitive, already lowered)
    let mut best: f64 = 0.0;
    let chars: Vec<char> = name.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        // Find start of a number
        if chars[i].is_ascii_digit() {
            let start = i;
            // Consume digits and optional decimal
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            // Check if followed by 'b'
            if i < chars.len() && chars[i] == 'b' {
                if let Ok(n) = name[start..i].parse::<f64>() {
                    if n > best {
                        best = n;
                    }
                }
            }
        }
        i += 1;
    }
    best
}

/// Select system prompt based on model parameter count (PMAT-198).
///
/// | Size | Prompt | Rationale |
/// |------|--------|-----------|
/// | <2B  | COMPACT | Avoids thinking loops, keeps tool format |
/// | 2-7B | MID | Tool names + format, no example JSON |
/// | 7B+  | FULL | Full table with examples + guidelines |
pub(super) fn scale_prompt_for_model(params_b: f64) -> String {
    if params_b < 2.0 {
        COMPACT_SYSTEM_PROMPT.to_string()
    } else if params_b < 7.0 {
        MID_SYSTEM_PROMPT.to_string()
    } else {
        CODE_SYSTEM_PROMPT.to_string()
    }
}

pub(super) fn map_error_to_exit_code(e: &crate::agent::result::AgentError) -> i32 {
    use crate::agent::result::AgentError;
    match e {
        AgentError::CircuitBreak(_) => exit_code::BUDGET_EXHAUSTED,
        AgentError::MaxIterationsReached => exit_code::MAX_TURNS,
        AgentError::CapabilityDenied { .. } => exit_code::SANDBOX_VIOLATION,
        _ => exit_code::AGENT_ERROR,
    }
}
