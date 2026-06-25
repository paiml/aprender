//! CCPA teacher.stream.ndjson -> apr-code `<tool_call>` SFT export.
//!
//! Harvests real Claude Code teacher trajectories captured by the
//! claude-code-parity-apr (CCPA) project and remaps the Anthropic-native tool
//! schema onto the apr-code tool schema (see
//! `crates/aprender-orchestrate/src/agent/code_prompts.rs::CODE_SYSTEM_PROMPT`).
//!
//! Each captured assistant `tool_use` turn becomes one `entrenar` `InstructSample`
//! JSONL record `{instruction, response, system}` where:
//!   * `system`      = the apr-code CODE_SYSTEM_PROMPT (the format the student must learn)
//!   * `instruction` = the running observation transcript up to that turn (the
//!                     agentic context: prior tool_calls + their tool_results)
//!   * `response`    = the literal `<tool_call>{"name":..,"input":..}</tool_call>`
//!                     envelope the student is supposed to emit.
//!
//! The student base model emits 0 tool_calls (Markdown / prose prior); the goal of
//! the spike is to verify that SFT on this corpus produces the 0 -> 1 tool_call flip.
//!
//! Tool remap (Anthropic native -> apr-code):
//!   Read  -> file_read   (file_path -> path)
//!   Write -> file_write  (file_path -> path, content -> content)
//!   Edit  -> file_edit   (file_path -> path, old_string -> old, new_string -> new)
//!   Bash  -> shell       (command -> command)
//!   Grep  -> grep        (pattern, path)
//!   Glob  -> glob        (pattern)
//! Tools with no apr-code equivalent (Task*, ToolSearch, Agent, AskUserQuestion) are dropped.

use serde::Serialize;
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// apr-code CODE_SYSTEM_PROMPT (kept in sync with
/// crates/aprender-orchestrate/src/agent/code_prompts.rs). Embedded here so the
/// converter is a self-contained data tool with no heavy crate deps.
const CODE_SYSTEM_PROMPT: &str = "You are apr code, a sovereign AI coding assistant. All inference runs locally — no data ever leaves the machine.\n\n## Tools\n\nYou have 9 tools. To use one, emit a <tool_call> block:\n\n<tool_call>\n{\"name\": \"tool_name\", \"input\": {\"param\": \"value\"}}\n</tool_call>\n\n| Tool | Use for | Example input |\n|------|---------|---------------|\n| file_read | Read a file | {\"path\": \"src/main.rs\"} |\n| file_write | Create/overwrite file | {\"path\": \"new.rs\", \"content\": \"fn main() {}\"} |\n| file_edit | Replace text in file | {\"path\": \"src/lib.rs\", \"old\": \"foo\", \"new\": \"bar\"} |\n| glob | Find files by pattern | {\"pattern\": \"src/**/*.rs\"} |\n| grep | Search file contents | {\"pattern\": \"TODO\", \"path\": \"src/\"} |\n| shell | Run a command | {\"command\": \"cargo test --lib\"} |\n| memory | Remember/recall facts | {\"action\": \"remember\", \"key\": \"bug\", \"value\": \"off-by-one\"} |\n| pmat_query | Search code by intent | {\"query\": \"error handling\", \"limit\": 5} |\n| rag | Search project docs | {\"query\": \"authentication flow\"} |\n\n## Guidelines\n\n- Read files before editing — understand first\n- Use file_edit for changes, file_write only for new files\n- Run tests after changes: shell with cargo test\n- Be concise — DO NOT narrate what you're about to do; just emit the <tool_call>\n- DO NOT use Markdown ```rust``` code blocks for file edits; ALWAYS use file_edit or file_write tool_calls";

/// Synthesized task framing per fixture family (the original user task text was
/// elided from most CCPA captures — the stream begins mid-conversation). This is
/// a faithful, generic description of the agentic coding setup that holds for
/// every fixture so the model learns "given a coding repo + observations, emit
/// the next tool_call".
const TASK_HEADER: &str =
    "You are working in a Rust project. Investigate, then fix the failing code so the tests pass. \
Emit exactly one <tool_call> for your next action.";

#[derive(Serialize)]
struct InstructSample {
    instruction: String,
    response: String,
    system: String,
}

/// A remapped apr-code tool call.
struct AprToolCall {
    name: String,
    input: Value,
}

/// Remap an Anthropic-native tool_use block to an apr-code tool call.
/// Returns None for tools that have no apr-code equivalent.
fn remap_tool(name: &str, input: &Value) -> Option<AprToolCall> {
    let obj = input.as_object()?;
    let get = |k: &str| obj.get(k).cloned();
    let (apr_name, apr_input) = match name {
        "Read" => {
            let mut m = serde_json::Map::new();
            m.insert("path".into(), get("file_path")?);
            ("file_read", Value::Object(m))
        }
        "Write" => {
            let mut m = serde_json::Map::new();
            m.insert("path".into(), get("file_path")?);
            m.insert("content".into(), get("content").unwrap_or(Value::String(String::new())));
            ("file_write", Value::Object(m))
        }
        "Edit" => {
            let mut m = serde_json::Map::new();
            m.insert("path".into(), get("file_path")?);
            m.insert("old".into(), get("old_string")?);
            m.insert("new".into(), get("new_string")?);
            ("file_edit", Value::Object(m))
        }
        "Bash" => {
            let mut m = serde_json::Map::new();
            m.insert("command".into(), get("command")?);
            ("shell", Value::Object(m))
        }
        "Grep" => {
            let mut m = serde_json::Map::new();
            m.insert("pattern".into(), get("pattern")?);
            if let Some(p) = get("path") {
                m.insert("path".into(), p);
            }
            ("grep", Value::Object(m))
        }
        "Glob" => {
            let mut m = serde_json::Map::new();
            m.insert("pattern".into(), get("pattern")?);
            ("glob", Value::Object(m))
        }
        _ => return None,
    };
    Some(AprToolCall { name: apr_name.to_string(), input: apr_input })
}

/// Render the apr-code response envelope for a tool call.
fn render_tool_call(tc: &AprToolCall) -> String {
    let body = serde_json::json!({ "name": tc.name, "input": tc.input });
    format!(
        "<tool_call>\n{}\n</tool_call>",
        serde_json::to_string(&body).unwrap_or_default()
    )
}

/// Truncate long observation text so a single tool_result doesn't blow the
/// instruction context.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max).collect();
        format!("{head}\n…[truncated]")
    }
}

/// Extract a flat text string from a tool_result content (string or block list).
fn tool_result_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| {
                if let Some(t) = b.get("text").and_then(Value::as_str) {
                    Some(t.to_string())
                } else if b.is_string() {
                    b.as_str().map(str::to_string)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

struct Stats {
    streams: usize,
    raw_tool_use: usize,
    remapped: usize,
    dropped_unmapped: usize,
    emitted: usize,
    deduped: usize,
}

fn process_stream(
    path: &Path,
    stats: &mut Stats,
    seen: &mut HashSet<String>,
    samples: &mut Vec<InstructSample>,
    curated_only: bool,
) {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return,
    };
    stats.streams += 1;

    // Running transcript of observations the student has seen so far.
    let mut transcript: Vec<String> = Vec::new();
    let mut first_turn_emitted = false;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let o: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ty = o.get("type").and_then(Value::as_str).unwrap_or("");
        match ty {
            "assistant" => {
                let content = o
                    .get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(Value::as_array);
                let Some(blocks) = content else { continue };
                for b in blocks {
                    if b.get("type").and_then(Value::as_str) != Some("tool_use") {
                        continue;
                    }
                    stats.raw_tool_use += 1;
                    let name = b.get("name").and_then(Value::as_str).unwrap_or("");
                    let empty = Value::Object(serde_json::Map::new());
                    let input = b.get("input").unwrap_or(&empty);
                    let Some(tc) = remap_tool(name, input) else {
                        stats.dropped_unmapped += 1;
                        continue;
                    };
                    stats.remapped += 1;
                    let response = render_tool_call(&tc);

                    // Build the instruction: task header + running observation trail.
                    let mut instr = String::from(TASK_HEADER);
                    if !transcript.is_empty() {
                        instr.push_str("\n\n## Observations so far\n\n");
                        // Keep the last few observations to bound context.
                        let tail_start = transcript.len().saturating_sub(6);
                        for obs in &transcript[tail_start..] {
                            instr.push_str(obs);
                            instr.push_str("\n\n");
                        }
                    }
                    instr.push_str("\nEmit the <tool_call> for your next action.");

                    // For the curated spike set: keep only the FIRST tool_call of
                    // each trajectory (cleanest "task -> first action" signal) plus
                    // dedup. For the full set: keep every remapped turn.
                    let keep = if curated_only { !first_turn_emitted } else { true };
                    first_turn_emitted = true;

                    if keep {
                        let dkey = format!("{}\u{0}{}", instr, response);
                        if seen.insert(dkey) {
                            samples.push(InstructSample {
                                instruction: instr,
                                response: response.clone(),
                                system: CODE_SYSTEM_PROMPT.to_string(),
                            });
                            stats.emitted += 1;
                        } else {
                            stats.deduped += 1;
                        }
                    }

                    // Record this action into the transcript for downstream turns.
                    transcript.push(format!("Action: {}", render_tool_call(&tc)));
                }
            }
            "user" => {
                let content = o
                    .get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(Value::as_array);
                let Some(blocks) = content else { continue };
                for b in blocks {
                    if b.get("type").and_then(Value::as_str) == Some("tool_result") {
                        let result = b.get("content").cloned().unwrap_or(Value::Null);
                        let txt = truncate(&tool_result_text(&result), 600);
                        transcript.push(format!("Result:\n{txt}"));
                    }
                }
            }
            _ => {}
        }
    }
}

fn find_streams(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().and_then(|n| n.to_str()) == Some("teacher.stream.ndjson") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut evidence_root =
        PathBuf::from("/home/noah/src/claude-code-parity-apr/evidence");
    let mut out_path = PathBuf::from("apr_code_sft.jsonl");
    let mut curated_only = true;
    let mut balanced = false;
    let mut per_tool: usize = 40;
    let mut limit: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--evidence" => {
                i += 1;
                evidence_root = PathBuf::from(&args[i]);
            }
            "--out" => {
                i += 1;
                out_path = PathBuf::from(&args[i]);
            }
            "--full" => {
                curated_only = false;
                balanced = false;
            }
            "--curated" => {
                curated_only = true;
                balanced = false;
            }
            "--balanced" => {
                // Stratified curated set: up to `per_tool` samples per apr-code tool,
                // drawn from ALL turns (with real observation context), not just first
                // actions. Gives a diverse 0->1 flip signal across every tool type.
                curated_only = false;
                balanced = true;
            }
            "--per-tool" => {
                i += 1;
                per_tool = args[i].parse().unwrap_or(40);
            }
            "--limit" => {
                i += 1;
                limit = args[i].parse().ok();
            }
            "--help" | "-h" => {
                eprintln!(
                    "ccpa-sft-export — CCPA teacher.stream.ndjson -> apr-code tool_call SFT JSONL\n\n\
                     Usage: ccpa-sft-export [--evidence DIR] [--out FILE.jsonl] [--curated|--balanced|--full] [--per-tool N] [--limit N]\n\n\
                       --curated     one (first) tool_call per trajectory, deduped (default; spike set)\n\
                       --balanced    stratified set: up to --per-tool per apr-code tool, with context (recommended spike set)\n\
                       --full        every remappable tool_call turn (full corpus)\n\
                       --per-tool N  cap per tool in --balanced mode (default 40)\n\
                       --limit N     cap emitted samples at N"
                );
                return;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let streams = find_streams(&evidence_root);
    let mut stats = Stats {
        streams: 0,
        raw_tool_use: 0,
        remapped: 0,
        dropped_unmapped: 0,
        emitted: 0,
        deduped: 0,
    };
    let mut seen = HashSet::new();
    let mut samples = Vec::new();

    for s in &streams {
        process_stream(s, &mut stats, &mut seen, &mut samples, curated_only);
    }

    // Balanced stratification: cap each tool type at `per_tool` for a diverse set.
    if balanced {
        use std::collections::HashMap;
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut kept = Vec::new();
        for s in samples {
            // tool name = first "name":"X" in the response envelope.
            let name = s
                .response
                .find("\"name\":\"")
                .and_then(|p| {
                    let rest = &s.response[p + 8..];
                    rest.find('"').map(|e| rest[..e].to_string())
                })
                .unwrap_or_default();
            let c = counts.entry(name).or_insert(0);
            if *c < per_tool {
                *c += 1;
                kept.push(s);
            }
        }
        samples = kept;
    }

    if let Some(n) = limit {
        samples.truncate(n);
    }

    // Write JSONL.
    let mut buf = String::new();
    for s in &samples {
        buf.push_str(&serde_json::to_string(s).expect("serialize sample"));
        buf.push('\n');
    }
    std::fs::write(&out_path, buf).expect("write jsonl");

    eprintln!("=== ccpa-sft-export ===");
    eprintln!("evidence root : {}", evidence_root.display());
    eprintln!("streams       : {}", stats.streams);
    eprintln!("raw tool_use  : {}", stats.raw_tool_use);
    eprintln!("remapped      : {} (dropped unmapped: {})", stats.remapped, stats.dropped_unmapped);
    eprintln!("deduped out   : {}", stats.deduped);
    eprintln!("mode          : {}", if curated_only { "curated" } else { "full" });
    eprintln!("samples written: {}", samples.len());
    eprintln!("output        : {}", out_path.display());
}
