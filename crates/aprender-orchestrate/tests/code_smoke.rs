//! Smoke test: `batuta code` end-to-end pipeline with MockDriver.
//!
//! Proves the full agent loop works: MockDriver → tool_use(file_read) → tool
//! executes → final response. This is the minimum viability test for apr code.
//!
//! Refs: PMAT-107, apr-code-feasibility-falsification.md §7 F-1

#![cfg(feature = "agents")]

use std::path::PathBuf;

use batuta::agent::capability::Capability;
use batuta::agent::driver::mock::MockDriver;
use batuta::agent::manifest::{AgentManifest, ModelConfig, ResourceQuota};
use batuta::agent::memory::InMemorySubstrate;
use batuta::agent::runtime::run_agent_loop;
use batuta::agent::tool::file::{FileEditTool, FileReadTool, FileWriteTool};
use batuta::agent::tool::search::{GlobTool, GrepTool};
use batuta::agent::tool::shell::ShellTool;
use batuta::agent::tool::ToolRegistry;
use batuta::serve::backends::PrivacyTier;

/// Build the same default manifest as `cli::code::build_default_manifest`.
fn code_manifest() -> AgentManifest {
    AgentManifest {
        name: "apr-code-test".to_string(),
        description: "Smoke test".to_string(),
        privacy: PrivacyTier::Standard,
        model: ModelConfig {
            system_prompt: "You are a coding assistant.".to_string(),
            max_tokens: 1024,
            temperature: 0.0,
            ..ModelConfig::default()
        },
        resources: ResourceQuota {
            max_iterations: 10,
            max_tool_calls: 20,
            max_cost_usd: 0.0,
            max_tokens_budget: None,
        },
        capabilities: vec![
            Capability::FileRead { allowed_paths: vec!["*".into()] },
            Capability::FileWrite { allowed_paths: vec!["*".into()] },
            Capability::Shell { allowed_commands: vec!["*".into()] },
            Capability::Memory,
        ],
        ..AgentManifest::default()
    }
}

/// Build the same tool registry as `cli::code::build_code_tools`.
fn code_tools() -> ToolRegistry {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(FileReadTool::new(vec!["*".into()])));
    tools.register(Box::new(FileWriteTool::new(vec!["*".into()])));
    tools.register(Box::new(FileEditTool::new(vec!["*".into()])));
    tools.register(Box::new(GlobTool::new(vec!["*".into()])));
    tools.register(Box::new(GrepTool::new(vec!["*".into()])));
    tools.register(Box::new(ShellTool::new(vec!["*".into()], cwd)));
    tools
}

/// Smoke test: agent loop with MockDriver that reads a file.
#[tokio::test]
async fn test_code_smoke_file_read() {
    // Create a temp file to read
    let dir = tempfile::TempDir::new().unwrap();
    let test_file = dir.path().join("hello.rs");
    std::fs::write(&test_file, "fn main() { println!(\"hello\"); }\n").unwrap();

    let manifest = code_manifest();
    let memory = InMemorySubstrate::new();
    let tools = code_tools();

    // MockDriver: first response is tool_use(file_read), second is final text
    let driver = MockDriver::tool_then_response(
        "file_read",
        serde_json::json!({"path": test_file.to_str().unwrap()}),
        "The file contains a hello world program.",
    );

    let result = run_agent_loop(&manifest, "Read hello.rs", &driver, &tools, &memory, None)
        .await
        .expect("agent loop failed");

    assert_eq!(result.text, "The file contains a hello world program.");
    assert!(result.iterations >= 2, "expected at least 2 iterations (tool_use + end_turn)");
    assert_eq!(result.tool_calls, 1, "expected 1 tool call");
}

/// Smoke test: agent loop with MockDriver that writes a file.
#[tokio::test]
async fn test_code_smoke_file_write() {
    let dir = tempfile::TempDir::new().unwrap();
    let target_file = dir.path().join("output.txt");

    let manifest = code_manifest();
    let memory = InMemorySubstrate::new();
    let tools = code_tools();

    let driver = MockDriver::tool_then_response(
        "file_write",
        serde_json::json!({
            "path": target_file.to_str().unwrap(),
            "content": "Hello from apr code!"
        }),
        "File written successfully.",
    );

    let result = run_agent_loop(&manifest, "Write output.txt", &driver, &tools, &memory, None)
        .await
        .expect("agent loop failed");

    assert_eq!(result.text, "File written successfully.");
    assert!(target_file.exists(), "file was not created");
    assert_eq!(std::fs::read_to_string(&target_file).unwrap(), "Hello from apr code!");
}

/// Smoke test: agent loop with MockDriver that edits a file.
#[tokio::test]
async fn test_code_smoke_file_edit() {
    let dir = tempfile::TempDir::new().unwrap();
    let test_file = dir.path().join("code.rs");
    std::fs::write(&test_file, "fn main() {\n    println!(\"hello\");\n}\n").unwrap();

    let manifest = code_manifest();
    let memory = InMemorySubstrate::new();
    let tools = code_tools();

    let driver = MockDriver::tool_then_response(
        "file_edit",
        serde_json::json!({
            "path": test_file.to_str().unwrap(),
            "old_string": "println!(\"hello\")",
            "new_string": "println!(\"world\")"
        }),
        "Changed hello to world.",
    );

    let result = run_agent_loop(&manifest, "Fix the greeting", &driver, &tools, &memory, None)
        .await
        .expect("agent loop failed");

    assert_eq!(result.text, "Changed hello to world.");
    let content = std::fs::read_to_string(&test_file).unwrap();
    assert!(content.contains("println!(\"world\")"), "edit not applied");
    assert!(!content.contains("println!(\"hello\")"), "old string still present");
}

/// Smoke test: agent loop with MockDriver that runs shell command.
#[tokio::test]
async fn test_code_smoke_shell() {
    let manifest = code_manifest();
    let memory = InMemorySubstrate::new();
    let tools = code_tools();

    let driver = MockDriver::tool_then_response(
        "shell",
        serde_json::json!({"command": "echo hello_from_agent"}),
        "Shell returned hello.",
    );

    let result =
        run_agent_loop(&manifest, "Run echo", &driver, &tools, &memory, None).await.expect("fail");

    assert_eq!(result.text, "Shell returned hello.");
    assert_eq!(result.tool_calls, 1);
}

/// Smoke test: agent loop with MockDriver that uses glob.
#[tokio::test]
async fn test_code_smoke_glob() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(dir.path().join("a.rs"), "").unwrap();
    std::fs::write(dir.path().join("b.rs"), "").unwrap();

    let manifest = code_manifest();
    let memory = InMemorySubstrate::new();
    let tools = code_tools();

    let driver = MockDriver::tool_then_response(
        "glob",
        serde_json::json!({
            "pattern": "*.rs",
            "path": dir.path().to_str().unwrap()
        }),
        "Found 2 rust files.",
    );

    let result = run_agent_loop(&manifest, "Find rs files", &driver, &tools, &memory, None)
        .await
        .expect("fail");

    assert_eq!(result.text, "Found 2 rust files.");
}

/// Smoke test: verify tool count matches spec (7 tools).
#[test]
fn test_code_tool_count() {
    let tools = code_tools();
    // file_read, file_write, file_edit, glob, grep, shell = 6
    // (memory is registered separately in cli::code, not in code_tools() here)
    assert_eq!(tools.len(), 6, "expected 6 tools in base code_tools()");
}

/// Smoke test: simple end_turn without tool use.
#[tokio::test]
async fn test_code_smoke_no_tools() {
    let manifest = code_manifest();
    let memory = InMemorySubstrate::new();
    let tools = code_tools();
    let driver = MockDriver::single_response("I can help you with coding tasks.");

    let result =
        run_agent_loop(&manifest, "Hello", &driver, &tools, &memory, None).await.expect("fail");

    assert_eq!(result.text, "I can help you with coding tasks.");
    assert_eq!(result.tool_calls, 0);
    assert_eq!(result.iterations, 1);
}

// ── Session Persistence Integration Tests ──

/// Test that SessionStore roundtrips messages through JSONL.
#[test]
fn test_session_roundtrip() {
    use batuta::agent::driver::{Message, ToolCall, ToolResultMsg};
    use batuta::agent::session::SessionStore;

    let store = SessionStore::create("integration-test").expect("create");

    // Write diverse message types
    let messages = vec![
        Message::User("Fix the bug".into()),
        Message::AssistantToolUse(ToolCall {
            id: "t1".into(),
            name: "file_read".into(),
            input: serde_json::json!({"path": "src/main.rs"}),
        }),
        Message::ToolResult(ToolResultMsg {
            tool_use_id: "t1".into(),
            content: "fn main() {}".into(),
            is_error: false,
        }),
        Message::Assistant("I found the bug.".into()),
    ];
    store.append_messages(&messages).expect("append");

    // Reload from disk
    let loaded = store.load_messages().expect("load");
    assert_eq!(loaded.len(), 4);
    assert!(matches!(&loaded[0], Message::User(s) if s == "Fix the bug"));
    assert!(matches!(&loaded[1], Message::AssistantToolUse(tc) if tc.name == "file_read"));
    assert!(matches!(&loaded[2], Message::ToolResult(tr) if tr.content == "fn main() {}"));
    assert!(matches!(&loaded[3], Message::Assistant(s) if s == "I found the bug."));

    let _ = std::fs::remove_dir_all(&store.dir);
}

/// Test that tool definitions are injected into prompt for local models.
#[test]
fn test_tool_definitions_in_prompt() {
    use batuta::agent::driver::chat_template::{format_prompt_with_template, ChatTemplate};
    use batuta::agent::driver::{CompletionRequest, Message, ToolDefinition};

    let tools = vec![ToolDefinition {
        name: "file_read".into(),
        description: "Read a file".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to read"}
            }
        }),
    }];

    let request = CompletionRequest {
        model: String::new(),
        messages: vec![Message::User("Read main.rs".into())],
        tools,
        max_tokens: 1024,
        temperature: 0.0,
        system: Some("You are a coding assistant.".into()),
    };

    let prompt = format_prompt_with_template(&request, ChatTemplate::ChatMl);

    // Tool definition must appear in the prompt
    assert!(prompt.contains("file_read"), "tool name missing from prompt");
    assert!(prompt.contains("Read a file"), "tool description missing");
    assert!(prompt.contains("path (string)"), "tool schema missing");
    assert!(prompt.contains("<tool_call>"), "tool call format missing");
    assert!(prompt.contains("tool_result"), "tool result format missing");
}

/// Test multi-turn history with run_agent_turn.
#[tokio::test]
async fn test_multi_turn_session_integration() {
    use batuta::agent::driver::Message;
    use batuta::agent::runtime::run_agent_turn;

    let manifest = code_manifest();
    let memory = InMemorySubstrate::new();
    let tools = code_tools();

    let driver = MockDriver::new(vec![
        batuta::agent::driver::CompletionResponse {
            text: "I see the project.".into(),
            stop_reason: batuta::agent::result::StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
        batuta::agent::driver::CompletionResponse {
            text: "The bug is on line 42.".into(),
            stop_reason: batuta::agent::result::StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    ]);

    let mut history: Vec<Message> = Vec::new();

    // Turn 1
    let r1 = run_agent_turn(
        &manifest,
        &mut history,
        "Look at the project",
        &driver,
        &tools,
        &memory,
        None,
    )
    .await
    .expect("turn 1");
    assert_eq!(r1.text, "I see the project.");
    assert_eq!(history.len(), 2); // User + Assistant

    // Turn 2 — should have context from turn 1
    let r2 = run_agent_turn(
        &manifest,
        &mut history,
        "Where is the bug?",
        &driver,
        &tools,
        &memory,
        None,
    )
    .await
    .expect("turn 2");
    assert_eq!(r2.text, "The bug is on line 42.");
    assert_eq!(history.len(), 4); // 2 from turn 1 + 2 from turn 2
}
