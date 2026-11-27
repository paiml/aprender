//! #4600 falsifiers: the prompt the driver receives is the prompt that was sent.

use super::*;
use crate::agent::capability::Capability;
use crate::agent::code_envelope::{envelope, CodeOutcome};
use crate::agent::driver::mock::MockDriver;
use crate::agent::driver::{CompletionRequest, CompletionResponse, LlmDriver, ToolDefinition};
use crate::agent::manifest::AgentManifest;
use crate::agent::memory::InMemorySubstrate;
use crate::agent::result::AgentLoopResult;
use crate::agent::result::{StopReason, TokenUsage};
use crate::agent::runtime::{run_agent_loop, run_agent_loop_with_nudge};
use crate::agent::tool::{Tool, ToolRegistry, ToolResult};
use crate::serve::backends::PrivacyTier;
use crate::serve::context::{ContextConfig, ContextWindow};

fn context() -> ContextManager {
    ContextManager::new(ContextConfig {
        window: ContextWindow::new(262_144, 4096),
        ..ContextConfig::default()
    })
}

fn prompt_200kb() -> String {
    let mut p = "The passphrase is TANGERINE-4417.\n".to_string();
    while p.len() < 200 * 1024 {
        p.push_str("filler text for the context rung. ");
    }
    p.truncate(200 * 1024);
    p
}

async fn run(prompt: &str, window: usize) -> Result<AgentLoopResult, AgentError> {
    let driver = MockDriver::single_response("ok").with_context_window(window);
    let mut manifest = AgentManifest::default();
    manifest.model.max_tokens = 4096;
    run_agent_loop(
        &manifest,
        prompt,
        &driver,
        &ToolRegistry::new(),
        &InMemorySubstrate::new(),
        None,
    )
    .await
}

/// FALSIFY-4600-001: a planted one-byte drop is `input_not_consumed` with a nonzero exit.
#[test]
fn falsify_4600_001_planted_one_byte_drop_is_refused() {
    let prompt = prompt_200kb();
    let cut = prompt[..prompt.len() - 1].to_string();
    let err = InputRecord::measure(&prompt, &[Message::User(cut)], &context())
        .expect_err("a one-byte drop passed");
    assert!(
        matches!(err, AgentError::InputNotConsumed { sent: 204_800, consumed: 204_799 }),
        "got {err:?}"
    );
    let code = crate::agent::code_prompts::map_error_to_exit_code(&err);
    assert_ne!(code, 0);
    let doc = envelope(
        None,
        Some(&CodeOutcome::from_agent_error(&err, code)),
        None,
        std::time::Duration::ZERO,
    );
    let doc: serde_json::Value = serde_json::from_str(&doc).expect("json");
    assert_eq!(doc["error"]["kind"], "input_not_consumed");
    assert_eq!(doc["is_error"], true);
}

/// A driver that counts the requests it is handed. It answers `ok` when called.
struct CountingDriver {
    window: usize,
    calls: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl LlmDriver for CountingDriver {
    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, AgentError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(answer("ok"))
    }
    fn context_window(&self) -> usize {
        self.window
    }
    fn privacy_tier(&self) -> PrivacyTier {
        PrivacyTier::Sovereign
    }
}

/// FALSIFY-4600-006: a prompt the loop cannot send whole is refused at `run_agent_loop` and the
/// driver is never called. Sliding-window truncation never cuts a message's content, so the
/// loop-level refusal for an oversized prompt is `context_overflow` (#4599), not
/// `input_not_consumed`; the byte-level refusal is covered by the `measure` falsifiers above.
#[tokio::test]
async fn falsify_4600_006_overflowing_prompt_refused_before_driver_call() {
    let driver = CountingDriver { window: 8_192, calls: std::sync::atomic::AtomicUsize::new(0) };
    let mut manifest = AgentManifest::default();
    manifest.model.max_tokens = 4096;
    let prompt = prompt_200kb();
    let err = run_agent_loop(
        &manifest,
        &prompt,
        &driver,
        &ToolRegistry::new(),
        &InMemorySubstrate::new(),
        None,
    )
    .await
    .expect_err("a 200 KB prompt fit a 4096-token input budget");
    assert!(matches!(err, AgentError::ContextOverflow { .. }), "got {err:?}");
    assert_eq!(
        driver.calls.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "the driver was called with a prompt the loop refused"
    );
}

/// FALSIFY-4600-002: a changed byte at the head, and a request with no prompt, are refused.
#[test]
fn falsify_4600_002_changed_or_missing_prompt_is_refused() {
    let changed = InputRecord::measure("abc", &[Message::User("xbc".into())], &context());
    assert!(matches!(changed, Err(AgentError::InputNotConsumed { sent: 3, consumed: 0 })));
    let missing = InputRecord::measure("abc", &[Message::Assistant("abc".into())], &context());
    assert!(matches!(missing, Err(AgentError::InputNotConsumed { sent: 3, consumed: 0 })));
}

/// FALSIFY-4600-003: a 200 KB prompt through the agent loop is recorded whole, and the four
/// fields reach the `-p --output-format json` document.
#[tokio::test]
async fn falsify_4600_003_200kb_prompt_is_recorded_whole() {
    let prompt = prompt_200kb();
    let r = run(&prompt, 262_144).await.expect("fits");
    assert_eq!(r.input.input_bytes_sent, 204_800);
    assert_eq!(r.input.input_bytes_consumed, 204_800);
    assert_eq!(r.input.prompt_sha256, format!("{:x}", Sha256::digest(prompt.as_bytes())));
    assert!(r.input.prompt_tokens >= 204_800 / 4, "{}", r.input.prompt_tokens);

    let doc = envelope(Some(&r), None, None, std::time::Duration::ZERO);
    let doc: serde_json::Value = serde_json::from_str(&doc).expect("json");
    assert_eq!(doc["input_bytes_sent"], 204_800);
    assert_eq!(doc["input_bytes_consumed"], 204_800);
    assert_eq!(doc["prompt_sha256"], r.input.prompt_sha256.as_str());
    assert_eq!(doc["prompt_tokens"], r.input.prompt_tokens);
}

/// FALSIFY-4600-004: with no loop result the fields are null (not measured), never 0.
#[test]
fn falsify_4600_004_unmeasured_is_null() {
    let doc = envelope(None, None, None, std::time::Duration::ZERO);
    let doc: serde_json::Value = serde_json::from_str(&doc).expect("json");
    for key in ["input_bytes_sent", "input_bytes_consumed", "prompt_sha256", "prompt_tokens"] {
        assert_eq!(doc.get(key), Some(&serde_json::Value::Null), "{key}");
    }
}

struct NoopTool;

#[async_trait::async_trait]
impl Tool for NoopTool {
    fn name(&self) -> &'static str {
        "noop"
    }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "noop".into(),
            description: "does nothing".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }
    async fn execute(&self, _input: serde_json::Value) -> ToolResult {
        ToolResult::success("")
    }
    fn required_capability(&self) -> Capability {
        Capability::Memory
    }
}

/// FALSIFY-4600-005: after the tool-use nudge, the record is still the caller's prompt.
#[tokio::test]
async fn falsify_4600_005_nudge_keeps_the_callers_record() {
    let driver = MockDriver::new(vec![answer("first"), answer("second")]);
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(NoopTool));
    let r = run_agent_loop_with_nudge(
        &AgentManifest::default(),
        "caller prompt",
        &driver,
        &tools,
        &InMemorySubstrate::new(),
        None,
    )
    .await
    .expect("runs");
    assert_eq!(r.text, "second", "the nudge turn ran");
    assert_eq!(r.input.input_bytes_sent, "caller prompt".len());
}

fn answer(text: &str) -> CompletionResponse {
    CompletionResponse {
        text: text.into(),
        stop_reason: StopReason::EndTurn,
        tool_calls: vec![],
        usage: TokenUsage::default(),
    }
}
