//! #4599 falsifiers: an over-budget prompt is never silently dropped.
//!
//! The #3715 release cells found `apr code -p` answering a 212 KB prompt with rc 0 and
//! `tokens_in 19`: the sliding window skipped the only user message whole and the model
//! answered an empty conversation. These run the agent loop against a driver that records
//! what it was actually sent.

use super::*;
use crate::agent::driver::mock::MockDriver;
use crate::agent::memory::InMemorySubstrate;
use crate::agent::result::TokenUsage;
use async_trait::async_trait;
use std::sync::Mutex;

/// The needle the #3715 cells plant at token 0 (`model_ladder_cells_produce.py` NEEDLE_WORD).
const NEEDLE: &str = "TANGERINE-4417";
/// Qwen3.5's GGUF `qwen35.context_length` (inv.enriched.jsonl of the #3715 pilot).
const QWEN35_CONTEXT: usize = 262_144;
/// The window `apr code` hard-coded before #4599.
const OLD_CODE_WINDOW: usize = 32_768;

/// The 8 failing #3715 cells: verb `code`, rungs 60k and consumer-max, thinking on and off,
/// on lambda and gx10. The prompt bytes are the pilot's own prompt files; both hosts and both
/// thinking modes send the same bytes per rung, so the rung alone decides whether it fits.
const FAILING_CELLS: [(&str, &str, &str, usize); 8] = [
    ("lambda", "60k", "off", 212_046),
    ("lambda", "60k", "on", 212_046),
    ("lambda", "consumer-max", "off", 510_772),
    ("lambda", "consumer-max", "on", 510_772),
    ("gx10", "60k", "off", 212_046),
    ("gx10", "60k", "on", 212_046),
    ("gx10", "consumer-max", "off", 510_772),
    ("gx10", "consumer-max", "on", 510_772),
];

/// Records the user messages of every request it is sent, and echoes the input it received as
/// `usage.input_tokens` under a byte-level tokenizer (1 byte = 1 token): exact, so a single lost
/// byte changes the count. (The 4-bytes-per-token estimate would not: 212,046 and 212,045 bytes
/// both estimate to 53,012.)
struct RecordingDriver {
    window: usize,
    seen: Mutex<Vec<String>>,
}

impl RecordingDriver {
    fn new(window: usize) -> Self {
        Self { window, seen: Mutex::new(Vec::new()) }
    }

    fn user_messages(&self) -> Vec<String> {
        self.seen.lock().expect("lock").clone()
    }
}

#[async_trait]
impl LlmDriver for RecordingDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, AgentError> {
        let mut seen = self.seen.lock().expect("lock");
        let mut input_tokens = 0u64;
        for m in &request.messages {
            if let Message::User(text) = m {
                input_tokens += text.len() as u64;
                seen.push(text.clone());
            }
        }
        Ok(CompletionResponse {
            text: NEEDLE.into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: TokenUsage { input_tokens, output_tokens: 0 },
        })
    }

    fn context_window(&self) -> usize {
        self.window
    }

    fn privacy_tier(&self) -> crate::serve::backends::PrivacyTier {
        crate::serve::backends::PrivacyTier::Sovereign
    }
}

/// A needle-at-token-0 prompt of exactly `bytes` bytes.
fn needle_prompt(bytes: usize) -> String {
    let mut p = format!("The passphrase is {NEEDLE}. Remember it.\n");
    while p.len() < bytes {
        p.push_str("filler text for the context rung. ");
    }
    p.truncate(bytes);
    p
}

fn manifest() -> AgentManifest {
    let mut m = AgentManifest::default();
    m.model.max_tokens = 4096;
    m
}

async fn run(prompt: &str, driver: &RecordingDriver) -> Result<AgentLoopResult, AgentError> {
    run_agent_loop(
        &manifest(),
        prompt,
        driver,
        &ToolRegistry::new(),
        &InMemorySubstrate::new(),
        None,
    )
    .await
}

/// The cell verdict: the model received exactly one user message, identical to the prompt
/// (needle at the head, tail present), and the echoed input token count equals the prompt's.
fn cell_verdict(prompt: &str, sent: &[String], echoed_input_tokens: u64) -> Result<(), String> {
    if sent.len() != 1 {
        return Err(format!("{} user messages reached the model, want 1", sent.len()));
    }
    if echoed_input_tokens != prompt.len() as u64 {
        return Err(format!("echoed input tokens {echoed_input_tokens} != {}", prompt.len()));
    }
    if !sent[0].starts_with("The passphrase is TANGERINE-4417") {
        return Err("the needle at token 0 is missing".into());
    }
    if sent[0] != prompt {
        return Err("the prompt the model received differs from the one sent (tail lost?)".into());
    }
    Ok(())
}

/// FALSIFY-4599-001: at the model's own context length, every failing #3715 cell sends the
/// whole prompt: echoed input token count == expected, needle and tail included.
#[tokio::test]
async fn falsify_4599_001_failing_cells_fit_the_models_window() {
    for (host, rung, think, bytes) in FAILING_CELLS {
        let prompt = needle_prompt(bytes);
        let driver = RecordingDriver::new(QWEN35_CONTEXT);
        let r = run(&prompt, &driver).await;
        let r = r.unwrap_or_else(|e| panic!("{host} {rung} think {think} ({bytes} B): {e}"));
        assert_eq!(r.usage.input_tokens, bytes as u64, "{host} {rung} think {think}: echoed");
        let v = cell_verdict(&prompt, &driver.user_messages(), r.usage.input_tokens);
        assert_eq!(v, Ok(()), "{host} {rung} think {think} ({bytes} B)");
    }
}

/// FALSIFY-4599-008: a planted 1-byte truncation is RED. For both rungs, a model that received
/// the prompt minus its last byte fails the cell verdict, on the count and on the content.
#[test]
fn falsify_4599_008_planted_one_byte_truncation_is_red() {
    for bytes in [212_046, 510_772] {
        let prompt = needle_prompt(bytes);
        let cut = prompt[..bytes - 1].to_string();
        let v = cell_verdict(&prompt, &[cut.clone()], cut.len() as u64);
        assert!(v.is_err(), "{bytes} B: a 1-byte truncation passed the cell verdict");
        // Content alone catches it too, if a driver echoed the right count by accident.
        assert!(cell_verdict(&prompt, &[cut], bytes as u64).is_err(), "{bytes} B: content");
        assert_eq!(cell_verdict(&prompt, &[prompt.clone()], bytes as u64), Ok(()));
    }
}

/// FALSIFY-4599-002: at the old 32K window the same cells are refused as `ContextOverflow`,
/// and the driver is never called: no answer to a conversation the prompt was dropped from.
#[tokio::test]
async fn falsify_4599_002_failing_cells_are_refused_not_dropped_at_32k() {
    for (host, rung, think, bytes) in FAILING_CELLS {
        let driver = RecordingDriver::new(OLD_CODE_WINDOW);
        let r = run(&needle_prompt(bytes), &driver).await;
        assert!(
            matches!(r, Err(AgentError::ContextOverflow { .. })),
            "{host} {rung} think {think} ({bytes} B) must be refused, got {r:?}"
        );
        assert!(driver.user_messages().is_empty(), "{host} {rung}: the driver was called");
    }
}

/// FALSIFY-4599-003: the issue's own boundary. A prompt over the 32K budget is refused; a 70 KB
/// prompt (the 20k rung, which passed) still goes through whole. The budget here is
/// 32768 − 4096 reserve = 28672 estimated tokens (4 bytes each, no system prompt or tools), so
/// 128 KiB is over it; in `apr code` the system prompt and tool schemas lower the boundary
/// further, to the ~105 KB the issue observed.
#[tokio::test]
async fn falsify_4599_003_prompt_over_budget_is_refused_at_32k() {
    let driver = RecordingDriver::new(OLD_CODE_WINDOW);
    let r = run(&needle_prompt(128 * 1024), &driver).await;
    assert!(matches!(r, Err(AgentError::ContextOverflow { .. })), "got {r:?}");
    assert!(driver.user_messages().is_empty());

    let driver = RecordingDriver::new(OLD_CODE_WINDOW);
    let r = run(&needle_prompt(70_678), &driver).await;
    assert!(r.is_ok(), "the 20k rung fits 32K: {r:?}");
    assert_eq!(driver.user_messages()[0].len(), 70_678);
}

/// FALSIFY-4599-004: the refusal is the envelope's named `context_overflow`, with a nonzero
/// exit code, never rc 0.
#[tokio::test]
async fn falsify_4599_004_refusal_is_named_and_nonzero() {
    let driver = MockDriver::single_response("unreachable").with_context_window(OLD_CODE_WINDOW);
    let r = run_agent_loop(
        &manifest(),
        &needle_prompt(212_046),
        &driver,
        &ToolRegistry::new(),
        &InMemorySubstrate::new(),
        None,
    )
    .await;
    let e = r.expect_err("a 212 KB prompt at 32K must be refused");
    let code = crate::agent::code_prompts::map_error_to_exit_code(&e);
    assert_ne!(code, 0, "a refusal exits nonzero");
    let outcome = crate::agent::code_envelope::CodeOutcome::from_agent_error(&e, code);
    assert_eq!(outcome.kind, "context_overflow");
    assert_eq!(outcome.exit_code, code);
}

fn tool_turn(prompt: &str, result_bytes: usize) -> Vec<Message> {
    vec![
        Message::User(prompt.to_string()),
        Message::AssistantToolUse(crate::agent::driver::ToolCall {
            id: "t1".into(),
            name: "file_read".into(),
            input: serde_json::json!({"path": "big.rs"}),
        }),
        Message::ToolResult(crate::agent::driver::ToolResultMsg {
            tool_use_id: "t1".into(),
            content: "x".repeat(result_bytes),
            is_error: false,
        }),
    ]
}

fn ctx_32k() -> crate::serve::context::ContextManager {
    use crate::serve::context::{ContextConfig, ContextManager, ContextWindow, TruncationStrategy};
    ContextManager::new(ContextConfig {
        window: ContextWindow::new(OLD_CODE_WINDOW, 4096),
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: false,
        min_messages: 2,
    })
}

/// FALSIFY-4599-006: the drop by another path (quorum, PR #4601). A 20K-token prompt whose own
/// 10K-token tool result arrives in a 28,672-token budget: the sliding window keeps the newest
/// (the tool result) and would evict the prompt. That is refused, never sent without the prompt.
#[test]
fn falsify_4599_006_tool_result_never_evicts_the_turns_prompt() {
    let msgs = tool_turn(&needle_prompt(80_000), 40_000);
    let r = truncate_messages(&msgs, &ctx_32k());
    assert!(matches!(r, Err(AgentError::ContextOverflow { .. })), "got {:?}", r.map(|m| m.len()));

    // Anti-vacuity: the same turn that fits is kept whole, prompt first.
    let msgs = tool_turn(&needle_prompt(40_000), 20_000);
    let kept = truncate_messages(&msgs, &ctx_32k()).expect("fits");
    assert_eq!(kept.len(), 3);
    assert!(matches!(&kept[0], Message::User(p) if p.len() == 40_000));
}

/// FALSIFY-4599-007: an older turn may still be evicted; only the CURRENT turn's prompt is pinned.
#[test]
fn falsify_4599_007_older_turns_may_still_be_evicted() {
    let mut msgs =
        vec![Message::User("x".repeat(100_000)), Message::Assistant("old answer".into())];
    msgs.extend(tool_turn("what is in big.rs?", 20_000));
    let kept = truncate_messages(&msgs, &ctx_32k()).expect("the current turn fits");
    // The 100 KB old prompt is evicted; the current turn is kept whole, prompt first. (A small
    // older message can survive past the evicted one: pre-existing, not #4599's.)
    assert!(!kept.iter().any(|m| matches!(m, Message::User(p) if p.len() == 100_000)));
    let turn = &kept[kept.len() - 3..];
    assert!(matches!(&turn[0], Message::User(p) if p == "what is in big.rs?"), "{kept:?}");
    assert!(matches!(&turn[2], Message::ToolResult(r) if r.content.len() == 20_000));
}
