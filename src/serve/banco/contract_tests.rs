//! Provable contract falsification tests for Banco.
//!
//! Generated from contracts/banco/*.yaml, then wired to real implementations.
//! Each test attempts to falsify a claim from the Banco spec.

use crate::serve::backends::{BackendSelector, PrivacyTier, ServingBackend};
use crate::serve::circuit_breaker::{CircuitBreakerConfig, CircuitState, CostCircuitBreaker};
use crate::serve::context::{ContextConfig, ContextManager, ContextWindow, TruncationStrategy};
use crate::serve::router::{RouterConfig, RoutingDecision, SpilloverRouter};
use crate::serve::templates::{ChatMessage, ChatTemplateEngine, TemplateFormat};

// ============================================================================
// F-BANCO-001: Privacy Tier Enforcement (privacy-enforcement-v1.yaml)
// ============================================================================

/// FALSIFY-PRIV-001: Sovereign blocks ALL remote backends.
/// Prediction: PrivacyTier::Sovereign.allows(b) = false for every remote backend.
/// If fails: New remote backend added without privacy gate update.
#[test]
#[allow(non_snake_case)]
fn falsify_PRIV_001_sovereign_blocks_all_remote() {
    let remote_backends = [
        ServingBackend::HuggingFace,
        ServingBackend::Together,
        ServingBackend::Replicate,
        ServingBackend::Anyscale,
        ServingBackend::Modal,
        ServingBackend::Fireworks,
        ServingBackend::Groq,
        ServingBackend::OpenAI,
        ServingBackend::Anthropic,
        ServingBackend::AzureOpenAI,
        ServingBackend::AwsBedrock,
        ServingBackend::GoogleVertex,
        ServingBackend::AwsLambda,
        ServingBackend::CloudflareWorkers,
    ];
    for backend in &remote_backends {
        assert!(
            !PrivacyTier::Sovereign.allows(*backend),
            "Sovereign must block {:?} but allows() returned true",
            backend
        );
    }
}

/// FALSIFY-PRIV-002: Sovereign allows ALL local backends.
/// Prediction: PrivacyTier::Sovereign.allows(b) = true for every local backend.
#[test]
#[allow(non_snake_case)]
fn falsify_PRIV_002_sovereign_allows_all_local() {
    let local_backends = [
        ServingBackend::Realizar,
        ServingBackend::Ollama,
        ServingBackend::LlamaCpp,
        ServingBackend::Llamafile,
        ServingBackend::Candle,
        ServingBackend::Vllm,
        ServingBackend::Tgi,
        ServingBackend::LocalAI,
    ];
    for backend in &local_backends {
        assert!(
            PrivacyTier::Sovereign.allows(*backend),
            "Sovereign must allow {:?} but allows() returned false",
            backend
        );
    }
}

/// FALSIFY-PRIV-003: Standard allows ALL backends (completeness).
/// Prediction: PrivacyTier::Standard.allows(b) = true for every variant.
#[test]
#[allow(non_snake_case)]
fn falsify_PRIV_003_standard_allows_all() {
    let all_backends = [
        ServingBackend::Realizar,
        ServingBackend::Ollama,
        ServingBackend::LlamaCpp,
        ServingBackend::Llamafile,
        ServingBackend::Candle,
        ServingBackend::Vllm,
        ServingBackend::Tgi,
        ServingBackend::LocalAI,
        ServingBackend::HuggingFace,
        ServingBackend::Together,
        ServingBackend::Replicate,
        ServingBackend::Anyscale,
        ServingBackend::Modal,
        ServingBackend::Fireworks,
        ServingBackend::Groq,
        ServingBackend::OpenAI,
        ServingBackend::Anthropic,
        ServingBackend::AzureOpenAI,
        ServingBackend::AwsBedrock,
        ServingBackend::GoogleVertex,
        ServingBackend::AwsLambda,
        ServingBackend::CloudflareWorkers,
    ];
    for backend in &all_backends {
        assert!(
            PrivacyTier::Standard.allows(*backend),
            "Standard must allow {:?} but allows() returned false",
            backend
        );
    }
}

/// FALSIFY-PRIV-004: BackendSelector with Sovereign recommends only locals.
/// Prediction: recommend() returns zero remote backends under Sovereign tier.
#[test]
#[allow(non_snake_case)]
fn falsify_PRIV_004_sovereign_recommend_all_local() {
    let selector = BackendSelector::new().with_privacy(PrivacyTier::Sovereign);
    let recommended = selector.recommend();
    for backend in &recommended {
        assert!(backend.is_local(), "Sovereign recommend() returned remote {:?}", backend);
    }
}

// ============================================================================
// F-BANCO-002: Budget Conservation (budget-conservation-v1.yaml)
// ============================================================================

/// FALSIFY-BUDGET-001: Budget exceeded → check returns Err.
/// Prediction: After recording costs totaling > budget, check() fails.
#[test]
#[allow(non_snake_case)]
fn falsify_BUDGET_001_over_budget_blocks() {
    let cb = CostCircuitBreaker::new(CircuitBreakerConfig::with_budget(1.0));
    cb.record(0.9);
    let result = cb.check(0.2);
    assert!(result.is_err(), "check(0.2) should fail after 0.9 recorded against $1.0 budget");
}

/// FALSIFY-BUDGET-002: Accumulated monotonically increases.
/// Prediction: Each record() call increases or maintains accumulated_usd().
#[test]
#[allow(non_snake_case)]
fn falsify_BUDGET_002_monotonic_accumulation() {
    let cb = CostCircuitBreaker::new(CircuitBreakerConfig::with_budget(100.0));
    let mut prev = cb.accumulated_usd();
    for i in 1..=20 {
        let cost = (i as f64) * 0.1;
        cb.record(cost);
        let current = cb.accumulated_usd();
        assert!(
            current >= prev,
            "accumulated went from {prev} to {current} after recording {cost}"
        );
        prev = current;
    }
}

/// FALSIFY-BUDGET-003: Per-request limit checked before budget.
/// Prediction: Expensive request rejected even with ample budget.
#[test]
#[allow(non_snake_case)]
fn falsify_BUDGET_003_per_request_limit_first() {
    let config = CircuitBreakerConfig {
        daily_budget_usd: 100.0,
        max_request_cost_usd: 1.0,
        ..Default::default()
    };
    let cb = CostCircuitBreaker::new(config);
    // Budget has $100, but per-request max is $1
    let result = cb.check(5.0);
    assert!(
        matches!(
            result,
            Err(crate::serve::circuit_breaker::CircuitBreakerError::RequestTooExpensive { .. })
        ),
        "Expected RequestTooExpensive, got {result:?}"
    );
}

/// FALSIFY-BUDGET-004: Open state blocks during cooldown.
/// Prediction: Circuit open + cooldown not elapsed → check returns Err(BudgetExceeded).
#[test]
#[allow(non_snake_case)]
fn falsify_BUDGET_004_open_blocks_during_cooldown() {
    let config = CircuitBreakerConfig {
        daily_budget_usd: 1.0,
        max_request_cost_usd: 5.0,
        cooldown_seconds: 3600,
        ..Default::default()
    };
    let cb = CostCircuitBreaker::new(config);
    cb.record(1.0); // Exhaust budget → Open
    assert_eq!(cb.state(), CircuitState::Open);
    let result = cb.check(0.01);
    assert!(result.is_err(), "Open circuit with active cooldown must block");
}

// ============================================================================
// F-BANCO-003: Routing Determinism (routing-determinism-v1.yaml)
// ============================================================================

/// FALSIFY-ROUTE-001: Empty queue routes local.
/// Prediction: route() = Local(Realizar) when no requests enqueued.
#[test]
#[allow(non_snake_case)]
fn falsify_ROUTE_001_empty_queue_local() {
    let router = SpilloverRouter::with_defaults();
    let decision = router.route();
    assert!(
        matches!(decision, RoutingDecision::Local(ServingBackend::Realizar)),
        "Empty queue should route Local(Realizar), got {decision:?}"
    );
}

/// FALSIFY-ROUTE-002: Same state → same decision (determinism).
/// Prediction: Two consecutive route() calls return identical result.
#[test]
#[allow(non_snake_case)]
fn falsify_ROUTE_002_deterministic() {
    let router = SpilloverRouter::with_defaults();
    // Add some queue depth
    router.start_request(ServingBackend::Realizar);
    router.start_request(ServingBackend::Realizar);
    let d1 = router.route();
    let d2 = router.route();
    assert_eq!(d1, d2, "route() must be deterministic: {d1:?} != {d2:?}");
}

/// FALSIFY-ROUTE-003: Sovereign spillover stays local.
/// Prediction: Under Sovereign config, spillover targets are all local.
#[test]
#[allow(non_snake_case)]
fn falsify_ROUTE_003_sovereign_spillover_local() {
    let router = SpilloverRouter::new(RouterConfig::sovereign());
    // Fill past threshold to trigger spillover
    for _ in 0..15 {
        router.start_request(ServingBackend::Realizar);
    }
    let decision = router.route();
    match decision {
        RoutingDecision::Spillover(b) => {
            assert!(b.is_local(), "Sovereign spillover to remote {:?}", b);
        }
        RoutingDecision::Local(_) | RoutingDecision::Reject(_) => {} // acceptable
    }
}

/// FALSIFY-ROUTE-004: Full queue + no spillover → Reject.
/// Prediction: Queue at max_depth with spillover disabled → Reject(QueueFull).
#[test]
#[allow(non_snake_case)]
fn falsify_ROUTE_004_full_queue_rejects() {
    let router = SpilloverRouter::new(RouterConfig {
        spillover_threshold: 2,
        max_queue_depth: 3,
        spillover_enabled: false,
        ..Default::default()
    });
    for _ in 0..3 {
        router.start_request(ServingBackend::Realizar);
    }
    let decision = router.route();
    assert!(
        matches!(decision, RoutingDecision::Reject(_)),
        "Full queue without spillover should reject, got {decision:?}"
    );
}

// ============================================================================
// F-BANCO-004: Context Enforcement (context-enforcement-v1.yaml)
// ============================================================================

/// FALSIFY-CTX-001: Truncated messages always fit.
/// Prediction: After SlidingWindow truncation, fits() returns true.
#[test]
#[allow(non_snake_case)]
fn falsify_CTX_001_truncated_fits() {
    let mgr = ContextManager::new(ContextConfig {
        window: ContextWindow::new(100, 20), // 80 available tokens
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: true,
        min_messages: 1,
    });
    // Create messages that exceed the window
    let msgs: Vec<ChatMessage> = (0..50)
        .map(|i| ChatMessage::user(format!("Message number {i} with some content padding")))
        .collect();
    assert!(!mgr.fits(&msgs), "Messages should exceed window");
    let truncated = mgr.truncate(&msgs).expect("SlidingWindow should not error");
    assert!(
        mgr.fits(&truncated),
        "Truncated messages must fit: {} tokens > {} available",
        mgr.estimate_tokens(&truncated),
        mgr.available_tokens()
    );
}

/// FALSIFY-CTX-002: Error strategy refuses truncation.
/// Prediction: Strategy=Error on oversized input → Err(ExceedsLimit).
#[test]
#[allow(non_snake_case)]
fn falsify_CTX_002_error_strategy_rejects() {
    let mgr = ContextManager::new(ContextConfig {
        window: ContextWindow::new(50, 10),
        strategy: TruncationStrategy::Error,
        preserve_system: true,
        min_messages: 1,
    });
    let msgs: Vec<ChatMessage> =
        (0..20).map(|i| ChatMessage::user(format!("Msg {i} padded content here"))).collect();
    let result = mgr.truncate(&msgs);
    assert!(result.is_err(), "Error strategy must reject oversized input");
}

/// FALSIFY-CTX-003: Token count monotonicity.
/// Prediction: Adding a message never decreases token estimate.
#[test]
#[allow(non_snake_case)]
fn falsify_CTX_003_token_monotonicity() {
    let mgr = ContextManager::default();
    let mut msgs = vec![ChatMessage::user("Hello")];
    let mut prev = mgr.estimate_tokens(&msgs);
    for i in 0..20 {
        msgs.push(ChatMessage::user(format!("Additional message {i}")));
        let current = mgr.estimate_tokens(&msgs);
        assert!(current >= prev, "Token count decreased: {prev} → {current}");
        prev = current;
    }
}

/// FALSIFY-CTX-004: Fitting messages pass through unchanged.
/// Prediction: Small messages that fit are returned unchanged by truncate.
#[test]
#[allow(non_snake_case)]
fn falsify_CTX_004_passthrough_when_fits() {
    let mgr = ContextManager::default(); // 4096 token window
    let msgs = vec![ChatMessage::user("Short message")];
    assert!(mgr.fits(&msgs));
    let result = mgr.truncate(&msgs).expect("should not error");
    assert_eq!(result.len(), msgs.len());
    assert_eq!(result[0].content, msgs[0].content);
}

// ============================================================================
// F-BANCO-005: Template Correctness (template-correctness-v1.yaml)
// ============================================================================

/// FALSIFY-TPL-001: Content preserved across all formats.
/// Prediction: Every message content appears as substring in formatted output.
#[test]
#[allow(non_snake_case)]
fn falsify_TPL_001_content_preserved() {
    let formats = [
        TemplateFormat::Llama2,
        TemplateFormat::Mistral,
        TemplateFormat::ChatML,
        TemplateFormat::Alpaca,
        TemplateFormat::Vicuna,
        TemplateFormat::Raw,
    ];
    let msgs = vec![
        ChatMessage::system("System prompt here"),
        ChatMessage::user("User query here"),
        ChatMessage::assistant("Assistant response here"),
    ];
    for fmt in &formats {
        let engine = ChatTemplateEngine::new(*fmt);
        let output = engine.apply(&msgs);
        for msg in &msgs {
            assert!(
                output.contains(&msg.content),
                "Format {:?} lost content {:?} in output {:?}",
                fmt,
                msg.content,
                output
            );
        }
    }
}

/// FALSIFY-TPL-002: Raw format is identity for single message.
/// Prediction: apply([user("hello")]) = "hello" for Raw format.
#[test]
#[allow(non_snake_case)]
fn falsify_TPL_002_raw_identity() {
    let engine = ChatTemplateEngine::new(TemplateFormat::Raw);
    let output = engine.apply(&[ChatMessage::user("hello")]);
    assert_eq!(output, "hello", "Raw format must be identity");
}

/// FALSIFY-TPL-003: All formats produce non-empty output.
/// Prediction: No format returns empty string for non-empty input.
#[test]
#[allow(non_snake_case)]
fn falsify_TPL_003_nonempty_output() {
    let formats = [
        TemplateFormat::Llama2,
        TemplateFormat::Mistral,
        TemplateFormat::ChatML,
        TemplateFormat::Alpaca,
        TemplateFormat::Vicuna,
        TemplateFormat::Raw,
    ];
    let msgs = vec![ChatMessage::user("x")];
    for fmt in &formats {
        let engine = ChatTemplateEngine::new(*fmt);
        let output = engine.apply(&msgs);
        assert!(!output.is_empty(), "Format {:?} produced empty output", fmt);
    }
}

/// FALSIFY-TPL-004: Unknown model name defaults to Raw.
/// Prediction: from_model_name("totally-fake-model") = Raw.
#[test]
#[allow(non_snake_case)]
fn falsify_TPL_004_unknown_defaults_raw() {
    assert_eq!(TemplateFormat::from_model_name("totally-fake-model-v99"), TemplateFormat::Raw);
    assert_eq!(TemplateFormat::from_model_name(""), TemplateFormat::Raw);
    assert_eq!(TemplateFormat::from_model_name("my-custom-7b"), TemplateFormat::Raw);
}
