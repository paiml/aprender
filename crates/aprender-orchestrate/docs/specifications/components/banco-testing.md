# Banco Testing Strategy: Full Probar Integration

> Parent: [banco-spec.md](banco-spec.md) §5
> Depends on: jugar-probar (1.0.x) with features: browser, tui, runtime
> Status: **L1 complete (353 tests). L2-L4 NOT YET IMPLEMENTED.**

---

## Why This Matters

UAT revealed "nothing works" in the browser — because 353 L1 unit tests all use `tower::oneshot()` (in-process, no TCP). They never validated that a real browser can connect to a real server and receive a real response. Probar L4 tests would have caught this immediately.

---

## Zero-JavaScript Testing Constraint

```
❌ FORBIDDEN in tests                     ✅ REQUIRED
────────────────────────────────────────────────────────────
• Playwright (Node.js)                    • jugar-probar (pure Rust CDP)
• Puppeteer                               • probar Browser::launch() + Page
• Selenium WebDriver                      • probar Locator + expect() API
• Cypress                                 • probar SoftAssertions
• Any npm test runner                     • cargo test --features browser
• Jest / Vitest / Mocha                   • probar TestSuite + TestHarness
```

Probar provides **every Playwright capability** without JavaScript:
- `Browser::launch()` → headless Chromium via Chrome DevTools Protocol
- `page.goto(url)` → navigation with auto-wait
- `page.locator(Selector::css("#input"))` → element selection with 5s auto-retry
- `expect(&locator).to_have_text("...")` → fluent assertions
- `page.screenshot()` → PNG capture for visual regression
- `page.evaluate(expr)` → JS execution in browser context (for testing only)
- `AccessibilityValidator::audit(&page)` → WCAG AA compliance
- `PixelCoverageTracker` → interaction coverage heatmaps
- `InputFuzzer` → Monte Carlo random input generation
- `SimulationConfig` → deterministic seed-based replay

This aligns with the broader Sovereign AI Stack policy: probar's CLAUDE.md states "JavaScript introduces non-determinism and GC pauses. Probar compiles to a single `.wasm` file with ZERO JS."

---

## Test Pyramid

```
┌──────────────────────────────────────────────────────────────┐
│                     Banco Test Pyramid                        │
├──────────────────────────────────────────────────────────────┤
│  L4: E2E Browser      probar Browser + CDP + Locators        │
│  L3: E2E TUI          probar MockTty + FrameAssertion        │
│  L2: API Integration   Real TCP server + HTTP client          │
│  L1: Unit              tower::oneshot (353 tests, COMPLETE)   │
├──────────────────────────────────────────────────────────────┤
│  Cross-cutting: load, fuzz, a11y, visual regression,          │
│                 pixel coverage, deterministic replay,          │
│                 network interception, WebSocket monitoring     │
└──────────────────────────────────────────────────────────────┘
```

---

## L1: Unit Tests (COMPLETE — 353 tests)

Uses `tower::ServiceExt::oneshot()` — in-process, no TCP, no probar needed.

| Suite | Count | What |
|-------|-------|------|
| BANCO_TYP | 12 | Type serde roundtrip |
| BANCO_STA | 8 | State init, health, models |
| BANCO_MID | 5 | Privacy/CORS middleware |
| BANCO_HDL | 7 | Core handler routing |
| P0/P1/P2 | 42 | Cross-cutting: OpenAI compat, probes, scoped keys, Ollama pull/delete |
| CONV | 15 | Conversation CRUD + export/import + disk persistence |
| INF | 24 | Inference engine, BPE decode, streaming |
| STOR | 17 | File storage, info endpoint, disk roundtrip |
| RECIPE | 14 | Recipe pipeline (7 step types), CSV/JSONL parsing |
| RAG | 14 | BM25 index, search, chat integration, trueno-rag backend |
| EVAL/TRAIN | 32 | Eval perplexity, training presets, SSE metrics, export |
| EXP | 9 | Experiment tracking, comparison, disk persistence |
| BATCH | 5 | Batch inference |
| MERGE | 12 | Model merge (TIES/DARE/SLERP/weighted) |
| TOOLS | 17 | Calculator, code execution, self-healing retry |
| MCP | 12 | JSON-RPC 2.0 initialize, tools/list, tools/call |
| AUDIO | 7 | Transcription, base64 decode, format listing |
| CMPL | 9 | Text completions, model detail, prompt array |
| EVENTS | 8 | EventBus emit/subscribe, WebSocket endpoint |
| UI | 6 | HTML content, chat API reference, endpoint serving |
| METRICS | 5 | Prometheus counters, gauges, exposition format |
| REGISTRY | 6 | Pacha pull/list/cache |
| CONTRACT | 20 | Falsification tests |
| MODEL_SLOT | 8 | Model load/unload, metadata extraction |

**Limitation:** L1 tests go through the full middleware stack (audit, auth, privacy, CORS) but never bind a TCP port. They cannot detect:
- Server startup/binding failures
- Browser JS fetch/SSE bugs
- WebSocket connection lifecycle
- BPE token decode issues visible only in browser rendering
- CORS preflight failures from real browsers

---

## L2: API Integration Tests (probar::llm — MANDATORY)

### What It Tests

Start a real Banco HTTP server on a random port. Use probar's `LlmClient` (purpose-built for OpenAI-compatible endpoints) to send real HTTP requests over TCP. Validate responses with `LlmAssertion` builders.

**probar::llm is not optional.** It provides typed chat request/response, SSE streaming with per-token timestamps, latency assertions, determinism checks, load testing, and scorecards — all in pure Rust. We wrote 353 unit tests that bypass TCP. L2 uses `LlmClient` to test the real HTTP path.

### Server Harness

```rust
// tests/banco_api.rs
use batuta::serve::banco::{router::create_banco_router, BancoStateInner};

async fn start_server() -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let state = BancoStateInner::with_defaults();
    let app = create_banco_router(state);
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (format!("http://127.0.0.1:{port}"), handle)
}
```

### Test Cases

| Test | Endpoint | Validates |
|------|----------|-----------|
| `e2e_health` | GET /health | 200, JSON with status=ok |
| `e2e_system_info` | GET /api/v1/system | endpoints count, version, privacy_tier |
| `e2e_chat_no_model` | POST /api/v1/chat/completions | 200, helpful "No model loaded" message |
| `e2e_chat_streaming` | POST /api/v1/chat/completions (stream=true) | SSE format, data: chunks, [DONE] terminator |
| `e2e_upload_rag_search` | POST upload → GET rag/search | File indexed, BM25 results returned |
| `e2e_tool_calculator` | POST /api/v1/tools/execute | calculator returns correct result |
| `e2e_mcp_initialize` | POST /api/v1/mcp | JSON-RPC 2.0, serverInfo, capabilities |
| `e2e_mcp_tool_call` | POST /api/v1/mcp (tools/call) | MCP content format, tool execution |
| `e2e_text_completions` | POST /v1/completions | OpenAI text_completion format |
| `e2e_model_detail` | GET /v1/models/:id | 200 for backends, 404 for unknown |
| `e2e_ollama_tags` | GET /api/tags | Ollama models array |
| `e2e_ollama_pull` | POST /api/pull | Ollama pull response with digest |
| `e2e_metrics_prometheus` | GET /api/v1/metrics | text/plain, Prometheus counters |
| `e2e_health_probes` | GET /health/live, /health/ready | 200 live, 503 ready (no model) |
| `e2e_conversations` | POST/GET/DELETE /api/v1/conversations | Full CRUD lifecycle |
| `e2e_cors_headers` | GET /health with Origin header | Access-Control-Allow-Origin present |
| `e2e_browser_ui` | GET / | 200, text/html, contains Banco + script |
| `e2e_websocket_connect` | GET /api/v1/ws (upgrade) | WebSocket handshake succeeds |
| `e2e_training_preset` | POST /api/v1/train/start | Preset expands, metrics returned |
| `e2e_merge_slerp` | POST /api/v1/models/merge | Strategy accepted, result returned |

### probar LlmClient Tests (PRIMARY L2 approach)

```rust
use jugar_probar::llm::{LlmClient, ChatRequest, ChatMessage, LlmAssertion};
use std::time::Duration;

#[tokio::test]
async fn test_chat_response_valid() {
    let (base, handle) = start_server().await;
    let client = LlmClient::new(&format!("{base}/v1"), "local");

    // Wait for server to be ready (polls /health)
    client.wait_ready(Duration::from_secs(5)).await.unwrap();

    // Send chat request, get timed response
    let request = ChatRequest {
        model: "local".into(),
        messages: vec![ChatMessage { role: Role::User, content: "Hello!".into() }],
        ..Default::default()
    };
    let timed = client.send(&request).await.unwrap();

    // Structural assertions
    let assertions = LlmAssertion::new()
        .assert_response_valid()
        .assert_latency_under(Duration::from_secs(30))
        .assert_token_count(Some(1), None); // at least 1 token
    assert!(assertions.run_all_pass(&timed));

    handle.abort();
}

#[tokio::test]
async fn test_chat_streaming_sse() {
    let (base, handle) = start_server().await;
    let client = LlmClient::new(&format!("{base}/v1"), "local");
    client.wait_ready(Duration::from_secs(5)).await.unwrap();

    let request = ChatRequest {
        model: "local".into(),
        messages: vec![ChatMessage { role: Role::User, content: "Count to 3".into() }],
        ..Default::default()
    };

    // Stream with per-token timestamps + TTFT measurement
    let streamed = client.chat_completion_stream(&request).await.unwrap();
    assert!(!streamed.content.is_empty(), "stream produced no content");
    assert!(streamed.ttft < Duration::from_secs(10), "TTFT too slow");
    assert!(!streamed.token_timestamps.is_empty(), "no token timestamps");

    handle.abort();
}

#[tokio::test]
async fn test_deterministic_output() {
    let (base, handle) = start_server().await;
    let client = LlmClient::new(&format!("{base}/v1"), "local");
    client.wait_ready(Duration::from_secs(5)).await.unwrap();

    let request = ChatRequest {
        model: "local".into(),
        messages: vec![ChatMessage { role: Role::User, content: "What is 1+1?".into() }],
        temperature: Some(0.0), // deterministic
        ..Default::default()
    };

    let r1 = client.send(&request).await.unwrap();
    let r2 = client.send(&request).await.unwrap();
    let result = jugar_probar::llm::assert_deterministic(&[r1.response, r2.response]);
    assert!(result.passed, "Non-deterministic output at temperature=0");

    handle.abort();
}
```

### probar LoadTest (Performance Regression)

```rust
use jugar_probar::llm::loadtest::{LoadTest, LoadTestConfig, RequestRate, ValidationMode};
use jugar_probar::llm::score::{compute_scorecard, format_markdown};

#[tokio::test]
#[ignore] // weekly CI
async fn test_banco_load() {
    let (base, handle) = start_server().await;

    let config = LoadTestConfig {
        concurrency: 10,
        duration: Duration::from_secs(30),
        warmup_duration: Duration::from_secs(5),
        prompts: vec![
            ChatRequest { model: "local".into(),
                messages: vec![ChatMessage { role: Role::User, content: "Hello!".into() }],
                ..Default::default() },
        ],
        rate: RequestRate::Poisson(5.0), // 5 req/s Poisson arrival
        stream: false,
        validate: ValidationMode::Basic, // inline correctness checks
        slo_latency_ms: Some(5000.0),    // 5s SLO
        slo_ttft_ms: Some(1000.0),       // 1s TTFT SLO
        ..Default::default()
    };

    let result = LoadTest::new(
        &format!("{base}/v1"), "local", config
    ).run().await.unwrap();

    assert_eq!(result.failed, 0, "requests failed under load");
    assert!(result.latency_p95_ms < 5000.0, "p95 > 5s SLO");
    assert!(result.throughput_rps > 1.0, "throughput < 1 rps");

    // Generate scorecard
    let card = compute_scorecard(&result);
    eprintln!("{}", format_markdown(&card));

    handle.abort();
}
```

### With probar WebSocket monitoring

```rust
use jugar_probar::websocket::WebSocketConnection;

#[tokio::test]
async fn test_websocket_events() {
    let (base, handle) = start_server().await;
    let ws_url = base.replace("http://", "ws://") + "/api/v1/ws";
    let conn = WebSocketConnection::connect(&ws_url).await.unwrap();

    // Should receive connected event
    let msg = conn.receive_timeout(Duration::from_secs(5)).await.unwrap();
    let data: serde_json::Value = msg.json().unwrap();
    assert_eq!(data["type"], "connected");

    handle.abort();
}
```

---

## L3: TUI Integration Tests (probar tui module)

### What It Tests

Terminal UI rendering using probar's `MockTty` + `FrameAssertion`. Validates text content, layout, and performance without a real terminal.

### Frame Assertions

```rust
use jugar_probar::tui::{MockTty, FrameAssertion};

#[test]
fn test_banco_banner_renders() {
    let mut tty = MockTty::new(120, 40);
    // Render banco startup banner
    render_banner(&mut tty, &test_state());

    let frame = tty.capture_frame();
    FrameAssertion::new(&frame)
        .to_have_text("Banco")?
        .to_have_text("Listening")?
        .to_have_text("/health")?;
}
```

### State Machine Playbook

```yaml
# tests/playbooks/banco-chat.yaml
version: "1.0"
machine:
  id: banco-chat
  initial: idle
  states:
    idle:
      invariants:
        - element_exists: "#input"
        - element_exists: "#send"
    waiting:
      invariants:
        - element_has_attribute: { selector: "#send", attr: "disabled" }
    response:
      invariants:
        - element_exists: ".msg.assistant"
  transitions:
    - { from: idle, to: waiting, event: click_send }
    - { from: waiting, to: response, event: fetch_complete }
    - { from: response, to: idle, event: input_focus }
```

### TUI Performance Budget

```rust
use jugar_probar::tui_load::{TuiLoadTest, TuiFrameMetrics};

#[test]
fn test_banco_tui_60fps() {
    let load = TuiLoadTest {
        item_count: 1000,
        frame_budget_ms: 16.6,
        timeout_ms: 5000,
    };
    let metrics = load.run(|tty| render_chat(tty, 1000)).unwrap();
    assert!(metrics.p95_frame_ms() < 16.6);
}
```

---

## L4: E2E Browser Tests (probar browser + CDP)

### What It Tests

Launch headless Chromium, navigate to `http://localhost:PORT/`, interact with the embedded chat UI, verify responses appear. This is the test that would have caught "nothing works."

### Browser Harness

```rust
use jugar_probar::browser::{Browser, BrowserConfig, Page};
use jugar_probar::locator::Selector;

async fn banco_browser(port: u16) -> (Browser, Page) {
    let browser = Browser::launch(BrowserConfig {
        headless: true,
        sandbox: false,
        ..Default::default()
    }).await.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(&format!("http://127.0.0.1:{port}/")).await.unwrap();
    (browser, page)
}
```

### Core Browser Tests

| Test | What | How |
|------|------|-----|
| `browser_ui_loads` | Page loads, title is "Banco" | `page.goto()`, check `<title>` |
| `browser_chat_input_exists` | Chat input field visible | `page.locator("#input").to_be_visible()` |
| `browser_send_button_exists` | Send button visible | `page.locator("#send").to_be_visible()` |
| `browser_system_info_displayed` | System info fetched on load | `page.locator(".msg.system")` contains version |
| `browser_model_status` | Model indicator shows loaded/not | `.model-info .dot` has class `on` or `off` |
| `browser_send_message` | Type + click Send → response appears | `fill("#input")`, `click("#send")`, wait for `.msg.assistant` |
| `browser_response_readable` | Response is decoded text, not BPE | `.msg.assistant` text does NOT contain `Ġ` or `Ċ` |
| `browser_settings_work` | Temperature slider changes value | `page.evaluate("document.getElementById('temp').value")` |
| `browser_rag_toggle` | RAG button toggles active class | `click("#rag-toggle")`, check `.active` class |
| `browser_file_upload` | Upload button triggers file dialog | `click(".upload-btn")`, verify dialog opens |
| `browser_websocket_connects` | WS status shows "connected" | `page.locator("#ws-status")` has class `connected` |
| `browser_conversations_load` | Sidebar populates with convos | `.convos .conv` count >= 0 |

### Example: The Test That Would Have Caught The Bug

```rust
#[tokio::test]
async fn browser_send_message_gets_response() {
    let (base, server) = start_server().await;
    let port = extract_port(&base);
    let (browser, mut page) = banco_browser(port).await;

    // Type a message
    page.locator(Selector::css("#input")).fill("Hello Banco!").await.unwrap();

    // Click send
    page.locator(Selector::css("#send")).click().await.unwrap();

    // Wait for assistant response to appear (up to 30s for model inference)
    let response = page.locator(Selector::css(".msg.assistant"))
        .with_timeout(Duration::from_secs(30));

    // THIS IS THE ASSERTION THAT WOULD HAVE CAUGHT "nothing works"
    expect(&response).to_be_visible().await.unwrap();

    // Verify response is readable text (not BPE garbage)
    let text = response.text_content().await.unwrap();
    assert!(!text.contains("Ġ"), "BPE tokens not decoded: {text}");
    assert!(!text.contains("Ċ"), "BPE tokens not decoded: {text}");
    assert!(!text.is_empty(), "Response is empty");

    browser.close().await.unwrap();
    server.abort();
}
```

### Visual Regression

```rust
use jugar_probar::visual_regression::{VisualRegressionTester, VisualRegressionConfig};

#[tokio::test]
async fn browser_chat_screen_visual() {
    let (browser, page) = banco_browser(port).await;
    let screenshot = page.screenshot().await.unwrap();

    let tester = VisualRegressionTester::new(VisualRegressionConfig {
        threshold: 1.0,
        baseline_dir: "tests/baselines/".into(),
        diff_dir: "tests/diffs/".into(),
        ..Default::default()
    });

    let result = tester.compare("chat-screen", &screenshot).unwrap();
    assert!(result.matches, "Visual diff: {:.2}%", result.diff_percentage);
}
```

### Accessibility Audit

```rust
use jugar_probar::accessibility::{AccessibilityValidator, AccessibilityConfig};

#[tokio::test]
async fn browser_wcag_aa() {
    let (browser, page) = banco_browser(port).await;
    let validator = AccessibilityValidator::new(AccessibilityConfig {
        check_contrast: true,
        check_focus: true,
        check_keyboard: true,
        min_contrast_text: 4.5,
        min_contrast_ui: 3.0,
        ..Default::default()
    });

    let report = validator.audit(&page).await.unwrap();
    assert!(report.passes_wcag_aa, "Violations: {:?}", report.violations);
}
```

---

## Cross-Cutting: Network Interception

Verify Sovereign mode makes zero external network calls:

```rust
use jugar_probar::network::{NetworkInterceptor, RequestPattern};

#[tokio::test]
async fn sovereign_no_external_calls() {
    let state = BancoStateInner::with_privacy(PrivacyTier::Sovereign);
    let (base, server) = start_server_with_state(state).await;
    let interceptor = NetworkInterceptor::new();

    // Make several requests
    post(&format!("{base}/api/v1/chat/completions"), chat_body()).await;
    post(&format!("{base}/api/v1/tools/execute"), calc_body()).await;

    // Assert zero external requests
    let external = interceptor.requests()
        .filter(|r| !r.url.starts_with("http://127.0.0.1"))
        .count();
    assert_eq!(external, 0, "Sovereign mode leaked external requests");

    server.abort();
}
```

---

## Cross-Cutting: Load Testing

```rust
use jugar_probar::llm::loadtest::{LoadTest, LoadTestConfig};

#[tokio::test]
#[ignore] // Weekly CI only
async fn banco_load_test() {
    let (base, server) = start_server().await;
    let config = LoadTestConfig {
        base_url: base,
        concurrent_requests: 10,
        request_rate: 5.0,
        duration: Duration::from_secs(30),
        dataset: vec![
            ChatRequest { messages: vec![ChatMessage::user("Hello!")], ..Default::default() },
            ChatRequest { messages: vec![ChatMessage::user("2+2?")], ..Default::default() },
        ],
    };

    let result = LoadTest::with_config(config).run().await.unwrap();
    assert_eq!(result.failed, 0);
    assert!(result.latency_stats.p95 < 5000.0, "p95 > 5s");
    assert!(result.throughput_rps > 1.0);

    server.abort();
}
```

---

## Cross-Cutting: Monte Carlo Fuzzing

```rust
use jugar_probar::fuzzer::{InputFuzzer, FuzzerConfig, Seed};

#[test]
#[ignore] // Weekly CI only
fn banco_fuzz_chat_input() {
    let mut fuzzer = InputFuzzer::with_config(Seed(42), FuzzerConfig {
        viewport_width: 1280.0,
        viewport_height: 720.0,
        key_probability: 0.8,
        ..Default::default()
    });

    for _ in 0..10_000 {
        let inputs = fuzzer.generate_valid_inputs();
        for input in inputs {
            // POST to chat — should never panic or 500
            let body = serde_json::json!({
                "messages": [{"role": "user", "content": input.as_text()}],
                "max_tokens": 8
            });
            // Verify no panic, no 500
        }
    }
}
```

---

## Cross-Cutting: Pixel Coverage

```rust
use jugar_probar::pixel_coverage::{PixelCoverageTracker, PngHeatmap};

#[tokio::test]
async fn banco_ui_coverage_80pct() {
    let mut tracker = PixelCoverageTracker::new((1280, 720), (64, 36));
    let (browser, mut page) = banco_browser(port).await;

    // Visit all interactive regions
    // Chat input, send button, sidebar, settings, RAG toggle, upload
    for selector in ["#input", "#send", "#rag-toggle", ".upload-btn", "#temp", "#maxtok"] {
        if let Ok(loc) = page.locator(Selector::css(selector)).await {
            if let Ok(bounds) = loc.bounding_box().await {
                tracker.record_region(bounds);
            }
        }
    }

    let report = tracker.generate_report();
    assert!(report.overall_coverage >= 0.80, "UI coverage: {:.1}%", report.overall_coverage * 100.0);

    PngHeatmap::render(&tracker, "tests/output/banco_coverage.png").unwrap();
}
```

---

## Cross-Cutting: Deterministic Replay

```rust
use jugar_probar::simulation::{SimulationConfig, Seed, run_simulation};

#[test]
fn banco_deterministic_replay() {
    let config = SimulationConfig::new()
        .with_seed(Seed::Fixed(42))
        .with_max_steps(300)
        .with_headless(true);

    let result1 = run_simulation(initial_state(), config.clone(), update_fn).unwrap();
    let result2 = run_simulation(initial_state(), config, update_fn).unwrap();

    assert_eq!(result1.final_state_hash, result2.final_state_hash,
        "Non-determinism detected: different state after same inputs");
}
```

---

## Implementation Status

| Level | Tests | Status | Blocking Issues Found |
|-------|-------|--------|----------------------|
| **L1 Unit** | 353 | **COMPLETE** | None (but missed BPE decode bug, browser JS bug) |
| **L2 API** | 0 | **NOT STARTED** | Need `pub mod router` (done), HTTP client dep |
| **L3 TUI** | 0 | **NOT STARTED** | No TUI app exists yet |
| **L4 Browser** | 0 | **NOT STARTED** | Would have caught "nothing works" |
| Load | 0 | **NOT STARTED** | — |
| Fuzz | 0 | **NOT STARTED** | — |
| Visual | 0 | **NOT STARTED** | — |
| a11y | 0 | **NOT STARTED** | — |

---

## CI Pipeline

```bash
# L1: Every commit (<30s)
cargo test --features banco,inference --lib banco

# L2: Pre-merge (<2min)
cargo test --features banco,inference --test banco_api

# L4: Nightly (<10min, requires Chrome)
cargo test --features banco,inference,agents-browser --test banco_e2e

# Load: Weekly (<5min)
cargo test --features banco,inference --test banco_load -- --ignored

# Fuzz: Weekly (<10min)
cargo test --features banco,inference --test banco_fuzz -- --ignored
```

---

## Cargo.toml Dependencies

```toml
[dev-dependencies]
jugar-probar = { version = "1.0", features = ["browser", "llm"] }
# browser: CDP headless Chrome (L4)
# llm: LlmClient, LlmAssertion, LoadTest, Scorecard (L2)
# Both are pure Rust, zero JavaScript.
```

No probar in production builds. All test code in `tests/` directory or behind `#[cfg(test)]`.

### Why jugar-probar, not raw HTTP clients

| Approach | LOC for "chat works" test | What it catches |
|----------|--------------------------|-----------------|
| Raw hyper client | ~40 lines (boilerplate) | HTTP status code only |
| `LlmClient::send()` + `LlmAssertion` | ~8 lines | Structure, content, latency, token count |
| `LlmClient::chat_completion_stream()` | ~5 lines | SSE parsing, TTFT, per-token timing |
| `LoadTest::run()` | ~15 lines | p95 latency, throughput, correctness under concurrency |

probar::llm was **purpose-built for testing OpenAI-compatible inference servers**. It speaks the exact protocol Banco serves.

---

## Lessons Learned

1. **353 unit tests are necessary but not sufficient.** They test the Rust code path but not the browser JS, TCP binding, SSE stream parsing, or BPE rendering.
2. **L4 browser tests are the UAT safety net.** The test `browser_send_message_gets_response` would have caught every issue the user reported.
3. **Probar's auto-waiting locators prevent flaky tests.** Instead of `sleep(5)`, use `locator.with_timeout(30s)` — probar polls until the element appears or the timeout expires.
4. **The BPE decode bug was invisible to unit tests** because they compare Rust strings, not what the browser renders. A browser test checking `.msg.assistant` text content would have shown garbled output immediately.
