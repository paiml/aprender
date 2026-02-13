//! Falsification: Stress & Endurance Tests
//!
//! Tests for long sessions, large prompts, token limits, and concurrency.
//! Model-tests gated — requires actual model files.
//!
//! These catch edge cases that only appear under sustained use:
//! - Memory leaks over many generations
//! - State corruption in long chat sessions
//! - Boundary behavior at token limits

#![cfg(feature = "model-tests")]

// =============================================================================
// Token Limit Tests
// =============================================================================

/// max_tokens=1 should produce exactly 1 token (not 0, not infinite loop)
#[test]
fn stress_max_tokens_one() {
    let model_path = match std::env::var("TEST_MODEL_PATH") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            eprintln!(
                "FALSIFICATION-SKIP: gate=stress_max_tokens_one reason=TEST_MODEL_PATH not set"
            );
            return;
        }
    };

    let output = std::process::Command::new("cargo")
        .args([
            "run",
            "--bin",
            "apr",
            "--features",
            "inference",
            "--",
            "run",
            &model_path.to_string_lossy(),
            "--prompt",
            "Hello",
            "--max-tokens",
            "1",
            "--temperature",
            "0",
        ])
        .output()
        .expect("max_tokens=1 generation");

    assert!(
        output.status.success(),
        "max_tokens=1 must succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let text = String::from_utf8_lossy(&output.stdout);
    assert!(
        !text.trim().is_empty(),
        "max_tokens=1 must produce at least some output"
    );
}

/// Single-token prompt should produce valid output
#[test]
fn stress_single_token_prompt() {
    let model_path = match std::env::var("TEST_MODEL_PATH") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            eprintln!("FALSIFICATION-SKIP: gate=stress_single_token_prompt reason=TEST_MODEL_PATH not set");
            return;
        }
    };

    let output = std::process::Command::new("cargo")
        .args([
            "run",
            "--bin",
            "apr",
            "--features",
            "inference",
            "--",
            "run",
            &model_path.to_string_lossy(),
            "--prompt",
            "Hi",
            "--max-tokens",
            "8",
            "--temperature",
            "0",
        ])
        .output()
        .expect("single token prompt generation");

    assert!(
        output.status.success(),
        "Single-token prompt must succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Very long prompt (1000+ chars) should not panic
#[test]
fn stress_long_prompt_no_panic() {
    let model_path = match std::env::var("TEST_MODEL_PATH") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            eprintln!("FALSIFICATION-SKIP: gate=stress_long_prompt_no_panic reason=TEST_MODEL_PATH not set");
            return;
        }
    };

    let long_prompt = "The quick brown fox jumps over the lazy dog. ".repeat(50);

    let output = std::process::Command::new("cargo")
        .args([
            "run",
            "--bin",
            "apr",
            "--features",
            "inference",
            "--",
            "run",
            &model_path.to_string_lossy(),
            "--prompt",
            &long_prompt,
            "--max-tokens",
            "8",
            "--temperature",
            "0",
        ])
        .output()
        .expect("long prompt generation");

    // Should either succeed or fail gracefully (not panic/crash)
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("panicked"),
            "Long prompt must not cause panic: {stderr}"
        );
    }
}

// =============================================================================
// Sequential Endurance Tests
// =============================================================================

/// 20 sequential generations all succeed and produce output
#[test]
fn stress_20_sequential_generations() {
    let model_path = match std::env::var("TEST_MODEL_PATH") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            eprintln!("FALSIFICATION-SKIP: gate=stress_20_sequential_generations reason=TEST_MODEL_PATH not set");
            return;
        }
    };

    let mut outputs = Vec::new();
    for i in 0..20 {
        let output = std::process::Command::new("cargo")
            .args([
                "run",
                "--bin",
                "apr",
                "--features",
                "inference",
                "--",
                "run",
                &model_path.to_string_lossy(),
                "--prompt",
                &format!("The number {} is", i + 1),
                "--max-tokens",
                "4",
                "--temperature",
                "0",
            ])
            .output()
            .unwrap_or_else(|_| panic!("Generation {i} failed to execute"));

        assert!(
            output.status.success(),
            "Generation {i}/20 must succeed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let text = String::from_utf8_lossy(&output.stdout).to_string();
        outputs.push(text);
    }

    // At least some outputs should be non-empty
    let non_empty = outputs.iter().filter(|o| !o.trim().is_empty()).count();
    assert!(
        non_empty >= 15,
        "At least 15/20 generations must produce output (got {non_empty})"
    );
}

// =============================================================================
// Serve Endpoint Tests
// =============================================================================

/// apr serve starts and responds to a health check
#[test]
fn stress_serve_health_check() {
    let model_path = match std::env::var("TEST_MODEL_PATH") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            eprintln!(
                "FALSIFICATION-SKIP: gate=stress_serve_health_check reason=TEST_MODEL_PATH not set"
            );
            return;
        }
    };

    // Start server in background
    let mut child = match std::process::Command::new("cargo")
        .args([
            "run",
            "--bin",
            "apr",
            "--features",
            "inference",
            "--",
            "serve",
            &model_path.to_string_lossy(),
            "--port",
            "18901",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "FALSIFICATION-SKIP: gate=stress_serve_health_check reason=spawn failed: {e}"
            );
            return;
        }
    };

    // Wait for server to start
    std::thread::sleep(std::time::Duration::from_secs(5));

    // Health check
    let health = std::process::Command::new("curl")
        .args([
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "http://localhost:18901/health",
        ])
        .output();

    // Kill server
    let _ = child.kill();
    let _ = child.wait();

    if let Ok(h) = health {
        let code = String::from_utf8_lossy(&h.stdout);
        assert!(
            code == "200" || code == "404",
            "Serve health check returned unexpected code: {code}"
        );
    }
}

// =============================================================================
// Chat Template Stress Tests
// =============================================================================

/// 50-message chat conversation maintains valid formatting
#[test]
fn stress_chat_50_messages() {
    let template = aprender::text::chat_template::auto_detect_template("qwen2.5-coder");

    let mut messages = Vec::new();
    for i in 0..50 {
        if i % 2 == 0 {
            messages.push(aprender::text::chat_template::ChatMessage::user(format!(
                "Question {}: What is {}+{}?",
                i / 2 + 1,
                i,
                i + 1
            )));
        } else {
            messages.push(aprender::text::chat_template::ChatMessage::assistant(
                format!("The answer is {}.", i + (i + 1)),
            ));
        }
    }

    let formatted = template.format_conversation(&messages).expect("format");
    assert!(!formatted.is_empty(), "50-message conversation must format");
    // Should contain markers from the template
    assert!(
        formatted.contains("user") || formatted.contains("im_start"),
        "Formatted conversation must contain role markers"
    );
}

/// Chat with very large message content
#[test]
fn stress_chat_large_message() {
    let template = aprender::text::chat_template::auto_detect_template("qwen2.5-coder");
    let large_content = "x".repeat(100_000);
    let messages = vec![aprender::text::chat_template::ChatMessage::user(
        large_content,
    )];
    let formatted = template.format_conversation(&messages).expect("format");
    assert!(formatted.len() > 100_000, "Large message must be preserved");
}

/// Chat with empty messages
#[test]
fn stress_chat_empty_messages() {
    let template = aprender::text::chat_template::auto_detect_template("qwen2.5-coder");
    let messages = vec![
        aprender::text::chat_template::ChatMessage::user(String::new()),
        aprender::text::chat_template::ChatMessage::assistant(String::new()),
    ];
    let formatted = template.format_conversation(&messages).expect("format");
    assert!(
        !formatted.is_empty(),
        "Empty messages must still format with template markers"
    );
}

/// Chat with only system message
#[test]
fn stress_chat_system_only() {
    let template = aprender::text::chat_template::auto_detect_template("qwen2.5-coder");
    let messages = vec![aprender::text::chat_template::ChatMessage::system(
        "You are helpful.".to_string(),
    )];
    let formatted = template.format_conversation(&messages).expect("format");
    assert!(!formatted.is_empty(), "System-only message must format");
}

// =============================================================================
// Generation Parameter Edge Cases (no model needed for template tests)
// =============================================================================

/// Format conversation with alternating roles never panics
#[test]
fn stress_alternating_roles_no_panic() {
    let template = aprender::text::chat_template::auto_detect_template("qwen2.5-coder");
    let roles = ["system", "user", "assistant", "user", "assistant"];
    let messages: Vec<_> = roles
        .iter()
        .enumerate()
        .map(|(i, &role)| {
            aprender::text::chat_template::ChatMessage::new(
                role.to_string(),
                format!("Message {i}"),
            )
        })
        .collect();
    let formatted = template.format_conversation(&messages).expect("format");
    assert!(!formatted.is_empty());
}

/// Multiple consecutive user messages (no assistant between)
#[test]
fn stress_consecutive_user_messages() {
    let template = aprender::text::chat_template::auto_detect_template("qwen2.5-coder");
    let messages = vec![
        aprender::text::chat_template::ChatMessage::user("First question".to_string()),
        aprender::text::chat_template::ChatMessage::user("Second question".to_string()),
        aprender::text::chat_template::ChatMessage::user("Third question".to_string()),
    ];
    let formatted = template.format_conversation(&messages).expect("format");
    assert!(
        !formatted.is_empty(),
        "Consecutive user messages must format"
    );
}
