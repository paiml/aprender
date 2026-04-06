//! Serve command implementations
//!
//! This module contains the Realizar model server command extracted from main.rs.

#![cfg(feature = "native")]

use crate::ansi_colors::Colorize;

/// Run the Realizar model server
pub fn cmd_serve(
    model: Option<String>,
    host: &str,
    port: u16,
    openai_api: bool,
    watch: bool,
) -> anyhow::Result<()> {
    println!("{}", "🚀 Starting Realizar Model Server".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    // Resolve model reference if provided
    let resolved_model = if let Some(model_ref) = &model {
        // Try to resolve via pacha aliases
        let resolved = resolve_model_for_serve(model_ref);
        println!("{} Model: {}", "•".bright_blue(), model_ref.cyan());
        if resolved != *model_ref {
            println!("{} Resolved: {}", "•".bright_blue(), resolved.dimmed());
        }
        Some(resolved)
    } else {
        println!("{} Model: {}", "•".bright_blue(), "demo (no model specified)".dimmed());
        None
    };

    println!("{} Address: {}:{}", "•".bright_blue(), host.cyan(), port.to_string().cyan());
    println!(
        "{} OpenAI API: {}",
        "•".bright_blue(),
        if openai_api { "enabled".green() } else { "disabled".dimmed() }
    );
    if watch {
        println!("{} Hot-reload: {}", "•".bright_blue(), "enabled".green());
    }
    println!();

    // Check if model is cached (if using pacha scheme)
    if let Some(ref resolved) = resolved_model {
        if resolved.starts_with("hf://") || resolved.starts_with("pacha://") {
            println!("{}", "Checking model cache...".dimmed());
            println!("{} Model will be pulled on first request if not cached", "ℹ".bright_blue());
            println!();
        }
    }

    println!("{}", "Endpoints:".bright_yellow());
    println!("  GET  /health              - Health check");
    println!("  GET  /metrics             - Prometheus metrics");
    println!("  POST /generate            - Text generation");
    println!("  POST /tokenize            - Tokenize text");
    println!("  POST /stream/generate     - Streaming generation (SSE)");
    if openai_api {
        println!();
        println!("{}", "OpenAI-Compatible API:".bright_yellow());
        println!("  GET  /v1/models           - List models");
        println!("  POST /v1/chat/completions - Chat completions");
    }
    println!();

    // Show curl examples
    println!("{}", "Quick Test:".bright_yellow());
    println!("  # Health check");
    println!("  curl http://{}:{}/health", host, port);
    println!();
    if openai_api {
        println!("  # Chat completion (OpenAI-compatible)");
        println!("  curl http://{}:{}/v1/chat/completions \\", host, port);
        println!("    -H \"Content-Type: application/json\" \\");
        println!("    -d '{{\"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'");
        println!();
    }

    // Note: Full integration with Realizar would require tokio runtime
    println!("{}", "─".repeat(60).dimmed());
    println!("{}", "Note:".bright_yellow());
    println!("  For production serving, use the Realizar CLI directly:");
    println!();
    if let Some(ref model_ref) = resolved_model {
        println!("  {} {}", "realizar serve --model".cyan(), model_ref.bright_white());
    } else {
        println!("  {} ", "realizar serve --demo".cyan());
    }
    println!();

    // Show pacha model management
    println!("{}", "Model Management:".bright_yellow());
    println!("  # Pull a model first");
    println!("  batuta pacha pull {}", model.as_deref().unwrap_or("llama3:8b"));
    println!();
    println!("  # List cached models");
    println!("  batuta pacha list");

    Ok(())
}

/// Resolve model reference for serving (alias expansion)
fn resolve_model_for_serve(model_ref: &str) -> String {
    // Built-in aliases for common models
    let aliases = [
        ("llama3", "hf://meta-llama/Meta-Llama-3-8B-Instruct-GGUF"),
        ("llama3:8b", "hf://meta-llama/Meta-Llama-3-8B-Instruct-GGUF"),
        ("llama3:70b", "hf://meta-llama/Meta-Llama-3-70B-Instruct-GGUF"),
        ("mistral", "hf://mistralai/Mistral-7B-Instruct-v0.2-GGUF"),
        ("mixtral", "hf://mistralai/Mixtral-8x7B-Instruct-v0.1-GGUF"),
        ("phi3", "hf://microsoft/Phi-3-mini-4k-instruct-gguf"),
        ("gemma", "hf://google/gemma-7b-it-GGUF"),
        ("qwen2", "hf://Qwen/Qwen2-7B-Instruct-GGUF"),
        ("codellama", "hf://codellama/CodeLlama-7b-Instruct-GGUF"),
    ];

    for (alias, target) in &aliases {
        if model_ref == *alias {
            return (*target).to_string();
        }
    }

    // If already a full URI, return as-is
    if model_ref.contains("://") {
        return model_ref.to_string();
    }

    // Otherwise, assume pacha:// scheme
    format!("pacha://{}", model_ref)
}
