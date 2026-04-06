//! Example: Running the Realizar HTTP API server
//!
//! This example demonstrates how to:
//! 1. Create a demo model
//! 2. Start the HTTP server
//! 3. Make API requests
//!
//! Implements the Local-Global SLM/APR/LLM Serving Spec:
//! - OpenAI-compatible API (§5.1)
//! - Native Realizar API (§5.2)
//!
//! Run with: cargo run --example api_server --features server

use std::net::SocketAddr;

use anyhow::Result;
use realizar::api::{create_router, AppState};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Realizar API Server Example ===\n");

    // Create demo model and tokenizer
    println!("Creating demo model...");
    let state = AppState::demo()?;
    println!("  ✓ Model created\n");

    // Create router
    let app = create_router(state);

    // Bind to localhost
    let addr: SocketAddr = "127.0.0.1:3000".parse()?;
    println!("Starting server on http://{}\n", addr);

    println!("=== Available Endpoints ===\n");

    println!("Health & Metrics:");
    println!("  GET  /health              - Health check");
    println!("  GET  /metrics             - Prometheus metrics\n");

    println!("OpenAI-Compatible API (§5.1):");
    println!("  GET  /v1/models           - List models");
    println!("  POST /v1/completions      - Text completions");
    println!("  POST /v1/chat/completions - Chat completions");
    println!("  POST /v1/embeddings       - Generate embeddings\n");

    println!("Native Realizar API (§5.2):");
    println!("  POST /realize/generate    - Streaming generation");
    println!("  POST /realize/batch       - Batch inference");
    println!("  POST /realize/embed       - Embeddings");
    println!("  GET  /realize/model       - Model metadata + lineage");
    println!("  POST /realize/reload      - Hot-reload model\n");

    println!("Legacy API:");
    println!("  POST /tokenize            - Tokenize text");
    println!("  POST /generate            - Generate text");
    println!("  POST /batch/tokenize      - Batch tokenize");
    println!("  POST /batch/generate      - Batch generate\n");

    println!("=== Example Requests ===\n");

    println!("# Health check");
    println!("curl http://127.0.0.1:3000/health\n");

    println!("# OpenAI chat completions");
    println!("curl -X POST http://127.0.0.1:3000/v1/chat/completions \\");
    println!("  -H 'Content-Type: application/json' \\");
    println!("  -d '{{");
    println!("    \"model\": \"default\",");
    println!("    \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}],");
    println!("    \"max_tokens\": 100");
    println!("  }}'\n");

    println!("# OpenAI completions");
    println!("curl -X POST http://127.0.0.1:3000/v1/completions \\");
    println!("  -H 'Content-Type: application/json' \\");
    println!(
        "  -d '{{\"model\": \"default\", \"prompt\": \"Once upon a\", \"max_tokens\": 50}}'\n"
    );

    println!("# OpenAI embeddings");
    println!("curl -X POST http://127.0.0.1:3000/v1/embeddings \\");
    println!("  -H 'Content-Type: application/json' \\");
    println!("  -d '{{\"input\": \"Hello world\", \"model\": \"default\"}}'\n");

    println!("# Native API - Model metadata");
    println!("curl http://127.0.0.1:3000/realize/model\n");

    println!("# Native API - Embeddings");
    println!("curl -X POST http://127.0.0.1:3000/realize/embed \\");
    println!("  -H 'Content-Type: application/json' \\");
    println!("  -d '{{\"input\": \"Semantic search query\"}}'\n");

    println!("# Native API - Hot reload");
    println!("curl -X POST http://127.0.0.1:3000/realize/reload \\");
    println!("  -H 'Content-Type: application/json' \\");
    println!("  -d '{{\"model\": \"new-model\"}}'\n");

    println!("Server running... (Press Ctrl+C to stop)\n");

    // Create listener and serve
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
