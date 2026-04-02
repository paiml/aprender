/// Run an axum server with graceful shutdown and standard banner.
/// GH-172-FIX: 16MB thread stack for worker threads — GPU forward pass with
/// large-vocab models (151K × 4B = 608KB logits) overflows the default 2MB stack.
#[cfg(feature = "inference")]
fn run_server_async(app: axum::Router, bind_addr: &str, label: &str) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(16 * 1024 * 1024) // 16 MB (default: 2 MB)
        .build()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = bind_addr.to_string();
    let label = label.to_string();

    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("{label} server listening on http://{bind_addr}")
                .green()
                .bold()
        );
        println!();
        println!("{}", "Endpoints:".cyan());
        println!("  GET  /health              - Health check");
        println!("  GET  /metrics             - Prometheus metrics");
        println!("  POST /generate            - Text generation");
        println!("  POST /v1/completions      - OpenAI-compatible");
        println!("  POST /v1/chat/completions - Chat completions");
        println!();
        println!("{}", "Press Ctrl+C to stop".dimmed());

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        Ok(())
    })
}
