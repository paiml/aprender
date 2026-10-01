//! The embedded-weights door, proven on the build that embeds.

#[test]
fn an_embedded_model_resolves_without_any_runtime_environment() {
    if aprender_mcp_chronos_lambda::EMBEDDED_WEIGHTS.is_empty() {
        println!(
            "EMBED SKIP: this build staged no model weights — stage models/chronos-bolt-tiny/f16 \
             to arm this gate"
        );
        return;
    }
    let model = aprender_mcp_chronos_lambda::resolve_model()
        .map(std::sync::Arc::new)
        .expect("embedded bytes must pass the verification ladder");
    let server = aprender_mcp_chronos_lambda::build_server(model, "embed-test", "0");
    assert!(server.is_ok(), "the embedded model must build the server");
}
