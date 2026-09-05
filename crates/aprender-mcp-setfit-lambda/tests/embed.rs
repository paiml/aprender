//! The embedded-model door, proven on the build that actually embeds.
//!
//! Self-gating on the build configuration rather than on an env var at RUN
//! time: `build.rs` stages `APRENDER_SETFIT_MODEL` into the binary at BUILD
//! time, so whether there is anything to test is a compile-time fact —
//! `EMBEDDED_MODEL.is_empty()` — and the test reads exactly that. A plain CI
//! build skips visibly; a deploy-shaped build
//! (`APRENDER_SETFIT_MODEL=… cargo test -p aprender-mcp-setfit-lambda`)
//! proves the same bytes the Lambda would serve load through the verification
//! ladder and build the server.

#[test]
fn an_embedded_model_resolves_without_any_runtime_environment() {
    if aprender_mcp_setfit_lambda::EMBEDDED_MODEL.is_empty() {
        println!(
            "EMBED SKIP: this build staged no model — set APRENDER_SETFIT_MODEL at \
             build time to arm this gate"
        );
        return;
    }
    let model = aprender_mcp_setfit_lambda::resolve_model()
        .expect("embedded bytes must pass the verification ladder");
    let server = aprender_mcp_setfit::build_server(model, "embed-test", "0");
    assert!(server.is_ok(), "the embedded model must build the server");
}
