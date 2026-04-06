//! Agent Routing Driver Demo
//!
//! Demonstrates the RoutingDriver (Heijunka: load leveling):
//! - PrimaryWithFallback strategy (local-first)
//! - Spillover metrics tracking
//! - Privacy tier inheritance
//! - PrimaryOnly and FallbackOnly strategies
//!
//! Run with: `cargo run --example agent_routing --features agents`

#[cfg(feature = "agents")]
#[tokio::main]
async fn main() {
    use batuta::agent::driver::mock::MockDriver;
    use batuta::agent::driver::router::{RoutingDriver, RoutingStrategy};
    use batuta::agent::driver::{CompletionRequest, CompletionResponse, LlmDriver};
    use batuta::agent::result::StopReason;
    use batuta::serve::backends::PrivacyTier;

    println!("Agent Routing Driver Demo (Heijunka)");
    println!("====================================");
    println!();

    let request = CompletionRequest {
        model: String::new(),
        messages: vec![],
        tools: vec![],
        max_tokens: 1024,
        temperature: 0.7,
        system: None,
    };

    // --- Primary succeeds (no spillover) ---
    println!("--- PrimaryWithFallback: Primary Succeeds ---");
    let primary = MockDriver::single_response("Local response");
    let fallback = MockDriver::single_response("Remote response");

    let router = RoutingDriver::new(Box::new(primary), Box::new(fallback));

    assert_eq!(router.privacy_tier(), PrivacyTier::Sovereign);
    println!("  Privacy tier: {:?}", router.privacy_tier());

    let result = router.complete(request.clone()).await;
    match &result {
        Ok(r) => println!("  Response: {}", r.text),
        Err(e) => println!("  Error: {e}"),
    }
    println!(
        "  Metrics: primary_attempts={}, spillovers={}",
        router.metrics().primary_attempts(),
        router.metrics().spillover_count(),
    );
    println!();

    // --- Metrics after successful primary ---
    println!("--- Metrics After Success ---");
    println!("  Primary attempts: {}", router.metrics().primary_attempts(),);
    println!("  Spillovers: {}", router.metrics().spillover_count(),);
    println!("  Fallback success rate: {:.1}%", router.metrics().fallback_success_rate() * 100.0,);
    println!();

    // --- PrimaryOnly strategy ---
    println!("--- PrimaryOnly Strategy ---");
    let primary = MockDriver::single_response("Local only");

    let router =
        RoutingDriver::primary_only(Box::new(primary)).with_strategy(RoutingStrategy::PrimaryOnly);

    let result = router.complete(request.clone()).await;
    match &result {
        Ok(r) => println!("  Response: {}", r.text),
        Err(e) => println!("  Error: {e}"),
    }
    println!("  Strategy: PrimaryOnly, spillovers={}", router.metrics().spillover_count(),);
    println!();

    // --- FallbackOnly strategy ---
    println!("--- FallbackOnly Strategy ---");
    let primary = MockDriver::single_response("Unused");
    let fallback = MockDriver::single_response("Remote only");

    let router = RoutingDriver::new(Box::new(primary), Box::new(fallback))
        .with_strategy(RoutingStrategy::FallbackOnly);

    let result = router.complete(request.clone()).await;
    match &result {
        Ok(r) => println!("  Response: {}", r.text),
        Err(e) => println!("  Error: {e}"),
    }
    println!("  Strategy: FallbackOnly");
    println!();

    // --- Privacy tier inheritance ---
    println!("--- Privacy Tier Inheritance ---");
    println!(
        "  Sovereign + Sovereign = {:?}",
        RoutingDriver::new(
            Box::new(MockDriver::single_response("a")),
            Box::new(MockDriver::single_response("b")),
        )
        .privacy_tier()
    );
    println!("  (Both MockDrivers default to Sovereign)");
    println!();

    // --- Multiple calls with metrics ---
    println!("--- Multiple Calls Metrics ---");
    let responses: Vec<CompletionResponse> = (1..=5)
        .map(|i| CompletionResponse {
            text: format!("Response {i}"),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        })
        .collect();
    let primary = MockDriver::new(responses);
    let router = RoutingDriver::primary_only(Box::new(primary));

    for i in 1..=5 {
        let _ = router.complete(request.clone()).await;
        println!("  After call {i}: primary_attempts={}", router.metrics().primary_attempts(),);
    }
    println!();

    println!("All routing demos completed.");
}

#[cfg(not(feature = "agents"))]
fn main() {
    eprintln!(
        "Enable `agents` feature: \
         cargo run --example agent_routing --features agents"
    );
    std::process::exit(1);
}
