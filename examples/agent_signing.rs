//! Agent Manifest Signing Demo
//!
//! Demonstrates Ed25519 manifest signing and verification:
//! - Generate signing keypair
//! - Sign manifest content (BLAKE3 hash + Ed25519)
//! - Verify signature (tamper detection)
//! - Serialize/deserialize signature as TOML sidecar
//!
//! Run with: `cargo run --example agent_signing --features agents`

#[cfg(feature = "agents")]
fn main() {
    use batuta::agent::signing::{
        sign_manifest, signature_from_toml, signature_to_toml, verify_manifest,
    };

    println!("Agent Manifest Signing (Ed25519 + BLAKE3)");
    println!("==========================================");
    println!();

    // Sample manifest content
    let manifest = r#"
name = "research-agent"
version = "0.2.0"

[model]
model_path = "/models/llama-3.2-8b.apr"
max_tokens = 4096
temperature = 0.7

[resources]
max_iterations = 30
max_tool_calls = 50
max_cost_usd = 0.0
privacy = "sovereign"

[[capabilities]]
type = "memory"
"#;

    // Generate keypair
    let signing_key = pacha::signing::SigningKey::generate();
    let verifying_key = signing_key.verifying_key();
    println!("Generated Ed25519 keypair");
    println!();

    // Sign the manifest
    let signature = sign_manifest(manifest, &signing_key, Some("build-pipeline"));
    println!("Signed manifest:");
    println!("  BLAKE3 hash: {}...", &signature.content_hash[..16]);
    println!("  Signature:   {}...", &signature.signature_hex[..16]);
    println!("  Signer:      {}", signature.signer.as_deref().unwrap_or("(none)"));
    println!();

    // Verify signature
    let result = verify_manifest(manifest, &signature, &verifying_key);
    match &result {
        Ok(()) => println!("Verification: PASS"),
        Err(e) => println!("Verification: FAIL — {e}"),
    }
    assert!(result.is_ok());
    println!();

    // Demonstrate tamper detection (Jidoka: stop on defect)
    let tampered = manifest.replace("30", "999999");
    let tamper_result = verify_manifest(&tampered, &signature, &verifying_key);
    match &tamper_result {
        Ok(()) => println!("Tamper detection: MISSED (bad!)"),
        Err(e) => println!("Tamper detection: CAUGHT — {e}"),
    }
    assert!(tamper_result.is_err());
    println!();

    // Wrong key detection
    let other_key = pacha::signing::SigningKey::generate();
    let other_vk = other_key.verifying_key();
    let wrong_key_result = verify_manifest(manifest, &signature, &other_vk);
    match &wrong_key_result {
        Ok(()) => println!("Wrong key detection: MISSED (bad!)"),
        Err(e) => println!("Wrong key detection: CAUGHT — {e}"),
    }
    assert!(wrong_key_result.is_err());
    println!();

    // TOML sidecar roundtrip
    let toml_str = signature_to_toml(&signature);
    println!("--- Signature TOML sidecar ---");
    println!("{toml_str}");

    let parsed = signature_from_toml(&toml_str).expect("TOML parse failed");
    assert_eq!(parsed.content_hash, signature.content_hash);
    assert_eq!(parsed.signature_hex, signature.signature_hex);
    assert_eq!(parsed.signer, signature.signer);
    println!("TOML roundtrip: PASS");
    println!();

    println!("All signing checks passed.");
}

#[cfg(not(feature = "agents"))]
fn main() {
    eprintln!(
        "Enable `agents` feature: \
         cargo run --example agent_signing --features agents"
    );
    std::process::exit(1);
}
