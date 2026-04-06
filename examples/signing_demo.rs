//! Pacha Model Signing Demo
//!
//! Demonstrates Ed25519 digital signatures for model integrity.
//!
//! Run with: `cargo run --example signing_demo`

use pacha::signing::{
    sign_model, verify_model, verify_model_with_key, Keyring, ModelSignature, SigningKey,
    VerifyingKey,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Pacha Model Signing Demo ===\n");

    // Simulate model data
    let model_data = generate_mock_model_data();
    println!("Model size: {} bytes", model_data.len());

    // 1. Key generation
    println!("\n--- Key Generation ---");
    let (signing_key, verifying_key) = demo_key_generation()?;

    // 2. Signing
    println!("\n--- Model Signing ---");
    let signature = demo_signing(&model_data, &signing_key)?;

    // 3. Verification
    println!("\n--- Signature Verification ---");
    demo_verification(&model_data, &signature, &verifying_key)?;

    // 4. Keyring management
    println!("\n--- Keyring Management ---");
    demo_keyring()?;

    // 5. Error cases
    println!("\n--- Error Handling ---");
    demo_error_cases(&model_data, &signature)?;

    println!("\n=== Demo Complete ===");
    Ok(())
}

fn generate_mock_model_data() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&[3, 0, 0, 0]);
    for i in 0..2000 {
        data.push((i * 7 % 256) as u8);
    }
    data
}

fn demo_key_generation() -> Result<(SigningKey, VerifyingKey), Box<dyn std::error::Error>> {
    // Generate a new signing key
    let signing_key = SigningKey::generate();

    // Derive the verifying (public) key
    let verifying_key = signing_key.verifying_key();

    println!("Generated Ed25519-like keypair:");
    println!("  Public key (hex):  {}...", &verifying_key.to_hex()[..32]);
    println!("  Private key: [REDACTED - 32 bytes]");

    // Keys can be serialized for storage
    let public_hex = verifying_key.to_hex();
    println!("\nPublic key length: {} hex chars", public_hex.len());

    // And deserialized
    let restored = VerifyingKey::from_hex(&public_hex)?;
    assert_eq!(verifying_key.to_hex(), restored.to_hex());
    println!("Key round-trip successful");

    // PEM format for file storage
    let pem = signing_key.to_pem();
    println!("\nPEM format (first line): {}", pem.lines().next().unwrap_or(""));

    Ok((signing_key, verifying_key))
}

fn demo_signing(
    model_data: &[u8],
    signing_key: &SigningKey,
) -> Result<ModelSignature, Box<dyn std::error::Error>> {
    // Sign the model
    let signature = sign_model(model_data, signing_key)?;

    println!("Model signed successfully:");
    println!("  Algorithm:    {}", signature.algorithm);
    println!("  Content hash: {}...", &signature.content_hash[..32]);
    println!("  Signer key:   {}...", &signature.signer_key[..32]);
    println!("  Timestamp:    {}", signature.timestamp);

    // Signature can be saved alongside the model
    // signature.save("model.sig")?;

    Ok(signature)
}

fn demo_verification(
    model_data: &[u8],
    signature: &ModelSignature,
    verifying_key: &VerifyingKey,
) -> Result<(), Box<dyn std::error::Error>> {
    // Basic verification (uses key embedded in signature)
    println!("Verifying with embedded key...");
    verify_model(model_data, signature)?;
    println!("  Verification PASSED");

    // Verification with expected key
    println!("Verifying with expected key...");
    verify_model_with_key(model_data, signature, verifying_key)?;
    println!("  Verification PASSED (key matches expected)");

    Ok(())
}

fn demo_keyring() -> Result<(), Box<dyn std::error::Error>> {
    // Create a keyring for managing trusted keys
    let mut keyring = Keyring::new();
    println!("Created empty keyring");

    // Generate some keypairs
    let alice_key = SigningKey::generate();
    let alice_pub = alice_key.verifying_key();

    let bob_key = SigningKey::generate();
    let bob_pub = bob_key.verifying_key();

    // Add to keyring (takes reference to VerifyingKey)
    keyring.add("alice@example.com", &alice_pub);
    keyring.add("bob@example.com", &bob_pub);
    println!("Added 2 trusted keys");

    // Look up keys
    let retrieved = keyring.get("alice@example.com")?;
    println!("Found Alice's key: {}...", &retrieved.to_hex()[..16]);

    // List all keys
    println!("\nKeyring contents:");
    for name in keyring.list() {
        println!("  - {}", name);
    }

    // Remove a key
    keyring.remove("bob@example.com");
    println!("\nRemoved Bob's key, {} key(s) remaining", keyring.list().len());

    // Set default key
    keyring.set_default("alice@example.com");
    let default = keyring.default_key()?;
    println!("Default key: {}...", &default.to_hex()[..16]);

    Ok(())
}

fn demo_error_cases(
    model_data: &[u8],
    signature: &ModelSignature,
) -> Result<(), Box<dyn std::error::Error>> {
    // Tampered data
    println!("Testing tampered data...");
    let mut tampered = model_data.to_vec();
    tampered[100] ^= 0xFF;
    match verify_model(&tampered, signature) {
        Ok(()) => println!("  Unexpectedly passed!"),
        Err(e) => println!("  Correctly failed: {}", e),
    }

    // Wrong key verification
    println!("Testing wrong key...");
    let wrong_key = SigningKey::generate().verifying_key();
    match verify_model_with_key(model_data, signature, &wrong_key) {
        Ok(()) => println!("  Unexpectedly passed!"),
        Err(e) => println!("  Correctly failed: {}", e),
    }

    // Invalid hex key
    println!("Testing invalid key format...");
    match VerifyingKey::from_hex("not-valid-hex") {
        Ok(_) => println!("  Unexpectedly passed!"),
        Err(e) => println!("  Correctly failed: {}", e),
    }

    println!("\nAll error cases handled correctly!");

    Ok(())
}
