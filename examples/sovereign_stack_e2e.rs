//! End-to-End Sovereign Stack Integration Example
//!
//! Demonstrates the full workflow from model registration to privacy-preserving inference:
//!
//! 1. **Content Addressing**: BLAKE3 hashing for model integrity
//! 2. **Signing**: Ed25519 digital signatures for model authenticity
//! 3. **Encryption**: ChaCha20-Poly1305 encryption for distribution
//! 4. **Privacy Tiers**: Sovereign/Private/Standard enforcement
//! 5. **Backend Selection**: Intelligent routing based on privacy requirements
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example sovereign_stack_e2e
//! ```
//!
//! ## Per Initial Release Specification §2-6

use batuta::serve::{BackendSelector, LatencyTier, PrivacyTier, ServingBackend};

fn main() -> anyhow::Result<()> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("     🛡️  Sovereign AI Stack - End-to-End Demo");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // Phase 1: Model Security with Pacha
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Phase 1: Model Security (Pacha)                             │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    demo_model_security()?;

    // =========================================================================
    // Phase 2: Privacy Tier Enforcement
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ Phase 2: Privacy Tier Enforcement (Batuta)                  │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    demo_privacy_tiers()?;

    // =========================================================================
    // Phase 3: Backend Selection
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ Phase 3: Backend Selection (Batuta Serve)                   │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    demo_backend_selection()?;

    // =========================================================================
    // Phase 4: Full Workflow Integration
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ Phase 4: Full Sovereign Stack Workflow                      │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    demo_full_workflow()?;

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("     ✅ All Sovereign Stack phases completed successfully!");
    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}

/// Compute BLAKE3 hash of data (content addressing)
fn blake3_hash(data: &[u8]) -> String {
    let hash = blake3::hash(data);
    hash.to_hex().to_string()
}

/// Demonstrate model security features from Pacha
fn demo_model_security() -> anyhow::Result<()> {
    use pacha::crypto::{decrypt_model, encrypt_model, is_encrypted};
    use pacha::signing::{sign_model, verify_model, SigningKey};

    // Simulated model data (in production, this would be a real GGUF/SafeTensors file)
    let model_data = b"[SIMULATED MODEL WEIGHTS - Llama 3 8B Q4_K_M]".repeat(100);
    println!("  📦 Model size: {} bytes", model_data.len());

    // -------------------------------------------------------------------------
    // 1. Content Addressing (BLAKE3)
    // -------------------------------------------------------------------------
    println!("\n  1️⃣  Content Addressing (BLAKE3)");
    let content_hash = blake3_hash(&model_data);
    println!("     Content Hash: {}...", &content_hash[..32]);
    println!("     ✓ Unique identifier for deduplication and integrity");

    // -------------------------------------------------------------------------
    // 2. Digital Signatures (Ed25519)
    // -------------------------------------------------------------------------
    println!("\n  2️⃣  Digital Signatures (Ed25519)");

    // Generate signing key pair
    let signing_key = SigningKey::generate();
    let verifying_key = signing_key.verifying_key();
    println!("     Public Key:  {}...", &verifying_key.to_hex()[..32]);

    // Sign the model
    let signature = sign_model(&model_data, &signing_key)?;
    println!("     Signature:   {}...", &signature.signature[..32]);

    // Verify the signature
    verify_model(&model_data, &signature)?;
    println!("     ✓ Signature VERIFIED - Model integrity confirmed");

    // -------------------------------------------------------------------------
    // 3. Encryption at Rest (ChaCha20-Poly1305)
    // -------------------------------------------------------------------------
    println!("\n  3️⃣  Encryption at Rest (ChaCha20-Poly1305)");

    let password = "sovereign-secret-key-2025";

    // Encrypt the model
    let encrypted = encrypt_model(&model_data, password)?;
    println!("     Encrypted size: {} bytes", encrypted.len());
    println!("     Is encrypted:   {}", is_encrypted(&encrypted));

    // Decrypt the model
    let decrypted = decrypt_model(&encrypted, password)?;
    assert_eq!(decrypted, model_data);
    println!("     ✓ Decryption successful - Data matches original");

    // Demonstrate tamper detection
    println!("\n  4️⃣  Tamper Detection");
    let mut tampered = encrypted.clone();
    if tampered.len() > 100 {
        tampered[100] ^= 0xFF; // Flip some bits
    }
    let tamper_result = decrypt_model(&tampered, password);
    match tamper_result {
        Err(_) => println!("     ✓ Tampered data REJECTED - AEAD authentication failed"),
        Ok(_) => println!("     ⚠ WARNING: Tampered data was not detected"),
    }

    Ok(())
}

/// Demonstrate privacy tier enforcement
fn demo_privacy_tiers() -> anyhow::Result<()> {
    println!("  Privacy tiers control which backends can be used:\n");

    // -------------------------------------------------------------------------
    // Sovereign Tier (Healthcare, Government)
    // -------------------------------------------------------------------------
    println!("  🔒 SOVEREIGN Tier (Healthcare, Government):");
    println!("     - Blocks ALL external API calls");
    println!("     - Only local inference allowed");

    let local_backends = [
        ServingBackend::Realizar,
        ServingBackend::Ollama,
        ServingBackend::LlamaCpp,
        ServingBackend::Llamafile,
    ];

    let sovereign_allowed: Vec<_> =
        local_backends.iter().filter(|b| PrivacyTier::Sovereign.allows(**b)).collect();
    println!("     Allowed backends: {:?}", sovereign_allowed);

    let blocked_by_sovereign = PrivacyTier::Sovereign.blocked_hosts();
    println!("     Blocked hosts: {} external APIs", blocked_by_sovereign.len());

    // -------------------------------------------------------------------------
    // Private Tier (Financial Services)
    // -------------------------------------------------------------------------
    println!("\n  🔐 PRIVATE Tier (Financial Services):");
    println!("     - VPC/dedicated endpoints only");
    println!("     - No public cloud APIs");

    let private_allows_azure = PrivacyTier::Private.allows(ServingBackend::AzureOpenAI);
    let private_allows_openai = PrivacyTier::Private.allows(ServingBackend::OpenAI);
    println!("     Allows Azure OpenAI: {}", private_allows_azure);
    println!("     Allows Public OpenAI: {}", private_allows_openai);

    // -------------------------------------------------------------------------
    // Standard Tier (General Use)
    // -------------------------------------------------------------------------
    println!("\n  🌐 STANDARD Tier (General Use):");
    println!("     - All backends available");
    println!("     - Public APIs allowed");

    let standard_blocked = PrivacyTier::Standard.blocked_hosts();
    println!("     Blocked hosts: {} (none)", standard_blocked.len());

    Ok(())
}

/// Demonstrate intelligent backend selection
fn demo_backend_selection() -> anyhow::Result<()> {
    println!("  Backend selection based on requirements:\n");

    // -------------------------------------------------------------------------
    // Scenario 1: Local-only inference
    // -------------------------------------------------------------------------
    println!("  📍 Scenario 1: Local-only (Sovereign)");
    let selector = BackendSelector::new()
        .with_privacy(PrivacyTier::Sovereign)
        .with_latency(LatencyTier::Interactive);

    let recommendations = selector.recommend();
    println!("     Recommended backends: {:?}", recommendations);
    println!("     ✓ External APIs blocked");

    // -------------------------------------------------------------------------
    // Scenario 2: Real-time latency
    // -------------------------------------------------------------------------
    println!("\n  ⚡ Scenario 2: Real-time latency");
    let selector = BackendSelector::new()
        .with_privacy(PrivacyTier::Standard)
        .with_latency(LatencyTier::RealTime);

    let recommendations = selector.recommend();
    println!("     Recommended backends: {:?}", recommendations);
    println!("     ✓ Prioritizing low-latency backends");

    // -------------------------------------------------------------------------
    // Scenario 3: Preferred backend
    // -------------------------------------------------------------------------
    println!("\n  💰 Scenario 3: Preferred local backend");
    let selector = BackendSelector::new()
        .with_privacy(PrivacyTier::Standard)
        .prefer(ServingBackend::Realizar)
        .prefer(ServingBackend::Ollama);

    let recommendations = selector.recommend();
    println!("     Recommended backends: {:?}", recommendations);
    println!("     ✓ Local backends prioritized");

    // -------------------------------------------------------------------------
    // Scenario 4: Backend validation (Poka-Yoke)
    // -------------------------------------------------------------------------
    println!("\n  🛡️  Scenario 4: Backend validation (Poka-Yoke)");
    let sovereign_selector = BackendSelector::new().with_privacy(PrivacyTier::Sovereign);

    let local_valid = sovereign_selector.validate(ServingBackend::Realizar);
    let cloud_valid = sovereign_selector.validate(ServingBackend::OpenAI);

    println!("     Realizar valid: {:?}", local_valid.is_ok());
    println!("     OpenAI valid:   {:?}", cloud_valid.is_ok());
    if let Err(e) = cloud_valid {
        println!("     Violation:      {:?}", e);
    }

    Ok(())
}

/// Demonstrate full sovereign stack workflow
fn demo_full_workflow() -> anyhow::Result<()> {
    use pacha::crypto::{decrypt_model, encrypt_model};
    use pacha::signing::{sign_model, verify_model, SigningKey};

    println!("  Full workflow: Register → Sign → Encrypt → Distribute → Verify → Serve\n");

    // Step 1: Model Registration
    println!("  Step 1: Model Registration");
    let model_data = b"[Sovereign Model Data]".repeat(50);
    let content_hash = blake3_hash(&model_data);
    println!("     ✓ Registered with hash: {}...", &content_hash[..16]);

    // Step 2: Sign for Distribution
    println!("\n  Step 2: Sign for Distribution");
    let signing_key = SigningKey::generate();
    let signature = sign_model(&model_data, &signing_key)?;
    println!("     ✓ Signed by: {}...", &signature.signer_key[..16]);

    // Step 3: Encrypt for Distribution
    println!("\n  Step 3: Encrypt for Distribution");
    let encrypted = encrypt_model(&model_data, "distribution-key")?;
    println!("     ✓ Encrypted: {} bytes", encrypted.len());

    // Step 4: Distribute (simulated)
    println!("\n  Step 4: Distribute to Air-Gapped Environment");
    println!("     ✓ Model bundle transferred offline");

    // Step 5: Verify at Load Time
    println!("\n  Step 5: Verify at Load Time");
    let decrypted = decrypt_model(&encrypted, "distribution-key")?;
    verify_model(&decrypted, &signature)?;
    println!("     ✓ Signature verified - Safe to load");

    // Step 6: Serve with Privacy Enforcement
    println!("\n  Step 6: Serve with Sovereign Privacy");
    let selector = BackendSelector::new()
        .with_privacy(PrivacyTier::Sovereign)
        .with_latency(LatencyTier::Interactive);

    let recommendations = selector.recommend();
    println!("     ✓ Serving via: {:?}", recommendations.first());
    println!("     ✓ External APIs BLOCKED - Data sovereignty maintained");

    Ok(())
}

// =============================================================================
// Integration Tests (Extreme TDD)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Content Addressing Tests
    // =========================================================================

    #[test]
    fn test_blake3_hash_deterministic() {
        let data = b"test model data";
        let hash1 = blake3_hash(data);
        let hash2 = blake3_hash(data);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_blake3_hash_different_for_different_data() {
        let data1 = b"model version 1";
        let data2 = b"model version 2";
        let hash1 = blake3_hash(data1);
        let hash2 = blake3_hash(data2);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_blake3_hash_length() {
        let data = b"test";
        let hash = blake3_hash(data);

        // BLAKE3 produces 64 hex characters (32 bytes)
        assert_eq!(hash.len(), 64);
    }

    // =========================================================================
    // Ed25519 Signature Tests
    // =========================================================================

    #[test]
    fn test_ed25519_sign_verify_roundtrip() {
        use pacha::signing::{sign_model, verify_model, SigningKey};

        let model_data = b"test model for signing";
        let signing_key = SigningKey::generate();

        let signature = sign_model(model_data, &signing_key).expect("signing should succeed");
        let result = verify_model(model_data, &signature);

        assert!(result.is_ok());
    }

    #[test]
    fn test_ed25519_rejects_tampered_data() {
        use pacha::signing::{sign_model, verify_model, SigningKey};

        let model_data = b"original model data";
        let signing_key = SigningKey::generate();
        let signature = sign_model(model_data, &signing_key).expect("signing should succeed");

        let tampered_data = b"tampered model data";
        let result = verify_model(tampered_data, &signature);

        assert!(result.is_err());
    }

    #[test]
    fn test_ed25519_signature_deterministic() {
        use pacha::signing::{sign_model, SigningKey};

        let model_data = b"deterministic test";
        let signing_key = SigningKey::generate();

        let sig1 = sign_model(model_data, &signing_key).expect("signing should succeed");
        let sig2 = sign_model(model_data, &signing_key).expect("signing should succeed");

        // Same key + same data = same signature (Ed25519 is deterministic)
        assert_eq!(sig1.signature, sig2.signature);
    }

    // =========================================================================
    // ChaCha20-Poly1305 Encryption Tests
    // =========================================================================

    #[test]
    fn test_chacha20_encrypt_decrypt_roundtrip() {
        use pacha::crypto::{decrypt_model, encrypt_model};

        let model_data = b"test model for encryption".repeat(10);
        let password = "test-password-123";

        let encrypted = encrypt_model(&model_data, password).expect("encryption should succeed");
        let decrypted = decrypt_model(&encrypted, password).expect("decryption should succeed");

        assert_eq!(decrypted, model_data);
    }

    #[test]
    fn test_chacha20_rejects_wrong_password() {
        use pacha::crypto::{decrypt_model, encrypt_model};

        let model_data = b"secret model data";
        let encrypted =
            encrypt_model(model_data, "correct-password").expect("encryption should succeed");

        let result = decrypt_model(&encrypted, "wrong-password");

        assert!(result.is_err());
    }

    #[test]
    fn test_chacha20_detects_tampering() {
        use pacha::crypto::{decrypt_model, encrypt_model};

        let model_data = b"model to be tampered";
        let encrypted = encrypt_model(model_data, "password").expect("encryption should succeed");

        let mut tampered = encrypted.clone();
        if tampered.len() > 50 {
            tampered[50] ^= 0xFF;
        }

        let result = decrypt_model(&tampered, "password");

        assert!(result.is_err());
    }

    #[test]
    fn test_encryption_detection() {
        use pacha::crypto::{encrypt_model, is_encrypted};

        let plain_data = b"not encrypted";
        let encrypted =
            encrypt_model(b"encrypted data", "pass").expect("encryption should succeed");

        assert!(!is_encrypted(plain_data));
        assert!(is_encrypted(&encrypted));
    }

    #[test]
    fn test_encryption_adds_overhead() {
        use pacha::crypto::encrypt_model;

        let small_data = b"small";
        let large_data = b"large".repeat(1000);

        let small_encrypted = encrypt_model(small_data, "p").expect("encryption should succeed");
        let large_encrypted = encrypt_model(&large_data, "p").expect("encryption should succeed");

        // Encrypted should be larger than original (salt + nonce + tag)
        assert!(small_encrypted.len() > small_data.len());
        assert!(large_encrypted.len() > large_data.len());
    }

    // =========================================================================
    // Privacy Tier Tests
    // =========================================================================

    #[test]
    fn test_sovereign_tier_blocks_external() {
        let selector = BackendSelector::new().with_privacy(PrivacyTier::Sovereign);

        // Should not allow public cloud backends
        assert!(!selector.is_valid(ServingBackend::OpenAI));
        assert!(!selector.is_valid(ServingBackend::Anthropic));
        assert!(!selector.is_valid(ServingBackend::HuggingFace));
    }

    #[test]
    fn test_sovereign_tier_allows_local() {
        let selector = BackendSelector::new().with_privacy(PrivacyTier::Sovereign);

        // Should allow local backends
        assert!(selector.is_valid(ServingBackend::Realizar));
        assert!(selector.is_valid(ServingBackend::Ollama));
        assert!(selector.is_valid(ServingBackend::LlamaCpp));
    }

    #[test]
    fn test_private_tier_allows_vpc() {
        // Private tier allows VPC/dedicated endpoints
        assert!(PrivacyTier::Private.allows(ServingBackend::AzureOpenAI));
        assert!(PrivacyTier::Private.allows(ServingBackend::AwsBedrock));

        // But blocks public APIs
        assert!(!PrivacyTier::Private.allows(ServingBackend::OpenAI));
    }

    #[test]
    fn test_standard_tier_allows_all() {
        // Standard tier should allow everything
        assert!(PrivacyTier::Standard.allows(ServingBackend::OpenAI));
        assert!(PrivacyTier::Standard.allows(ServingBackend::Realizar));
        assert!(PrivacyTier::Standard.allows(ServingBackend::AzureOpenAI));
    }

    #[test]
    fn test_privacy_tier_blocked_hosts() {
        let sovereign_blocked = PrivacyTier::Sovereign.blocked_hosts();
        let standard_blocked = PrivacyTier::Standard.blocked_hosts();

        // Sovereign should block many hosts
        assert!(sovereign_blocked.len() > 5);
        assert!(sovereign_blocked.contains(&"api.openai.com"));

        // Standard should block none
        assert!(standard_blocked.is_empty());
    }

    // =========================================================================
    // Backend Selection Tests
    // =========================================================================

    #[test]
    fn test_backend_selector_recommend_non_empty() {
        let selector = BackendSelector::new().with_privacy(PrivacyTier::Standard);

        let recommendations = selector.recommend();

        // Should return at least one backend
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_backend_selector_prefer_works() {
        let selector = BackendSelector::new()
            .with_privacy(PrivacyTier::Standard)
            .prefer(ServingBackend::Realizar);

        let recommendations = selector.recommend();

        // Preferred backend should be first
        assert_eq!(recommendations.first(), Some(&ServingBackend::Realizar));
    }

    #[test]
    fn test_backend_selector_disable_works() {
        let selector = BackendSelector::new()
            .with_privacy(PrivacyTier::Standard)
            .disable(ServingBackend::OpenAI);

        // Disabled backend should not be valid
        assert!(!selector.is_valid(ServingBackend::OpenAI));
    }

    #[test]
    fn test_backend_selector_validate_returns_error_for_violation() {
        let selector = BackendSelector::new().with_privacy(PrivacyTier::Sovereign);

        let result = selector.validate(ServingBackend::OpenAI);

        assert!(result.is_err());
    }

    #[test]
    fn test_backend_selector_validate_ok_for_valid() {
        let selector = BackendSelector::new().with_privacy(PrivacyTier::Sovereign);

        let result = selector.validate(ServingBackend::Realizar);

        assert!(result.is_ok());
    }

    // =========================================================================
    // Full Workflow Integration Tests
    // =========================================================================

    #[test]
    fn test_full_workflow_sign_encrypt_verify() {
        use pacha::crypto::{decrypt_model, encrypt_model};
        use pacha::signing::{sign_model, verify_model, SigningKey};

        // 1. Create model
        let model_data = b"integration test model";

        // 2. Sign
        let signing_key = SigningKey::generate();
        let signature = sign_model(model_data, &signing_key).expect("signing should succeed");

        // 3. Encrypt
        let encrypted =
            encrypt_model(model_data, "integration-key").expect("encryption should succeed");

        // 4. Decrypt
        let decrypted =
            decrypt_model(&encrypted, "integration-key").expect("decryption should succeed");

        // 5. Verify
        let result = verify_model(&decrypted, &signature);

        assert!(result.is_ok());
        assert_eq!(decrypted, model_data);
    }

    #[test]
    fn test_full_workflow_sovereign_deployment() {
        use pacha::crypto::{decrypt_model, encrypt_model};
        use pacha::signing::{sign_model, verify_model, SigningKey};

        // Simulate air-gapped deployment
        let model_data = b"sovereign deployment model";

        // Producer side: sign and encrypt
        let signing_key = SigningKey::generate();
        let verifying_key = signing_key.verifying_key();
        let signature = sign_model(model_data, &signing_key).expect("signing should succeed");
        let encrypted =
            encrypt_model(model_data, "air-gap-key").expect("encryption should succeed");

        // Consumer side: decrypt and verify
        let decrypted =
            decrypt_model(&encrypted, "air-gap-key").expect("decryption should succeed");
        verify_model(&decrypted, &signature).expect("verification should succeed");

        // Serve with sovereign privacy
        let selector = BackendSelector::new().with_privacy(PrivacyTier::Sovereign);

        // Verify external APIs are blocked
        assert!(!selector.is_valid(ServingBackend::OpenAI));

        // Verify public key matches
        assert_eq!(signature.signer_key, verifying_key.to_hex());
    }

    #[test]
    fn test_workflow_hash_then_sign() {
        use pacha::signing::{sign_model, verify_model, SigningKey};

        let model_data = b"model to hash and sign";

        // Hash for content addressing
        let hash = blake3_hash(model_data);
        assert_eq!(hash.len(), 64);

        // Sign for authenticity
        let signing_key = SigningKey::generate();
        let signature = sign_model(model_data, &signing_key).expect("signing should succeed");

        // Verify
        verify_model(model_data, &signature).expect("verification should succeed");

        // Hash in signature should match our computed hash
        assert_eq!(signature.content_hash, hash);
    }
}
