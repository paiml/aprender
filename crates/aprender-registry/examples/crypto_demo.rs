//! Pacha Encryption at Rest Demo
//!
//! Demonstrates model encryption and decryption for secure distribution.
//!
//! Run with: `cargo run --example crypto_demo`

use pacha::crypto::{
    decrypt_model, encrypt_model, encrypt_model_with_config, get_version, is_encrypted,
    EncryptionConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Pacha Encryption at Rest Demo ===\n");

    // Simulate model data
    let model_data = generate_mock_model_data();
    println!("Original model size: {} bytes", model_data.len());

    // 1. Basic encryption/decryption
    println!("\n--- Basic Encryption ---");
    demo_basic_encryption(&model_data)?;

    // 2. Custom configuration
    println!("\n--- Custom Configuration ---");
    demo_custom_config(&model_data)?;

    // 3. File format inspection
    println!("\n--- Format Inspection ---");
    demo_format_inspection(&model_data)?;

    // 4. Error handling
    println!("\n--- Error Handling ---");
    demo_error_handling(&model_data)?;

    println!("\n=== Demo Complete ===");
    Ok(())
}

fn generate_mock_model_data() -> Vec<u8> {
    // Simulate a small model with header and weights
    let mut data = Vec::new();

    // GGUF-like header
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&[3, 0, 0, 0]); // Version
    data.extend_from_slice(&[0; 8]); // Tensor count
    data.extend_from_slice(&[0; 8]); // KV count

    // Simulated weights (random-ish data)
    for i in 0..1000 {
        data.push((i % 256) as u8);
    }

    data
}

fn demo_basic_encryption(model_data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let password = "my-secure-password-2024";

    // Encrypt
    println!("Encrypting with password...");
    let encrypted = encrypt_model(model_data, password)?;
    println!("Encrypted size: {} bytes", encrypted.len());
    println!("Overhead: {} bytes (header + auth tag)", encrypted.len() - model_data.len());

    // Verify it's encrypted
    assert!(is_encrypted(&encrypted));
    println!("File correctly identified as encrypted");

    // Decrypt
    println!("Decrypting...");
    let decrypted = decrypt_model(&encrypted, password)?;

    // Verify round-trip
    assert_eq!(model_data, &decrypted);
    println!("Round-trip successful: original == decrypted");

    Ok(())
}

fn demo_custom_config(model_data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let password = "custom-config-password";

    // High-security configuration
    let high_security = EncryptionConfig {
        memory_cost_kib: 131072, // 128 MB
        time_cost: 5,
        parallelism: 8,
    };

    println!("Using high-security config:");
    println!("  Memory: {} KiB", high_security.memory_cost_kib);
    println!("  Iterations: {}", high_security.time_cost);
    println!("  Parallelism: {}", high_security.parallelism);

    let encrypted = encrypt_model_with_config(model_data, password, &high_security)?;
    println!("Encrypted size: {} bytes", encrypted.len());

    // Fast configuration for development
    let fast_config = EncryptionConfig {
        memory_cost_kib: 16384, // 16 MB
        time_cost: 2,
        parallelism: 2,
    };

    println!("\nUsing fast config:");
    println!("  Memory: {} KiB", fast_config.memory_cost_kib);
    println!("  Iterations: {}", fast_config.time_cost);
    println!("  Parallelism: {}", fast_config.parallelism);

    let encrypted_fast = encrypt_model_with_config(model_data, password, &fast_config)?;
    println!("Encrypted size: {} bytes", encrypted_fast.len());

    Ok(())
}

fn demo_format_inspection(model_data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let password = "inspection-password";
    let encrypted = encrypt_model(model_data, password)?;

    // Check if data is encrypted
    println!("Is original encrypted? {}", is_encrypted(model_data));
    println!("Is encrypted file encrypted? {}", is_encrypted(&encrypted));

    // Get encryption version
    let version = get_version(&encrypted)?;
    println!("Encryption format version: {}", version);

    // Show header structure
    println!("\nEncrypted file structure:");
    println!("  Magic:   PACHAENC (8 bytes)");
    println!("  Version: {} (1 byte)", version);
    println!("  Salt:    32 bytes (key derivation)");
    println!("  Nonce:   12 bytes (encryption)");
    println!("  Data:    {} bytes (ciphertext)", model_data.len());
    println!("  Tag:     16 bytes (authentication)");
    println!("  Total:   {} bytes", encrypted.len());

    Ok(())
}

fn demo_error_handling(model_data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let password = "correct-password";
    let encrypted = encrypt_model(model_data, password)?;

    // Wrong password
    println!("Testing wrong password...");
    match decrypt_model(&encrypted, "wrong-password") {
        Ok(_) => println!("  Unexpectedly succeeded!"),
        Err(e) => println!("  Correctly failed: {}", e),
    }

    // Corrupted data
    println!("Testing corrupted data...");
    let mut corrupted = encrypted.clone();
    corrupted[60] ^= 0xFF; // Flip some bits
    match decrypt_model(&corrupted, password) {
        Ok(_) => println!("  Unexpectedly succeeded!"),
        Err(e) => println!("  Correctly failed: {}", e),
    }

    // Truncated data
    println!("Testing truncated data...");
    let truncated = &encrypted[..encrypted.len() - 20];
    match decrypt_model(truncated, password) {
        Ok(_) => println!("  Unexpectedly succeeded!"),
        Err(e) => println!("  Correctly failed: {}", e),
    }

    // Empty password
    println!("Testing empty password...");
    match encrypt_model(model_data, "") {
        Ok(_) => println!("  Unexpectedly succeeded!"),
        Err(e) => println!("  Correctly failed: {}", e),
    }

    println!("\nAll error cases handled correctly!");

    Ok(())
}
