//! Pacha crypto command implementations
//!
//! This module contains cryptographic operations extracted from pacha/mod.rs:
//! keygen, sign, verify, encrypt, decrypt

use crate::ansi_colors::Colorize;
use std::io::{self, Write};

// ============================================================================
// PACHA-CLI-014: Keygen Command
// ============================================================================

pub fn cmd_keygen(output: Option<&str>, identity: Option<&str>, force: bool) -> anyhow::Result<()> {
    use pacha::signing::{Keyring, SigningKey};

    println!("{}", "🔑 Generate Signing Key".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    // Determine output paths
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let pacha_dir = format!("{home}/.pacha");
    let default_key_path = format!("{pacha_dir}/signing-key.pem");
    let key_path = output.unwrap_or(&default_key_path);
    let public_path = format!("{key_path}.pub");
    let keyring_path = format!("{pacha_dir}/keyring.json");

    // Check if key exists
    if std::path::Path::new(key_path).exists() && !force {
        println!("{} Key already exists at {}", "⚠".yellow(), key_path.cyan());
        println!("Use {} to overwrite", "--force".yellow());
        return Ok(());
    }

    // Create directory if needed
    std::fs::create_dir_all(&pacha_dir)?;

    // Generate key pair
    println!("Generating Ed25519 key pair...");
    let signing_key = SigningKey::generate();
    let verifying_key = signing_key.verifying_key();

    // Save private key
    std::fs::write(key_path, signing_key.to_pem())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(key_path, std::fs::Permissions::from_mode(0o600))?;
    }

    // Save public key
    std::fs::write(&public_path, verifying_key.to_pem())?;

    // Add to keyring if identity provided
    if let Some(id) = identity {
        let mut keyring = Keyring::load(&keyring_path).unwrap_or_default();
        keyring.add(id, &verifying_key);
        keyring.set_default(id);
        keyring.save(&keyring_path)?;
        println!("Identity:    {}", id.cyan());
    }

    println!();
    println!("{} Key pair generated:", "✓".bright_green().bold());
    println!("  Private:  {}", key_path.cyan());
    println!("  Public:   {}", public_path.cyan());
    println!();
    println!("Public key (hex):");
    println!("  {}", verifying_key.to_hex().dimmed());
    println!();
    println!("{} Keep your private key secure! Anyone with it can sign models.", "⚠".yellow());

    Ok(())
}

// ============================================================================
// PACHA-CLI-015: Sign Command
// ============================================================================

pub fn cmd_sign(
    model: &str,
    key_path: Option<&str>,
    output: Option<&str>,
    identity: Option<&str>,
) -> anyhow::Result<()> {
    use pacha::signing::{sign_model_with_id, SigningKey};

    println!("{}", "✍️  Sign Model".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    // Determine key path
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let default_key_path = format!("{home}/.pacha/signing-key.pem");
    let key_file = key_path.unwrap_or(&default_key_path);

    // Check key exists
    if !std::path::Path::new(key_file).exists() {
        println!("{} Signing key not found at {}", "✗".red(), key_file.cyan());
        println!("Run {} first", "batuta pacha keygen".cyan());
        return Ok(());
    }

    // Load signing key
    println!("Loading signing key...");
    let key_pem = std::fs::read_to_string(key_file)?;
    let signing_key =
        SigningKey::from_pem(&key_pem).map_err(|e| anyhow::anyhow!("Failed to load key: {e}"))?;

    // Determine model path
    let model_path = if std::path::Path::new(model).exists() {
        model.to_string()
    } else {
        // Try to resolve from cache
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{home}/.pacha/models/{model}")
    };

    if !std::path::Path::new(&model_path).exists() {
        println!("{} Model not found: {}", "✗".red(), model.cyan());
        return Ok(());
    }

    // Load model data
    println!("Loading model: {}", model_path.cyan());
    let model_data = std::fs::read(&model_path)?;
    let size_mb = model_data.len() as f64 / (1024.0 * 1024.0);
    println!("Model size:   {:.1} MB", size_mb);

    // Sign model
    println!("Signing...");
    let signature = sign_model_with_id(&model_data, &signing_key, identity.map(String::from))
        .map_err(|e| anyhow::anyhow!("Failed to sign: {e}"))?;

    // Determine output path
    let sig_path = output.map(String::from).unwrap_or_else(|| format!("{model_path}.sig"));

    // Save signature
    signature.save(&sig_path).map_err(|e| anyhow::anyhow!("Failed to save signature: {e}"))?;

    println!();
    println!("{} Model signed successfully:", "✓".bright_green().bold());
    println!("  Signature: {}", sig_path.cyan());
    println!(
        "  Hash:      {}",
        signature.content_hash.get(..16).unwrap_or(&signature.content_hash).dimmed()
    );
    println!(
        "  Signer:    {}",
        signature.signer_key.get(..16).unwrap_or(&signature.signer_key).dimmed()
    );
    if let Some(id) = &signature.signer_id {
        println!("  Identity:  {}", id.as_str().cyan());
    }

    Ok(())
}

// ============================================================================
// PACHA-CLI-016: Verify Command
// ============================================================================

pub fn cmd_verify(
    model: &str,
    signature_path: Option<&str>,
    expected_key: Option<&str>,
) -> anyhow::Result<()> {
    use pacha::signing::{verify_model, verify_model_with_key, ModelSignature, VerifyingKey};

    println!("{}", "🔍 Verify Model Signature".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    // Determine model path
    let model_path = if std::path::Path::new(model).exists() {
        model.to_string()
    } else {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{home}/.pacha/models/{model}")
    };

    if !std::path::Path::new(&model_path).exists() {
        println!("{} Model not found: {}", "✗".red(), model.cyan());
        return Ok(());
    }

    // Determine signature path
    let sig_path = signature_path.map(String::from).unwrap_or_else(|| format!("{model_path}.sig"));

    if !std::path::Path::new(&sig_path).exists() {
        println!("{} Signature not found: {}", "✗".red(), sig_path.cyan());
        return Ok(());
    }

    println!("Model:     {}", model_path.cyan());
    println!("Signature: {}", sig_path.cyan());
    println!();

    // Load model and signature
    println!("Loading model...");
    let model_data = std::fs::read(&model_path)?;

    println!("Loading signature...");
    let signature = ModelSignature::load(&sig_path)
        .map_err(|e| anyhow::anyhow!("Failed to load signature: {e}"))?;

    println!();
    println!("Signature details:");
    println!("  Algorithm: {}", signature.algorithm.cyan());
    println!(
        "  Hash:      {}",
        signature.content_hash.get(..16).unwrap_or(&signature.content_hash).dimmed()
    );
    println!(
        "  Signer:    {}",
        signature.signer_key.get(..16).unwrap_or(&signature.signer_key).dimmed()
    );
    if let Some(id) = &signature.signer_id {
        println!("  Identity:  {}", id.as_str().cyan());
    }
    println!();

    // Verify
    println!("Verifying...");
    let result = if let Some(key_hex) = expected_key {
        let expected =
            VerifyingKey::from_hex(key_hex).map_err(|e| anyhow::anyhow!("Invalid key: {e}"))?;
        verify_model_with_key(&model_data, &signature, &expected)
    } else {
        verify_model(&model_data, &signature)
    };

    match result {
        Ok(()) => {
            println!();
            println!(
                "{} Signature is {}",
                "✓".bright_green().bold(),
                "VALID".bright_green().bold()
            );
            if expected_key.is_some() {
                println!("  Signed by expected key");
            }
        }
        Err(e) => {
            println!();
            println!("{} Signature is {} - {}", "✗".red().bold(), "INVALID".red().bold(), e);
            return Err(anyhow::anyhow!("Signature verification failed"));
        }
    }

    Ok(())
}

// ============================================================================
// PACHA-CLI-017: Encrypt Command
// ============================================================================

pub fn cmd_encrypt(
    model: &str,
    output: Option<&str>,
    password_env: Option<&str>,
) -> anyhow::Result<()> {
    use pacha::crypto::{encrypt_model, is_encrypted};

    println!("{}", "🔐 Encrypt Model".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    // Determine model path
    let model_path = if std::path::Path::new(model).exists() {
        model.to_string()
    } else {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{home}/.pacha/models/{model}")
    };

    if !std::path::Path::new(&model_path).exists() {
        println!("{} Model not found: {}", "✗".red(), model.cyan());
        return Ok(());
    }

    // Determine output path
    let output_path = output.map(String::from).unwrap_or_else(|| format!("{model_path}.enc"));

    println!("Model:  {}", model_path.cyan());
    println!("Output: {}", output_path.cyan());
    println!();

    // Get password
    let password = if let Some(env_var) = password_env {
        std::env::var(env_var)
            .map_err(|_| anyhow::anyhow!("Environment variable {} not set", env_var))?
    } else {
        // Prompt for password
        print!("Enter encryption password: ");
        io::stdout().flush()?;
        let mut password = String::new();
        io::stdin().read_line(&mut password)?;
        password.trim().to_string()
    };

    if password.is_empty() {
        println!("{} Password cannot be empty", "✗".red());
        return Err(anyhow::anyhow!("Empty password"));
    }

    // Load model
    println!("Loading model...");
    let model_data = std::fs::read(&model_path)?;

    // Check if already encrypted
    if is_encrypted(&model_data) {
        println!("{} Model is already encrypted", "⚠".yellow());
        return Ok(());
    }

    let size_mb = model_data.len() as f64 / (1024.0 * 1024.0);
    println!("Model size: {:.2} MB", size_mb);

    // Encrypt
    println!("Encrypting...");
    let encrypted = encrypt_model(&model_data, &password)
        .map_err(|e| anyhow::anyhow!("Encryption failed: {e}"))?;

    // Write output
    std::fs::write(&output_path, &encrypted)?;

    let encrypted_mb = encrypted.len() as f64 / (1024.0 * 1024.0);
    println!();
    println!("{} Model encrypted successfully", "✓".bright_green().bold());
    println!("  Output: {}", output_path.cyan());
    println!("  Size:   {:.2} MB", encrypted_mb);
    println!();
    println!("{}", "To decrypt, run:".dimmed());
    println!("  batuta pacha decrypt {} --password-env MODEL_KEY", output_path);

    Ok(())
}

// ============================================================================
// PACHA-CLI-018: Decrypt Command
// ============================================================================

pub fn cmd_decrypt(
    file: &str,
    output: Option<&str>,
    password_env: Option<&str>,
) -> anyhow::Result<()> {
    use pacha::crypto::{decrypt_model, is_encrypted};

    println!("{}", "🔓 Decrypt Model".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    if !std::path::Path::new(file).exists() {
        println!("{} File not found: {}", "✗".red(), file.cyan());
        return Ok(());
    }

    // Determine output path
    let output_path = output.map(String::from).unwrap_or_else(|| {
        if let Some(stripped) = file.strip_suffix(".enc") {
            stripped.to_string()
        } else {
            format!("{file}.decrypted")
        }
    });

    println!("Input:  {}", file.cyan());
    println!("Output: {}", output_path.cyan());
    println!();

    // Load encrypted file
    println!("Loading encrypted file...");
    let encrypted_data = std::fs::read(file)?;

    // Verify it's encrypted
    if !is_encrypted(&encrypted_data) {
        println!("{} File does not appear to be encrypted", "✗".red());
        return Err(anyhow::anyhow!("Not an encrypted file"));
    }

    let size_mb = encrypted_data.len() as f64 / (1024.0 * 1024.0);
    println!("Encrypted size: {:.2} MB", size_mb);

    // Get password
    let password = if let Some(env_var) = password_env {
        std::env::var(env_var)
            .map_err(|_| anyhow::anyhow!("Environment variable {} not set", env_var))?
    } else {
        // Prompt for password
        print!("Enter decryption password: ");
        io::stdout().flush()?;
        let mut password = String::new();
        io::stdin().read_line(&mut password)?;
        password.trim().to_string()
    };

    if password.is_empty() {
        println!("{} Password cannot be empty", "✗".red());
        return Err(anyhow::anyhow!("Empty password"));
    }

    // Decrypt
    println!("Decrypting...");
    let decrypted = decrypt_model(&encrypted_data, &password)
        .map_err(|e| anyhow::anyhow!("Decryption failed: {e}"))?;

    // Write output
    std::fs::write(&output_path, &decrypted)?;

    let decrypted_mb = decrypted.len() as f64 / (1024.0 * 1024.0);
    println!();
    println!("{} Model decrypted successfully", "✓".bright_green().bold());
    println!("  Output: {}", output_path.cyan());
    println!("  Size:   {:.2} MB", decrypted_mb);

    Ok(())
}
