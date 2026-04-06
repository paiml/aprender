#![allow(clippy::all, clippy::pedantic, clippy::nursery)]
//! TLS Example
//!
//! Demonstrates TLS-encrypted remote execution using self-signed certificates.
//!
//! # Setup
//!
//! 1. Generate test certificates:
//!    ```bash
//!    ./scripts/generate-test-certs.sh ./certs
//!    ```
//!
//! 2. Run the example:
//!    ```bash
//!    cargo run --example tls_example --features remote-tls
//!    ```

#[cfg(feature = "remote-tls")]
use repartir::executor::tls::TlsConfig;

fn main() -> repartir::error::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🔐 Repartir v1.1 - TLS Encrypted Remote Execution");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    #[cfg(feature = "remote-tls")]
    {
        println!("Loading TLS configuration...");
        println!();

        // Build TLS configuration
        let config = TlsConfig::builder()
            .client_cert("./certs/client.pem")
            .client_key("./certs/client.key")
            .server_cert("./certs/server.pem")
            .server_key("./certs/server.key")
            .ca_cert("./certs/ca.pem")
            .build()?;

        println!("✅ TLS Configuration loaded successfully!");
        println!();
        println!("Client configuration:");
        match config.client_config() {
            Ok(_) => println!("  ✓ Client TLS enabled"),
            Err(_) => println!("  ✗ Client TLS not configured"),
        }

        println!();
        println!("Server configuration:");
        match config.server_config() {
            Ok(_) => println!("  ✓ Server TLS enabled"),
            Err(_) => println!("  ✗ Server TLS not configured"),
        }

        println!();
        println!("🔒 TLS Security Features:");
        println!("  • End-to-end encryption (TLS 1.3)");
        println!("  • Certificate-based authentication");
        println!("  • Perfect forward secrecy");
        println!("  • Protection against MITM attacks");
        println!();
        println!("⚠️  Note: These are self-signed certificates for TESTING ONLY");
        println!("   For production, use certificates from a trusted CA");
        println!();
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }

    #[cfg(not(feature = "remote-tls"))]
    {
        println!("❌ TLS feature not enabled");
        println!();
        println!("To enable TLS support, rebuild with:");
        println!("  cargo run --example tls_example --features remote-tls");
        println!();
        println!("Or add to your Cargo.toml:");
        println!("  repartir = {{ version = \"0.1\", features = [\"remote-tls\"] }}");
    }

    Ok(())
}
