//! TLS configuration for secure remote execution.
//!
//! Provides TLS/SSL encryption for remote executor connections using rustls.

use crate::error::{RepartirError, Result};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use tokio_rustls::rustls;
use tracing::{debug, info};

/// TLS configuration for remote executor.
///
/// Provides both client and server TLS configurations using rustls.
/// Supports certificate-based authentication for mutual TLS (mTLS).
#[derive(Clone)]
pub struct TlsConfig {
    /// Client configuration for connecting to remote workers.
    client_config: Option<Arc<rustls::ClientConfig>>,
    /// Server configuration for accepting connections.
    server_config: Option<Arc<rustls::ServerConfig>>,
}

impl TlsConfig {
    /// Creates a new TLS configuration builder.
    #[must_use]
    pub const fn builder() -> TlsConfigBuilder {
        TlsConfigBuilder {
            client_cert_path: None,
            client_key_path: None,
            server_cert_path: None,
            server_key_path: None,
            ca_cert_path: None,
        }
    }

    /// Returns the client configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if client TLS is not configured.
    pub fn client_config(&self) -> Result<Arc<rustls::ClientConfig>> {
        self.client_config.clone().ok_or_else(|| RepartirError::InvalidTask {
            reason: "Client TLS not configured".to_string(),
        })
    }

    /// Returns the server configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if server TLS is not configured.
    pub fn server_config(&self) -> Result<Arc<rustls::ServerConfig>> {
        self.server_config.clone().ok_or_else(|| RepartirError::InvalidTask {
            reason: "Server TLS not configured".to_string(),
        })
    }
}

/// Builder for TLS configuration.
#[allow(clippy::struct_field_names)]
pub struct TlsConfigBuilder {
    client_cert_path: Option<String>,
    client_key_path: Option<String>,
    server_cert_path: Option<String>,
    server_key_path: Option<String>,
    ca_cert_path: Option<String>,
}

impl TlsConfigBuilder {
    /// Sets the client certificate path.
    #[must_use]
    pub fn client_cert<S: Into<String>>(mut self, path: S) -> Self {
        self.client_cert_path = Some(path.into());
        self
    }

    /// Sets the client private key path.
    #[must_use]
    pub fn client_key<S: Into<String>>(mut self, path: S) -> Self {
        self.client_key_path = Some(path.into());
        self
    }

    /// Sets the server certificate path.
    #[must_use]
    pub fn server_cert<S: Into<String>>(mut self, path: S) -> Self {
        self.server_cert_path = Some(path.into());
        self
    }

    /// Sets the server private key path.
    #[must_use]
    pub fn server_key<S: Into<String>>(mut self, path: S) -> Self {
        self.server_key_path = Some(path.into());
        self
    }

    /// Sets the CA certificate path for verification.
    #[must_use]
    pub fn ca_cert<S: Into<String>>(mut self, path: S) -> Self {
        self.ca_cert_path = Some(path.into());
        self
    }

    /// Builds the TLS configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Certificate or key files cannot be read
    /// - Certificate or key parsing fails
    /// - TLS configuration is invalid
    pub fn build(self) -> Result<TlsConfig> {
        let mut client_config = None;
        let mut server_config = None;

        // Build client configuration if cert/key provided
        if let (Some(cert_path), Some(key_path)) = (&self.client_cert_path, &self.client_key_path) {
            debug!("Loading client TLS certificate from {cert_path}");
            let certs = load_certs(cert_path)?;
            let key = load_private_key(key_path)?;

            // Build config with or without CA certificate
            let config = if let Some(ca_path) = &self.ca_cert_path {
                debug!("Loading CA certificate from {ca_path}");
                let ca_certs = load_certs(ca_path)?;
                let mut root_store = rustls::RootCertStore::empty();
                for cert in ca_certs {
                    root_store.add(cert).map_err(|e| RepartirError::InvalidTask {
                        reason: format!("Failed to add CA certificate: {e}"),
                    })?;
                }
                rustls::ClientConfig::builder()
                    .with_root_certificates(root_store)
                    .with_client_auth_cert(certs, key)
                    .map_err(|e| RepartirError::InvalidTask {
                        reason: format!("Failed to build client TLS config with CA: {e}"),
                    })?
            } else {
                // No CA cert - disable verification (development only!)
                let mut config = rustls::ClientConfig::builder()
                    .with_root_certificates(rustls::RootCertStore::empty())
                    .with_client_auth_cert(certs, key)
                    .map_err(|e| RepartirError::InvalidTask {
                        reason: format!("Failed to build client TLS config: {e}"),
                    })?;
                config.dangerous().set_certificate_verifier(Arc::new(NoCertificateVerification));
                config
            };

            info!("Client TLS configured");
            client_config = Some(Arc::new(config));
        }

        // Build server configuration if cert/key provided
        if let (Some(cert_path), Some(key_path)) = (&self.server_cert_path, &self.server_key_path) {
            debug!("Loading server TLS certificate from {cert_path}");
            let certs = load_certs(cert_path)?;
            let key = load_private_key(key_path)?;

            let config = rustls::ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(certs, key)
                .map_err(|e| RepartirError::InvalidTask {
                    reason: format!("Failed to build server TLS config: {e}"),
                })?;

            info!("Server TLS configured");
            server_config = Some(Arc::new(config));
        }

        Ok(TlsConfig { client_config, server_config })
    }
}

/// Loads certificates from a PEM file.
fn load_certs<P: AsRef<Path>>(path: P) -> Result<Vec<CertificateDer<'static>>> {
    let file = File::open(path.as_ref()).map_err(RepartirError::Io)?;
    let mut reader = BufReader::new(file);

    let certs: Vec<CertificateDer<'static>> =
        rustls_pemfile::certs(&mut reader).collect::<std::result::Result<Vec<_>, _>>().map_err(
            |e| RepartirError::InvalidTask { reason: format!("Failed to parse certificates: {e}") },
        )?;

    Ok(certs)
}

/// Loads a private key from a PEM file.
fn load_private_key<P: AsRef<Path>>(path: P) -> Result<PrivateKeyDer<'static>> {
    let file = File::open(path.as_ref()).map_err(RepartirError::Io)?;
    let mut reader = BufReader::new(file);

    // Try reading as PKCS#8 first, then RSA, then EC
    let key = rustls_pemfile::private_key(&mut reader)
        .map_err(|e| RepartirError::InvalidTask {
            reason: format!("Failed to parse private key: {e}"),
        })?
        .ok_or_else(|| RepartirError::InvalidTask {
            reason: "No private key found in file".to_string(),
        })?;

    Ok(key)
}

/// Certificate verifier that accepts all certificates (INSECURE - development only).
#[derive(Debug)]
struct NoCertificateVerification;

impl rustls::client::danger::ServerCertVerifier for NoCertificateVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::panic
)]
mod tests {
    use super::*;
    use rustls::client::danger::ServerCertVerifier;
    use std::io::Write;

    // Test certificate and key (minimal self-signed cert for testing)
    const TEST_CERT_PEM: &str = r"-----BEGIN CERTIFICATE-----
MIIBkTCB+wIJAKHHCgVZU0HHMA0GCSqGSIb3DQEBCwUAMBExDzANBgNVBAMMBnRl
c3RDQTAeFw0yNDAxMDEwMDAwMDBaFw0yNTAxMDEwMDAwMDBaMBExDzANBgNVBAMM
BnRlc3RDQTCBnzANBgkqhkiG9w0BAQEFAAOBjQAwgYkCgYEA2dQ7A+PqJqZ5Bnxm
Hn3Zp0oqL4nGqH3lxLKp5Qq+Wm1K8j9YKpZqWGxKpQqFl3mXqH5nGlKpQqZWGxKp
QqFl3mXqH5nGlKpQqZWGxKpQqFl3mXqH5nGlKpQqZWGxKpQqFl3mXqH5nGlKpQqZ
WGxKpQqFl3mXqH5nGlKpQqECAwEAATANBgkqhkiG9w0BAQsFAAOBgQB1nWBqH3mX
qH5nGlKpQqZWGxKpQqFl3mXqH5nGlKpQqZWGxKpQqFl3mXqH5nGlKpQqZWGxKpQq
Fl3mXqH5nGlKpQqZWGxKpQqFl3mXqH5nGlKpQqZWGxKpQqFl3mXqH5nGlKpQqZWG
xKpQqFl3mXqH5nGlKpQqZWGxKpQqFl3mXqH5nGlKpQqZWGxKpQqFg==
-----END CERTIFICATE-----";

    const TEST_KEY_PEM: &str = r"-----BEGIN PRIVATE KEY-----
MIICdwIBADANBgkqhkiG9w0BAQEFAASCAmEwggJdAgEAAoGBANnUOwPj6iameQZ8
Zh592adKKi+Jxqh95cSyqeUKvlptSvI/WCqWalhsSqUKhZd5l6h+ZxpSqUKmVhsS
qUKhZd5l6h+ZxpSqUKmVhsSqUKhZd5l6h+ZxpSqUKmVhsSqUKhZd5l6h+ZxpSqUK
mVhsSqUKhZd5l6h+ZxpSqUKhAgMBAAECgYAqH3mXqH5nGlKpQqZWGxKpQqFl3mXq
H5nGlKpQqZWGxKpQqFl3mXqH5nGlKpQqZWGxKpQqFl3mXqH5nGlKpQqZWGxKpQqF
l3mXqH5nGlKpQqZWGxKpQqFl3mXqH5nGlKpQqZWGxKpQqFl3mXqH5nGlKpQqZWGx
KpQqFl3mXqH5nGlKpQqZQJBAP8xL6h+ZxpSqUKmVhsSqUKhZd5l6h+ZxpSqUKmVh
sSqUKhZd5l6h+ZxpSqUKmVhsSqUKhZd5l6h+ZxpSqUKmVhsCQQDaVDsD4+ompnkG
fGYefdmnSiovic aofqH5nGlKpQqZWGxKpQqFl3mXqH5nGlKpQqZWGxKpQqFl3mXq
H5nGlKpQqZWGxAkEA2lQ7A+PqJqZ5BnxmHn3Zp0oqL4nGqH6h+ZxpSqUKmVhsSqU
KhZd5l6h+ZxpSqUKmVhsSqUKhZd5l6h+ZxpSqUKmVhsQJANpUOwPj6iameQZ8Zh5
92adKKi+Jxqh+ZxpSqUKmVhsSqUKhZd5l6h+ZxpSqUKmVhsSqUKhZd5l6h+ZxpSq
UKmVhsQJBANpUOwPj6iameQZ8Zh592adKKi+Jxqh+ZxpSqUKmVhsSqUKhZd5l6h+
ZxpSqUKmVhsSqUKhZd5l6h+ZxpSqUKmVhsQ=
-----END PRIVATE KEY-----";

    fn create_temp_file(content: &str) -> std::path::PathBuf {
        let mut temp_file = std::env::temp_dir();
        let timestamp =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        temp_file.push(format!("repartir_test_{}_{}.pem", std::process::id(), timestamp));

        let mut file = File::create(&temp_file).unwrap();
        file.write_all(content.as_bytes()).unwrap();

        temp_file
    }

    #[test]
    fn test_tls_config_builder() {
        let _config = TlsConfig::builder()
            .client_cert("client.pem")
            .client_key("client.key")
            .server_cert("server.pem")
            .server_key("server.key")
            .ca_cert("ca.pem");

        // Builder should compile and be chainable (compilation is the test)
    }

    #[test]
    fn test_empty_tls_config() {
        let config = TlsConfig::builder().build();

        // Should succeed even with no certs (they're optional)
        assert!(config.is_ok());

        let cfg = config.unwrap();

        // But trying to use configs should fail
        assert!(cfg.client_config().is_err());
        assert!(cfg.server_config().is_err());
    }

    #[test]
    fn test_load_certs_missing_file() {
        let result = load_certs("nonexistent.pem");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_private_key_missing_file() {
        let result = load_private_key("nonexistent.key");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_certs_empty_file() {
        let temp_file = create_temp_file("");
        let result = load_certs(&temp_file);
        std::fs::remove_file(temp_file).ok();
        // Empty file should succeed but return empty vec (rustls behavior)
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_private_key_no_key() {
        let temp_file = create_temp_file("");
        let result = load_private_key(&temp_file);
        std::fs::remove_file(temp_file).ok();
        // Empty file with no key should fail
        assert!(result.is_err());
    }

    #[test]
    fn test_tls_config_missing_client_cert() {
        // Missing certificate file should fail
        let config = TlsConfig::builder()
            .client_cert("nonexistent_cert.pem")
            .client_key("nonexistent_key.pem")
            .build();

        assert!(config.is_err());
    }

    #[test]
    fn test_tls_config_missing_server_cert() {
        // Missing certificate file should fail
        let config = TlsConfig::builder()
            .server_cert("nonexistent_cert.pem")
            .server_key("nonexistent_key.pem")
            .build();

        assert!(config.is_err());
    }

    #[test]
    fn test_tls_config_partial_client_cert_only() {
        // Only cert without key - build should skip (no error, no config)
        let cert_file = create_temp_file(TEST_CERT_PEM);

        let config = TlsConfig::builder().client_cert(cert_file.to_str().unwrap()).build();

        std::fs::remove_file(cert_file).ok();

        // Should succeed (partial config is optional)
        assert!(config.is_ok());
        let cfg = config.unwrap();
        assert!(cfg.client_config().is_err()); // No key, so no client config
    }

    #[test]
    fn test_tls_config_partial_server_key_only() {
        // Only key without cert - build should skip
        let key_file = create_temp_file(TEST_KEY_PEM);

        let config = TlsConfig::builder().server_key(key_file.to_str().unwrap()).build();

        std::fs::remove_file(key_file).ok();

        // Should succeed (partial config is optional)
        assert!(config.is_ok());
        let cfg = config.unwrap();
        assert!(cfg.server_config().is_err()); // No cert, so no server config
    }

    #[test]
    fn test_tls_config_clone() {
        let config = TlsConfig::builder().build().unwrap();
        let cloned = config.clone();
        // Use both values to satisfy clippy
        let _ = (config, cloned);
    }

    #[test]
    fn test_no_certificate_verification() {
        let verifier = NoCertificateVerification;

        // Test supported schemes
        let schemes = verifier.supported_verify_schemes();
        assert!(!schemes.is_empty());
        assert!(schemes.contains(&rustls::SignatureScheme::RSA_PKCS1_SHA256));
        assert!(schemes.contains(&rustls::SignatureScheme::ECDSA_NISTP256_SHA256));
        assert!(schemes.contains(&rustls::SignatureScheme::ED25519));
    }

    #[test]
    fn test_builder_partial_client() {
        // Only client cert, no key - should succeed (build() validates)
        let _builder = TlsConfig::builder().client_cert("cert.pem");
    }

    #[test]
    fn test_builder_partial_server() {
        // Only server key, no cert - should succeed (build() validates)
        let _builder = TlsConfig::builder().server_key("key.pem");
    }
}
