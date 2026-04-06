# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email security@paiml.com with details
3. Include steps to reproduce if possible

We will acknowledge receipt within 48 hours and provide a timeline for a fix.

## Security Measures

- `#![deny(unsafe_code)]` enforced project-wide
- `cargo-deny` for dependency license and advisory checks
- `cargo-audit` in CI pipeline
- BLAKE3 content-addressed storage for integrity verification
- Optional Ed25519 signing for artifact authenticity
- Optional ChaCha20-Poly1305 encryption at rest
