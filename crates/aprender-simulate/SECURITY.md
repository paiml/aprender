# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it via GitHub Security Advisories:

1. Go to https://github.com/paiml/simular/security/advisories
2. Click "New draft security advisory"
3. Provide details about the vulnerability

We will respond within 48 hours and work with you to understand and address the issue.

## Security Practices

- No `unsafe` code in simulation-critical paths
- All dependencies audited with `cargo audit`
- WASM sandboxing for browser execution
- No network access in core library
- Deterministic RNG prevents timing attacks
