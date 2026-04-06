# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.2.x   | Yes                |
| < 0.2   | No                 |

## Reporting a Vulnerability

If you discover a security vulnerability in alimentar, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email security concerns to the maintainers
3. Include a description of the vulnerability and steps to reproduce

We will acknowledge receipt within 48 hours and provide a timeline for a fix.

## Security Measures

- `cargo audit` runs in CI on every push and weekly
- `cargo deny` enforces license compliance and bans known-vulnerable crates
- `unsafe` code is denied at the crate level (`unsafe_code = "deny"`)
- All dependencies are sourced from crates.io (no git dependencies)
- Format encryption uses `aes-gcm` with `argon2` key derivation
- Format signing uses `ed25519-dalek`

## Dependencies

We use `cargo-deny` to enforce:
- No yanked crates
- License compliance (MIT/Apache-2.0/BSD compatible)
- No wildcard dependencies
- All crates from crates.io registry
