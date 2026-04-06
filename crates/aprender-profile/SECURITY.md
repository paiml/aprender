# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.9.x   | :white_check_mark: |
| < 0.9   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in renacer, please report it
responsibly by emailing **info@paiml.com**.

Do **not** open a public GitHub issue for security vulnerabilities.

We will acknowledge receipt within 48 hours and provide an initial
assessment within 5 business days.

## Security Measures

- `unsafe_code` is denied at workspace level via `[workspace.lints.rust]`
- All unsafe blocks require `#[allow(unsafe_code)]` with SAFETY comments
- Dependencies audited weekly via `cargo-audit` and `cargo-deny`
- Memory-mapped I/O operations are read-only where possible
- ptrace operations are sandboxed to traced child processes only
