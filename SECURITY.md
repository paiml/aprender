# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.29.x  | Yes       |
| < 0.29  | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in aprender, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email: **security@paiml.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Impact assessment (e.g., arbitrary code execution, data leak, denial of service)
- Affected versions

## Response Timeline

- **Acknowledgment**: within 48 hours
- **Initial assessment**: within 7 days
- **Fix or mitigation**: within 30 days for critical issues

## Scope

The following are in scope for security reports:

- Arbitrary code execution via model loading (GGUF, SafeTensors, APR formats)
- Memory safety violations (despite `unsafe_code = "forbid"` lint)
- Denial of service via crafted model files (OOM, infinite loops)
- Path traversal in `apr pull`, `apr import`, or `apr export`
- Information disclosure via `apr serve` API
- Supply chain issues in dependencies (report via `cargo deny`)

## Security Measures

- `unsafe_code = "forbid"` enforced workspace-wide via `[workspace.lints.rust]`
- `cargo deny check advisories` in CI — blocks known CVEs
- `cargo deny check licenses` — MIT-only dependency policy
- 540 provable contracts with falsification tests
- Fuzz targets in `fuzz/` directory
