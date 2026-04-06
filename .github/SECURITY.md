# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please report security vulnerabilities by emailing security@paiml.com.

Do NOT create public GitHub issues for security vulnerabilities.

### What to Include

1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

### Response Timeline

- Initial response: Within 48 hours
- Status update: Within 7 days
- Fix deployment: Within 30 days (critical) or 90 days (other)

## Security Practices

### Code Review

- All changes require code review via pull request
- Security-sensitive changes require security team review
- CODEOWNERS file enforces required reviewers

### Dependencies

- Dependencies audited with `cargo audit`
- Automated dependency updates via Dependabot
- No known vulnerabilities in dependency tree

### Testing

- Security-focused tests in `tests/security/`
- Fuzz testing for parsers
- Input validation tests

## Responsible Disclosure

We follow responsible disclosure practices and will:
- Acknowledge receipt of vulnerability reports
- Provide regular updates on remediation progress
- Credit reporters (unless they prefer anonymity)
- Not take legal action against good-faith security research
