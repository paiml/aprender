# Project Governance

This document describes the governance structure for the simular project.

## Maintainers

The project is maintained by the PAIML team. Maintainers are listed in `.github/CODEOWNERS`.

### Responsibilities

- Review and merge pull requests
- Triage issues
- Make release decisions
- Enforce code quality standards

## Decision Making

### Technical Decisions

Technical decisions are documented in Architecture Decision Records (ADRs) in `docs/adr/`.

Process:
1. Create ADR draft
2. Discuss in GitHub issue
3. Review by maintainers
4. Accept or reject with documented rationale

### Breaking Changes

Breaking changes require:
1. ADR documenting the change
2. Deprecation period of at least one minor version
3. Migration guide in CHANGELOG
4. Approval from at least one maintainer

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

### Pull Request Process

1. Fork and create feature branch
2. Write tests (95% coverage required)
3. Update documentation
4. Submit PR with descriptive title
5. Address review feedback
6. Maintainer merges when approved

## Code of Conduct

All participants must follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Release Process

1. Update CHANGELOG.md
2. Bump version in Cargo.toml
3. Create git tag (e.g., v1.0.0)
4. GitHub Actions publishes to crates.io
5. Announce release

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
