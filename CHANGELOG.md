# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-11-28

### Added

- Initial release of Pacha model/data/recipe registry
- **Model Registry**: Semantic versioning, Model Cards (Mitchell et al. 2019), lifecycle stages
- **Data Registry**: Dataset versioning, Datasheets (Gebru et al. 2021), W3C PROV-DM provenance
- **Recipe Registry**: Training recipes in TOML, hyperparameter specifications
- **Content-Addressed Storage**: BLAKE3 hashing, automatic deduplication
- **Lineage Tracking**: Fine-tuning, distillation, quantization, pruning, merging
- **Experiment Tracking**: Run status, metrics logging, hardware info
- **CLI**: `pacha` binary with model/data/recipe/run commands
- **SQLite Metadata**: Persistent registry database

### Features

- `compression` - Zstd compression for stored artifacts (default)
- `cli` - Command-line interface (default)
- `encryption` - AES-256-GCM encryption for artifacts
- `lineage-graph` - trueno-graph integration for lineage visualization

[Unreleased]: https://github.com/paiml/pacha/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/paiml/pacha/releases/tag/v0.1.0
