# Alimentar - Data Loading Library

[Introduction](./introduction.md)

# Getting Started

- [Installation](./getting-started/installation.md)
- [Quick Start](./getting-started/quick-start.md)
- [Core Concepts](./getting-started/core-concepts.md)

# Architecture

- [Overview](./architecture/overview.md)
- [Design Principles](./architecture/design-principles.md)
- [Module Structure](./architecture/module-structure.md)

# Dataset API

- [ArrowDataset](./dataset/arrow-dataset.md)
- [Loading Data](./dataset/loading-data.md)
  - [CSV Files](./dataset/csv-files.md)
  - [JSON/JSONL Files](./dataset/json-files.md)
  - [Parquet Files](./dataset/parquet-files.md)
- [Dataset Operations](./dataset/operations.md)
- [Streaming Datasets](./dataset/streaming.md)

# Canonical Datasets

- [Overview](./datasets/overview.md)
- [Iris](./datasets/iris.md)
- [MNIST](./datasets/mnist.md)
- [Fashion-MNIST](./datasets/fashion-mnist.md)
- [CIFAR-10](./datasets/cifar10.md)
- [CIFAR-100](./datasets/cifar100.md)

# DataLoader

- [Overview](./dataloader/overview.md)
- [Batching](./dataloader/batching.md)
- [Shuffling](./dataloader/shuffling.md)
- [Drop Last](./dataloader/drop-last.md)
- [Iteration Patterns](./dataloader/iteration-patterns.md)

# Transforms

- [Transform Trait](./transforms/transform-trait.md)
- [Built-in Transforms](./transforms/built-in.md)
  - [Select](./transforms/select.md)
  - [Drop](./transforms/drop.md)
  - [Rename](./transforms/rename.md)
  - [Filter](./transforms/filter.md)
  - [Map](./transforms/map.md)
  - [Cast](./transforms/cast.md)
  - [FillNull](./transforms/fillnull.md)
  - [Normalize](./transforms/normalize.md)
  - [Sample](./transforms/sample.md)
  - [Shuffle](./transforms/shuffle.md)
  - [Sort](./transforms/sort.md)
  - [Unique](./transforms/unique.md)
  - [Take/Skip](./transforms/take-skip.md)
- [Chaining Transforms](./transforms/chaining.md)
- [Custom Transforms](./transforms/custom.md)

# Storage Backends

- [Backend Trait](./backends/backend-trait.md)
- [Local Storage](./backends/local.md)
- [Memory Backend](./backends/memory.md)
- [S3-Compatible](./backends/s3.md)
- [HTTP Backend](./backends/http.md)

# Registry

- [Overview](./registry/overview.md)
- [Dataset Format (.ald)](./registry/format.md)
- [Publishing Datasets](./registry/publishing.md)
- [Pulling Datasets](./registry/pulling.md)
- [Versioning](./registry/versioning.md)
- [Search and Discovery](./registry/search.md)

# HuggingFace Hub Integration

- [Overview](./hf-hub/overview.md)
- [Importing Datasets](./hf-hub/importing.md)
- [Publishing Datasets](./hf-hub/publishing.md)
- [Cache Management](./hf-hub/cache.md)
- [API Reference](./hf-hub/api-reference.md)

# WASM Serving

- [Overview](./serve/overview.md)
- [ServeableContent Trait](./serve/serveable-content.md)
- [Content Plugins](./serve/plugins.md)
  - [Dataset Plugin](./serve/dataset-plugin.md)
  - [Raw Source Plugin](./serve/raw-source-plugin.md)
  - [Course Plugin](./serve/course-plugin.md)
  - [Book Plugin](./serve/book-plugin.md)
- [Browser Integration](./serve/browser-integration.md)
- [Schema System](./serve/schema-system.md)

# CLI Reference

- [Overview](./cli/overview.md)
- [alimentar info](./cli/info.md)
- [alimentar head](./cli/head.md)
- [alimentar schema](./cli/schema.md)
- [alimentar view](./cli/view.md)
- [alimentar convert](./cli/convert.md)
- [alimentar registry](./cli/registry.md)

# Examples

- [ML Data Pipeline](./examples/ml-pipeline.md)
- [Data Preprocessing](./examples/preprocessing.md)
- [Format Conversion](./examples/format-conversion.md)
- [Streaming Large Files](./examples/streaming.md)
- [Registry Workflow](./examples/registry-workflow.md)

# 100 Executable Examples

- [Overview](./100-examples/overview.md)
- [Basic Loading (1-10)](./100-examples/basic-loading.md)
- [DataLoader & Batching (11-20)](./100-examples/dataloader-batching.md)
- [Streaming & Memory (21-30)](./100-examples/streaming-memory.md)
- [Transforms Pipeline (31-45)](./100-examples/transforms-pipeline.md)
- [Quality & Validation (46-55)](./100-examples/quality-validation.md)
- [Drift Detection (56-65)](./100-examples/drift-detection.md)
- [Federated & Splitting (66-75)](./100-examples/federated-splitting.md)
- [HuggingFace Hub (76-85)](./100-examples/huggingface-hub.md)
- [CLI & REPL (86-95)](./100-examples/cli-repl.md)
- [Edge Cases & WASM (96-100)](./100-examples/edge-cases-wasm.md)

# Development Guide

- [Contributing](./development/contributing.md)
- [Extreme TDD](./development/extreme-tdd.md)
- [Quality Gates](./development/quality-gates.md)
- [Testing](./development/testing.md)
- [Code Review](./development/code-review.md)

# Ecosystem Integration

- [Trueno (SIMD/GPU)](./ecosystem/trueno.md)
- [Aprender (ML)](./ecosystem/aprender.md)
- [Assetgen (Content)](./ecosystem/assetgen.md)

# Appendix

- [Glossary](./appendix/glossary.md)
- [FAQ](./appendix/faq.md)
- [Changelog](./appendix/changelog.md)
- [Migration Guide](./appendix/migration-guide.md)
