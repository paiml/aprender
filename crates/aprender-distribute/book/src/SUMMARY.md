# Repartir - Sovereign AI-Grade Distributed Computing

[Introduction](./introduction.md)

# Getting Started

- [Installation](./getting-started/installation.md)
- [Quick Start](./getting-started/quick-start.md)
- [First Distributed Task](./getting-started/first-task.md)
- [Core Concepts](./getting-started/core-concepts.md)

# Architecture

- [Overview](./architecture/overview.md)
- [Design Principles](./architecture/design-principles.md)
  - [Pure Rust Everything](./architecture/pure-rust.md)
  - [Message-Oriented](./architecture/message-oriented.md)
  - [Work Stealing](./architecture/work-stealing.md)
- [Task Model](./architecture/task-model.md)
- [Scheduler Architecture](./architecture/scheduler.md)
- [Fault Tolerance](./architecture/fault-tolerance.md)

# Executors

- [Executor Overview](./executors/overview.md)
- [CPU Executor](./executors/cpu.md)
  - [Process Execution](./executors/cpu-process.md)
  - [Thread Pool](./executors/cpu-threadpool.md)
  - [Environment & Args](./executors/cpu-env.md)
- [GPU Executor](./executors/gpu.md)
  - [WGSL Shaders](./executors/gpu-wgsl.md)
  - [Buffer Management](./executors/gpu-buffers.md)
  - [Compute Pipelines](./executors/gpu-pipelines.md)
  - [GPU Examples](./executors/gpu-examples.md)
- [Remote Executor](./executors/remote.md)
  - [TCP Transport](./executors/remote-tcp.md)
  - [TLS Security](./executors/remote-tls.md)
  - [Binary Distribution](./executors/remote-binaries.md)

# Scheduler & Messaging

- [Task Scheduler](./scheduler/scheduler.md)
- [Work-Stealing Algorithm](./scheduler/work-stealing.md)
- [Priority Queues](./scheduler/priorities.md)
- [PubSub Messaging](./messaging/pubsub.md)
- [PushPull Messaging](./messaging/pushpull.md)
- [Message Patterns](./messaging/patterns.md)

# Examples

- [CPU Task Execution](./examples/cpu-task.md)
- [GPU Compute Shader](./examples/gpu-compute.md)
- [Remote Distributed Work](./examples/remote-work.md)
- [Heterogeneous Pool](./examples/heterogeneous-pool.md)
- [Machine Learning Pipeline](./examples/ml-pipeline.md)
- [Scientific Computing](./examples/scientific.md)

# Development Guide

- [Contributing](./development/contributing.md)
- [Iron Lotus Framework](./development/iron-lotus.md)
  - [Jidoka (Quality Gates)](./development/jidoka.md)
  - [Genchi Genbutsu (Transparency)](./development/genchi-genbutsu.md)
  - [Kaizen (Continuous Improvement)](./development/kaizen.md)
- [Certeza Testing](./development/certeza.md)
  - [Tier 1: Sub-Second](./development/tier1-tests.md)
  - [Tier 2: Commit Gates](./development/tier2-tests.md)
  - [Tier 3: Mutation Testing](./development/tier3-tests.md)
- [Testing Guide](./development/testing.md)
- [Quality Gates](./development/quality-gates.md)
- [Code Review Checklist](./development/code-review.md)

# Advanced Topics

- [Binary Signing](./advanced/binary-signing.md)
- [Supply Chain Security](./advanced/supply-chain.md)
- [Custom Executors](./advanced/custom-executors.md)
- [Performance Tuning](./advanced/performance-tuning.md)
- [Distributed ML Patterns](./advanced/distributed-ml.md)
- [Fault Recovery Strategies](./advanced/fault-recovery.md)

# Ecosystem Integration

- [Trueno Integration](./ecosystem/trueno.md)
- [Aprender (ML Workloads)](./ecosystem/aprender.md)
- [PAIML Stack](./ecosystem/paiml-stack.md)
- [Renacer (Process Lifecycle)](./ecosystem/renacer.md)

# Specifications

- [Iron Lotus Enhanced Spec](./specifications/iron-lotus-spec.md)
- [Sovereign AI Considerations](./specifications/sovereign-ai.md)
- [Academic Foundations](./specifications/academic-foundations.md)
- [Roadmap](./specifications/roadmap.md)

# Appendix

- [Glossary](./appendix/glossary.md)
- [References](./appendix/references.md)
- [FAQ](./appendix/faq.md)
- [Changelog](./appendix/changelog.md)
- [Comparison with Ray/Dask](./appendix/comparison.md)
