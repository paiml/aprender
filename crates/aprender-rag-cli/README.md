# aprender-rag-cli

Command-line interface for the **Trueno-RAG** pipeline (Retrieval-Augmented Generation)
in the aprender monorepo. Installs the `trueno-rag` binary.

This crate is a thin CLI wrapper over [`aprender-rag`](../aprender-rag) (lib name
`trueno-rag`), exposing document chunking, hybrid (BM25 + dense) retrieval, reranking, and
context assembly from the terminal.

## Usage

```bash
cargo run -p aprender-rag-cli -- --help
```

## Features

Feature flags are forwarded to `aprender-rag`:

- `embeddings` — embedding generation
- `nemotron` — Nemotron reranker
- `transcription` — audio transcription ingest
- `eval` — retrieval evaluation metrics

## Relation to the stack

Part of the [paiml/aprender](https://github.com/paiml/aprender) monorepo (APR-MONO
consolidation; relocated to flat layout under `crates/` per spec §S). The RAG engine itself
lives in [`crates/aprender-rag`](../aprender-rag); this crate only provides the binary
entry point.
