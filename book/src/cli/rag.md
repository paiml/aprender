<!-- PCU: cli-rag | contract: contracts/apr-page-cli-rag-v1.yaml -->

# apr rag

Pure-Rust retrieval-augmented generation pipeline: index documents, query them
with dense/sparse/hybrid retrieval, transcribe media into the corpus, extract
video keyframes, and evaluate retrieval quality.

**Category**: Data

## Was `trueno-rag`

This command surface was published as a standalone binary named `trueno-rag`.
That name advertises `trueno`, a project that is now `crates/aprender-compute`
inside this monorepo, and it is not a name anyone would guess from
`cargo install aprender-rag-cli`. The pipeline is unchanged — `apr rag` calls
`aprender_rag_cli::run`, the entry point that binary's `main` used to inline —
but the binary is gone.

| Before | Now |
|--------|-----|
| `trueno-rag index --path docs/ --output idx/` | `apr rag index --path docs/ --output idx/` |
| `trueno-rag query "vectors" --index idx/` | `apr rag query "vectors" --index idx/` |
| `trueno-rag transcribe --path media/ -r` | `apr rag transcribe --path media/ -r` |
| `trueno-rag extract-frames --path video/` | `apr rag extract-frames --path video/` |
| `trueno-rag demo` | `apr rag demo` |
| `trueno-rag info` | `apr rag info` |
| `trueno-rag eval …` (feature `eval`) | `apr rag eval …` (feature `eval`) |

One deliberate difference: **`demo --query` has no `-q` short form.** `apr` has
a global `-q/--quiet`, and clap refuses an entire command when two arguments
claim one short — `apr rag demo` would not have printed so much as its help.
`--query` is otherwise unchanged, and `apr rag query <QUERY>` (the non-demo
path) takes its query positionally and is unaffected.

## Synopsis

```text
apr rag <COMMAND>

  demo                Run a demo RAG query over a built-in corpus
  index               Index documents from a file or directory
  query <QUERY>       Query an index
  transcribe          Batch transcribe media files to .srt sidecars
  extract-frames      Extract video keyframes at scene changes (needs ffmpeg)
  info                Show pipeline info: chunkers, embedders, formats
  eval                Evaluation framework (requires --features eval)
```

### `index`

| Argument | Default | Meaning |
|----------|---------|---------|
| `-p`, `--path <PATH>` | required | File or directory to index |
| `-o`, `--output <DIR>` | required | Output directory for the index |
| `--chunk-size <N>` | `512` | Chunk size in characters (recursive chunker) |
| `--chunk-overlap <N>` | `64` | Chunk overlap in characters |
| `--dimension <N>` | `256` | Embedding dimension (tfidf embedder only) |
| `-e`, `--embedder <KIND>` | `tfidf` | `tfidf` or `semantic` (needs `embeddings`) |
| `-m`, `--model <MODEL>` | `mini-lm` | `mini-lm`, `bge-small`, `bge-base` |
| `-r`, `--recursive` | `false` | Scan subdirectories |
| `--chunk-strategy <S>` | `auto` | `auto`, `recursive`, `timestamp` |
| `-j`, `--jobs <N>` | `1` | Parallel loading jobs |
| `--manifest` | `false` | Write a JSON manifest of files and chunks |
| `--exclude <GLOB>` | none | Exclude glob, repeatable |
| `--dedup` | `false` | Drop chunks with identical content |
| `--sqlite` | `false` | Also export a SQLite+FTS5 index (needs `sqlite`) |
| `--incremental` | `false` | Re-index only changed files (requires `--sqlite`) |

### `query`

| Argument | Default | Meaning |
|----------|---------|---------|
| `<QUERY>` | required | Query string (positional) |
| `-i`, `--index <DIR>` | required | Index directory |
| `-t`, `--top-k <N>` | `5` | Number of results |
| `-f`, `--format <FMT>` | `text` | `text` or `json` |
| `--mode <MODE>` | `hybrid` | `dense`, `sparse` (BM25), `hybrid` |
| `--fusion <STRATEGY>` | `rrf` | `rrf`, `linear`, `dbsf` (hybrid only) |
| `--fusion-k <F>` | unset | RRF `k`, or Linear `dense_weight` |
| `--candidates <N>` | `50` | Candidates per source for hybrid retrieval |
| `--rerank <MODE>` | `none` | `none` or `lexical` |
| `--hyde` | `false` | HyDE query expansion (needs `ANTHROPIC_API_KEY` + `eval`) |

### `transcribe`

`-p/--path` (required), `-r/--recursive`, `--skip-existing` (default `true`),
`-j/--jobs` (`1`), `-m/--model` (Whisper `.apr` path), `-b/--backend`
(`cpu`, `gpu`, `cuda`; default `cpu`), `--dry-run`, `--prompt`, `--hotwords`,
`--exclude` (repeatable).

### `extract-frames`

`-p/--path` (required), `-r/--recursive`, `--threshold` (`0.3`),
`--min-interval` (`5.0` seconds), `-j/--jobs` (`4`), `--skip-existing`
(default `true`), `--dry-run`, `--exclude` (repeatable).

### `demo`

`--query` (default `"What is machine learning?"`), `-t/--top-k` (`3`).

### `eval`

Available only when built `--features eval`. Subcommands: `sample`, `generate`,
`retrieve`, `judge`, `metrics`, `compare`, `gate`. `generate` and `judge` call
the Claude API and need `ANTHROPIC_API_KEY`; `sample`, `retrieve`, `metrics`,
`compare` and `gate` are offline.

## Example

<!-- example-cost: trivial -->
```bash
apr rag --help
apr rag info
```

Index a directory and query it:

<!-- example-cost: trivial -->
```bash
mkdir -p /tmp/apr-rag-docs
echo 'Vectors are arrays of numbers used in retrieval.' > /tmp/apr-rag-docs/a.txt
apr rag index --path /tmp/apr-rag-docs --output /tmp/apr-rag-idx
apr rag query "vectors" --index /tmp/apr-rag-idx --top-k 1
```

## Full help

Run `apr rag --help`, or `apr rag <SUBCOMMAND> --help`, for the complete
option list.

## See also

- Command surface: [`crates/aprender-rag-cli/src/lib.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-rag-cli/src/lib.rs)
- Engine: [`crates/aprender-rag/`](https://github.com/paiml/aprender/blob/main/crates/aprender-rag/src/lib.rs)
- Contract: [`contracts/apr-page-cli-rag-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-rag-v1.yaml)
