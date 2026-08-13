<!-- PCU: cli-embed-viz | contract: contracts/apr-lint-producers-v1.yaml -->

# apr debug embed-viz

Project a model's token-embedding table to 2-D.

**Category**: Inspection

## Synopsis

```text
apr debug embed-viz --model <MODEL> [--tensor NAME] [--projection pca|random]
                    [--seed N] [--limit N] [--tokens FILE] [-o FILE] [--force]
```

## What it produces

It reads a **real** token-embedding tensor out of a GGUF, APR or SafeTensors
model — dequantising as needed via `RosettaStone::load_tensor_f32` — and writes
the `token_id,token_str,x,y` CSV that `apr embed-viz-lint` parses. One row per
**token**.

| `--projection` | Method |
|----------------|--------|
| `pca` (default) | Exact PCA onto the top 2 principal components. Deterministic. |
| `random` | Seeded Johnson–Lindenstrauss projection. Deterministic in `--seed`, cheap at any hidden size. |
| `umap` | **Refused.** This binary implements no UMAP, and will not label a different algorithm's output `umap`. |

`token_str` comes from the model's own GGUF vocabulary, or from `--tokens`.
When neither is available every row carries the literal `<unresolved>` — a
marker that claims nothing. Token text is escaped (`,` → `\x2c`, `"` → `\x22`,
`\` → `\\`, CR/LF → `\r`/`\n`) so a token containing a comma cannot silently
shift the column count the classifier counts.

## The vocabulary axis is chosen per format

This is the part that is easy to get wrong, and was:

| Format | Reported shape of the embedding table |
|--------|----------------------------------------|
| GGUF | `[hidden, vocab]` — GGML `ne` order, `ne[0]` is the contiguous dimension |
| APR, SafeTensors | `[vocab, hidden]` — row-major |

The payload is `[vocab][hidden]` in all three, so only the reported axis *order*
differs. Taking `shape[0]` as the vocabulary for every format made
`Qwen3.5-0.8B-Q4_K_M.gguf` — whose `token_embd.weight` reports `[1024, 248320]`
— emit 1024 rows for a 248320-token vocabulary, and `--projection pca` never
returned because it was handed a 248320-wide covariance problem.

Fixing the axes made the row count correct and PCA tractable, but **it did not
make full-vocabulary PCA fast**. Measured on that model with a release binary,
at the correct hidden size of 1024:

| rows | wall |
|------|------|
| 5,000 | 82.6s |
| 20,000 | 176.9s |
| 248,320 (whole vocab) | does not finish in 300s; ~26-30 min extrapolated |

So on a large vocabulary the default `--projection pca` still needs either
`--limit` or patience. Use `--projection random` (2.1s for the full vocab on the
same model) when you want the whole vocabulary quickly.

Note that `token_str` looks **correct either way**: it is resolved by row index
from the vocabulary list, so it cannot reveal this. The row count against the
real vocabulary size can. `apr embed-viz-lint --expected-vocab-size` is
therefore not a formality — run it.

As a backstop the producer refuses outright when a model declares more tokens
than the chosen axis has rows: every token must have an embedding row, and
padding only ever goes the other way.

## Example

<!-- example-cost: trivial -->
```bash
apr debug embed-viz --help
```

Producing the observation `apr embed-viz-lint` reads, then checking it against
the model's real vocabulary size:

<!-- example-cost: model-required model: Qwen3.5-0.8B-Q4_K_M.gguf -->
```bash
apr debug embed-viz --model Qwen3.5-0.8B-Q4_K_M.gguf \
    --projection random --seed 42 -o emb.csv
apr embed-viz-lint --csv-file emb.csv --expected-vocab-size 248320
```

The determinism gate wants two runs at one seed:

<!-- example-cost: model-required model: Qwen3.5-0.8B-Q4_K_M.gguf -->
```bash
apr debug embed-viz --model Qwen3.5-0.8B-Q4_K_M.gguf --seed 42 --limit 500 -o a.csv
apr debug embed-viz --model Qwen3.5-0.8B-Q4_K_M.gguf --seed 42 --limit 500 -o b.csv
apr embed-viz-lint --csv-file a.csv --csv-file-b b.csv
```

`--limit` caps the number of tokens projected, which is what you want on a
large vocabulary. PCA's cost here is dominated by the ROW count, not the hidden
size: at a fixed hidden size of 1024, 5,000 rows takes 82.6s and 20,000 takes
176.9s — roughly linear in rows, ~6.3ms per row. That is why the full 248,320-token
vocabulary does not finish inside a 300s budget.

## Full help

Run `apr debug embed-viz --help` for the complete option list.

## See also

- Consumer: [`apr embed-viz-lint`](./embed-viz-lint.md)
- Parent command: [`apr debug`](./debug.md)
- Source: [`crates/apr-cli/src/commands/embed_viz.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/embed_viz.rs)
- Layout contract: [`contracts/tensor-layout-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/tensor-layout-v1.yaml)
- Contract: [`contracts/apr-lint-producers-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-lint-producers-v1.yaml)
