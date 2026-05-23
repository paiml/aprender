<!-- PCU: cli-qa | contract: contracts/apr-page-cli-qa-v1.yaml -->

# apr qa

Falsifiable QA checklist for model releases

**Category**: Quality & Evaluation

## Synopsis

```text
apr qa [OPTIONS]
```

## Example

```bash
apr qa qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

## What this does

`apr qa` runs the eight falsifiable ship-gates on a model: golden output, throughput,
Ollama parity, GPU-vs-CPU speedup, tensor contract, cross-format parity, PTX parity,
and metadata plausibility. Every gate has a measurable pass/fail condition — there
are no judgment calls. This is the FIRST tool to run on any model regression, and
the CI gate that blocks a HF or crates.io publish.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--assert-tps N` | Fail if throughput < N tok/s | `--assert-tps 100` |
| `--assert-speedup R` | Fail if speedup vs Ollama < R | `--assert-speedup 1.0` |
| `--iterations N` | Bench iterations (default 10) | `--iterations 20` |
| `--max-tokens N` | Generate budget per iteration | `--max-tokens 64` |
| `--json` | Emit one JSON envelope for CI | `--json` |
| `--previous-report FILE` | Regression check against prior run | `--previous-report last.json` |
| `--regression-threshold R` | Allowed degradation ratio | `--regression-threshold 0.05` |

## Common workflows

**Block a release until every gate is GREEN.**

```bash
apr qa qwen2.5-coder-1.5b.apr \
    --assert-tps 100 --assert-speedup 1.0 --json > qa-report.json
jq '.gates[] | select(.status != "PASS")' qa-report.json
```

**Regression-detect against the previous nightly build.**

```bash
apr qa qwen2.5-coder-1.5b.apr --json --previous-report nightly-prev.json \
    --regression-threshold 0.10
```

## Troubleshooting

- **Golden output fails on a known-good model** — re-pull the model; the golden
  fixture is content-hashed against the canonical model in the contract. See
  [`apr qa first, hypotheses second`](https://github.com/paiml/aprender/issues/202).
- **Throughput gate fails on first run** — increase `--warmup` to 5+; cold-cache
  numbers are not representative.
- **Cross-format parity skipped** — pass `--safetensors-path` to enable
  GGUF/SafeTensors comparison (F-QUAL-032 contract).

## See also

- Source: [`crates/apr-cli/src/commands/qa.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/qa.rs)
- Contract: [`contracts/apr-page-cli-qa-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-qa-v1.yaml)

