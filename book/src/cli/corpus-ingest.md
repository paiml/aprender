<!-- PCU: cli-corpus-ingest | contract: contracts/apr-cli-commands-v1.yaml -->

# apr corpus-ingest

Dry-run scaffolding for the SHIP-TWO-001 MODEL-2 Python-code pretraining corpus
ingest pipeline. Reads and validates
`contracts/dataset-thestack-python-v1.yaml` (C-DATA-THESTACK-PYTHON) and emits
a planned manifest.

**Category**: Data

This was `apr-corpus-ingest`, a second `[[bin]]` in the `apr-cli` crate. That
binary is gone, so `cargo install aprender` places exactly one program in
`~/.cargo/bin`.

## Synopsis

```text
apr corpus-ingest plan [--contract FILE] [--output-dir DIR]
apr corpus-ingest validate-contract <PATH>
```

## Arguments

### `plan`

| Flag | Default | Meaning |
|------|---------|---------|
| `--contract <FILE>` | `contracts/dataset-thestack-python-v1.yaml` | Corpus contract to read |
| `--output-dir <DIR>` | `output` | Where `dry-run-manifest.yaml` is written |

### `validate-contract`

| Argument | Required | Meaning |
|----------|----------|---------|
| `<PATH>` | yes | Corpus contract to validate |

Pure validation: parses the YAML, asserts the six required top-level keys
(`source`, `license_whitelist`, `pii_scrub`, `deduplication`, `split`,
`budget`) and the declared minimum counts (7 invariants, 5 falsification tests,
5 gates). Writes no files. Exits `0` on pass and non-zero on failure.

## What it does NOT do

This is scaffolding, and it is honest about it. There is **no network access**
and no download. `plan` prints the six planned pipeline steps and writes a
manifest whose fields are TODO placeholders for the real ingest to fill in:

1. pin the HF dataset `revision_sha`, record `raw_tar_sha256`
2. license filter against the SPDX whitelist
3. PII scrub (file-level rejection on pattern match)
4. MinHash-LSH dedup — shingle 5, 128 permutations, seed 42
5. hash-by-`file_sha256` deterministic train/val split
6. emit shards, manifest, provenance, `corpus_sha256`

There is deliberately no `run` subcommand: a missing subcommand is a louder
signal than a stubbed one that lies.

## Examples

<!-- example-cost: trivial -->
```bash
apr corpus-ingest validate-contract contracts/dataset-thestack-python-v1.yaml
```

<!-- example-cost: trivial -->
```bash
apr corpus-ingest plan --output-dir /tmp/ingest-plan
cat /tmp/ingest-plan/dry-run-manifest.yaml
```

## Exit codes

`0` on success, `4` (invalid input) when the contract is missing, unparseable,
or fails its structural minimums.

## Full help

Run `apr corpus-ingest --help`, or `apr corpus-ingest <SUBCOMMAND> --help`, for
the complete option list.

## See also

- Contract: [`contracts/dataset-thestack-python-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/dataset-thestack-python-v1.yaml)
- Source: [`crates/apr-cli/src/commands/corpus_ingest.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/corpus_ingest.rs)
