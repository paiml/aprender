<!-- PCU: cli-beat-run | contract: contracts/apr-cli-commands-v1.yaml -->

# apr beat-run

Evaluate a beat-benchmark contract against a measured value (PMAT-741)

**Category**: Quality

## Synopsis

```text
apr beat-run <CONTRACT> [--measured <VALUE>] [--json]
```

`apr beat-run` is the falsifiable runner behind the four-pillar "replace **and**
beat" mission. A beat contract pins an incumbent's baseline — the number
scikit-learn, PyTorch, Unsloth or Ollama actually produces — together with the
threshold `apr` must clear to claim a win. This command reads that contract and,
given a measurement, returns the verdict.

Two modes:

- **Without `--measured`** it reports the contract's pinned parameters and exits
  0. Use this to see what a beat currently claims.
- **With `--measured`** it computes the verdict and exits **non-zero on a
  regression or an unjudgeable contract**, so it can gate CI directly.

The verdict is not computed here. It comes from
`aprender_contracts::schema::Beat::evaluate`, the single source of truth, so the
CLI and the contract engine cannot drift into disagreeing about whether a beat
was won.

## Examples

<!-- example-cost: trivial -->
```bash
apr beat-run --help
```

Report what a beat contract pins, without judging anything:

<!-- example-cost: trivial -->
```bash
apr beat-run contracts/beat-sklearn-iris-v1.yaml
```

Judge a measurement and gate on it — this is the form CI uses:

<!-- example-cost: trivial -->
```bash
apr beat-run contracts/beat-sklearn-iris-v1.yaml --measured 0.973
echo "exit=$?"   # non-zero means REGRESSED or unjudgeable
```

## Exit codes

| exit | meaning |
|-----:|---------|
| 0 | the beat was WON, or no `--measured` value was supplied |
| non-zero | REGRESSED, or the contract cannot judge the value it was given |

An **unjudgeable** contract exits non-zero on purpose. A beat that cannot decide
is not a beat that passed: silently treating "I could not tell" as a win is the
failure mode this runner exists to prevent.

A contract path that does not exist reports `File not found`, not a format
error — an earlier version printed `Invalid APR format:` and sent readers
looking for a model that was never involved.

## See also

- [`apr bench`](./bench.md) — produce the measurement this command judges
- [`apr qa`](./qa.md) — falsifiable QA gates on a model artifact
- Source: [`crates/apr-cli/src/commands/beat_run.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/beat_run.rs)
- Contract: [`contracts/apr-cli-commands-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-cli-commands-v1.yaml)
