<!-- PCU: cli-test | contract: contracts/apr-page-cli-test-v1.yaml -->

# apr test

Test harness for web, LLM, media and replay — powered by probador.

Renamed from `apr probar` (Spanish "to try"), which named the *verb* and so said
nothing about the subject. Follows the `apr data` precedent: a plain English noun
for the command, the Spanish name kept for the engine. **`apr probar` still works
as a hidden alias**, so existing scripts are unaffected.

## What it tests

| group | under test |
|---|---|
| **web** | the WASM/browser build and its runtime (`serve`, `build`, `watch`, `comply`, `stress`) |
| **llm** | inference correctness, throughput and cost against an endpoint (`test`, `load`, `bench`, `sweep`, `score`) |
| **media** | rendered output against ground truth (`av-sync`, `audio`, `video`, `animation`) |
| **replay** | the runner itself — recording, state machines, reporting (`record`, `playbook`, `coverage`, `report`) |

Only `tensor` is routed through `apr` today (PMAT-481 visual regression); the
rest are reached via the `aprender-test-cli` binary and land under `apr test` as
they are delegated.

**Category**: Tools & Integration

## Synopsis

```text
apr test [OPTIONS]
```

## Example

<!-- example-cost: trivial -->
```bash
apr test --help
```

## Full help

Run `apr test --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/probar.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/probar.rs)
- Contract: [`contracts/apr-page-cli-test-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-test-v1.yaml)
