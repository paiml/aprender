<!-- PCU: cli-tune | contract: contracts/apr-page-cli-tune-v1.yaml -->

# apr tune

ML tuning: LoRA/QLoRA configuration, memory planning, and HPO (GH-176, SPEC-TUNE-2026-001)

**Category**: Training

## Synopsis

```text
apr tune [OPTIONS]
```

## Example

<!-- example-cost: model-required model: qwen2.5-coder-0.5b -->
```bash
apr tune qwen2.5-coder-0.5b --data train.jsonl
```

## Full help

Run `apr tune --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/tune.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/tune.rs)
- Contract: [`contracts/apr-page-cli-tune-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-tune-v1.yaml)
