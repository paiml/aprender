<!-- PCU: cli-compare-hf | contract: contracts/apr-page-cli-compare-hf-v1.yaml -->

# apr compare-hf

Compare APR model against HuggingFace source

**Category**: Quality & Evaluation

## Synopsis

```text
apr compare-hf [OPTIONS]
```

## Example

<!-- example-cost: model-required model: model.apr -->
```bash
apr compare-hf model.apr --hf-repo Qwen/Qwen2.5-Coder-1.5B-Instruct
```

## Full help

Run `apr compare-hf --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/compare_hf.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/compare_hf.rs)
- Contract: [`contracts/apr-page-cli-compare-hf-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-compare-hf-v1.yaml)
