<!-- PCU: cli-finetune | contract: contracts/apr-page-cli-finetune-v1.yaml -->

# apr finetune

Fine-tune model with LoRA/QLoRA (GH-244)

**Category**: Training

## Synopsis

```text
apr finetune [OPTIONS]
```

## Example

```bash
apr finetune qwen2.5-coder-0.5b --data train.jsonl --epochs 3
```

## Full help

Run `apr finetune --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/finetune.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/finetune.rs)
- Contract: [`contracts/apr-page-cli-finetune-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-finetune-v1.yaml)

