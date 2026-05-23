<!-- PCU: cli-distill | contract: contracts/apr-page-cli-distill-v1.yaml -->

# apr distill

Knowledge distillation (teacher -> student) (GH-247, ALB-011)

**Category**: Training

## Synopsis

```text
apr distill [OPTIONS]
```

## Example

```bash
apr distill --teacher 7b.apr --student 0.5b.apr --data train.jsonl
```

## Full help

Run `apr distill --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/distill.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/distill.rs)
- Contract: [`contracts/apr-page-cli-distill-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-distill-v1.yaml)

<!-- TODO: walkthrough -->
