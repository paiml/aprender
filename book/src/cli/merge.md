<!-- PCU: cli-merge | contract: contracts/apr-page-cli-merge-v1.yaml -->

# apr merge

Merge multiple models

**Category**: Model Transform

## Synopsis

```text
apr merge [OPTIONS]
```

## Example

```bash
apr merge model1.apr model2.apr --strategy weighted --weights 0.7,0.3 -o merged.apr
```

## Full help

Run `apr merge --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/merge.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/merge.rs)
- Contract: [`contracts/apr-page-cli-merge-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-merge-v1.yaml)

