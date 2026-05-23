<!-- PCU: cli-import | contract: contracts/apr-page-cli-import-v1.yaml -->

# apr import

Import from external formats (hf://org/repo, local files, URLs)

**Category**: Model Transform

## Synopsis

```text
apr import [OPTIONS]
```

## Example

```bash
apr import hf://openai/whisper-tiny -o whisper.apr --arch whisper
```

## Full help

Run `apr import --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/import.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/import.rs)
- Contract: [`contracts/apr-page-cli-import-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-import-v1.yaml)

<!-- TODO: walkthrough -->
