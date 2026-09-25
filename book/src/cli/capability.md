<!-- PCU: cli-capability | contract: contracts/apr-page-cli-capability-v1.yaml -->

# apr capability

What this build can and cannot do, and why. The answer is read from the model-capability contract.

**Category**: Inspection

## Synopsis

```text
apr capability [--json]
```

## What it prints

The model-capability registry that this binary was **built** from. The contract `contracts/apr-model-capability-v1.yaml` is embedded at compile time through its packaged mirror, `crates/apr-cli/contracts/apr-model-capability-v1.yaml`. So a published binary reports its own facts, not a file that happens to be on the host. Every unsupported operation prints the reason it is unsupported. `--json` emits the contract's own sections verbatim, so a consumer reads the same field names the contract declares.

## Example

<!-- example-cost: trivial -->
```bash
apr capability
apr capability --json | jq 'keys'
```

## Full help

Run `apr capability --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/capability.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/capability.rs)
- Contract: [`contracts/apr-model-capability-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-model-capability-v1.yaml)
- Page contract: [`contracts/apr-page-cli-capability-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-capability-v1.yaml)
