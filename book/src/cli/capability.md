<!-- PCU: cli-capability | contract: contracts/apr-page-cli-capability-v1.yaml -->

# apr capability

What this build can and cannot do, and why — read from the capability contract (`contracts/apr-model-capability-v1.yaml`).

**Category**: Inspection

## Synopsis

```text
apr capability [--json]
```

## What it prints

One line per operation in the contract's registry. An operation this build does not support always prints the contract's reason for it, never a bare "no". The contract is embedded in the binary at compile time, so the answer describes the build you are running, not the tree you are reading. `--json` emits the contract's own sections verbatim, so a consumer reads the same field names the contract declares.

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
- Contract: [`contracts/apr-page-cli-capability-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-capability-v1.yaml)
- Registry: [`contracts/apr-model-capability-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-model-capability-v1.yaml)
