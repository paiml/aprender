<!-- PCU: cli-capability | contract: contracts/apr-page-cli-capability-v1.yaml -->

# apr capability

What this build can and cannot do, and why — read from the capability contract (#3856)

**Category**: Other

## Synopsis

```text
apr capability [OPTIONS]
```

## Example

```bash
apr capability --json
```

Prints the capability registry compiled into this build, read from `contracts/apr-model-capability-v1.yaml` (#3856). It has three sections:

- `ops`: each model operation, and whether the GPU path supports it.
- `quant_types`: each GGML quantization type (name and type id), and whether a GPU kernel exists for it.
- `op_implementation`: where an operation is implemented (symbol, path), its status, and the evidence for it.

Without `--json`, the same registry prints as a table. Each unsupported entry gives its reason, for example "GPU uses the RMSNorm path; LayerNorm models fall back to CPU". Use `-v` for dispatch resolution and per-command detail.

## Full help

Run `apr capability --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/capability.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/capability.rs)
- Contract: [`contracts/apr-page-cli-capability-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-capability-v1.yaml)

