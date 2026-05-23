<!-- PCU: cli-ptx | contract: contracts/apr-page-cli-ptx-v1.yaml -->

# apr ptx

PTX analysis and bug detection (register pressure, roofline)

**Category**: Tools & Integration

## Synopsis

```text
apr ptx [OPTIONS]
```

## Example

```bash
apr ptx --gpu-arch sm_89 kernel.ptx
```

## Full help

Run `apr ptx --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/ptx.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/ptx.rs)
- Contract: [`contracts/apr-page-cli-ptx-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-ptx-v1.yaml)

