<!-- PCU: cli-qa-playbook | contract: contracts/apr-page-cli-qa-playbook-v1.yaml -->

# apr qa-playbook

Model QA playbook runner: certify, run, score.

**Category**: Tools

This command surface previously shipped only as the standalone `apr-qa` binary.
Its command enum lived in a `main.rs`, which is importable by nothing, so `apr`
had no route to it at all. `apr qa-playbook` and `apr-qa` now call the same
`dispatch` function, so the two surfaces cannot drift.

## Synopsis

```text
apr qa-playbook <COMMAND>
```

## Subcommands

| Path |
|------|
| `apr qa-playbook certify` |
| `apr qa-playbook run` |
| `apr qa-playbook tools` |
| `apr qa-playbook generate` |
| `apr qa-playbook score` |
| `apr qa-playbook report` |
| `apr qa-playbook list` |
| `apr qa-playbook lock-playbooks` |
| `apr qa-playbook tickets` |
| `apr qa-playbook parity` |
| `apr qa-playbook export-csv` |
| `apr qa-playbook export-evidence` |
| `apr qa-playbook bootstrap` |
| `apr qa-playbook validate-contract` |
| `apr qa-playbook kernel-coverage` |

Every one of these is locked by `FALSIFY-CLI-006`: the list above and the built
binary are asserted to agree in both directions.

## Example

<!-- example-cost: trivial -->
```bash
apr qa-playbook --help
```

## Full help

Run `apr qa-playbook --help` for the complete option list.

## See also

- Source: [`crates/aprender-qa-cli/src/cli.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-qa-cli/src/cli.rs)
- Registry: [`contracts/apr-cli-commands-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-cli-commands-v1.yaml)
