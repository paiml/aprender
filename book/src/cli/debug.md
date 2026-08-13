<!-- PCU: cli-debug | contract: contracts/apr-page-cli-debug-v1.yaml -->

# apr debug

Simple debugging output ("drama" mode available)

**Category**: Inspection

## Synopsis

```text
apr debug [FILE] [OPTIONS]
apr debug embed-viz --model <MODEL> [OPTIONS]
```

`FILE` is optional because a subcommand brings its own input. `apr debug` with
neither a file nor a subcommand refuses rather than exiting 0 having done
nothing.

## Example

<!-- example-cost: model-required model: qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -->
```bash
apr debug qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

## Full help

Run `apr debug --help` for the complete option list.

## See also

- Subcommand: [`apr debug embed-viz`](./embed-viz.md)
- Source: [`crates/apr-cli/src/commands/debug.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/debug.rs)
- Contract: [`contracts/apr-page-cli-debug-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-debug-v1.yaml)
