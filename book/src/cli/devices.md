<!-- PCU: cli-devices | contract: contracts/apr-page-cli-devices-v1.yaml -->

# apr devices

Enumerate the compute backends this build can drive on this host, and say why each one that cannot be driven is unavailable.

**Category**: Hardware

## Synopsis

```text
apr devices [--json]
```

## What it prints

One line per backend in `{cpu, cuda, wgpu, metal, hip}`, every time, so the absence of a line is itself a defect. A backend is **ready** only when its driver loaded, a device was enumerated, and a context was created; otherwise the line names the refusal (`not compiled in`, `driver not found`, `no device`, `software rasteriser`, `reserve exceeds device memory`). Two identically named devices are told apart by an ordinal (`#1`, `#2`). The registry — not a compile-time flag — is what `apr run`, `apr serve` and `apr bench` resolve a backend from, so this block is the same fact those commands act on.

## Example

<!-- example-cost: trivial -->
```bash
apr devices
apr devices --json | jq '[.entries[] | select(.status == "ready")] | length'
```

The JSON document validates against `contracts/schemas/apr-devices-v1.schema.json` (`schema: apr-devices-v1`).

## Full help

Run `apr devices --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/devices.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/devices.rs)
- Registry: [`crates/aprender-compute/src/registry/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-compute/src/registry/mod.rs)
- Contract: [`contracts/apr-backend-registry-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-backend-registry-v1.yaml)
- Page contract: [`contracts/apr-page-cli-devices-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-devices-v1.yaml)
