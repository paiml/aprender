<!-- PCU: cli-kernel | contract: contracts/apr-lint-producers-v1.yaml -->

# apr kernel

Kernel-level parity measurements.

**Category**: Analysis

## Synopsis

```text
apr kernel parity [--impl tiled|flash2] [--ref naive] [--seq-len N]
                  [--num-heads N] [--num-kv-heads N] [--head-dim N]
                  [--seed N] [--json] [-o FILE] [--force]
```

## What `parity` measures

`--impl tiled` runs the in-tree tiled online-softmax attention kernel
(`realizar::brick::FlashAttentionBrick`) and a naive reference — a materialised
score row with a max-subtracted softmax — over the same seeded Q/K/V, then
reports `max_abs_diff` and `cosine_sim` between the two outputs. Two
independent implementations, so the comparison can genuinely fail.

The regime is a decode step: one query position attending over a `seq_len`-long
KV cache. The emitted body says so.

`--impl flash2` names the pinned `hf-kernels-community:flash-attn2@<sha>` CUDA
kernel. **This binary embeds no such kernel**, so asking for it is refused with
a non-zero exit — never answered by the tiled path under flash2's name. That
matters because CRUX-L-02 pins `kernel_source` to `pkg@sha` precisely so a
provenance line cannot be borrowed.

`--impl flash2` with `--head-dim` outside {64, 128} is refused at dispatch, and
that refusal is itself a capturable observation: the `{"error": ...}` body it
writes is what `apr attn-parity-lint --head-dim-error-file` reads.

## Example

<!-- example-cost: trivial -->
```bash
apr kernel parity --impl tiled --ref naive --seq-len 16 --num-heads 2 \
    --num-kv-heads 2 --head-dim 64 --json
```

Feeding the result to its lint — one body discharges both gates:

<!-- example-cost: trivial -->
```bash
apr kernel parity --impl tiled --ref naive --seq-len 16 --json -o /tmp/parity.json --force
apr attn-parity-lint --parity-file /tmp/parity.json --provenance-file /tmp/parity.json
```

## Full help

Run `apr kernel parity --help` for the complete option list.

## See also

- Consumer: [`apr attn-parity-lint`](./attn-parity-lint.md)
- Source: [`crates/apr-cli/src/commands/kernel_parity.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/kernel_parity.rs)
- Contract: [`contracts/apr-lint-producers-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-lint-producers-v1.yaml)
