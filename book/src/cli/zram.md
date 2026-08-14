<!-- PCU: cli-zram | contract: contracts/apr-page-cli-zram-v1.yaml -->

# apr zram

zram device management — a `zramctl` replacement — plus SIMD-accelerated
compression benchmarking.

**Category**: Hardware

## Was `trueno-zram`

This command surface was published as a standalone binary named `trueno-zram`.
That name advertises `trueno`, a project that is now `crates/aprender-compute`
inside this monorepo. The commands are unchanged — `apr zram` calls
`aprender_zram_cli::run`, the same dispatch the binary's `main` performed — but
the binary is gone.

| Before | Now |
|--------|-----|
| `trueno-zram create -d 0 -s 4G -a zstd` | `apr zram create -d 0 -s 4G -a zstd` |
| `trueno-zram remove -d 0 --force` | `apr zram remove -d 0 --force` |
| `trueno-zram --format json status` | `apr zram --format json status` |
| `trueno-zram benchmark --pages 10000` | `apr zram benchmark --pages 10000` |

One deliberate difference, and it fixes a shipped defect:

> **`benchmark -p` now means `--pattern`, and `--pages` is long-form only.**
>
> `--pages` derived `-p` from its field name while `--pattern` asked for
> `short = 'p'` explicitly. Two arguments cannot own one short. Under
> `debug_assertions` clap refused the whole command, so `trueno-zram benchmark`
> **panicked outright in any debug build**; in the release build
> `cargo install` produces, the collision was silent and `-p` resolved to
> `--pages`, so `--pattern`'s declared short was dead and `benchmark -p text`
> died in an integer parser:
>
> ```text
> error: invalid value 'text' for '--pages <PAGES>': invalid digit found in string
> ```
>
> The explicit declaration wins. `--pages <N>` is unchanged.

## Synopsis

```text
apr zram [--format <FORMAT>] <COMMAND>

  create      Create and configure a zram device
  remove      Remove a zram device
  status      Show zram device status
  benchmark   Run compression benchmarks
```

| Global argument | Default | Meaning |
|-----------------|---------|---------|
| `--format <FORMAT>` | `table` | `table`, `json`, or `raw` (scripting). Consulted by `status`, which is the only subcommand with tabular output. |

### `create`

| Argument | Default | Meaning |
|----------|---------|---------|
| `-d`, `--device <N>` | `0` | Device number; refused outside `0..=16` |
| `-s`, `--size <SIZE>` | required | Device size, e.g. `4G`, `512M`, `ram/2` |
| `-a`, `--algorithm <ALG>` | `lz4` | Compression algorithm (`lz4`, `zstd`) |
| `--streams <N>` | `0` | Compression streams; `0` means auto |

### `remove`

| Argument | Default | Meaning |
|----------|---------|---------|
| `-d`, `--device <N>` | `0` | Device number to remove |
| `-f`, `--force` | off | Remove even if the device is in use |

### `status`

| Argument | Default | Meaning |
|----------|---------|---------|
| `-d`, `--device <N>` | all devices | Show one device instead of every one |

### `benchmark`

| Argument | Default | Meaning |
|----------|---------|---------|
| `--pages <N>` | `10000` | Number of pages to compress. Long form only — see above |
| `-a`, `--algorithm <ALG>` | `all` | `lz4`, `zstd`, or `all` |
| `-p`, `--pattern <PAT>` | `mixed` | `zero`, `random`, `text`, `mixed` |

`create` and `remove` write to `/sys/block/zram*` and need root.
`status` and `benchmark` do not.

## Example

<!-- example-cost: trivial -->
```bash
apr zram --help
apr zram benchmark --pages 32 --pattern text --algorithm lz4
```

Inspect existing devices as JSON:

<!-- example-cost: trivial -->
```bash
apr zram --format json status
```

Creating and removing devices is destructive and needs root:

<!-- example-cost: destructive -->
```bash
apr zram create --device 0 --size 4G --algorithm zstd
```

## Full help

Run `apr zram --help`, or `apr zram <SUBCOMMAND> --help`, for the complete
option list.

## See also

- Command surface: [`crates/aprender-zram-cli/src/lib.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-zram-cli/src/lib.rs)
- Engine: [`crates/aprender-zram-core/`](https://github.com/paiml/aprender/blob/main/crates/aprender-zram-core/src/lib.rs)
- Contract: [`contracts/apr-page-cli-zram-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-zram-v1.yaml)
