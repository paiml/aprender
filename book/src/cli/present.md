# apr present

Presentar WASM app operations: serve, bundle, scaffold, check, score, gate,
deploy.

**Category**: Observability & Pipeline

## Synopsis

```text
apr present <SUBCOMMAND> [OPTIONS]
```

## Where this came from

`apr present` is the whole of the former `presentar` binary
(`aprender-present-cli`). `presentar` is a pre-consolidation name; published,
it installed `~/.cargo/bin/presentar` alongside the identically-named binary
the standalone `presentar` crate ships, and whichever was installed last won.

It is a **container** subcommand rather than seven top-level verbs because
four of the seven names already mean something else in `apr`:
`apr present serve` serves a WASM app, `apr serve` serves a model;
`apr present check` validates a manifest, `apr check` runs a model self-test;
`apr present score` scores a manifest, [`apr score`](./score.md) scores a Rust
crate.

## Subcommands

### `apr present serve`

Static development server with optional hot reload.

| Option | Default | Description |
|--------|---------|-------------|
| `-p, --port <PORT>` | `8080` | Port to serve on |
| `-d, --dir <DIR>` | `www` | Directory to serve |
| `-w, --watch` | off | Watch for changes, rebuild WASM, and hot-reload browsers |

`--watch` needs `aprender-present-cli`'s `dev-server` feature. Without it the
server still starts and a warning explains that hot reload is off — the
binary's behaviour, unchanged. The hot-reload WebSocket listens on
`ws://localhost:35729`.

### `apr present bundle`

Build the optimized release WASM bundle with `wasm-pack`, then shrink it with
`wasm-opt -Oz`.

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output <DIR>` | `dist` | Output directory |
| `--no-optimize` | off | Skip the `wasm-opt` pass |

### `apr present new`

Scaffold a project directory containing `app.yaml` and `www/index.html`.

| Argument | Description |
|----------|-------------|
| `<NAME>` | Project name; also the directory created |

### `apr present check`

Parse a manifest and report its name, version, data-source count and section
count. Exits 1 if the file cannot be read or does not parse.

| Argument | Default | Description |
|----------|---------|-------------|
| `[MANIFEST]` | `app.yaml` | Path to the manifest file |

### `apr present score`

Quality-score a manifest across six dimensions — structural (25), performance
(20), accessibility (20), data (15), documentation (10), consistency (10) —
and emit a letter grade from `A+` down to `F`.

| Option | Default | Description |
|--------|---------|-------------|
| `[MANIFEST]` | `app.yaml` | Path to the manifest file |
| `-f, --format <FORMAT>` | `text` | `text`, `json`, or `badge` (an SVG shields-style badge) |
| `--badge <FILE>` | none | Also write the SVG badge to this file |

> **Changed on rehome.** The binary compared `--format` against `json` and
> `badge` and fell through to the text renderer for anything else, so
> `--format jsno` printed a text report and exited 0. The three formats it can
> actually produce are now the three the parser accepts.

### `apr present gate`

Enforce quality thresholds on a manifest. Exits 1 when the grade is below
`--min-grade`, when the score is below `--min-score`, or when `--strict` is
set and any warning was raised.

| Option | Default | Description |
|--------|---------|-------------|
| `[MANIFEST]` | `app.yaml` | Path to the manifest file |
| `-m, --min-grade <GRADE>` | `B` | Minimum passing grade: `F`, `D`, `C`, `B`, `A` (suffixes `+`/`-` accepted) |
| `-s, --min-score <N>` | none | Minimum score, 0-100 |
| `--strict` | off | Promote warnings to failures |

### `apr present deploy`

Publish a built bundle to a hosting target.

| Option | Default | Description |
|--------|---------|-------------|
| `-s, --source <DIR>` | `dist` | Source directory to deploy |
| `-t, --target <TARGET>` | `s3` | `s3`, `cloudflare`, `vercel`, `netlify`, or `local` |
| `-b, --bucket <NAME>` | none | S3 bucket, Cloudflare project name, or `local` destination directory |
| `--distribution <ID>` | none | CloudFront distribution to invalidate after an S3 upload |
| `--region <REGION>` | `us-east-1` | AWS region for S3 |
| `--dry-run` | off | Print the plan; upload nothing |
| `--skip-build` | off | Deploy existing files without bundling first |

`--bucket` is required for `s3`. For `cloudflare` it names the Pages project
(default `presentar-app`); for `local` it is the destination directory
(default `/var/www/html`); `vercel` and `netlify` ignore it.

> **Changed on rehome.** An unknown `--target` was already refused, but only
> *after* a full `wasm-pack` release build had run. It is now refused at parse
> time, before anything is built.

## Examples

```bash
apr present new my-dashboard
```

<!-- example-cost: interactive -->
```bash
apr present serve --port 3000 --dir www --watch
```

```bash
apr present check app.yaml
apr present score app.yaml --format json
apr present gate app.yaml --min-grade A- --strict
```

```bash
apr present bundle --output dist
apr present deploy --target s3 --bucket my-bucket --region eu-west-1 --dry-run
```

## See also

- [`apr score`](./score.md) — quality score for a Rust crate (a different scorer)
- [`apr serve`](./serve.md) — model inference server
- Source: [`crates/apr-cli/src/commands/present.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/present.rs)
- Implementation: [`crates/aprender-present-cli/src/lib.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-present-cli/src/lib.rs)
