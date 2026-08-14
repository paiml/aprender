<!-- PCU: cli-db | contract: contracts/apr-cli-commands-v1.yaml -->

# apr db

The embedded analytics database: SQL over Parquet, GPU-first with a SIMD
fallback, served over HTTP.

**Category**: Serving

This was the standalone `trueno-db` binary. That binary is gone; the server is
reached as `apr db serve`.

## Synopsis

```text
apr db serve --config <FILE>
```

## Arguments

| Flag | Required | Meaning |
|------|----------|---------|
| `--config <FILE>` | yes | Path to the YAML configuration file |

`--config` was the standalone binary's only argument, and it is still the only
one. A path that does not exist is refused with exit `3` (not found) before any
runtime is started.

## Configuration

```yaml
listen: "0.0.0.0:5433"          # required — no default
data_dir: "/opt/trueno-db/data" # default
max_memory_mb: 2048             # default
max_connections: 128            # default
wal_enabled: true               # default
sync_mode: "normal"             # default
compaction_interval_secs: 0     # default (0 = disabled)
```

`listen` has **no default**: a config without it is refused rather than bound
to an address the operator never chose.

`max_connections`, `wal_enabled`, `sync_mode` and `compaction_interval_secs`
are parsed and validated but not yet acted on. That is pre-existing behaviour
of this server, unchanged by the move into `apr`, and is recorded here rather
than left for a reader to discover.

## Routes

| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/health` | `OK` |
| `GET` | `/status` | JSON: version, data dir, memory cap, loaded row count |
| `POST` | `/query` | JSON: `{ "columns": [...], "rows": [[...]], "row_count": N }` |

`POST /query` takes `{"sql": "..."}`. A SQL parse failure answers `400`; an
execution failure answers `500`.

Every `*.parquet` file in `data_dir` is loaded at startup. The server shuts down
cleanly on `SIGTERM` or Ctrl+C.

## Example

<!-- example-cost: trivial -->
```bash
cat > /tmp/db.yaml <<'YAML'
listen: "127.0.0.1:5433"
data_dir: "/tmp/apr-db-data"
YAML

apr db serve --config /tmp/db.yaml
```

<!-- example-cost: trivial -->
```bash
curl -s localhost:5433/health
curl -s localhost:5433/status
curl -s localhost:5433/query -H 'content-type: application/json' \
     -d '{"sql": "SELECT count(*) FROM events"}'
```

## Full help

Run `apr db serve --help` for the complete option list.

## See also

- Model serving (a different server): [`apr serve`](./serve.md)
- Source: [`crates/aprender-db/src/server.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-db/src/server.rs)
