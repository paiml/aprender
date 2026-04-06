# `batuta reset`

Reset workflow state to start the transpilation pipeline from scratch.

## Synopsis

```bash
batuta reset [OPTIONS]
```

## Description

The reset command clears the workflow state file, allowing you to re-run the pipeline from Phase 1. By default, it prompts for confirmation before resetting.

## Options

| Option | Description |
|--------|-------------|
| `--yes` | Skip confirmation prompt |
| `-v, --verbose` | Enable verbose output |
| `-h, --help` | Print help |

## Examples

### Interactive Reset

```bash
$ batuta reset

⚠️  This will reset all workflow state.
Are you sure? (y/N): y

✅ Workflow state reset. Run `batuta analyze` to start over.
```

### Non-Interactive

```bash
$ batuta reset --yes
```

## See Also

- [`batuta status`](./cli-status.md) - Check current state
- [`batuta init`](./cli-init.md) - Re-initialize project

---

**Previous:** [`batuta status`](./cli-status.md)
**Next:** [`batuta oracle`](./cli-oracle.md)
