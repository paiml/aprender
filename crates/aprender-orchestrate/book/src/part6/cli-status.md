# `batuta status`

Show current workflow phase and pipeline progress.

## Synopsis

```bash
batuta status [OPTIONS]
```

## Description

The status command displays the current state of the 5-phase transpilation pipeline, showing which phases are completed, in progress, or pending. It reads the workflow state from the `.batuta-state.json` file.

## Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose output |
| `-h, --help` | Print help |

## Examples

```bash
$ batuta status

📊 Workflow Status

Phase 1: Analysis       ✅ Completed
Phase 2: Transpilation  ✅ Completed
Phase 3: Optimization   ✅ Completed
Phase 4: Validation     🔄 In Progress
Phase 5: Deployment     ⏳ Pending

Overall: 3/5 phases completed
```

## See Also

- [`batuta reset`](./cli-reset.md) - Reset workflow state
- [Workflow State Management](../part5/workflow-state.md)

---

**Previous:** [`batuta report`](./cli-report.md)
**Next:** [`batuta reset`](./cli-reset.md)
