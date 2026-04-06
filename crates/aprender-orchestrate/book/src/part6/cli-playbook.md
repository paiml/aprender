# `batuta playbook`

Deterministic pipeline orchestration with BLAKE3 content-addressable caching.

## Synopsis

```bash
batuta playbook <COMMAND> [OPTIONS]
```

## Commands

| Command | Description |
|---------|-------------|
| `run` | Execute a playbook pipeline |
| `validate` | Parse, check refs, detect cycles |
| `status` | Show pipeline execution status from lock file |
| `lock` | Display lock file contents |

---

## `batuta playbook run`

Execute a playbook pipeline. Stages run in topological order based on data dependencies (`deps`/`outs` matching) and explicit `after` edges. BLAKE3 hashes determine cache hits; only invalidated stages re-execute.

### Usage

```bash
batuta playbook run <PLAYBOOK_PATH> [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--stages <STAGES>` | Comma-separated list of stages to run (default: all) |
| `--force` | Force re-run, ignoring cache |
| `-p, --param <KEY=VALUE>` | Override a parameter (repeatable) |

### Examples

```bash
# Run all stages
batuta playbook run pipeline.yaml

# Force re-run ignoring cache
batuta playbook run pipeline.yaml --force

# Override parameters
batuta playbook run pipeline.yaml -p model=large -p chunk_size=1024

# Run only specific stages
batuta playbook run pipeline.yaml --stages extract,transcribe
```

### Output

Each stage prints its status:

```
Running playbook: pipeline.yaml
  extract RUNNING (no lock file found)
  extract COMPLETED (1.2s)
  transcribe RUNNING (upstream stage 'extract' was re-run)
  transcribe COMPLETED (3.4s)
  summarize CACHED

Done: 2 run, 1 cached, 0 failed (4.6s)
```

Cache miss reasons are displayed inline:

| Reason | Meaning |
|--------|---------|
| `no lock file found` | First run, no previous cache |
| `cmd_hash changed` | Command text was modified |
| `dep '...' hash changed` | Input file contents changed |
| `params_hash changed` | Parameter values changed |
| `upstream stage '...' was re-run` | A dependency stage was re-executed |
| `forced re-run (--force)` | `--force` flag was passed |
| `stage is frozen` | Stage has `frozen: true` |
| `output '...' is missing` | Expected output file was deleted |

### Lock File

After execution, a `.lock.yaml` file is written alongside the playbook (e.g., `pipeline.lock.yaml`). This file stores per-stage BLAKE3 hashes for cache decisions on subsequent runs. Lock file writes are atomic (temp file + rename) to prevent corruption.

---

## `batuta playbook validate`

Parse and validate a playbook without executing it. Checks structural constraints, template references, and DAG acyclicity.

### Usage

```bash
batuta playbook validate <PLAYBOOK_PATH>
```

### Checks Performed

1. **Schema version** must be `"1.0"`
2. **Name** must not be empty
3. **Stages** must have non-empty `cmd`
4. **`after` references** must point to existing stages (no self-references)
5. **Template references** (`{{params.key}}`, `{{deps[N].path}}`, `{{outs[N].path}}`) must resolve
6. **DAG** must be acyclic (no circular dependencies)
7. **Warnings** for stages with no outputs (always re-run)

### Example

```bash
$ batuta playbook validate pipeline.yaml
Validating: pipeline.yaml
Playbook 'my-pipeline' is valid
  Stages: 5
  Params: 3
```

---

## `batuta playbook status`

Display pipeline execution status from the lock file.

### Usage

```bash
batuta playbook status <PLAYBOOK_PATH>
```

### Example

```bash
$ batuta playbook status pipeline.yaml
Playbook: my-pipeline (pipeline.yaml)
Version: 1.0
Stages: 3

Lock file: batuta 0.7.2 (2026-03-01T12:00:00Z)
------------------------------------------------------------
  extract              COMPLETED    1.2s
  transcribe           COMPLETED    3.4s
  summarize            COMPLETED    0.1s
```

---

## `batuta playbook lock`

Display the raw lock file contents in YAML format.

### Usage

```bash
batuta playbook lock <PLAYBOOK_PATH>
```

---

## Playbook YAML Schema

```yaml
version: "1.0"
name: my-pipeline
params:
  model: "whisper-base"
  chunk_size: 512
targets:
  gpu-box:
    host: "gpu-box.local"
    ssh_user: noah
    cores: 32
    memory_gb: 288
stages:
  extract:
    cmd: "ffmpeg -i {{deps[0].path}} {{outs[0].path}}"
    deps:
      - path: /data/input.mp4
    outs:
      - path: /data/audio.wav
  transcribe:
    cmd: "whisper --model {{params.model}} {{deps[0].path}} > {{outs[0].path}}"
    deps:
      - path: /data/audio.wav
    outs:
      - path: /data/transcript.txt
    params:
      - model
    after:
      - extract
policy:
  failure: stop_on_first    # Jidoka: stop on first error
  validation: checksum       # BLAKE3 content validation
  lock_file: true            # Persist cache state
```

### Template Variables

| Pattern | Resolves to |
|---------|-------------|
| `{{params.key}}` | Global parameter value |
| `{{deps[N].path}}` | Nth dependency path |
| `{{outs[N].path}}` | Nth output path |

### Granular Parameter Invalidation

Stages only invalidate when *their* referenced parameters change. The effective param keys are the union of:

1. Template-extracted refs (`{{params.model}}` in `cmd`)
2. Explicitly declared keys (`params: [model]` on the stage)

A change to `chunk_size` does not invalidate a stage that only references `model`.

### Frozen Stages

Stages with `frozen: true` always report CACHED unless `--force` is passed. Use this for stages whose outputs are committed artifacts that should never be regenerated.

### Execution Policy

| Policy | Options | Default |
|--------|---------|---------|
| `failure` | `stop_on_first`, `continue_independent` | `stop_on_first` |
| `validation` | `checksum`, `none` | `checksum` |
| `lock_file` | `true`, `false` | `true` |

### Event Log

Each run appends timestamped JSONL events to a `.events.jsonl` file alongside the playbook. Events include `run_started`, `stage_started`, `stage_completed`, `stage_cached`, `stage_failed`, `run_completed`, and `run_failed`.
