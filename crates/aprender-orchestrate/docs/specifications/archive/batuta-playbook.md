# Batuta Playbook Specification v1.0

**Status:** Draft
**Authors:** PAIML Engineering
**Date:** 2026-02-16
**Refs:** BATUTA-PLAYBOOK-001
**Ticket Prefix:** BATUTA-PB-XXX

## Abstract

Batuta Playbook introduces deterministic, YAML-defined workflow execution to the Sovereign AI Stack. Playbooks declare multi-stage data pipelines as directed acyclic graphs (DAGs) with content-addressed caching, hash-based invalidation, and distributed execution across heterogeneous machines. The design synthesizes DVC's lock-file determinism [1], Argo's artifact system [2], APR-QA's YAML schema validation [3], Probar's state-machine invariants [4], Snakemake's wildcard expansion and resource-aware scheduling [9], Nextflow's hash-bucketed work directories [10], and Temporal's event-sourced execution log [11] — orchestrated through batuta with repartir execution and pacha artifact tracking.

**Core Invariants:**
- Determinism: identical inputs (deps + params + code) always produce identical outputs
- Idempotency: re-running a playbook skips stages whose input hashes match the lock file
- Resumability: failed pipelines resume from the last successful stage via Parquet checkpoints
- Provenance: every artifact is content-addressed (BLAKE3) with full lineage in pacha
- Shell safety: all `cmd` fields are purified through rash (bashrs) before execution — deterministic, idempotent POSIX sh
- Compliance: pmat (paiml-mcp-agent-toolkit) quality gates enforced at pipeline boundaries — TDG grading, mutation testing, coverage, defect prediction

---

## 1. Introduction

### 1.1 Motivation

The Sovereign AI Stack has compute (repartir), ML (aprender/realizar/whisper-apr), storage (trueno-db, alimentar), and registry (pacha) — but no declarative way to compose them into reproducible multi-stage workflows. Current pain points:

- **No pipeline definition language**: Building a transcription corpus requires hand-written shell scripts with no caching, no resume, no provenance, no safety guarantees
- **No multi-machine orchestration**: repartir can distribute tasks but has no DAG scheduler to express "extract on machine A, transcribe on machine B"
- **No determinism guarantee**: Re-running a pipeline may produce different results if inputs changed silently
- **No artifact lineage**: No record of which inputs produced which outputs at which versions

### 1.2 Design Principles

Following Toyota Way principles:

| Principle | Application |
|-----------|-------------|
| **Genchi Genbutsu** | Hash-based validation — trust checksums, not timestamps |
| **Jidoka** | Stop-on-first-failure by default; never propagate bad data downstream |
| **Muda Elimination** | Skip unchanged stages (DVC-style invalidation); zero wasted compute |
| **Heijunka** | Load-balanced parallel fan-out via repartir work-stealing |
| **Poka-Yoke** | JSON Schema validation of playbook YAML before execution |
| **Kaizen** | Lock files enable incremental pipeline improvement with guaranteed reproducibility |

### 1.3 Prior Art

| System | Strength Adopted | Weakness Avoided |
|--------|-----------------|------------------|
| **DVC** [1] | Hash-based lock files, implicit DAG from deps/outs, params separation | No parallelism, no distributed execution |
| **Argo Workflows** [2] | Typed artifacts, retry with backoff, memoization | YAML verbosity, Kubernetes dependency |
| **Dagster** [5] | Asset-centric thinking, staleness detection | Python-only, no YAML definition |
| **APR-QA** [3] | YAML + JSON Schema, lock files (SHA256), failure policies | Domain-specific to model qualification |
| **Probar** [4] | State machine invariants, setup/steps/teardown lifecycle | Single-threaded, UI-testing domain |
| **Ansible** [6] | Declarative state (`state: present`), `changed_when` | No DAG, no caching |
| **GitHub Actions** [7] | Simple `needs` DAG, clean YAML | Untyped outputs, no content-addressing |
| **Rash (bashrs)** [15] | Shell purification (deterministic, idempotent POSIX sh), security linting | — (sovereign stack component) |
| **PMAT** [16] | TDG grading, quality gates, mutation testing, coverage gaps, defect prediction, doc validation | — (sovereign stack component) |
| **Snakemake** [9] | Wildcard expansion, resource-aware scheduling, checkpoints | Python-embedded DSL, no YAML |
| **Nextflow** [10] | Channel dataflow, hash-bucketed work dirs, resumability | Groovy DSL, opaque work directory tree |
| **Temporal** [11] | Event-sourced execution log, deterministic replay | Heavy runtime (Go/Java), server dependency |
| **Makeflow** [12] | Simple file-based DAG, resource annotations | No content-addressing, no distributed state |
| **CWL** [13] | Portable workflow description, typed I/O | Verbose, no caching |

### 1.4 Scope

**In scope:**
- YAML playbook definition language with JSON Schema
- DAG construction from deps/outs (implicit) and explicit `after` edges
- Content-addressed caching with BLAKE3 lock files
- Multi-machine execution via repartir remote executor
- Parallel fan-out for file-per-job stages
- Wildcard expansion in paths for automatic per-sample DAG generation (Snakemake pattern)
- Resource-aware scheduling with per-stage resource declarations
- Frozen stages that are never re-executed (DVC pattern)
- Append-only execution event log for crash recovery and audit (Temporal pattern)
- Hash-bucketed work directory isolation for parallel stages (Nextflow pattern)
- Artifact registration in pacha with W3C PROV-DM lineage
- CLI: `batuta playbook run`, `status`, `lock`, `validate`, `visualize`

**Out of scope (v1.0):**
- Cron/schedule-based execution (use system cron + `batuta playbook run`)
- Kubernetes/container orchestration (use Argo for K8s workloads)
- GUI/web dashboard (use job-flow TUI via repartir)
- Real-time streaming pipelines (batch-oriented only)
- Dynamic DAG via checkpoints (v2.0 — Snakemake pattern where a checkpoint stage's output determines downstream DAG structure)

---

## 2. YAML Schema

### 2.1 Playbook Definition

```yaml
# playbooks/video-corpus.yaml
version: "1.0"
name: "course-video-transcription-corpus"
description: "Extract, transcribe, chunk, embed, and index 1.6TB of course videos"

# --- Global Parameters ---
# Params are first-class dependencies. Changing a param invalidates only
# stages that reference it (DVC-style granular invalidation).
params:
  whisper_model: "moonshine-tiny"
  chunk_strategy: "semantic"
  chunk_size: 512
  embedding_model: "bge-small-en"
  hnsw_m: 16
  hnsw_ef_construction: 200
  bm25_weight: 0.3
  dense_weight: 0.7

# --- Machine Targets ---
# Named execution targets with connection details.
# Resolved via SSH config (~/.ssh/config) or inline.
targets:
  workstation:
    host: localhost
  intel:
    host: intel                    # resolves via ~/.ssh/config
    ssh_user: noah
    cores: 32
    memory_gb: 288

# --- Pipeline Stages ---
# DAG is implicit: if stage B lists path X in deps and stage A lists
# path X in outs, then B depends on A. No manual dependency declaration
# needed (DVC pattern). Use 'after' for explicit ordering without data deps.
stages:
  extract_audio:
    description: "Extract 16kHz mono WAV from video files"
    cmd: >
      ffmpeg -i {{input}} -vn -ar 16000 -ac 1 -f wav {{output}}
    deps:
      - path: /mnt/nvme-raid0/mac-backup/RecordedCourses/
        type: directory
    outs:
      - path: /mnt/nvme-raid0/corpus/audio/
        type: audio
    target: workstation
    parallel:
      strategy: per_file            # one job per input file
      glob: "**/*.{mp4,mov,mkv,avi,webm}"
      max_workers: 8

  transfer_audio:
    description: "Sync extracted audio to intel server over 10GbE"
    cmd: >
      rsync -av --progress
      -e "ssh -c aes128-gcm@openssh.com -o Compression=no"
      {{deps[0].path}} intel:{{outs[0].path}}
    deps:
      - path: /mnt/nvme-raid0/corpus/audio/
        type: audio
    outs:
      - path: /home/noah/corpus/audio/
        type: audio
        remote: intel
    target: workstation

  transcribe:
    description: "Transcribe audio with Moonshine ASR"
    cmd: >
      whisper-apr transcribe
      --model {{params.whisper_model}}
      --output-json
      {{input}}
    deps:
      - path: /home/noah/corpus/audio/
        type: audio
    params:
      - whisper_model
    outs:
      - path: /home/noah/corpus/transcripts/
        type: transcript
    target: intel
    parallel:
      strategy: per_file
      glob: "*.wav"
      max_workers: 16               # half of 32 threads (whisper uses ~2 threads/file)
    retry:
      limit: 3
      policy: on_failure
      backoff:
        initial: 5s
        factor: 2

  chunk:
    description: "Semantic chunking of transcripts"
    cmd: >
      trueno-rag chunk
      --strategy {{params.chunk_strategy}}
      --size {{params.chunk_size}}
      --input {{deps[0].path}}
      --output {{outs[0].path}}
    deps:
      - path: /home/noah/corpus/transcripts/
        type: transcript
    params:
      - chunk_strategy
      - chunk_size
    outs:
      - path: /home/noah/corpus/chunks/
        type: chunks
    target: intel

  embed:
    description: "Generate dense embeddings for each chunk"
    cmd: >
      aprender embed
      --model {{params.embedding_model}}
      --input {{deps[0].path}}
      --output {{outs[0].path}}
    deps:
      - path: /home/noah/corpus/chunks/
        type: chunks
    params:
      - embedding_model
    outs:
      - path: /home/noah/corpus/embeddings/
        type: embeddings
    target: intel

  index:
    description: "Build HNSW vector index in trueno-db"
    cmd: >
      trueno-db index
      --hnsw-m {{params.hnsw_m}}
      --hnsw-ef {{params.hnsw_ef_construction}}
      --input {{deps[0].path}}
      --output {{outs[0].path}}
    deps:
      - path: /home/noah/corpus/embeddings/
        type: embeddings
    params:
      - hnsw_m
      - hnsw_ef_construction
    outs:
      - path: /home/noah/corpus/vectors.db
        type: vector_store
    target: intel

# --- Compliance Gates (PMAT) ---
compliance:
  pre_flight:
    - tdg:
        path: ../whisper.apr/src/
        min_grade: C                  # ensure ASR engine isn't degraded
        fail_on: violation
    - coverage:
        path: ../whisper.apr/
        min_percent: 85

  post_flight:
    - quality_gate:
        min_grade: B
        fail_on: violation
    - documentation:
        validate_claims: true         # hallucination detection on generated docs

# --- Execution Policy ---
policy:
  failure: stop_on_first            # Jidoka: halt on first stage failure
  validation: checksum              # BLAKE3 verify outputs after each stage
  lock_file: true                   # Generate/update lock file after each stage
```

#### 2.1.1 Wildcard Expansion (Snakemake Pattern)

Paths may contain `{name}` wildcards that expand into per-sample DAG nodes at runtime. This is more expressive than glob-only parallel fan-out: wildcards generate **distinct DAG stages** per matched value, each with independent caching and retry.

```yaml
stages:
  transcribe_{sample}:
    description: "Transcribe a single audio file"
    cmd: >
      whisper-apr transcribe
      --model {{params.whisper_model}}
      --output-json
      /corpus/audio/{sample}.wav
    deps:
      - path: /corpus/audio/{sample}.wav
    outs:
      - path: /corpus/transcripts/{sample}.json
    target: intel
    retry:
      limit: 3
      policy: on_failure
```

**Resolution algorithm:**
1. Scan `deps` for paths containing `{name}` wildcards
2. Glob the filesystem to enumerate concrete values (e.g., `{sample}` → `["lecture-01", "lecture-02", ...]`)
3. Expand stage into N concrete stages, each with its own cache_key in the lock file
4. Insert expanded stages into the DAG in place of the wildcard stage
5. Downstream stages referencing wildcard outputs collect all expanded outputs

**When to use wildcards vs parallel fan-out:**

| Feature | `parallel: per_file` (§4.3) | Wildcards `{sample}` |
|---------|----------------------------|---------------------|
| Granularity | One lock file entry per stage | One lock file entry per sample |
| Caching | Stage-level (all or nothing) | Per-sample (fine-grained) |
| Retry | Per-job within stage | Per-sample stage |
| DAG visibility | Opaque parallel block | Expanded in DAG visualization |
| Use case | Large uniform batches | Heterogeneous processing per sample |

#### 2.1.2 Resource Declarations (Snakemake/Nextflow Pattern)

Each stage may declare resource requirements. The scheduler uses these to bin-pack stages onto targets and cap concurrency to avoid oversubscription.

```yaml
stages:
  transcribe:
    cmd: "whisper-apr transcribe ..."
    resources:
      cores: 2                      # threads per job
      memory_gb: 4                  # RAM per job
      gpu: 1                        # GPU devices per job (0 = CPU only)
      disk_gb: 10                   # scratch disk per job
      timeout: 3600s                # wall-clock timeout per job
    target: intel
```

**Scheduling rules:**
- The scheduler never launches more concurrent jobs than `target.cores / stage.resources.cores`
- If `memory_gb` is declared, total allocated memory across running stages on a target never exceeds `target.memory_gb`
- `gpu` resources are allocated exclusively (no oversubscription)
- `timeout` kills the job with SIGTERM after the specified duration (distinct from retry — timeout counts as a failure)
- If no resources are declared, defaults: `cores: 1`, `memory_gb: 0` (unlimited), `gpu: 0`, `timeout: 0` (unlimited)

#### 2.1.3 Frozen Stages (DVC Pattern)

A stage marked `frozen: true` is **never re-executed**, even if its deps or params change. This is useful for expensive stages whose outputs are manually verified and should not be accidentally invalidated.

```yaml
stages:
  train_baseline:
    cmd: "aprender train --epochs 100 --output model.apr"
    deps:
      - path: /corpus/training-data/
    outs:
      - path: /models/baseline-v1.apr
    frozen: true                     # never re-run, even if deps change
```

**Behavior:**
- `frozen: true` stages are always CACHED regardless of cache_key
- `--force` flag overrides frozen (explicit user intent)
- `batuta playbook status` shows frozen stages with a lock icon: `[FROZEN]`
- Changing `frozen: true → false` allows the stage to participate in normal invalidation
- The lock file preserves the last-known hashes for frozen stages

### 2.2 Lock File

Auto-generated after each successful stage execution. Committed to git as the reproducibility proof.

```yaml
# playbooks/video-corpus.lock.yaml (auto-generated)
schema: "1.0"
playbook: "course-video-transcription-corpus"
generated_at: "2026-02-16T14:30:00Z"
generator: "batuta 0.7.0"
blake3_version: "1.5"

params_hash: "blake3:e4d909c290d0fb1ca068..."   # hash of all params

stages:
  extract_audio:
    status: completed
    started_at: "2026-02-16T12:00:00Z"
    completed_at: "2026-02-16T13:05:22Z"
    duration_seconds: 3922
    target: workstation
    deps:
      - path: /mnt/nvme-raid0/mac-backup/RecordedCourses/
        hash: "blake3:a1b2c3d4e5f6..."
        file_count: 6452
        total_bytes: 1717986918400
    params_hash: "blake3:0000000000000000..."    # no params referenced
    outs:
      - path: /mnt/nvme-raid0/corpus/audio/
        hash: "blake3:f6e5d4c3b2a1..."
        file_count: 6452
        total_bytes: 172000000000
    cmd_hash: "blake3:aabbccdd..."               # hash of resolved cmd template
    cache_key: "blake3:1234567890ab..."           # hash(cmd_hash + deps_hash + params_hash)

  transfer_audio:
    status: completed
    started_at: "2026-02-16T13:05:23Z"
    completed_at: "2026-02-16T13:14:50Z"
    duration_seconds: 567
    target: workstation
    deps:
      - path: /mnt/nvme-raid0/corpus/audio/
        hash: "blake3:f6e5d4c3b2a1..."           # matches extract_audio.outs[0].hash
    params_hash: "blake3:0000000000000000..."
    outs:
      - path: /home/noah/corpus/audio/
        hash: "blake3:f6e5d4c3b2a1..."           # identical (rsync preserves content)
        remote: intel
    cmd_hash: "blake3:eeff0011..."
    cache_key: "blake3:abcdef123456..."

  transcribe:
    status: completed
    started_at: "2026-02-16T13:14:51Z"
    completed_at: "2026-02-16T21:14:51Z"
    duration_seconds: 28800
    target: intel
    deps:
      - path: /home/noah/corpus/audio/
        hash: "blake3:f6e5d4c3b2a1..."
    params_hash: "blake3:789abc..."               # hash of whisper_model="moonshine-tiny"
    outs:
      - path: /home/noah/corpus/transcripts/
        hash: "blake3:112233..."
        file_count: 6452
        total_bytes: 4500000000
    cmd_hash: "blake3:445566..."
    cache_key: "blake3:778899..."
    retries: 0

  chunk:
    status: completed
    started_at: "2026-02-16T21:14:52Z"
    completed_at: "2026-02-16T21:44:52Z"
    duration_seconds: 1800
    target: intel
    deps:
      - path: /home/noah/corpus/transcripts/
        hash: "blake3:112233..."
    params_hash: "blake3:aabb..."                 # hash of chunk_strategy + chunk_size
    outs:
      - path: /home/noah/corpus/chunks/
        hash: "blake3:ccddee..."
        file_count: 385000
        total_bytes: 2100000000
    cmd_hash: "blake3:ff0011..."
    cache_key: "blake3:223344..."

  embed:
    status: completed
    started_at: "2026-02-16T21:44:53Z"
    completed_at: "2026-02-16T23:44:53Z"
    duration_seconds: 7200
    target: intel
    deps:
      - path: /home/noah/corpus/chunks/
        hash: "blake3:ccddee..."
    params_hash: "blake3:5566..."                 # hash of embedding_model
    outs:
      - path: /home/noah/corpus/embeddings/
        hash: "blake3:7788..."
        file_count: 385000
        total_bytes: 58000000000
    cmd_hash: "blake3:9900..."
    cache_key: "blake3:aabb..."

  index:
    status: completed
    started_at: "2026-02-16T23:44:54Z"
    completed_at: "2026-02-17T00:14:54Z"
    duration_seconds: 1800
    target: intel
    deps:
      - path: /home/noah/corpus/embeddings/
        hash: "blake3:7788..."
    params_hash: "blake3:ccdd..."                 # hash of hnsw_m + hnsw_ef_construction
    outs:
      - path: /home/noah/corpus/vectors.db
        hash: "blake3:eeff..."
        total_bytes: 62000000000
    cmd_hash: "blake3:0011..."
    cache_key: "blake3:2233..."
```

### 2.3 Invalidation Rules

A stage is **skipped** if and only if:

```
current_cache_key == lock_file[stage].cache_key
```

Where:

```
cache_key = BLAKE3(cmd_hash || deps_hash || params_hash)
cmd_hash  = BLAKE3(resolved command template after param substitution)
deps_hash = BLAKE3(dep_1_hash || dep_2_hash || ...)
dep_hash  = BLAKE3(file contents)  # for files
          | BLAKE3(dir_listing + file_hashes)  # for directories
params_hash = BLAKE3(sorted(key=value pairs for referenced params only))
```

**Granular param invalidation**: If stage `transcribe` references `whisper_model` and stage `chunk` references `chunk_strategy` + `chunk_size`, then changing `whisper_model` invalidates `transcribe` and all downstream stages — but does NOT invalidate `chunk` if `chunk`'s deps hash hasn't changed (since transcribe's output hash would differ, chunk would see a new deps_hash and also re-run).

This is the DVC model: invalidation propagates through the DAG via content hashes.

### 2.4 Cache Invalidation Explanation (Anti-Nextflow Pattern)

A major pain point in Nextflow is opaque cache misses — users cannot determine *why* a stage re-ran. Batuta MUST log a human-readable explanation for every cache miss.

**CLI output on invalidation:**

```
  [2/6] transcribe (intel)
        cache MISS — invalidation reason:
          ├── params_hash changed: whisper_model "moonshine-tiny" → "moonshine-base"
          └── deps_hash unchanged: /home/noah/corpus/audio/ [blake3:f6e5d4c3]

  [3/6] chunk (intel)
        cache MISS — invalidation reason:
          └── deps_hash changed: /home/noah/corpus/transcripts/ (upstream 'transcribe' re-ran)
```

**Explanation types:**

| Reason | Message |
|--------|---------|
| No lock file | `no lock file found — first run` |
| Stage not in lock | `stage not found in lock file — new stage` |
| cmd changed | `cmd_hash changed (command template modified)` |
| deps changed | `deps_hash changed: {path} [old → new]` |
| params changed | `params_hash changed: {param_name} "{old}" → "{new}"` |
| upstream re-ran | `deps_hash changed: {path} (upstream '{stage}' re-ran)` |
| forced | `--force flag specified` |
| frozen | `FROZEN — skipping despite changes` |

This is stored in the execution event log (§4.7) for post-hoc debugging.

---

## 3. DAG Construction

### 3.1 Implicit DAG from Deps/Outs

The primary DAG construction method. No manual dependency declaration required.

**Algorithm:**

```
1. Build output_map: path → stage_name for all stages' outs
2. For each stage S:
     For each dep D in S.deps:
       If D.path exists in output_map:
         Add edge: output_map[D.path] → S
       Else:
         D is an external input (no upstream stage)
3. Topological sort (Kahn's algorithm from batuta stack/graph.rs)
4. If cycle detected: abort with error listing the cycle
```

**Example from video-corpus.yaml:**

```
extract_audio
  outs: /mnt/nvme-raid0/corpus/audio/
       │
       ▼  (transfer_audio.deps matches extract_audio.outs)
transfer_audio
  outs: /home/noah/corpus/audio/
       │
       ▼  (transcribe.deps matches transfer_audio.outs)
transcribe
  outs: /home/noah/corpus/transcripts/
       │
       ▼  (chunk.deps matches transcribe.outs)
chunk
  outs: /home/noah/corpus/chunks/
       │
       ▼  (embed.deps matches chunk.outs)
embed
  outs: /home/noah/corpus/embeddings/
       │
       ▼  (index.deps matches embed.outs)
index
  outs: /home/noah/corpus/vectors.db
```

### 3.2 Explicit Ordering

For stages that need ordering without data dependencies, use the `after` field:

```yaml
stages:
  notify:
    cmd: "curl -X POST https://hooks.slack.com/... -d '{\"text\": \"Pipeline complete\"}'"
    after:
      - index                     # explicit: run after index, no data dep
```

### 3.3 Parallel Branches

Stages without dependencies (or with independent deps) execute in parallel:

```yaml
stages:
  extract_video_metadata:
    deps:
      - path: /mnt/nvme-raid0/mac-backup/RecordedCourses/
    outs:
      - path: /mnt/nvme-raid0/corpus/metadata.json
    target: workstation

  extract_audio:
    deps:
      - path: /mnt/nvme-raid0/mac-backup/RecordedCourses/
    outs:
      - path: /mnt/nvme-raid0/corpus/audio/
    target: workstation

  # Both stages depend only on external input → execute in parallel
```

---

## 4. Execution Model

### 4.1 Stage Lifecycle

Each stage follows an atomic lifecycle:

```
PENDING → HASHING → CACHED|RUNNING → VALIDATING → COMPLETED|FAILED
              │            │
              │            └── (cache_key matches lock) → CACHED (skip)
              │
              └── Compute BLAKE3 of deps + params + cmd
```

**State transitions:**

| From | To | Trigger |
|------|----|---------|
| PENDING | HASHING | DAG predecessor completed |
| HASHING | CACHED | cache_key matches lock file |
| HASHING | RUNNING | cache_key mismatch or no lock file |
| RUNNING | VALIDATING | cmd exited with code 0 |
| RUNNING | FAILED | cmd exited non-zero (after retries exhausted) |
| VALIDATING | COMPLETED | output hashes computed and lock file updated |
| VALIDATING | FAILED | output validation failed (missing files, etc.) |

### 4.2 Target Dispatch

Stages declare a `target` machine. The executor resolves this:

```rust
match stage.target {
    "localhost" | "workstation" => {
        // Local execution via tokio::process::Command
        execute_local(&stage.cmd).await
    }
    remote_target => {
        // Remote execution via repartir RemoteExecutor (TCP)
        // Falls back to SSH if repartir worker not running
        let executor = RemoteExecutor::connect(
            resolve_target(remote_target)?
        ).await?;
        executor.submit(stage.cmd).await
    }
}
```

**Connection resolution:**

1. Check `targets` section in playbook YAML
2. Fall back to `~/.ssh/config` host lookup
3. Fall back to direct hostname/IP

### 4.3 Parallel Fan-Out

Stages with `parallel` configuration split work across files:

```yaml
parallel:
  strategy: per_file              # one job per matching file
  glob: "**/*.wav"                # file pattern to match in deps directory
  max_workers: 16                 # cap concurrent jobs
```

**Execution:**

```
1. Enumerate files matching glob in deps directory
2. Load .manifest.jsonl from output directory (if exists from prior run)
3. Filter: skip files whose input hash matches a manifest entry (§4.6.2)
4. Create job queue from remaining (unprocessed) files
5. Dispatch to repartir work-stealing pool (max_workers threads)
6. Each job:
   a. Substitute {{input}} and {{output}} in cmd template
   b. Execute cmd
   c. On success: atomically append to .manifest.jsonl
   d. On failure: apply retry policy per-job
7. Wait for all jobs to complete (or fail with partial manifest)
8. Hash entire output directory for lock file
```

See §4.6.2 for per-file idempotency guarantees and resume behavior.

**Template variables for parallel stages:**

| Variable | Meaning |
|----------|---------|
| `{{input}}` | Current input file path |
| `{{output}}` | Derived output file path (same basename, different directory/extension) |
| `{{input_stem}}` | Filename without extension |
| `{{input_dir}}` | Parent directory of input file |

### 4.4 Retry Policy

```yaml
retry:
  limit: 3                        # max retry attempts
  policy: on_failure               # retry on any non-zero exit
  backoff:
    initial: 5s                    # first retry wait
    factor: 2                      # exponential backoff multiplier
    max: 300s                      # cap backoff at 5 minutes
```

**Policies:**

| Policy | Behavior |
|--------|----------|
| `on_failure` | Retry on any non-zero exit code |
| `on_timeout` | Retry only on timeout (not other failures) |
| `on_transient` | Retry on exit codes 1, 75, 137, 143 (transient signals) |
| `never` | No retries (default) |

### 4.5 Failure Policy

Pipeline-level failure handling:

| Policy | Behavior |
|--------|----------|
| `stop_on_first` | Jidoka: halt entire pipeline on first stage failure (default) |
| `continue_on_failure` | Mark failed stage, continue independent branches |
| `collect_all` | Run all possible stages, report failures at end |

### 4.5.1 Shell Compliance via Rash (bashrs)

All `cmd` fields are **mandatory** purified through rash before execution. This eliminates an entire class of non-determinism and safety bugs at the shell layer.

**Pipeline (applied automatically during stage execution):**

```
1. Template resolution: {{params.whisper_model}} → "moonshine-tiny"
2. Rash purification: bash → deterministic, idempotent POSIX sh
3. Execution: purified command runs via tokio::process or repartir
```

**What rash enforces:**
- All variables quoted (prevents word splitting, injection)
- Non-deterministic patterns eliminated (`$RANDOM`, `$$`, timestamps)
- Operations made idempotent (`mkdir` → `mkdir -p`, `ln` → `ln -sf`, `rm` → `rm -f`)
- POSIX compliance (passes shellcheck)
- Deterministic output ordering (sorted globs, stable pipes)

**Example purification:**

```bash
# User writes in playbook cmd:
ffmpeg -i $input -vn -ar 16000 -ac 1 -f wav $output

# Rash purifies to:
ffmpeg -i "${input}" -vn -ar 16000 -ac 1 -f wav "${output}"
```

**Integration point:**

```rust
use rash::purify;

fn execute_stage(stage: &Stage, resolved_cmd: &str) -> Result<()> {
    // Purify through rash before execution
    let purified = purify::transform(resolved_cmd, &purify::Config {
        enforce_posix: true,
        enforce_idempotent: true,
        enforce_deterministic: true,
        shellcheck: true,
    })?;

    // Execute purified command
    execute_cmd(&purified, stage.target.as_deref())
}
```

**Escape hatch**: For commands that legitimately need non-POSIX behavior (rare), stages can opt out:

```yaml
stages:
  special:
    cmd: "some-command-that-needs-bash-extensions"
    shell: raw                      # bypass rash purification (default: "rash")
```

When `shell: raw` is used, rash still **lints** the command and emits warnings, but does not transform it. The event log (§4.7) records that the stage ran unpurified.

**Validation during `batuta playbook validate`:**
- All `cmd` fields are parsed through rash's linter
- Security violations (SEC001-SEC008) are **errors** — playbook fails validation
- Determinism violations (DET rules) are **warnings** unless `shell: raw`
- Idempotency violations (IDEM rules) are **warnings** unless `shell: raw`

---

## 4.6 Idempotency

Runs MUST be idempotent: executing a playbook N times with identical inputs produces the same result as executing it once. This requires guarantees at three levels.

#### 4.6.1 Pipeline-Level Idempotency (Cache-Skip)

If `cache_key` matches the lock file, the stage does not execute. A completed pipeline re-run with unchanged inputs executes zero stages — all hit cache. This is the primary idempotency mechanism.

```
Run 1: extract_audio(RUNNING) → transcribe(RUNNING) → chunk(RUNNING) → ...
Run 2: extract_audio(CACHED)  → transcribe(CACHED)  → chunk(CACHED)  → ...  [0 stages executed]
```

#### 4.6.2 Per-File Idempotency for Parallel Stages

The §4.3 parallel fan-out MUST track completion at the **per-file** level, not just per-stage. This prevents re-processing completed files after a partial failure.

**Manifest file**: Each parallel stage maintains a `.manifest.jsonl` in its output directory. One line per completed file:

```jsonl
{"input":"/corpus/audio/lecture-01.wav","output":"/corpus/transcripts/lecture-01.json","hash":"blake3:aabb...","completed_at":"2026-02-16T14:00:00Z"}
{"input":"/corpus/audio/lecture-02.wav","output":"/corpus/transcripts/lecture-02.json","hash":"blake3:ccdd...","completed_at":"2026-02-16T14:00:05Z"}
```

**Execution with manifest (revised §4.3 step 1-6):**

```
1. Enumerate files matching glob in deps directory
2. Load manifest (if exists) from previous partial run
3. Filter out files whose input hash matches manifest entry → already done
4. Create job queue from remaining files only
5. Dispatch to repartir work-stealing pool
6. Each completed job appends to manifest atomically
7. On full completion: hash entire output directory for lock file
8. On partial failure: manifest preserves progress for resume
```

**Resume behavior**: If `extract_audio` completes 3,000 of 6,452 files then crashes:
- Re-run loads manifest → 3,000 entries
- Hashes the 3,000 input files → all match manifest
- Creates job queue for remaining 3,452 files only
- Resumes exactly where it left off

**Manifest invalidation**: If a dep file's content hash changes (video was re-encoded), the manifest entry for that file is invalidated and the file is re-processed.

#### 4.6.3 Command Idempotency

The playbook system cannot enforce that arbitrary commands are idempotent — but it provides mechanisms to detect and handle non-determinism:

**Deterministic commands** (preferred):
- `ffmpeg` with fixed parameters: deterministic (same input → same output bytes)
- `whisper-apr` with fixed model + seed: deterministic
- `rsync`: idempotent by design (only transfers differences)

**Non-deterministic commands** (must be declared):
```yaml
stages:
  train:
    cmd: "aprender train --epochs 10 {input}"
    deterministic: false            # explicit declaration
    # When deterministic=false:
    # - Stage always re-runs (never cached)
    # - Output hash is recorded but not used for cache_key
    # - Downstream stages use output hash for their cache_key
    #   (so they re-run only if training actually produced different output)
```

Default is `deterministic: true`. If a stage is marked `deterministic: true` but produces different output hashes across runs with identical inputs, the executor emits a warning:

```
⚠️  Stage 'transcribe' is marked deterministic but produced different output hash:
    Previous: blake3:aabb...
    Current:  blake3:ccdd...
    Consider setting 'deterministic: false' or fixing the non-determinism.
```

#### 4.6.4 Idempotency Invariants

The following invariants MUST hold:

| Invariant | Description |
|-----------|-------------|
| **I1** | Running a fully-cached playbook executes zero commands |
| **I2** | A partial parallel failure resumes from the last incomplete file, not from the beginning |
| **I3** | Changing param P only invalidates stages that declare P in their `params` list, plus their transitive downstream |
| **I4** | Two concurrent runs of the same playbook produce the same final state (lock file arbitration via file locks) |
| **I5** | `--force` on a stage re-runs it and all downstream, but does not affect upstream |
| **I6** | Adding a new stage to the YAML does not invalidate existing cached stages |

#### 4.6.5 Concurrent Run Safety

Multiple `batuta playbook run` invocations on the same playbook MUST be safe:

- Lock file writes use `flock(2)` advisory locks (or platform equivalent)
- Manifest appends are atomic (write to temp file, rename)
- If a run detects another run in progress, it either:
  - Waits for the lock (default: `concurrency: wait`)
  - Aborts with error (`concurrency: fail`)
  - Declared in policy:

```yaml
policy:
  concurrency: wait               # wait | fail
```

### 4.7 Execution Event Log (Temporal Pattern)

Every playbook run produces an append-only event log. This serves three purposes:
1. **Crash recovery**: Replay the log to determine pipeline state without re-hashing
2. **Audit trail**: Full history of what ran, when, why, and what changed
3. **Debugging**: Answer "why did stage X re-run?" after the fact (§2.4)

**Log format**: JSONL (one event per line), stored alongside the lock file:

```
playbooks/video-corpus.events.jsonl
```

**Event types:**

```jsonl
{"ts":"2026-02-16T14:00:00Z","event":"run_started","playbook":"course-video-transcription-corpus","run_id":"r-a1b2c3","batuta_version":"0.7.0"}
{"ts":"2026-02-16T14:00:01Z","event":"stage_cached","stage":"extract_audio","cache_key":"blake3:1234...","reason":"cache_key matches lock"}
{"ts":"2026-02-16T14:00:01Z","event":"stage_started","stage":"transcribe","target":"intel","cache_miss_reason":"params_hash changed: whisper_model \"moonshine-tiny\" → \"moonshine-base\""}
{"ts":"2026-02-16T22:00:01Z","event":"stage_completed","stage":"transcribe","duration_seconds":28800,"outs_hash":"blake3:5566..."}
{"ts":"2026-02-16T22:00:02Z","event":"stage_failed","stage":"embed","exit_code":137,"retry_attempt":1,"error":"OOM killed"}
{"ts":"2026-02-16T22:00:02Z","event":"stage_retry","stage":"embed","attempt":2,"backoff_seconds":10}
{"ts":"2026-02-16T23:00:00Z","event":"run_completed","run_id":"r-a1b2c3","stages_run":4,"stages_cached":2,"stages_failed":0,"total_seconds":32400}
```

**Rust types:**

```rust
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum PipelineEvent {
    RunStarted { playbook: String, run_id: String, batuta_version: String },
    StageCached { stage: String, cache_key: String, reason: String },
    StageStarted { stage: String, target: String, cache_miss_reason: String },
    StageCompleted { stage: String, duration_seconds: u64, outs_hash: String },
    StageFailed { stage: String, exit_code: i32, retry_attempt: u32, error: String },
    StageRetry { stage: String, attempt: u32, backoff_seconds: u64 },
    RunCompleted { run_id: String, stages_run: u32, stages_cached: u32, stages_failed: u32, total_seconds: u64 },
}
```

**Log rotation**: The event log grows monotonically. A `batuta playbook log --compact` command prunes old run events, keeping only the latest N runs (default: 10).

### 4.8 Work Directory Isolation (Nextflow Pattern)

Parallel stages execute each job in an isolated, hash-bucketed work directory. This prevents jobs from interfering with each other and enables clean resumability.

**Work directory structure:**

```
.batuta/work/
├── ab/
│   └── cd1234ef5678.../           # BLAKE3(stage_name + input_path)[0:12]
│       ├── .command.sh            # resolved command for this job
│       ├── .exitcode              # exit code (written atomically on completion)
│       ├── .stdout                # captured stdout
│       ├── .stderr                # captured stderr
│       └── output/                # symlinked or moved to final outs path on success
├── ef/
│   └── 9876543210ab.../
│       └── ...
```

**Algorithm:**
1. For each parallel job, compute `work_hash = BLAKE3(stage_name || input_file_path)`
2. Create work directory: `.batuta/work/{work_hash[0:2]}/{work_hash[2:12]}/`
3. Write `.command.sh` with the resolved command
4. Execute command with `cwd` set to the work directory
5. Capture stdout/stderr to `.stdout`/`.stderr`
6. On success: write `.exitcode`, move/symlink outputs to final path, append to manifest
7. On failure: write `.exitcode`, leave work dir intact for debugging

**Benefits:**
- Jobs cannot accidentally overwrite each other's intermediate files
- Failed jobs leave their work directory intact for `batuta playbook debug --stage transcribe --job lecture-42`
- Successful work directories can be cleaned: `batuta playbook clean --keep-latest`
- Work directory hash enables instant lookup: given an input file, find its work directory in O(1)

**Cleanup policy:**

```yaml
policy:
  work_dir: .batuta/work            # default location
  clean_on_success: false           # keep work dirs after success (default)
  # batuta playbook clean removes work dirs for completed stages
```

### 4.9 Compliance Gates via PMAT (paiml-mcp-agent-toolkit)

Playbooks may declare **compliance gates** — quality checks enforced between stages using pmat. Gates are first-class pipeline citizens: they produce artifacts, participate in caching, and block downstream stages on failure.

#### 4.9.1 Gate Declaration

```yaml
stages:
  transcribe:
    cmd: "whisper-apr transcribe --model {{params.whisper_model}} {{input}}"
    deps:
      - path: /corpus/audio/
    outs:
      - path: /corpus/transcripts/
    target: intel

  # Compliance gate: validate transcription quality before chunking
  validate_transcripts:
    gate: true                       # marks this as a compliance gate
    cmd: >
      pmat quality-gate
      --path {{deps[0].path}}
      --min-grade B
      --fail-on-violation
      --output-format json
      --output {{outs[0].path}}
    deps:
      - path: /corpus/transcripts/
    outs:
      - path: /corpus/reports/transcript-quality.json
        type: metrics
    target: intel

  chunk:
    deps:
      - path: /corpus/transcripts/   # data dependency
      - path: /corpus/reports/transcript-quality.json  # gate dependency
    outs:
      - path: /corpus/chunks/
```

#### 4.9.2 Built-in Gate Types

Playbooks support shorthand for common pmat compliance checks via the `compliance` block:

```yaml
# --- Pipeline-Level Compliance ---
compliance:
  # Applied before pipeline starts
  pre_flight:
    - tdg:
        path: ../whisper.apr/src/
        min_grade: B
        fail_on: violation
    - coverage:
        path: ../whisper.apr/
        min_percent: 85

  # Applied after pipeline completes (on all output artifacts)
  post_flight:
    - quality_gate:
        min_grade: B
        fail_on: violation
    - mutation:
        threshold: 80               # ≥80% mutation kill rate
    - defect_prediction:
        max_severity: medium         # block on high/critical defects
    - documentation:
        validate_claims: true        # detect hallucinations via Semantic Entropy
```

**Expansion**: The `compliance` block expands into synthetic stages injected into the DAG:

```
compliance.pre_flight → [user stages] → compliance.post_flight
```

Pre-flight gates run before any user stage. Post-flight gates run after all user stages complete. Each gate is a normal stage with caching, retry, and event logging.

#### 4.9.3 Available PMAT Checks

| Check | pmat Command | What It Validates |
|-------|-------------|-------------------|
| `tdg` | `pmat analyze tdg` | Technical Debt Grade (A+ to F, 6 orthogonal metrics) |
| `quality_gate` | `pmat quality-gate` | Composite quality check (TDG + coverage + complexity) |
| `mutation` | `pmat mutate` | Test suite quality via mutation testing (kill rate %) |
| `coverage` | `pmat query --coverage-gaps` | Line coverage and ROI-ranked gap analysis |
| `complexity` | `pmat analyze complexity` | Cyclomatic complexity bounds |
| `defect_prediction` | `pmat analyze defect-prediction` | Fault patterns (unwrap, panic, unsafe, boundary) |
| `dead_code` | `pmat analyze dead-code` | Unused functions, types, imports |
| `duplication` | `pmat analyze duplication` | Code clones via MinHash + LSH |
| `documentation` | `pmat validate_documentation` | Hallucination detection, broken refs, claim validation |
| `satd` | `pmat analyze satd` | Self-admitted technical debt (TODO, FIXME, HACK) |

#### 4.9.4 Gate Behavior

- **Caching**: Gates are cached like any stage. If deps haven't changed, the gate is skipped. This prevents redundant quality scans on unchanged code.
- **Failure**: A failed gate follows the pipeline failure policy (§4.5). With `stop_on_first` (default), a failed gate halts the pipeline — Jidoka.
- **Artifacts**: Gate output (JSON reports) is registered in pacha as `type: metrics`. Not cached for invalidation purposes (metrics are informational, not data dependencies).
- **Event log**: Gate results are recorded in the execution event log (§4.7):

```jsonl
{"ts":"2026-02-16T14:00:02Z","event":"gate_passed","stage":"validate_transcripts","gate":"quality_gate","grade":"B+","details":{"tdg":"B+","coverage":92.1,"mutation_kill_rate":87.3}}
{"ts":"2026-02-16T14:00:03Z","event":"gate_failed","stage":"pre_flight_tdg","gate":"tdg","grade":"D","min_required":"B","violations":["src/parser.rs: F (complexity=42)","src/utils.rs: D (duplication=0.35)"]}
```

#### 4.9.5 MCP Integration

For interactive development, pmat's 19 MCP tools are available to AI agents working alongside batuta. When an agent (Claude Code, Cline) is developing pipeline stages, it can query pmat in real-time:

```
Agent → pmat MCP → analyze_technical_debt(file)  → TDG grade
Agent → pmat MCP → quality_gate(path)             → pass/fail
Agent → pmat MCP → mutation_test(target)           → kill rate
Agent → pmat MCP → validate_documentation(doc)     → hallucination check
```

This creates a feedback loop: the agent writes code, pmat validates it, the agent fixes issues before they reach the pipeline gate.

---

## 5. Artifact Tracking

### 5.1 Pacha Integration

Every stage output is registered in pacha as a versioned artifact:

```rust
// After stage completion
let artifact = pacha::Artifact {
    name: format!("{}/{}", playbook.name, stage.name),
    version: SemanticVersion::from_hash(&outs_hash),
    content_hash: outs_hash,            // BLAKE3
    content_size: total_bytes,
    artifact_type: stage.outs[0].type,  // "audio", "transcript", etc.
    metadata: json!({
        "playbook": playbook.name,
        "stage": stage.name,
        "target": stage.target,
        "params": stage.resolved_params,
        "duration_seconds": duration,
    }),
};
registry.register_artifact(&artifact)?;
```

### 5.2 Lineage Graph

W3C PROV-DM lineage recorded in pacha:

```
Entity(corpus/audio)
  wasGeneratedBy(extract_audio)
  wasDerivedFrom(RecordedCourses)

Entity(corpus/transcripts)
  wasGeneratedBy(transcribe)
  wasDerivedFrom(corpus/audio)
  wasAttributedTo(whisper-apr/moonshine-tiny)

Entity(corpus/chunks)
  wasGeneratedBy(chunk)
  wasDerivedFrom(corpus/transcripts)

Entity(corpus/embeddings)
  wasGeneratedBy(embed)
  wasDerivedFrom(corpus/chunks)
  wasAttributedTo(aprender/bge-small-en)

Entity(corpus/vectors.db)
  wasGeneratedBy(index)
  wasDerivedFrom(corpus/embeddings)
```

### 5.3 Artifact Types

| Type | Description | Content |
|------|-------------|---------|
| `directory` | Generic directory | Any files |
| `audio` | Audio files | WAV, FLAC, MP3 |
| `transcript` | ASR output | JSON with timestamps, text |
| `chunks` | Chunked text | JSON with chunk boundaries |
| `embeddings` | Dense vectors | Binary f32 arrays or .npy |
| `vector_store` | HNSW index | trueno-db format |
| `model` | ML model | .apr, .gguf, .safetensors |
| `dataset` | Training data | .ald, .parquet, .csv |
| `metrics` | Evaluation results | JSON (not cached) |

### 5.4 Git-Backed Artifact Storage

Pipeline artifacts split into two tiers: **metadata** (git-tracked) and **blobs** (local-only). The lock file bridges them — it contains BLAKE3 hashes that prove blob integrity without storing blobs in git.

#### 5.4.1 What Lives in Git

```
playbooks/
├── video-corpus.yaml              # pipeline definition (authored)
├── video-corpus.lock.yaml         # reproducibility proof — all BLAKE3 hashes
├── video-corpus.events.jsonl      # execution audit trail
└── reports/
    ├── transcript-quality.json    # pmat compliance gate output
    └── post-flight.json           # post-flight quality report
```

These files are small (KB–low MB), text-based, and diffable. They are the **intellectual property record**: what ran, when, with what inputs, what quality gates passed, and what hashes the outputs produced.

**Commit policy**: `batuta playbook run` auto-stages metadata files after successful completion. The user commits explicitly (batuta never auto-commits).

```
$ batuta playbook run playbooks/video-corpus.yaml
  ...
  ✓ Pipeline complete. 6 stages, 0 failures, 43,200s total.

  Git-trackable files updated:
    modified: playbooks/video-corpus.lock.yaml
    modified: playbooks/video-corpus.events.jsonl
    new:      playbooks/reports/transcript-quality.json

  Run 'git add playbooks/ && git commit' to version this run.
```

#### 5.4.2 What Stays Local-Only

```
/home/noah/corpus/                 # on intel (288GB RAM, 3.6TB NVMe)
├── audio/                         # 172 GB — 6,452 WAV files
├── transcripts/                   # 4.5 GB — 6,452 JSON files
├── chunks/                        # 2.1 GB — 385K chunk files
├── embeddings/                    # 58 GB — 385K f32 vectors
└── vectors.db                     # 62 GB — HNSW index

.batuta/work/                      # hash-bucketed work directories (§4.8)
```

These are large binary artifacts. They are **never** pushed to GitHub. Their integrity is proven by the BLAKE3 hashes in the lock file.

#### 5.4.3 Verification Without Blobs

Any machine with the lock file can verify local blobs match:

```bash
# Verify all artifacts match lock file hashes
batuta playbook lock playbooks/video-corpus.yaml --verify

  [1/6] extract_audio: /mnt/nvme-raid0/corpus/audio/
        lock: blake3:f6e5d4c3b2a1...  local: blake3:f6e5d4c3b2a1...  ✓
  [2/6] transfer_audio: intel:/home/noah/corpus/audio/
        lock: blake3:f6e5d4c3b2a1...  local: blake3:f6e5d4c3b2a1...  ✓
  ...
  ✓ All 6 stages verified. Corpus integrity confirmed.
```

If a blob is missing or corrupted, `--verify` reports exactly which stage output is affected:

```
  [3/6] transcribe: intel:/home/noah/corpus/transcripts/
        lock: blake3:112233...  local: blake3:aabbcc...  ✗ MISMATCH
        → 3 files differ. Run 'batuta playbook run --stage transcribe --force' to regenerate.
```

#### 5.4.4 .gitignore Integration

`batuta playbook init` generates a `.gitignore` alongside the playbook:

```gitignore
# Batuta: large binary artifacts (local-only, verified via lock file)
/corpus/
.batuta/work/

# Batuta: track these (metadata, proofs, reports)
!playbooks/*.yaml
!playbooks/*.lock.yaml
!playbooks/*.events.jsonl
!playbooks/reports/
```

#### 5.4.5 Storage Tiers

| Tier | Location | Backed by | Size | Examples |
|------|----------|-----------|------|----------|
| **Git** | GitHub repo | git + remote | KB–MB | `.yaml`, `.lock.yaml`, `.events.jsonl`, compliance JSON |
| **Local** | NVMe on target machine | filesystem + BLAKE3 proof | GB–TB | audio, transcripts, embeddings, vectors, models |
| **Pacha** | Local registry | BLAKE3 + W3C PROV-DM | KB | artifact metadata, lineage graph, version history |

The lock file is the bridge: it lives in git and contains the BLAKE3 hashes that prove local blob integrity. Pacha adds lineage (which input produced which output) on top.

#### 5.4.6 Disaster Recovery

If the intel server's NVMe dies, recovery is deterministic:

1. Lock file in git has every hash → you know exactly what existed
2. Source videos on workstation are untouched (read-only deps)
3. `batuta playbook run` on a fresh machine re-derives everything from source
4. Lock file `--verify` confirms the regenerated corpus matches the original hashes (for deterministic stages)

No cloud backup needed. The pipeline definition *is* the backup — the blobs are derived artifacts.

---

## 6. CLI Interface

### 6.1 Commands

```bash
# Initialize playbook directory with .gitignore
batuta playbook init playbooks/video-corpus.yaml

# Run a playbook (full pipeline or resume from lock file)
batuta playbook run playbooks/video-corpus.yaml

# Run a specific stage only (and its upstream deps if needed)
batuta playbook run playbooks/video-corpus.yaml --stage transcribe

# Run with parameter override (does not modify YAML file)
batuta playbook run playbooks/video-corpus.yaml --set whisper_model=moonshine-base

# Dry run: show DAG, hash comparison, what would execute
batuta playbook run playbooks/video-corpus.yaml --dry-run

# Validate playbook YAML against schema (poka-yoke)
batuta playbook validate playbooks/video-corpus.yaml

# Show pipeline status from lock file
batuta playbook status playbooks/video-corpus.yaml

# Visualize DAG (terminal or SVG)
batuta playbook visualize playbooks/video-corpus.yaml
batuta playbook visualize playbooks/video-corpus.yaml --format svg -o pipeline.svg

# Force re-run a stage (ignore cache)
batuta playbook run playbooks/video-corpus.yaml --stage transcribe --force

# Lock file management
batuta playbook lock playbooks/video-corpus.yaml          # generate/update lock
batuta playbook lock playbooks/video-corpus.yaml --verify  # verify lock matches current state

# Show artifact lineage
batuta playbook lineage playbooks/video-corpus.yaml

# Monitor execution (repartir job-flow TUI)
batuta playbook run playbooks/video-corpus.yaml --tui

# View execution event log
batuta playbook log playbooks/video-corpus.yaml                  # last run
batuta playbook log playbooks/video-corpus.yaml --all            # all runs
batuta playbook log playbooks/video-corpus.yaml --compact --keep 5  # prune old runs

# Debug a failed parallel job via work directory
batuta playbook debug playbooks/video-corpus.yaml --stage transcribe --job lecture-42.wav

# Clean work directories for completed stages
batuta playbook clean playbooks/video-corpus.yaml
batuta playbook clean playbooks/video-corpus.yaml --keep-latest   # keep most recent work dirs
```

### 6.2 Output Format

```
$ batuta playbook run playbooks/video-corpus.yaml

  Batuta Playbook v1.0 — course-video-transcription-corpus
  ═══════════════════════════════════════════════════════════

  DAG: extract_audio → transfer_audio → transcribe → chunk → embed → index

  [1/6] extract_audio (workstation)
        deps: /mnt/nvme-raid0/mac-backup/RecordedCourses/ [blake3:a1b2c3]
        RUNNING — 6,452 files, 8 workers
        ████████████████████░░░░░░░░ 72% (4,645/6,452) — 48:22 elapsed, ~18:30 remaining

  [2/6] transfer_audio (workstation → intel)
        PENDING — waiting for extract_audio

  [3/6] transcribe (intel)
        CACHED — cache_key blake3:778899 matches lock file ✓

  ...
```

### 6.3 TUI Mode

When `--tui` is passed, launch repartir's job-flow TUI with playbook-aware rendering:

```
┌─ Batuta Playbook: course-video-transcription-corpus ────────────────────────┐
│                                                                              │
│  Stage            Target       Status      Progress    Duration   Workers    │
│  ─────────────    ──────────   ────────    ────────    ────────   ───────    │
│  extract_audio    workstation  RUNNING     72%         48:22      8/8        │
│  transfer_audio   workstation  PENDING     —           —          —          │
│  transcribe       intel        CACHED      100%        —          —          │
│  chunk            intel        PENDING     —           —          —          │
│  embed            intel        PENDING     —           —          —          │
│  index            intel        PENDING     —           —          —          │
│                                                                              │
│  Nodes: workstation ● online (load: 0.85)  intel ● online (load: 0.12)      │
│  Cache hits: 1/6    Estimated remaining: 9h 22m                              │
│                                                                              │
│  [q]uit  [r]etry failed  [↑↓] select stage  [Enter] show logs               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Architecture

### 7.1 Component Integration

```
batuta playbook run video-corpus.yaml
  │
  ├── [1] Parse & Validate
  │     ├── serde_yaml::from_str (YAML parsing)
  │     ├── JSON Schema validation (poka-yoke, from APR-QA pattern)
  │     ├── Param resolution & template expansion
  │     └── Rash purification of all cmd fields (§4.5.1)
  │
  ├── [2] Build DAG
  │     ├── Expand compliance.pre_flight/post_flight into synthetic stages
  │     ├── Construct output_map from all stages' outs
  │     ├── Match deps against output_map → edges
  │     ├── Add explicit 'after' edges
  │     ├── Topological sort (batuta stack/graph.rs — Kahn's algorithm)
  │     └── Cycle detection → abort with error
  │
  ├── [3] Hash & Invalidate
  │     ├── Load lock file (if exists)
  │     ├── For each stage in topological order:
  │     │     ├── Compute deps_hash (BLAKE3 of dep files/directories)
  │     │     ├── Compute params_hash (BLAKE3 of referenced param values)
  │     │     ├── Compute cmd_hash (BLAKE3 of resolved command)
  │     │     ├── cache_key = BLAKE3(cmd_hash || deps_hash || params_hash)
  │     │     └── Compare against lock_file[stage].cache_key
  │     └── Mark stages as CACHED or PENDING
  │
  ├── [4] Execute (topological order, parallel where independent)
  │     ├── For each PENDING stage:
  │     │     ├── Resolve target → local or repartir RemoteExecutor
  │     │     ├── If parallel: fan-out via repartir work-stealing pool
  │     │     ├── Execute cmd (tokio::process or repartir TCP)
  │     │     ├── Apply retry policy on failure
  │     │     ├── On success: hash outputs, update lock file
  │     │     ├── Register artifact in pacha (BLAKE3, lineage)
  │     │     └── On failure: apply pipeline failure policy (Jidoka)
  │     └── Checkpoint after each stage (repartir Parquet)
  │
  └── [5] Finalize
        ├── Write final lock file
        ├── Register pipeline run in pacha (experiment tracking)
        ├── Print summary (stages run, cached, failed, duration)
        └── Exit code: 0 (success) or 1 (failure)
```

### 7.2 Rust Types

```rust
// --- Playbook YAML types ---

#[derive(Debug, Deserialize, Serialize)]
pub struct Playbook {
    pub version: String,
    pub name: String,
    pub description: Option<String>,
    pub params: HashMap<String, serde_yaml::Value>,
    pub targets: HashMap<String, Target>,
    pub stages: IndexMap<String, Stage>,    // preserve YAML order
    pub policy: Policy,
    pub compliance: Option<Compliance>,     // §4.9
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Target {
    pub host: String,
    pub ssh_user: Option<String>,
    pub cores: Option<u32>,
    pub memory_gb: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Stage {
    pub description: Option<String>,
    pub cmd: String,
    pub deps: Vec<Dependency>,
    pub outs: Vec<Output>,
    pub params: Option<Vec<String>>,        // param keys this stage depends on
    pub target: Option<String>,              // default: localhost
    pub parallel: Option<ParallelConfig>,
    pub retry: Option<RetryConfig>,
    pub after: Option<Vec<String>>,          // explicit ordering edges
    pub deterministic: Option<bool>,          // default: true (§4.6.3)
    pub frozen: Option<bool>,                 // default: false (§2.1.3)
    pub resources: Option<ResourceConfig>,    // §2.1.2
    pub shell: Option<ShellMode>,             // default: Rash (§4.5.1)
    pub gate: Option<bool>,                   // compliance gate stage (§4.9)
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ShellMode {
    Rash,                                      // purify through rash (default)
    Raw,                                       // bypass purification, lint-only
}

// --- Compliance gates (§4.9) ---

#[derive(Debug, Deserialize, Serialize)]
pub struct Compliance {
    pub pre_flight: Option<Vec<ComplianceCheck>>,
    pub post_flight: Option<Vec<ComplianceCheck>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ComplianceCheck {
    Tdg { path: PathBuf, min_grade: String, fail_on: String },
    QualityGate { min_grade: String, fail_on: String },
    Mutation { threshold: u32 },
    Coverage { path: PathBuf, min_percent: u32 },
    Complexity { max_cyclomatic: u32 },
    DefectPrediction { max_severity: String },
    DeadCode { fail_on: String },
    Duplication { max_ratio: f64 },
    Documentation { validate_claims: bool },
    Satd { max_count: Option<u32> },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Dependency {
    pub path: PathBuf,
    #[serde(rename = "type")]
    pub dep_type: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Output {
    pub path: PathBuf,
    #[serde(rename = "type")]
    pub out_type: Option<String>,
    pub remote: Option<String>,              // target name if output is on remote
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ResourceConfig {
    pub cores: Option<u32>,                   // threads per job (default: 1)
    pub memory_gb: Option<u32>,               // RAM per job (default: unlimited)
    pub gpu: Option<u32>,                     // GPU devices per job (default: 0)
    pub disk_gb: Option<u32>,                 // scratch disk per job
    pub timeout: Option<String>,              // wall-clock timeout (e.g. "3600s")
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ParallelConfig {
    pub strategy: ParallelStrategy,
    pub glob: Option<String>,
    pub max_workers: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelStrategy {
    PerFile,                                  // one job per matched file
    Chunked { chunk_size: usize },           // N files per job
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RetryConfig {
    pub limit: u32,
    pub policy: RetryPolicy,
    pub backoff: Option<BackoffConfig>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RetryPolicy {
    OnFailure,
    OnTimeout,
    OnTransient,
    Never,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct BackoffConfig {
    pub initial: String,                      // e.g. "5s"
    pub factor: u32,
    pub max: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Policy {
    pub failure: FailurePolicy,
    pub validation: ValidationPolicy,
    pub lock_file: bool,
    pub concurrency: ConcurrencyPolicy,       // §4.6.5
    pub work_dir: Option<PathBuf>,            // default: .batuta/work (§4.8)
    pub clean_on_success: Option<bool>,       // default: false (§4.8)
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConcurrencyPolicy {
    Wait,                                      // wait for lock (default)
    Fail,                                      // abort if another run active
}

// --- Per-file manifest for parallel stages (§4.6.2) ---

#[derive(Debug, Deserialize, Serialize)]
pub struct ManifestEntry {
    pub input: PathBuf,
    pub output: PathBuf,
    pub input_hash: String,                    // BLAKE3 of input file
    pub output_hash: String,                   // BLAKE3 of output file
    pub completed_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailurePolicy {
    StopOnFirst,                              // Jidoka (default)
    ContinueOnFailure,
    CollectAll,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationPolicy {
    Checksum,                                 // BLAKE3 verify outputs
    None,
}

// --- Lock file types ---

#[derive(Debug, Deserialize, Serialize)]
pub struct LockFile {
    pub schema: String,
    pub playbook: String,
    pub generated_at: DateTime<Utc>,
    pub generator: String,
    pub params_hash: String,
    pub stages: IndexMap<String, StageLock>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StageLock {
    pub status: StageStatus,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub duration_seconds: Option<u64>,
    pub target: String,
    pub deps: Vec<DepLock>,
    pub params_hash: String,
    pub outs: Vec<OutLock>,
    pub cmd_hash: String,
    pub cache_key: String,
    pub retries: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DepLock {
    pub path: PathBuf,
    pub hash: String,
    pub file_count: Option<u64>,
    pub total_bytes: Option<u64>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OutLock {
    pub path: PathBuf,
    pub hash: String,
    pub file_count: Option<u64>,
    pub total_bytes: Option<u64>,
    pub remote: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    Pending,
    Hashing,
    Cached,
    Running,
    Validating,
    Completed,
    Failed,
}
```

### 7.3 Module Structure

```
src/
├── playbook/
│   ├── mod.rs              # Public API: run(), validate(), status()
│   ├── types.rs            # Playbook, Stage, LockFile, PipelineEvent (§7.2)
│   ├── parser.rs           # YAML parsing + JSON Schema validation
│   ├── dag.rs              # DAG construction from deps/outs (§3)
│   ├── wildcard.rs         # {name} wildcard expansion into concrete stages (§2.1.1)
│   ├── hasher.rs           # BLAKE3 hashing for files, dirs, params
│   ├── cache.rs            # Lock file load/compare/update + invalidation explanation (§2.3, §2.4)
│   ├── executor.rs         # Stage execution with target dispatch (§4.2)
│   ├── scheduler.rs        # Resource-aware stage scheduling (§2.1.2)
│   ├── parallel.rs         # Per-file fan-out via repartir (§4.3)
│   ├── workdir.rs          # Hash-bucketed work directory isolation (§4.8)
│   ├── eventlog.rs         # Append-only execution event log (§4.7)
│   ├── template.rs         # {{param}} and {{input}}/{{output}} resolution
│   ├── shell.rs            # Rash purification + ShellMode dispatch (§4.5.1)
│   ├── lineage.rs          # Pacha artifact registration (§5)
│   ├── compliance.rs       # PMAT quality gate expansion + execution (§4.9)
│   └── schema.yaml         # JSON Schema for playbook validation
├── cli/
│   └── playbook.rs         # CLI subcommands (§6)
```

---

## 8. Implementation Phases

### Phase 1: Core (BATUTA-PB-001 through PB-005)

| Ticket | Description | Deps |
|--------|-------------|------|
| BATUTA-PB-001 | YAML parser + types.rs (§2.1, §7.2) | serde_yaml |
| BATUTA-PB-002 | DAG builder from deps/outs (§3.1) | stack/graph.rs |
| BATUTA-PB-003 | BLAKE3 hasher for files/dirs/params (§2.3) | blake3 crate |
| BATUTA-PB-004 | Lock file read/write/compare with invalidation explanation (§2.2, §2.4) | PB-003 |
| BATUTA-PB-005 | Local stage executor with template resolution (§4.1, §4.2) | PB-001..004 |

**Milestone:** `batuta playbook run` works for single-machine, sequential pipelines with cache miss explanations.

### Phase 2: Distribution (BATUTA-PB-006 through PB-010)

| Ticket | Description | Deps |
|--------|-------------|------|
| BATUTA-PB-006 | Remote target dispatch via repartir TCP (§4.2) | repartir |
| BATUTA-PB-007 | SSH fallback for targets without repartir worker (§4.2) | PB-006 |
| BATUTA-PB-008 | Parallel fan-out with per_file strategy (§4.3) | repartir pool |
| BATUTA-PB-009 | Retry with exponential backoff (§4.4) | PB-005 |
| BATUTA-PB-010 | Hash-bucketed work directory isolation (§4.8) | PB-008 |

**Milestone:** `batuta playbook run` works across workstation + intel with parallel stages and isolated work directories.

### Phase 3: Tracking & Observability (BATUTA-PB-011 through PB-015)

| Ticket | Description | Deps |
|--------|-------------|------|
| BATUTA-PB-011 | Append-only execution event log (§4.7) | PB-005 |
| BATUTA-PB-012 | Pacha artifact registration (§5.1) | pacha |
| BATUTA-PB-013 | W3C PROV-DM lineage recording (§5.2) | pacha lineage |
| BATUTA-PB-014 | Pipeline run tracking in pacha experiments (§5.1) | pacha |
| BATUTA-PB-015 | CLI status/lineage/visualize commands (§6.1) | PB-012..014 |

**Milestone:** Full provenance tracking, audit trail, and artifact registry.

### Phase 4: Advanced Scheduling (BATUTA-PB-016 through PB-019)

| Ticket | Description | Deps |
|--------|-------------|------|
| BATUTA-PB-016 | Resource-aware scheduler with bin-packing (§2.1.2) | PB-006 |
| BATUTA-PB-017 | Wildcard expansion into concrete DAG stages (§2.1.1) | PB-002 |
| BATUTA-PB-018 | Frozen stage support (§2.1.3) | PB-004 |
| BATUTA-PB-019 | `batuta playbook clean` and work directory management (§4.8) | PB-010 |

**Milestone:** Resource-aware scheduling, wildcard DAGs, and frozen stages.

### Phase 5: Compliance (BATUTA-PB-020 through PB-023)

| Ticket | Description | Deps |
|--------|-------------|------|
| BATUTA-PB-020 | Compliance block parser + synthetic stage expansion (§4.9.1, §4.9.2) | PB-001, PB-002 |
| BATUTA-PB-021 | Gate stage execution with pmat CLI integration (§4.9.3) | PB-005, pmat |
| BATUTA-PB-022 | Gate event log entries and metrics artifact registration (§4.9.4) | PB-011, PB-012 |
| BATUTA-PB-023 | MCP bridge: pmat MCP tools accessible from batuta agents (§4.9.5) | pmat MCP |

**Milestone:** Full pmat compliance gates with pre-flight/post-flight quality enforcement.

### Phase 6: Polish (BATUTA-PB-024 through PB-027)

| Ticket | Description | Deps |
|--------|-------------|------|
| BATUTA-PB-024 | JSON Schema validation for playbook YAML (§2.1) | PB-001 |
| BATUTA-PB-025 | TUI mode via repartir job-flow (§6.3) | repartir TUI |
| BATUTA-PB-026 | `--dry-run` mode (§6.1) | PB-005 |
| BATUTA-PB-027 | SVG DAG visualization (§6.1) | PB-002 |

**Milestone:** Production-ready with validation, monitoring, and visualization.

---

## 9. Validation & Testing

### 9.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    // DAG construction
    #[test]
    fn test_implicit_dag_from_deps_outs();       // §3.1
    #[test]
    fn test_cycle_detection_aborts();             // §3.1
    #[test]
    fn test_explicit_after_edges();               // §3.2
    #[test]
    fn test_parallel_branches_detected();         // §3.3

    // Invalidation
    #[test]
    fn test_unchanged_stage_cached();             // §2.3
    #[test]
    fn test_param_change_invalidates_downstream();// §2.3
    #[test]
    fn test_dep_change_propagates_through_dag();  // §2.3
    #[test]
    fn test_unrelated_param_no_invalidation();    // §2.3

    // Lock file
    #[test]
    fn test_lock_file_roundtrip();                // §2.2
    #[test]
    fn test_lock_file_missing_triggers_full_run();// §2.2

    // Execution
    #[test]
    fn test_stage_lifecycle_transitions();         // §4.1
    #[test]
    fn test_retry_with_backoff();                  // §4.4
    #[test]
    fn test_jidoka_stops_on_first_failure();       // §4.5
    #[test]
    fn test_continue_on_failure_runs_branches();   // §4.5

    // Template
    #[test]
    fn test_param_substitution();
    #[test]
    fn test_parallel_input_output_variables();

    // Wildcards (§2.1.1)
    #[test]
    fn test_wildcard_expansion_generates_concrete_stages();
    #[test]
    fn test_wildcard_downstream_collects_all_expanded();
    #[test]
    fn test_wildcard_per_sample_caching();

    // Resources (§2.1.2)
    #[test]
    fn test_resource_scheduler_respects_core_limit();
    #[test]
    fn test_resource_scheduler_gpu_exclusive();
    #[test]
    fn test_resource_timeout_kills_job();

    // Frozen (§2.1.3)
    #[test]
    fn test_frozen_stage_always_cached();
    #[test]
    fn test_frozen_stage_force_override();

    // Event log (§4.7)
    #[test]
    fn test_event_log_records_all_transitions();
    #[test]
    fn test_event_log_survives_crash();

    // Work directory (§4.8)
    #[test]
    fn test_work_dir_isolation_no_crosstalk();
    #[test]
    fn test_work_dir_hash_deterministic();

    // Invalidation explanation (§2.4)
    #[test]
    fn test_cache_miss_explains_param_change();
    #[test]
    fn test_cache_miss_explains_upstream_rerun();

    // Rash shell compliance (§4.5.1)
    #[test]
    fn test_cmd_purified_through_rash();
    #[test]
    fn test_unquoted_vars_get_quoted();
    #[test]
    fn test_shell_raw_bypasses_purification();
    #[test]
    fn test_security_violation_fails_validation();

    // Git-backed storage (§5.4)
    #[test]
    fn test_lock_file_lists_git_trackable_files();
    #[test]
    fn test_verify_detects_blob_mismatch();
    #[test]
    fn test_init_generates_gitignore();
    #[test]
    fn test_blobs_never_in_git_trackable_list();

    // PMAT compliance gates (§4.9)
    #[test]
    fn test_compliance_pre_flight_expands_to_stages();
    #[test]
    fn test_compliance_post_flight_depends_on_all_user_stages();
    #[test]
    fn test_gate_failure_halts_pipeline_jidoka();
    #[test]
    fn test_gate_cached_when_deps_unchanged();
    #[test]
    fn test_gate_metrics_registered_in_pacha();
    #[test]
    fn test_tdg_gate_blocks_below_min_grade();
    #[test]
    fn test_mutation_gate_blocks_below_threshold();
}
```

### 9.2 Integration Tests

```rust
#[cfg(test)]
mod integration {
    #[test]
    fn test_full_pipeline_local_execution();       // 3-stage pipeline on localhost
    #[test]
    fn test_resume_from_lock_file();               // fail stage 2, resume skips stage 1
    #[test]
    fn test_param_change_partial_rerun();           // change param, verify correct stages rerun
    #[test]
    fn test_pacha_artifacts_registered();           // verify artifacts in pacha after pipeline
}
```

### 9.3 Property-Based Tests

```rust
#[cfg(test)]
mod proptest_tests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_dag_toposort_is_valid(stages in arbitrary_stages()) {
            // For any valid DAG, topological sort produces valid ordering
        }

        #[test]
        fn test_cache_key_deterministic(deps in arbitrary_deps(), params in arbitrary_params()) {
            // Same inputs always produce same cache_key
            let key1 = compute_cache_key(&deps, &params);
            let key2 = compute_cache_key(&deps, &params);
            prop_assert_eq!(key1, key2);
        }

        #[test]
        fn test_invalidation_propagates(dag in arbitrary_dag(), changed_stage in 0..dag.len()) {
            // Changing a stage invalidates all downstream, no upstream
        }
    }
}
```

### 9.4 Falsification Checklist

Following Popperian methodology from APR-QA:

| # | Hypothesis | Falsification Method |
|---|-----------|---------------------|
| 1 | Lock file guarantees determinism | Run pipeline twice with same inputs, verify identical lock files |
| 2 | Cache skip is correct | Modify one dep, verify only downstream stages rerun |
| 3 | Param granularity works | Change param A, verify stages referencing param B are unaffected |
| 4 | DAG construction is sound | Generate random deps/outs, verify no false edges or missing edges |
| 5 | Parallel fan-out produces same output as sequential | Run same stage with max_workers=1 and max_workers=16, compare hashes |
| 6 | Remote execution matches local | Run same cmd on localhost and via repartir TCP, compare output hash |
| 7 | Retry is bounded | Set retry limit=3, fail 10 times, verify exactly 3 retries |
| 8 | Jidoka stops propagation | Fail stage 2 of 5, verify stages 3-5 never start |
| 9 | Lock file survives process crash | Kill during stage 3, verify lock has stages 1-2 completed, 3 absent |
| 10 | BLAKE3 collision resistance | Hash 100K distinct files, verify zero collisions |
| 11 | Per-file resume after partial failure | Kill parallel stage at 50%, resume, verify only remaining files process | §4.6.2 |
| 12 | Manifest tracks completed files | Process 100 files, kill at 60, verify manifest has exactly 60 entries |
| 13 | Non-deterministic stage never caches | Mark stage `deterministic: false`, run twice, verify it executes both times |
| 14 | Concurrent runs are safe | Launch two `batuta playbook run` simultaneously, verify no corruption |
| 15 | Adding new stage preserves cache | Add stage to YAML, verify existing cached stages still skip |
| 16 | Wildcard expansion is correct | Expand `{sample}` over 100 files, verify 100 concrete stages with correct paths | §2.1.1 |
| 17 | Resource scheduler prevents oversubscription | Declare 4-core jobs on 8-core target, verify max 2 concurrent jobs | §2.1.2 |
| 18 | Frozen stage never re-runs | Change deps of frozen stage, verify it stays CACHED | §2.1.3 |
| 19 | Event log is append-only | Run 3 pipelines, verify log has all 3 run_started/run_completed pairs in order | §4.7 |
| 20 | Work directories are isolated | Run 2 parallel jobs, verify no file in common between work dirs | §4.8 |
| 21 | Cache miss explanation is accurate | Change param, verify CLI output names the specific param that changed | §2.4 |
| 22 | Frozen + force overrides frozen | Run `--force` on frozen stage, verify it executes | §2.1.3 |
| 23 | Rash purifies all commands | Write cmd with unquoted `$var`, verify executed command has `"${var}"` | §4.5.1 |
| 24 | Security linting blocks dangerous commands | Write cmd with `eval $user_input`, verify `batuta playbook validate` rejects it | §4.5.1 |
| 25 | Shell raw still lints | Set `shell: raw`, verify warnings emitted but command executes | §4.5.1 |
| 26 | Pre-flight gate blocks pipeline | Set `tdg.min_grade: A`, feed code with grade D, verify pipeline never starts | §4.9 |
| 27 | Post-flight gate runs after all stages | Add post-flight quality_gate, verify it executes after last user stage | §4.9 |
| 28 | Gate is cached like normal stage | Run pipeline twice with same code, verify gate skipped on second run | §4.9.4 |
| 29 | Gate failure produces metrics artifact | Fail a TDG gate, verify JSON report registered in pacha as type: metrics | §4.9.4 |
| 30 | Lock file verifies blob integrity | Corrupt one output file, run `--verify`, confirm it reports the exact mismatch | §5.4.3 |
| 31 | Git-trackable list excludes blobs | Run pipeline, check git-trackable output contains only metadata files, zero binary artifacts | §5.4.1 |
| 32 | Disaster recovery reproduces corpus | Delete all blobs, re-run pipeline from source, `--verify` confirms hashes match original lock file | §5.4.6 |

---

## 10. Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| YAML parse time | <10ms for 100-stage playbook | Interactive CLI feedback |
| DAG construction | <1ms for 100 stages | Kahn's algorithm is O(V+E) |
| BLAKE3 hash rate | >1 GB/s per core | BLAKE3 benchmark [8] |
| Lock file I/O | <5ms read/write | Small YAML file |
| Stage dispatch overhead | <50ms per stage | Dominated by actual cmd execution |
| Remote dispatch overhead | <100ms per stage | TCP connection + serialization |
| Parallel fan-out overhead | <10ms per job enqueue | repartir work-stealing is O(1) |
| TUI refresh rate | 10 fps | repartir job-flow TUI baseline |

For the video corpus pipeline (6,452 files, 1.6TB):
- Hash overhead: ~1.6TB / 1 GB/s = ~27 minutes for full dep hash (first run)
- Subsequent runs: only re-hash changed files (mtime-based pre-filter, then BLAKE3 for modified)
- Lock file size: ~50KB for 6 stages

---

## 11. References

[1] DVC (Data Version Control). "Pipeline Files (dvc.yaml)." https://dvc.org/doc/user-guide/project-structure/dvcyaml-files

[2] Argo Workflows. "DAG Template." https://argo-workflows.readthedocs.io/en/latest/walk-through/dag/

[3] PAIML. "APR-QA Model Qualification Playbooks." Internal specification, 2025.

[4] PAIML. "Probar Playbook Specification (SCXML-style)." Internal specification, 2025.

[5] Dagster. "Software-Defined Assets." https://dagster.io/blog/software-defined-assets

[6] Ansible. "Intro to Playbooks." https://docs.ansible.com/ansible/latest/playbook_guide/playbooks_intro.html

[7] GitHub. "Workflow Syntax for GitHub Actions." https://docs.github.com/actions/using-workflows/workflow-syntax-for-github-actions

[8] BLAKE3 Team. "BLAKE3: One Function, Fast Everywhere." https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf

[9] Mölder, F. et al. (2021). "Sustainable data analysis with Snakemake." F1000Research. DOI: 10.12688/f1000research.29032.2

[10] Di Tommaso, P. et al. (2017). "Nextflow enables reproducible computational workflows." Nature Biotechnology, 35(4), 316-319. DOI: 10.1038/nbt.3820

[11] Temporal Technologies. "How Temporal Works." https://docs.temporal.io/temporal

[12] Albrecht, M. et al. (2012). "Makeflow: A Portable Abstraction for Data Intensive Computing." SWEET '12. DOI: 10.1145/2443416.2443417

[13] Amstutz, P. et al. (2016). "Common Workflow Language, v1.0." Specification. https://www.commonwl.org/v1.0/

[14] Sculley, D. et al. (2015). "Hidden Technical Debt in Machine Learning Systems." NeurIPS 2015.

[15] PAIML. "Rash (bashrs): Shell Safety and Purification Tool." Bash → deterministic, idempotent POSIX sh. Internal, 2024.

[16] PAIML. "PMAT (paiml-mcp-agent-toolkit): Code Analysis and Quality Gate Toolkit." TDG grading, mutation testing, coverage, defect prediction, doc validation. v3.3.0. Internal, 2025.
