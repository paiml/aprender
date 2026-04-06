# Banco Infrastructure: Forjar Integration

> Parent: [banco-spec.md](banco-spec.md) §5
> Depends on: forjar (IaC), probador CLI (optional)

---

## Why

Banco Phase 1-4 specs describe a single-binary workbench. But production AI teams need:
- Multi-machine GPU clusters provisioned reproducibly
- Model artifacts cached with cryptographic provenance
- Drift detection when someone manually installs a driver or swaps a model
- Rollback when a deployment breaks inference
- Audit trail for compliance (who deployed what model, when)

Forjar provides all of this as a Rust-native IaC tool with BLAKE3-hashed state, deterministic DAG execution, and Jidoka failure isolation. Banco integrates forjar for everything beyond the single binary.

---

## Integration Points

### 1. `batuta serve --banco` remains the workbench

Forjar doesn't replace Banco — it provisions the machines that run Banco. The split:

| Concern | Tool |
|---------|------|
| HTTP API, chat, training, UI | Banco (batuta) |
| Machine provisioning, packages, drivers | Forjar |
| Model download + cache + provenance | Forjar content-addressed store |
| Service management (start/stop Banco) | Forjar service resource |
| Drift detection (config, models, drivers) | Forjar tripwire |
| Multi-machine orchestration | Forjar rolling/canary apply |
| Secret management (API keys, JWT keys) | Forjar secrets (age encrypted) |

### 2. Banco queries forjar for infrastructure state

```
GET /api/v1/system (Banco Phase 2, extended)
{
  "privacy_tier": "Sovereign",
  "version": "0.7.2",
  "model_loaded": true,
  "infrastructure": {
    "managed_by": "forjar",
    "generation": 7,
    "last_apply": "2026-03-19T10:00:00Z",
    "drift_status": "clean",
    "machines": 2,
    "gpu_drivers": "cuda-12.4",
    "compliance": "passing"
  }
}
```

Banco reads forjar's state directory (`state/{machine}/state.lock.yaml`) for live infrastructure status. No forjar daemon needed — state is file-based in Git.

---

## Blueprint: banco-workbench.yaml

A forjar config that provisions a complete Banco AI workbench on bare metal.

```yaml
version: "1.0"
name: banco-workbench
description: "Sovereign AI workbench: GPU drivers, Banco binary, model cache"

params:
  banco_version: "0.7.2"
  banco_port: 8090
  privacy_tier: sovereign
  models_dir: /var/lib/banco/models
  data_dir: /var/lib/banco
  gpu_persistence: true

machines:
  workbench:
    hostname: ml-workstation
    addr: 127.0.0.1      # or remote: 192.168.50.10
    user: deploy
    arch: x86_64

resources:
  # GPU drivers + CUDA
  gpu-setup:
    type: gpu
    machine: workbench
    driver: nvidia
    cuda_version: "12.4"
    persistence_mode: "{{params.gpu_persistence}}"

  # Banco binary (from cargo or content-addressed store)
  banco-binary:
    type: package
    machine: workbench
    provider: cargo
    packages: [batuta]
    features: [banco, inference, ml]
    depends_on: [gpu-setup]

  # Data directory
  banco-data:
    type: file
    machine: workbench
    state: directory
    path: "{{params.data_dir}}"
    mode: "0750"

  # Config file
  banco-config:
    type: file
    machine: workbench
    path: "{{params.data_dir}}/config.toml"
    content: |
      [server]
      host = "0.0.0.0"
      port = {{params.banco_port}}
      privacy_tier = "{{params.privacy_tier}}"
      [storage]
      data_dir = "{{params.data_dir}}"
    depends_on: [banco-data]

  # Model cache directory
  model-cache:
    type: file
    machine: workbench
    state: directory
    path: "{{params.models_dir}}"
    mode: "0755"
    depends_on: [banco-data]

  # Pre-pull a default model
  default-model:
    type: model
    machine: workbench
    name: meta-llama/Llama-3-8B-Instruct-GGUF
    source: huggingface
    format: gguf
    quantization: q4_k_m
    cache_dir: "{{params.models_dir}}"
    depends_on: [model-cache]

  # Banco systemd service
  banco-service:
    type: service
    machine: workbench
    name: banco
    enabled: true
    exec_start: >
      batuta serve --banco
      --host 0.0.0.0
      --port {{params.banco_port}}
      --model {{params.models_dir}}/llama-3-8b-q4_k_m.gguf
    restart: on-failure
    depends_on: [banco-binary, banco-config, default-model]

  # Firewall: only expose Banco port
  firewall:
    type: network
    machine: workbench
    allow: [{port: "{{params.banco_port}}", proto: tcp}]
    deny_default: true
    depends_on: [banco-service]

policy:
  failure: stop_on_first
  tripwire: true
  lock_file: true
```

### Deploy

```bash
forjar validate -f banco-workbench.yaml
forjar plan -f banco-workbench.yaml
forjar apply -f banco-workbench.yaml
```

---

## Recipes: Reusable Infrastructure Components

### GPU Compute Node Recipe

```yaml
# recipes/gpu-compute.yaml
name: gpu-compute
version: "1.0"
inputs:
  cuda_version: { type: string, default: "12.4" }
  persistence: { type: bool, default: true }
resources:
  drivers:
    type: gpu
    driver: nvidia
    cuda_version: "{{inputs.cuda_version}}"
    persistence_mode: "{{inputs.persistence}}"
  packages:
    type: package
    provider: apt
    packages: [build-essential, pkg-config, libssl-dev]
```

### Model Cache Recipe

```yaml
# recipes/model-cache.yaml
name: model-cache
version: "1.0"
inputs:
  cache_dir: { type: string, required: true }
  models: { type: list, default: [] }
resources:
  cache_dir:
    type: file
    state: directory
    path: "{{inputs.cache_dir}}"
  models:
    type: model
    for_each: "{{inputs.models}}"
    name: "{{item.name}}"
    source: "{{item.source}}"
    format: "{{item.format}}"
    cache_dir: "{{inputs.cache_dir}}"
    depends_on: [cache_dir]
```

Use in main config:
```yaml
recipes:
  - name: gpu-compute
    inputs: { cuda_version: "12.4" }
  - name: model-cache
    inputs:
      cache_dir: /var/lib/banco/models
      models:
        - { name: "meta-llama/Llama-3-8B-Instruct-GGUF", source: huggingface, format: gguf }
        - { name: "microsoft/Phi-3-mini-4k-instruct-gguf", source: huggingface, format: gguf }
```

---

## Content-Addressed Model Store

Forjar's content-addressed store gives Banco reproducible model management:

```
/var/lib/forjar/store/
  <blake3-hash-a>/
    meta.yaml     # provenance: source URL, download date, format, quant
    content/
      llama-3-8b-q4_k_m.gguf
  <blake3-hash-b>/
    meta.yaml
    content/
      phi-3-mini-q4_k_m.gguf
```

Benefits:
- **Dedup**: Same model referenced by multiple workbenches shares one store entry
- **Provenance**: `meta.yaml` records source, download time, BLAKE3 hash, purity level
- **Rollback**: Previous model versions retained as prior store entries
- **Transfer**: `forjar cache push/pull` syncs models between machines via copia delta (minimal bandwidth for large files)
- **Integrity**: BLAKE3 hash verified on every access — detects corruption or tampering

Banco's `/api/v1/models/load` can reference store paths:
```json
{"model": "forjar:///var/lib/forjar/store/<hash>/content/model.gguf"}
```

---

## Drift Detection for AI Infrastructure

```bash
# Check: has anything changed since last apply?
forjar drift -f banco-workbench.yaml

# Output:
# ✓ gpu-setup: clean
# ✗ banco-config: DRIFTED (config.toml modified manually)
# ✓ default-model: clean (BLAKE3 hash matches)
# ✗ banco-service: DRIFTED (service restarted outside forjar)
```

Banco's `/api/v1/system` endpoint exposes drift status when forjar state is available:

```json
{
  "infrastructure": {
    "drift_status": "drifted",
    "drifted_resources": ["banco-config", "banco-service"],
    "last_drift_check": "2026-03-19T12:00:00Z"
  }
}
```

### Tripwire Events

Forjar logs all state changes to `state/{machine}/events.jsonl`:
```jsonl
{"ts":"2026-03-19T10:00:00Z","action":"apply","resource":"default-model","result":"created","hash":"abc123"}
{"ts":"2026-03-19T12:30:00Z","action":"drift","resource":"banco-config","result":"drifted","detail":"content changed"}
```

Banco's audit log (cross-cutting §10) can merge with forjar events for a unified timeline.

---

## Multi-Machine: Team Workbench

```yaml
machines:
  gpu-1:
    hostname: ml-gpu-1
    addr: 192.168.50.10
    roles: [gpu-compute, banco-primary]
  gpu-2:
    hostname: ml-gpu-2
    addr: 192.168.50.11
    roles: [gpu-compute, banco-replica]

resources:
  # Shared model cache via NFS
  model-nfs:
    type: mount
    machine: [gpu-1, gpu-2]
    source: "nfs-server:/models"
    target: /mnt/models
    fstype: nfs4

  # Banco on both machines
  banco:
    type: service
    machine: [gpu-1, gpu-2]
    name: banco
    exec_start: "batuta serve --banco --model /mnt/models/llama3.gguf"
    depends_on: [model-nfs]
```

Deploy with rolling strategy:
```bash
forjar apply -f team-workbench.yaml --rolling 1
# Applies to gpu-1 first, verifies health, then gpu-2
```

---

## Banco API Extensions for Forjar

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/api/v1/infra/status` | Forjar state summary (generation, drift, machines) |
| GET | `/api/v1/infra/drift` | Detailed drift report |
| GET | `/api/v1/infra/history` | Forjar event log (recent applies, drifts) |
| POST | `/api/v1/infra/apply` | Trigger `forjar apply` from Banco UI (admin only) |

These endpoints read forjar's file-based state — no forjar daemon required.

---

## Secret Management

Forjar's age-encrypted secrets store API keys, JWT signing keys, and model credentials:

```bash
# Encrypt a secret
forjar secrets set banco-jwt-key --value "$(openssl rand -base64 32)" -f banco.yaml

# Secret referenced in config
resources:
  banco-config:
    type: file
    content: |
      [auth]
      jwt_secret = "{{secrets.banco-jwt-key}}"
```

Banco never stores secrets in its own `~/.banco/` — forjar manages them with age encryption and Ed25519 key pairs.

---

## Implementation Priority

| Priority | Feature | Phase |
|----------|---------|-------|
| P1 | Blueprint: single-machine banco-workbench.yaml | Phase 2 |
| P1 | Model store integration (forjar:// URI in model load) | Phase 2 |
| P2 | /api/v1/infra/status endpoint (read forjar state) | Phase 2 |
| P2 | Drift status in /api/v1/system | Phase 2 |
| P2 | Secret management for auth keys | Phase 2 |
| P3 | Multi-machine blueprint + rolling deploy | Phase 3 |
| P3 | Recipes for GPU compute, model cache | Phase 3 |
| P3 | /api/v1/infra/apply from UI | Phase 4 |
| P3 | Unified audit trail (Banco + forjar events) | Phase 3 |
