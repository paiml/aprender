# LLaMA Training Observability & Tracing

This guide covers the **Phase 4** observability stack for LLaMA training in entrenar, using **renacer** for
syscall-level tracing, **OpenTelemetry** for distributed tracing, and **Jaeger** for trace visualization.

## Overview

The entrenar LLaMA implementation includes a comprehensive observability stack that enables:

- **Performance Profiling**: Identify bottlenecks in training (I/O, compute, syscalls)
- **Distributed Tracing**: Visualize forward/backward passes with OpenTelemetry + Jaeger
- **Anomaly Detection**: Catch hardware issues (GPU throttling, disk contention) in real-time
- **ML-Based Analysis**: Cluster-based outlier detection using KMeans

## Quick Start

### 1. Basic Profiling (Function-Level)

Profile LLaMA training to identify hot paths:

```bash
make profile-llama
```

This runs:
```bash
renacer --function-time --source --stats-extended -- \
  cargo run --release --example llama2-train --config examples/llama2/configs/124m.toml --epochs 1
```

**Output Example:**
```
Function Profiling Summary:
========================
Total functions profiled: 234
Total syscalls: 4,231

Top 10 Hot Paths (by total time):
  1. llama2::load_checkpoint  - 45.2% (1.2s, 67 syscalls) ⚠️ SLOW I/O
  2. entrenar::backward       - 32.1% (850ms, 2345 syscalls)
  3. trueno::matmul_simd      - 12.4% (330ms, 123 syscalls)
```

### 2. OTLP Tracing with Jaeger

Visualize training traces in Jaeger UI:

**Step 1: Start Jaeger Backend**
```bash
docker-compose -f docker-compose-jaeger.yml up -d
```

**Step 2: Run Training with OTLP Export**
```bash
make profile-llama-otlp
```

**Step 3: View Traces**
Open [http://localhost:16686](http://localhost:16686) in your browser.

You'll see:
- **Service**: `llama-training`
- **Root Span**: `process: llama2-train`
  - **Child Span**: `forward_pass` (234ms)
    - `attention_layer_0` (45ms)
      - `syscall: read` (12ms)
      - `compute: trueno_matmul` (28ms)
    - `attention_layer_1` (43ms)
  - **Child Span**: `backward_pass` (421ms)
  - **Child Span**: `optimizer_step` (89ms)

### 3. Real-Time Anomaly Detection

Detect anomalies during training:

```bash
make profile-llama-anomaly
```

This generates `.pmat/llama-training-profile.json` with ML analysis.

**Output Example:**
```
⚠️ ANOMALY: read took 5234 μs (4.2σ from baseline 102.3 μs) - 🟡 Medium
⚠️ ANOMALY: matmul took 12345 μs (6.3σ from baseline 1234.5 μs) - 🔴 High

=== Real-Time Anomaly Detection Report ===
Total anomalies detected: 23
Severity Distribution:
  🔴 High (>5.0σ):   3 anomalies (likely hardware issues)
  🟡 Medium (4-5σ):  8 anomalies (investigate)
  🟢 Low (3-4σ):    12 anomalies (noise)
```

### 4. Post-Training Analysis

Analyze the ML anomaly detection results:

```bash
./scripts/analyze_training.sh
```

This parses `.pmat/llama-training-profile.json` and provides:
- Clustering quality (silhouette score)
- High-latency cluster identification
- Outlier detection with z-scores
- Actionable recommendations

## Architecture

### Tracing Stack Components

```
┌─────────────────────────────────────────────────┐
│  LLaMA Training (llama2-train)                  │
│  ┌───────────────────────────────────────────┐  │
│  │  entrenar autograd                        │  │
│  │  ├─ forward()                             │  │
│  │  ├─ backward()                            │  │
│  │  └─ optimizer.step()                      │  │
│  └───────────────────────────────────────────┘  │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │  renacer       │
         │  (strace++)    │
         └────────┬───────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
  ┌──────────┐      ┌──────────────┐
  │ Function │      │ OTLP gRPC    │
  │ Profiler │      │ (port 4317)  │
  └──────────┘      └───────┬──────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Jaeger       │
                    │  (all-in-one) │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Jaeger UI    │
                    │  :16686       │
                    └───────────────┘
```

### Key Technologies

- **renacer**: System call tracer with ML anomaly detection
- **OpenTelemetry (OTLP)**: Distributed tracing protocol
- **Jaeger**: Trace visualization and storage
- **KMeans Clustering**: ML-based anomaly detection (via aprender)

## Use Cases

### Use Case 1: Identify I/O Bottlenecks

**Problem**: Training is slower than expected.

**Solution**:
```bash
make profile-llama
```

**Analysis**:
- Look for high percentage in `load_checkpoint` or file I/O
- Check syscall counts for `read`, `write`, `fsync`
- If >40% time in I/O: optimize checkpoint loading

### Use Case 2: Debug GPU Throttling

**Problem**: Training speed varies significantly between runs.

**Solution**:
```bash
make profile-llama-anomaly
./scripts/analyze_training.sh
```

**Analysis**:
- Check for **High severity** anomalies (>5.0σ)
- Look for outliers in `compute` operations
- Silhouette score <0.5 indicates inconsistent performance

### Use Case 3: Visualize Forward/Backward Pass Timing

**Problem**: Need to understand where time is spent in each training iteration.

**Solution**:
```bash
docker-compose -f docker-compose-jaeger.yml up -d
make profile-llama-otlp
```

**Analysis**:
- Open Jaeger UI: [http://localhost:16686](http://localhost:16686)
- Select service: `llama-training`
- Find traces, examine span hierarchy
- Identify slow spans (red/orange in UI)

### Use Case 4: Compare Training Runs

**Problem**: Want to compare performance before/after optimization.

**Solution**:
```bash
# Baseline
make profile-llama-anomaly
mv .pmat/llama-training-profile.json .pmat/baseline.json

# After optimization
make profile-llama-anomaly
mv .pmat/llama-training-profile.json .pmat/optimized.json

# Compare
./scripts/analyze_training.sh .pmat/baseline.json
./scripts/analyze_training.sh .pmat/optimized.json
```

## Configuration

### Jaeger Sampling Strategies

Edit `jaeger-sampling.json` to control trace sampling:

```json
{
  "service_strategies": [
    {
      "service": "llama-training",
      "type": "probabilistic",
      "param": 1.0  // Sample 100% (development)
    }
  ],
  "default_strategy": {
    "type": "probabilistic",
    "param": 0.1  // Sample 10% (production)
  }
}
```

### Anomaly Detection Thresholds

Adjust anomaly sensitivity in the Makefile:

```makefile
renacer --anomaly-threshold 3.0   # Lower = more sensitive (default: 3.0)
        --ml-clusters 5           # Number of KMeans clusters (default: 5)
```

## Metrics Reference

### Function Profiling Metrics

- **Total functions profiled**: Number of distinct functions traced
- **Total syscalls**: System calls made during execution
- **Top Hot Paths**: Functions ranked by total time spent
- **Syscall count**: Number of syscalls per function

### OTLP Trace Metrics

- **Span duration**: Time spent in each operation (ms)
- **Span hierarchy**: Parent-child relationships (forward → attention → matmul)
- **Service name**: Identifier for the traced service (`llama-training`)

### Anomaly Detection Metrics

- **Severity Levels**:
  - 🔴 **High (>5.0σ)**: Likely hardware issues, investigate immediately
  - 🟡 **Medium (4-5σ)**: Worth investigating, may impact performance
  - 🟢 **Low (3-4σ)**: Noise, can usually ignore

- **Silhouette Score**: Clustering quality
  - **>0.7**: Excellent cluster separation
  - **0.5-0.7**: Good separation
  - **<0.5**: Poor separation, noisy/inconsistent performance

### ML Anomaly Metrics

- **Clusters**: Number of distinct performance patterns detected
- **High-latency cluster**: Syscalls grouped by high latency
- **Outliers**: Individual syscalls with extreme z-scores (>3.0σ)

## Troubleshooting

### Renacer Not Found

```bash
cargo install renacer
```

Or build from source:
```bash
git clone https://github.com/durbanlegend/renacer
cd renacer
cargo install --path .
```

### Jaeger Not Starting

Check Docker is running:
```bash
docker ps
docker-compose -f docker-compose-jaeger.yml logs
```

### OTLP Connection Failed

Ensure Jaeger is listening on port 4317:
```bash
docker-compose -f docker-compose-jaeger.yml ps
netstat -an | grep 4317
```

### No Anomalies Detected

Possible causes:
- Training run too short (increase epochs)
- System too consistent (try longer benchmark)
- Threshold too high (lower `--anomaly-threshold`)

## Advanced Topics

### Custom OTLP Exporters

Export traces to other backends (Tempo, Zipkin, etc.):

```bash
renacer --otlp-endpoint http://tempo:4317 \
        --otlp-service-name llama-training \
        --trace-compute \
        -- cargo run --release --example llama2-train
```

### Source Mapping for Transpiled Code

If using Python→Rust transpilation (via Depyler):

```bash
# 1. Generate source map
depyler finetune.py --output finetune_rs.rs --source-map finetune.sourcemap.json

# 2. Run with source mapping
renacer --transpiler-map finetune.sourcemap.json \
        --function-time \
        --source \
        -- ./finetune_rs
```

Output shows original Python source locations:
```
read(3, buf, 1024) = 1024  [finetune.py:42 in load_dataset]
write(1, "epoch 1", 7) = 7  [finetune.py:89 in train_loop]
```

### Continuous Profiling in CI/CD

Add profiling to CI pipeline:

```yaml
# .github/workflows/ci.yml
- name: Profile LLaMA training
  run: |
    make profile-llama-anomaly
    ./scripts/analyze_training.sh

- name: Upload profile
  uses: actions/upload-artifact@v3
  with:
    name: training-profile
    path: .pmat/llama-training-profile.json
```

## Quality Gates

Phase 4 acceptance criteria:

- ✅ **Renacer profiling identifies top 3 bottlenecks**
- ✅ **OTLP traces viewable in Jaeger UI**
- ✅ **Anomaly detection catches hardware issues**
- ✅ **Documentation includes example traces** (this document)

## References

- [OpenTelemetry](https://opentelemetry.io/) - Distributed tracing standard
- [Jaeger](https://www.jaegertracing.io/) - Trace visualization
- [Aprender](https://crates.io/crates/aprender) - ML library (KMeans clustering)

## Next Steps

After setting up observability:

1. **Optimize identified bottlenecks** (I/O, compute, memory)
2. **Set up continuous profiling** in CI/CD
3. **Create performance dashboards** (Grafana + Prometheus)
4. **Implement performance regression tests** using baseline profiles

---

**Built with EXTREME TDD** 🦀⚡

Following renacer tracing patterns for production-ready ML infrastructure.
