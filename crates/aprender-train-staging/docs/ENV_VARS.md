# Environment Variables

## Training Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ENTRENAR_LOG` | Log level (trace, debug, info, warn, error) | `info` |
| `ENTRENAR_SEED` | Global random seed for reproducibility | None |
| `ENTRENAR_CHECKPOINT_DIR` | Directory for checkpoint saves | `./checkpoints` |
| `ENTRENAR_DATA_DIR` | Default data directory | `./data` |

## GPU Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU device selection | All |
| `CUDA_LAUNCH_BLOCKING` | Synchronous CUDA for debugging | `0` |
| `ENTRENAR_GPU_MEMORY_LIMIT` | GPU memory limit in bytes | Unlimited |

## Distributed Training

| Variable | Description | Default |
|----------|-------------|---------|
| `ENTRENAR_WORLD_SIZE` | Number of distributed workers | `1` |
| `ENTRENAR_RANK` | Current worker rank | `0` |
| `ENTRENAR_LOCAL_RANK` | Local GPU rank | `0` |
| `ENTRENAR_COORDINATOR_ADDR` | Coordinator address for DDP | `0.0.0.0:9000` |

## Sovereign Mode

| Variable | Description | Default |
|----------|-------------|---------|
| `ENTRENAR_OFFLINE` | Disable all network access | `false` |
| `ENTRENAR_DATA_RESIDENCY` | Allowed data residency regions | None |
| `ENTRENAR_AUDIT_LOG` | Path to audit log file | None |
