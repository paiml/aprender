## cold start (fresh process, rayon threads 6, keep_mmap false)

| step | t since start ms | RSS MB |
|---|---|---|
| process start | 4 | 9 |
| mmap GGUF | 21 | 52 |
| base (embeddings, norms, lm_head) | 184 | 2963 |
| owned layers | 402 | 6765 |
| after dropping mmap | 420 | 3885 |
| **first decision (87 tokens)** | 819 (decision 396) | 3933 |

## steady state (5 rounds per request)

| request | questions | tokens (sum of rows) | p50 ms | p95 ms |
|---|---|---|---|---|
| 0 | 1 | 87 | 375 | 383 |
| 1 | 1 | 81 | 352 | 354 |
| 2 | 1 | 84 | 356 | 358 |
| 3 | 1 | 85 | 367 | 368 |
| 4 | 1 | 84 | 354 | 356 |
| 5 | 1 | 85 | 368 | 373 |
| 6 | 3 | 145 | 711 | 715 |
| 7 | 1 | 36 | 185 | 188 |
| 8 | 1 | 42 | 207 | 208 |
| 9 | 1 | 915 | 3729 | 3768 |

peak-so-far RSS 4094 MB; worst |dp| vs torch fp32 1.31e-6
