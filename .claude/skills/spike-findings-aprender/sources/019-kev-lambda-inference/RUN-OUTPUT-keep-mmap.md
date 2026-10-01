## cold start (fresh process, rayon threads 6, keep_mmap true)

| step | t since start ms | RSS MB |
|---|---|---|
| process start | 3 | 9 |
| mmap GGUF | 20 | 52 |
| base (embeddings, norms, lm_head) | 197 | 2963 |
| owned layers | 419 | 6765 |
| after keeping mmap | 421 | 6765 |
| **first decision (87 tokens)** | 809 (decision 385) | 6813 |

## steady state (5 rounds per request)

| request | questions | tokens (sum of rows) | p50 ms | p95 ms |
|---|---|---|---|---|
| 0 | 1 | 87 | 373 | 374 |
| 1 | 1 | 81 | 353 | 355 |
| 2 | 1 | 84 | 356 | 357 |
| 3 | 1 | 85 | 369 | 370 |
| 4 | 1 | 84 | 355 | 356 |
| 5 | 1 | 85 | 369 | 370 |
| 6 | 3 | 145 | 707 | 709 |
| 7 | 1 | 36 | 184 | 186 |
| 8 | 1 | 42 | 205 | 207 |
| 9 | 1 | 915 | 3733 | 3736 |

peak-so-far RSS 6997 MB; worst |dp| vs torch fp32 1.31e-6
