# Spike 007 — run output

## Bench (`chronos-mcp --bench models/tiny models/small-f16`, Apple M4 Pro)


### chronos-bolt-tiny — 8652672 params, weights F32, load 35 ms (parse + f32 decode + transposes)

| variant | forward ms (2048 ctx) | max abs Δ vs all-GEMM (q, 64 steps) |
|---|---|---|
| plain loops (spike 005 default) | 37.0 | 9.5e-7 |
| GEMM projections + FF + embedding, single rows via trueno gemv | 19.7 | 9.5e-7 |
| + attention scores/context via GEMM | 18.6 | 9.5e-7 |
| + single-row projections as contiguous dot8 instead of gemv | 18.5 | 0.0e0 |
| + rayon-parallel `blis::gemm` (crate feature `parallel`, 10 cores) | 17.8 | 0.0e0 |

Stage breakdown (2048 ctx, all-GEMM, dot8 single rows): patch embedding 1.3 ms · encoder 14.7 ms · decoder (1 token) 2.3 ms · quantile head 0.1 ms

| context | horizon | forwards | predict ms (median of 5) |
|---|---|---|---|
| 100 | 12 | 1 | 2.7 |
| 100 | 64 | 1 | 2.7 |
| 100 | 365 | 46 | 193.5 |
| 512 | 12 | 1 | 6.2 |
| 512 | 64 | 1 | 6.2 |
| 512 | 365 | 46 | 343.4 |
| 2048 | 12 | 1 | 18.6 |
| 2048 | 64 | 1 | 18.6 |
| 2048 | 365 | 46 | 850.9 |

### chronos-bolt-small — 47718016 params, weights F16, load 181 ms (parse + f32 decode + transposes)

| variant | forward ms (2048 ctx) | max abs Δ vs all-GEMM (q, 64 steps) |
|---|---|---|
| plain loops (spike 005 default) | 229.4 | 9.5e-7 |
| GEMM projections + FF + embedding, single rows via trueno gemv | 102.6 | 9.5e-7 |
| + attention scores/context via GEMM | 98.5 | 9.5e-7 |
| + single-row projections as contiguous dot8 instead of gemv | 98.6 | 0.0e0 |
| + rayon-parallel `blis::gemm` (crate feature `parallel`, 10 cores) | 91.9 | 0.0e0 |

Stage breakdown (2048 ctx, all-GEMM, dot8 single rows): patch embedding 4.6 ms · encoder 79.4 ms · decoder (1 token) 14.1 ms · quantile head 0.2 ms

| context | horizon | forwards | predict ms (median of 5) |
|---|---|---|---|
| 100 | 12 | 1 | 15.7 |
| 100 | 64 | 1 | 15.9 |
| 100 | 365 | 46 | 1119.9 |
| 512 | 12 | 1 | 35.3 |
| 512 | 64 | 1 | 35.2 |
| 512 | 365 | 46 | 1909.7 |
| 2048 | 12 | 1 | 98.2 |
| 2048 | 64 | 1 | 98.2 |
| 2048 | 365 | 46 | 4536.0 |

# Embedded builds — binary size and cold start (Apple M4 Pro)

| embedded weights | binary MB | stripped MB |
|---|---|---|
| tiny | 41.9 | 40.5 |
| tiny-f16 | 24.4 | 23.1 |
| small-f16 | 103.2 | 101.8 |
| none (runtime path) | 7.0 | – |

## cold start, embedded tiny-f16 (spawn → initialize → forecast Peyton h=64 over stdio)

| run | exec → initialize response ms | → forecast response ms | server load ms (reported) |
|---|---|---|---|
| 1 | 64 | 85 | see stderr banner |
| 2 | 33 | 53 | see stderr banner |
| 3 | 33 | 52 | see stderr banner |

## cold start, embedded small-f16 (spawn → initialize → forecast Peyton h=64 over stdio)

| run | exec → initialize response ms | → forecast response ms | server load ms (reported) |
|---|---|---|---|
| 1 | 347 | 448 | see stderr banner |
| 2 | 179 | 279 | see stderr banner |
| 3 | 180 | 280 | see stderr banner |

