# Mixture-of-Experts (MOE) Overview

Realizar implements inference-time MOE infrastructure following Toyota Production System principles.

## Key Components

| Module | Purpose | Citation |
|--------|---------|----------|
| `moe.rs` | Capacity Factor routing | Fedus et al. (2022) [^1] |
| `stats.rs` | A/B testing with log-transform | Box et al. (2005) [^2] |
| `memory.rs` | mlock for hot experts | Dean & Barroso (2013) [^3] |
| `registry.rs` | Lock-free reads via ArcSwap | McKenney (2011) [^4] |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MOE HOSTING LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Traffic Router → A/B Splitter → MOE Router → Model Registry    │
│       ↓              ↓              ↓              ↓            │
│  [Requests]     [Cohorts]     [Experts]      [Models]           │
└─────────────────────────────────────────────────────────────────┘
```

## Toyota Way Principles Applied

- **Jidoka**: Andon triggers with auto-rollback
- **Just-in-Time**: Lazy model loading with ArcSwap
- **Heijunka**: Capacity Factor load balancing
- **Kaizen**: A/B testing for continuous improvement
- **Genchi Genbutsu**: Memory pinning prevents page faults

## Performance

- MOE routing: **1.7M routes/sec**
- Registry reads (ArcSwap): **18.9M reads/sec**
- Capacity overflow fallback: **100% accuracy**

See [Reproducible Benchmarks](./benchmarks.md) for methodology and replication.

## References

[^1]: Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *JMLR*, 23, 1-39. doi:10.48550/arXiv.2101.03961

[^2]: Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2005). *Time Series Analysis: Forecasting and Control* (4th ed.). Wiley.

[^3]: Dean, J., & Barroso, L. A. (2013). The Tail at Scale. *Communications of the ACM*, 56(2), 74-80. doi:10.1145/2408776.2408794

[^4]: McKenney, P. E. (2011). Is Parallel Programming Hard, And, If So, What Can You Do About It?
