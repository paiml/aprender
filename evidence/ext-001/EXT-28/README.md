# EXT-28 — C2 speed arms (aprender#4410)

EXT-001 §10 C2: `apr serve` and each competitor arm serve the blessed model
(`ours-q4k.gguf`, sha256 `97784e90405056f4529564a811acb333d2303b09ce493a361d75b5797e1158af`,
Qwen3.5-4B Q4_K_M, text-only) and are measured by one client at m=1. Absolute values
only; the apr/llama.cpp comparison exists only inside the EXT-19 ledger ratchet (T28).

| File | What |
|------|------|
| `c2_cell.sh` | the harness: every arm started under `env -i`, pinned to CPUs 0-15 with 16 threads, all arms **resident**, measured iterations **interleaved** (the order rotates each round), one warmup, N=5, `max_tokens` 32, temperature 0 |
| `assemble_cell.sh` | one record dir → one `cell.json` receipt (`speed_arms::CellReceipt`) |
| `prompt.txt` | the 681-byte prompt every arm is sent |
| `ollama-blob.json` | why Ollama is refused (S-14): the `qwen3.5:4b` blob carries 393 vision tensors of 834 and its server loads a CLIP encoder; ours carries 0 |
| `cells/*.json` | the five REX cells this run had no host for, recorded `not_run` |
| `lambda-cpu/{r1,r2,r3,plant,r4}/` | five records on lambda-cpu (GPU hidden): per-arm JSON, `cell.json`, and `logs/` (server logs + the raw timestamped SSE streams the comparator `artifact_sha256` covers) |

## Arms

- **apr** 0.70.0 @ 99c6c61b4 (`plant`: the same source plus a 1 s sleep per decode step)
- **llama.cpp** d1d3c3396 — the reference arm
- **mistral.rs** v0.9.4 @ 4400935, `serve --cpu --format gguf`
- **Ollama** 0.34.4 — refused, see `ollama-blob.json`

## lambda-cpu, medians of 5 (shared host, load average 30–55 during the run)

| record | arm | load ms | TTFT ms | ITL ms | e2e ms | decode tok/s | peak RSS MiB |
|---|---|---|---|---|---|---|---|
| r1 | apr | 74341 | 69228 | 473.1 | 85725 | 1.8236 | 7957 |
| r1 | llama.cpp | 5569 | 575 | 205.2 | 6827 | 4.3048 | 4441 |
| r1 | mistral.rs | 62518 | 10547 | 348.1 | 24456 | 1.9897 | 8154 |
| r2 | apr | 245785 | 92680 | 737.5 | 118202 | 0.8541 | 7938 |
| r2 | llama.cpp | 15202 | 601 | 216.2 | 7962 | 3.9375 | 4442 |
| r2 | mistral.rs | 106605 | 18407 | 375.6 | 29590 | 2.4143 | 8132 |
| r3 | apr | 85015 | 74574 | 409.6 | 87061 | 1.8045 | 7938 |
| r3 | llama.cpp | 7414 | 509 | 217.1 | 7239 | 4.0207 | 4441 |
| r3 | mistral.rs | 94756 | 10644 | 354.9 | 24518 | 1.9462 | 8132 |
| plant | apr (plant) | 59969 | 74167 | 2061.3 | 135832 | 0.4471 | 7956 |
| plant | llama.cpp | 23437 | 661 | 205.3 | 7962 | 3.5613 | 4441 |
| plant | mistral.rs | 59412 | 22376 | 484.4 | 37676 | 1.7647 | 8120 |
| r4 | apr | 121185 | 92332 | 500.8 | 111128 | 1.3832 | 7939 |
| r4 | llama.cpp | 10892 | 506 | 215.1 | 7210 | 4.1237 | 4441 |
| r4 | mistral.rs | 87078 | 15285 | 393.9 | 29883 | 1.8495 | 8129 |

## FALSIFY-EXT-022: RED on this host

`cargo test -p apr-cli --lib falsify_ext_022_planted_sleep_red` judges r1–r3 as the
floor, then the plant and r4. The plant is RED, as it must be. **The unpatched
control r4 is also RED**, so the test fails: on this host the ratchet cannot tell
a 1 s-per-step regression from host load.

Per-iteration apr decode tok/s (unpatched, same binary sha `678647001a49…`) spans
0.53–2.22 within and across records; llama.cpp stays within 2.8–4.7. Interleaving
put every arm under the same load, but the load does not hit them alike: the
rayon-based engines (apr, mistral.rs) lose far more to contention than llama.cpp,
so the load does not cancel. An earlier sequential pass (arms run one after
another) failed the same way.

What would make it hold is a method decision, not a tolerance edit made after
seeing the data: a quiet-host rule for records, or a contention-robust statistic
chosen in advance (e.g. best-of-N per arm). It is escalated, not decided here.
