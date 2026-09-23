# PMAT-3714 R1 — implementation receipt

Ticket: #3714 (P0, 0.69.1). This row: done_when 1 and 3. `Refs #3714`, not a closing PR. Code commit `5615e7afe`; branch head adds the capacity re-lift, the fragment and the aggregate.

## Binaries (proven built from the commit, not labelled by intent)
| host | GPU | driver | `apr --version` | test binary |
|---|---|---|---|---|
| lambda | RTX 4090, sm_89 | 580.119.02 | `apr 0.69.0 (5615e7afe)` | release `--features cuda` lib tests of the same tree, snapshotted to `eb-3714-bins/r1f` |
| gx10 | GB10, sm_121, aarch64 | 590.48.01 | `apr 0.69.0 (5615e7afe)` | same, built on gx10 from the same commit (git bundle), `cmp`-verified snapshot |

Every GPU cell ran under the GPU lock: `flock /tmp/apr-gpu.lock choom -n 1000`, or `gpu-q --prio 1`, which takes that lock.

## Before (225b2a9ab, measured on both hosts)
- `apr run --gpu`: rc 14 on both files and both hosts, and CUDA was never attempted.
- `apr qa --json`: rc 5 with a **0-byte** report.
- `apr parity`: rc 12, REFUSED.

Gap list: https://github.com/paiml/aprender/issues/3714#issuecomment-5763628862

## After, at 5615e7afe

### GPU forward vs the CPU specification: 64 real positions + 1 decode step
The reference is `forward_single_qwen3_moe_with_cache` with FP32 activations. Every position is held to cosine ≥ 0.9999, including position 0.
| host | file | probe | min cosine (all 65 positions) | argmax mismatches | CPU / GPU wall |
|---|---|---|---|---|---|
| lambda | Coder-30B-A3B | plain text | **1.000000** | **0** | 6.2 s / 0.84 s |
| lambda | Coder-30B-A3B | `<\|im_start\|>user\n` + text | **1.000000** | **0** | 5.1 s / 0.79 s |
| lambda | Instruct-2507 | `<\|im_start\|>user\n` + text | **1.000000** | **0** | 5.1 s / 0.78 s |
| gx10 | Coder-30B-A3B | plain text | **1.000000** | **0** | 24.8 s / 1.89 s |
| gx10 | Coder-30B-A3B | `<\|im_start\|>user\n` + text | **1.000000** | **0** | 43.4 s / 1.83 s |
| gx10 | Instruct-2507 | `<\|im_start\|>user\n` + text | **1.000000** | **0** | 23.2 s / 1.89 s |

### The oracle still bites: routing faults injected in test builds only
A control and three faults, the same GPU model and the same reference, on both hosts. lambda and gx10 agree to 5 decimals; the one 6th-decimal difference is shown:
| fault | min cosine | argmax mismatches | exact-reference check | runtime F2 (0.95 / 0.98) |
|---|---|---|---|---|
| none (control) | 1.000000 | 0/65 | PASS | ACCEPT |
| WrongExpert (ids shifted by one) | −0.158221 | 65/65 | REJECT | REJECT |
| DropTopExpert (top weight zeroed) | −0.222673 | 62/65 | REJECT | REJECT |
| UniformWeights (router weights ignored) | 0.486669 (gx10) / 0.486670 (lambda) | 26/65 | REJECT | REJECT |

### Why the reference is FP32 (the oracle change), measured
- Coder, chat start, GPU vs the **Q8_K**-activation CPU: min cosine 0.985436 with 1 argmax mismatch. Against the FP32-activation CPU: 1.000000, 0 mismatches.
- Instruct-2507, runtime F2 judged against Q8_K (the first cut): **REJECTED** at position 1 (cosine 0.8725, argmax 25 ≠ 4102), and it fell back to CPU (rc 14). Judged against FP32 (the table above): 1.000000.
- A thread-local scope on the caller alone did NOT reach the QKV `rayon::join` or the expert `par_iter`: cosine 0.9954 with 1 argmax flip. The fix runs the reference on a dedicated pool whose workers carry the flag. Unit-tested across join, par_iter and nested joins.

### `apr run --gpu` (prompt "What is the capital of France? Answer briefly.", 32 tokens)
| host | file | rc | backend line | F2 | output | decode |
|---|---|---|---|---|---|---|
| lambda | Instruct-2507 | **0** | CUDA RTX 4090, weights resident in 929 ms | 50 positions, min cosine 1.0000 | "The capital of France is Paris." | 87 tok/s incl. token-by-token prefill |
| lambda | Coder-30B-A3B | **0** | CUDA RTX 4090, 942 ms | 50 positions, 1.0000 | "Paris" | 88 tok/s |
| gx10 | Coder-30B-A3B | **0** | CUDA GB10, 3974 ms | 50 positions, 1.0000 | "Paris" | 36.5 tok/s |
| gx10 | Instruct-2507 | **0** | CUDA GB10, 3634 ms | 50 positions, 1.0000 | "The capital of France is Paris." | 35.7 tok/s |

### `apr qa --json` (the ladder's flags: `--offline --skip-throughput --skip-ollama --skip-gpu-speedup --skip-ptx-parity --skip-gpu-state --skip-format-parity`)
| host | file | rc | report | capability_match | golden_output | served by |
|---|---|---|---|---|---|---|
| lambda | Instruct-2507 | 0 | 12 gates, 5 executed | PASS | PASS | 3 of 3 golden runs print `Backend: GPU` + F2 1.0000 |
| lambda | Coder-30B-A3B | 0 | 12 gates, 5 executed | PASS | PASS | 3 of 3 on the GPU |
| gx10 | Coder-30B-A3B | 0 | 12 gates, 5 executed | PASS | PASS | 3 of 3 on the GPU, 36 tok/s |
| gx10 | Instruct-2507 | 0 | 12 gates, 5 executed | PASS | PASS | 3 of 3 on the GPU |

The golden gate's PASS text on this base still reads "(GPU hybrid forward, #3090)". That is main's stale string, and #3711 (aprender-0e, in the 0.69.1 fold) replaces it with the backend taken from `used_gpu`. The `Backend: GPU` lines are what show the GPU served.

### Unit tests (device-free; they run in the workspace lane)
- `route_top_k` rank, renorm and tie rules.
- FP32 scope across join, par_iter and nested joins; restored on panic.
- capacity adapter: a busy 4090 → CoTenant with the arithmetic; the measured gx10 → Fits; a starved GB10 → refused naming MemAvailable.
- `dispatch_gate` records an erroring gate as FAILED and continues.
- qa's MoE predicate reads `general.architecture` through the ONE bounded-prefix header reader, `model_header::gguf_arch_and_tensors` (#3750 PR A, which this branch is stacked on). The first cut called realizar's `MappedGGUFModel::from_path`, whose MAP_POPULATE + mlock faults in the whole file (#3761). On Instruct-2507 (18.5 GB) the helper path peaks at **58,836 kB** RSS for the whole test process in 0.08 s. The qa rows above were measured with the populated map, and the routing decision is identical.
- **Stacked on #3750 PR A** (`efe74818d`, AGREED 3/3), which folds first. Quorum base = `efe74818d`. The R1 commits are cherry-picked from `PMAT-3714-qwen3moe-cuda` (their `-x` trailers name the originals). The only resolved conflict is `qa_capability.rs`, where #3750 A moved the header parse into `model_header`.
- Existing `qwen3_moe` (38) and `moe` (139) unit tests unchanged-green.

## Not measured / not in this row
- (Closed at 21:52Z.) **Instruct-2507 on gx10:** the cop ruled to copy the file. The copy used `ionice -c3 nice -n 19 rsync --partial` with gx10 free = 116G ≥ the 100G guard, and the sha256 was verified identical on both hosts before use (`6c997b8af17debdfb01d890214400ccbab00db6acc0ba8da5de1cc906c4774d0`). The file is added to gx10's inventory for #3714. Measured at gpu-q prio 1, hold 21:45:59Z–21:52:32Z (`date -u`): all three cells are in the tables above.
- done_when 2 (`apr parity` for qwen3moe) is R2. done_when 4 (ladder rungs) is R4, with aprender-62.
- `apr chat --gpu` / `apr serve` for qwen3moe still use the CPU chain. They need a persistent resident model.
- RMSNorm ε (aprender-37's defect: modules cached without ε): not triggered here, because the model builds on a fresh executor and every norm launch uses the model's ε = 1e-6. Measured: position 0 = `<|im_start|>`, cosine 1.000000. It would matter if this model ever shared an executor with the dense path.
- Throughput is v1 (host routing, 48 syncs per token, float GEMVs): 87 tok/s on the 4090, 36 on GB10. R3's device kernels (oxide-first) target it.
