# #3817 — `apr run --gpu` on qwen3moe refuses pre-load by name

Measured by aprender-c7 on lambda (x86_64, NVIDIA GeForce RTX 4090 sm_89, 24564 MiB), 2026-09-22.
Binaries snapshotted outside every cargo target dir before use, per the cop's standing ruling.

| role | binary | sha256 (24) |
|---|---|---|
| BEFORE | `apr 0.69.0 (9f8836c71)` = `release/0.69.1-batch-2` | `6148b5618b907899fafc0c95` |
| AFTER | `apr 0.69.0 (3f26744ee)` = this branch | `3f480d107aec7d20f9ffe966` (first build) |

## done_when 1 — refused pre-load, by name

| | BEFORE `9f8836c71` | AFTER this branch |
|---|---|---|
| exit code | **14** — "gpu was forced and selected, but its runtime attempt on this model failed and the generation ran on CPU" | **12** (`NotImplemented`) |
| stdout | `=== APR Run ===`, `Source: …`, then the fallback error | **0 bytes** |
| when | after loading the model and generating on the CPU | before any load, any print |
| cost | 18 GB touched | **40 MB, 0.03 s** |

AFTER, both files, `--prompt "What is 2+2?" --max-tokens 8 --gpu`:

```
Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf   rc=12 elapsed=0.03s maxRSS=40256kB  stdout=0 bytes
Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf  rc=12 elapsed=0.02s maxRSS=40268kB  stdout=0 bytes
```

stderr, both:

```
error: Not implemented: this build has no CUDA forward for architecture 'qwen3moe' (canonical
'qwen3_moe'): the mixture-of-experts GPU path is #3714 and lands in 0.70.0. 0.69.1 runs this
architecture on the CPU only. Re-run without --gpu to use the CPU path deliberately, or use a dense
Q4_K model on the GPU. This is a refusal, not a fallback: nothing was loaded and nothing was generated.
```

**A correction I had to make to my own first implementation, because the first version's refusal was
correct and still paid for the load.** `declared_architecture` originally called
`MappedGGUFModel::from_path`, which maps with `MAP_POPULATE` (PMAT-304, matching llama.cpp) and
therefore *synchronously pre-faults every page*. Measured: rc 12, stdout empty — and **maxRSS
18,155,520 kB, 2.18 s warm / 8.77 s cold**. The refusal was right and the claim "nothing was loaded"
was wrong. It now reads a bounded GGUF **prefix** (1 MiB, then 8, then 64, stopping there because
past that it is reading weights): **40 MB, 0.03 s**, a 450× drop in resident memory. A pre-load
refusal that faults in the file it refuses is the load followed by a message.

## done_when 2 — the working CPU path is untouched

`apr run <moe> --prompt "What is 2+2?" --max-tokens 12 --no-gpu`, AFTER: **rc 0**, output
`2 + 2 = 4.`, 8.82 s. BEFORE on the same file: rc 0, `2 + 2 = 4.`, 8.00 s. Unchanged.

## done_when 3 — `apr qa` emits gates instead of an empty document

| | BEFORE | AFTER |
|---|---|---|
| exit | 5 | 5 (the model is still unsupported on the GPU — correctly red) |
| JSON | **0 bytes** | **3498 bytes, 12 gates** |
| `capability_match` | **absent** | **FAIL**, carrying the named refusal above |
| `golden_output` | **absent** | present (skipped, reason recorded) |

BEFORE aborted out of `run_qa` with
`Validation failed: CPU generation failed: Invalid shape: matmul weight has EMPTY data buffer
(in_dim=2048, out_dim=16384, qtype=0) … (2) a MoE parent FFN tensor whose real weights live in
per-expert slices the loader has not wired in` — the dense loader, raised as a hard error, so the
process exited 5 having printed nothing. Absence read as conformance to anything parsing the report.

Three changes, and the first one is general: `dispatch_gate` now turns a gate that cannot RUN into a
FAILED gate carrying the error, so **no gate can empty the report again, for any model**.
`is_cpu_only_architecture` admits an architecture whose CUDA forward was never built, and
`capability_match` refuses by name rather than reporting "all N required ops supported by GPU" — which
would have been *true and misleading*: no kernel is missing, the path was not built.

Full gate list is `qa-after.json`. Note the trade-off, stated rather than hidden: with
`capability_match` red, `golden_output` and `throughput` skip with "Skipped due to capability match
failure" rather than running on the CPU. They are present with a reason, which is what done_when 3
asks for; the CPU path is proven separately and directly by the `--no-gpu` run above.

## done_when 4 — every other architecture is unchanged

Case table, run under one GPU lock, `--gpu --max-tokens 6`:

| control | architecture | rc |
|---|---|---|
| `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` | qwen2 (dense Q4_K) | **0** |
| `Qwen3.5-0.8B-Q4_K_M.gguf` | qwen35 (hybrid rung) | **0** |
| `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | llama (non-qwen) | **0** |

The unit case table is `every_other_architecture_is_unaffected_by_the_refusal` (8 architectures) and
`architectures_with_a_cuda_forward_are_not_refused` (9), both classifying by asking the predicate.

## done_when 5 — the mutant

`removing_the_refusal_would_restore_the_silent_cpu_fallback` is RED the moment
`no_cuda_forward_reason` stops answering for qwen3_moe, and says what it caught: *"MUTANT CAUGHT:
--gpu on qwen3moe no longer refuses. Without this refusal the run loads 18 GB, generates on the CPU,
and exits 14 AFTER the fact — the user asked for the GPU and was told the wrong thing about their own
hardware (#3817)."* Deleting the `"qwen3_moe" =>` arm is exactly what #3714 will do, so the refusal
turns itself off with the fix rather than being a second thing to remember.

## done_when 7 — does 30B-A3B Q4_K_M fit the 4090? **The tool cannot answer today, and that is the finding.**

`apr serve plan` on this file reports:

```
Parameters:  3.35B          File size: 17697 MB
Model weights (Q4_K): 1599 MB      KV cache (batch=1, 4096 seq): 1536 MB
Total: 3679 MB / 24564 MB (15.0%)   ✓ BUDGET-001: VRAM fits
```

**1599 MB of weights for a 17,697 MB file is wrong by 11×.** It is budgeting the *active* parameters
(A3B = 3.35B) and a MoE must hold **every** expert resident. Control on a dense model, same binary:
`qwen2.5-coder-1.5b` reports 847 MB of weights for a 1066 MB file — plausible. So the error is
specific to MoE, not a general miscalibration.

Therefore: **do not read a fit verdict off `apr serve plan` for a MoE model.** The honest arithmetic
from measured quantities is weights ≈ 17.3 GiB of a 24564 MiB card ≈ 72% before any KV cache, which
makes the short rungs plausible and the long rungs not — but stating a rung-by-rung verdict on top of
a planner that is 11× out would be the false arithmetic claim the ticket explicitly warns against.
The fit question is recorded here, unanswered, with the reason it cannot yet be answered. It matters
beyond this row: #3791's ruling requires a capacity preflight built on `capacity::plan`.

## done_when 6 — inventory and notes

- `contracts/model-capability-ladder-v1.yaml`: the `qwen3moe` representative is removed from
  `long_rungs_for.representatives`, with the reason inline (not a fit question — no CUDA forward
  exists; restore with the GPU forward, not before). `pv validate`: 0 errors, 0 warnings.
- `README.md` Beat 8: a block stating by name that 0.69.1 does not support `qwen3moe` on CUDA, that
  `apr run` works and `apr run --gpu` refuses with exit 12, pointing at #3714.
- `CHANGELOG.md` under Unreleased: the same, with the before/after and the #3817 reference.

## Tests

`cargo test -p apr-cli --lib --features cuda` — **7333 passed, 0 failed, 12 ignored**.
`cargo test -p aprender-serve --lib` — **15967 passed, 0 failed, 59 ignored**.
`cargo clippy -p apr-cli --lib -- -D warnings` rc 0; `-p aprender-serve --lib` rc 0; `cargo fmt --all --check` rc 0.

**Not mine, found on the way, reported rather than fixed here:** `cargo clippy --features cuda` fails
on `crates/aprender-compute/src/matrix/ops/arithmetic.rs:413` — `unused import: GemmOp`. It is present
unchanged on `release/0.69.1-batch-2` (this branch touches no file in that crate), so it is a property
of the base, not of this row.
