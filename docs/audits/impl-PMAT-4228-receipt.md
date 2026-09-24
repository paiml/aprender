# PMAT-4228 — implementation receipt: parallel attention and delta rule in the batched prefill

Ticket #4228 (child of CRUX #4223 T1): batched Qwen3.5 CPU prefill over a
multi-row K-quant GEMM.

- The GEMMs and the layer-major prefill landed on `batch/0.70.0` from
  aprender-88: `b209a0150`, `f3ba1635c`, `eac28855d`, `c91ae0dae`.
- This follow-up is branch `feat/4228-prefill-parallel` (`b2583f668`, on
  `161f45b70`). It changes one thing: the two per-token steps the batched
  prefill still ran serially now run in parallel.

## What changed (`crates/aprender-serve/src/gguf/inference/forward/forward_qwen35.rs`)

**Attention.** `attention_mix_token` is split into two halves:

- `attention_append_token` (mutating): q/gate split, the per-head norms, the
  partial NEOX RoPE and the KV append.
- `attention_attend_token` (read-only): scores, softmax, the weighted V sum and
  the sigmoid gate.

In the per-token path, `attention_mix_token` is now simply append followed by
attend.

`attention_mix_rows` does this for one chunk:

1. Appends every row's K/V, in order.
2. Attends the rows in parallel with rayon. Row `t` reads positions
   `0..=pos0+t` only, which are exactly the rows the per-token path would have
   appended by then.

**Delta rule.** The body of `delta_rule_recurrence_gqa` for one value head is
now `delta_rule_head`. The per-token recurrence calls it once per head, so it
is unchanged.

`deltanet_mix_rows` does this for one chunk:

1. Runs `deltanet_prep_token` in token order, because the conv window is
   shared state. This step covers the conv, SiLU, the q/k L2 norms, `dt` and the
   beta sigmoid.
2. Lets each value head walk every token on its own thread.
3. Applies the gated RMS norm per token.

This is safe because a head's slice of `ssm_states[il]` is read and written by
that head alone. On GQA files, head `h` reads key head `h % num_k_heads`, as
before.

**Order of arithmetic.** For every (token, head), the per-token path's
arithmetic runs in the per-token path's order, so no reduction order changes.

## Evidence

**Bitwise identity.** `APR_QWEN35_REQUIRE_MODEL=1`, release build, run on this
branch.

- `falsify_4228_004_prefill_bit_identical_0_8b` and
  `falsify_4228_005_prefill_bit_identical_4b` pass. They cover logits, conv,
  ssm, K and V; two prefills (135 = 2×64+7 tokens, then 5 from position 135) across
  chunk boundaries; and one decode after.
- The same filter covers the whole `qwen35` set: 48 passed, 1 ignored (the
  perf probe).

**Planted mutants.** Each was planted in the committed tree and restored by an
EXIT trap; `git status` was clean after.

| Mutant | 0.8B test | 4B test |
|---|---|---|
| M1: every row attends at `pos0` (not `pos0+t`) | fails: "prefill logits differ" | fails: "prefill logits differ" |
| M2: `dt[kh]` for `dt[h]` in the per-head walk | passes (ratio 1, so the mutant is equivalent there) | fails |

**Throughput.**

- **Setup:**
  - Probe: `qwen35_prefill_throughput_probe` on `Qwen3.5-4B-Q4_K_M.gguf`, the same GGUF throughout.
  - Host: lambda-vector, a Threadripper 7960X with 24 cores and 48 threads.
  - Runs were serialized with `flock /tmp/cb4228-cpu.lock`, and load1 was read at the start and end of every run.
- **Binaries:** both are test binaries built with `--release`, each copied
  right after its build; `cmp` confirmed they differ.
  - **base** = `161f45b70`, i.e. `batch/0.70.0`.
  - **new** = `b2583f668`.
- **Order:** new and base runs alternated.
- **Logs:** `/mnt/nvme-raid0/tmp/cb4228p/{quiet,ab}.log`.

Quiet window (load1 at run start 6.8–35.5, mostly under 24):

| prompt | base, batched (tok/s) | new, batched (tok/s) | new / base |
|---|---|---|---|
| 850  | 25.5 / 25.5 / 25.7 | 52.2 / 54.5 / 54.1 | 2.1× |
| 4096 | 10.8 / 11.0 / 11.5 | 52.0 / 45.1 / 50.9 | 4.4–4.7× |

Batched against per-token, in the same process, on the new binary at 850
tokens (load1 at start 6.8 / 29.6 / 33.8):

| run | batched (tok/s) | per-token (tok/s) | speedup |
|---|---|---|---|
| 1 | 54.5 | 6.5 | 8.37× |
| 2 | 54.0 | 6.2 | 8.68× |
| 3 | 54.4 | 6.0 | 9.04× |

On base, prefill tok/s fell by more than half from 850 to 4096 tokens, because
attention was serial and O(n²). On new, it holds flat at about 50 tok/s.

Contended runs (load1 56–110) kept the same direction: base 11.8/14.6/13.9 vs
new 18.6/26.9/25.2 at 850, and base 8.5/8.9/10.8 vs new 22.9/26.8/51.5 at 4096.

## Checks

- `cargo fmt --all -- --check`: clean.
- `cargo clippy -p aprender-serve --lib -- -D warnings`: exit 0.

## Not in this row

- The 0.8B throughput was not re-measured; its bitwise tests pass.
- The per-token path at 4096 tokens was not timed. At about 6 tok/s it would
  take more than 10 minutes per run, and the quiet window was closing.
- Throughput was measured with the in-process probe, not `apr run`.
- My earlier duplicate branch `feat/4228-qwen35-cpu-prefill` (pushed WIP on
  `rc/0.69.3-rc.2`) is superseded by aprender-88's commits and this branch. It
  needs no PR.
