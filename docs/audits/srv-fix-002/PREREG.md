# SRV-FIX-002 gx10 measurement — pre-registration (#4504, measures #4484)

Written and pushed **before** any measurement. Nothing below changes after the first run;
a deviation is recorded as a deviation, not an edit.

## Subject
- Code: `origin/79/4484-flash-prefill` @ `3b8fc87675` (split-KV flash prefill, default on).
  This branch adds only this file and the receipts; it makes no change to the kernel.
- Cell: gx10 (GB10, sm_121), `apr` built there from that sha with `--features cuda`;
  the binary's sha256 and `apr --version` are recorded.
- Model: `Qwen3.5-4B-Q4_K_M.gguf` (sha256 recorded at the first run).
- Paths: `APR_QWEN35_PREFILL_ATTENTION=flash` vs `=f32` (cuBLAS f32, the reference).
  GPU use is proven by the stderr line naming the attention path and by nsys kernel names,
  never by `CUDA_VISIBLE_DEVICES`.

## Prompts (5, fixed)
Built deterministically from files of this tree at `3b8fc87675` (no private text), each
ending in "Summarise the code above in one sentence.". Target lengths: ~30, ~2k, ~8k,
~16k and ~30k tokens. The generator command and each prompt's sha256 go in the receipt.

## P1 — greedy parity (pass/fail)
Greedy decoding, 64 new tokens. PASS iff the generated text is **byte-identical** between
`flash` and `f32` on **5/5** prompts. Any difference on any prompt is FAIL (reported with
the first differing token index), not "close enough".

## P2 — planted mutant (the test must be able to fail)
In a scratch copy only, `c_exp` in `prefill_flash_attention.rs::build_ptx` is multiplied
by 1.25 (a wrong softmax scale). P1 re-run with that binary must report **≥1 differing
prompt**. If the mutant passes P1, P1 is declared vacuous and no parity claim is made.

## P3 — speed vs llama.cpp (the #4504 bar)
30k-token prompt, same GGUF, same GPU, sequential runs (never co-located). `nsys profile`
+ `nsys stats --report cuda_gpu_kern_sum` for both `apr` (flash) and llama.cpp
(`/mnt/nvme-raid0/llama.cpp-d1d3c3396`, version+sha recorded). Attention kernel time is
the sum over kernels whose name contains the flash/attention kernel names (listed in the
receipt). PASS iff apr's attention time ≤ 2.0× llama.cpp's. Also reported, not gated:
end-to-end prefill time for flash vs f32 vs llama.cpp.

## Receipts
`docs/audits/srv-fix-002/gx10/` on this branch, summary comment on #4484 and #4504.
Timings, sha's and verdicts only; no completions beyond the parity diff index.
