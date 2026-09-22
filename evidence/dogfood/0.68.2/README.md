# 0.68.2 — first-green of the train's host receipts (#3731)

These receipts were produced by `scripts/release/host_receipt.sh 0.68.2 <host>`, run from a kit
on each host against the PUBLISHED crate (`cargo install aprender --version =0.68.2 --locked`
into a scratch root; `.crate` sha256 published = measured = `5dc9e307…44ae85` on every host),
per the cop's ruling on #3731 (2026-09-21): a gate never green on its own target may not block, so
the producer was proved on a real target before its blocks could STOP a train. Every block that
went green here BLOCKS from 0.69.1; every one that could not is listed in
`scripts/check_multiplatform_dogfood.sh`'s `REPORT_ONLY` with the issue that owns its refusal.

| host | kit (the script at) | run | notes |
|---|---|---|---|
| lambda | kit6, `--gpu-prio 8` | 2026-09-21 21:38Z → 23:51Z | parity block VALID (4 cpu bands lock-free; the accel probe under the per-band lock). kit6 predates the `parity.parity` unwrap fix, so the file it wrote nested the block; this copy applies exactly that transformation (`receipt["parity"] = receipt["parity"]["parity"]`) and nothing else. |
| intel | kit4 | 2026-09-21 18:27Z → 19:57Z | parity refused: band[16] subject rate 0 (#2855); bench block PRESENT (this run passed H12; the two earlier runs on intel refused at 9.6 tok/s, so bench stays REPORT for 0.69.1, #3758) |
| mini | kit7 (JSON twins, no GPU rule) | 2026-09-21 22:05Z → 23:44Z | parity refused: band[8]/band[16] subject rates 0 (#2855); bench refused (H12, #3758) |
| gx10 | kit6, `--gpu-prio 8` | 2026-09-21 21:38Z → 02:12Z (2026-09-22) | parity refused: the four cpu bands ran lock-free, then the accel probe was NOT admitted by the GPU queue within GPUQ_WAIT=3600 s (release-priority tickets held the lock), so the receipt says the accel lane is UNMEASURED rather than "0 layers"; the same starvation refused this run's `generate` and bench (gpu-q exit 75), so `generate` is null here. gx10's kit3 run (2026-09-21, `/tmp/tmp.m90Y4OUpIi/.receipt-work/generate.log` on gx10) had measured it green: `apr run --format json` rc 0, `"text": "4"`, `"used_gpu": false`, 12.2 s, after `Backend: wgpu (Vulkan)` and the same cosine-0.955 rejection |

Nothing else in these files was edited. The producers' full logs (`.receipt-work/`) were kept on
each host for diagnosis; the refusal reasons are in each receipt's `bench_attempt` /
`parity_attempt`.

This directory is NOT read by the gate for any cut (it reads `evidence/dogfood/<version being
cut>/`); it is the evidence that the producer and the gate's new blocks were green on a real
target before they were allowed to block.

## lambda.json is not in this directory

lambda's receipt is 424 KB (its parity block carries every replicate's per-request rows), which is
larger than a reviewable diff, so it is kept out of the tree: https://gist.github.com/noahgift/47ff689975e9752afc58abb37872a71d (sha256 `e497e26a345df3520c64ad9fbb3663ea53e344aae20d37961ff72fa76850cec4`). The other three
receipts are beside this file. `check_multiplatform_dogfood.sh` over all four (the preview in the
PMAT-3731 fragment) was run with lambda.json present.
