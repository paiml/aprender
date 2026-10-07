# Row 10: leg A for each E1 route (a wgpu arm, a Qwen3.5 arm)

Read at origin/main 4098007133 (newer than the 11f844a772 pin of rows 9-19; the cites
below are at 4098007133). `G:` cites are
`crates/aprender-serve/src/infer/gguf_gpu_generate.rs`. No build.

## Leg A today
`apr parity` picks one of three CPU-vs-CUDA pairs by architecture
(`crates/apr-cli/src/commands/parity_hybrid.rs:16-25`): Hybrid (Qwen35 CPU vs
Qwen35Cuda), Moe, Dense (`forward_single_with_cache` vs `forward_gpu_resident`). Without
the cuda feature it refuses (`crates/apr-cli/src/commands/parity_03.rs:214-217`). So no
leg A runs on mini (no CUDA) and no leg A measures a wgpu forward anywhere.

## Finding R10-1: a CPU-vs-wgpu pair already exists, inside the routes
Both wgpu generate routes run FALSIFY-CPU-GPU-006 before decoding: CPU
`forward_single_with_cache` against wgpu `forward_layer`, lockstep for 3 steps
(`APR_WGPU_PARITY_STEPS`, 1-16), advancing both on the CPU argmax, falling back to the
CPU with a tagged message below cosine 0.99 (`try_wgpu_generate` G:160, probe G:233-325; `try_apr_wgpu_inference` G:624, probe G:742-824; gates at
G:304 and G:803). That is a leg-A measurement with no receipt. A wgpu arm for
`apr parity` should be this probe lifted into one shared function, called by both
routes and by `apr parity --backend wgpu`, so leg A measures the code the route runs
(the FALSIFY-BPM-019 shape), not a copy.

## Finding R10-2: the route gate and E1 use different thresholds
The routes accept 0.99 (G:304, G:803); E1 is 0.995 (BPM thresholds). A route can pass
its own gate and fail E1. With #3757's measured 0.955 both fail today, so a wgpu
generate falls back to the CPU after the probe (loud, E4 kind F). When P4 lifts the
cosine, the gap 0.99-0.995 becomes a window where a route serves on wgpu and E1 calls
it FAIL. Fix: the shared function takes the threshold from one place (the BPM
thresholds), not a literal.

## Finding R10-3: four copies of output norm + LM head, and the probe skips `pick`
Output RMSNorm and the host LM head are written out four times in G: probe 1 (G:290-300),
route 1's `pick` (G:343-356), probe 2 (G:788-800), route 2's `pick` (G:846-858). The
probe compares logits it computed itself, not the ones the route's `pick` sees. A fix
to one copy (row 12 step 4 moves the LM head to the device) can leave the probe
measuring the old path. Same drift shape as R14-3.

## Finding R10-4: generate reaches the hardcoded RoPE theta
Both routes' forward closures call `fwd.forward_layer` (G:337, G:840), and so do both
probes (G:282, G:780). That answers row 12's open item: T5 (generate) reaches the
`rope_theta = 1e6` line too (#4832, R12-1), as do serve and batch. The probe does not
catch it: CPU and wgpu each apply their own theta, so on a Llama model the probe sees a
cosine drop and falls back, which reads as "wgpu is inaccurate", not "wrong config".

## Qwen3.5 arm
- CPU leg A exists (Hybrid arm) but steps one token at a time while run and serve
  prefill in one batched call (R1 §13): leg A does not measure the prefill the routes
  run. The arm needs a batched-prefill step: feed the prompt through the batched path,
  then compare per-token decode.
- A wgpu Qwen3.5 arm waits for row 6 (no wgpu forward in `Qwen35Session`).

## Draft for the row-10 ticket
1. `wgpu_parity_probe(model, fwd, tokens, steps, threshold) -> Vec<StepMetric>` in
   aprender-serve, used by both routes (G:233, G:742) and returning metrics, not only a
   pass bit. One shared `logits_from_hidden` replaces the four copies.
2. `apr parity` gets `ParityArm::Wgpu`, built under the gpu feature without cuda, so
   mini and intel can run it; it writes the same report as the other arms
   (`emit_parity_outcome`, `parity_03.rs`) plus the adapter block from row 13.
3. One threshold source (BPM), used by the routes and by E1.
4. Hybrid arm: a batched-prefill mode matching run and serve.

## Falsifiers
| id | claim | how |
|---|---|---|
| F10-1 | the route and `apr parity --backend wgpu` measure the same function | a planted change in the shared probe changes both outputs; a change only in the route's copy is impossible (no copy left) |
| F10-2 | the probe compares the logits the route's `pick` sees | planted LM-head bug in `logits_from_hidden` fails the probe |
| F10-3 | one threshold | route and E1 checker read the same constant; a cosine of 0.993 fails both |
| F10-4 | leg A runs without cuda | `apr parity --backend wgpu` on a gpu-only build exits 0 or 1 with a report, never `cuda feature required` |
| F10-5 | the Qwen3.5 arm prefills like run | a prompt of 32 tokens is measured through the batched path; per-token-only arms are refused for E1 |

## Open
- Which activation mode `forward_single_with_cache` uses vs RQ-5's `fp32_act` CPU
  reference [U]; if it is not fp32_act, the arm must call the RQ-5 reference.
- Route 2 is the .apr entry (`try_apr_wgpu_inference`, G:624); route 1 is GGUF (`try_wgpu_generate`, G:160). Both need the shared probe.
