# Row 17: sampling on the wgpu decoder (#3760)

Read at origin/main 11f844a772. `G:` cites are
`crates/aprender-serve/src/infer/gguf_gpu_generate.rs`, `S:` cites
`crates/aprender-serve/src/sampling.rs`. No build.

## Today
- A sampled request (`temperature > 0` and `top_k != 1`) skips the wgpu decoder:
  `wgpu_can_serve` (G:67-74) prints `WGPU_SAMPLING_NOTICE` (G:61-63) and returns false,
  at both call sites (G:536, G:600). The CPU loop runs it. Loud, so E4 lists it as
  T8, not as a silent fallback.
- The decoder is already split for this: `wgpu_greedy_decode` (G:133) takes
  `forward: FnMut(u32, usize) -> Vec<f32>` and `pick: FnMut(&[f32]) -> u32`
  (G:137-138). The greedy-only part is the `pick` the callers pass, not the loop.
- The CPU sampler is pure and seeded: `draw(logits, temperature, top_k, top_p, r)`
  (S:75), `draw_seeded(.., &mut StdRng)` (S:57), with a default seed when none is named
  (S:18). `S:198` already tests that a seed fully determines the draws.

## Finding R17-1: while the LM head is on the host, the fix is a closure
The logits that reach `pick` are host `Vec<f32>` (row 12: LM head and argmax are on
the host today). So sampling on wgpu is: build `StdRng` from the request seed, pass
`|l| sampling::draw_seeded(l, t, k, p, &mut rng)` as `pick`, and drop the
`wgpu_can_serve` gate. No shader, no device change.

## Finding R17-2: row 12 step 5 (device argmax) would undo it
Row 12 step 5 moves argmax to the device and reads back 4 bytes a token. A sampled
request then still needs the logits. Rule for row 12: device argmax only on the greedy
route; a sampled route reads back the logits (`vocab x 4` bytes a token) and draws on
the host. A device top-k (read back k pairs) is a later speed item, not a 0.73 need.
Ordering: row 17 lands before row 12 step 5, and F12-4's ticket must keep the sampled
read back path.

## Finding R17-3: what parity means for a sampled run
A sampled wgpu run and a sampled CPU run with one seed need not give the same tokens:
E1 allows cosine 0.995 on the logits, and a draw near a CDF boundary can flip. So
token equality is not the test. What is testable:
- the draw is the same function on both routes (same `r` and same logits give the same
  token; already `S:198` for the CPU),
- the wgpu route consumes the RNG exactly once per token, like the CPU route, so the
  streams stay aligned until the first divergent token,
- the first divergent step's logits still pass E1's cosine (the divergence is a
  boundary draw, not drift).

## Draft for the row-17 ticket
1. Pass a seeded `draw_seeded` closure as `pick` at G:536 and G:600; remove
   `wgpu_can_serve` and `WGPU_SAMPLING_NOTICE` (E4 T8 then closes).
2. The receipt for a sampled run records `seed`, `temperature`, `top_k`, `top_p`.
3. Keep the greedy closure (`argmax`, first max wins, S:36) for `is_greedy` requests
   (S:27), so E1's greedy cells do not change.

## Falsifiers
| id | claim | how |
|---|---|---|
| F17-1 | a sampled forced-wgpu run stays on wgpu | no `WGPU_SAMPLING_NOTICE` on stderr, `used_gpu=true` (E4 T8's falsifier, inverted) |
| F17-2 | one seed gives one wgpu token stream | two wgpu runs, same seed: equal tokens; seeds differ: streams differ within N tokens [U N] |
| F17-3 | the wgpu `pick` draws once per token | a counting RNG wrapper: draws == generated tokens |
| F17-4 | greedy unchanged | a greedy request through the new code gives E1's greedy tokens, same receipt |
| F17-5 | at the first token where wgpu and CPU differ (same seed), the logits cosine is >= 0.995 | log both at that step |

## Open
- Whether `--top-p` reaches `gen_config` and `config` at both call sites [U].
- R17-2 must be written into row 12's ticket; noted in `R12-wgpu-residency.md` step 5.
