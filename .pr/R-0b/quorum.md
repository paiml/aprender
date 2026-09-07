# R-0b review-only quorum (P3) — 2026-09-07

Lane: one agy quorum lane (agy 1.1.27, conversation 22cc3192-c3fd-4e01-be57-39d875657a4e, 527 s,
277,200 input tokens), review-only, over an exported .git-free tree at a0269f3c1 and the diff
d89db05a2..a0269f3c1 (28 files). Receipt: `docs/audits/quorum/PMAT-1073-lane-1.json`.
Dispatched through `paiml-agy-delegate` (resumed once at its turn cap).

## Verdict
Lane: **do-not-implement-as-written** on axis 6; PASS on axes 1–5. Delegate: **revise**.

- PASS (cited): `resolve_in` never lets a forced accelerator resolve to cpu; run/chat and serve
  precedence preserved (`accel.rs` normalisation, `serve/mod.rs` `accelerator_request`); the four
  rewritten serve tests still falsify #2696 over fixtures; the three decompositions are
  behaviour-preserving; the effective-config `resolved` block is honest because `matches_loaded`
  exposes a mismatch.
- FAIL (cited, then **measured by A on this box**): `crates/aprender-serve/src/infer/gguf_gpu_generate.rs:375-391`
  — after `selected: wgpu …` is announced, a runtime failure of `try_wgpu_generate` falls to CPU,
  prints only under `--verbose`, and returns `Ok((tokens, false))`. Measured: `apr run <0.5B
  q4_k_m> --gpu` printed `selected: wgpu device[0]=NVIDIA GeForce RTX 4090`, `Backend: wgpu
  (Vulkan)`, then (verbose only) `Backend: CPU (wgpu unavailable: Format error: Unsupported
  quantization type 6 for WGPU dequant)`, and exited 0. Quiet run: 11 lines, no fallback notice.
  The row claim "never downgrades a forced one" was false end to end.

## Dissent (delegate vs lane and vs the receipt), accepted
- The receipt's stated cause for `compiled("wgpu")` being true on default builds was wrong:
  `wgpu = ["inference"]` implies one way only; the real cause is aprender-serve's unconditional
  `trueno = { features = ["gpu"] }`. Corrected in the receipt.
- Verdict calibration: REG-OB-004 is scoped to the CLI boundary and the code satisfies it; the
  overreach was the row's prose claim. Resolution below.

## Fold (A's decision)
Fix at the apr-cli boundary, which R-0b owns and which is hook-editable: realizar already returns
`used_gpu` per run. After generation and BEFORE any output, a forced accelerator whose runtime
attempt fell to CPU is refused (`BackendUnavailable`, exit 14, the reason named); a default
selection that fell to CPU prints `selected: cpu (fallback …)` unconditionally so the last
`selected:` line is always what ran. The realizar-side pre-generation refusal (so the CPU run is
not spent first) needs `try_wgpu_generate` (cognitive 49) and `try_apr_wgpu_inference` (69)
decomposed — pre-existing debt in a hook-charged file — filed as a follow-up.

## Self-refuted by the delegate (recorded so it is not re-raised)
`Request::wanted()` returning Cpu on `no_gpu` does not invert GH-326 (accel.rs normalises first);
`--backend metal|hip|gpu` are not reachable CLI inputs (`BACKEND_VALUES`).
