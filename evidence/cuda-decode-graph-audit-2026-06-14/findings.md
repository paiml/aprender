# Live cuda decode — graph capture/replay staleness audit (2026-06-14)

Adversarial hunt over the live single-request/batched cuda decode (graph capture/replay, cuda KV
cache, q8 dequant, cross-request reset, prefill handoff), with a built-in LIVENESS gate.
**27 verified → 5 REAL: 3 "live", 2 dead-path.** Hand-verified below — the "live" ones are real
graph-staleness bugs but with NARROW triggers (batched m>1 + long/growing sequences), and one is
behind an off-by-default flag. Not tick fixes; a focused GPU-validated follow-up (like PMAT-764).

## Root cause (verified): batched decode graphs keyed by batch size M ONLY
`batched_decode_graphs: HashMap<usize, CudaGraphExec>` (executor/mod.rs:472) — insert(m,…)
(par-121.rs:57), get(&m)/contains_key(&m) (par-121.rs:413, batched_forward.rs:427). A graph
captured for batch size M at one seq_len is REPLAYED for all later decodes at M **regardless of
seq_len**. So anything baked into the graph from the capture-time seq_len (attention-algo choice,
max_chunks, buffer pointers) goes stale as the sequence grows. That is the common mechanism
behind all three findings.

## REAL but narrow-trigger (focused follow-up; need GPU validation under batched + seq>1024)

- **[1]/[3] (high) Stale `batched_kv_lengths` baked into the graphed attention dispatch.**
  `dispatch_attention` (graph_dispatch.rs:247-253) chooses flash-decode vs incremental from
  `batched_kv_lengths`; `batched_incremental_attention_into` (batch.rs:240-250) skips the CPU-side
  length update during capture. With the graph keyed by M only, the flash-vs-incremental decision
  and `max_chunks` are fixed at capture-time seq_len. As the sequence grows past the flash
  threshold (1024) the replayed graph can run the wrong kernel / too few chunks → incomplete
  attention → wrong output. LIVENESS: `use_graph_dispatch()` (GRAPH_DISPATCH) is **ON by default**
  (graph_decode.rs:145-147), so the per-layer graph path is live — BUT only for **batched
  (m>1)** decode and only once seq crosses the threshold. Single-request (c=1) and short batched
  sequences are unaffected (consistent with the coherent default output we observe).
- **[2] (high, but OFF by default) Q8 buffer realloc inside workspace early-return doesn't clear
  graphs.** `init_batched_workspace` (workspace.rs ~127-152) can reallocate the Q8 buffer (new GPU
  address) in its early-return branch without clearing `batched_decode_graphs`, so a replayed
  graph holds a stale Q8 pointer → garbage. This is on `forward_batched_to_token_ids_graphed`,
  gated by **`BATCHED_GRAPH=1` (off by default**, eager preferred per PMAT-056). Low default impact.

## DEAD-PATH (deprioritize — verify-not-live confirmed by the hunt)
- flash_attention_cached.rs GQA/write-position findings — the verifier flagged these as not on
  the live serve decode path.

## Candidate fix direction (for the focused effort)
Key `batched_decode_graphs` by (M, seq-bucket) OR invalidate/re-capture when the attention-algo
threshold (flash vs incremental, max_chunks) would change, OR refresh `batched_kv_lengths` and
recompute the dispatch decision outside the captured region so it isn't baked. For [2]: clear
`batched_decode_graphs` whenever the Q8 (or any workspace) buffer is reallocated, including the
early-return branch. ALL need GPU validation under batched (m>1) + sequences crossing 1024.

## Method / lesson
Liveness-gated hunt worked: it separated 3 "live" from 2 dead-path up front. Hand-verifying the
graph cache key (M-only) confirmed the staleness mechanism is real, while the trigger analysis
(batched + seq>1024; [2] flag-gated) shows these are narrow, not the everyday c=1 path — so a
careful GPU-validated fix, not a rushed patch.
