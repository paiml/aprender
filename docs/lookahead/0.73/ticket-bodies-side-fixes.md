# 0.73 L3 side-fix ticket bodies, S1 to S3 and S5 (DRAFT, not filed; la-73, 2026-10-03; S5 2026-10-04)

These are four fixes found while writing the 0.73 drafts. None of them blocks 0.73: no 0.73 model reaches the S1 or S2 code, S3 deletes two files that nothing compiles, and S5 makes a guard catch every such file. (S4, a pv lint rule, has no body here.)
- **Filing:** as PROPOSE-TICKET lines next to P1..P5 (`ticket-bodies-P1-P5.md`). S1..S3 went out on 2026-10-04 at 03:32Z, after LIVE 0.70.1; S5 goes out after the commit that adds it. L3 never mints (C277).
- **Format:** the P-body shape. Each acceptance names a falsifier that is red at 316dee2cd4, a control that shows the test can tell the two paths apart, and the mutations that must turn it red.
- **Citations:** at origin/main 316dee2cd4. Every defect marked [V] was read there.
- **Contract:** S1 and S2 amend `cpu-q4k-activation-quant-v1`, the contract that `fused_gate_up.rs:173` names. The draft is `contracts-draft/cpu-q4k-activation-quant-v1.yaml` (1.1.0). It adds the equation `activation_path_selector`, the obligations AQ-SEL-005 and AQ-SEL-006, and FALSIFY-AQ-005..007.

## The selector that S1 and S2 restore
A Q4_K matvec on f32 input takes f32 activations when any of these holds, and Q8_K activations otherwise:
1. the calling thread runs inside `with_fp32_activations` (`fp32_activations_scoped()`, parallel_k.rs:210);
2. `DIRECT_FP32_GEMV=1`;
3. the entry checks for crushed blocks and `has_crushed_block(x)` is true (crates/aprender-serve/src/quantize/mod.rs:409). A 256-block is crushed when max|x| / second|x| ≥ 8.0 (:404).

`fused_q4k_parallel_matvec_into` implements 1 and 2 (parallel_k.rs:303-304). `matvec_into_honest` adds 3 and notes each switch (ffn_block.rs:784-795). Three comments promise the selector:
- inside the scope, "every Q4_K matvec it reaches" takes the direct FP32 path (parallel_k.rs:214);
- the scope works "as `DIRECT_FP32_GEMV=1` makes every thread's do" (parallel_k.rs:202-204);
- on a crushed block, "the Q4_K drivers run the f32-activation dot ... for that call" (crates/aprender-serve/src/quantize/mod.rs:394-395).

S1 and S2 fix the two entries that break these promises.

---

## S1 — `fused_gate_up_q4k_into`: follow the selector, and make the #2971 crushed fallback f32
**Refs** #2971 (L0-1b), #3714 (scope), #3999 (found in 0.73 R1 §11a and §11b). Contract: `cpu-q4k-activation-quant-v1` 1.1.0 draft, AQ-SEL-005 and AQ-SEL-006.

**Defect [V]** `fused_gate_up_q4k_into` (fused_gate_up.rs:177) reads neither the scope nor the env var.
- **Crushed input, default route.** The fallback (fused_gate_up.rs:189-213) calls `fused_q4k_parallel_matvec_into` for gate and for up. Outside the scope that function quantizes to Q8_K (parallel_k.rs:303-304, then :326), so the remedy changes nothing. The fallback never calls `note_crushed_fallback` either, so the `[crushed-block]` count that a speed basis cites (crates/aprender-serve/src/quantize/mod.rs:431-432) misses these calls.
- **Uncrushed input, inside the scope or with `DIRECT_FP32_GEMV=1`.** Phase 1 quantizes to Q8_K (fused_gate_up.rs:214-224), then `fused_q4k_q8k_ffn_up_gate_into` runs (:227).

**Reach** The only caller is `fused_gate_up_matmul_into` (fused_matmul_into.rs:163). Its Q4_K arm (:186-187) runs only for seq_len 1 with matching qtypes and dims. That function is called from:
- the non-fused gated FFN branch (ffn_block.rs:58), taken by Gemma-1, LayerNorm models and models with no FFN norm;
- `scratch_q8k_up_gate` when `!use_q8k_path` (crates/aprender-serve/src/gguf/inference/forward/results.rs:103).

No 0.73 model reaches it (R1 §11b). Neither scope caller at 316dee2cd4 reaches it either (apr-cli parity_moe.rs:110, qwen3_moe_dispatch.rs:209).

**Scope**
- Add `pub(crate) fn q4k_takes_f32() -> bool` to parallel_k.rs. It returns `fp32_activations_scoped() || std::env::var("DIRECT_FP32_GEMV").as_deref() == Ok("1")`. parallel_k.rs:303-304 calls it, and its behaviour there does not change.
- Replace fused_gate_up.rs:189-213 with:
  ```rust
  // L0-1b (#2971) and #3714: the Q4_K selector, as fused_q4k_parallel_matvec_into applies it.
  let crushed = super::has_crushed_block(activations);
  if crushed || super::parallel_k::q4k_takes_f32() {
      if crushed {
          // One note per projection, on the calling thread: the rate counts matvecs.
          super::note_crushed_fallback(in_dim, out_dim);
          super::note_crushed_fallback(in_dim, out_dim);
      }
      let (g, u) = rayon::join(
          || super::fused_q4k_parallel_matvec_f32_into(gate_weight_data, activations, in_dim, out_dim, gate_output),
          || super::fused_q4k_parallel_matvec_f32_into(up_weight_data, activations, in_dim, out_dim, up_output),
      );
      g?;
      return u;
  }
  ```
- The doc at fused_gate_up.rs:171-176 says the function always pre-quantizes to Q8_K. Change it to name the selector.
- Add a test-only counter to quantize/mod.rs: a `#[cfg(test)]` thread-local `Cell<usize>`. `note_crushed_fallback` bumps it first thing, before the `APR_CRUSHED_TRACE` check. Because it is thread-local, tests running in parallel never see each other's notes.

**Acceptance** The tests go in `fp32_activation_scope_tests` (parallel_k.rs:502-545), next to the scope's own tests.
- **Inputs.**
  - in_dim 512 and out_dim 8.
  - Weights follow the `create_q4k_weights` pattern (parallel_k_test_helpers.rs:7).
  - Uncrushed x is a smooth sequence that is not made of powers of two, with max/second < 2 in every block.
  - Crushed x has the layer-26 shape in block 0: 146.68, then 7.28, then a 0.5 fill (ratio 20.1; `the_layer_26_block_is_crushed` builds it with `block_with`, crates/aprender-serve/src/quantize/mod.rs:444 and :452-454). Block 1 is uncrushed.
  - The reference is `fused_q4k_parallel_matvec_f32_into` for each projection.
  - Each test first asserts that `DIRECT_FP32_GEMV` is unset, so the environment cannot make a control pass vacuously.
- **FALSIFY-AQ-006.** Inside `with_fp32_activations`, on uncrushed x, gate and up are bit-identical to the reference. It is red at 316dee2cd4, where `fused_q4k_q8k_ffn_up_gate_into` runs. Control: the same call outside the scope differs from the reference.
- **FALSIFY-AQ-007.** Outside the scope, on crushed x, gate and up are bit-identical to the reference, and exactly 2 notes are counted. Uncrushed x gives 0 notes. It is red at 316dee2cd4: Q8_K, and 0 notes. Control: two `fused_q4k_parallel_matvec_into` calls (today's fallback) on crushed x differ from the reference.
- Notes are counted only outside the scope. Inside it the call runs on a pool worker, and the test thread's counter cannot see that thread's notes.
- **Mutations.** Each must turn a test red, and the PR body records the run:
  - drop `q4k_takes_f32()` from the new condition → AQ-006;
  - make `q4k_takes_f32` return `false` → AQ-006; make it return `true` → the AQ-006 control;
  - put `fused_q4k_parallel_matvec_into` back in the fallback → AQ-007;
  - delete one note or both → the AQ-007 count;
  - move the notes into the `rayon::join` closures → the AQ-007 count. Called from a thread outside any pool, `join` runs both closures on pool workers. That is why the sketch notes before the join.
- `cargo test -p aprender-serve --lib quantize` and `cargo test -p aprender-contracts --lib` pass.

**Blocked by** Nothing. S2 can share the PR; whichever of S1 and S2 lands first adds `q4k_takes_f32`.
**Hosts** x86 CI lib tests. No model, no GPU.
**Out of scope**
- Entries that receive activations already quantized to Q8_K (`scratch_q8k_up_gate`, crates/aprender-serve/src/gguf/inference/forward/results.rs:18-33; the traced forward, traced.rs:88). Their callers choose Q8_K themselves.
- Caching the env read. It is read on every call today (parallel_k.rs:304), and the helper keeps that.
- AQ-EQ-001's wording. It says the Q8_K dot is within 0.1% of the f32 dot for every input (`applies_to: all`), which is false on crushed blocks: L0-1b measured 13% low on the layer-26 gate and up outputs (crates/aprender-serve/src/quantize/mod.rs:388-393). The draft records this in the selector's invariants and leaves the obligation's wording to the contract owner.

---

## S2 — `fused_q4k_multirow_matmul_f32_into`: follow the scope
**Refs** #3714 (scope), #4228 (multirow), #3999 (R1 §11a). Contract: `cpu-q4k-activation-quant-v1` 1.1.0 draft, AQ-SEL-005.

**Defect [V]** The multirow entry (q4k_q8k_multirow.rs:161) takes f32 only when `DIRECT_FP32_GEMV=1` (:178-179). With m > 1 and in_dim a multiple of 256, a call inside the scope still quantizes each token row to Q8_K (:194-200) and runs the Q8_K multirow kernel (:201).
- The scope's doc says every Q4_K matvec inside it is f32 (parallel_k.rs:214).
- The multirow's own doc names only the env var (q4k_q8k_multirow.rs:154-157).

**Reach** The only caller is `matmul_rows` (forward_qwen35.rs:1416, call at :1425). Only `prefill_chunk_qwen35` uses that (:1228). Only `forward_prefill_qwen35` calls it (:1200, call at :1223), and only its tests call that (forward_qwen35_prefill_tests.rs). Neither scope caller at 316dee2cd4 runs a qwen35 prefill. So this is a latent trap, not a live bug: an `fp32_act` reference over a qwen35 chunked prefill would run its m > 1 chunks on Q8_K and still report itself as exact.

**Scope**
- q4k_q8k_multirow.rs:178 becomes `let direct_fp32 = q4k_takes_f32();`. The file is `include!`d into parallel_k.rs (:500), so the helper needs no import.
- Update the doc at q4k_q8k_multirow.rs:154-157. The per-row fallback runs when m == 1, when in_dim is not a multiple of 256, or when the selector picks f32 (inside `with_fp32_activations`, or with `DIRECT_FP32_GEMV=1`).
- The bit-identity claim at q4k_q8k_multirow.rs:154-155 still holds: each token row goes through `fused_q4k_parallel_matvec_into`, which takes f32 under the same predicate.

**Acceptance**
- **FALSIFY-AQ-005.** With m = 3, in_dim 512 and out_dim 8, inside `with_fp32_activations`, every token row is bit-identical to `fused_q4k_parallel_matvec_f32_into` on that row. It is red at 316dee2cd4, where the Q8_K multirow runs.
  - Control: outside the scope, the same call differs from that reference.
  - Token 1 carries the layer-26 crushed block, so the Q8_K error is large, not a single ulp.
  - The test asserts that `DIRECT_FP32_GEMV` is unset.
- **Mutations:** restore the env-only read at q4k_q8k_multirow.rs:178 → AQ-005; drop `direct_fp32` from the :179 predicate → AQ-005.
- **Interaction:** NEON-Q4K-008 calls this entry outside the scope on purpose (R3-test-skeletons.md §4c: entry G, and "Why the prefill entries"). S2 does not change what it measures.
- The commands listed for S1 pass.

**Blocked by** Nothing. It shares `q4k_takes_f32` with S1.
**Hosts** x86 CI lib tests.
**Out of scope**
- A crushed check in the multirow. #2971 scoped its remedy to the reference forward (direct_f32.rs:6-7). Whether a default-route chunked prefill needs one is an open question, and nothing here proposes it.
- P1..P5. Their receipts read the path from the trace (BPM-008, f008d), so they neither need S1 or S2 nor block on them.

---

## S3 — delete the orphans `quantize/fused_q4k.rs` and `quantize/fused_q.rs`
**Refs** #3999 (R3 §13 rows 1 and 9; R1 §11a). This body covers both orphan PROPOSE-TICKET lines (07:49Z and 18:21Z). They can be filed as one ticket or two, because the acceptance splits cleanly by file.

**Defect [V]** No `mod` or `include!` names either file, so neither compiles. Both still look live.
- `fused_q4k.rs` (360 lines) is a byte-identical copy of `q4k_dot_avx2.rs`, which `fused_k.rs:370` includes. A NEON arm added to the copy compiles away (R3 §13 row 1).
- `fused_q.rs` (327 lines) is an older copy of `q5k_q6k_matvec.rs` (included at parallel_k.rs:498) that has since diverged: the first 114 lines match, then 155 lines differ.
  - It holds a stale, uncompiled `fused_q4k_q8k_ffn_up_gate_into` (fused_q.rs:253) beside the live one (q5k_q6k_matvec.rs:380). That is the kernel S1 routes around, and a search for it finds two definitions.
- Four falsify tests cite `fused_q.rs` as the CPU path (listed below), and `crates/aprender-serve/.pmat-baseline.json` lists both files.

**Scope**
- Delete `crates/aprender-serve/src/quantize/fused_q4k.rs` and `crates/aprender-serve/src/quantize/fused_q.rs`.
- Keep `quantize/tests/fused_q4k.rs`. The `include!("fused_q4k.rs")` at `quantize/tests/dequantize.rs:509` resolves next to the including file, so it names this test file. The file is live and listed in `scripts/include_fmt_baseline.txt:1479`.
- Re-point the four doc-cites:
  - falsify_q6k_fp_accumulator_order_001.rs:101, falsify_q6k_activation_amplification_002.rs:84 and falsify_q6k_chain_length_003.rs:104 name `q5k_q6k_matvec.rs::fused_q6k_parallel_matvec` (:34);
  - falsify_q4k_bisect_dequant_007.rs:128 ("CPU fused") names `parallel_k.rs::fused_q4k_parallel_matvec` (:190).
- In `crates/aprender-serve/.pmat-baseline.json`, drop the entries keyed `./src/quantize/fused_q.rs` (`.pmat-baseline.json:80471`) and `./src/quantize/fused_q4k.rs` (`.pmat-baseline.json:117217`). Keep `./src/quantize/tests/fused_q4k.rs` (`.pmat-baseline.json:38151`).
- NEON-Q4K-006 used `fused_q4k.rs` as its orphan control. On this branch the control now plants its own unnamed file (R3-test-skeletons.md §4a; the neon-q4k-q6k-v1 draft), so the delete does not weaken that probe.

**Acceptance** These are one-shot checks recorded in the PR body, not a contract falsifier.
- **Compiled code is unchanged.** The output of `cargo test -p aprender-serve --lib -- --list` is identical before and after, compared by sha256.
- **No cite is left.** `git grep -n -E 'quantize/fused_q\.rs|quantize/fused_q4k\.rs' -- crates` prints nothing.
- **Mutations:**
  - delete `q4k_dot_avx2.rs` instead of its byte-identical twin → the build fails at the `fused_k.rs:370` include;
  - also delete `quantize/tests/fused_q4k.rs` → the build fails at the `tests/dequantize.rs:509` include;
  - leave one doc-cite or one baseline entry → the no-cite check prints it.

**Blocked by** Nothing.
**Hosts** x86 CI. The NEON-Q4K-006 probe itself runs later, with NEON-Q4K-000.
**Out of scope** Any other unreferenced file. This ticket deletes only the two that R3 §13 verified; S5 covers the rest and the guard.

---

## S5 — every src file must compile: widen the #3809 dark-file guard to the whole tree
**Refs** #3999 (R5 F-R5-6, and L25 row 5). #3809, the dark-test-file guard this widens. #4502: its 7981643bfa (2026-10-01) deleted 3 parity files that never compiled. S3 deletes two more files of this class; S5 is the guard and the rest.

**Defect [V]** 138 of the 8,264 `.rs` files in the workspace's packages (53,211 lines, in 12 packages) are reached by no `mod`, `include!` or `#[path]` from any package root, so nothing compiles them (`orphan_census.py`; the list is `orphan-census-316dee2cd4.tsv`). All 138 sit under a package's `src/`. aprender-serve holds 111 of them (36,286 lines). 27 of the 138 (15,241 lines) are copies: a compiled file holds their code, with the same text in 24 cases and the same text up to `//` comments and whitespace in 3. The TSV's twin column names each compiled copy.
- **The #3809 guard sees 42.** `scripts/check_src_test_files_wired.sh` looks only at files that hold a test attribute. It counts a file as declared when any file of its crate names it, without asking whether that file compiles. Its baseline, `scripts/src_test_files_unwired_baseline.txt`, has 42 entries, all of them orphans, and `scripts/check_baseline_ratchets.sh:150` keeps that set shrink-only. The other 96 orphans are 27 test files whose only declarers are dark themselves, and 69 files with no test attribute.
- **They read as live code** (Rule 8). Two R5 cells cited uncompiled copies (F-R5-6). The v0.69.3 merge-back #4338 converted the argmax module compiles in the uncompiled copy, reduces.rs, and the compiled copy kept the raw form.
- **Live files point into them** [V at 316dee2cd4]:
  - `contracts/realizar/binding.yaml:250-252` marks `paged_attention_into` `implemented`, but its only body is in the uncompiled `cuda/executor/incremental_attention.rs`. `contracts/binding-allowlist.json:339-340` already lists the symbol as a ghost.
  - `contracts/apr-model-capability-v1.yaml:81`, and its copy `crates/apr-cli/contracts/apr-model-capability-v1.yaml:81`, cite the `gelu_host` in the uncompiled `cuda/executor/kernel.rs`. The compiled `gelu_host` and `fused_swiglu_host` are at `crates/aprender-serve/src/cuda/executor/rope_indirect.rs:112` and `:138`.
  - The doc comment at `crates/aprender-serve/tests/falsify_swiglu_cpu_cuda_005.rs:112` names the uncompiled `kernel.rs` copy of `fused_swiglu_host`.
  - `falsify_007_no_catch_all_in_dispatch_sites` (`crates/aprender-serve/src/quantize/contract_tests.rs:611`) reads 8 dispatch files as text and panics when one is missing. One of them (`:621`) is the uncompiled `layers/transformer_layer_indexed.rs`, whose 3 fns are all defined in the compiled `indexed_transformer.rs`, which the test also reads. So the commit that deletes the file must also drop it from that list.

**Prior art** Reuse one of these; do not write a third resolver.
- The #3809 guard and its baseline, above.
- `module_of()` at `scripts/check_tree_reader_tests.sh:112`, a shell resolver for one file's module path.
- The syn walk in `crates/aprender-contracts/src/ontology/extract/code.rs`: `crate_roots` (`:458`) finds a crate's roots, `splice` (`:498`) splices `include!` files in, and `child_file` (`:958`) resolves a `mod` and follows `#[path]` (`:964`).
- `orphan_census.py` on this branch, as the oracle to diff the guard against. It has 33 self-test cases and 18 mutants, and it lists the sites it cannot resolve: one at 316dee2cd4, the `#[path = "."]` in `crates/aprender-core/src/nn/quantization_tests.rs`, checked by hand.
- `fn_census.py` on this branch, which reuses `orphan_census.py`'s resolver and sorts the dark files by their units (below). It has 24 self-test cases and 24 mutants. At aca6f2d7f6 (2026-09-27) it finds 144 dark files, and the 138 of them still dark at 316dee2cd4 get the same row at both pins.

**Scope**
- Widen the guard from files with a test attribute to every `.rs` under `crates/*/src`. A file counts as declared only when a chain of `mod`, `include!` or `#[path]` reaches it from a package root: `src/lib.rs`, `src/main.rs`, `build.rs`, `src/bin/`, the manifest's `path =` entries, and the test, bench and example targets.
- Keep the files that are still dark in a new shrink-only baseline (for example `scripts/src_files_unreachable_baseline.txt`), registered in `check_baseline_ratchets.sh` beside the #3809 one, which it can replace.
- Delete in batches. The first batch is the 27 copies: deleting a `=` copy loses nothing and deleting a `~` copy loses only its `//` comments, so they need no decision. For the other 111, `fn_census.py` compares each unit of a dark file with the units of the compiled files, by its text from the item keyword on, without `pub`, attributes, comments or whitespace. A unit is a fn or test; a struct, enum, union, type, const, static or `macro_rules` item; an `impl` or `trait` header; or the code outside all of them. Each file lands in one of three groups (the list is `fn-census-316dee2cd4.tsv`), and all 27 copies land in the first, which checks the tool against the twin column:
  - **covered**, 20 files (7,334 lines): every unit has a compiled copy with the same text. They need no decision and are the second batch.
  - **stale**, 31 files (10,229 lines): every unit has a compiled unit of the same kind and name under the same `impl` or `trait` header, but 73 of their 347 units differ from it in text. `unit_history.py` follows each such unit back, on main and, before its crate's import, in the source repo, and finds which copy moved after the two last had the same text (the list is `unit-history-316dee2cd4.tsv`; it needs `--source crates/aprender-serve=<realizar clone> --source crates/aprender-orchestrate=<batuta clone>`). It keys a unit by header, kind and name, so the two `adaptive_attention_with_cache` fns in `gguf/inference/cache.rs` count once, and 72 units remain. In 35 of them (`live`) only the compiled copy moved, so the dark copy is just old, and the 15 files that hold only such units need no decision. The other 37, in 16 files, moved only in the dark copy (`dark`, 18), in both copies (`both`, 17), or never had the same text as the compiled copy (`never`, 2). For each commit that moved a `dark` or `both` unit in the dark copy, `dark_fix_lines.py` lists the code lines it added that the dark file still holds: 10 of the 28 (file, commit) pairs are carried in full by a compiled holder of the unit, and the other 18 hold 79 lines that no compiled holder has. Read at 316dee2cd4, they fall as below. The first five bullets, 9 pairs, are code that has never compiled, and each needs a decision before its file goes. The rest need no decision: the deleting commit names where each fix went, and the last bullet adds one doc fix.
    - **Graph capture** (PMAT-374, realizar de21747e): the dark reduces.rs captures in `ThreadLocal` mode. Its compiled namesake `try_graph_capture` (`crates/aprender-serve/src/cuda/executor/layers/par-062.rs:6`) still captures in `Global` mode (:17), but nothing calls it: the decode graph, on by default from compute capability 8.9 (`crates/aprender-serve/src/cuda/executor/layers/graphed_capture.rs:393-394`), is built by hand without stream capture (`crates/aprender-serve/src/cuda/executor/layers/manual_graph.rs:75`). The Global captures that do run are both opt-in: `crates/aprender-serve/src/cuda/executor/layers/par-121.rs:37` under `BATCHED_GRAPH=1` (`crates/aprender-serve/src/gguf/cuda/generate_batched_streaming.rs:461`), and `crates/aprender-serve/src/cuda/executor/layers/prefill.rs:1106` under `PREFILL_GRAPH=1` (:27). GPU-ORD-4 (`crates/aprender-gpu/src/driver/cuda_tests/cuda_graph_tests.rs:40`) is why that matters: Global mode makes the legacy-stream sync calls illegal in the whole process. Decide whether the two opt-in sites move to ThreadLocal, a GPU change with its own ticket, and drop the uncalled `try_graph_capture` with reduces.rs.
    - **Decode event** (PMAT-283, realizar 921eece0): `init_decode_event` (`crates/aprender-serve/src/cuda/executor/workspace.rs:240`) is called, and its event recorded, only in the dark reduces.rs, and its reader `decode_event_complete` (`crates/aprender-serve/src/cuda/executor/workspace.rs:252`) has no caller. Decide whether the compiled decode path records the event, or the two fns go.
    - **Fused QKV** (GH-288, realizar 8e2f6900): the only dispatch of `fused_qkv_hw_dp4a_q4k_gemv_into` (`crates/aprender-serve/src/cuda/executor/q4k_mwv_gemv.rs:527`) is in the dark `cuda/executor/layers/rope.rs`, so that kernel has never run in this repo. It is a speed change, not a fix: port the dispatch with a GPU A/B, or drop it on purpose.
    - **Contract checks** (realizar 2ee35521, dafc2fa5 and bb8461e3; 5 pairs): five macros of `crates/aprender-serve/src/generated_contracts.rs` (:11810, :20371, :20393, :20495 and :24280, in the order below) are called only in dark files, so their checks have never compiled: `contract_pre_cuda_graph_guard!` in reduces.rs, `contract_pre_attention_sublayer!` and `contract_post_attention_sublayer!` in `forward_block_traced` (`gpu/scheduler/forward_from_model.rs`), `contract_pre_hybrid_block!` in `is_hybrid_attention` (`safetensors/safetensors_config.rs`) and `contract_pre_tensor_count!` in `tensor_count` (`safetensors/tensor.rs`). Call each from the compiled namesake (in `gpu/scheduler/gpu_forward_pass.rs`, `safetensors/safetensors_parser.rs` and `safetensors/shard.rs`, and for the graph guard at the capture sites above), or say in its contract that nothing calls it.
    - **Argmax** (#4338, 5fa13fe8c6): only the dark reduces.rs has this change to `gpu_argmax` and `batched_gpu_argmax`. It is F-R5-6, and the reduces.rs bullet below applies.
    - **Ported in other words**, 5 pairs. Gemma2's post-norms in `gguf/inference/forward/debug.rs` came in 1a1bf6fe65; the compiled code applies them at `crates/aprender-serve/src/gguf/inference/forward/ffn_block.rs:564` and `crates/aprender-serve/src/gguf/inference/forward/forward_cached.rs:82`. GH-306's fused gate_up weight for Phi-3.5 (realizar 0702e7a5, also in debug.rs) is at `crates/aprender-serve/src/gguf/inference/forward/ffn_block.rs:33`, `crates/aprender-serve/src/gguf/inference/forward/adaptive_ffn.rs:51/506` and `crates/aprender-serve/src/gguf/config.rs:658`. PMAT-409's `FORCE_FP16_CACHE=1` override (realizar c38a15e2, in rope.rs and `cuda/executor/layers/workspace_attention.rs`) is at `crates/aprender-serve/src/gguf/cuda/mod.rs:360-363`; the dark lines it lacks are debug prints. The `unwrap()` removal of e75ec33ade reached reduces.rs and par-062.rs alike, and par-062.rs has no `unwrap()` left.
    - **Hygiene**, 4 pairs: PMAT-364's debug conditions in rope.rs (realizar 84a3b4b2); `.unwrap()` to `.expect()` in the tests of the dark `convert/writing.rs` and `generate/make.rs` (realizar 2b3935fd); and `write_bench_json` in the dark `cli/benchmark.rs` (realizar c6ccf1a8), which returns a JSON error as `RealizarError::IoError` where the compiled copy (`crates/aprender-serve/src/cli/display_utils.rs:142`) calls `.expect` on it (:175).
    - **Never had the same text**, 2 units, read by hand. The dark `scatter_single_kv_to_batched` in `cuda/executor/cache_from_cache_from_kv_cache.rs` is the single contiguous copy that PMAT-044 (realizar 784a12d9) replaced with one copy per head, because the KV cache layout is `[num_kv_heads, max_len, head_dim]`; the compiled copy (`crates/aprender-serve/src/cuda/executor/kv_cache_gpu_init.rs:578`) has the fix. The dark `analyze` in `crates/aprender-cbtop/src/latency_distribution/analysis.rs` is the compiled one (`crates/aprender-cbtop/src/latency_distribution/mod.rs:64`) with its min, max, jitter and moments moved into helpers.
    - **ALiBi slope doc**: since its compiled namesakes existed, one commit changed the dark `layers/a_li_bi.rs` without them, ce39f30899 (PMAT-858), which corrected the slope formula in a doc comment. The compiled `crates/aprender-serve/src/layers/alibi.rs:27` has the right `2^(-8(h+1)/n)`, as its code does (:53), but `crates/aprender-serve/src/layers/scaled_rope.rs:327` still says `2^(-8h/n)`: fix that line in the commit that deletes a_li_bi.rs.
  - **unique**, 60 files (20,407 lines): 826 units have no compiled copy: 636 tests, 169 fns and 21 other units. For 786 of them (all of the tests and other units, and 129 of the fns) no compiled unit of that kind has the name at all; the other 40 fns have a namesake under another header. Each one needs a decision: port it, or drop it on purpose.
- The dark files this ticket names, at 316dee2cd4:
  - `gguf/inference/forward/debug.rs` (629 lines) and `gguf/inference/forward/cache.rs` (385) are stale. Each fn name in them is also defined in a compiled file, but 4 of the 9 fns in debug.rs and 3 of the 4 in cache.rs differ in text from their namesakes. One of the 4 is `forward_single_with_cache`, whose live copy is at `crates/aprender-serve/src/gguf/inference/forward/ffn_block.rs:479`.
  - `cuda/executor/incremental_attention.rs` (208) and `cuda/executor/layers/reduces.rs` (550), from F-R5-6. incremental_attention.rs is unique: `paged_attention_into` has no compiled twin, so port it or drop its binding first. Either way, the allowlist's ghost entry goes in the same commit. reduces.rs is stale: 4 of its 5 fns differ from their namesakes in `cuda/executor/layers/par-062.rs`, `gpu_argmax` and `batched_gpu_argmax` among them.
  - `cuda/executor/kernel.rs` (393) is stale: 4 of its 9 fns differ from their namesakes in `rope_indirect.rs` (`rope_neox_indirect_into`, `add_residual_gpu`, `q4k_gemv_gpu` and `tensor_core_q4k_gemm`). `gelu_host` and `fused_swiglu_host`, which the cites above name, have the same text in both files. Move the two contract cites to `rope_indirect.rs` first.
  - `cuda/executor/layers/transformer_layer_indexed.rs`, `apr/mapped_apr_model.rs` and `apr/loading.rs` are stale in one fn each: `transformer_layer_indexed` differs from its namesake in `cuda/executor/layers/indexed_transformer.rs`, `dtype_to_qtype` from the one in `apr/special_tokens.rs`, and `encode_text` from the one in `apr/tokenizer_loading.rs`.
- Fix every reference in the commit that deletes its file. Re-derive the list at the PR head with `git grep -n -F '<path below src/>'`. Known at 316dee2cd4, besides the four bullets above:
  - `contracts/apr-merge-runnable-v1.yaml:66` names the uncompiled `apr/mapped_apr_model.rs` (260 lines);
  - `contracts/decode-hot-path-first-tokens-diagnostic-v1.yaml:196-197` records reduces.rs as dead;
  - three doc comments: `crates/aprender-serve/src/gguf/inference/forward/forward_qwen3_moe.rs:289` takes a line of the uncompiled debug.rs as its reference; `crates/aprender-serve/src/convert/q4k_converter_helpers.rs:117` names the uncompiled `apr/loading.rs` (319 lines), while the live `load_embedded_bpe_tokenizer` is in `apr/tokenizer_loading.rs`; `crates/aprender-core/src/text/bpe/tests_encode_decode.rs:19` names the uncompiled `models/qwen2/tests.rs` (389 lines, in the #3809 baseline);
  - the #3809 baseline, and the root and aprender-serve `.pmat-baseline.json`.
- Leave the roadmap notes that name deleted files (`docs/roadmaps/entries/PMAT-3759.yaml:22`, `docs/roadmaps/roadmap.yaml:21589`). They are history.

**Acceptance** The guard stays; the rest are one-shot checks recorded in the PR body.
- **Compiled code is unchanged.** For each crate a batch touches, `cargo test -p <crate> --lib -- --list` prints the same list before and after, compared by sha256.
- **The guard turns RED in its new scope (Rule 4).** Each of these makes it fail: a new non-test file that nothing names; a test file whose only parent is dark; a dark chain of two files; a commented-out `mod` line; an `include!` written in a dark file. It stays GREEN on a `#[path]` inside an inline module, an `include!` of a sibling file, and an `r#` module name.
- **Rule 7.** Those cases ship as a table, run by a `--self-test` mode that CI calls.
- **The baseline only shrinks.** A new entry fails `check_baseline_ratchets.sh`.
- **No cite is left.** For each deleted file, `git grep -n -F '<path below src/>'` prints only the roadmap lines.
- **The census agrees.** At the PR head, `orphan_census.py` lists exactly the files left in the new baseline.

**Blocked by** Nothing. S3 can land first or fold into the first batch.
**Hosts** x86 CI. The guard is a script and needs no GPU; no cargo on lambda.
**Out of scope** `crates/aprender-present` and `crates/aprender-test`. Their manifests declare a workspace and no package, and the root `Cargo.toml` excludes both, so no package root can reach the 2 files under them (32,270 lines). Whether to keep them is a separate call. The aprender-test file is a `~` copy of `crates/aprender-test-lib/src/generated_contracts.rs` (the census twin line).

---

## Status (2026-10-04)
- None of S1..S3 or S5 is a 0.73 gate. `falsifier-landing-map.md` lists AQ-005..007 separately from the 42 0.73 falsifiers.
- Filing: S1..S3 went out as PROPOSE-TICKET lines on 2026-10-04 at 03:32Z, after LIVE 0.70.1. S5 went out at 06:12Z, after its commit. The cop mints.
