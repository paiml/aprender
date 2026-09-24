# Quorum brief — PR #4311 (B1 batch: serve/perf/runtime), head e1993673d
Scope: #4311 folds 9 PRs (#4046, #4224, #4229, #4170, #4203, #4297, #4304, #4213, #4147), each already reviewed on its own PR.
The full diff vs main is 15 MB, so THIS round judges only the FOLD DELTA: the commits the batch author wrote to make the fold build and pass CI.
Stated intent of each commit:
- 892f62b9c: regenerate roadmap aggregate (fold duplicated keys) + 2 missing fragments.
- d5c5b1d35: clear six fold defects CI named (fmt/clippy/contract/book/ratchet renumber/test-identity).
- c938fd7cc: guard-tree: a guard run via ci_run_explicit_test_commands.sh --run <dir> fragments counts as wired; clippy-census message truncation + self-test fix.
- 566ac9ee5: regenerate census count/README (generated; shown for count only).
- 68c842319: renumber qwen35-batched-prefill falsification IDs to FALSIFY-QBP-001..005 (sequential-ID integrity rule).
- e1993673d: re-extract contracts/contracts.nt (generated; excluded).
Judge: does each commit do what it says, without weakening a gate, deleting a test's teeth, or changing runtime semantics beyond the fix? Default FAIL on a weakened gate or an unbacked claim.
Return JSON: {"verdict":"PASS|FAIL|do-not-implement-as-written","summary":"...","findings":[{"file":"","line":0,"claim":"","grounding":"cited|measured|asserted","fix":""}]}

=== commit 892f62b9c

=== commit 892f62b9c
892f62b9c6fdedbcf2de1f600332f2e6b7545cb7 fix(roadmap): regenerate the aggregate from fragments — the fold duplicated keys

diff --git a/docs/roadmaps/entries/PMAT-4250.yaml b/docs/roadmaps/entries/PMAT-4250.yaml
new file mode 100644
index 000000000..15e851181
--- /dev/null
+++ b/docs/roadmaps/entries/PMAT-4250.yaml
@@ -0,0 +1,17 @@
+- id: PMAT-4250
+  github_issue: 4250
+  item_type: task
+  title: '0.69.4 G0: serve-level test fails unless apr serve prefills Qwen3.5 batched (#4250)'
+  status: planned
+  priority: medium
+  assigned_to: aprender-ac
+  created: 2026-09-24T10:04:37Z
+  updated: 2026-09-24T10:04:37Z
+  spec: null
+  acceptance_criteria: []
+  phases: []
+  subtasks: []
+  estimated_effort: null
+  labels:
+  - kind:code
+  notes: null
diff --git a/docs/roadmaps/entries/PMAT-4263.yaml b/docs/roadmaps/entries/PMAT-4263.yaml
new file mode 100644
index 000000000..c51e042c3
--- /dev/null
+++ b/docs/roadmaps/entries/PMAT-4263.yaml
@@ -0,0 +1,9 @@
+- id: PMAT-4263
+  github_issue: 4263
+  item_type: task
+  title: "one engine (0.69.3): every Qwen3.5 verb through realizar::session::Session"
+  status: inprogress
+  priority: critical
+  assigned_to: aprender-ac
+  labels:
+  - kind:code

=== commit d5c5b1d35
d5c5b1d35af2341bf7d4751578f7124298b39185 fix(b1): clear the six fold defects #4311's CI named

diff --git a/book/src/SUMMARY.md b/book/src/SUMMARY.md
index 3886c7641..539233f76 100644
--- a/book/src/SUMMARY.md
+++ b/book/src/SUMMARY.md
@@ -449,2 +449,3 @@
 - [aprender::regularization](./lib/regularization.md)
+- [aprender::release_section](./lib/release_section.md)
 - [aprender::scoring](./lib/scoring.md)
diff --git a/book/src/lib/release_section.md b/book/src/lib/release_section.md
new file mode 100644
index 000000000..4efb1a0c7
--- /dev/null
+++ b/book/src/lib/release_section.md
@@ -0,0 +1,27 @@
+<!-- PCU: lib-release_section | contract: contracts/apr-page-lib-release_section-v1.yaml -->
+
+# Module: `aprender::release_section`
+
+Public module of the `aprender-core` crate.
+
+The README's release-verification matrix, rendered from the model-ladder receipts
+(`evidence/dogfood/models/<version>/<host>.json`) rather than typed by hand (#3769).
+`render` turns a set of `Receipt`s (each carrying its `Rung`s) into the Markdown section;
+the committed README section is gated to equal that render.
+
+## Source
+
+[`crates/aprender-core/src/release_section.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/release_section.rs) or directory.
+
+## Example
+
+<!-- example-cost: trivial -->
+```rust
+use aprender::release_section::{render, Receipt, Rung};
+// See `cargo doc -p aprender-core --open` for full API reference.
+```
+
+## Full API
+
+Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
+[docs.rs/aprender](https://docs.rs/aprender) for the published version.
diff --git a/ci/explicit-test-commands.d/460-clippy-member-ratchet-self-test.cmd b/ci/explicit-test-commands.d/480-clippy-member-ratchet-self-test.cmd
similarity index 100%
rename from ci/explicit-test-commands.d/460-clippy-member-ratchet-self-test.cmd
rename to ci/explicit-test-commands.d/480-clippy-member-ratchet-self-test.cmd
diff --git a/ci/explicit-test-commands.d/470-clippy-member-ratchet.cmd b/ci/explicit-test-commands.d/490-clippy-member-ratchet.cmd
similarity index 100%
rename from ci/explicit-test-commands.d/470-clippy-member-ratchet.cmd
rename to ci/explicit-test-commands.d/490-clippy-member-ratchet.cmd
diff --git a/contracts/apr-page-lib-release_section-v1.yaml b/contracts/apr-page-lib-release_section-v1.yaml
new file mode 100644
index 000000000..d1bebab3d
--- /dev/null
+++ b/contracts/apr-page-lib-release_section-v1.yaml
@@ -0,0 +1,38 @@
+metadata:
+  version: 1.0.0
+  kind: schema
+  description: |
+    PCU contract for book/src/lib/release_section.md (aprender::release_section module reference).
+    Enforces: page exists, references the module, has a runnable example.
+  references:
+    - "docs/specifications/book-completeness-spec.md"
+
+equations:
+  pcu_header_present:
+    formula: "file_contains(book/src/lib/release_section.md, 'PCU: lib-release_section')"
+  module_mentioned:
+    formula: "file_contains(book/src/lib/release_section.md, 'aprender::release_section')"
+  example_block_present:
+    formula: "file_contains_fenced(book/src/lib/release_section.md, language='rust')"
+
+falsification_tests:
+  - id: FALSIFY-PAGE-LIB-RELEASE_SECTION-001
+    name: stub_exists
+    prediction: "book/src/lib/release_section.md exists on disk"
+    test_harness: "test -f book/src/lib/release_section.md"
+    expected_output: "exit 0"
+    if_fails: "Run scripts/gen-lib-chapter-stubs.sh to regenerate."
+
+  - id: FALSIFY-PAGE-LIB-RELEASE_SECTION-002
+    name: module_mentioned
+    prediction: "page references the module at least once"
+    test_harness: "grep -q 'aprender::release_section' book/src/lib/release_section.md"
+    expected_output: "exit 0"
+    if_fails: "Add a mention of aprender::release_section to the chapter."
+
+  - id: FALSIFY-PAGE-LIB-RELEASE_SECTION-003
+    name: runnable_example
+    prediction: "page has at least one rust code block"
+    test_harness: "grep -c '^```rust' book/src/lib/release_section.md"
+    expected_output: "stdout >= 1"
+    if_fails: "Add a rust code block with a runnable example."
diff --git a/contracts/kernel-fusion-v1.yaml b/contracts/kernel-fusion-v1.yaml
index a9e87d6f2..903e9b7fb 100644
--- a/contracts/kernel-fusion-v1.yaml
+++ b/contracts/kernel-fusion-v1.yaml
@@ -132,3 +132,3 @@ fusion_decisions:
       - V projection GEMV
-    call_site: crates/aprender-serve/src/cuda/kernels_generate_gemm_cuda.rs:371
+    call_site: crates/aprender-serve/src/cuda/kernels_generate_gemm_cuda.rs:381
     saves_per_layer: 2
@@ -145,3 +145,3 @@ fusion_decisions:
       - SwiGLU activation
-    call_site: crates/aprender-serve/src/cuda/kernels_generate_gemm_cuda.rs:374
+    call_site: crates/aprender-serve/src/cuda/kernels_generate_gemm_cuda.rs:384
     saves_per_layer: 2
@@ -185,3 +185,3 @@ fusion_decisions:
       - Up Q4K GEMV (separate input load)
-    call_site: crates/aprender-serve/src/cuda/kernels_generate_gemm_cuda.rs:377
+    call_site: crates/aprender-serve/src/cuda/kernels_generate_gemm_cuda.rs:387
     saves_per_layer: 1
diff --git a/crates/aprender-gpu/src/kernels/gdn/prefill_flash_attention.rs b/crates/aprender-gpu/src/kernels/gdn/prefill_flash_attention.rs
index 383597095..ef473b1ea 100644
--- a/crates/aprender-gpu/src/kernels/gdn/prefill_flash_attention.rs
+++ b/crates/aprender-gpu/src/kernels/gdn/prefill_flash_attention.rs
@@ -161,2 +161,87 @@ fn quad_reduce(
 
+/// The registers the Q-staging prologue reads.
+struct QStage {
+    q_ptr: VirtualReg,
+    rows: VirtualReg,
+    q0: VirtualReg,
+    lane: VirtualReg,
+    lane16: VirtualReg,
+    lane_hi: VirtualReg,
+    q_warp_base: VirtualReg,
+    head_col: VirtualReg,
+    q_stride: u32,
+}
+
+/// Stage each 128-wide half of the warp's Q rows as f16 in shared memory and
+/// lift it into the 16 A-fragments (8 k-steps per half) the QK^T MMAs consume.
+fn stage_q_fragments(
+    ctx: &mut crate::ptx::builder::KernelBuilder<'_>,
+    q: &QStage,
+) -> Vec<[VirtualReg; 4]> {
+    let QStage {
+        q_ptr,
+        rows,
+        q0,
+        lane,
+        lane16,
+        lane_hi,
+        q_warp_base,
+        head_col,
+        q_stride,
+    } = *q;
+    let mut qfrag: Vec<[VirtualReg; 4]> = Vec::with_capacity(16);
+    for half in 0..2u32 {
+        ctx.bar_sync(0);
+        let it = ctx.mov_u32_imm(0);
+        let lbl = format!("fa_qstage_{half}");
+        let end = format!("fa_qstage_end_{half}");
+        ctx.label(&lbl);
+        let n = ctx.mov_u32_imm(BR * 128 / 32);
+        let more = ctx.setp_lt_u32(it, n);
+        ctx.branch_if_not(more, &end);
+        let it32 = ctx.mul_u32(it, 32);
+        let idx = ctx.add_u32_reg(it32, lane);
+        let seven = ctx.mov_u32_imm(7);
+        let r = ctx.shr_u32(idx, seven); // / 128
+        let m127 = ctx.mov_u32_imm(127);
+        let c = ctx.and_u32(idx, m127);
+        let qrow = ctx.add_u32_reg(q0, r);
+        let val = ctx.mov_f32_imm(0.0);
+        let in_rows = ctx.setp_lt_u32(qrow, rows);
+        let skip = format!("fa_qload_skip_{half}");
+        ctx.branch_if_not(in_rows, &skip);
+        let rowel = ctx.mul_u32(qrow, q_stride);
+        let col = ctx.add_u32(c, half * 128);
+        let colh = ctx.add_u32_reg(head_col, col);
+        let el = ctx.add_u32_reg(rowel, colh);
+        let off = ctx.mul_wide_u32(el, 4);
+        let addr = ctx.add_u64(q_ptr, off);
+        let x = ctx.ld_global_f32(addr);
+        ctx.mov_f32_reg(val, x);
+        ctx.label(&skip);
+        let h = ctx.cvt_f16_f32(val);
+        let srow = ctx.mul_u32(r, Q_ROW_H);
+        let sel = ctx.add_u32_reg(srow, c);
+        let sb = ctx.mul_u32(sel, 2);
+        let sa = ctx.add_u32_reg(q_warp_base, sb);
+        let sa64 = ctx.cvt_u64_u32(sa);
+        ctx.st_shared_f16(sa64, h);
+        ctx.add_u32_inplace(it, 1);
+        ctx.branch(&lbl);
+        ctx.label(&end);
+        ctx.bar_sync(0);
+        // A-fragment of k-step kk: lane reads row lane%16, cols kk*16 + (lane/16)*8.
+        let lrow = ctx.mul_u32(lane16, Q_ROW_H);
+        let lcol = ctx.mul_u32(lane_hi, 8);
+        let lel = ctx.add_u32_reg(lrow, lcol);
+        let lb = ctx.mul_u32(lel, 2);
+        let lbase = ctx.add_u32_reg(q_warp_base, lb);
+        for kk in 0..8u32 {
+            let a = ctx.add_u32(lbase, kk * 16 * 2);
+            qfrag.push(ctx.ldmatrix_x4(a));
+        }
+    }
+    qfrag
+}
+
 impl Kernel for PrefillFlashAttention256Kernel {
@@ -218,54 +303,16 @@ impl Kernel for PrefillFlashAttention256Kernel {
                 let head_col = ctx.mul_u32(head, FLASH_HEAD_DIM);
-                let mut qfrag: Vec<[VirtualReg; 4]> = Vec::with_capacity(16);
-                for half in 0..2u32 {
-                    ctx.bar_sync(0);
-                    let it = ctx.mov_u32_imm(0);
-                    let lbl = format!("fa_qstage_{half}");
-                    let end = format!("fa_qstage_end_{half}");
-                    ctx.label(&lbl);
-                    let n = ctx.mov_u32_imm(BR * 128 / 32);
-                    let more = ctx.setp_lt_u32(it, n);
-                    ctx.branch_if_not(more, &end);
-                    let it32 = ctx.mul_u32(it, 32);
-                    let idx = ctx.add_u32_reg(it32, lane);
-                    let seven = ctx.mov_u32_imm(7);
-                    let r = ctx.shr_u32(idx, seven); // / 128
-                    let m127 = ctx.mov_u32_imm(127);
-                    let c = ctx.and_u32(idx, m127);
-                    let qrow = ctx.add_u32_reg(q0, r);
-                    let val = ctx.mov_f32_imm(0.0);
-                    let in_rows = ctx.setp_lt_u32(qrow, rows);
-                    let skip = format!("fa_qload_skip_{half}");
-                    ctx.branch_if_not(in_rows, &skip);
-                    let rowel = ctx.mul_u32(qrow, q_stride);
-                    let col = ctx.add_u32(c, half * 128);
-                    let colh = ctx.add_u32_reg(head_col, col);
-                    let el = ctx.add_u32_reg(rowel, colh);
-                    let off = ctx.mul_wide_u32(el, 4);
-                    let addr = ctx.add_u64(q_ptr, off);
-                    let x = ctx.ld_global_f32(addr);
-                    ctx.mov_f32_reg(val, x);
-                    ctx.label(&skip);
-                    let h = ctx.cvt_f16_f32(val);
-                    let srow = ctx.mul_u32(r, Q_ROW_H);
-                    let sel = ctx.add_u32_reg(srow, c);
-                    let sb = ctx.mul_u32(sel, 2);
-                    let sa = ctx.add_u32_reg(q_warp_base, sb);
-                    let sa64 = ctx.cvt_u64_u32(sa);
-                    ctx.st_shared_f16(sa64, h);
-                    ctx.add_u32_inplace(it, 1);
-                    ctx.branch(&lbl);
-                    ctx.label(&end);
-                    ctx.bar_sync(0);
-                    // A-fragment of k-step kk: lane reads row lane%16, cols kk*16 + (lane/16)*8.
-                    let lrow = ctx.mul_u32(lane16, Q_ROW_H);
-                    let lcol = ctx.mul_u32(lane_hi, 8);
-                    let lel = ctx.add_u32_reg(lrow, lcol);
-                    let lb = ctx.mul_u32(lel, 2);
-                    let lbase = ctx.add_u32_reg(q_warp_base, lb);
-                    for kk in 0..8u32 {
-                        let a = ctx.add_u32(lbase, kk * 16 * 2);
-                        qfrag.push(ctx.ldmatrix_x4(a));
-                    }
-                }
+                let qfrag = stage_q_fragments(
+                    ctx,
+                    &QStage {
+                        q_ptr,
+                        rows,
+                        q0,
+                        lane,
+                        lane16,
+                        lane_hi,
+                        q_warp_base,
+                        head_col,
+                        q_stride,
+                    },
+                );
 
diff --git a/crates/aprender-serve/src/api/cuda_batch_scheduler.rs b/crates/aprender-serve/src/api/cuda_batch_scheduler.rs
index 1b36580fd..0bb3c890f 100644
--- a/crates/aprender-serve/src/api/cuda_batch_scheduler.rs
+++ b/crates/aprender-serve/src/api/cuda_batch_scheduler.rs
@@ -142,14 +142,3 @@ fn generate_single_request_inner(cuda_model: &mut OwnedQuantizedModelCuda, req:
         });
-        match result {
-            Ok(_) => {
-                for t in tokens {
-                    if req.token_tx.try_send(Ok(t)).is_err() {
-                        break;
-                    }
-                }
-            },
-            Err(e) => {
-                let _ = req.token_tx.try_send(Err(e.to_string()));
-            },
-        }
+        send_buffered_turn(result.map(|_| tokens), &req.token_tx);
     } else {
@@ -164,2 +153,23 @@ fn generate_single_request_inner(cuda_model: &mut OwnedQuantizedModelCuda, req:
 
+/// The non-streaming turn's delivery: its tokens in order once the turn is
+/// done (stopping at the first closed-channel send), or its error.
+#[cfg(feature = "cuda")]
+fn send_buffered_turn<E: std::fmt::Display>(
+    result: std::result::Result<Vec<u32>, E>,
+    token_tx: &tokio::sync::mpsc::Sender<Result<u32, String>>,
+) {
+    match result {
+        Ok(tokens) => {
+            for t in tokens {
+                if token_tx.try_send(Ok(t)).is_err() {
+                    break;
+                }
+            }
+        },
+        Err(e) => {
+            let _ = token_tx.try_send(Err(e.to_string()));
+        },
+    }
+}
+
 /// Spawn the batch scheduler background task.
diff --git a/crates/aprender-serve/src/api/tests_engine_identity.rs b/crates/aprender-serve/src/api/tests_engine_identity.rs
index 2a25d871f..acf68d8e1 100644
--- a/crates/aprender-serve/src/api/tests_engine_identity.rs
+++ b/crates/aprender-serve/src/api/tests_engine_identity.rs
@@ -31,2 +31,11 @@ const ARCHES: &[(&str, Arch)] = &[
 
+/// `ArchForward` impls that are not an architecture of their own but another
+/// row's forward over a borrowed model: (impl type, the `Session` row it runs).
+/// The scan counts each as its row, so an adapter must still be declared here.
+const ADAPTERS: &[(&str, &str)] = &[
+    // #4280: the CUDA batch scheduler's single-request path, the dense CUDA
+    // forward borrowing the scheduler's model for one turn.
+    ("BorrowedCudaForward", "DenseForward"),
+];
+
 #[derive(Debug, Clone, Copy, PartialEq, Eq)]
@@ -264,2 +273,16 @@ fn apr_run_on_a_dense_gguf_enters_the_one_engine() {
 
+/// The implementing type's name when `line` opens an `impl ArchForward for X`.
+fn arch_forward_impl_type(line: &str) -> Option<String> {
+    let line = line.trim_start();
+    let rest = line
+        .strip_prefix("impl crate::session::ArchForward for ")
+        .or_else(|| line.strip_prefix("impl ArchForward for "))?;
+    let ty: String = rest
+        .chars()
+        .take_while(|c| c.is_alphanumeric() || *c == '_')
+        .collect();
+    let row = ADAPTERS.iter().find(|(adapter, _)| *adapter == ty);
+    Some(row.map_or(ty, |(_, row)| (*row).to_string()))
+}
+
 /// Every `impl ArchForward for X` in production source is exactly the set of
@@ -280,15 +303,3 @@ fn every_arch_forward_is_a_session_row() {
                 let text = std::fs::read_to_string(&path).expect("read source");
-                for line in text.lines() {
-                    let line = line.trim_start();
-                    let rest = line
-                        .strip_prefix("impl crate::session::ArchForward for ")
-                        .or_else(|| line.strip_prefix("impl ArchForward for "));
-                    if let Some(rest) = rest {
-                        let ty: String = rest
-                            .chars()
-                            .take_while(|c| c.is_alphanumeric() || *c == '_')
-                            .collect();
-                        found.insert(ty);
-                    }
-                }
+                found.extend(text.lines().filter_map(arch_forward_impl_type));
             }
diff --git a/crates/aprender-serve/src/gguf/inference/forward/dense_session.rs b/crates/aprender-serve/src/gguf/inference/forward/dense_session.rs
index c42303ffe..b915a8bc5 100644
--- a/crates/aprender-serve/src/gguf/inference/forward/dense_session.rs
+++ b/crates/aprender-serve/src/gguf/inference/forward/dense_session.rs
@@ -145,3 +145,3 @@ impl DenseForward {
         let context_length = self.context_length;
-        match &mut self.backend {
+        let (capacity, fresh) = match &mut self.backend {
             Backend::Cpu { model, cache } => {
@@ -154,10 +154,4 @@ impl DenseForward {
                     .max(positions);
-                if let Some(cache) = cache {
-                    // Grown in place: what it holds stays held.
-                    cache.grow_to(capacity);
-                    self.capacity = capacity;
-                    return Ok(());
-                }
-                *cache = Some(OwnedQuantizedKVCache::from_config(&model.config, capacity));
-                self.capacity = capacity;
+                // Grown in place: what it holds stays held.
+                (capacity, size_cache(cache, &model.config, capacity))
             },
@@ -174,14 +168,8 @@ impl DenseForward {
                 }
-                if let Some(cache) = cache {
-                    // The device cache is sized once, at build; only the host
-                    // side grows, in place.
-                    cache.grow_to(positions);
-                    self.capacity = positions;
-                    return Ok(());
-                }
-                *cache = Some(OwnedQuantizedKVCache::from_config(
-                    &model.model().config,
+                // The device cache is sized once, at build; only the host
+                // side grows, in place.
+                (
                     positions,
-                ));
-                self.capacity = positions;
+                    size_cache(cache, &model.model().config, positions),
+                )
             },
@@ -190,4 +178,7 @@ impl DenseForward {
             },
+        };
+        self.capacity = capacity;
+        if fresh {
+            self.held = 0;
         }
-        self.held = 0;
         Ok(())
@@ -518,2 +509,16 @@ pub fn dense_turn(
 
+/// Grow `cache` in place to `capacity`, or build it at `capacity` when there is
+/// none. Returns `true` when it was built fresh (so it holds nothing yet).
+fn size_cache(
+    cache: &mut Option<OwnedQuantizedKVCache>,
+    config: &crate::gguf::GGUFConfig,
+    capacity: usize,
+) -> bool {
+    if let Some(cache) = cache {
+        cache.grow_to(capacity);
+        return false;
+    }
+    *cache = Some(OwnedQuantizedKVCache::from_config(config, capacity));
+    true
+}
 #[cfg(test)]
diff --git a/scripts/check_baseline_ratchets.sh b/scripts/check_baseline_ratchets.sh
index 5c77ec092..95d46601a 100644
--- a/scripts/check_baseline_ratchets.sh
+++ b/scripts/check_baseline_ratchets.sh
@@ -83,2 +83,3 @@ classify() { # classify <basename> -> "<kind>[<TAB>reason]", rc 1 if unclassifie
         complexity_baseline.txt)                 printf 'keyed2\n' ;;
+        clippy_member_baseline.txt)              printf 'keyed\n' ;;   # #4152: <crate>|<kind>|<lint><TAB><count>, shrink-only (scripts/check_clippy_member_ratchet.sh)
         fabricated_baseline_rust_sites.txt)      printf 'set\n' ;;

=== commit c938fd7cc
c938fd7cc12782c8aa3089f7c99868cba4ea8571 fix(b1): guard-tree — a guard run by an explicit-test-command fragment is wired; the clippy census says what it cut

diff --git a/scripts/check_clippy_member_ratchet.sh b/scripts/check_clippy_member_ratchet.sh
index de49667af..945df78e4 100755
--- a/scripts/check_clippy_member_ratchet.sh
+++ b/scripts/check_clippy_member_ratchet.sh
@@ -140,3 +140,4 @@ for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
     where = f'{span.get("file_name", "?")}:{span.get("line_start", "?")}'
-    print(f"      {code} at {where}: {m.get('message', '')[:160]}")
+    text = m.get("message", "")
+    print(f"      {code} at {where}: " + (text if len(text) <= 160 else f"{text[:160]} ... and {len(text) - 160} more chars"))
     n += 1
@@ -227,3 +228,3 @@ self_test() {
     printf '{"reason":"compiler-message","package_id":"path+file:///x/crates/aprender-core#0.69.0","target":{"kind":["lib"]},"message":{"level":"warning","code":{"code":"clippy::x"}}}\n{"reason":"build-finished","success":true}\n' > "$t/verid.json"
-    if census "$t/verid.json" | grep -q '^aprender-core|lib|clippy::x'; then printf '  ok    a version-only package id names the crate\n'
+    if grep -q '^aprender-core|lib|clippy::x' <<< "$(census "$t/verid.json")"; then printf '  ok    a version-only package id names the crate\n'
     else printf '  BROKE a version-only package id was not named by its directory\n'; fails=$((fails + 1)); fi
diff --git a/scripts/check_guards_are_wired.sh b/scripts/check_guards_are_wired.sh
index 65b62af35..f676ff7f7 100755
--- a/scripts/check_guards_are_wired.sh
+++ b/scripts/check_guards_are_wired.sh
@@ -137,5 +137,29 @@ dogfood_declared_missing() {
 # Guards named by no workflow, one per line, sorted.
+# explicit_cmd_wired ROOT -- guards run by an EXPLICIT TEST COMMAND fragment.
+#
+# #4152: a cargo-using guard can be wired as ci/explicit-test-commands.d/NNN-*.cmd,
+# which ci.yml's workspace-test shards execute via
+#     bash scripts/ci_run_explicit_test_commands.sh --run ci/explicit-test-commands.d
+# The workflow never names the guard, so the per-guard scan below reports it dark
+# although CI runs it on every PR. Same rule as everywhere here -- EXECUTION, not
+# mention -- applied twice: a workflow must INVOKE the runner with --run on the
+# directory, and the fragment's line must INVOKE the guard (a `#` comment in a .cmd
+# wires nothing).
+explicit_cmd_wired() {
+    local root="$1" lines dir f
+    lines=$(grep -rh --include='*.yml' --include='*.yaml' -- 'ci_run_explicit_test_commands.sh' \
+                "$root"/.github/workflows/ 2>/dev/null | sed 's/#.*$//' | _not_a_name_line) || lines=''
+    for dir in $(sed -nE 's/.*ci_run_explicit_test_commands\.sh[[:space:]]+--run[[:space:]]+([^[:space:]"'"'"']+).*/\1/p' <<< "$lines" | LC_ALL=C sort -u); do
+        for f in "$root/$dir"/*.cmd; do
+            [ -f "$f" ] || continue
+            sed 's/#.*$//' "$f" \
+                | grep -oE "(^|[[:space:];&|(])((ba)?sh[[:space:]]+|\\./)?[^[:space:]]*scripts/[A-Za-z0-9_.-]+\\.sh([[:space:]]|$)" \
+                | grep -oE '[A-Za-z0-9_.-]+\.sh' || true
+        done
+    done | LC_ALL=C sort -u
+}
+
 unwired_in() {
     local root="$1" g base seen="" dispatched declared
-    dispatched=" $(dispatcher_wired "$root" | tr '\n' ' ') "
+    dispatched=" $(dispatcher_wired "$root" | tr '\n' ' ') $(explicit_cmd_wired "$root" | tr '\n' ' ') "
     declared=" $(dogfood_declared "$root" | sed 's|.*/||' | tr '\n' ' ') "
@@ -377,4 +401,30 @@ if [ "${1:-}" = "--self-test" ]; then
 
+    # ── Rows 14-16: WIRED BY AN EXPLICIT TEST COMMAND FRAGMENT (#4152) ──────
+    # Row 14: a .cmd fragment invokes the guard and a workflow runs the fragment
+    #         directory with --run -> wired.
+    # Row 15: the control -- drop the workflow's --run line and it is dark again.
+    # Row 16: a fragment that only NAMES the guard in a comment wires nothing.
+    mkdir -p "$TD3/ci/cmds"
+    printf '# the fragment\nbash scripts/check_dark.sh\n' > "$TD3/ci/cmds/010-dark.cmd"
+    printf '# bash scripts/check_release_gate.sh\n' > "$TD3/ci/cmds/020-comment.cmd"
+    printf 'jobs:\n  t:\n    steps:\n      - run: |\n          bash scripts/ci_run_explicit_test_commands.sh --run ci/cmds --shard "1/3"\n' \
+        > "$TD3/.github/workflows/tests.yml"
+    got14=$(unwired_in "$TD3" | tr '\n' ' ')
+    if [ "$got14" = "check_release_gate.sh " ]; then
+        printf 'ok    row 14 a guard run by an explicit-test-command fragment counts as wired\n'
+        printf 'ok    row 16 a fragment that names a guard only in a comment wires nothing\n'
+    else
+        printf 'FAIL  rows 14/16 got [%s], expected [check_release_gate.sh ]\n' "$got14"; fails=1
+    fi
+    printf 'jobs:\n  t:\n    steps:\n      - run: bash scripts/ci_run_explicit_test_commands.sh --list ci/cmds\n' \
+        > "$TD3/.github/workflows/tests.yml"
+    got15=$(unwired_in "$TD3" | tr '\n' ' ')
+    if [ "$got15" = "check_dark.sh check_release_gate.sh " ]; then
+        printf 'ok    row 15 a workflow that only --lists the fragments wires nothing (the control for row 14)\n'
+    else
+        printf 'FAIL  row 15 got [%s], expected [check_dark.sh check_release_gate.sh ]\n' "$got15"; fails=1
+    fi
+
     [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
-    printf '\nSELF-TEST PASSED (13/13)\n'
+    printf '\nSELF-TEST PASSED (16/16)\n'
     exit 0

=== commit 566ac9ee5
566ac9ee5e1655cbe5116560decbe1970d242937 fix(b1): regenerate contracts/census.json + README CONTRACT_COUNT (1834 -> 1836) for the contracts the fold added

diff --git a/README.md b/README.md
index ab563020e..a863ee79a 100644
--- a/README.md
+++ b/README.md
@@ -43,3 +43,3 @@ publishing — all backed by YAML provable contracts that fail CI on drift.
 | Workspace crates | **79** workspace crates | `cargo metadata --no-deps` (NOT `ls crates/` — 4 are `exclude`d, 1 has no Cargo.toml) |
-| Provable contracts | **<!-- CONTRACT_COUNT_START -->1834<!-- CONTRACT_COUNT_END -->** provable contracts | `contracts/census.json` `.n_files` — the set `pv lint` walks (`pv census`, ONT-001 ONT-1; regenerated by `make contracts`, written by `make readme-sync`, guarded by `scripts/check_readme_claims.sh`) |
+| Provable contracts | **<!-- CONTRACT_COUNT_START -->1836<!-- CONTRACT_COUNT_END -->** provable contracts | `contracts/census.json` `.n_files` — the set `pv lint` walks (`pv census`, ONT-001 ONT-1; regenerated by `make contracts`, written by `make readme-sync`, guarded by `scripts/check_readme_claims.sh`) |
 | CLI commands | **111** CLI commands | `contracts/apr-cli-commands-v1.yaml` §`commands` (parse the list; `apr --help` prints 112 because it lists `help` itself, and `grep -c '^  - name:'` gives 117 — other same-indent `name:` keys exist in the file) |
@@ -323,3 +323,3 @@ falsification_tests:
 
-The tree carries <!-- CONTRACT_COUNT_START -->1834<!-- CONTRACT_COUNT_END --> contracts across inference, training, quantization, attention, FFN,
+The tree carries <!-- CONTRACT_COUNT_START -->1836<!-- CONTRACT_COUNT_END --> contracts across inference, training, quantization, attention, FFN,
 tokenization, model formats, CLI safety — and this README itself.

=== commit 68c842319
68c842319a347ac1e60524c6e8cd27d6cca72f8c fix(b1): qwen35-batched-prefill falsification IDs are numbered (FALSIFY-QBP-001..005)

diff --git a/contracts/qwen35-batched-prefill-v1.yaml b/contracts/qwen35-batched-prefill-v1.yaml
index 56047dee2..0dca4ce6e 100644
--- a/contracts/qwen35-batched-prefill-v1.yaml
+++ b/contracts/qwen35-batched-prefill-v1.yaml
@@ -51,3 +51,3 @@ proof_obligations:
     statement: "The chunk scan is bitwise-equal to T per-token launches for the 0.8B, 9B and 27B head shapes."
-    discharged_by: FALSIFY-QBP-SCAN-BITWISE
+    discharged_by: FALSIFY-QBP-001
   - id: PO-QBP-002
@@ -55,3 +55,3 @@ proof_obligations:
     statement: "The conv1d/L2/gates/RoPE row kernels are bitwise-equal to their per-token twins."
-    discharged_by: FALSIFY-QBP-ROWS-BITWISE
+    discharged_by: FALSIFY-QBP-002
   - id: PO-QBP-003
@@ -59,3 +59,3 @@ proof_obligations:
     statement: "Batched prefill agrees with the per-token path on logits and every state buffer, in one call and split, and decode continues identically."
-    discharged_by: FALSIFY-QBP-PREFILL-EQ
+    discharged_by: FALSIFY-QBP-003
   - id: PO-QBP-005
@@ -63,3 +63,3 @@ proof_obligations:
     statement: "The flash kernel matches exact attention over f16 inputs for the 0.8B, 9B and 27B head shapes, with a cached prefix, a partial block and dozens of key tiles; the flash prefill matches the per-token path within its budget."
-    discharged_by: FALSIFY-QBP-FLASH
+    discharged_by: FALSIFY-QBP-005
   - id: PO-QBP-004
@@ -67,5 +67,5 @@ proof_obligations:
     statement: "A request that cannot fit is refused with its budget before any upload; a co-tenant-caused refusal is distinguishable from a device limit."
-    discharged_by: FALSIFY-QBP-CAPACITY-TABLE
+    discharged_by: FALSIFY-QBP-004
 falsification_tests:
-  - id: FALSIFY-QBP-SCAN-BITWISE
+  - id: FALSIFY-QBP-001
     rule: "Chunk scan is the per-token recurrence"
@@ -74,3 +74,3 @@ falsification_tests:
     if_fails: "The scan reordered an accumulation, contracted differently, or mis-addressed a token row or head"
-  - id: FALSIFY-QBP-ROWS-BITWISE
+  - id: FALSIFY-QBP-002
     rule: "Row twins are their per-token kernels"
@@ -79,3 +79,3 @@ falsification_tests:
     if_fails: "A row stride, a per-row position or a per-head index is wrong"
-  - id: FALSIFY-QBP-PREFILL-EQ
+  - id: FALSIFY-QBP-003
     rule: "Batched prefill is the per-token path"
@@ -84,3 +84,3 @@ falsification_tests:
     if_fails: "A layer op, a KV row write, the attention base offset or the chunking is wrong"
-  - id: FALSIFY-QBP-CAPACITY-TABLE
+  - id: FALSIFY-QBP-004
     rule: "Capacity refuses before loading, classified"
@@ -89,3 +89,3 @@ falsification_tests:
     if_fails: "The budget arithmetic or the refusal classification is wrong"
-  - id: FALSIFY-QBP-FLASH
+  - id: FALSIFY-QBP-005
     rule: "Flash attention is causal attention"
@@ -105,2 +105,2 @@ qa_gate:
   pass_criteria: "All 5 falsification tests pass on every architecture the path ships on (sm_89, sm_121)"
-  falsification: "Scale the scan's output immediate by 1.0001: FALSIFY-QBP-SCAN-BITWISE must go RED (measured 2026-09-21: RED at flat index 0)"
+  falsification: "Scale the scan's output immediate by 1.0001: FALSIFY-QBP-001 must go RED (measured 2026-09-21: RED at flat index 0)"
