---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
reviewed: 2026-09-28T00:55:45Z
depth: standard
files_reviewed: 80
files_reviewed_list:
  - .github/workflows/ci.yml
  - .gitignore
  - CLAUDE.md
  - Cargo.toml
  - Makefile
  - README.md
  - contracts/aprender/binding.yaml
  - contracts/decide-apr-v1.yaml
  - contracts/decide-tool-boundary-v1.yaml
  - contracts/laya-finetune-gate-v1.yaml
  - contracts/laya-parity-v1.yaml
  - crates/apr-format/src/v2/mod.rs
  - crates/apr-format/src/v2/reader_impl.rs
  - crates/apr-format/src/v2/tests.rs
  - crates/aprender-core/Cargo.toml
  - crates/aprender-core/src/models/mod.rs
  - crates/aprender-core/src/models/modernbert/config.rs
  - crates/aprender-core/src/models/modernbert/embeddings.rs
  - crates/aprender-core/src/models/modernbert/encoder.rs
  - crates/aprender-core/src/models/modernbert/gemm.rs
  - crates/aprender-core/src/models/modernbert/layer.rs
  - crates/aprender-core/src/models/modernbert/load.rs
  - crates/aprender-core/src/models/modernbert/mod.rs
  - crates/aprender-core/tests/monorepo_invariants.rs
  - crates/aprender-decide/Cargo.toml
  - crates/aprender-decide/examples/pack_laya.rs
  - crates/aprender-decide/src/artifact.rs
  - crates/aprender-decide/src/artifact/determinism.rs
  - crates/aprender-decide/src/artifact/ladder.rs
  - crates/aprender-decide/src/artifact/tests.rs
  - crates/aprender-decide/src/laya/builder.rs
  - crates/aprender-decide/src/laya/head.rs
  - crates/aprender-decide/src/laya/mod.rs
  - crates/aprender-decide/src/laya/scorer.rs
  - crates/aprender-decide/src/laya/temperature.rs
  - crates/aprender-decide/src/laya/tests.rs
  - crates/aprender-decide/src/lib.rs
  - crates/aprender-decide/src/pack.rs
  - crates/aprender-decide/src/task.rs
  - crates/aprender-decide/src/test_support.rs
  - crates/aprender-decide/src/verify.rs
  - crates/aprender-decide/src/verify/tests.rs
  - crates/aprender-decide/tests/common/mod.rs
  - crates/aprender-decide/tests/demo_run.rs
  - crates/aprender-decide/tests/fail_closed_vectors.rs
  - crates/aprender-decide/tests/laya_parity.rs
  - crates/aprender-decide/tests/python_records.rs
  - crates/aprender-decide/tests/ui.rs
  - crates/aprender-decide/tests/ui/decider_no_constructor.rs
  - crates/aprender-decide/tests/ui/decider_no_constructor.stderr
  - crates/aprender-decide/tests/ui/decider_struct_literal.rs
  - crates/aprender-decide/tests/ui/decider_struct_literal.stderr
  - crates/aprender-mcp-decide-lambda/.pmcp/.gitignore
  - crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml.template
  - crates/aprender-mcp-decide-lambda/Cargo.toml
  - crates/aprender-mcp-decide-lambda/examples/probe.rs
  - crates/aprender-mcp-decide-lambda/src/lib.rs
  - crates/aprender-mcp-decide-lambda/src/main.rs
  - crates/aprender-mcp-decide-lambda/src/probe.rs
  - crates/aprender-mcp-decide-lambda/src/s3.rs
  - crates/aprender-mcp-decide-lambda/src/tests.rs
  - crates/aprender-mcp-decide/Cargo.toml
  - crates/aprender-mcp-decide/src/lib.rs
  - crates/aprender-mcp-decide/src/main.rs
  - crates/aprender-mcp-decide/src/tests.rs
  - crates/aprender-mcp-decide/tests/e2e_stdio.rs
  - justfile
  - scripts/laya_deploy/cargo_pmcp_resolver_proof.rs
  - scripts/laya_train/.gitignore
  - scripts/laya_train/.python-version
  - scripts/laya_train/common.py
  - scripts/laya_train/contract.py
  - scripts/laya_train/data.py
  - scripts/laya_train/fixtures.py
  - scripts/laya_train/gate.py
  - scripts/laya_train/lifecycle.py
  - scripts/laya_train/metrics.py
  - scripts/laya_train/prepare_stance.py
  - scripts/laya_train/pyproject.toml
  - scripts/laya_train/train.py
findings:
  critical: 1
  warning: 9
  info: 8
  total: 18
status: issues_found
---

# Phase 8: Code Review Report

**Reviewed:** 2026-09-28T00:55:45Z
**Depth:** standard
**Files Reviewed:** 80
**Status:** issues_found

## Summary

I reviewed the 80 files at standard depth. Coverage:

- the ModernBERT encoder in aprender-core, checked against HF ModernBERT semantics: embeddings, pre-norm layers, fused Wqkv, rotate-half RoPE, the inclusive local window, and GeGLU;
- Laya's builder, head, scorer and temperature;
- the decide-apr-v1 artifact writer and load ladder;
- the packer and the Rust gate verifier;
- the thin MCP server and its Lambda shim, including the S3 loader and the probe;
- the Python back office;
- the justfile and Makefile additions.

The numerics are careful. Every refusal path I traced returns a typed error rather than panicking, with one exception (WR-04). Production code contains no `unwrap`.

The defects are at the trust boundaries.

- **Eligibility gap (the blocker).** The deploy-eligibility check (`verify_path`) binds the artifact's `recipe_id` and a few manifest *fields* to the run dir. It never binds those fields to the blobs the artifact actually embeds. It also never checks the `base` identity that every classify response serves.
- **Ladder gap.** The load ladder's "every entry" structural rungs can be sidestepped with duplicate index names.
- **Misleading tool description.** The served tool description tells LLM clients that long texts are truncated. At the deployed tier they are refused.

Items already recorded as deferred were not re-reported: the verify.rs split (D-ITEM-08-12-A), sha2 asm (08-17-C), the cold-start margin (08-17-E), auth-off (08-18-A), x86_64 parity (08-13-A), and CI wiring (08-12-B/C).

## Critical Issues

### CR-01: deploy eligibility does not bind the manifest's served identity, calibration or gate fields to the artifact's own blobs

**File:** `crates/aprender-decide/src/verify.rs:2395-2424`, `crates/aprender-decide/src/artifact.rs:1312-1401` (rung 4), `crates/aprender-decide/src/artifact.rs:1466-1472` (mint)

**Issue:** `verify_path` is documented as decide-apr-v1 `deploy_eligibility`, "the ONLY eligibility check". It claims to bind the manifest to the run and data dirs. In practice it compares only these manifest *fields*: `recipe_id`, `gate.report_sha256`, `inputs_sha256` and `labels`.

Rung 4 of the ladder ties the embedded blobs only to `manifest.blobs[*].sha256` and ties `recipe_id` to the recipe blob. Nothing, in either the ladder or `verify_path`, checks any of the following:

- `manifest.gate.report_sha256` against the embedded `decide.gate_report_json` blob hash;
- `manifest.inputs_sha256.task_json` / `.tokenizer_json` against the embedded task and tokenizer blob hashes;
- `manifest.base` (and `variant`) against the `base` / `variant` parsed from the embedded, sha-bound recipe blob;
- `manifest.calibration.*` (t_fitted / t_applied / clamp_hit) against the report or the embedded agent config's `temperature_by_options[bucket]`. The verifier never checks the report's calibration block either.

As a result, a crafted `.apr` whose weights re-score the eval rows correctly can pass `pack_laya verify` and get `deploy_eligible: true` while it does any of these:

- embeds a gate report other than the one verified;
- declares a different `base`;
- carries a fabricated calibration record.

`manifest.base.display()` is copied verbatim into `ModelIdentity.base`, which every `classify` response returns as `model.base` (D-11). The served base identity is therefore never verified. The deploy pins the file's sha256, so whatever the manifest claims is exactly what ships.

**Fix:** Add cross-checks in rung 4 so every consumer of a `Decider` gets them. `verify_path` should also compare the embedded blobs, not just the manifest fields.

```rust
// rung4_structure, after the blob hashes are verified:
let recipe: crate::pack::Recipe = serde_json::from_slice(recipe)
    .map_err(|e| ArtifactError::ConfigBlob { blob: RECIPE_BLOB, reason: e.to_string() })?;
if manifest.base != recipe.base || manifest.variant != recipe.variant {
    return Err(ArtifactError::ManifestDisagreesWithBlob { field: "base/variant" });
}
let blob_sha = |name| manifest.blobs.iter().find(|b| b.name == name).map(|b| b.sha256.as_str());
if Some(manifest.gate.report_sha256.as_str()) != blob_sha(GATE_REPORT_BLOB)
    || Some(manifest.inputs_sha256.task_json.as_str()) != blob_sha(TASK_BLOB)
    || Some(manifest.inputs_sha256.tokenizer_json.as_str()) != blob_sha(TOKENIZER_BLOB)
{
    return Err(ArtifactError::ManifestDisagreesWithBlob { field: "blob hashes" });
}
// and parse the gate-report blob: gate.{pass,margin,ece_post} and calibration.* must equal it,
// and calibration.t_applied must equal clamp(agent.temperature_by_options[bucket]).
```

In `verify_path`, also require `calibration.t_applied` in the report to equal the temperature the loaded Decider applies (`Laya::temperature()`).

## Warnings

### WR-01: duplicate tensor names slip past the "every entry" rungs 4 and 5

**File:** `crates/apr-format/src/v2/reader_impl.rs:131`, `crates/aprender-decide/src/artifact.rs:764-782`, `1369-1384`, `1407-1425`

**Issue:** The index sort check uses a strict `<`, so two entries with the SAME name are accepted, and `AprV2Writer::add_tensor` happily writes them. In the ladder:

- `check_name_set` sees a duplicate as a member of the expected set, so it is neither missing nor unexpected.
- Rung 4(c) loops over names and calls `reader.get_tensor(name)`, which returns the FIRST match. The second entry's dtype, `size == product(shape) x width` and data bounds are therefore never checked.
- Rung 5 iterates the entries but reads each one's data through `get_tensor_data(&entry.name)`, again the first match. The second entry's bytes are never scanned for NaN or Inf.

A verified artifact can therefore carry an arbitrary unvalidated payload, including an out-of-file range. That contradicts the contract's "tensor set is a bijection" and "every entry" claims.

**Fix:** Refuse duplicates. Either change the reader to `<=` (a strictly increasing index), or add a check in rung 4:

```rust
if index.windows(2).any(|w| w[0] == w[1]) {
    return Err(ArtifactError::UnexpectedTensor { name: /* the repeated name */ });
}
```

Also iterate `reader.tensor_index()` entries directly in rungs 4(c) and 5, not by name.

### WR-02: `verify_run` is public and mints `deploy_eligible: true` without binding `packed` to `inputs`

**File:** `crates/aprender-decide/src/verify.rs:2326-2336`

**Issue:** `verify_run(inputs, packed, ...)` returns a `VerifyReport { deploy_eligible: true }`. It never checks that `packed`'s manifest (`recipe_id`, report hash, input hashes, labels) describes `inputs`, which is the binding `verify_path` performs. The doc says "bytes packed from inputs", but nothing enforces it. Its only callers are tests; `pack_for_serving` inlines its own sequence. It remains a second public eligibility door with weaker guarantees than the one the contract names.

**Fix:** Make it `pub(crate)` (or `#[cfg(test)]`). Alternatively, factor `verify_path`'s manifest binding (lines 2404-2421) into a helper and call it from `verify_run` too.

### WR-03: the served tool description promises truncation that is unreachable at the deployed tier

**File:** `crates/aprender-mcp-decide/src/lib.rs:499-505`

**Issue:** The description every MCP client's LLM reads says: "Long texts are truncated by the model itself to its window, and each such result reports `truncated: true`." At the contracted 3 008 MB tier, however, `max_total_tokens` is 120, and a truncated row is always `max_len` = 512 tokens. `decide-tool-boundary-v1.yaml:79-80` says so itself: "a text long enough to be truncated ... is always refused by the budget, so `truncated: true` is unreachable here".

Clients are told to send whole long documents. They then get `classify_max_total_tokens` validation errors instead of a truncated decision.

**Fix:** Derive the sentence from the limits and the model's `max_len`. For example, when `max_total_tokens < manifest().agent.max_len`, say "a text whose built row exceeds {max_total_tokens} tokens (about N words of state) is refused; send a shorter excerpt", and emit the truncation sentence only when a full row fits the budget.

### WR-04: `ModernBertLayer::forward` panics on an empty row (the public API promises typed errors)

**File:** `crates/aprender-core/src/models/modernbert/layer.rs:290-350`

**Issue:** With `l == 0` and an empty `x`, every `check_len` passes (0 == 0 x d). `Linear::forward` returns `Vec::new()` for `m == 0`, so `qkv` is empty. `&qkv[d..]` at line 343 then panics with "range start index d out of range". The module docs state that each primitive "returns a typed ModernBertError instead of panicking". `ModernBertEncoder::forward` guards against empty input, but `ModernBertLayer::forward` is `pub` and has no such guard.

**Fix:**

```rust
if l == 0 {
    return Err(ModernBertError::EmptyInput);
}
```

Put this at the top of `forward_with_rope`, and add a unit test for the case.

### WR-05: the Python back office splits JSONL on Unicode line boundaries, so its own prepared data can be refused and Rust diverges

**File:** `scripts/laya_train/data.py:147`, `scripts/laya_train/prepare_stance.py:60`, `102-103`

**Issue:** `prepare_stance.jsonl_bytes` writes with `json.dumps(..., ensure_ascii=False)`, which emits U+0085, U+2028 and U+2029 raw inside strings. That is valid JSON. `data.load_rows` and `prepare_stance.read_jsonl`, however, split with `str.splitlines()`, which breaks lines at exactly those code points. So any tweet containing NEL (common in mis-decoded cp1252 text, and explicitly anticipated by `_WHITE_SPACE` in data.py:45-47) or U+2028 splits one row into two invalid-JSON lines, and the trainer refuses its own data. Rust's `parse_rows` uses `str::lines()` (`\n` only), so the two sides disagree about what a row is.

**Fix:** Split on `"\n"` only, in both files:

```python
text = path.read_bytes().decode("utf-8")
lines = text.split("\n")
if lines and lines[-1] == "":
    lines.pop()
for n, line in enumerate(lines, 1):
    line = line.removesuffix("\r")
    ...
```

Alternatively, write with `ensure_ascii=True`.

### WR-06: a dead loopback server leaves the Lambda container permanently failing

**File:** `crates/aprender-mcp-decide-lambda/src/main.rs:78-97`, `162-211`

**Issue:** After a successful load, `LOADED` holds `base_url` forever. If the spawned loopback task ever ends (its `JoinHandle` completes), the watcher only logs `loopback server task ended`. Every later request then fails `req.send()` with a connect error and returns `Err` (a 500 from the runtime). No re-load is attempted, and the warm container keeps being routed traffic until Lambda recycles it.

**Fix:** When the loopback task ends, make the process exit (e.g. `std::process::exit(1)` in the watcher after logging) so Lambda replaces the environment. Alternatively, store an `AtomicBool` "alive" flag in `Loaded` and return 503 with `kind: "loopback_down"`, so the failure is at least visible and fast.

### WR-07: `_laya-grant-check` claims "nothing broader" but inspects only inline policies and a fixed list of wildcards; `laya-grant`'s header is stale

**File:** `justfile:2341-2419`

**Issue:**

- **Scope of the check.** `laya-grant` lists only the role's INLINE policies. Attached managed policies (`list-attached-role-policies`), permission boundaries, `NotAction` and `NotResource` are never examined.
- **Wildcard detection.** Only four exact strings count as "broad": `*`, `arn:aws:s3:::<bucket>/*`, `arn:aws:s3:::<bucket>`, `arn:aws:s3:::*`. So `arn:aws:s3:::aprender-*/*`, `arn:aws:s3:::<bucket>/decide/*`, or `s3:Get*` on another bucket all pass.
- **The success message.** "grant ok ... allows s3:GetObject on <prefix> and nothing broader" is not established by the check.
- **Stale header.** The recipe header (lines 2341-2344) still says it creates an inline policy `aprender-decide-weights-<env>` with `s3:ListBucket` and must be RE-RUN after a redeploy. The body is read-only and *refuses* that legacy policy.

**Fix:**

- Add `aws iam list-attached-role-policies`, and fail when any attached policy grants `s3:*`-class actions.
- Treat any `*` / `?` wildcard in an S3 action or resource other than the exact `want` ARN as broad.
- Handle `NotAction` / `NotResource` as broad.
- Rewrite the header to describe the read-only check.

### WR-08: the Rust verifier does not enforce the calibration-slice fraction the contract requires

**File:** `crates/aprender-decide/src/verify.rs:1072-1090`, `1101-1138`

**Issue:** laya-finetune-gate-v1 `calibration_fit_bounded` (lines 460-467) requires the slice to be "at least calibration_slice_fraction of the shots and at least calibration_slice_min_per_class rows per class". `split_disjointness` says "the Rust verifier re-derives all of this". `check_slice_classes` checks only `min_per_class` and "at least one fit row". `VerifyPolicy` has no fraction field at all. A report whose `slice_ids` hold just `min_per_class` rows per class (for example 2 of 64) therefore verifies, even though the Python trainer (`data.calibration_split`, which uses `max(min_per_class, ceil(fraction x n_class))`) could never have produced it.

**Fix:** Add `calibration_slice_fraction: f64` to `VerifyPolicy`, read it from the contract in `pack_laya.rs` and `tests/common/mod.rs`, and require:

```rust
let need = (min_per_class as usize).max((fraction * in_class as f64).ceil() as usize);
if in_slice < need || in_slice >= in_class { return Err(slice_invalid(...)); }
```

### WR-09: the Python trainer truncates `pack_rescore_noise_k` with `int()`

**File:** `scripts/laya_train/contract.py:53-61`

**Issue:** `noise_policy()` returns `int(p["constants"]["pack_rescore_noise_k"])`, while Rust reads the same key as `f64` (`pack_laya.rs:109`) and requires `rec.k.to_bits() == policy.rescore_noise_k.to_bits()` (`verify.rs:1393`).

The contract value is 4 today, so nothing breaks yet. Any non-integer amendment would change that:

- Python would silently record, and multiply by, a truncated k (for example 2.5 becomes 2).
- The records would then fail every Rust verify with `RescoreNoiseInvalid k`.
- Worse, the Python-side bound written into the record would be tighter than the contract's.

The same class of defect exists in `seed_selection_decl()`, which applies `int(rank_scale)`.

**Fix:** Read both as `float(...)`, and refuse (raise) when a value is not what the contract declares, rather than coercing it.

## Info

### IN-01: `argmax` says "NaN never wins", but a NaN at index 0 does

**File:** `crates/aprender-decide/src/laya/mod.rs:476-479`, `crates/aprender-decide/src/verify.rs:1252-1255`, `1350-1353`

**Issue:** The fold starts at `m = 0`, and `p[i] > NaN` is always false, so `argmax([NaN, 0.9])` returns 0. The callers validate finiteness first, so the case is unreachable today. The doc comment is still wrong.

**Fix:** Either correct the comment, or start from the first finite index.

### IN-02: `pack_laya inspect` reads the file unbounded before the rung-1 cap

**File:** `crates/aprender-decide/examples/pack_laya.rs:263`

**Issue:** `std::fs::read(&apr)` loads the whole file. Only then does `inspect_manifest` apply the 1.25 GiB cap. Every other door uses `read_decide_apr_bytes_bounded`.

**Fix:** Open the file and call `artifact::read_decide_apr_bytes_bounded(file, Some(metadata.len()))`.

### IN-03: `labels_in_order` is a substring search, so a label can match inside unrelated words

**File:** `crates/aprender-mcp-decide-lambda/src/probe.rs:301-310`

**Issue:** `description[from..].find(label)` matches `none` inside "nonetheless", and it matches labels inside the question text that precedes the list. The check can therefore pass for a description whose `[a, b, c]` list is in the wrong order. It is deploy evidence (`description_has_labels_in_order`).

**Fix:** Parse the `The labels, in this order: [...]` segment and compare it as a list.

### IN-04: stale comments contradict current behaviour

**File:** `crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml.template:15-17`, `23`; `crates/aprender-decide/tests/ui.rs:8-9`

**Issue:**

- The template still says the tell that the right binary shipped is the GET health body naming the package, but D-ITEM-08-17-A established that GET never reaches the bootstrap on pmcp.run. It also still says the deploy root is still to be settled by plan 08-10.
- `tests/ui.rs` says the target is dark in CI, but ci.yml now runs it.

**Fix:** Update both comments to match current behaviour.

### IN-05: a crate-wide `#![allow(clippy::disallowed_methods)]` switches off the `unwrap` ban for whole crates

**File:** `crates/aprender-mcp-decide/src/lib.rs:42`, `crates/aprender-mcp-decide-lambda/src/lib.rs:22`, `crates/aprender-mcp-decide-lambda/src/main.rs:18`

**Issue:** These allows are needed only for the `schemars` derive and `json!` expansions. As written, any future `.unwrap()` in these files passes clippy, which weakens the `.clippy.toml` gate that CLAUDE.md relies on.

**Fix:** Move `ClassifyArgs` and the `json!` call sites into a small module that carries the allow, or use item-level allows where the expansion permits it.

### IN-06: the Lambda response can carry duplicate CORS headers

**File:** `crates/aprender-mcp-decide-lambda/src/main.rs:216-229`

**Issue:** `access-control-allow-origin: *` is set first, and then every upstream header is appended, so a pmcp server that sets its own CORS header produces two values.

**Fix:** Skip upstream `access-control-*` headers, or use `headers_mut().insert` so one value overrides the other.

### IN-07: an aprender-core dev-dependency is not workspace-pinned

**File:** `crates/aprender-core/Cargo.toml:226`

**Issue:** `safetensors = "0.4"` is a literal version, while `aprender-decide` uses `safetensors = { workspace = true }`. The two can drift.

**Fix:** Use `{ workspace = true }`.

### IN-08: the trainer crashes with a traceback when early stopping never saw a finite monitor

**File:** `scripts/laya_train/train.py:365-367`, `scripts/laya_train/gate.py:135-137`

**Issue:** `EarlyStopper.record()` raises a bare `ValueError` when every epoch's calibration NLL is NaN. `train_seed` does not catch it, so the run dies with a traceback and exit 1. It does not produce the documented `REFUSED ...` line with exit 2.

**Fix:** Catch it and call `refuse("REFUSED early-stopping: ...")`.

---

_Reviewed: 2026-09-28T00:55:45Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
