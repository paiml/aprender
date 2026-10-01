---
phase: 04
phase_name: apr-artifact-and-production-parity
reviewed: 2026-08-16T00:00:00Z
review_round: 2
depth: standard
diff_base: 87b780eefabb4dbb6f5e64efe2f5b6698d058e73
files_reviewed: 35
files_reviewed_list:
  - crates/apr-cli/src/commands/builder.rs
  - crates/apr-cli/src/commands/construction.rs
  - crates/apr-cli/src/commands/data_contrastive.rs
  - crates/apr-cli/src/commands/eval/mod.rs
  - crates/apr-cli/src/commands/eval/setfit.rs
  - crates/apr-cli/src/commands/inspect.rs
  - crates/apr-cli/src/commands/inspect_output_json.rs
  - crates/apr-cli/src/commands/inspect_setfit.rs
  - crates/apr-cli/src/commands/mod.rs
  - crates/apr-cli/src/commands/predict.rs
  - crates/apr-cli/src/commands/serve/handlers.rs
  - crates/apr-cli/src/commands/setfit_train.rs
  - crates/apr-cli/src/dispatch_analysis.rs
  - crates/apr-cli/src/extended_commands.rs
  - crates/apr-cli/src/lib.rs
  - crates/apr-cli/src/lib_dispatch_coverage.rs
  - crates/apr-cli/src/lib_parse_rosetta_02.rs
  - crates/apr-cli/src/lib_verbose_inheritance_parse.rs
  - crates/apr-cli/src/setfit_commands.rs
  - crates/apr-cli/src/setfit_io.rs
  - crates/apr-cli/src/setfit_tag.rs
  - crates/aprender-serve/src/api/mod.rs
  - crates/aprender-serve/src/api/mod_app_state_gpu.rs
  - crates/aprender-serve/src/api/mod_app_state_new.rs
  - crates/aprender-serve/src/api/router.rs
  - crates/aprender-serve/src/api/setfit_handlers.rs
  - crates/aprender-serve/src/api/types.rs
  - crates/aprender-train/src/train/setfit/apr_codec.rs
  - crates/aprender-train/src/train/setfit/apr_evaluate.rs
  - crates/aprender-train/src/train/setfit/apr_reload.rs
  - crates/aprender-train/src/train/setfit/credential.rs
  - crates/aprender-train/src/train/setfit/evaluate.rs
  - crates/aprender-train/src/train/setfit/lock.rs
  - crates/aprender-train/src/train/setfit/mod.rs
  - crates/aprender-train/src/train/setfit/verify.rs
findings:
  critical: 0
  warning: 9
  info: 6
  total: 15
closed_since_round_1:
  - CR-01
  - CR-02
  - WR-02
  - IN-03
status: findings
---

# Phase 04: Code Review Report (round 2 — re-review)

**Reviewed:** 2026-08-16
**Depth:** standard (per-file, Rust-specific, cross-file into `aprender-core` / `aprender-serve`)
**Files Reviewed:** 35
**Status:** findings — **0 Critical**, 9 Warning, 6 Info

## Summary

Both round-1 Critical findings are **genuinely closed**, and the fixes are correct rather than
cosmetic — I traced each one to its single call site and to the sibling implementation it was
supposed to mirror. `WR-02` and `IN-03` are closed too. The cleanup pass (`b47acc4fe`) is, on
the six items the brief named, **correct in all six**; the verification for each is recorded
below so the next reader does not have to re-derive it.

The re-review found **four new Warnings**, three of which are consequences of the cleanup and
the CR-02 fix themselves:

* the `AuthGate` that closed CR-02 is layered **outside** `CorsLayer::permissive()`, so with
  `APR_API_KEY` set the classify surface now 401s every browser CORS preflight and every
  unauthenticated health probe (**WR-07**);
* the new `MAX_TAG_METADATA_BYTES` fail-open makes `apr predict` and `apr eval` state, in
  writing, that a genuinely tagged artifact is "not a SetFit classifier" — and the message
  points the operator at `apr inspect`, which will show them the opposite (**WR-08**);
* `apr inspect`'s own metadata read — bounded by this phase, but only by the file's length —
  still allocates up to 4 GiB from an attacker-controlled `u32`, which is the exact hazard the
  new 16 MiB constant was introduced to refuse two files away (**WR-09**);
* `apr eval --lock-out` runs the entire multi-candidate sweep before discovering the lock file
  already exists, inverting the ordering discipline `setfit_train.rs`'s own module header
  states and enforces (**WR-10**).

Five round-1 Warnings and four round-1 Infos are re-confirmed present, unchanged.

**Project-rule compliance:** zero `unwrap()` in every Phase 4 source file (verified by count
across all 13 new/changed non-test modules); zero `unsafe`; zero SATD markers. Note that
`crates/apr-cli/src/lib.rs:9-16` carries a crate-wide
`#![allow(clippy::all, clippy::pedantic, clippy::disallowed_methods, unused_imports, dead_code, …)]`,
so the `unwrap()` ban and the unused-import lint are **not machine-enforced anywhere in
`apr-cli`** — this phase's compliance is by discipline, not by gate. That is the mechanism
behind IN-06.

---

## Verification of the round-1 fixes and the cleanup pass

| Item | Verdict | Evidence |
|---|---|---|
| **CR-01** label-map gate | **CLOSED — correct** | `eval/setfit.rs:435-444` compares `credential.model().ordered_labels()` against `dataset.label_names()` and returns `Err` **before** `evaluate_test` at line 446. It mirrors the library gate byte-for-byte in *semantics*: `apr_evaluate.rs:219-229` compares the same two accessors, and both deliberately read the **rebuilt head**, not the document's copy. Not bypassable: `evaluate_test` has exactly one call site (446) and `run_test` exactly one (238). The gate sits after `mint_test_token`/`grant`, which is fine — no scoring happens in between. |
| **CR-02** AuthGate on `/v1/classify` | **CLOSED for the bypass** | `handlers.rs:1474-1477` wraps the whole `create_router_with_config(...)` router in `super::auth::layer(super::auth::AuthGate::from_env(), …)`. `auth::layer<S>` (`serve/auth.rs:176-184`) applies `from_fn_with_state(Arc<AuthGate>, apply)` to the router, and `apply` (`auth.rs:139-165`) checks every request with no route exemption — so `/v1/classify` is genuinely gated. `AuthGate::from_env()` is now *called* on this path, so the `auth.rs:70-72` "routes are unauthenticated" warning is reachable. **But the layer's position introduces WR-07.** The fifth source-scan needle the round-1 fix suggested (`handlers.rs:1691-1709`) was **not** added, so nothing pins the gate against silent removal. |
| **WR-02** predictable temp file | **CLOSED — correct** | The hand-rolled writer is deleted. `eval/setfit.rs:622` delegates to `crate::commands::setfit_train::atomic_write`, which uses `temp_path` (`setfit_train.rs:132-141`: `.{stem}.tmp.{pid}.{ordinal}` off a shared `AtomicU64`) and `fill_and_sync` (`148-151`: `create_new(true)`). All three sub-issues — concurrent clobber, silent leftover reuse, symlink-following `create(true)` — are gone. |
| **IN-03** third hex encoder | **CLOSED** | `hex_of` deleted; `config_hash_of` now calls `aprender_contrastive_data::hash::hex` (`eval/setfit.rs:566`), the same encoder `setfit_train.rs:52` uses. |
| **write_lock double-refusal / force semantics** | **Correct, no defect** | `force=true` → both checks skip → `rename` replaces. `force=false` + pre-existing → bespoke "a lock is a COMMITMENT" message fires first, before bytes are produced. `force=false` + file appears mid-run → `atomic_write`'s `refuse_existing_output` catches it. No path double-refuses and no path inverts `force`. Temp-file safety strictly improved. |
| **`setfit_tag.rs` bound ordering** | **Correct** | `end > file_len` (line 124) precedes `declared > MAX_TAG_METADATA_BYTES` (line 141). A container declaring a block past its own EOF still errors (`InvalidFormat`); a merely-large in-bounds block returns `Ok(None)`. `saturating_add` (123) means a near-`u64::MAX` offset saturates and trips the EOF check rather than wrapping. Ordering is right. **The fail-open's downstream effect is WR-08.** |
| **`handlers.rs` `.ok().flatten().is_some()`** | **Correct — matches the deleted copy exactly** | I recovered the deleted `setfit_apr_model_type_tag` (removed in `b47acc4fe`): it returned `Option<String>` built entirely from `.ok()?` chains, so *every* failure produced `None` and fell through. `.ok().flatten()` reproduces that disposition precisely. The shared reader is strictly **stronger** than the copy — the copy had only the 16 MiB cap and no `end > file_len` check at all. No failure is swallowed that the previous code surfaced. |
| **`inspect.rs` / `setfit_tag.rs` `.remove()`** | **Correct** | `inspect.rs:571,578`: `meta.custom` is never read after 578 and `MetadataInfo` (66-130) has no `custom` field, so nothing downstream expects either key. `setfit_tag.rs:168`: `meta` is a function-local `let Ok(mut meta) = …` consumed at the `Ok(Some(…))` return. Both moves are safe. |
| **`predict.rs::preview` running counter** | **Correct — boundary unchanged** | The old guard was `out.chars().count() >= MAX`, and `out` already held `"\\n"` as **two** chars for an escaped control character. The new counter adds `2` for `\n`/`\r`/`\t` and `1` otherwise, which is the same quantity. The check remains at loop head, `'…'` is still pushed on break. Truncation is identical for every input including the escaped-control case. |
| **`apr_reload.rs` `Deserialize::deserialize(&…)`** | **Correct — identical error behaviour** | `serde_json` implements `Deserializer<'de>` for `&'de Value` with `Error = serde_json::Error`, the same associated error type `from_value` produces, and neither form carries line/column data. Field handling, `deny_unknown_fields` and message text are unchanged; only the deep clone disappears. |

---

## Warnings

### WR-01 — STILL OPEN: `atomic_write` and `write_lock` clobber a file that appears in the check→rename window, and both doc comments deny it

**File:** `crates/apr-cli/src/commands/setfit_train.rs:176-207`

`refuse_existing_output` (177) then `fill_and_sync` then `fs::rename(&temp, target)` (183).
`fs::rename` on Unix replaces the destination unconditionally, so a file created during the
temp write — seconds, for a ~90 MB artifact plus `fsync` — is destroyed without `--force`.
`refuse_existing_output`'s doc (191-194) still claims "the write-time check is what makes the
guarantee true for a file that appeared while the run was going." It narrows the window; it
does not close it. `write_lock` now inherits the same window via delegation.

**Fix:** take the destination itself with `O_CREAT|O_EXCL` before the temp write and rename
over your own placeholder, so the kernel adjudicates the race:

```rust
if !force {
    fs::OpenOptions::new().write(true).create_new(true).open(target)
        .map_err(|e| if e.kind() == std::io::ErrorKind::AlreadyExists {
            CliError::ValidationFailed(format!(
                "Refusing to replace existing file {} (pass --force to replace it)",
                target.display()))
        } else { CliError::Io(e) })?;
}
```

Or amend both doc comments to say the guarantee is best-effort.

### WR-03 — STILL OPEN: `config_hash_of` silently collapses to `sha256("")` on serialization failure

**File:** `crates/apr-cli/src/commands/eval/setfit.rs:563-567`

```rust
let bytes = serde_json::to_vec(requested).unwrap_or_default();
aprender_contrastive_data::hash::hex(&Sha256::digest(&bytes).into())
```

A failed serialize yields an empty slice, so every affected candidate is labelled with the
constant `e3b0c442…b855` in the committed lock and in `--json`.
`SelectionLock::from_candidates` refuses duplicate `artifact_hash` but **not** duplicate
`config_hash` (`lock.rs:269-280`), so the lock commits and nothing notices. The value is a
label, not a gate, so this is not an authorization defect — but it is a silent fail-open on an
identity value in a module whose thesis is that identity values are never defaulted, and
`to_vec` over a `serde_json::Value` is effectively infallible, so the `unwrap_or_default` buys
nothing.

**Fix:** return `Result<String>` and map the error to
`CliError::Aprender("the artifact's requested_config sub-document did not serialize: {error}")`.

### WR-04 — STILL OPEN: an arity mismatch is reported as `SelectionLabelOutOfRange`

**File:** `crates/aprender-train/src/train/setfit/evaluate.rs:378-386`

```rust
if truth.len() != predicted.len() {
    return Err(SetFitTrainError::SelectionLabelOutOfRange {
        label: predicted.len(), classes: truth.len(),
    });
}
```

That variant's `Display` (`mod.rs:1414-1418`) renders *"selection names class N but the dataset
declares M classes"* and sends the operator to check `--selection` against `--data`. Nothing
about the selection is wrong; the classifier returned a different number of results than there
were rows. The variant is `PartialEq`-matchable, so a caller can match on a diagnosis nothing
produced.

**Fix:** add a variant to the already-`#[non_exhaustive]` enum (`mod.rs:1125`), so this is not
a breaking change:

```rust
PredictionArityMismatch { truth_rows: usize, predicted_rows: usize },
```

### WR-05 — STILL OPEN: `apr inspect` shows no SetFit section for a tagged artifact whose `setfit` document is missing

**Files:** `crates/apr-cli/src/commands/inspect.rs:577-581`, `crates/apr-cli/src/commands/inspect_setfit.rs:42-43`

`inspect.rs` populates `setfit_doc` only when the tag **and** the custom key are both present,
and `build_setfit_inspection` early-returns on `let doc = doc?;`, so the whole APR-05 section
vanishes. `setfit_tag::read_setfit_tag` is deliberately the opposite (`setfit_tag.rs:53-59`):
an artifact tagged `setfit` with no custom key is still routed to the SetFit path. So
`apr predict` (`predict.rs:114-120`) and `apr serve` (`handlers.rs:774-778`) both fail on such
a file while `apr inspect` — CLAUDE.md's mandated first diagnostic — reports a healthy plain
APR. This is exactly the outcome `inspect_setfit.rs:33-36` says it exists to avoid.

**Fix:** carry the tag decision separately from the document and emit the section with `Null`
fields plus a `document_present: false` marker when the document is absent.

### WR-06 — STILL OPEN: `evaluate_test` scores an unknown predicted label as merely "wrong", and its `zip` cannot detect an over-long response

**File:** `crates/apr-cli/src/commands/eval/setfit.rs:513-535`

Two gaps against `apr_evaluate.rs:245-275`:

1. `labels.iter().position(|label| label == result.label())` yields `None` for a label outside
   the artifact's ordered set; `None == Some(row.label)` is `false`, so the row counts as
   **incorrect**. If it happened for every row the command reports `accuracy = 0.000000` as a
   measurement. The sibling refuses this by name with `UnknownPredictedLabel`.
2. `for (row, result) in chunk.iter().zip(response.results())` with `seen += 1` **inside** the
   loop means `seen` can never exceed `rows.len()`, so the guard at 530-535 is structurally
   unable to fire when the response is *longer* than the chunk. The sibling counts results
   independently and then compares totals, catching both directions.

Note this is now the *only* remaining hole in the test path — CR-01's label-map gate closed the
larger one — but an artifact whose head grows a label the loader admits still produces a number
rather than a refusal.

**Fix:** mirror the sibling — `ok_or_else` on the `position`, and accumulate
`returned += response.results().len()` outside the zip, comparing `returned` against
`rows.len()` at the end.

### WR-07 — NEW: the CR-02 auth gate is layered outside `CorsLayer::permissive()`, so every browser preflight and every unauthenticated health probe now gets 401

**Files:** `crates/apr-cli/src/commands/serve/handlers.rs:1474-1477`, `crates/aprender-serve/src/api/router.rs:166-168`

`create_router_with_config` finishes with `router.layer(cors).with_state(state)` (router.rs:167-168).
`handlers.rs:1474` then applies `super::auth::layer(...)` to the returned router. In axum the
**last-added layer is the outermost** (axum 0.7 `docs/middleware.md`, "Ordering": *"First
`layer_three` receives the request … then passes it onto `layer_two`"*). So the request order is
**auth → CORS → handler**, and the auth middleware short-circuits before `CorsLayer` ever runs.

Concrete failure, with `APR_API_KEY` set and `apr serve classifier.apr` running:

1. A browser app at `https://app.example.com` calls
   `fetch(url, {method:'POST', headers:{'Authorization':'Bearer k','Content-Type':'application/json'}})`.
2. Because of the non-simple headers the browser first sends
   `OPTIONS /v1/classify` — and per the Fetch spec a preflight carries **no** `Authorization`
   header.
3. `auth::apply` (`auth.rs:145-165`) sees `header_value == None`, `check_bearer` returns
   `false`, and it returns `401` **constructed inside the auth layer**, so the response never
   passes back through `CorsLayer` and carries no `Access-Control-Allow-Origin`.
4. The browser aborts. GH-671's CORS support is dead for every authenticated classifier
   deployment.

Secondary, same root cause: `/health`, `/health/live`, `/health/ready` and `/metrics` are also
inside the gate. The startup banner at `handlers.rs:1493` advertises
`GET /health/ready - Readiness (reports the artifact hash)` without saying it needs a bearer
token, and `AppState::model_loaded()` was just taught (`mod_app_state_gpu.rs:410-419`) to return
`true` for a classifier specifically so *"a k8s rollout would … admit a classifier deployment"* —
which it now cannot, because the probe gets 401 instead of 200. `build_apr_cpu_router` does not
exhibit the CORS half of this (it mounts no `CorsLayer`), so this is specific to the new path.

**Fix:** put the gate **inside** the CORS layer, or exempt preflight and the liveness surface.
The cheapest correct form is to apply auth to the router *before* CORS wraps it — which requires
a small `router.rs` change — or to short-circuit in `apply`:

```rust
// auth.rs::apply, before the bearer check
if req.method() == axum::http::Method::OPTIONS {
    return next.run(req).await;   // let CorsLayer answer the preflight
}
```

and, if health probes are meant to stay open, add the same early return for
`/health`, `/health/live`, `/health/ready`. Whichever is chosen, the startup banner should say
which routes require the token.

### WR-08 — NEW: `MAX_TAG_METADATA_BYTES`'s fail-open makes `apr predict` and `apr eval` state that a genuinely tagged artifact is not a SetFit classifier

**File:** `crates/apr-cli/src/setfit_tag.rs:133-143`

```rust
if declared > MAX_TAG_METADATA_BYTES {
    return Ok(None);
}
```

The comment justifying `Ok(None)` says *"the caller falls through to the pre-existing APR path,
which has its own and better diagnosis for an unusual container."* That is true for exactly one
of the three consumers.

* `apr serve` (`handlers.rs:774-778`) — falls through to `start_apr_server`. The comment holds.
* `apr predict` (`predict.rs:114-120`) — has **no** plain-APR path. It returns
  `InvalidFormat("{path}: not a SetFit classifier. …")`, and `SUPPORTED_FAMILIES`
  (`predict.rs:44-47`) appends *"run `apr inspect <FILE>` to see what it actually is."*
* `apr eval --task classify` (`dispatch_analysis.rs:759-809`) — falls past the `tagged` branch
  and refuses the flags with *"`--selection` applies only to a setfit-apr-v1 artifact, and X
  **does not carry the SetFit tag**"*, which is a false statement about a file that does.

Concrete failure: an artifact whose `setfit` document exceeds 16 MiB — reachable by a large
`vocab_remap` / HF name map plus six probe embeddings, and trivially constructible by an
attacker who wants a classifier to be misidentified. `apr predict` tells the operator it is not
a classifier and sends them to `apr inspect`, which reads the same block (it has no such cap —
see WR-09), finds `model_type: "setfit"` and renders the **full APR-05 section**. The two tools
contradict each other and the error message routes the operator into the contradiction.

This is precisely the failure mode the module's own header refuses in the adjacent case
(`setfit_tag.rs:56-59`): *"rather than silently falling back to 'plain APR', which would report
a corrupt classifier as a healthy generic model."*

Related, and worth deciding at the same time: the three consumers disagree on error disposition.
`dispatch_analysis.rs:759` and `predict.rs:114` propagate `read_setfit_tag`'s `Err` with `?`;
`handlers.rs:774` swallows it with `.ok()`. Each is individually defensible, but "ONE detection
point" currently means one reader with three dispositions.

**Fix:** make the over-cap case distinguishable from "not tagged". Return the tag decision even
when the document is not read, e.g.

```rust
pub(crate) struct SetFitTag {
    pub(crate) doc: Option<serde_json::Value>,
    /// True when the block was in-bounds for the file but over MAX_TAG_METADATA_BYTES,
    /// so `doc` is absent for a reason that is NOT "this is not a SetFit artifact".
    pub(crate) metadata_over_cap: bool,
}
```

and read `model_type` from a bounded prefix, or — simpler and consistent with the module's own
stated bias — return `Err(CliError::InvalidFormat(..))` naming the cap, letting `serve` keep its
`.ok()` fall-through while `predict`/`eval` say something true.

### WR-09 — NEW: `apr inspect` still allocates up to 4 GiB from an attacker-controlled `u32`

**File:** `crates/apr-cli/src/commands/inspect.rs:548-566`

The bound this phase added is:

```rust
let fits = reader.get_ref().metadata()
    .is_ok_and(|m| header.metadata_offset.saturating_add(declared) <= m.len());
if !fits { return MetadataInfo::default(); }
let mut metadata_bytes = vec![0u8; header.metadata_size as usize];
```

`metadata_size` is a `u32` read straight out of the file (`apr-format/src/v2/header_impl.rs:80`),
`from_bytes` performs no range validation and does **not** verify the header checksum
(`inspect.rs:521` computes `checksum_valid` but nothing branches on it), so an attacker controls
the value up to `0xFFFF_FFFF` ≈ 4 GiB. The only bound is the file's own length — and a file long
enough is free:

```bash
truncate -s 4297M evil.apr          # sparse, zero disk cost
# write a 64-byte APR\0 header with metadata_offset=64, metadata_size=0xFFFFFFFF
apr inspect evil.apr                # allocates ~4 GiB, read_exact fills it, then
                                    # from_json fails and the command reports a plain APR
```

The command *succeeds* after burning ~4 GiB RSS. On a memory-constrained host or in a container
with a memory limit this is an OOM-kill of the tool CLAUDE.md mandates as diagnostic step 1.

This is in scope because it is code this phase wrote, and because the phase reached the opposite
conclusion two files away: `setfit_tag.rs:133-143` argues that a file-length bound is *not*
sufficient (*"a 30 GB APR whose metadata block declares 20 MiB passes (2) and would otherwise be
read in full just to answer one yes/no question"*) and adds an absolute 16 MiB cap. The deleted
`serve` copy named `inspect.rs:517` as the counterexample it refused to be. The gap is now the
only place in the phase's read surface without an absolute cap.

**Fix:** apply the same absolute cap the tag reader uses — the constant already exists:

```rust
if declared > crate::setfit_tag::MAX_TAG_METADATA_BYTES {   // make it pub(crate)
    return MetadataInfo::default();
}
```

`MetadataInfo::default()` is already the outcome for an unparseable block, so the observable
behaviour for a legitimate file is unchanged.

### WR-10 — NEW: `apr eval --lock-out` runs the whole candidate sweep before discovering the lock file already exists

**File:** `crates/apr-cli/src/commands/eval/setfit.rs:318-352`

`run_validation` loads and evaluates the primary artifact (321) and then every `--candidate` in
a loop (329-341) — each iteration a full bounded read, the eight-rung load ladder, and a
classify pass over the whole validation split — then calls `create_selection_lock` (347) and only
then `write_lock` (352), whose first act is `destination.exists() && !force`.

Concrete failure: `apr eval M.apr --candidate A.apr --candidate B.apr --candidate C.apr
--lock-out selection.json` on a directory where `selection.json` already exists spends four full
artifact loads and four validation sweeps and then exits with *"selection.json already exists.
A selection lock is a COMMITMENT…"* — a fact it could have reported before reading a byte.

This directly inverts the discipline the sibling module states and enforces
(`setfit_train.rs:12-20`): *"parse the config, merge the overrides, resolve the device, refuse an
existing output — all four BEFORE a single row of `--data` is read… Refusing the output path last
of the four is what stops a twenty-minute run from ending in 'I will not overwrite that file'."*
`setfit_train.rs:199-207` exposes `refuse_existing_output` as a standalone function precisely so
it can be called early; `apr eval` never calls it early.

**Fix:** pre-flight in `check_split_flags` (or immediately after it, before step (2) reads the
dataset):

```rust
if let Some(destination) = args.lock_out {
    if destination.exists() && !args.force {
        return Err(CliError::ValidationFailed(format!("{} already exists. …", destination.display())));
    }
}
```

The existing check in `write_lock` stays — it is what covers a file that appeared during the run.

---

## Info

### IN-01 — STILL OPEN: `config_hash_of`'s canonicality argument is wrong for this build

**File:** `crates/apr-cli/src/commands/eval/setfit.rs:557-562`

The doc claims the hash is canonical *"because … `serde_json::Map` preserves object order as
read."* That is true only with the `preserve_order` feature, which is not enabled
(`crates/apr-cli/Cargo.toml`, plain `serde_json = "1.0"`). Without it `Map` is a `BTreeMap` and
sorts keys. The output is still deterministic, so nothing is broken today — but
`CONFIG_HASH_DERIVATION` is published in `--json` as the recipe Phase 5 reproduces, and the
stated recipe is not the one the code runs. It is also a feature-unification hazard: any crate
in the graph enabling `preserve_order` silently changes every `config_hash` in every lock.

### IN-02 — STILL OPEN: two of the credential's three values are write-only in the lock chain

**Files:** `crates/aprender-train/src/train/setfit/lock.rs:594-617` and `796-816`

`create_selection_lock` records all three credential values; `mint_test_token` re-checks only
`artifact_hash` (line 606), and `grant` re-checks `artifact_hash` plus the dataset fingerprint
(802, 809). Neither ever compares the lock's recorded `selection_semantic_hash` / `ledger_hash`
against the credential presenting itself. Combined with `from_canonical_bytes`'s admission that
a hand-written, internally consistent lock passes its own integrity check (`lock.rs:419-429`), a
lock's recorded selection provenance can name a selection the artifact was never trained under.
Nothing wrong is reported today because `run_test`'s row does not surface those hashes — hence
Info. `mint_test_token` already holds the credential; adding the two equalities is three lines.

### IN-04 — STILL OPEN: `read_selection_manifest` was widened to `pub(crate)` and still reads unbounded

**File:** `crates/apr-cli/src/commands/data_contrastive.rs:610-611`

```rust
pub(crate) fn read_selection_manifest(path: &Path) -> Result<SelectionManifest> {
    let bytes = fs::read(path).map_err(...)?;
```

Every other reader this phase touched applies a stat-then-stream bound — `setfit_io.rs:67-91`,
`predict.rs:236-271`, `eval/setfit.rs:626-663`, `setfit_tag.rs:118-152`. This one takes an
operator-supplied path and reads it whole. The visibility change adds no new external entry
point (`apr data select` already reached it), but it is now the only unbounded read on the
`apr setfit train` and `apr eval` ingest paths.

### IN-05 — STILL OPEN: `RetainedArtifactBytes`'s `Debug` mitigation is partial at the level it claims

**File:** `crates/aprender-train/src/train/setfit/mod.rs:388-410`

The newtype's doc scopes its claim to `{:?}` on any `SetFitRun`, but `SetFitRun` also derives
`Debug` and holds `encoder: SetFitMiniLm` and `dataset: PreparedDataset<Canonical>`, so `{:?}` on
a run already renders every encoder tensor and the whole corpus — orders of magnitude past the
1.8 MB the newtype removes. The newtype is correct and worth keeping; narrow the claim to
`ArtifactVerifiedEvidence` (where it *is* true) or give `SetFitRun` a hand-written `Debug` too.

### IN-06 — NEW: dead `Write` import left by the cleanup pass, invisible to the compiler

**File:** `crates/apr-cli/src/commands/eval/setfit.rs:40`

```rust
use std::io::{Read as _, Write as _};
```

The cleanup deleted the hand-rolled writer (the only `write_all`/`sync_all` in the file), so
`Write` is now unused — `Read` is still needed by `file.take(..).read_to_end(..)` at 654-656.
This does not surface as a warning because `crates/apr-cli/src/lib.rs:9-16` carries a crate-wide
`#![allow(clippy::all, clippy::pedantic, clippy::disallowed_methods, unused_imports, dead_code,
unused_variables, unreachable_code, unused_assignments)]`. Verified empirically:
`cargo check -p apr-cli --lib --features setfit` recompiles the crate and emits **zero**
diagnostics for `apr-cli/src`.

Worth flagging beyond the one import: that blanket allow means the project's `unwrap()` ban
(`.clippy.toml` `disallowed-methods`) and the pedantic lint set are **not enforced in `apr-cli`
at all**. This phase's `apr-cli` code happens to be clean (0 `unwrap()` across all seven new
modules, verified by count), but nothing would have caught it if it were not.

**Fix:** drop `Write as _`. Separately, consider a `#![deny]` island or a
`#[allow(...)]`-per-module scheme so new Phase 4 modules are actually linted.

### IN-07 — NEW: `--force` is accepted and silently ignored on three `apr eval` paths

**Files:** `crates/apr-cli/src/commands/eval/setfit.rs:246-288`, `crates/apr-cli/src/dispatch_analysis.rs:788-809`

`check_split_flags` refuses `--selection-lock` on validation and `--lock-out` / `--candidate` on
test, and `dispatch_classify_eval` refuses `--selection`, `--lock-out`, `--selection-lock` and
`--candidate` on a non-tagged artifact. `--force` is refused on none of them. So
`apr eval M.apr --task classify --split test --force`, and `--force` on a non-SetFit artifact,
are both accepted and do nothing. `--force` only has meaning together with `--lock-out`.

`check_split_flags`'s own doc states the rule this misses: *"A flag silently ignored is worse
than a flag refused."*

**Fix:** add `("--force", args.force)` to the dispatch-level refusal table, and refuse
`--force` without `--lock-out` in `check_split_flags`.

---

## Round-1 → round-2 delta

| ID | Round 1 | Round 2 |
|---|---|---|
| CR-01 label-map gate | Critical | **CLOSED** — fix verified correct and unbypassable |
| CR-02 unauthenticated `/v1/classify` | Critical | **CLOSED** — gate applies; see WR-07 for the layering consequence |
| WR-01 rename clobber window | Warning | STILL OPEN |
| WR-02 predictable temp file | Warning | **CLOSED** — verified |
| WR-03 `config_hash_of` fail-open | Warning | STILL OPEN |
| WR-04 wrong error variant for arity | Warning | STILL OPEN |
| WR-05 `inspect` hides SetFit section | Warning | STILL OPEN |
| WR-06 unknown label / `zip` arity | Warning | STILL OPEN |
| WR-07 auth outside CORS | — | **NEW** |
| WR-08 tag cap fail-open misdiagnoses | — | **NEW** |
| WR-09 `inspect` 4 GiB allocation | — | **NEW** |
| WR-10 late `--lock-out` no-clobber check | — | **NEW** |
| IN-01 canonicality doc wrong | Info | STILL OPEN |
| IN-02 write-only credential values | Info | STILL OPEN |
| IN-03 third hex encoder | Info | **CLOSED** — verified |
| IN-04 unbounded manifest read | Info | STILL OPEN |
| IN-05 partial `Debug` claim | Info | STILL OPEN |
| IN-06 dead `Write` import + unenforced lints | — | **NEW** |
| IN-07 `--force` silently ignored | — | **NEW** |

---

_Reviewed: 2026-08-16_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard — re-review (round 2)_
_Diff base: 87b780eefabb4dbb6f5e64efe2f5b6698d058e73..HEAD_
