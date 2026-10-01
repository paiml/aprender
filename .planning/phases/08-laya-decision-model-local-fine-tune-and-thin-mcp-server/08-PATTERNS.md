# Phase 8: Laya Decision Model: Local Fine-Tune and Thin MCP Server - Pattern Map

**Mapped:** 2026-09-25
**Files analyzed:** 22 new or modified files, grouped by module
**Analogs found:** 20 / 22 (every analog path below was checked with `git ls-files`; none is a gitignored mirror)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `crates/aprender-core/src/models/modernbert/{mod,config,embeddings,layer,encoder,load,gemm}.rs` | model | transform (forward) | `crates/aprender-core/src/models/bert/*` (shape) + spike `.claude/skills/spike-findings-aprender/sources/025-laya-rust-forward-parity/src/laya.rs` (numerics) | role-match (shape) / exact (numerics) |
| `crates/aprender-core/src/models/mod.rs` (modify) | config/registration | - | same file, lines 33-36 | exact |
| `crates/aprender-decide/Cargo.toml` + `src/lib.rs` | library/service | request-response | `crates/aprender-mcp-setfit/Cargo.toml` (manifest), `crates/aprender-core/src/setfit/mod.rs` (lib door) | role-match |
| `crates/aprender-decide/src/task.rs` | utility (parser) | transform | none in-tree for order-preserving visitor; RESEARCH Pitfall 2 | no analog |
| `crates/aprender-decide/src/artifact.rs` | model/schema (writer + load ladder) | file-I/O | `crates/aprender-core/src/setfit/artifact.rs` (`write_setfit_apr` 811, `read_setfit_apr_bytes_bounded` 1922, `load_setfit_apr` 2038) | exact |
| `crates/aprender-decide/src/laya/{mod,head,scorer,builder,temperature}.rs` | model | transform | spike `025-.../src/laya.rs` (Builder 341, temperature 325, softmax_t 333) + `026-.../src/lib.rs` (`render_options` 175) | exact (spike lift) |
| `crates/aprender-decide/examples/pack_laya.rs` | utility (packer) | batch/file-I/O | `crates/aprender-core/src/setfit/artifact.rs::write_setfit_apr` | role-match |
| `crates/aprender-mcp-decide/{Cargo.toml,src/lib.rs,src/main.rs}` | controller (MCP server) | request-response | `crates/aprender-mcp-setfit/*` | exact |
| `crates/aprender-mcp-decide/tests/e2e_stdio.rs` | test | request-response | `crates/aprender-mcp-setfit/tests/e2e_stdio.rs` | exact |
| `crates/aprender-mcp-decide-lambda/{Cargo.toml,src/lib.rs,src/main.rs}` | controller (Lambda bootstrap) | request-response (loopback proxy) | `crates/aprender-mcp-chronos-lambda/*` | exact |
| `crates/aprender-mcp-decide-lambda/src/s3.rs` (cold-start loader) | service | file-I/O (S3 ranged GET, in memory) | spike `026-.../src/lib.rs` `s3_download` lines 96-130 | role-match (must change: into memory, not `/tmp`) |
| `crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml` | config | - | `crates/aprender-mcp-chronos-lambda/.pmcp/deploy.toml` | exact |
| `contracts/decide-apr-v1.yaml` | contract | - | `contracts/setfit-apr-v1.yaml` | exact |
| `contracts/laya-parity-v1.yaml` | contract | - | `contracts/chronos-bolt-parity-v1.yaml` | exact |
| `contracts/decide-tool-boundary-v1.yaml` | contract | - | `contracts/forecast-tool-boundary-v1.yaml` | exact |
| `scripts/laya_train/{pyproject.toml,uv.lock,.python-version,train.py,gate.py,fixtures.py}` | back-office script | batch | `scripts/setfit_fixtures/{pyproject.toml,.python-version,uv.lock}` + spike `024-.../tools/ft_laya.py` | role-match |
| `justfile` (modify: `laya-train`, `laya-pack`, `pmcp-decide-deploy`) | config | - | `justfile` lines 255-302 (`pmcp-train-deploy`, `pmcp-chronos-deploy`), 327+ (`fetch-chronos-tiny`) | exact |
| Root `Cargo.toml` (modify members) | config | - | `Cargo.toml` lines 64-65 | exact |
| `crates/aprender-core/tests/monorepo_invariants.rs` (modify `deployment_unit_bins`) | test | - | same file lines 312-320 | exact |
| `CLAUDE.md` (modify realizar-first table) | docs | - | SetFit and Forecasting rows + their paragraphs | exact |
| `crates/aprender-{decide,mcp-decide,mcp-decide-lambda}/README.md` | docs | - | `crates/aprender-mcp-setfit/README.md`, `crates/aprender-mcp-chronos-lambda/README.md` | exact |
| `deploy-extensions/lib/decide-weights-stack.ts` (optional) | config (CDK) | - | `deploy-extensions/lib/setfit-training-stack.ts` | role-match |

## Pattern Assignments

### `crates/aprender-core/src/models/modernbert/` (model, transform)

**Shape analog:** `crates/aprender-core/src/models/bert/mod.rs` (43 lines). Copy the module layout, the architecture doc block, and the re-export list:
```rust
pub mod config;
pub mod cross_encoder;
pub mod embeddings;
pub mod encoder;
pub mod layer;
pub mod load;

pub use config::{BertConfig, BertConfigError};
pub use embeddings::BertEmbeddings;
pub use encoder::BertEncoder;
pub use layer::BertLayer;
pub use load::{expected_bert_tensor_names, BertLoadError};
```
For ModernBERT, drop `cross_encoder`, add `gemm`, and export `ModernBertConfig`, `ModernBertEncoder`, `expected_modernbert_tensor_names`, `ModernBertLoadError`.

**Loader analog:** `crates/aprender-core/src/models/bert/load.rs` lines 29-80. Copy the imports and the "expected names is the import-load contract" function:
```rust
use crate::format::v2::AprV2Reader;
// issue #2231: `get_tensor_as_f32` is the re-attached `AprV2DequantExt` method.
use crate::format::AprV2DequantExt;

#[must_use]
pub fn expected_bert_tensor_names(config: &BertConfig, with_pooler: bool, classifier_prefix: &str) -> Vec<String> {
    let mut names = Vec::new();
    names.push("bert.embeddings.word_embeddings.weight".to_string());
    for idx in 0..config.num_layers {
        let p = format!("bert.encoder.layer.{idx}");
        ...
```
ModernBERT version takes a `prefix: &str` argument (RESEARCH: "generic prefix-aware loader") so `aprender-decide` can load `model.`-prefixed Laya tensors. F16 is widened by `get_tensor_as_f32` (`format/dequant_ext.rs:85-90`); no hand-written f16 code.

**Numerics source:** do NOT reuse bert's post-norm layer. Lift from spike `025-.../src/laya.rs`: `Linear` + `gemm_blis` (39-57), `layer_norm` (67), `gelu` (81, swap `libm::erf` for `batuta_common::math::erfc_precise` per RESEARCH and re-prove parity), `rope` (87), `attention` with the local window |i-j| <= 64 (107), `EncLayer` (143). `load_tensors` (181) is replaced by the prefix-aware `AprV2ReaderRef` loader above. Core GEMM call from the spike (line 57):
```rust
trueno::blis::gemm_blis(rows, m, k, &self.w[r0 * k..(r0 + rows) * k], &xt, c, ...)
```

**Registration** (`crates/aprender-core/src/models/mod.rs` lines 33-36):
```rust
pub mod bert;
pub mod qwen2;
pub use bert::{BertConfig, BertEncoder, CrossEncoder};
```
Add `pub mod modernbert;` and a matching `pub use`.

---

### `crates/aprender-decide/src/artifact.rs` (schema writer + load ladder, file-I/O)

**Analog:** `crates/aprender-core/src/setfit/artifact.rs` (5483 lines; read with offsets).

**Imports + contract-resident constants** (lines 55-86):
```rust
use super::tokenizer::sha256_hex;
use crate::format::v2::{AprV2Metadata, AprV2Reader, AprV2Writer, TensorDType, V2FormatError, VERSION_V2};

pub const ARTIFACT_SCHEMA: &str = "setfit-apr-v1";
pub const ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const MODEL_TYPE_TAG: &str = "setfit";
pub const CUSTOM_METADATA_KEY: &str = "setfit";
pub const TOKENIZER_BLOB_TENSOR: &str = "tokenizer.blob";
```
Mirror as `"decide-apr-v1"`, `CUSTOM_METADATA_KEY = "decide"`, and add a `task.json`, recipe, temperatures, gate report and input sha256 map in the one metadata document (D-17). The base is a declared field (D-04). From `aprender-decide`, the imports are `aprender::format::v2::...` (or `apr_format::v2` directly; RESEARCH names `AprV2Writer`/`AprV2ReaderRef` in apr-format).

**Writer rung order** (`write_setfit_apr`, lines 811-850). Copy the numbered, fail-before-write structure:
```rust
pub fn write_setfit_apr(view: &SetFitArtifactView) -> Result<Vec<u8>, SetFitArtifactError> {
    // (1) NAMES ... validate_view_structure(view, &hf_name_map)?;
    // (2) EVERY f32 ... scan_view_floats(view)?;
    // (3) TOKENIZER IDENTITY, BEFORE THE REBUILD.
    let observed = sha256_hex(&view.tokenizer_bytes);
    if observed != view.architecture.tokenizer_sha256 { return Err(...TokenizerHashMismatch {..}); }
    // (4) PROBES, computed from the view's OWN parts
    // (5) THE ONE DOCUMENT.  (6) NULL WALK.
    // (7) THE CONTAINER. Nothing has been written until here
    write_container(view, &hf_name_map, doc)
}
```
Laya adds a rung: re-score every eval row in Rust and refuse if any |dp| > 1e-5 against the Python F16-reload probabilities (RESEARCH primary recommendation).

**Bounded read** (lines 1922-1960). Copy exactly; this is the door both servers call:
```rust
pub fn read_setfit_apr_bytes_bounded<R: std::io::Read>(reader: R, declared_len: Option<u64>) -> Result<Vec<u8>, SetFitArtifactError> {
    read_setfit_apr_bytes_bounded_within(reader, declared_len, &ArtifactLimits::CONTRACTED)
}
// (a) THE DECLARED LENGTH, BEFORE THE READER IS TOUCHED ... ArtifactTooLarge { what: "declared_length", observed, cap }
// (b) THE READ ITSELF, BOUNDED AT cap + 1 ANYWAY
```
The cap must admit the ~0.85 GB artifact. Also mirror `artifact_sha256_hex` (line 781), which is the model-identity hash for D-11, and `load_*_apr` (line 2038) as the only load door.

---

### `crates/aprender-decide/src/laya/*` (model, transform)

**Analog:** spike `.claude/skills/spike-findings-aprender/sources/025-laya-rust-forward-parity/src/laya.rs`, specifically `HeadLayer` (153), `Laya` (162), `temperature(cfg, qtype, k)` (325), `softmax_t` (333) and `Builder` (341). Also spike `026-.../src/lib.rs` `render_options` (175) and `render_criterion` (167), with its refusal rules: unknown types, fewer than 2 options, and options beyond `head_max_len` are refused, not defaulted. Lift these into the named files, one struct per file. Replace every `unwrap()` with `ok_or_else(..)?`, because the spike code is not lint-clean.

---

### `crates/aprender-mcp-decide/` (MCP server, request-response)

**Analog:** `crates/aprender-mcp-setfit/` (exact).

**Cargo.toml** (setfit Cargo.toml, whole file): copy `publish = false` and its comment, the `[lib]` + `[[bin]]` pair, `pmcp = { version = "2.9", features = ["streamable-http", "schema-generation"] }`, `schemars = "1.0"`, and `[lints] workspace = true`. Replace the `aprender` dep with `aprender-decide = { path = "../aprender-decide" }`.

**lib.rs header and lint allow** (setfit lib.rs lines 1-33): keep the "One model, one tool, on purpose" and "Bounds ownership" doc sections, plus:
```rust
#![allow(clippy::disallowed_methods)]
```
**Re-export for transport crates** (line 47):
```rust
pub use aprender::setfit::VerifiedSetFitModel as Model;
```
**Strict args** (lines 69-79): `#[derive(Debug, Deserialize, JsonSchema)] #[serde(deny_unknown_fields)] pub struct ClassifyArgs { pub texts: Vec<String>, ... }`. For decide, `texts` is the only field (D-09: the question and labels come from `task.json`).

**Load doors** (lines 84-144): `ModelLoadError { Io, Artifact }` with `Display`, `load_model_from_path` (open, then metadata len, then bounded read, then load ladder), and `load_model_from_bytes`. The Lambda crate needs the from-bytes door because S3 bytes stay in memory.

**precheck** (lines 155-174). Here the count bound and the byte bound name the contract file and key in the message:
```rust
if args.texts.len() > MAX_BATCH_TEXTS {
    return Err(pmcp::Error::validation(format!(
        "batch of {} texts exceeds max_batch_texts {MAX_BATCH_TEXTS} \
         (contracts/setfit-apr-v1.yaml item 11); split the batch", args.texts.len())));
}
```
Decide adds the token-budget bound (RESEARCH trap 4) and points at `contracts/decide-tool-boundary-v1.yaml`.

**Error taxonomy** (lines 180-187): caller-fixable errors map to `validation`, everything else to `internal`.

**build_server** (lines 197-222): `Server::builder().name().version().capabilities(ServerCapabilities::tools_only()).tool_typed_with_description::<ClassifyArgs,_,_>(TOOL_NAME, TOOL_DESCRIPTION, move |args,_extra| { ... tokio::task::spawn_blocking(move || model.classify(&document)) ... serde_json::to_value(&response) })`. Keep `spawn_blocking`, since a 512-token row takes seconds.

**Unit tests** (lines 224-321): copy all six test shapes: unknown key refused, args convert to the document, over-bound batch refused, single oversized text refused, maximal legal request passes, and schema is strict (`additionalProperties == false`).

**main.rs** (setfit main.rs, 86 lines): copy it as-is. It covers `--model` / env-var fallback, stderr-only logging ("stdout belongs to the protocol"), `ExitCode::from(2)` for usage errors and `run_stdio()`. Rename the env var to `APRENDER_DECIDE_MODEL`.

**tests/e2e_stdio.rs** (setfit e2e, lines 1-40): copy the env-gated SKIP (println + early return, never `#[ignore]`), `env!("CARGO_BIN_EXE_...")` pinning, `KillOnDrop`, and the long `INIT_TIMEOUT` for debug-build load. Remember that the test is dark until someone adds it to the ci.yml `--test` line, and editing CI needs a check-in.

---

### `crates/aprender-mcp-decide-lambda/` (Lambda bootstrap, loopback proxy)

**Analog:** `crates/aprender-mcp-chronos-lambda/` (exact).

**Cargo.toml** (whole, 39 lines): copy `publish = false`, `[[bin]] name = "bootstrap"` with the platform-constant comment, `lambda_http = "0.13"`, `reqwest` (rustls, no default features), `once_cell`, and `tracing*`. Add `aws-sdk-s3`/`aws-config` at the LOCKED versions (1.137.0 / 1.8.18; do not `cargo update`, per RESEARCH).

**lib.rs** (38 lines): a shared `server_config()` returning `StreamableHttpServerConfig::stateless()`, plus `resolve_model()` and `build_server()` delegating to the server crate. Drop `EMBEDDED_WEIGHTS` (D-18: never bake weights). `resolve_model` becomes async S3-in-memory, falling back to an env path for local runs, the same way chronos's `resolve_model` falls back to `CHRONOS_MODEL_DIR` (`crates/aprender-mcp-chronos/src/lib.rs:67-78`).

**main.rs** (187 lines): copy the whole thing, including `BASE_URL`/`HTTP` `OnceCell`s, `server_name()` from `PMCP_SERVER_ID`, `start_http_in_background` on 127.0.0.1, `ensure_server_started`, the GET health check, OPTIONS CORS, and the header-filtered proxy that drops `host`, `transfer-encoding` and `content-length`. Change `DEFAULT_SERVER_NAME` and the health message.

**S3 loader:** spike `026-.../src/lib.rs` lines 96-130 has the parallel 64 MiB ranged GETs, a semaphore of 16 and 5-attempt retry. **Must change:** it writes to a `/tmp` file via `write_all_at`. Lambda `/tmp` is 512 MB and cargo-pmcp cannot raise it (RESEARCH trap 3), so write each part into a pre-sized `Vec<u8>` at offset `s`, verify the pinned sha256, then call `load_model_from_bytes`. Replace `content_length().unwrap_or(0)` with a refusal.

**.pmcp/deploy.toml** (chronos, 62 lines): copy the file and change `[server] name`, `memory_mb = 10240`, and the header comment. Keep `timeout_seconds = 30`; the API Gateway ceiling cannot be raised. Add `[environment]` keys for the S3 URI and pinned sha256.

---

### Contracts

- **`contracts/decide-apr-v1.yaml`**: copy the top-level layout of `contracts/setfit-apr-v1.yaml`: `contract:` / `metadata:` (1-79, with the description paragraphs and the PROVABILITY / ANNOTATION / BINDING HONESTY blocks, lines 43-60), `equations:` (80), `proof_obligations:` (1017), `falsification_tests:` (1110), `kani_harnesses:` (1250) and `qa_gate:` (1283). The storage map binds head tensors to named tensors, not to metadata (the "HEAD IS TENSORS" paragraph).
- **`contracts/laya-parity-v1.yaml`**: copy `contracts/chronos-bolt-parity-v1.yaml` (539 lines): `constants:` (73), `equations:` (87) and `falsification_tests:` (364). FALSIFY-CHRONOS-001/002 are the template, with a per-arch bar, a printed measured max|delta|, `if_fails` diagnosis by ladder rung, and the "do not fit a bar from one side" lesson. Bars: probs <= 1e-5 and ids 14/14.
- **`contracts/decide-tool-boundary-v1.yaml`**: copy `contracts/forecast-tool-boundary-v1.yaml` (1607 lines): `constants:` (64), `door_surface:` (378), `equations:` (740, e.g. `fit_server_bounds`, `holiday_design_cost_bounded`, which is the composition-bound precedent for the token budget), `falsification_tests:` (1179). Put the gate thresholds here too (ECE ceiling, zero-shot margin, D-07), declared before any run.

**Contract-mirror test** (`crates/aprender-forecast/src/types.rs:745-767`, with helper `constant_u64` at `crates/aprender-forecast/src/test_support.rs:213`):
```rust
#[test]
fn bounds_match_contract() {
    assert_eq!(MIN_POINTS as u64, constant_u64("forecast-tool-boundary-v1", "fit_min_points"),
        "types::MIN_POINTS must equal constants.fit_min_points in forecast-tool-boundary-v1");
```
Every Rust bound in `aprender-mcp-decide` gets the same kind of test against `decide-tool-boundary-v1`. Copy the helper into that crate, because the forecast one is `pub(crate)`.

---

### `scripts/laya_train/` (uv project, batch)

**Analog:** `scripts/setfit_fixtures/pyproject.toml`. Copy the conventions: an upper-bounded `requires-python = ">=3.13,<3.14"`, a `.python-version` of `3.13.7`, exact `==` pins with a comment saying why each is pinned, constraints labelled as constraints, and `uv.lock` committed. Laya is a git dependency at `4066d5d5fbf08b66c6757ddeedbd797bd7655bc0`; transformers is `5.17.0`. `train.py` productises spike `024-.../tools/ft_laya.py`. Record the device from torch (D-03). Compute gate, calibration and parity on the F16-rounded reload (RESEARCH trap 5).

### `justfile` recipes

**Analog:** `justfile` lines 255-302. `pmcp-train-deploy` is the model for a recipe that is a `#!/usr/bin/env bash` + `set -euo pipefail` script with `ulimit -n 65536`, precondition `test -f` checks and explicit ERROR messages. Its header comment (255-278) documents the cargo-pmcp `find_lambda_package_dir` fallback, which is the exact wrong-server trap in RESEARCH trap 1. `pmcp-decide-deploy` must also refuse when there is no passing gate report (D-07) and run a post-deploy identity probe. The plain form is `pmcp-chronos-deploy` (299-302). For pinned fetch-and-verify, use `fetch-chronos-tiny` (327+) with its `chronos_rev` / `chronos_dir` variables (36-38).

### `crates/aprender-core/tests/monorepo_invariants.rs` (modify)

Lines 312-320, `deployment_unit_bins`. Append:
```rust
"aprender-mcp-decide",         // Laya decision server (Phase 8 D-15)
"aprender-mcp-decide-lambda",  // its AWS Lambda custom runtime (bootstrap)
```
Keep `aprender-decide` lib-only, with no bin. Note that the chronos-lambda entry uses the package name while its bin is `bootstrap`; copy that convention.

### `CLAUDE.md` (modify)

Add a row after "Time-Series Forecasting" in the realizar-first table, and a paragraph in the style of "The Forecasting row is a second deliberate EXCEPTION (Phase 6 D-07)". Every repo path cited must exist, or FALSIFY-DOCS-CLAUDE-001 fails. Do not widen the SafeTensors carve-out.

## Shared Patterns

### Thin-server bounds ownership
**Source:** `crates/aprender-mcp-setfit/src/lib.rs` lines 20-26 and 155-174, plus `contracts/forecast-tool-boundary-v1.yaml` `door_surface`.
**Apply to:** `aprender-mcp-decide`, `aprender-mcp-decide-lambda`. The transport checks the bounds and the library door re-checks nothing. Every refusal names the contract file and key, and never echoes the offending payload.

### Lint compliance
**Source:** `#![allow(clippy::disallowed_methods)]` at the top of setfit `lib.rs` (line 33) and chronos-lambda `main.rs` (line 9), used only for the `json!`/schemars expansions; `#[cfg(test)] #[allow(clippy::disallowed_methods)]` on test modules.
**Apply to:** all new Rust files. There is no `unwrap()` anywhere else, so spike code must be cleaned when lifted.

### Blocking inference off the async loop
**Source:** setfit `build_server`, `tokio::task::spawn_blocking(move || model.classify(&document))`.
**Apply to:** the decide `classify` handler.

### One hash path / model identity
**Source:** `crates/aprender-core/src/setfit/artifact.rs:781` `artifact_sha256_hex`, delegating to `sha256_hex`.
**Apply to:** the artifact writer, the S3 loader's pinned-hash check, and the `classify` response identity (D-11).

### Stateless streamable HTTP
**Source:** `crates/aprender-mcp-chronos-lambda/src/lib.rs` `server_config()` -> `StreamableHttpServerConfig::stateless()`.
**Apply to:** decide-lambda.

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `crates/aprender-decide/src/task.rs` | parser | transform | No in-tree order-preserving `MapAccess` visitor. It must not go through `serde_json::Value`/`Map`, because `preserve_order` is feature-unified by pmcp and absent standalone (RESEARCH trap 2). Build it from the RESEARCH probe, and test it under both feature shapes |
| `crates/aprender-decide/src/lib.rs` `DecisionMethod` seam | trait | - | No existing method-neutral decision trait. Keep it to the one Laya implementation, designed for Kev as the known second (CONTEXT specifics) |

## Metadata

**Analog search scope:** `crates/aprender-mcp-setfit`, `crates/aprender-mcp-chronos`, `crates/aprender-mcp-chronos-lambda`, `crates/aprender-mcp-setfit-lambda`, `crates/aprender-core/src/{models/bert,setfit}`, `crates/aprender-forecast/src/types.rs`, `contracts/`, `scripts/setfit_fixtures`, `justfile`, `.claude/skills/spike-findings-aprender/sources/02{4,5,6}-*` (git-tracked)
**Files scanned:** ~30
**Pattern extraction date:** 2026-09-25
