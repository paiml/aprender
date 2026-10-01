# Phase 8 — /code-review max findings (2026-09-28)

Source: `/code-review max --fix` over `origin/gsd/phase-2-contract-gate...HEAD`, run after
08-REVIEW.md. Copied here from the session scratchpad so it survives; the verdict IDs below
(V<n>-<x>) are the reviewer's.

## Already applied in commit ea940faec (do NOT re-plan these; verify they hold)

1. `verify_path` refuses an artifact the run does not re-pack byte-for-byte (`ArtifactNotFromRun`) — the VERIFY side of CR-01. The LOAD side (rung-4 manifest↔blob cross-checks) is NOT done.
2. S3 part retry resumes at the first missing byte (V4-a).
3. eval set missing a criterion refused — Rust `check_eval_coverage`, data.py `eval-class-coverage` (V7-c).
4. `MAX_NUM_HIDDEN_LAYERS` 1024, `MAX_HEAD_LAYERS` 64 (V1-a).
5. Rung 2 `MAX_METADATA_BYTES` 1 MiB (V2-a).
6. classify refusals never echo caller text (V5-a; the JSON-RPC -32603 side note is NOT fixed).
7. PARTIAL (V6-c): base weights hashed on scored bytes, base tokenizer bound (`BaseTokenizer`). base `encoder/config.json` and `rl_agent_config.json` remain UNBOUND — needs sha256 pins in the gate contract's base block.
8. One SHA-256 per cold start: `HashedArtifact`, `Decider::load_hashed` (V3-a, V3-b).
9. `check_served_task_fits` at build_server (V5-d).
10. Unsupported activation / rope_type / rope_scaling refused (V10-a).
11. `load_tensor` F16/F32 + exact size only (V10-c).
12. train.py pins input hashes at startup, refuses a changed data dir (V13-b).
13. laya-deploy check (6a): env vs weights bucket before any live change (V11-d).
15. shift probe dropped while shift.jsonl present → `ShiftProbeMismatch` (V6-a).

SKIPPED (#14 = V4-b): `DOWNLOAD_DEADLINE` 25 s is stale (sha+build measured 7.8-11 s at
3,008 MB; the #8 fix removes ~3 s of that) and the handler never reads the Lambda invocation
deadline. Choosing the budget needs a live cold measurement.

## Verdicts (full list; CONFIRMED items not in the list above are open)

V8-a REFUTED (t_applied: 1e-3 ceiling refuses any realistic wrong T; provenance-only, low)
V8-b REFUTED (argmax compare matches contract formula; label_index == argmax today)
V8-c CONFIRMED low (write_atomic symlink follow via File::create; reproduced; limited exposure)
V8-d CONFIRMED low (ProbsRowCoverage Display drops which)
V11-a CONFIRMED low (justfile:1387 `^SKIP:` misses laya_parity.rs:274 "SKIP ladder rung:"; ladder rung never runs, LEG OK; ladder bin exists locally at .planning/spikes/025-laya-rust-forward-parity/fixtures/laya-en_ladder.bin)
V11-b CONFIRMED low (justfile:2366 for-list $(aws ...) masks failure -> false "missing grant" diagnosis; containment by design; teardown 2593 mislabels any delete failure as expected-absent)
V11-c CONFIRMED low (sha256() rtk proxy in 3 recipes; rtk output byte-identical; foreign rtk -> abort / H="rtk")
V11-d CONFIRMED medium (laya-deploy never checks env vs config bucket before `cargo pmcp deploy`; mismatch replaces live fn then contain -> outage)
V12-a CONFIRMED low (probe ok() ignores response labels; justfile:1865 comment claims probe rules out wrong binary w/ labels in order)
V12-b CONFIRMED low (pack_laya as_u64 vs tests as_f64 as u64; `ece_bins: 15.0` -> prod refuses all, tests green; fail-closed)
V12-c CONFIRMED low (pack_laya args() panics exit 101 on non-UTF-8; also lambda examples/probe.rs:76)
V12-d REFUTED (logits assert w/ length check precedes; softmax finite)
V3-a CONFIRMED medium (double sha256 per cold load; sha_ms/build_ms ratio 0.47-0.51; IdentityMismatch unreachable; fix: pinned door computing digest once)
V3-b CONFIRMED low (lambda identity tests self-referential; golden laya_tiny.apr.sha256 exists; fix: compare lambda tests to golden)
V3-c CONFIRMED low (misleading lib.rs:356-357 comment; "lock" wording still roughly accurate; NO semantic regression in LoadOnce)
V3-d CONFIRMED low (read_local_bounded outer stream TooLarge dead at contracted cap; kind "read")
V10-a CONFIRMED medium (hidden_activation/rope_type/rope_scaling ignored; silu 0.28, yarn 0.55, linear 1.0 max diff vs 1e-4 bar; all in-tree configs gelu/default -> refusing others breaks nothing)
V10-b CONFIRMED low (ModernBertLayer::forward builds RopeTable(l) before check_len; l=usize::MAX/2 panics; no in-tree caller)
V10-c CONFIRMED medium (load_tensor widens any dtype; AprQ4 0-byte / NaN-scale loads as zeros via public from_apr; decide shielded)
V6-a CONFIRMED low (check_shift_probe (None,None) ignores data.shift; lifecycle.py enforces it in Python; no test)
V6-b CONFIRMED low (check_base compares only sha256; revision/checkpoint/repo -> served base string; tiny tests rely on gap)
V6-c CONFIRMED medium (base_dir encoder/agent/tokenizer unhashed; model.safetensors re-read unhashed after ladder (TOCTOU); tokenizer binding available via inputs_sha256.tokenizer_json)
V6-d CONFIRMED low (early_stopping + whole recipe block never compared in Rust)
V6-e CONFIRMED low (shift/top-level/per_seed f_avg never recomputed)
V4-a CONFIRMED high (8 s timeout on whole 64 MiB part + restart at 0; 6/8 parts cut per live cold load; latent all-parts-cut -> s3_deadline every cold start; SDK has 5 s stalled-stream guard already)
V4-b CONFIRMED medium (DOWNLOAD_DEADLINE 25 s stale (sha+build 7.8-11 s); invocation deadline never read)
V4-c PLAUSIBLE low (HEAD/PUT/other paths trigger load; edge behaviour unknown)
V4-d REFUTED (retry-every-request is documented intent)
V1-a CONFIRMED medium (reproduced via real Decider::load_bytes: 2,692-byte .apr, num_hidden_layers=2^62 -> SIGABRT; u64::MAX -> capacity overflow panic; 3e6 layers -> 3.53 GB; head_layers 1e6 -> 1.26 GB; derived count = 6L + 12H + 20; Lambda path pinned)
V1-b CONFIRMED low (task.rs O(N^2) duplicate scan; 80k criteria 46.8 s debug / 5.8 s release; no max-criteria bound anywhere)
V13-a CONFIRMED low (load_rows UnicodeDecodeError / lone-surrogate UnicodeEncodeError -> traceback rc 1, not REFUSED rc 2; reproduced)
V13-b CONFIRMED medium (train.py inputs_sha256 + task.json copy taken at END from disk; demonstrated: report binds edited train.jsonl hash; pack/verify cannot see it)
V13-c CONFIRMED low (EarlyStopper best-anchored vs contract :542 running-min formula; contract :211/:216/FALSIFY-009 disagree among themselves)
V13-d PLAUSIBLE low (PyYAML `1e-6` str; latent — no current value; chain demonstrated in-memory)
V13-e CONFIRMED low (NFC Unicode 17 vs 15.1; fails closed late)
V2-a CONFIRMED medium (rung 2 never bounds metadata_size; AprV2Metadata::from_json -> Value tree + flatten copy; 16 MiB -> 0.84 GB, 64 MiB -> 2.82 GB peak; MAX_METADATA_SIZE unused; S3 path shielded by pin)
V2-b CONFIRMED low (index_capacity reserves 3.6x bytes-to-EOF; lazy on macOS; decide shielded by rung 2)
V2-c CONFIRMED low (forged_tensor_count test passes with the fix reverted on macOS — mutation-verified)
V9-a PLAUSIBLE low (tokenizer.json truncation/padding applied by tokenizers encode, disabled by transformers per call; empirically shown with injected truncation; pinned base + fixture have null -> latent)
V9-b CONFIRMED low (probe-replay classify failure labelled "6 rebuild" instead of rung 7)
V9-c REFUTED (contract marker_rule explicitly accepts zero-room)
V9-d CONFIRMED low (doc-only: rung 2 doc says "or version" but version never checked; contract doesn't require it)
V9-e REFUTED (inspect documented as manifest-only identity, never eligibility)
V7-a CONFIRMED low (f32 vs f64 macro-F1 at exact 0.05 margin -> PassDisagrees / per_seed.pass refusal; fail-closed; ~1e-6 per seed at 459 rows)
V7-b CONFIRMED low (rank_key f32 ECE vs f64: 0.0425%/seed, ~0.13%/production run; fail-closed; contract premise false)
V7-c CONFIRMED medium (eval.jsonl missing a criterion is never refused; gate FAIL-OPEN: reproduced ft worse on every present class passes with margin 0.19 because zs predicted the absent class once)
V5-a CONFIRMED medium (typed-tool serde error echoes caller text on the wire — reproduced on shipped stdio binary with a 200 KiB echo; side: every refusal goes out as JSON-RPC -32603 Internal (pmcp create_response hardcodes INTERNAL_ERROR) though contract says internal code on a bound refusal is a defect)
V5-b CONFIRMED low (Busy unreachable: stdio single worker + unbounded mpsc; HTTP Mutex<Server>; Lambda impact nil; contract invariant false)
V5-c CONFIRMED low (count not first: pmcp parses frame + Vec<String> first; 3M empty texts -> 764 MB RSS on stdio; 1 MiB cap is ServerCore-only)
V5-d CONFIRMED medium (served task's min row tokens never measured at build_server; 57 today, 6-token margin; 61-120 -> every 2-text call refused while advertising 1..=2)
V14-a CONFIRMED low (resolver EREs have no case table; doc comments / cfg(test) helpers / just vars resolve; latent; fn/recipe EREs pre-existed in phase6)
V14-b PLAUSIBLE low (shell resolver instead of extending pv; pv's existing checks don't fit)
V14-c CONFIRMED medium (Phase 8 adds a new offender to each of 3 CI-required guards: cascade TIERS (aprender-decide), hand-rolled parser (aprender-mcp-decide), duplicate `bootstrap` bin; all 3 already red at base)
V14-d REFUTED (no server reads safetensors; BUT "servers never call pack" comments false: artifact.rs + working-tree task.rs call crate::pack::sha256_hex on the load path)

## Candidate detail (reviewer notes)

# Candidates (raw, by angle)

## B (removed behavior)
B1 lambda/src/lib.rs:446 — cold path hashes the 846 MB artifact twice (pin, then ladder rung-8 mint); IdentityMismatch check is f(bytes)!=f(bytes). ~3.5 s at cold start vs 650 ms margin.
B2 lambda/src/tests.rs:47 + s3.rs:673/686/768 — dedup removed the independent sha2 oracle; lambda identity tests compare the code under test with itself (mcp-decide keeps an independent sha2 dev-dep for exactly this).

## A1 (line-by-line verify/pack)
A1-1 verify.rs:2239 / pack.rs:799-821 — zero-shot base scored from base-dir config/agent/tokenizer files never hash-bound (only model.safetensors hashed by check_base, then re-read later: TOCTOU).
A1-2 verify.rs:2053 — check_shift_probe returns Ok on (None, None) without looking at data.shift: a data dir WITH shift.jsonl verifies with the probe deleted from the report (contract: present exactly when shift.jsonl is).
A1-3 verify.rs:930 — check_base compares only recipe.base.sha256; family/checkpoint/repo/revision unchecked but become the served `base` identity string.
A1-4 verify.rs:2333 — write_atomic: predictable temp name + File::create follows symlinks (no O_EXCL).
A1-5 verify.rs:1985 — rank_key floors f32-accumulated Rust ECE; Python ranks on f64 ECE; exact rank-key equality can refuse an honest seed (contract premise "agree to 1e-12" false).
A1-6 pack.rs:88 — recipe.early_stopping parsed but never compared to the contract block (only seed_selection is).
A1-7 verify.rs:2129 — check_shift_probe never recomputes shift f_avg though the contract lists it.
A1-8 verify.rs:629 — Display of ProbsRowCoverage drops `which` (which of 7 files).

## E (wrapper/proxy)
E1 = B1 (double sha at cold start; IdentityMismatch tautological).
E2 mcp-decide lib.rs:555 — classify_max_pending (Busy) unreachable: pmcp serializes dispatch (stdio single worker w/ unbounded channel; HTTP holds Mutex<Server> across tool future) -> unbounded queue, contract classify_admission violated; tests call Admission directly.
E3 mcp-decide lib.rs:646 — pmcp typed-tool deserialization error echoes caller text ("Invalid arguments for tool 'classify': invalid type: string \"<text>\"...") — violates refusal_names_bound / ASVS V7; untested path.
E4 lambda main.rs:162 — HEAD/other methods/paths trigger the full cold load before proxying.
E5 lambda lib.rs:513 — resolve_local: over-cap stream surfaces as kind "read" (inner reader refuses first); TooLarge{stream} branch dead at contracted cap.

## Reuse
R1 verify.rs:1005 normalize_text/normalized_sha256 duplicate aprender-contrastive-data hash::normalized_hash (nfc-trim-ws-v1).
R2 lambda lib.rs:499 read_local_bounded re-implements rung 1; mcp-decide load_model_from_path == Decider::load_path.
R3 Makefile:2557 contract-audit-phase8 ~130-line copy of phase6 (already diverged).
R4 lambda main.rs:196 loopback proxy = 3rd copy of chronos/setfit lambda handler.
R5 tests/fail_closed_vectors.rs:90 duplicates tests/common (policy(), contract(), workspace_root, sha256_file).
R6 probe.rs:639 build_maximal_request re-implements check_token_budget (plain sum vs saturating fold).
R7 fixtures.py:356 calibration_slice re-implements data.calibration_split (group-unaware); slice hash / task.json bytes inlined.
R8 justfile:2305 laya-upload duplicates laya-deploy eligibility parse; bucket naming x4; sha256() x3.

## Altitude
AL1 = B1 (named alt: Decider::load_bytes_pinned hashing once).
AL2 pack_laya.rs:96 — contract->VerifyPolicy mapping x4; prod uses as_u64 (refuses `15.0`), tests use as_f64 as u64 -> tests can't catch prod mapping bugs. Alt: VerifyPolicy::from_contracts.
AL3 apr-format reader_impl.rs:93 — index_capacity bounds in on-disk units (20 B) but TensorIndexEntry is ~72 B in memory -> forged count reserves ~3.6x file size (abort on strict-overcommit hosts) for non-decide consumers. Alt: Vec::new().
AL4 lambda main.rs:191 — loopback TCP proxy vs in-process pmcp::axum::router_with_config + oneshot.
AL5 reader_impl.rs:131 — WR-01 root cause is reader's non-strict `<` (known WR-01, reframed).
AL6 = R3.
AL7 justfile:2228/1387 — laya-deploy-selftest prints SKIP then DEPLOY SELFTEST OK; laya-verify-suite greps `^SKIP:` but laya_parity prints `SKIP ladder rung: ...` (no colon after SKIP) -> LEG OK on a skipped rung.
AL8 verify.rs:1313 — rescore compares argmax(&d.probabilities) not d.label_index (never observes the served decision); ties verifier's numpy mirror to laya's argmax.

## A4 (line-by-line MCP servers)
A4-1 = B1.
A4-2 s3.rs:297 — ATTEMPT_TIMEOUT (8 s) bounds a whole 64 MiB part, retry restarts at byte 0; with bandwidth shared by 13 parts a part needs ~9.5 s -> progress-making parts cut; live evidence: 6-8 attempts cut at 8000 ms, download 13.5-17.8 s vs ~9.2 s.
A4-3 = E3.
A4-4 s3.rs:34 — DOWNLOAD_DEADLINE 25 s justified as "30 s cap minus ~5 s sha+build" but measured sha+build is 7.8-11 s at 3008 MB.
A4-5 probe.rs:105 — ProbeReport::ok() ignores the tools/call `labels` (never compared to expected); README says probe checks label order.
A4-6 main.rs:127 — (≈E4) GET health ok:true without validating config; HEAD triggers the cold load.
A4-7 decide-tool-boundary-v1.yaml:298 — "no refusal contains any non-empty caller text" is unsatisfiable (texts ["a"]).
A4-8 decide-tool-boundary-v1.yaml:185 — count "checked FIRST before any text is read" is false: pmcp TypedTool deserializes the whole array first; 1 MiB args cap not enforced by pmcp::Server; stdio uncapped.

## C1 (Rust cross-file)
C1-1 artifact.rs:1356 (+ modernbert/config.rs:318, artifact.rs:738, load.rs:95) — rung 4(b) allocates from author-controlled num_hidden_layers/head_layers before bounding vs MAX_TENSOR_COUNT: a 2 KB .apr with num_hidden_layers=2^50 aborts (Vec<bool> of 1 PiB in validate) / OOM instead of a typed refusal.
C1-2 = A1-1 + A1-3 (base_dir config/agent/tokenizer unbound; recipe.base revision/repo/... unchecked).
C1-3 = A1-2.
C1-4 mcp-decide lib.rs:260 — served_task_min_row_tokens (57) assumption never measured on the loaded artifact; long-prefix task -> every 2-text call refused while description advertises 1..=2.
C1-5 = B1 (IdentityMismatch dead; new doc comment wrong).

## A2 (line-by-line artifact/laya)
A2-1 = C1-1.
A2-2 artifact.rs:1225/1282 — rung 2 never bounds the header-declared metadata section; AprV2ReaderRef::from_bytes parses up to ~1 GiB JSON into a Value tree (16-36x) before rung 3; apr-format MAX_METADATA_SIZE (16 MiB) unenforced.
A2-3 artifact.rs:1296 — inspect_manifest stops at rung 3; `pack_laya inspect` prints recipe_id/labels (identity) never bound to the blobs.
A2-4 laya/builder.rs:93 — Tokenizer::from_bytes keeps tokenizer.json truncation/padding; HF Python disables them per call -> Rust truncates state silently (truncated=false) for a base shipping truncation. Fix: with_truncation(None)/with_padding(None).
A2-5 artifact.rs:859 — probe-replay classify failure mapped to ArtifactError::Rebuild ("6 rebuild") instead of rung 7.
A2-6 tests/laya_parity.rs:248 — NaN-masking fold (m.max(d) drops an earlier NaN) + zip without length check in the probs rung.
A2-7 laya/builder.rs:207 — marker rule doesn't require room for text: a long-prefix task loads and scores every request on zero caller tokens; truncated=true for empty text.
A2-8 artifact.rs:341 — header version never checked (doc claims rung 2 refuses a bad version).

## D1 (Rust pitfalls decide/core)
D1-1 = C1-1 (repro: num_hidden_layers 4.6e18 -> SIGABRT; u64::MAX -> capacity overflow panic).
D1-2 task.rs:111 — duplicate-criterion check is O(N^2) (`pairs.iter().any(...)` per entry); 100k criteria 8.5 s; hostile task blob hangs rung 4(d). Fix: HashSet.
D1-3 = AL3 (index_capacity uses EOF window, 20 B on-disk vs 72 B in-memory).
D1-4 = A1-2.
D1-5 = A1-5 (measured: 1/2000 random 459x3 sets flip floor(ece*1e4)).
D1-6 = A1-4.
D1-7 examples/pack_laya.rs:326 — std::env::args() panics on non-UTF-8 arg -> exit 101, not documented exit 2. Fix: args_os.
D1-8 modernbert/config.rs:155 — RawConfig ignores hidden_activation and rope_parameters.*.rope_type/scaling: other activation or RoPE scaling silently computed as exact GELU + default RoPE (contradicts "refused by name").

## D2 (Rust async/IO pitfalls servers)
D2-1 = A4-2. D2-2 = B1. D2-3 = E2.
D2-4 lambda main.rs:162 — no invocation-deadline check: load + proxied classify can run past the 30 s timeout -> Lambda resets env, loaded model lost; DOWNLOAD_DEADLINE derivation stale (≈A4-4).
D2-5 lambda main.rs:164 — deterministic load failures (hash_mismatch/load/identity) retried from scratch on every request (846 MB download each) — cost amplification.
D2-6 = E4.

## Simplification
S1 = B1 (+ three pub names for one hash fn: pack::sha256_hex, artifact::artifact_sha256_hex, lambda sha256_hex).
S2 verify.rs:966 — check_inputs re-runs hash checks from_run_dir already made on the same bytes (hash_matches copy of pack::check_hash, 2 unreachable arms, hash_file_name table).
S3 = R2/E5 (read_local_bounded; + lambda pub build_server forwards to its own alias).
S4 = R5/AL2 (policy copies).
S5 verify.rs:2294 — VerifyReport/GateFailure duplicate 11 evidence fields; deploy_eligible always true; RescoreStats.n unread.
S6 lambda lib.rs:615 — LoadOnce: `get_or_try_init(|| { performed = true; load() })` simpler; Arc redundant; STALE DOCS still describe the removed lock (main.rs:7-11 "behind a load lock ... leaves the lock re-armed", s3.rs:10-12 "hold the load lock", test name retry_after_failure_rearms_the_load_lock); load_log_line (true, None) arm unreachable; LoadTimeline.cpu_part unread; ModelSource::kind() no callers.
S7 task.rs:163 — Task.sha256 computed on every parse but read only by a test (dedup made task.rs import back-office pack); ModelIdentity.base_decl, BuiltRow.tokens, Laya.options, Laya::temperature() write-only/unused.
S8 verify.rs:1305 — NaN-propagating max hand-copied 3x, probs-row rule 2x, train-collision map 2x; working-tree alias `max_abs` adds a 2nd name rather than removing a copy (rename row_max_abs -> max_abs).

## Efficiency
EF1 = B1.
EF2 modernbert/load.rs:151 — F16->F32 widening per element via runtime fp16 detection + out-of-line call on aarch64-linux; use half's convert_to_f32_slice (+ rayon). est 0.5-1 s cold.
EF3 modernbert/load.rs:156 — finiteness scanned twice (rung 5 on F16 bits, then widened f32 in load_tensor, provably redundant).
EF4 laya/mod.rs:501 — rows scored one full forward each (weights re-packed per row); batch row-independent ops or prepack.
EF5 modernbert/embeddings.rs:10 — full [50368 x 1024] embedding table widened to f32 (206 MB resident, +103 MB RSS) though only <=120 rows gathered.
EF6 artifact.rs:1010 — pack builds + probes the FT model twice (throwaway container + rung 6-7 re-run).
EF7 verify.rs:2383 — verify_path materializes all checkpoint tensors (0.85 GB) only to use checkpoint_sha256; peak ~3.4 GB.
EF8 verify.rs:937 — base safetensors read twice; hashed bytes != scored bytes (TOCTOU) (≈A1-1 part).

## C2 (Python<->Rust<->contract)
C2-1 verify.rs:1610 — Rust gate uses f32 macro-F1 (aprender-core f1_score -> f32) vs Python f64: at the inclusive margin boundary verdicts flip (constructed 9-row example: exact margin 1/20 -> Python PASS, Rust FAIL -> PassDisagrees; 724 flip pairs found); gate.py comment "same subtraction the Rust verifier does" false; per_seed.pass flip refuses whole run.
C2-2 = A1-3 + A1-6.
C2-3 = A1-5 (constructed ece 0.0679999937 py vs 0.0680000111 rust -> key 679 vs 680).
C2-4 contract.py:71/227 — PyYAML reads `1e-6` (no dot) as a string -> copied into recipe/gate-report as JSON strings -> Rust refuses Schema after a multi-hour run. Latent (no current value).
C2-5 verify.rs:1006 — NFC Unicode version skew: Rust unicode-normalization 0.1.25 (Unicode 17) vs Python 3.13.7 unicodedata 15.1 -> split-disjointness hashes differ for Unicode 16/17 composition pairs (U+113C2 x2 -> U+113C5).
C2-6 data.py:147 — invalid UTF-8 / lone surrogate rows -> uncaught UnicodeDecodeError/EncodeError traceback exit 1 (not REFUSED exit 2); Rust refuses typed.
C2-7 = AL7 (justfile:1387 `^SKIP:` misses `SKIP ladder rung:`).
C2-8 pack.rs:394 — rescore-noise sets[].t_applied parsed but never cross-checked vs calibration.t_applied in Rust (Python self-test checks it) -> noise from a different temperature inflates the bound up to 1e-3.

## A5 (line-by-line Python/just)
A5-1 metrics.py:64 — macro-F1 averages over y ∪ pred labels; nothing refuses an eval.jsonl missing a criterion -> margin decided by label coverage (one-row change passes 0.05 margin).
A5-2 = C2-1 (30-row example: exact 0.05 margin -> Python FAIL(0.04999999999999993), Rust PASS -> PassDisagrees; 11/33 constructed pairs disagree).
A5-3 = A1-5 (real demo_s64 seeds: delta up to 2.7e-8).
A5-4 train.py:871 — inputs_sha256 + task.json copy taken from data dir at END of the multi-hour run, not over the bytes read at start (TOCTOU).
A5-5 justfile:1775 — laya-deploy never checks env vs config bucket; mismatch refused only after going live (then contained).
A5-6 gate.py:124 — EarlyStopper compares to value at last improving epoch; contract formula uses running min of ALL prior epochs -> different restored epoch.
A5-7 = AL7.

## A3 (line-by-line modernbert)
A3-1 = C1-1 (measured: 4.6e18 -> SIGABRT; 5e6 layers -> 3.54 GB RSS in name set).
A3-2 = D1-8 (measured: silu + yarn/linear rope_type configs load Ok).
A3-3 modernbert/layer.rs:304 — public ModernBertLayer::forward allocates RoPE table from caller's `l` BEFORE checking x.len()==l*d -> huge l OOM/SIGKILL (debug overflow panic) instead of typed InputShape.
A3-4 modernbert/load.rs:151 — load_tensor accepts any dtype get_tensor_as_f32 widens; AprQ4 zero-pads truncated data and zeroes NaN scales -> truncated/non-finite Q4 weights load as zeros (core public API; decide shielded by F16-only rung).
A3-5 apr-format v2/tests.rs:497 — forged_tensor_count regression test cannot fail on macOS (309 GB reservation succeeds lazily) -> green with the fix reverted.

## D3 (Python/shell pitfalls)
D3-1 = AL7 (+ RUST_TEST_THREADS=1 prefixes "test x ... SKIP:" -> `^SKIP:` catches nothing; laya-resolver-proof sed `^RESOLVED` same column-0 assumption).
D3-2 = A1-5. D3-3 = C2-6.
D3-4 justfile:1545/1754/2296 — sha256() helper uses `rtk proxy shasum` whenever any `rtk` is on PATH: deploy identity pin depends on a dev proxy (name collision -> recipes abort). rtk hook never rewrites commands inside recipes, so the proxy adds nothing.
D3-5 prepare_stance.py:154 — .DS_Store in out dir makes a byte-identical re-run fail "differs" (also tree_sha256 in checkpoint dirs). Low.

## Conventions
CV1 Makefile:2658/2647 — contract-audit-phase8 guard EREs ship no must-match/must-not-match case table (CLAUDE.md VD#7); doc comments / cfg(test) helpers / justfile variables RESOLVE as definitions.
CV2 crates/aprender-decide/Cargo.toml etc — new violators of CI-required guards: aprender-decide publishable but not in cascade-publish.sh TIERS; aprender-mcp-decide hand-rolled argv; 4th `bootstrap` bin not allowlisted (guards already red at base).
CV3 Makefile:2614 — contract-audit-phase8 re-implements `pv verify-bindings` in awk/grep (CLAUDE.md "Never work around pv with a shell script"); pv --verify-bindings reports 6/13 (laya-train-selftest ghost).
CV4 justfile:2366 — `for NAME in $(aws iam list-role-policies ...)` hides the aws failure under set -e -> IAM read error reported as missing grant -> laya-deploy `contain` throttles a healthy deploy; justfile:2593 delete-role-policy failure (2>/dev/null) reported as "no legacy policy (expected)".
CV5 CLAUDE.md:268 — "safetensors ... never by a server" asserted, not enforced (normal dep; pack is pub in crates the servers link).
CV6 artifact.rs:1312 etc — 12 new fns exceed max_complexity 10 (.pmat-gates.toml).
