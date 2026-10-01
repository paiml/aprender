---
phase: 04
slug: apr-artifact-and-production-parity
status: verified
threats_open: 0
threats_total: 97
threats_closed: 89
threats_accepted: 8
asvs_level: 1
block_on: high
block_on_tripped: false
created: 2026-08-17T00:07:15Z
audited_at_commit: f37a0607bcd3404eb2b0492039b82cf9019471a6
register_authored_at_plan_time: true
audit_mode: verify-mitigations
run_by: gsd-security-auditor
note: >-
  Register origin is plan-time: 21 of 22 plans authored a <threat_model> block, so the
  auditor verified mitigations rather than building a retroactive STRIDE register. Plan
  04-17 is the sole exception — it shipped with no threat model and left T-04-65..69
  unallocated; those five IDs were authored during this audit and four of the five were
  already mitigated in shipped code. A further five IDs (T-04-92..96) were authored for
  review findings that had a security dimension and no plan-time row. No declared
  mitigation was found ABSENT. Three plan-time rows are PARTIAL and, with the five newly
  registered rows, are accepted with rationale below; none reaches high severity, so
  block_on: high was not tripped.
row_provenance:
  plan_time: 87        # 86 numbered + the shared T-04-SC, across 21 <threat_model> blocks
  authored_04_17_gap: 5   # T-04-65..69
  authored_review_findings: 5   # T-04-92..96 (IN-04, WR-03, WR-05, WR-06, IN-02)
---

# Phase 04 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
> Audited at `f37a0607b` in verify-mitigations mode against 92 rows across 4 crates,
> `contracts/`, `Makefile` and `scripts/`.

**Evidence rule for this document.** Every status claim below cites a `file:line` or a
command with its real exit status. No row is marked CLOSED on the strength of a
SUMMARY.md assertion — every SUMMARY in this phase asserts its own threats mitigated, and
that assertion is what was audited, not evidence for it. Where a mitigation is partially
present it is recorded as PARTIAL rather than rounded in either direction.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| hostile artifact bytes → core loader | The primary Phase 4 attack surface. Every rung of `read_setfit_apr_parts` is a STRIDE mitigation. | Untrusted APR container: header, declared index, encoder + head tensors, embedded JSON sub-documents |
| hostile file → caller allocation | The size bound must precede the read, not follow it. | Declared length (`u32`/`u64`), attacker-controlled |
| network client → `POST /v1/classify` | Untrusted request bodies. Two independent bounds: `MAX_REQUEST_BODY_BYTES` bounds the PARSE, `MAX_BATCH_TEXTS` bounds the WORK. Neither subsumes the other. | Batch of caller texts, arbitrary length and content |
| orchestrator / k8s probe → `GET /health/ready` | Readiness carries `classifier_artifact_sha256` + `classifier_verified`, which a deployment uses to decide WHICH artifact answered. | Model identity (hash), verification state |
| user filesystem / config → trainer | Untrusted paths and config become validated typed values here. | Dataset paths, TOML config, CLI overrides |
| lock file bytes → lock object | Untrusted filesystem content becomes the record that gates canonical-test access (TRN-07 / D-16). | `SelectionLock`: artifact hash, dataset fingerprint, selection semantic + ledger hashes |
| eval → canonical test data | The leakage boundary. Test split reachable only via a PRIOR durable lock → mint → grant. | Canonical test corpus |
| execution layer → `ClassifyResponse.backend` | The identity string reaches every API, CLI and HTTP consumer as a claim about which kernel ran. | Backend identity string |
| contract registry → auditor / gate | `contracts/aprender/binding.yaml` is what a human or gate reads to decide whether an equation is implemented. A row naming a nonexistent symbol while claiming `implemented` is a false attestation. | Equation → symbol bindings |
| gate surface (`Makefile`, `scripts/`) → developer trust | Every SAFE-01/SAFE-02 claim is mediated by these files. A gate that cannot fail is an authenticity defect one level above the code. | Exit statuses, lint reports |
| shell script → filesystem via hardcoded `/tmp` | Predictable temp paths in a world-writable directory are a symlink-attack surface. 24 measured SEC013 findings. | Temp file contents |
| operator-supplied path → `apr inspect` / `predict` / `eval` | The APR container is untrusted input; its 64-byte header including `u32 metadata_size` is attacker-controlled. | Header, metadata block |

---

## Threat Register

97 rows: 87 authored at plan time (86 numbered + the shared `T-04-SC`), 5 authored during
this audit for the 04-17 gap (`T-04-65..69`), and 5 authored for review findings that had
a security dimension and no plan-time row (`T-04-92..96`).

**89 CLOSED · 8 ACCEPTED with rationale · 0 OPEN · 0 absent.** Of the 89 closed, 81 are
closed by a mitigation verified in shipped source and 8 are closed by disposition (an
`accept` whose rationale was re-verified at HEAD).

### 04-01 — contract

| Threat ID | Category | Component | Disposition | Mitigation as verified | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| T-04-01 | Tampering | `contracts/setfit-apr-v1.yaml` | mitigate | `Makefile:1826` `PHASE4_CONTRACTS`; `:1786` in `$(CONTRACTS)`; `contract-audit-phase4` `:2010`, wired **blocking** in tier3 | closed |
| T-04-02 | Repudiation | tolerance provenance | mitigate | `setfit-apr-v1.yaml:765,773,775` — tolerances cited from `setfit-encoder-conformance-v1`, not re-derived | closed |
| T-04-37 | Repudiation | backend identity grammar | mitigate | contract `:864-871` forbids `Backend::AVX2`/`NEON`/`select_backend*`; records the size-dispatch evidence | closed |
| T-04-38 | Spoofing | head smuggled as metadata | mitigate | contract `:108,:113`; `artifact.rs:1202-1203` head in derived expected set; shape gate `:2384-2435`; negative `:4846-4851` | closed |
| T-04-58 | DoS | over-refusing null guard | mitigate | contract `:230-266` = 4 paths; `artifact.rs:184-200`; completeness gate `bundle_tests.rs:1395-1470` walks all five against the SHIPPED allowlist | closed |
| T-04-SC | Tampering | package installs | accept | **Measured**: `git diff d66678e7a..HEAD -- Cargo.lock` rc=0 → 1 insertion (`+ "toml 0.8.23"`), zero `+name = ` lines ⇒ zero new `[[package]]` | closed |

### 04-02 — writer

| Threat ID | Category | Component | Disposition | Mitigation as verified | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| T-04-03 | Tampering | metadata determinism | mitigate | `artifact.rs:1115-1116` one custom key; `:1123` `created_at: None`; BTreeMap `:38`; cross-process test `:4190` | closed |
| T-04-04 | Info Disclosure | probe records | mitigate | `artifact.rs:222-245` six synthetic strings; `:3685` asserts contract-resident inputs only; rung-7 refusal `:5262` | closed |
| T-04-05 | Tampering | non-finite float → null JSON | mitigate | `scan_view_floats:947-975`; `guard_subdocument_nulls:1338`; `observed_null_paths:1306`; typed `NonFiniteValue:469` | closed |
| T-04-64 | Tampering | unwalked sub-document | mitigate | `WALKED_SUBDOCUMENTS:130-136` includes `resolved_config` + `provenance`; counted REJECT test `:3916` | closed |

### 04-03 — loader ladder

| Threat ID | Category | Component | Disposition | Mitigation as verified | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| T-04-06 | DoS | allocation bomb | mitigate | `artifact.rs:1926-1934` declared-length refusal **before the reader is touched**; `:1943` `take(cap+1)`; `:1941` reservation clamped; rung2 `:2056`. Residual I-01 noted: `min(declared,cap)` can pre-reserve 256 MiB | closed |
| T-04-07 | Tampering | corrupted-but-CRC-valid | mitigate | `:2001` SHA-256 identity; `rung7_rebuild:2651`; `rung8_replay_probes:2720` | closed |
| T-04-08 | Spoofing | foreign APR masquerading | mitigate | `rung4_document:2153-2199` tag → one custom key → `schema`/`schema_version` **before** typed parse; `deny_unknown_fields` ×4 | closed |
| T-04-09 | Tampering | NaN weight poisoning | mitigate | `rung6_finite_payloads:2562-2590` scans tensors `:2579` + intercepts `:2580` + probe expectations `:2581`; NaN-visible comparator `:5302` | closed |
| T-04-10 | EoP | classify without verification | mitigate | private fields `:1732-1743`; derive-list guard `:5446-5459` bans `Default`/`Deserialize`/`Clone`; out-of-crate trybuild `setfit_verified_model_constructed.rs` | closed |
| T-04-44 | Spoofing | headless artifact reaching classify | mitigate | `check_head_shapes:2384-2435` vs `ordered_labels`/`head.n_features`, called at rung5 `:2240` | closed |

### 04-04 — classify envelope

| Threat ID | Category | Component | Disposition | Mitigation as verified | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| T-04-11 | DoS | unbounded batch | mitigate | `classify.rs:49` `MAX_BATCH_TEXTS=256`; typed `BatchTooLarge:624` | closed |
| T-04-12 | Repudiation | backend misreporting | mitigate | `classify.rs:647` identity from `encode_batch_traced`; no-ctor/no-setter guard `:1741`; no-capability-detection guard `:1775` | closed |
| T-04-13 | Tampering | NaN → null in JSON | mitigate | private fields `:404-410`; `#[serde(into, try_from)]` `:235,:403`; validating `TryFrom:378` | closed |
| T-04-59 | Repudiation | a gate whose filter selects nothing | mitigate | modules `envelope:864`, `backend:1434`, `classify_path:1849`; filter + floor `Makefile:2330-2335` (floor 45) | closed |

### 04-05 — codec

| Threat ID | Category | Component | Disposition | Mitigation as verified | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| T-04-14 | Spoofing | codec reporting a foreign payload as its own | mitigate | trusted `decode` re-checks format id `verify.rs:266-275` (both checks kept — the redundancy is load-bearing) | closed |
| T-04-15 | Repudiation | codec hashing its own output | mitigate | `artifact_hash` stays a trusted free fn `verify.rs:244-246`; codec's only `Sha256` use is inside `#[cfg(test)] mod fixture` (`apr_codec.rs:359`) | closed |
| T-04-16 | Tampering | nondeterministic serialize breaking closure | mitigate | `apr_codec.rs:964` double-closure + two-writes-identical; plus `artifact.rs:4190` cross-process | closed |
| T-04-45 | Repudiation | silently defaulted bundle field | mitigate | 20-field per-field bijection `:886`; defaulted-field negative `:1252` names the field | closed |

### 04-06 — `apr setfit train`

| Threat ID | Category | Component | Disposition | Mitigation as verified | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| T-04-17 | Tampering | config injection | mitigate | `config.rs:770,797` `deny_unknown_fields`; `setfit_train.rs:307-314` merge → `to_request` → `SetFitTrainConfig::new` | closed |
| T-04-18 | Repudiation | device silently falling back | mitigate | `resolve_requested_device:350-366` called `:616` **before** ingest `:621`; typed `CudaNotAvailable`, no silent fallback | closed |
| T-04-19 | Tampering | partial / clobbered output | mitigate | `atomic_write:176-188`; `refuse_existing_output:226-233`; exactly one `fs::rename` (`:183`) | closed |
| T-04-50 | DoS | hostile artifact read unbounded by a CLI adapter | mitigate | `setfit_io.rs:67-91` stat → open → core's bounded door. `fs::read(` count in `predict.rs`/`inspect.rs`/`eval/setfit.rs`/`setfit_io.rs` = 0,0,0,0 | closed |

### 04-07 — generic consumers

| Threat ID | Category | Component | Disposition | Mitigation as verified | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| T-04-20 | Spoofing | untagged APR treated as SetFit | mitigate | `setfit_tag.rs:194-198` typed-tag only; `artifact.rs:2156-2161` | closed |
| T-04-21 | Info Disclosure | canonical test leakage via eval | mitigate | chain `eval/setfit.rs:305 → 355 → 415 → 424`; no-bypass source scan `eval/setfit_tests.rs:513-545` drives 5 doors, forbids 5 forged types, non-vacuity `:507` | closed |
| T-04-22 | Tampering | corrupted artifact reaching prediction | mitigate | Full ladder on every **prediction-bearing** surface: `predict.rs:173`, eval via `apr_reload.rs:325`, serve `handlers.rs:1438`. `apr inspect` deliberately renders metadata without the ladder, bounded instead by T-04-70's cap | closed (scoped) |
| T-04-52 | Tampering | edited lock granting wrong-model access | mitigate | `lock.rs:604-607` `StaleLock`; `:801-812` `TokenModelMismatch` + `TokenDatasetMismatch` | closed |

### 04-08 — serve

| Threat ID | Category | Component | Disposition | Mitigation as verified | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| T-04-23 | DoS | unbounded HTTP batch / body | mitigate | `setfit_handlers.rs:152` batch; `router.rs:137-138` `DefaultBodyLimit::max(classify_body_limit_bytes())`; core re-check `classify.rs:624` | closed |
| T-04-24 | Spoofing | serving an unverified model | mitigate | `handlers.rs:1435` bounded read → `:1438` `load_setfit_apr` → `:1468` slot; slot type is `Option<Arc<VerifiedSetFitModel>>` | closed |
| T-04-25 | Repudiation | readiness reporting a hash the model lacks | mitigate | `router.rs:248-252` hash read off the loaded model object | closed |
| T-04-26 | Info Disclosure | auth gap on `api::router` paths | **accept** | Pre-existing, user-deferred (CONTEXT). Rationale re-verified: `grep auth\|AuthGate\|bearer router.rs` → **rc=1 (absent)**. See Accepted Risks | closed by disposition |

### 04-09 — three-surface parity

| Threat ID | Category | Component | Disposition | Mitigation as verified | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| T-04-27 | Repudiation | vacuous parity gate | mitigate | `setfit_parity.rs:1597-1604` in-band skewed negative through the public validating constructor | closed |
| T-04-28 | Tampering | golden drift | mitigate | `:1275-1340` SHA-256 manifest + flipped-byte negative | closed |
| T-04-29 | Spoofing | stale / shadowed `apr` binary | mitigate | `:615,:2038` `CARGO_BIN_EXE_apr`; zero PATH lookups; rationale `:1118` | closed |
| T-04-53 | Repudiation | surfaces compared on different inputs | mitigate | `:33` one `ClassifyRequestDocument`; `:472-478` per-leg input fidelity | closed |
| T-04-61 | Repudiation | a SAFE-02 run leg proving less than it reads | mitigate | `apr-cli/Cargo.toml:306-311`; `Makefile:502-504` | closed |

### 04-10 — gates

| Threat ID | Category | Component | Disposition | Mitigation as verified | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| T-04-30 | Repudiation | zero-match filters exiting 0 | mitigate | `assert_tests_ran` `Makefile:2055-2064`, **43 call sites** | closed |
| T-04-31 | Repudiation | rc read through a pipe | mitigate | **31** direct `rc=$$?` captures in the setfit recipe range; **zero** `\| tee` status reads (CLAUDE.md rule 1) | closed |
| T-04-32 | Spoofing | boundary grep matching nothing forever | mitigate | MUST-MATCH control `Makefile:2650-2661` fails if the pattern reads 0 (CLAUDE.md rule 7) | closed |
| T-04-56 | Repudiation | a recipe that cannot run what it names | mitigate | one positional filter per invocation, e.g. `:2330 --lib setfit::classify::` | closed |
| T-04-57 | Tampering | undeclared contract edit | mitigate | `git log -- contracts/setfit-apr-v1.yaml` → **one commit** `488e307d5` in its whole history | closed |
| T-04-62 | Repudiation | cited evidence with no guarded target | mitigate | `setfit-config-tests:2351`, `setfit-evaluate-tests:2364` | closed |

### 04-11 — CI proposal + audit

| Threat ID | Category | Component | Disposition | Mitigation as verified | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| T-04-33 | EoP | autonomous CI edit | mitigate | Proposal is a patch FILE `04-11-ci-setfit.patch`; `ci.yml` has exactly one Phase-4 commit `57f7823ab`, **after** the human ruling | closed |
| T-04-34 | Repudiation | unearned requirement checkboxes | mitigate | `04-11-SUMMARY.md:96-99` difflib diff vs the real file (zero differences); `:339` seven amendments; `:462` 24-row accounting | closed |
| **T-04-35** | Tampering | mutation gate scoped to nothing | mitigate | **PARTIAL** — mechanism correct for `aprender-serve` only (99 mutants listed, `baseline: ok`, 593 s). Three crates never run; no aggregate score. See Accepted Risks | **accepted** |
| T-04-63 | Repudiation | tier-covered surface silently absent from CI | mitigate | `ci.yml:305-321` accounts for every 04-10 target by inclusion or written rationale | closed |

### 04-12 / 04-13 / 04-14 / 04-15

| Threat ID | Category | Component | Disposition | Mitigation as verified | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| T-04-36 | Repudiation | public-API-only lifecycle claim while CLI types leak | mitigate | `setfit_apr_lifecycle.rs:568-610` with a **positive control** `:583-588` and self-non-vacuity `:602-610`; `make setfit-api-boundary` | closed |
| T-04-51 | Repudiation | envelope forcing consumers to reach for fields | mitigate | The missing accessor was surfaced as `04-12-BLOCKED.md`, **not** fixed by widening visibility | closed |
| T-04-39 | Spoofing | caller-supplied provenance | mitigate | `bundle.rs:539-547` takes `&Selection`; no `String` provenance parameter exists | closed |
| T-04-40 | Tampering | silent schema drift | mitigate | `BUNDLE_SCHEMA_VERSION=2` `:106`; typed `UnsupportedSchemaVersion` `:655-656` | closed |
| T-04-41 | DoS | oversized lock payload | mitigate | `MAX_SELECTION_LOCK_BYTES` `lock.rs:70`; bound on the **raw slice** `:467-473`, step (1) before serde at step (2) | closed |
| T-04-42 | Tampering | edited lock file | mitigate | rebuild-through-constructor `:508-513`; canonical-bytes equality `:536-540`; doc `:419-429` refuses to overclaim | closed |
| T-04-43 | EoP | override bypassing validation | mitigate | `config.rs:648` returns a REQUEST; `pub fn set_*` / `*_mut(` count = **0** | closed |
| T-04-54 | Repudiation | passes unit-by-unit, fails end to end | mitigate | `setfit_cli_lifecycle.rs` — 8 `CARGO_BIN_EXE_apr` spawns chained on each other's files | closed |
| T-04-55 | Repudiation | untested compatibility claim (A3) | mitigate | Generic tools executed: `tensors:1241`, `inspect:1272`, `qa:1291,:1330`; the one nonzero verdict recorded, not suppressed | closed |

### 04-16 — fresh-process reload door

| Threat ID | Category | Component | Disposition | Mitigation as verified | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| T-04-46 | EoP | a second minting path with weaker evidence | mitigate | **Closed by substitution.** The declared "re-enters `verify_artifact`" did not ship — rustc-refused (`E0063`/`E0451`), written up `apr_reload.rs:13-31` + `04-16-BLOCKED.md`. Shipped equivalent: `load_setfit_apr` `:325` is the only minting path; `apr_reload_module_fabricates_no_evidence_and_mints_no_state` `apr_reload_tests.rs:476` | closed |
| T-04-47 | Spoofing | reloading against a dataset/selection it was not trained on | mitigate | Three ordered gates `apr_reload.rs:347-351 / 360-365 / 373-377`, each naming both values | closed |
| T-04-48 | Tampering | an artifact that reloads into something else | mitigate | **Closed by substitution.** The declared re-hash gate is deliberately absent; measured basis: `artifact.rs:2001` sets `artifact_sha256` over the *input slice*, so the gate would compare a value with itself. Rationale `apr_reload.rs:56-65` | closed |
| T-04-49 | Repudiation | restating the recorded device as this host's | mitigate | `grep device apr_reload.rs` → no output; `credential.rs` → 0. No device claim restated | closed |
| T-04-60 | Repudiation | a check declined on a false rationale | mitigate | `apr_reload_the_ledger_rationale_is_the_true_one` `apr_reload_tests.rs:558-575` asserts the false rationale ABSENT and the `Selection::replay` citation present | closed |

### 04-17 — **register gap filled during this audit** (T-04-65…69)

Plan 04-17 shipped with no `<threat_model>` block. Two new public doors landed:
`SetFitRun::<ArtifactReloadedAndVerified>::into_artifact_bytes` and the sealed
`pub trait SetFitCredential`. Four of the five threats below were **already mitigated in
shipped code** — the gap was in the record, not the implementation.

| Threat ID | Category | Component | Disposition | Mitigation as verified | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| **T-04-65** | Info Disclosure | `RetainedArtifactBytes` `Debug` claim | mitigate | **PARTIAL** — the newtype is correct (`mod.rs:404-410` prints length only, no accessor), but `mod.rs:526-534` `#[derive(Debug)] SetFitRun` still renders `encoder` + `dataset`. The doc at `:388-399` claims more than holds. Review finding IN-05, confirmed. See Accepted Risks | **accepted** |
| T-04-66 | EoP | `into_artifact_bytes` reachability | mitigate | `impl SetFitRun<ArtifactReloadedAndVerified>` `mod.rs:1006-1013`; that state is producible only by `verify_artifact` or the sealed reload door | closed |
| T-04-67 | Tampering | door returns re-serialized bytes | mitigate | `mod.rs:1011` moves `self.evidence.artifact_bytes.0`; `verify_into_artifact_bytes_are_the_hashed_bytes` `verify_tests.rs:648` **re-hashes** and compares to `artifact_hash()`; surface guard `:605-631` pins exactly one `pub fn` and forbids a third `impl` block | closed |
| T-04-68 | Spoofing | forged `SetFitCredential` | mitigate | `credential.rs:84` `pub trait SetFitCredential: sealed::Sealed`, `mod sealed` private `:65-68`; trybuild `tests/ui/setfit_external_credential_impl.rs` + committed `.stderr` (`E0277`); implementor count pinned at **2** `credential_tests.rs:252-256` | closed |
| T-04-69 | DoS | ~1.8 MB buffer retained beside the live run | mitigate | Consuming `self` enforced by borrowck and asserted by exact signature `verify_tests.rs:619`; measured peak-RSS delta recorded `verify.rs:675-693` | closed |

### 04-18 / 04-19 / 04-20 — gap closure

| Threat ID | Category | Component | Disposition | Mitigation as verified | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| T-04-70 | DoS | `inspect.rs::read_metadata` ~4 GiB allocation | mitigate | `inspect.rs:651-656` applies `setfit_tag::MAX_TAG_METADATA_BYTES` **before** `vec![0u8; ..]` `:657`; shared constant (4 refs each file); boundary tests `inspect_tests.rs:453,:485`, `predict_tests.rs:112` | closed |
| T-04-71 | Spoofing / Repudiation | over-cap `Ok(None)` denying a tagged classifier | mitigate | `setfit_tag.rs:173-180` typed `InvalidFormat` naming the cap (was `Ok(None)`); `four_consumers_agree_about_one_over_cap_file` `setfit_tag_tests.rs:317` | closed |
| T-04-72 | Information Disclosure | the new over-cap error message | **accept** | `setfit_tag.rs:174-179` names declared size, cap, path; explicitly states the block was **not read** | closed by disposition |
| T-04-73 | Tampering | header checksum computed but unused | **accept** | Fact unchanged, **coordinates moved**: computed `inspect.rs:553`, stored `:563`; branch grep → **rc=1 (no branch anywhere)**. Plan cited `:521`, now `:553` after 04-18's insert | closed by disposition |
| T-04-74 | Repudiation | `binding.yaml` `backend_identity` ghost symbol | mitigate | `binding.yaml:1455-1459` now names `aprender::setfit::encoder` / `ExecutionBackend::identity`; resolution check `classify.rs:1659-1680`; ghost-path grep over `contracts/` → **rc=1** | closed |
| T-04-75 | Spoofing | `ExecutionBackend` identity value | mitigate | `classify.rs:1741`, `:1775` — registry and guards now name the same symbol | closed |
| T-04-76 | Tampering | the registry edit itself | mitigate | `setfit-apr-v1.yaml` still one commit in its whole history | closed |
| T-04-77 | Information Disclosure | — | **accept** | YAML registry + test module only; no data path, no input, no network | closed by disposition |
| T-04-78 | DoS | candidate sweep before an argument-time refusal | mitigate | Pre-flight `refuse_existing_lock` `eval/setfit.rs:224` sited **above** `read_attested_canonical` `:229` | closed |
| T-04-79 | Tampering | existing committed `SelectionLock` at `--lock-out` | mitigate | Three ordered checks `:224`, `:638`, `setfit_train.rs:177`. **Executed**: `cargo test -p apr-cli --features setfit --lib eval::setfit` → **rc=0, 20 passed**, incl. `write_lock_still_refuses_a_destination_that_appeared_mid_run` and `eval_setfit_writes_the_lock_atomically_through_exactly_one_rename` | closed |
| T-04-80 | Tampering | check → `fs::rename` window in `atomic_write` | **accept** | **Explicitly NOT fixed.** `setfit_train.rs:216-222` states the residual in-source: "`rename` replaces its destination unconditionally … closing it needs the destination taken with `O_CREAT\|O_EXCL`". The source does not mislead. This is review finding WR-01 | closed by disposition |
| T-04-81 | Information Disclosure | the refusal message | **accept** | `eval/setfit.rs:621-630` names only the path and `--force` | closed by disposition |

### 04-21 / 04-22 — gap closure

| Threat ID | Category | Component | Disposition | Mitigation as verified | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| T-04-82 | DoS | transport batch bound at `:152` | mitigate | At-the-bound test `setfit_handlers.rs:771-780` reads the bound from the exported constant and asserts the body sits under `classify_body_limit_bytes()` | closed |
| T-04-83 | Repudiation | a test NAME asserting boundary coverage | mitigate | One-over test kept `:727-735`; the at-the-bound test that earns the name added `:771` | closed |
| T-04-84 | Spoofing | readiness reporting no resident classifier | *mitigate if measured surviving* | **Closed by measurement, no test added.** The mutant's subject `has_setfit_model` was deleted at `b47acc4fe`; the successor `AppState::setfit_model -> None` measured **CAUGHT** (D-04-21-A). Writing a test for deleted code was correctly declined | closed |
| T-04-85 | EoP | `/v1/classify` auth and CORS layering | **accept** | Rationale **re-verified at HEAD** and now demonstrably true: `auth.rs:193-201` `layer_public_ops`; `apply_except_public_ops:204-215` exempts **only** `Method::OPTIONS` and `/health*`; `/v1/classify` still hits `apply`; applied `handlers.rs:1474` | closed by disposition |
| T-04-86 | Tampering | mutation-run vacuity | mitigate | `04-21-SUMMARY.md:75-80` — 21 mutants tested (3 missed, 5 caught, 13 unviable), `baseline: ok`, `--features setfit` engaged, with filter-engagement proof (unfiltered suite is 51-red, so `baseline ok` proves the filter) | closed |
| T-04-87 | Repudiation | `bashrs-lint-makefile` exiting 0 unconditionally | mitigate | `Makefile:1325` `bashrs make lint Makefile > log 2>&1; rc=$$?` — the `\|\| echo` is gone. Fails on missing tool `:1318-1323`, unparseable report `:1343-1348`, above baseline `:1350-1356` | closed |
| T-04-88 | Repudiation | reporting an unavailable check as passing | mitigate | `deferred-items.md` §8 — seven explicit NOT-RUN entries incl. `bashrs gate` marked `NOT RUN (vacuous)` | closed |
| **T-04-89** | Tampering | hardcoded `/tmp` in `scripts/` | **transfer** | **PARTIAL** — content is actionable (SEC013 ×24, SEC014 ×24, SEC020 ×4, SEC006 ×2, fix order, required shape, in `deferred-items.md` §4 Group A + §5) but the owner is a **role** on an unfiled ticket. See Accepted Risks | **accepted** |
| T-04-90 | DoS | wiring a permanently-red gate into a blocking tier | mitigate | `lint-scripts` **deliberately not wired** (tier3 comment +119); only `bashrs-scoped-lint` wired (+125), green today | closed |
| T-04-91 | Tampering | mechanical SC2086 fix over 180 sites | **accept, deferred** | `deferred-items.md` Group D requires a must-match / must-not-match table before merge (CLAUDE.md rule 7); word-splitting rationale intact | closed by disposition |

### Newly registered during this audit — review findings with a security dimension

These four had **no register row** at plan time. They are recorded here rather than left
in a review file nothing reads. All are accepted for Phase 4 with the rationale below.

| Threat ID | Category | Component | Disposition | Finding | Status |
|-----------|----------|-----------|-------------|---------|--------|
| **T-04-92** | DoS | `data_contrastive.rs:614-615` `read_selection_manifest` | **accept** | **IN-04 / unregistered surface.** Bare `fs::read` on an operator-supplied path — no stat, no cap. The only unbounded read left on the `apr setfit train` / `apr eval` ingest paths; this phase widened its visibility. Every other reader this phase touched is bounded (`setfit_io.rs:67`, `predict.rs`, `eval/setfit.rs:626`, `setfit_tag.rs:118`). T-04-50 is scoped to artifact + lock files and does not cover it. Severity Low-Medium under ASVS L1 (local allocation DoS on the operator's own path; no privilege or data crossing) | **accepted** |
| T-04-93 | Repudiation | `eval/setfit.rs:572` `config_hash` | accept | **WR-03.** `to_vec(..).unwrap_or_default()` fails open to `sha256("")`. `config_hash` is a published identity **label**, not a gate — `lock.rs:269-280` refuses duplicate `artifact_hash` but not duplicate `config_hash`. Marginal. 3-line fix (`Result<String>`) available | accepted |
| T-04-94 | Repudiation | `inspect.rs:671-673` `setfit_doc` | accept | **WR-05.** Set only when tag **and** custom key are both present, so a tagged artifact with a missing document renders as a healthy plain APR while `predict`/`serve` fail on it — diagnostic misdirection on the tool CLAUDE.md mandates as step 1. Same contradiction class as WR-08, which was closed | accepted |
| T-04-95 | Tampering | `eval/setfit.rs:532-536` label arity | accept | **WR-06.** `position()` → `None` scores as *wrong*, and `seen += 1` sits **inside** the `zip`, so the arity guard at `:539` structurally cannot fire on an over-long response. An artifact whose head grows a label the loader admits yields `accuracy = 0.000000` as a *measurement* rather than a typed refusal. Sibling `apr_evaluate.rs:245-275` refuses by name. Not an access-control defect — CR-01's label-map gate `:435-444` still fires first | accepted |
| T-04-96 | Spoofing | `lock.rs:606`, `:802`, `:809` lock-chain provenance | accept | **IN-02.** Re-checks `artifact_hash` (`:606`) and artifact + dataset (`:802,:809`), but never compares the lock's recorded `selection_semantic_hash`/`ledger_hash` against the presenting credential. Combined with `from_canonical_bytes`'s own admission (`:419-429`), a hand-written internally consistent lock can record selection provenance the artifact was never trained under. Bounded: `apr_reload.rs:347-377` gates the *credential* on all three against the caller's own `Selection`. 3-line fix available | accepted |

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-01 | T-04-SC | Zero new registry packages this phase — verified by measurement, not assertion: `git diff d66678e7a..HEAD -- Cargo.lock` rc=0 yields 1 insertion and zero `+name = ` lines. Nothing to audit. | User | 2026-08-17 |
| AR-02 | T-04-26 | Auth gap on `api::router` paths is **pre-existing** and was explicitly deferred by user decision in `04-CONTEXT.md`. Re-verified absent at HEAD (`grep` rc=1). Phase 4 inherits it via `POST /v1/classify` rather than introducing it, and bounds that route on both body bytes and batch size before any compute. **Scope note:** `apply_except_public_ops` exempts all `/health*`, and `/health/ready` now carries `classifier_artifact_sha256` (`router.rs:249-252`) — model identity added by 04-08, i.e. new since the "both were fully public before" argument was written. The exemption is correct for k8s admission; the newly disclosed value is folded into this acceptance explicitly rather than left implicit. | User | 2026-08-17 |
| AR-03 | T-04-72, T-04-77, T-04-81 | Error-message content. Each names only values the operator already supplied or can `stat` (path, declared size, cap, `--force`). T-04-72's refusal happens **before** the block is read, so no artifact content is available to echo. | User | 2026-08-17 |
| AR-04 | T-04-73 | `inspect.rs:553` computes `checksum_valid` and `:563` stores it; a branch grep returns rc=1 — nothing anywhere branches on it. Out of 04-18's scope and untouched by it. **The 16 MiB cap bounds resource use; it does not authenticate the header.** Belongs to whoever closes the APR-02 residual I-01. | User | 2026-08-17 |
| AR-05 | T-04-80 | **WR-01.** `fs::rename` (`setfit_train.rs:183`) replaces its destination unconditionally, so a file created during the temp write is destroyed without `--force`. 04-20 narrowed the window by refusing earlier and **explicitly did not claim to close it**; the residual is stated in-source at `:216-222`. Correct fix is taking the destination with `O_CREAT\|O_EXCL`. The source does not mislead a reader. | User | 2026-08-17 |
| AR-06 | T-04-85 | `/v1/classify` auth + CORS layering, closed by CR-02 and WR-07. Re-verified at HEAD: `auth.rs:204-215` exempts only `Method::OPTIONS` and `/health*`; `/v1/classify` still hits `apply`. The rationale is now demonstrably true, which it was not at plan time. | User | 2026-08-17 |
| AR-07 | T-04-91 | Mechanical SC2086 quoting over 180 sites changes word-splitting behaviour and can silently alter what a script does. Deferred in `deferred-items.md` Group D, gated on the fixer shipping a must-match / must-not-match case table — CLAUDE.md rule 7, which caught all five wrong `apr`-invocation patterns where review caught none. | User | 2026-08-17 |
| **AR-08** | **T-04-35** | **PARTIAL.** Per-crate mutation mechanism is correct and was executed for `aprender-serve` (99 mutants enumerated, `baseline: ok`, 593 s). `aprender-core`, `aprender-train` and `apr-cli` were never run and **no aggregate adjusted score exists**. D-04-11-B measured ≥10 h wall clock for all 890 mutants — beyond any single session. Already human-ruled *"Accepted, deferred → a standalone compute ticket"* at `04-VERIFICATION.md:520`. **A later reader must not infer a Phase 4 mutation score: there is none.** Severity Low — assurance coverage, not a code vulnerability. | User | 2026-08-17 |
| **AR-09** | **T-04-89** | **PARTIAL.** Transfer *content* is complete and actionable in `deferred-items.md` §4 Group A + §5 — counts (SEC013 ×24, SEC014 ×24, SEC020 ×4, SEC006 ×2), representative files, fix order, and the required per-site-judgement shape (no `sed`). What is missing is the **named owner**: §5 names a *role* ("whoever owns `scripts/` maintenance") on a ticket filed nowhere — no ID, no assignee. A transfer to an unidentifiable party is not a completed transfer. `git diff -- scripts/` is empty, so nothing regressed and the exposure is exactly as measured. Severity Medium **in dev tooling**, outside the shipped-product ASVS L1 surface. **To close: file a ticket and record its ID in `deferred-items.md` §5.** | User | 2026-08-17 |
| **AR-10** | **T-04-65** | **PARTIAL.** `RetainedArtifactBytes` renders `{ len }` only and carries no accessor (`mod.rs:404-410`) — the newtype is correct. But `mod.rs:526-534` `#[derive(Debug)] SetFitRun<S>` holds `encoder: SetFitMiniLm` and `dataset: PreparedDataset<Canonical>`, so `{:?}` on any run still renders every encoder tensor and the whole canonical corpus — orders of magnitude past the 1.82 MB the newtype removes. The doc at `mod.rs:388-399` scopes its claim to "`{:?}` on any `SetFitRun`", **which is broader than what holds**. Review finding IN-05, confirmed by measurement. Severity Low — requires a `{:?}` on a run reaching a log path. **To close: narrow the doc claim to `ArtifactVerifiedEvidence`, or hand-write `Debug` for `SetFitRun`.** | User | 2026-08-17 |
| **AR-11** | **T-04-92** | **IN-04, previously unregistered.** `data_contrastive.rs:614-615` `read_selection_manifest` does a bare `fs::read` on an operator-supplied path with no stat and no cap. Local allocation DoS on a path the operator supplied themselves; no privilege boundary and no data crossing, so it does not reach high severity under ASVS L1. Accepted for Phase 4 and registered so it stops being invisible. **To close: route it through a stat + cap like `setfit_io.rs:67-91`.** | User | 2026-08-17 |
| AR-12 | T-04-93, T-04-94, T-04-95, T-04-96 | WR-03, WR-05, WR-06 and IN-02 — four review findings with a genuine security dimension and no plan-time register row. All are Low or Marginal, all are bounded by a gate that fires first (see each row above), and each has a small named fix. Registered here so they are dispositioned rather than merely recorded in a review file. | User | 2026-08-17 |

**Not security-relevant — ruled on and closed without a register row:** WR-04
(`evaluate.rs:378-386`, error-taxonomy defect on a branch the source now documents as
unreachable from either caller); IN-01 (`preserve_order` grep → rc=1, so `Map` is a
`BTreeMap` and output is deterministic; the doc/behaviour mismatch is real but harmless
today); IN-06's unused `Write` import (`eval/setfit.rs:40`); IN-07 (`--force` without
`--lock-out` fails closed and does nothing dangerous).

**Escalated beyond this phase — project-level control gap (not a Phase 4 threat).**
`apr-cli/src/lib.rs:9-16` carries a blanket `#![allow(...)]` that disables
`clippy::disallowed_methods` crate-wide — i.e. **the `unwrap()` ban is not
machine-enforced anywhere in `apr-cli`**. This phase's zero-`unwrap()` compliance is
therefore by discipline, not by gate, and is the mechanism by which a Phase 4 control
could regress unobserved. Recorded as D-04-14-B; correctly out of Phase 4's scope.

---

## Register Corrections

Recorded so a future reader does not mis-audit these rows.

| # | Correction |
|---|------------|
| 1 | **04-17 had no `<threat_model>`.** T-04-65…69 were unallocated and are authored above. Four of the five were already mitigated in shipped code — the gap was in the record. `ROADMAP.md` also omits 04-17 entirely (verification W-02), so the phase's least-visible plan is invisible in two artifacts. |
| 2 | **T-04-46 and T-04-48 shipped a different mechanism than declared.** Both substitutions are sound — one rustc-witnessed (`E0063`/`E0451`), one measurement-backed — and both are written up in-source. But the register still names the superseded mechanism, so a reader grepping for `verify_artifact` in `apr_reload.rs` finds nothing and would wrongly call them OPEN. The verified mechanism is recorded in the table above. |
| 3 | **T-04-73's cited line moved**: `inspect.rs:521 → :553`, because 04-18 inserted the cap block above it. Nothing keyed on the number broke; the register did not follow. |
| 4 | **`04-VERIFICATION.md`'s WR-02 entry is stale, not wrong.** It measured a real defect at `0fb47958f`; the code was deleted at `b47acc4fe`. See the audit trail note below. |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Accepted | Run By |
|------------|---------------|--------|------|----------|--------|
| 2026-08-17 | 97 | 89 | 0 | 8 | gsd-security-auditor (verify-mitigations, ASVS L1, block_on: high) |

### Audit 2026-08-17 — notes

**Scope.** 87 plan-time rows across 21 `<threat_model>` blocks, + 5 authored for the 04-17
gap, + 5 authored for previously unregistered review findings. Audited at `f37a0607b`
(clean tree apart from untracked `.serena/`). Read-only over `crates/`, `contracts/`,
`Makefile`, `scripts/`.

**Result.** No declared mitigation was found **absent**. Every `mitigate` row resolved to a
mechanism read in shipped source. Two headline gates were *executed* rather than inferred:
`cargo test -p apr-cli --features setfit --lib eval::setfit` → rc=0, 20 passed; the
loader/parity suites → rc=0, 15 passed. Three plan-time rows are PARTIAL (AR-08, AR-09,
AR-10); five newly registered rows are accepted (AR-11, AR-12).
**No finding reaches high severity, so `block_on: high` was not tripped.**

**WR-02 contradiction — resolved. `04-VERIFICATION.md` is the stale record.**
`04-REVIEW.md` listed WR-02 under `closed_since_round_1`; `04-VERIFICATION.md` said it was
still present at `eval/setfit.rs:620-636`. Both were true when written. Measured three ways:

1. `git show 0fb47958f:crates/apr-cli/src/commands/eval/setfit.rs` **does** contain the
   hand-rolled writer — `.{name}.tmp` with no pid/ordinal, `.create(true).truncate(true)`.
   The verification measured a real thing at its own commit.
2. It was deleted at `b47acc4fe` (*"refactor(04): cleanup pass — one detector, one atomic
   writer"*). `git log 0fb47958f..HEAD -- eval/setfit.rs` → `9949f982f`, `b47acc4fe`.
3. At HEAD, `eval/setfit.rs:637-666` `write_lock` delegates to
   `setfit_train::atomic_write`, whose `temp_path` (`setfit_train.rs:132-141`) is
   `.{stem}.tmp.{pid}.{ordinal}` off a `static AtomicU64`, and whose `fill_and_sync`
   (`:144-151`) opens with **`.create_new(true)`** — O_EXCL. It neither follows a symlink
   to an existing target nor reuses a crashed run's leftover. The only `create(true)` and
   `.tmp` strings remaining in `eval/setfit.rs` are at `:660` and `:662`, **inside the
   comment describing the deleted code** — which is what a text-only re-measurement would
   have tripped on.

**No symlink-attack surface remains on the selection-lock write path.** What remains is
the separate, registered, accepted T-04-80/WR-01 (`fs::rename` replaces unconditionally),
whose residual is stated in-source at `setfit_train.rs:216-222`.

**Residual noted under T-04-06.** The rung-1 cap refuses an over-cap *declared* length
before the reader is touched, but the reservation is `min(declared, cap)`, so a declared
length at the cap pre-reserves 256 MiB. Bounded and not a bypass; recorded as APR-02
residual I-01 alongside T-04-73.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer) — 97/97
- [x] Accepted risks documented in Accepted Risks Log — AR-01 … AR-12
- [x] `threats_open: 0` confirmed — 81 closed by verified mitigation, 8 closed by re-verified disposition, 8 accepted with rationale
- [x] `status: verified` set in frontmatter
- [x] `block_on: high` not tripped — no finding reaches high severity under ASVS L1

**Approval:** verified 2026-08-17

### Open follow-ups (tracked, not blocking)

These do not block Phase 4. They are the concrete closure actions behind AR-08…AR-11.

1. **T-04-89 / AR-09** — file a ticket for the `scripts/` `/tmp` hardening and record its ID in `deferred-items.md` §5. Content is ready; only the owner is missing.
2. **T-04-92 / AR-11** — bound `read_selection_manifest` (`data_contrastive.rs:614`) through a stat + cap, matching `setfit_io.rs:67-91`.
3. **T-04-65 / AR-10** — narrow the `Debug` claim at `mod.rs:388-399`, or hand-write `Debug` for `SetFitRun`.
4. **T-04-35 / AR-08** — the standalone per-crate `cargo-mutants` compute ticket (≥10 h). Until it runs, **Phase 4 has no mutation score.**
