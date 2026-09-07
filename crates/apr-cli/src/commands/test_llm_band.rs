//! `apr test llm bench --band` — PERF-025: the §4.4 protocol, reachable.
//!
//! # The defect this closes
//!
//! PERF-024 landed the conformant measurement protocol — `llm::band::run_band`,
//! the §4.4.2 warmup/quiesce/termination rule, `perf_gate::window`'s drain
//! accounting, the §4.4.4 bootstrap — and PERF-026 landed the receipt producer
//! (`perf_gate::drain::BandInput`, `perf_gate::receipt::ReceiptInput`).
//! **Nothing called either of them.** On `50d2bc2bb`,
//! `git grep "ReceiptInput\|BandInput\|DerivedBand\|perf_gate::"` outside the
//! module returned rc=1: zero callers. `apr test llm bench` still ran
//! `LoadTest::run`, whose termination rule is "stop after `--duration`
//! seconds" — no minimum sample count, no warmup-then-quiesce, no drain
//! accounting, no §4.4.6 tokenization block.
//!
//! The other half of the same defect: `scripts/perf_gate.sh`'s real mode
//! (`--host/--phase/--workload/--receipt`) was invoked only as `--selftest`
//! (`ci.yml:1011`), because nothing in the repo could write the receipt its
//! real mode reads. So the repo simultaneously held a conformant protocol, a
//! conformant receipt producer and a gate — and could not produce one
//! conformant measurement. Both halves close here.
//!
//! # This is a MODE, not a second harness
//!
//! `scripts/check_no_competing_harnesses.sh` names `apr test llm bench` the
//! canonical entrypoint, at `BASELINE=0`, shrink-only. This extends that
//! subcommand and drives the same `LlmClient` the legacy mode drives, so
//! PERF-009's one-entrypoint rule holds by construction rather than by promise.
//!
//! # There is deliberately no flag that shrinks the window
//!
//! `BandConfig::relaxed` exists for unit tests and is unreachable from the CLI.
//! The `max(30, 8c)` sample floor and the 60 s wall-clock floor are the entire
//! difference between a load test and a measurement; a `--band-duration` would
//! be the shortest path back to a gate that cannot fail. `--replicates` is the
//! only knob, it defaults to §4.4.2's `N = 3`, and going below that is written
//! into the receipt directory as a stated violation rather than silently taken.
//!
//! # One receipt per replicate, and why
//!
//! `ReceiptInput.bands` holds **one entry per concurrency**, and
//! `perf_gate.sh`'s Arm A indexes it with `{b["concurrency"]: b for b in
//! bands}` — a Python dict comprehension. Emitting `N` bands that all say
//! `concurrency: 1` would therefore be silently reduced to whichever one landed
//! last: a receipt that looks like three measurements and is judged as one.
//! Collapsing the replicates into a "representative" band instead would mean
//! picking a median of three, which is exactly the estimator that reported
//! 2.91x for a cell whose real answer was 1.21x (#2567).
//!
//! So each replicate gets its own complete, independently judgeable receipt,
//! `receipt.r1.json` … `receipt.rN.json`. The cell's verdict is the conjunction
//! over them, which is a decision for the runner, not something this producer
//! should pre-collapse.

use crate::error::{CliError, Result};
use apr_test::llm::band::{run_band, BandRun, RequestExtra};
use apr_test::llm::client::{ChatRequest, LlmClient};
use apr_test::llm::{assert_prompt_tokens_in_band, PromptTokenBand};
use apr_test::perf_gate::protocol::{BandConfig, Outcome};
use apr_test::perf_gate::receipt::now_utc_millis;
use apr_test::perf_gate::{
    sha256_file, write_samples_gz, BandContext, BandInput, BatchInvariance, BatchInvarianceWitness,
    ClientIdentity, ComparatorIdentity, ComparatorStatus, ComputeClass, JoinKey, KvBlock, Ladder,
    Lane, LaneConfig, ModelIdentity, ProtocolParams, ProtocolSource, Provenance, ReceiptInput,
    RequestOutcome, RequestSample, RunId, SamplesFile, SlotsAdmitted, StreamMode, SubjectIdentity,
    TokenizationBlock, Workload, CLOCK_SOURCE_SYSTEM_REALTIME, REPLICATES, SCHEMA_VERSION,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use super::test_llm::{describe_workload, resolve_corpus};

/// Arguments for one §4.4-conformant band sweep.
///
/// A struct rather than a 20-argument function, for the reason `BenchArgs` is
/// one: positional arguments of the same type are where a caller transposes
/// `accelerator` and `quantization` and the receipt records a lie.
pub struct BandArgs<'a> {
    /// Endpoint under measurement.
    pub url: &'a str,
    /// Model name sent in the request body.
    pub model: &'a str,
    /// Concurrency levels, comma-separated, e.g. `1,4,8,16`.
    pub bands: &'a str,
    /// §4.4.2 replicates per cell.
    pub replicates: usize,
    /// Directory receiving the receipts and the gzipped JSONL samples.
    pub receipt: &'a Path,
    /// §4.3 workload identifier.
    pub workload: &'a str,
    /// Join key: which machine measured.
    pub host: &'a str,
    /// Join key: which accelerator served the request.
    pub accelerator: &'a str,
    /// Join key: which quantization the served model uses.
    pub quantization: &'a str,
    /// The dispatch path the SERVER took.
    pub compute_class: &'a str,
    /// The SERVER's build features.
    pub server_features: &'a [String],
    /// §4.4.6 counting method.
    pub tokenization: &'a str,
    /// §4.4.6 digest, required for `client_tokenizer`.
    pub tokenizer_sha256: Option<&'a str>,
    /// §4.4.6 — do the counts include special tokens?
    pub counts_special_tokens: bool,
    /// §4.4.6 — do the counts include the echoed prompt?
    pub counts_prompt_echo: bool,
    /// Commit under measurement. REQUIRED for `--band` (PP-21).
    pub commit: Option<&'a str>,
    /// Streaming responses. REQUIRED for `--band` (PP-27).
    pub stream: bool,
    /// Named prompt profile.
    pub profile: &'a str,
    /// Prompt file, overriding the profile.
    pub prompts: Option<&'a Path>,
    /// Who owes the comparator measurement this producer refuses to invent.
    pub comparator_owner: &'a str,
    /// §4.3 / PP-3 — the comparator lane's endpoint. When present, every band
    /// is measured on BOTH lanes inside one invocation, interleaved, and joined
    /// on a shared `run_id`.
    pub comparator_url: Option<&'a str>,
    /// The model name the comparator server expects in the request body.
    /// Defaults to [`BandArgs::model`]: most servers ignore the field, and
    /// `llama-server` serves whatever it was launched with.
    pub comparator_model: Option<&'a str>,
    /// PP-20 — the comparator build's upstream commit.
    pub comparator_commit: Option<&'a str>,
    /// PP-20 — the `cmake` line the comparator was configured with.
    pub comparator_cmake: Option<&'a str>,
    /// PP-20 — the comparator binary's digest, 64 lowercase hex.
    pub comparator_sha256: Option<&'a str>,
    /// PP-20 — the instant after which every ratio against this pin is
    /// `COMPARATOR_STALE`, as canonical UTC `YYYY-MM-DDTHH:MM:SS.mmmZ`.
    pub comparator_pin_expiry: Option<&'a str>,
    /// PP-21 — sign the receipt with this key id after writing it.
    pub key_id: Option<&'a str>,
    /// PP-21 — keyring passed through to `scripts/perf_receipt_sign.sh`.
    pub keyring: Option<&'a Path>,
    /// PP-26 — `witness.json` from `scripts/perf041_batched_parity_probe.py`.
    pub witness_json: Option<&'a Path>,
    /// PP-18 — the `apr serve` binary that served the bands, when it is not
    /// this same binary. Hashed into `provenance.subject`.
    pub subject_binary: Option<&'a Path>,

    // ------------------------------------------------------------------
    // PP-22 / §5.3 — the comparator lane's configuration, as DECLARED by
    // whoever launched it.
    //
    // llama.cpp's `GET /props` does not report `n_batch`, flash attention or
    // the KV cache type, so `lane_config_from_props` left all three `None` on
    // every real run. `JoinKey::refuse_mismatch` treats `None == None` as
    // agreement, so the join always succeeded — and `refuse_cripple`, which
    // asks whether `n_batch == Some(1)`, could NEVER fire against a real
    // llama-server. The `-b 1` cripple that manufactured a 2.39x overstatement
    // once (llama_pin.toml:129-165) was unrefusable in practice.
    //
    // The launcher knows the argv it used. These flags are that knowledge,
    // declared; `/props` OVERRIDES them wherever it does report a value, and a
    // disagreement between the two is refused outright rather than resolved.
    // ------------------------------------------------------------------
    /// §5.3 — `-b` the comparator was launched with. `1` is refused.
    pub comparator_n_batch: Option<u32>,
    /// §5.3 — per-slot context (`-c / -np`) the comparator was launched with.
    pub comparator_n_ctx_slot: Option<u32>,
    /// §5.3 — `-fa`: `on`, `off` or `auto`. `auto` records nothing, because the
    /// launcher does not know what the server resolved it to.
    pub comparator_fa: Option<&'a str>,
    /// §5.3 — `-ctk/-ctv`, e.g. `f16`.
    pub comparator_kv_type: Option<&'a str>,
}

/// PP-22 / §5.3 — the comparator lane as the LAUNCHER declared it.
///
/// # Errors
/// When `--comparator-fa` is not one of `on`, `off`, `auto`.
fn declared_lane_config(args: &BandArgs<'_>) -> Result<LaneConfig> {
    let fa = match args.comparator_fa {
        None | Some("auto") => None,
        Some("on") => Some(true),
        Some("off") => Some(false),
        Some(other) => {
            return Err(CliError::InvalidInput(format!(
                "--comparator-fa {other:?}: expected on, off or auto. `auto` records NOTHING,                  because a launcher that passed `-fa auto` does not know what the server                  resolved it to, and a guessed join-key field silently joins two different                  configurations."
            )))
        }
    };
    Ok(LaneConfig {
        n_ctx_slot: args.comparator_n_ctx_slot,
        kv_type: args.comparator_kv_type.map(ToString::to_string),
        fa,
        n_batch: args.comparator_n_batch,
    })
}

/// PP-22 / PP-2 — the declared lane and the lane's own `GET /props`, reconciled.
///
/// `/props` **wins** wherever it reports a value: it is the server's own
/// account of itself, and PP-2 asks for the configuration TAKEN. A declared
/// value the server contradicts is refused rather than overwritten, for the
/// same reason `reconcile_compute_class` refuses one: the disagreement usually
/// means the wrong server is being measured, and picking a winner hides that.
/// Where `/props` reports nothing, the declaration stands — that is the whole
/// point of the flags, since llama.cpp reports none of `n_batch`, `fa` or the
/// KV type.
fn reconcile_lane(declared: &LaneConfig, reported: &LaneConfig) -> Result<LaneConfig> {
    fn pick<T: PartialEq + std::fmt::Debug + Clone>(
        field: &str,
        declared: Option<&T>,
        reported: Option<&T>,
    ) -> Result<Option<T>> {
        match (declared, reported) {
            (Some(d), Some(r)) if d != r => Err(CliError::InvalidInput(format!(
                "--comparator-{field} declares {d:?} but the comparator's own GET /props reports                  {r:?} — PP-2 takes the server's report and refuses the contradiction rather                  than silently overwriting it: the two disagreeing usually means the flags                  describe a different server than the one being measured."
            ))),
            (_, Some(r)) => Ok(Some(r.clone())),
            (d, None) => Ok(d.cloned()),
        }
    }
    Ok(LaneConfig {
        n_ctx_slot: pick(
            "n-ctx-slot",
            declared.n_ctx_slot.as_ref(),
            reported.n_ctx_slot.as_ref(),
        )?,
        kv_type: pick(
            "kv-type",
            declared.kv_type.as_ref(),
            reported.kv_type.as_ref(),
        )?,
        fa: pick("fa", declared.fa.as_ref(), reported.fa.as_ref())?,
        n_batch: pick(
            "n-batch",
            declared.n_batch.as_ref(),
            reported.n_batch.as_ref(),
        )?,
    })
}

/// `1,4,8,16` into concurrency levels.
///
/// A zero or unparseable level is refused where it was typed.
/// `BandConfig::conformant` clamps `0` up to `1`, which would measure a
/// different band than the operator asked for and then label it with the number
/// they typed.
fn parse_bands(spec: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for raw in spec.split(',') {
        let token = raw.trim();
        if token.is_empty() {
            continue;
        }
        let c: usize = token.parse().map_err(|_| {
            CliError::InvalidInput(format!("--bands: {token:?} is not a concurrency level"))
        })?;
        if c == 0 {
            return Err(CliError::InvalidInput(
                "--bands: 0 is not a concurrency level; a band with no workers measures nothing"
                    .to_string(),
            ));
        }
        // `BandInput.concurrency` is a `u32`. Refusing the excess here keeps
        // that conversion infallible, so the receipt cannot end up labelled
        // with a concurrency the operator did not ask for -- the same
        // silent-substitution defect as the clamped zero above.
        if u32::try_from(c).is_err() {
            return Err(CliError::InvalidInput(format!(
                "--bands: {c} exceeds the receipt's u32 concurrency field"
            )));
        }
        out.push(c);
    }
    if out.is_empty() {
        return Err(CliError::InvalidInput(
            "--bands named no concurrency levels".to_string(),
        ));
    }
    Ok(out)
}

/// §4.4.6 — `method` has no default, so an unknown value is refused rather than
/// falling back to one of the two.
fn build_tokenization(args: &BandArgs<'_>) -> Result<TokenizationBlock> {
    let block = match args.tokenization {
        "server_usage" => TokenizationBlock::ServerUsage {
            counts_special_tokens: args.counts_special_tokens,
            counts_prompt_echo: args.counts_prompt_echo,
        },
        "client_tokenizer" => {
            let sha = args.tokenizer_sha256.ok_or_else(|| {
                CliError::InvalidInput(
                    "--tokenization client_tokenizer requires --tokenizer-sha256".to_string(),
                )
            })?;
            TokenizationBlock::ClientTokenizer {
                tokenizer_sha256: sha.to_string(),
                counts_special_tokens: args.counts_special_tokens,
                counts_prompt_echo: args.counts_prompt_echo,
            }
        }
        other => {
            return Err(CliError::InvalidInput(format!(
                "--tokenization {other:?}: expected server_usage or client_tokenizer (§4.4.6 \
                 gives `method` no default)"
            )))
        }
    };
    block.validate().map_err(CliError::InvalidInput)?;
    Ok(block)
}

/// PP-21 — the commit under measurement, refused where it was typed.
///
/// `--commit` used to be optional and defaulted to the literal `"UNPINNED"`.
/// A receipt is evidence about a build; PP-21's staleness arm asserts
/// `receipt.commit ⊇ commit-under-test`, and `"UNPINNED"` satisfies no such
/// containment while still rendering a complete, signed-looking document. The
/// default is gone, and a value that is not a full 40-hex object name is
/// refused before the sweep spends 20 minutes producing a receipt nothing can
/// judge.
fn require_commit(commit: Option<&str>) -> Result<String> {
    let commit = commit.ok_or_else(|| {
        CliError::InvalidInput(
            "--band requires --commit <40-hex>: PP-21's staleness arm asserts that the receipt's \
             commit contains the commit under test, and a receipt that does not name a build is \
             a declared-state artifact rather than evidence. Use `git rev-parse HEAD`."
                .to_string(),
        )
    })?;
    if commit.eq_ignore_ascii_case("UNPINNED") {
        return Err(CliError::InvalidInput(
            "--commit UNPINNED: that was this producer's old default and it is exactly the \
             fabrication PP-21 exists to stop — a receipt that renders, signs and reads as \
             evidence while naming no build. Pass the real object name."
                .to_string(),
        ));
    }
    if commit.len() != 40
        || !commit
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(CliError::InvalidInput(format!(
            "--commit {commit:?}: expected a 40-character lowercase hex object name. An \
             abbreviated sha cannot be tested for ancestry (PP-18) without guessing which \
             object it names."
        )));
    }
    Ok(commit.to_string())
}

/// PP-27 — `--band` without `--stream` measures a band it cannot report on.
///
/// A non-streaming band has no per-token arrival times at all, so `dec`,
/// `ttft` and `itl` are undefined; §5.1 states "streaming required". Both
/// committed evidence runs were taken without it (`invocation.txt` for lambda
/// and gx10 contain zero `--stream`), and the producer merely printed a NOTE
/// and carried on. A note is not a refusal.
fn require_stream(stream: bool) -> Result<()> {
    if stream {
        return Ok(());
    }
    Err(CliError::InvalidInput(
        "--band requires --stream: §5.1 makes streaming mandatory and PP-27 requires the \
         server's `stream_mode` declaration plus the client's ttft/e2e witness. Without \
         per-token arrival times dec, ttft and itl are UNDEFINED, so the band cannot report \
         two of the three metrics PP-4 requires of it. This used to be a printed NOTE, and \
         both committed evidence runs took it."
            .to_string(),
    ))
}

/// §5.1 / PP-33 — the protocol block for this run.
///
/// `window_ms`, `quiesce_ms`, `cooldown_ms`, `n_predict` and the sampler come
/// from `perf-matrix.yaml`; `replicates` and `interleaved` are what this
/// invocation ACTUALLY did. The matrix declares `replicates_min` — a floor —
/// and a receipt that copied the floor instead of the count it ran would claim
/// five replicates for a three-replicate run.
///
/// `interleaved` is `true` only when a comparator lane exists. §4.3's
/// interleaving is the alternation of two lanes; a single-lane run did not
/// alternate, and recording `true` from the matrix would put a protocol the
/// run did not follow on the wire. A single-lane band is therefore
/// `NONCONFORMANT-VALID` — a record, never a parity baseline — which is the
/// honest posture for a receipt with no comparator in it.
fn protocol_params(replicates: usize, interleaved: bool) -> ProtocolParams {
    protocol_params_with_source(replicates, interleaved).0
}

/// [`protocol_params`], with PP-33's provenance: did the block come from
/// `perf-matrix.yaml`, or from the compiled-in spec fallback and why?
///
/// `ProtocolParams::effective()` swallowed the error and nothing ever called
/// `source()`, so a checkout whose matrix did not parse put the Rust constants
/// on the wire under a `protocol:` block the receipt presented as the matrix's.
/// The producer now prints the answer before the first request and, on the
/// fallback, names it in `unproduced_fields`.
fn protocol_params_with_source(
    replicates: usize,
    interleaved: bool,
) -> (ProtocolParams, ProtocolSource) {
    let (effective, source) = ProtocolParams::effective_with_source();
    (
        ProtocolParams {
            replicates: u32::try_from(replicates).unwrap_or(u32::MAX),
            interleaved,
            ..effective
        },
        source,
    )
}

/// PP-2 — the facts the §5.2 endpoint reports, read out of its verbatim body.
///
/// Every field is `Option`: "the server did not report it" is a fact recorded
/// here rather than filled in. PP-13 makes a harness-computed substitute for a
/// server-reported field schema-fatal, so there is no default anywhere below.
#[derive(Debug, Default, PartialEq)]
struct ServerFacts {
    compute_class: Option<String>,
    build_features: Vec<String>,
    slots_admitted: Option<u32>,
    started_utc: Option<String>,
    build_commit: Option<String>,
    kv: Option<KvBlock>,
    model_file: Option<ModelIdentity>,
}

fn as_u32(v: Option<&Value>) -> Option<u32> {
    u32::try_from(v?.as_u64()?).ok()
}

fn as_u64(v: Option<&Value>) -> Option<u64> {
    v?.as_u64()
}

fn as_string(v: Option<&Value>) -> Option<String> {
    Some(v?.as_str()?.to_string())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// PMAT-973 / #2756 — the label and the sha256 of the prompt-corpus bytes
/// that earned it. A `Workload` recorded with no [`WorkloadBinding`] is a
/// declaration; one recorded WITH it is a claim the receipt can be checked
/// against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorkloadBinding {
    /// The `--workload` label this binding was checked against.
    pub(crate) label: Workload,
    /// sha256 of exactly the bytes read from `prompts_file`.
    pub(crate) corpus_sha256: String,
    /// Prompt records after the `_meta` header line.
    pub(crate) prompt_count: usize,
}

/// The first line's `_meta.corpus`, when the line parses and declares one.
///
/// Unrelated to [`resolve_corpus`]'s private `CorpusMeta` (which does not
/// even carry this field) — this is checked at the wire-format level, on
/// purpose: binding a label to a corpus must not depend on which fields a
/// deserializer happens to know about today.
fn parse_meta_corpus(first_line: &str) -> Option<String> {
    let value: Value = serde_json::from_str(first_line).ok()?;
    value
        .get("_meta")?
        .get("corpus")?
        .as_str()
        .map(str::to_string)
}

/// Bind `label` to the prompt file actually loaded, or refuse.
///
/// Accepted only when `prompts_file`'s own `_meta.corpus` equals `label`'s
/// wire token (§ DAG row I-25): a `--workload W1` run whose sent prompts are
/// not the labelled W1 corpus — `--profile short`'s one prompt, or a file
/// with no `_meta` header at all — is refused rather than silently recorded
/// under a name it did not earn.
///
/// # Errors
/// [`CliError::InvalidFormat`] when `prompts_file` cannot be read or hashed;
/// [`CliError::ValidationFailed`] when the file names a different corpus, or
/// names none.
pub(crate) fn bind_workload(label: Workload, prompts_file: &Path) -> Result<WorkloadBinding> {
    let label_token = label.wire_token();
    let content = std::fs::read_to_string(prompts_file).map_err(|e| {
        CliError::InvalidFormat(format!("prompt corpus {}: {e}", prompts_file.display()))
    })?;
    let mut lines = content.lines();
    let corpus = lines.next().and_then(parse_meta_corpus);
    match corpus.as_deref() {
        Some(actual) if actual == label_token => {}
        Some(actual) => {
            return Err(CliError::ValidationFailed(format!(
                "--workload {label_token}: {} is labelled {actual}, not {label_token}; a corpus \
                 label needs its corpus",
                prompts_file.display()
            )));
        }
        None => {
            return Err(CliError::ValidationFailed(format!(
                "--workload {label_token}: {} is not a labelled corpus (no `_meta.corpus` \
                 header); a corpus label needs its corpus",
                prompts_file.display()
            )));
        }
    }
    let corpus_sha256 = sha256_file(prompts_file)
        .map_err(|e| CliError::InvalidFormat(format!("hashing {}: {e}", prompts_file.display())))?;
    let prompt_count = lines.filter(|l| !l.trim().is_empty()).count();
    Ok(WorkloadBinding {
        label,
        corpus_sha256,
        prompt_count,
    })
}

/// False when `workload` is a corpus label (every [`Workload`] variant is)
/// but the receipt carries no `corpus_sha256` — a label the receipt did not
/// bind to the bytes it measured.
pub(crate) fn receipt_accepts_workload(_workload: Workload, corpus_sha256: Option<&str>) -> bool {
    corpus_sha256.is_some()
}

impl ServerFacts {
    /// Read `GET /v1/effective-config`. A body this harness cannot understand
    /// yields empty facts rather than an error: an older `apr serve` simply
    /// does not route the path, and the caller records that as a declared
    /// input.
    fn read(body: Option<&Value>) -> Self {
        let Some(body) = body else {
            return Self::default();
        };
        Self {
            compute_class: as_string(body.get("compute_class")),
            build_features: body
                .get("build_features")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(ToString::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            slots_admitted: as_u32(body.pointer("/scheduler/slots_admitted")),
            started_utc: as_string(body.pointer("/server/started_utc")),
            build_commit: as_string(body.pointer("/server/build_commit")),
            kv: Self::kv(body),
            model_file: Self::model_file(body),
        }
    }

    /// Arm D's memory block, built when the server reported the two BYTE
    /// figures. The two counters may be absent.
    ///
    /// `apr serve` reports `admission_rejected` and `preempted_swap` as `null`:
    /// it has no KV-admission refusal path and no swap path, so there is no
    /// quantity for either to denote (`effective_config.rs`'s `KvReport` says
    /// so in its own doc). While all four were required the whole block was
    /// dropped — including the two figures the server DID report — so `kv`
    /// could never be produced at all and Arm D was permanently blind.
    ///
    /// A null counter stays null: [`KvBlock`] carries it as `None` and
    /// [`ReceiptInput::render`] names it in `unproduced_fields`. Substituting
    /// `0` would be the fabrication the previous comment warned about, because
    /// Arm D reads `admission_rejected > 0` as evidence and cannot tell a
    /// counted zero from an uncounted one.
    fn kv(body: &Value) -> Option<KvBlock> {
        Some(KvBlock::from_server_report(
            as_u64(body.pointer("/kv/bytes_used"))?,
            as_u64(body.pointer("/kv/bytes_reserved"))?,
            as_u64(body.pointer("/kv/admission_rejected")),
            as_u64(body.pointer("/kv/preempted_swap")),
        ))
    }

    /// PP-23's input. The digest must be the 64 lowercase hex characters
    /// `Provenance::validate` demands: a server reporting a content hash in
    /// some other encoding is reported as absent rather than reshaped, because
    /// a reshaped digest is one that cannot be checked against the file.
    fn model_file(body: &Value) -> Option<ModelIdentity> {
        let sha256 = as_string(body.pointer("/model/content_hash"))?;
        if !is_sha256(&sha256) {
            return None;
        }
        Some(ModelIdentity {
            path: as_string(body.pointer("/model/path"))?,
            sha256,
            bytes: as_u64(body.pointer("/model/size_bytes"))?,
        })
    }
}

/// PP-2 — the declared input must not contradict the server's own report.
///
/// A run that declares `--compute-class cuda` against a server reporting `cpu`
/// is not a mislabelled receipt; it is a receipt about a different execution
/// path than the one that ran. The refusal happens before the first sampled
/// request, because under §12's spend rule the measurement is gone either way
/// and the only thing left to save is the operator's next twenty minutes.
fn reconcile_compute_class(declared: &str, reported: Option<&str>) -> Result<()> {
    match reported {
        Some(r) if r != declared => Err(CliError::InvalidInput(format!(
            "--compute-class {declared:?} but GET /v1/effective-config reports {r:?} — PP-2 \
             requires the dispatch path TAKEN, read from the process. The server's report wins \
             and the declared value is refused rather than silently overwritten, because the \
             disagreement usually means the wrong server is being measured."
        ))),
        _ => Ok(()),
    }
}

/// PP-2's other half: a declared server feature the server does not report.
fn reconcile_features(declared: &[String], reported: &[String]) -> Result<()> {
    if reported.is_empty() {
        return Ok(());
    }
    let missing: Vec<&str> = declared
        .iter()
        .filter(|d| !reported.iter().any(|r| r == *d))
        .map(String::as_str)
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    Err(CliError::InvalidInput(format!(
        "--server-feature {missing:?} not in the server's own build_features {reported:?} — \
         PP-2 refuses a claim about a path the build cannot take"
    )))
}

/// PP-22 — the comparator lane's configuration, read from its `GET /props`.
///
/// Every field stays `None` unless the comparator actually reported it:
/// `JoinKey::refuse_mismatch` treats an absent field as NOT matching a present
/// one, so a guessed value would silently make two different configurations
/// join. `n_ctx_slot` is `default_generation_settings.n_ctx` when present —
/// that is llama.cpp's PER-SLOT context — and `n_ctx / total_slots` otherwise.
fn lane_config_from_props(props: Option<&Value>) -> LaneConfig {
    let Some(props) = props else {
        return LaneConfig::default();
    };
    let settings = props.pointer("/default_generation_settings");
    let per_slot = settings.and_then(|s| as_u32(s.get("n_ctx"))).or_else(|| {
        let total = as_u32(props.get("n_ctx"))?;
        let slots = as_u32(props.get("total_slots"))?;
        total.checked_div(slots)
    });
    LaneConfig {
        n_ctx_slot: per_slot,
        kv_type: settings
            .and_then(|s| as_string(s.get("cache_type_k")))
            .or_else(|| as_string(props.get("cache_type_k"))),
        fa: settings
            .and_then(|s| s.get("flash_attn"))
            .or_else(|| props.get("flash_attn"))
            .and_then(Value::as_bool),
        n_batch: settings
            .and_then(|s| as_u32(s.get("n_batch")))
            .or_else(|| as_u32(props.get("n_batch"))),
    }
}

/// §5.3 — a comparator serving one request at a time is not serving the band.
///
/// Refused here, from the lane's OWN `/props`, before a single request is
/// issued. [`JoinKey::refuse_cripple`] catches it again at join time — but by
/// then the measurement has been spent against a configuration that once
/// manufactured a 2.39x overstatement (`llama_pin.toml:129-165`), and under
/// §12's spend rule there is no re-running it.
fn lane_refuse_cripple(lane: &LaneConfig) -> Result<()> {
    if lane.n_batch == Some(1) {
        return Err(CliError::InvalidInput(
            "§5.3: the comparator's own GET /props reports n_batch = 1, which switches \
             llama.cpp's batching OFF. That configuration is a cripple, not a lane serving the \
             band; it manufactured a 2.39x overstatement once (llama_pin.toml:129-165). Relaunch \
             the comparator with -b >= 2."
                .to_string(),
        ));
    }
    Ok(())
}

/// PP-24 — the comparator lane's admitted slot count, from `GET /props`.
fn comparator_slots(props: Option<&Value>) -> Option<u32> {
    as_u32(props?.get("total_slots"))
}

/// PP-20 — the comparator pin, or a refusal naming what is missing.
///
/// §5.3 pins the comparator's commit, `cmake` line, binary digest and an
/// expiry. All four or none: a lane with three of them cannot be checked for
/// staleness, and `provenance.comparator` is the ONLY place the per-band
/// `GET /props` response is retained — so a partial pin would silently discard
/// the configuration evidence §5.3 requires.
fn comparator_identity(args: &BandArgs<'_>, props: Option<Value>) -> Result<ComparatorIdentity> {
    let missing: Vec<&str> = [
        ("--comparator-commit", args.comparator_commit),
        ("--comparator-cmake", args.comparator_cmake),
        ("--comparator-sha256", args.comparator_sha256),
        ("--comparator-pin-expiry", args.comparator_pin_expiry),
    ]
    .into_iter()
    .filter(|(_, v)| v.is_none())
    .map(|(name, _)| name)
    .collect();
    if !missing.is_empty() {
        return Err(CliError::InvalidInput(format!(
            "--comparator-url needs the §5.3 pin too; missing {missing:?}. PP-20 marks every \
             ratio COMPARATOR_STALE once the pin expires, and a pin that does not exist can \
             never expire — a comparator lane with no pin produces ratios nothing can date. \
             `provenance.comparator` is also the only field that retains the lane's `GET \
             /props` body, which §5.3 requires per band."
        )));
    }
    Ok(ComparatorIdentity {
        commit: args.comparator_commit.unwrap_or_default().to_string(),
        cmake: args.comparator_cmake.unwrap_or_default().to_string(),
        sha256: args.comparator_sha256.unwrap_or_default().to_string(),
        pin_expiry: args.comparator_pin_expiry.unwrap_or_default().to_string(),
        props: props.unwrap_or(Value::Null),
    })
}

/// PP-26 — the per-band batch-invariance witnesses, keyed by `c`.
///
/// Read from `witness.json` as `scripts/perf041_batched_parity_probe.py`
/// writes it. The verdict is taken from the probe's own `result` token rather
/// than recomputed here: the probe holds the token streams, this producer does
/// not, and a witness derived from anything else would be a second opinion
/// with no evidence under it.
fn load_witness(path: Option<&Path>) -> Result<BTreeMap<u32, BatchInvarianceWitness>> {
    let Some(path) = path else {
        return Ok(BTreeMap::new());
    };
    let text = std::fs::read_to_string(path)
        .map_err(|e| CliError::InvalidInput(format!("--witness-json {}: {e}", path.display())))?;
    let doc: Value = serde_json::from_str(&text).map_err(|e| {
        CliError::InvalidFormat(format!("--witness-json {}: not JSON: {e}", path.display()))
    })?;
    let bands = doc.get("bands").and_then(Value::as_array).ok_or_else(|| {
        CliError::InvalidFormat(format!(
            "--witness-json {}: no `bands` array — this is not a perf041 witness",
            path.display()
        ))
    })?;
    let source = format!(
        "scripts/perf041_batched_parity_probe.py ({})",
        path.display()
    );
    let mut out = BTreeMap::new();
    for band in bands {
        let c = as_u32(band.get("c")).ok_or_else(|| {
            CliError::InvalidFormat(format!(
                "--witness-json {}: a band entry has no concurrency `c`",
                path.display()
            ))
        })?;
        out.insert(c, witness_of(band, &source, path)?);
    }
    Ok(out)
}

/// One perf041 band entry as a [`BatchInvarianceWitness`].
fn witness_of(band: &Value, source: &str, path: &Path) -> Result<BatchInvarianceWitness> {
    let result = band.get("result").and_then(Value::as_str).unwrap_or("");
    // An unknown verdict token is REFUSED, never mapped to the failing side.
    // A silent fallback would turn a probe this producer does not understand
    // into a correctness verdict it never gave.
    let batch_invariance = match result {
        "PASS" => BatchInvariance::Pass,
        "FAIL" => BatchInvariance::Fail,
        "UNMEASURABLE" => BatchInvariance::Unmeasurable,
        other => {
            return Err(CliError::InvalidFormat(format!(
                "--witness-json {}: band result {other:?} is not one of PASS, FAIL, \
                 UNMEASURABLE (PP-26)",
                path.display()
            )))
        }
    };
    Ok(BatchInvarianceWitness {
        batch_invariance,
        divergence_at: as_u32(band.get("divergence_at")),
        intra_agree_to: as_u32(band.get("intra_agree_to")),
        max_constant_run: as_u32(band.get("max_constant_run")),
        declared_min: as_u32(band.get("declared_min")).unwrap_or_default(),
        m_formed: as_u32(band.get("m_formed")).unwrap_or_default(),
        source: source.to_string(),
    })
}

/// §4.2.2 — who measured, what served, and what the pinned comparator was.
///
/// `binary_*` stay the CLIENT's, unchanged from v2.2 so existing readers keep
/// working, and [`Provenance::client`] says so in a field that cannot be
/// misread. The SUBJECT's feature set is the server's own `build_features`
/// when the §5.2 endpoint answered — `bench_receipt.py` uses `feature_set` to
/// refuse a `compute_class` the build cannot reach, and pointing that check at
/// the measuring binary instead of the measured one would make it read green
/// while checking nothing.
fn build_provenance(input: ProvenanceInput<'_>) -> Result<Provenance> {
    let ProvenanceInput {
        args,
        commit,
        started_utc,
        facts,
        server_config,
        comparator,
        notes,
    } = input;
    let exe = std::env::current_exe()
        .map_err(|e| CliError::InvalidInput(format!("cannot resolve current_exe: {e}")))?;
    let binary_sha256 = sha256_file(&exe)
        .map_err(|e| CliError::InvalidInput(format!("cannot hash {}: {e}", exe.display())))?;
    let compute_class = ComputeClass::from_str(args.compute_class)
        .map_err(|e| CliError::InvalidInput(format!("--compute-class: {e}")))?;

    reconcile_compute_class(args.compute_class, facts.compute_class.as_deref())?;
    reconcile_features(args.server_features, &facts.build_features)?;

    // PP-18: the subject binary. `apr serve` and `apr test llm bench` are the
    // same `apr`, so `current_exe` is the common case — but it is a DECLARED
    // fact, not an observed one, and it is named as such in
    // `unproduced_fields` unless the operator pointed at the served binary.
    let subject_path = args
        .subject_binary
        .map_or_else(|| exe.clone(), Path::to_path_buf);
    let subject_sha256 = sha256_file(&subject_path).map_err(|e| {
        CliError::InvalidInput(format!("cannot hash {}: {e}", subject_path.display()))
    })?;
    if args.subject_binary.is_none() {
        notes.push(format!(
            "PP-18 provenance.subject.path/sha256 — DECLARED, not observed. No `--subject-binary` \
             was given, so the receipt names the client binary ({}) as the served one. That is \
             correct when `apr serve` and `apr test llm bench` are the same build and wrong \
             otherwise; pass --subject-binary to make it an observation.",
            exe.display()
        ));
    }
    let subject_feature_set = if facts.build_features.is_empty() {
        notes.push(
            "PP-2 provenance.subject.feature_set — source \"declared\": `GET \
             /v1/effective-config` reported no build_features (or was not routed), so the set is \
             the operator's `--server-feature` flags. A declared feature set cannot falsify a \
             compute_class claim; a server-reported one can."
                .to_string(),
        );
        args.server_features.to_vec()
    } else {
        facts.build_features.clone()
    };
    // PP-30 / §5.1: the harness clock and the SERVER's own start instant, side
    // by side. A server that started after this run's clock read restarted
    // between the two, which means the warm state §5.1's warmup produced is
    // gone — a fact a reader must be able to see without re-parsing
    // `server_config`, and one no band-level number reveals.
    match facts.started_utc.as_deref() {
        Some(server_start) => notes.push(format!(
            "§5.1 server start — the subject server reports started_utc {server_start}; this run \
             read its own clock at {started_utc}. A server start LATER than the run start means \
             the server restarted mid-preparation and the warmup's warm state is gone."
        )),
        None => notes.push(
            "PP-30 server.started_utc — the subject server reported no start instant, so its \
             uptime at measurement time cannot be reconstructed from this receipt."
                .to_string(),
        ),
    }
    if facts.compute_class.is_none() {
        notes.push(format!(
            "PP-2 provenance.compute_class — source \"declared\": the server did not report a \
             dispatch path, so {:?} is the operator's `--compute-class`. PP-2 asks for the path \
             TAKEN, read from the process.",
            args.compute_class
        ));
    }
    // PP-18: the subject's commit is the SERVER's when it reported one.
    let subject_commit = facts.build_commit.clone().unwrap_or_else(|| {
        notes.push(
            "PP-18 provenance.subject.commit — source \"declared\": the server reported no \
             build_commit, so the commit under test stands in for it and the ancestry assertion \
             is vacuous for the subject lane."
                .to_string(),
        );
        commit.to_string()
    });

    let prov = Provenance {
        binary_path: exe.display().to_string(),
        binary_sha256: binary_sha256.clone(),
        resolution: "current_exe".to_string(),
        compute_class,
        host: args.host.to_string(),
        accelerator: args.accelerator.to_string(),
        model: args.model.to_string(),
        quantization: args.quantization.to_string(),
        feature_set: subject_feature_set.clone(),
        started_utc: started_utc.to_string(),
        clock_source: CLOCK_SOURCE_SYSTEM_REALTIME.to_string(),
        subject: SubjectIdentity {
            path: subject_path.display().to_string(),
            sha256: subject_sha256,
            commit: subject_commit,
            feature_set: subject_feature_set,
        },
        client: ClientIdentity {
            path: exe.display().to_string(),
            sha256: binary_sha256,
            commit: commit.to_string(),
            // PP-3 — the fourth input to `RunId::derive`, on the wire so a
            // reader can recompute the id the receipt states. Without it
            // "derived, therefore reproducible from the receipt" was a comment
            // no reader could check.
            pid: std::process::id(),
        },
        comparator,
        server_config,
        model_file: facts.model_file.clone(),
    };
    prov.validate().map_err(CliError::InvalidInput)?;
    Ok(prov)
}

/// The seven things [`build_provenance`] needs, as a struct rather than seven
/// positional arguments of overlapping types.
struct ProvenanceInput<'a> {
    args: &'a BandArgs<'a>,
    commit: &'a str,
    started_utc: &'a str,
    facts: &'a ServerFacts,
    server_config: Option<Value>,
    comparator: Option<ComparatorIdentity>,
    notes: &'a mut Vec<String>,
}

/// §4.7.1 — this cell's comparator posture when there is no comparator lane.
///
/// With `--comparator-url` absent this producer measures ONE lane, so it emits
/// `UNMEASURED` with an owner and Arm B reports and skips. An `agg_ratio`
/// synthesised here is precisely the fabrication the epic exists to remove;
/// the only way to a `MEASURED` posture is [`BandInput::join_status`], which
/// needs a second lane from the same invocation.
fn comparator_status(args: &BandArgs<'_>, lane: Lane) -> ComparatorStatus {
    let reason = match (lane, args.comparator_url) {
        (Lane::Llama, _) => "PP-3: this band IS the comparator lane. A ratio is carried by the \
             SUBJECT band, against this one as its baseline; the comparator of the comparator is \
             the subject, and recording it here would make the receipt a chain of baselines."
            .to_string(),
        (Lane::Apr, None) => "PP-3/PP-25: a ratio needs a baseline band from the SAME run, joined \
             on the PP-22 key and driven by this same client binary. This invocation was given \
             no --comparator-url, so it measured one lane and declares the ratio unmeasured \
             rather than deriving one."
            .to_string(),
        (Lane::Apr, Some(url)) => format!(
            "PP-3: the comparator lane at {url} was measured in this same invocation but the \
             join has not been formed for this band yet. A band left in this posture at render \
             time means the join was REFUSED; the refusal replaces this text and says which rule \
             refused it."
        ),
    };
    ComparatorStatus::unmeasured(args.comparator_owner, reason)
}

/// One `run_band` result as the receipt producer's input.
///
/// `RequestSample` records offsets from the band origin in **seconds**;
/// `RequestOutcome` records them in **milliseconds**. The conversion is here,
/// once, rather than at each field's use.
///
/// `ttft_ms` is a duration from issue to first token, while `token_times_ms`
/// are absolute offsets from the band origin — the two differ by `issued_ms`,
/// and `RequestOutcome::itl_gaps_ms` differences the latter, so mixing the
/// conventions would leave the gaps correct and TTFT wrong by the request's
/// start offset.
///
/// `extras` carries the three per-request facts `RequestSample` has no field
/// for (PP-4's server prefill, PP-28's issued `n_predict`, PP-27's declared
/// mode). It is index-aligned with `samples`; a missing entry yields `None`
/// for those fields rather than another request's values.
/// `run.protocol_violations` is carried onto the band, not merely printed.
///
/// `report_run` printed each one to stdout and dropped it. §4.4.2's departures
/// — a window that closed below the `max(30, 8c)` sample floor, a warmup that
/// did not complete — are facts about the measurement, and the receipt is the
/// only thing `perf_gate.sh` ever sees. A violation the operator watched scroll
/// past and the receipt did not carry is a receipt that reads conformant.
fn band_input(run: &BandRun, comparator: ComparatorStatus) -> BandInput {
    let requests = run
        .samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let extra = run.extras.get(i).copied().unwrap_or_default();
            RequestOutcome {
                issued_ms: s.start_s * 1000.0,
                settled_ms: s.end_s * 1000.0,
                outcome: s.outcome,
                generated_tokens: s.generated_tokens,
                prompt_tokens: s.prompt_tokens,
                expected_tokens: extra.expected_tokens,
                ttft_ms: s.token_times_s.first().map(|t| (t - s.start_s) * 1000.0),
                prefill_ms: extra.prefill_ms,
                in_flight_at_start: u32::try_from(s.in_flight_at_start).unwrap_or(u32::MAX),
                token_times_ms: s.token_times_s.iter().map(|t| t * 1000.0).collect(),
            }
        })
        .collect();
    let mut input = BandInput::new(
        // Infallible: `parse_bands` refused anything a `u32` cannot hold, so
        // there is no clamp here to silently relabel the band.
        u32::try_from(run.config.concurrency).unwrap_or(u32::MAX),
        run.window.window_ms,
        requests,
        comparator,
    );
    if let Some(mode) = run.stream_mode {
        input = input.stream_mode(mode);
    }
    input.conformance_violations(run.protocol_violations.clone())
}

/// Print one lane's headline numbers as it finishes.
fn report_run(concurrency: usize, k: usize, replicates: usize, lane: &str, run: &BandRun) {
    println!(
        "  c={concurrency} replicate {}/{replicates} lane {lane}",
        k + 1
    );
    println!(
        "    agg {:.2} tok/s  decode {:.2} tok/s  requested {}  completed {}  timeouts {}  \
         drain {:.1} ms  peak_in_flight {}  stream_mode {:?}",
        run.metrics.agg_tok_s,
        run.metrics.decode_tok_s,
        run.metrics.requested,
        run.metrics.completed,
        run.metrics.timeouts,
        run.window.drain_ms,
        run.window.client_peak_in_flight,
        run.stream_mode
    );
    for v in &run.protocol_violations {
        println!("    ! {v}");
    }
}

/// One band of one replicate, both lanes.
struct BandRow {
    concurrency: usize,
    subject: BandRun,
    comparator: Option<BandRun>,
}

/// Everything the receipt needs that does not change between replicates.
struct ReceiptShell<'a> {
    args: &'a BandArgs<'a>,
    tokenization: TokenizationBlock,
    provenance: Provenance,
    protocol: ProtocolParams,
    workload: Workload,
    commit: String,
    run_id: RunId,
    ladder: Ladder,
    lane: LaneConfig,
    witness: BTreeMap<u32, BatchInvarianceWitness>,
    kv: Option<KvBlock>,
    notes: Vec<String>,
    /// PMAT-973 / #2756 — sha256 of the prompt-corpus bytes that earned
    /// `workload`'s label. `Some` only when [`bind_workload`] accepted it;
    /// `prepare` refuses the run rather than construct a shell with `None`.
    corpus_sha256: Option<String>,
}

impl ReceiptShell<'_> {
    /// PP-22 — the join key for one band of one lane.
    ///
    /// `window_ms` is the DECLARED protocol window, not the measured one. Two
    /// lanes never close their windows on the same millisecond — each closes
    /// when BOTH its sample floor and its wall-clock floor are met — so a key
    /// built from the measured value could never match and every join would
    /// refuse for a reason PP-22 is not about. What PP-22 asks is whether the
    /// two lanes ran the same protocol: "a 30 s window against a 60 s one
    /// compares different amounts of thermal drift".
    fn join_key(&self, concurrency: u32) -> JoinKey {
        JoinKey {
            host: self.provenance.host.clone(),
            workload: self.workload,
            band: concurrency,
            model: self.provenance.model.clone(),
            quant: self.provenance.quantization.clone(),
            tokenization: self.tokenization.method(),
            window_ms: self.protocol.window_ms,
            replicates: self.protocol.replicates,
            interleaved: self.protocol.interleaved,
            n_ctx_slot: self.lane.n_ctx_slot,
            kv_type: self.lane.kv_type.clone(),
            fa: self.lane.fa,
            n_batch: self.lane.n_batch,
            n_predict: self.protocol.n_predict,
        }
    }

    fn band_context(&self) -> BandContext {
        BandContext {
            schema_version: SCHEMA_VERSION,
            replicates: self.protocol.replicates,
            interleaved: self.protocol.interleaved,
            comparator_stale: self.provenance.comparator_is_stale(),
            ..BandContext::default()
        }
    }

    /// Attach the §5.1 pin, the correctness witness, the lane configuration
    /// and the retained samples file to one lane's band.
    ///
    /// # PP-26: the witness goes on the SUBJECT lane only
    ///
    /// `witness.json` comes from `scripts/perf041_batched_parity_probe.py`,
    /// which probes **`apr serve`**: it asks whether the subject returns the
    /// same tokens under batching as it does alone. This function used to copy
    /// that verdict onto the comparator band too, so a subject-side PASS
    /// silently vouched for `llama-server` — a correctness claim about a server
    /// nothing had probed. The comparator is the ORACLE the question is asked
    /// against; its band carries `witness: null` and is exempt from the
    /// requirement (`BandInput::invalid_correctness`), rather than borrowing a
    /// verdict that is not about it.
    fn dress(
        &self,
        input: BandInput,
        lane: Lane,
        replicate: usize,
        samples_file: SamplesFile,
    ) -> BandInput {
        let mut input = input
            .replicate(u32::try_from(replicate + 1).unwrap_or(u32::MAX))
            .n_predict(self.protocol.n_predict)
            .lane(self.lane.clone())
            .role(lane)
            .samples_file(samples_file);
        if lane == Lane::Apr {
            if let Some(w) = self.witness.get(&input.concurrency) {
                input = input.witness(w.clone());
            }
        }
        input
    }

    /// PP-3 / PP-22 / P-5 — the subject band with its same-run comparator
    /// joined, or the subject band alone when there is no comparator lane.
    ///
    /// # A refused join never discards the measurement
    ///
    /// This used to return the join's `Err`, which propagated out of
    /// `write_replicate` and aborted `run_bands` — so a two-lane run with a
    /// `c > 1` band and no `--witness-json` measured every band, for every
    /// replicate, and then threw the whole sweep away at receipt-write time
    /// with nothing on disk. The join legitimately refuses there (an
    /// `INVALID-CORRECTNESS` band reports no throughput, so there is no `agg`
    /// to divide) — but under §12's spend rule the measurement is gone either
    /// way, and the only thing a late refusal can still destroy is the record
    /// of what was measured.
    ///
    /// So a refused join becomes the honest posture it describes: `UNMEASURED`,
    /// carrying the refusal's own text as its reason. Nothing is fabricated —
    /// there is still no baseline and no ratio, which is exactly what the
    /// refusal said — and the receipt is written, with the band rendering
    /// `INVALID-CORRECTNESS` (and no throughput) or `NONCONFORMANT-VALID` as
    /// the rules require.
    fn joined(&self, subject: BandInput, comparator: Option<BandInput>) -> BandInput {
        let Some(comparator) = comparator else {
            return subject;
        };
        let subject_key = self.join_key(subject.concurrency);
        let comparator_key = self.join_key(comparator.concurrency);
        let status = match BandInput::join_status_in(
            &subject,
            &comparator,
            &subject_key,
            &comparator_key,
            // PP-3: ONE run id for both lanes. They are two lanes of one
            // invocation; a ratio against a baseline from another run divides
            // two thermal states, two free-VRAM figures and two schedulers.
            (&self.run_id, &self.run_id),
            &self.band_context(),
        ) {
            Ok(status) => status,
            Err(refusal) => {
                println!("  ! c={} join refused: {refusal}", subject.concurrency);
                ComparatorStatus::unmeasured(
                    self.args.comparator_owner,
                    format!(
                        "PP-3: the comparator lane ran in this same invocation and the join was \
                         REFUSED, so this band has no baseline and no ratio: {refusal}"
                    ),
                )
            }
        };
        BandInput {
            comparator: status,
            ..subject
        }
    }

    /// Write one replicate's receipt and its per-band sample files.
    fn write_replicate(&self, replicate: usize, rows: &[BandRow]) -> Result<PathBuf> {
        let mut bands = Vec::with_capacity(rows.len());
        for row in rows {
            let c = row.concurrency;
            let file = write_samples(self.args.receipt, "", c, replicate, &row.subject)?;
            let subject = self.dress(
                band_input(&row.subject, comparator_status(self.args, Lane::Apr)),
                Lane::Apr,
                replicate,
                file,
            );
            let comparator = match &row.comparator {
                None => None,
                Some(run) => {
                    let file = write_samples(self.args.receipt, "comparator.", c, replicate, run)?;
                    Some(self.dress(
                        band_input(run, comparator_status(self.args, Lane::Llama)),
                        Lane::Llama,
                        replicate,
                        file,
                    ))
                }
            };
            bands.push(self.joined(subject, comparator));
        }

        let mut input = ReceiptInput::new(
            self.run_id.clone(),
            self.provenance.clone(),
            self.tokenization.clone(),
            self.workload,
            self.protocol,
            self.commit.clone(),
            self.ladder.clone(),
            bands,
        );
        input.kv = self.kv;
        let rendered = render_with_notes(&input, &self.notes)
            .and_then(|json| attach_corpus_sha256(&json, self.corpus_sha256.as_deref()))
            .map_err(|e| CliError::ValidationFailed(format!("receipt r{}: {e}", replicate + 1)))?;

        let path = self
            .args
            .receipt
            .join(format!("receipt.r{}.json", replicate + 1));
        std::fs::write(&path, rendered.as_bytes())
            .map_err(|e| CliError::InvalidFormat(format!("writing {}: {e}", path.display())))?;
        println!("receipt  {} ({} bytes)", path.display(), rendered.len());
        sign_receipt(self.args, &path)?;
        Ok(path)
    }
}

/// §4.4.5 — one lane's raw samples, gzipped beside the receipt.
fn write_samples(
    dir: &Path,
    prefix: &str,
    concurrency: usize,
    replicate: usize,
    run: &BandRun,
) -> Result<SamplesFile> {
    let path = dir.join(format!(
        "{prefix}samples.c{concurrency}.r{}.jsonl.gz",
        replicate + 1
    ));
    let file = write_samples_gz(&path, &run.samples)
        .map_err(|e| CliError::InvalidFormat(format!("writing {}: {e}", path.display())))?;
    println!(
        "samples  {} ({} rows, {} bytes)",
        file.path.display(),
        file.rows,
        file.bytes
    );
    Ok(file)
}

/// Render the receipt and append this run's provenance notes to
/// `unproduced_fields`.
///
/// `ReceiptInput` owns the notes it can derive from its own contents; these
/// are the ones only the PRODUCER knows — that a compute class was declared
/// rather than read, that a subject binary was assumed. Naming them in
/// `unproduced_fields` is what keeps "declared" and "observed" distinguishable
/// inside the file, which is the whole of PP-13.
///
/// # Errors
/// When the receipt does not render, or carries no `unproduced_fields` array.
fn render_with_notes(
    input: &ReceiptInput,
    notes: &[String],
) -> std::result::Result<String, String> {
    let mut value = input.render()?;
    if !notes.is_empty() {
        let field = value
            .get_mut("unproduced_fields")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| "receipt has no `unproduced_fields` array to extend".to_string())?;
        field.extend(notes.iter().map(|n| Value::String(n.clone())));
    }
    serde_json::to_string_pretty(&value).map_err(|e| format!("serialising receipt: {e}"))
}

/// PMAT-973 / #2756 — add `corpus_sha256` to an already-rendered receipt.
///
/// A second pass over the rendered JSON, the same shape as
/// [`render_with_notes`]'s `unproduced_fields` patch, rather than a field on
/// `ReceiptInput` itself: `ReceiptInput` lives in `aprender-test-lib` and is
/// shared with `bench_receipt.py`'s schema on the other side of the wire;
/// this producer's own binding is not that contract's business.
///
/// # Errors
/// When `rendered` is not valid JSON.
fn attach_corpus_sha256(
    rendered: &str,
    corpus_sha256: Option<&str>,
) -> std::result::Result<String, String> {
    let Some(sha) = corpus_sha256 else {
        return Ok(rendered.to_string());
    };
    let mut value: Value =
        serde_json::from_str(rendered).map_err(|e| format!("re-reading rendered receipt: {e}"))?;
    if let Some(map) = value.as_object_mut() {
        map.insert("corpus_sha256".to_string(), Value::String(sha.to_string()));
    }
    serde_json::to_string_pretty(&value).map_err(|e| format!("serialising receipt: {e}"))
}

/// PP-21 — sign the receipt in place, failing closed.
///
/// Shells out to `scripts/perf_receipt_sign.sh`, which calls
/// `scripts/lib/receipt_sig.py` — the single authority
/// `perf_gate.sh --phase release` verifies against. Porting the HMAC here
/// would give the payload two constructions that could drift, and a signature
/// that verifies against the wrong payload is worse than an unsigned receipt.
///
/// A non-zero exit is fatal: an unsigned receipt the producer reported as
/// signed is exactly the document `ArmC-sig` exists to catch.
fn sign_receipt(args: &BandArgs<'_>, path: &Path) -> Result<()> {
    let Some(key_id) = args.key_id else {
        return Ok(());
    };
    let mut cmd = std::process::Command::new("bash");
    cmd.arg("scripts/perf_receipt_sign.sh")
        .arg("--receipt")
        .arg(path)
        .arg("--key-id")
        .arg(key_id)
        .arg("--out")
        .arg(path);
    if let Some(keyring) = args.keyring {
        cmd.arg("--keyring").arg(keyring);
    }
    let status = cmd.status().map_err(|e| {
        CliError::InvalidInput(format!(
            "PP-21: could not run scripts/perf_receipt_sign.sh (run this from the repository \
             root): {e}"
        ))
    })?;
    if !status.success() {
        return Err(CliError::ValidationFailed(format!(
            "PP-21: scripts/perf_receipt_sign.sh exited {} signing {} — the receipt on disk is \
             UNSIGNED and `perf_gate.sh --phase release` will fail it. Refusing to report a \
             signed receipt that is not one.",
            status.code().unwrap_or(-1),
            path.display()
        )));
    }
    println!("signed   {} (key {key_id})", path.display());
    Ok(())
}

/// The two lanes and the corpus each is issued.
struct Lanes {
    subject: LlmClient,
    subject_prompts: Vec<ChatRequest>,
    comparator: Option<LlmClient>,
    comparator_prompts: Vec<ChatRequest>,
    prompt_band: Option<PromptTokenBand>,
}

/// PP-25 — build both lanes from ONE corpus and ONE client type.
///
/// The only difference between the two request streams is the model name the
/// two servers expect in the body; everything else — prompt text, sampler pin,
/// `max_tokens` — is byte-identical, because §4.4.8 refuses a ratio taken by
/// two clients over two corpora.
async fn open_lanes(args: &BandArgs<'_>) -> Result<Lanes> {
    let corpus = resolve_corpus(args.profile, args.prompts)?;
    let prompt_band = corpus.band;
    let subject_prompts: Vec<ChatRequest> = corpus
        .requests
        .into_iter()
        .map(|mut p| {
            p.model = args.model.to_string();
            p
        })
        .collect();
    let comparator_model = args.comparator_model.unwrap_or(args.model);
    let comparator_prompts: Vec<ChatRequest> = subject_prompts
        .iter()
        .cloned()
        .map(|mut p| {
            p.model = comparator_model.to_string();
            p
        })
        .collect();

    let subject = LlmClient::new(args.url, args.model);
    subject.health_check().await.map_err(|e| {
        CliError::InferenceFailed(format!("endpoint {} is not ready: {e}", args.url))
    })?;
    let comparator = match args.comparator_url {
        None => None,
        Some(url) => {
            let c = LlmClient::new(url, comparator_model);
            c.health_check().await.map_err(|e| {
                CliError::InferenceFailed(format!("comparator {url} is not ready: {e}"))
            })?;
            Some(c)
        }
    };
    Ok(Lanes {
        subject,
        subject_prompts,
        comparator,
        comparator_prompts,
        prompt_band,
    })
}

/// PP-2 / §5.3 — read both servers' configuration BEFORE the first sampled
/// request, and keep both bodies verbatim.
async fn read_configuration(
    lanes: &Lanes,
    args: &BandArgs<'_>,
) -> Result<(Option<Value>, ServerFacts, Option<Value>)> {
    let server_config = lanes
        .subject
        .get_json("/v1/effective-config")
        .await
        .map_err(|e| CliError::InferenceFailed(format!("GET /v1/effective-config: {e}")))?;
    let facts = ServerFacts::read(server_config.as_ref());
    let props = match (&lanes.comparator, args.comparator_url) {
        (Some(c), Some(_)) => c
            .get_json("/props")
            .await
            .map_err(|e| CliError::InferenceFailed(format!("comparator GET /props: {e}")))?,
        _ => None,
    };
    Ok((server_config, facts, props))
}

/// Everything refusable before the first sampled request, refused.
///
/// Under §12's spend rule the measurement is gone once it has been taken, so
/// the only thing an early refusal can still save is the operator's next
/// twenty minutes — and the only thing a LATE refusal saves is nothing at all.
async fn prepare<'a>(
    args: &'a BandArgs<'a>,
    levels: &[usize],
) -> Result<(ReceiptShell<'a>, Lanes)> {
    require_stream(args.stream)?;
    let commit = require_commit(args.commit)?;
    let tokenization = build_tokenization(args)?;
    let workload = Workload::from_str(args.workload)
        .map_err(|e| CliError::InvalidInput(format!("--workload: {e}")))?;
    // PMAT-973 / #2756 — a workload label the receipt cannot falsify is not
    // provenance: refuse before the first request rather than record a label
    // the sent prompts never earned.
    let corpus_sha256 = match args.prompts {
        Some(path) => Some(bind_workload(workload, path)?.corpus_sha256),
        None => {
            let corpus = resolve_corpus(args.profile, None)?;
            return Err(CliError::ValidationFailed(format!(
                "--workload {}: the prompt set sent is not the {} corpus ({}); a corpus label \
                 needs its corpus",
                args.workload,
                args.workload,
                describe_workload(args.profile, None, corpus.requests.len())
            )));
        }
    };
    let witness = load_witness(args.witness_json)?;
    let (protocol, protocol_source) =
        protocol_params_with_source(args.replicates, args.comparator_url.is_some());
    // PP-33 — say which of the two sources supplied the protocol block, once,
    // before anything is measured against it.
    println!("{}", protocol_source.announcement());
    // §5.3 — the declared comparator lane, parsed before any connection is
    // opened so a bad `--comparator-fa` costs nothing.
    let declared_lane = declared_lane_config(args)?;
    // PP-26 — the operator learns what an absent witness costs BEFORE the
    // first request, not in a note appended to a receipt twenty minutes later.
    if witness.is_empty() {
        println!(
            "NOTE     no --witness-json: PP-26 makes every c>1 band INVALID-CORRECTNESS and it \
             will report NO throughput. Run scripts/perf041_batched_parity_probe.py first."
        );
    }
    std::fs::create_dir_all(args.receipt).map_err(|e| {
        CliError::InvalidFormat(format!("creating {}: {e}", args.receipt.display()))
    })?;

    // PP-30: the clock is read once, before the first request; every band of
    // every replicate belongs to that instant's run.
    let started_utc = now_utc_millis();
    let lanes = open_lanes(args).await?;
    let (server_config, facts, props) = read_configuration(&lanes, args).await?;
    // §5.3 — the launcher's declaration and the server's own report, reconciled
    // BEFORE the first sampled request. `/props` wins where it reports; a
    // contradiction is refused; and a `-b 1` cripple is refused whichever of
    // the two named it, which is the only way that refusal can fire at all
    // against a llama-server (its `/props` reports no `n_batch`).
    let lane = reconcile_lane(&declared_lane, &lane_config_from_props(props.as_ref()))?;
    lane_refuse_cripple(&lane)?;
    let comparator_slots = comparator_slots(props.as_ref());

    let mut notes: Vec<String> = Vec::new();
    if let Some(note) = protocol_source.unproduced_note() {
        notes.push(note);
    }
    let pin = match args.comparator_url {
        None => None,
        Some(_) => Some(comparator_identity(args, props)?),
    };
    let provenance = build_provenance(ProvenanceInput {
        args,
        commit: &commit,
        started_utc: &started_utc,
        facts: &facts,
        server_config,
        comparator: pin,
        notes: &mut notes,
    })?;
    if witness.is_empty() {
        notes.push(
            "PP-26 witness — no --witness-json was given, so no band carries a batch-invariance \
             witness. Every band at c>1 is INVALID-CORRECTNESS and reports no throughput at all: \
             §7.0 asks \"were the tokens right?\" before \"how fast?\", and an unwitnessed batched \
             decode answered neither."
                .to_string(),
        );
    }

    let declared: Vec<u32> = levels
        .iter()
        .map(|c| u32::try_from(*c).unwrap_or(u32::MAX))
        .collect();
    let ladder = Ladder::derive(
        &declared,
        SlotsAdmitted {
            apr: facts.slots_admitted,
            llama: comparator_slots,
        },
    );
    let run_id = RunId::derive(
        &started_utc,
        args.host,
        &provenance.client.sha256,
        std::process::id(),
    );
    let shell = ReceiptShell {
        args,
        tokenization,
        provenance,
        protocol,
        workload,
        commit,
        run_id,
        ladder,
        lane,
        witness,
        kv: facts.kv,
        notes,
        corpus_sha256,
    };
    announce(&shell, levels, &started_utc);
    Ok((shell, lanes))
}

/// What this invocation is about to do, on stdout, before it starts.
fn announce(shell: &ReceiptShell<'_>, levels: &[usize], started_utc: &str) {
    let args = shell.args;
    println!("protocol PP-LLAMA-001 v3.0 §5.1 (closed-loop, conformant)");
    println!("run_id   {}", shell.run_id.as_str());
    println!("started  {started_utc}");
    println!("endpoint {}", args.url);
    match args.comparator_url {
        Some(u) => println!(
            "compare  {u} (model {}) — interleaved A,B per band",
            args.comparator_model.unwrap_or(args.model)
        ),
        None => println!("compare  none — every band's comparator_status is UNMEASURED"),
    }
    println!("workload {}", shell.workload.wire_token());
    println!(
        "prompts  {}",
        describe_workload(args.profile, args.prompts, 0)
    );
    println!(
        "bands    {levels:?} x {} replicate(s); warmup 2c, quiesce {} ms, cooldown {} ms, window \
         closes when BOTH max(30, 8c) samples AND {} ms wall-clock are met",
        args.replicates,
        shell.protocol.quiesce_ms,
        shell.protocol.cooldown_ms,
        shell.protocol.window_ms
    );
    println!(
        "ladder   declared {:?} derived {:?} (slots apr={:?} llama={:?})",
        shell.ladder.declared,
        shell.ladder.derived,
        shell.ladder.slots_admitted.apr,
        shell.ladder.slots_admitted.llama
    );
    if args.replicates < REPLICATES {
        println!(
            "!        --replicates {} is below §4.3's n={REPLICATES}; n=3 sizes an effect and \
             bounds no variance, so every band is NONCONFORMANT-VALID",
            args.replicates
        );
    }
    for skipped in levels.iter().filter(|c| {
        !shell
            .ladder
            .derived
            .contains(&u32::try_from(**c).unwrap_or(u32::MAX))
    }) {
        println!(
            "SKIP     c={skipped} is above the derived ladder — a band wider than what the \
             servers admitted measures a queue, not a server (PP-24)"
        );
    }
}

/// One lane of one band, with everything the report line needs.
struct LaneRun<'a> {
    client: &'a LlmClient,
    prompts: &'a [ChatRequest],
    band: &'a BandConfig,
    tokenization: &'a TokenizationBlock,
    prompt_band: Option<PromptTokenBand>,
    concurrency: usize,
    replicate: usize,
    replicates: usize,
    lane: &'static str,
}

/// Measure one lane of one band and assert §5.1's prompt-length band on it.
async fn measure(run: LaneRun<'_>) -> Result<BandRun> {
    let (c, k) = (run.concurrency, run.replicate);
    let lane = run.lane;
    let measured = run_band(
        run.client,
        run.prompts,
        run.band,
        run.tokenization.clone(),
        true,
    )
    .await
    .map_err(|e| {
        CliError::InferenceFailed(format!("{lane} band c={c} replicate {}: {e}", k + 1))
    })?;
    check_prompt_band(run.prompt_band, run.prompts.len(), &measured.samples).map_err(|e| {
        CliError::InvalidInput(format!("{lane} band c={c} replicate {}: {e}", k + 1))
    })?;
    report_run(c, k, run.replicates, lane, &measured);
    Ok(measured)
}

/// §4.3 — every band of every replicate, both lanes, interleaved.
///
/// The cooldown is taken before every lane except the very first of the run,
/// which makes the whole sequence A,B,A,B,… with a pause at each transition.
/// Without it the second lane inherits the first lane's thermal state and free
/// VRAM — the exact drift interleaving exists to cancel — so a single-lane run
/// takes no cooldown at all: there is no transition to cool across.
async fn sweep(shell: &ReceiptShell<'_>, lanes: &Lanes, levels: &[usize]) -> Result<Vec<PathBuf>> {
    let args = shell.args;
    let cooldown = std::time::Duration::from_millis(shell.protocol.cooldown_ms);
    let mut first_lane = true;
    let mut written = Vec::new();
    for k in 0..args.replicates {
        let mut rows = Vec::with_capacity(levels.len());
        for &c in levels {
            let band = BandConfig::conformant(c);
            let common = LaneRun {
                client: &lanes.subject,
                prompts: &lanes.subject_prompts,
                band: &band,
                tokenization: &shell.tokenization,
                prompt_band: lanes.prompt_band,
                concurrency: c,
                replicate: k,
                replicates: args.replicates,
                lane: "subject",
            };
            cool(&mut first_lane, cooldown, lanes.comparator.is_some()).await;
            let subject = measure(common).await?;
            let comparator = match &lanes.comparator {
                None => None,
                Some(client) => {
                    cool(&mut first_lane, cooldown, true).await;
                    Some(
                        measure(LaneRun {
                            client,
                            prompts: &lanes.comparator_prompts,
                            band: &band,
                            tokenization: &shell.tokenization,
                            prompt_band: lanes.prompt_band,
                            concurrency: c,
                            replicate: k,
                            replicates: args.replicates,
                            lane: "comparator",
                        })
                        .await?,
                    )
                }
            };
            rows.push(BandRow {
                concurrency: c,
                subject,
                comparator,
            });
        }
        written.push(shell.write_replicate(k, &rows)?);
    }
    Ok(written)
}

/// PP-24 — the bands this run may actually measure.
///
/// A band wider than what either server admitted measures a queue, not a
/// server, and `ReceiptInput::render` refuses to write one. Skipping is not
/// silence: [`announce`] names each skipped band and the `ladder` block on the
/// wire carries both the declared set and the derived one, so the receipt
/// records what was asked for as well as what ran.
///
/// When neither lane reported a slot count the derived ladder IS the declared
/// one (`Ladder::derive`), so nothing is skipped on no evidence.
fn runnable_bands(levels: &[usize], ladder: &Ladder) -> Vec<usize> {
    levels
        .iter()
        .copied()
        .filter(|c| {
            ladder
                .derived
                .contains(&u32::try_from(*c).unwrap_or(u32::MAX))
        })
        .collect()
}

/// §5.1 — the pause between two lanes. No-op for the first lane of the run,
/// and for a single-lane run, which has no lane transition to cool across.
async fn cool(first_lane: &mut bool, cooldown: std::time::Duration, interleaved: bool) {
    if !interleaved {
        return;
    }
    if *first_lane {
        *first_lane = false;
        return;
    }
    tokio::time::sleep(cooldown).await;
}

/// Run the §5.1 protocol over every requested band and write the receipts.
///
/// # Errors
/// When `--commit` or `--stream` is missing, when a comparator lane is asked
/// for without its §5.3 pin, when either endpoint is unreachable, when the
/// server contradicts a declared input (PP-2), when any band fails, or when a
/// receipt cannot be rendered, signed or written.
pub async fn run_bands(args: BandArgs<'_>) -> Result<()> {
    let levels = parse_bands(args.bands)?;
    if args.replicates == 0 {
        return Err(CliError::InvalidInput(
            "--replicates 0 measures nothing".to_string(),
        ));
    }
    let (shell, lanes) = prepare(&args, &levels).await?;
    let runnable = runnable_bands(&levels, &shell.ladder);
    if runnable.is_empty() {
        return Err(CliError::InvalidInput(format!(
            "PP-24: none of the requested bands {levels:?} is within the derived ladder {:?} \
             (slots apr={:?} llama={:?}) — every band would measure a queue rather than a server",
            shell.ladder.derived,
            shell.ladder.slots_admitted.apr,
            shell.ladder.slots_admitted.llama
        )));
    }
    let written = sweep(&shell, &lanes, &runnable).await?;

    println!("\nnext");
    for path in &written {
        println!(
            "  scripts/perf_gate.sh --host {} --phase merge --workload {} --receipt {}",
            args.host,
            shell.workload.wire_token(),
            path.display()
        );
    }
    Ok(())
}

/// Assert §4.3.1's prompt-length band against what the server actually counted.
///
/// # The defect this closes (PERF-056, #2778)
///
/// `prompts-w1.jsonl`'s `_meta.token_count_note` says, in the file, that
/// "the 512 +/-8 of 4.3.1 is asserted by the harness against the model's own
/// tokenizer at measurement time". No line of code did that. The generator
/// says so too and cannot do it (it runs no tokenizer); the loader says the
/// harness does it; the harness did nothing. A workload 200 tokens off its
/// declared shape would have been measured, receipted and ratcheted as W1.
///
/// This runs after each band because there is nowhere earlier it CAN run: the
/// counts are the server's, reported per request. Under I-9 the run is spent
/// either way — so the point is not to save the replicate, it is to stop the
/// receipt being written as if the workload had been W1.
///
/// `RequestSample::index` is the monotone ISSUE index; the corpus is consumed
/// modulo its length (`band.rs`'s `prompts[slot.0 % prompts.len()]`), so the
/// prompt index is recovered the same way. Only `Completed` samples are read:
/// a failed or timed-out request carries `prompt_tokens: 0` by construction,
/// and counting those would red every band that saw one timeout.
fn check_prompt_band(
    prompt_band: Option<PromptTokenBand>,
    corpus_len: usize,
    samples: &[RequestSample],
) -> std::result::Result<(), String> {
    let Some(prompt_band) = prompt_band else {
        return Ok(());
    };
    if corpus_len == 0 {
        return Ok(());
    }
    let observed: Vec<(usize, u32)> = samples
        .iter()
        .filter(|s| s.outcome == Outcome::Completed)
        .map(|s| (s.index % corpus_len, s.prompt_tokens))
        .collect();
    assert_prompt_tokens_in_band(prompt_band, &observed)
}

/// §4.3's `n`, re-exported so the CLI default and the spec constant cannot
/// drift apart.
///
/// **Five, not three.** v2.2 §4.4.2 asked for `N = 3`; PP-LLAMA-001 v3.0 §4.3
/// reverses that: "`n = 3` sizes an effect and bounds no variance, no
/// σ-dependent status changes at `n < 5`". Three replicates give a t bound with
/// two degrees of freedom, whose 95% one-sided multiplier is 2.920 — wide
/// enough that a real regression and a quiet run are indistinguishable.
#[must_use]
pub const fn default_replicates() -> usize {
    REPLICATES
}

/// PMAT-973 / #2756 — `bind_workload`/`receipt_accepts_workload` pinned
/// against the real W1 corpus and the `--profile short` shape (no `_meta`).
#[cfg(test)]
#[path = "test_llm_band_workload_binding_tests.rs"]
mod workload_corpus_binding;

#[cfg(test)]
mod tests {
    use super::*;
    use apr_test::perf_gate::BandStatus;

    // ---------------------------------------------------------------------
    // §4.3.1's PROMPT-LENGTH BAND, at the point of measurement (PERF-056,
    // #2778). `prompts-w1.jsonl` promises in its own `_meta` that "the 512
    // +/-8 of 4.3.1 is asserted by the harness against the model's own
    // tokenizer at measurement time". This is the harness. Until PERF-056 it
    // did nothing of the kind, and a W1 receipt could be written over a
    // workload of any shape.
    //
    //  server-reported prompt_tokens over the band | must  | why
    //  --------------------------------------------|-------|-----------------
    //  every completed request at 512              | OK    |
    //  504 / 520 at the edges       [BOUNDARY]     | OK    | the band is a band
    //  one request at 521                          | ERR   | names the prompt
    //  a FAILED request's structural 0  [BOUNDARY] | OK    | not a workload defect
    //  no band declared (a --profile) [BOUNDARY]   | OK    | nothing claimed it
    // ---------------------------------------------------------------------

    const BAND_512: PromptTokenBand = PromptTokenBand {
        target: 512,
        tolerance: 8,
    };

    fn sample(index: usize, prompt_tokens: u32, outcome: Outcome) -> RequestSample {
        RequestSample {
            index,
            worker: 0,
            start_s: 0.0,
            end_s: 1.0,
            token_times_s: Vec::new(),
            generated_tokens: 128,
            prompt_tokens,
            outcome,
            in_flight_at_start: 1,
            drained: false,
        }
    }

    #[test]
    fn prompt_band_in_band_run_is_accepted() {
        let s: Vec<RequestSample> = (0..32)
            .map(|i| sample(i, 512, Outcome::Completed))
            .collect();
        check_prompt_band(Some(BAND_512), 256, &s).expect("512 is dead centre");
    }

    #[test]
    fn prompt_band_edges_stay_green() {
        // DISCRIMINATION. 504 and 520 are conformant W1. A run reddened here
        // is a gate that cannot pass, which is the mirror of the defect above.
        let s: Vec<RequestSample> = (0..32)
            .map(|i| sample(i, if i % 2 == 0 { 504 } else { 520 }, Outcome::Completed))
            .collect();
        check_prompt_band(Some(BAND_512), 256, &s).expect("both edges are INSIDE 512 +/- 8");
    }

    #[test]
    fn prompt_band_one_out_of_band_request_fails_the_band_naming_the_prompt() {
        let mut s: Vec<RequestSample> = (0..32)
            .map(|i| sample(i, 512, Outcome::Completed))
            .collect();
        // Issue index 269 over a 256-prompt corpus is prompt 13 -- the modulo
        // the worker loop uses, so the message points at the record an
        // operator would actually edit.
        s[7] = sample(269, 521, Outcome::Completed);
        let err = check_prompt_band(Some(BAND_512), 256, &s)
            .expect_err("521 is one token past the high edge");
        assert!(
            err.contains("prompt 13"),
            "must name the CORPUS prompt: {err}"
        );
        assert!(err.contains("521"), "must give the actual length: {err}");
        // REVERT -> GREEN.
        s[7] = sample(269, 512, Outcome::Completed);
        check_prompt_band(Some(BAND_512), 256, &s).expect("reverted run is in band");
    }

    #[test]
    fn prompt_band_ignores_non_completed_samples() {
        // DISCRIMINATION, and the one that would otherwise red every band that
        // saw a single timeout: a failed or abandoned request carries
        // `prompt_tokens: 0` BY CONSTRUCTION (band.rs's `sample_from`), not
        // because the workload was wrong.
        let mut s: Vec<RequestSample> = (0..32)
            .map(|i| sample(i, 512, Outcome::Completed))
            .collect();
        s.push(sample(99, 0, Outcome::Failed));
        s.push(sample(100, 0, Outcome::Timeout));
        s.push(sample(101, 0, Outcome::AbandonedAtDrain));
        check_prompt_band(Some(BAND_512), 256, &s)
            .expect("a failed request's structural zero is not an out-of-band prompt");
    }

    #[test]
    fn prompt_band_absent_declares_nothing_and_asserts_nothing() {
        // A built-in `--profile` corpus declares no band. Inventing 512 +/- 8
        // for it would be the fabricated-threshold defect this epic is named
        // after, and would red every non-W1 run.
        let s = vec![sample(0, 37, Outcome::Completed)];
        check_prompt_band(None, 256, &s).expect("no claim, no rule");
    }

    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    fn args<'a>(bands: &'a str, tokenization: &'a str) -> BandArgs<'a> {
        BandArgs {
            url: "http://127.0.0.1:8080",
            model: "qwen2.5-coder-1.5b-instruct",
            bands,
            replicates: 5,
            receipt: Path::new("/tmp/perf-025-receipt"),
            workload: "W1",
            host: "lambda",
            accelerator: "rtx-4090",
            quantization: "Q4_K_M",
            compute_class: "cpu",
            server_features: &[],
            tokenization,
            tokenizer_sha256: None,
            counts_special_tokens: true,
            counts_prompt_echo: false,
            commit: Some(COMMIT),
            stream: true,
            profile: "medium",
            prompts: None,
            comparator_owner: "perf-gate",
            comparator_url: None,
            comparator_model: None,
            comparator_commit: None,
            comparator_cmake: None,
            comparator_sha256: None,
            comparator_pin_expiry: None,
            key_id: None,
            keyring: None,
            witness_json: None,
            subject_binary: None,
            comparator_n_batch: None,
            comparator_n_ctx_slot: None,
            comparator_fa: None,
            comparator_kv_type: None,
        }
    }

    #[test]
    fn bands_parse_in_the_order_given() {
        assert_eq!(parse_bands("1,4,8,16").expect("valid"), vec![1, 4, 8, 16]);
        assert_eq!(parse_bands(" 2 , 3 ").expect("spaces"), vec![2, 3]);
    }

    /// `BandConfig::conformant` clamps 0 up to 1, so a zero that reaches it
    /// measures c=1 while the receipt says the operator asked for 0.
    #[test]
    fn a_zero_band_is_rejected_rather_than_clamped() {
        let err = parse_bands("1,0,4").expect_err("must reject");
        assert!(err.to_string().contains("measures nothing"), "{err}");
    }

    /// A level a `u32` cannot hold must be refused where it was typed rather
    /// than clamped into a receipt that names a band nobody ran.
    #[test]
    fn a_band_wider_than_the_receipt_field_is_rejected() {
        let too_wide = u64::from(u32::MAX) + 1;
        let err = parse_bands(&format!("1,{too_wide}")).expect_err("must reject");
        assert!(err.to_string().contains("u32"), "{err}");
        assert!(parse_bands(&format!("{}", u32::MAX)).is_ok());
    }

    #[test]
    fn an_unparseable_band_is_rejected() {
        assert!(parse_bands("1,four").is_err());
        assert!(parse_bands("").is_err());
        assert!(parse_bands(",,").is_err());
    }

    /// §4.4.6 gives `method` no default; an unknown value must not fall back.
    #[test]
    fn an_unknown_tokenization_method_is_refused() {
        let err = build_tokenization(&args("1", "guess")).expect_err("must reject");
        let msg = err.to_string();
        assert!(msg.contains("server_usage"), "{msg}");
        assert!(msg.contains("client_tokenizer"), "{msg}");
    }

    #[test]
    fn client_tokenizer_without_a_digest_is_refused() {
        let err =
            build_tokenization(&args("1", "client_tokenizer")).expect_err("digest is required");
        assert!(err.to_string().contains("tokenizer-sha256"), "{err}");
    }

    #[test]
    fn server_usage_needs_no_digest() {
        assert!(build_tokenization(&args("1", "server_usage")).is_ok());
    }

    /// Build provenance from `a`, with no server config behind it.
    fn provenance_of(a: &BandArgs<'_>) -> Result<Provenance> {
        let mut notes = Vec::new();
        build_provenance(ProvenanceInput {
            args: a,
            commit: COMMIT,
            started_utc: "2026-09-02T10:11:12.345Z",
            facts: &ServerFacts::default(),
            server_config: None,
            comparator: None,
            notes: &mut notes,
        })
    }

    /// The provenance the CLI builds must be one `bench_receipt.py` accepts: a
    /// real 64-hex digest of the binary that is actually running.
    #[test]
    fn provenance_hashes_the_running_binary() {
        let prov = provenance_of(&args("1", "server_usage")).expect("current_exe must hash");
        assert_eq!(prov.binary_sha256.len(), 64);
        assert_eq!(prov.resolution, "current_exe");
        assert_eq!(prov.client.sha256, prov.binary_sha256, "PP-25: same binary");
        assert!(prov.validate().is_ok());
    }

    #[test]
    fn an_unknown_compute_class_is_refused_where_it_was_typed() {
        let mut a = args("1", "server_usage");
        a.compute_class = "tpu";
        let err = provenance_of(&a).expect_err("must reject");
        assert!(err.to_string().contains("compute-class"), "{err}");
    }

    /// PP-2: declaring `cuda` without the server having been built with it is
    /// a claim about a path the build cannot take.
    #[test]
    fn cuda_without_the_server_feature_is_refused() {
        let mut a = args("1", "server_usage");
        a.compute_class = "cuda";
        assert!(provenance_of(&a).is_err());

        let features = vec!["cuda".to_string()];
        let mut ok = args("1", "server_usage");
        ok.compute_class = "cuda";
        ok.server_features = &features;
        provenance_of(&ok).expect("cuda declared and built is legal");
    }

    /// An empty join key must be refused before the sweep, not after it.
    #[test]
    fn a_blank_join_key_is_refused() {
        let mut a = args("1", "server_usage");
        a.host = "";
        let err = provenance_of(&a).expect_err("host is required");
        assert!(err.to_string().contains("host"), "{err}");
    }

    /// PP-30 — the receipt carries the instant the run started and the clock it
    /// came from, and both are the shape `Provenance::validate` accepts.
    #[test]
    fn provenance_carries_the_run_start_and_its_clock() {
        let prov = provenance_of(&args("1", "server_usage")).expect("builds");
        assert_eq!(prov.started_utc, "2026-09-02T10:11:12.345Z");
        assert_eq!(prov.clock_source, CLOCK_SOURCE_SYSTEM_REALTIME);
        // The producer's own clock must produce a value the validator accepts;
        // otherwise every real run fails at receipt-write time.
        let mut live = prov.clone();
        live.started_utc = now_utc_millis();
        assert!(live.validate().is_ok(), "{}", live.started_utc);
    }

    #[test]
    fn an_unknown_workload_is_refused() {
        assert!(Workload::from_str("W3").is_err());
    }

    // =====================================================================
    // PP-21 / PP-27 — what `--band` now REFUSES before it spends a run.
    //
    //  input                                     | must  | why
    //  ------------------------------------------|-------|-----------------
    //  --commit absent                           | ERR   | PP-21 staleness
    //  --commit UNPINNED (the old default)       | ERR   | names no build
    //  --commit abbreviated / uppercase          | ERR   | PP-18 ancestry
    //  --commit 40 lowercase hex     [BOUNDARY]  | OK    |
    //  --stream absent                           | ERR   | PP-27, §5.1
    //  --stream present              [BOUNDARY]  | OK    |
    // =====================================================================

    #[test]
    fn band_mode_refuses_to_run_without_a_commit() {
        let err = require_commit(None).expect_err("PP-21 needs a commit");
        assert!(err.to_string().contains("--commit"), "{err}");
        assert!(err.to_string().contains("PP-21"), "{err}");
    }

    #[test]
    fn the_unpinned_commit_default_is_refused() {
        // This literal WAS the default: `args.commit.unwrap_or("UNPINNED")`.
        for spelling in ["UNPINNED", "unpinned", "UnPinned"] {
            let err = require_commit(Some(spelling)).expect_err("{spelling} must be refused");
            assert!(err.to_string().contains("UNPINNED"), "{err}");
        }
    }

    #[test]
    fn a_commit_that_is_not_a_full_object_name_is_refused() {
        for bad in [
            "0123456",                                   // abbreviated
            "0123456789ABCDEF0123456789ABCDEF01234567",  // uppercase
            "0123456789abcdef0123456789abcdef0123456",   // 39
            "0123456789abcdef0123456789abcdef012345678", // 41
            "zzzz456789abcdef0123456789abcdef01234567",  // not hex
        ] {
            require_commit(Some(bad)).expect_err(&format!("{bad} must be refused"));
        }
        // REVERT -> GREEN.
        assert_eq!(require_commit(Some(COMMIT)).expect("valid"), COMMIT);
    }

    #[test]
    fn band_mode_refuses_to_run_without_stream() {
        let err = require_stream(false).expect_err("PP-27 requires streaming");
        assert!(err.to_string().contains("--stream"), "{err}");
        assert!(err.to_string().contains("PP-27"), "{err}");
        require_stream(true).expect("streaming is the conformant case");
    }

    // =====================================================================
    // PP-2 / PP-13 / PP-24 — reading GET /v1/effective-config.
    //
    // Every field is read from the SERVER's own report or recorded absent.
    // The must-fire cases are the ones where a plausible default would have
    // been indistinguishable from a measurement.
    // =====================================================================

    fn effective_config() -> Value {
        serde_json::json!({
            "schema_version": 1,
            "compute_class": "cuda",
            "build_features": ["cuda", "inference"],
            "server": {
                "started_utc": "2026-09-02T09:00:00.000Z",
                "build_commit": "89abcdef0123456789abcdef0123456789abcdef"
            },
            "scheduler": {"slots_admitted": 8, "max_in_flight": 8},
            "kv": {
                "bytes_used": 1024, "bytes_reserved": 4096,
                "admission_rejected": 2, "preempted_swap": 0
            },
            "model": {
                "path": "/models/qwen.gguf",
                "size_bytes": 1_073_741_824_u64,
                "content_hash": "aa".repeat(32)
            }
        })
    }

    #[test]
    fn the_effective_config_endpoint_supplies_every_server_reported_field() {
        let f = ServerFacts::read(Some(&effective_config()));
        assert_eq!(f.compute_class.as_deref(), Some("cuda"));
        assert_eq!(f.build_features, vec!["cuda", "inference"]);
        assert_eq!(f.slots_admitted, Some(8));
        assert_eq!(f.started_utc.as_deref(), Some("2026-09-02T09:00:00.000Z"));
        assert_eq!(
            f.build_commit.as_deref(),
            Some("89abcdef0123456789abcdef0123456789abcdef")
        );
        assert_eq!(
            f.kv,
            Some(KvBlock::from_server_report(1024, 4096, Some(2), Some(0))),
            "Arm D's block is the SERVER's four figures"
        );
        let m = f.model_file.expect("model file");
        assert_eq!(m.bytes, 1_073_741_824);
        assert_eq!(m.sha256, "aa".repeat(32));
    }

    #[test]
    fn a_server_that_does_not_route_the_endpoint_reports_nothing_rather_than_zero() {
        // MUST-NOT-FIRE for the whole block: an older build simply does not
        // have the route, and `get_json` returns None. Every field must then be
        // ABSENT -- a zero kv block would be read by Arm D as a measurement.
        let f = ServerFacts::read(None);
        assert_eq!(f, ServerFacts::default());
        assert!(f.kv.is_none(), "no report is not a report of zero");
        assert!(f.slots_admitted.is_none());
        assert!(f.model_file.is_none());
    }

    /// Arm D — a missing BYTE figure is no block; a missing COUNTER is a
    /// counter this server does not count, and the block survives.
    ///
    /// `apr serve` reports `admission_rejected` and `preempted_swap` as null:
    /// there is no KV-admission refusal path and no swap path, so neither has a
    /// quantity to denote (`effective_config.rs`'s `KvReport`). While all four
    /// were required, `kv` was dropped on EVERY real run — the two byte figures
    /// the server did report went with it and Arm D was permanently blind.
    ///
    /// The counters stay `None` rather than becoming `0`: Arm D reads
    /// `admission_rejected > 0` as evidence and cannot tell a counted zero from
    /// an uncounted one. `ReceiptInput::render` names them in
    /// `unproduced_fields`, and `perf_gate.sh:752` already lists a null counter
    /// as missing.
    #[test]
    fn a_missing_kv_byte_figure_is_no_block_but_an_uncounted_counter_is_null() {
        let body = effective_config();
        // MUST-FIRE: without a byte figure there is no memory report at all.
        for missing in ["bytes_used", "bytes_reserved"] {
            let mut partial = body.clone();
            partial["kv"]
                .as_object_mut()
                .expect("kv object")
                .remove(missing);
            assert!(
                ServerFacts::read(Some(&partial)).kv.is_none(),
                "kv without {missing} must be absent, not defaulted"
            );
        }
        // MUST-NOT-FIRE: an absent or explicitly null counter keeps the block,
        // with the counter recorded as uncounted.
        for absent in ["admission_rejected", "preempted_swap"] {
            for partial in [
                {
                    let mut b = body.clone();
                    b["kv"].as_object_mut().expect("kv object").remove(absent);
                    b
                },
                {
                    let mut b = body.clone();
                    b["kv"][absent] = serde_json::Value::Null;
                    b
                },
            ] {
                let kv = ServerFacts::read(Some(&partial))
                    .kv
                    .unwrap_or_else(|| panic!("kv survives an uncounted {absent}"));
                assert_eq!(
                    kv.uncounted_fields(),
                    vec![format!("kv.{absent}")],
                    "…and says which counter it is"
                );
            }
        }
        // REVERT -> GREEN: all four reported, nothing uncounted.
        let complete = ServerFacts::read(Some(&body)).kv.expect("complete block");
        assert!(complete.uncounted_fields().is_empty());
    }

    #[test]
    fn a_model_digest_that_is_not_sha256_is_reported_absent_not_reshaped() {
        let mut body = effective_config();
        body["model"]["content_hash"] = serde_json::json!("blake3:deadbeef");
        assert!(
            ServerFacts::read(Some(&body)).model_file.is_none(),
            "a digest that cannot be checked against the file is not provenance"
        );
    }

    #[test]
    fn a_compute_class_the_server_contradicts_is_refused() {
        // MUST-FIRE (PP-2): the operator says cpu, the process says cuda.
        let err = reconcile_compute_class("cpu", Some("cuda")).expect_err("must refuse");
        assert!(err.to_string().contains("PP-2"), "{err}");
        // MUST-NOT-FIRE: agreement, and silence.
        reconcile_compute_class("cuda", Some("cuda")).expect("agreement");
        reconcile_compute_class("cpu", None).expect("an old server reports nothing");
    }

    #[test]
    fn a_declared_server_feature_the_build_lacks_is_refused() {
        let reported = vec!["cpu".to_string()];
        let declared = vec!["cuda".to_string()];
        let err = reconcile_features(&declared, &reported).expect_err("must refuse");
        assert!(err.to_string().contains("cuda"), "{err}");
        reconcile_features(&declared, &[]).expect("an old server reports nothing");
        reconcile_features(&declared, &["cuda".to_string(), "x".to_string()]).expect("subset");
    }

    #[test]
    fn the_server_report_supplies_the_subject_identity_and_the_declared_one_is_named() {
        let features = vec!["cuda".to_string()];
        let mut a = args("1", "server_usage");
        a.compute_class = "cuda";
        a.server_features = &features;
        let facts = ServerFacts::read(Some(&effective_config()));
        let mut notes = Vec::new();
        let prov = build_provenance(ProvenanceInput {
            args: &a,
            commit: COMMIT,
            started_utc: "2026-09-02T10:11:12.345Z",
            facts: &facts,
            server_config: Some(effective_config()),
            comparator: None,
            notes: &mut notes,
        })
        .expect("builds");
        assert_eq!(prov.subject.feature_set, vec!["cuda", "inference"]);
        assert_eq!(
            prov.subject.commit, "89abcdef0123456789abcdef0123456789abcdef",
            "PP-18: the SERVER's build commit, not the commit under test"
        );
        assert!(prov.server_config.is_some(), "PP-2: verbatim");
        assert!(
            notes
                .iter()
                .any(|n| n.contains("PP-18") && n.contains("DECLARED")),
            "the assumed subject binary must be named: {notes:?}"
        );
        assert!(
            !notes.iter().any(|n| n.contains("source \"declared\"")),
            "with a server report nothing else is declared: {notes:?}"
        );
        assert!(
            notes.iter().any(|n| n.contains("2026-09-02T09:00:00.000Z")),
            "PP-30: the server's own start instant must be recorded beside the run's: {notes:?}"
        );
        assert!(prov.validate().is_ok());
    }

    #[test]
    fn without_the_endpoint_the_declared_inputs_are_named_in_the_notes() {
        // The other polarity: no server report, so compute_class, feature_set
        // and the subject commit are all the operator's word and each says so.
        let mut notes = Vec::new();
        let a = args("1", "server_usage");
        build_provenance(ProvenanceInput {
            args: &a,
            commit: COMMIT,
            started_utc: "2026-09-02T10:11:12.345Z",
            facts: &ServerFacts::default(),
            server_config: None,
            comparator: None,
            notes: &mut notes,
        })
        .expect("builds");
        let joined = notes.join("\n");
        assert!(joined.contains("subject.feature_set"), "{joined}");
        assert!(joined.contains("compute_class"), "{joined}");
        assert!(joined.contains("subject.commit"), "{joined}");
        assert!(joined.contains("declared"), "{joined}");
    }

    // =====================================================================
    // §5.3 / PP-22 — the comparator lane's configuration, from its /props.
    // =====================================================================

    /// llama.cpp's `GET /props` as it really answers: no `n_batch`, no
    /// `flash_attn`, no `cache_type_k`. This is why the declared flags exist.
    fn props_without_batch() -> Value {
        serde_json::json!({
            "default_generation_settings": { "n_ctx": 1024 },
            "n_ctx": 8192,
            "total_slots": 8
        })
    }

    fn props(n_batch: u32) -> Value {
        serde_json::json!({
            "total_slots": 4,
            "n_ctx": 4096,
            "default_generation_settings": {
                "n_ctx": 1024, "n_batch": n_batch,
                "flash_attn": true, "cache_type_k": "f16"
            }
        })
    }

    #[test]
    fn the_lane_configuration_is_read_from_the_comparators_own_props() {
        let lane = lane_config_from_props(Some(&props(2048)));
        assert_eq!(lane.n_ctx_slot, Some(1024), "PER-SLOT context, not total");
        assert_eq!(lane.kv_type.as_deref(), Some("f16"));
        assert_eq!(lane.fa, Some(true));
        assert_eq!(lane.n_batch, Some(2048));
        assert_eq!(comparator_slots(Some(&props(2048))), Some(4));
    }

    #[test]
    fn a_props_response_that_reports_nothing_leaves_every_lane_field_absent() {
        // PP-22: an absent field does NOT match a present one, so a guessed
        // value would silently join two different configurations.
        let lane = lane_config_from_props(Some(&serde_json::json!({})));
        assert_eq!(lane, LaneConfig::default());
        assert_eq!(lane_config_from_props(None), LaneConfig::default());
        assert_eq!(comparator_slots(None), None);
    }

    #[test]
    fn n_ctx_slot_falls_back_to_total_context_over_slots() {
        let lane = lane_config_from_props(Some(&serde_json::json!({
            "total_slots": 4, "n_ctx": 4096
        })));
        assert_eq!(lane.n_ctx_slot, Some(1024));
    }

    #[test]
    fn a_b1_comparator_is_refused_before_a_single_request_is_issued() {
        // MUST-FIRE (§5.3): `-b 1` switches llama.cpp's batching OFF and once
        // manufactured a 2.39x overstatement. `JoinKey::refuse_cripple` catches
        // it at join time -- by which point the measurement has been spent.
        let err = lane_refuse_cripple(&lane_config_from_props(Some(&props(1))))
            .expect_err("a -b 1 comparator must be refused");
        assert!(err.to_string().contains("2.39x"), "{err}");
        // REVERT -> GREEN.
        lane_refuse_cripple(&lane_config_from_props(Some(&props(2)))).expect("-b 2 serves");
    }

    // =====================================================================
    // PP-22 / §5.3 — the DECLARED comparator lane, and why it has to exist.
    // =====================================================================

    /// `--comparator-fa` is a three-valued token, and `auto` records NOTHING.
    ///
    /// A launcher that passed `-fa auto` does not know what the server resolved
    /// it to, and a guessed join-key field silently joins two different
    /// configurations — `JoinKey::refuse_mismatch` treats `None` as NOT
    /// matching a value, which is the behaviour that keeps an unknown honest.
    #[test]
    fn the_declared_flash_attention_flag_is_three_valued_and_auto_records_nothing() {
        let with = |fa: Option<&'static str>| -> Result<LaneConfig> {
            let mut a = args("1", "server_usage");
            a.comparator_fa = fa;
            declared_lane_config(&a)
        };
        assert_eq!(with(Some("on")).expect("on").fa, Some(true));
        assert_eq!(with(Some("off")).expect("off").fa, Some(false));
        assert_eq!(with(Some("auto")).expect("auto").fa, None);
        assert_eq!(with(None).expect("unset").fa, None);
        let err = with(Some("yes")).expect_err("an unknown token must be refused");
        assert!(err.to_string().contains("on, off or auto"), "{err}");
    }

    /// MUST-FIRE (§5.3), the point of the flags: a `-b 1` comparator is refused
    /// **before a single request is issued**, from the DECLARATION.
    ///
    /// llama.cpp's `GET /props` reports no `n_batch` at all, so
    /// `lane_config_from_props` left it `None` on every real run and
    /// `lane_refuse_cripple` — which asks whether `n_batch == Some(1)` — could
    /// never fire against a real llama-server. `JoinKey::refuse_cripple` at
    /// join time was equally blind for the same reason. The cripple that once
    /// manufactured a 2.39x overstatement was, in practice, unrefusable.
    #[test]
    fn a_declared_b1_comparator_is_refused_before_a_single_request_is_issued() {
        let reported = lane_config_from_props(Some(&props_without_batch()));
        assert_eq!(
            reported.n_batch, None,
            "the fixture must reproduce llama.cpp's silence, or this proves nothing"
        );

        let mut crippled = args("1", "server_usage");
        crippled.comparator_url = Some("http://127.0.0.1:8081");
        crippled.comparator_n_batch = Some(1);
        let lane = reconcile_lane(
            &declared_lane_config(&crippled).expect("declared"),
            &reported,
        )
        .expect("nothing to contradict");
        let err = lane_refuse_cripple(&lane).expect_err("-b 1 must be refused");
        assert!(err.to_string().contains("2.39x"), "{err}");

        // REVERT -> GREEN: the same run declared with `-b 2048` serves the band.
        let mut ok = args("1", "server_usage");
        ok.comparator_url = Some("http://127.0.0.1:8081");
        ok.comparator_n_batch = Some(2048);
        let lane =
            reconcile_lane(&declared_lane_config(&ok).expect("declared"), &reported).expect("ok");
        lane_refuse_cripple(&lane).expect("-b 2048 serves");
        assert_eq!(lane.n_batch, Some(2048), "and it reaches the join key");
    }

    /// PP-2 — `/props` OVERRIDES a declaration wherever it reports one, and a
    /// contradiction between the two is refused rather than resolved.
    #[test]
    fn the_servers_own_props_win_and_a_contradiction_is_refused() {
        let mut a = args("1", "server_usage");
        a.comparator_n_batch = Some(2048);
        a.comparator_n_ctx_slot = Some(1024);
        a.comparator_kv_type = Some("f16");
        a.comparator_fa = Some("on");
        let declared = declared_lane_config(&a).expect("declared");

        // Silence: the declaration stands, which is the whole point.
        let quiet = reconcile_lane(
            &declared,
            &lane_config_from_props(Some(&props_without_batch())),
        )
        .expect("nothing contradicts");
        assert_eq!(quiet.n_batch, Some(2048));
        assert_eq!(quiet.kv_type.as_deref(), Some("f16"));
        assert_eq!(quiet.fa, Some(true));

        // Agreement: the reported value is taken, indistinguishably.
        let agreeing = reconcile_lane(&declared, &lane_config_from_props(Some(&props(2048))))
            .expect("agreement");
        assert_eq!(agreeing.n_batch, Some(2048));

        // MUST-FIRE: a disagreement is refused, naming both values.
        let err = reconcile_lane(&declared, &lane_config_from_props(Some(&props(512))))
            .expect_err("2048 declared against 512 reported");
        let msg = err.to_string();
        assert!(msg.contains("--comparator-n-batch"), "{msg}");
        assert!(msg.contains("2048") && msg.contains("512"), "{msg}");
        assert!(msg.contains("PP-2"), "{msg}");
    }

    // =====================================================================
    // PP-20 — a comparator lane must carry its pin.
    // =====================================================================

    #[test]
    fn a_comparator_lane_without_its_pin_is_refused() {
        let mut a = args("1", "server_usage");
        a.comparator_url = Some("http://127.0.0.1:8081");
        let err = comparator_identity(&a, Some(props(2048))).expect_err("PP-20 needs the pin");
        let msg = err.to_string();
        for flag in [
            "--comparator-commit",
            "--comparator-cmake",
            "--comparator-sha256",
            "--comparator-pin-expiry",
        ] {
            assert!(msg.contains(flag), "must name {flag}: {msg}");
        }
    }

    #[test]
    fn a_pinned_comparator_lane_keeps_its_props_verbatim() {
        let expiry = "2026-12-01T00:00:00.000Z";
        let sha = "bb".repeat(32);
        let mut a = args("1", "server_usage");
        a.comparator_url = Some("http://127.0.0.1:8081");
        a.comparator_commit = Some("39173bcac0123456789abcdef0123456789abcde");
        a.comparator_cmake = Some("cmake -B build -DGGML_CUDA=ON");
        a.comparator_sha256 = Some(&sha);
        a.comparator_pin_expiry = Some(expiry);
        let id = comparator_identity(&a, Some(props(2048))).expect("pinned");
        assert_eq!(id.pin_expiry, expiry);
        assert_eq!(
            id.props.pointer("/default_generation_settings/n_batch"),
            Some(&serde_json::json!(2048)),
            "§5.3 stores the lane's /props verbatim"
        );
    }

    // =====================================================================
    // PP-26 — the batch-invariance witness, read from perf041's witness.json.
    // =====================================================================

    fn witness_file(dir: &std::path::Path, body: &str) -> PathBuf {
        let path = dir.join("witness.json");
        std::fs::write(&path, body).expect("write witness");
        path
    }

    #[test]
    fn the_witness_json_supplies_one_verdict_per_band() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = witness_file(
            dir.path(),
            r#"{"probe":"perf041","bands":[
                 {"c":1,"result":"PASS","divergence_at":null,"declared_min":64,"m_formed":1},
                 {"c":4,"result":"FAIL","divergence_at":0,"declared_min":64,"m_formed":4},
                 {"c":8,"result":"UNMEASURABLE","divergence_at":null,"declared_min":64,"m_formed":1}
               ]}"#,
        );
        let w = load_witness(Some(&path)).expect("loads");
        assert_eq!(w[&1].batch_invariance, BatchInvariance::Pass);
        assert!(w[&1].passed());
        assert_eq!(w[&4].batch_invariance, BatchInvariance::Fail);
        assert_eq!(w[&4].divergence_at, Some(0));
        assert_eq!(w[&4].m_formed, 4);
        assert_eq!(w[&8].batch_invariance, BatchInvariance::Unmeasurable);
        assert!(
            !w[&8].passed(),
            "Unmeasurable is on the failing side of P-4"
        );
        assert!(w[&1].source.contains("perf041"), "{}", w[&1].source);
    }

    #[test]
    fn an_unknown_witness_verdict_is_refused_rather_than_mapped() {
        // A silent fallback would turn a probe this producer does not
        // understand into a correctness verdict it never gave.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = witness_file(
            dir.path(),
            r#"{"bands":[{"c":4,"result":"SKIP","declared_min":64,"m_formed":4}]}"#,
        );
        let err = load_witness(Some(&path)).expect_err("SKIP is not a PP-26 verdict");
        assert!(err.to_string().contains("SKIP"), "{err}");
    }

    #[test]
    fn a_document_that_is_not_a_perf041_witness_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = witness_file(dir.path(), r#"{"probe":"perf041"}"#);
        let err = load_witness(Some(&path)).expect_err("no bands array");
        assert!(err.to_string().contains("bands"), "{err}");
        assert!(load_witness(None).expect("absent is legal").is_empty());
    }

    // =====================================================================
    // §12 row 7 / PP-3 — the JOIN. This is the test that was inverted.
    //
    // The rule this replaces was "the producer must NEVER emit a comparator
    // ratio", and it was right while the producer measured one lane. v3.0 §4.3
    // gives it two, interleaved, inside ONE invocation — so the rule becomes
    // "MEASURED only from a same-run baseline that passes every receipt rule;
    // UNMEASURED with an owner otherwise". Both polarities are tested here,
    // because inverting the old test alone would leave the fabrication it was
    // written to stop untested.
    // =====================================================================

    fn shell_for<'a>(a: &'a BandArgs<'a>, interleaved: bool) -> ReceiptShell<'a> {
        let mut notes = Vec::new();
        let provenance = build_provenance(ProvenanceInput {
            args: a,
            commit: COMMIT,
            started_utc: "2026-09-02T10:11:12.345Z",
            facts: &ServerFacts::default(),
            server_config: None,
            comparator: None,
            notes: &mut notes,
        })
        .expect("provenance builds");
        ReceiptShell {
            args: a,
            tokenization: build_tokenization(a).expect("tokenization"),
            provenance,
            protocol: protocol_params(a.replicates, interleaved),
            workload: Workload::W1,
            commit: COMMIT.to_string(),
            run_id: RunId::derive("2026-09-02T10:11:12.345Z", "lambda", &"cc".repeat(32), 4242),
            ladder: Ladder::derive(
                &[1, 4],
                SlotsAdmitted {
                    apr: None,
                    llama: None,
                },
            ),
            lane: LaneConfig::default(),
            witness: BTreeMap::new(),
            kv: None,
            notes,
            corpus_sha256: None,
        }
    }

    /// A streamed, live, n_predict-honouring band at `c`, with `scale` on its
    /// per-request rate so two lanes can differ by a known factor.
    fn synthetic_band(c: u32, scale: f64) -> BandInput {
        let requests: Vec<RequestOutcome> = (0..8)
            .map(|i| {
                let issued = f64::from(i) * 100.0;
                let dur = 800.0 / scale + f64::from(i);
                let ttft = dur * 0.05;
                let times: Vec<f64> = (0..128)
                    .map(|k| issued + ttft + f64::from(k) * (dur - ttft) / 128.0)
                    .collect();
                RequestOutcome {
                    issued_ms: issued,
                    settled_ms: issued + dur,
                    outcome: Outcome::Completed,
                    generated_tokens: 128,
                    prompt_tokens: 512,
                    expected_tokens: Some(128),
                    ttft_ms: Some(ttft),
                    prefill_ms: Some(dur * 0.04),
                    in_flight_at_start: c,
                    token_times_ms: times,
                }
            })
            .collect();
        BandInput::new(
            c,
            1000.0,
            requests,
            ComparatorStatus::unmeasured("perf-gate", "x"),
        )
        .n_predict(128)
        .stream_mode(StreamMode::Live)
        .witness(BatchInvarianceWitness::compare(&[1, 2, 3], &[1, 2, 3], 2))
    }

    /// MUST-NOT-FIRE, the v2.2 posture that survives: with no comparator lane
    /// the producer emits UNMEASURED with an owner and no ratio exists.
    #[test]
    fn the_comparator_is_unmeasured_without_a_comparator_url() {
        let a = args("1", "server_usage");
        assert!(a.comparator_url.is_none());
        let status = comparator_status(&a, Lane::Apr);
        assert_eq!(status.wire_token(), "UNMEASURED");
        match status {
            ComparatorStatus::Unmeasured { owner, .. } => assert_eq!(owner, "perf-gate"),
            ComparatorStatus::NotApplicable { .. } => panic!("must not be permanent"),
            ComparatorStatus::Measured(_) => panic!("one lane cannot produce a ratio"),
        }
        // And the whole band renders with no baseline and no ratios.
        let shell = shell_for(&a, false);
        let joined = shell.joined(synthetic_band(1, 1.0), None);
        assert_eq!(joined.comparator.wire_token(), "UNMEASURED");
    }

    /// MUST-NOT-FIRE for the new rule: two lanes of ONE invocation, matching
    /// keys, join and produce a ratio that is subject over comparator.
    #[test]
    fn the_comparator_is_measured_only_from_a_same_run_baseline() {
        let mut a = args("1", "server_usage");
        a.comparator_url = Some("http://127.0.0.1:8081");
        let shell = shell_for(&a, true);
        // The subject is twice as fast per request as the comparator.
        let joined = shell.joined(synthetic_band(1, 2.0), Some(synthetic_band(1, 1.0)));
        match joined.comparator {
            ComparatorStatus::Measured(join) => {
                assert_eq!(join.baseline().concurrency, 1);
                assert!(
                    join.ratios().agg.point > 1.0,
                    "the ratio is subject/comparator: {:?}",
                    join.ratios().agg
                );
                assert!(
                    join.baseline().run_id.is_some(),
                    "PP-3: the baseline names its run"
                );
            }
            other => panic!("expected MEASURED, got {}", other.wire_token()),
        }
    }

    /// MUST-FIRE (PP-3): a baseline from another run is refused. Two runs saw
    /// two thermal states, two free-VRAM figures and two schedulers.
    #[test]
    fn a_baseline_from_another_run_is_refused() {
        let mut a = args("1", "server_usage");
        a.comparator_url = Some("http://127.0.0.1:8081");
        let shell = shell_for(&a, true);
        let other = RunId::derive("2026-09-03T10:11:12.345Z", "lambda", &"cc".repeat(32), 4242);
        assert_ne!(other, shell.run_id, "the fixture must use two run ids");
        let key = shell.join_key(1);
        let err = BandInput::join_status_in(
            &synthetic_band(1, 2.0),
            &synthetic_band(1, 1.0),
            &key,
            &key,
            (&shell.run_id, &other),
            &shell.band_context(),
        )
        .expect_err("a cross-run baseline must be refused");
        assert!(err.contains("PP-3"), "{err}");
        // REVERT -> GREEN: the same two lanes under one run id join.
        BandInput::join_status_in(
            &synthetic_band(1, 2.0),
            &synthetic_band(1, 1.0),
            &key,
            &key,
            (&shell.run_id, &shell.run_id),
            &shell.band_context(),
        )
        .expect("same run joins");
    }

    /// MUST-FIRE (PP-22): the two lanes must be the same band.
    #[test]
    fn a_c4_subject_against_a_c16_comparator_is_refused() {
        let mut a = args("1", "server_usage");
        a.comparator_url = Some("http://127.0.0.1:8081");
        let shell = shell_for(&a, true);
        let err = BandInput::join_status_in(
            &synthetic_band(4, 1.0),
            &synthetic_band(16, 1.0),
            &shell.join_key(4),
            &shell.join_key(16),
            (&shell.run_id, &shell.run_id),
            &shell.band_context(),
        )
        .expect_err("c=4 against c=16 compares two offered loads");
        assert!(err.contains("band"), "{err}");
    }

    /// PP-22 — the key both lanes are compared on carries the DECLARED window,
    /// not the measured one. Two lanes never close on the same millisecond, so
    /// a key built from `BandInput::window_ms` would refuse every honest join.
    #[test]
    fn the_join_key_carries_the_declared_window_not_the_measured_one() {
        let a = args("1", "server_usage");
        let shell = shell_for(&a, true);
        assert_eq!(shell.join_key(1).window_ms, shell.protocol.window_ms);
        assert_ne!(
            shell.join_key(1).window_ms,
            synthetic_band(1, 1.0).window_ms.round() as u64,
            "the fixture's measured window must differ, or this proves nothing"
        );
    }

    // =====================================================================
    // §4.3 / §5.1 — the protocol block records what this run DID.
    // =====================================================================

    #[test]
    fn the_protocol_block_records_the_replicates_actually_run() {
        // The matrix declares a FLOOR (`replicates_min`). A receipt that copied
        // the floor would claim five replicates for a three-replicate run.
        assert_eq!(protocol_params(3, true).replicates, 3);
        assert_eq!(protocol_params(9, true).replicates, 9);
        // And the rest of the block still comes from the matrix.
        assert_eq!(
            protocol_params(3, true).window_ms,
            ProtocolParams::effective().window_ms
        );
        assert_eq!(protocol_params(3, true).n_predict, 128);
        assert!(protocol_params(3, true).sampler.ignore_eos, "PP-28");
    }

    #[test]
    fn a_single_lane_run_is_not_interleaved() {
        // §4.3's interleaving is the ALTERNATION of two lanes. A one-lane run
        // did not alternate, and copying `interleaved: true` out of the matrix
        // would put a protocol the run did not follow on the wire.
        assert!(!protocol_params(5, false).interleaved);
        assert!(protocol_params(5, true).interleaved);
    }

    /// The CLI default must not drift from §4.3's n.
    #[test]
    fn the_replicate_default_is_the_spec_constant() {
        // INVERTED from `== 3`: v3.0 §4.3 replaces v2.2 §4.4.2's N=3 with
        // n >= 5, because "n = 3 sizes an effect and bounds no variance".
        assert_eq!(default_replicates(), 5);
        assert_eq!(default_replicates(), REPLICATES);
    }

    // =====================================================================
    // The producer's own notes reach the receipt.
    // =====================================================================

    #[test]
    fn producer_notes_are_appended_to_unproduced_fields() {
        let a = args("1", "server_usage");
        let shell = shell_for(&a, false);
        let input = ReceiptInput::new(
            shell.run_id.clone(),
            shell.provenance.clone(),
            shell.tokenization.clone(),
            shell.workload,
            shell.protocol,
            shell.commit.clone(),
            Ladder::derive(
                &[1],
                SlotsAdmitted {
                    apr: None,
                    llama: None,
                },
            ),
            vec![synthetic_band(1, 1.0)],
        );
        let note = "PP-2 provenance.compute_class — source \"declared\"".to_string();
        let rendered = render_with_notes(&input, std::slice::from_ref(&note)).expect("renders");
        let value: Value = serde_json::from_str(&rendered).expect("valid JSON");
        let fields = value["unproduced_fields"]
            .as_array()
            .expect("unproduced_fields is an array");
        assert!(
            fields.iter().any(|f| f.as_str() == Some(note.as_str())),
            "the producer's note must reach the receipt: {fields:?}"
        );
        // MUST-NOT-FIRE: with no notes the rendering is the receipt's own.
        let bare = render_with_notes(&input, &[]).expect("renders");
        assert_eq!(bare, input.render_string().expect("renders"));
    }

    // =====================================================================
    // PP-24 — the ladder the run actually measures.
    //
    //  slots_admitted            | declared      | runs        | why
    //  --------------------------|---------------|-------------|-----------
    //  neither lane reported     | 1,4,8,16      | 1,4,8,16    | no evidence
    //  apr 8, llama 16           | 1,4,8,16      | 1,4,8       | min = 8
    //  apr 16, llama 4           | 1,4,8,16      | 1,4         | min = 4
    //  apr 1                     | 4,8           | (none)      | refused
    // =====================================================================

    fn ladder(apr: Option<u32>, llama: Option<u32>, declared: &[u32]) -> Ladder {
        Ladder::derive(declared, SlotsAdmitted { apr, llama })
    }

    #[test]
    fn every_declared_band_runs_when_neither_lane_reported_a_slot_count() {
        // MUST-NOT-FIRE: narrowing the ladder on no evidence would silently
        // drop bands that ran perfectly well.
        let l = ladder(None, None, &[1, 4, 8, 16]);
        assert_eq!(runnable_bands(&[1, 4, 8, 16], &l), vec![1, 4, 8, 16]);
    }

    #[test]
    fn a_band_wider_than_the_smaller_lanes_admission_is_skipped() {
        // MUST-FIRE: c=16 against a subject admitting 8 measures a queue.
        assert_eq!(
            runnable_bands(&[1, 4, 8, 16], &ladder(Some(8), Some(16), &[1, 4, 8, 16])),
            vec![1, 4, 8],
        );
        // The COMPARATOR can be the binding lane just as well as the subject.
        assert_eq!(
            runnable_bands(&[1, 4, 8, 16], &ladder(Some(16), Some(4), &[1, 4, 8, 16])),
            vec![1, 4],
        );
        // BOUNDARY: `c == cap` is admitted; `c == cap + 1` is not.
        assert_eq!(runnable_bands(&[8], &ladder(Some(8), None, &[8])), vec![8]);
        assert!(runnable_bands(&[9], &ladder(Some(8), None, &[9])).is_empty());
    }

    // =====================================================================
    // The whole wire, once: a two-lane receipt with a baseline and ratios.
    // =====================================================================

    /// PP-3 / P-5 — a joined band renders `baseline` and `ratios` on the wire,
    /// and the baseline carries the run id both lanes share.
    ///
    /// This is the shape `scripts/lib/bench_receipt.py` reads: a ratio is
    /// representable ONLY inside `ratios`, beside a `baseline` object that
    /// itself passed every receipt rule. A bare `agg_ratio` scalar beside a
    /// band -- what `perf_receipt.py` used to emit -- is unrepresentable here
    /// because `ComparatorStatus::Measured` has no public constructor.
    #[test]
    fn a_two_lane_receipt_carries_its_baseline_and_ratios_on_the_wire() {
        let mut a = args("1", "server_usage");
        a.comparator_url = Some("http://127.0.0.1:8081");
        let shell = shell_for(&a, true);
        let band = shell.joined(synthetic_band(1, 2.0), Some(synthetic_band(1, 1.0)));
        let input = ReceiptInput::new(
            shell.run_id.clone(),
            shell.provenance.clone(),
            shell.tokenization.clone(),
            shell.workload,
            shell.protocol,
            shell.commit.clone(),
            Ladder::derive(
                &[1],
                SlotsAdmitted {
                    apr: None,
                    llama: None,
                },
            ),
            vec![band],
        );
        let value: Value = serde_json::from_str(&render_with_notes(&input, &[]).expect("renders"))
            .expect("valid JSON");

        assert_eq!(value["schema_version"], 3);
        assert_eq!(value["run_id"], shell.run_id.as_str());
        assert_eq!(value["commit"], COMMIT);
        assert_eq!(value["protocol"]["interleaved"], true);
        assert_eq!(value["protocol"]["n_predict"], 128);
        assert_eq!(value["protocol"]["sampler"]["ignore_eos"], true);
        assert_eq!(value["short_of_n_predict"], 0, "PP-28: 8 of 8 at n_predict");

        let b = &value["bands"][0];
        assert_eq!(b["comparator_status"], "MEASURED");
        assert_eq!(b["status"], "MEASURED");
        assert_eq!(b["stream_mode"], "live");
        assert_eq!(b["stream_witness"]["verdict"], "live");
        assert_eq!(b["baseline"]["run_id"], shell.run_id.as_str(), "PP-3");
        assert!(
            b["baseline"]["baseline"].is_null(),
            "the baseline must not carry a baseline of its own"
        );
        assert!(
            b["ratios"]["agg"]["point"].as_f64().unwrap_or(0.0) > 1.0,
            "the ratio is subject/comparator: {}",
            b["ratios"]
        );
        assert!(b["prefill_tok_per_sec"].is_number(), "PP-4: all three");
        assert_eq!(b["prefill_source"], "server", "PP-13");
        assert!(b["decode_tok_per_sec"].is_number(), "PP-4: all three");
        assert!(b["join_key"].is_object(), "PP-22");
    }

    // =====================================================================
    // §12's spend rule — a refused join NEVER discards the measurement.
    // =====================================================================

    /// MUST-FIRE: a two-lane run with a `c > 1` band and no `--witness-json`
    /// writes a receipt.
    ///
    /// The join legitimately refuses there — an `INVALID-CORRECTNESS` band
    /// reports no throughput, so there is no `agg` to divide — but `joined`
    /// returned that `Err`, it propagated out of `write_replicate` and aborted
    /// `run_bands`. So the sweep measured every band of every replicate on both
    /// lanes and then threw the whole thing away at receipt-write time, with
    /// nothing on disk. Under §12's spend rule the measurement is gone either
    /// way; the only thing a late refusal can still destroy is the record.
    ///
    /// Nothing is fabricated to achieve this: the band renders
    /// `INVALID-CORRECTNESS`, emits NO throughput, carries no baseline and no
    /// ratio, and its comparator reason is the refusal's own text.
    #[test]
    fn an_unwitnessed_c_gt_1_band_still_writes_its_receipt() {
        let mut a = args("4", "server_usage");
        a.comparator_url = Some("http://127.0.0.1:8081");
        let shell = shell_for(&a, true);
        assert!(
            shell.witness.is_empty(),
            "no --witness-json in this fixture"
        );

        // Both lanes measured, c=4, no witness: the join has nothing to divide.
        let unwitnessed = |scale: f64| -> BandInput {
            let b = synthetic_band(4, scale);
            BandInput { witness: None, ..b }
        };
        let band = shell.joined(unwitnessed(2.0), Some(unwitnessed(1.0)));
        assert_eq!(
            band.comparator.wire_token(),
            "UNMEASURED",
            "a refused join has no baseline — but it is a posture, not an abort"
        );
        match &band.comparator {
            ComparatorStatus::Unmeasured { reason, .. } => assert!(
                reason.contains("REFUSED"),
                "the refusal's own text is the reason: {reason}"
            ),
            other => panic!("expected UNMEASURED, got {}", other.wire_token()),
        }

        let input = ReceiptInput::new(
            shell.run_id.clone(),
            shell.provenance.clone(),
            shell.tokenization.clone(),
            shell.workload,
            shell.protocol,
            shell.commit.clone(),
            Ladder::derive(
                &[4],
                SlotsAdmitted {
                    apr: None,
                    llama: None,
                },
            ),
            vec![band],
        );
        let value: Value = serde_json::from_str(
            &render_with_notes(&input, &[]).expect("THE POINT: the receipt is written"),
        )
        .expect("valid JSON");
        let b = &value["bands"][0];
        assert_eq!(b["status"], "INVALID-CORRECTNESS");
        assert!(
            b.get("aggregate_tok_per_sec").is_none() && b.get("decode_tok_per_sec").is_none(),
            "and it reports NO throughput: {b}"
        );
        assert!(b["baseline"].is_null() && b["ratios"].is_null());
    }

    /// PP-26 — the witness attaches to the SUBJECT lane and to no other.
    ///
    /// `dress` copied `witness.json` onto the comparator band too, so a
    /// subject-side PASS from `perf041_batched_parity_probe.py` — which probes
    /// `apr serve` — silently vouched for `llama-server`, a server nothing had
    /// probed. The comparator is the ORACLE the question is asked against, so
    /// its band carries `witness: null` and is exempt from the requirement.
    #[test]
    fn the_batch_invariance_witness_attaches_to_the_subject_lane_only() {
        let a = args("4", "server_usage");
        let mut shell = shell_for(&a, true);
        shell.witness.insert(
            4,
            BatchInvarianceWitness::compare(&[1, 2, 3], &[1, 2, 3], 2),
        );
        let file = SamplesFile {
            path: PathBuf::from("samples.c4.r1.jsonl.gz"),
            sha256: "aa".repeat(32),
            bytes: 10,
            rows: 8,
        };
        let bare = |c: u32| -> BandInput {
            BandInput {
                witness: None,
                ..synthetic_band(c, 1.0)
            }
        };

        let subject = shell.dress(bare(4), Lane::Apr, 0, file.clone());
        assert!(
            subject.witness.is_some(),
            "the SUBJECT is what PP-26 is a claim about"
        );
        assert_eq!(subject.role, Lane::Apr);
        let subject_band = subject.derive_in(&shell.band_context()).expect("renders");
        assert_ne!(
            subject_band.status,
            BandStatus::InvalidCorrectness,
            "…and a passing witness lets it report"
        );
        assert!(subject_band.aggregate_tok_per_sec.is_some());

        let comparator = shell.dress(bare(4), Lane::Llama, 0, file);
        assert!(
            comparator.witness.is_none(),
            "the ORACLE borrows no verdict: a subject-side PASS is not about it"
        );
        assert_eq!(comparator.role, Lane::Llama);
        let derived = comparator
            .derive_in(&shell.band_context())
            .expect("renders");
        assert_ne!(
            derived.status,
            BandStatus::InvalidCorrectness,
            "and it is not INVALID-CORRECTNESS for lacking one"
        );
        assert!(derived.aggregate_tok_per_sec.is_some());
    }

    /// §4.4.2 — the driver's protocol violations reach the BAND, not just
    /// stdout.
    ///
    /// `report_run` printed each one and dropped it. A window that closed below
    /// the `max(30, 8c)` sample floor, or a warmup that did not complete, is a
    /// fact about the measurement; `perf_gate.sh` only ever sees the receipt,
    /// so a violation the operator watched scroll past and the receipt did not
    /// carry is a receipt that reads conformant.
    #[test]
    fn a_driver_protocol_violation_reaches_the_band_and_not_only_stdout() {
        let samples: Vec<RequestSample> =
            (0..8).map(|i| sample(i, 512, Outcome::Completed)).collect();
        let run = |violations: Vec<String>| -> BandRun {
            BandRun {
                config: BandConfig::conformant(1),
                client_model: apr_test::perf_gate::ClientModel::ClosedLoop,
                tokenization: TokenizationBlock::ServerUsage {
                    counts_special_tokens: true,
                    counts_prompt_echo: false,
                },
                metrics: apr_test::perf_gate::BandMetrics::from_samples(1, &samples),
                window: apr_test::perf_gate::WindowReport {
                    requested: samples.len(),
                    window_ms: 60_000.0,
                    drain_ms: 0.0,
                    client_peak_in_flight: 1,
                    suspect: Vec::new(),
                },
                samples: samples.clone(),
                extras: vec![RequestExtra::default(); samples.len()],
                stream_mode: Some(StreamMode::Live),
                agg_ci: None,
                warmup_completed: 2,
                protocol_violations: violations,
            }
        };

        // MUST-NOT-FIRE: a clean run carries nothing.
        let clean = band_input(
            &run(Vec::new()),
            comparator_status(&args("1", "server_usage"), Lane::Apr),
        );
        assert!(clean.conformance_violations.is_empty());

        // MUST-FIRE: the violation text is on the band, and it renders.
        let text = "window closed after 8 samples, below the max(30, 8c) floor";
        let violated = band_input(
            &run(vec![text.to_string()]),
            comparator_status(&args("1", "server_usage"), Lane::Apr),
        );
        assert_eq!(violated.conformance_violations, vec![text.to_string()]);
        let derived = violated.derive().expect("renders");
        assert_eq!(derived.status, BandStatus::NonconformantValid);
        assert!(
            derived.unproduced.iter().any(|u| u.contains(text)),
            "the violation text must reach unproduced_fields: {:?}",
            derived.unproduced
        );
    }

    /// PP-3 — the comparator lane's band says it IS the comparator lane.
    ///
    /// Both lanes were given the same reason string, which on a two-lane run
    /// read "This invocation was given no --comparator-url" — false on its
    /// face, and printed beside a band measured against the comparator it
    /// claimed did not exist.
    #[test]
    fn the_comparator_reason_names_the_lane_it_is_about() {
        let mut a = args("1", "server_usage");
        let solo = comparator_status(&a, Lane::Apr);
        a.comparator_url = Some("http://127.0.0.1:8081");
        let subject = comparator_status(&a, Lane::Apr);
        let oracle = comparator_status(&a, Lane::Llama);

        let reason = |s: &ComparatorStatus| match s {
            ComparatorStatus::Unmeasured { reason, .. } => reason.clone(),
            other => panic!("expected UNMEASURED, got {}", other.wire_token()),
        };
        assert!(
            reason(&solo).contains("no --comparator-url"),
            "{}",
            reason(&solo)
        );
        assert!(
            !reason(&subject).contains("no --comparator-url"),
            "a two-lane run must not claim it was given none: {}",
            reason(&subject)
        );
        assert!(
            reason(&subject).contains("http://127.0.0.1:8081"),
            "{}",
            reason(&subject)
        );
        assert!(
            reason(&oracle).contains("IS the comparator lane"),
            "{}",
            reason(&oracle)
        );
    }

    /// PP-33 — the protocol block's source travels with it.
    ///
    /// `ProtocolParams::effective()` swallowed the matrix error and nothing
    /// called `source()`, so a checkout whose matrix did not parse put the
    /// compiled-in Rust constants on the wire under a `protocol:` block the
    /// receipt presented as `perf-matrix.yaml`'s.
    #[test]
    fn the_protocol_block_says_where_it_came_from() {
        let (params, source) = protocol_params_with_source(5, true);
        assert_eq!(params.replicates, 5, "the run's own count, not the floor");
        assert!(params.interleaved);
        assert!(
            !source.announcement().is_empty(),
            "the producer prints one line either way"
        );
        match &source {
            ProtocolSource::Matrix => assert!(
                source.unproduced_note().is_none(),
                "the matrix supplied them; nothing is unproduced"
            ),
            ProtocolSource::SpecFallback(_) => {
                let note = source.unproduced_note().expect("a fallback is unproduced");
                assert!(note.contains("PP-33"), "{note}");
            }
        }
        // Whichever the shipped matrix does, the fallback's note is the thing
        // that reaches `unproduced_fields`, so it is asserted directly.
        let fallback = ProtocolSource::SpecFallback("the matrix does not parse".to_string());
        assert!(fallback
            .announcement()
            .contains("spec fallback because the matrix does not parse"));
        assert!(fallback
            .unproduced_note()
            .expect("note")
            .contains("NOT scripts/perf-matrix.yaml"));
    }

    /// PP-3 — `provenance.client.pid` is on the receipt, so the `run_id` it
    /// states can be recomputed from its own contents.
    #[test]
    fn the_client_pid_reaches_the_receipt_and_the_run_id_recomputes() {
        let a = args("1", "server_usage");
        let prov = provenance_of(&a).expect("provenance builds");
        assert_eq!(
            prov.client.pid,
            std::process::id(),
            "the receipt names the process that measured"
        );
        let derived = RunId::derive(
            &prov.started_utc,
            &prov.host,
            &prov.client.sha256,
            prov.client.pid,
        );
        assert_eq!(
            derived.as_str().len(),
            32,
            "and every input to the id is on the wire"
        );
    }

    /// The other polarity of the same render: one lane, no baseline, no ratios,
    /// and a status that says the band is a record rather than a baseline.
    #[test]
    fn a_one_lane_receipt_carries_no_ratio_at_all() {
        let a = args("1", "server_usage");
        let shell = shell_for(&a, false);
        let band = shell.joined(synthetic_band(1, 1.0), None);
        let input = ReceiptInput::new(
            shell.run_id.clone(),
            shell.provenance.clone(),
            shell.tokenization.clone(),
            shell.workload,
            shell.protocol,
            shell.commit.clone(),
            Ladder::derive(
                &[1],
                SlotsAdmitted {
                    apr: None,
                    llama: None,
                },
            ),
            vec![band],
        );
        let value: Value = serde_json::from_str(&render_with_notes(&input, &[]).expect("renders"))
            .expect("valid JSON");
        let b = &value["bands"][0];
        assert_eq!(b["comparator_status"], "UNMEASURED");
        assert!(b["baseline"].is_null());
        assert!(b["ratios"].is_null());
        assert_eq!(value["protocol"]["interleaved"], false);
        assert_eq!(
            b["status"], "NONCONFORMANT-VALID",
            "§4.3: a run that did not alternate two lanes did not interleave"
        );
    }
}
