//! Standardized prompt profiles for LLM benchmarking.
//!
//! Provides deterministic prompt sets with calibrated input/output token counts
//! for reproducible benchmarks. Follows HuggingFace Inference-Benchmarker
//! methodology with fixed-length profiles.

use super::client::{ChatMessage, ChatRequest, Role};
use std::path::Path;

/// Standardized prompt profiles for benchmarking.
///
/// Each profile targets a specific input/output token count to ensure
/// comparable results across backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptProfile {
    /// ~10 input tokens, max_tokens=1. TTFT-only measurement (prefill speed).
    Micro,
    /// ~32 input tokens, max_tokens=32. Quick latency check.
    Short,
    /// ~128 input tokens, max_tokens=128. Standard comparison (default).
    Medium,
    /// ~512 input tokens, max_tokens=256. Sustained decode measurement.
    Long,
}

impl PromptProfile {
    /// Parse a profile name (case-insensitive).
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "micro" => Some(Self::Micro),
            "short" => Some(Self::Short),
            "medium" => Some(Self::Medium),
            "long" => Some(Self::Long),
            _ => None,
        }
    }
}

/// Load the built-in prompts for a given profile.
///
/// All prompts use temperature=0.0 for deterministic output.
pub fn load_profile(profile: PromptProfile) -> Vec<ChatRequest> {
    match profile {
        PromptProfile::Micro => vec![micro_prompt()],
        PromptProfile::Short => vec![short_prompt()],
        PromptProfile::Medium => vec![medium_prompt()],
        PromptProfile::Long => vec![long_prompt()],
    }
}

/// Load a benchmark prompt corpus from a **JSONL** file.
///
/// # The format is JSONL, and this is the file that says so
///
/// One JSON object per line, no enclosing array, no YAML:
///
/// ```jsonl
/// {"_meta":{"corpus":"W1","provenance":"SYNTHETIC ..."}}
/// {"id":0,"prompt":"// w1-0000\nlet mut ...","max_tokens":128,"temperature":0.0,"seed":0,"target_prompt_tokens":512}
/// ```
///
/// `APR-PERF-GATE-001` v2.2 §4.3.1 and §4.3.2 both name a `.jsonl` corpus, and
/// `prompts-w2.jsonl` has been on main in that shape since W2 landed. This
/// loader previously parsed YAML with a top-level `prompts:` key and the CLI
/// help advertised "a JSON array" — three formats, no two matching, and the
/// consequence was that **the only committed corpus in the tree could not be
/// read by the only loader in the tree.** The spec is the authority; the two
/// implementations were the drift, so both moved to JSONL.
///
/// # Records
///
/// `prompt` is the only required field. `id` and `target_prompt_tokens` are
/// carried by the corpus for the receipt and ignored here. Unknown fields are
/// **rejected**, not ignored: a record that says `content:` (the old YAML key)
/// or `max_token:` would otherwise load with a silently defaulted budget, and
/// a benchmark whose generation budget silently defaulted is measuring
/// something other than what its receipt claims.
///
/// # Errors
///
/// Every failure names JSONL and the offending line. In particular an empty
/// corpus is an error, never an empty `Vec` — a load-test that issues zero
/// requests and reports success is the vacuous pass this gate exists to
/// remove.
pub fn load_from_file(path: &Path) -> Result<Vec<ChatRequest>, String> {
    load_corpus(path).map(|c| c.requests)
}

/// Load a corpus **and the invariants its own `_meta` header declares**.
///
/// # The defect this closes (PERF-056, #2778)
///
/// `prompts-w1.jsonl`'s `_meta` block promises, in prose, that the corpus is
/// 256 distinct prompts of a fixed shape at `target_prompt_tokens = 512`
/// with `tolerance_tokens = 8`, `max_tokens = 128`, `temperature = 0.0` and
/// `seed = 0`. The loader **discarded that block entirely** — it parsed the
/// line only far enough to notice the `_meta` key and skip it. Every promise
/// in it was therefore unenforced, and a corpus hand-edited (or regenerated
/// with different flags) out of agreement with its own header would have
/// loaded silently and been measured as W1.
///
/// A `_meta` block right about one property and unchecked about another is
/// worse than one claiming nothing, because a reader believes it.
///
/// # What is enforced here, and what is not
///
/// Everything checkable **without a tokenizer** is enforced at load:
/// `count`, per-record agreement with `_meta` on `max_tokens`, `temperature`,
/// `seed` and `target_prompt_tokens`, `prompts_distinct`, and — when the
/// generator recorded a `body_words` — the whitespace-word shape of every
/// prompt.
///
/// **A word count is not a token count**, and this function does not pretend
/// otherwise. §4.3.1's `512 ± 8` is a property of the *model's* tokenizer, and
/// no tokenizer is reachable here: the corpus stores raw text and the model is
/// a GGUF that is not in this tree. The band is therefore *returned*, in
/// [`Corpus::band`], and asserted against real per-request counts by
/// [`assert_prompt_tokens_in_band`] at measurement time — which is exactly
/// what `_meta.token_count_note` says happens, and until PERF-056 did not.
///
/// # Errors
/// As [`load_from_file`], plus any disagreement between the `_meta` header and
/// the records that follow it. Every such message names the offending prompt.
pub fn load_corpus(path: &Path) -> Result<Corpus, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    parse_corpus(&content).map_err(|e| format!("{}: {e}", path.display()))
}

/// A loaded corpus: the requests, and the §4.3.1 band its header declared.
#[derive(Debug, Clone)]
pub struct Corpus {
    /// The prompts, in file order.
    pub requests: Vec<ChatRequest>,
    /// `target_prompt_tokens ± tolerance_tokens`, when the corpus declared
    /// both. `None` for a corpus with no `_meta` header (W2 has none) or one
    /// that declares no band — such a corpus imposes no length invariant, and
    /// inventing one for it would be the fabricated-threshold defect this
    /// epic is named after.
    pub band: Option<PromptTokenBand>,
}

/// §4.3.1's prompt-length invariant, exactly as a corpus declares it.
///
/// Half-open ranges and off-by-ones are how a `± 8` band silently becomes a
/// `± 7` one, so the edges are named and tested rather than inlined at each
/// use: `contains` is INCLUSIVE at both ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptTokenBand {
    /// `_meta.target_prompt_tokens` — 512 for W1.
    pub target: u32,
    /// `_meta.tolerance_tokens` — 8 for W1.
    pub tolerance: u32,
}

impl PromptTokenBand {
    /// Inclusive lower edge.
    #[must_use]
    pub const fn lo(self) -> u32 {
        self.target.saturating_sub(self.tolerance)
    }

    /// Inclusive upper edge.
    #[must_use]
    pub const fn hi(self) -> u32 {
        self.target.saturating_add(self.tolerance)
    }

    /// True when `n` is inside the band. Both edges are IN.
    #[must_use]
    pub const fn contains(self, n: u32) -> bool {
        n >= self.lo() && n <= self.hi()
    }
}

impl std::fmt::Display for PromptTokenBand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\u{a7}4.3.1 prompt_tokens = {} +/- {} (inclusive {}..={})",
            self.target,
            self.tolerance,
            self.lo(),
            self.hi()
        )
    }
}

/// Assert every observed prompt-token count sits inside the declared band.
///
/// `observed` is `(prompt index in the corpus, prompt_tokens the server
/// reported)`, one entry per **completed** sampled request. This is the
/// assertion `_meta.token_count_note` promised and nothing made:
///
/// > "The 512 +/-8 of 4.3.1 is asserted by the harness against the model's own
/// > tokenizer at measurement time"
///
/// # Why a failure names the prompt
///
/// The remedy differs per cause and the number alone does not distinguish
/// them. A corpus 40 tokens long everywhere means `--body-words` needs
/// retuning and the corpus regenerating; one prompt out of 256 out of band
/// means that record was edited. A bare "out of band" sends the reader to
/// re-run the measurement, which under I-9 is the one thing that cannot fix it.
///
/// # Errors
/// When any observed count is outside the band, when no completed request
/// reported a count at all, or when every count is zero — which is what a
/// server emitting no `usage` block looks like, and is an instrumentation gap
/// rather than an out-of-band corpus. The three are separate messages because
/// they have three different remedies.
pub fn assert_prompt_tokens_in_band(
    band: PromptTokenBand,
    observed: &[(usize, u32)],
) -> Result<(), String> {
    if observed.is_empty() {
        return Err(format!(
            "{band} could not be checked: no completed request reported a prompt-token              count. An unverifiable invariant is not a satisfied one"
        ));
    }
    if observed.iter().all(|&(_, n)| n == 0) {
        return Err(format!(
            "{band} could not be checked: all {} completed requests reported              prompt_tokens = 0, which is what a server that emits no `usage` block looks              like. This is an INSTRUMENTATION gap in the server or the transport, not a              corpus that is out of band -- re-measuring will not change it",
            observed.len()
        ));
    }
    let bad: Vec<(usize, u32)> = observed
        .iter()
        .copied()
        .filter(|&(_, n)| !band.contains(n))
        .collect();
    if bad.is_empty() {
        return Ok(());
    }
    let shown: Vec<String> = bad
        .iter()
        .take(8)
        .map(|&(i, n)| format!("prompt {i}: {n} tokens"))
        .collect();
    let more = if bad.len() > shown.len() {
        format!(", and {} more", bad.len() - shown.len())
    } else {
        String::new()
    };
    Err(format!(
        "{band}: {} of {} completed requests are OUTSIDE the band -- {}{more}. \
         Retune `scripts/gen_prompts_w1.py --body-words` and regenerate the corpus; \
         this is a workload defect, and the band measured under it is not W1",
        bad.len(),
        observed.len(),
        shown.join("; ")
    ))
}

/// Parse a JSONL corpus body, discarding the `_meta` block. Kept so the
/// pre-PERF-056 format-contract table below still binds to the shape rules
/// alone; every production path goes through [`parse_corpus`].
#[cfg(test)]
fn parse_jsonl(content: &str) -> Result<Vec<ChatRequest>, String> {
    parse_corpus(content).map(|c| c.requests)
}

/// The body of [`load_corpus`]: format contract, then `_meta` contract.
fn parse_corpus(content: &str) -> Result<Corpus, String> {
    reject_non_jsonl_shape(content)?;

    let mut meta: Option<CorpusMeta> = None;
    let mut records: Vec<PromptRecord> = Vec::new();
    for (idx, raw) in content.lines().enumerate() {
        let lineno = idx + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if is_meta_line(line) {
            // The corpus's own provenance header. Allowed on line 1 only, so a
            // `_meta` object cannot hide in the middle of a corpus and quietly
            // shrink the workload by one request.
            if lineno == 1 {
                meta = Some(parse_meta(line)?);
                continue;
            }
            return Err(format!(
                "line {lineno}: a `_meta` header record is only allowed on line 1 \
                 of a JSONL corpus"
            ));
        }
        let record: PromptRecord = serde_json::from_str(line).map_err(|e| {
            format!("line {lineno}: expected one JSON object per line (JSONL), got: {e}")
        })?;
        records.push(record);
    }

    if records.is_empty() {
        return Err(
            "contains no prompt records — a benchmark corpus that loads zero prompts \
             would report success having issued no requests"
                .to_string(),
        );
    }
    let band = match &meta {
        Some(m) => {
            enforce_meta_contract(m, &records)?;
            m.band()
        }
        None => None,
    };
    Ok(Corpus {
        requests: records
            .into_iter()
            .map(PromptRecord::into_request)
            .collect(),
        band,
    })
}

/// A `// w1-NNNN\n` header is two whitespace-delimited words (`//` and
/// `w1-NNNN`). Named because it is the one place the load-time shape check is
/// coupled to `scripts/gen_prompts_w1.py`'s header format.
const HEADER_WHITESPACE_WORDS: usize = 2;

/// Deserialize the line-1 `_meta` header.
///
/// Unknown fields are ALLOWED here, unlike [`PromptRecord`]: the header
/// deliberately carries prose (`provenance`, `distinctness_rationale`,
/// `template_boundary_open`) that no code reads and every reader does.
fn parse_meta(line: &str) -> Result<CorpusMeta, String> {
    #[derive(serde::Deserialize)]
    struct MetaLine {
        #[serde(rename = "_meta")]
        meta: CorpusMeta,
    }
    serde_json::from_str::<MetaLine>(line)
        .map(|m| m.meta)
        .map_err(|e| format!("line 1: `_meta` header is not readable: {e}"))
}

/// The enforceable half of the `_meta` header.
#[derive(Debug, Default, serde::Deserialize)]
struct CorpusMeta {
    count: Option<usize>,
    body_words: Option<usize>,
    target_prompt_tokens: Option<u32>,
    tolerance_tokens: Option<u32>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    seed: Option<u64>,
    /// §5.1 / PP-28 — W1 pins `ignore_eos: true` so `completion_tokens ==
    /// n_predict` on every retained sample. Without the header field the
    /// corpus could not DECLARE the pin, so nothing could check it: a record
    /// that quietly dropped `ignore_eos` produced a band whose work per request
    /// was whatever the model decided to stop after, and an Arm A floor
    /// committed over that drifts with the model's stopping behaviour rather
    /// than with the server's throughput.
    ignore_eos: Option<bool>,
    prompts_distinct: Option<bool>,
}

impl CorpusMeta {
    /// The declared band, when BOTH edges are declared.
    ///
    /// A target with no tolerance is not a band: reading the missing tolerance
    /// as `0` would turn `512` into an exact-equality assertion no real
    /// tokenizer satisfies, and defaulting it to `8` would invent §4.3.1's
    /// number in a file that did not state it.
    fn band(&self) -> Option<PromptTokenBand> {
        match (self.target_prompt_tokens, self.tolerance_tokens) {
            (Some(target), Some(tolerance)) => Some(PromptTokenBand { target, tolerance }),
            _ => None,
        }
    }
}

/// Enforce every promise the `_meta` header makes that is checkable without a
/// tokenizer. See [`load_corpus`] for what is deliberately NOT checked here.
///
/// Split into one predicate per promise rather than one long chain: the header
/// gains fields over time, and a single function that grows a branch per field
/// is how the fifth one gets added without a case in the table below it.
fn enforce_meta_contract(meta: &CorpusMeta, records: &[PromptRecord]) -> Result<(), String> {
    check_count(meta, records)?;
    for (i, r) in records.iter().enumerate() {
        check_record(meta, i, r)?;
    }
    check_distinct(meta, records)
}

/// `_meta.count` against the records that followed it.
fn check_count(meta: &CorpusMeta, records: &[PromptRecord]) -> Result<(), String> {
    match meta.count {
        Some(want) if want != records.len() => Err(format!(
            "`_meta.count` = {want} but the corpus carries {} prompt records",
            records.len()
        )),
        _ => Ok(()),
    }
}

/// One record against the header. `who` names the record the way the FILE
/// names it, so the message points at a line an operator can find.
fn check_record(meta: &CorpusMeta, i: usize, r: &PromptRecord) -> Result<(), String> {
    let who =
        r.id.map_or_else(|| format!("record {}", i + 1), |v| format!("prompt {v}"));
    agree(&who, "max_tokens", meta.max_tokens, r.max_tokens)?;
    agree(&who, "seed", meta.seed, r.seed)?;
    agree(&who, "ignore_eos", meta.ignore_eos, r.ignore_eos)?;
    agree(
        &who,
        "target_prompt_tokens",
        meta.target_prompt_tokens,
        r.target_prompt_tokens,
    )?;
    check_temperature(&who, meta.temperature, r.temperature)?;
    check_body_words(&who, meta.body_words, &r.prompt)
}

/// A record field that must equal the header's, when both are present. A
/// header that declares nothing constrains nothing — silence is not a claim.
fn agree<T: PartialEq + std::fmt::Display>(
    who: &str,
    field: &str,
    want: Option<T>,
    got: Option<T>,
) -> Result<(), String> {
    match (want, got) {
        (Some(w), Some(g)) if w != g => {
            Err(format!("{who}: {field} = {g} but `_meta.{field}` = {w}"))
        }
        _ => Ok(()),
    }
}

/// As [`agree`], but for the one field that is a float. `==` on `f64` is what
/// the workspace lints allow here and would still be wrong: `0.0` written five
/// ways must compare equal.
fn check_temperature(who: &str, want: Option<f64>, got: Option<f64>) -> Result<(), String> {
    match (want, got) {
        (Some(w), Some(g)) if (w - g).abs() > f64::EPSILON => Err(format!(
            "{who}: temperature = {g} but `_meta.temperature` = {w}"
        )),
        _ => Ok(()),
    }
}

/// The whitespace-word shape `scripts/gen_prompts_w1.py` recorded.
///
/// THE ONLY LENGTH CHECK POSSIBLE BEFORE A TOKENIZER EXISTS, and it is not
/// §4.3.1's band — see [`load_corpus`] and [`assert_prompt_tokens_in_band`].
fn check_body_words(who: &str, body_words: Option<usize>, prompt: &str) -> Result<(), String> {
    let Some(body_words) = body_words else {
        return Ok(());
    };
    let want = body_words + HEADER_WHITESPACE_WORDS;
    let got = prompt.split_whitespace().count();
    if want == got {
        return Ok(());
    }
    Err(format!(
        "{who}: {got} whitespace-delimited words, but `_meta.body_words` = \
         {body_words} plus a {HEADER_WHITESPACE_WORDS}-word header declares \
         {want}. A word count is NOT a token count and this is not \
         \u{a7}4.3.1's band — it is the corpus failing to have the shape its own \
         header describes, which is the only length check possible before a \
         tokenizer exists"
    ))
}

/// `_meta.prompts_distinct`, which until PERF-056 was enforced by exactly one
/// test against exactly one committed file and by no loader at all.
fn check_distinct(meta: &CorpusMeta, records: &[PromptRecord]) -> Result<(), String> {
    if meta.prompts_distinct != Some(true) {
        return Ok(());
    }
    let mut seen: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::with_capacity(records.len());
    for (i, r) in records.iter().enumerate() {
        if let Some(&first) = seen.get(r.prompt.as_str()) {
            return Err(format!(
                "`_meta.prompts_distinct` is true but records {} and {} carry the same \
                 prompt. Identical prompts would let prefix caching, not the scheduler, \
                 drive Arm A's scaling_efficiency — agg(c) would rise with c for a \
                 reason that is not batching",
                first + 1,
                i + 1
            ));
        }
        seen.insert(r.prompt.as_str(), i);
    }
    Ok(())
}

/// True when `line` is a JSONL provenance header (`{"_meta": ...}`).
fn is_meta_line(line: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(line)
        .ok()
        .and_then(|v| v.as_object().map(|o| o.contains_key("_meta")))
        .unwrap_or(false)
}

/// Refuse the two formats this loader used to be confused with, naming JSONL.
///
/// Without this, a YAML corpus fails with a serde_json message about line 1
/// and a JSON array fails with one about a trailing bracket — both true, and
/// neither tells the operator that the format they wrote is not the format
/// this file wants.
fn reject_non_jsonl_shape(content: &str) -> Result<(), String> {
    let Some(first) = content.lines().map(str::trim).find(|l| !l.is_empty()) else {
        return Err("is empty — expected JSONL, one JSON object per line".to_string());
    };
    if first.starts_with('[') {
        return Err(
            "looks like a JSON array — expected JSONL: one JSON object per line, \
             with no enclosing `[ ]` and no commas between records"
                .to_string(),
        );
    }
    if !first.starts_with('{') {
        return Err(format!(
            "line 1 does not begin a JSON object (it begins {:?}) — expected JSONL: \
             one JSON object per line. The YAML `prompts:` corpus format this loader \
             used to accept is no longer supported; regenerate as JSONL",
            first.chars().take(16).collect::<String>()
        ));
    }
    Ok(())
}

/// One record of a JSONL prompt corpus.
///
/// `deny_unknown_fields` is deliberate — see [`load_from_file`].
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptRecord {
    /// The prompt text. The only required field.
    prompt: String,
    /// Message role. Defaults to `user`.
    role: Option<String>,
    /// Generation budget for this request.
    max_tokens: Option<u32>,
    /// Sampling temperature. Defaults to `0.0` (greedy).
    temperature: Option<f64>,
    /// Sampling seed, so the run is reproducible (§4.4.4).
    seed: Option<u64>,
    /// Suppress end-of-sequence stopping, so the token count per request is
    /// pinned to `max_tokens` (§4.3.1).
    ignore_eos: Option<bool>,
    /// Recorded by the corpus for the receipt, and used to NAME this record in
    /// any `_meta`-contract failure. A message that says "record 137" when the
    /// file says `"id":136` costs the reader the one lookup the message existed
    /// to save.
    id: Option<u64>,
    /// Recorded by the corpus. Checked against `_meta.target_prompt_tokens` at
    /// load (agreement, not length); the actual TOKEN count is measured by the
    /// harness against the model's own tokenizer and asserted by
    /// [`assert_prompt_tokens_in_band`], because no tokenizer is reachable here.
    target_prompt_tokens: Option<u32>,
}

impl PromptRecord {
    fn into_request(self) -> ChatRequest {
        ChatRequest {
            model: String::new(),
            messages: vec![ChatMessage {
                role: self.role.as_deref().map_or(Role::User, parse_role),
                content: self.prompt,
            }],
            temperature: Some(self.temperature.unwrap_or(0.0)),
            max_tokens: self.max_tokens,
            stream: Some(false),
            seed: self.seed,
            ignore_eos: self.ignore_eos,
            // PP-27: `LlmClient::wire_request` sets this when the request
            // actually streams; a corpus record never asks for it.
            stream_options: None,
        }
    }
}

fn parse_role(s: &str) -> Role {
    match s.to_lowercase().as_str() {
        "system" => Role::System,
        "assistant" => Role::Assistant,
        _ => Role::User,
    }
}

// --- Built-in prompt profiles ---

fn micro_prompt() -> ChatRequest {
    ChatRequest {
        model: String::new(),
        messages: vec![ChatMessage {
            role: Role::User,
            content: "Say hello.".to_string(),
        }],
        temperature: Some(0.0),
        max_tokens: Some(1),
        stream: Some(false),
        seed: None,
        ignore_eos: None,
        // PP-27: set on the wire by `LlmClient::wire_request` when (and only
        // when) the request actually streams; a profile never asks for it.
        stream_options: None,
    }
}

fn short_prompt() -> ChatRequest {
    ChatRequest {
        model: String::new(),
        messages: vec![ChatMessage {
            role: Role::User,
            content: "Explain what a hash table is and why it provides O(1) average lookup time."
                .to_string(),
        }],
        temperature: Some(0.0),
        max_tokens: Some(32),
        stream: Some(false),
        seed: None,
        ignore_eos: None,
        // PP-27: set on the wire by `LlmClient::wire_request` when (and only
        // when) the request actually streams; a profile never asks for it.
        stream_options: None,
    }
}

fn medium_prompt() -> ChatRequest {
    ChatRequest {
        model: String::new(),
        messages: vec![ChatMessage {
            role: Role::User,
            content: "Write a detailed explanation of how binary search works, \
                      including its time complexity, when to use it, and common \
                      pitfalls. Include a step-by-step example with the array \
                      [2, 5, 8, 12, 16, 23, 38, 56, 72, 91] searching for 23. \
                      Explain why the algorithm requires a sorted array and what \
                      happens if the array is unsorted. Discuss the difference \
                      between iterative and recursive implementations and their \
                      respective trade-offs in terms of stack usage and performance."
                .to_string(),
        }],
        temperature: Some(0.0),
        max_tokens: Some(128),
        stream: Some(false),
        seed: None,
        ignore_eos: None,
        // PP-27: set on the wire by `LlmClient::wire_request` when (and only
        // when) the request actually streams; a profile never asks for it.
        stream_options: None,
    }
}

fn long_prompt() -> ChatRequest {
    ChatRequest {
        model: String::new(),
        messages: vec![ChatMessage {
            role: Role::User,
            content: "You are a systems programming expert. Write a comprehensive \
                      guide on implementing a memory allocator in Rust. Cover the \
                      following topics in detail:\n\n\
                      1. The difference between stack and heap allocation, including \
                      how the operating system manages virtual memory pages and how \
                      the brk/mmap system calls work on Linux.\n\n\
                      2. The design of a simple bump allocator, including its \
                      advantages (fast allocation, no fragmentation tracking) and \
                      disadvantages (no individual deallocation, memory waste).\n\n\
                      3. The free list allocator design pattern, explaining how \
                      freed blocks are tracked, how coalescing adjacent free blocks \
                      works, and the trade-offs between first-fit, best-fit, and \
                      worst-fit allocation strategies.\n\n\
                      4. The buddy system allocator, explaining how power-of-two \
                      block sizes enable efficient splitting and merging, and how \
                      this approach reduces external fragmentation at the cost of \
                      internal fragmentation.\n\n\
                      5. How Rust's ownership system and the GlobalAlloc trait \
                      interact with custom allocators. Show how to implement the \
                      GlobalAlloc trait and register a custom allocator using \
                      #[global_allocator].\n\n\
                      6. Thread safety considerations: how to make an allocator \
                      thread-safe using Mutex or atomic operations, and the \
                      performance implications of lock contention in multi-threaded \
                      workloads. Discuss arena-per-thread strategies.\n\n\
                      7. Real-world allocator designs like jemalloc, mimalloc, and \
                      tcmalloc. Explain their key innovations and when you would \
                      choose one over another.\n\n\
                      Include code examples for each allocator type."
                .to_string(),
        }],
        temperature: Some(0.0),
        max_tokens: Some(256),
        stream: Some(false),
        seed: None,
        ignore_eos: None,
        // PP-27: set on the wire by `LlmClient::wire_request` when (and only
        // when) the request actually streams; a profile never asks for it.
        stream_options: None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_from_name() {
        assert_eq!(
            PromptProfile::from_name("micro"),
            Some(PromptProfile::Micro)
        );
        assert_eq!(
            PromptProfile::from_name("short"),
            Some(PromptProfile::Short)
        );
        assert_eq!(
            PromptProfile::from_name("medium"),
            Some(PromptProfile::Medium)
        );
        assert_eq!(PromptProfile::from_name("long"), Some(PromptProfile::Long));
        assert_eq!(
            PromptProfile::from_name("MEDIUM"),
            Some(PromptProfile::Medium)
        );
        assert_eq!(PromptProfile::from_name("unknown"), None);
    }

    #[test]
    fn test_micro_profile() {
        let prompts = load_profile(PromptProfile::Micro);
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].max_tokens, Some(1));
        assert_eq!(prompts[0].temperature, Some(0.0));
    }

    #[test]
    fn test_short_profile() {
        let prompts = load_profile(PromptProfile::Short);
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].max_tokens, Some(32));
    }

    #[test]
    fn test_medium_profile() {
        let prompts = load_profile(PromptProfile::Medium);
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].max_tokens, Some(128));
    }

    #[test]
    fn test_long_profile() {
        let prompts = load_profile(PromptProfile::Long);
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].max_tokens, Some(256));
    }

    // ---------------------------------------------------------------------
    // JSONL format contract (PERF-039).
    //
    // MUTATION TABLE. Every row is a corpus in a format this loader must
    // REFUSE, and each asserts that the refusal message NAMES the expected
    // format. The defect being excluded is not "wrong format accepted" alone —
    // it is a loader that reads zero prompts and lets the caller proceed to
    // report a successful benchmark having issued no requests. The rows marked
    // "was silently ..." describe what the pre-PERF-039 YAML loader did with
    // that same input.
    //
    //  input shape                          | must | message must name
    //  -------------------------------------|------|------------------------
    //  JSON array `[{...}]`                  | ERR  | "JSONL"
    //  YAML `prompts:` (the OLD format)      | ERR  | "JSONL"
    //  empty file                            | ERR  | "JSONL"
    //  whitespace-only file                  | ERR  | "JSONL"
    //  well-formed JSONL, zero records       | ERR  | "zero prompts"
    //  `_meta` header alone, no records      | ERR  | "zero prompts"
    //  `_meta` on a line other than 1        | ERR  | "line 1"
    //  record missing `prompt`               | ERR  | line number
    //  record with `content:` (old YAML key) | ERR  | line number
    //  record with a typo'd field            | ERR  | line number
    //  the committed prompts-w1.jsonl        | OK   | 256 records
    // ---------------------------------------------------------------------

    /// Assert `body` is refused and the message mentions `needle`.
    fn refused(body: &str, needle: &str) -> String {
        let err = parse_jsonl(body)
            .err()
            .unwrap_or_else(|| panic!("must be REFUSED, was accepted: {body:?}"));
        assert!(
            err.to_lowercase().contains(&needle.to_lowercase()),
            "message must name {needle:?}, got: {err}"
        );
        err
    }

    #[test]
    fn jsonl_array_is_refused_naming_jsonl() {
        // Binds to the SHAPE guard, not to the per-line parser: with the guard
        // disarmed a `[` line is still refused, just by a message that does not
        // tell the operator their whole file is the wrong shape. Asserting the
        // generic word "JSONL" alone was green under that mutation.
        let err = refused(r#"[{"prompt":"hi","max_tokens":4}]"#, "JSONL");
        assert!(
            err.contains("JSON array"),
            "must name the shape, got: {err}"
        );
    }

    #[test]
    fn jsonl_yaml_is_refused_naming_jsonl() {
        // The exact shape the pre-PERF-039 loader accepted.
        let err = refused(
            "prompts:\n  - role: user\n    content: \"What is 2+2?\"\n    max_tokens: 16\n",
            "JSONL",
        );
        assert!(
            err.contains("no longer supported"),
            "must say the YAML corpus format is gone, got: {err}"
        );
    }

    #[test]
    fn jsonl_empty_file_is_refused() {
        refused("", "JSONL");
        refused("   \n\n\t\n", "JSONL");
    }

    #[test]
    fn jsonl_zero_records_is_refused_not_empty_vec() {
        // THE vacuous-pass row: a syntactically fine corpus carrying no work.
        // It must be an error, never `Ok(vec![])`.
        let err = refused("{\"_meta\":{\"corpus\":\"W1\"}}\n", "zero prompts");
        assert!(err.contains("no prompt records"), "got: {err}");
    }

    #[test]
    fn jsonl_meta_outside_line_one_is_refused() {
        refused(
            "{\"prompt\":\"a\"}\n{\"_meta\":{\"corpus\":\"W1\"}}\n{\"prompt\":\"b\"}\n",
            "line 1",
        );
    }

    #[test]
    fn jsonl_record_missing_prompt_is_refused_with_line_number() {
        refused("{\"prompt\":\"a\"}\n{\"max_tokens\":4}\n", "line 2");
    }

    #[test]
    fn jsonl_old_yaml_content_key_is_refused_with_line_number() {
        // `content` was the YAML loader's key. Converting a YAML corpus to
        // JSONL naively produces exactly this, and it must not load with a
        // silently defaulted prompt.
        refused("{\"role\":\"user\",\"content\":\"hi\"}\n", "line 1");
    }

    #[test]
    fn jsonl_typoed_field_is_refused_not_defaulted() {
        // `max_token` (singular) would otherwise deserialize to
        // `max_tokens: None` and the benchmark would run an unbudgeted
        // generation while its receipt claimed 128.
        refused("{\"prompt\":\"a\",\"max_token\":128}\n", "line 1");
    }

    #[test]
    fn jsonl_happy_path_carries_every_field() {
        let reqs = parse_jsonl(
            "{\"_meta\":{\"corpus\":\"W1\"}}\n\
             {\"id\":0,\"prompt\":\"hi\",\"max_tokens\":128,\"temperature\":0.0,\
             \"seed\":7,\"ignore_eos\":true,\"target_prompt_tokens\":512}\n\
             {\"prompt\":\"bye\"}\n",
        )
        .expect("valid JSONL");
        assert_eq!(reqs.len(), 2, "_meta must not be counted as a request");
        assert_eq!(reqs[0].max_tokens, Some(128));
        assert_eq!(reqs[0].seed, Some(7));
        assert_eq!(reqs[0].ignore_eos, Some(true));
        assert_eq!(reqs[0].messages[0].role, Role::User);
        assert_eq!(reqs[0].messages[0].content, "hi");
        // Defaults: greedy, no seed, no ignore_eos, no budget.
        assert_eq!(reqs[1].temperature, Some(0.0));
        assert_eq!(reqs[1].seed, None);
        assert_eq!(reqs[1].ignore_eos, None);
        assert_eq!(reqs[1].max_tokens, None);
    }

    #[test]
    fn jsonl_blank_lines_are_skipped_not_counted() {
        let reqs = parse_jsonl("{\"prompt\":\"a\"}\n\n{\"prompt\":\"b\"}\n\n")
            .expect("blank lines are legal in JSONL");
        assert_eq!(reqs.len(), 2);
    }

    #[test]
    fn jsonl_role_is_honored() {
        let reqs = parse_jsonl("{\"prompt\":\"s\",\"role\":\"system\"}\n").expect("valid");
        assert_eq!(reqs[0].messages[0].role, Role::System);
    }

    /// The committed W1 corpus must load through THIS loader.
    ///
    /// Before PERF-039 the only committed corpus in the tree
    /// (`prompts-w2.jsonl`) could not be read by the only loader in the tree,
    /// and nothing noticed because nothing tried. This test is what makes the
    /// corpus and the loader a single artifact rather than two.
    #[test]
    fn committed_w1_corpus_loads() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../aprender-serve/benchmarks/qwen-coder/prompts-w1.jsonl");
        let reqs =
            load_from_file(&path).unwrap_or_else(|e| panic!("committed W1 corpus must load: {e}"));
        assert_eq!(reqs.len(), 256, "§4.4.2 needs 8*16 sampled + 2*16 warmup");
        for r in &reqs {
            assert_eq!(r.max_tokens, Some(128), "§4.3.1 tg128");
            assert_eq!(r.temperature, Some(0.0), "§4.3.1 greedy");
            assert_eq!(r.seed, Some(0), "§4.3.1 seed = 0");
        }
        let distinct: std::collections::HashSet<&str> = reqs
            .iter()
            .map(|r| r.messages[0].content.as_str())
            .collect();
        assert_eq!(
            distinct.len(),
            reqs.len(),
            "identical prompts would let prefix caching drive Arm A"
        );
    }

    /// The committed W2 corpus must load through the SAME loader.
    #[test]
    fn committed_w2_corpus_loads() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../aprender-serve/benchmarks/qwen-coder/prompts-w2.jsonl");
        let reqs =
            load_from_file(&path).unwrap_or_else(|e| panic!("committed W2 corpus must load: {e}"));
        assert_eq!(reqs.len(), 100);
    }

    // ---------------------------------------------------------------------
    // §4.3.1's PROMPT-LENGTH INVARIANT (PERF-056, #2778).
    //
    // `prompts-w1.jsonl`'s `_meta` block promised, in two places, that the
    // harness asserts `prompt_tokens = 512 ± 8` against the model's own
    // tokenizer at measurement time. Nothing did. `grep -rn
    // "target_prompt_tokens\|tolerance_tokens" --include="*.rs" --include="*.sh"
    // --include="*.py"` on 9d45b927d found the generator that writes the
    // promise, the loader field documented as "never asserted here", and no
    // assertion anywhere — not in `perf_gate.sh`, not in `bench_receipt.py`,
    // not in `perf_gate/`, whose own module docs list the "`prompt_tokens =
    // 512 ± 8` assertion" under NOT IMPLEMENTED.
    //
    // MUTATION TABLE. Rows marked BOUNDARY are the discrimination cases: they
    // must stay GREEN, or the band is not a band but a point.
    //
    //  observed prompt_tokens                | must  | message must name
    //  --------------------------------------|-------|--------------------
    //  512 (dead centre)                     | OK    |
    //  504 = 512-8         [BOUNDARY]        | OK    |
    //  520 = 512+8         [BOUNDARY]        | OK    |
    //  503 = 512-9                           | ERR   | the prompt and 503
    //  521 = 512+9                           | ERR   | the prompt and 521
    //  one of 256 out of band                | ERR   | THAT prompt's index
    //  every count 0 (no `usage` block)      | ERR   | "INSTRUMENTATION"
    //  no completed request at all           | ERR   | "could not be checked"
    // ---------------------------------------------------------------------

    const W1_BAND: PromptTokenBand = PromptTokenBand {
        target: 512,
        tolerance: 8,
    };

    #[test]
    fn band_edges_are_inclusive_on_both_sides() {
        assert_eq!(W1_BAND.lo(), 504);
        assert_eq!(W1_BAND.hi(), 520);
        assert!(W1_BAND.contains(512), "dead centre");
        assert!(W1_BAND.contains(504), "BOUNDARY: the low edge is IN");
        assert!(W1_BAND.contains(520), "BOUNDARY: the high edge is IN");
        assert!(!W1_BAND.contains(503), "one below the low edge is OUT");
        assert!(!W1_BAND.contains(521), "one above the high edge is OUT");
    }

    #[test]
    fn band_boundary_corpus_stays_green() {
        // THE DISCRIMINATION CASE. A corpus sitting exactly on both edges is
        // conformant. A `<`/`>` slip in either comparison reds this row while
        // leaving every out-of-band row below still red, which is how a
        // tightened band passes review as a fixed one.
        let edges: Vec<(usize, u32)> = (0..64)
            .map(|i| (i, if i % 2 == 0 { 504 } else { 520 }))
            .collect();
        assert_prompt_tokens_in_band(W1_BAND, &edges).expect("504 and 520 are INSIDE 512 ± 8");
        assert_prompt_tokens_in_band(W1_BAND, &[(0, 512)]).expect("dead centre");
    }

    #[test]
    fn band_one_perturbed_prompt_is_refused_naming_it() {
        // THE NAMED MUTATION: 256 prompts in band, prompt 137 perturbed to
        // 521 — one token past the high edge.
        let mut observed: Vec<(usize, u32)> = (0..256).map(|i| (i, 512)).collect();
        observed[137].1 = 521;
        let err = assert_prompt_tokens_in_band(W1_BAND, &observed)
            .expect_err("521 is OUTSIDE 512 ± 8 and must be refused");
        assert!(err.contains("prompt 137"), "must name the prompt: {err}");
        assert!(err.contains("521"), "must give the actual length: {err}");
        assert!(err.contains("504"), "must state the band it failed: {err}");
        assert!(
            err.contains("1 of 256"),
            "must say how much of the corpus is affected — one record is an \
             edit, all 256 is a retune: {err}"
        );

        // REVERT -> GREEN. The same corpus with 137 put back is accepted, so
        // the row above binds to the perturbation and not to the fixture.
        observed[137].1 = 512;
        assert_prompt_tokens_in_band(W1_BAND, &observed).expect("reverted corpus is in band");
    }

    #[test]
    fn band_low_side_is_refused_too() {
        // Both polarities. A guard checking only the upper edge is green on
        // the failure mode the generator actually predicts: `--body-words`
        // tuned too low.
        let err =
            assert_prompt_tokens_in_band(W1_BAND, &[(0, 503)]).expect_err("503 is OUTSIDE 512 ± 8");
        assert!(err.contains("prompt 0") && err.contains("503"), "{err}");
    }

    #[test]
    fn band_all_zero_counts_are_an_instrumentation_gap_not_an_out_of_band_corpus() {
        // A server that emits no `usage` block reports 0 for every request.
        // Calling that "corpus out of band" sends the reader to regenerate the
        // corpus, which cannot fix it. Different cause, different message.
        let err = assert_prompt_tokens_in_band(W1_BAND, &[(0, 0), (1, 0), (2, 0)])
            .expect_err("an unverifiable invariant is not a satisfied one");
        assert!(err.contains("INSTRUMENTATION"), "{err}");
        assert!(err.contains("usage"), "{err}");
    }

    #[test]
    fn band_with_no_observations_is_refused_not_vacuously_passed() {
        let err = assert_prompt_tokens_in_band(W1_BAND, &[])
            .expect_err("zero observations must not pass");
        assert!(err.contains("could not be checked"), "{err}");
    }

    // ---------------------------------------------------------------------
    // THE `_meta` CONTRACT (PERF-056). Before this, the header was parsed only
    // far enough to notice the `_meta` key and skip the line: every promise in
    // it — count, distinctness, max_tokens, temperature, seed, the declared
    // band — was unenforced prose.
    //
    //  mutation of prompts-w1.jsonl's header/records | must | names
    //  ----------------------------------------------|------|-------------
    //  _meta.count disagrees with the record count    | ERR  | both numbers
    //  a record's max_tokens != _meta.max_tokens      | ERR  | the prompt id
    //  a record's seed != _meta.seed                  | ERR  | the prompt id
    //  a record's temperature != _meta.temperature    | ERR  | the prompt id
    //  a record's ignore_eos != _meta.ignore_eos      | ERR  | the prompt id
    //  a record's target_prompt_tokens disagrees      | ERR  | the prompt id
    //  two identical prompts, prompts_distinct: true  | ERR  | both records
    //  a prompt's word count != body_words + 2        | ERR  | the prompt id
    //  the committed corpus, unmutated  [BOUNDARY]    | OK   | 256 records
    // ---------------------------------------------------------------------

    /// The committed W1 header, with `{OVERRIDE}` splice points.
    fn w1_meta(extra: &str) -> String {
        format!(
            "{{\"_meta\":{{\"count\":2,\"max_tokens\":128,\"temperature\":0.0,\"seed\":0,\
             \"ignore_eos\":true,\"target_prompt_tokens\":512,\"tolerance_tokens\":8,\
             \"prompts_distinct\":true{extra}}}}}\n"
        )
    }

    fn w1_rec(id: u64, prompt: &str) -> String {
        format!(
            "{{\"id\":{id},\"max_tokens\":128,\"temperature\":0.0,\"seed\":0,\
             \"ignore_eos\":true,\"target_prompt_tokens\":512,\"prompt\":\"{prompt}\"}}\n"
        )
    }

    #[test]
    fn meta_contract_committed_shape_is_accepted() {
        // BOUNDARY row: the un-mutated shape must load, or every red row below
        // proves only that the fixture is broken.
        let body = w1_meta("") + &w1_rec(0, "alpha") + &w1_rec(1, "beta");
        let c = parse_corpus(&body).expect("the committed shape must load");
        assert_eq!(c.requests.len(), 2);
        assert_eq!(
            c.band,
            Some(PromptTokenBand {
                target: 512,
                tolerance: 8
            })
        );
    }

    #[test]
    fn meta_contract_count_disagreement_is_refused() {
        let body = w1_meta("") + &w1_rec(0, "alpha");
        let err = parse_corpus(&body).expect_err("count 2 with 1 record must be refused");
        assert!(
            err.contains("`_meta.count` = 2") && err.contains("1 prompt records"),
            "{err}"
        );
    }

    #[test]
    fn meta_contract_per_record_disagreements_are_refused_naming_the_prompt() {
        // A record silently disagreeing with its own header is how a corpus
        // regenerated with different flags gets measured as W1. Each row
        // mutates ONE field of prompt 1 and must be refused NAMING prompt 1 —
        // "a record is wrong" would leave the reader to find which.
        for (original, mutated, needle) in [
            ("\"max_tokens\":128", "\"max_tokens\":64", "max_tokens = 64"),
            ("\"seed\":0", "\"seed\":7", "seed = 7"),
            (
                "\"temperature\":0.0",
                "\"temperature\":0.7",
                "temperature = 0.7",
            ),
            (
                "\"target_prompt_tokens\":512",
                "\"target_prompt_tokens\":2048",
                "target_prompt_tokens = 2048",
            ),
            // PP-28: the fourth pinned sampler field. Added with the same
            // `agree` check as the three above, in the same table, so the
            // must-fire and its revert-to-green are proven the same way.
            (
                "\"ignore_eos\":true",
                "\"ignore_eos\":false",
                "ignore_eos = false",
            ),
        ] {
            let bad = w1_rec(1, "beta").replace(original, mutated);
            assert!(bad.contains(mutated), "fixture did not mutate: {original}");
            let body = w1_meta("") + &w1_rec(0, "alpha") + &bad;
            let err = parse_corpus(&body)
                .err()
                .unwrap_or_else(|| panic!("{mutated} must be REFUSED, was accepted"));
            assert!(err.contains("prompt 1"), "must name the prompt: {err}");
            assert!(err.contains(needle), "must give both values: {err}");
            // REVERT -> GREEN, per row, so each binds to its own mutation.
            let good = w1_meta("") + &w1_rec(0, "alpha") + &w1_rec(1, "beta");
            parse_corpus(&good).expect("reverted row must load");
        }
    }

    #[test]
    fn meta_contract_distinctness_is_enforced_not_merely_documented() {
        // `_meta.distinctness_rationale` explains that identical prompts would
        // let prefix caching, not the scheduler, drive Arm A's
        // scaling_efficiency. Only ONE test asserted it, and only against the
        // committed file — an operator-supplied corpus, or a regenerated one,
        // was unchecked.
        let body = w1_meta("") + &w1_rec(0, "same") + &w1_rec(1, "same");
        let err = parse_corpus(&body).expect_err("duplicate prompts must be refused");
        assert!(err.contains("records 1 and 2"), "must name both: {err}");
        assert!(err.contains("prefix caching"), "must say why: {err}");
        // DISCRIMINATION: with the flag absent, duplicates are legal. A corpus
        // that never claimed distinctness must not be judged on it.
        let permissive =
            "{\"_meta\":{\"count\":2}}\n".to_string() + &w1_rec(0, "same") + &w1_rec(1, "same");
        parse_corpus(&permissive).expect("no claim, no rule");
    }

    #[test]
    fn meta_contract_body_words_shape_is_enforced() {
        // The one LENGTH check possible before a tokenizer exists: the
        // whitespace-word shape the generator recorded. `// w1-NNNN` is two
        // words, so `body_words: 3` declares five.
        let body = w1_meta(",\"body_words\":3") + &w1_rec(0, "// w1-0000 a b c");
        let mut ok = body.replace("\"count\":2", "\"count\":1");
        parse_corpus(&ok).expect("5 words == body_words 3 + 2-word header");
        // PERTURB: one word short.
        ok = ok.replace("// w1-0000 a b c", "// w1-0000 a b");
        let err = parse_corpus(&ok).expect_err("4 words must be refused");
        assert!(err.contains("prompt 0"), "must name the prompt: {err}");
        assert!(
            err.contains("4 whitespace"),
            "must give the actual count: {err}"
        );
        assert!(
            err.contains("NOT a token count"),
            "must not let a word count masquerade as §4.3.1's band: {err}"
        );
    }

    #[test]
    fn meta_contract_band_needs_both_edges() {
        // A target with no tolerance is not a band. Defaulting the tolerance
        // to 0 would assert exact equality; defaulting it to 8 would invent
        // §4.3.1's number in a corpus that never stated it.
        let body = "{\"_meta\":{\"target_prompt_tokens\":512}}\n".to_string() + &w1_rec(0, "a");
        assert_eq!(parse_corpus(&body).expect("loads").band, None);
    }

    #[test]
    fn committed_w1_corpus_declares_the_512_pm_8_band() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../aprender-serve/benchmarks/qwen-coder/prompts-w1.jsonl");
        let corpus = load_corpus(&path).expect("committed W1 corpus must load");
        assert_eq!(
            corpus.band,
            Some(PromptTokenBand {
                target: 512,
                tolerance: 8
            }),
            "§4.3.1's band must reach the harness, not stop at the header"
        );
        assert_eq!(corpus.requests.len(), 256);
    }

    #[test]
    fn committed_w1_corpus_is_refused_when_one_prompt_is_perturbed() {
        // THE FILE-LEVEL MUTATION. Read the committed corpus, drop one word
        // from prompt 137, and the loader must refuse it NAMING that prompt.
        // Reverting restores GREEN, so this binds to the perturbation.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../aprender-serve/benchmarks/qwen-coder/prompts-w1.jsonl");
        let good = std::fs::read_to_string(&path).expect("committed corpus is readable");
        parse_corpus(&good).expect("unmutated: GREEN");

        let mut lines: Vec<String> = good.lines().map(ToString::to_string).collect();
        // Line 0 is `_meta`; line 138 is `"id":137`.
        let victim = &mut lines[138];
        assert!(
            victim.contains("\"id\":137"),
            "fixture drifted: {}",
            &victim[..40]
        );
        let cut = victim.rfind(' ').expect("a 498-word prompt has spaces");
        victim.replace_range(cut..=cut, "");
        let mutated = lines.join("\n") + "\n";

        let err = parse_corpus(&mutated).expect_err("a perturbed prompt must be REFUSED");
        assert!(err.contains("prompt 137"), "must name the prompt: {err}");
        assert!(
            err.contains("497 whitespace"),
            "must give the actual length: {err}"
        );

        // REVERT -> GREEN.
        parse_corpus(&good).expect("reverted: GREEN");
    }

    #[test]
    fn test_load_from_file_missing() {
        let result = load_from_file(Path::new("/nonexistent/prompts.jsonl"));
        assert!(result.is_err());
    }

    #[test]
    fn test_all_profiles_have_deterministic_settings() {
        for profile in [
            PromptProfile::Micro,
            PromptProfile::Short,
            PromptProfile::Medium,
            PromptProfile::Long,
        ] {
            let prompts = load_profile(profile);
            for p in &prompts {
                assert_eq!(
                    p.temperature,
                    Some(0.0),
                    "Profile {profile:?} should be deterministic"
                );
                assert_eq!(p.stream, Some(false));
                assert!(
                    p.max_tokens.is_some(),
                    "Profile {profile:?} should set max_tokens"
                );
            }
        }
    }

    // ---------------------------------------------------------------------
    // PP-28 — `ignore_eos` is the fourth field of §5.1's sampler pin, and the
    // one the corpus could not previously carry.
    //
    // Before this change `CorpusMeta` had no `ignore_eos` field, so the header
    // could not DECLARE the pin and `check_record` could not compare against
    // it — and no record in the committed corpus carried the field at all. A
    // W1 band therefore ran with EOS active: each request generated whatever
    // the model decided to stop after rather than exactly `max_tokens`, so the
    // work per band was not pinned and `completion_tokens == n_predict` was
    // uncheckable. `short_of_n_predict` counts exactly that, and it counts
    // nothing if the wire never carried the flag.
    // ---------------------------------------------------------------------

    /// MUST-FIRE: a record that opts OUT of the header's pin is refused, and
    /// the message names the record an operator would edit.
    ///
    /// This is the same `agree` rule `max_tokens` and `seed` already have.
    /// Silence still constrains nothing — a header that declares no
    /// `ignore_eos` claims nothing about it — so what is refused here is
    /// DISAGREEMENT, which is the only shape a corpus file can be wrong in
    /// without the reader being able to see it.
    #[test]
    fn corpus_without_ignore_eos_is_refused_for_w1() {
        let unpinned = w1_rec(1, "beta").replace("\"ignore_eos\":true", "\"ignore_eos\":false");
        assert!(
            unpinned.contains("\"ignore_eos\":false"),
            "fixture: {unpinned}"
        );
        let body = w1_meta("") + &w1_rec(0, "alpha") + &unpinned;
        let err = parse_corpus(&body).expect_err("an unpinned record must be REFUSED");
        assert!(err.contains("prompt 1"), "must name the prompt: {err}");
        assert!(
            err.contains("ignore_eos = false"),
            "must give the value: {err}"
        );
        assert!(
            err.contains("`_meta.ignore_eos` = true"),
            "and the header's: {err}"
        );

        // REVERT -> GREEN, so the refusal binds to the mutation and not to the
        // fixture.
        let good = w1_meta("") + &w1_rec(0, "alpha") + &w1_rec(1, "beta");
        parse_corpus(&good).expect("the pinned corpus loads");
    }

    /// MUST-NOT-FIRE, at file level: the corpus this repo actually ships pins
    /// `ignore_eos` in every record AND declares it in `_meta`.
    ///
    /// Both halves matter. A corpus whose records carry the flag but whose
    /// header does not declare it passes `agree` vacuously — the enforcement
    /// only exists where the header makes a claim.
    #[test]
    fn committed_w1_corpus_pins_ignore_eos() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../aprender-serve/benchmarks/qwen-coder/prompts-w1.jsonl");
        let text = std::fs::read_to_string(&path).expect("committed corpus is readable");
        let mut lines = text.lines();
        let header = lines.next().expect("line 1 is the `_meta` header");
        assert!(
            header.contains("\"ignore_eos\":true"),
            "`_meta` must DECLARE the pin, or `agree` enforces nothing: {}",
            &header[..header.len().min(160)]
        );

        let reqs = load_from_file(&path).expect("committed W1 corpus must load");
        assert_eq!(reqs.len(), 256);
        for (i, r) in reqs.iter().enumerate() {
            assert_eq!(
                r.ignore_eos,
                Some(true),
                "record {i} does not pin ignore_eos; §5.1 W1 requires it on the wire"
            );
            // The other three of the four PP-28 fields, asserted here too so a
            // regeneration that drops one is caught by the same test that owns
            // the sampler pin.
            assert_eq!(r.max_tokens, Some(128));
            assert_eq!(r.temperature, Some(0.0));
            assert_eq!(r.seed, Some(0));
        }
    }
}
