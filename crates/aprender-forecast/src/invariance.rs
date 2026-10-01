//! The D-19 no-argument bitwise invariance gate: a delivery that ADDS optional arguments
//! must be byte-identical for a caller who passes none.
//!
//! That is Forecast Coach's acceptance condition for a one-line tag bump, and it is the
//! only thing that makes "additive change" a checkable claim rather than a promise. The
//! mechanism is a bit-exact signature over the deterministic fields of a
//! [`ForecastResponse`], captured through the PUBLIC door on a commit at which the new
//! code does not exist, committed, and re-checked afterwards.
//!
//! # What the signature covers, and what it deliberately does not
//!
//! `fit_seconds` and `predict_seconds` are EXCLUDED. They are wall-clock: a signature that
//! included them could never reproduce, which would make the gate vacuous in the opposite
//! direction — permanently red, therefore permanently ignored. Everything else is included.
//!
//! Every f64 is hashed by `to_bits()`, never by a formatted decimal. A printed comparison
//! silently accepts any change below the print precision, which is exactly the class of
//! drift this gate exists to catch. Negative zero is normalised to positive zero so `-0.0`
//! and `0.0` do not read as a change; every other bit pattern, NaN payloads included, is
//! significant.
//!
//! `components` and `diagnostics` are walked as JSON. The walk sorts object keys itself
//! (see [`json`]'s `Object` arm) rather than relying on `serde_json::Map`'s own iteration
//! order: that order is a BTreeMap's key-sorted iteration by default, but flips to an
//! IndexMap's insertion order the instant any crate in the build graph enables the
//! `preserve_order` cargo feature — and cargo feature unification applies that choice
//! workspace-wide, not only to the crate that asked for it (feature-unification-divergence,
//! `.planning/debug/resolved/feature-unification-divergence.md`). `components` is built by
//! inserting in computation order, so before this fix the signature silently depended on
//! whichever backing store serde_json happened to be compiled with; explicit sorting means
//! a future reordering of those inserts — or a future dependency enabling `preserve_order`
//! — cannot read as a behaviour change either way.
//!
//! # Three parts, three DIFFERENT claims — and the exposure each one carries
//!
//! SC2 does not ask for a green invariance table. It asks for one that is *proven able to
//! fail*, and says in as many words that a green gate with no falsification probe beside it
//! does not satisfy the criterion. A `signature` that returned a constant would produce
//! exactly the same green table as a correct one. So the gate is four statements, not one,
//! and they are deliberately not four ways of saying the same thing:
//!
//! | Part | Claim it CARRIES | Claim it does NOT carry | Arch exposure |
//! |---|---|---|---|
//! | the committed baseline ([`every_baseline_case_reproduces_its_signature`]) | the eight door cases answer today what they answered at a commit where `regressors.rs` did not exist — the only INDEPENDENT HISTORICAL evidence in this gate | nothing about whether the signature can detect a change | **arch-keyed** (skips loudly on an unrecorded arch) |
//! | part A ([`part_a_every_case_is_deterministic_through_the_door`]) | the same host, twice, answers identically — so a baseline mismatch is a real difference and never run-to-run noise | nothing about the past: it compares now against now | **unconditional** |
//! | part B ([`part_b_the_signature_is_proven_able_to_fail`]) | the signature MOVES for a change of one unit in the last place, and for the band and `diagnostics` fields nothing else exercises | nothing about the forecast being correct — only that the detector detects | **unconditional** |
//! | part C ([`part_c_the_splice_is_inert_at_zero_regressors`]) | SPLICE INERTNESS: `regressors::splice` at zero regressors changes neither the design nor the forecast, measured bit for bit, bands included | it is NOT a pre-change comparison — both sides call the SAME post-change [`crate::prophet::predict`], so a regression common to the empty-regressor branch moves both sides equally and part C stays green | unconditional (but not independent of the current `predict`) |
//!
//! **Parts A and B must never be made conditional on the running architecture, and must
//! never be `#[ignore]`d.** Part A compares one host against ITSELF and part B mutates a
//! response in memory; neither touches libm, so neither can legitimately differ across
//! runners. Only the cross-commit baseline comparison is arch-keyed, for the libm reason
//! below. This distinction is written down because the tempting repair for a cross-platform
//! red — relaxing whatever is red — would remove exactly the two parts SC2 rests on.
//! [`parts_a_and_b_are_unconditional`] enforces it against this module's own source.
//!
//! # Why the baseline is keyed by architecture
//!
//! `by_arch` is keyed on [`std::env::consts::ARCH`], and this is a correctness requirement
//! rather than future-proofing. `prophet::feature_row` calls `f64::sin`/`f64::cos`, which
//! resolve to the platform libm, and the NeuralProphet arm trains on the f32 autograd whose
//! GEMM routing is arch-specific. CI's `workspace-test` job runs on self-hosted X64 Linux
//! and is a REQUIRED check on protected `main`, while these signatures were captured on
//! aarch64 macOS. A single-architecture baseline compared unconditionally would be red in
//! CI on day one, and a gate that is red for a reason nobody intended gets disabled.
//!
//! The repo already carries this precedent: `quantiles_abs_f32_nonaarch64` is a separate
//! bar precisely because x86 and aarch64 differ bitwise.
//!
//! Absence of the FILE is a DEFECT and panics. Absence of the running ARCH inside an
//! otherwise valid file is a skip WITH A STATED REASON and a stated way to close it — see
//! [`every_baseline_case_reproduces_its_signature`]. The companion test
//! [`the_baseline_records_at_least_one_architecture`] forbids a file that lost its content,
//! so "nothing to compare" can never be how this passes.

use crate::test_support::{fixture_path, load_json, read_csv};
use crate::types::{ForecastArgs, ForecastResponse, HolidayArg};

// ------------------------------------------------------------- the hasher ----

/// FNV-1a 64. Chosen for being trivially reproducible in any language, not for strength:
/// this is a change detector, not a security primitive.
pub struct Hasher(u64);

impl Hasher {
    #[must_use]
    pub fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    pub fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= u64::from(x);
            self.0 = self.0.wrapping_mul(0x1000_0000_01b3);
        }
    }

    /// Hash a length prefix, so `["a", "bc"]` and `["ab", "c"]` cannot collide.
    fn len(&mut self, n: usize) {
        let n = u64::try_from(n).expect("a collection length fits in u64");
        self.bytes(&n.to_le_bytes());
    }

    pub fn f64(&mut self, v: f64) {
        // Normalise the two zeros so -0.0 and 0.0 do not read as a change; every other
        // bit pattern, NaN payloads included, is significant.
        let v = if v == 0.0 { 0.0 } else { v };
        self.bytes(&v.to_bits().to_le_bytes());
    }

    pub fn f64s(&mut self, vs: &[f64]) {
        self.len(vs.len());
        for &v in vs {
            self.f64(v);
        }
    }

    pub fn str(&mut self, s: &str) {
        self.len(s.len());
        self.bytes(s.as_bytes());
    }

    #[must_use]
    pub fn finish(&self) -> u64 {
        self.0
    }

    /// Hash a JSON object: its entry count, then its entries in KEY ORDER.
    ///
    /// THE one place the signature's object ordering is decided. It is decided HERE rather
    /// than inherited from `serde_json::Map`'s iteration order, because that order is a
    /// `BTreeMap`'s (key-sorted) by default but an `IndexMap`'s (insertion) the moment ANY
    /// crate in the build graph enables `serde_json/preserve_order` — cargo feature
    /// unification then applies that choice to every crate linking serde_json, not only the
    /// one that asked for it. `pmcp` does exactly that.
    ///
    /// Both map-walking sites route through here deliberately. [`signature_with`] walks
    /// `components` directly instead of through [`json`], so sorting only inside [`json`]
    /// left that site hashing in insertion order — one operation with two implementations is
    /// how this gate began hashing `components` under one rule and `diagnostics` under
    /// another. A third map-typed field must call this, not copy it.
    fn json_map(&mut self, o: &serde_json::Map<String, serde_json::Value>) {
        self.len(o.len());
        let mut entries: Vec<_> = o.iter().collect();
        entries.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        for (k, v) in entries {
            self.str(k);
            json(self, v);
        }
    }
}

/// Walk a JSON value, tagging each variant so a string `"1"` and a number `1` differ.
fn json(h: &mut Hasher, v: &serde_json::Value) {
    match v {
        serde_json::Value::Null => h.bytes(b"n"),
        serde_json::Value::Bool(b) => {
            h.bytes(b"b");
            h.bytes(&[u8::from(*b)]);
        }
        serde_json::Value::Number(n) => {
            h.bytes(b"#");
            h.f64(n.as_f64().unwrap_or(f64::NAN));
        }
        serde_json::Value::String(s) => {
            h.bytes(b"s");
            h.str(s);
        }
        serde_json::Value::Array(a) => {
            h.bytes(b"[");
            h.len(a.len());
            for x in a {
                json(h, x);
            }
        }
        serde_json::Value::Object(o) => {
            h.bytes(b"{");
            h.json_map(o);
        }
    }
}

/// A bit-exact signature over every deterministic field of a [`ForecastResponse`].
///
/// `fit_seconds` and `predict_seconds` are excluded — see the module docs.
#[must_use]
pub fn signature(r: &ForecastResponse) -> u64 {
    signature_with(r, &r.diagnostics)
}

/// The ONE hashing body. `signature` is this with the response's own `diagnostics`; part A's
/// `budget_hit` triage is this with a masked copy.
///
/// `ForecastResponse` is not `Clone` (it is a serialisation type), so the triage cannot
/// clone-and-edit. Factoring the body is the alternative to writing the field list twice —
/// and a second field list is precisely how a gate starts hashing less than it claims.
fn signature_with(r: &ForecastResponse, diagnostics: &serde_json::Value) -> u64 {
    let mut h = Hasher::new();
    h.str(&r.model);
    h.str(&r.freq);
    h.len(r.n_history);
    h.len(r.ds.len());
    for d in &r.ds {
        h.str(d);
    }
    h.f64s(&r.yhat);
    h.f64s(&r.yhat_lower);
    h.f64s(&r.yhat_upper);
    h.f64s(&r.trend);
    // `components` is walked HERE rather than through `json`, so it does not reach that
    // function's `Object` arm. The door inserts these keys in PROPHET'S COMPUTATION ORDER
    // (`additive_terms`, `multiplicative_terms`, `holidays`, seasonality names), which is not
    // sorted order — hence `json_map`, which is the single place that decides it.
    h.json_map(&r.components);
    json(&mut h, diagnostics);
    h.finish()
}

// --------------------------------------------------------- the door cases ----

/// The shape of one door case, beyond the series and the horizon.
#[derive(Clone, Copy, Debug)]
enum Shape {
    /// Prophet, everything defaulted.
    ProphetDefault,
    /// Prophet on a monthly series (`freq: "MS"`).
    ProphetMonthly,
    /// Prophet, monthly, multiplicative seasonality.
    ProphetMonthlyMultiplicative,
    /// Prophet, logistic growth (the `cap` is derived from the series).
    ProphetLogistic,
    /// Prophet with one holiday carrying a `[-1, +1]` window.
    ProphetHolidayWindows,
    /// The NeuralProphet arm at a given lag count.
    NeuralProphet(usize),
}

/// The eight cases, ported from `sources/012-no-arg-bitwise-invariance/src/main.rs`.
///
/// There is deliberately NO `retail/neuralprophet/MS` case: `forecast.rs` refuses
/// `freq != "D"` on that arm (pinned by `neuralprophet_refuses_non_daily_freq`), so the
/// spike swapped it for a second DAILY series. A case that is refused rather than computed
/// would record the signature of an error path and prove nothing about invariance.
const CASES: [(&str, &str, usize, Shape); 8] = [
    (
        "peyton/prophet/default",
        "peyton_manning.csv",
        30,
        Shape::ProphetDefault,
    ),
    (
        "air/prophet/multiplicative",
        "air_passengers.csv",
        12,
        Shape::ProphetMonthlyMultiplicative,
    ),
    (
        "retail/prophet/default",
        "retail_sales.csv",
        12,
        Shape::ProphetMonthly,
    ),
    (
        "wp_log_R/prophet/logistic",
        "wp_log_R.csv",
        30,
        Shape::ProphetLogistic,
    ),
    (
        "peyton/prophet/holidays+windows",
        "peyton_manning.csv",
        30,
        Shape::ProphetHolidayWindows,
    ),
    (
        "peyton/neuralprophet/lag0",
        "peyton_manning.csv",
        30,
        Shape::NeuralProphet(0),
    ),
    (
        "peyton/neuralprophet/lag7",
        "peyton_manning.csv",
        30,
        Shape::NeuralProphet(7),
    ),
    (
        "wp_log_R/neuralprophet/lag0",
        "wp_log_R.csv",
        30,
        Shape::NeuralProphet(0),
    ),
];

/// Build one case's request through the PUBLIC argument type only.
///
/// `read_csv` already strips quoted fields and carriage returns and sorts/de-duplicates by
/// `ds`, which `wp_log_R.csv` needs because it is not chronological.
fn args_for(csv: &str, horizon: usize, shape: Shape) -> ForecastArgs {
    let (ds, y) = read_csv(csv);
    let mut args = ForecastArgs {
        ds,
        y,
        horizon,
        seed: Some(42),
        ..ForecastArgs::default()
    };
    match shape {
        Shape::ProphetDefault => {}
        Shape::ProphetMonthly => args.freq = Some("MS".into()),
        Shape::ProphetMonthlyMultiplicative => {
            args.freq = Some("MS".into());
            args.seasonality_mode = Some("multiplicative".into());
        }
        Shape::ProphetLogistic => {
            args.growth = Some("logistic".into());
            args.cap = Some(args.y.iter().fold(f64::MIN, |m, v| m.max(*v)) * 1.2);
        }
        Shape::ProphetHolidayWindows => {
            args.holidays = Some(vec![HolidayArg {
                name: "playoff".into(),
                dates: vec![
                    "2010-01-16".into(),
                    "2014-01-12".into(),
                    "2016-01-17".into(),
                ],
                lower_window: -1,
                upper_window: 1,
            }]);
        }
        Shape::NeuralProphet(n_lags) => {
            args.model = Some("neuralprophet".into());
            args.n_lags = Some(n_lags);
        }
    }
    args
}

/// The name of the committed baseline, read through `test_support::fixture_path`.
const BASELINE: &str = "invariance_baseline.json";

/// The recipe printed when the running architecture has no recorded entry. It is a single
/// line on purpose: a skip whose reason scrolls away is a silent skip.
const CAPTURE_RECIPE: &str = "INVARIANCE_BASELINE_MODE=capture cargo test -p aprender-forecast --lib invariance::capture_baseline -- --ignored --nocapture";

// --------------------------------------------------------------- capture ----

/// Write (or extend) the committed baseline for the RUNNING architecture.
///
/// `#[ignore]`d and additionally gated on `INVARIANCE_BASELINE_MODE=capture`, following
/// `np::wall`'s `NP_WALL_MODE` convention: this is a CAPTURE, not an assertion, and a
/// capture that can run by accident can overwrite the evidence it is supposed to preserve.
///
/// **It must be run on a commit at which the change under test does not yet exist.** A
/// baseline captured afterwards is circular evidence — it would record the new behaviour
/// and then congratulate the new behaviour for matching it. `captured_at_commit` is
/// `git rev-parse HEAD` taken BEFORE the capture is committed, so the recorded value is
/// already the parent of the commit that carries the file; that value IS the phase base and
/// must not have `^` applied to it again.
///
/// Capturing a SECOND architecture (the X64 CI runner) means checking out
/// `captured_at_commit` on that host and re-running this, then carrying the merged file
/// forward. Existing architectures are preserved, so the two captures compose.
#[test]
#[ignore = "capture, not an assertion; run with INVARIANCE_BASELINE_MODE=capture -- --ignored"]
fn capture_baseline() {
    let mode = std::env::var("INVARIANCE_BASELINE_MODE").unwrap_or_default();
    assert_eq!(
        mode, "capture",
        "refusing to overwrite the committed baseline without \
         INVARIANCE_BASELINE_MODE=capture; the recipe is: {CAPTURE_RECIPE}"
    );

    let head = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("git rev-parse HEAD must run: the baseline's provenance is not optional");
    assert!(head.status.success(), "git rev-parse HEAD failed");
    let head = String::from_utf8(head.stdout)
        .expect("git rev-parse HEAD is utf-8")
        .trim()
        .to_string();

    // CLAUDE.md rule 2: ONE owner for this token. A second derivation here is exactly the
    // drift `profile_token`'s doc comment guards against — a debug run labelled `release`
    // is stamped into the committed baseline and turns it into a confident wrong answer.
    let profile = crate::sc1_wall::profile_token();
    let arch = std::env::consts::ARCH;

    let mut cases = serde_json::Map::new();
    for (label, csv, horizon, shape) in CASES {
        let args = args_for(csv, horizon, shape);
        let r = crate::forecast::forecast(&args)
            .unwrap_or_else(|e| panic!("case {label} must succeed through the door: {e}"));
        let sig = signature(&r);
        println!(
            "INVARIANCE CAPTURE: case={label} arch={arch} profile={profile} signature={sig:016x}"
        );
        // Built field by field rather than with `serde_json::json!`: that macro expands to
        // an internal `.unwrap()` for runtime values, which `.clippy.toml` bans outright
        // (GH-41). `Value::String` is the same result with no hidden unwrap.
        let mut entry = serde_json::Map::new();
        entry.insert(
            "signature".into(),
            serde_json::Value::String(format!("{sig:016x}")),
        );
        entry.insert("arch".into(), serde_json::Value::String(arch.to_string()));
        entry.insert(
            "profile".into(),
            serde_json::Value::String(profile.to_string()),
        );
        entry.insert(
            "captured_at_commit".into(),
            serde_json::Value::String(head.clone()),
        );
        cases.insert(label.to_string(), serde_json::Value::Object(entry));
    }

    let path = fixture_path(BASELINE);
    // Preserve any architecture already recorded, and keep the ORIGINAL top-level
    // provenance: the top-level `captured_at_commit` is the pre-change commit the whole
    // baseline describes, so a later capture on a second host must be taken at that same
    // commit rather than silently re-dating the file.
    let mut doc: serde_json::Map<String, serde_json::Value> = if path.is_file() {
        let raw = std::fs::read_to_string(&path).expect("existing baseline is readable");
        serde_json::from_str(&raw).expect("existing baseline is a JSON object")
    } else {
        serde_json::Map::new()
    };
    if !doc.contains_key("captured_at_commit") {
        doc.insert(
            "captured_at_commit".into(),
            serde_json::Value::String(head.clone()),
        );
        doc.insert(
            "captured_on".into(),
            serde_json::Value::String(iso_date_utc()),
        );
    }
    // ---- the two halves must describe ONE COMPUTATION, measured rather than inferred ----
    //
    // This check used to be `recorded == head`, and it made its own documented workflow
    // IMPOSSIBLE: the only commit it accepted was `captured_at_commit` (c850e62aa), and this
    // file does not exist at that commit — it was added by bfc21cbd9. So the harness could
    // not run at the only commit the guard allowed, while 06.1-05's checkpoint simultaneously
    // instructed an operator to capture "at the CURRENT commit". The two directions
    // contradicted, and three capture attempts on x86_64 hit exactly this (measured by the
    // phase orchestrator, `06.1-x64-baseline-measurement.md`, finding 3).
    //
    // The INTENT was right and is kept. A commit id was only ever a PROXY for "the
    // computation tree is unchanged", so the fix is to measure the thing itself: the fresh
    // signatures for the running architecture were just computed above, so if this file
    // already records this architecture and every one of them still reproduces, then HEAD
    // computes bit-for-bit what the file describes and capturing here composes — whatever
    // the commit ids say. That is STRICTLY STRONGER evidence than commit-id equality, which
    // could hold while a dependency moved under it.
    //
    // When the running architecture is NOT recorded there is nothing local that can
    // establish the tree matches, and this correctly still refuses. That case is what
    // `import_baseline` exists for: a signature for an architecture this host cannot execute
    // can only arrive with stated provenance, never with a local proof.
    if let Some(recorded) = doc.get("captured_at_commit").and_then(|v| v.as_str()) {
        if recorded != head {
            let already: Option<&serde_json::Map<String, serde_json::Value>> = doc
                .get("by_arch")
                .and_then(|v| v.as_object())
                .and_then(|m| m.get(arch))
                .and_then(|v| v.as_object());
            let reproduces = already.is_some_and(|prev| {
                prev.len() == CASES.len()
                    && cases.iter().all(|(label, fresh)| {
                        let fresh = fresh["signature"].as_str();
                        let old = prev.get(label).and_then(|v| v["signature"].as_str());
                        fresh.is_some() && fresh == old
                    })
            });
            assert!(
                reproduces,
                "this baseline describes commit {recorded}, HEAD is {head}, and arch={arch} \
                 does not reproduce the recorded signatures at HEAD — so the two halves of \
                 the file would describe DIFFERENT computations. Either this is a deliberate \
                 behaviour change, in which case re-capture the WHOLE file at its own \
                 pre-change commit, or you are importing an architecture this host cannot \
                 execute, in which case use import_baseline: {IMPORT_RECIPE}"
            );
            println!(
                "INVARIANCE CAPTURE: HEAD {head} differs from the baseline's commit \
                 {recorded}, but all {} recorded arch={arch} cases reproduce at HEAD, so the \
                 computation tree is unchanged and this capture composes.",
                CASES.len()
            );
        }
    }
    doc.insert(
        "note".into(),
        serde_json::Value::String(String::from(
            "Evidence for ONE tag bump, not a permanent certificate. These signatures \
             describe the tree at `captured_at_commit`; a later intentional behaviour \
             change re-captures them at its own pre-change commit. Keyed by \
             std::env::consts::ARCH because libm and the f32 GEMM routing differ bitwise \
             across architectures.",
        )),
    );
    let by_arch = doc
        .entry("by_arch")
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()))
        .as_object_mut()
        .expect("by_arch is an object");
    by_arch.insert(arch.to_string(), serde_json::Value::Object(cases));

    let text = serde_json::to_string_pretty(&doc).expect("baseline serialises");
    std::fs::write(&path, text + "\n").expect("baseline is writable");
    println!("INVARIANCE CAPTURE: wrote {}", path.display());
}

/// The recipe for [`import_baseline`], printed by the guard that refuses a capture it
/// cannot verify locally. One line, for the reason [`CAPTURE_RECIPE`] is one line.
const IMPORT_RECIPE: &str = "INVARIANCE_BASELINE_MODE=import INVARIANCE_BASELINE_IMPORT=<payload.json> cargo test -p aprender-forecast --lib invariance::import_baseline -- --ignored --nocapture";

/// Record signatures for an architecture THIS HOST CANNOT EXECUTE, with stated provenance.
///
/// # Why a second entry point exists at all
///
/// [`capture_baseline`] computes. It is the right tool whenever the host can run the
/// architecture being recorded, and it is the only tool that produces evidence rather than
/// transcribing it. But a cross-architecture baseline is, by construction, a claim about a
/// machine that is not this one: no local check can establish a foreign architecture's
/// signatures, and no amount of fixing `capture_baseline`'s guard changes that. The honest
/// alternative to an import path is a hand-edited fixture, which has no guard at all.
///
/// So this path is deliberately narrow, and every one of its refusals is a guard the hand
/// edit would not have had:
///
/// * it REFUSES to import the architecture it is running on — that one must be CAPTURED,
///   which is what stops this from becoming a way around [`capture_baseline`];
/// * it REFUSES to overwrite an architecture already recorded;
/// * it requires the payload's case labels to equal [`CASES`] EXACTLY, both directions, so a
///   short payload cannot land a partial architecture that
///   [`the_baseline_records_at_least_one_architecture`] would then reject;
/// * it requires every signature to be sixteen lowercase hex digits, the shape
///   [`signature`] emits;
/// * it requires a non-empty `measured_by`, stamped into EVERY entry, so the committed file
///   says plainly that these were transcribed rather than computed here.
///
/// # What actually verifies an imported entry
///
/// Nothing here does, and the doc comment must not pretend otherwise.
/// [`every_baseline_case_reproduces_its_signature`] verifies it — on the first host of that
/// architecture that runs the suite. An imported entry that is wrong turns that gate RED
/// there and names the case, which is strictly better than the alternative it replaces: a
/// gate that SKIPS forever on the architecture the repo's own CI runs, reporting nothing
/// either way.
#[test]
#[ignore = "import, not an assertion; run with INVARIANCE_BASELINE_MODE=import -- --ignored"]
fn import_baseline() {
    let mode = std::env::var("INVARIANCE_BASELINE_MODE").unwrap_or_default();
    assert_eq!(
        mode, "import",
        "refusing to touch the committed baseline without INVARIANCE_BASELINE_MODE=import; \
         the recipe is: {IMPORT_RECIPE}"
    );
    let payload_path = std::env::var("INVARIANCE_BASELINE_IMPORT").unwrap_or_default();
    assert!(
        !payload_path.is_empty(),
        "INVARIANCE_BASELINE_IMPORT must name the payload file; the recipe is: {IMPORT_RECIPE}"
    );
    let raw = std::fs::read_to_string(&payload_path)
        .unwrap_or_else(|e| panic!("the import payload {payload_path} must be readable: {e}"));
    let payload: serde_json::Value = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("the import payload {payload_path} must be JSON: {e}"));

    let field = |k: &str| -> String {
        payload
            .get(k)
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("the import payload must carry a string `{k}`"))
            .trim()
            .to_string()
    };
    let arch = field("arch");
    let profile = field("profile");
    let captured_at_commit = field("captured_at_commit");
    let measured_by = field("measured_by");
    assert!(
        !arch.is_empty() && !profile.is_empty() && !captured_at_commit.is_empty(),
        "arch, profile and captured_at_commit must all be non-empty"
    );
    assert!(
        measured_by.len() >= 40,
        "`measured_by` is the whole reason an import is admissible: it must state the host, \
         the method and what was and was not controlled, in enough words to be checkable. \
         Got {} characters",
        measured_by.len()
    );

    // THE ANTI-BYPASS GUARD. What this host can execute, it must MEASURE.
    assert_ne!(
        arch,
        std::env::consts::ARCH,
        "refusing to IMPORT arch={arch}, which is the architecture this host runs: capture \
         it instead, so the file records evidence rather than a transcription. The recipe \
         is: {CAPTURE_RECIPE}"
    );

    let payload_cases = payload
        .get("cases")
        .and_then(|v| v.as_object())
        .expect("the import payload must carry a `cases` object");
    let want: std::collections::BTreeSet<&str> = CASES.iter().map(|(l, _, _, _)| *l).collect();
    let got: std::collections::BTreeSet<&str> = payload_cases.keys().map(String::as_str).collect();
    let missing: Vec<&&str> = want.difference(&got).collect();
    let extra: Vec<&&str> = got.difference(&want).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "the payload must name EXACTLY the {} door cases — a partial architecture is one \
         the gate would reject anyway, and an extra case is a label that no longer exists.\n  \
         MISSING: {missing:?}\n  EXTRA: {extra:?}",
        CASES.len()
    );

    let mut cases = serde_json::Map::new();
    for (label, _, _, _) in CASES {
        let sig = payload_cases
            .get(label)
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("case {label} must be a hex signature string"))
            .trim()
            .to_string();
        assert!(
            sig.len() == 16
                && sig
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
            "case {label}: {sig:?} is not sixteen lowercase hex digits, which is the shape \
             `signature` emits — a mistyped signature must fail HERE and not silently become \
             a red gate on another host"
        );
        println!("INVARIANCE IMPORT: case={label} arch={arch} profile={profile} signature={sig}");
        let mut entry = serde_json::Map::new();
        entry.insert("signature".into(), serde_json::Value::String(sig));
        entry.insert("arch".into(), serde_json::Value::String(arch.clone()));
        entry.insert("profile".into(), serde_json::Value::String(profile.clone()));
        entry.insert(
            "captured_at_commit".into(),
            serde_json::Value::String(captured_at_commit.clone()),
        );
        // Stamped on EVERY entry, not once at the top: an entry read in isolation must say
        // that it was transcribed and by what method.
        entry.insert(
            "measured_by".into(),
            serde_json::Value::String(measured_by.clone()),
        );
        cases.insert((*label).to_string(), serde_json::Value::Object(entry));
    }

    let path = fixture_path(BASELINE);
    let existing = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("an import EXTENDS a committed baseline; {BASELINE}: {e}"));
    let mut doc: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&existing).expect("existing baseline is a JSON object");
    let by_arch = doc
        .entry("by_arch")
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()))
        .as_object_mut()
        .expect("by_arch is an object");
    assert!(
        !by_arch.contains_key(&arch),
        "arch={arch} is already recorded; an import must never silently replace measured \
         signatures. Remove the entry deliberately first if it is genuinely being re-taken."
    );
    by_arch.insert(arch.clone(), serde_json::Value::Object(cases));

    let text = serde_json::to_string_pretty(&doc).expect("baseline serialises");
    std::fs::write(&path, text + "\n").expect("baseline is writable");
    println!(
        "INVARIANCE IMPORT: wrote arch={arch} into {}",
        path.display()
    );
}

/// `YYYY-MM-DD` for today, UTC, without adding a calendar dependency (D-17).
fn iso_date_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("the clock is after 1970")
        .as_secs();
    let days = i64::try_from(secs / 86_400).expect("days since epoch fit in i64");
    // `days_from_civil(1970, 1, 1)` is 0 by definition — the epoch is the origin.
    crate::dates::format_ymd(days)
}

// ------------------------------------------------------------ the gate ----

/// Read the committed baseline. Absence is a DEFECT, never a skip.
fn baseline() -> serde_json::Value {
    load_json(BASELINE)
}

/// Every recorded pre-change case reproduces its signature, on the architecture it was
/// recorded for.
///
/// When the running arch IS recorded, all eight cases are compared and the recorded count
/// is asserted to be exactly 8 — so a file that lost cases cannot pass by comparing fewer.
/// When it is NOT recorded, this prints one loud line naming the running arch, the archs
/// that are recorded and the exact capture recipe, and returns. That is a skip with a
/// stated reason and a stated way to close it; [`the_baseline_records_at_least_one_architecture`]
/// is what stops it from degenerating into a silent pass.
#[test]
fn every_baseline_case_reproduces_its_signature() {
    let doc = baseline();
    let by_arch = doc["by_arch"]
        .as_object()
        .expect("the baseline must carry a by_arch object");
    let arch = std::env::consts::ARCH;

    let Some(recorded) = by_arch.get(arch).and_then(|v| v.as_object()) else {
        let known: Vec<&str> = by_arch.keys().map(String::as_str).collect();
        println!(
            "INVARIANCE SKIP: no baseline recorded for arch={arch} (recorded: {known:?}); \
             this gate is NOT proving anything on this host. Capture one with: {CAPTURE_RECIPE}"
        );
        return;
    };

    assert_eq!(
        recorded.len(),
        CASES.len(),
        "the baseline for arch={arch} records {} cases; it must record all {} door cases, \
         or the gate is only watching part of the surface",
        recorded.len(),
        CASES.len()
    );

    // Accumulate EVERY case's outcome across the FULL loop and assert ONCE at the end. An
    // `assert_eq!` inside this loop panics at the first mismatch and never evaluates the
    // remaining cases, so a fix that repairs case 1 but not cases 2..N would read as fully
    // green right up until case 2 is reached on a later run. Measured, not assumed: this is
    // exactly the blind spot that made "only peyton/prophet/default is affected" an
    // early-termination artifact rather than a finding (see debug session evidence s7).
    let mut mismatches: Vec<String> = Vec::new();
    for (label, csv, horizon, shape) in CASES {
        let want = recorded
            .get(label)
            .and_then(|v| v["signature"].as_str())
            .unwrap_or_else(|| panic!("baseline for arch={arch} must record case {label}"));
        let args = args_for(csv, horizon, shape);
        let r = crate::forecast::forecast(&args)
            .unwrap_or_else(|e| panic!("case {label} must succeed through the door: {e}"));
        let got = format!("{:016x}", signature(&r));
        if got != want {
            mismatches.push(format!("  case {label}: recorded {want}, got {got}"));
        }
    }

    assert!(
        mismatches.is_empty(),
        "{} of {} cases on arch={arch} no longer reproduce their pre-change signature. A \
         caller who passes NO new argument is getting a different answer, so this delivery \
         is not a one-line tag bump for them.\n{}",
        mismatches.len(),
        CASES.len(),
        mismatches.join("\n")
    );
}

/// The baseline records at least one architecture, and every architecture it records is
/// complete.
///
/// This is the anti-vacuity companion to [`every_baseline_case_reproduces_its_signature`]:
/// that test skips when the running arch is absent, so without this one a baseline that was
/// emptied — or written with `by_arch: {}` — would sail through on every host.
#[test]
fn the_baseline_records_at_least_one_architecture() {
    let doc = baseline();
    let by_arch = doc["by_arch"]
        .as_object()
        .expect("the baseline must carry a by_arch object");
    assert!(
        !by_arch.is_empty(),
        "by_arch is empty: the gate would skip on every host and prove nothing"
    );
    for (arch, cases) in by_arch {
        let cases = cases
            .as_object()
            .unwrap_or_else(|| panic!("by_arch.{arch} must be an object of cases"));
        assert_eq!(
            cases.len(),
            CASES.len(),
            "by_arch.{arch} records {} cases; every recorded arch must carry all {}",
            cases.len(),
            CASES.len()
        );
        for (label, entry) in cases {
            let sig = entry["signature"]
                .as_str()
                .unwrap_or_else(|| panic!("by_arch.{arch}.{label}.signature must be a string"));
            assert_eq!(
                sig.len(),
                16,
                "by_arch.{arch}.{label}.signature must be 16 lowercase hex digits, got {sig:?}"
            );
            assert!(
                sig.chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "by_arch.{arch}.{label}.signature must be lowercase hex, got {sig:?}"
            );
        }
    }
    assert!(
        doc.get("captured_at_commit")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|s| !s.trim().is_empty()),
        "the baseline must record captured_at_commit: a signature with no provenance \
         cannot be traced to the tree it describes"
    );
}

/// The signature is able to FAIL — a guard that cannot go red is theatre.
///
/// Three mutations at the smallest representable scale (1 ULP) plus a structural one, then
/// a restore. Plan 06.1-04 owns the full mutation ladder; this is the in-plan floor that
/// stops the gate from being committed vacuous.
#[test]
fn the_signature_detects_a_one_ulp_change() {
    let args = args_for("peyton_manning.csv", 30, Shape::ProphetDefault);
    let mut r = crate::forecast::forecast(&args).expect("the default case must succeed");
    let clean = signature(&r);

    let orig = r.yhat[0];
    r.yhat[0] = f64::from_bits(orig.to_bits() + 1);
    assert_ne!(signature(&r), clean, "a 1-ULP change in yhat[0] must show");
    r.yhat[0] = orig;

    let last = r.trend.len() - 1;
    let orig_t = r.trend[last];
    r.trend[last] = f64::from_bits(orig_t.to_bits() + 1);
    assert_ne!(
        signature(&r),
        clean,
        "a 1-ULP change in trend[last] must show"
    );
    r.trend[last] = orig_t;

    r.components.insert(
        "phantom".into(),
        serde_json::Value::Array(vec![serde_json::Value::Null]),
    );
    assert_ne!(signature(&r), clean, "an extra component key must show");
    r.components.remove("phantom");

    assert_eq!(
        signature(&r),
        clean,
        "reverting every mutation must return the original signature, or the detector is \
         reacting to something other than what was changed"
    );
}

// ------------------------------------------------- part A: determinism ----

/// The four hashed f64 arrays, in the signature's own order, so the divergence walk and the
/// signature cannot disagree about which fields or which order they are looking at.
fn float_fields(r: &ForecastResponse) -> [(&'static str, &[f64]); 4] {
    [
        ("yhat", r.yhat.as_slice()),
        ("yhat_lower", r.yhat_lower.as_slice()),
        ("yhat_upper", r.yhat_upper.as_slice()),
        ("trend", r.trend.as_slice()),
    ]
}

/// This response's `diagnostics` with `lbfgs.budget_hit` forced to a fixed value.
///
/// Used ONLY by the mismatch triage below, never by the gate itself: the key stays inside
/// the signature (see [`mismatch_report`] for why removing it would be a permanent
/// weakening).
fn diagnostics_with_budget_hit_masked(r: &ForecastResponse) -> serde_json::Value {
    let mut d = r.diagnostics.clone();
    if let Some(flag) = d.get_mut("lbfgs").and_then(|l| l.get_mut("budget_hit")) {
        *flag = serde_json::Value::Bool(false);
    }
    d
}

/// A diagnostic for two runs of one case whose signatures disagree.
///
/// A bare "signatures differ" sends the reader back to the whole response. This is written
/// for the place the failure is hardest to chase — CI, where nobody can re-run it
/// interactively — so it names the first divergent BIT, the scale of the divergence per
/// field, and whether the one wall-clock-dependent flag inside `diagnostics` accounts for
/// it on its own.
fn mismatch_report(label: &str, a: &ForecastResponse, b: &ForecastResponse) -> String {
    let mut s = format!(
        "case {label} produced two different signatures on the SAME host, from two calls \
         made back to back with the same arguments: {:016x} then {:016x}.\n",
        signature(a),
        signature(b)
    );

    // (1) The FIRST divergent element, by field, index and bit pattern.
    let mut first: Option<String> = None;
    for ((field, xa), (_, xb)) in float_fields(a).into_iter().zip(float_fields(b)) {
        if xa.len() != xb.len() {
            first = Some(format!(
                "  first divergence: {field} has {} values in run 1 and {} in run 2 — a \
                 STRUCTURAL break, not arithmetic drift\n",
                xa.len(),
                xb.len()
            ));
            break;
        }
        if let Some(i) = xa
            .iter()
            .zip(xb)
            .position(|(p, q)| p.to_bits() != q.to_bits())
        {
            let (p, q) = (xa[i], xb[i]);
            first = Some(format!(
                "  first divergence: {field}[{i}]  run1={p:e} 0x{:016x}  run2={q:e} \
                 0x{:016x}  abs diff {:e}\n",
                p.to_bits(),
                q.to_bits(),
                (p - q).abs()
            ));
            break;
        }
    }
    s.push_str(&first.unwrap_or_else(|| {
        String::from(
            "  first divergence: NONE of yhat, yhat_lower, yhat_upper or trend differs by a \
             single bit, so the difference is in ds, components or diagnostics — look \
             there, not at the arithmetic\n",
        )
    }));

    // (2) The whole-response max absolute difference, per field, so a one-ULP libm
    //     difference reads differently from a structural break.
    s.push_str("  max abs difference per field:");
    for ((field, xa), (_, xb)) in float_fields(a).into_iter().zip(float_fields(b)) {
        let m = xa
            .iter()
            .zip(xb)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0_f64, f64::max);
        s.push_str(&format!(" {field}={m:e}"));
    }
    s.push('\n');

    // (3) The budget_hit note, as a POINTER and not as a subtraction.
    let masked_a = signature_with(a, &diagnostics_with_budget_hit_masked(a));
    let masked_b = signature_with(b, &diagnostics_with_budget_hit_masked(b));
    let verdict = if masked_a == masked_b {
        "EQUAL"
    } else {
        "STILL UNEQUAL"
    };
    s.push_str(&format!(
        "  with diagnostics.lbfgs.budget_hit masked to a fixed value, the two signatures \
         are {verdict}.\n\
         \x20   READ THAT AS A POINTER, NOT AS A SUBTRACTION. budget_hit records whether \
         fit::fit_prophet's cooperative round-boundary budget (FIT_BUDGET_SECS, \
         fit.rs:104 and fit.rs:112) was hit, and hitting it BREAKS the optimisation loop \
         — so when the flag flips, the parameters, the iteration count, the objective, \
         the predictions and the bands all differ too.\n\
         \x20   MASKED-EQUAL means: the ONLY difference is the flag, so the two fits \
         genuinely agreed and this is a loaded machine rather than a regression.\n\
         \x20   MASKED-STILL-UNEQUAL does NOT mean the opposite. A REAL budget hit changes \
         far more than the flag, so this line cannot rule an environment effect out; it \
         can only ever rule one IN.\n\
         \x20   budget_hit is deliberately NOT carved out of the signature. SC2 puts \
         diagnostics inside the comparison, and removing one key to dodge a failure mode \
         that has never been observed weakens the gate permanently in exchange for a \
         convenience.\n"
    ));
    s
}

/// Part A: every recorded door case is DETERMINISTIC — the same host, twice, same answer.
///
/// This is the claim that makes the committed baseline mean something. Without it, a
/// baseline mismatch has two readings — "the code changed" and "this run was noisy" — and
/// the second reading is always available to whoever does not want to believe the first.
///
/// UNCONDITIONAL on every architecture, by construction: it compares one host against
/// itself and never against a recorded value, so libm differences cannot reach it. See the
/// module docs for which part carries which exposure.
#[test]
fn part_a_every_case_is_deterministic_through_the_door() {
    // Accumulate across the FULL loop and assert ONCE, for the same reason
    // `every_baseline_case_reproduces_its_signature` does: an `assert!` inside the loop stops
    // at the first mismatch and never evaluates the remaining cases, so the failure reports
    // "1 of 8" whatever the true blast radius is. Part A is the WORSE place to leave that —
    // it is the unconditional, cross-arch test, the one most likely to fire on a runner with
    // a different libm, and a misread blast radius there sends the next investigation at a
    // single case when the class is general.
    let mut exercised = 0usize;
    let mut nondeterministic: Vec<String> = Vec::new();
    for (label, csv, horizon, shape) in CASES {
        let args = args_for(csv, horizon, shape);
        let first = crate::forecast::forecast(&args)
            .unwrap_or_else(|e| panic!("case {label} must succeed through the door: {e}"));
        let second = crate::forecast::forecast(&args)
            .unwrap_or_else(|e| panic!("case {label} must succeed on its second call: {e}"));
        if signature(&first) != signature(&second) {
            nondeterministic.push(mismatch_report(label, &first, &second));
        }
        exercised += 1;
    }
    assert!(
        nondeterministic.is_empty(),
        "{} of {} cases are NOT deterministic through the door.\n{}",
        nondeterministic.len(),
        CASES.len(),
        nondeterministic.join("\n")
    );
    // Non-vacuity, pinned and PRINTED: a determinism table that silently shrank would
    // otherwise report green while proving less. Same discipline as the poisson sweep's
    // checked-lambda count.
    assert!(
        exercised == CASES.len() && exercised == 8,
        "part A exercised {exercised} cases; it must exercise all 8 recorded door cases \
         (CASES.len() = {})",
        CASES.len()
    );
}

/// The two wall-clock fields are proven EXCLUDED, not merely documented as excluded.
///
/// Without this, "we left the timings out" is a claim about the code rather than a property
/// of the function. And the failure it guards against is vacuity in the OPPOSITE direction:
/// a signature carrying wall-clock can never reproduce, so the gate would be permanently
/// red, therefore permanently ignored, which is exactly as useless as permanently green.
#[test]
fn the_signature_ignores_the_two_wall_clock_fields() {
    let args = args_for("retail_sales.csv", 12, Shape::ProphetMonthly);
    let mut r = crate::forecast::forecast(&args).expect("the retail case must succeed");
    let before = signature(&r);
    let (fit0, predict0) = (r.fit_seconds, r.predict_seconds);
    assert!(
        fit0.is_finite() && predict0.is_finite(),
        "the door must report finite timings, got fit_seconds={fit0} \
         predict_seconds={predict0}"
    );

    r.fit_seconds = fit0 + 1.0;
    r.predict_seconds = predict0 + 2.0;
    assert!(
        r.fit_seconds.to_bits() != fit0.to_bits()
            && r.predict_seconds.to_bits() != predict0.to_bits(),
        "the mutation must actually change both fields, or this test proves nothing"
    );
    assert!(
        signature(&r) == before,
        "replacing BOTH wall-clock fields changed the signature, so the gate is hashing \
         time; it could never reproduce and would be red on every host forever"
    );
}

// ------------------------------------- part B: the falsification probe ----

/// Step a finite f64 to its next representable neighbour — one unit in the last place.
///
/// Computed from the BIT PATTERN with `to_bits` / `from_bits`, never by adding a chosen
/// epsilon, and that distinction is the whole probe. A 1-ULP change is BELOW print
/// precision, which is exactly what a formatted comparison silently accepts; an epsilon
/// large enough to be reliable across magnitudes would also be large enough to show up in
/// a printed decimal, and would therefore test a weaker property than the one claimed.
fn one_ulp_step(v: f64) -> f64 {
    let bits = v.to_bits();
    let stepped = f64::from_bits(bits.wrapping_add(1));
    assert!(
        stepped.to_bits() != bits,
        "the ULP step must change the bit pattern of {v:e}"
    );
    assert!(
        stepped.is_finite(),
        "the ULP step of {v:e} left the finite range; probe a different element"
    );
    stepped
}

/// The mutation MUST move the signature.
fn detected(clean: u64, r: &ForecastResponse, what: &str) {
    assert!(
        signature(r) != clean,
        "mutation [{what}] did NOT change the signature, so the signature is hashing less \
         than the field list claims. A green invariance table produced by a detector that \
         cannot see this change is indistinguishable from a broken harness — which is the \
         exact condition SC2 refuses to accept as satisfied"
    );
}

/// Reverting it MUST bring the signature back.
fn restored(clean: u64, r: &ForecastResponse, what: &str) {
    assert!(
        signature(r) == clean,
        "reverting mutation [{what}] did not restore the clean signature, so the probe is \
         reacting to something other than what it changed and proves nothing about that \
         field"
    );
}

/// Part B: the signature is PROVEN ABLE TO FAIL, on six mutation shapes, each reverted.
///
/// SC2's wording is the specification here: a green gate with no falsification probe beside
/// it does not satisfy the criterion. A `signature` that returned a constant, or that
/// quietly skipped a field, would produce exactly the same green table as a correct one on
/// the baseline comparison, on part A AND on part C. Part B is the only thing in this
/// module that makes the difference observable.
///
/// **Mutations 4 to 6 are not padding.** SC2 puts the uncertainty bands and `diagnostics`
/// INSIDE the comparison, and plan 06.1-03 adds regressor keys to `diagnostics` whose
/// absence-when-unused is the mechanism that keeps D-19 free. Nothing else in this module
/// exercises those three fields: their inclusion is claimed by `signature`'s field list and,
/// without these three mutations, tested by nothing.
///
/// UNCONDITIONAL on every architecture, by construction: it perturbs a response already in
/// memory and never crosses a libm boundary.
#[test]
fn part_b_the_signature_is_proven_able_to_fail() {
    let args = args_for("peyton_manning.csv", 30, Shape::ProphetDefault);
    let mut r = crate::forecast::forecast(&args)
        .expect("the peyton/prophet/default case must succeed through the door");
    let clean = signature(&r);
    let mut shapes = 0usize;

    // ---- 1. the FIRST yhat value, one unit in the last place ----
    let last_yhat = r.yhat.len() - 1;
    let last_trend = r.trend.len() - 1;
    let last_upper = r.yhat_upper.len() - 1;
    assert!(
        last_yhat > 0 && last_trend > 0 && last_upper > 0,
        "the probe needs a multi-row response; got yhat {}, trend {}, yhat_upper {}",
        r.yhat.len(),
        r.trend.len(),
        r.yhat_upper.len()
    );

    let orig = r.yhat[0];
    r.yhat[0] = one_ulp_step(orig);
    detected(clean, &r, "1: yhat[0] + 1 ULP");
    r.yhat[0] = orig;
    restored(clean, &r, "1: yhat[0]");
    shapes += 1;

    // ---- 2. the LAST trend value, one unit in the last place ----
    let orig = r.trend[last_trend];
    r.trend[last_trend] = one_ulp_step(orig);
    detected(clean, &r, "2: trend[last] + 1 ULP");
    r.trend[last_trend] = orig;
    restored(clean, &r, "2: trend[last]");
    shapes += 1;

    // ---- 3. one extra key in `components` (structural, not numeric) ----
    r.components.insert(
        "phantom".into(),
        serde_json::Value::Array(vec![serde_json::Value::Null]),
    );
    detected(clean, &r, "3: an extra components key");
    r.components.remove("phantom");
    restored(clean, &r, "3: the extra components key");
    shapes += 1;

    // ---- 4. the FIRST yhat_lower value — the LOWER band, inside the comparison ----
    let orig = r.yhat_lower[0];
    r.yhat_lower[0] = one_ulp_step(orig);
    detected(clean, &r, "4: yhat_lower[0] + 1 ULP");
    r.yhat_lower[0] = orig;
    restored(clean, &r, "4: yhat_lower[0]");
    shapes += 1;

    // ---- 5. the LAST yhat_upper value — the UPPER band, inside the comparison ----
    let orig = r.yhat_upper[last_upper];
    r.yhat_upper[last_upper] = one_ulp_step(orig);
    detected(clean, &r, "5: yhat_upper[last] + 1 ULP");
    r.yhat_upper[last_upper] = orig;
    restored(clean, &r, "5: yhat_upper[last]");
    shapes += 1;

    // ---- 6. one value inside `diagnostics`, in two sub-cases: numeric and boolean ----
    let obj = r.diagnostics["lbfgs"]["objective"]
        .as_f64()
        .expect("diagnostics.lbfgs.objective must be a number");
    let number = |v: f64| {
        serde_json::Value::Number(
            serde_json::Number::from_f64(v).expect("a finite f64 is a JSON number"),
        )
    };
    r.diagnostics["lbfgs"]["objective"] = number(one_ulp_step(obj));
    detected(clean, &r, "6a: diagnostics.lbfgs.objective + 1 ULP");
    r.diagnostics["lbfgs"]["objective"] = number(obj);
    restored(clean, &r, "6a: diagnostics.lbfgs.objective");

    let flag = r.diagnostics["lbfgs"]["budget_hit"]
        .as_bool()
        .expect("diagnostics.lbfgs.budget_hit must be a boolean");
    r.diagnostics["lbfgs"]["budget_hit"] = serde_json::Value::Bool(!flag);
    detected(clean, &r, "6b: diagnostics.lbfgs.budget_hit flipped");
    r.diagnostics["lbfgs"]["budget_hit"] = serde_json::Value::Bool(flag);
    restored(clean, &r, "6b: diagnostics.lbfgs.budget_hit");
    shapes += 1;

    // Non-vacuity, for the same reason part A pins its case count.
    assert!(
        shapes == 6,
        "part B exercised {shapes} mutation shapes; it must exercise all 6 — two point \
         estimates, one structural, BOTH uncertainty bands and one diagnostics leaf — or \
         the fields SC2 names are claimed by the field list and tested by nothing"
    );
    assert!(
        signature(&r) == clean,
        "after reverting every mutation the signature must equal the clean value captured \
         before the first one; a probe that cannot get back is testing the wrong thing"
    );
}

/// The ONE deliberate insensitivity in the hasher, stated as a property rather than a
/// comment: `-0.0` and `0.0` hash the same. Every other bit pattern is significant.
#[test]
fn the_signature_normalises_negative_zero() {
    let args = args_for("retail_sales.csv", 12, Shape::ProphetMonthly);
    let mut r = crate::forecast::forecast(&args).expect("the retail case must succeed");
    let last = r.trend.len() - 1;

    r.yhat[0] = 0.0;
    r.trend[last] = 0.0;
    let positive = signature(&r);

    r.yhat[0] = -0.0;
    r.trend[last] = -0.0;
    assert!(
        r.yhat[0].to_bits() != 0.0_f64.to_bits() && r.trend[last].to_bits() != 0.0_f64.to_bits(),
        "the two zeros must really differ in their BIT patterns, or this test is comparing \
         a value with itself"
    );
    assert!(
        signature(&r) == positive,
        "-0.0 must hash as 0.0 in both yhat and trend. This is the hasher's single \
         deliberate blind spot and it is deliberate because a sign flip on an arithmetic \
         zero is not a behaviour change; every OTHER bit pattern, NaN payloads included, \
         is significant"
    );
}

// ------------------------------------- part C: the mechanism, measured ----

/// Two f64 slices are equal BY BITS, with a diagnostic naming the first divergent element.
///
/// Not an epsilon comparison. The claim part C makes is bit equality, and an epsilon would
/// accept exactly the drift class this module exists to catch.
fn bitwise_equal(series: &str, field: &str, a: &[f64], b: &[f64]) {
    assert!(
        a.len() == b.len(),
        "{series}: {field} has {} values without the splice and {} after splicing zero \
         regressors; the splice changed a LENGTH, which is a structural break",
        a.len(),
        b.len()
    );
    if let Some(i) = a
        .iter()
        .zip(b)
        .position(|(p, q)| p.to_bits() != q.to_bits())
    {
        let (p, q) = (a[i], b[i]);
        panic!(
            "{series}: {field}[{i}] is not bit-identical — unspliced {p:e} 0x{:016x} vs \
             spliced-with-nothing {q:e} 0x{:016x}, abs diff {:e}. Splicing ZERO regressors \
             must be inert; if it is not, D-19 is not free and every caller who passes no \
             new argument is getting a different answer",
            p.to_bits(),
            q.to_bits(),
            (p - q).abs()
        );
    }
}

/// A bit-exact fingerprint of one fitted parameter vector.
fn params_fingerprint(p: &crate::prophet::Params) -> u64 {
    let mut h = Hasher::new();
    h.f64(p.k);
    h.f64(p.m);
    h.f64s(&p.delta);
    h.f64s(&p.beta);
    h.f64(p.sigma_obs);
    h.finish()
}

/// Part C: `regressors::splice` at ZERO regressors is inert — MEASURED, not asserted.
///
/// # What this part claims, and what it explicitly does not
///
/// It compares the PRE-SPLICE design against the spliced-with-nothing design, through the
/// SAME (post-change) [`crate::prophet::predict`]. It therefore establishes **SPLICE
/// INERTNESS at zero regressors** and nothing wider. It does NOT establish that the
/// `predict` signature change itself was inert: both sides call the same post-change
/// function, so a regression common to the empty-channel branch moves both outputs equally
/// and part C stays green.
///
/// That second claim — the historical one — is carried by the committed baseline in
/// `invariance_baseline.json`, which was captured on a commit at which `regressors.rs` did
/// not exist and is the only INDEPENDENT historical evidence in this gate. The two are
/// deliberately different statements; see the module docs for the per-part table.
///
/// # part_c_is_a_measurement_not_an_assertion
///
/// The reason D-19 is free is a COLUMN-ORDER fact: regressor columns append, so no existing
/// column index moves. A comment saying "appending is harmless" is not a gate — it is the
/// same claim the gate exists to check, written where nothing can check it. So this part
/// runs the real `make_design`, the real `splice` and the real `predict` on three real
/// series and compares the results bit for bit.
///
/// # The bands are INSIDE the comparison
///
/// The spike's part C covered point estimates only, because its prototype computed no
/// bands, and the reference flags that the band path resamples changepoints around a `yhat`
/// that depends on `beta` and `X` — so the bands must be added once the feature lands
/// in-crate. This test is that close-out: `yhat_lower` and `yhat_upper` are compared here,
/// not excused.
///
/// # Each series is FIT ONCE
///
/// Both `predict` calls take the SAME `Params`. Fitting each design independently would
/// fold L-BFGS's own run-to-run behaviour into a test whose claim is about the DESIGN, and
/// the budget check at `fit.rs:104` / `fit.rs:112` BREAKS the optimisation loop on
/// wall-clock — which changes the parameters, the iteration count, the objective, the
/// predictions and the bands, not merely a flag. Masking `budget_hit` cannot subtract that,
/// because by the time the flag differs the parameters already differ. Fitting once removes
/// the fit from the variable set entirely, so the only input that differs between the two
/// calls is the design, which is exactly the claim.
#[test]
fn part_c_the_splice_is_inert_at_zero_regressors() {
    use crate::prophet::{auto_seasonalities, make_design, predict, Mode, Spec};

    /// `(csv, horizon, freq)` — the three series, as the prophet door would take them.
    const SERIES: [(&str, usize, &str); 3] = [
        ("peyton_manning.csv", 30, "D"),
        ("retail_sales.csv", 12, "MS"),
        ("air_passengers.csv", 12, "MS"),
    ];
    const SEED: u64 = 42;

    let mut series_compared = 0usize;
    for (csv, horizon, freq) in SERIES {
        let (ds_text, y) = read_csv(csv);
        let ds: Vec<i64> = ds_text.iter().map(|s| crate::dates::parse_ymd(s)).collect();
        let fut = crate::dates::future_days(ds[ds.len() - 1], horizon, freq)
            .unwrap_or_else(|e| panic!("{csv}: the future grid must build: {e}"));

        // The prophet arm's defaulted spec, built the way `forecast.rs:317` builds it.
        let spec = Spec::default_linear(auto_seasonalities(&ds, 10.0, Mode::Additive));
        assert!(
            !spec.seasonalities.is_empty(),
            "{csv}: the defaulted spec must carry at least one seasonality, or this \
             series exercises an almost-empty design and proves very little"
        );

        // ---- THE DESIGN: one straight from make_design, one spliced with nothing ----
        let plain = make_design(&ds, &y, &spec);
        let mut spliced = make_design(&ds, &y, &spec);
        crate::regressors::splice(&mut spliced, &[], &[]);

        assert!(
            plain.k == spliced.k && plain.k > 0,
            "{csv}: k must survive the empty splice unchanged and be non-zero, got {} vs {}",
            plain.k,
            spliced.k
        );
        assert!(
            plain.cols.len() == spliced.cols.len(),
            "{csv}: the column COUNT changed across an empty splice, {} vs {}",
            plain.cols.len(),
            spliced.cols.len()
        );
        for (i, (want, got)) in plain.cols.iter().zip(&spliced.cols).enumerate() {
            assert!(
                want.name == got.name
                    && want.component == got.component
                    && want.mode == got.mode
                    && want.prior_scale.to_bits() == got.prior_scale.to_bits()
                    && want.holiday == got.holiday,
                "{csv}: column {i} changed across an empty splice: {want:?} vs {got:?}"
            );
        }
        bitwise_equal(csv, "design.x", &plain.x, &spliced.x);
        bitwise_equal(csv, "design.s_a", &plain.s_a, &spliced.s_a);
        bitwise_equal(csv, "design.s_m", &plain.s_m, &spliced.s_m);
        bitwise_equal(
            csv,
            "design.prior_scales",
            &plain.prior_scales,
            &spliced.prior_scales,
        );

        // ---- THE FORECAST: fit ONCE, the same Params and the same seed into both ----
        let (params, _info) = crate::fit::fit_prophet(&plain, 8);
        let before = params_fingerprint(&params);
        let none = crate::regressors::RegressorChannel::NONE;
        // Both seeds read from the same binding and are compared below, so a band
        // difference can only come from the design and never from the RNG.
        let seeds = [SEED, SEED];
        let unspliced = predict(&plain, &params, &fut, seeds[0], &none)
            .unwrap_or_else(|e| panic!("{csv}: predict on the unspliced design: {e}"));
        let after_splice = predict(&spliced, &params, &fut, seeds[1], &none)
            .unwrap_or_else(|e| panic!("{csv}: predict on the spliced design: {e}"));
        assert!(
            seeds[0] == seeds[1],
            "{csv}: both predict calls must receive the SAME seed"
        );
        assert!(
            params_fingerprint(&params) == before,
            "{csv}: the fitted parameters changed between the two predict calls; the only \
             input allowed to differ is the design"
        );

        assert!(
            unspliced.yhat.len() == horizon,
            "{csv}: expected {horizon} predicted rows, got {}",
            unspliced.yhat.len()
        );
        bitwise_equal(csv, "yhat", &unspliced.yhat, &after_splice.yhat);
        bitwise_equal(csv, "trend", &unspliced.trend, &after_splice.trend);
        bitwise_equal(
            csv,
            "yhat_lower",
            &unspliced.yhat_lower,
            &after_splice.yhat_lower,
        );
        bitwise_equal(
            csv,
            "yhat_upper",
            &unspliced.yhat_upper,
            &after_splice.yhat_upper,
        );
        // Non-vacuity for the band half: comparing two degenerate zero-width bands would
        // pass without the band path ever having produced anything.
        assert!(
            unspliced
                .yhat_upper
                .iter()
                .zip(&unspliced.yhat_lower)
                .any(|(hi, lo)| hi > lo),
            "{csv}: the uncertainty band has zero width everywhere, so comparing it \
             proves nothing about the band path"
        );

        let mut shared = 0usize;
        for (name, left) in &unspliced.components {
            let Some((_, right)) = after_splice.components.iter().find(|(n, _)| n == name) else {
                panic!("{csv}: component {name} exists without the splice and is missing after it")
            };
            bitwise_equal(csv, name, left, right);
            shared += 1;
        }
        assert!(
            unspliced.components.len() == after_splice.components.len(),
            "{csv}: the component COUNT changed across an empty splice, {} vs {}",
            unspliced.components.len(),
            after_splice.components.len()
        );
        assert!(
            shared >= 3,
            "{csv}: only {shared} components were compared; the spike measured 3 shared \
             components and a mechanism test that compared none would report green"
        );

        series_compared += 1;
    }

    assert!(
        series_compared == SERIES.len() && series_compared == 3,
        "part C compared {series_compared} series; it must compare all 3 (SERIES.len() = {})",
        SERIES.len()
    );
}

/// Parts A and B are unconditional — enforced against this module's own source.
///
/// The cross-commit baseline comparison is arch-keyed for a real libm reason, so a
/// cross-platform red WILL eventually appear in CI. The tempting repair is to relax
/// whatever is red. This test makes that repair impossible to apply to the two parts SC2
/// actually rests on: part A compares one host against itself and part B perturbs a
/// response in memory, so neither can legitimately differ across runners, and neither may
/// be gated behind `#[ignore]` or an architecture branch.
#[test]
fn parts_a_and_b_are_unconditional() {
    const SRC: &str = include_str!("invariance.rs");
    // Assembled at runtime so the needles appear only in THIS function's body and cannot
    // make the test find itself.
    let arch_needle = concat!("consts", "::ARCH");
    let ignore_needle = concat!("#[", "ignore");

    for name in [
        "part_a_every_case_is_deterministic_through_the_door",
        "part_b_the_signature_is_proven_able_to_fail",
    ] {
        // The trailing `(` matters: a bare `fn {name}` prefix-matches a RENAMED function,
        // so `part_a_..._door` silently resolved to `part_a_..._doorX` and the MISSING
        // direction never fired. Also measured, not reasoned.
        let fn_at = SRC
            .find(&format!("fn {name}("))
            .unwrap_or_else(|| panic!("MISSING test fn {name} in this module's own source"));
        // Walk BACK to the `#[test]` attribute. An extraction that starts at `fn` is blind
        // to exactly the gating this test exists to forbid, because `#[ignore]` sits
        // BEFORE `fn`, not inside the body — MEASURED, not reasoned: with the naive
        // `fn`-anchored slice, adding `#[ignore = "slow"]` to part A left this test GREEN.
        // (CLAUDE.md Verification Discipline 4 and 7: extending a guard's scope requires
        // re-mutating in the new scope, and the pattern is wrong until the case table says
        // otherwise.)
        let start = SRC[..fn_at]
            .rfind("#[test]")
            .unwrap_or_else(|| panic!("{name} is not preceded by a #[test] attribute"));
        assert!(
            !SRC[start..fn_at].contains("fn "),
            "the attribute walk-back for {name} crossed another function; the extracted \
             region does not belong to it"
        );
        let open = SRC[fn_at..]
            .find('{')
            .unwrap_or_else(|| panic!("{name} has no body"))
            + fn_at;
        let mut depth = 0usize;
        let mut end = None;
        for (off, ch) in SRC[open..].char_indices() {
            if ch == '{' {
                depth += 1;
            } else if ch == '}' {
                depth -= 1;
                if depth == 0 {
                    end = Some(open + off + 1);
                    break;
                }
            }
        }
        let end = end.unwrap_or_else(|| panic!("unbalanced braces in {name}"));
        let body = &SRC[start..end];
        // Self-check on the extractor: over-capturing would silently weaken the assertions
        // below into a scan of some other function.
        assert!(
            !body.contains("\n#[test]"),
            "the body extractor over-captured {name}; it swallowed the next test"
        );
        assert!(
            !body.contains(arch_needle),
            "{name} branches on the running architecture. It must not: it is \
             arch-independent by construction, and making it conditional is how a \
             cross-platform red gets silenced by removing the half of the gate that SC2 \
             rests on"
        );
        assert!(
            !body.contains(ignore_needle),
            "{name} carries an ignore attribute, so part of the release gate is \
             unreachable from a plain `cargo test --lib` run"
        );
    }
}
