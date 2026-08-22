#![allow(clippy::disallowed_methods)]
//! Popperian Falsification Tests -- Showcase Spec v10.4.0 (119 Gates)
//!
//! This file implements ALL 119 falsification gates from:
//!   docs/specifications/archive/qwen2.5-coder-showcase-demo.md
//!
//! **GATED BY `model-tests` FEATURE** — these tests do NOT run with `cargo test`.
//! Many tests load GGUF/SafeTensors models, start servers, call ollama, and use GPU.
//! Running all at once WILL OOM the system.
//!
//! Run with: `cargo test --features model-tests --test falsification_spec_v10_tests <TEST_NAME>`
//! Never run the entire file at once without filtering.
//!
//! "We do not try to prove our theories are true, but to show that they
//!  are false." -- K. Popper (1963)
#![cfg(feature = "model-tests")]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use aprender::format::layout_contract::{
    enforce_embedding_contract, enforce_import_contract, enforce_matmul_contract, LayoutContract,
};
use aprender::format::model_family::{
    build_default_registry, Activation, AttentionType, MlpType, NormType, PositionalEncoding,
    KNOWN_FAMILIES,
};
use aprender::format::rosetta::FormatType;
use aprender::format::validated_tensors::{RowMajor, ValidatedEmbedding, ValidatedWeight};
use tempfile::NamedTempFile;

// =============================================================================
// Helpers
// =============================================================================

fn collect_rs_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if !dir.exists() || !dir.is_dir() {
        return files;
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Skip target/ and hidden directories
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if name.starts_with('.') || name == "target" {
                    continue;
                }
                files.extend(collect_rs_files(&path));
            } else if path.extension().map_or(false, |ext| ext == "rs") {
                files.push(path);
            }
        }
    }
    files
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

// -----------------------------------------------------------------------------
// Crate-anchored source resolution (aprender#2522)
// -----------------------------------------------------------------------------
//
// 38 of this suite's gates failed for one shared reason: they asserted "symbol X
// appears in FILE Y" against a path literal. Three separate refactors moved the
// files without touching the symbols --
//
//   * APR-MONO moved the whole tree from `<root>/src/` to `crates/<crate>/src/`,
//   * #2231/#2236 lifted `src/format/types.rs` out into the `apr-format` crate,
//   * apr-cli split `lib.rs` into `validate.rs` + `commands_enum.rs` and
//     `commands/qa.rs` into `qa_report.rs` + `commands/output_verification.rs`.
//
// -- so every one of those gates failed while the property it names is still
// true. A path literal is the wrong anchor: which FILE holds a symbol is churn,
// which CRATE holds it is the boundary this architecture actually defends
// (aprender = training, realizar = inference, trueno = compute). Anchoring to
// the crate is therefore both more durable AND a stronger statement.
//
// `crate_src_text` fails loudly on a missing crate rather than returning an
// empty string, because "grep found nothing" and "there was nothing to grep"
// must never be the same verdict -- that equivalence is what let 38 of these
// report a symbol as absent when it was merely elsewhere.

/// Directory of a workspace crate. Panics if the crate is not where it claims.
fn crate_dir(crate_name: &str) -> PathBuf {
    let dir = project_root().join("crates").join(crate_name);
    assert!(
        dir.is_dir(),
        "crate `{crate_name}` is not at crates/{crate_name}. A crate rename must \
         update this suite; it must not silently make a gate vacuous."
    );
    dir
}

/// Every `.rs` file under a crate's `src/`, concatenated.
///
/// Use this instead of reading one hardcoded file: it survives an intra-crate
/// file move, which is the churn that broke 13 of the 38 (#2522).
fn crate_src_text(crate_name: &str) -> String {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static CACHE: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("crate_src_text cache poisoned");
    if let Some(hit) = guard.get(crate_name) {
        return (*hit).to_string();
    }

    let src = crate_dir(crate_name).join("src");
    assert!(
        src.is_dir(),
        "crate `{crate_name}` has no src/ directory at {}",
        src.display()
    );
    // Production sources only: a symbol that exists ONLY in a test file does not
    // satisfy "the implementation exists", which is what every caller asserts.
    let files: Vec<PathBuf> = collect_rs_files(&src)
        .into_iter()
        .filter(|p| is_scannable_production_source(p))
        .collect();
    assert!(
        !files.is_empty(),
        "crate `{crate_name}` src/ contains no .rs files -- refusing to evaluate a \
         gate against an empty corpus (a vacuous pass is worse than a failure)"
    );
    let mut text = String::new();
    for path in files {
        if let Ok(content) = std::fs::read_to_string(&path) {
            text.push_str(&content);
            text.push('\n');
        }
    }
    let leaked: &'static str = Box::leak(text.into_boxed_str());
    guard.insert(crate_name.to_string(), leaked);
    leaked.to_string()
}

/// The showcase spec markdown. It moved to `docs/specifications/archive/` and
/// five gates died on the old path; both locations are accepted, and neither
/// missing is a pass.
fn spec_text() -> String {
    let candidates = [
        project_root()
            .join("docs")
            .join("specifications")
            .join("archive")
            .join("qwen2.5-coder-showcase-demo.md"),
        project_root()
            .join("docs")
            .join("specifications")
            .join("qwen2.5-coder-showcase-demo.md"),
    ];
    for path in &candidates {
        if let Ok(content) = std::fs::read_to_string(path) {
            return content;
        }
    }
    panic!(
        "qwen2.5-coder-showcase-demo.md found at none of: {}",
        candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// This suite's own source: the root file plus every `includes/` fragment it
/// pulls in. The suite -- not the archived markdown -- is the living record of
/// which gates exist.
fn suite_source_text() -> String {
    let tests_dir = crate_dir("aprender-core").join("tests");
    let mut text = std::fs::read_to_string(tests_dir.join("falsification_spec_v10_tests.rs"))
        .expect("suite root readable");
    for name in suite_include_names() {
        let path = tests_dir.join("includes").join(&name);
        text.push_str(
            &std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("suite include {} unreadable: {e}", path.display())),
        );
        text.push('\n');
    }
    text
}

/// Every file this suite's root `include!`s, in declaration order.
fn suite_include_names() -> Vec<String> {
    let root = crate_dir("aprender-core")
        .join("tests")
        .join("falsification_spec_v10_tests.rs");
    let text = std::fs::read_to_string(&root).expect("suite root readable");
    let mut names = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix(concat!("include", "!(\"includes/")) else {
            continue;
        };
        if let Some(end) = rest.find('"') {
            names.push(rest[..end].to_string());
        }
    }
    names
}

/// A crate's build script plus every `build_*.rs` fragment it `include!`s.
///
/// aprender-core's build.rs is a dispatcher; the code that emits the const
/// proofs lives in `build_codegen.rs`, pulled in with `include!`. Reading
/// build.rs alone made F-PROVE-002 assert against a file that never contained
/// what it was looking for.
fn build_script_text(crate_name: &str) -> String {
    let dir = crate_dir(crate_name);
    let mut text = std::fs::read_to_string(dir.join("build.rs"))
        .unwrap_or_else(|e| panic!("{crate_name}/build.rs unreadable: {e}"));
    let includes: Vec<String> = text
        .lines()
        .filter_map(|l| {
            let rest = l.trim().strip_prefix(concat!("include", "!(\""))?;
            let end = rest.find('"')?;
            Some(rest[..end].to_string())
        })
        .collect();
    for name in includes {
        if let Ok(extra) = std::fs::read_to_string(dir.join(&name)) {
            text.push('\n');
            text.push_str(&extra);
        }
    }
    text
}

/// Source files that are the SUBJECT of a scan, not part of its universe.
///
/// `f_contract_006` asserted "no `struct ColumnMajor` exists anywhere" and then
/// matched the three string literals in its own body that spell the pattern. A
/// scan that includes its own definition can never pass. Test trees, examples
/// and benches are excluded for the same class of reason: an example that
/// deliberately compares column-major against row-major is evidence the gate is
/// working, not a violation of it.
fn is_scannable_production_source(path: &Path) -> bool {
    let s = path.to_string_lossy().replace('\\', "/");
    if s.contains("/tests/") || s.contains("/examples/") || s.contains("/benches/") {
        return false;
    }
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    !(name == "tests.rs" || name.starts_with("tests_") || name.contains("_tests"))
}

/// Every production `.rs` file in the workspace (see the exclusions above).
fn production_rs_files() -> Vec<PathBuf> {
    let mut files = Vec::new();
    for dir in [project_root().join("src"), project_root().join("crates")] {
        files.extend(
            collect_rs_files(&dir)
                .into_iter()
                .filter(|p| is_scannable_production_source(p)),
        );
    }
    assert!(
        !files.is_empty(),
        "production source scan found 0 files -- a scan over an empty universe \
         passes every assertion put to it"
    );
    files
}

/// A line with its trailing `//` comment removed.
///
/// `f_dod_005` bans a `_ => ...F32` catch-all. It skipped whole-line comments
/// and not trailing ones, so `_ => 4, // Default F32` -- a byte-WIDTH default,
/// no dtype in the code at all -- was reported as a silent dtype fallback. Four
/// of its seven violations were comments.
fn strip_trailing_comment(line: &str) -> &str {
    match line.find("//") {
        Some(pos) => &line[..pos],
        None => line,
    }
}

/// `F32` as a whole identifier token, not as a substring.
///
/// `contains("F32")` matched `ImmF32` in the PTX builder --
/// `_ => panic!("Expected ImmF32 source")` was reported as a silent dtype
/// fallback to F32. It is neither silent nor a fallback nor F32.
fn contains_f32_token(haystack: &str) -> bool {
    let bytes = haystack.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = haystack[from..].find("F32") {
        let start = from + rel;
        let end = start + 3;
        let before_ok =
            start == 0 || !(bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_');
        let after_ok =
            end >= bytes.len() || !(bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_');
        if before_ok && after_ok {
            return true;
        }
        from = end;
    }
    false
}

fn search_dir_for_file(search_root: &Path, filename: &str) -> Option<PathBuf> {
    let mut dirs_to_visit = vec![search_root.to_path_buf()];
    while let Some(dir) = dirs_to_visit.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                dirs_to_visit.push(path);
            } else if path.file_name().map_or(false, |n| n == filename) {
                return Some(path);
            }
        }
    }
    None
}

/// Candidate cargo target directories, most authoritative first.
///
/// `find_generated_file` used to look only in `<project_root>/target`. On this
/// repo that directory does not exist: `.cargo/config.toml` and every worktree
/// wrapper redirect `CARGO_TARGET_DIR` elsewhere, so the finder returned `None`
/// unconditionally. F-PROVE-007 failed on it; F-PROVE-003/004/005 wrote their
/// `None` branch as `if let Some(..)` and so passed VACUOUSLY -- three gates
/// reporting `ok` while never opening the file they exist to check.
///
/// `current_exe()` is the one source that cannot be wrong: this very test binary
/// lives at `<target>/<profile>/deps/`, whatever `--target-dir` was passed.
fn target_dir_candidates() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        // <target>/<profile>/deps/<binary>
        if let Some(target) = exe
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
        {
            roots.push(target.to_path_buf());
        }
    }
    if let Ok(dir) = std::env::var("CARGO_TARGET_DIR") {
        roots.push(PathBuf::from(dir));
    }
    roots.push(project_root().join("target"));
    roots
}

fn find_generated_file(filename: &str) -> Option<PathBuf> {
    for target_dir in target_dir_candidates() {
        for profile in &["debug", "release"] {
            let search_root = target_dir.join(profile).join("build");
            if !search_root.exists() {
                continue;
            }
            if let Some(found) = search_dir_for_file(&search_root, filename) {
                return Some(found);
            }
        }
    }
    None
}

// =============================================================================
// Model fixture helpers
// =============================================================================

/// Model directory: uses MODEL_DIR env var, or ./models/ relative to project root
fn model_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("MODEL_DIR") {
        PathBuf::from(dir)
    } else {
        project_root().join("models")
    }
}

/// Get path to the 0.5B GGUF model (fastest for testing)
fn gguf_model_path() -> Option<PathBuf> {
    let path = model_dir().join("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf");
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Get path to the 0.5B APR model (validates it can be read by apr CLI)
fn apr_model_path() -> Option<PathBuf> {
    let path = model_dir().join("qwen2.5-coder-0.5b-instruct-q4_k_m.apr");
    if !path.exists() {
        return None;
    }
    // Validate APR file is usable (not corrupt)
    let bin = apr_binary();
    let output = Command::new(&bin)
        .args(["tensors", path.to_str().unwrap()])
        .output()
        .ok()?;
    if output.status.success() {
        Some(path)
    } else {
        eprintln!(
            "SKIP: APR model exists but is corrupt/incompatible: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        None
    }
}

/// Get path to SafeTensors model directory (0.5B)
fn safetensors_model_dir() -> Option<PathBuf> {
    // GH-327: Use MODEL_DIR or HOME-relative path, never hard-coded absolute paths
    let candidates: Vec<PathBuf> = [
        std::env::var("MODEL_DIR")
            .ok()
            .map(|d| PathBuf::from(d).join("qwen2.5-coder-0.5b-instruct")),
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join("models/qwen2.5-coder-0.5b-instruct")),
    ]
    .into_iter()
    .flatten()
    .collect();

    candidates
        .into_iter()
        .find(|path| path.join("model.safetensors").exists())
}

/// Find the apr CLI binary (release preferred, then debug)
fn apr_binary() -> PathBuf {
    // GH-301: Use CARGO_TARGET_DIR if set, then standard target/, then PATH.
    // #2522: `CARGO_TARGET_DIR` is not set when cargo is invoked with the
    // `--target-dir` FLAG, which is how every worktree here builds -- so this
    // fell through to `<project_root>/target`, which does not exist, and then to
    // a bare `apr` on PATH. That is exactly the stale-binary trap CLAUDE.md
    // Step 0 forbids. `target_dir_candidates()` leads with `current_exe()`.
    let target_bases = target_dir_candidates();
    for base in &target_bases {
        let release = base.join("release").join("apr");
        if release.exists() {
            return release;
        }
        let debug = base.join("debug").join("apr");
        if debug.exists() {
            return debug;
        }
    }
    PathBuf::from("apr")
}

/// Find ollama binary if installed
fn which_ollama() -> Option<PathBuf> {
    let output = Command::new("which").arg("ollama").output().ok()?;
    if output.status.success() {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if path.is_empty() {
            None
        } else {
            Some(PathBuf::from(path))
        }
    } else {
        None
    }
}

/// Run apr CLI command, return (success, stdout, stderr)
fn run_apr(args: &[&str]) -> (bool, String, String) {
    let bin = apr_binary();
    let output = Command::new(&bin)
        .args(args)
        .current_dir(project_root())
        .output()
        .unwrap_or_else(|e| panic!("Failed to run apr at {}: {}", bin.display(), e));
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

/// Skip test if model not available (returns from calling function).
///
/// #2522: a skipped gate returns early and the harness prints `ok`. 30 call
/// sites do this, so the suite could report "140 passed" while proving nothing
/// -- the exact vacuity the repo's skip-class doctrine forbids ("a skip must be
/// counted and bounded, never silent").
///
/// This macro is only the tidiest form of that skip. `f_meta_002_skip_class_is_bounded`
/// bounds the whole class -- these 30 plus the 32 hand-written
/// `eprintln!("SKIP: ..."); return;` pairs elsewhere in the suite -- as a
/// shrink-only ratchet. Counting only the macro would have been a ratchet on
/// half the universe.
macro_rules! require_model {
    ($path_opt:expr, $name:expr) => {
        match $path_opt {
            Some(p) => p,
            None => {
                eprintln!(
                    "SKIP[fixture]: {} not found. Set MODEL_DIR or download with `apr pull`",
                    $name
                );
                return;
            }
        }
    };
}

// =============================================================================
// Section META: the suite's gates on itself (aprender#2522)
// =============================================================================

/// Every site in this suite at which a `#[test]` can return without asserting.
///
/// SHRINK-ONLY. Each one is a gate that prints `ok` while proving nothing,
/// which is how this suite could be 38-red and silently vacuous for months
/// while reporting to nobody.
///
/// #2627: the first version of this ratchet counted `require_model!(` sites
/// alone and called that "the fixture-skip class". It is not the class. A
/// `require_model!` is only the tidiest of the early returns here; the suite
/// also skips on `which_ollama().is_none()`, on `!apr_ok`, on a server that
/// never became ready, on a GPU that is absent -- hand-written
/// `eprintln!("SKIP: ..."); return;` pairs that the macro-only count could not
/// see. Bounding 30 of them and naming the bound after the whole class is the
/// same defect the suite exists to catch: a measurement whose universe is
/// smaller than the thing it claims to cover.
///
/// The universe is now BOTH halves, measured together (see `count_skip_sites`):
///
/// | half | measured 2026-08-22 |
/// |---|---|
/// | `require_model!(` fixture gates | 30 |
/// | other early `return;` inside a `#[test]` | 32 |
/// | **total** | **62** |
///
/// What this still does NOT cover, stated rather than implied: a gate that
/// asserts something trivially true, and a gate whose `assert!` is inside a
/// conditional that never fires. Those are vacuous without returning early,
/// and no textual count reaches them.
const SKIP_SITE_BASELINE: usize = 62;

/// How far below the baseline a measurement may drift before the ratchet
/// demands it be re-recorded. A gain that is not locked in is a gain that gets
/// silently spent.
const SKIP_SITE_SLACK: usize = 5;

/// Count the two halves of the silent-skip class in one fragment.
///
/// Returns `(require_model_sites, early_returns_in_tests)`.
///
/// The second half needs to know whether a `return;` sits inside a `#[test]`
/// function or inside a plain helper: `collect_command_names` in the CLI
/// fragment recurses and returns early three times as ordinary control flow,
/// and counting those would make the ratchet fire on a refactor that changed
/// no gate.
///
/// A test's extent is delimited by rustfmt's layout -- from the `fn` line that
/// follows `#[test]` to the next line that is exactly `}` -- and deliberately
/// NOT by counting braces. Brace counting was tried first and undercounted
/// f_ollama_00 by four: `s.split([',', '}'])` on line 116 carries a `'}'` CHAR
/// LITERAL, which drove the depth negative and ended the function 53 lines
/// early. A miscount here is invisible in exactly the way this ratchet exists
/// to prevent, so the rule that needs no lexer wins.
fn count_skip_sites(text: &str) -> (usize, usize) {
    let fixture = text.matches(concat!("require_model", "!(")).count();

    let mut early = 0usize;
    let mut saw_test_attr = false;
    let mut in_test = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if in_test {
            if line == "}" {
                in_test = false;
                saw_test_attr = false;
            } else if !trimmed.starts_with("//") && trimmed.contains("return;") {
                early += 1;
            }
            continue;
        }
        if trimmed.starts_with("#[test]") {
            saw_test_attr = true;
        } else if saw_test_attr && trimmed.starts_with("fn ") {
            in_test = true;
        } else if saw_test_attr && !trimmed.is_empty() && !trimmed.starts_with('#') {
            // Other attributes (#[ignore], #[should_panic]) may sit between
            // #[test] and fn; anything else means that #[test] was not ours.
            saw_test_attr = false;
        }
    }

    (fixture, early)
}

#[test]
fn f_meta_001_every_include_is_included() {
    // A fragment in tests/includes/ that the root file does not `include!` is
    // dead code that looks like coverage. This is the same defect as the suite
    // itself being named in no workflow, one level down.
    let includes_dir = crate_dir("aprender-core").join("tests").join("includes");
    let declared: std::collections::HashSet<String> = suite_include_names().into_iter().collect();
    assert!(
        !declared.is_empty(),
        "F-META-001: the suite root declares no include!() fragments"
    );

    let mut orphans = Vec::new();
    for name in &declared {
        let path = includes_dir.join(name);
        assert!(
            path.is_file(),
            "F-META-001: suite includes `{name}`, which does not exist at {}",
            path.display()
        );
    }
    // Fragments belonging to THIS suite must all be wired in. Other suites keep
    // their own fragments in the same directory, so only the v10 family and the
    // f_* fragments this suite owns are in scope.
    for entry in std::fs::read_dir(&includes_dir)
        .expect("includes/ readable")
        .flatten()
    {
        let name = entry.file_name().to_string_lossy().to_string();
        let is_ours = name.starts_with("falsification_spec_v10_");
        if is_ours && !declared.contains(&name) {
            orphans.push(name);
        }
    }
    assert!(
        orphans.is_empty(),
        "F-META-001: falsification_spec_v10 fragments that no include!() reaches: {}",
        orphans.join(", ")
    );
}

#[test]
fn f_meta_002_skip_class_is_bounded() {
    // The skip class may only shrink. A new early return inside a #[test] is a
    // new gate that reports `ok` when its precondition is absent, and CI has
    // neither models, nor ollama, nor a GPU.
    //
    // Counted over the `includes/` fragments only. Counting the root file too
    // would make THIS function part of its own universe -- the same
    // self-matching defect F-CONTRACT-006 shipped with.
    let tests_dir = crate_dir("aprender-core").join("tests");
    let mut fixture = 0usize;
    let mut early = 0usize;
    let names = suite_include_names();
    assert!(
        !names.is_empty(),
        "F-META-002: the suite declares no fragments -- an empty universe is \
         not a measurement"
    );
    for name in names {
        let text = std::fs::read_to_string(tests_dir.join("includes").join(&name))
            .unwrap_or_else(|e| panic!("suite include {name} unreadable: {e}"));
        let (f, e) = count_skip_sites(&text);
        fixture += f;
        early += e;
    }
    let sites = fixture + early;

    // The counter itself must not be inert. `require_model!` is known to be a
    // real, non-empty half of this class; if the scan returns zero for it the
    // measurement is broken, not the suite.
    assert!(
        fixture > 0 && early > 0,
        "F-META-002: counted {fixture} fixture gates and {early} other early \
         returns. A zero on either half means count_skip_sites stopped matching, \
         not that the skips went away."
    );

    assert!(
        sites <= SKIP_SITE_BASELINE,
        "F-META-002: {sites} silent-skip sites ({fixture} `require_model!` + \
         {early} other early returns inside #[test]) against a baseline of \
         {SKIP_SITE_BASELINE}. A gate that returns before asserting reports `ok` \
         while proving nothing -- convert it to a real assertion instead of \
         raising the baseline."
    );
    assert!(
        sites + SKIP_SITE_SLACK >= SKIP_SITE_BASELINE,
        "F-META-002: silent-skip sites fell to {sites} ({fixture} fixture + \
         {early} other). Lower SKIP_SITE_BASELINE to {sites} so the gain is \
         locked in."
    );
}

/// Join backslash-continued shell lines into one logical command.
///
/// A real CI step wraps: `--features model-tests` on one line, `--test <target>`
/// on the next. A per-PHYSICAL-line scan calls that unwired -- the first version
/// of F-META-003 did exactly that and went red against a correctly wired
/// workflow, which is the same defect the gate exists to catch, one level up.
fn fold_shell_continuations(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for raw in text.lines() {
        let piece = if buf.is_empty() {
            raw.to_string()
        } else {
            format!("{} {}", buf, raw.trim_start())
        };
        match piece.strip_suffix('\\') {
            Some(head) => buf = head.to_string(),
            None => {
                buf.clear();
                out.push(piece);
            }
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

#[test]
fn f_meta_003_suite_is_named_by_a_workflow() {
    // The whole point of #2522. This suite carries 140 gates and appeared in no
    // GitHub workflow, so 38 failures sat unseen for months. A test that asserts
    // its own wiring cannot be silently unwired: deleting the CI step turns this
    // gate red in whatever else still runs it.
    let workflows = project_root().join(".github").join("workflows");
    let mut wired = false;
    let mut scanned = 0usize;
    for entry in std::fs::read_dir(&workflows)
        .expect(".github/workflows readable")
        .flatten()
    {
        let path = entry.path();
        if path.extension().map_or(true, |e| e != "yml" && e != "yaml") {
            continue;
        }
        scanned += 1;
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in fold_shell_continuations(&text) {
            // EXECUTION, not mention: a name inside a `#` comment is not wiring.
            let code = match line.find('#') {
                Some(pos) => line[..pos].to_string(),
                None => line,
            };
            if code.contains("--test falsification_spec_v10_tests") && code.contains("--features") {
                wired = true;
            }
        }
    }
    assert!(
        scanned > 5,
        "F-META-003: scanned only {scanned} workflow files -- the scan is broken, \
         not the wiring"
    );
    assert!(
        wired,
        "F-META-003: no workflow runs `--test falsification_spec_v10_tests` with a \
         `--features` flag. This suite is dark again (aprender#2522)."
    );
}

// =============================================================================
// Section 0: Ground Truth Testing (F-GT-*)
// These require model fixtures (SafeTensors BF16, 7B)
// =============================================================================

include!("includes/falsification_spec_v10_ground_truth.rs");
include!("includes/falsification_spec_v10_cli_interface.rs");
include!("includes/falsification_spec_v10_model_spec.rs");
include!("includes/falsification_spec_v10_checklist.rs");
include!("includes/f_ollama_00.rs");
include!("includes/falsification_spec_v10_definition_of_done.rs");
include!("includes/falsification_spec_v10_ml_diagnostics.rs");
include!("includes/f_trueno_00.rs");
include!("includes/f_realize_0.rs");
include!("includes/falsification_spec_v10_contract_model.rs");
include!("includes/falsification_spec_v10_qwen2_7b_params.rs");
