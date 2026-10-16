//! `enrich` and `measure` end to end against the fake `apr` the parity harness drives, each
//! output compared byte for byte with what the original wrote for the same fixture
//! (`tests/data/ladder-cells/`, from `scripts/lib/model_ladder_cells_produce.py`).

use std::path::{Path, PathBuf};

use super::*;

#[path = "ladder_cells_fake_apr.rs"]
mod fake_apr;

const LADDER: &str = "ladder:
  hosts: [{id: t, required: true}]
  cells:
    verbs: [run, chat, serve, code]
    long_rungs_for: {families: [qwen2], representatives: {}}
";

const LADDER_DECLAIMED: &str = "ladder:
  hosts: [{id: t, required: true}, {id: u, required: true}]
  cells:
    declaimed: [{arch: qwen2, host: t, issue: 4590, until: \"0.70.1\", why: fixture}]
    verbs: [run, chat, serve, code]
    long_rungs_for: {families: [qwen2], representatives: {}}
";

const RUNGS: &str = r#"{"schema": "apr-release-context-rungs/v1",
 "consumers": [{"consumer": "c", "max_prompt_tokens": 7000}],
 "rungs": [{"id": "4k", "tokens": 4096}, {"id": "8k", "tokens": 8192}]}
"#;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn golden(name: &str) -> String {
    let p = manifest().join("tests/data/ladder-cells").join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// A fixture dir of its own, with an `apr` that runs the fake in one FAKE_APR_MODE.
struct Fx {
    dir: PathBuf,
}

impl Fx {
    fn new(name: &str, mode: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("lcp-e2e-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("fixture dir");
        let apr = dir.join("apr");
        // The fake is this test binary re-executed (`fake_apr_entry`): the CI test image has no
        // python3. A test killed mid-run (a timeout, a mutant that hangs) never drops its Serve,
        // and the serve has a process group of its own: `timeout` bounds such an orphan.
        let exe = std::env::current_exe().expect("test binary");
        let entry = fake_apr::entry_name();
        let run = format!(
            "timeout -s KILL 120 \"{}\" --exact {entry} --nocapture",
            exe.display()
        );
        let script = format!(
            "#!/bin/sh\nFAKE_APR_MODE={mode}; export FAKE_APR_MODE\n\
             FAKE_APR_ARGV=$(printf '%s\\n' \"$@\"); export FAKE_APR_ARGV\n\
             FAKE_APR_ACT=\"$0.act.$$\"; export FAKE_APR_ACT\n\
             case \"$1\" in serve) exec {run} >/dev/null 2>&1;; esac\n\
             {run} >/dev/null 2>&1\n\
             cat \"$FAKE_APR_ACT\"; cat \"$FAKE_APR_ACT.err\" >&2\n\
             rc=$(cat \"$FAKE_APR_ACT.rc\" 2>/dev/null || echo 99)\n\
             rm -f \"${{FAKE_APR_ACT:?}}\" \"${{FAKE_APR_ACT:?}}.err\" \"${{FAKE_APR_ACT:?}}.rc\"\n\
             exit \"$rc\"\n"
        );
        std::fs::write(&apr, script).expect("apr wrapper");
        let mut perm = std::fs::metadata(&apr).expect("apr meta").permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perm, 0o755);
        std::fs::set_permissions(&apr, perm).expect("chmod apr");
        let files = [
            ("ladder.yaml", LADDER.to_owned()),
            ("ladder-declaimed.yaml", LADDER_DECLAIMED.to_owned()),
            ("rungs.json", RUNGS.to_owned()),
            (
                "rungs-empty.json",
                "{\"schema\": \"apr-release-context-rungs/v1\", \"rungs\": []}\n".to_owned(),
            ),
            (
                "inventory.jsonl",
                format!(
                    "{{\"file\":\"m.gguf\",\"sha256\":\"{}\",\"bytes\":1000}}\n",
                    "0".repeat(64)
                ),
            ),
            ("inv.jsonl", golden("enrich.jsonl")),
            ("models.txt", format!("m.gguf|{}/m.gguf\n", dir.display())),
            ("m.gguf", "gguf body".to_owned()),
        ];
        for (name, body) in files {
            std::fs::write(dir.join(name), body).expect("fixture file");
        }
        Self { dir }
    }

    fn p(&self, name: &str) -> String {
        self.dir.join(name).display().to_string()
    }

    fn common(&self, inventory: &str) -> Common {
        Common {
            apr: self.p("apr"),
            inventory: self.p(inventory),
            models: self.p("models.txt"),
            out: self.p("out"),
        }
    }

    fn measure(&self, ladder: &str, rungs: &str) -> Measure {
        Measure {
            common: self.common("inv.jsonl"),
            ladder: self.p(ladder),
            rungs: self.p(rungs),
            work: self.dir.display().to_string(),
            lock: None,
            host: None,
            lock_wait: 1800,
            timeout: 60,
            serve_ceiling: 10,
            max_tokens: 64,
            max_tokens_thinking: 1024,
            chars_per_token: 4.5,
            only: None,
        }
    }

    fn out(&self) -> String {
        std::fs::read_to_string(self.dir.join("out")).expect("out written")
    }
}

impl Drop for Fx {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// `name` is the fixture dir: tests run in parallel, so no two may share one.
fn measured(
    name: &str,
    mode: &str,
    set: impl FnOnce(&mut Measure),
) -> (Result<String, (u8, String)>, Fx) {
    let fx = Fx::new(name, mode);
    let mut a = fx.measure("ladder.yaml", "rungs.json");
    set(&mut a);
    (run(&Sub::Measure(a)), fx)
}

fn same_as_original(mode: &str, pass: usize, refused: usize) {
    let (got, fx) = measured(&format!("measure-{mode}"), mode, |_| {});
    assert_eq!(
        got,
        Ok(format!(
            "cells: 16 row(s), {pass} pass, {refused} refused\n"
        )),
        "{mode}"
    );
    assert_eq!(fx.out(), golden(&format!("measure-{mode}.json")), "{mode}");
}

#[test]
fn measure_good() {
    same_as_original("good", 16, 0);
}

#[test]
fn measure_fellback() {
    same_as_original("fellback", 16, 0);
}

#[test]
fn measure_noneedle() {
    same_as_original("noneedle", 0, 0);
}

#[test]
fn measure_noclose() {
    same_as_original("noclose", 8, 0);
}

#[test]
fn measure_refuse() {
    same_as_original("refuse", 8, 8);
}

#[test]
fn measure_undercount() {
    same_as_original("undercount", 16, 0);
}

#[test]
fn measure_codenull() {
    same_as_original("codenull", 16, 0);
}

#[test]
fn measure_off_budget_leaves_room() {
    let (got, fx) = measured("measure-think", "good", |a| a.max_tokens_thinking = 30_000);
    assert_eq!(got, Ok("cells: 16 row(s), 16 pass, 0 refused\n".to_owned()));
    assert_eq!(fx.out(), golden("measure-think.json"));
}

#[test]
fn measure_only_the_inventory_model() {
    let (got, fx) = measured("measure-only-inv", "good", |a| {
        a.only = Some("inv:m.gguf".to_owned())
    });
    assert_eq!(got, Ok("cells: 16 row(s), 16 pass, 0 refused\n".to_owned()));
    assert_eq!(fx.out(), golden("measure-good.json"));
}

#[test]
fn measure_only_a_rung_id_refuses() {
    let (got, fx) = measured("measure-only-4k", "good", |a| {
        a.only = Some("4k".to_owned())
    });
    let why =
        "cells: refused -- --only '4k' names no inventory model; cells are measured per model";
    assert_eq!(got, Err((2, why.to_owned())));
    assert!(!Path::new(&fx.p("out")).exists());
}

#[test]
fn measure_refuses_rungs_the_judge_fails() {
    let fx = Fx::new("measure-empty", "good");
    let got = run(&Sub::Measure(fx.measure("ladder.yaml", "rungs-empty.json")));
    let why = "cells: refused -- FAIL  the context rungs file names no rung -- it owes nothing, \
               which is not a pass";
    assert_eq!(got, Err((1, why.to_owned())));
}

#[test]
fn measure_declaimed_on_this_host_measures_nothing() {
    let fx = Fx::new("measure-dect", "good");
    let mut a = fx.measure("ladder-declaimed.yaml", "rungs.json");
    a.host = Some("t".to_owned());
    let got = run(&Sub::Measure(a));
    assert_eq!(got, Ok("cells: 0 row(s), 0 pass, 0 refused\n".to_owned()));
    assert_eq!(fx.out(), "[]");
}

#[test]
fn measure_declaimed_on_another_host_measures_all() {
    let fx = Fx::new("measure-decu", "good");
    let mut a = fx.measure("ladder-declaimed.yaml", "rungs.json");
    a.host = Some("u".to_owned());
    let got = run(&Sub::Measure(a));
    assert_eq!(got, Ok("cells: 16 row(s), 16 pass, 0 refused\n".to_owned()));
    assert_eq!(fx.out(), golden("measure-good.json"));
}

#[test]
fn enrich_matches_the_original() {
    let fx = Fx::new("enrich", "good");
    assert_eq!(
        run(&Sub::Enrich(fx.common("inventory.jsonl"))),
        Ok(String::new())
    );
    assert_eq!(fx.out(), golden("enrich.jsonl"));
}

#[test]
fn enrich_a_missing_inventory_raises() {
    let fx = Fx::new("enrich-missing", "good");
    let got = run(&Sub::Enrich(fx.common("no-such.jsonl"))).expect_err("raises");
    assert_eq!(got.0, 1);
    assert!(got.1.starts_with("FileNotFoundError: "), "{}", got.1);
}

#[test]
fn a_serve_that_never_answers_is_rc_1() {
    let port = free_port().expect("a port");
    let argv: Vec<String> = ["sleep", "30"].iter().map(|s| (*s).to_owned()).collect();
    let s = Serve::start(&argv, port, 1).expect("starts");
    let item = loads(&golden("enrich.jsonl")).expect("item");
    let row = base_row(&item, &Py::str("serve"), "off", &Py::str("4k"), 64).expect("row");
    let got = measure_serve(&s, "p", "off", 64, row.clone(), 5).expect("a row");
    let mut want = row;
    want.set("rc", Py::Int(1));
    want.set("reason", Py::str("apr serve never answered /health"));
    assert_eq!(got, want);
}
