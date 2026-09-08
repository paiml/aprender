// PMAT-1094 / #3055 — no `*-lint --json` field may be a Rust `Debug` rendering.
//
// aprender#2377(6) already decided the shape and fixed ONE file
// (`audio_inspect_lint.rs`): "these were `format!(\"{x:?}\")`, which puts a Rust Debug
// rendering inside a JSON *string* — a consumer asking for the sample rate got the
// characters `Ok { rate: 16000 }`. The outcome enums are internally-tagged Serialize, so
// each is now an object with a `status` discriminant and its real fields."
//
// The rest of the family kept the defect, which is why the surface has THREE shapes for one
// outcome field (measured on `origin/main`, #3055):
//
//     {"status":"ok","efficiency":0.875}    ddp-metrics-lint   — the value, serialised
//     "Ok"                                  prometheus-lint    — Debug of a unit variant
//     "Ok { code: 134 }"                    nccl-diag-lint     — Debug of a struct variant
//
// The last two are the SAME defect: `format!("{o:?}")`. A unit variant renders as a bare
// word and a struct variant renders as Rust source; neither is JSON a consumer can parse.
//
// This test reads the sources rather than running the binaries, because the alternative is a
// fixture per lint (ten of them) and the rule is a property of the code, not of any input.
#[cfg(test)]
mod lint_json_outcome_shape {
    use std::path::{Path, PathBuf};

    fn commands_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/commands")
    }

    /// The text of every `serde_json::json!({ … })` block in `src`, with comments stripped —
    /// a comment quoting the defect (as `audio_inspect_lint.rs` does, deliberately) must not
    /// read as the defect.
    fn json_blocks(src: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut depth = 0usize;
        let mut cur = String::new();
        for line in src.lines() {
            let code = line.split("//").next().unwrap_or("");
            if depth == 0 {
                if code.contains("serde_json::json!") {
                    depth = 1;
                    cur.clear();
                    cur.push_str(code);
                }
                continue;
            }
            cur.push('\n');
            cur.push_str(code);
            if code.contains("});") {
                out.push(std::mem::take(&mut cur));
                depth = 0;
            }
        }
        out
    }

    #[test]
    fn no_lint_serialises_an_outcome_as_a_debug_string() {
        let dir = commands_dir();
        let mut offenders: Vec<String> = Vec::new();
        let mut scanned = 0usize;
        for entry in std::fs::read_dir(&dir).expect("src/commands is readable") {
            let path = entry.expect("dir entry").path();
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
            if !name.ends_with("_lint.rs") {
                continue;
            }
            scanned += 1;
            let src = std::fs::read_to_string(&path).expect("read lint source");
            for block in json_blocks(&src) {
                for line in block.lines() {
                    if line.contains(":?}") && line.contains("format!") {
                        offenders.push(format!("{name}: {}", line.trim()));
                    }
                }
            }
        }
        assert!(
            scanned >= 10,
            "the scan found only {scanned} *_lint.rs files — a detector that finds nothing is broken, not a pass"
        );
        assert!(
            offenders.is_empty(),
            "{} lint --json field(s) render a Rust Debug string instead of the serialised outcome \
             (aprender#2377(6), #3055). Serialise the value; give its enum \
             `#[derive(serde::Serialize)]` + `#[serde(tag = \"status\", rename_all = \"snake_case\")]`:\n  {}",
            offenders.len(),
            offenders.join("\n  ")
        );
    }
}
