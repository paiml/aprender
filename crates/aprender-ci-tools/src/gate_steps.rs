//! `gate-steps RESULTS...` (PMAT-5759, FLOW-009 LG-9): the step-time table of one gate run.
//! Reads the x86-main driver's `results.json` and prints one JSON line per executed step,
//! `{"step":"<section>/<step name>","seconds":N,"category":"build|test|lint|content|network"}`,
//! the row the gate ledger takes. A skipped step (0 s) is not a row.

use serde_json::Value;

/// The LG-9 category of a step, from its section and step name. Lint is the default for a
/// guard section; a step that only fetches, installs or uploads is `network`.
pub fn category(section: &str, step: &str) -> &'static str {
    let s = format!("{section} {step}").to_ascii_lowercase();
    let has = |ws: &[&str]| ws.iter().any(|w| s.contains(w));
    if has(&[
        "upload", "download", "install", "fetch", "pull", "checkout", "cache",
    ]) {
        "network"
    } else if has(&[
        "nextest",
        "workspace-test",
        "cargo test",
        "test-tier",
        "shard",
    ]) {
        "test"
    } else if has(&["build", "compile", "cargo check", "cargo build", "link"]) {
        "build"
    } else if has(&["render", "video", "audio", "encode"]) {
        "content"
    } else {
        "lint"
    }
}

#[derive(serde::Serialize)]
struct Row {
    step: String,
    seconds: f64,
    category: &'static str,
}

/// One JSON line per executed step of every section in `results`.
pub fn rows(results: &str) -> Result<String, String> {
    let v: Value = serde_json::from_str(results).map_err(|e| format!("results.json: {e}"))?;
    let obj = v.as_object().ok_or("results.json: not an object")?;
    let mut out = String::new();
    for (section, r) in obj {
        for st in r
            .get("steps")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let secs = st.get("seconds").and_then(Value::as_f64).unwrap_or(0.0);
            if st.get("outcome").and_then(Value::as_str) == Some("skipped") || secs <= 0.0 {
                continue;
            }
            let name = st.get("name").and_then(Value::as_str).unwrap_or("");
            let row = Row {
                step: format!("{section}/{name}"),
                seconds: secs,
                category: category(section, name),
            };
            out.push_str(&serde_json::to_string(&row).map_err(|e| e.to_string())?);
            out.push('\n');
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: &str = r#"{
      "guard-tree": {"result":"success","steps":[
        {"index":0,"name":"check_x.sh","outcome":"success","seconds":12.5},
        {"index":1,"name":"skipped one","outcome":"skipped","seconds":0}]},
      "workspace-test-shard[1,3]": {"result":"success","steps":[
        {"index":0,"name":"cargo nextest run","outcome":"success","seconds":900.0},
        {"index":1,"name":"upload-artifact","outcome":"success","seconds":3.0}]}
    }"#;

    #[test]
    fn one_row_per_executed_step_and_none_for_skipped() {
        let out = rows(R).expect("rows");
        assert_eq!(out.lines().count(), 3, "{out}");
        assert!(!out.contains("skipped one"));
    }

    #[test]
    fn row_carries_step_seconds_and_category() {
        let out = rows(R).expect("rows");
        let first: Value = serde_json::from_str(out.lines().next().expect("line")).expect("json");
        assert_eq!(first["step"], "guard-tree/check_x.sh");
        assert_eq!(first["seconds"], 12.5);
        assert_eq!(first["category"], "lint");
    }

    #[test]
    fn categories_follow_the_step() {
        assert_eq!(
            category("workspace-test-shard[1,3]", "cargo nextest run"),
            "test"
        );
        assert_eq!(category("guard-cargo", "upload-artifact"), "network");
        assert_eq!(category("guard-cargo", "cargo build --release"), "build");
        assert_eq!(category("guard-tree", "check_foo.sh"), "lint");
    }

    #[test]
    fn bad_input_is_refused() {
        assert!(rows("not json").is_err());
        assert!(rows("[1]").is_err());
    }
}
