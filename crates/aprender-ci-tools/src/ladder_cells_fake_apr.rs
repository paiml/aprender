//! The fake `apr` the ladder-cells e2e tests drive: a Rust port of
//! `scripts/lib/model_ladder_cells_produce_cases/fake_apr.py` (which stays for the parity
//! harness). The CI test image has no python3, so the wrapper `apr` the fixture writes
//! re-executes THIS test binary at `fake_apr_entry`. The verbs run / chat / code / inspect
//! answer through `verb`; `serve` binds the port and answers `GET /health` and chat
//! completions as the Python fake did. One defect is switched on by FAKE_APR_MODE:
//! good | fellback | noneedle | refuse | noclose | undercount | codenull.

use std::io::{Read, Write};
use std::path::PathBuf;

use serde_json::{json, Value};

const NEEDLE: &str = "TANGERINE-4417";
const TMPL: &str = "{% if enable_thinking %}<think>{% endif %}{% if add_generation_prompt %}<|im_start|>assistant{% endif %}";
const REFUSED_BODY: &str =
    "{\"error\": \"GPU capacity refused: = 29612 MiB, against 24000 MiB free of 24564 MiB\"}";
const REFUSAL: &str = "error: GPU capacity refused: weights 20000 MiB + KV 9000 MiB (8192 positions x 4 B at F32) \
                       + workspace 100 MiB + overhead 512 MiB = 29612 MiB, against 24000 MiB free of 24564 MiB \
                       (discrete GPU: cuMemGetInfo). Pass --no-gpu to run the CPU forward instead\n";

/// One call's `(stdout, stderr, rc)`.
type Called = (String, String, i32);

/// The libtest name of `fake_apr_entry`, for the wrapper's `--exact`.
pub(super) fn entry_name() -> String {
    let path = module_path!();
    let local = path.split_once("::").map_or(path, |(_, rest)| rest);
    format!("{local}::fake_apr_entry")
}

/// `(prompt tokens, text)`, or the pre-load capacity refusal's stderr text.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn answer(mode: &str, prompt: &str, thinking: &str) -> Result<(usize, String), String> {
    let cpt = if mode == "undercount" { 6.0 } else { 3.0 };
    let tok = (prompt.chars().count() as f64 / cpt) as usize;
    if mode == "refuse" && tok >= 8192 {
        return Err(REFUSAL.to_owned());
    }
    let on = thinking == "on";
    let noclose = mode == "noclose";
    let think = match (on, noclose) {
        (false, _) => "",
        (true, true) => "<think>checking the first line",
        (true, false) => "<think>checking the first line</think>",
    };
    let body = match (mode == "noneedle", noclose && on) {
        (true, _) => "Paris.",
        (false, true) => "",
        (false, false) => NEEDLE,
    };
    Ok((tok, format!("{think}{body}")))
}

fn opt<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let at = args.iter().position(|a| a == name)?;
    args.get(at + 1).map(String::as_str)
}

fn backend(mode: &str) -> Value {
    let gpu = mode != "fellback";
    let ran = if gpu { "gpu" } else { "cpu" };
    json!({"requested": "gpu", "ran": ran, "fell_back": !gpu})
}

fn ok(stdout: String) -> Called {
    (stdout, String::new(), 0)
}

fn refused(why: String) -> Called {
    (String::new(), why, 1)
}

fn inspect() -> Called {
    let md = json!({"general.architecture": "qwen2", "qwen2.context_length": "32768",
        "qwen2.block_count": "28", "qwen2.attention.head_count": "12",
        "qwen2.attention.head_count_kv": "2", "qwen2.embedding_length": "1536",
        "tokenizer.chat_template": TMPL, "general.file_type": "15"});
    ok(format!(
        "{}\n",
        json!({"architecture": "qwen2", "metadata": md})
    ))
}

fn run(mode: &str, args: &[String], thinking: &str) -> Called {
    let prompt = std::fs::read_to_string(opt(args, "--input").unwrap_or("")).unwrap_or_default();
    match answer(mode, &prompt, thinking) {
        Err(why) => refused(why),
        Ok((tok, text)) => ok(format!(
            "{}\n",
            json!({"text": text, "prompt_tokens": tok, "used_gpu": mode != "fellback",
                "backend": backend(mode)})
        )),
    }
}

fn chat(mode: &str, stdin: &str, thinking: &str) -> Called {
    let first = stdin.split('\n').next().unwrap_or("");
    match answer(mode, first, thinking) {
        Err(why) => refused(why),
        Ok((_, text)) => ok(format!("{text}\n{}\n", json!({"backend": backend(mode)}))),
    }
}

fn code(mode: &str, stdin: &str, thinking: &str) -> Called {
    match answer(mode, stdin, thinking) {
        Err(why) => refused(why),
        Ok((_, text)) => {
            let seen = if mode == "codenull" {
                Value::Null
            } else {
                backend(mode)
            };
            ok(format!(
                "{}\n",
                json!({"type": "result", "subtype": "success", "result": text,
                    "session_id": "x", "duration_ms": 1, "backend": seen})
            ))
        }
    }
}

/// The `inspect` / `run` / `chat` / `code` verbs; `serve` is `serve`.
fn verb(mode: &str, args: &[String], stdin: &str) -> Called {
    let thinking = opt(args, "--thinking").unwrap_or("off");
    match args.first().map(String::as_str) {
        Some("inspect") => inspect(),
        Some("run") => run(mode, args, thinking),
        Some("chat") => chat(mode, stdin, thinking),
        Some("code") => code(mode, stdin, thinking),
        other => (
            String::new(),
            format!("fake apr: unknown verb {}\n", other.unwrap_or("")),
            2,
        ),
    }
}

/// The reply to one request on the fake serve: `(status, JSON body)`.
fn reply(mode: &str, method: &str, body: &[u8]) -> (u16, String) {
    if method == "GET" {
        return (200, json!({"status": "ok"}).to_string());
    }
    let req: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
    let on = req["chat_template_kwargs"]["enable_thinking"]
        .as_bool()
        .unwrap_or(false);
    let content = req["messages"][0]["content"].as_str().unwrap_or("");
    match answer(mode, content, if on { "on" } else { "off" }) {
        // json.dumps spelled this body with ": ", and the producer keeps it verbatim as the reason.
        Err(_) => (503, REFUSED_BODY.to_owned()),
        Ok((tok, text)) => (
            200,
            json!({"choices": [{"message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": tok}, "used_gpu": mode != "fellback"})
            .to_string(),
        ),
    }
}

/// The request line's method and the body, read in full off the connection.
fn read_request(conn: &mut std::net::TcpStream) -> (String, Vec<u8>) {
    let mut raw = Vec::new();
    let mut buf = [0u8; 4096];
    let mut head_end = None;
    while head_end.is_none() {
        match conn.read(&mut buf) {
            Ok(n) if n > 0 => raw.extend_from_slice(&buf[..n]),
            _ => break,
        }
        head_end = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|at| at + 4);
    }
    let head_end = head_end.unwrap_or(raw.len());
    let head = String::from_utf8_lossy(&raw[..head_end]).into_owned();
    let method = head.split_whitespace().next().unwrap_or("GET").to_owned();
    let want = head
        .lines()
        .filter_map(|l| l.split_once(':'))
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    while raw.len() < head_end + want {
        match conn.read(&mut buf) {
            Ok(n) if n > 0 => raw.extend_from_slice(&buf[..n]),
            _ => break,
        }
    }
    (method, raw.split_off(head_end))
}

/// Serve on `port` until killed, one connection at a time and closing after each reply
/// (HTTP/1.0), as `http.server.HTTPServer` did.
fn serve(mode: &str, port: u16) {
    let listener = std::net::TcpListener::bind(("127.0.0.1", port)).expect("bind the fake serve");
    for mut conn in listener.incoming().flatten() {
        let (method, body) = read_request(&mut conn);
        let (status, text) = reply(mode, &method, &body);
        let head = format!(
            "HTTP/1.0 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            text.len()
        );
        let _ = conn.write_all(format!("{head}{text}").as_bytes());
    }
}

/// `<out><ext>`: the sibling file the wrapper reads a call's stderr and rc from.
fn beside(out: &std::ffi::OsStr, ext: &str) -> PathBuf {
    let mut p = out.to_owned();
    p.push(ext);
    PathBuf::from(p)
}

/// The entry the wrapper `apr` re-executes this binary at (`--exact`): FAKE_APR_ACT names the
/// file the call's stdout goes to (`.err` and `.rc` beside it); `serve` answers on the network
/// and never returns. Run as an ordinary test FAKE_APR_ACT is unset, and it checks the fake's
/// own answers instead.
#[test]
fn fake_apr_entry() {
    let Some(out) = std::env::var_os("FAKE_APR_ACT") else {
        let (tok, text) = answer("good", "abcdef", "off").expect("answers");
        assert_eq!((tok, text.as_str()), (2, NEEDLE));
        let refusal = answer("refuse", &"x".repeat(3 * 8192), "off").expect_err("refuses");
        assert!(refusal.contains("GPU capacity refused"), "{refusal}");
        return;
    };
    let mode = std::env::var("FAKE_APR_MODE").unwrap_or_else(|_| "good".to_owned());
    let argv = std::env::var("FAKE_APR_ARGV").unwrap_or_default();
    let args: Vec<String> = argv.lines().map(str::to_owned).collect();
    if args.first().map(String::as_str) == Some("serve") {
        serve(
            &mode,
            opt(&args, "--port")
                .and_then(|p| p.parse().ok())
                .expect("--port"),
        );
        return;
    }
    let mut stdin = String::new();
    let _ = std::io::stdin().read_to_string(&mut stdin);
    let (stdout, stderr, rc) = verb(&mode, &args, &stdin);
    std::fs::write(PathBuf::from(&out), stdout).expect("fake stdout");
    std::fs::write(beside(&out, ".err"), stderr).expect("fake stderr");
    std::fs::write(beside(&out, ".rc"), rc.to_string()).expect("fake rc");
}
