//! #4961: the wire formats of the one HF upload path, and its requests as a
//! local fake Hub sees them. No test here reaches the network.

use super::*;
use std::io::{BufRead, BufReader, Write as _};
use std::net::TcpListener;
use std::sync::mpsc;

const TOKEN: &str = "hf_secret_test_token";

/// A scripted reply: status, extra headers, body.
type Reply = (u16, Vec<(&'static str, String)>, String);

/// One request as the fake Hub received it.
#[derive(Debug)]
struct Seen {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}

/// A fake Hub on 127.0.0.1 that answers `replies` in order, one per
/// connection, and hands back what it received. `{base}` in a reply body or
/// header is replaced by the server's own base URL.
fn fake_hub(replies: Vec<Reply>) -> (String, mpsc::Receiver<Seen>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let base = format!("http://{}", listener.local_addr().expect("addr"));
    let (tx, rx) = mpsc::channel();
    let b = base.clone();
    std::thread::spawn(move || {
        for (status, headers, body) in replies {
            let Ok((stream, _)) = listener.accept() else {
                return;
            };
            let mut r = BufReader::new(stream.try_clone().expect("clone"));
            let mut line = String::new();
            r.read_line(&mut line).expect("request line");
            let mut parts = line.split_whitespace();
            let method = parts.next().unwrap_or_default().to_string();
            let path = parts.next().unwrap_or_default().to_string();
            let mut seen_headers = BTreeMap::new();
            loop {
                let mut h = String::new();
                r.read_line(&mut h).expect("header");
                let h = h.trim_end();
                if h.is_empty() {
                    break;
                }
                if let Some((k, v)) = h.split_once(':') {
                    seen_headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
                }
            }
            let len: usize = seen_headers
                .get("content-length")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let mut req_body = vec![0; len];
            r.read_exact(&mut req_body).expect("body");
            let body = body.replace("{base}", &b);
            let mut out = format!(
                "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n",
                body.len()
            );
            for (k, v) in &headers {
                out.push_str(&format!("{k}: {}\r\n", v.replace("{base}", &b)));
            }
            out.push_str("\r\n");
            out.push_str(&body);
            let mut w = stream;
            w.write_all(out.as_bytes()).expect("reply");
            let _ = tx.send(Seen {
                method,
                path,
                headers: seen_headers,
                body: req_body,
            });
        }
    });
    (base, rx)
}

fn ok(body: &str) -> Reply {
    (200, Vec::new(), body.to_string())
}

fn next(rx: &mpsc::Receiver<Seen>) -> Seen {
    rx.recv_timeout(std::time::Duration::from_secs(10))
        .expect("the fake Hub saw a request")
}

#[test]
fn wire_formats() {
    assert_eq!(
        redact("https://s3.x/obj?X-Amz-Signature=abc#f"),
        "https://s3.x/obj"
    );
    assert_eq!(rev_segment("rc/v0.1.0-rc.1"), "rc%2Fv0.1.0-rc.1");
    assert_eq!(rev_segment("a%2Fb"), "a%252Fb");
    assert_eq!(
        next_link(Some("<https://h/api?cursor=2>; rel=\"next\"")).as_deref(),
        Some("https://h/api?cursor=2")
    );
    assert_eq!(next_link(Some("<https://h/p1>; rel=\"prev\"")), None);
    assert_eq!(next_link(None), None);

    let tree = parse_tree(&json!([
        {"type": "file", "path": "a.apr", "size": 3, "oid": "p", "lfs": {"oid": "s", "size": 3}},
        {"type": "file", "path": "README.md", "size": 6, "oid": "g"},
        {"type": "directory", "path": "d", "oid": "t"}
    ]))
    .expect("tree");
    assert_eq!(tree.len(), 2);
    assert_eq!(tree[0].size, 3);
    assert_eq!(tree[0].git_oid, "p");
    assert_eq!(tree[0].lfs_sha256.as_deref(), Some("s"));
    assert_eq!(tree[1].lfs_sha256, None);
    assert!(parse_tree(&json!({"error": "x"})).is_err());

    let refs = parse_refs(
        &json!({"branches": [{"name": "main", "targetCommit": "c1"}],
        "tags": [{"name": "v1.0.0", "targetCommit": "c2"}], "converts": []}),
    );
    assert_eq!(refs.branches["main"], "c1");
    assert_eq!(refs.tags["v1.0.0"], "c2");
    assert_eq!(parse_refs(&json!({})), Refs::default());

    let body = commit_body(
        "s",
        &[
            CommitOp::Lfs {
                path: "a.apr".into(),
                sha256: "ab".into(),
                size: 3,
            },
            CommitOp::File {
                path: "R".into(),
                bytes: b"hi".to_vec(),
            },
            CommitOp::Delete { path: "old".into() },
        ],
    );
    assert!(
        body.ends_with('\n'),
        "every NDJSON line ends with a newline"
    );
    let lines: Vec<Value> = body
        .lines()
        .map(|l| serde_json::from_str(l).expect("ndjson line"))
        .collect();
    assert_eq!(lines.len(), 4);
    assert_eq!(lines[0]["key"], "header");
    assert_eq!(lines[0]["value"]["summary"], "s");
    assert_eq!(lines[1]["key"], "lfsFile");
    assert_eq!(lines[1]["value"]["oid"], "ab");
    assert_eq!(lines[1]["value"]["algo"], "sha256");
    assert_eq!(lines[1]["value"]["size"], 3);
    assert_eq!(lines[2]["key"], "file");
    assert_eq!(lines[2]["value"]["encoding"], "base64");
    assert_eq!(lines[2]["value"]["content"], "aGk=");
    assert_eq!(lines[3]["key"], "deletedFile");
    assert_eq!(lines[3]["value"]["path"], "old");
}

#[test]
fn the_token_is_never_printed() {
    let api = RepoApi::new("https://huggingface.co/", "org/m", TOKEN);
    let dbg = format!("{api:?}");
    assert!(!dbg.contains(TOKEN), "{dbg}");
    assert!(dbg.contains("<redacted>"), "{dbg}");
    assert!(
        dbg.contains("\"https://huggingface.co\""),
        "trailing / trimmed: {dbg}"
    );
    let src = include_str!("repo_api.rs");
    assert!(!src.contains(concat!("env", "::var")), "repo_api reads env");
}

#[test]
fn refs_tree_branch_tag_hit_the_named_revision() {
    let (base, rx) = fake_hub(vec![
        ok(r#"{"branches":[{"name":"main","targetCommit":"c1"}],"tags":[]}"#),
        (
            200,
            vec![(
                "Link",
                "<{base}/api/models/org/m/tree/page2>; rel=\"next\"".into(),
            )],
            r#"[{"type":"file","path":"a","size":1,"oid":"o1"}]"#.into(),
        ),
        ok(r#"[{"type":"file","path":"b","size":2,"oid":"o2","lfs":{"oid":"s2"}}]"#),
        ok("{}"),
        ok("{}"),
    ]);
    let api = RepoApi::new(&base, "org/m", TOKEN);

    assert_eq!(api.refs().expect("refs").branches["main"], "c1");
    let s = next(&rx);
    assert_eq!(
        (s.method.as_str(), s.path.as_str()),
        ("GET", "/api/models/org/m/refs")
    );
    assert_eq!(s.headers["authorization"], format!("Bearer {TOKEN}"));
    assert!(s.headers["user-agent"].starts_with("aprender/"));

    let files = api.tree("rc/v0.1.0-rc.1").expect("tree");
    assert_eq!(files.len(), 2, "both pages are read");
    assert_eq!(files[1].lfs_sha256.as_deref(), Some("s2"));
    assert_eq!(
        next(&rx).path,
        "/api/models/org/m/tree/rc%2Fv0.1.0-rc.1?recursive=true"
    );
    assert_eq!(next(&rx).path, "/api/models/org/m/tree/page2");

    api.create_branch("rc/v0.1.0-rc.1", "main").expect("branch");
    let s = next(&rx);
    assert_eq!(
        (s.method.as_str(), s.path.as_str()),
        ("POST", "/api/models/org/m/branch/rc%2Fv0.1.0-rc.1")
    );
    let v: Value = serde_json::from_slice(&s.body).expect("json");
    assert_eq!(v["startingPoint"], "main");

    api.create_tag("main", "v0.1.0").expect("tag");
    let s = next(&rx);
    assert_eq!(s.path, "/api/models/org/m/tag/main");
    let v: Value = serde_json::from_slice(&s.body).expect("json");
    assert_eq!(v["tag"], "v0.1.0");
}

#[test]
fn preupload_and_commit_at_a_named_revision() {
    let (base, rx) = fake_hub(vec![
        ok(
            r#"{"files":[{"path":"a.apr","uploadMode":"lfs"},{"path":"R","uploadMode":"regular"}]}"#,
        ),
        ok(r#"{"commitOid":"abc123","commitUrl":"x"}"#),
    ]);
    let api = RepoApi::new(&base, "org/m", TOKEN);
    let big = vec![7u8; 600];
    let answer = api
        .preupload(
            "rc/v1",
            &[
                PreuploadFile {
                    path: "a.apr",
                    size: 600,
                    sample: &big,
                },
                PreuploadFile {
                    path: "R",
                    size: 2,
                    sample: b"hi",
                },
            ],
        )
        .expect("preupload");
    assert_eq!(answer.len(), 2);
    assert_eq!(answer[0]["uploadMode"], "lfs");
    let s = next(&rx);
    assert_eq!(s.path, "/api/models/org/m/preupload/rc%2Fv1");
    let v: Value = serde_json::from_slice(&s.body).expect("json");
    assert_eq!(v["files"][0]["size"], 600);
    assert_eq!(
        v["files"][0]["sample"],
        base64_encode(&big[..512]),
        "the sample is capped at 512 bytes"
    );
    assert_eq!(v["files"][1]["sample"], "aGk=");
    assert!(api.preupload("main", &[]).expect("empty").is_empty());

    let oid = api
        .commit("rc/v1", "rc 1", &[CommitOp::Delete { path: "old".into() }])
        .expect("commit");
    assert_eq!(oid, "abc123");
    let s = next(&rx);
    assert_eq!(s.path, "/api/models/org/m/commit/rc%2Fv1");
    assert_eq!(s.headers["content-type"], "application/x-ndjson");
    assert_eq!(
        String::from_utf8(s.body).expect("utf8"),
        commit_body("rc 1", &[CommitOp::Delete { path: "old".into() }])
    );
}

#[test]
fn upload_lfs_puts_once_and_skips_a_held_object() {
    let dir = tempfile::tempdir().expect("tmp");
    let file = dir.path().join("w.bin");
    std::fs::write(&file, b"weights!").expect("w");
    let (base, rx) = fake_hub(vec![
        ok(r#"{"objects":[{"oid":"s1","size":8,"actions":{
            "upload":{"href":"{base}/put/s1?X-Amz-Signature=sig","header":{"x-amz-meta":"m"}},
            "verify":{"href":"{base}/verify/s1"}}}]}"#),
        ok(""),
        ok("{}"),
        ok(r#"{"objects":[{"oid":"s1","size":8}]}"#),
    ]);
    let api = RepoApi::new(&base, "org/m", TOKEN);
    assert!(api
        .upload_lfs("s1", 8, LfsBody::File(&file))
        .expect("upload"));
    let s = next(&rx);
    assert_eq!(s.path, "/org/m.git/info/lfs/objects/batch");
    assert_eq!(s.headers["content-type"], LFS_JSON);
    let v: Value = serde_json::from_slice(&s.body).expect("json");
    assert_eq!(v["transfers"], json!(["basic"]));
    assert_eq!(v["objects"][0]["oid"], "s1");
    let s = next(&rx);
    assert_eq!(
        (s.method.as_str(), s.path.as_str()),
        ("PUT", "/put/s1?X-Amz-Signature=sig")
    );
    assert_eq!(s.body, b"weights!");
    assert_eq!(s.headers["x-amz-meta"], "m", "the action headers are sent");
    assert!(
        !s.headers.contains_key("authorization"),
        "the token never goes to the presigned URL"
    );
    assert_eq!(next(&rx).path, "/verify/s1");

    assert!(!api
        .upload_lfs("s1", 8, LfsBody::Bytes(b"weights!"))
        .expect("held"));
    assert_eq!(next(&rx).path, "/org/m.git/info/lfs/objects/batch");
}

#[test]
fn errors_name_the_path_never_the_token_or_the_signature() {
    let (base, _rx) = fake_hub(vec![
        (403, Vec::new(), "forbidden".into()),
        ok(
            r#"{"objects":[{"oid":"s","size":1,"actions":{"upload":{"href":"{base}/put?X-Amz-Signature=sig"}}}]}"#,
        ),
        (500, Vec::new(), "boom".into()),
        ok(
            r#"{"objects":[{"oid":"s","size":1,"actions":{"upload":{"href":"{base}/p","header":{"chunk_size":"5"}}}}]}"#,
        ),
        ok(r#"{"objects":[{"oid":"s","size":1,"error":{"code":422,"message":"bad oid"}}]}"#),
        ok(r#"{"nope":1}"#),
    ]);
    let api = RepoApi::new(&base, "org/m", TOKEN);
    let bytes = LfsBody::Bytes(b"x");
    let errors: Vec<String> = vec![
        api.refs().expect_err("403").to_string(),
        api.upload_lfs("s", 1, bytes)
            .expect_err("put 500")
            .to_string(),
        api.upload_lfs("s", 1, bytes)
            .expect_err("multipart")
            .to_string(),
        api.upload_lfs("s", 1, bytes)
            .expect_err("object error")
            .to_string(),
        api.commit("main", "s", &[])
            .expect_err("no oid")
            .to_string(),
        RepoApi::new("http://127.0.0.1:9", "org/m", TOKEN)
            .tree("main")
            .expect_err("refused")
            .to_string(),
    ];
    let has = |i: usize, parts: &[&str]| {
        for p in parts {
            assert!(
                errors[i].contains(p),
                "error {i} lacks {p:?}: {}",
                errors[i]
            );
        }
    };
    has(
        0,
        &["GET", "/api/models/org/m/refs", "HTTP 403", "forbidden"],
    );
    has(1, &["PUT", "/put", "HTTP 500"]);
    has(2, &["multipart"]);
    has(3, &["bad oid"]);
    has(4, &["commitOid"]);
    has(5, &["GET", "/api/models/org/m/tree/main"]);
    for e in &errors {
        assert!(!e.contains(TOKEN), "the token is printed: {e}");
        assert!(
            !e.contains("Signature"),
            "the presigned query is printed: {e}"
        );
    }
}
