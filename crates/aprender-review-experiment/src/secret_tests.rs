//! Falsifiers for `trace-admission-secret-v1` (PRA-001 T6, G-SEC / G-SEC-FN).
//!
//! Every canary is BUILT at run time, from pieces or from a seeded generator,
//! so no key-shaped literal sits in the source for a repo scanner to flag.

// `json!` expands to an `unwrap` of an infallible `to_value`.
#![allow(clippy::disallowed_methods)]

use super::*;
use sha2::{Digest, Sha256};

/// A splitmix64 stream: the same seed yields the same canaries on every run.
struct Gen(u64);

impl Gen {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `n` characters drawn from `alphabet`.
    fn take(&mut self, alphabet: &str, n: usize) -> String {
        let a = alphabet.as_bytes();
        (0..n)
            .map(|_| char::from(a[(self.next() % a.len() as u64) as usize]))
            .collect()
    }
}

const UPPER: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const B32: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
const ALNUM: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const LOWER_NUM: &str = "abcdefghijklmnopqrstuvwxyz0123456789";
const HEX: &str = "0123456789abcdef";
const DIGITS: &str = "0123456789";

fn b64url() -> String {
    [ALNUM, "-_"].concat()
}

fn b64() -> String {
    [ALNUM, "+/"].concat()
}

fn hit(scanner: Scanner, rule: &'static str) -> Hit {
    Hit { scanner, rule }
}

/// The AWS documentation key: gitleaks allowlists it (`.+EXAMPLE$`), which is
/// exactly why the builtin scanner must catch it on its own.
fn planted_aws_key() -> String {
    ["AKIA", "IOSFODNN7", "EXAMPLE"].concat()
}

/// The 20 canaries, each labelled with the ONE rule it is planted for.
fn canaries() -> Vec<(Hit, String)> {
    use Scanner::{Builtin as B, Gitleaks as G};
    let mut g = Gen(0x5EC0_0001);
    let pem = |g: &mut Gen, kind: &str| {
        let body = (0..4)
            .map(|_| g.take(&b64(), 64))
            .collect::<Vec<_>>()
            .join("\n");
        format!("-----BEGIN {kind}PRIVATE KEY-----\n{body}\n-----END {kind}PRIVATE KEY-----\n")
    };
    let jwt = |g: &mut Gen| {
        format!(
            "Authorization: Bearer eyJ{}.eyJ{}.{}\n",
            g.take(ALNUM, 30),
            g.take(ALNUM, 60),
            g.take(&b64url(), 43)
        )
    };
    vec![
        // builtin
        (
            hit(B, "aws-access-key-id"),
            format!("export AWS_ACCESS_KEY_ID={}", planted_aws_key()),
        ),
        (
            hit(B, "aws-secret-access-key"),
            format!(
                "aws configure set aws_secret_access_key \"{}\"",
                g.take(&b64(), 40)
            ),
        ),
        (
            hit(B, "github-token"),
            ["gho_", &g.take(ALNUM, 36)].concat(),
        ),
        (hit(B, "private-key"), pem(&mut g, "OPENSSH ")),
        (
            hit(B, "anthropic-key"),
            ["sk-ant-", "admin01-", &g.take(ALNUM, 40)].concat(),
        ),
        (
            hit(B, "slack-token"),
            ["xoxp-", &g.take(DIGITS, 11), "-", &g.take(ALNUM, 24)].concat(),
        ),
        (hit(B, "jwt"), jwt(&mut g)),
        (
            hit(B, "high-entropy-assignment"),
            format!("client_secret: \"{}\"", g.take(ALNUM, 32)),
        ),
        // gitleaks
        (
            hit(G, "aws-access-token"),
            format!("key {}{} ", "ASIA", g.take(B32, 16)),
        ),
        (hit(G, "github-pat"), ["ghp_", &g.take(ALNUM, 36)].concat()),
        (
            hit(G, "github-fine-grained-pat"),
            ["github_pat_", &g.take(ALNUM, 82)].concat(),
        ),
        (hit(G, "private-key"), pem(&mut g, "")),
        (hit(G, "jwt"), jwt(&mut g)),
        (
            hit(G, "generic-api-key"),
            format!("api_token = \"{}{}\"", g.take(ALNUM, 30), g.take(DIGITS, 2)),
        ),
        (
            hit(G, "slack-bot-token"),
            [
                "xoxb-",
                &g.take(DIGITS, 12),
                "-",
                &g.take(DIGITS, 12),
                "-",
                &g.take(ALNUM, 24),
            ]
            .concat(),
        ),
        (
            hit(G, "stripe-access-token"),
            format!("{}{} ", "sk_live_", g.take(ALNUM, 24)),
        ),
        (
            hit(G, "gitlab-pat"),
            ["glpat-", &g.take(ALNUM, 20)].concat(),
        ),
        (
            hit(G, "npm-access-token"),
            format!("{}{} ", "npm_", g.take(LOWER_NUM, 36)),
        ),
        (
            hit(G, "gcp-api-key"),
            format!("{}{} ", "AIza", g.take(&b64url(), 35)),
        ),
        (
            hit(G, "digitalocean-pat"),
            format!("{}{} ", "dop_v1_", g.take(HEX, 64)),
        ),
    ]
}

/// The canary set covers what G-SEC names: AWS, GitHub PAT, private key, JWT,
/// high-entropy generic; 20 of them, each labelled with a distinct rule.
#[test]
fn canary_set_is_twenty_distinct_labelled_rules() {
    let c = canaries();
    assert_eq!(c.len(), 20);
    let mut labels: Vec<Hit> = c.iter().map(|(h, _)| *h).collect();
    labels.sort();
    labels.dedup();
    assert_eq!(labels.len(), 20, "each canary names a distinct rule");
    for (h, _) in &c {
        match h.scanner {
            Scanner::Builtin => assert!(BUILTIN.iter().any(|(r, _)| r == &h.rule)),
            Scanner::Gitleaks => assert!(gitleaks().rules.iter().any(|r| r.id == h.rule)),
        }
    }
    assert!(c[0].1.contains(&planted_aws_key()));
}

/// FALSIFY-TAS-001 (G-SEC): 100% canary recall. Each canary fires the rule it is
/// labelled with, bare and inside a captured JSONL row.
#[test]
fn falsify_tas_001_every_canary_fires_its_labelled_rule() {
    for (label, text) in canaries() {
        let row = serde_json::json!({"lane": "haiku-4-5", "output": text}).to_string();
        for t in [&text, &row] {
            assert!(scan(t).contains(&label), "{label:?} missed: {t}");
        }
    }
}

/// FALSIFY-TAS-002 (G-SEC-FN): remove one rule and the canary planted for it
/// goes RED: its scanner then reports NOTHING on it, so no other rule of that
/// scanner is quietly carrying the recall check above.
#[test]
fn falsify_tas_002_removing_a_rule_turns_its_canary_red() {
    for (label, text) in canaries() {
        let rest: Vec<Hit> = scan_without(&text, &[label])
            .into_iter()
            .filter(|h| h.scanner == label.scanner)
            .collect();
        assert_eq!(
            rest,
            Vec::<Hit>::new(),
            "{label:?} with its rule removed: {text}"
        );
    }
}

/// FALSIFY-TAS-003: the AWS documentation key is allowlisted by gitleaks and
/// caught by the builtin scanner; the union quarantines it. Removing the
/// builtin rule leaves NOTHING to catch it: the union is load-bearing.
#[test]
fn falsify_tas_003_aws_doc_key_needs_the_builtin_scanner() {
    let text = format!("export AWS_ACCESS_KEY_ID={}", planted_aws_key());
    let aws = hit(Scanner::Builtin, "aws-access-key-id");
    assert_eq!(scan(&text), vec![aws]);
    assert!(scan_without(&text, &[aws]).is_empty());
}

/// The vendored ruleset is the released v8.30.1 file, byte for byte.
#[test]
fn vendored_gitleaks_toml_is_sha_pinned() {
    let got = format!("{:x}", Sha256::digest(GITLEAKS_TOML.as_bytes()));
    assert_eq!(got, GITLEAKS_SHA256);
}

/// Every gitleaks rule with a regex compiles, and the only rules without one
/// are the listed path-only rules: no rule is silently dropped.
#[test]
fn every_gitleaks_rule_compiles() {
    let gl = gitleaks();
    assert!(gl.failed.is_empty(), "failed to compile: {:?}", gl.failed);
    let declared = GITLEAKS_TOML
        .lines()
        .filter(|l| l.trim() == "[[rules]]")
        .count();
    assert_eq!(declared, 222);
    assert_eq!(gl.rules.len() + PATH_ONLY_RULES.len(), declared);
    for id in PATH_ONLY_RULES {
        assert!(gl.rules.iter().all(|r| r.id != *id));
        assert!(GITLEAKS_TOML.contains(&format!("id = \"{id}\"")));
    }
}

/// Go's literal `{` becomes an escaped one; real repetitions are untouched.
#[test]
fn go_regex_translation_case_table() {
    for (go, rust) in [
        (r"^\$(?:\d+|{\d+})$", r"^\$(?:\d+|\{\d+})$"),
        (r"a{2,5}", r"a{2,5}"),
        (r"a{2}", r"a{2}"),
        (r"[{]x", r"[{]x"),
        (r"{{x}}", r"\{\{x}}"),
        (r"a{,5}", r"a\{,5}"),
        (r"\{\d{0,2}}", r"\{\d{0,2}}"),
    ] {
        assert_eq!(go_to_rust(go), rust, "{go}");
        assert!(Regex::new(&go_to_rust(go)).is_ok(), "{go}");
    }
}

/// Builtin case table: must-match rows fire exactly their rule; must-not-match
/// rows fire nothing in the builtin scanner.
#[test]
fn builtin_case_table() {
    let builtin = |t: &str| -> Vec<&'static str> {
        scan(t)
            .into_iter()
            .filter(|h| h.scanner == Scanner::Builtin)
            .map(|h| h.rule)
            .collect()
    };
    let hits = [
        (planted_aws_key(), "aws-access-key-id"),
        (["ASIA", "Y34FZKBOKMUTVV7A"].concat(), "aws-access-key-id"),
        (["ghp_", &"a".repeat(36)].concat(), "github-token"),
        (["github_pat_", &"B".repeat(40)].concat(), "github-token"),
        (
            ["-----BEGIN RSA ", "PRIVATE KEY-----"].concat(),
            "private-key",
        ),
        (["-----BEGIN ", "PRIVATE KEY-----"].concat(), "private-key"),
        (
            ["sk-ant-", "api03-", &"x".repeat(40)].concat(),
            "anthropic-key",
        ),
        (
            ["xoxb-", "123456789012-", &"q".repeat(24)].concat(),
            "slack-token",
        ),
    ];
    let escaped = format!("{{\"output\": \"\\u0041{}\"}}", &planted_aws_key()[1..]);
    assert!(
        !escaped.contains("AKIA"),
        "the fixture must hide the literal"
    );
    assert_eq!(builtin(&escaped), vec!["aws-access-key-id"]);
    for (text, kind) in &hits {
        assert_eq!(builtin(text), vec![*kind], "must match: {text}");
    }
    let misses = [
        "AKIA".to_string(),
        ["AKIA", "iosfodnn7example"].concat(),
        ["XAKIA", "IOSFODNN7EXAMPLE"].concat(),
        ["AKIA", "IOSFODNN7EXAMPLEX"].concat(),
        "ghp_short".to_string(),
        "-----BEGIN PUBLIC KEY-----".to_string(),
        "the aws_secret_access_key field must be set".to_string(),
        "sk-ant is the Anthropic key prefix".to_string(),
        "eyJhbGciOiJIUzI1NiJ9 alone is a header".to_string(),
        format!("token = {}", "0123456789abcdef".repeat(4)),
    ];
    for text in &misses {
        assert!(builtin(text).is_empty(), "must not match: {text}");
    }
}

/// Ordinary review traffic fires neither scanner: quarantine is for secrets,
/// not for every row that mentions a token.
#[test]
fn ordinary_review_rows_are_clean() {
    let upper = Gen(7).take(UPPER, 8);
    for text in [
        "LGTM: the bound check is inclusive now.".to_string(),
        "let token = lexer.next_token()?;".to_string(),
        "+    password: String,\n-    passwd: &'static str,".to_string(),
        format!("sha256 {}", "0123456789abcdef".repeat(4)),
        "fn secret_scan(rows: &str) -> Vec<Hit> {".to_string(),
        format!("see crates/aprender-serve/src/api/{upper}.rs"),
        "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1,3 +1,3 @@\n-x\n+y".to_string(),
    ] {
        assert!(scan(&text).is_empty(), "{:?} fired on: {text}", scan(&text));
    }
}

/// gitleaks semantics the canaries alone do not pin: the entropy floor drops a
/// low-entropy match, and an allowlist conditioned (AND) on a path we do not
/// have allows nothing, so the scan fails safe.
#[test]
fn gitleaks_semantics_case_table() {
    let gl_rules = |t: &str| -> Vec<&'static str> {
        scan(t)
            .into_iter()
            .filter(|h| h.scanner == Scanner::Gitleaks)
            .map(|h| h.rule)
            .collect()
    };
    // `ghp_aaaa…` matches github-pat's regex but not its entropy floor of 3.
    let flat = ["ghp_", &"a".repeat(36)].concat();
    assert!(!gl_rules(&flat).contains(&"github-pat"), "{flat}");
    // generic-api-key's `SRC…="…"` allowlist applies only to BitBake paths
    // (condition AND); with no path it must not suppress the hit.
    let mut g = Gen(0x5EC0_0002);
    let bitbake = format!(
        "SRC_API_TOKEN = \"{}{}\"",
        g.take(ALNUM, 30),
        g.take(DIGITS, 2)
    );
    assert!(gl_rules(&bitbake).contains(&"generic-api-key"), "{bitbake}");
}
