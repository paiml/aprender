//! What is published: the latest release, the green nightly, and ancestry.
//!
//! Parsing is pure (`release_candidate`, `nightly_candidate`, `ancestry_of`);
//! only [`Net`] touches the network, so tests hand in a fake.

use crate::decide::{Ancestry, Candidate, Source};
use crate::Product;
use serde_json::Value;

/// One HTTP GET. `Ok(None)` is a 404; every other failure is `Err`.
pub trait Net {
    fn get(&self, url: &str) -> Result<Option<Vec<u8>>, String>;
}

/// The real network: ureq with a hard timeout.
pub struct Http {
    agent: ureq::Agent,
}

impl Http {
    #[must_use]
    pub fn new(timeout: std::time::Duration, user_agent: &str) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout(timeout)
            .user_agent(user_agent)
            .build();
        Self { agent }
    }
}

impl Net for Http {
    fn get(&self, url: &str) -> Result<Option<Vec<u8>>, String> {
        match self.agent.get(url).call() {
            Ok(r) => {
                let mut buf = Vec::new();
                std::io::Read::read_to_end(&mut r.into_reader(), &mut buf)
                    .map_err(|e| format!("{url}: {e}"))?;
                Ok(Some(buf))
            }
            Err(ureq::Error::Status(404, _)) => Ok(None),
            Err(e) => Err(format!("{url}: {e}")),
        }
    }
}

fn json(bytes: &[u8]) -> Option<Value> {
    serde_json::from_slice(bytes).ok()
}

/// Fill `{bin}`, `{tag}`, `{version}` and `{target}` in an asset template.
#[must_use]
pub fn asset_name(template: &str, bin: &str, tag: &str, target: &str) -> String {
    template
        .replace("{bin}", bin)
        .replace("{tag}", tag)
        .replace("{version}", tag.trim_start_matches('v'))
        .replace("{target}", target)
}

/// A release's asset for this product and target, if it has one. A release
/// without this bin's asset is no candidate: never announce a build that
/// `update` could not install.
#[must_use]
pub fn release_candidate(body: &[u8], p: &Product, target: &str) -> Option<Candidate> {
    candidate_of(&json(body)?, p, target)
}

/// The line a release manager writes into an `-rc` prerelease's notes once
/// the hand-smoke passed. Until it is there the prerelease is not installable.
pub const HAND_SMOKE_MARKER: &str = "Hand-smoke: PASS";

fn hand_smoked(v: &Value) -> bool {
    v.get("body").and_then(Value::as_str).is_some_and(|b| {
        b.lines().any(|l| {
            l.trim_start_matches(|c: char| c == '#' || c == '*' || c == '-' || c.is_whitespace())
                .starts_with(HAND_SMOKE_MARKER)
        })
    })
}

/// Whether a `/releases` entry may be installed: not a draft, a semver tag
/// (the rolling `nightly` release is a prerelease with no version — it comes
/// in through its manifest instead), and a prerelease only once hand-smoked.
/// Every intermediate release is dogfooded (operator, 2026-09-24: "we use all
/// intermediate releases for dogfood").
fn eligible(v: &Value) -> bool {
    let flag = |k: &str| v.get(k).and_then(Value::as_bool);
    let Some(tag) = v.get("tag_name").and_then(Value::as_str) else {
        return false;
    };
    if flag("draft") != Some(false) || semver::Version::parse(tag.trim_start_matches('v')).is_err()
    {
        return false;
    }
    match flag("prerelease") {
        Some(false) => true,
        Some(true) => hand_smoked(v),
        None => false,
    }
}

/// The highest-versioned eligible release in a `/releases` list that carries
/// this target's asset. Semver orders `0.69.1 < 0.69.3-rc.1 < 0.69.3`, so a
/// hand-smoked rc supersedes the last release and the final tag supersedes it.
#[must_use]
pub fn newest_release_candidate(body: &[u8], p: &Product, target: &str) -> Option<Candidate> {
    json(body)?
        .as_array()?
        .iter()
        .filter(|v| eligible(v))
        .filter_map(|v| candidate_of(v, p, target))
        .filter_map(|c| Some((semver::Version::parse(&c.version).ok()?, c)))
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, c)| c)
}

fn candidate_of(v: &Value, p: &Product, target: &str) -> Option<Candidate> {
    let tag = v.get("tag_name")?.as_str()?;
    let want = asset_name(p.release_asset?, p.bin, tag, target);
    let url = v
        .get("assets")?
        .as_array()?
        .iter()
        .find(|a| a.get("name").and_then(Value::as_str) == Some(want.as_str()))?
        .get("browser_download_url")?
        .as_str()?;
    Some(Candidate {
        source: Source::Release,
        version: tag.trim_start_matches('v').to_string(),
        git_ref: tag.to_string(),
        asset_url: url.to_string(),
        sha256: None,
        bin_sha256: None,
    })
}

/// First whitespace-separated token of a `--version` line that is a semver.
fn version_in(line: &str) -> Option<String> {
    line.split_whitespace()
        .map(|t| t.trim_matches(|c: char| c == '(' || c == ')' || c == ','))
        .find(|t| semver::Version::parse(t.trim_start_matches('v')).is_ok())
        .map(|t| t.trim_start_matches('v').to_string())
}

/// This target's tool entry in an `aprender-nightly-manifest/v1` (#4189), if
/// the target is green and carries both digests.
#[must_use]
pub fn nightly_candidate(body: &[u8], p: &Product, target: &str) -> Option<Candidate> {
    let v = json(body)?;
    if v.get("schema")?.as_str()? != "aprender-nightly-manifest/v1" {
        return None;
    }
    let t = v.get("targets")?.get(target)?;
    if t.get("status")?.as_str()? != "green" {
        return None;
    }
    let tool = t.get("tools")?.get(p.bin)?;
    let s = |k: &str| tool.get(k).and_then(Value::as_str).map(str::to_string);
    let asset = s("asset")?;
    Some(Candidate {
        source: Source::Nightly,
        version: version_in(&s("version_output")?)?,
        git_ref: t.get("green_sha")?.as_str()?.to_string(),
        asset_url: format!(
            "https://github.com/{}/releases/download/nightly/{asset}",
            p.repo
        ),
        sha256: Some(s("sha256")?),
        bin_sha256: Some(s("bin_sha256")?),
    })
}

/// The `status` field of a GitHub compare response.
#[must_use]
pub fn ancestry_of(body: &[u8]) -> Option<Ancestry> {
    match json(body)?.get("status")?.as_str()? {
        "ahead" => Some(Ancestry::Ahead),
        "behind" => Some(Ancestry::Behind),
        "identical" => Some(Ancestry::Identical),
        "diverged" => Some(Ancestry::Diverged),
        _ => None,
    }
}

pub fn latest_release(
    net: &dyn Net,
    p: &Product,
    target: &str,
) -> Result<Option<Candidate>, String> {
    if p.release_asset.is_none() {
        return Ok(None);
    }
    // Not /releases/latest: GitHub leaves every prerelease out of it, and a
    // hand-smoked -rc is installable.
    let url = format!(
        "https://api.github.com/repos/{}/releases?per_page=30",
        p.repo
    );
    Ok(net
        .get(&url)?
        .and_then(|b| newest_release_candidate(&b, p, target)))
}

pub fn nightly(net: &dyn Net, p: &Product, target: &str) -> Result<Option<Candidate>, String> {
    if !p.nightly {
        return Ok(None);
    }
    let url = format!(
        "https://github.com/{}/releases/download/nightly/nightly-manifest.json",
        p.repo
    );
    Ok(net
        .get(&url)?
        .and_then(|b| nightly_candidate(&b, p, target)))
}

/// Ancestry by the compare API; any failure is `None` (unknown), which the
/// decision treats conservatively.
pub fn ancestry(net: &dyn Net, repo: &str, base: &str, head: &str) -> Option<Ancestry> {
    let url = format!("https://api.github.com/repos/{repo}/compare/{base}...{head}");
    net.get(&url).ok().flatten().and_then(|b| ancestry_of(&b))
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to Result::unwrap
mod tests {
    use super::*;

    const P: Product = Product {
        bin: "apr",
        repo: "paiml/aprender",
        version: "0.69.0",
        build_sha: None,
        release_asset: Some("{bin}-{tag}-{target}-cpu.tar.gz"),
        nightly: true,
    };
    const T: &str = "x86_64-unknown-linux-gnu";

    fn release_json(assets: &[&str]) -> Vec<u8> {
        let a: Vec<Value> = assets
            .iter()
            .map(|n| serde_json::json!({"name": n, "browser_download_url": format!("https://dl/{n}")}))
            .collect();
        serde_json::to_vec(&serde_json::json!({"tag_name": "v0.69.1", "assets": a})).expect("json")
    }

    #[test]
    fn release_candidate_rows() {
        let c = release_candidate(
            &release_json(&["apr-v0.69.1-x86_64-unknown-linux-gnu-cpu.tar.gz"]),
            &P,
            T,
        )
        .expect("asset for this target is a candidate");
        assert_eq!(
            (c.version.as_str(), c.git_ref.as_str()),
            ("0.69.1", "v0.69.1")
        );
        assert_eq!(
            c.asset_url,
            "https://dl/apr-v0.69.1-x86_64-unknown-linux-gnu-cpu.tar.gz"
        );
        assert!(
            release_candidate(
                &release_json(&["apr-v0.69.1-aarch64-unknown-linux-gnu-cpu.tar.gz"]),
                &P,
                T
            )
            .is_none(),
            "a release without THIS target's asset is no candidate"
        );
        assert!(release_candidate(b"not json", &P, T).is_none());
        let no_release = Product {
            release_asset: None,
            ..P
        };
        assert!(release_candidate(&release_json(&["x"]), &no_release, T).is_none());
    }

    /// One `/releases` entry: (tag, prerelease, draft, body, has this target's asset).
    fn rel(tag: &str, pre: bool, draft: bool, body: &str, asset: bool) -> Value {
        let name = format!("apr-{tag}-{T}-cpu.tar.gz");
        let assets = if asset {
            vec![
                serde_json::json!({"name": name, "browser_download_url": format!("https://dl/{name}")}),
            ]
        } else {
            vec![]
        };
        serde_json::json!({"tag_name": tag, "prerelease": pre, "draft": draft, "body": body, "assets": assets})
    }

    fn newest(list: &[Value]) -> Option<String> {
        let body = serde_json::to_vec(&Value::Array(list.to_vec())).expect("json");
        newest_release_candidate(&body, &P, T).map(|c| c.git_ref)
    }

    /// Every intermediate release is dogfooded; an -rc only once hand-smoked.
    #[test]
    fn newest_release_candidate_case_table() {
        let smoked =
            "Pre-release of 0.69.3.\n\n## Hand-smoke: PASS on lambda-labs (apr serve, qwen3.5)\n";
        let bullet = "notes\n- **Hand-smoke: PASS** 2026-09-24\n";
        let unsmoked = "Pre-release of 0.69.3. Use it to dogfood today.";
        let r0691 = rel("v0.69.1", false, false, "", true);
        let nightly = rel("nightly", true, false, smoked, true);
        let rows: Vec<(&str, Vec<Value>, Option<&str>)> = vec![
            ("R1 release only", vec![r0691.clone()], Some("v0.69.1")),
            (
                "R2 rc WITHOUT hand-smoke is not installable",
                vec![
                    rel("v0.69.3-rc.1", true, false, unsmoked, true),
                    r0691.clone(),
                ],
                Some("v0.69.1"),
            ),
            (
                "R3 hand-smoked rc supersedes the release",
                vec![
                    rel("v0.69.3-rc.1", true, false, smoked, true),
                    r0691.clone(),
                ],
                Some("v0.69.3-rc.1"),
            ),
            (
                "R4 marker as a markdown bullet counts",
                vec![
                    rel("v0.69.3-rc.1", true, false, bullet, true),
                    r0691.clone(),
                ],
                Some("v0.69.3-rc.1"),
            ),
            (
                "R5 final tag supersedes its rc",
                vec![
                    rel("v0.69.3", false, false, "", true),
                    rel("v0.69.3-rc.2", true, false, smoked, true),
                    r0691.clone(),
                ],
                Some("v0.69.3"),
            ),
            (
                "R6 rc.10 > rc.2 by semver, not by string",
                vec![
                    rel("v0.69.3-rc.2", true, false, smoked, true),
                    rel("v0.69.3-rc.10", true, false, smoked, true),
                ],
                Some("v0.69.3-rc.10"),
            ),
            (
                "R7 the rolling nightly release is never a release candidate",
                vec![nightly.clone(), r0691.clone()],
                Some("v0.69.1"),
            ),
            (
                "R8 a draft is never installable",
                vec![rel("v0.69.4", false, true, smoked, true), r0691.clone()],
                Some("v0.69.1"),
            ),
            (
                "R9 smoked rc without THIS target's asset is skipped",
                vec![
                    rel("v0.69.3-rc.1", true, false, smoked, false),
                    r0691.clone(),
                ],
                Some("v0.69.1"),
            ),
            (
                "R10 list order does not matter",
                vec![
                    r0691.clone(),
                    rel("v0.69.3-rc.1", true, false, smoked, true),
                ],
                Some("v0.69.3-rc.1"),
            ),
            (
                "R11 marker mid-sentence is not a hand-smoke",
                vec![
                    rel(
                        "v0.69.3-rc.1",
                        true,
                        false,
                        "hand-smoke pending; not Hand-smoke: PASS yet",
                        true,
                    ),
                    r0691.clone(),
                ],
                Some("v0.69.1"),
            ),
            (
                "R13 a Hand-smoke: FAIL line is not installable",
                vec![
                    rel(
                        "v0.69.3-rc.1",
                        true,
                        false,
                        "Hand-smoke: FAIL apr-v0.69.3-rc.1-x86_64-unknown-linux-gnu-cuda.tar.gz used_gpu=false",
                        true,
                    ),
                    r0691.clone(),
                ],
                Some("v0.69.1"),
            ),
            (
                "R14 the line the release driver writes (aprender-36's format)",
                vec![
                    rel(
                        "v0.69.3-rc.1",
                        true,
                        false,
                        "Pre-release.\nHand-smoke: PASS apr-v0.69.3-rc.1-x86_64-unknown-linux-gnu-cuda.tar.gz sha256=ab12 version=apr 0.69.3-rc.1 commit=7ff50ec2a==tag used_gpu=true\n",
                        true,
                    ),
                    r0691.clone(),
                ],
                Some("v0.69.3-rc.1"),
            ),
            (
                "R12 nothing eligible",
                vec![nightly, rel("v0.69.3-rc.1", true, false, unsmoked, true)],
                None,
            ),
        ];
        for (name, list, want) in rows {
            assert_eq!(newest(&list).as_deref(), want, "{name}");
        }
        assert!(
            newest_release_candidate(b"{\"tag_name\":\"v1.0.0\"}", &P, T).is_none(),
            "an object is not a /releases list"
        );
    }

    fn manifest(status: &str, schema: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "schema": schema,
            "targets": {T: {"status": status, "green_sha": "abc123",
                "tools": {"apr": {"asset": "apr-x86_64-unknown-linux-gnu.tar.gz", "sha256": "aa",
                    "bin_sha256": "bb", "version_output": "apr 0.69.1 (abc1234)"}}}}
        }))
        .expect("json")
    }

    #[test]
    fn nightly_candidate_rows() {
        let c = nightly_candidate(&manifest("green", "aprender-nightly-manifest/v1"), &P, T)
            .expect("green");
        assert_eq!(
            (
                c.version.as_str(),
                c.git_ref.as_str(),
                c.sha256.as_deref(),
                c.bin_sha256.as_deref()
            ),
            ("0.69.1", "abc123", Some("aa"), Some("bb"))
        );
        assert_eq!(c.asset_url, "https://github.com/paiml/aprender/releases/download/nightly/apr-x86_64-unknown-linux-gnu.tar.gz");
        assert!(
            nightly_candidate(&manifest("red", "aprender-nightly-manifest/v1"), &P, T).is_none(),
            "red target"
        );
        assert!(
            nightly_candidate(&manifest("green", "v2"), &P, T).is_none(),
            "unknown schema"
        );
        assert!(nightly_candidate(
            &manifest("green", "aprender-nightly-manifest/v1"),
            &P,
            "riscv64"
        )
        .is_none());
        let pv = Product { bin: "pv", ..P };
        assert!(
            nightly_candidate(&manifest("green", "aprender-nightly-manifest/v1"), &pv, T).is_none(),
            "bin absent"
        );
    }

    #[test]
    fn version_and_ancestry_parsing() {
        assert_eq!(
            version_in("pv 0.69.0 (aprender provable-contracts verifier)").as_deref(),
            Some("0.69.0")
        );
        assert_eq!(
            version_in("apr v0.69.1+no-git").as_deref(),
            Some("0.69.1+no-git")
        );
        assert_eq!(version_in("nothing here"), None);
        for (s, a) in [
            ("ahead", Some(Ancestry::Ahead)),
            ("behind", Some(Ancestry::Behind)),
            ("identical", Some(Ancestry::Identical)),
            ("diverged", Some(Ancestry::Diverged)),
            ("weird", None),
        ] {
            assert_eq!(
                ancestry_of(format!("{{\"status\":\"{s}\"}}").as_bytes()),
                a,
                "{s}"
            );
        }
        assert_eq!(
            asset_name("{bin}-{tag}-{target}.tar.gz", "pv", "v1.2.3", "t"),
            "pv-v1.2.3-t.tar.gz"
        );
        assert_eq!(
            asset_name("{bin}-{version}.zip", "pv", "v1.2.3", "t"),
            "pv-1.2.3.zip"
        );
    }
}
