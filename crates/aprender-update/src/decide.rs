//! The pure decision: is there a build newer than the running one, and which?
//!
//! No I/O happens here. Ancestry questions ("is commit B ahead of A on main?")
//! are answered by the caller, so the whole rule set is a case table.

use semver::Version;

/// Where a candidate build comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Release,
    Nightly,
}

/// A build that could replace the running binary.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Candidate {
    pub source: Source,
    /// Crate version the build reports (`0.69.1`).
    pub version: String,
    /// Git ref that identifies the build for ancestry: a release tag (`v0.69.1`)
    /// or a nightly's `green_sha`.
    pub git_ref: String,
    /// Download URL of the tarball.
    pub asset_url: String,
    /// Expected sha256 of the tarball, when the source states it inline (the
    /// nightly manifest does). A release publishes it as `<asset>.sha256`.
    pub sha256: Option<String>,
    /// Expected sha256 of the extracted executable (nightly manifest only).
    pub bin_sha256: Option<String>,
}

/// The running binary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Current {
    pub version: String,
    /// Commit the binary was built from, when it knows it (#4219).
    pub build_sha: Option<String>,
}

/// How `head` relates to `base` on the repository's history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ancestry {
    /// `head` contains `base` plus more commits.
    Ahead,
    /// `head` is an ancestor of `base`.
    Behind,
    Identical,
    /// Neither contains the other (e.g. a release branch).
    Diverged,
}

/// Outcome of a check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    UpToDate,
    Available(Candidate),
}

fn parse(v: &str) -> Option<Version> {
    Version::parse(v.trim_start_matches('v')).ok()
}

/// A release is newer when its version is greater AND, if the running build
/// knows its commit, the tag does not sit at or behind that commit.
fn release_is_newer(
    cur: &Current,
    c: &Candidate,
    ancestry: &dyn Fn(&str, &str) -> Option<Ancestry>,
) -> bool {
    let (Some(have), Some(want)) = (parse(&cur.version), parse(&c.version)) else {
        return false;
    };
    if want <= have {
        return false;
    }
    match &cur.build_sha {
        None => true,
        Some(sha) => !matches!(
            ancestry(sha, &c.git_ref),
            Some(Ancestry::Behind | Ancestry::Identical)
        ),
    }
}

/// A nightly is newer only when ancestry proves it: its commit is ahead of the
/// running build's commit. Without a build SHA, only a strictly greater
/// version counts; an equal version is "unknown", and unknown never nags.
fn nightly_is_newer(
    cur: &Current,
    c: &Candidate,
    ancestry: &dyn Fn(&str, &str) -> Option<Ancestry>,
) -> bool {
    match &cur.build_sha {
        Some(sha) => ancestry(sha, &c.git_ref) == Some(Ancestry::Ahead),
        None => matches!((parse(&cur.version), parse(&c.version)), (Some(h), Some(w)) if w > h),
    }
}

/// Pick the newest build that is strictly newer than `cur`: the newer of the
/// release and the nightly, where the nightly wins only when it is provably
/// ahead of the release tag. Never a downgrade.
#[must_use]
pub fn decide(
    cur: &Current,
    release: Option<&Candidate>,
    nightly: Option<&Candidate>,
    ancestry: &dyn Fn(&str, &str) -> Option<Ancestry>,
) -> Decision {
    let rel = release.filter(|c| release_is_newer(cur, c, ancestry));
    let nig = nightly.filter(|c| nightly_is_newer(cur, c, ancestry));
    let pick = match (rel, nig) {
        (Some(r), Some(n)) => {
            if ancestry(&r.git_ref, &n.git_ref) == Some(Ancestry::Ahead) {
                n
            } else {
                r
            }
        }
        (Some(r), None) => r,
        (None, Some(n)) => n,
        (None, None) => return Decision::UpToDate,
    };
    Decision::Available(pick.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(v: &str) -> Candidate {
        Candidate {
            source: Source::Release,
            version: v.into(),
            git_ref: format!("v{v}"),
            asset_url: String::new(),
            sha256: None,
            bin_sha256: None,
        }
    }
    fn nig(v: &str, sha: &str) -> Candidate {
        Candidate {
            source: Source::Nightly,
            version: v.into(),
            git_ref: sha.into(),
            asset_url: String::new(),
            sha256: Some("00".into()),
            bin_sha256: Some("11".into()),
        }
    }
    fn cur(v: &str, sha: Option<&str>) -> Current {
        Current {
            version: v.into(),
            build_sha: sha.map(Into::into),
        }
    }
    /// History used by every row: `old` < `v0.69.1` < `mid` < `new` on main;
    /// `rb` is a release-branch commit diverged from main.
    fn hist(base: &str, head: &str) -> Option<Ancestry> {
        const ORDER: [&str; 4] = ["old", "v0.69.1", "mid", "new"];
        if base == "offline" || head == "offline" {
            return None;
        }
        if base == head {
            return Some(Ancestry::Identical);
        }
        if base == "rb" || head == "rb" {
            return Some(Ancestry::Diverged);
        }
        let (b, h) = (
            ORDER.iter().position(|x| *x == base)?,
            ORDER.iter().position(|x| *x == head)?,
        );
        Some(if h > b {
            Ancestry::Ahead
        } else {
            Ancestry::Behind
        })
    }
    fn offline(_: &str, _: &str) -> Option<Ancestry> {
        None
    }
    fn src(d: &Decision) -> Option<(Source, String)> {
        match d {
            Decision::UpToDate => None,
            Decision::Available(c) => Some((c.source, c.git_ref.clone())),
        }
    }

    /// The case table. Each row: name, current, release, nightly, ancestry, expected pick.
    #[test]
    fn decide_case_table() {
        type Row = (
            &'static str,
            Current,
            Option<Candidate>,
            Option<Candidate>,
            fn(&str, &str) -> Option<Ancestry>,
            Option<(Source, &'static str)>,
        );
        let r = Some((Source::Release, "v0.69.1"));
        let n = Some((Source::Nightly, "new"));
        let rows: Vec<Row> = vec![
            (
                "newer release, no sha",
                cur("0.69.0", None),
                Some(rel("0.69.1")),
                None,
                hist,
                r,
            ),
            (
                "equal release",
                cur("0.69.1", None),
                Some(rel("0.69.1")),
                None,
                hist,
                None,
            ),
            (
                "older release is never a downgrade",
                cur("0.70.0", None),
                Some(rel("0.69.1")),
                None,
                hist,
                None,
            ),
            (
                "greater tag BEHIND the running commit is not newer",
                cur("0.69.0", Some("mid")),
                Some(rel("0.69.1")),
                None,
                hist,
                None,
            ),
            (
                "greater tag ahead of the running commit",
                cur("0.69.0", Some("old")),
                Some(rel("0.69.1")),
                None,
                hist,
                r,
            ),
            (
                "greater tag diverged (release branch) is newer",
                cur("0.69.0", Some("rb")),
                Some(rel("0.69.1")),
                None,
                hist,
                r,
            ),
            (
                "nightly ahead of running commit",
                cur("0.69.1", Some("mid")),
                None,
                Some(nig("0.69.1", "new")),
                hist,
                n,
            ),
            (
                "nightly at the running commit",
                cur("0.69.1", Some("new")),
                None,
                Some(nig("0.69.1", "new")),
                hist,
                None,
            ),
            (
                "nightly behind the running commit",
                cur("0.69.1", Some("new")),
                None,
                Some(nig("0.69.1", "mid")),
                hist,
                None,
            ),
            (
                "no sha, same version nightly: unknown never nags",
                cur("0.69.1", None),
                None,
                Some(nig("0.69.1", "new")),
                hist,
                None,
            ),
            (
                "no sha, greater version nightly",
                cur("0.69.0", None),
                None,
                Some(nig("0.70.0", "new")),
                hist,
                n,
            ),
            (
                "both newer, nightly ahead of tag -> nightly",
                cur("0.69.0", Some("old")),
                Some(rel("0.69.1")),
                Some(nig("0.69.1", "new")),
                hist,
                n,
            ),
            (
                "both newer, nightly behind tag -> release",
                cur("0.69.0", None),
                Some(rel("0.69.1")),
                Some(nig("0.70.0", "old")),
                hist,
                r,
            ),
            (
                "offline: release by semver, nightly unprovable",
                cur("0.69.0", Some("old")),
                Some(rel("0.69.1")),
                Some(nig("0.69.1", "new")),
                offline,
                r,
            ),
            (
                "offline with sha, nightly only: no nag",
                cur("0.69.1", Some("mid")),
                None,
                Some(nig("0.69.1", "new")),
                offline,
                None,
            ),
            (
                "unparseable current version: no nag",
                cur("dev", None),
                Some(rel("0.69.1")),
                None,
                hist,
                None,
            ),
            (
                "nothing published",
                cur("0.69.0", None),
                None,
                None,
                hist,
                None,
            ),
        ];
        let mut bad = Vec::new();
        for (name, c, re, ni, anc, want) in rows {
            let got = src(&decide(&c, re.as_ref(), ni.as_ref(), &anc));
            let want = want.map(|(s, g)| (s, g.to_string()));
            if got != want {
                bad.push(format!("{name}: got {got:?}, want {want:?}"));
            }
        }
        assert!(bad.is_empty(), "decide rows failed:\n{}", bad.join("\n"));
    }
}
