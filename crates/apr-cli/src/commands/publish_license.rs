//! Where the model card's `license:` comes from (hf-rc-publish-v1 HRP-003).
//!
//! `apr publish` used to default `--license` to `mit`. Publishing a fine-tune of
//! an Apache-2.0 base with no flag therefore shipped the derivative under a
//! licence its base does not grant. The licence now comes from `--license`, or
//! else from the base model found in DIRECTORY, or the publish refuses.

use crate::error::CliError;
use std::fs;
use std::path::{Path, PathBuf};

/// The licence the card will carry, and the evidence for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedLicense {
    pub(crate) spdx: String,
    /// `--license`, or the file the base licence was read from.
    pub(crate) source: String,
}

/// Resolve the card licence. Order: `--license`, the `license` field of an
/// `.apr` artifact's metadata, the `license:` key of a README.md front matter,
/// then a LICENSE file whose text names a known licence. With none of those,
/// refuse: there is no default for a derivative.
pub(crate) fn resolve_license(
    flag: Option<&str>,
    files: &[PathBuf],
    companion_files: &[PathBuf],
) -> Result<ResolvedLicense, CliError> {
    let base = base_license(files, companion_files)?;
    if let Some(value) = flag {
        if let Some(why) = crate::commands::spdx::reject_reason("--license", value) {
            return Err(CliError::ValidationFailed(format!("apr publish: {why}")));
        }
        if let Some(b) = base
            .as_ref()
            .filter(|b| !b.spdx.eq_ignore_ascii_case(value))
        {
            eprintln!(
                "apr publish: warning: --license {value} differs from the base licence {} ({}). \
                 The card uses --license; make sure the base licence allows it.",
                b.spdx, b.source
            );
        }
        return Ok(ResolvedLicense {
            spdx: value.to_string(),
            source: "--license".to_string(),
        });
    }
    let found = base.ok_or_else(|| {
        CliError::ValidationFailed(
            "apr publish: no --license given and no base licence found in DIRECTORY \
             (.apr metadata `license`, README.md front matter `license:`, or a LICENSE file). \
             Pass --license <SPDX>; there is no default, because a derivative must not ship \
             under a licence its base does not grant (hf-rc-publish-v1 HRP-003)."
                .to_string(),
        )
    })?;
    if let Some(why) = crate::commands::spdx::reject_reason(&found.source, &found.spdx) {
        return Err(CliError::ValidationFailed(format!(
            "apr publish: base licence {why}. Pass --license <SPDX> (HRP-003)."
        )));
    }
    Ok(found)
}

fn base_license(
    files: &[PathBuf],
    companion_files: &[PathBuf],
) -> Result<Option<ResolvedLicense>, CliError> {
    for f in files.iter().filter(|f| has_ext(f, "apr")) {
        if let Some(spdx) = apr_metadata_license(f)? {
            return Ok(Some(found(spdx, f)));
        }
    }
    let named = |name: &str| {
        companion_files
            .iter()
            .find(|p| p.file_name().and_then(|n| n.to_str()) == Some(name))
    };
    if let Some(readme) = named("README.md") {
        if let Some(spdx) = fs::read_to_string(readme)
            .ok()
            .and_then(|t| front_matter_license(&t))
        {
            return Ok(Some(found(spdx, readme)));
        }
    }
    for name in ["LICENSE", "LICENSE.md", "LICENSE.txt"] {
        if let Some(p) = named(name) {
            if let Some(spdx) = fs::read_to_string(p)
                .ok()
                .and_then(|t| license_text_spdx(&t))
            {
                return Ok(Some(found(spdx.to_string(), p)));
            }
        }
    }
    Ok(None)
}

fn found(spdx: String, path: &Path) -> ResolvedLicense {
    ResolvedLicense {
        spdx,
        source: path.display().to_string(),
    }
}

fn has_ext(p: &Path, ext: &str) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

/// The `license` of an APR v2 file, read from the header and metadata block
/// only (a model can be tens of GB). Not an APR v2 file: `None`.
fn apr_metadata_license(path: &Path) -> Result<Option<String>, CliError> {
    use aprender::format::v2::{
        AprV2Header, AprV2Metadata, HEADER_SIZE_V2, MAGIC_V2, MAX_METADATA_SIZE,
    };
    use std::io::{Read, Seek, SeekFrom};

    let fail =
        |what: &str| CliError::ValidationFailed(format!("apr publish: {}: {what}", path.display()));
    let mut file = fs::File::open(path).map_err(|e| fail(&format!("cannot open: {e}")))?;
    let mut head = [0u8; HEADER_SIZE_V2];
    if file.read_exact(&mut head).is_err() || head[0..4] != MAGIC_V2 {
        return Ok(None);
    }
    let header =
        AprV2Header::from_bytes(&head).map_err(|e| fail(&format!("bad APR v2 header: {e}")))?;
    let size = header.metadata_size as usize;
    if size > MAX_METADATA_SIZE {
        return Err(fail(&format!("APR v2 metadata too large ({size} bytes)")));
    }
    let mut buf = vec![0u8; size];
    file.seek(SeekFrom::Start(header.metadata_offset))
        .and_then(|_| file.read_exact(&mut buf))
        .map_err(|e| fail(&format!("cannot read APR v2 metadata: {e}")))?;
    let meta =
        AprV2Metadata::from_json(&buf).map_err(|e| fail(&format!("bad APR v2 metadata: {e}")))?;
    Ok(meta.license.filter(|l| !l.trim().is_empty()))
}

/// `license:` from a YAML front matter block (`---` … `---`) at the top of a
/// README, as Hugging Face model cards carry it.
fn front_matter_license(text: &str) -> Option<String> {
    let mut lines = text.lines();
    if lines.next()?.trim() != "---" {
        return None;
    }
    for line in lines {
        let line = line.trim_end();
        if line.trim() == "---" {
            return None;
        }
        if let Some(v) = line.strip_prefix("license:") {
            let v = v.trim().trim_matches(|c| c == '"' || c == '\'');
            return (!v.is_empty()).then(|| v.to_string());
        }
    }
    None
}

/// The SPDX id of a LICENSE file's text, for the licences whose text
/// identifies them unambiguously. Anything else: `None`.
fn license_text_spdx(text: &str) -> Option<&'static str> {
    let head: String = text.chars().take(2000).collect();
    if head.contains("Apache License") && head.contains("Version 2.0") {
        Some("Apache-2.0")
    } else if head.contains("MIT License")
        || head.contains("Permission is hereby granted, free of charge")
    {
        Some("MIT")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir_with(files: &[(&str, &str)]) -> (tempfile::TempDir, Vec<PathBuf>) {
        let d = tempfile::tempdir().expect("tempdir");
        let paths = files
            .iter()
            .map(|(name, body)| {
                let p = d.path().join(name);
                fs::write(&p, body).expect("write");
                p
            })
            .collect();
        (d, paths)
    }

    fn apr_with_license(dir: &Path, license: Option<&str>) -> PathBuf {
        use aprender::format::v2::{AprV2Metadata, AprV2Writer};
        let mut meta = AprV2Metadata::default();
        meta.license = license.map(str::to_string);
        let mut w = AprV2Writer::new(meta);
        w.add_f32_tensor("w", vec![2], &[1.0, 2.0]);
        let p = dir.join("model.apr");
        fs::write(&p, w.write().expect("write apr")).expect("write");
        p
    }

    const APACHE: &str = "\n                                 Apache License\n                           Version 2.0, January 2004\n";

    /// FALSIFY-HRP-003: with no --license, an Apache-2.0 base gives an
    /// apache-2.0 card. The old `default_value = "mit"` turns this RED.
    #[test]
    fn falsify_hrp_003_derivative_inherits_base_license_not_mit() {
        let (_d, comp) = dir_with(&[("LICENSE", APACHE)]);
        let r = resolve_license(None, &[], &comp).expect("resolves");
        assert_eq!(r.spdx, "Apache-2.0");
        assert!(r.source.ends_with("LICENSE"), "{}", r.source);

        let (_d, comp) = dir_with(&[(
            "README.md",
            "---\nlicense: apache-2.0\ntags: [x]\n---\n# m\n",
        )]);
        let r = resolve_license(None, &[], &comp).expect("resolves");
        assert_eq!(r.spdx, "apache-2.0");

        let d = tempfile::tempdir().expect("tempdir");
        let apr = apr_with_license(d.path(), Some("Apache-2.0"));
        let r = resolve_license(None, &[apr], &[]).expect("resolves");
        assert_eq!(r.spdx, "Apache-2.0");
    }

    /// FALSIFY-HRP-003: no flag and no base licence refuses; it never
    /// falls back to mit.
    #[test]
    fn falsify_hrp_003_no_license_anywhere_is_refused() {
        let d = tempfile::tempdir().expect("tempdir");
        let apr = apr_with_license(d.path(), None);
        let (_d2, comp) = dir_with(&[
            ("README.md", "# no front matter\nlicense: mit\n"),
            ("LICENSE", "All rights reserved."),
            ("config.json", "{}"),
        ]);
        let err = resolve_license(None, &[apr], &comp).expect_err("must refuse");
        let msg = format!("{err}");
        assert!(
            msg.contains("HRP-003") && msg.contains("--license"),
            "{msg}"
        );
    }

    #[test]
    fn apr_metadata_beats_readme_and_flag_beats_both() {
        let d = tempfile::tempdir().expect("tempdir");
        let apr = apr_with_license(d.path(), Some("Apache-2.0"));
        let (_d2, comp) = dir_with(&[("README.md", "---\nlicense: mit\n---\n")]);
        let r = resolve_license(None, &[apr.clone()], &comp).expect("resolves");
        assert_eq!(r.spdx, "Apache-2.0");
        let r = resolve_license(Some("MIT"), &[apr], &comp).expect("resolves");
        assert_eq!((r.spdx.as_str(), r.source.as_str()), ("MIT", "--license"));
    }

    #[test]
    fn non_spdx_base_license_is_refused() {
        let (_d, comp) = dir_with(&[("README.md", "---\nlicense: other\n---\n")]);
        let err = resolve_license(None, &[], &comp).expect_err("must refuse");
        assert!(format!("{err}").contains("not a recognised SPDX"), "{err}");
    }

    #[test]
    fn license_text_detection() {
        assert_eq!(license_text_spdx(APACHE), Some("Apache-2.0"));
        assert_eq!(license_text_spdx("MIT License\n\nCopyright"), Some("MIT"));
        assert_eq!(license_text_spdx("Apache License\nVersion 1.1"), None);
        assert_eq!(
            front_matter_license("---\nlicense: \"apache-2.0\"\n---"),
            Some("apache-2.0".into())
        );
        assert_eq!(
            front_matter_license("---\ntags: []\n---\nlicense: mit"),
            None
        );
    }
}
