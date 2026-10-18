use std::path::Path;

type Doc = serde_json::Value;

fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start_matches([' ', '\t']).len()
}

/// The key a block-mapping line opens, with plain or quoted spelling normalised.
fn key_of(line: &str) -> Option<&str> {
    let (key, _) = line.trim_start().split_once(':')?;
    Some(key.trim().trim_matches(['"', '\'']))
}

/// Drop the `locked_level` key of the top-level `metadata:` block and nothing else.
/// `None` when the key is not written as a line of that block (flow style, absent).
fn drop_locked_level(content: &str) -> Option<String> {
    let lines: Vec<&str> = content.split_inclusive('\n').collect();
    let start = lines.iter().position(|l| {
        !l.starts_with([' ', '\t', '#'])
            && key_of(l) == Some("metadata")
            && l.trim_end().ends_with(':')
    })?;
    let mut key_indent = None;
    let mut hit = None;
    for (i, line) in lines.iter().enumerate().skip(start + 1) {
        let body = line.trim();
        if body.is_empty() || body.starts_with('#') {
            continue;
        }
        let ind = indent_of(line);
        if ind == 0 {
            break;
        }
        let want = *key_indent.get_or_insert(ind);
        if ind == want && key_of(line) == Some("locked_level") {
            hit = Some(i);
            break;
        }
    }
    let at = hit?;
    Some(
        lines
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != at)
            .map(|(_, l)| *l)
            .collect(),
    )
}

fn metadata_without_lock(doc: &Doc) -> Doc {
    let mut copy = doc.clone();
    if let Some(m) = copy.get_mut("metadata").and_then(Doc::as_object_mut) {
        m.remove("locked_level");
    }
    copy
}

pub fn run(contract: &Path, reason: &str) -> Result<(), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(contract)?;

    // Parse to verify it's valid
    let before: Doc = crate::yaml::from_str(&content)?;

    // Only metadata.locked_level is removed
    let locked = before
        .get("metadata")
        .and_then(Doc::as_object)
        .is_some_and(|m| m.contains_key("locked_level"));
    if !locked {
        println!("Contract has no locked_level -- nothing to unlock.");
        return Ok(());
    }

    // Edit the text (a Value round trip would re-sort keys and drop comments), then prove the
    // edit: the result must parse, have lost metadata.locked_level, and differ nowhere else.
    let new_content = drop_locked_level(&content).ok_or(
        "metadata.locked_level is not a plain line of the metadata block (flow style?); \
         edit the file by hand",
    )?;
    let after: Doc = crate::yaml::from_str(&new_content)?;
    let gone = after
        .get("metadata")
        .and_then(Doc::as_object)
        .is_some_and(|m| !m.contains_key("locked_level"));
    if !gone || after != metadata_without_lock(&before) {
        return Err("unlock would change more than metadata.locked_level; file left as is".into());
    }
    std::fs::write(contract, &new_content)?;

    println!("Unlocked: {}", contract.display());
    println!("Reason: {reason}");
    println!("Warning: The contract can now regress below its previous locked level.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::run;

    const LOCKED: &str = "# header comment\nmetadata:\n  version: 1.0.0\n  locked_level: 3\n  author: x\nequations:\n  zeta:\n    formula: a\n  alpha:\n    formula: b\n";

    fn unlock(text: &str) -> Result<String, Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("c.yaml");
        std::fs::write(&path, text)?;
        run(&path, "test")?;
        Ok(std::fs::read_to_string(&path)?)
    }

    #[test]
    fn removes_only_the_locked_level_line_and_keeps_every_other_byte() {
        let out = unlock(LOCKED).expect("unlock");
        let want = LOCKED.replace("  locked_level: 3\n", "");
        assert_eq!(out, want);
        // Key order and the comment survive: a Value round trip would have sorted zeta/alpha.
        assert!(out.starts_with("# header comment\n"));
        assert!(out.find("zeta").expect("zeta") < out.find("alpha").expect("alpha"));
    }

    #[test]
    fn a_contract_without_locked_level_is_left_untouched() {
        let text = "metadata:\n  version: 1.0.0\n";
        assert_eq!(unlock(text).expect("unlock"), text);
    }

    #[test]
    fn invalid_yaml_is_refused_and_the_file_is_not_rewritten() {
        let text = "metadata: [unclosed\n  locked_level: 3\n";
        assert!(unlock(text).is_err());
    }

    #[test]
    fn a_top_level_locked_level_key_is_not_metadata_and_is_kept() {
        let text = "locked_level: 3\nmetadata:\n  version: 1.0.0\n";
        assert_eq!(unlock(text).expect("unlock"), text);
    }

    #[test]
    fn nested_decoys_are_kept_and_only_the_metadata_key_goes() {
        let text = "metadata:\n  locked_level: 3\n  notes:\n    locked_level: 9\nequations:\n  e:\n    locked_level: 7\n";
        let out = unlock(text).expect("unlock");
        assert_eq!(out, text.replacen("  locked_level: 3\n", "", 1));
    }

    #[test]
    fn a_decoy_without_the_real_key_is_a_noop_without_unlocked_message() {
        let text = "metadata:\n  version: 1.0.0\nequations:\n  e:\n    locked_level: 7\n";
        assert_eq!(unlock(text).expect("unlock"), text);
    }

    #[test]
    fn a_quoted_key_is_removed() {
        let text = "metadata:\n  \"locked_level\": 3\n  version: 1.0.0\n";
        assert_eq!(
            unlock(text).expect("unlock"),
            "metadata:\n  version: 1.0.0\n"
        );
    }

    #[test]
    fn flow_style_metadata_is_refused_and_left_untouched() {
        let dir = tempfile::tempdir().expect("dir");
        let path = dir.path().join("c.yaml");
        let text = "metadata: {version: 1.0.0, locked_level: 3}\n";
        std::fs::write(&path, text).expect("write");
        assert!(run(&path, "t").is_err());
        assert_eq!(std::fs::read_to_string(&path).expect("read"), text);
    }
}
