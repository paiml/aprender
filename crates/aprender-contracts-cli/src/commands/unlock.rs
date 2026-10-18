use std::path::Path;

pub fn run(contract: &Path, reason: &str) -> Result<(), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(contract)?;

    // Parse to verify it's valid
    let contract_obj: serde_json::Value = crate::yaml::from_str(&content)?;

    // Only metadata.locked_level is removed
    let locked = contract_obj
        .get("metadata")
        .and_then(serde_json::Value::as_object)
        .is_some_and(|m| m.contains_key("locked_level"));
    if !locked {
        println!("Contract has no locked_level -- nothing to unlock.");
        return Ok(());
    }

    // Write back: drop the indented `locked_level:` line and keep every other byte, so key
    // order and comments survive (a Value round trip would re-sort the keys).
    let new_content: String = content
        .split_inclusive('\n')
        .filter(|l| !(l.starts_with([' ', '\t']) && l.trim_start().starts_with("locked_level:")))
        .collect();
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
}
