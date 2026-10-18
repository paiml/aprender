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
