//! SUMMARY.md updater for mdBook contract pages.
//!
//! Manages the auto-generated contract section in SUMMARY.md using
//! HTML comment markers to delimit the managed region.

/// Update SUMMARY.md content by inserting/replacing the contract section.
///
/// Uses `<!-- CONTRACTS:BEGIN -->` and `<!-- CONTRACTS:END -->` markers.
/// Contract entries are sorted alphabetically. Content outside the markers
/// is preserved unchanged.
///
/// `contract_stems` should be the sorted list of contract filenames without `.yaml`.
pub fn update_summary(existing: &str, contract_stems: &[&str]) -> String {
    let begin_marker = "<!-- CONTRACTS:BEGIN -->";
    let end_marker = "<!-- CONTRACTS:END -->";

    let mut sorted_stems: Vec<&str> = contract_stems.to_vec();
    sorted_stems.sort_unstable();

    let mut contract_section = String::new();
    contract_section.push_str(begin_marker);
    contract_section.push('\n');
    for stem in &sorted_stems {
        contract_section.push_str(&format!("- [{stem}](contracts/{stem}.md)\n"));
    }
    contract_section.push_str(end_marker);

    if let Some(begin_pos) = existing.find(begin_marker) {
        if let Some(end_pos) = existing.find(end_marker) {
            let end_of_marker = end_pos + end_marker.len();
            // Consume trailing newline if present
            let end_of_marker = if existing[end_of_marker..].starts_with('\n') {
                end_of_marker + 1
            } else {
                end_of_marker
            };
            let mut result = String::new();
            result.push_str(&existing[..begin_pos]);
            result.push_str(&contract_section);
            result.push('\n');
            result.push_str(&existing[end_of_marker..]);
            return result;
        }
    }

    // Markers not found — append before final newlines
    let trimmed = existing.trim_end();
    let mut result = String::new();
    result.push_str(trimmed);
    result.push_str("\n\n");
    result.push_str(&contract_section);
    result.push('\n');
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_when_no_markers() {
        let existing = "# Summary\n\n- [Introduction](./introduction.md)\n";
        let result = update_summary(existing, &["softmax-kernel-v1", "rmsnorm-kernel-v1"]);

        assert!(result.contains("<!-- CONTRACTS:BEGIN -->"));
        assert!(result.contains("<!-- CONTRACTS:END -->"));
        assert!(result.contains("- [rmsnorm-kernel-v1](contracts/rmsnorm-kernel-v1.md)"));
        assert!(result.contains("- [softmax-kernel-v1](contracts/softmax-kernel-v1.md)"));
        // Original content preserved
        assert!(result.contains("- [Introduction](./introduction.md)"));
    }

    #[test]
    fn replaces_existing_markers() {
        let existing = "# Summary\n\n- [Intro](./intro.md)\n\n\
            <!-- CONTRACTS:BEGIN -->\n\
            - [old](contracts/old.md)\n\
            <!-- CONTRACTS:END -->\n\n\
            - [References](./references.md)\n";

        let result = update_summary(existing, &["softmax-kernel-v1"]);

        assert!(result.contains("- [softmax-kernel-v1](contracts/softmax-kernel-v1.md)"));
        assert!(!result.contains("- [old](contracts/old.md)"));
        // Content outside markers preserved
        assert!(result.contains("- [Intro](./intro.md)"));
        assert!(result.contains("- [References](./references.md)"));
    }

    #[test]
    fn sorts_stems_alphabetically() {
        let result = update_summary("", &["z-kernel-v1", "a-kernel-v1", "m-kernel-v1"]);

        let a_pos = result.find("a-kernel-v1").unwrap();
        let m_pos = result.find("m-kernel-v1").unwrap();
        let z_pos = result.find("z-kernel-v1").unwrap();
        assert!(a_pos < m_pos);
        assert!(m_pos < z_pos);
    }

    #[test]
    fn empty_stems_produces_empty_section() {
        let result = update_summary("# Summary\n", &[]);

        assert!(result.contains("<!-- CONTRACTS:BEGIN -->"));
        assert!(result.contains("<!-- CONTRACTS:END -->"));
        // No contract links
        assert!(!result.contains("](contracts/"));
    }
}
