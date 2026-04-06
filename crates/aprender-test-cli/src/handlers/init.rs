//! Init command handler

use crate::config::CliConfig;
use crate::InitArgs;
use std::path::Path;

/// Execute the init command
pub fn execute_init(_config: &CliConfig, args: &InitArgs) {
    println!("Initializing Probar project in: {}", args.path.display());

    if args.force {
        println!("Force mode enabled - overwriting existing files");
    }

    // Create project directory if it doesn't exist
    if let Err(e) = std::fs::create_dir_all(&args.path) {
        eprintln!("Failed to create directory: {e}");
        return;
    }

    // Create basic test file
    let test_file = args.path.join("tests").join("basic_test.rs");
    if let Some(parent) = test_file.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let test_content = generate_probar_config();
    if !test_file.exists() || args.force {
        let _ = std::fs::write(&test_file, test_content);
        println!("Created: {}", test_file.display());
    }

    println!("Probar project initialized successfully!");
}

/// Generate the default Probar test configuration file content
#[must_use]
pub const fn generate_probar_config() -> &'static str {
    r#"//! Basic Probar test
use jugar_probar::prelude::*;

#[test]
fn test_example() {
    let result = TestResult::pass("example_test");
    assert!(result.passed);
}
"#
}

/// Check if a path is a valid init target
#[must_use]
pub fn is_valid_init_path(path: &Path) -> bool {
    // Path should either not exist or be an empty directory
    if !path.exists() {
        return true;
    }

    if !path.is_dir() {
        return false;
    }

    // Check if directory is empty (except for hidden files)
    match std::fs::read_dir(path) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if !name_str.starts_with('.') {
                    return false; // Non-hidden file found
                }
            }
            true
        }
        Err(_) => false,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_generate_probar_config() {
        let config = generate_probar_config();
        assert!(config.contains("use jugar_probar::prelude::*"));
        assert!(config.contains("fn test_example()"));
        assert!(config.contains("TestResult::pass"));
    }

    #[test]
    fn test_is_valid_init_path_nonexistent() {
        let temp = TempDir::new().unwrap();
        let nonexistent = temp.path().join("new_project");
        assert!(is_valid_init_path(&nonexistent));
    }

    #[test]
    fn test_is_valid_init_path_empty_dir() {
        let temp = TempDir::new().unwrap();
        assert!(is_valid_init_path(temp.path()));
    }

    #[test]
    fn test_is_valid_init_path_nonempty_dir() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("existing.rs"), "// content").unwrap();
        assert!(!is_valid_init_path(temp.path()));
    }

    #[test]
    fn test_is_valid_init_path_dir_with_hidden_files() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join(".gitignore"), "target/").unwrap();
        assert!(is_valid_init_path(temp.path()));
    }

    #[test]
    fn test_is_valid_init_path_file() {
        let temp = TempDir::new().unwrap();
        let file_path = temp.path().join("file.txt");
        std::fs::write(&file_path, "content").unwrap();
        assert!(!is_valid_init_path(&file_path));
    }

    #[test]
    fn test_execute_init_creates_directory() {
        let temp = TempDir::new().unwrap();
        let project_path = temp.path().join("new_project");

        let config = CliConfig::default();
        let args = InitArgs {
            path: project_path.clone(),
            force: false,
        };

        execute_init(&config, &args);

        assert!(project_path.exists());
        assert!(project_path.join("tests").exists());
        assert!(project_path.join("tests").join("basic_test.rs").exists());
    }

    #[test]
    fn test_execute_init_force_overwrites() {
        let temp = TempDir::new().unwrap();
        let project_path = temp.path().join("existing_project");
        std::fs::create_dir_all(project_path.join("tests")).unwrap();

        let old_content = "// old content";
        std::fs::write(
            project_path.join("tests").join("basic_test.rs"),
            old_content,
        )
        .unwrap();

        let config = CliConfig::default();
        let args = InitArgs {
            path: project_path.clone(),
            force: true,
        };

        execute_init(&config, &args);

        let content =
            std::fs::read_to_string(project_path.join("tests").join("basic_test.rs")).unwrap();
        assert!(content.contains("jugar_probar"));
        assert!(!content.contains("old content"));
    }

    #[test]
    fn test_execute_init_does_not_overwrite_without_force() {
        let temp = TempDir::new().unwrap();
        let project_path = temp.path().join("existing_project");
        std::fs::create_dir_all(project_path.join("tests")).unwrap();

        let old_content = "// old content that should remain";
        std::fs::write(
            project_path.join("tests").join("basic_test.rs"),
            old_content,
        )
        .unwrap();

        let config = CliConfig::default();
        let args = InitArgs {
            path: project_path.clone(),
            force: false,
        };

        execute_init(&config, &args);

        let content =
            std::fs::read_to_string(project_path.join("tests").join("basic_test.rs")).unwrap();
        assert!(content.contains("old content that should remain"));
    }
}
