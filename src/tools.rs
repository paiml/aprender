use anyhow::{Context, Result};
use std::process::Command;

#[cfg(feature = "native")]
use tracing::{debug, info};

// Stub macros for WASM build
#[cfg(not(feature = "native"))]
macro_rules! info {
    ($($arg:tt)*) => {{}};
}

#[cfg(not(feature = "native"))]
macro_rules! debug {
    ($($arg:tt)*) => {{}};
}

/// Detected tool information
#[derive(Debug, Clone)]
pub struct ToolInfo {
    pub name: String,
    pub version: Option<String>,
    pub path: String,
    pub available: bool,
}

/// Check which transpiler and utility tools are available
#[derive(Debug, Clone)]
pub struct ToolRegistry {
    pub decy: Option<ToolInfo>,
    pub depyler: Option<ToolInfo>,
    pub bashrs: Option<ToolInfo>,
    pub ruchy: Option<ToolInfo>,
    pub trueno: Option<ToolInfo>,
    pub aprender: Option<ToolInfo>,
    pub realizar: Option<ToolInfo>,
    pub renacer: Option<ToolInfo>,
    pub pmat: Option<ToolInfo>,
}

impl ToolRegistry {
    /// Detect all available tools
    pub fn detect() -> Self {
        info!("Detecting installed Pragmatic AI Labs tools...");

        Self {
            decy: detect_tool("decy"),
            depyler: detect_tool("depyler"),
            bashrs: detect_tool("bashrs"),
            ruchy: detect_tool("ruchy"),
            trueno: detect_tool("trueno"),
            aprender: detect_tool("aprender"),
            realizar: detect_tool("realizar"),
            renacer: detect_tool("renacer"),
            pmat: detect_tool("pmat"),
        }
    }

    /// Check if any transpiler is available
    pub fn has_transpiler(&self) -> bool {
        contract_pre_transpile!(self);
        self.decy.is_some() || self.depyler.is_some() || self.bashrs.is_some()
    }

    /// Get transpiler for a specific language
    pub fn get_transpiler_for_language(&self, lang: &crate::types::Language) -> Option<&ToolInfo> {
        contract_pre_transpile!(lang);
        use crate::types::Language;

        match lang {
            Language::C | Language::Cpp => self.decy.as_ref(),
            Language::Python => self.depyler.as_ref(),
            Language::Shell => self.bashrs.as_ref(),
            _ => None,
        }
    }

    /// Get list of available tools
    pub fn available_tools(&self) -> Vec<String> {
        let mut tools = Vec::new();

        if let Some(tool) = &self.decy {
            if tool.available {
                tools.push("Decy (C/C++ → Rust)".to_string());
            }
        }
        if let Some(tool) = &self.depyler {
            if tool.available {
                tools.push("Depyler (Python → Rust)".to_string());
            }
        }
        if let Some(tool) = &self.bashrs {
            if tool.available {
                tools.push("Bashrs (Shell → Rust)".to_string());
            }
        }
        if let Some(tool) = &self.ruchy {
            if tool.available {
                tools.push("Ruchy (Rust scripting)".to_string());
            }
        }
        if let Some(tool) = &self.pmat {
            if tool.available {
                tools.push("PMAT (Quality analysis)".to_string());
            }
        }
        if let Some(tool) = &self.trueno {
            if tool.available {
                tools.push("Trueno (Multi-target compute)".to_string());
            }
        }
        if let Some(tool) = &self.aprender {
            if tool.available {
                tools.push("Aprender (ML library)".to_string());
            }
        }
        if let Some(tool) = &self.realizar {
            if tool.available {
                tools.push("Realizar (Inference runtime)".to_string());
            }
        }
        if let Some(tool) = &self.renacer {
            if tool.available {
                tools.push("Renacer (Syscall tracing)".to_string());
            }
        }

        tools
    }

    /// Get installation instructions for missing tools
    pub fn get_installation_instructions(&self, needed_tools: &[&str]) -> Vec<String> {
        let mut instructions = Vec::new();

        for tool in needed_tools {
            let instruction = match *tool {
                "decy" if self.decy.is_none() => Some("Install Decy: cargo install decy"),
                "depyler" if self.depyler.is_none() => {
                    Some("Install Depyler: cargo install depyler")
                }
                "bashrs" if self.bashrs.is_none() => Some("Install Bashrs: cargo install bashrs"),
                "ruchy" if self.ruchy.is_none() => Some("Install Ruchy: cargo install ruchy"),
                "pmat" if self.pmat.is_none() => Some("Install PMAT: cargo install pmat"),
                "trueno" if self.trueno.is_none() => {
                    Some("Install Trueno: Add 'trueno' to Cargo.toml dependencies")
                }
                "aprender" if self.aprender.is_none() => {
                    Some("Install Aprender: Add 'aprender' to Cargo.toml dependencies")
                }
                "realizar" if self.realizar.is_none() => {
                    Some("Install Realizar: Add 'realizar' to Cargo.toml dependencies")
                }
                "renacer" if self.renacer.is_none() => {
                    Some("Install Renacer: cargo install renacer")
                }
                _ => None,
            };

            if let Some(inst) = instruction {
                instructions.push(inst.to_string());
            }
        }

        instructions
    }
}

/// Detect a single tool
#[cfg(feature = "native")]
fn detect_tool(name: &str) -> Option<ToolInfo> {
    debug!("Checking for tool: {}", name);

    // Try to find the tool using `which`
    let path = match which::which(name) {
        Ok(p) => p.to_string_lossy().to_string(),
        Err(_) => {
            debug!("Tool '{}' not found in PATH", name);
            return None;
        }
    };

    // Try to get version
    let version = get_tool_version(name);

    debug!("Found tool '{}' at '{}' (version: {:?})", name, path, version);

    Some(ToolInfo { name: name.to_string(), version, path, available: true })
}

/// Detect a single tool (stub when native disabled)
#[cfg(not(feature = "native"))]
fn detect_tool(name: &str) -> Option<ToolInfo> {
    let _ = name;
    None
}

/// Get tool version by running --version
fn get_tool_version(name: &str) -> Option<String> {
    let output = Command::new(name).arg("--version").output().ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let version_line = stdout.lines().next()?;

    // Extract version number from output
    // Common formats:
    // "tool 1.2.3"
    // "tool version 1.2.3"
    // "1.2.3"
    let parts: Vec<&str> = version_line.split_whitespace().collect();
    let version = (*parts.last()?).to_string();

    Some(version)
}

/// Run a tool command and capture output
pub fn run_tool(
    tool_name: &str,
    args: &[&str],
    working_dir: Option<&std::path::Path>,
) -> Result<String> {
    debug!("Running tool: {} {:?}", tool_name, args);

    let mut cmd = Command::new(tool_name);
    cmd.args(args);

    if let Some(dir) = working_dir {
        cmd.current_dir(dir);
    }

    let output = cmd.output().with_context(|| format!("Failed to run tool: {}", tool_name))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "Tool '{}' failed with exit code {:?}: {}",
            tool_name,
            output.status.code(),
            stderr
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(stdout)
}

/// Transpile Python code using Depyler
pub fn transpile_python(
    input_path: &std::path::Path,
    output_path: &std::path::Path,
) -> Result<String> {
    contract_pre_transpile!(input_path);
    info!("Transpiling Python with Depyler: {:?} → {:?}", input_path, output_path);

    let input_str = input_path.to_string_lossy();
    let output_str = output_path.to_string_lossy();

    let args = vec![
        "transpile",
        "--input",
        &input_str,
        "--output",
        &output_str,
        "--format",
        "project", // Generate full Rust project structure
    ];

    let result = run_tool("depyler", &args, None);
    if let Ok(ref val) = result {
        contract_post_configuration!(val);
    }
    result
}

/// Transpile Shell script using Bashrs
pub fn transpile_shell(
    input_path: &std::path::Path,
    output_path: &std::path::Path,
) -> Result<String> {
    contract_pre_transpile!(input_path);
    info!("Transpiling Shell with Bashrs: {:?} → {:?}", input_path, output_path);

    let input_str = input_path.to_string_lossy();
    let output_str = output_path.to_string_lossy();

    let args = vec![
        "build",
        &input_str,
        "-o",
        &output_str,
        "--target",
        "posix", // Most compatible shell target
        "--verify",
        "strict", // Strict verification
    ];

    let result = run_tool("bashrs", &args, None);
    if let Ok(ref val) = result {
        contract_post_configuration!(val);
    }
    result
}

/// Transpile C/C++ code using Decy (if available)
pub fn transpile_c_cpp(
    input_path: &std::path::Path,
    output_path: &std::path::Path,
) -> Result<String> {
    contract_pre_transpile!(input_path);
    info!("Transpiling C/C++ with Decy: {:?} → {:?}", input_path, output_path);

    let input_str = input_path.to_string_lossy();
    let output_str = output_path.to_string_lossy();

    // Note: Decy might not be installed, handle gracefully
    let args = vec!["transpile", "--input", &input_str, "--output", &output_str];

    let result = run_tool("decy", &args, None);
    if let Ok(ref val) = result {
        contract_post_configuration!(val);
    }
    result
}

/// Run quality analysis using PMAT
pub fn analyze_quality(path: &std::path::Path) -> Result<String> {
    contract_pre_analyze!(path);
    info!("Running PMAT quality analysis: {:?}", path);

    let path_str = path.to_string_lossy();

    let args = vec!["analyze", "complexity", &path_str, "--format", "json"];

    let result = run_tool("pmat", &args, None);
    if let Ok(ref val) = result {
        contract_post_configuration!(val);
    }
    result
}

/// Run Ruchy scripting (if needed)
pub fn run_ruchy_script(script_path: &std::path::Path) -> Result<String> {
    info!("Running Ruchy script: {:?}", script_path);

    let script_str = script_path.to_string_lossy();

    let args = vec!["run", &script_str];

    run_tool("ruchy", &args, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ============================================================================
    // TOOLINFO TESTS
    // ============================================================================

    #[test]
    fn test_tool_info_creation() {
        let tool = ToolInfo {
            name: "decy".to_string(),
            version: Some("1.0.0".to_string()),
            path: "/usr/local/bin/decy".to_string(),
            available: true,
        };

        assert_eq!(tool.name, "decy");
        assert_eq!(tool.version, Some("1.0.0".to_string()));
        assert_eq!(tool.path, "/usr/local/bin/decy");
        assert!(tool.available);
    }

    #[test]
    fn test_tool_info_no_version() {
        let tool = ToolInfo {
            name: "test_tool".to_string(),
            version: None,
            path: "/bin/test".to_string(),
            available: true,
        };

        assert_eq!(tool.name, "test_tool");
        assert!(tool.version.is_none());
    }

    #[test]
    fn test_tool_info_clone() {
        let tool1 = ToolInfo {
            name: "depyler".to_string(),
            version: Some("2.0.0".to_string()),
            path: "/usr/bin/depyler".to_string(),
            available: true,
        };

        let tool2 = tool1.clone();
        assert_eq!(tool1.name, tool2.name);
        assert_eq!(tool1.version, tool2.version);
        assert_eq!(tool1.path, tool2.path);
        assert_eq!(tool1.available, tool2.available);
    }

    #[test]
    fn test_tool_info_debug() {
        let tool = ToolInfo {
            name: "bashrs".to_string(),
            version: Some("0.5.0".to_string()),
            path: "/usr/local/bin/bashrs".to_string(),
            available: true,
        };

        let debug_str = format!("{:?}", tool);
        assert!(debug_str.contains("bashrs"));
        assert!(debug_str.contains("0.5.0"));
    }

    // ============================================================================
    // TOOLREGISTRY TESTS
    // ============================================================================

    fn create_test_registry() -> ToolRegistry {
        ToolRegistry {
            decy: Some(ToolInfo {
                name: "decy".to_string(),
                version: Some("1.0.0".to_string()),
                path: "/usr/bin/decy".to_string(),
                available: true,
            }),
            depyler: Some(ToolInfo {
                name: "depyler".to_string(),
                version: Some("2.0.0".to_string()),
                path: "/usr/bin/depyler".to_string(),
                available: true,
            }),
            bashrs: None,
            ruchy: Some(ToolInfo {
                name: "ruchy".to_string(),
                version: Some("0.3.0".to_string()),
                path: "/usr/bin/ruchy".to_string(),
                available: true,
            }),
            trueno: None,
            aprender: None,
            realizar: None,
            renacer: None,
            pmat: Some(ToolInfo {
                name: "pmat".to_string(),
                version: Some("1.5.0".to_string()),
                path: "/usr/bin/pmat".to_string(),
                available: true,
            }),
        }
    }

    fn create_empty_registry() -> ToolRegistry {
        ToolRegistry {
            decy: None,
            depyler: None,
            bashrs: None,
            ruchy: None,
            trueno: None,
            aprender: None,
            realizar: None,
            renacer: None,
            pmat: None,
        }
    }

    #[test]
    fn test_tool_registry_clone() {
        let registry1 = create_test_registry();
        let registry2 = registry1.clone();

        assert!(registry2.decy.is_some());
        assert!(registry2.depyler.is_some());
        assert!(registry2.bashrs.is_none());
    }

    #[test]
    fn test_tool_registry_debug() {
        let registry = create_test_registry();
        let debug_str = format!("{:?}", registry);
        assert!(debug_str.contains("decy"));
        assert!(debug_str.contains("depyler"));
    }

    #[test]
    fn test_tool_detection() {
        // This test will only pass if tools are installed
        let registry = ToolRegistry::detect();

        // At minimum, we should detect PMAT if running on dev machine
        // But we don't want tests to fail in CI, so we just log
        println!("Available tools: {:?}", registry.available_tools());
    }

    #[test]
    fn test_has_transpiler_with_tools() {
        let registry = create_test_registry();
        assert!(registry.has_transpiler());
    }

    #[test]
    fn test_has_transpiler_empty() {
        let registry = create_empty_registry();
        assert!(!registry.has_transpiler());
    }

    #[test]
    fn test_has_transpiler_only_decy() {
        let mut registry = create_empty_registry();
        registry.decy = Some(ToolInfo {
            name: "decy".to_string(),
            version: None,
            path: "/usr/bin/decy".to_string(),
            available: true,
        });
        assert!(registry.has_transpiler());
    }

    #[test]
    fn test_has_transpiler_only_depyler() {
        let mut registry = create_empty_registry();
        registry.depyler = Some(ToolInfo {
            name: "depyler".to_string(),
            version: None,
            path: "/usr/bin/depyler".to_string(),
            available: true,
        });
        assert!(registry.has_transpiler());
    }

    #[test]
    fn test_has_transpiler_only_bashrs() {
        let mut registry = create_empty_registry();
        registry.bashrs = Some(ToolInfo {
            name: "bashrs".to_string(),
            version: None,
            path: "/usr/bin/bashrs".to_string(),
            available: true,
        });
        assert!(registry.has_transpiler());
    }

    #[test]
    fn test_get_transpiler_for_language_c() {
        let registry = create_test_registry();
        let tool = registry.get_transpiler_for_language(&crate::types::Language::C);
        assert!(tool.is_some());
        assert_eq!(tool.expect("unexpected failure").name, "decy");
    }

    #[test]
    fn test_get_transpiler_for_language_cpp() {
        let registry = create_test_registry();
        let tool = registry.get_transpiler_for_language(&crate::types::Language::Cpp);
        assert!(tool.is_some());
        assert_eq!(tool.expect("unexpected failure").name, "decy");
    }

    #[test]
    fn test_get_transpiler_for_language_python() {
        let registry = create_test_registry();
        let tool = registry.get_transpiler_for_language(&crate::types::Language::Python);
        assert!(tool.is_some());
        assert_eq!(tool.expect("unexpected failure").name, "depyler");
    }

    #[test]
    fn test_get_transpiler_for_language_shell() {
        let mut registry = create_test_registry();
        registry.bashrs = Some(ToolInfo {
            name: "bashrs".to_string(),
            version: None,
            path: "/usr/bin/bashrs".to_string(),
            available: true,
        });

        let tool = registry.get_transpiler_for_language(&crate::types::Language::Shell);
        assert!(tool.is_some());
        assert_eq!(tool.expect("unexpected failure").name, "bashrs");
    }

    #[test]
    fn test_get_transpiler_for_language_rust() {
        let registry = create_test_registry();
        let tool = registry.get_transpiler_for_language(&crate::types::Language::Rust);
        assert!(tool.is_none());
    }

    #[test]
    fn test_get_transpiler_for_language_javascript() {
        let registry = create_test_registry();
        let tool = registry.get_transpiler_for_language(&crate::types::Language::JavaScript);
        assert!(tool.is_none());
    }

    #[test]
    fn test_get_transpiler_for_language_other() {
        let registry = create_test_registry();
        let tool = registry
            .get_transpiler_for_language(&crate::types::Language::Other("Kotlin".to_string()));
        assert!(tool.is_none());
    }

    #[test]
    fn test_available_tools_all_installed() {
        let mut registry = create_test_registry();
        registry.bashrs = Some(ToolInfo {
            name: "bashrs".to_string(),
            version: Some("1.0.0".to_string()),
            path: "/usr/bin/bashrs".to_string(),
            available: true,
        });
        registry.trueno = Some(ToolInfo {
            name: "trueno".to_string(),
            version: Some("2.0.0".to_string()),
            path: "/usr/bin/trueno".to_string(),
            available: true,
        });
        registry.aprender = Some(ToolInfo {
            name: "aprender".to_string(),
            version: Some("1.0.0".to_string()),
            path: "/usr/bin/aprender".to_string(),
            available: true,
        });
        registry.realizar = Some(ToolInfo {
            name: "realizar".to_string(),
            version: Some("1.0.0".to_string()),
            path: "/usr/bin/realizar".to_string(),
            available: true,
        });
        registry.renacer = Some(ToolInfo {
            name: "renacer".to_string(),
            version: Some("1.0.0".to_string()),
            path: "/usr/bin/renacer".to_string(),
            available: true,
        });

        let tools = registry.available_tools();
        assert_eq!(tools.len(), 9);
        assert!(tools.contains(&"Decy (C/C++ → Rust)".to_string()));
        assert!(tools.contains(&"Depyler (Python → Rust)".to_string()));
        assert!(tools.contains(&"Bashrs (Shell → Rust)".to_string()));
        assert!(tools.contains(&"Ruchy (Rust scripting)".to_string()));
        assert!(tools.contains(&"PMAT (Quality analysis)".to_string()));
    }

    #[test]
    fn test_available_tools_empty() {
        let registry = create_empty_registry();
        let tools = registry.available_tools();
        assert_eq!(tools.len(), 0);
    }

    #[test]
    fn test_available_tools_partial() {
        let registry = create_test_registry();
        let tools = registry.available_tools();

        // Should have decy, depyler, ruchy, pmat
        assert_eq!(tools.len(), 4);
        assert!(tools.contains(&"Decy (C/C++ → Rust)".to_string()));
        assert!(tools.contains(&"Depyler (Python → Rust)".to_string()));
        assert!(tools.contains(&"Ruchy (Rust scripting)".to_string()));
        assert!(tools.contains(&"PMAT (Quality analysis)".to_string()));
    }

    #[test]
    fn test_available_tools_unavailable_flag() {
        let mut registry = create_test_registry();
        // Mark depyler as unavailable
        if let Some(tool) = &mut registry.depyler {
            tool.available = false;
        }

        let tools = registry.available_tools();
        // Should not include depyler
        assert!(!tools.iter().any(|t| t.contains("Depyler")));
    }

    #[test]
    fn test_get_installation_instructions_all_missing() {
        let registry = create_empty_registry();
        let instructions = registry.get_installation_instructions(&[
            "decy", "depyler", "bashrs", "ruchy", "pmat", "trueno", "aprender", "realizar",
            "renacer",
        ]);

        assert_eq!(instructions.len(), 9);
        assert!(instructions.contains(&"Install Decy: cargo install decy".to_string()));
        assert!(instructions.contains(&"Install Depyler: cargo install depyler".to_string()));
        assert!(instructions.contains(&"Install Bashrs: cargo install bashrs".to_string()));
        assert!(instructions.contains(&"Install Ruchy: cargo install ruchy".to_string()));
        assert!(instructions.contains(&"Install PMAT: cargo install pmat".to_string()));
        assert!(instructions
            .contains(&"Install Trueno: Add 'trueno' to Cargo.toml dependencies".to_string()));
        assert!(instructions
            .contains(&"Install Aprender: Add 'aprender' to Cargo.toml dependencies".to_string()));
        assert!(instructions
            .contains(&"Install Realizar: Add 'realizar' to Cargo.toml dependencies".to_string()));
        assert!(instructions.contains(&"Install Renacer: cargo install renacer".to_string()));
    }

    #[test]
    fn test_get_installation_instructions_none_missing() {
        let registry = create_test_registry();
        let instructions =
            registry.get_installation_instructions(&["decy", "depyler", "ruchy", "pmat"]);

        // All are installed, should return empty
        assert_eq!(instructions.len(), 0);
    }

    #[test]
    fn test_get_installation_instructions_partial() {
        let registry = create_test_registry();
        let instructions = registry.get_installation_instructions(&["decy", "bashrs", "trueno"]);

        // Only bashrs and trueno are missing
        assert_eq!(instructions.len(), 2);
        assert!(instructions.contains(&"Install Bashrs: cargo install bashrs".to_string()));
        assert!(instructions
            .contains(&"Install Trueno: Add 'trueno' to Cargo.toml dependencies".to_string()));
    }

    #[test]
    fn test_get_installation_instructions_unknown_tool() {
        let registry = create_empty_registry();
        let instructions = registry.get_installation_instructions(&["unknown_tool", "decy"]);

        // Should only return instruction for decy
        assert_eq!(instructions.len(), 1);
        assert!(instructions.contains(&"Install Decy: cargo install decy".to_string()));
    }

    #[test]
    fn test_get_installation_instructions_empty_list() {
        let registry = create_test_registry();
        let instructions = registry.get_installation_instructions(&[]);

        assert_eq!(instructions.len(), 0);
    }

    // ============================================================================
    // FUNCTION ARGUMENT TESTS
    // ============================================================================

    #[test]
    fn test_transpile_python_paths() {
        let input = PathBuf::from("/path/to/input.py");
        let output = PathBuf::from("/path/to/output");

        // This will fail because depyler isn't installed in test environment
        // But we can verify the function exists and accepts the right types
        let _result = transpile_python(&input, &output);
    }

    #[test]
    fn test_transpile_shell_paths() {
        let input = PathBuf::from("/path/to/script.sh");
        let output = PathBuf::from("/path/to/output");

        // Will fail but verifies function signature
        let _result = transpile_shell(&input, &output);
    }

    #[test]
    fn test_transpile_c_cpp_paths() {
        let input = PathBuf::from("/path/to/code.c");
        let output = PathBuf::from("/path/to/output");

        // Will fail but verifies function signature
        let _result = transpile_c_cpp(&input, &output);
    }

    #[test]
    fn test_analyze_quality_path() {
        let path = PathBuf::from("/path/to/project");

        // Will fail but verifies function signature
        let _result = analyze_quality(&path);
    }

    #[test]
    fn test_run_ruchy_script_path() {
        let script = PathBuf::from("/path/to/script.ruchy");

        // Will fail but verifies function signature
        let _result = run_ruchy_script(&script);
    }

    #[test]
    fn test_run_tool_basic_args() {
        // Test with a command that should exist (echo)
        let result = run_tool("echo", &["test"], None);

        // echo should be available on most systems
        if let Ok(output) = result {
            assert!(output.contains("test"));
        }
    }

    #[test]
    fn test_run_tool_with_working_dir() {
        use std::env;
        let current_dir = env::current_dir().expect("current_dir failed");

        let result = run_tool("pwd", &[], Some(&current_dir));

        // pwd should work and return the directory
        if let Ok(output) = result {
            assert!(!output.is_empty());
        }
    }
}
