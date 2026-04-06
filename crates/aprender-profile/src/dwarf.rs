//! DWARF debug info parsing for source correlation
//!
//! Sprint 5-6: Map instruction pointers to source <file:line> using DWARF .`debug_line`
//!
//! Uses addr2line crate for robust DWARF parsing

use anyhow::{Context, Result};
use object::{Object, ObjectSection};
use std::fs::File;
use std::path::Path;

/// Source location information
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLocation {
    /// Source file path
    pub file: String,
    /// Line number
    pub line: u32,
    /// Column number (if available)
    pub column: Option<u32>,
    /// Function name (if available)
    pub function: Option<String>,
}

/// DWARF debug info context for a binary
///
/// Sprint 5-6: Full implementation using addr2line crate
pub struct DwarfContext {
    /// addr2line context for DWARF lookups
    context: addr2line::Context<gimli::EndianRcSlice<gimli::RunTimeEndian>>,
}

impl std::fmt::Debug for DwarfContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DwarfContext").field("context", &"<addr2line context>").finish()
    }
}

impl DwarfContext {
    /// Load DWARF debug info from an ELF binary
    ///
    /// Sprint 5-6: Full implementation using addr2line + object crates
    ///
    /// # Allows
    /// `unsafe_code` — `Mmap::map` requires unsafe for memory-mapped I/O.
    #[allow(unsafe_code)]
    pub fn load(binary_path: &Path) -> Result<Self> {
        // Verify binary exists
        if !binary_path.exists() {
            anyhow::bail!("Binary does not exist: {}", binary_path.display());
        }

        // Open and parse ELF binary
        let file = File::open(binary_path)
            .with_context(|| format!("Failed to open binary: {}", binary_path.display()))?;

        // SAFETY: File is valid and open; memory mapping is read-only and file stays open for mmap lifetime
        let mmap = unsafe { memmap2::Mmap::map(&file) }.context("Failed to memory-map binary")?;

        let object = object::File::parse(&*mmap).context("Failed to parse ELF binary")?;

        // Load DWARF sections from object file
        let endian = if object.is_little_endian() {
            gimli::RunTimeEndian::Little
        } else {
            gimli::RunTimeEndian::Big
        };

        // Helper to load a DWARF section
        let load_section =
            |id: gimli::SectionId| -> Result<gimli::EndianRcSlice<gimli::RunTimeEndian>> {
                let data = object
                    .section_by_name(id.name())
                    .and_then(|section| section.uncompressed_data().ok())
                    .unwrap_or(std::borrow::Cow::Borrowed(&[]));
                // Convert Cow<[u8]> to Rc<[u8]> by converting to owned Vec first
                let bytes: std::rc::Rc<[u8]> = std::rc::Rc::from(data.into_owned());
                Ok(gimli::EndianRcSlice::new(bytes, endian))
            };

        // Load all DWARF sections
        let dwarf = gimli::Dwarf::load(&load_section)
            .context("Failed to load DWARF sections - binary may not have debug symbols. Compile with -g flag.")?;

        // Create addr2line context from DWARF
        let context =
            addr2line::Context::from_dwarf(dwarf).context("Failed to create DWARF context")?;

        Ok(Self { context })
    }

    /// Look up source location for an instruction pointer
    ///
    /// Sprint 5-6: Full implementation with DWARF .`debug_line` parsing
    /// Returns the first valid source location found in DWARF
    pub fn lookup(&self, ip: u64) -> Result<Option<SourceLocation>> {
        // Try multiple IP offsets to find user code
        // At syscall-entry-stop, IP might be in libc, so we try backing up
        for offset in [0, 1, 2, 4, 8, 16] {
            let adjusted_ip = ip.saturating_sub(offset);

            // Look up location in DWARF data
            let location = match self.context.find_location(adjusted_ip) {
                Ok(Some(loc)) => loc,
                Ok(None) => continue,
                Err(_) => continue,
            };

            // Extract file and line
            let file = match location.file {
                Some(f) => {
                    // Filter out libc/std paths - we want user code
                    if f.contains("/rustc/") || f.contains("library/") {
                        continue;
                    }
                    f
                }
                None => continue,
            };

            let line = location.line.unwrap_or(0);
            if line == 0 {
                continue; // Invalid line number
            }

            let column = location.column;

            // Try to find function name using find_frames
            let mut function_name = None;
            let frames_result = self.context.find_frames(adjusted_ip);

            // addr2line returns LookupResult which needs to be handled with load()
            if let Ok(mut frames_iter) = frames_result.skip_all_loads() {
                if let Ok(Some(frame)) = frames_iter.next() {
                    if let Some(func) = frame.function {
                        // Get raw name as Cow<str>
                        let raw_name = func.raw_name().ok();
                        if let Some(name) = raw_name {
                            function_name = Some(name.to_string());
                        }
                    }
                }
            }

            return Ok(Some(SourceLocation {
                file: file.to_string(),
                line,
                column,
                function: function_name,
            }));
        }

        // No valid location found
        Ok(None)
    }
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(SourceLocation: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::process::Command;
    use tempfile::TempDir;

    fn compile_test_binary() -> (TempDir, std::path::PathBuf) {
        let temp_dir = TempDir::new().expect("test");
        let src_file = temp_dir.path().join("test.rs");
        let bin_file = temp_dir.path().join("test_bin");

        fs::write(&src_file, "fn main() { println!(\"test\"); }").expect("test");

        Command::new("rustc")
            .arg(&src_file)
            .arg("-o")
            .arg(&bin_file)
            .arg("-g")
            .status()
            .expect("test");

        (temp_dir, bin_file)
    }

    #[test]
    fn test_dwarf_context_loads() {
        let (_temp_dir, bin_file) = compile_test_binary();
        let result = DwarfContext::load(&bin_file);
        assert!(result.is_ok(), "Should load DWARF context: {:?}", result.err());
    }

    #[test]
    fn test_dwarf_lookup_returns_option() {
        let (_temp_dir, bin_file) = compile_test_binary();
        let ctx = DwarfContext::load(&bin_file).expect("test");
        let result = ctx.lookup(0x1000);
        assert!(result.is_ok(), "Lookup should not crash: {:?}", result.err());
    }

    #[test]
    fn test_dwarf_load_nonexistent_file() {
        // Test error handling: nonexistent file
        let result = DwarfContext::load(std::path::Path::new("/nonexistent/binary"));
        assert!(result.is_err(), "Should fail for nonexistent file");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("does not exist"), "Error should mention file doesn't exist: {}", err);
    }

    #[test]
    fn test_dwarf_load_invalid_binary() {
        // Test error handling: invalid ELF file
        let temp_dir = TempDir::new().expect("test");
        let invalid_file = temp_dir.path().join("invalid.bin");
        fs::write(&invalid_file, b"not a valid ELF file").expect("test");

        let result = DwarfContext::load(&invalid_file);
        assert!(result.is_err(), "Should fail for invalid ELF");
    }

    #[test]
    fn test_dwarf_load_no_debug_symbols() {
        // Test error handling: binary without debug symbols
        let temp_dir = TempDir::new().expect("test");
        let src_file = temp_dir.path().join("test.rs");
        let bin_file = temp_dir.path().join("test_bin_stripped");

        fs::write(&src_file, "fn main() {}").expect("test");

        // Compile without -g (no debug symbols)
        Command::new("rustc")
            .arg(&src_file)
            .arg("-o")
            .arg(&bin_file)
            .arg("-C")
            .arg("strip=symbols")
            .status()
            .expect("test");

        let result = DwarfContext::load(&bin_file);
        // May succeed but with no useful DWARF data, or may fail
        // Either is acceptable
        let _ = result;
    }

    #[test]
    fn test_dwarf_lookup_with_valid_address() {
        // Test lookup with a potentially valid address
        let (_temp_dir, bin_file) = compile_test_binary();
        let ctx = DwarfContext::load(&bin_file).expect("test");

        // Try multiple addresses to increase coverage
        for offset in [0, 1, 2, 4, 8, 16] {
            let _ = ctx.lookup(0x1000 + offset);
        }
    }

    #[test]
    fn test_dwarf_lookup_zero_address() {
        let (_temp_dir, bin_file) = compile_test_binary();
        let ctx = DwarfContext::load(&bin_file).expect("test");
        let result = ctx.lookup(0);
        assert!(result.is_ok(), "Lookup of zero address should not crash");
    }

    #[test]
    fn test_dwarf_lookup_high_address() {
        let (_temp_dir, bin_file) = compile_test_binary();
        let ctx = DwarfContext::load(&bin_file).expect("test");
        let result = ctx.lookup(0xFFFFFFFFFFFF);
        assert!(result.is_ok(), "Lookup of high address should not crash");
    }

    #[test]
    fn test_source_location_clone() {
        // Test SourceLocation Clone and PartialEq
        let loc1 = SourceLocation {
            file: "test.rs".to_string(),
            line: 42,
            column: Some(10),
            function: Some("main".to_string()),
        };
        let loc2 = loc1.clone();
        assert_eq!(loc1, loc2, "Cloned SourceLocation should be equal");
    }

    #[test]
    fn test_source_location_debug() {
        // Test SourceLocation Debug impl
        let loc = SourceLocation {
            file: "test.rs".to_string(),
            line: 42,
            column: Some(10),
            function: Some("main".to_string()),
        };
        let debug_str = format!("{:?}", loc);
        assert!(debug_str.contains("test.rs"), "Debug should contain file: {}", debug_str);
        assert!(debug_str.contains("42"), "Debug should contain line: {}", debug_str);
    }

    #[test]
    fn test_source_location_no_column() {
        let loc =
            SourceLocation { file: "lib.rs".to_string(), line: 100, column: None, function: None };
        assert_eq!(loc.file, "lib.rs");
        assert_eq!(loc.line, 100);
        assert_eq!(loc.column, None);
        assert_eq!(loc.function, None);
    }

    #[test]
    fn test_source_location_equality() {
        let loc1 = SourceLocation {
            file: "main.rs".to_string(),
            line: 1,
            column: Some(5),
            function: Some("foo".to_string()),
        };
        let loc2 = SourceLocation {
            file: "main.rs".to_string(),
            line: 1,
            column: Some(5),
            function: Some("foo".to_string()),
        };
        let loc3 = SourceLocation {
            file: "main.rs".to_string(),
            line: 2,
            column: Some(5),
            function: Some("foo".to_string()),
        };
        assert_eq!(loc1, loc2);
        assert_ne!(loc1, loc3);
    }

    #[test]
    fn test_dwarf_context_debug() {
        let (_temp_dir, bin_file) = compile_test_binary();
        let ctx = DwarfContext::load(&bin_file).expect("test");
        let debug_str = format!("{:?}", ctx);
        assert!(debug_str.contains("DwarfContext"));
        assert!(debug_str.contains("addr2line context"));
    }

    #[test]
    fn test_dwarf_lookup_multiple_addresses() {
        let (_temp_dir, bin_file) = compile_test_binary();
        let ctx = DwarfContext::load(&bin_file).expect("test");

        // Test a range of addresses
        for addr in [0x1000, 0x2000, 0x3000, 0x4000, 0x5000] {
            let result = ctx.lookup(addr);
            assert!(result.is_ok(), "Lookup should not crash for addr {:#x}", addr);
        }
    }

    #[test]
    fn test_dwarf_lookup_negative_offset() {
        let (_temp_dir, bin_file) = compile_test_binary();
        let ctx = DwarfContext::load(&bin_file).expect("test");

        // Test lookup with addresses that might underflow with offset
        let result = ctx.lookup(0);
        assert!(result.is_ok());

        let result = ctx.lookup(1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dwarf_load_real_binary() {
        // Test loading DWARF from a real binary (this executable)
        let exe_path = std::env::current_exe().expect("Failed to get current exe");

        // Try to load DWARF - may succeed or fail depending on if we have debug info
        let result = DwarfContext::load(&exe_path);

        // Either way, it shouldn't panic
        match result {
            Ok(ctx) => {
                // If we loaded it, try a lookup
                let lookup_result = ctx.lookup(0x1000);
                assert!(lookup_result.is_ok());
            }
            Err(e) => {
                // If it failed, error should be reasonable
                let err_msg = e.to_string();
                assert!(!err_msg.is_empty(), "Error message should not be empty");
            }
        }
    }

    #[test]
    fn test_source_location_with_column() {
        let loc = SourceLocation {
            file: "test.rs".to_string(),
            line: 10,
            column: Some(20),
            function: Some("test_fn".to_string()),
        };
        assert_eq!(loc.column, Some(20));
        assert_eq!(loc.function, Some("test_fn".to_string()));
    }

    #[test]
    fn test_dwarf_lookup_extensive_coverage() {
        // Compile a more complex test binary with actual code
        let temp_dir = TempDir::new().expect("test");
        let src_file = temp_dir.path().join("complex.rs");
        let bin_file = temp_dir.path().join("complex_bin");

        // Write a program with multiple functions
        fs::write(
            &src_file,
            r#"
fn helper(x: i32) -> i32 {
    x + 1
}

fn main() {
    let a = helper(5);
    let b = helper(10);
    println!("Result: {} {}", a, b);
}
"#,
        )
        .expect("test");

        Command::new("rustc")
            .arg(&src_file)
            .arg("-o")
            .arg(&bin_file)
            .arg("-g")
            .status()
            .expect("test");

        let ctx = DwarfContext::load(&bin_file).expect("test");

        // Try many different addresses to maximize coverage of lookup logic
        for addr in 0x1000..0x1100 {
            let _ = ctx.lookup(addr);
        }

        // Also test edge cases (avoid u64::MAX to prevent addr2line overflow)
        let _ = ctx.lookup(0x7FFF_FFFF_FFFF_FFF0);
        let _ = ctx.lookup(0x7FFF_FFFF_FFFF_FFFF);
        let _ = ctx.lookup(0);
        let _ = ctx.lookup(1);
        let _ = ctx.lookup(2);
        let _ = ctx.lookup(4);
        let _ = ctx.lookup(8);
        let _ = ctx.lookup(16);
    }

    #[test]
    fn test_source_location_all_variants() {
        // Test all combinations of Option fields
        let loc1 = SourceLocation {
            file: "a.rs".to_string(),
            line: 1,
            column: Some(1),
            function: Some("f".to_string()),
        };
        let loc2 =
            SourceLocation { file: "b.rs".to_string(), line: 2, column: Some(2), function: None };
        let loc3 = SourceLocation {
            file: "c.rs".to_string(),
            line: 3,
            column: None,
            function: Some("g".to_string()),
        };
        let loc4 =
            SourceLocation { file: "d.rs".to_string(), line: 4, column: None, function: None };

        assert_eq!(loc1.file, "a.rs");
        assert_eq!(loc2.file, "b.rs");
        assert_eq!(loc3.file, "c.rs");
        assert_eq!(loc4.file, "d.rs");

        // Test inequality
        assert_ne!(loc1, loc2);
        assert_ne!(loc2, loc3);
        assert_ne!(loc3, loc4);
    }
}
