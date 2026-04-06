//! Sanitizer output parser and PTX source mapping.

use std::collections::HashMap;

use super::types::{MemoryViolation, MemoryViolationType, SourceLocation};

// ============================================================================
// Sanitizer Output Parser
// ============================================================================

/// Parser for compute-sanitizer output
pub struct SanitizerParser;

impl SanitizerParser {
    /// Parse compute-sanitizer output and extract violations
    pub fn parse(output: &str) -> Vec<MemoryViolation> {
        let mut violations = Vec::new();
        let lines: Vec<&str> = output.lines().collect();

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i];

            // Look for "Invalid __global__ read" or similar patterns
            if line.contains("Invalid __") {
                if let Some(violation) = Self::parse_violation(&lines[i..]) {
                    violations.push(violation);
                }
            }

            i += 1;
        }

        violations
    }

    fn parse_violation(lines: &[&str]) -> Option<MemoryViolation> {
        let first_line = lines.first()?;
        let violation_type = Self::classify_violation(first_line);

        let mut kernel_name = String::from("unknown");
        let mut sass_offset = 0u64;
        let mut thread = (0u32, 0u32, 0u32);
        let mut block = (0u32, 0u32, 0u32);
        let mut address = 0u64;

        for line in lines.iter().skip(1).take(10) {
            if let Some((name, offset)) = Self::parse_kernel_location(line) {
                kernel_name = name;
                sass_offset = offset;
            }
            if let Some(parsed) = Self::parse_thread_block(line) {
                thread = parsed.0;
                block = parsed.1;
            }
            if let Some(addr) = Self::parse_address(line) {
                address = addr;
            }
        }

        Some(MemoryViolation {
            violation_type,
            kernel_name,
            sass_offset,
            thread,
            block,
            address,
            raw_message: lines.iter().take(5).copied().collect::<Vec<_>>().join("\n"),
        })
    }

    /// Classify violation type from the first line of a sanitizer error.
    fn classify_violation(line: &str) -> MemoryViolationType {
        if line.contains("__shared__ read") {
            MemoryViolationType::InvalidSharedRead {
                size: Self::extract_size(line).unwrap_or(4),
            }
        } else if line.contains("__shared__ write") {
            MemoryViolationType::InvalidSharedWrite {
                size: Self::extract_size(line).unwrap_or(4),
            }
        } else if line.contains("__global__ read") {
            MemoryViolationType::InvalidGlobalRead {
                size: Self::extract_size(line).unwrap_or(4),
            }
        } else if line.contains("__global__ write") {
            MemoryViolationType::InvalidGlobalWrite {
                size: Self::extract_size(line).unwrap_or(4),
            }
        } else if line.contains("misaligned") {
            MemoryViolationType::MisalignedAccess { addr: 0 }
        } else {
            MemoryViolationType::Other(line.to_string())
        }
    }

    /// Parse "at kernel_name+0xOFFSET" from a sanitizer line.
    fn parse_kernel_location(line: &str) -> Option<(String, u64)> {
        let at_pos = line.find(" at ")?;
        let rest = &line[at_pos + 4..];
        let plus_pos = rest.find("+0x")?;
        let kernel_name = rest[..plus_pos].trim().to_string();
        let hex_str = &rest[plus_pos + 3..];
        let hex_end = hex_str
            .find(|c: char| !c.is_ascii_hexdigit())
            .unwrap_or(hex_str.len());
        let offset = u64::from_str_radix(&hex_str[..hex_end], 16).unwrap_or(0);
        Some((kernel_name, offset))
    }

    /// Parse "Address 0xXXXX" from a sanitizer line.
    fn parse_address(line: &str) -> Option<u64> {
        let addr_pos = line.find("Address 0x")?;
        let rest = &line[addr_pos + 10..];
        let hex_end = rest
            .find(|c: char| !c.is_ascii_hexdigit())
            .unwrap_or(rest.len());
        Some(u64::from_str_radix(&rest[..hex_end], 16).unwrap_or(0))
    }

    fn extract_size(line: &str) -> Option<usize> {
        // Parse "of size N bytes"
        if let Some(pos) = line.find("of size ") {
            let rest = &line[pos + 8..];
            let num_end = rest.find(' ').unwrap_or(rest.len());
            rest[..num_end].parse().ok()
        } else {
            None
        }
    }

    fn parse_thread_block(line: &str) -> Option<((u32, u32, u32), (u32, u32, u32))> {
        // Parse "by thread (X,Y,Z) in block (X,Y,Z)"
        let parse_triple = |s: &str| -> Option<(u32, u32, u32)> {
            let s = s.trim_start_matches('(').trim_end_matches(')');
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() == 3 {
                Some((
                    parts[0].parse().ok()?,
                    parts[1].parse().ok()?,
                    parts[2].parse().ok()?,
                ))
            } else {
                None
            }
        };

        let thread_pos = line.find("thread (")?;
        let thread_end = line[thread_pos..].find(')')?;
        let thread_str = &line[thread_pos + 7..thread_pos + thread_end + 1];

        let block_pos = line.find("block (")?;
        let block_end = line[block_pos..].find(')')?;
        let block_str = &line[block_pos + 6..block_pos + block_end + 1];

        Some((parse_triple(thread_str)?, parse_triple(block_str)?))
    }
}

// ============================================================================
// PTX Source Location Mapping
// ============================================================================

/// PTX source map for kernel debugging
pub struct PtxSourceMap {
    /// PTX source code
    ptx_source: String,
    /// Map from PTX line number to source location
    line_map: HashMap<u32, SourceLocation>,
    /// Map from label name to PTX line number
    pub(super) label_lines: HashMap<String, u32>,
}

impl PtxSourceMap {
    /// Create a new source map from PTX source
    pub fn new(ptx_source: &str) -> Self {
        let mut map = Self {
            ptx_source: ptx_source.to_string(),
            line_map: HashMap::new(),
            label_lines: HashMap::new(),
        };
        map.parse_ptx();
        map
    }

    fn parse_ptx(&mut self) {
        for (line_num, line) in self.ptx_source.lines().enumerate() {
            let line_num = (line_num + 1) as u32;

            // Look for labels
            let trimmed = line.trim();
            if trimmed.ends_with(':') && !trimmed.starts_with("//") {
                let label = trimmed.trim_end_matches(':');
                self.label_lines.insert(label.to_string(), line_num);
            }

            // Look for .loc directives (if present)
            // Format: .loc file_id line_num [column]
            if trimmed.starts_with(".loc ") {
                // Parse .loc directive
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 3 {
                    if let Ok(src_line) = parts[2].parse::<u32>() {
                        // For now, just store the line number
                        // A full implementation would track file_id mapping
                        self.line_map.insert(
                            line_num,
                            SourceLocation {
                                file: "kernel.rs".to_string(),
                                line: src_line,
                                column: parts.get(3).and_then(|s| s.parse().ok()),
                                function: None,
                            },
                        );
                    }
                }
            }
        }
    }

    /// Get the label containing a given PTX line
    pub fn label_at_line(&self, target_line: u32) -> Option<&str> {
        let mut best_label = None;
        let mut best_line = 0;

        for (label, &line) in &self.label_lines {
            if line <= target_line && line > best_line {
                best_line = line;
                best_label = Some(label.as_str());
            }
        }

        best_label
    }

    /// Get PTX lines around a label
    pub fn context_around_label(&self, label: &str, context_lines: usize) -> Option<String> {
        let &label_line = self.label_lines.get(label)?;
        let lines: Vec<&str> = self.ptx_source.lines().collect();

        let start = (label_line as usize).saturating_sub(context_lines);
        let end = ((label_line as usize) + context_lines).min(lines.len());

        let mut result = String::new();
        for (i, line) in lines[start..end].iter().enumerate() {
            let actual_line = start + i + 1;
            let marker = if actual_line == label_line as usize {
                ">>>"
            } else {
                "   "
            };
            result.push_str(&format!("{} {:4}: {}\n", marker, actual_line, line));
        }

        Some(result)
    }
}
