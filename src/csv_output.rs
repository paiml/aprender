//! CSV output format for syscall traces
//!
//! Sprint 17: CSV output for spreadsheet analysis and machine parsing

/// CSV record for a single syscall event
#[derive(Debug, Clone)]
pub struct CsvSyscall {
    pub name: String,
    pub arguments: String,
    pub result: i64,
    pub duration_us: Option<u64>,
    pub source_location: Option<String>,
}

/// CSV output formatter
#[derive(Debug)]
pub struct CsvOutput {
    syscalls: Vec<CsvSyscall>,
    include_timing: bool,
    include_source: bool,
}

impl CsvOutput {
    /// Create a new CSV output formatter
    pub fn new(include_timing: bool, include_source: bool) -> Self {
        Self { syscalls: Vec::new(), include_timing, include_source }
    }

    /// Add a syscall to the output
    pub fn add_syscall(&mut self, syscall: CsvSyscall) {
        self.syscalls.push(syscall);
    }

    /// Generate CSV header row based on enabled flags
    fn header(&self) -> String {
        let mut headers = vec!["syscall", "arguments", "result"];

        if self.include_timing {
            headers.push("duration");
        }

        if self.include_source {
            headers.push("source_location");
        }

        headers.join(",")
    }

    /// Escape CSV field (handle commas, quotes, newlines)
    fn escape_field(field: &str) -> String {
        // If field contains comma, quote, or newline, wrap in quotes and escape quotes
        if field.contains(',') || field.contains('"') || field.contains('\n') {
            format!("\"{}\"", field.replace('"', "\"\""))
        } else {
            field.to_string()
        }
    }

    /// Format a syscall as CSV row
    fn format_syscall(&self, syscall: &CsvSyscall) -> String {
        let mut fields = vec![
            Self::escape_field(&syscall.name),
            Self::escape_field(&syscall.arguments),
            syscall.result.to_string(),
        ];

        if self.include_timing {
            if let Some(duration) = syscall.duration_us {
                fields.push(format!("{duration}us"));
            } else {
                fields.push(String::new());
            }
        }

        if self.include_source {
            if let Some(ref source) = syscall.source_location {
                fields.push(Self::escape_field(source));
            } else {
                fields.push(String::new());
            }
        }

        fields.join(",")
    }

    /// Generate CSV output as string
    pub fn to_csv(&self) -> String {
        let mut output = String::new();

        // Add header
        output.push_str(&self.header());
        output.push('\n');

        // Add each syscall
        for syscall in &self.syscalls {
            output.push_str(&self.format_syscall(syscall));
            output.push('\n');
        }

        output
    }
}

/// CSV statistics output formatter (for -c mode)
#[derive(Debug)]
pub struct CsvStatsOutput {
    stats: Vec<CsvStat>,
}

#[derive(Debug, Clone)]
pub struct CsvStat {
    pub syscall: String,
    pub calls: u64,
    pub errors: u64,
    pub total_time_us: Option<u64>,
}

impl CsvStatsOutput {
    /// Create a new CSV stats output formatter
    pub fn new() -> Self {
        Self { stats: Vec::new() }
    }

    /// Add a statistic
    pub fn add_stat(&mut self, stat: CsvStat) {
        self.stats.push(stat);
    }

    /// Generate CSV output for statistics
    pub fn to_csv(&self, include_timing: bool) -> String {
        let mut output = String::new();

        // Header
        if include_timing {
            output.push_str("syscall,calls,errors,total_time\n");
        } else {
            output.push_str("syscall,calls,errors\n");
        }

        // Stats rows
        for stat in &self.stats {
            output.push_str(&stat.syscall);
            output.push(',');
            output.push_str(&stat.calls.to_string());
            output.push(',');
            output.push_str(&stat.errors.to_string());

            if include_timing {
                output.push(',');
                if let Some(time_us) = stat.total_time_us {
                    output.push_str(&format!("{time_us}us"));
                } else {
                    output.push_str("0us");
                }
            }

            output.push('\n');
        }

        output
    }
}

impl Default for CsvStatsOutput {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_basic_header() {
        let output = CsvOutput::new(false, false);
        assert_eq!(output.header(), "syscall,arguments,result");
    }

    #[test]
    fn test_csv_header_with_timing() {
        let output = CsvOutput::new(true, false);
        assert_eq!(output.header(), "syscall,arguments,result,duration");
    }

    #[test]
    fn test_csv_header_with_source() {
        let output = CsvOutput::new(false, true);
        assert_eq!(output.header(), "syscall,arguments,result,source_location");
    }

    #[test]
    fn test_csv_header_all_flags() {
        let output = CsvOutput::new(true, true);
        assert_eq!(output.header(), "syscall,arguments,result,duration,source_location");
    }

    #[test]
    fn test_csv_escape_field_simple() {
        assert_eq!(CsvOutput::escape_field("hello"), "hello");
    }

    #[test]
    fn test_csv_escape_field_with_comma() {
        assert_eq!(CsvOutput::escape_field("hello,world"), "\"hello,world\"");
    }

    #[test]
    fn test_csv_escape_field_with_quote() {
        assert_eq!(CsvOutput::escape_field("say \"hi\""), "\"say \"\"hi\"\"\"");
    }

    #[test]
    fn test_csv_format_syscall_basic() {
        let output = CsvOutput::new(false, false);
        let syscall = CsvSyscall {
            name: "write".to_string(),
            arguments: "1, \"hello\", 5".to_string(),
            result: 5,
            duration_us: None,
            source_location: None,
        };

        let row = output.format_syscall(&syscall);
        assert_eq!(row, "write,\"1, \"\"hello\"\", 5\",5");
    }

    #[test]
    fn test_csv_format_syscall_with_timing() {
        let output = CsvOutput::new(true, false);
        let syscall = CsvSyscall {
            name: "read".to_string(),
            arguments: "3, buf, 1024".to_string(),
            result: 42,
            duration_us: Some(1500),
            source_location: None,
        };

        let row = output.format_syscall(&syscall);
        assert_eq!(row, "read,\"3, buf, 1024\",42,1500us");
    }

    #[test]
    fn test_csv_format_syscall_with_source() {
        let output = CsvOutput::new(false, true);
        let syscall = CsvSyscall {
            name: "openat".to_string(),
            arguments: "AT_FDCWD, \"/tmp/test\", O_RDONLY".to_string(),
            result: 3,
            duration_us: None,
            source_location: Some("src/main.rs:42".to_string()),
        };

        let row = output.format_syscall(&syscall);
        assert_eq!(row, "openat,\"AT_FDCWD, \"\"/tmp/test\"\", O_RDONLY\",3,src/main.rs:42");
    }

    #[test]
    fn test_csv_to_csv_output() {
        let mut output = CsvOutput::new(false, false);
        output.add_syscall(CsvSyscall {
            name: "write".to_string(),
            arguments: "1, \"test\", 4".to_string(),
            result: 4,
            duration_us: None,
            source_location: None,
        });
        output.add_syscall(CsvSyscall {
            name: "exit_group".to_string(),
            arguments: "0".to_string(),
            result: 0,
            duration_us: None,
            source_location: None,
        });

        let csv = output.to_csv();
        assert!(csv.contains("syscall,arguments,result"));
        assert!(csv.contains("write,\"1, \"\"test\"\", 4\",4"));
        assert!(csv.contains("exit_group,0,0"));
    }

    #[test]
    fn test_csv_stats_basic() {
        let mut stats = CsvStatsOutput::new();
        stats.add_stat(CsvStat {
            syscall: "write".to_string(),
            calls: 5,
            errors: 0,
            total_time_us: None,
        });

        let csv = stats.to_csv(false);
        assert!(csv.contains("syscall,calls,errors"));
        assert!(csv.contains("write,5,0"));
    }

    #[test]
    fn test_csv_stats_with_timing() {
        let mut stats = CsvStatsOutput::new();
        stats.add_stat(CsvStat {
            syscall: "read".to_string(),
            calls: 10,
            errors: 2,
            total_time_us: Some(5000),
        });

        let csv = stats.to_csv(true);
        assert!(csv.contains("syscall,calls,errors,total_time"));
        assert!(csv.contains("read,10,2,5000us"));
    }
}
