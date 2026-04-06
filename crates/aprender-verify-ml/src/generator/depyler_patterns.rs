//! Depyler-targeted pattern generators
//!
//! Generates Python programs that exercise problematic transpilation patterns
//! identified in depyler:
//! - File/Stdout type unification (File vs Stdout need Box<dyn Write>)
//! - serde_json::Value misuse
//! - Method mapping gaps (file.readlines(), iteration)
//! - Context manager patterns (with open() as f:)

use crate::generator::GeneratedCode;
use crate::Language;

/// Generates Python programs with file I/O patterns
#[derive(Debug, Clone)]
pub struct FileIOPatternGenerator {
    max_depth: usize,
}

impl FileIOPatternGenerator {
    /// Create a new file I/O pattern generator
    #[must_use]
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }

    /// Generate all file I/O patterns
    #[must_use]
    pub fn generate(&self) -> Vec<GeneratedCode> {
        let mut programs = Vec::new();

        // Level 1: Basic stdout/file write patterns
        programs.extend(self.generate_basic_write_patterns());

        // Level 2: File handle passing patterns
        if self.max_depth >= 2 {
            programs.extend(self.generate_handle_passing_patterns());
        }

        // Level 3: Context manager patterns
        if self.max_depth >= 3 {
            programs.extend(self.generate_context_manager_patterns());
        }

        // Level 4: Mixed I/O type patterns (the hardest cases)
        if self.max_depth >= 4 {
            programs.extend(self.generate_mixed_io_patterns());
        }

        programs
    }

    /// Basic write patterns - stdout vs file
    fn generate_basic_write_patterns(&self) -> Vec<GeneratedCode> {
        vec![
            // Pattern 1: sys.stdout.write()
            GeneratedCode {
                code: r#"import sys

def main():
    sys.stdout.write("hello\n")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 1,
                features: vec!["sys_stdout_write".to_string()],
            },
            // Pattern 2: print() function
            GeneratedCode {
                code: r#"def main():
    print("hello")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 1,
                features: vec!["print".to_string()],
            },
            // Pattern 3: file.write()
            GeneratedCode {
                code: r#"def main():
    f = open("output.txt", "w")
    f.write("hello\n")
    f.close()
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 1,
                features: vec!["file_write".to_string()],
            },
            // Pattern 4: sys.stderr.write()
            GeneratedCode {
                code: r#"import sys

def main():
    sys.stderr.write("error\n")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 1,
                features: vec!["sys_stderr_write".to_string()],
            },
        ]
    }

    /// Patterns where file handles are passed to functions
    fn generate_handle_passing_patterns(&self) -> Vec<GeneratedCode> {
        vec![
            // Pattern 5: Function accepting file handle
            GeneratedCode {
                code: r#"def write_to(f, msg: str):
    f.write(msg)

def main():
    f = open("out.txt", "w")
    write_to(f, "hello\n")
    f.close()
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 2,
                features: vec!["file_handle_param".to_string()],
            },
            // Pattern 6: Function accepting stdout
            GeneratedCode {
                code: r#"import sys

def write_to(f, msg: str):
    f.write(msg)

def main():
    write_to(sys.stdout, "hello\n")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 2,
                features: vec!["stdout_handle_param".to_string()],
            },
            // Pattern 7: Function returning file handle
            GeneratedCode {
                code: r#"def get_output():
    return open("out.txt", "w")

def main():
    f = get_output()
    f.write("hello\n")
    f.close()
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 2,
                features: vec!["file_handle_return".to_string()],
            },
            // Pattern 8: Conditional file vs stdout
            GeneratedCode {
                code: r#"import sys

def main():
    use_file = True
    if use_file:
        f = open("out.txt", "w")
    else:
        f = sys.stdout
    f.write("hello\n")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 2,
                features: vec!["conditional_io_type".to_string()],
            },
        ]
    }

    /// Context manager patterns (with statement)
    fn generate_context_manager_patterns(&self) -> Vec<GeneratedCode> {
        vec![
            // Pattern 9: Basic with open
            GeneratedCode {
                code: r#"def main():
    with open("output.txt", "w") as f:
        f.write("hello\n")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 3,
                features: vec!["context_manager_write".to_string()],
            },
            // Pattern 10: with open for reading
            GeneratedCode {
                code: r#"def main():
    with open("input.txt", "r") as f:
        content = f.read()
    print(content)
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 3,
                features: vec!["context_manager_read".to_string()],
            },
            // Pattern 11: Nested with statements
            GeneratedCode {
                code: r#"def main():
    with open("input.txt", "r") as fin:
        with open("output.txt", "w") as fout:
            fout.write(fin.read())
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 3,
                features: vec!["nested_context_manager".to_string()],
            },
            // Pattern 12: Multiple context managers
            GeneratedCode {
                code: r#"def main():
    with open("in.txt", "r") as fin, open("out.txt", "w") as fout:
        fout.write(fin.read())
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 3,
                features: vec!["multiple_context_manager".to_string()],
            },
            // Pattern 13: readlines() iteration
            GeneratedCode {
                code: r#"def main():
    with open("input.txt", "r") as f:
        for line in f.readlines():
            print(line)
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 3,
                features: vec!["readlines_iteration".to_string()],
            },
            // Pattern 14: Direct file iteration
            GeneratedCode {
                code: r#"def main():
    with open("input.txt", "r") as f:
        for line in f:
            print(line)
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 3,
                features: vec!["file_iteration".to_string()],
            },
        ]
    }

    /// Mixed I/O type patterns (most challenging for depyler)
    fn generate_mixed_io_patterns(&self) -> Vec<GeneratedCode> {
        vec![
            // Pattern 15: Function that works with any writable
            GeneratedCode {
                code: r#"import sys

def log_message(output, msg: str):
    output.write(f"[LOG] {msg}\n")

def main():
    log_message(sys.stdout, "to stdout")
    with open("log.txt", "w") as f:
        log_message(f, "to file")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 4,
                features: vec!["polymorphic_writer".to_string()],
            },
            // Pattern 16: List of outputs
            GeneratedCode {
                code: r#"import sys

def main():
    outputs = [sys.stdout, open("out.txt", "w")]
    for out in outputs:
        out.write("hello\n")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 4,
                features: vec!["writer_list".to_string()],
            },
            // Pattern 17: Conditional context manager
            GeneratedCode {
                code: r#"import sys

def get_output(use_file: bool):
    if use_file:
        return open("out.txt", "w")
    return sys.stdout

def main():
    out = get_output(True)
    out.write("hello\n")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 4,
                features: vec!["conditional_output_factory".to_string()],
            },
            // Pattern 18: Streaming filter (like csv_filter example)
            GeneratedCode {
                code: r#"import sys

def filter_lines(input_stream, output_stream, predicate):
    for line in input_stream:
        if predicate(line):
            output_stream.write(line)

def main():
    with open("data.txt", "r") as f:
        filter_lines(f, sys.stdout, lambda x: len(x) > 5)
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 4,
                features: vec!["stream_filter".to_string()],
            },
            // Pattern 19: Log analyzer pattern
            GeneratedCode {
                code: r#"import sys

def analyze_log(log_file: str):
    counts = {}
    with open(log_file, "r") as f:
        for line in f:
            level = line.split()[0] if line.strip() else "UNKNOWN"
            counts[level] = counts.get(level, 0) + 1
    return counts

def main():
    result = analyze_log("app.log")
    for level, count in result.items():
        sys.stdout.write(f"{level}: {count}\n")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 4,
                features: vec!["log_analyzer".to_string()],
            },
            // Pattern 20: I/O streams with type hints
            GeneratedCode {
                code: r#"import sys
from typing import TextIO

def process(input: TextIO, output: TextIO):
    for line in input:
        output.write(line.upper())

def main():
    with open("in.txt", "r") as fin:
        process(fin, sys.stdout)
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 4,
                features: vec!["typed_io_streams".to_string()],
            },
        ]
    }
}

/// Generates Python programs with JSON/dict patterns
#[derive(Debug, Clone)]
pub struct JsonDictPatternGenerator {
    max_depth: usize,
}

impl JsonDictPatternGenerator {
    /// Create a new JSON/dict pattern generator
    #[must_use]
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }

    /// Generate all JSON/dict patterns
    #[must_use]
    pub fn generate(&self) -> Vec<GeneratedCode> {
        let mut programs = Vec::new();

        // Level 1: Basic dict operations
        programs.extend(self.generate_basic_dict_patterns());

        // Level 2: JSON parsing/serialization
        if self.max_depth >= 2 {
            programs.extend(self.generate_json_patterns());
        }

        // Level 3: Nested structures
        if self.max_depth >= 3 {
            programs.extend(self.generate_nested_patterns());
        }

        programs
    }

    /// Basic dictionary patterns
    fn generate_basic_dict_patterns(&self) -> Vec<GeneratedCode> {
        vec![
            // Pattern 1: Dict literal
            GeneratedCode {
                code: r#"def main():
    d = {"key": "value", "count": 42}
    print(d["key"])
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 1,
                features: vec!["dict_literal".to_string()],
            },
            // Pattern 2: Dict get with default
            GeneratedCode {
                code: r#"def main():
    d = {"a": 1}
    val = d.get("b", 0)
    print(val)
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 1,
                features: vec!["dict_get_default".to_string()],
            },
            // Pattern 3: Dict iteration
            GeneratedCode {
                code: r#"def main():
    d = {"a": 1, "b": 2}
    for k, v in d.items():
        print(f"{k}: {v}")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 1,
                features: vec!["dict_items_iteration".to_string()],
            },
            // Pattern 4: Dict comprehension
            GeneratedCode {
                code: r"def main():
    nums = [1, 2, 3]
    d = {str(n): n * n for n in nums}
    print(d)
"
                .to_string(),
                language: Language::Python,
                ast_depth: 1,
                features: vec!["dict_comprehension".to_string()],
            },
        ]
    }

    /// JSON parsing and serialization patterns
    fn generate_json_patterns(&self) -> Vec<GeneratedCode> {
        vec![
            // Pattern 5: json.loads
            GeneratedCode {
                code: r#"import json

def main():
    data = json.loads('{"name": "test", "value": 42}')
    print(data["name"])
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 2,
                features: vec!["json_loads".to_string()],
            },
            // Pattern 6: json.dumps
            GeneratedCode {
                code: r#"import json

def main():
    data = {"name": "test", "value": 42}
    output = json.dumps(data)
    print(output)
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 2,
                features: vec!["json_dumps".to_string()],
            },
            // Pattern 7: json.load from file
            GeneratedCode {
                code: r#"import json

def main():
    with open("data.json", "r") as f:
        data = json.load(f)
    print(data)
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 2,
                features: vec!["json_load_file".to_string()],
            },
            // Pattern 8: json.dump to file
            GeneratedCode {
                code: r#"import json

def main():
    data = {"name": "test", "value": 42}
    with open("output.json", "w") as f:
        json.dump(data, f)
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 2,
                features: vec!["json_dump_file".to_string()],
            },
        ]
    }

    /// Nested dict/JSON patterns
    fn generate_nested_patterns(&self) -> Vec<GeneratedCode> {
        vec![
            // Pattern 9: Nested dict access
            GeneratedCode {
                code: r#"def main():
    data = {"user": {"name": "alice", "age": 30}}
    print(data["user"]["name"])
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 3,
                features: vec!["nested_dict_access".to_string()],
            },
            // Pattern 10: Dict with list values
            GeneratedCode {
                code: r#"def main():
    data = {"items": [1, 2, 3], "tags": ["a", "b"]}
    for item in data["items"]:
        print(item)
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 3,
                features: vec!["dict_list_values".to_string()],
            },
            // Pattern 11: Dynamic key access
            GeneratedCode {
                code: r#"def get_value(d: dict, key: str):
    return d.get(key)

def main():
    data = {"a": 1, "b": 2}
    key = "a"
    print(get_value(data, key))
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 3,
                features: vec!["dynamic_key_access".to_string()],
            },
            // Pattern 12: JSON with mixed types (problematic for serde)
            GeneratedCode {
                code: r#"import json

def main():
    data = json.loads('{"str": "hello", "num": 42, "arr": [1, 2], "obj": {"nested": true}}')
    print(type(data["str"]))
    print(type(data["num"]))
    print(type(data["arr"]))
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 3,
                features: vec!["json_mixed_types".to_string()],
            },
        ]
    }
}

/// Generates Python programs with context manager patterns
#[derive(Debug, Clone)]
pub struct ContextManagerPatternGenerator {
    max_depth: usize,
}

impl ContextManagerPatternGenerator {
    /// Create a new context manager pattern generator
    #[must_use]
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }

    /// Generate all context manager patterns
    #[must_use]
    pub fn generate(&self) -> Vec<GeneratedCode> {
        let mut programs = Vec::new();

        // Level 1: Basic with statements
        programs.extend(self.generate_basic_with_patterns());

        // Level 2: Custom context managers
        if self.max_depth >= 2 {
            programs.extend(self.generate_custom_context_manager_patterns());
        }

        // Level 3: Exception handling in context
        if self.max_depth >= 3 {
            programs.extend(self.generate_exception_patterns());
        }

        programs
    }

    /// Basic with statement patterns
    fn generate_basic_with_patterns(&self) -> Vec<GeneratedCode> {
        vec![
            // Pattern 1: Basic file context
            GeneratedCode {
                code: r#"def main():
    with open("test.txt", "w") as f:
        f.write("hello")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 1,
                features: vec!["basic_file_context".to_string()],
            },
            // Pattern 2: Without as clause
            GeneratedCode {
                code: r#"from contextlib import suppress

def main():
    with suppress(FileNotFoundError):
        with open("missing.txt", "r") as f:
            print(f.read())
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 1,
                features: vec!["suppress_context".to_string()],
            },
            // Pattern 3: Multiple items in single with
            GeneratedCode {
                code: r#"def main():
    with open("a.txt", "r") as a, open("b.txt", "w") as b:
        b.write(a.read())
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 1,
                features: vec!["multiple_with_items".to_string()],
            },
        ]
    }

    /// Custom context manager patterns
    fn generate_custom_context_manager_patterns(&self) -> Vec<GeneratedCode> {
        vec![
            // Pattern 4: Class-based context manager
            GeneratedCode {
                code: r"class Timer:
    def __enter__(self):
        self.start = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = 100
        return False

def main():
    with Timer() as t:
        x = 1 + 1
    print(t.elapsed)
"
                .to_string(),
                language: Language::Python,
                ast_depth: 2,
                features: vec!["class_context_manager".to_string()],
            },
            // Pattern 5: contextlib.contextmanager decorator
            GeneratedCode {
                code: r#"from contextlib import contextmanager

@contextmanager
def managed_resource():
    print("acquiring")
    yield "resource"
    print("releasing")

def main():
    with managed_resource() as r:
        print(r)
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 2,
                features: vec!["contextmanager_decorator".to_string()],
            },
        ]
    }

    /// Exception handling in context managers
    fn generate_exception_patterns(&self) -> Vec<GeneratedCode> {
        vec![
            // Pattern 6: Exception in context body
            GeneratedCode {
                code: r#"def main():
    try:
        with open("test.txt", "w") as f:
            f.write("hello")
            raise ValueError("test error")
    except ValueError as e:
        print(f"caught: {e}")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 3,
                features: vec!["exception_in_context".to_string()],
            },
            // Pattern 7: Context manager that suppresses exceptions
            GeneratedCode {
                code: r#"class Suppressor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True  # Suppress all exceptions

def main():
    with Suppressor():
        raise ValueError("this is suppressed")
    print("continued after exception")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 3,
                features: vec!["exception_suppression".to_string()],
            },
        ]
    }
}

/// Combined generator for all depyler-problematic patterns
#[derive(Debug, Clone)]
pub struct DepylerPatternGenerator {
    max_depth: usize,
}

impl DepylerPatternGenerator {
    /// Create a new combined depyler pattern generator
    #[must_use]
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }

    /// Generate all patterns that exercise depyler problem areas
    #[must_use]
    pub fn generate(&self) -> Vec<GeneratedCode> {
        let mut programs = Vec::new();

        // File I/O patterns
        let file_gen = FileIOPatternGenerator::new(self.max_depth);
        programs.extend(file_gen.generate());

        // JSON/dict patterns
        let json_gen = JsonDictPatternGenerator::new(self.max_depth);
        programs.extend(json_gen.generate());

        // Context manager patterns
        let ctx_gen = ContextManagerPatternGenerator::new(self.max_depth);
        programs.extend(ctx_gen.generate());

        programs
    }

    /// Generate patterns with statistics
    #[must_use]
    pub fn generate_with_stats(&self) -> (Vec<GeneratedCode>, DepylerPatternStats) {
        let file_gen = FileIOPatternGenerator::new(self.max_depth);
        let file_patterns = file_gen.generate();

        let json_gen = JsonDictPatternGenerator::new(self.max_depth);
        let json_patterns = json_gen.generate();

        let ctx_gen = ContextManagerPatternGenerator::new(self.max_depth);
        let ctx_patterns = ctx_gen.generate();

        let stats = DepylerPatternStats {
            file_io_count: file_patterns.len(),
            json_dict_count: json_patterns.len(),
            context_manager_count: ctx_patterns.len(),
            total_count: file_patterns.len() + json_patterns.len() + ctx_patterns.len(),
        };

        let mut programs = file_patterns;
        programs.extend(json_patterns);
        programs.extend(ctx_patterns);

        (programs, stats)
    }
}

/// Statistics about generated depyler patterns
#[derive(Debug, Clone)]
pub struct DepylerPatternStats {
    /// Number of file I/O patterns
    pub file_io_count: usize,
    /// Number of JSON/dict patterns
    pub json_dict_count: usize,
    /// Number of context manager patterns
    pub context_manager_count: usize,
    /// Total pattern count
    pub total_count: usize,
}

/// Advanced pattern generator targeting the 4 remaining depyler issues:
/// 1. Option → Path unwrapping
/// 2. Box<dyn Write> type unification
/// 3. serde_json::Value inference for locals
/// 4. Context manager __enter__() translation
#[derive(Debug, Clone)]
pub struct AdvancedDepylerPatternGenerator {
    max_depth: usize,
}

impl AdvancedDepylerPatternGenerator {
    /// Create a new advanced pattern generator
    #[must_use]
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }

    /// Generate all advanced patterns
    #[must_use]
    pub fn generate(&self) -> Vec<GeneratedCode> {
        let mut programs = Vec::new();

        // Issue 1: Option → Path patterns
        programs.extend(self.generate_option_path_patterns());

        // Issue 2: Box<dyn Write> patterns
        if self.max_depth >= 2 {
            programs.extend(self.generate_trait_object_patterns());
        }

        // Issue 3: serde_json::Value local patterns
        if self.max_depth >= 3 {
            programs.extend(self.generate_dynamic_value_patterns());
        }

        // Issue 4: Context manager __enter__ patterns
        if self.max_depth >= 4 {
            programs.extend(self.generate_enter_exit_patterns());
        }

        programs
    }

    /// Issue 1: Option<String> needs unwrapping before use as path
    fn generate_option_path_patterns(&self) -> Vec<GeneratedCode> {
        vec![
            // Pattern: Optional filename parameter
            GeneratedCode {
                code: r#"def process_file(filename: str = None):
    if filename is None:
        filename = "default.txt"
    with open(filename, "r") as f:
        return f.read()

def main():
    result = process_file()
    print(result)
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 1,
                features: vec!["option_path_default".to_string()],
            },
            // Pattern: Optional from argparse-style
            GeneratedCode {
                code: r#"def get_config_path(override: str = None) -> str:
    if override:
        return override
    return "config.json"

def main():
    path = get_config_path()
    with open(path, "r") as f:
        print(f.read())
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 1,
                features: vec!["option_path_override".to_string()],
            },
            // Pattern: .get() returning optional path
            GeneratedCode {
                code: r#"def main():
    config = {"output": "result.txt"}
    path = config.get("output", "default.txt")
    with open(path, "w") as f:
        f.write("data")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 1,
                features: vec!["option_path_dict_get".to_string()],
            },
            // Pattern: or-expression for path fallback
            GeneratedCode {
                code: r#"import os

def main():
    path = os.environ.get("OUTPUT_FILE") or "output.txt"
    with open(path, "w") as f:
        f.write("result")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 1,
                features: vec!["option_path_or_fallback".to_string()],
            },
        ]
    }

    /// Issue 2: Box<dyn Write> needed for File vs Stdout unification
    fn generate_trait_object_patterns(&self) -> Vec<GeneratedCode> {
        vec![
            // Pattern: Function returns either File or stdout (CRASH case!)
            GeneratedCode {
                code: r#"import sys

def get_writer(use_stdout: bool):
    if use_stdout:
        return sys.stdout
    return open("output.txt", "w")

def main():
    writer = get_writer(True)
    writer.write("hello\n")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 2,
                features: vec!["trait_object_conditional_return".to_string()],
            },
            // Pattern: Ternary returning different write types
            GeneratedCode {
                code: r#"import sys

def main():
    verbose = True
    out = sys.stdout if verbose else open("log.txt", "w")
    out.write("message\n")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 2,
                features: vec!["trait_object_ternary".to_string()],
            },
            // Pattern: Function accepting any writable and called with both
            GeneratedCode {
                code: r#"import sys

def write_report(output, data: str):
    output.write(f"Report: {data}\n")
    output.write("=" * 40 + "\n")

def main():
    write_report(sys.stdout, "summary")
    with open("report.txt", "w") as f:
        write_report(f, "detailed")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 2,
                features: vec!["trait_object_param_both_types".to_string()],
            },
            // Pattern: List of mixed writers
            GeneratedCode {
                code: r#"import sys

def broadcast(writers: list, msg: str):
    for w in writers:
        w.write(msg)

def main():
    with open("log.txt", "w") as f:
        broadcast([sys.stdout, f], "broadcast message\n")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 2,
                features: vec!["trait_object_list_mixed".to_string()],
            },
            // Pattern: Store writer in variable, use conditionally
            GeneratedCode {
                code: r#"import sys

class Logger:
    def __init__(self, filename: str = None):
        if filename:
            self.output = open(filename, "w")
        else:
            self.output = sys.stdout

    def log(self, msg: str):
        self.output.write(f"[LOG] {msg}\n")

def main():
    logger = Logger()
    logger.log("test message")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 2,
                features: vec!["trait_object_class_field".to_string()],
            },
        ]
    }

    /// Issue 3: serde_json::Value inferred for locals that should be typed
    fn generate_dynamic_value_patterns(&self) -> Vec<GeneratedCode> {
        vec![
            // Pattern: Local from json.loads used as dict
            GeneratedCode {
                code: r#"import json

def parse_config(json_str: str) -> str:
    config = json.loads(json_str)
    name = config["name"]
    return name

def main():
    result = parse_config('{"name": "test"}')
    print(result)
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 3,
                features: vec!["value_json_to_dict_field".to_string()],
            },
            // Pattern: json value iteration
            GeneratedCode {
                code: r#"import json

def get_items(json_str: str) -> list:
    data = json.loads(json_str)
    items = data["items"]
    return [item["name"] for item in items]

def main():
    result = get_items('{"items": [{"name": "a"}, {"name": "b"}]}')
    print(result)
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 3,
                features: vec!["value_json_nested_iteration".to_string()],
            },
            // Pattern: Type narrowing from json
            GeneratedCode {
                code: r#"import json

def process_response(json_str: str):
    response = json.loads(json_str)
    if response["status"] == "ok":
        data = response["data"]
        count = data["count"]
        return count
    return 0

def main():
    result = process_response('{"status": "ok", "data": {"count": 42}}')
    print(result)
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 3,
                features: vec!["value_json_conditional_access".to_string()],
            },
            // Pattern: Mixed dict literal and json
            GeneratedCode {
                code: r#"import json

def merge_configs(base: dict, override_json: str) -> dict:
    override = json.loads(override_json)
    result = base.copy()
    result.update(override)
    return result

def main():
    base = {"debug": False, "port": 8080}
    merged = merge_configs(base, '{"debug": true}')
    print(merged["debug"])
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 3,
                features: vec!["value_dict_json_merge".to_string()],
            },
        ]
    }

    /// Issue 4: Context manager __enter__/__exit__ translation
    fn generate_enter_exit_patterns(&self) -> Vec<GeneratedCode> {
        vec![
            // Pattern: __enter__ returns self
            GeneratedCode {
                code: r#"class Connection:
    def __init__(self, host: str):
        self.host = host
        self.connected = False

    def __enter__(self):
        self.connected = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connected = False
        return False

    def query(self, sql: str) -> str:
        return f"Result from {self.host}"

def main():
    with Connection("localhost") as conn:
        result = conn.query("SELECT 1")
        print(result)
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 4,
                features: vec!["enter_returns_self".to_string()],
            },
            // Pattern: __enter__ returns different object
            GeneratedCode {
                code: r#"class FileManager:
    def __init__(self, filename: str):
        self.filename = filename
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, "w")
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False

def main():
    with FileManager("output.txt") as f:
        f.write("managed write")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 4,
                features: vec!["enter_returns_different".to_string()],
            },
            // Pattern: Nested context managers with different __enter__ types
            GeneratedCode {
                code: r#"class Timer:
    def __enter__(self):
        self.start = 0
        return self

    def __exit__(self, *args):
        self.elapsed = 100
        return False

def main():
    with Timer() as t:
        with open("data.txt", "w") as f:
            f.write("timed write")
    print(f"Elapsed: {t.elapsed}")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 4,
                features: vec!["enter_nested_different_types".to_string()],
            },
            // Pattern: __exit__ with exception handling
            GeneratedCode {
                code: r#"class TransactionManager:
    def __init__(self):
        self.committed = False

    def __enter__(self):
        print("Starting transaction")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.committed = True
            print("Committed")
        else:
            print("Rolled back")
        return False

    def execute(self, query: str):
        print(f"Executing: {query}")

def main():
    with TransactionManager() as tx:
        tx.execute("INSERT INTO users VALUES (1)")
"#
                .to_string(),
                language: Language::Python,
                ast_depth: 4,
                features: vec!["exit_exception_handling".to_string()],
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_io_generator_basic() {
        let gen = FileIOPatternGenerator::new(1);
        let programs = gen.generate();
        assert_eq!(programs.len(), 4);
        assert!(programs.iter().all(|p| p.language == Language::Python));
    }

    #[test]
    fn test_file_io_generator_full_depth() {
        let gen = FileIOPatternGenerator::new(4);
        let programs = gen.generate();
        assert_eq!(programs.len(), 20);

        // Check feature coverage
        let features: Vec<_> = programs.iter().flat_map(|p| &p.features).collect();
        assert!(features.iter().any(|f| f.contains("stdout")));
        assert!(features.iter().any(|f| f.contains("context_manager")));
        assert!(features.iter().any(|f| f.contains("polymorphic")));
    }

    #[test]
    fn test_json_dict_generator_basic() {
        let gen = JsonDictPatternGenerator::new(1);
        let programs = gen.generate();
        assert_eq!(programs.len(), 4);
    }

    #[test]
    fn test_json_dict_generator_full_depth() {
        let gen = JsonDictPatternGenerator::new(3);
        let programs = gen.generate();
        assert_eq!(programs.len(), 12);

        let features: Vec<_> = programs.iter().flat_map(|p| &p.features).collect();
        assert!(features.iter().any(|f| f.contains("json_loads")));
        assert!(features.iter().any(|f| f.contains("nested")));
    }

    #[test]
    fn test_context_manager_generator_basic() {
        let gen = ContextManagerPatternGenerator::new(1);
        let programs = gen.generate();
        assert_eq!(programs.len(), 3);
    }

    #[test]
    fn test_context_manager_generator_full_depth() {
        let gen = ContextManagerPatternGenerator::new(3);
        let programs = gen.generate();
        assert_eq!(programs.len(), 7);
    }

    #[test]
    fn test_combined_generator() {
        let gen = DepylerPatternGenerator::new(4);
        let programs = gen.generate();
        assert_eq!(programs.len(), 39); // 20 + 12 + 7
    }

    #[test]
    fn test_combined_generator_with_stats() {
        let gen = DepylerPatternGenerator::new(4);
        let (programs, stats) = gen.generate_with_stats();

        assert_eq!(stats.file_io_count, 20);
        assert_eq!(stats.json_dict_count, 12);
        assert_eq!(stats.context_manager_count, 7);
        assert_eq!(stats.total_count, 39);
        assert_eq!(programs.len(), stats.total_count);
    }

    #[test]
    fn test_patterns_have_valid_python_syntax() {
        let gen = DepylerPatternGenerator::new(4);
        let programs = gen.generate();

        // All patterns should have def main(): or class definition
        for prog in &programs {
            assert!(
                prog.code.contains("def main()") || prog.code.contains("class "),
                "Pattern missing main function: {}",
                prog.code
            );
        }
    }

    #[test]
    fn test_file_io_features_are_tagged() {
        let gen = FileIOPatternGenerator::new(4);
        let programs = gen.generate();

        // All programs should have at least one feature tag
        for prog in &programs {
            assert!(
                !prog.features.is_empty(),
                "Pattern missing features: {}",
                prog.code
            );
        }
    }

    #[test]
    fn test_depth_constraint_respected() {
        // Depth 1 should only give basic patterns
        let gen1 = FileIOPatternGenerator::new(1);
        let programs1 = gen1.generate();
        assert!(programs1.iter().all(|p| p.ast_depth <= 1));

        // Depth 4 should include deep patterns
        let gen4 = FileIOPatternGenerator::new(4);
        let programs4 = gen4.generate();
        assert!(programs4.iter().any(|p| p.ast_depth == 4));
    }

    // Tests for AdvancedDepylerPatternGenerator

    #[test]
    fn test_advanced_generator_option_path() {
        let gen = AdvancedDepylerPatternGenerator::new(1);
        let programs = gen.generate();
        assert_eq!(programs.len(), 4); // 4 option/path patterns

        let features: Vec<_> = programs.iter().flat_map(|p| &p.features).collect();
        assert!(features.iter().any(|f| f.contains("option_path")));
    }

    #[test]
    fn test_advanced_generator_trait_objects() {
        let gen = AdvancedDepylerPatternGenerator::new(2);
        let programs = gen.generate();
        assert_eq!(programs.len(), 9); // 4 option + 5 trait object

        let features: Vec<_> = programs.iter().flat_map(|p| &p.features).collect();
        assert!(features.iter().any(|f| f.contains("trait_object")));
    }

    #[test]
    fn test_advanced_generator_json_value() {
        let gen = AdvancedDepylerPatternGenerator::new(3);
        let programs = gen.generate();
        assert_eq!(programs.len(), 13); // 4 + 5 + 4 json value

        let features: Vec<_> = programs.iter().flat_map(|p| &p.features).collect();
        assert!(features.iter().any(|f| f.contains("value_json")));
    }

    #[test]
    fn test_advanced_generator_full_depth() {
        let gen = AdvancedDepylerPatternGenerator::new(4);
        let programs = gen.generate();
        assert_eq!(programs.len(), 17); // 4 + 5 + 4 + 4 context manager

        let features: Vec<_> = programs.iter().flat_map(|p| &p.features).collect();
        assert!(features.iter().any(|f| f.contains("enter_")));
    }

    #[test]
    fn test_advanced_patterns_target_depyler_issues() {
        let gen = AdvancedDepylerPatternGenerator::new(4);
        let programs = gen.generate();

        // Should have patterns for all 4 issue categories
        let features: Vec<_> = programs.iter().flat_map(|p| &p.features).cloned().collect();

        // Issue 1: Option → Path
        assert!(features.iter().any(|f| f.contains("option_path")));
        // Issue 2: Box<dyn Write>
        assert!(features.iter().any(|f| f.contains("trait_object")));
        // Issue 3: serde_json::Value
        assert!(features.iter().any(|f| f.contains("value_json")));
        // Issue 4: __enter__/__exit__
        assert!(features.iter().any(|f| f.contains("enter_")));
    }

    #[test]
    fn test_advanced_patterns_have_main() {
        let gen = AdvancedDepylerPatternGenerator::new(4);
        let programs = gen.generate();

        for prog in &programs {
            assert!(
                prog.code.contains("def main()") || prog.code.contains("class "),
                "Pattern missing main function or class: {}",
                prog.features.first().unwrap_or(&"unknown".to_string())
            );
        }
    }
}
