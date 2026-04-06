//! Integration tests for Python doctest extraction.
//!
//! These tests define the expected behavior for the doctest parser.
//! Phase 1 (Red): All tests should FAIL until implementation is complete.

#![cfg(feature = "doctest")]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use alimentar::{Dataset, DocTest, DocTestCorpus, DocTestParser};

// =============================================================================
// Basic Doctest Extraction
// =============================================================================

#[test]
fn test_simple_doctest() {
    let source = r#"
def add(a, b):
    """Add two numbers.

    >>> add(1, 2)
    3
    """
    return a + b
"#;

    let parser = DocTestParser::new();
    let doctests = parser.parse_source(source, "math");

    assert_eq!(doctests.len(), 1);
    assert_eq!(doctests[0].module, "math");
    assert_eq!(doctests[0].function, "add");
    assert_eq!(doctests[0].input, ">>> add(1, 2)");
    assert_eq!(doctests[0].expected, "3");
}

#[test]
fn test_multiline_input() {
    let source = r#"
def greet(name):
    """Greet someone.

    >>> greet(
    ...     "World"
    ... )
    'Hello, World!'
    """
    return f"Hello, {name}!"
"#;

    let parser = DocTestParser::new();
    let doctests = parser.parse_source(source, "hello");

    assert_eq!(doctests.len(), 1);
    assert_eq!(doctests[0].input, ">>> greet(\n...     \"World\"\n... )");
    assert_eq!(doctests[0].expected, "'Hello, World!'");
}

#[test]
fn test_multiline_expected_output() {
    let source = r#"
def show_list():
    """Show a list.

    >>> show_list()
    [1,
     2,
     3]
    """
    return [1, 2, 3]
"#;

    let parser = DocTestParser::new();
    let doctests = parser.parse_source(source, "lists");

    assert_eq!(doctests.len(), 1);
    assert_eq!(doctests[0].expected, "[1,\n 2,\n 3]");
}

#[test]
fn test_multiple_doctests_in_one_function() {
    let source = r#"
def calculate(x):
    """Perform calculations.

    >>> calculate(2)
    4
    >>> calculate(3)
    9
    >>> calculate(-1)
    1
    """
    return x * x
"#;

    let parser = DocTestParser::new();
    let doctests = parser.parse_source(source, "calc");

    assert_eq!(doctests.len(), 3);
    assert_eq!(doctests[0].expected, "4");
    assert_eq!(doctests[1].expected, "9");
    assert_eq!(doctests[2].expected, "1");
}

#[test]
fn test_doctest_without_output() {
    let source = r#"
def set_value(x):
    """Set a value (no return).

    >>> set_value(42)
    >>> print("done")
    done
    """
    pass
"#;

    let parser = DocTestParser::new();
    let doctests = parser.parse_source(source, "setter");

    // Should extract both: one with no output, one with output
    assert_eq!(doctests.len(), 2);
    assert_eq!(doctests[0].input, ">>> set_value(42)");
    assert_eq!(doctests[0].expected, "");
    assert_eq!(doctests[1].expected, "done");
}

// =============================================================================
// Class and Method Doctests
// =============================================================================

#[test]
fn test_class_doctest() {
    let source = r#"
class Counter:
    """A simple counter.

    >>> c = Counter()
    >>> c.value
    0
    """
    def __init__(self):
        self.value = 0
"#;

    let parser = DocTestParser::new();
    let doctests = parser.parse_source(source, "counter");

    assert_eq!(doctests.len(), 2);
    assert_eq!(doctests[0].function, "Counter");
}

#[test]
fn test_method_doctest() {
    let source = r#"
class Stack:
    def push(self, item):
        """Push an item onto the stack.

        >>> s = Stack()
        >>> s.push(1)
        >>> s.push(2)
        >>> s.items
        [1, 2]
        """
        self.items.append(item)
"#;

    let parser = DocTestParser::new();
    let doctests = parser.parse_source(source, "stack");

    // Standard Python doctest: each >>> line is a separate test
    assert_eq!(doctests.len(), 4);
    assert_eq!(doctests[0].function, "Stack.push");
    assert_eq!(doctests[0].input, ">>> s = Stack()");
    assert_eq!(doctests[0].expected, "");
    assert_eq!(doctests[3].input, ">>> s.items");
    assert_eq!(doctests[3].expected, "[1, 2]");
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_doctest_with_exception() {
    let source = r#"
def divide(a, b):
    """Divide two numbers.

    >>> divide(10, 2)
    5.0
    >>> divide(1, 0)
    Traceback (most recent call last):
        ...
    ZeroDivisionError: division by zero
    """
    return a / b
"#;

    let parser = DocTestParser::new();
    let doctests = parser.parse_source(source, "math");

    assert_eq!(doctests.len(), 2);
    assert!(doctests[1].expected.contains("ZeroDivisionError"));
}

#[test]
fn test_doctest_with_ellipsis() {
    let source = r#"
def get_object():
    """Get an object.

    >>> get_object()  # doctest: +ELLIPSIS
    <object at 0x...>
    """
    return object()
"#;

    let parser = DocTestParser::new();
    let doctests = parser.parse_source(source, "obj");

    assert_eq!(doctests.len(), 1);
    assert!(doctests[0].input.contains("doctest: +ELLIPSIS"));
}

#[test]
fn test_prompt_inside_string_literal_not_matched() {
    // This is the edge case mentioned in the refinements
    let source = r#"
def show_prompt():
    """Show a prompt example.

    >>> print("Type >>> to continue")
    Type >>> to continue
    """
    pass
"#;

    let parser = DocTestParser::new();
    let doctests = parser.parse_source(source, "prompt");

    // Should only find ONE doctest, not be confused by >>> inside the string
    assert_eq!(doctests.len(), 1);
    assert_eq!(doctests[0].input, ">>> print(\"Type >>> to continue\")");
    assert_eq!(doctests[0].expected, "Type >>> to continue");
}

#[test]
fn test_indented_doctest_in_nested_function() {
    let source = r#"
def outer():
    def inner():
        """Inner function.

        >>> inner()
        42
        """
        return 42
    return inner
"#;

    let parser = DocTestParser::new();
    let doctests = parser.parse_source(source, "nested");

    assert_eq!(doctests.len(), 1);
    assert_eq!(doctests[0].function, "inner");
}

#[test]
fn test_empty_source() {
    let parser = DocTestParser::new();
    let doctests = parser.parse_source("", "empty");
    assert!(doctests.is_empty());
}

#[test]
fn test_source_without_doctests() {
    let source = r#"
def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    # No docstring at all
    return a - b
"#;

    let parser = DocTestParser::new();
    let doctests = parser.parse_source(source, "nodoc");
    assert!(doctests.is_empty());
}

// =============================================================================
// Module-level Doctests
// =============================================================================

#[test]
fn test_module_docstring() {
    let source = r#"
"""Module-level documentation.

>>> import this_module
>>> this_module.VERSION
'1.0.0'
"""

VERSION = '1.0.0'
"#;

    let parser = DocTestParser::new();
    let doctests = parser.parse_source(source, "mymodule");

    assert_eq!(doctests.len(), 2);
    assert_eq!(doctests[0].function, "__module__");
}

// =============================================================================
// Corpus Operations
// =============================================================================

#[test]
fn test_corpus_merge() {
    let mut corpus1 = DocTestCorpus::new("cpython", "3.12.0");
    corpus1.push(DocTest::new("os", "getcwd", ">>> getcwd()", "'/home'"));

    let mut corpus2 = DocTestCorpus::new("cpython", "3.12.0");
    corpus2.push(DocTest::new("sys", "exit", ">>> exit(0)", ""));

    corpus1.merge(corpus2);

    assert_eq!(corpus1.len(), 2);
}

#[test]
fn test_corpus_to_parquet_roundtrip() {
    use alimentar::ArrowDataset;
    use tempfile::tempdir;

    let mut corpus = DocTestCorpus::new("numpy", "1.26.0");
    corpus.push(DocTest::new(
        "numpy.core",
        "array",
        ">>> array([1,2,3])",
        "array([1, 2, 3])",
    ));

    let dataset = corpus.to_dataset().expect("should create dataset");
    assert_eq!(dataset.len(), 1);

    // Write to parquet and read back
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("doctests.parquet");
    dataset.to_parquet(&path).expect("should write parquet");

    let loaded = ArrowDataset::from_parquet(&path).expect("should read parquet");
    assert_eq!(loaded.len(), 1);
}
