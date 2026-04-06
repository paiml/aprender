#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::doc_markdown
)]
//! Python Doctest Extraction Example
//!
//! Demonstrates extracting Python doctests from source code and converting
//! them to Arrow/Parquet format for ML training data.
//!
//! Run with: cargo run --example doctest_extraction --features doctest

use std::path::Path;

use alimentar::{Dataset, DocTest, DocTestCorpus, DocTestParser};

fn main() -> alimentar::Result<()> {
    println!("=== Alimentar Doctest Extraction Example ===\n");

    // 1. Parse doctests from a Python source string
    println!("1. Parsing doctests from source string");
    println!("   ─────────────────────────────────────");

    let python_source = r#"
def factorial(n):
    """Calculate factorial of n.

    >>> factorial(0)
    1
    >>> factorial(1)
    1
    >>> factorial(5)
    120
    >>> factorial(10)
    3628800
    """
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def fibonacci(n):
    """Return the nth Fibonacci number.

    >>> fibonacci(0)
    0
    >>> fibonacci(1)
    1
    >>> fibonacci(10)
    55
    >>> [fibonacci(i) for i in range(8)]
    [0, 1, 1, 2, 3, 5, 8, 13]
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)


class Calculator:
    """A simple calculator.

    >>> calc = Calculator()
    >>> calc.add(2, 3)
    5
    >>> calc.multiply(4, 5)
    20
    """

    def add(self, a, b):
        """Add two numbers.

        >>> Calculator().add(1, 2)
        3
        >>> Calculator().add(-1, 1)
        0
        """
        return a + b

    def multiply(self, a, b):
        """Multiply two numbers.

        >>> Calculator().multiply(3, 4)
        12
        >>> Calculator().multiply(0, 100)
        0
        """
        return a * b
"#;

    let parser = DocTestParser::new();
    let doctests = parser.parse_source(python_source, "math_utils");

    println!("   Found {} doctests:\n", doctests.len());

    for (i, dt) in doctests.iter().enumerate() {
        println!("   [{}] {}::{}", i + 1, dt.module, dt.function);
        println!(
            "       Input:    {}",
            dt.input.replace('\n', "\n                 ")
        );
        println!(
            "       Expected: {}",
            if dt.expected.is_empty() {
                "(no output)"
            } else {
                &dt.expected
            }
        );
        println!();
    }

    // 2. Create a corpus and convert to Arrow
    println!("\n2. Creating DocTestCorpus");
    println!("   ──────────────────────");

    let mut corpus = DocTestCorpus::new("example", "1.0.0");

    // Add doctests from parsing
    for dt in doctests {
        corpus.push(dt);
    }

    // Add some manual doctests
    corpus.push(DocTest::new("builtins", "len", ">>> len([1, 2, 3])", "3"));

    corpus.push(DocTest::new(
        "builtins",
        "sorted",
        ">>> sorted([3, 1, 2])",
        "[1, 2, 3]",
    ));

    println!("   Corpus source:  {}", corpus.source);
    println!("   Corpus version: {}", corpus.version);
    println!("   Total doctests: {}", corpus.len());

    // 3. Convert to Arrow RecordBatch
    println!("\n3. Converting to Arrow RecordBatch");
    println!("   ────────────────────────────────");

    let batch = corpus.to_record_batch()?;
    println!("   Rows:    {}", batch.num_rows());
    println!("   Columns: {}", batch.num_columns());
    println!("   Schema:");
    for field in batch.schema().fields() {
        println!(
            "     - {}: {} (nullable: {})",
            field.name(),
            field.data_type(),
            field.is_nullable()
        );
    }

    // 4. Convert to ArrowDataset and demonstrate operations
    println!("\n4. Working with ArrowDataset");
    println!("   ──────────────────────────");

    let dataset = corpus.to_dataset()?;
    println!("   Dataset length: {} rows", dataset.len());

    // Iterate over batches
    println!("   Iterating over batches:");
    for (i, batch) in dataset.iter().enumerate() {
        println!("     Batch {}: {} rows", i, batch.num_rows());
    }

    // 5. Save to Parquet (using temp directory)
    println!("\n5. Saving to Parquet");
    println!("   ──────────────────");

    let temp_dir = std::env::temp_dir();
    let parquet_path = temp_dir.join("example_doctests.parquet");

    dataset.to_parquet(&parquet_path)?;
    println!("   Saved to: {}", parquet_path.display());

    // Get file size
    let metadata = std::fs::metadata(&parquet_path).expect("read metadata");
    println!("   File size: {} bytes", metadata.len());

    // 6. Load back from Parquet
    println!("\n6. Loading from Parquet");
    println!("   ─────────────────────");

    let loaded = alimentar::ArrowDataset::from_parquet(&parquet_path)?;
    println!("   Loaded {} rows", loaded.len());
    println!("   Schema matches: {}", loaded.schema() == dataset.schema());

    // 7. Demonstrate merging corpora
    println!("\n7. Merging Corpora");
    println!("   ────────────────");

    let mut corpus_a = DocTestCorpus::new("source_a", "1.0");
    corpus_a.push(DocTest::new("mod_a", "func_a", ">>> a()", "1"));
    corpus_a.push(DocTest::new("mod_a", "func_b", ">>> b()", "2"));

    let mut corpus_b = DocTestCorpus::new("source_b", "2.0");
    corpus_b.push(DocTest::new("mod_b", "func_c", ">>> c()", "3"));
    corpus_b.push(DocTest::new("mod_b", "func_d", ">>> d()", "4"));

    println!("   Corpus A: {} doctests", corpus_a.len());
    println!("   Corpus B: {} doctests", corpus_b.len());

    corpus_a.merge(corpus_b);
    println!("   After merge: {} doctests", corpus_a.len());

    // 8. Summary statistics
    println!("\n8. Corpus Statistics");
    println!("   ──────────────────");

    // Count by function
    let mut by_function: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for dt in &corpus.doctests {
        *by_function.entry(&dt.function).or_insert(0) += 1;
    }

    println!("   Doctests by function:");
    let mut counts: Vec<_> = by_function.iter().collect();
    counts.sort_by(|a, b| b.1.cmp(a.1));
    for (func, count) in counts.iter().take(5) {
        println!("     {}: {}", func, count);
    }

    // Count with/without expected output
    let with_output = corpus
        .doctests
        .iter()
        .filter(|dt| !dt.expected.is_empty())
        .count();
    let without_output = corpus.len() - with_output;
    println!("\n   With expected output:    {}", with_output);
    println!("   Without expected output: {}", without_output);

    // Cleanup
    let _ = std::fs::remove_file(&parquet_path);

    println!("\n=== Example Complete ===");
    Ok(())
}

/// Example: Parse doctests from a file (not run, just for documentation)
#[allow(dead_code)]
fn parse_from_file_example() -> alimentar::Result<()> {
    let parser = DocTestParser::new();

    // Parse a single file
    let doctests = parser.parse_file(Path::new("example.py"), "example")?;
    println!("Found {} doctests", doctests.len());

    Ok(())
}

/// Example: Parse doctests from a directory (not run, just for documentation)
#[allow(dead_code)]
fn parse_from_directory_example() -> alimentar::Result<()> {
    let parser = DocTestParser::new();

    // Parse entire Python project
    let corpus =
        parser.parse_directory(Path::new("/path/to/python/project"), "myproject", "v1.0.0")?;

    println!("Extracted {} doctests from myproject", corpus.len());

    // Save to parquet
    corpus
        .to_dataset()?
        .to_parquet(Path::new("myproject_doctests.parquet"))?;

    Ok(())
}
