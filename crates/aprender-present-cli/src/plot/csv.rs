//! Table reader for the `plot` verb's data input: RFC 4180 CSV with `#` comment lines.
//!
//! PMAT-6917 (contract `present-plot-spec-v1`). Pure Rust, no dependencies.
//!
//! Rules, all pinned by tests:
//! * A line whose FIRST character is `#` is skipped, before and after the header.
//!   A `#` inside a record (or inside a quoted field) is data.
//! * A completely empty line is skipped. A one-column empty cell must be written `""`.
//! * The first record is the header; header names must be unique.
//! * Quoting follows RFC 4180: `"a,b"`, doubled quotes `""`, quoted newlines.
//!   `\n` and `\r\n` both end a record; a leading UTF-8 byte-order mark is dropped.
//! * A column is [`Values::Numeric`] iff every non-empty cell parses as a finite `f64`
//!   (cells are NOT trimmed: `" 3"` is text). An empty cell is `None`. Otherwise the
//!   column is [`Values::Text`]. A header-only table has empty `Numeric` columns.
//!
//! Total: no input panics; every failure is a [`CsvError`] naming the 1-based source line.

use std::fmt;

/// One typed column.
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    pub name: String,
    pub values: Values,
}

/// Column contents.
#[derive(Debug, Clone, PartialEq)]
pub enum Values {
    /// `None` is an empty cell.
    Numeric(Vec<Option<f64>>),
    Text(Vec<String>),
}

impl Values {
    fn len(&self) -> usize {
        match self {
            Self::Numeric(v) => v.len(),
            Self::Text(v) => v.len(),
        }
    }
}

/// A parsed table: every column has the same length.
#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    pub columns: Vec<Column>,
}

impl Table {
    /// The column called `name`, if any.
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// Number of data rows (header excluded).
    pub fn row_count(&self) -> usize {
        self.columns.first().map_or(0, |c| c.values.len())
    }
}

/// Why a table was rejected. Every variant names the 1-based source line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CsvError {
    /// No header row: the input is empty or holds only comments / blank lines.
    /// `line` is the line the input ended on.
    NoHeader { line: usize },
    /// A record with a different number of fields than the header.
    Ragged {
        line: usize,
        expected: usize,
        got: usize,
    },
    /// Two header cells with the same name.
    DuplicateColumn { line: usize, name: String },
    /// A quoted field that never closes; `line` is where it opened.
    UnterminatedQuote { line: usize },
    /// Text right after a closing quote (`"a"b`).
    StrayQuote { line: usize },
}

impl fmt::Display for CsvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoHeader { line } => write!(
                f,
                "line {line}: no header row (the input is empty or only comments)"
            ),
            Self::Ragged {
                line,
                expected,
                got,
            } => write!(f, "line {line}: expected {expected} fields, got {got}"),
            Self::DuplicateColumn { line, name } => {
                write!(f, "line {line}: duplicate column name `{name}`")
            }
            Self::UnterminatedQuote { line } => {
                write!(f, "line {line}: unterminated quoted field")
            }
            Self::StrayQuote { line } => {
                write!(f, "line {line}: unexpected text after a closing quote")
            }
        }
    }
}

impl std::error::Error for CsvError {}

struct Reader {
    cs: Vec<char>,
    i: usize,
    line: usize,
}

impl Reader {
    fn peek(&self, n: usize) -> Option<char> {
        self.cs.get(self.i + n).copied()
    }

    fn at_eol(&self) -> bool {
        self.peek(0) == Some('\n') || (self.peek(0) == Some('\r') && self.peek(1) == Some('\n'))
    }

    fn eat_eol(&mut self) {
        self.i += if self.peek(0) == Some('\r') { 2 } else { 1 };
        self.line += 1;
    }

    /// Skip comment lines and blank lines; true when a record starts here.
    fn seek_record(&mut self) -> bool {
        loop {
            if self.at_eol() {
                self.eat_eol();
            } else if self.peek(0) == Some('#') {
                while self.peek(0).is_some() && !self.at_eol() {
                    self.i += 1;
                }
                if self.peek(0).is_some() {
                    self.eat_eol();
                }
            } else {
                return self.peek(0).is_some();
            }
        }
    }

    fn quoted(&mut self) -> Result<String, CsvError> {
        let opened = self.line;
        self.i += 1;
        let mut out = String::new();
        loop {
            match self.peek(0) {
                None => return Err(CsvError::UnterminatedQuote { line: opened }),
                Some('"') if self.peek(1) == Some('"') => {
                    out.push('"');
                    self.i += 2;
                }
                Some('"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(c) => {
                    self.line += usize::from(c == '\n');
                    out.push(c);
                    self.i += 1;
                }
            }
        }
    }

    fn field(&mut self) -> Result<String, CsvError> {
        if self.peek(0) == Some('"') {
            let out = self.quoted()?;
            if self.peek(0).is_some() && self.peek(0) != Some(',') && !self.at_eol() {
                return Err(CsvError::StrayQuote { line: self.line });
            }
            return Ok(out);
        }
        let mut out = String::new();
        while let Some(c) = self.peek(0).filter(|c| *c != ',') {
            if self.at_eol() {
                break;
            }
            out.push(c);
            self.i += 1;
        }
        Ok(out)
    }

    /// One record, starting on a non-skipped line; consumes its end of line.
    fn record(&mut self) -> Result<Vec<String>, CsvError> {
        let mut fields = vec![self.field()?];
        while self.peek(0) == Some(',') {
            self.i += 1;
            fields.push(self.field()?);
        }
        if self.at_eol() {
            self.eat_eol();
        }
        Ok(fields)
    }
}

fn typed(name: String, cells: Vec<String>) -> Column {
    let numbers: Option<Vec<Option<f64>>> = cells
        .iter()
        .map(|c| {
            if c.is_empty() {
                Some(None)
            } else {
                c.parse::<f64>().ok().filter(|v| v.is_finite()).map(Some)
            }
        })
        .collect();
    let values = match numbers {
        Some(v) => Values::Numeric(v),
        None => Values::Text(cells),
    };
    Column { name, values }
}

/// Read a table from CSV text. See the module docs for the exact rules.
pub fn read_table(text: &str) -> Result<Table, CsvError> {
    let mut r = Reader {
        cs: text
            .strip_prefix('\u{feff}')
            .unwrap_or(text)
            .chars()
            .collect(),
        i: 0,
        line: 1,
    };
    if !r.seek_record() {
        return Err(CsvError::NoHeader { line: r.line });
    }
    let header_line = r.line;
    let header = r.record()?;
    for (i, name) in header.iter().enumerate() {
        if header[..i].contains(name) {
            return Err(CsvError::DuplicateColumn {
                line: header_line,
                name: name.clone(),
            });
        }
    }
    let mut cells: Vec<Vec<String>> = vec![Vec::new(); header.len()];
    while r.seek_record() {
        let line = r.line;
        let row = r.record()?;
        if row.len() != header.len() {
            return Err(CsvError::Ragged {
                line,
                expected: header.len(),
                got: row.len(),
            });
        }
        for (col, cell) in cells.iter_mut().zip(row) {
            col.push(cell);
        }
    }
    Ok(Table {
        columns: header
            .into_iter()
            .zip(cells)
            .map(|(n, c)| typed(n, c))
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn ok(text: &str) -> Table {
        read_table(text).expect("the table reads")
    }

    fn err(text: &str) -> CsvError {
        read_table(text).expect_err("the table is rejected")
    }

    fn num(v: &[Option<f64>]) -> Values {
        Values::Numeric(v.to_vec())
    }

    fn text(v: &[&str]) -> Values {
        Values::Text(v.iter().map(|s| (*s).to_string()).collect())
    }

    // ── typing rule ──

    #[test]
    fn a_column_is_numeric_iff_every_nonempty_cell_is_a_finite_number() {
        let t = ok("n,t,m,e\n1,a,1,\n2.5,b,x,\n-3e2,c,,\n");
        assert_eq!(
            t.column("n").map(|c| c.values.clone()),
            Some(num(&[Some(1.0), Some(2.5), Some(-300.0)]))
        );
        assert_eq!(
            t.column("t").map(|c| c.values.clone()),
            Some(text(&["a", "b", "c"]))
        );
        assert_eq!(
            t.column("m").map(|c| c.values.clone()),
            Some(text(&["1", "x", ""])),
            "one non-number makes the column text"
        );
        assert_eq!(
            t.column("e").map(|c| c.values.clone()),
            Some(num(&[None, None, None])),
            "all-empty is numeric"
        );
        assert_eq!(t.row_count(), 3);
        assert_eq!(t.column("missing"), None);
    }

    #[test]
    fn empty_cells_are_none_in_numeric_columns() {
        let t = ok("x\n1\n\"\"\n3\n");
        assert_eq!(t.columns[0].values, num(&[Some(1.0), None, Some(3.0)]));
    }

    #[test]
    fn non_finite_and_untrimmed_cells_are_text() {
        for cell in [
            "NaN", "inf", "-inf", "Infinity", "1e999", " 3", "3 ", "1,5", "0x10", "--1",
        ] {
            let t = ok(&format!("x\n1\n\"{cell}\"\n"));
            assert!(
                matches!(t.columns[0].values, Values::Text(_)),
                "{cell:?} must make the column text"
            );
        }
        for cell in ["+5", ".5", "5.", "-0", "1E3", "00012"] {
            let t = ok(&format!("x\n{cell}\n"));
            assert!(
                matches!(t.columns[0].values, Values::Numeric(_)),
                "{cell:?} must be numeric"
            );
        }
    }

    #[test]
    fn a_header_only_table_has_empty_numeric_columns() {
        let t = ok("a,b\n");
        assert_eq!(t.row_count(), 0);
        assert_eq!(t.columns.len(), 2);
        assert_eq!(t.columns[0].values, num(&[]));
    }

    // ── (4) comment lines ──

    #[test]
    fn hash_lines_are_skipped_before_and_after_the_header() {
        let t = ok("# title\n# more\na,b\n1,2\n# in the middle\n3,4\n#\n# tail");
        assert_eq!(t.columns[0].values, num(&[Some(1.0), Some(3.0)]));
        assert_eq!(t.columns[1].values, num(&[Some(2.0), Some(4.0)]));
        assert_eq!(
            ok("# c\r\na\r\n1\r\n# c\r\n2\r\n").columns[0].values,
            num(&[Some(1.0), Some(2.0)])
        );
    }

    #[test]
    fn a_hash_that_is_not_first_on_its_line_is_data() {
        let t = ok("a,b\n x,#y\n#z,w\n\"#q\",1\n");
        assert_eq!(t.columns[0].values, text(&[" x", "#q"]));
        assert_eq!(t.columns[1].values, text(&["#y", "1"]));
    }

    #[test]
    fn a_hash_inside_a_quoted_field_is_data_even_at_line_start() {
        let t = ok("a\n\"line one\n#not a comment\"\n");
        assert_eq!(t.columns[0].values, text(&["line one\n#not a comment"]));
    }

    // ── RFC 4180 ──

    #[test]
    fn quoting_commas_doubled_quotes_and_newlines() {
        let t = ok("a,b,c\n\"x,y\",\"say \"\"hi\"\"\",\"two\nlines\"\n");
        assert_eq!(t.columns[0].values, text(&["x,y"]));
        assert_eq!(t.columns[1].values, text(&["say \"hi\""]));
        assert_eq!(t.columns[2].values, text(&["two\nlines"]));
    }

    #[test]
    fn crlf_lf_and_missing_final_newline_agree() {
        let want = ok("a,b\n1,2\n3,4\n");
        assert_eq!(ok("a,b\r\n1,2\r\n3,4\r\n"), want);
        assert_eq!(ok("a,b\n1,2\n3,4"), want);
        assert_eq!(ok("a,b\r\n1,2\r\n3,4"), want);
        assert_eq!(ok("\u{feff}a,b\n1,2\n3,4\n"), want, "BOM dropped");
    }

    #[test]
    fn blank_lines_are_skipped_and_empty_trailing_fields_are_kept() {
        let t = ok("\n\na,b\n\n1,\n\r\n,2\n");
        assert_eq!(t.columns[0].values, num(&[Some(1.0), None]));
        assert_eq!(t.columns[1].values, num(&[None, Some(2.0)]));
    }

    #[test]
    fn a_lone_carriage_return_is_data() {
        let t = ok("a\nx\ry\n");
        assert_eq!(t.columns[0].values, text(&["x\ry"]));
    }

    #[test]
    fn a_quote_in_the_middle_of_an_unquoted_field_is_data() {
        let t = ok("a\nab\"c\n");
        assert_eq!(t.columns[0].values, text(&["ab\"c"]));
    }

    // ── errors name cause and 1-based line ──

    #[test]
    fn no_header_reports_the_line_the_input_ended_on() {
        assert_eq!(err(""), CsvError::NoHeader { line: 1 });
        assert_eq!(err("# only\n# comments\n"), CsvError::NoHeader { line: 3 });
        assert_eq!(err("\n\n"), CsvError::NoHeader { line: 3 });
        assert_eq!(err("# no newline"), CsvError::NoHeader { line: 1 });
        assert!(err("").to_string().contains("no header row"));
        assert!(err("").to_string().starts_with("line 1:"));
    }

    #[test]
    fn ragged_rows_name_the_line_and_both_counts() {
        let e = err("a,b\n1,2\n3\n");
        assert_eq!(
            e,
            CsvError::Ragged {
                line: 3,
                expected: 2,
                got: 1
            }
        );
        assert!(
            e.to_string().contains("line 3") && e.to_string().contains("expected 2 fields, got 1"),
            "{e}"
        );
        assert_eq!(
            err("# c\na,b\n1,2,3\n"),
            CsvError::Ragged {
                line: 3,
                expected: 2,
                got: 3
            }
        );
        // a quoted newline moves the line counter: the bad row starts on line 4
        assert_eq!(
            err("a,b\n\"x\ny\",1\n2\n"),
            CsvError::Ragged {
                line: 4,
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn duplicate_column_names_name_the_header_line() {
        let e = err("# c\nx,y,x\n1,2,3\n");
        assert_eq!(
            e,
            CsvError::DuplicateColumn {
                line: 2,
                name: "x".into()
            }
        );
        assert!(
            e.to_string().contains("line 2") && e.to_string().contains("`x`"),
            "{e}"
        );
        assert_eq!(
            err(",a,\n"),
            CsvError::DuplicateColumn {
                line: 1,
                name: String::new()
            }
        );
    }

    #[test]
    fn unterminated_quotes_name_the_opening_line() {
        let e = err("a\n1\n\"never closed\nmore\n");
        assert_eq!(e, CsvError::UnterminatedQuote { line: 3 });
        assert!(
            e.to_string().contains("line 3") && e.to_string().contains("unterminated"),
            "{e}"
        );
        assert_eq!(err("\"a"), CsvError::UnterminatedQuote { line: 1 });
        assert_eq!(err("a\n\"x\"\"\n"), CsvError::UnterminatedQuote { line: 2 });
    }

    #[test]
    fn text_after_a_closing_quote_is_a_stray_quote() {
        let e = err("a,b\n1,2\n\"x\"y,3\n");
        assert_eq!(e, CsvError::StrayQuote { line: 3 });
        assert!(
            e.to_string().contains("line 3") && e.to_string().contains("closing quote"),
            "{e}"
        );
        assert_eq!(err("\"x\" ,b\n"), CsvError::StrayQuote { line: 1 });
        // a quote closing on a later line reports that line
        assert_eq!(err("a\n\"x\ny\"z\n"), CsvError::StrayQuote { line: 3 });
        let _: &dyn std::error::Error = &e;
    }

    // ── (5) no panic, and a writer/reader round trip ──

    fn write_field(out: &mut String, cell: &str) {
        if cell.is_empty() || cell.contains([',', '"', '\n', '\r']) || cell.starts_with('#') {
            out.push('"');
            out.push_str(&cell.replace('"', "\"\""));
            out.push('"');
        } else {
            out.push_str(cell);
        }
    }

    fn write_table(header: &[String], rows: &[Vec<String>], eol: &str) -> String {
        let mut out = String::new();
        for row in std::iter::once(header).chain(rows.iter().map(Vec::as_slice)) {
            for (i, cell) in row.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_field(&mut out, cell);
            }
            out.push_str(eol);
        }
        out
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn read_table_never_panics_on_arbitrary_strings(src in any::<String>()) {
            let _ = read_table(&src);
        }

        #[test]
        fn read_table_never_panics_on_arbitrary_bytes(bytes in proptest::collection::vec(any::<u8>(), 0..200)) {
            let _ = read_table(&String::from_utf8_lossy(&bytes));
        }

        #[test]
        fn read_table_never_panics_on_csv_soup(src in "[a-c0-9,\"\n\r# .-]{0,120}") {
            match read_table(&src) {
                Ok(t) => {
                    let n = t.row_count();
                    prop_assert!(t.columns.iter().all(|c| c.values.len() == n));
                }
                Err(e) => {
                    let line = match e {
                        CsvError::NoHeader { line }
                        | CsvError::Ragged { line, .. }
                        | CsvError::DuplicateColumn { line, .. }
                        | CsvError::UnterminatedQuote { line }
                        | CsvError::StrayQuote { line } => line,
                    };
                    prop_assert!(line >= 1);
                    let prefix = format!("line {line}:");
                    prop_assert!(e.to_string().starts_with(&prefix));
                }
            }
        }

        #[test]
        fn a_written_table_reads_back_cell_for_cell(
            ncols in 1usize..5,
            raw in proptest::collection::vec(proptest::collection::vec(any::<String>(), 4), 0..6),
            crlf in any::<bool>(),
        ) {
            let header: Vec<String> = (0..ncols).map(|i| format!("c{i}")).collect();
            let rows: Vec<Vec<String>> = raw.iter().map(|r| r[..ncols].to_vec()).collect();
            let src = write_table(&header, &rows, if crlf { "\r\n" } else { "\n" });
            let t = read_table(&src).expect("a written table reads back");
            prop_assert_eq!(t.row_count(), rows.len());
            for (j, col) in t.columns.iter().enumerate() {
                prop_assert_eq!(&col.name, &header[j]);
                let cells: Vec<String> = rows.iter().map(|r| r[j].clone()).collect();
                match &col.values {
                    Values::Text(v) => prop_assert_eq!(v, &cells),
                    Values::Numeric(v) => {
                        for (got, cell) in v.iter().zip(&cells) {
                            match got {
                                None => prop_assert!(cell.is_empty()),
                                Some(x) => prop_assert_eq!(cell.parse::<f64>().ok(), Some(*x)),
                            }
                        }
                    }
                }
            }
        }
    }
}
