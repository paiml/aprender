# Doctest Corpus QA Checklist for Publication

**Version:** 1.0.0
**Status:** Active
**Last Updated:** 2025-11-29
**Applicable To:** CPython stdlib doctest corpus, CITL training datasets

## Executive Summary

This checklist provides 100 verification points for Quality Assurance teams to validate doctest corpora before publication to Hugging Face Hub or integration into Compiler-in-the-Loop (CITL) training pipelines. Each section is grounded in peer-reviewed research on data quality, ML dataset construction, and NLP corpus validation.

---

## Section 1: Schema Integrity (Points 1-12)

*Based on Batini & Scannapieco [1] data quality dimensions framework.*

| # | Check | Pass Criteria | Severity |
|---|-------|---------------|----------|
| 1 | Schema version documented | `schema_version` field present in metadata | Critical |
| 2 | All required columns present | `source`, `version`, `module`, `function`, `input`, `expected` | Critical |
| 3 | Column data types correct | All text columns are UTF-8 strings | Critical |
| 4 | No schema drift from baseline | Schema hash matches registered version | High |
| 5 | Nullable columns documented | `signature` explicitly marked nullable | Medium |
| 6 | Column order consistent | Matches published schema specification | Low |
| 7 | No phantom columns | No undocumented columns present | Medium |
| 8 | Primary key uniqueness | `(module, function, input)` tuple is unique or documented duplicates | High |
| 9 | Foreign key integrity | `module` values match known stdlib modules | Medium |
| 10 | Encoding validation | No invalid UTF-8 sequences | Critical |
| 11 | BOM handling | No byte-order marks in text fields | Medium |
| 12 | Line ending normalization | Consistent `\n` (no `\r\n` contamination) | Medium |

---

## Section 2: Completeness (Points 13-24)

*Based on Pipino et al. [2] information quality assessment methodology.*

| # | Check | Pass Criteria | Severity |
|---|-------|---------------|----------|
| 13 | Minimum corpus size | >= 1,000 doctest examples | High |
| 14 | Module coverage breadth | >= 30 distinct modules | High |
| 15 | Function coverage depth | >= 200 distinct functions | High |
| 16 | Input field non-null rate | 100% (no null inputs) | Critical |
| 17 | Expected field population | >= 60% non-empty expected values | Medium |
| 18 | Source attribution complete | 100% of rows have source identifier | Critical |
| 19 | Version tracking complete | 100% of rows have version string | Critical |
| 20 | Multi-line input coverage | >= 10% of inputs span multiple lines | Medium |
| 21 | Multi-line output coverage | >= 5% of expected outputs span multiple lines | Medium |
| 22 | Exception doctest coverage | >= 20 exception-raising examples | Medium |
| 23 | Class method coverage | >= 50 class/method doctests | Medium |
| 24 | Module-level doctest coverage | >= 5 module-level doctests | Low |

---

## Section 3: Accuracy (Points 25-40)

*Based on Redman [3] data quality improvement principles and Wang & Strong [4] data quality framework.*

| # | Check | Pass Criteria | Severity |
|---|-------|---------------|----------|
| 25 | Prompt prefix correctness | All inputs start with `>>>` or `...` | Critical |
| 26 | Continuation marker validity | `...` only follows `>>>` lines | High |
| 27 | No prose in expected output | Expected field contains no English sentences | Critical |
| 28 | Python literal validity | Expected outputs parse as valid Python literals or exceptions | High |
| 29 | Exception format correctness | Exception outputs match `ErrorType: message` pattern | High |
| 30 | Traceback format validity | Traceback blocks follow Python format | Medium |
| 31 | No prompt leakage in output | Expected field contains no `>>>` markers | Critical |
| 32 | String quote consistency | Quoted strings use valid Python quoting | Medium |
| 33 | Numeric precision preserved | Float outputs maintain source precision | Medium |
| 34 | Collection format validity | Lists/dicts/tuples are syntactically valid | High |
| 35 | Boolean literal correctness | Only `True`/`False` (not `true`/`false`) | High |
| 36 | None literal correctness | Only `None` (not `null`/`nil`) | High |
| 37 | Ellipsis handling | `...` in output marked as doctest directive | Medium |
| 38 | Whitespace preservation | Significant whitespace in outputs preserved | Medium |
| 39 | Unicode correctness | Unicode characters properly encoded | High |
| 40 | Escape sequence validity | `\n`, `\t`, etc. are valid escape sequences | Medium |

---

## Section 4: Consistency (Points 41-52)

*Based on Rahm & Do [5] data cleaning methodology.*

| # | Check | Pass Criteria | Severity |
|---|-------|---------------|----------|
| 41 | Module naming convention | Lowercase with dots (e.g., `os.path`) | High |
| 42 | Function naming convention | snake_case or PascalCase for classes | High |
| 43 | Method naming format | `ClassName.method_name` format | High |
| 44 | Source identifier stability | Single source value per extraction | Medium |
| 45 | Version format consistency | Semantic versioning or git SHA | Medium |
| 46 | Input indentation normalized | Consistent indentation in multi-line inputs | Medium |
| 47 | Output indentation normalized | Consistent indentation in multi-line outputs | Medium |
| 48 | No trailing whitespace | Fields trimmed of trailing spaces | Low |
| 49 | No leading whitespace anomalies | Consistent leading space handling | Low |
| 50 | Duplicate detection | < 5% exact duplicate rows | High |
| 51 | Near-duplicate detection | < 10% semantically equivalent examples | Medium |
| 52 | Cross-module consistency | Same function in different contexts consistent | Medium |

---

## Section 5: Semantic Validity (Points 53-68)

*Based on Gebru et al. [6] Datasheets for Datasets framework and Bender & Friedman [7] data statements.*

| # | Check | Pass Criteria | Severity |
|---|-------|---------------|----------|
| 53 | Input-output semantic alignment | Outputs are plausible for given inputs | Critical |
| 54 | Type inference feasibility | I/O pairs allow type inference | High |
| 55 | No hallucinated outputs | Outputs match actual Python behavior | Critical |
| 56 | Side-effect documentation | Side-effecting examples flagged | Medium |
| 57 | State dependency documentation | Stateful examples documented | Medium |
| 58 | Import statement handling | Required imports documented or included | Medium |
| 59 | Context independence | Examples runnable without hidden setup | High |
| 60 | Error message accuracy | Exception messages match Python version | Medium |
| 61 | Platform-specific handling | OS-specific outputs flagged | Medium |
| 62 | Version-specific behavior | Python version differences documented | High |
| 63 | Deprecated API flagging | Deprecated functions marked | Medium |
| 64 | Security-sensitive filtering | No credential/path exposure | Critical |
| 65 | PII detection | No personally identifiable information | Critical |
| 66 | License compliance | Source license permits redistribution | Critical |
| 67 | Attribution requirements | Proper attribution to CPython/source | High |
| 68 | Temporal validity | Examples valid for target Python version | High |

---

## Section 6: Distribution Quality (Points 69-80)

*Based on Sambasivan et al. [8] "Everyone wants to do the model work" and data quality research.*

| # | Check | Pass Criteria | Severity |
|---|-------|---------------|----------|
| 69 | Module distribution balance | No single module > 20% of corpus | Medium |
| 70 | Function distribution balance | No single function > 5% of corpus | Medium |
| 71 | Input length distribution | Documented mean, median, std dev | Low |
| 72 | Output length distribution | Documented mean, median, std dev | Low |
| 73 | Empty output ratio documented | Percentage of no-output examples | Medium |
| 74 | Exception example ratio | 5-15% exception examples | Medium |
| 75 | Simple vs complex ratio | Mix of single-line and multi-line | Medium |
| 76 | Numeric example coverage | >= 100 numeric computation examples | Medium |
| 77 | String manipulation coverage | >= 100 string operation examples | Medium |
| 78 | Collection operation coverage | >= 100 list/dict/set examples | Medium |
| 79 | Boolean logic coverage | >= 50 boolean expression examples | Low |
| 80 | Control flow inference coverage | Examples that imply conditionals | Low |

---

## Section 7: Provenance & Reproducibility (Points 81-88)

*Based on Hutchinson et al. [9] dataset documentation best practices.*

| # | Check | Pass Criteria | Severity |
|---|-------|---------------|----------|
| 81 | Source commit SHA recorded | Git SHA of source Python version | High |
| 82 | Extraction timestamp recorded | ISO 8601 extraction datetime | High |
| 83 | Extraction tool version | alimentar version documented | High |
| 84 | Extraction parameters logged | CLI arguments/config preserved | Medium |
| 85 | Reproducibility verification | Re-extraction produces identical output | Critical |
| 86 | Checksum validation | SHA-256 of parquet file documented | High |
| 87 | Row count verification | Documented row count matches actual | High |
| 88 | Schema fingerprint | Schema hash for drift detection | Medium |

---

## Section 8: Format & Interoperability (Points 89-96)

*Based on Apache Arrow and Parquet specifications.*

| # | Check | Pass Criteria | Severity |
|---|-------|---------------|----------|
| 89 | Parquet version compatibility | Parquet 2.x format | High |
| 90 | Compression codec documented | ZSTD/Snappy/None specified | Medium |
| 91 | Row group size appropriate | 100K-1M rows per group | Low |
| 92 | Arrow schema embedded | Schema in Parquet metadata | High |
| 93 | HuggingFace compatibility | Loads with `datasets.load_dataset()` | Critical |
| 94 | Pandas compatibility | Loads with `pd.read_parquet()` | High |
| 95 | Polars compatibility | Loads with `pl.read_parquet()` | Medium |
| 96 | DuckDB compatibility | Queryable with DuckDB | Medium |

---

## Section 9: Publication Readiness (Points 97-100)

*Based on Bender et al. [10] on the dangers of stochastic parrots and responsible AI.*

| # | Check | Pass Criteria | Severity |
|---|-------|---------------|----------|
| 97 | Dataset card complete | README.md with full documentation | Critical |
| 98 | License file present | LICENSE file in repository | Critical |
| 99 | Intended use documented | Training purposes clearly stated | High |
| 100 | Limitations documented | Known gaps and biases described | High |

---

## Scoring Rubric

| Grade | Score Range | Publication Decision |
|-------|-------------|---------------------|
| A | 95-100 points | Publish immediately |
| B | 85-94 points | Publish with documented caveats |
| C | 70-84 points | Remediation required before publication |
| D | 50-69 points | Major rework needed |
| F | < 50 points | Do not publish |

**Severity Weights:**
- Critical: 2 points each (failure blocks publication)
- High: 1.5 points each
- Medium: 1 point each
- Low: 0.5 points each

---

## References

[1] Batini, C., & Scannapieco, M. (2016). *Data and Information Quality: Dimensions, Principles and Techniques*. Springer. https://doi.org/10.1007/978-3-319-24106-7

[2] Pipino, L. L., Lee, Y. W., & Wang, R. Y. (2002). Data quality assessment. *Communications of the ACM*, 45(4), 211-218. https://doi.org/10.1145/505248.506010

[3] Redman, T. C. (1996). *Data Quality for the Information Age*. Artech House.

[4] Wang, R. Y., & Strong, D. M. (1996). Beyond accuracy: What data quality means to data consumers. *Journal of Management Information Systems*, 12(4), 5-33. https://doi.org/10.1080/07421222.1996.11518099

[5] Rahm, E., & Do, H. H. (2000). Data cleaning: Problems and current approaches. *IEEE Data Engineering Bulletin*, 23(4), 3-13.

[6] Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86-92. https://doi.org/10.1145/3458723

[7] Bender, E. M., & Friedman, B. (2018). Data statements for natural language processing: Toward mitigating system bias and enabling better science. *Transactions of the Association for Computational Linguistics*, 6, 587-604. https://doi.org/10.1162/tacl_a_00041

[8] Sambasivan, N., Kapania, S., Highfill, H., Akrong, D., Paritosh, P., & Aroyo, L. M. (2021). "Everyone wants to do the model work, not the data work": Data cascades in high-stakes AI. *Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems*, 1-15. https://doi.org/10.1145/3411764.3445518

[9] Hutchinson, B., Smart, A., Hber, A., Paullada, A., Denton, E., & Gebru, T. (2021). Towards accountability for machine learning datasets: Practices from software engineering and infrastructure. *Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency*, 560-575. https://doi.org/10.1145/3442188.3445918

[10] Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the dangers of stochastic parrots: Can language models be too big? *Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency*, 610-623. https://doi.org/10.1145/3442188.3445922

---

## Appendix A: Quick Validation Commands

```bash
# Schema check
alimentar schema /path/to/doctests.parquet

# Quality report
alimentar quality check /path/to/doctests.parquet

# Row count verification
alimentar info /path/to/doctests.parquet

# Sample inspection
alimentar head /path/to/doctests.parquet -n 20

# Checksum generation
sha256sum /path/to/doctests.parquet
```

## Appendix B: Prose Detection Verification (ALIM-R001)

```bash
# Run prose detection example to verify Poka-Yoke
cargo run --example prose_detection --features doctest

# Expected: All test cases pass
# - Prose sentences detected as prose (true)
# - Code outputs not detected as prose (false)
```

## Appendix C: Automated QA Integration

```yaml
# .github/workflows/dataset-qa.yml
name: Dataset QA
on: [push]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run QA Checks
        run: |
          alimentar quality check data/doctests.parquet
          alimentar schema data/doctests.parquet --validate
```
