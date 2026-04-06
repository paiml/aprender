# Plan: Create `databricks-scala-ground-truth-corpus`

## Overview
Create a new ground truth repository at `~/src/databricks-scala-ground-truth-corpus` following the established PAIML corpus conventions. Scala sbt project covering Spark fundamentals, ML/MLflow, Delta Lake, and Structured Streaming. Full PMAT compliance. Push to `paiml/databricks-scala-ground-truth-corpus` on GitHub.

## Step 1: Install Scala toolchain via Coursier
- `curl -fL https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz | gzip -d > cs && chmod +x cs && ./cs setup --yes`
- This installs JDK, sbt, scala, scalafmt in `~/.local/share/coursier`
- Verify: `sbt --version`, `scala --version`, `java --version`

## Step 2: Create repo and scaffold sbt project
- `mkdir ~/src/databricks-scala-ground-truth-corpus && cd` into it
- `git init`
- Create sbt project structure:

```
databricks-scala-ground-truth-corpus/
├── build.sbt                      # Multi-module sbt build
├── project/
│   ├── build.properties           # sbt version
│   └── plugins.sbt                # scoverage, scalafmt, wartremover, sbt-jmh
├── .scalafmt.conf                 # Scalafmt config
├── .scalafix.conf                 # Scalafix rules
├── src/
│   ├── main/scala/com/paiml/databricks/
│   │   ├── spark/                 # Domain 1: Spark fundamentals
│   │   │   ├── DataFrameOps.scala      # DataFrame transformations
│   │   │   ├── SqlOps.scala            # Spark SQL operations
│   │   │   ├── UdfRegistry.scala       # UDF patterns
│   │   │   ├── WindowFunctions.scala   # Window/analytical functions
│   │   │   └── JoinPatterns.scala      # Join strategies
│   │   ├── ml/                    # Domain 2: ML/MLflow
│   │   │   ├── FeatureEngineering.scala   # Feature transforms
│   │   │   ├── PipelineBuilder.scala      # MLlib pipeline construction
│   │   │   ├── ModelEvaluation.scala      # Evaluation metrics
│   │   │   └── HyperparamTuning.scala    # CrossValidator/TrainValidationSplit
│   │   ├── delta/                 # Domain 3: Delta Lake
│   │   │   ├── DeltaTableOps.scala     # CRUD, MERGE, time travel
│   │   │   ├── ChangeDataCapture.scala # CDC patterns
│   │   │   └── SchemaEvolution.scala   # Schema enforcement/evolution
│   │   └── streaming/             # Domain 4: Structured Streaming
│   │       ├── StreamProcessor.scala     # readStream/writeStream
│   │       ├── WindowedAggregation.scala # Tumbling/sliding windows
│   │       └── StreamingJoin.scala       # Stream-stream/stream-static joins
│   └── test/scala/com/paiml/databricks/
│       ├── spark/
│       │   ├── DataFrameOpsSpec.scala
│       │   ├── SqlOpsSpec.scala
│       │   ├── UdfRegistrySpec.scala
│       │   ├── WindowFunctionsSpec.scala
│       │   └── JoinPatternsSpec.scala
│       ├── ml/
│       │   ├── FeatureEngineeringSpec.scala
│       │   ├── PipelineBuilderSpec.scala
│       │   ├── ModelEvaluationSpec.scala
│       │   └── HyperparamTuningSpec.scala
│       ├── delta/
│       │   ├── DeltaTableOpsSpec.scala
│       │   ├── ChangeDataCaptureSpec.scala
│       │   └── SchemaEvolutionSpec.scala
│       └── streaming/
│           ├── StreamProcessorSpec.scala
│           ├── WindowedAggregationSpec.scala
│           └── StreamingJoinSpec.scala
├── oracle/                        # Golden outputs for Popperian falsification
│   ├── spark/
│   ├── ml/
│   ├── delta/
│   └── streaming/
├── specs/                         # Domain specifications
│   ├── spark-fundamentals.md
│   ├── ml-mlflow.md
│   ├── delta-lake.md
│   └── structured-streaming.md
├── book/                          # mdBook documentation
│   ├── book.toml
│   └── src/
│       ├── SUMMARY.md
│       ├── introduction.md
│       ├── spark-fundamentals.md
│       ├── ml-mlflow.md
│       ├── delta-lake.md
│       └── structured-streaming.md
```

## Step 3: Configure build.sbt
- Scala 2.12.x (Spark 3.x compatibility)
- Dependencies: spark-core, spark-sql, spark-mllib, delta-lake, scalatest, scalacheck
- Plugins: sbt-scoverage (95% target), wartremover, scalafmt, sbt-jmh
- Resolver for Delta Lake

## Step 4: Implement source modules (4 domains, 14 files)
Each module implements real Databricks/Spark patterns with:
- Pure functions where possible (testable without SparkSession)
- Builder patterns for pipeline construction
- Case class models for type-safe schemas
- Comprehensive ScalaDoc

## Step 5: Implement test suites (14 spec files)
- ScalaTest FlatSpec + Matchers style
- ScalaCheck property-based testing for pure functions
- SharedSparkSession trait for integration tests
- Oracle golden output comparison where applicable
- Target: 95%+ line coverage

## Step 6: PMAT compliance files
- `pmat.toml` — Quality gates, commit rules, Certeza tiers
- `.pmat/project.toml` — Project metadata
- `CLAUDE.md` — Development guidelines (following TGI-GTC pattern)
- `Makefile` — 4-tier quality gates (fmt, lint, test, coverage, mutants)
- `.gitignore` — Scala/sbt/IDE ignores

## Step 7: Documentation
- `README.md` — Badges, overview, domain structure, quick start, quality standards
- `QA-CHECKLIST.md` — Falsification test checklist
- `LICENSE` — Apache 2.0
- `book/` — mdBook with domain chapters

## Step 8: Create GitHub repo and push
- `gh repo create paiml/databricks-scala-ground-truth-corpus --public --description "Scala ground truth corpus for Databricks course - Spark, ML, Delta Lake, Streaming"`
- Set remote, commit all files, push to main
- Verify with `gh repo view`

## Constraints
- All commits directly to master/main (per CLAUDE.md rules)
- Commit format: `feat|fix|test|docs: message (Refs DSGTC-XXX)`
- 95% test coverage minimum
- Zero scalac warnings
- ScalaCheck property tests for all pure functions
