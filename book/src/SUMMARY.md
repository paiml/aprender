# EXTREME TDD - The Aprender Guide

[Introduction](./introduction.md)

# Core Methodology

- [What is EXTREME TDD?](./methodology/what-is-extreme-tdd.md)
- [The RED-GREEN-REFACTOR Cycle](./methodology/red-green-refactor.md)
- [Test-First Philosophy](./methodology/test-first-philosophy.md)
- [Zero Tolerance Quality](./methodology/zero-tolerance.md)

# The RED Phase

- [Writing Failing Tests First](./red-phase/failing-tests-first.md)
- [Test Categories](./red-phase/test-categories.md)
  - [Unit Tests](./red-phase/unit-tests.md)
  - [Integration Tests](./red-phase/integration-tests.md)
  - [Property-Based Tests](./red-phase/property-based-tests.md)
- [Verification Strategy](./red-phase/verification-strategy.md)

# The GREEN Phase

- [Minimal Implementation](./green-phase/minimal-implementation.md)
- [Making Tests Pass](./green-phase/making-tests-pass.md)
- [Avoiding Over-Engineering](./green-phase/avoiding-over-engineering.md)
- [The Simplest Thing That Works](./green-phase/simplest-thing.md)

# The REFACTOR Phase

- [Refactoring with Confidence](./refactor-phase/refactoring-with-confidence.md)
- [Code Quality Improvements](./refactor-phase/code-quality.md)
- [Performance Optimization](./refactor-phase/performance-optimization.md)
- [Documentation](./refactor-phase/documentation.md)

# Advanced Testing

- [Property-Based Testing](./advanced-testing/property-based-testing.md)
  - [Proptest Fundamentals](./advanced-testing/proptest-fundamentals.md)
  - [Strategies and Generators](./advanced-testing/strategies-generators.md)
  - [Testing Invariants](./advanced-testing/testing-invariants.md)
- [Mutation Testing](./advanced-testing/mutation-testing.md)
  - [What is Mutation Testing?](./advanced-testing/what-is-mutation-testing.md)
  - [Using cargo-mutants](./advanced-testing/using-cargo-mutants.md)
  - [Mutation Score Targets](./advanced-testing/mutation-score-targets.md)
  - [Killing Mutants](./advanced-testing/killing-mutants.md)
- [Fuzzing](./advanced-testing/fuzzing.md)
- [Benchmark Testing](./advanced-testing/benchmark-testing.md)

# Quality Gates

- [Pre-Commit Hooks](./quality-gates/pre-commit-hooks.md)
- [Continuous Integration](./quality-gates/continuous-integration.md)
- [Code Formatting (rustfmt)](./quality-gates/code-formatting.md)
- [Linting (clippy)](./quality-gates/linting-clippy.md)
- [Coverage Measurement](./quality-gates/coverage-measurement.md)
- [Complexity Analysis](./quality-gates/complexity-analysis.md)
- [Technical Debt Gradient (TDG)](./quality-gates/tdg-score.md)

# Toyota Way Principles

- [Overview](./toyota-way/overview.md)
- [Kaizen (Continuous Improvement)](./toyota-way/kaizen.md)
- [Genchi Genbutsu (Go and See)](./toyota-way/genchi-genbutsu.md)
- [Jidoka (Built-in Quality)](./toyota-way/jidoka.md)
- [PDCA Cycle](./toyota-way/pdca-cycle.md)
- [Respect for People](./toyota-way/respect-for-people.md)

# Machine Learning Fundamentals

## Supervised Learning

- [Linear Regression Theory](./ml-fundamentals/linear-regression.md)
- [Regularization Theory](./ml-fundamentals/regularization.md)
- [Logistic Regression Theory](./ml-fundamentals/logistic-regression.md)
- [K-Nearest Neighbors (kNN) Theory](./ml-fundamentals/knn.md)
- [Naive Bayes Theory](./ml-fundamentals/naive-bayes.md)
- [Bayesian Inference Theory](./ml-fundamentals/bayesian-inference.md)
- [Support Vector Machines (SVM) Theory](./ml-fundamentals/svm.md)
- [Decision Trees Theory](./ml-fundamentals/decision-trees.md)
- [Ensemble Methods Theory](./ml-fundamentals/ensemble-methods.md)

## Unsupervised Learning

- [K-Means Clustering Theory](./ml-fundamentals/kmeans-clustering.md)
- [Principal Component Analysis (PCA) Theory](./ml-fundamentals/pca.md)
- [t-SNE (t-Distributed Stochastic Neighbor Embedding) Theory](./ml-fundamentals/tsne.md)

## Model Evaluation

- [Regression Metrics Theory](./ml-fundamentals/regression-metrics.md)
- [Classification Metrics Theory](./ml-fundamentals/classification-metrics.md)
- [Cross-Validation Theory](./ml-fundamentals/cross-validation.md)

## Optimization

- [Gradient Descent Theory](./ml-fundamentals/gradient-descent.md)
- [Advanced Optimizers Theory](./ml-fundamentals/advanced-optimizers.md)
- [Metaheuristics Theory](./ml-fundamentals/metaheuristics.md)

## AutoML

- [AutoML: Automated Machine Learning](./ml-fundamentals/automl.md)

## Learning Paradigms

- [Compiler-in-the-Loop Learning](./ml-fundamentals/compiler-in-the-loop.md)

## Preprocessing

- [Feature Scaling Theory](./ml-fundamentals/feature-scaling.md)

## Graph Algorithms

- [Graph Algorithms Theory](./ml-fundamentals/graph-algorithms.md)
- [Graph Pathfinding Theory](./ml-fundamentals/graph-pathfinding.md)
- [Graph Components and Traversal](./ml-fundamentals/graph-components-traversal.md)
- [Graph Link Prediction and Community Detection](./ml-fundamentals/graph-link-prediction.md)

## Statistics

- [Descriptive Statistics Theory](./ml-fundamentals/descriptive-statistics.md)

## Pattern Mining

- [Apriori Algorithm Theory](./ml-fundamentals/apriori.md)

# Real-World Examples from Aprender

- [Case Study: Linear Regression](./examples/linear-regression.md)
- [Case Study: Boston Housing](./examples/boston-housing.md)
- [Case Study: Cross-Validation](./examples/cross-validation.md)
- [Case Study: Grid Search Hyperparameter Tuning](./examples/grid-search-tuning.md)
- [Case Study: AutoML Clustering (TPE)](./examples/automl-clustering.md)
- [Case Study: Random Forest](./examples/random-forest.md)
- [Case Study: Random Forest Iris](./examples/random-forest-iris.md)
- [Case Study: Decision Tree Iris](./examples/decision-tree-iris.md)
- [Case Study: Model Serialization](./examples/model-serialization.md)
- [Case Study: Model Format (.apr)](./examples/model-format.md)
- [The .apr Format: A Five Whys Deep Dive](./examples/apr-format-deep-dive.md)
- [Case Study: Model Bundling and Memory Paging](./examples/model-bundling-paging.md)
- [Case Study: Tracing Memory Paging with Renacer](./examples/tracing-memory-paging.md)
- [Case Study: Bundle Trace Demo](./examples/bundle-trace-demo.md)
- [Case Study: Synthetic Data Generation](./examples/synthetic-data-generation.md)
- [Case Study: Code-Aware EDA](./examples/code-eda.md)
- [Case Study: Code Feature Extraction](./examples/code-feature-extractor.md)
- [Case Study: KMeans Clustering](./examples/kmeans-clustering.md)
- [Case Study: DBSCAN Clustering](./examples/dbscan-clustering.md)
- [Case Study: Hierarchical Clustering](./examples/hierarchical-clustering.md)
- [Case Study: GMM Clustering](./examples/gmm-clustering.md)
- [Case Study: Iris Clustering](./examples/iris-clustering.md)
- [Case Study: Logistic Regression](./examples/logistic-regression.md)
- [Case Study: KNN Iris](./examples/knn-iris.md)
- [Case Study: Naive Bayes Iris](./examples/naive-bayes-iris.md)
- [Case Study: Beta-Binomial Bayesian Inference](./examples/beta-binomial-inference.md)
- [Case Study: Gamma-Poisson Bayesian Inference](./examples/gamma-poisson-inference.md)
- [Case Study: Normal-InverseGamma Bayesian Inference](./examples/normal-inverse-gamma-inference.md)
- [Case Study: Dirichlet-Multinomial Bayesian Inference](./examples/dirichlet-multinomial-inference.md)
- [Case Study: Bayesian Linear Regression](./examples/bayesian-linear-regression.md)
- [Case Study: Bayesian Logistic Regression](./examples/bayesian-logistic-regression.md)
- [Case Study: Negative Binomial GLM (Overdispersed Counts)](./examples/negative-binomial-glm.md)
- [Case Study: SVM Iris](./examples/svm-iris.md)
- [Case Study: Gradient Boosting Iris](./examples/gbm-iris.md)
- [Case Study: Regularized Regression](./examples/regularized-regression.md)
- [Case Study: Optimizer Demo](./examples/optimizer-demo.md)
- [Case Study: Batch Optimization](./examples/batch-optimization.md)
- [Case Study: Convex Optimization (FISTA + Coordinate Descent)](./examples/convex-optimization.md)
- [Case Study: Constrained Optimization (Projected GD + Augmented Lagrangian + Interior Point)](./examples/constrained-optimization.md)
- [Case Study: ADMM Optimization (Distributed ML + Federated Learning)](./examples/admm-optimization.md)
- [Case Study: Differential Evolution (Metaheuristics)](./examples/differential-evolution.md)
- [Case Study: Metaheuristics Optimization](./examples/metaheuristics-optimization.md)
- [Case Study: Ant Colony Optimization (TSP)](./examples/aco-tsp.md)
- [Case Study: Tabu Search (TSP)](./examples/tabu-tsp.md)
- [Case Study: aprender-tsp Sub-Crate](./examples/tsp-solver-crate.md)
- [Case Study: Predator-Prey Optimization](./examples/predator-prey-optimization.md)
- [Case Study: DataFrame Basics](./examples/dataframe-basics.md)
- [Case Study: Data Preprocessing with Scalers](./examples/data-preprocessing-scalers.md)
- [Case Study: Graph Social Network](./examples/graph-social-network.md)
- [Case Study: Community Detection with Louvain](./examples/community-detection.md)
- [Case Study: Comprehensive Graph Algorithms](./examples/graph-algorithms-comprehensive.md)
- [Case Study: Descriptive Statistics](./examples/descriptive-statistics.md)
- [Case Study: Bayesian Blocks Histogram](./examples/bayesian-blocks-histogram.md)
- [Case Study: PCA Iris](./examples/pca-iris.md)
- [Case Study: Isolation Forest Anomaly Detection](./examples/isolation-forest-anomaly.md)
- [Case Study: Local Outlier Factor (LOF)](./examples/lof-anomaly.md)
- [Case Study: Spectral Clustering](./examples/spectral-clustering.md)
- [Case Study: t-SNE Visualization](./examples/tsne-visualization.md)
- [Case Study: Market Basket Analysis (Apriori)](./examples/market-basket-apriori.md)
- [Case Study: ARIMA Time Series Forecasting](./examples/time-series-forecasting.md)
- [Case Study: Text Preprocessing for NLP](./examples/text-preprocessing.md)
- [Case Study: Text Classification with TF-IDF](./examples/text-classification.md)
- [Case Study: Advanced NLP (Similarity, Entities, Summarization)](./examples/advanced-nlp.md)
- [Case Study: XOR Neural Network (Deep Learning)](./examples/xor-neural-network.md)
- [Case Study: XOR Training](./examples/xor-training.md)
- [Case Study: Neural Network Training Pipeline](./examples/neural-network-training.md)
- [Case Study: Classification Training](./examples/classification-training.md)
- [Case Study: Advanced NLP](./examples/nlp-advanced.md)
- [Case Study: Topic & Sentiment Analysis](./examples/topic-sentiment-analysis.md)
- [Case Study: Content-Based Recommendations](./examples/recommend-content.md)
- [Case Study: AI Shell Completion](./examples/shell-completion.md)
- [Case Study: Shell Completion Benchmarks](./examples/shell-completion-benchmarks.md)
- [Case Study: Publishing Shell Models to HF Hub](./examples/shell-hf-hub-publishing.md)
- [Case Study: Model Encryption Tiers](./examples/shell-encryption-tiers.md)
- [Case Study: Shell Encryption Demo](./examples/shell-encryption-demo.md)
- [Case Study: Shell Model Format](./examples/shell-model-format.md)
- [Case Study: Mixture of Experts (MoE)](./examples/mixture-of-experts.md)
- [Developer's Guide: Shell History Models](./examples/shell-history-developer-guide.md)
- [Building Custom Error Classifiers](./examples/custom-error-classifier.md)
- [Case Study: CITL Automated Program Repair](./examples/citl-automated-repair.md)
- [Case Study: Batuta - Automated Migration to Aprender](./examples/batuta-integration.md)

# Sprint-Based Development

- [Sprint Planning](./sprints/sprint-planning.md)
- [Sprint Execution](./sprints/sprint-execution.md)
- [Sprint Review](./sprints/sprint-review.md)
- [Sprint Retrospective](./sprints/sprint-retrospective.md)
- [Issue Management](./sprints/issue-management.md)

# Anti-Hallucination Enforcement

- [Test-Backed Examples](./anti-hallucination/test-backed-examples.md)
- [Example Verification](./anti-hallucination/example-verification.md)
- [CI Validation](./anti-hallucination/ci-validation.md)
- [Documentation Testing](./anti-hallucination/documentation-testing.md)

# Tools and Setup

- [Development Environment](./tools/development-environment.md)
- [cargo test](./tools/cargo-test.md)
- [cargo clippy](./tools/cargo-clippy.md)
- [cargo fmt](./tools/cargo-fmt.md)
- [cargo mutants](./tools/cargo-mutants.md)
- [proptest](./tools/proptest.md)
- [criterion](./tools/criterion.md)
- [pmat (Toyota AI Toolkit)](./tools/pmat.md)

# Best Practices

- [Error Handling](./best-practices/error-handling.md)
- [API Design](./best-practices/api-design.md)
- [Builder Pattern](./best-practices/builder-pattern.md)
- [Type Safety](./best-practices/type-safety.md)
- [Performance Considerations](./best-practices/performance.md)
- [Documentation Standards](./best-practices/documentation-standards.md)

# Metrics and Measurement

- [Test Coverage](./metrics/test-coverage.md)
- [Mutation Score](./metrics/mutation-score.md)
- [Cyclomatic Complexity](./metrics/cyclomatic-complexity.md)
- [Code Churn](./metrics/code-churn.md)
- [Build Times](./metrics/build-times.md)
- [TDG Score Breakdown](./metrics/tdg-breakdown.md)

# Common Pitfalls

- [Skipping Tests](./pitfalls/skipping-tests.md)
- [Insufficient Test Coverage](./pitfalls/insufficient-coverage.md)
- [Ignoring Warnings](./pitfalls/ignoring-warnings.md)
- [Over-Mocking](./pitfalls/over-mocking.md)
- [Flaky Tests](./pitfalls/flaky-tests.md)
- [Technical Debt Accumulation](./pitfalls/technical-debt.md)

# Appendix

- [Glossary](./appendix/glossary.md)
- [References](./appendix/references.md)
- [Further Reading](./appendix/further-reading.md)
- [Contributing to This Book](./appendix/contributing.md)
