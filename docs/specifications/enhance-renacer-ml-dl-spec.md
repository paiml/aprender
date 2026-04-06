# Enhanced ML/DL Integration Specification for Renacer

**Version:** 1.1.0
**Sprint:** 48 (Planned)
**Author:** Pragmatic AI Labs
**Status:** Draft
**Toyota Way Principle:** *Genchi Genbutsu* (Go and see for yourself)
**Review Integration:** Toyota Way Review Response (2025-11-24)

---

## Executive Summary

This specification identifies **low-hanging fruit integrations** between renacer (syscall tracer) and aprender (ML library v0.10.0) to enhance anomaly detection [^19], performance prediction, and behavioral analysis. Following Toyota Way principles, particularly *Muda* (elimination of waste), we prioritize the `.apr` model serialization format—a critical capability that is currently **unused MUDA**.

**Critical Finding:** aprender v0.10.0 provides a mature `.apr` model serialization format with encryption, signatures, quantization, and cross-platform compatibility. Renacer's anomaly detection models are currently trained fresh on every run—pure waste (*Muda*).

---

## 1. Current State Analysis

### 1.1 Existing aprender Integration in Renacer

| Module | aprender Feature | Purpose |
|--------|------------------|---------|
| `ml_anomaly.rs` | `KMeans` | Syscall clustering for anomaly detection [^12] |
| `ml_anomaly.rs` | `Matrix` | Feature matrix construction |
| `regression/statistics.rs` | `DescriptiveStats` | Percentile/median computation |
| `regression/statistics.rs` | `ttest_ind` | Statistical significance testing [^13] |

**Gap Analysis:** Renacer uses ~4% of aprender's capabilities. The v0.10.0 release adds significant ML/DL features that remain untapped, representing *Muda* (waste) in our current tooling.

### 1.2 Toyota Way Assessment

Following *Jidoka* (build quality in) [^20], we assess each integration opportunity against:

1. **Value to customer** - Does it improve tracing insights?
2. **Technical risk** - Is the algorithm well-understood?
3. **Implementation effort** - Can it be done in <1 sprint?
4. **Test coverage** - Can we achieve 95%+ coverage?
5. **Standardization** - Does it promote consistent practices and reduce custom code?

---

## 2. Critical MUDA Elimination: `.apr` Model Persistence

### 2.0 The `.apr` Format Integration (HIGHEST PRIORITY)

**Priority:** P0 (CRITICAL - Eliminates Major Waste)
**Effort:** 2 days
**Toyota Principle:** *Muda* (Eliminate waste)

**Current State (WASTE):**
- Renacer trains KMeans/IsolationForest models **on every run**
- Model training adds 100-500ms latency per trace
- No model reuse across runs or projects
- No baseline models for "normal" behavior

**aprender's `.apr` Format Capabilities:**

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Serialization** | `save()` / `load()` | Persist trained models |
| **Compression** | Zstd compression | 50-80% size reduction |
| **Encryption** | AES-256-GCM | Secure model storage [^16] |
| **Signatures** | Ed25519 | Tamper detection [^15] |
| **Quantization** | Q8_0, Q4_0 | 4x size reduction [^14] |
| **Metadata** | Training info, provenance | Model lineage |
| **GGUF Export** | Ollama compatibility | Cross-tool sharing |

**Proposed Implementation:**

```rust
use aprender::format::{save, load, SaveOptions, Compression};
use aprender::cluster::KMeans;

/// Save trained anomaly model for reuse
pub fn save_anomaly_model(model: &KMeans, path: &Path) -> Result<()> {
    let options = SaveOptions::new()
        .compression(Compression::Zstd)
        .with_metadata("renacer_version", env!("CARGO_PKG_VERSION"))
        .with_metadata("trained_on", chrono::Utc::now().to_rfc3339());

    save(model, path, options)?;
    Ok(())
}

/// Load pre-trained model (fast startup)
pub fn load_anomaly_model(path: &Path) -> Result<KMeans> {
    load::<KMeans>(path)
}

/// Load with memory mapping (zero-copy for large models)
pub fn load_anomaly_model_mmap(path: &Path) -> Result<KMeans> {
    aprender::format::load_mmap::<KMeans>(path)
}
```

**CLI Integration:**

```bash
# Train and save model from trace data
renacer --ml-anomaly --save-model baseline.apr -- ./my-app

# Load pre-trained model (instant startup)
renacer --ml-anomaly --load-model baseline.apr -- ./my-app

# Compare against baseline
renacer --ml-anomaly --baseline baseline.apr -- ./my-app-v2
```

**Scientific Foundation:**
> Sculley, D., Holt, G., Golovin, D., et al. (2015). Hidden technical debt in machine learning systems. In *Advances in Neural Information Processing Systems* (pp. 2503-2511). [^11]

**MUDA Eliminated:**

| Waste Type | Before | After |
|------------|--------|-------|
| **Waiting** | 100-500ms training per run | <10ms model load |
| **Overprocessing** | Retrain identical models | Reuse cached models |
| **Defects** | Inconsistent baselines | Versioned, signed models |
| **Inventory** | No model history | Model registry possible |

**Benefits:**
- **10-50x faster startup** when using pre-trained models
- **Consistent baselines** across CI/CD runs [^21]
- **Model versioning** for regression tracking
- **Cross-project sharing** of "normal" syscall patterns
- **Secure storage** with encryption and signatures

---

## 3. Low-Hanging Fruit Integrations

### 3.1 Isolation Forest for Outlier Detection

**Priority:** P0 (Immediate)
**Effort:** 2-3 days
**Toyota Principle:** *Heijunka* (Level the workload)

**Current State:**
- Renacer has `isolation_forest.rs` with a custom implementation
- aprender v0.10.0 provides `IsolationForest` with optimized ensemble

**Proposed Change:**
Replace custom implementation with aprender's `IsolationForest`:

```rust
use aprender::cluster::IsolationForest;

pub fn detect_anomalous_syscalls(features: &Matrix<f64>) -> Vec<f64> {
    let iso_forest = IsolationForest::new()
        .n_estimators(100)
        .contamination(0.1)
        .fit(features);

    iso_forest.score_samples(features)
}
```

**Scientific Foundation:**
> Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In *2008 Eighth IEEE International Conference on Data Mining* (pp. 413-422). IEEE. doi:10.1109/ICDM.2008.17 [^1]

**Benefits:**
- Eliminates 200+ lines of custom code
- Leverages optimized implementation
- Consistent API with other aprender algorithms
- Enhances *Poka-yoke* (error-proofing) by using a robust, tested implementation

---

### 3.2 DBSCAN for Syscall Pattern Discovery

**Priority:** P0 (Immediate)
**Effort:** 1-2 days
**Toyota Principle:** *Kaizen* (Continuous improvement)

**Use Case:**
Discover natural clusters in syscall sequences without specifying k (unlike KMeans).

**Implementation:**

```rust
use aprender::cluster::DBSCAN;

pub fn discover_syscall_patterns(features: &Matrix<f64>) -> Vec<i32> {
    let dbscan = DBSCAN::new()
        .eps(0.5)           // Distance threshold
        .min_samples(5)     // Minimum cluster size
        .fit(features);

    dbscan.labels()  // -1 indicates noise/anomaly
}
```

**Scientific Foundation:**
> Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *KDD* (Vol. 96, No. 34, pp. 226-231). [^2]

**Benefits:**
- No need to pre-specify number of clusters
- Identifies noise points as potential anomalies
- Handles arbitrary cluster shapes

---

### 3.3 Local Outlier Factor (LOF) for Density-Based Anomaly Detection

**Priority:** P1 (High)
**Effort:** 2 days
**Toyota Principle:** *Genchi Genbutsu* (Go and see)

**Use Case:**
Detect syscalls that deviate from local neighborhood density, complementing global methods.

**Implementation:**

```rust
use aprender::cluster::LocalOutlierFactor;

pub fn compute_lof_scores(features: &Matrix<f64>) -> Vec<f64> {
    let lof = LocalOutlierFactor::new()
        .n_neighbors(20)
        .fit(features);

    lof.negative_outlier_factor()
}
```

**Scientific Foundation:**
> Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: identifying density-based local outliers. In *Proceedings of the 2000 ACM SIGMOD international conference on Management of data* (pp. 93-104). [^3]

**Benefits:**
- Captures local deviations that global methods miss
- Robust to varying cluster densities
- Interpretable outlier scores

---

### 3.4 PCA for Dimensionality Reduction in High-Cardinality Syscall Analysis

**Priority:** P1 (High)
**Effort:** 1 day
**Toyota Principle:** *Muda* (Eliminate waste)

**Use Case:**
Reduce high-dimensional syscall feature vectors for visualization and faster clustering.

**Implementation:**

```rust
use aprender::preprocessing::PCA;

pub fn reduce_syscall_dimensions(features: &Matrix<f64>, n_components: usize) -> Matrix<f64> {
    let pca = PCA::new(n_components)
        .fit(features);

    pca.transform(features)
}
```

**Scientific Foundation:**
> Pearson, K. (1901). LIII. On lines and planes of closest fit to systems of points in space. *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, 2(11), 559-572. [^4]

**Benefits:**
- Reduces noise and computation time
- Enables 2D/3D visualization of syscall patterns
- Improves clustering performance

---

### 3.5 Gaussian Mixture Models for Soft Clustering

**Priority:** P2 (Medium)
**Effort:** 2 days
**Toyota Principle:** *Hansei* (Reflection)

**Use Case:**
Assign probabilistic cluster memberships to syscalls, enabling "fuzzy" anomaly scores.

**Implementation:**

```rust
use aprender::cluster::{GaussianMixture, CovarianceType};

pub fn soft_cluster_syscalls(features: &Matrix<f64>, n_components: usize) -> Vec<Vec<f64>> {
    let gmm = GaussianMixture::new()
        .n_components(n_components)
        .covariance_type(CovarianceType::Full)
        .fit(features);

    gmm.predict_proba(features)  // Probability per cluster
}
```

**Scientific Foundation:**
> Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Methodological)*, 39(1), 1-22. [^5]

**Benefits:**
- Provides uncertainty quantification
- Handles overlapping syscall patterns
- Enables probabilistic anomaly thresholds

---

### 3.6 ARIMA for Syscall Rate Forecasting

**Priority:** P2 (Medium)
**Effort:** 3 days
**Toyota Principle:** *Nemawashi* (Build consensus)

**Use Case:**
Forecast expected syscall rates to detect deviations from predicted behavior.

**Implementation:**

```rust
use aprender::time_series::ARIMA;

pub fn forecast_syscall_rate(history: &[f64], steps: usize) -> Vec<f64> {
    let arima = ARIMA::new(2, 1, 2)  // ARIMA(2,1,2)
        .fit(history);

    arima.forecast(steps)
}
```

**Scientific Foundation:**
> Box, G. E., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. San Francisco: Holden-Day. [^6]

**Benefits:**
- Detects trending anomalies (gradual degradation)
- Enables proactive alerting
- Handles non-stationary syscall rates

---

### 3.7 StandardScaler for Feature Normalization

**Priority:** P0 (Immediate)
**Effort:** 0.5 days
**Toyota Principle:** *Seiri* (Sort/Organize)

**Use Case:**
Normalize syscall features before clustering to prevent scale bias.

**Implementation:**

```rust
use aprender::preprocessing::StandardScaler;

pub fn normalize_features(features: &Matrix<f64>) -> Matrix<f64> {
    let scaler = StandardScaler::new()
        .with_mean(true)
        .with_std(true)
        .fit(features);

    scaler.transform(features)
}
```

**Scientific Foundation:**
> Jain, A., Nandakumar, K., & Ross, A. (2005). Score normalization in multimodal biometric systems. *Pattern Recognition*, 38(12), 2270-2285. [^7]

**Benefits:**
- Prevents features with large magnitudes from dominating
- Improves convergence of clustering algorithms
- Standard practice for ML pipelines

---

### 3.8 Silhouette Score for Cluster Quality Assessment

**Priority:** P1 (High)
**Effort:** 0.5 days
**Toyota Principle:** *Poka-yoke* (Error-proofing)

**Use Case:**
Automatically validate clustering quality and select optimal k.

**Implementation:**

```rust
use aprender::metrics::silhouette_score;

pub fn evaluate_clustering(features: &Matrix<f64>, labels: &[i32]) -> f64 {
    silhouette_score(features, labels)
}

pub fn find_optimal_k(features: &Matrix<f64>, k_range: std::ops::Range<usize>) -> usize {
    k_range
        .map(|k| {
            let kmeans = KMeans::new(k).fit(features);
            (k, silhouette_score(features, &kmeans.labels()))
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(k, _)| k)
        .unwrap_or(3)
}
```

**Scientific Foundation:**
> Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65. [^8]

**Benefits:**
- Quantifies cluster separation
- Enables automatic hyperparameter selection
- Validates anomaly detection quality

---

### 3.9 Random Forest for Syscall Classification

**Priority:** P2 (Medium)
**Effort:** 3 days
**Toyota Principle:** *Monozukuri* (The art of making things)

**Use Case:**
Classify syscalls into behavioral categories (normal, suspicious, malicious).

**Implementation:**

```rust
use aprender::tree::RandomForestClassifier;

pub struct SyscallClassifier {
    forest: RandomForestClassifier,
}

impl SyscallClassifier {
    pub fn train(features: &Matrix<f64>, labels: &[usize]) -> Self {
        let forest = RandomForestClassifier::new()
            .n_estimators(100)
            .max_depth(Some(10))
            .fit(features, labels);

        Self { forest }
    }

    pub fn predict(&self, features: &Matrix<f64>) -> Vec<usize> {
        self.forest.predict(features)
    }
}
```

**Scientific Foundation:**
> Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32. [^9]

**Benefits:**
- Robust to overfitting
- Handles high-dimensional features
- Provides feature importance rankings

---

### 3.10 Cross-Validation for Model Selection

**Priority:** P1 (High)
**Effort:** 1 day
**Toyota Principle:** *Yokoten* (Horizontal deployment)

**Use Case:**
Validate anomaly detection models before deployment.

**Implementation:**

```rust
use aprender::model_selection::{KFold, cross_validate};

pub fn validate_anomaly_model<M: Estimator>(
    model: &M,
    features: &Matrix<f64>,
    labels: &[usize],
) -> CrossValidationResult {
    let kfold = KFold::new(5).shuffle(true);
    cross_validate(model, features, labels, &kfold)
}
```

**Scientific Foundation:**
> Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. *Journal of the Royal Statistical Society: Series B (Methodological)*, 36(2), 111-133. [^10]

**Benefits:**
- Prevents overfitting to training data
- Provides variance estimates
- Industry standard for model validation

---

## 4. Implementation Roadmap

### Phase 0: MUDA Elimination (Sprint 48 - IMMEDIATE)
*Toyota Principle: Muda (Eliminate waste)*

| Task | Effort | Dependencies |
|------|--------|--------------|
| Implement `.apr` model save/load | 2 days | None |
| Add `--save-model` CLI flag | 0.5 days | `.apr` integration |
| Add `--load-model` CLI flag | 0.5 days | `.apr` integration |
| Add `--baseline` CLI flag | 0.5 days | `.apr` integration |
| Model versioning & metadata | 0.5 days | `.apr` integration |

**Deliverables:**
- Zero-waste model persistence
- 10-50x faster startup with pre-trained models
- Consistent baselines across CI/CD runs
- Secure model storage with encryption/signatures

### Phase 1: Foundation (Sprint 49)
*Toyota Principle: Genba (The real place)*

| Task | Effort | Dependencies |
|------|--------|--------------|
| Replace IsolationForest | 2 days | Phase 0 |
| Add StandardScaler | 0.5 days | None |
| Add Silhouette Score | 0.5 days | None |
| Add DBSCAN | 1 day | StandardScaler |

**Deliverables:**
- 4 new aprender integrations
- 95%+ test coverage
- Benchmark comparison with custom implementations

### Phase 2: Enhancement (Sprint 50)
*Toyota Principle: Kaizen (Continuous improvement)*

| Task | Effort | Dependencies |
|------|--------|--------------|
| Add LOF | 2 days | StandardScaler |
| Add PCA | 1 day | StandardScaler |
| Add GMM | 2 days | StandardScaler |
| Add Cross-Validation | 1 day | Any model |

**Deliverables:**
- 4 additional integrations
- Probabilistic anomaly detection
- Model validation framework

### Phase 3: Advanced (Sprint 51)
*Toyota Principle: Hansei (Reflection)*

| Task | Effort | Dependencies |
|------|--------|--------------|
| Add ARIMA | 3 days | None |
| Add RandomForest | 3 days | StandardScaler |
| Unified ML Pipeline | 2 days | All above |

**Deliverables:**
- Time series forecasting
- Supervised classification
- End-to-end ML pipeline

---

## 5. Quality Gates

### 5.1 Definition of Done (Toyota Way: Jidoka)

Each integration must satisfy these criteria, embodying *Jidoka* by building quality directly into the process and stopping for abnormalities:

1. **Unit Tests:** 95%+ coverage [^18]
2. **Property Tests:** 5+ proptest cases [^17]
3. **Benchmark:** Performance comparison with baseline
4. **Documentation:** API docs + usage example
5. **Integration Test:** End-to-end with real syscall data

### 5.2 Acceptance Criteria

```rust
#[test]
fn test_aprender_integration_quality() {
    // Must not increase binary size by >5%
    assert!(binary_size_increase() < 0.05);

    // Must not regress P99 latency
    assert!(p99_latency_ms() < 10.0);

    // Must improve anomaly detection F1 score
    assert!(f1_score() > baseline_f1_score());
}
```

---

## 6. Risk Assessment

### 6.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API breaking changes | Low | Medium | Pin aprender version |
| Performance regression | Medium | High | Benchmark all changes |
| Increased binary size | Medium | Low | Feature-gate new modules |

### 6.2 Toyota Way Mitigations

- **Andon (Stop the line):** Automated CI gates prevent regression
- **Poka-yoke (Error-proofing):** Type-safe aprender APIs
- **Nemawashi (Consensus):** Code review for all changes

---

## 7. References

[^1]: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In *2008 Eighth IEEE International Conference on Data Mining* (pp. 413-422). IEEE. doi:10.1109/ICDM.2008.17

[^2]: Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *KDD* (Vol. 96, No. 34, pp. 226-231).

[^3]: Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: identifying density-based local outliers. In *Proceedings of the 2000 ACM SIGMOD international conference on Management of data* (pp. 93-104).

[^4]: Pearson, K. (1901). LIII. On lines and planes of closest fit to systems of points in space. *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, 2(11), 559-572.

[^5]: Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Methodological)*, 39(1), 1-22.

[^6]: Box, G. E., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. San Francisco: Holden-Day.

[^7]: Jain, A., Nandakumar, K., & Ross, A. (2005). Score normalization in multimodal biometric systems. *Pattern Recognition*, 38(12), 2270-2285.

[^8]: Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.

[^9]: Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

[^10]: Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. *Journal of the Royal Statistical Society: Series B (Methodological)*, 36(2), 111-133.

[^11]: Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J. F., & Dennison, D. (2015). Hidden technical debt in machine learning systems. In *Advances in Neural Information Processing Systems* (Vol. 28, pp. 2503-2511). NIPS.

[^12]: MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings of the fifth Berkeley symposium on mathematical statistics and probability* (Vol. 1, No. 14, pp. 281-297).

[^13]: Student. (1908). The probable error of a mean. *Biometrika*, 6(1), 1-25.

[^14]: Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., ... & Adam, H. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2704-2713.

[^15]: Bernstein, D. J., Duif, N., Lange, T., Schwabe, P., & Yang, B. Y. (2012). High-speed high-security signatures. *Journal of Cryptographic Engineering*, 2(2), 77-89.

[^16]: Daemen, J., & Rijmen, V. (2002). *The Design of Rijndael: AES - The Advanced Encryption Standard*. Springer-Verlag.

[^17]: Claessen, K., & Hughes, J. (2000). QuickCheck: a lightweight tool for random testing of Haskell programs. In *Proceedings of the fifth ACM SIGPLAN international conference on Functional programming* (pp. 268-279).

[^18]: Hoare, C. A. R. (1969). An axiomatic basis for computer programming. *Communications of the ACM*, 12(10), 576-580.

[^19]: Forrest, S., Hofmeyr, S. A., Somayaji, A., & Longstaff, T. A. (1996). A sense of self for unix processes. In *Proceedings of the 1996 IEEE Symposium on Security and Privacy* (pp. 120-128). IEEE.

[^20]: Liker, J. K. (2004). *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. McGraw-Hill.

[^21]: Peng, R. D. (2011). Reproducible research in computational science. *Science*, 334(6060), 1226-1227.

---

## 8. Appendix: Toyota Way Principles Applied

| Principle | Japanese | Application in This Spec |
|-----------|----------|--------------------------|
| Genchi Genbutsu | 現地現物 | Analyze actual renacer code before proposing changes |
| Kaizen | 改善 | Incremental improvements over 3 sprints |
| Jidoka | 自働化 | Automated quality gates prevent defects |
| Heijunka | 平準化 | Level workload across phases |
| Muda | 無駄 | Eliminate redundant custom implementations |
| Hansei | 反省 | Reflect on each phase before proceeding |
| Nemawashi | 根回し | Build consensus via code review |
| Poka-yoke | ポカヨケ | Type-safe APIs prevent misuse |
| Andon | アンドン | CI gates stop builds on failure |
| Yokoten | 横展 | Deploy patterns horizontally across modules |

---

## 9. Conclusion

This specification identifies **11 low-hanging fruit integrations** that can be implemented in **4 sprints** with minimal risk. By leveraging aprender v0.10.0's mature ML algorithms and `.apr` model persistence format, renacer can:

1. **Eliminate MUDA** by persisting trained models instead of retraining every run
2. **Reduce code complexity** by replacing custom implementations
3. **Improve anomaly detection** with proven algorithms
4. **Enable new capabilities** (time series, classification, probabilistic clustering)
5. **Maintain quality** through rigorous testing and Toyota Way principles

**CRITICAL:** The recommended starting point is **Phase 0** (Sprint 48), implementing `.apr` model persistence to eliminate the most egregious waste. This single integration provides 10-50x startup improvement and enables all subsequent phases to benefit from reusable models.

Following Phase 0, **Phase 1** (Sprint 49) focuses on Isolation Forest, StandardScaler, Silhouette Score, and DBSCAN—all of which provide immediate value with minimal integration effort.
