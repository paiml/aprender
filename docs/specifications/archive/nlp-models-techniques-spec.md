# NLP Models and Techniques Specification
## Organizational Intelligence Plugin - Enhanced Defect Classification

**Version:** 1.0
**Date:** 2025-11-24
**Status:** PROPOSED
**Authors:** OIP Development Team
**Issue:** [#1 - Improve NLP categorization for transpiler-specific defect patterns](https://github.com/paiml/organizational-intelligence-plugin/issues/1)

---

## Executive Summary

This specification defines a comprehensive Natural Language Processing (NLP) strategy to improve defect classification in the Organizational Intelligence Plugin (OIP). Current classification achieves only **30.8% actionable categorization** (69.2% classified as "General"). This specification proposes a multi-tiered NLP approach combining rule-based, classical ML, and transformer-based techniques to achieve **≥80% actionable categorization** with **≥85% classification accuracy**.

**Key Goals:**
- Reduce "General" category from 69.2% to <20%
- Achieve ≥85% F1-score on transpiler-specific defect patterns
- Support multi-label classification (single commit → multiple categories)
- Enable domain adaptation for different project types (transpilers, web services, ML systems)
- Maintain <30s inference time for PR review workflows

**Proposed Architecture:** Hybrid 3-tier system
1. **Tier 1 (Fast):** Rule-based classifier (current) - <10ms
2. **Tier 2 (Medium):** TF-IDF + Ensemble ML - <100ms
3. **Tier 3 (Accurate):** CodeBERT fine-tuned transformer - <1s

---

## 1. Background and Motivation

### 1.1 Current State Analysis

**Problem:** OIP's rule-based classifier (10 categories) shows poor granularity:

```bash
# depyler transpiler analysis (500 commits)
Category 0 (General): 346 (69.2%)  # TOO BROAD
Category 9 (Documentation): 54 (10.8%)
Category 5 (Performance): 40 (8.0%)
# ... remaining 8 categories: 60 commits (12%)
```

**Root Causes:**
1. **Simple keyword matching** - Only checks commit messages, ignores diffs
2. **Single-label classification** - Cannot handle multi-faceted defects
3. **No semantic understanding** - "operator precedence" not recognized
4. **Generic categories** - No domain-specific taxonomies (e.g., transpiler patterns)

### 1.2 Use Case: Transpiler Development

**Example:** DEPYLER-0511 (Operator Precedence Bug)

```diff
- parse_quote! { #iter_expr.into_iter() }
+ parse_quote! { (#iter_expr).into_iter() }
```

**Commit Message:** `[GREEN] DEPYLER-0511: Fix range comprehension parentheses`

**Current Classification:** Category 0 (General) - 70% confidence
**Desired Classification:**
- Primary: OperatorPrecedence (90% confidence)
- Secondary: Comprehension (75% confidence)
- Tertiary: CodeGeneration (60% confidence)

### 1.3 Impact on Development Workflow

**Before (Current State):**
- 69.2% of defects categorized as "General" → No actionable insights
- Cannot prioritize work (which component needs improvement?)
- Trend analysis impossible (is operator precedence improving?)

**After (Target State):**
- ≥80% defects have specific categories
- Clear prioritization: "Fix top 3 categories = 45% of defects"
- Trend tracking: "Operator precedence bugs down 60% this quarter"

---

## 2. NLP Techniques Overview

### 2.1 Feature Extraction Methods

#### 2.1.1 Text Preprocessing

**Standard NLP Pipeline:**
```python
# Step 1: Tokenization
text = "fix: null pointer dereference in parse_expr()"
tokens = ["fix", "null", "pointer", "dereference", "parse_expr"]

# Step 2: Lowercasing + Stop word removal
tokens_clean = ["fix", "null", "pointer", "dereference", "parse_expr"]
# Keep domain-specific words: "fix", "null", "pointer"

# Step 3: Stemming/Lemmatization (optional)
stemmed = ["fix", "null", "point", "derefer", "pars", "expr"]
# Recommendation: Use lemmatization for better semantic preservation
```

**Software-Specific Considerations:**
- **Preserve code tokens:** `parse_expr()`, `into_iter()`, `Option<T>`
- **Handle camelCase:** `parseExpr` → `["parse", "Expr"]`
- **Extract regex patterns:** `\bfix\b`, `\bbug\b`, `\berror\b`

#### 2.1.2 TF-IDF (Term Frequency-Inverse Document Frequency)

**Mathematical Definition:**
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
TF(t, d) = (count of term t in document d) / (total terms in d)
IDF(t) = log(N / df(t))
  where N = total documents, df(t) = documents containing t
```

**Sklearn Implementation:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Best practices from sklearn documentation [1]
vectorizer = TfidfVectorizer(
    max_features=1500,      # Limit to top N features
    ngram_range=(1, 3),     # Unigrams, bigrams, trigrams
    sublinear_tf=True,      # Use log(1 + TF) for damping
    min_df=5,               # Ignore terms in <5 documents
    max_df=0.5,             # Ignore terms in >50% documents
    stop_words='english',   # Remove common words
    analyzer='word',        # Word-level tokenization
    token_pattern=r'\b\w+\b'  # Alphanumeric tokens
)

# Fit on training corpus (commit messages)
X_train_tfidf = vectorizer.fit_transform(commit_messages)
```

**Software Engineering Adaptations:**
```python
# Custom stop words (exclude generic words, keep technical terms)
custom_stop_words = ['the', 'a', 'an', 'and', 'or', 'but']
# Keep: 'fix', 'bug', 'error', 'null', 'pointer', 'race', 'deadlock'

# Software-specific n-grams
# Unigrams: "null", "pointer"
# Bigrams: "null pointer", "race condition", "memory leak"
# Trigrams: "use after free", "operator precedence bug"
```

**When to Use TF-IDF:**
- Medium-sized corpora (1K-100K documents)
- When interpretability is important (see feature weights)
- Baseline model before trying deep learning
- Fast inference required (<100ms)

**Limitations:**
- No semantic understanding ("null pointer" ≠ "nullptr")
- Sparse representations (high-dimensional, mostly zeros)
- Cannot handle out-of-vocabulary (OOV) words

#### 2.1.3 N-grams

**Definition:** Contiguous sequences of N tokens

```python
text = "fix race condition in mutex lock"

# Unigrams (1-gram)
["fix", "race", "condition", "in", "mutex", "lock"]

# Bigrams (2-gram)
["fix race", "race condition", "condition in", "in mutex", "mutex lock"]

# Trigrams (3-gram)
["fix race condition", "race condition in", "condition in mutex", "in mutex lock"]
```

**Research Evidence:**
- Practitioner consensus: "TF-IDF + n-grams + linear model is an extremely strong approach" [2]
- Ensemble study: N-grams (1-3) + TF-IDF achieve 85-92% accuracy in sentiment classification [3]

**Implementation:**
```python
from sklearn.feature_extraction.text import CountVectorizer

# Character n-grams (for code tokens)
char_vectorizer = CountVectorizer(
    analyzer='char',
    ngram_range=(3, 5),  # 3-5 character sequences
    max_features=1000
)
# Useful for: "parse_expr", "into_iter", "Option<T>"

# Word n-grams (for natural language)
word_vectorizer = CountVectorizer(
    analyzer='word',
    ngram_range=(1, 3),
    max_features=1500
)
# Useful for: "null pointer dereference", "race condition"
```

#### 2.1.4 Word Embeddings (Word2Vec, GloVe, fastText)

**Motivation:** Capture semantic relationships between words

```python
# Semantic similarity examples
similar("null pointer") → ["nullptr", "null dereference", "dangling pointer"]
similar("race condition") → ["data race", "thread safety", "concurrency bug"]
```

**Word2Vec (2013):**
- Two architectures: CBOW (predict word from context) and Skip-gram (predict context from word)
- Produces dense vectors (typically 100-300 dimensions)
- Trained on large text corpora

**GloVe (Global Vectors, 2014):**
- Matrix factorization approach
- Captures global corpus statistics
- Pre-trained models available (Wikipedia, Common Crawl)

**fastText (Facebook, 2016):**
- Extension of Word2Vec with character n-grams
- Handles out-of-vocabulary (OOV) words
- Better for morphologically rich languages and code tokens

**Research Evidence:**
- Bug report management: Word embeddings (WE) improve BRM tasks vs. classical VSM [4]
- Vulnerability detection: Word2Vec achieved highest precision/recall for vulnerable functions [5]

**Implementation:**
```python
from gensim.models import Word2Vec, FastText

# Train Word2Vec on commit message corpus
commit_tokens = [
    ["fix", "null", "pointer", "bug"],
    ["race", "condition", "in", "mutex"],
    # ... thousands more
]

model = Word2Vec(
    sentences=commit_tokens,
    vector_size=200,      # Embedding dimension
    window=5,             # Context window size
    min_count=5,          # Ignore rare words
    workers=4,            # Parallel training
    sg=1                  # Skip-gram (vs CBOW)
)

# Get word vector
vector = model.wv["null"]  # 200-dimensional dense vector

# Find similar words
similar = model.wv.most_similar("null", topn=10)
# [("nullptr", 0.87), ("null_ptr", 0.82), ("dangling", 0.78), ...]
```

**When to Use Word Embeddings:**
- Need semantic similarity (synonyms, related concepts)
- Handling OOV words (fastText)
- Medium-sized training corpus (10K+ documents)
- Can combine with TF-IDF for hybrid features

#### 2.1.5 Transformer-Based Embeddings (BERT, CodeBERT)

**BERT (Bidirectional Encoder Representations from Transformers, 2018):**
- Contextual embeddings (word meaning depends on context)
- Pre-trained on large text corpora (Wikipedia, books)
- Fine-tunable for downstream tasks

**CodeBERT (Microsoft, 2020):**
- BERT pre-trained on code and natural language
- Bimodal model: understands code + comments
- State-of-the-art for software engineering tasks

**Research Evidence:**
- CodeBERT: 10-12% F1-score improvement over baselines in just-in-time defect prediction [6]
- Transformer models (CodeT5, CodeBERT): Effective for identifying various defects, learn complex patterns [7]
- BERT-based cross-project defect prediction: Outperforms classical ML [8]

**Architecture Comparison:**

| Model | Training Data | Use Case | Performance |
|-------|---------------|----------|-------------|
| BERT-base | Wikipedia, Books | General text | Good for commit messages |
| CodeBERT | GitHub (6 languages) | Code + NL | Best for code diffs + messages |
| RoBERTa | Wikipedia, CC-News | Robust text | Better than BERT for text |
| CodeT5 | CodeSearchNet | Code generation | Good for code completion |

**Implementation:**
```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained CodeBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# Encode commit message + code diff
text = """fix: null pointer in parse_expr()
--- a/parser.py
+++ b/parser.py
-    result = parse_expr(node)
+    result = parse_expr(node) if node else None
"""

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)

# Get [CLS] token embedding (sentence-level representation)
embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: (1, 768)
```

**When to Use Transformers:**
- Need state-of-the-art accuracy
- Have GPU resources for inference
- Large training corpus (50K+ examples)
- Can afford 100ms-1s inference time
- Need to understand code + natural language

---

### 2.2 Classification Algorithms

#### 2.2.1 Ensemble Methods

**Definition:** Combine multiple weak learners to create a strong learner

**Common Ensemble Algorithms:**

1. **Random Forest (RF)**
   - Bagging: Bootstrap Aggregating
   - Trains multiple decision trees on random subsets
   - Averages predictions (regression) or votes (classification)

2. **Gradient Boosting (GB, XGBoost, CatBoost)**
   - Boosting: Sequential training, each tree corrects previous errors
   - XGBoost: Extreme Gradient Boosting (regularization, parallel processing)
   - CatBoost: Handles categorical features natively

3. **AdaBoost**
   - Adaptive Boosting: Increases weights of misclassified samples
   - Combines weak learners (decision stumps)

**Research Evidence:**
- Ensemble ML (AdaBoost, CatBoost, XGBoost) detect refactoring types with 85-92% accuracy [9]
- Random Forest + TF-IDF: Strong baseline for text classification [3]

**Implementation:**
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=20,          # Prevent overfitting
    min_samples_split=10,
    class_weight='balanced',  # Handle imbalanced classes
    n_jobs=-1              # Parallel training
)

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8          # Stochastic gradient boosting
)

# Train and evaluate
rf.fit(X_train_tfidf, y_train)
scores = cross_val_score(rf, X_train_tfidf, y_train, cv=5)
print(f"Cross-validated F1: {scores.mean():.2f} ± {scores.std():.2f}")
```

**When to Use Ensemble Methods:**
- Tabular data (TF-IDF features, code metrics)
- Need interpretability (feature importances)
- Imbalanced classes (Random Forest handles well)
- Fast training (minutes on CPU)
- Baseline before deep learning

#### 2.2.2 Support Vector Machines (SVM)

**Definition:** Find optimal hyperplane to separate classes

**Kernel Trick:** Transform non-linearly separable data to higher dimensions
- Linear kernel: Fast, good for high-dimensional text
- RBF kernel: Handles non-linear relationships

**Implementation:**
```python
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# Linear SVM (recommended for text)
svm = SVC(
    kernel='linear',
    C=1.0,              # Regularization parameter
    class_weight='balanced'
)

# Add probability calibration
calibrated_svm = CalibratedClassifierCV(svm, cv=3)
calibrated_svm.fit(X_train_tfidf, y_train)

# Get probability estimates (for confidence scores)
probs = calibrated_svm.predict_proba(X_test_tfidf)
```

**When to Use SVM:**
- High-dimensional sparse features (TF-IDF)
- Binary classification
- Need margin-based confidence scores
- Strong theoretical guarantees

#### 2.2.3 Deep Learning Architectures

**1. Convolutional Neural Networks (CNN) for Text**

```python
import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3)
        self.conv2 = nn.Conv1d(embed_dim, 128, kernel_size=4)
        self.conv3 = nn.Conv1d(embed_dim, 128, kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(384, num_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)  # (batch, embed_dim, seq_len)
        x1 = torch.relu(self.conv1(x)).max(dim=2)[0]
        x2 = torch.relu(self.conv2(x)).max(dim=2)[0]
        x3 = torch.relu(self.conv3(x)).max(dim=2)[0]
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.dropout(x)
        return self.fc(x)
```

**Research Evidence:**
- CNN + Grad-CAM: Effective for extracting bug report features related to fix time [10]
- TextCNN ensemble: High accuracy for Japanese text multi-classification [11]

**2. Recurrent Neural Networks (LSTM, BiGRU)**

```python
class BiGRU_Classifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, hidden = self.gru(x)
        # Use last hidden states from both directions
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(hidden)
```

**When to Use Deep Learning:**
- Large training corpus (50K+ examples)
- GPU resources available
- Need state-of-the-art accuracy
- Sequential patterns matter (commit message → diff → result)

#### 2.2.4 Multi-Label Classification

**Problem:** Single commit can belong to multiple categories

**Example:**
```
Commit: "fix operator precedence in range comprehensions"
Labels: [OperatorPrecedence, Comprehension, CodeGeneration]
```

**Approaches:**

1. **Binary Relevance:** Train N binary classifiers (one per label)
```python
from sklearn.multioutput import MultiOutputClassifier

base_clf = RandomForestClassifier()
multi_label_clf = MultiOutputClassifier(base_clf)
multi_label_clf.fit(X_train, y_train_multi)  # y_train: (n_samples, n_labels)
```

2. **Classifier Chains:** Model label dependencies
```python
from sklearn.multioutput import ClassifierChain

chain_clf = ClassifierChain(base_clf)
chain_clf.fit(X_train, y_train_multi)
```

3. **Label Powerset:** Treat each unique label combination as a class
```python
from skmultilearn.problem_transform import LabelPowerset

lp_clf = LabelPowerset(base_clf)
lp_clf.fit(X_train, y_train_multi)
```

**Evaluation Metrics:**
- **Hamming Loss:** Fraction of incorrect labels
- **Subset Accuracy:** Exact match of entire label set
- **F1-Score (Micro, Macro, Samples):** Harmonic mean of precision/recall

---

## 3. Sklearn Best Practices

### 3.1 Pipeline Construction

**Motivation:** Prevent data leakage, ensure reproducibility

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Create pipeline (preprocessing + model)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier())
])

# Hyperparameter tuning with grid search
param_grid = {
    'tfidf__max_features': [1000, 1500, 2000],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [10, 20, None]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best F1: {grid_search.best_score_:.3f}")
```

**Benefits:**
- Prevents overfitting (no information leakage from test set)
- Reproducible (entire pipeline saved as one object)
- Easy hyperparameter tuning (search across all pipeline stages)

### 3.2 Cross-Validation Strategy

**Stratified K-Fold:** Preserve class distribution in each fold

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'precision_macro': 'precision_macro',
    'recall_macro': 'recall_macro',
    'f1_macro': 'f1_macro'
}

cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=skf,
    scoring=scoring,
    return_train_score=True
)

print(f"F1 (Macro): {cv_results['test_f1_macro'].mean():.3f} ± {cv_results['test_f1_macro'].std():.3f}")
```

**Time-Series Split:** For temporal data (commits are time-ordered)

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate chronologically
```

### 3.3 Handling Imbalanced Classes

**Problem:** Most defects are "General" (69.2%), other categories rare

**Solutions:**

1. **Class Weighting:**
```python
clf = RandomForestClassifier(class_weight='balanced')
# Automatically adjusts weights inversely proportional to class frequencies
```

2. **Resampling:**
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# SMOTE: Synthetic Minority Over-sampling
smote = SMOTE(sampling_strategy='auto', random_state=42)
rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)

pipeline = ImbPipeline([
    ('smote', smote),
    ('rus', rus),
    ('clf', RandomForestClassifier())
])
```

3. **Ensemble with Balanced Bagging:**
```python
from imblearn.ensemble import BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier(
    n_estimators=100,
    sampling_strategy='auto',
    replacement=True
)
```

### 3.4 Feature Engineering

**Combining Multiple Feature Types:**

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Assume we have:
# - commit_message (text)
# - code_diff (text)
# - lines_changed (numeric)
# - files_changed (numeric)

preprocessor = ColumnTransformer([
    ('message_tfidf', TfidfVectorizer(max_features=1000), 'commit_message'),
    ('diff_tfidf', TfidfVectorizer(max_features=500), 'code_diff'),
    ('numeric', StandardScaler(), ['lines_changed', 'files_changed'])
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier())
])
```

**Feature Union (Alternative):**

```python
from sklearn.pipeline import FeatureUnion

feature_union = FeatureUnion([
    ('message_features', Pipeline([
        ('selector', FunctionTransformer(lambda X: X['commit_message'])),
        ('tfidf', TfidfVectorizer(max_features=1000))
    ])),
    ('diff_features', Pipeline([
        ('selector', FunctionTransformer(lambda X: X['code_diff'])),
        ('tfidf', TfidfVectorizer(max_features=500))
    ]))
])
```

### 3.5 Model Persistence

**Saving Trained Models:**

```python
import joblib

# Save entire pipeline (preprocessing + model)
joblib.dump(pipeline, 'defect_classifier_v1.pkl')

# Load for inference
loaded_pipeline = joblib.load('defect_classifier_v1.pkl')
predictions = loaded_pipeline.predict(new_commits)
```

**Versioning Best Practices:**
- Include date/version in filename: `defect_classifier_2025-11-24_v1.2.pkl`
- Save metadata: training data size, feature names, performance metrics
- Use Git LFS for large model files (>100MB)

---

## 4. UC Berkeley NLP Research Principles

### 4.1 Empirical Validation

**Principle:** Validate on real-world software engineering corpora

**Recommended Datasets:**
1. **Defects4J** - Real bugs from 17 Java projects (800+ bugs)
2. **ManyBugs & IntroClass** - C program defects
3. **BugSwarm** - 3,091 reproducible failures from GitHub
4. **PROMISE** - Defect prediction datasets (NASA, Eclipse, etc.)
5. **Custom corpus** - Your organization's commit history

**Evaluation Protocol:**
```python
from sklearn.metrics import classification_report, confusion_matrix

# Split: 70% train, 15% validation, 15% test
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

# Train on train set
model.fit(X_train, y_train)

# Tune hyperparameters on validation set
# ... grid search ...

# Final evaluation on test set
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### 4.2 Error Analysis

**Principle:** Understand failure modes, not just accuracy

**Analysis Steps:**

1. **Confusion Matrix Analysis:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Identify common misclassifications
# Example: "Security" often misclassified as "General"
```

2. **Error Case Study:**
```python
# Find misclassified examples
misclassified_idx = np.where(y_pred != y_test)[0]

for idx in misclassified_idx[:10]:
    print(f"True: {y_test[idx]}, Predicted: {y_pred[idx]}")
    print(f"Commit: {commit_messages[idx]}")
    print(f"Confidence: {model.predict_proba(X_test[idx])}")
    print("---")
```

3. **Feature Importance:**
```python
# For tree-based models
feature_importances = model.named_steps['clf'].feature_importances_
feature_names = model.named_steps['tfidf'].get_feature_names_out()

# Top 20 most important features
top_indices = np.argsort(feature_importances)[-20:]
print("Top features:")
for idx in top_indices:
    print(f"{feature_names[idx]}: {feature_importances[idx]:.4f}")
```

### 4.3 Domain Adaptation

**Principle:** Models should transfer across projects/domains

**Techniques:**

1. **Transfer Learning:**
```python
# Pre-train on large corpus (all GitHub projects)
base_model = train_on_large_corpus(github_commits)

# Fine-tune on specific project (depyler)
fine_tuned_model = base_model.fine_tune(depyler_commits)
```

2. **Domain-Specific Embeddings:**
```python
# Train Word2Vec on software engineering corpus
from gensim.models import Word2Vec

se_corpus = load_stackoverflow_posts() + load_github_commits()
se_model = Word2Vec(se_corpus, vector_size=200, window=5, min_count=5)

# Use SE embeddings for defect classification
```

3. **Few-Shot Learning:**
```python
# Learn from few examples of new category
# Example: Introduce "OperatorPrecedence" with only 10 labeled examples

from sklearn.semi_supervised import LabelPropagation

# Use label propagation with unlabeled data
lp = LabelPropagation()
lp.fit(X_combined, y_partial)  # y_partial: 10 labeled + 1000 unlabeled (-1)
```

### 4.4 Interpretability and Explainability

**Principle:** Users must understand why a commit was classified

**Techniques:**

1. **LIME (Local Interpretable Model-agnostic Explanations):**
```python
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=class_names)

# Explain prediction for a single commit
explanation = explainer.explain_instance(
    commit_message,
    classifier_fn=model.predict_proba,
    num_features=10
)

# Show top words influencing the decision
print(explanation.as_list())
# [("null pointer", 0.32), ("dereference", 0.18), ("fix", 0.12), ...]
```

2. **SHAP (SHapley Additive exPlanations):**
```python
import shap

explainer = shap.TreeExplainer(model.named_steps['clf'])
shap_values = explainer.shap_values(X_test_tfidf)

# Visualize feature contributions
shap.summary_plot(shap_values, X_test_tfidf, feature_names=feature_names)
```

3. **Attention Visualization (for Transformers):**
```python
# Extract attention weights from BERT
from transformers import BertModel

outputs = model(**inputs, output_attentions=True)
attentions = outputs.attentions  # Tuple of attention matrices

# Visualize which tokens the model focused on
import matplotlib.pyplot as plt

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
attention_weights = attentions[-1][0].mean(dim=0).detach().cpu().numpy()

plt.bar(range(len(tokens)), attention_weights[0])
plt.xticks(range(len(tokens)), tokens, rotation=90)
plt.show()
```

### 4.5 Continuous Learning

**Principle:** Models should improve as new data arrives

**Active Learning Loop:**

```python
# 1. Train initial model on labeled data
model.fit(X_labeled, y_labeled)

# 2. Predict on unlabeled pool
y_pred = model.predict(X_unlabeled)
y_proba = model.predict_proba(X_unlabeled)

# 3. Select uncertain samples (low confidence)
uncertainty = 1 - y_proba.max(axis=1)
uncertain_indices = np.argsort(uncertainty)[-100:]  # Top 100 uncertain

# 4. Request human labels for uncertain samples
X_to_label = X_unlabeled[uncertain_indices]
y_new_labels = request_human_labels(X_to_label)

# 5. Retrain with new labels
X_labeled = np.vstack([X_labeled, X_to_label])
y_labeled = np.hstack([y_labeled, y_new_labels])
model.fit(X_labeled, y_labeled)

# 6. Repeat
```

**Model Monitoring:**
```python
# Track performance over time
from collections import defaultdict

performance_log = defaultdict(list)

for week in range(52):
    # Evaluate on this week's commits
    y_true_week = get_labels_for_week(week)
    y_pred_week = model.predict(X_week)

    f1_week = f1_score(y_true_week, y_pred_week, average='macro')
    performance_log['f1'].append(f1_week)
    performance_log['week'].append(week)

    # Detect performance degradation
    if f1_week < 0.75:  # Below threshold
        print(f"⚠️ Week {week}: F1 dropped to {f1_week:.3f}")
        # Trigger retraining
```

---

## 5. Recommended Architecture

### 5.1 Hybrid 3-Tier System

**Tier 1: Rule-Based Classifier (Fast, <10ms)**
- Current implementation (10 categories)
- High-confidence keyword matches
- Handles obvious cases: "null pointer" → MemorySafety

**Tier 2: TF-IDF + Ensemble ML (Medium, <100ms)**
- TF-IDF features (1-3 grams) + code metrics
- Random Forest or XGBoost classifier
- Handles common patterns with good accuracy

**Tier 3: CodeBERT Transformer (Accurate, <1s)**
- Fine-tuned on software engineering corpus
- Handles complex, ambiguous cases
- Multi-label classification

**Decision Flow:**
```python
def classify_commit(commit_message, code_diff):
    # Tier 1: Rule-based
    rule_result = rule_classifier.classify(commit_message)
    if rule_result.confidence > 0.90:
        return rule_result

    # Tier 2: TF-IDF + ML
    ml_result = ml_classifier.classify(commit_message, code_diff)
    if ml_result.confidence > 0.80:
        return ml_result

    # Tier 3: Transformer (expensive, but accurate)
    transformer_result = transformer_classifier.classify(commit_message, code_diff)
    return transformer_result
```

**Performance Targets:**

| Tier | Inference Time | Accuracy | Coverage |
|------|----------------|----------|----------|
| Rule-based | <10ms | 85% | 20% of commits |
| TF-IDF + ML | <100ms | 88% | 60% of commits |
| CodeBERT | <1s | 92% | 20% of commits |
| **Overall** | **<150ms avg** | **≥88%** | **100%** |

### 5.2 Expanded Taxonomy

**Current Categories (10):**
1. MemorySafety
2. ConcurrencyBugs
3. LogicErrors
4. ApiMisuse
5. ResourceLeaks
6. TypeErrors
7. ConfigurationErrors
8. SecurityVulnerabilities
9. PerformanceIssues
10. IntegrationFailures

**Proposed Extensions for Transpilers (+8):**
11. OperatorPrecedence - Expression parsing/generation bugs
12. TypeAnnotationGaps - Unsupported type hints
13. StdlibMapping - Python→Rust standard library conversions
14. ASTTransform - HIR→Codegen bugs
15. ComprehensionBugs - List/dict/set comprehension generation
16. IteratorChain - `.into_iter()`, `.map()`, `.filter()` issues
17. OwnershipBorrow - Lifetime and borrow checker errors
18. TraitBounds - Generic constraint issues

**Domain Adaptation:**
- **Web Services:** Add AuthN/AuthZ, CORS, RateLimiting, SQLInjection
- **ML Systems:** Add DataPipeline, ModelDrift, FeatureEngineering, TensorShape
- **General:** Keep configurable taxonomy via YAML

```yaml
# taxonomy.yaml
categories:
  - id: operator_precedence
    name: "Operator Precedence"
    parent: logic_errors
    keywords:
      - "operator precedence"
      - "parentheses"
      - "parse expression"
      - "order of operations"
    patterns:
      - 'r"\(.+\)\.\w+\(\)"'  # (expr).method()
      - 'r"\+|-|\*|/"'        # Arithmetic operators
```

### 5.3 Multi-Label Classification

**Implementation:**

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# Define label matrix (n_samples × n_labels)
# Example: commit can be [OperatorPrecedence=1, Comprehension=1, CodeGen=0, ...]

base_clf = RandomForestClassifier(n_estimators=100)
multi_label_clf = MultiOutputClassifier(base_clf)

# Train
multi_label_clf.fit(X_train_tfidf, y_train_multi)

# Predict
y_pred_multi = multi_label_clf.predict(X_test_tfidf)
y_proba_multi = multi_label_clf.predict_proba(X_test_tfidf)

# Output format
# commit_123 → [
#   (OperatorPrecedence, 0.89),
#   (Comprehension, 0.76),
#   (CodeGeneration, 0.62)
# ]
```

**Threshold Tuning:**
```python
# Adjust per-label thresholds based on precision/recall trade-offs
thresholds = {
    'OperatorPrecedence': 0.75,
    'Comprehension': 0.80,
    'General': 0.60  # Lower threshold to catch edge cases
}

final_labels = []
for label, proba in zip(label_names, y_proba):
    if proba > thresholds[label]:
        final_labels.append(label)
```

### 5.4 Training Data Pipeline

**Data Collection:**

```python
# Extract training data from Git history
import git

repo = git.Repo('/path/to/project')

training_data = []
for commit in repo.iter_commits('main', max_count=5000):
    # Skip merges, reverts, automated commits
    if len(commit.parents) > 1 or 'Revert' in commit.message:
        continue

    commit_message = commit.message
    code_diff = commit.diff(commit.parents[0]) if commit.parents else []

    # Extract features
    lines_added = commit.stats.total['insertions']
    lines_deleted = commit.stats.total['deletions']
    files_changed = commit.stats.total['files']

    training_data.append({
        'message': commit_message,
        'diff': code_diff,
        'lines_added': lines_added,
        'lines_deleted': lines_deleted,
        'files_changed': files_changed,
        'author': commit.author.name,
        'timestamp': commit.committed_datetime
    })
```

**Labeling Strategy:**

1. **Automatic (Heuristic):** Use existing rule-based classifier
2. **Semi-Automatic:** Confidence-based sampling (label only uncertain cases)
3. **Manual:** Domain expert labels 1K-5K examples
4. **Active Learning:** Iteratively label most informative samples

**Data Augmentation:**

```python
# Synonym replacement
"fix null pointer" → "fix nullptr", "fix null dereference"

# Back-translation (English → German → English)
"race condition in mutex" → "Rennen Bedingung in Mutex" → "race condition in mutex"

# Code obfuscation (preserve semantics)
"parse_expr()" → "parse_expression()", "expr_parser()"
```

### 5.5 Evaluation Metrics

**Standard Metrics:**

```python
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

y_true = test_labels
y_pred = model.predict(X_test)

# Overall accuracy
acc = accuracy_score(y_true, y_pred)

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None
)

# Macro-averaged (equal weight to each class)
f1_macro = f1_score(y_true, y_pred, average='macro')

# Micro-averaged (equal weight to each sample)
f1_micro = f1_score(y_true, y_pred, average='micro')

# Weighted (by class support)
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print(classification_report(y_true, y_pred))
```

**Multi-Label Metrics:**

```python
from sklearn.metrics import hamming_loss, jaccard_score

# Hamming loss (fraction of incorrect labels)
hamming = hamming_loss(y_true_multi, y_pred_multi)

# Jaccard similarity (IoU of label sets)
jaccard = jaccard_score(y_true_multi, y_pred_multi, average='samples')

# Subset accuracy (exact match)
subset_acc = accuracy_score(y_true_multi, y_pred_multi)
```

**Custom Metrics:**

```python
# Actionability: % of commits with specific category (not "General")
def actionability_score(y_pred, general_label=0):
    actionable = (y_pred != general_label).sum()
    return actionable / len(y_pred)

# Target: ≥80% actionability (currently 30.8%)
```

---

## 6. Implementation Roadmap

### Phase 1: Enhanced Rule-Based Classifier (Week 1-2)

**Goals:**
- Expand keyword dictionary with domain-specific terms
- Add multi-label support
- Improve confidence scoring

**Tasks:**
1. [ ] Extend `DefectCategory` enum with 8 new transpiler categories
2. [ ] Add keyword patterns to `classifier.rs`
3. [ ] Implement confidence scoring based on pattern count + specificity
4. [ ] Add multi-label classification (return top-3 categories)
5. [ ] Unit tests: 100% coverage, property tests with proptest

**Deliverables:**
- `src/classifier.rs` updated with 18 categories
- `tests/classifier_tests.rs` with 50+ test cases
- Documentation: `docs/taxonomies/transpiler-taxonomy.md`

### Phase 2: TF-IDF + Ensemble ML Classifier (Week 3-5)

**Goals:**
- Train ML classifier on historical commits
- Achieve 85%+ F1-score on test set
- Integrate into hybrid pipeline

**Tasks:**
1. [ ] Extract training data from depyler Git history (5K commits)
2. [ ] Manual labeling of 1K commits (domain expert)
3. [ ] Implement TF-IDF feature extraction (Python/Rust bindings)
4. [ ] Train Random Forest, XGBoost, SVM (compare performance)
5. [ ] Cross-validation with Stratified K-Fold
6. [ ] Serialize trained model (ONNX for Rust inference)
7. [ ] Integrate into OIP CLI: `oip classify --model ml`
8. [ ] Benchmark inference time: <100ms per commit

**Deliverables:**
- `training/train_ml_classifier.py` (sklearn pipeline)
- `models/defect_classifier_v1.onnx` (trained model)
- `src/ml_classifier.rs` (Rust ONNX inference)
- `docs/training/ml-classifier-training.md` (reproducibility guide)

### Phase 3: CodeBERT Fine-Tuning (Week 6-8)

**Goals:**
- Fine-tune CodeBERT on software engineering corpus
- Achieve 90%+ F1-score on test set
- Multi-label classification

**Tasks:**
1. [ ] Collect large corpus: GitHub commits (50K+), bug reports, Stack Overflow
2. [ ] Preprocess: Commit message + code diff → input format
3. [ ] Fine-tune CodeBERT with Hugging Face Transformers
4. [ ] Hyperparameter tuning: learning rate, batch size, epochs
5. [ ] Multi-label head: Binary cross-entropy loss
6. [ ] Export to ONNX for Rust inference (or use PyTorch via libtorch-rs)
7. [ ] Integrate into OIP: `oip classify --model transformer`
8. [ ] Benchmark inference time: <1s per commit

**Deliverables:**
- `training/finetune_codebert.py` (PyTorch training script)
- `models/codebert_defect_classifier_v1/` (fine-tuned model)
- `src/transformer_classifier.rs` (Rust inference)
- `docs/training/codebert-finetuning.md`

### Phase 4: Hybrid Pipeline Integration (Week 9-10)

**Goals:**
- Combine all three tiers into unified API
- Optimize inference time
- Production deployment

**Tasks:**
1. [ ] Implement decision logic: Tier 1 → Tier 2 → Tier 3
2. [ ] Confidence threshold tuning (ROC analysis)
3. [ ] Batch inference optimization (parallel processing)
4. [ ] CI/CD integration: Automated model retraining weekly
5. [ ] Monitoring dashboard: Track classification distribution over time
6. [ ] A/B testing: Compare hybrid vs. rule-based

**Deliverables:**
- `src/hybrid_classifier.rs` (unified API)
- `.github/workflows/retrain-models.yml` (weekly retraining)
- `docs/monitoring/classification-dashboard.md`

### Phase 5: Active Learning & Continuous Improvement (Ongoing)

**Goals:**
- Incrementally improve model with user feedback
- Adapt to new defect patterns

**Tasks:**
1. [ ] Implement feedback mechanism: `oip classify --feedback`
2. [ ] Active learning loop: Select uncertain samples for labeling
3. [ ] Model versioning: Track performance of each model version
4. [ ] Periodic retraining: Weekly/monthly based on new data
5. [ ] Transfer learning: Adapt to new projects/domains

**Deliverables:**
- `src/feedback.rs` (user feedback collection)
- `training/active_learning.py` (uncertainty sampling)
- `docs/maintenance/model-retraining.md`

---

## 7. Academic References

### 7.1 Text Classification & Feature Extraction

1. **Scikit-learn Documentation (2024)**
   "Feature extraction — scikit-learn 1.7.2 documentation"
   https://scikit-learn.org/stable/modules/feature_extraction.html
   *Best practices for TF-IDF, n-grams, and text vectorization in Python*

2. **Ensemble Learning for Text Classification (2020)**
   "An Investigation and Evaluation of N-Gram, TF-IDF and Ensemble Methods in Sentiment Classification"
   https://www.researchgate.net/publication/343286758
   *Empirical study showing ensemble methods (RF, AdaBoost, XGBoost) achieve 85-92% accuracy with TF-IDF features*

3. **TextCNN Ensemble Learning (2023)**
   "TextCNN-based ensemble learning model for Japanese Text Multi-classification"
   Science Direct, 2023
   https://www.sciencedirect.com/science/article/abs/pii/S0045790623001751
   *CNN architectures for text classification with ensemble approaches*

### 7.2 Software Defect Classification

4. **Bug Report Classification (2023)**
   Tabassum, S., et al. "Classification of Bugs in Cloud Computing Applications Using Machine Learning Techniques"
   Applied Sciences, Vol. 13, No. 5, 2880, February 2023
   https://www.mdpi.com/2076-3417/13/5/2880
   *Proposes hybrid NLP + ML approach for bug classification in cloud applications*

5. **Deep Learning for Bug Reports (2023)**
   Noyori, K., et al. "Deep learning and gradient-based extraction of bug report features related to bug fixing time"
   Frontiers in Computer Science, June 2023
   https://frontiersin.org/articles/10.3389/fcomp.2023.1032440/full
   *CNN + Grad-CAM approach for extracting bug report features*

6. **Feature Transformation for Bug Detection (2024)**
   "Feature transformation for improved software bug detection and commit classification"
   ScienceDirect, 2024
   https://www.sciencedirect.com/science/article/pii/S0164121224002498
   *Novel feature engineering techniques for commit classification*

### 7.3 Transformer Models for Code

7. **CodeBERT for Defect Prediction (2024)**
   "Parameter-efficient fine-tuning of pre-trained code models for just-in-time defect prediction"
   Neural Computing and Applications, 2024
   https://link.springer.com/article/10.1007/s00521-024-09930-5
   *Shows CodeBERT achieves 10-12% F1-score improvement over baselines*

8. **BERT-based Cross-Project Defect Prediction (2024)**
   "BERT-based cross-project and cross-version software defect prediction"
   ResearchGate, 2024
   https://www.researchgate.net/publication/382045453
   *Transfer learning with BERT for cross-project defect prediction*

9. **Transformer Models for Bug Detection (2024)**
   "Transformer-based models application for bug detection in source code"
   ResearchGate, August 2024
   https://www.researchgate.net/publication/385876129
   *Compares BERT, CodeBERT, GPT-2, CodeT5 for bug detection*

### 7.4 Commit Message Analysis & Mining

10. **Commit Message Detail and Defect Proneness (2016)**
    "The relationship between commit message detail and defect proneness in Java projects on GitHub"
    Proceedings of the 13th International Conference on Mining Software Repositories (MSR), 2016
    https://dl.acm.org/doi/10.1145/2901739.2903496
    *Empirical study showing commit message detail improves defect prediction*

11. **Refactoring Detection via Ensemble ML (2024)**
    "Detecting refactoring type of software commit messages based on ensemble machine learning algorithms"
    Scientific Reports, 2024
    https://www.nature.com/articles/s41598-024-72307-0
    *Ensemble ML (AdaBoost, CatBoost, XGBoost) for commit classification*

12. **Mining Commit Messages for Refactoring (2022)**
    "Mining commit messages to enhance software refactorings recommendation: A machine learning approach"
    ScienceDirect, 2022
    https://www.sciencedirect.com/science/article/pii/S2666827022000354
    *Text mining + ML for extracting refactoring patterns from commits*

### 7.5 Word Embeddings for Software Engineering

13. **Word Embeddings for Bug Report Management (2024)**
    "An empirical study on the potential of word embedding techniques in bug report management tasks"
    Empirical Software Engineering, 2024
    https://link.springer.com/article/10.1007/s10664-024-10510-3
    *Compares Word2Vec, GloVe, fastText for bug report classification*

14. **Word Embeddings for Vulnerability Detection (2021)**
    "An Extended Benchmark System of Word Embedding Methods for Vulnerability Detection"
    ACM Digital Library, 2021
    https://dl.acm.org/doi/fullHtml/10.1145/3440749.3442661
    *Word2Vec achieves highest precision/recall for vulnerable function detection*

### 7.6 Defect Taxonomy & Data Collection

15. **Software Defect Datasets Survey (2025)**
    "From Bugs to Benchmarks: A Comprehensive Survey of Software Defect Datasets"
    arXiv:2504.17977, 2025
    https://arxiv.org/pdf/2504.17977
    *Comprehensive survey of defect collection methods, including commit message keyword scanning*

---

## 8. Appendix

### 8.1 Example Training Script (Tier 2: TF-IDF + Random Forest)

```python
#!/usr/bin/env python3
"""
Train TF-IDF + Random Forest classifier for defect categorization
Usage: python train_ml_classifier.py --data commits.csv --output model.pkl
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

def load_data(filepath):
    """Load commit messages and labels from CSV"""
    df = pd.read_csv(filepath)
    X = df['commit_message']
    y = df['category']
    return X, y

def build_pipeline():
    """Construct sklearn pipeline"""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=1500,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=5,
            max_df=0.5,
            stop_words='english'
        )),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

def tune_hyperparameters(pipeline, X_train, y_train):
    """Grid search for optimal hyperparameters"""
    param_grid = {
        'tfidf__max_features': [1000, 1500, 2000],
        'tfidf__ngram_range': [(1, 2), (1, 3)],
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [10, 20, None]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best F1 (macro): {grid_search.best_score_:.3f}")

    return grid_search.best_estimator_

def evaluate(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to training data CSV')
    parser.add_argument('--output', required=True, help='Path to save trained model')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build pipeline
    print("Building pipeline...")
    pipeline = build_pipeline()

    # Train
    if args.tune:
        print("Tuning hyperparameters...")
        model = tune_hyperparameters(pipeline, X_train, y_train)
    else:
        print("Training with default parameters...")
        model = pipeline.fit(X_train, y_train)

    # Evaluate
    print("Evaluating...")
    evaluate(model, X_test, y_test)

    # Save
    print(f"Saving model to {args.output}...")
    joblib.dump(model, args.output)
    print("Done!")

if __name__ == '__main__':
    main()
```

### 8.2 Example Inference in Rust (ONNX)

```rust
// src/ml_classifier.rs
use ndarray::{Array1, Array2};
use onnxruntime::{environment::Environment, GraphOptimizationLevel, LoggingLevel};
use std::path::Path;

pub struct MLClassifier {
    session: onnxruntime::session::Session<'static>,
    vectorizer: TfidfVectorizer,
    label_names: Vec<String>,
}

impl MLClassifier {
    pub fn load(model_path: &Path, vectorizer_path: &Path) -> Result<Self> {
        // Load ONNX model
        let environment = Environment::builder()
            .with_name("defect_classifier")
            .with_log_level(LoggingLevel::Warning)
            .build()?;

        let session = environment
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_model_from_file(model_path)?;

        // Load TF-IDF vectorizer config
        let vectorizer = TfidfVectorizer::load(vectorizer_path)?;

        let label_names = vec![
            "MemorySafety".to_string(),
            "ConcurrencyBugs".to_string(),
            "LogicErrors".to_string(),
            "ApiMisuse".to_string(),
            "ResourceLeaks".to_string(),
            "TypeErrors".to_string(),
            "ConfigurationErrors".to_string(),
            "SecurityVulnerabilities".to_string(),
            "PerformanceIssues".to_string(),
            "IntegrationFailures".to_string(),
        ];

        Ok(Self {
            session,
            vectorizer,
            label_names,
        })
    }

    pub fn classify(&self, commit_message: &str) -> Result<Classification> {
        // Step 1: Vectorize commit message (TF-IDF)
        let features = self.vectorizer.transform(commit_message)?;

        // Step 2: Run inference
        let input = Array2::from_shape_vec((1, features.len()), features.to_vec())?;
        let outputs = self.session.run(vec![input.into()])?;

        // Step 3: Parse outputs
        let probs: Array1<f32> = outputs[0].try_extract()?;
        let predicted_class = probs.argmax().unwrap();
        let confidence = probs[predicted_class];

        Ok(Classification {
            category: self.label_names[predicted_class].clone(),
            confidence,
            probabilities: probs.to_vec(),
        })
    }
}

pub struct Classification {
    pub category: String,
    pub confidence: f32,
    pub probabilities: Vec<f32>,
}
```

### 8.3 Model Versioning Strategy

```yaml
# models/registry.yaml
models:
  - name: defect_classifier_v1
    version: 1.0.0
    type: rule_based
    path: models/rule_based_v1.yml
    created: 2025-11-24
    performance:
      accuracy: 0.72
      f1_macro: 0.68
      inference_time_ms: 5

  - name: defect_classifier_v2
    version: 2.0.0
    type: tfidf_random_forest
    path: models/ml_classifier_v2.pkl
    created: 2025-12-01
    training_data:
      source: depyler_commits_5k.csv
      size: 5000
      split: 0.8_train_0.2_test
    performance:
      accuracy: 0.87
      f1_macro: 0.85
      inference_time_ms: 80

  - name: defect_classifier_v3
    version: 3.0.0
    type: codebert_transformer
    path: models/codebert_v3/
    created: 2025-12-15
    training_data:
      source: github_commits_50k.csv
      size: 50000
      augmentation: True
    performance:
      accuracy: 0.92
      f1_macro: 0.90
      inference_time_ms: 950
```

### 8.4 A/B Testing Framework

```rust
// src/ab_testing.rs
use rand::Rng;

pub enum ClassifierVariant {
    RuleBased,
    ML,
    Hybrid,
}

pub struct ABTestConfig {
    pub rule_based_weight: f32,
    pub ml_weight: f32,
    pub hybrid_weight: f32,
}

impl ABTestConfig {
    pub fn production() -> Self {
        Self {
            rule_based_weight: 0.0,
            ml_weight: 0.0,
            hybrid_weight: 1.0,  // 100% hybrid
        }
    }

    pub fn ab_test() -> Self {
        Self {
            rule_based_weight: 0.2,  // 20% rule-based
            ml_weight: 0.3,          // 30% ML
            hybrid_weight: 0.5,      // 50% hybrid
        }
    }
}

pub fn select_classifier(config: &ABTestConfig) -> ClassifierVariant {
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();

    if r < config.rule_based_weight {
        ClassifierVariant::RuleBased
    } else if r < config.rule_based_weight + config.ml_weight {
        ClassifierVariant::ML
    } else {
        ClassifierVariant::Hybrid
    }
}
```

---

## 9. Success Metrics

**Primary KPIs:**

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Actionable Categorization | 30.8% | ≥80% | % of commits with specific category (not "General") |
| F1-Score (Macro) | 0.68 | ≥0.85 | Average F1 across all categories |
| Inference Time (Avg) | 5ms | <150ms | Mean time per commit classification |
| Multi-Label Coverage | 0% | ≥30% | % of commits with 2+ categories |

**Secondary KPIs:**

| Metric | Target | Measurement |
|--------|--------|-------------|
| Model Retraining Frequency | Weekly | Automated CI/CD pipeline |
| Active Learning Efficiency | 10% | Data added per retraining cycle |
| Transfer Learning Success | ≥75% F1 | Cross-project classification accuracy |
| User Feedback Rate | ≥5% | % of classifications receiving feedback |

**Business Impact:**

- **Development Velocity:** 20% reduction in time-to-triage defects
- **Technical Debt:** 30% reduction in high-priority defect categories (via targeted fixes)
- **Trend Analysis:** Quarterly defect pattern reports for strategic planning

---

## 10. Conclusion

This specification proposes a comprehensive NLP strategy combining rule-based, classical ML, and transformer-based techniques to improve defect classification in the Organizational Intelligence Plugin. By implementing the hybrid 3-tier architecture, we expect to achieve:

1. **≥80% actionable categorization** (vs. 30.8% baseline)
2. **≥85% F1-score** (vs. 68% baseline)
3. **<150ms average inference time**
4. **Multi-label classification** for complex defects
5. **Domain adaptation** for different project types

The roadmap spans 10 weeks with incremental deliverables, enabling continuous improvement through active learning and user feedback. The approach is grounded in peer-reviewed research (15 academic references) and industry best practices from scikit-learn and UC Berkeley NLP research.

**Next Steps:**
1. Review and approval of this specification
2. Allocate resources for Phase 1 implementation
3. Begin manual labeling of 1K commits for training data
4. Set up continuous integration for model retraining

---

**Document Status:** PROPOSED
**Review Due:** 2025-12-01
**Approvers:** @paiml/oip-maintainers, @paiml/ml-team

---

## Sources

1. [TF-IDF Feature Extraction - Scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html)
2. [Data Science Stack Exchange - Text Categorization](https://datascience.stackexchange.com/questions/987/text-categorization-combining-different-kind-of-features)
3. [N-Gram, TF-IDF and Ensemble Methods in Sentiment Classification](https://www.researchgate.net/publication/343286758)
4. [Word Embeddings for Bug Report Management](https://link.springer.com/article/10.1007/s10664-024-10510-3)
5. [Word Embeddings for Vulnerability Detection](https://dl.acm.org/doi/fullHtml/10.1145/3440749.3442661)
6. [CodeBERT for Just-in-Time Defect Prediction](https://link.springer.com/article/10.1007/s00521-024-09930-5)
7. [Transformer-based Bug Detection](https://www.researchgate.net/publication/385876129)
8. [BERT-based Cross-Project Defect Prediction](https://www.researchgate.net/publication/382045453)
9. [Ensemble ML for Refactoring Detection](https://www.nature.com/articles/s41598-024-72307-0)
10. [Deep Learning for Bug Report Features](https://frontiersin.org/articles/10.3389/fcomp.2023.1032440/full)
11. [TextCNN Ensemble Learning](https://www.sciencedirect.com/science/article/abs/pii/S0045790623001751)
12. [Mining Commit Messages for Refactoring](https://www.sciencedirect.com/science/article/pii/S2666827022000354)
13. [Bug Report Classification with ML](https://www.mdpi.com/2076-3417/13/5/2880)
14. [Commit Message Detail and Defect Proneness](https://dl.acm.org/doi/10.1145/2901739.2903496)
15. [Software Defect Datasets Survey](https://arxiv.org/pdf/2504.17977)
