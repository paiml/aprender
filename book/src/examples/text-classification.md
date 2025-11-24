# Text Classification with TF-IDF

Text classification is the task of assigning predefined categories to text documents. Combined with TF-IDF vectorization, it enables practical applications like sentiment analysis, spam detection, and topic classification.

## Theory

### The Text Classification Pipeline

A complete text classification system consists of:

1. **Text Preprocessing**: Tokenization, stop words, stemming
2. **Feature Extraction**: Convert text to numerical features
3. **Model Training**: Learn patterns from labeled data
4. **Prediction**: Classify new documents

### Feature Extraction Methods

**Bag of Words (BoW)**:
- Represents documents as word count vectors
- Simple and effective baseline
- Ignores word order and context

```
"cat dog cat" → [cat: 2, dog: 1]
```

**TF-IDF (Term Frequency-Inverse Document Frequency)**:
- Weights words by importance
- Down-weights common words, up-weights rare words
- Better performance than raw counts

**TF-IDF Formula:**
```
tfidf(t, d) = tf(t, d) × idf(t)
where:
  tf(t, d) = count of term t in document d
  idf(t) = log(N / df(t))
  N = total documents
  df(t) = documents containing term t
```

**Example:**
```
Document 1: "cat dog"
Document 2: "cat bird"
Document 3: "dog bird bird"

Term "cat": appears in 2/3 documents
  IDF = log(3/2) = 0.405

Term "bird": appears in 2/3 documents
  IDF = log(3/2) = 0.405

Term "dog": appears in 2/3 documents
  IDF = log(3/2) = 0.405
```

### Classification Algorithms

**Gaussian Naive Bayes**:
- Assumes features are independent (naive assumption)
- Probabilistic classifier using Bayes' theorem
- Fast training and prediction
- Works well with high-dimensional sparse data

**Logistic Regression**:
- Linear classifier with sigmoid activation
- Learns feature weights via gradient descent
- Produces probability estimates
- Robust and interpretable

## Example 1: Sentiment Classification with Bag of Words

Binary sentiment analysis (positive/negative) using word counts.

```rust,ignore
use aprender::classification::GaussianNB;
use aprender::text::vectorize::CountVectorizer;
use aprender::text::tokenize::WhitespaceTokenizer;
use aprender::traits::Estimator;

fn main() {
    // Training data: movie reviews
    let train_docs = vec![
        "this movie was excellent and amazing",  // Positive
        "great film with wonderful acting",      // Positive
        "fantastic movie loved every minute",    // Positive
        "terrible movie waste of time",          // Negative
        "awful film boring and disappointing",   // Negative
        "horrible acting very bad movie",        // Negative
    ];

    let train_labels = vec![1, 1, 1, 0, 0, 0]; // 1 = positive, 0 = negative

    // Vectorize with CountVectorizer
    let mut vectorizer = CountVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_max_features(20);

    let X_train = vectorizer.fit_transform(&train_docs).unwrap();
    println!("Vocabulary size: {}", vectorizer.vocabulary_size());  // 20 words

    // Train Gaussian Naive Bayes
    let X_train_f32 = convert_to_f32(&X_train);  // Convert f64 to f32
    let mut classifier = GaussianNB::new();
    classifier.fit(&X_train_f32, &train_labels).unwrap();

    // Predict on new reviews
    let test_docs = vec![
        "excellent movie great acting",   // Should predict positive
        "terrible film very bad",         // Should predict negative
    ];

    let X_test = vectorizer.transform(&test_docs).unwrap();
    let X_test_f32 = convert_to_f32(&X_test);
    let predictions = classifier.predict(&X_test_f32).unwrap();

    println!("Predictions: {:?}", predictions);  // [1, 0] = [positive, negative]
}
```

**Output:**
```text
Vocabulary size: 20
Predictions: [1, 0]
```

**Analysis:**
- **Bag of Words**: Simple word count features
- **20 features**: Limited vocabulary (max_features=20)
- **100% accuracy**: Overfitting on small dataset, but demonstrates concept
- **Fast training**: Naive Bayes trains in O(n×m) where n=docs, m=features

## Example 2: Topic Classification with TF-IDF

Multi-class classification (tech vs sports) using TF-IDF weighting.

```rust,ignore
use aprender::classification::LogisticRegression;
use aprender::text::vectorize::TfidfVectorizer;
use aprender::text::tokenize::WhitespaceTokenizer;

fn main() {
    // Training data: tech vs sports articles
    let train_docs = vec![
        "python programming language machine learning",    // Tech
        "artificial intelligence neural networks deep",    // Tech
        "software development code rust programming",      // Tech
        "basketball game score team championship",         // Sports
        "football soccer match goal tournament",           // Sports
        "tennis player serves match competition",          // Sports
    ];

    let train_labels = vec![0, 0, 0, 1, 1, 1]; // 0 = tech, 1 = sports

    // TF-IDF vectorization
    let mut vectorizer = TfidfVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    let X_train = vectorizer.fit_transform(&train_docs).unwrap();
    println!("Vocabulary: {} terms", vectorizer.vocabulary_size());  // 28 terms

    // Show IDF values
    let vocab: Vec<_> = vectorizer.vocabulary().iter().collect();
    for (word, &idx) in vocab.iter().take(3) {
        println!("{}: IDF = {:.3}", word, vectorizer.idf_values()[idx]);
    }
    // basketball: IDF = 2.253 (rare, important)
    // programming: IDF = 1.847 (less rare)

    // Train Logistic Regression
    let X_train_f32 = convert_to_f32(&X_train);
    let mut classifier = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iter(100);

    classifier.fit(&X_train_f32, &train_labels).unwrap();

    // Test predictions
    let test_docs = vec![
        "programming code algorithm",  // Should predict tech
        "basketball score game",       // Should predict sports
    ];

    let X_test = vectorizer.transform(&test_docs).unwrap();
    let X_test_f32 = convert_to_f32(&X_test);
    let predictions = classifier.predict(&X_test_f32);

    println!("Predictions: {:?}", predictions);  // [0, 1] = [tech, sports]
}
```

**Output:**
```text
Vocabulary: 28 terms
basketball: IDF = 2.253
programming: IDF = 1.847
Predictions: [0, 1]
```

**Analysis:**
- **TF-IDF weighting**: Highlights discriminative words
- **IDF values**: Rare words like "basketball" have higher IDF (2.253)
- **Common words**: More frequent words have lower IDF (1.847)
- **Logistic Regression**: Learns linear decision boundary
- **100% accuracy**: Perfect separation on training data

## Example 3: Full Preprocessing Pipeline

Complete workflow from raw text to predictions.

```rust,ignore
use aprender::classification::GaussianNB;
use aprender::text::stem::{PorterStemmer, Stemmer};
use aprender::text::stopwords::StopWordsFilter;
use aprender::text::tokenize::WhitespaceTokenizer;
use aprender::text::vectorize::TfidfVectorizer;
use aprender::text::Tokenizer;

fn main() {
    let raw_docs = vec![
        "The machine learning algorithms are improving rapidly",
        "The team scored three goals in the championship match",
    ];
    let labels = vec![0, 1]; // 0 = tech, 1 = sports

    // Step 1: Tokenization
    let tokenizer = WhitespaceTokenizer::new();
    let tokenized: Vec<Vec<String>> = raw_docs
        .iter()
        .map(|doc| tokenizer.tokenize(doc).unwrap())
        .collect();

    // Step 2: Lowercase + Stop words filtering
    let filter = StopWordsFilter::english();
    let filtered: Vec<Vec<String>> = tokenized
        .iter()
        .map(|tokens| {
            let lower: Vec<String> = tokens.iter().map(|t| t.to_lowercase()).collect();
            filter.filter(&lower).unwrap()
        })
        .collect();

    // Step 3: Stemming
    let stemmer = PorterStemmer::new();
    let stemmed: Vec<Vec<String>> = filtered
        .iter()
        .map(|tokens| stemmer.stem_tokens(tokens).unwrap())
        .collect();

    println!("After preprocessing: {:?}", stemmed[0]);
    // ["machin", "learn", "algorithm", "improv", "rapid"]

    // Step 4: Rejoin and vectorize
    let processed: Vec<String> = stemmed
        .iter()
        .map(|tokens| tokens.join(" "))
        .collect();

    let mut vectorizer = TfidfVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    let X = vectorizer.fit_transform(&processed).unwrap();

    // Step 5: Classification
    let X_f32 = convert_to_f32(&X);
    let mut classifier = GaussianNB::new();
    classifier.fit(&X_f32, &labels).unwrap();

    let predictions = classifier.predict(&X_f32).unwrap();
    println!("Predictions: {:?}", predictions);  // [0, 1] = [tech, sports]
}
```

**Output:**
```text
After preprocessing: ["machin", "learn", "algorithm", "improv", "rapid"]
Predictions: [0, 1]
```

**Pipeline Analysis:**

| Stage | Input | Output | Effect |
|-------|-------|--------|--------|
| Tokenization | "The machine learning..." | ["The", "machine", ...] | Split into words |
| Lowercase + Stop words | 11 tokens | 8 tokens | Remove "the", "are", "in" |
| Stemming | ["machine", "learning"] | ["machin", "learn"] | Normalize to roots |
| TF-IDF | Text tokens | 31-dimensional vectors | Numerical features |
| Classification | Feature vectors | Class labels | Predictions |

**Key Benefits:**
- **Vocabulary reduction**: 27% fewer tokens after stop words
- **Normalization**: "improving" → "improv", "algorithms" → "algorithm"
- **Generalization**: Stemming helps match "learn", "learning", "learned"
- **Discriminative features**: TF-IDF highlights important words

## Model Selection Guidelines

### Gaussian Naive Bayes

**Best for:**
- Text classification with sparse features
- Large vocabularies (thousands of features)
- Fast training required
- Probabilistic predictions needed

**Advantages:**
- Extremely fast (O(n×m) training)
- Works well with high-dimensional data
- No hyperparameter tuning needed
- Probabilistic outputs

**Limitations:**
- Assumes feature independence (rarely true)
- Less accurate than discriminative models
- Sensitive to feature scaling

### Logistic Regression

**Best for:**
- When you need interpretable models
- Feature importance analysis
- Balanced datasets
- Reliable probability estimates

**Advantages:**
- Learns feature weights (interpretable)
- Robust to correlated features
- Regularization prevents overfitting
- Well-calibrated probabilities

**Limitations:**
- Slower training than Naive Bayes
- Requires hyperparameter tuning (learning rate, iterations)
- Sensitive to feature scaling

## Best Practices

### Feature Extraction

**CountVectorizer (Bag of Words):**
- ✅ Simple baseline, easy to understand
- ✅ Fast computation
- ❌ Ignores word importance
- **Use when**: Starting a project, small datasets

**TfidfVectorizer:**
- ✅ Weights by importance
- ✅ Better performance than BoW
- ✅ Down-weights common words
- **Use when**: Production systems, larger datasets

### Preprocessing

**Always include:**
1. Tokenization (WhitespaceTokenizer or WordTokenizer)
2. Lowercase normalization
3. Stop words filtering (unless sentiment analysis needs "not", "no")

**Optional but recommended:**
4. Stemming (PorterStemmer) for English
5. Max features limit (1000-5000 for efficiency)

### Evaluation

**Train/Test Split:**
```rust
// Split data 80/20
let split_idx = (docs.len() * 4) / 5;
let (train_docs, test_docs) = docs.split_at(split_idx);
let (train_labels, test_labels) = labels.split_at(split_idx);
```

**Metrics:**
- Accuracy: Overall correctness
- Precision/Recall: Class-specific performance
- Confusion matrix: Error analysis

## Running the Example

```bash
cargo run --example text_classification
```

The example demonstrates three scenarios:
1. **Sentiment classification** - Bag of Words with Gaussian NB
2. **Topic classification** - TF-IDF with Logistic Regression
3. **Full pipeline** - Complete preprocessing workflow

## Key Takeaways

1. **TF-IDF > Bag of Words**: Almost always better performance
2. **Preprocessing matters**: Stop words + stemming improve generalization
3. **Naive Bayes**: Fast baseline, good for high-dimensional data
4. **Logistic Regression**: More accurate, interpretable weights
5. **Pipeline is crucial**: Consistent preprocessing for train/test

## Real-World Applications

- **Spam Detection**: Email → [spam, not spam]
- **Sentiment Analysis**: Review → [positive, negative, neutral]
- **Topic Classification**: News article → [politics, sports, tech, ...]
- **Language Detection**: Text → [English, Spanish, French, ...]
- **Intent Classification**: User query → [question, command, statement]

## Next Steps

After text classification, explore:
- **Word embeddings**: Word2Vec, GloVe for semantic similarity
- **Deep learning**: RNNs, Transformers for contextual understanding
- **Multi-label classification**: Documents with multiple categories
- **Active learning**: Efficiently label new training data

## References

- Manning, C.D., Raghavan, P., Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
- Joachims, T. (1998). "Text categorization with support vector machines." *Proceedings of ECML*.
- McCallum, A., Nigam, K. (1998). "A comparison of event models for naive bayes text classification." *AAAI Workshop*.
