# Text Preprocessing for NLP

Text preprocessing is the fundamental first step in Natural Language Processing (NLP) that transforms raw text into a structured format suitable for machine learning. This chapter demonstrates the core preprocessing techniques: tokenization, stop words filtering, and stemming.

## Theory

### The NLP Preprocessing Pipeline

Raw text data is noisy and unstructured. A typical preprocessing pipeline includes:

1. **Tokenization**: Split text into individual units (words, characters)
2. **Normalization**: Convert to lowercase, handle punctuation
3. **Stop Words Filtering**: Remove common words with little semantic value
4. **Stemming/Lemmatization**: Reduce words to their root form
5. **Vectorization**: Convert text to numerical features (TF-IDF, embeddings)

### Tokenization

**Definition**: The process of breaking text into smaller units called tokens.

**Tokenization Strategies:**

- **Whitespace Tokenization**: Split on Unicode whitespace (spaces, tabs, newlines)
  ```
  "Hello, world!" → ["Hello,", "world!"]
  ```

- **Word Tokenization**: Split on whitespace and separate punctuation
  ```
  "Hello, world!" → ["Hello", ",", "world", "!"]
  ```

- **Character Tokenization**: Split into individual characters
  ```
  "NLP" → ["N", "L", "P"]
  ```

### Stop Words Filtering

**Stop words** are common words (e.g., "the", "is", "at", "on") that:
- Appear frequently in text
- Carry minimal semantic meaning
- Can be removed to reduce noise and computational cost

**Example:**
```
Input:  "The quick brown fox jumps over the lazy dog"
Output: ["quick", "brown", "fox", "jumps", "lazy", "dog"]
```

**Benefits:**
- Reduces vocabulary size by 30-50%
- Improves signal-to-noise ratio
- Speeds up downstream ML algorithms
- Focuses on content words (nouns, verbs, adjectives)

### Stemming

**Stemming** reduces words to their root form by removing suffixes using heuristic rules.

**Porter Stemming Algorithm:**
Applies sequential rules to strip common English suffixes:

1. **Plural removal**: "cats" → "cat"
2. **Gerund removal**: "running" → "run"
3. **Comparative removal**: "happier" → "happi"
4. **Derivational endings**: "happiness" → "happi"

**Characteristics:**
- Fast and simple (rule-based)
- May produce non-words ("studies" → "studi")
- Good enough for information retrieval and search
- Language-specific rules

**vs. Lemmatization:**
Lemmatization uses dictionaries to return actual words ("running" → "run", "better" → "good"), but stemming is faster and often sufficient for ML tasks.

## Example 1: Tokenization Strategies

Comparing different tokenization approaches for the same text.

```rust,ignore
use aprender::text::tokenize::{WhitespaceTokenizer, WordTokenizer, CharTokenizer};
use aprender::text::Tokenizer;

fn main() {
    let text = "Hello, world! Natural Language Processing is amazing.";

    // Whitespace tokenization
    let whitespace_tokenizer = WhitespaceTokenizer::new();
    let tokens = whitespace_tokenizer.tokenize(text).unwrap();
    println!("Whitespace: {:?}", tokens);
    // ["Hello,", "world!", "Natural", "Language", "Processing", "is", "amazing."]

    // Word tokenization
    let word_tokenizer = WordTokenizer::new();
    let tokens = word_tokenizer.tokenize(text).unwrap();
    println!("Word: {:?}", tokens);
    // ["Hello", ",", "world", "!", "Natural", "Language", "Processing", "is", "amazing", "."]

    // Character tokenization
    let char_tokenizer = CharTokenizer::new();
    let tokens = char_tokenizer.tokenize("NLP").unwrap();
    println!("Character: {:?}", tokens);
    // ["N", "L", "P"]
}
```

**Output:**
```text
Whitespace: ["Hello,", "world!", "Natural", "Language", "Processing", "is", "amazing."]
Word: ["Hello", ",", "world", "!", "Natural", "Language", "Processing", "is", "amazing", "."]
Character: ["N", "L", "P"]
```

**Analysis:**
- **Whitespace**: 7 tokens, preserves punctuation
- **Word**: 10 tokens, separates punctuation
- **Character**: 3 tokens, character-level analysis

## Example 2: Stop Words Filtering

Removing common words to reduce noise and improve signal.

```rust,ignore
use aprender::text::stopwords::StopWordsFilter;
use aprender::text::tokenize::WhitespaceTokenizer;
use aprender::text::Tokenizer;

fn main() {
    let text = "The quick brown fox jumps over the lazy dog in the garden";

    // Tokenize
    let tokenizer = WhitespaceTokenizer::new();
    let tokens = tokenizer.tokenize(text).unwrap();
    println!("Original: {:?}", tokens);
    // ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "in", "the", "garden"]

    // Filter English stop words
    let filter = StopWordsFilter::english();
    let filtered = filter.filter(&tokens).unwrap();
    println!("Filtered: {:?}", filtered);
    // ["quick", "brown", "fox", "jumps", "lazy", "dog", "garden"]

    let reduction = 100.0 * (1.0 - filtered.len() as f64 / tokens.len() as f64);
    println!("Reduction: {:.1}%", reduction);  // 41.7%

    // Custom stop words
    let custom_filter = StopWordsFilter::new(vec!["fox", "dog", "garden"]);
    let custom_filtered = custom_filter.filter(&filtered).unwrap();
    println!("Custom filtered: {:?}", custom_filtered);
    // ["quick", "brown", "jumps", "lazy"]
}
```

**Output:**
```text
Original: ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "in", "the", "garden"]
Filtered: ["quick", "brown", "fox", "jumps", "lazy", "dog", "garden"]
Reduction: 41.7%
Custom filtered: ["quick", "brown", "jumps", "lazy"]
```

**Analysis:**
- Removed 5 stop words ("the", "over", "in")
- 41.7% reduction in token count
- Custom filtering enables domain-specific preprocessing

## Example 3: Stemming (Word Normalization)

Reducing words to their root form using Porter stemmer.

```rust,ignore
use aprender::text::stem::{PorterStemmer, Stemmer};

fn main() {
    let stemmer = PorterStemmer::new();

    // Single word stemming
    println!("running → {}", stemmer.stem("running").unwrap());  // "run"
    println!("studies → {}", stemmer.stem("studies").unwrap());  // "studi"
    println!("happiness → {}", stemmer.stem("happiness").unwrap());  // "happi"
    println!("easily → {}", stemmer.stem("easily").unwrap());  // "easili"

    // Batch stemming
    let words = vec!["running", "jumped", "flying", "studies", "cats", "quickly"];
    let stemmed = stemmer.stem_tokens(&words).unwrap();
    println!("Original: {:?}", words);
    println!("Stemmed:  {:?}", stemmed);
    // ["run", "jump", "flying", "studi", "cat", "quickli"]
}
```

**Output:**
```text
running → run
studies → studi
happiness → happi
easily → easili
Original: ["running", "jumped", "flying", "studies", "cats", "quickly"]
Stemmed:  ["run", "jump", "flying", "studi", "cat", "quickli"]
```

**Analysis:**
- Normalizes word variations: "running"/"run", "studies"/"studi"
- May produce non-words: "happiness" → "happi"
- Groups semantically similar words together
- Reduces vocabulary size for ML models

## Example 4: Complete Preprocessing Pipeline

End-to-end pipeline combining tokenization, normalization, filtering, and stemming.

```rust,ignore
use aprender::text::stem::{PorterStemmer, Stemmer};
use aprender::text::stopwords::StopWordsFilter;
use aprender::text::tokenize::WordTokenizer;
use aprender::text::Tokenizer;

fn main() {
    let document = "The students are studying machine learning algorithms. \
                    They're analyzing different classification models and \
                    comparing their performances on various datasets.";

    // Step 1: Tokenization
    let tokenizer = WordTokenizer::new();
    let tokens = tokenizer.tokenize(document).unwrap();
    println!("Tokens: {} items", tokens.len());  // 21 tokens

    // Step 2: Lowercase normalization
    let lowercase_tokens: Vec<String> = tokens
        .iter()
        .map(|t| t.to_lowercase())
        .collect();

    // Step 3: Stop words filtering
    let filter = StopWordsFilter::english();
    let filtered_tokens = filter.filter(&lowercase_tokens).unwrap();
    println!("After filtering: {} items", filtered_tokens.len());  // 16 tokens

    // Step 4: Stemming
    let stemmer = PorterStemmer::new();
    let stemmed_tokens = stemmer.stem_tokens(&filtered_tokens).unwrap();

    println!("Final: {:?}", stemmed_tokens);
    // ["stud", "studi", "machin", "learn", "algorithm", ".", "they'r",
    //  "analyz", "differ", "classif", "model", "compar", "perform",
    //  "variou", "dataset", "."]

    let reduction = 100.0 * (1.0 - stemmed_tokens.len() as f64 / tokens.len() as f64);
    println!("Total reduction: {:.1}%", reduction);  // 23.8%
}
```

**Output:**
```text
Tokens: 21 items
After filtering: 16 items
Final: ["stud", "studi", "machin", "learn", "algorithm", ".", "they'r", "analyz", "differ", "classif", "model", "compar", "perform", "variou", "dataset", "."]
Total reduction: 23.8%
```

**Pipeline Analysis:**

| Stage | Token Count | Change |
|-------|-------------|--------|
| Original | 21 | - |
| Lowercase | 21 | 0% |
| Stop words | 16 | -23.8% |
| Stemming | 16 | 0% |

**Key Transformations:**
- "students" → "stud"
- "studying" → "studi"
- "machine" → "machin"
- "learning" → "learn"
- "algorithms" → "algorithm"
- "analyzing" → "analyz"
- "classification" → "classif"

## Best Practices

### When to Use Each Technique

**Tokenization:**
- Whitespace: Quick analysis, sentiment analysis
- Word: Most NLP tasks, classification, named entity recognition
- Character: Character-level models, language modeling

**Stop Words Filtering:**
- ✅ Information retrieval, topic modeling, keyword extraction
- ❌ Sentiment analysis (negation words like "not" matter)
- ❌ Question answering (question words like "what", "where")

**Stemming:**
- ✅ Search engines, information retrieval
- ✅ Text classification with large vocabularies
- ❌ Tasks requiring exact word meaning
- Consider lemmatization for better quality (at cost of speed)

### Pipeline Recommendations

**Fast & Simple (Search/Retrieval):**
```
Text → Whitespace → Lowercase → Stop words → Stemming
```

**High Quality (Classification):**
```
Text → Word tokenization → Lowercase → Stop words → Lemmatization
```

**Character-Level (Language Models):**
```
Text → Character tokenization → No further preprocessing
```

## Running the Example

```bash
cargo run --example text_preprocessing
```

The example demonstrates four scenarios:
1. **Tokenization strategies** - Comparing whitespace, word, and character tokenizers
2. **Stop words filtering** - English and custom stop word removal
3. **Stemming** - Porter algorithm for word normalization
4. **Full pipeline** - Complete preprocessing workflow

## Key Takeaways

1. **Preprocessing is crucial**: Directly impacts ML model performance
2. **Pipeline matters**: Order of operations affects results
3. **Trade-offs exist**: Speed vs. quality, simplicity vs. accuracy
4. **Domain-specific**: Customize for your task (sentiment vs. search)
5. **Reproducibility**: Same pipeline for training and inference

## Next Steps

After preprocessing, text is ready for:
- **Vectorization**: Bag of Words, TF-IDF, word embeddings
- **Feature engineering**: N-grams, POS tags, named entities
- **Model training**: Classification, clustering, topic modeling

## References

- Porter, M.F. (1980). "An algorithm for suffix stripping." *Program*, 14(3), 130-137.
- Manning, C.D., Raghavan, P., Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
- Jurafsky, D., Martin, J.H. (2023). *Speech and Language Processing* (3rd ed.).
