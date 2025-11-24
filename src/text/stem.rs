//! Stemming algorithms for text normalization.
//!
//! Stemming reduces words to their root form by removing suffixes.
//! For example: "running" → "run", "better" → "better", "studies" → "studi".
//!
//! This module provides:
//! - Porter Stemmer (simplified implementation of the classic algorithm)
//! - Trait-based design for extensibility
//!
//! # Examples
//!
//! ```
//! use aprender::text::stem::{Stemmer, PorterStemmer};
//!
//! let stemmer = PorterStemmer::new();
//!
//! // Stem individual words
//! assert_eq!(stemmer.stem("running").expect("stem should succeed"), "run");
//! assert_eq!(stemmer.stem("studies").expect("stem should succeed"), "studi");
//!
//! // Stem multiple words
//! let words = vec!["running", "jumped", "easily"];
//! let stemmed = stemmer.stem_tokens(&words).expect("stem should succeed");
//! assert_eq!(stemmed, vec!["run", "jump", "easili"]);
//! ```
//!
//! # References
//!
//! Porter, M.F. (1980). "An algorithm for suffix stripping."
//! Program, 14(3), 130-137.

use crate::AprenderError;

/// Trait for stemming algorithms.
///
/// Stemmers reduce words to their root form by removing suffixes.
///
/// # Examples
///
/// ```
/// use aprender::text::stem::{Stemmer, PorterStemmer};
///
/// let stemmer = PorterStemmer::new();
/// assert_eq!(stemmer.stem("running").expect("stem should succeed"), "run");
/// assert_eq!(stemmer.stem("flies").expect("stem should succeed"), "fli");
/// ```
pub trait Stemmer {
    /// Stem a single word to its root form.
    ///
    /// # Arguments
    ///
    /// * `word` - Input word to stem
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - Stemmed word
    /// * `Err(AprenderError)` - If stemming fails
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::stem::{Stemmer, PorterStemmer};
    ///
    /// let stemmer = PorterStemmer::new();
    /// assert_eq!(stemmer.stem("running").expect("stem should succeed"), "run");
    /// ```
    fn stem(&self, word: &str) -> Result<String, AprenderError>;

    /// Stem multiple tokens.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Input tokens to stem
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<String>)` - Stemmed tokens
    /// * `Err(AprenderError)` - If stemming fails
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::stem::{Stemmer, PorterStemmer};
    ///
    /// let stemmer = PorterStemmer::new();
    /// let words = vec!["running", "flies", "easily"];
    /// let stemmed = stemmer.stem_tokens(&words).expect("stem should succeed");
    /// assert_eq!(stemmed, vec!["run", "fli", "easili"]);
    /// ```
    fn stem_tokens<S: AsRef<str>>(&self, tokens: &[S]) -> Result<Vec<String>, AprenderError> {
        tokens
            .iter()
            .map(|token| self.stem(token.as_ref()))
            .collect()
    }
}

/// Simplified Porter Stemmer implementation.
///
/// This is a streamlined version of the classic Porter Stemmer algorithm,
/// implementing the most common suffix removal rules from Steps 1-5.
///
/// # Examples
///
/// ```
/// use aprender::text::stem::{Stemmer, PorterStemmer};
///
/// let stemmer = PorterStemmer::new();
///
/// // Common suffixes
/// assert_eq!(stemmer.stem("running").expect("stem should succeed"), "run");
/// assert_eq!(stemmer.stem("flies").expect("stem should succeed"), "fli");
/// assert_eq!(stemmer.stem("studying").expect("stem should succeed"), "studi");
///
/// // Preserves short words
/// assert_eq!(stemmer.stem("sky").expect("stem should succeed"), "sky");
/// assert_eq!(stemmer.stem("is").expect("stem should succeed"), "is");
/// ```
#[derive(Debug, Clone, Default)]
pub struct PorterStemmer;

impl PorterStemmer {
    /// Create a new Porter Stemmer.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::stem::PorterStemmer;
    ///
    /// let stemmer = PorterStemmer::new();
    /// ```
    pub fn new() -> Self {
        Self
    }

    /// Check if a character is a vowel (a, e, i, o, u).
    fn is_vowel(c: char) -> bool {
        matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
    }

    /// Calculate the "measure" of a word (number of VC sequences).
    ///
    /// The measure is roughly the number of syllables in the word.
    /// Examples: "tree" = 0, "trees" = 0, "trouble" = 1, "troubles" = 1
    fn measure(word: &str) -> usize {
        let mut count = 0;
        let mut prev_is_vowel = false;

        for c in word.chars() {
            let is_vowel = Self::is_vowel(c);
            if !is_vowel && prev_is_vowel {
                count += 1;
            }
            prev_is_vowel = is_vowel;
        }

        count
    }

    /// Check if the word ends with a double consonant (e.g., "ss", "tt").
    fn ends_with_double_consonant(word: &str) -> bool {
        if word.len() < 2 {
            return false;
        }
        let chars: Vec<char> = word.chars().collect();
        let last = chars[chars.len() - 1];
        let second_last = chars[chars.len() - 2];
        !Self::is_vowel(last) && last == second_last
    }

    /// Check if the word ends with CVC pattern (consonant-vowel-consonant)
    /// where the final consonant is not w, x, or y.
    fn ends_with_cvc(word: &str) -> bool {
        if word.len() < 3 {
            return false;
        }
        let chars: Vec<char> = word.chars().collect();
        let len = chars.len();
        let last = chars[len - 1];
        let middle = chars[len - 2];
        let first = chars[len - 3];

        !Self::is_vowel(last)
            && Self::is_vowel(middle)
            && !Self::is_vowel(first)
            && !matches!(last, 'w' | 'x' | 'y')
    }

    /// Replace suffix if conditions are met.
    fn replace_suffix(word: &str, suffix: &str, replacement: &str, min_measure: usize) -> String {
        if let Some(stem) = word.strip_suffix(suffix) {
            if Self::measure(stem) > min_measure {
                return format!("{stem}{replacement}");
            }
        }
        word.to_string()
    }
}

impl Stemmer for PorterStemmer {
    #[allow(clippy::too_many_lines)]
    fn stem(&self, word: &str) -> Result<String, AprenderError> {
        // Convert to lowercase for processing
        let mut word = word.to_lowercase();

        // Skip very short words
        if word.len() <= 2 {
            return Ok(word);
        }

        // Step 1a: plurals and -ed, -ing
        if word.ends_with("sses") || word.ends_with("ies") {
            word = word[..word.len() - 2].to_string();
        } else if word.ends_with("ss") {
            // Keep ss
        } else if word.ends_with('s') && word.len() > 1 {
            word = word[..word.len() - 1].to_string();
        }

        // Step 1b: -ed, -ing
        let mut step1b_flag = false;
        if word.ends_with("eed") {
            let stem = &word[..word.len() - 3];
            if Self::measure(stem) > 0 {
                word = format!("{stem}ee");
            }
        } else if word.ends_with("ed") {
            let stem = &word[..word.len() - 2];
            if stem.chars().any(Self::is_vowel) {
                word = stem.to_string();
                step1b_flag = true;
            }
        } else if word.ends_with("ing") {
            let stem = &word[..word.len() - 3];
            if stem.chars().any(Self::is_vowel) {
                word = stem.to_string();
                step1b_flag = true;
            }
        }

        // Step 1b continued: handle special cases after removing -ed/-ing
        if step1b_flag {
            if word.ends_with("at") || word.ends_with("bl") || word.ends_with("iz") {
                word.push('e');
            } else if Self::ends_with_double_consonant(&word)
                && !word.ends_with('l')
                && !word.ends_with('s')
                && !word.ends_with('z')
            {
                word.pop();
            } else if Self::measure(&word) == 1 && Self::ends_with_cvc(&word) {
                word.push('e');
            }
        }

        // Step 1c: -y
        if word.ends_with('y') && word.len() > 1 {
            let stem = &word[..word.len() - 1];
            if stem.chars().any(Self::is_vowel) {
                word = format!("{stem}i");
            }
        }

        // Step 2: common suffixes
        word = Self::replace_suffix(&word, "ational", "ate", 0);
        word = Self::replace_suffix(&word, "tional", "tion", 0);
        word = Self::replace_suffix(&word, "enci", "ence", 0);
        word = Self::replace_suffix(&word, "anci", "ance", 0);
        word = Self::replace_suffix(&word, "izer", "ize", 0);
        word = Self::replace_suffix(&word, "abli", "able", 0);
        word = Self::replace_suffix(&word, "alli", "al", 0);
        word = Self::replace_suffix(&word, "entli", "ent", 0);
        word = Self::replace_suffix(&word, "eli", "e", 0);
        word = Self::replace_suffix(&word, "ousli", "ous", 0);
        word = Self::replace_suffix(&word, "ization", "ize", 0);
        word = Self::replace_suffix(&word, "ation", "ate", 0);
        word = Self::replace_suffix(&word, "ator", "ate", 0);
        word = Self::replace_suffix(&word, "alism", "al", 0);
        word = Self::replace_suffix(&word, "iveness", "ive", 0);
        word = Self::replace_suffix(&word, "fulness", "ful", 0);
        word = Self::replace_suffix(&word, "ousness", "ous", 0);
        word = Self::replace_suffix(&word, "aliti", "al", 0);
        word = Self::replace_suffix(&word, "iviti", "ive", 0);
        word = Self::replace_suffix(&word, "biliti", "ble", 0);

        // Step 3: more suffixes
        word = Self::replace_suffix(&word, "icate", "ic", 0);
        word = Self::replace_suffix(&word, "ative", "", 0);
        word = Self::replace_suffix(&word, "alize", "al", 0);
        word = Self::replace_suffix(&word, "iciti", "ic", 0);
        word = Self::replace_suffix(&word, "ical", "ic", 0);
        word = Self::replace_suffix(&word, "ful", "", 0);
        word = Self::replace_suffix(&word, "ness", "", 0);

        // Step 4: remove suffixes in longer words
        #[allow(clippy::if_same_then_else)]
        if Self::measure(&word) > 1 {
            if word.ends_with("al") {
                word = word[..word.len() - 2].to_string();
            } else if word.ends_with("ance") {
                word = word[..word.len() - 4].to_string();
            } else if word.ends_with("ence") {
                word = word[..word.len() - 4].to_string();
            } else if word.ends_with("er") {
                word = word[..word.len() - 2].to_string();
            } else if word.ends_with("ic") {
                word = word[..word.len() - 2].to_string();
            } else if word.ends_with("able") {
                word = word[..word.len() - 4].to_string();
            } else if word.ends_with("ible") {
                word = word[..word.len() - 4].to_string();
            } else if word.ends_with("ant") {
                word = word[..word.len() - 3].to_string();
            } else if word.ends_with("ement") {
                word = word[..word.len() - 5].to_string();
            } else if word.ends_with("ment") {
                word = word[..word.len() - 4].to_string();
            } else if word.ends_with("ent") {
                word = word[..word.len() - 3].to_string();
            } else if word.ends_with("ion") && word.len() > 3 {
                let prev = word.chars().nth(word.len() - 4);
                if matches!(prev, Some('s' | 't')) {
                    word = word[..word.len() - 3].to_string();
                }
            } else if word.ends_with("ou") {
                word = word[..word.len() - 2].to_string();
            } else if word.ends_with("ism") {
                word = word[..word.len() - 3].to_string();
            } else if word.ends_with("ate") {
                word = word[..word.len() - 3].to_string();
            } else if word.ends_with("iti") {
                word = word[..word.len() - 3].to_string();
            } else if word.ends_with("ous") {
                word = word[..word.len() - 3].to_string();
            } else if word.ends_with("ive") {
                word = word[..word.len() - 3].to_string();
            } else if word.ends_with("ize") {
                word = word[..word.len() - 3].to_string();
            }
        }

        // Step 5a: remove -e
        if word.ends_with('e') {
            let stem = &word[..word.len() - 1];
            let m = Self::measure(stem);
            if m > 1 || (m == 1 && !Self::ends_with_cvc(stem)) {
                word = stem.to_string();
            }
        }

        // Step 5b: remove double l
        if word.ends_with("ll") && Self::measure(&word) > 1 {
            word.pop();
        }

        Ok(word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== PorterStemmer Tests ==========

    #[test]
    fn test_porter_basic_plurals() {
        let stemmer = PorterStemmer::new();
        assert_eq!(stemmer.stem("cats").expect("stem should succeed"), "cat");
        assert_eq!(stemmer.stem("ponies").expect("stem should succeed"), "poni");
        assert_eq!(
            stemmer.stem("caresses").expect("stem should succeed"),
            "caress"
        );
    }

    #[test]
    fn test_porter_ed_ing() {
        let stemmer = PorterStemmer::new();
        assert_eq!(stemmer.stem("running").expect("stem should succeed"), "run");
        assert_eq!(stemmer.stem("jumped").expect("stem should succeed"), "jump");
        assert_eq!(
            stemmer.stem("skating").expect("stem should succeed"),
            "skate"
        );
    }

    #[test]
    fn test_porter_y_suffix() {
        let stemmer = PorterStemmer::new();
        assert_eq!(stemmer.stem("happy").expect("stem should succeed"), "happi");
        assert_eq!(stemmer.stem("sky").expect("stem should succeed"), "sky");
    }

    #[test]
    fn test_porter_common_words() {
        let stemmer = PorterStemmer::new();
        assert_eq!(
            stemmer.stem("studies").expect("stem should succeed"),
            "studi"
        );
        assert_eq!(
            stemmer.stem("studying").expect("stem should succeed"),
            "studi"
        );
        assert_eq!(stemmer.stem("flies").expect("stem should succeed"), "fli");
    }

    #[test]
    fn test_porter_short_words() {
        let stemmer = PorterStemmer::new();
        assert_eq!(stemmer.stem("is").expect("stem should succeed"), "is");
        assert_eq!(stemmer.stem("as").expect("stem should succeed"), "as");
        assert_eq!(stemmer.stem("at").expect("stem should succeed"), "at");
    }

    #[test]
    fn test_porter_empty_string() {
        let stemmer = PorterStemmer::new();
        assert_eq!(stemmer.stem("").expect("stem should succeed"), "");
    }

    #[test]
    fn test_porter_uppercase() {
        let stemmer = PorterStemmer::new();
        // Porter stemmer converts to lowercase
        assert_eq!(stemmer.stem("RUNNING").expect("stem should succeed"), "run");
        assert_eq!(stemmer.stem("Cats").expect("stem should succeed"), "cat");
    }

    #[test]
    fn test_porter_technical_words() {
        let stemmer = PorterStemmer::new();
        assert_eq!(
            stemmer.stem("computational").expect("stem should succeed"),
            "comput"
        );
        assert_eq!(
            stemmer.stem("relational").expect("stem should succeed"),
            "rel"
        );
    }

    #[test]
    fn test_stem_tokens() {
        let stemmer = PorterStemmer::new();
        let words = vec!["running", "cats", "easily"];
        let stemmed = stemmer
            .stem_tokens(&words)
            .expect("stem_tokens should succeed");
        assert_eq!(stemmed, vec!["run", "cat", "easili"]);
    }

    #[test]
    fn test_stem_tokens_empty() {
        let stemmer = PorterStemmer::new();
        let words: Vec<&str> = vec![];
        let stemmed = stemmer
            .stem_tokens(&words)
            .expect("stem_tokens should succeed");
        assert_eq!(stemmed, Vec::<String>::new());
    }

    #[test]
    fn test_stem_tokens_mixed() {
        let stemmer = PorterStemmer::new();
        let words = vec!["machine", "learning", "algorithms", "are", "powerful"];
        let stemmed = stemmer
            .stem_tokens(&words)
            .expect("stem_tokens should succeed");
        assert_eq!(stemmed, vec!["machin", "learn", "algorithm", "ar", "pow"]);
    }

    #[test]
    fn test_porter_default() {
        let stemmer = PorterStemmer;
        assert_eq!(stemmer.stem("running").expect("stem should succeed"), "run");
    }

    // ========== Helper Function Tests ==========

    #[test]
    fn test_is_vowel() {
        assert!(PorterStemmer::is_vowel('a'));
        assert!(PorterStemmer::is_vowel('e'));
        assert!(PorterStemmer::is_vowel('i'));
        assert!(PorterStemmer::is_vowel('o'));
        assert!(PorterStemmer::is_vowel('u'));
        assert!(!PorterStemmer::is_vowel('b'));
        assert!(!PorterStemmer::is_vowel('z'));
    }

    #[test]
    fn test_measure() {
        assert_eq!(PorterStemmer::measure("tree"), 0);
        assert_eq!(PorterStemmer::measure("trees"), 1);
        assert_eq!(PorterStemmer::measure("trouble"), 1);
        assert_eq!(PorterStemmer::measure("troubles"), 2);
    }

    #[test]
    fn test_ends_with_double_consonant() {
        assert!(PorterStemmer::ends_with_double_consonant("hopp"));
        assert!(PorterStemmer::ends_with_double_consonant("hiss"));
        assert!(!PorterStemmer::ends_with_double_consonant("hope"));
        assert!(!PorterStemmer::ends_with_double_consonant("hi"));
    }

    #[test]
    fn test_ends_with_cvc() {
        assert!(PorterStemmer::ends_with_cvc("hop"));
        assert!(!PorterStemmer::ends_with_cvc("hoop"));
        assert!(!PorterStemmer::ends_with_cvc("hi"));
    }
}
