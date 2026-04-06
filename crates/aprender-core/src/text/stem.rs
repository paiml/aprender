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
    #[must_use]
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

impl PorterStemmer {
    /// Step 1: Handle plurals, past tense, and progressive suffixes
    fn apply_step1(word: &mut String) {
        // Step 1a: plurals
        if word.ends_with("sses") || word.ends_with("ies") {
            word.truncate(word.len() - 2);
        } else if !word.ends_with("ss") && word.ends_with('s') && word.len() > 1 {
            word.truncate(word.len() - 1);
        }

        // Step 1b: -eed, -ed, -ing
        let mut step1b_flag = false;
        if word.ends_with("eed") {
            let stem = &word[..word.len() - 3];
            if Self::measure(stem) > 0 {
                word.truncate(word.len() - 1); // eed → ee
            }
        } else if word.ends_with("ed") {
            let stem_len = word.len() - 2;
            if word[..stem_len].chars().any(Self::is_vowel) {
                word.truncate(stem_len);
                step1b_flag = true;
            }
        } else if word.ends_with("ing") {
            let stem_len = word.len() - 3;
            if word[..stem_len].chars().any(Self::is_vowel) {
                word.truncate(stem_len);
                step1b_flag = true;
            }
        }

        if step1b_flag {
            Self::apply_step1b_fixup(word);
        }

        // Step 1c: -y → -i
        if word.ends_with('y')
            && word.len() > 1
            && word[..word.len() - 1].chars().any(Self::is_vowel)
        {
            word.truncate(word.len() - 1);
            word.push('i');
        }
    }

    /// Step 1b fixup: adjust word after removing -ed/-ing
    fn apply_step1b_fixup(word: &mut String) {
        if word.ends_with("at") || word.ends_with("bl") || word.ends_with("iz") {
            word.push('e');
        } else if Self::ends_with_double_consonant(word)
            && !word.ends_with('l')
            && !word.ends_with('s')
            && !word.ends_with('z')
        {
            word.pop();
        } else if Self::measure(word) == 1 && Self::ends_with_cvc(word) {
            word.push('e');
        }
    }

    /// Step 4: Remove suffixes from words with measure > 1
    fn apply_step4(word: &mut String) {
        if Self::measure(word) <= 1 {
            return;
        }
        // Suffix → trim length (checked in order of specificity)
        const SUFFIXES: &[(&str, usize)] = &[
            ("ement", 5),
            ("ance", 4),
            ("ence", 4),
            ("able", 4),
            ("ible", 4),
            ("ment", 4),
            ("ant", 3),
            ("ent", 3),
            ("ism", 3),
            ("ate", 3),
            ("iti", 3),
            ("ous", 3),
            ("ive", 3),
            ("ize", 3),
            ("al", 2),
            ("er", 2),
            ("ic", 2),
            ("ou", 2),
        ];
        for &(suffix, trim) in SUFFIXES {
            if word.ends_with(suffix) {
                word.truncate(word.len() - trim);
                return;
            }
        }
        // Special case: -ion requires preceding s or t
        if word.ends_with("ion")
            && word.len() > 3
            && matches!(word.as_bytes().get(word.len() - 4), Some(b's' | b't'))
        {
            word.truncate(word.len() - 3);
        }
    }
}

impl Stemmer for PorterStemmer {
    fn stem(&self, word: &str) -> Result<String, AprenderError> {
        let mut word = word.to_lowercase();
        if word.len() <= 2 {
            return Ok(word);
        }

        Self::apply_step1(&mut word);

        // Step 2: derivational suffixes
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

        // Step 3: more derivational suffixes
        word = Self::replace_suffix(&word, "icate", "ic", 0);
        word = Self::replace_suffix(&word, "ative", "", 0);
        word = Self::replace_suffix(&word, "alize", "al", 0);
        word = Self::replace_suffix(&word, "iciti", "ic", 0);
        word = Self::replace_suffix(&word, "ical", "ic", 0);
        word = Self::replace_suffix(&word, "ful", "", 0);
        word = Self::replace_suffix(&word, "ness", "", 0);

        Self::apply_step4(&mut word);

        // Step 5a: remove final -e
        if word.ends_with('e') {
            let m = Self::measure(&word[..word.len() - 1]);
            if m > 1 || (m == 1 && !Self::ends_with_cvc(&word[..word.len() - 1])) {
                word.truncate(word.len() - 1);
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
#[path = "stem_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests_stem_contract.rs"]
mod tests_stem_contract;
