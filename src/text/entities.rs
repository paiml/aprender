//! Pattern-based entity extraction.
//!
//! This module provides rule-based extraction of common entities:
//! - Email addresses
//! - URLs (http/https)
//! - Phone numbers (US format)
//! - Mentions (@username)
//! - Hashtags (#topic)
//! - Keywords (TF-IDF based)
//!
//! # Quick Start
//!
//! ```
//! use aprender::text::entities::EntityExtractor;
//!
//! let extractor = EntityExtractor::new();
//! let text = "Contact john@example.com or visit https://example.com";
//!
//! let entities = extractor.extract(text).unwrap();
//! assert!(!entities.emails.is_empty());
//! assert!(!entities.urls.is_empty());
//! ```

use crate::AprenderError;

/// Extracted entities from text.
#[derive(Debug, Clone, PartialEq)]
pub struct Entities {
    /// Email addresses found
    pub emails: Vec<String>,
    /// URLs found (http/https)
    pub urls: Vec<String>,
    /// Phone numbers found
    pub phone_numbers: Vec<String>,
    /// Mentions (@username)
    pub mentions: Vec<String>,
    /// Hashtags (#topic)
    pub hashtags: Vec<String>,
    /// Potential named entities (capitalized words)
    pub named_entities: Vec<String>,
}

impl Entities {
    /// Create empty entities.
    #[must_use]
    pub fn new() -> Self {
        Self {
            emails: Vec::new(),
            urls: Vec::new(),
            phone_numbers: Vec::new(),
            mentions: Vec::new(),
            hashtags: Vec::new(),
            named_entities: Vec::new(),
        }
    }

    /// Check if any entities were found.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.emails.is_empty()
            && self.urls.is_empty()
            && self.phone_numbers.is_empty()
            && self.mentions.is_empty()
            && self.hashtags.is_empty()
            && self.named_entities.is_empty()
    }

    /// Count total entities found.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.emails.len()
            + self.urls.len()
            + self.phone_numbers.len()
            + self.mentions.len()
            + self.hashtags.len()
            + self.named_entities.len()
    }
}

impl Default for Entities {
    fn default() -> Self {
        Self::new()
    }
}

/// Pattern-based entity extractor.
///
/// Uses regular expressions and heuristics to extract common entities.
///
/// # Examples
///
/// ```
/// use aprender::text::entities::EntityExtractor;
///
/// let extractor = EntityExtractor::new();
/// let text = "Email me at test@example.com or call 555-123-4567";
///
/// let entities = extractor.extract(text).unwrap();
/// assert_eq!(entities.emails.len(), 1);
/// assert_eq!(entities.phone_numbers.len(), 1);
/// ```
#[derive(Debug)]
pub struct EntityExtractor {
    /// Extract named entities (capitalized words)
    extract_named_entities: bool,
}

impl EntityExtractor {
    /// Create a new entity extractor.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::entities::EntityExtractor;
    ///
    /// let extractor = EntityExtractor::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            extract_named_entities: true,
        }
    }

    /// Set whether to extract named entities.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::entities::EntityExtractor;
    ///
    /// let extractor = EntityExtractor::new().with_named_entities(false);
    /// ```
    #[must_use]
    pub fn with_named_entities(mut self, enable: bool) -> Self {
        self.extract_named_entities = enable;
        self
    }

    /// Extract all entities from text.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to process
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::entities::EntityExtractor;
    ///
    /// let extractor = EntityExtractor::new();
    /// let text = "Visit https://example.com or email info@example.com";
    ///
    /// let entities = extractor.extract(text).unwrap();
    /// assert!(!entities.urls.is_empty());
    /// assert!(!entities.emails.is_empty());
    /// ```
    pub fn extract(&self, text: &str) -> Result<Entities, AprenderError> {
        let mut entities = Entities::new();

        entities.emails = EntityExtractor::extract_emails(text);
        entities.urls = EntityExtractor::extract_urls(text);
        entities.phone_numbers = self.extract_phone_numbers(text);
        entities.mentions = EntityExtractor::extract_mentions(text);
        entities.hashtags = EntityExtractor::extract_hashtags(text);

        if self.extract_named_entities {
            entities.named_entities = EntityExtractor::extract_capitalized_words(text);
        }

        Ok(entities)
    }

    /// Extract email addresses.
    fn extract_emails(text: &str) -> Vec<String> {
        let mut emails = Vec::new();

        // Simple email regex: word@word.word
        for word in text.split_whitespace() {
            if EntityExtractor::is_email(word) {
                emails.push(word.to_string());
            }
        }

        emails
    }

    /// Check if string looks like an email.
    fn is_email(s: &str) -> bool {
        // Must contain exactly one @
        let at_count = s.chars().filter(|&c| c == '@').count();
        if at_count != 1 {
            return false;
        }

        // Split on @
        let parts: Vec<&str> = s.split('@').collect();
        if parts.len() != 2 {
            return false;
        }

        let local = parts[0];
        let domain = parts[1];

        // Local part must not be empty
        if local.is_empty() {
            return false;
        }

        // Domain must contain a dot and not be empty
        if !domain.contains('.') || domain.is_empty() {
            return false;
        }

        // Domain parts must not be empty
        let domain_parts: Vec<&str> = domain.split('.').collect();
        if domain_parts.iter().any(|p| p.is_empty()) {
            return false;
        }

        true
    }

    /// Extract URLs (http/https).
    fn extract_urls(text: &str) -> Vec<String> {
        let mut urls = Vec::new();

        for word in text.split_whitespace() {
            if word.starts_with("http://") || word.starts_with("https://") {
                // Remove trailing punctuation
                let url = word.trim_end_matches(|c: char| c.is_ascii_punctuation());
                if !url.is_empty() {
                    urls.push(url.to_string());
                }
            }
        }

        urls
    }

    /// Extract phone numbers (US format: XXX-XXX-XXXX).
    #[allow(clippy::unused_self)]
    fn extract_phone_numbers(&self, text: &str) -> Vec<String> {
        let mut phones = Vec::new();

        for word in text.split_whitespace() {
            if self.is_phone_number(word) {
                phones.push(word.to_string());
            }
        }

        phones
    }

    /// Check if string looks like a phone number.
    #[allow(clippy::unused_self)]
    fn is_phone_number(&self, s: &str) -> bool {
        // Phone formats: 123-456-7890 or (123) 456-7890
        let digits: String = s.chars().filter(char::is_ascii_digit).collect();

        // Must have 10 digits
        if digits.len() != 10 {
            return false;
        }

        // Common formats
        let formats = ["###-###-####", "##########", "(###) ###-####"];

        for format in &formats {
            if Self::matches_phone_format(s, format) {
                return true;
            }
        }

        false
    }

    /// Check if string matches phone format.
    fn matches_phone_format(s: &str, format: &str) -> bool {
        if s.len() != format.len() {
            return false;
        }

        for (c1, c2) in s.chars().zip(format.chars()) {
            match c2 {
                '#' => {
                    if !c1.is_ascii_digit() {
                        return false;
                    }
                }
                _ => {
                    if c1 != c2 {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Extract mentions (@username).
    fn extract_mentions(text: &str) -> Vec<String> {
        let mut mentions = Vec::new();

        for word in text.split_whitespace() {
            if word.starts_with('@') && word.len() > 1 {
                // Remove trailing punctuation
                let mention = word[1..].trim_end_matches(|c: char| c.is_ascii_punctuation());
                if !mention.is_empty() && mention.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    mentions.push(format!("@{mention}"));
                }
            }
        }

        mentions
    }

    /// Extract hashtags (#topic).
    fn extract_hashtags(text: &str) -> Vec<String> {
        let mut hashtags = Vec::new();

        for word in text.split_whitespace() {
            if word.starts_with('#') && word.len() > 1 {
                // Remove trailing punctuation
                let hashtag = word[1..].trim_end_matches(|c: char| c.is_ascii_punctuation());
                if !hashtag.is_empty() && hashtag.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    hashtags.push(format!("#{hashtag}"));
                }
            }
        }

        hashtags
    }

    /// Extract capitalized words (potential named entities).
    fn extract_capitalized_words(text: &str) -> Vec<String> {
        let mut entities = Vec::new();

        for word in text.split_whitespace() {
            // Clean word (remove punctuation)
            let clean: String = word.chars().filter(|c| c.is_alphabetic()).collect();

            if clean.is_empty() {
                continue;
            }

            // Check if capitalized (first letter uppercase, rest lowercase)
            let first_char = clean.chars().next().expect("checked is_empty");
            if first_char.is_uppercase() && clean.len() > 1 {
                // Not all caps (probably not an acronym)
                if !clean.chars().skip(1).all(char::is_uppercase) {
                    entities.push(clean);
                }
            }
        }

        // Remove duplicates while preserving order
        let mut seen = std::collections::HashSet::new();
        entities.retain(|e| seen.insert(e.clone()));

        entities
    }
}

impl Default for EntityExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_emails() {
        let extractor = EntityExtractor::new();
        let text = "Contact john@example.com or jane@test.org for info";

        let entities = extractor.extract(text).expect("should succeed");
        assert_eq!(entities.emails.len(), 2);
        assert!(entities.emails.contains(&"john@example.com".to_string()));
        assert!(entities.emails.contains(&"jane@test.org".to_string()));
    }

    #[test]
    fn test_extract_urls() {
        let extractor = EntityExtractor::new();
        let text = "Visit https://example.com or http://test.org";

        let entities = extractor.extract(text).expect("should succeed");
        assert_eq!(entities.urls.len(), 2);
        assert!(entities.urls.contains(&"https://example.com".to_string()));
        assert!(entities.urls.contains(&"http://test.org".to_string()));
    }

    #[test]
    fn test_extract_phone_numbers() {
        let extractor = EntityExtractor::new();
        let text = "Call 555-123-4567 or 987-654-3210";

        let entities = extractor.extract(text).expect("should succeed");
        assert_eq!(entities.phone_numbers.len(), 2);
        assert!(entities.phone_numbers.contains(&"555-123-4567".to_string()));
        assert!(entities.phone_numbers.contains(&"987-654-3210".to_string()));
    }

    #[test]
    fn test_extract_mentions() {
        let extractor = EntityExtractor::new();
        let text = "Thanks @john_doe and @jane for your help!";

        let entities = extractor.extract(text).expect("should succeed");
        assert_eq!(entities.mentions.len(), 2);
        assert!(entities.mentions.contains(&"@john_doe".to_string()));
        assert!(entities.mentions.contains(&"@jane".to_string()));
    }

    #[test]
    fn test_extract_hashtags() {
        let extractor = EntityExtractor::new();
        let text = "Check out #rust and #machinelearning topics";

        let entities = extractor.extract(text).expect("should succeed");
        assert_eq!(entities.hashtags.len(), 2);
        assert!(entities.hashtags.contains(&"#rust".to_string()));
        assert!(entities.hashtags.contains(&"#machinelearning".to_string()));
    }

    #[test]
    fn test_extract_named_entities() {
        let extractor = EntityExtractor::new();
        let text = "John Smith visited Paris and London last week";

        let entities = extractor.extract(text).expect("should succeed");
        assert!(!entities.named_entities.is_empty());
        assert!(entities.named_entities.contains(&"John".to_string()));
        assert!(entities.named_entities.contains(&"Smith".to_string()));
        assert!(entities.named_entities.contains(&"Paris".to_string()));
    }

    #[test]
    fn test_empty_entities() {
        let entities = Entities::new();
        assert!(entities.is_empty());
        assert_eq!(entities.total_count(), 0);
    }

    // ====================================================================
    // Additional coverage tests for uncovered branches
    // ====================================================================

    #[test]
    fn test_with_named_entities_disabled() {
        let extractor = EntityExtractor::new().with_named_entities(false);
        let text = "John Smith visited Paris and London";

        let entities = extractor.extract(text).expect("should succeed");
        assert!(
            entities.named_entities.is_empty(),
            "Named entities should not be extracted when disabled"
        );
    }

    #[test]
    fn test_entities_default() {
        let entities = Entities::default();
        assert!(entities.is_empty());
        assert_eq!(entities.total_count(), 0);
    }

    #[test]
    fn test_extractor_default() {
        let extractor = EntityExtractor::default();
        let entities = extractor.extract("Hello World").expect("should succeed");
        // Default should have named entity extraction enabled
        assert!(!entities.named_entities.is_empty());
    }

    #[test]
    fn test_is_empty_partial() {
        let mut entities = Entities::new();
        // Only add one type of entity
        entities.emails.push("test@example.com".to_string());
        assert!(!entities.is_empty());
        assert_eq!(entities.total_count(), 1);
    }

    #[test]
    fn test_total_count_mixed() {
        let mut entities = Entities::new();
        entities.emails.push("a@b.com".to_string());
        entities.urls.push("https://x.com".to_string());
        entities.phone_numbers.push("555-123-4567".to_string());
        entities.mentions.push("@user".to_string());
        entities.hashtags.push("#tag".to_string());
        entities.named_entities.push("John".to_string());
        assert_eq!(entities.total_count(), 6);
        assert!(!entities.is_empty());
    }

    #[test]
    fn test_invalid_email_no_at() {
        let extractor = EntityExtractor::new();
        let text = "Not an email: noatsign.com";
        let entities = extractor.extract(text).expect("should succeed");
        assert!(entities.emails.is_empty());
    }

    #[test]
    fn test_invalid_email_multiple_at() {
        let extractor = EntityExtractor::new();
        let text = "Bad email: a@@b.com";
        let entities = extractor.extract(text).expect("should succeed");
        assert!(entities.emails.is_empty());
    }

    #[test]
    fn test_invalid_email_empty_local() {
        let extractor = EntityExtractor::new();
        let text = "Bad email: @domain.com";
        let entities = extractor.extract(text).expect("should succeed");
        // Starts with @ so it will be treated as a mention, not email
        assert!(entities.emails.is_empty());
    }

    #[test]
    fn test_invalid_email_no_dot_in_domain() {
        let extractor = EntityExtractor::new();
        let text = "Bad email: user@localhost";
        let entities = extractor.extract(text).expect("should succeed");
        assert!(entities.emails.is_empty());
    }

    #[test]
    fn test_invalid_email_empty_domain_part() {
        let extractor = EntityExtractor::new();
        let text = "Bad email: user@.com";
        let entities = extractor.extract(text).expect("should succeed");
        assert!(entities.emails.is_empty());
    }

    #[test]
    fn test_url_with_trailing_punctuation() {
        let extractor = EntityExtractor::new();
        let text = "Visit https://example.com.";
        let entities = extractor.extract(text).expect("should succeed");
        assert_eq!(entities.urls.len(), 1);
        assert_eq!(entities.urls[0], "https://example.com");
    }

    #[test]
    fn test_url_http_scheme() {
        let extractor = EntityExtractor::new();
        let text = "Visit http://insecure.example.com/path?q=1";
        let entities = extractor.extract(text).expect("should succeed");
        assert_eq!(entities.urls.len(), 1);
    }

    #[test]
    fn test_no_url_without_scheme() {
        let extractor = EntityExtractor::new();
        let text = "Visit www.example.com";
        let entities = extractor.extract(text).expect("should succeed");
        assert!(entities.urls.is_empty());
    }

    #[test]
    fn test_phone_number_wrong_digit_count() {
        let extractor = EntityExtractor::new();
        // 9 digits instead of 10
        let text = "Call 555-123-456";
        let entities = extractor.extract(text).expect("should succeed");
        assert!(entities.phone_numbers.is_empty());
    }

    #[test]
    fn test_phone_number_11_digits() {
        let extractor = EntityExtractor::new();
        let text = "Call 1-555-123-4567";
        let entities = extractor.extract(text).expect("should succeed");
        // Has 11 digits, not 10
        assert!(entities.phone_numbers.is_empty());
    }

    #[test]
    fn test_phone_number_wrong_format_10_digits() {
        let extractor = EntityExtractor::new();
        // 10 digits but in a format that doesn't match any known pattern
        let text = "Call 55.512.34567";
        let entities = extractor.extract(text).expect("should succeed");
        assert!(entities.phone_numbers.is_empty());
    }

    #[test]
    fn test_mention_with_trailing_punctuation() {
        let extractor = EntityExtractor::new();
        let text = "Thanks @user123!";
        let entities = extractor.extract(text).expect("should succeed");
        assert_eq!(entities.mentions.len(), 1);
        assert_eq!(entities.mentions[0], "@user123");
    }

    #[test]
    fn test_mention_bare_at_sign() {
        let extractor = EntityExtractor::new();
        let text = "Just an @ sign";
        let entities = extractor.extract(text).expect("should succeed");
        assert!(entities.mentions.is_empty());
    }

    #[test]
    fn test_mention_with_special_chars() {
        let extractor = EntityExtractor::new();
        // Mention with non-alphanumeric/underscore char should not match
        let text = "Hello @user-name";
        let entities = extractor.extract(text).expect("should succeed");
        // "user-name" after trim of '-' yields "user" which passes alphanumeric check
        // but "user-name" itself has '-' so the trim removes trailing '-' leaving... let's verify
        // Actually: word is "@user-name", then word[1..] = "user-name",
        // trim_end_matches(punctuation) on "user-name" keeps it because '-' is ascii punctuation,
        // so it trims the trailing part. Actually '-' IS ascii_punctuation.
        // "user-name".trim_end_matches(|c: char| c.is_ascii_punctuation()) = "user-name"
        // Wait, trim_end only trims from the end. 'e' is not punctuation so it stops at 'e'.
        // So we get "user-name", and then chars().all(alphanumeric or _) fails on '-'.
        assert!(entities.mentions.is_empty());
    }

    #[test]
    fn test_hashtag_with_trailing_punctuation() {
        let extractor = EntityExtractor::new();
        let text = "Check #rustlang!";
        let entities = extractor.extract(text).expect("should succeed");
        assert_eq!(entities.hashtags.len(), 1);
        assert_eq!(entities.hashtags[0], "#rustlang");
    }

    #[test]
    fn test_hashtag_bare_hash() {
        let extractor = EntityExtractor::new();
        let text = "Just a # sign";
        let entities = extractor.extract(text).expect("should succeed");
        assert!(entities.hashtags.is_empty());
    }

    #[test]
    fn test_hashtag_with_special_chars() {
        let extractor = EntityExtractor::new();
        let text = "Tag #rust-lang";
        let entities = extractor.extract(text).expect("should succeed");
        // Similar to mentions: "rust-lang" has '-' which is ascii_punctuation
        // trim_end_matches on "rust-lang": 'g' is not punctuation, so no trimming
        // then all(alphanumeric or _) fails on '-'
        assert!(entities.hashtags.is_empty());
    }

    #[test]
    fn test_named_entity_all_caps_excluded() {
        let extractor = EntityExtractor::new();
        let text = "The NASA and FBI are organizations";
        let entities = extractor.extract(text).expect("should succeed");
        // "NASA" and "FBI" are all-caps, should be excluded as acronyms
        assert!(!entities.named_entities.contains(&"NASA".to_string()));
        assert!(!entities.named_entities.contains(&"FBI".to_string()));
    }

    #[test]
    fn test_named_entity_single_char_excluded() {
        let extractor = EntityExtractor::new();
        let text = "I went to A store";
        let entities = extractor.extract(text).expect("should succeed");
        // Single uppercase char "I" and "A" have len==1, excluded
        assert!(!entities.named_entities.contains(&"I".to_string()));
        assert!(!entities.named_entities.contains(&"A".to_string()));
    }

    #[test]
    fn test_named_entity_deduplication() {
        let extractor = EntityExtractor::new();
        let text = "John met John and John again";
        let entities = extractor.extract(text).expect("should succeed");
        // "John" should appear only once
        let john_count = entities
            .named_entities
            .iter()
            .filter(|e| *e == "John")
            .count();
        assert_eq!(john_count, 1);
    }

    #[test]
    fn test_named_entity_numbers_only_word() {
        let extractor = EntityExtractor::new();
        let text = "Number 123 here";
        let entities = extractor.extract(text).expect("should succeed");
        // "123" has no alphabetic characters, should be skipped
        assert!(!entities.named_entities.contains(&"123".to_string()));
    }

    #[test]
    fn test_entities_clone_eq() {
        let mut entities = Entities::new();
        entities.emails.push("a@b.com".to_string());
        let cloned = entities.clone();
        assert_eq!(entities, cloned);
    }

    #[test]
    fn test_entities_debug() {
        let entities = Entities::new();
        let debug = format!("{:?}", entities);
        assert!(debug.contains("Entities"));
    }

    #[test]
    fn test_extractor_debug() {
        let extractor = EntityExtractor::new();
        let debug = format!("{:?}", extractor);
        assert!(debug.contains("EntityExtractor"));
    }

    #[test]
    fn test_empty_text_extraction() {
        let extractor = EntityExtractor::new();
        let entities = extractor.extract("").expect("should succeed");
        assert!(entities.is_empty());
    }

    #[test]
    fn test_extract_all_entity_types() {
        let extractor = EntityExtractor::new();
        // Note: phone number must be a standalone whitespace-separated token
        // (not adjacent to period) since the extractor splits on whitespace
        let text = "Contact john@example.com at https://example.com \
                    and call 555-123-4567 now. Ping @johndoe about #rust. \
                    John visited Paris.";
        let entities = extractor.extract(text).expect("should succeed");

        assert!(!entities.emails.is_empty());
        assert!(!entities.urls.is_empty());
        assert!(!entities.phone_numbers.is_empty());
        assert!(!entities.mentions.is_empty());
        assert!(!entities.hashtags.is_empty());
        assert!(!entities.named_entities.is_empty());
        assert!(entities.total_count() >= 6);
    }

    #[test]
    fn test_phone_matches_format_length_mismatch() {
        let extractor = EntityExtractor::new();
        // "12345" has 5 digits, won't match 10-digit phone
        let text = "Number 12345";
        let entities = extractor.extract(text).expect("should succeed");
        assert!(entities.phone_numbers.is_empty());
    }

    #[test]
    fn test_phone_format_non_digit_in_digit_position() {
        let extractor = EntityExtractor::new();
        // Right length (12 chars for ###-###-####) but non-digit in digit position
        let text = "Call ABC-DEF-GHIJ";
        let entities = extractor.extract(text).expect("should succeed");
        assert!(entities.phone_numbers.is_empty());
    }
}
