//! Trie data structure for fast prefix matching

use std::collections::HashMap;

/// Trie node
#[derive(Default)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    is_end: bool,
    count: u32,
}

/// Trie for fast prefix-based command lookup
pub struct Trie {
    root: TrieNode,
}

impl Trie {
    pub fn new() -> Self {
        Self {
            root: TrieNode::default(),
        }
    }

    /// Insert a command into the trie
    pub fn insert(&mut self, word: &str) {
        let mut node = &mut self.root;

        for ch in word.chars() {
            node = node.children.entry(ch).or_default();
        }

        node.is_end = true;
        node.count += 1;
    }

    /// Find all commands matching a prefix, sorted by frequency
    pub fn find_prefix(&self, prefix: &str, limit: usize) -> Vec<String> {
        // Navigate to prefix node
        let mut node = &self.root;

        for ch in prefix.chars() {
            match node.children.get(&ch) {
                Some(n) => node = n,
                None => return vec![],
            }
        }

        // Collect all completions
        let mut results = Vec::new();
        self.collect_words(node, prefix.to_string(), &mut results);

        // Sort by count (descending)
        results.sort_by(|a, b| b.1.cmp(&a.1));

        // Return just the strings
        results.into_iter().take(limit).map(|(s, _)| s).collect()
    }

    fn collect_words(&self, node: &TrieNode, current: String, results: &mut Vec<(String, u32)>) {
        if node.is_end {
            results.push((current.clone(), node.count));
        }

        // Limit search depth for performance
        if results.len() >= 100 {
            return;
        }

        for (ch, child) in &node.children {
            let mut next = current.clone();
            next.push(*ch);
            self.collect_words(child, next, results);
        }
    }
}

impl Default for Trie {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_find() {
        let mut trie = Trie::new();
        trie.insert("git status");
        trie.insert("git commit");
        trie.insert("git push");
        trie.insert("grep pattern");

        let results = trie.find_prefix("git ", 10);
        assert_eq!(results.len(), 3);

        let results = trie.find_prefix("grep", 10);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_frequency_ordering() {
        let mut trie = Trie::new();
        trie.insert("git status");
        trie.insert("git status");
        trie.insert("git status");
        trie.insert("git commit");

        let results = trie.find_prefix("git ", 10);
        assert_eq!(results[0], "git status"); // Most frequent first
    }

    #[test]
    fn test_no_match() {
        let mut trie = Trie::new();
        trie.insert("git status");

        let results = trie.find_prefix("docker ", 10);
        assert!(results.is_empty());
    }
}
