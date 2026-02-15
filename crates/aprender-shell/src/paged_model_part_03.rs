
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_ngram_segment_new() {
        let segment = NgramSegment::new("git".to_string());
        assert_eq!(segment.prefix, "git");
        assert!(segment.ngrams.is_empty());
        assert_eq!(segment.size_bytes, 0);
    }

    #[test]
    fn test_ngram_segment_add() {
        let mut segment = NgramSegment::new("git".to_string());
        segment.add("git".to_string(), "commit".to_string(), 1);
        segment.add("git".to_string(), "commit".to_string(), 1);
        segment.add("git".to_string(), "push".to_string(), 1);

        assert_eq!(segment.ngrams.len(), 1);
        let git_nexts = segment.ngrams.get("git").unwrap();
        assert_eq!(git_nexts.get("commit"), Some(&2));
        assert_eq!(git_nexts.get("push"), Some(&1));
    }

    #[test]
    fn test_ngram_segment_serialization() {
        let mut segment = NgramSegment::new("cargo".to_string());
        segment.add("cargo".to_string(), "build".to_string(), 5);
        segment.add("cargo".to_string(), "test".to_string(), 3);
        segment.add("cargo build".to_string(), "--release".to_string(), 2);

        let bytes = segment.to_bytes();
        let restored = NgramSegment::from_bytes(&bytes).unwrap();

        assert_eq!(restored.prefix, "cargo");
        assert_eq!(restored.ngrams.len(), 2);
        assert_eq!(restored.ngrams.get("cargo").unwrap().get("build"), Some(&5));
        assert_eq!(restored.ngrams.get("cargo").unwrap().get("test"), Some(&3));
        assert_eq!(
            restored.ngrams.get("cargo build").unwrap().get("--release"),
            Some(&2)
        );
    }

    #[test]
    fn test_paged_model_new() {
        let model = PagedMarkovModel::new(3, 10);
        assert_eq!(model.ngram_size(), 3);
        assert!(model.memory_limit() >= MIN_MEMORY_LIMIT);
    }

    #[test]
    fn test_paged_model_train() {
        let commands = vec![
            "git status".to_string(),
            "git commit -m test".to_string(),
            "git push".to_string(),
            "cargo build".to_string(),
            "cargo test".to_string(),
        ];

        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&commands);

        assert_eq!(model.total_commands(), 5);
        assert_eq!(model.vocab_size(), 5);

        // Should have git and cargo segments
        assert!(model.segments.contains_key("git"));
        assert!(model.segments.contains_key("cargo"));
    }

    #[test]
    fn test_paged_model_suggest() {
        let commands = vec![
            "git status".to_string(),
            "git status".to_string(),
            "git commit -m fix".to_string(),
            "git push".to_string(),
        ];

        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&commands);

        let suggestions = model.suggest("git ", 3);
        assert!(!suggestions.is_empty());

        // status appears twice, should be suggested
        let has_status = suggestions.iter().any(|(s, _)| s.contains("status"));
        assert!(has_status);
    }

    #[test]
    fn test_paged_model_save_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apbundle");

        // Create and train model
        let commands = vec![
            "git status".to_string(),
            "git commit".to_string(),
            "cargo build".to_string(),
        ];

        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&commands);
        model.save(&path).unwrap();

        // Load model
        let mut loaded = PagedMarkovModel::load(&path, 10).unwrap();

        assert_eq!(loaded.total_commands(), 3);
        assert_eq!(loaded.vocab_size(), 3);
        assert_eq!(loaded.ngram_size(), 3);

        // Test suggestions work after load
        let suggestions = loaded.suggest("git ", 3);
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_paged_model_stats() {
        let commands = vec![
            "git status".to_string(),
            "cargo build".to_string(),
            "docker run".to_string(),
        ];

        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&commands);

        let stats = model.stats();
        assert_eq!(stats.n, 3);
        assert_eq!(stats.total_commands, 3);
        assert_eq!(stats.vocab_size, 3);
        assert_eq!(stats.total_segments, 3); // git, cargo, docker
    }

    #[test]
    fn test_paged_model_on_demand_loading() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("ondemand.apbundle");

        // Create model with many segments
        let commands = vec![
            "git status".to_string(),
            "git commit".to_string(),
            "cargo build".to_string(),
            "cargo test".to_string(),
            "docker run".to_string(),
            "kubectl get pods".to_string(),
        ];

        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&commands);
        model.save(&path).unwrap();

        // Load with small memory limit
        let mut loaded = PagedMarkovModel::load(&path, 1).unwrap();

        // Initially no segments loaded
        assert_eq!(loaded.stats().loaded_segments, 0);

        // Query git commands - should load git segment
        let _ = loaded.suggest("git ", 3);
        assert!(loaded.segments.contains_key("git"));

        // Query cargo commands - should load cargo segment
        let _ = loaded.suggest("cargo ", 3);
        assert!(loaded.segments.contains_key("cargo"));
    }

    #[test]
    fn test_paged_model_prefetch_hint() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("prefetch.apbundle");

        let commands = vec!["git status".to_string(), "cargo build".to_string()];

        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&commands);
        model.save(&path).unwrap();

        let mut loaded = PagedMarkovModel::load(&path, 10).unwrap();

        // Hint that we'll need git segment
        loaded.prefetch_hint("git");

        // Should still work after hint
        let suggestions = loaded.suggest("git ", 3);
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_paged_model_top_commands() {
        let commands = vec![
            "git status".to_string(),
            "git status".to_string(),
            "git status".to_string(),
            "cargo build".to_string(),
            "cargo build".to_string(),
            "docker run".to_string(),
        ];

        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&commands);

        let top = model.top_commands(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "git status");
        assert_eq!(top[0].1, 3);
        assert_eq!(top[1].0, "cargo build");
        assert_eq!(top[1].1, 2);
    }

    #[test]
    fn test_paged_model_empty_commands() {
        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&[]);

        assert_eq!(model.total_commands(), 0);
        assert_eq!(model.vocab_size(), 0);

        let suggestions = model.suggest("git ", 3);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_ngram_segment_empty_bytes() {
        let result = NgramSegment::from_bytes(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_paged_model_stats_display() {
        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&["git status".to_string()]);

        let stats = model.stats();
        let display = format!("{stats}");

        assert!(display.contains("N-gram size:"));
        assert!(display.contains("Total commands:"));
        assert!(display.contains("Memory limit:"));
    }
}
