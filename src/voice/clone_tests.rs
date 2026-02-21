
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloning_config_default() {
        let config = CloningConfig::default();
        assert!((config.min_reference_duration - 3.0).abs() < f32::EPSILON);
        assert!((config.max_reference_duration - 30.0).abs() < f32::EPSILON);
        assert_eq!(config.sample_rate, 22050);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_cloning_config_few_shot() {
        let config = CloningConfig::few_shot();
        assert!((config.min_reference_duration - 3.0).abs() < f32::EPSILON);
        assert!((config.max_reference_duration - 10.0).abs() < f32::EPSILON);
        assert!(!config.enable_adaptation);
    }

    #[test]
    fn test_cloning_config_zero_shot() {
        let config = CloningConfig::zero_shot();
        assert!((config.min_reference_duration - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cloning_config_with_adaptation() {
        let config = CloningConfig::with_adaptation();
        assert!(config.enable_adaptation);
        assert!((config.min_reference_duration - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cloning_config_validation() {
        let mut config = CloningConfig::default();

        config.min_reference_duration = 0.0;
        assert!(config.validate().is_err());

        config.min_reference_duration = 3.0;
        config.max_reference_duration = 2.0; // Less than min
        assert!(config.validate().is_err());

        config.max_reference_duration = 30.0;
        config.similarity_threshold = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cloning_config_samples() {
        let config = CloningConfig::default();
        let min_samples = config.min_reference_samples();
        let max_samples = config.max_reference_samples();

        assert_eq!(min_samples, (3.0 * 22050.0) as usize);
        assert_eq!(max_samples, (30.0 * 22050.0) as usize);
    }

    #[test]
    fn test_voice_profile_new() {
        let profile = VoiceProfile::new("speaker_1".to_string());
        assert_eq!(profile.speaker_id(), "speaker_1");
        assert!(profile.embedding().is_none());
        assert!(!profile.is_ready());
        assert!(!profile.is_adapted());
    }

    #[test]
    fn test_voice_profile_with_embedding() {
        let emb = SpeakerEmbedding::from_vec(vec![1.0, 2.0, 3.0]);
        let profile = VoiceProfile::with_embedding("speaker_2".to_string(), emb);

        assert_eq!(profile.speaker_id(), "speaker_2");
        assert!(profile.is_ready());
        assert!((profile.quality_score() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_voice_profile_setters() {
        let mut profile = VoiceProfile::new("test".to_string());

        profile.set_embedding(SpeakerEmbedding::from_vec(vec![1.0, 2.0]));
        assert!(profile.is_ready());

        profile.set_reference_duration(5.5);
        assert!((profile.reference_duration() - 5.5).abs() < f32::EPSILON);

        profile.set_quality_score(0.8);
        assert!((profile.quality_score() - 0.8).abs() < f32::EPSILON);

        // Test clamping
        profile.set_quality_score(1.5);
        assert!((profile.quality_score() - 1.0).abs() < f32::EPSILON);

        profile.set_adapted(true);
        assert!(profile.is_adapted());
    }

    #[test]
    fn test_voice_profile_similarity() {
        let emb1 = SpeakerEmbedding::from_vec(vec![1.0, 0.0, 0.0]);
        let emb2 = SpeakerEmbedding::from_vec(vec![1.0, 0.0, 0.0]);
        let emb3 = SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]);

        let profile1 = VoiceProfile::with_embedding("a".to_string(), emb1);
        let profile2 = VoiceProfile::with_embedding("b".to_string(), emb2);
        let profile3 = VoiceProfile::with_embedding("c".to_string(), emb3);

        let sim_same = profile1.similarity(&profile2).expect("similarity failed");
        assert!((sim_same - 1.0).abs() < 1e-6);

        let sim_diff = profile1.similarity(&profile3).expect("similarity failed");
        assert!((sim_diff - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_voice_profile_similarity_missing_embedding() {
        let profile1 = VoiceProfile::new("a".to_string());
        let profile2 =
            VoiceProfile::with_embedding("b".to_string(), SpeakerEmbedding::from_vec(vec![1.0]));

        assert!(profile1.similarity(&profile2).is_err());
        assert!(profile2.similarity(&profile1).is_err());
    }

    #[test]
    fn test_voice_profile_similarity_dimension_mismatch() {
        let profile1 = VoiceProfile::with_embedding(
            "a".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0, 0.0]),
        );
        let profile2 = VoiceProfile::with_embedding(
            "b".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0, 0.0, 0.0]),
        );

        assert!(profile1.similarity(&profile2).is_err());
    }

    #[test]
    fn test_yourtts_cloner_stub() {
        let cloner = YourTtsCloner::default_config();
        let audio = vec![0.0_f32; 22050 * 5]; // 5 seconds
        assert!(cloner.create_profile(&audio, "test").is_err());
    }

    #[test]
    fn test_yourtts_cloner_empty_audio() {
        let cloner = YourTtsCloner::default_config();
        let result = cloner.create_profile(&[], "test");
        assert!(matches!(result, Err(VoiceError::InvalidAudio(_))));
    }

    #[test]
    fn test_yourtts_cloner_short_audio() {
        let cloner = YourTtsCloner::default_config();
        let audio = vec![0.0_f32; 1000]; // Too short
        let result = cloner.create_profile(&audio, "test");
        assert!(matches!(result, Err(VoiceError::InvalidAudio(_))));
    }

    #[test]
    fn test_yourtts_synthesize() {
        let cloner = YourTtsCloner::default_config();
        let profile = VoiceProfile::with_embedding(
            "test".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0; 256]),
        );

        // Empty text
        assert!(cloner.synthesize("", &profile).is_err());

        // Not implemented
        assert!(cloner.synthesize("Hello world", &profile).is_err());
    }

    #[test]
    fn test_yourtts_synthesize_not_ready() {
        let cloner = YourTtsCloner::default_config();
        let profile = VoiceProfile::new("test".to_string()); // No embedding

        let result = cloner.synthesize("Hello", &profile);
        assert!(matches!(result, Err(VoiceError::ModelNotLoaded)));
    }

    #[test]
    fn test_yourtts_adapt() {
        let mut cloner = YourTtsCloner::default_config();
        let mut profile = VoiceProfile::with_embedding(
            "test".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0; 256]),
        );
        let audio = vec![0.0_f32; 22050];

        // Adaptation disabled
        assert!(cloner.adapt(&mut profile, &audio).is_err());

        // Enable adaptation
        cloner = YourTtsCloner::new(CloningConfig::with_adaptation());
        // Still fails (not implemented)
        assert!(cloner.adapt(&mut profile, &audio).is_err());
    }

    #[test]
    fn test_sv2tts_encoder_stub() {
        let encoder = Sv2TtsSpeakerEncoder::default_config();
        assert_eq!(encoder.embedding_dim(), 256);

        let audio = vec![0.0_f32; 16000];
        assert!(encoder.encode(&audio).is_err());
    }

    #[test]
    fn test_sv2tts_encoder_empty() {
        let encoder = Sv2TtsSpeakerEncoder::default_config();
        assert!(matches!(
            encoder.encode(&[]),
            Err(VoiceError::InvalidAudio(_))
        ));
    }

    #[test]
    fn test_verify_same_speaker() {
        let profile1 = VoiceProfile::with_embedding(
            "a".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0, 0.0, 0.0]),
        );
        let profile2 = VoiceProfile::with_embedding(
            "b".to_string(),
            SpeakerEmbedding::from_vec(vec![0.99, 0.1, 0.0]), // Similar
        );
        let profile3 = VoiceProfile::with_embedding(
            "c".to_string(),
            SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]), // Different
        );

        assert!(verify_same_speaker(&profile1, &profile2, 0.9).expect("verify failed"));
        assert!(!verify_same_speaker(&profile1, &profile3, 0.5).expect("verify failed"));
    }

    #[test]
    fn test_estimate_quality() {
        // Empty audio
        assert_eq!(estimate_quality(&[], 16000), 0.0);

        // Very short audio
        let short = vec![0.5_f32; 16000]; // 1 second
        let quality = estimate_quality(&short, 16000);
        assert!(quality > 0.0 && quality < 1.0);

        // Optimal duration
        let optimal = vec![0.5_f32; 16000 * 10]; // 10 seconds
        let quality2 = estimate_quality(&optimal, 16000);
        assert!(quality2 >= quality);
    }

    #[test]
    fn test_estimate_quality_zero_sample_rate() {
        let audio = vec![0.5_f32; 16000];
        assert_eq!(estimate_quality(&audio, 0), 0.0);
    }

    #[test]
    fn test_merge_profiles() {
        let profile1 = VoiceProfile::with_embedding(
            "speaker".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0, 0.0]),
        );
        let profile2 = VoiceProfile::with_embedding(
            "speaker".to_string(),
            SpeakerEmbedding::from_vec(vec![0.0, 1.0]),
        );

        let merged = merge_profiles(&[profile1, profile2]).expect("merge failed");
        assert!(merged.is_ready());

        let emb = merged.embedding().expect("no embedding");
        assert!((emb.as_slice()[0] - 0.5).abs() < 1e-6);
        assert!((emb.as_slice()[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_merge_profiles_empty() {
        let profiles: Vec<VoiceProfile> = vec![];
        assert!(merge_profiles(&profiles).is_err());
    }

    #[test]
    fn test_merge_profiles_no_embeddings() {
        let profiles = vec![
            VoiceProfile::new("a".to_string()),
            VoiceProfile::new("b".to_string()),
        ];
        assert!(merge_profiles(&profiles).is_err());
    }

    #[test]
    fn test_merge_profiles_dimension_mismatch() {
        let profile1 = VoiceProfile::with_embedding(
            "a".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0, 0.0]),
        );
        let profile2 = VoiceProfile::with_embedding(
            "b".to_string(),
            SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]),
        );

        assert!(merge_profiles(&[profile1, profile2]).is_err());
    }

    #[test]
    fn test_merge_profiles_metrics() {
        let mut profile1 = VoiceProfile::with_embedding(
            "speaker".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0]),
        );
        profile1.set_reference_duration(5.0);
        profile1.set_quality_score(0.8);

        let mut profile2 = VoiceProfile::with_embedding(
            "speaker".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0]),
        );
        profile2.set_reference_duration(10.0);
        profile2.set_quality_score(0.6);
        profile2.set_adapted(true);

        let merged = merge_profiles(&[profile1, profile2]).expect("merge failed");
        assert!((merged.reference_duration() - 15.0).abs() < f32::EPSILON);
        assert!((merged.quality_score() - 0.7).abs() < 1e-6);
        assert!(merged.is_adapted());
    }
}
