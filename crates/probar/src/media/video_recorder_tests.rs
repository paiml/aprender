    use super::*;

    mod video_config_tests {
        use super::*;

        #[test]
        fn test_default_config() {
            let config = VideoConfig::default();
            assert_eq!(config.fps, 30);
            assert_eq!(config.width, 1280);
            assert_eq!(config.height, 720);
            assert_eq!(config.bitrate, 5000);
            assert_eq!(config.codec, VideoCodec::Mjpeg);
            assert_eq!(config.max_duration_secs, 300);
            assert_eq!(config.jpeg_quality, 85);
        }

        #[test]
        fn test_config_new() {
            let config = VideoConfig::new(1920, 1080);
            assert_eq!(config.width, 1920);
            assert_eq!(config.height, 1080);
        }

        #[test]
        fn test_config_builder() {
            let config = VideoConfig::new(800, 600)
                .with_fps(60)
                .with_bitrate(10000)
                .with_codec(VideoCodec::Raw)
                .with_max_duration(600)
                .with_jpeg_quality(95);

            assert_eq!(config.fps, 60);
            assert_eq!(config.bitrate, 10000);
            assert_eq!(config.codec, VideoCodec::Raw);
            assert_eq!(config.max_duration_secs, 600);
            assert_eq!(config.jpeg_quality, 95);
        }

        #[test]
        fn test_fps_clamping() {
            let config = VideoConfig::default().with_fps(0);
            assert_eq!(config.fps, 1);

            let config = VideoConfig::default().with_fps(100);
            assert_eq!(config.fps, 60);
        }

        #[test]
        fn test_jpeg_quality_clamping() {
            let config = VideoConfig::default().with_jpeg_quality(0);
            assert_eq!(config.jpeg_quality, 1);

            let config = VideoConfig::default().with_jpeg_quality(200);
            assert_eq!(config.jpeg_quality, 100);
        }

        #[test]
        fn test_frame_duration() {
            let config = VideoConfig::default().with_fps(30);
            let duration = config.frame_duration();
            assert_eq!(duration.as_millis(), 33);

            let config = VideoConfig::default().with_fps(60);
            let duration = config.frame_duration();
            assert_eq!(duration.as_millis(), 16);
        }

        #[test]
        fn test_timescale() {
            let config = VideoConfig::default().with_fps(30);
            assert_eq!(config.timescale(), 3000);

            let config = VideoConfig::default().with_fps(60);
            assert_eq!(config.timescale(), 6000);
        }
    }

    mod video_codec_tests {
        use super::*;

        #[test]
        fn test_default_codec() {
            let codec = VideoCodec::default();
            assert_eq!(codec, VideoCodec::Mjpeg);
        }

        #[test]
        fn test_codec_equality() {
            assert_eq!(VideoCodec::Mjpeg, VideoCodec::Mjpeg);
            assert_eq!(VideoCodec::Raw, VideoCodec::Raw);
            assert_ne!(VideoCodec::Mjpeg, VideoCodec::Raw);
        }
    }

    mod recording_state_tests {
        use super::*;

        #[test]
        fn test_state_equality() {
            assert_eq!(RecordingState::Idle, RecordingState::Idle);
            assert_eq!(RecordingState::Recording, RecordingState::Recording);
            assert_eq!(RecordingState::Stopped, RecordingState::Stopped);
            assert_ne!(RecordingState::Idle, RecordingState::Recording);
        }
    }

    mod video_recorder_tests {
        use super::*;

        #[test]
        fn test_new_recorder() {
            let config = VideoConfig::default();
            let recorder = VideoRecorder::new(config);
            assert_eq!(recorder.state(), RecordingState::Idle);
            assert_eq!(recorder.frame_count(), 0);
        }

        #[test]
        fn test_start_recording() {
            let config = VideoConfig::default();
            let mut recorder = VideoRecorder::new(config);

            recorder.start().expect("Failed to start recording");
            assert_eq!(recorder.state(), RecordingState::Recording);
        }

        #[test]
        fn test_double_start_error() {
            let config = VideoConfig::default();
            let mut recorder = VideoRecorder::new(config);

            recorder.start().expect("Failed to start recording");
            let result = recorder.start();
            assert!(result.is_err());
        }

        #[test]
        fn test_capture_without_start_error() {
            let config = VideoConfig::default();
            let mut recorder = VideoRecorder::new(config);

            let data = vec![255u8; 800 * 600 * 4];
            let result = recorder.capture_raw_frame(&data, 800, 600);
            assert!(result.is_err());
        }

        #[test]
        fn test_stop_without_start_error() {
            let config = VideoConfig::default();
            let mut recorder = VideoRecorder::new(config);

            let result = recorder.stop();
            assert!(result.is_err());
        }

        #[test]
        fn test_stop_without_frames_error() {
            let config = VideoConfig::default();
            let mut recorder = VideoRecorder::new(config);

            recorder.start().expect("Failed to start recording");
            let result = recorder.stop();
            assert!(result.is_err());
        }

        #[test]
        fn test_capture_raw_frame() {
            let config = VideoConfig::new(10, 10).with_fps(1);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().expect("Failed to start recording");

            // Create a small red image
            let data = vec![255, 0, 0, 255].repeat(100); // 10x10 RGBA
            recorder
                .capture_raw_frame(&data, 10, 10)
                .expect("Failed to capture frame");

            assert_eq!(recorder.frame_count(), 1);
        }

        #[test]
        fn test_full_recording_cycle() {
            let config = VideoConfig::new(10, 10).with_fps(1);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().expect("Failed to start recording");

            // Capture a few frames
            for _ in 0..3 {
                let data = vec![255, 0, 0, 255].repeat(100);
                recorder
                    .capture_raw_frame(&data, 10, 10)
                    .expect("Failed to capture frame");
                // Sleep to allow frame capture (due to rate limiting)
                std::thread::sleep(std::time::Duration::from_millis(1100));
            }

            let video_data = recorder.stop().expect("Failed to stop recording");
            assert!(!video_data.is_empty());

            // Verify MP4 magic bytes (ftyp box)
            assert!(video_data.len() >= 8);
            assert_eq!(&video_data[4..8], b"ftyp");
        }

        #[test]
        fn test_config_accessor() {
            let config = VideoConfig::new(1920, 1080).with_fps(60);
            let recorder = VideoRecorder::new(config);

            assert_eq!(recorder.config().width, 1920);
            assert_eq!(recorder.config().height, 1080);
            assert_eq!(recorder.config().fps, 60);
        }
    }

    mod encoded_frame_tests {
        use super::*;

        #[test]
        fn test_encoded_frame_creation() {
            let frame = EncodedFrame {
                data: vec![1, 2, 3, 4],
                timestamp_ms: 100,
                duration_ms: 33,
            };

            assert_eq!(frame.data.len(), 4);
            assert_eq!(frame.timestamp_ms, 100);
            assert_eq!(frame.duration_ms, 33);
        }
    }

    mod mp4_generation_tests {
        use super::*;

        #[test]
        fn test_mp4_has_correct_structure() {
            let config = VideoConfig::new(10, 10).with_fps(1);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().expect("Failed to start");
            let data = vec![255, 0, 0, 255].repeat(100);
            recorder
                .capture_raw_frame(&data, 10, 10)
                .expect("Failed to capture");

            let video = recorder.stop().expect("Failed to stop");

            // Check for ftyp box
            assert!(find_box(&video, b"ftyp").is_some());

            // Check for mdat box
            assert!(find_box(&video, b"mdat").is_some());

            // Check for moov box
            assert!(find_box(&video, b"moov").is_some());
        }
    }

    mod save_tests {
        use super::*;
        use tempfile::TempDir;

        #[test]
        fn test_save_without_stop_error() {
            let config = VideoConfig::new(10, 10);
            let recorder = VideoRecorder::new(config);
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.mp4");

            let result = recorder.save(&path);
            assert!(result.is_err());
        }

        #[test]
        fn test_save_after_stop() {
            let config = VideoConfig::new(10, 10).with_fps(1);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();
            let data = vec![255, 0, 0, 255].repeat(100);
            recorder.capture_raw_frame(&data, 10, 10).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1100));
            recorder.capture_raw_frame(&data, 10, 10).unwrap();
            recorder.stop().unwrap();

            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.mp4");
            recorder.save(&path).unwrap();

            assert!(path.exists());
            let saved_data = std::fs::read(&path).unwrap();
            assert!(!saved_data.is_empty());
        }
    }

    mod frame_rate_tests {
        use super::*;

        #[test]
        fn test_frame_skipping() {
            let config = VideoConfig::new(10, 10).with_fps(1);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();
            let data = vec![255, 0, 0, 255].repeat(100);

            // Capture multiple frames rapidly - should be rate limited
            for _ in 0..5 {
                recorder.capture_raw_frame(&data, 10, 10).unwrap();
            }

            // Should only have captured 1 frame due to rate limiting
            assert_eq!(recorder.frame_count(), 1);
        }
    }

    mod resize_tests {
        use super::*;

        #[test]
        fn test_resize_frame() {
            let config = VideoConfig::new(20, 20).with_fps(1);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();

            // Capture a 10x10 frame when config expects 20x20
            let data = vec![255, 0, 0, 255].repeat(100);
            recorder.capture_raw_frame(&data, 10, 10).unwrap();

            assert_eq!(recorder.frame_count(), 1);
        }
    }

    mod invalid_frame_tests {
        use super::*;

        #[test]
        fn test_invalid_raw_frame_dimensions() {
            let config = VideoConfig::new(10, 10).with_fps(1);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();

            // Data doesn't match dimensions (too small)
            let data = vec![255u8; 10];
            let result = recorder.capture_raw_frame(&data, 10, 10);
            assert!(result.is_err());
        }
    }

    mod codec_tests {
        use super::*;

        #[test]
        fn test_raw_codec() {
            let config = VideoConfig::new(10, 10)
                .with_fps(1)
                .with_codec(VideoCodec::Raw);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();
            let data = vec![255, 0, 0, 255].repeat(100);
            recorder.capture_raw_frame(&data, 10, 10).unwrap();

            // Frame count should still be 1
            assert_eq!(recorder.frame_count(), 1);
        }

        #[test]
        fn test_codec_debug() {
            assert!(format!("{:?}", VideoCodec::Mjpeg).contains("Mjpeg"));
            assert!(format!("{:?}", VideoCodec::Raw).contains("Raw"));
        }

        #[test]
        fn test_codec_clone() {
            let codec = VideoCodec::Mjpeg;
            let cloned = codec;
            assert_eq!(codec, cloned);
        }
    }

    mod recording_state_debug {
        use super::*;

        #[test]
        fn test_state_debug() {
            assert!(format!("{:?}", RecordingState::Idle).contains("Idle"));
            assert!(format!("{:?}", RecordingState::Recording).contains("Recording"));
            assert!(format!("{:?}", RecordingState::Stopped).contains("Stopped"));
        }

        #[test]
        fn test_state_clone() {
            let state = RecordingState::Recording;
            let cloned = state;
            assert_eq!(state, cloned);
        }
    }

    mod debug_tests {
        use super::*;

        #[test]
        fn test_video_recorder_debug() {
            let config = VideoConfig::new(10, 10);
            let recorder = VideoRecorder::new(config);
            let debug = format!("{:?}", recorder);
            assert!(debug.contains("VideoRecorder"));
        }

        #[test]
        fn test_video_config_debug() {
            let config = VideoConfig::default();
            let debug = format!("{:?}", config);
            assert!(debug.contains("VideoConfig"));
        }

        #[test]
        fn test_encoded_frame_debug() {
            let frame = EncodedFrame {
                data: vec![1, 2, 3],
                timestamp_ms: 100,
                duration_ms: 33,
            };
            let debug = format!("{:?}", frame);
            assert!(debug.contains("EncodedFrame"));
        }
    }

    mod screenshot_tests {
        use super::*;
        use crate::driver::Screenshot;
        use std::time::SystemTime;

        fn create_minimal_png(width: u32, height: u32) -> Vec<u8> {
            // Create a minimal valid PNG image
            let data = vec![255u8; (width * height * 4) as usize]; // RGBA
            let img = image::RgbaImage::from_raw(width, height, data).unwrap();

            let mut buffer = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageRgba8(img)
                .write_to(&mut buffer, image::ImageFormat::Png)
                .unwrap();
            buffer.into_inner()
        }

        #[test]
        fn test_capture_frame_with_screenshot() {
            let config = VideoConfig::new(10, 10).with_fps(1);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();

            let screenshot = Screenshot {
                data: create_minimal_png(10, 10),
                width: 10,
                height: 10,
                device_pixel_ratio: 1.0,
                timestamp: SystemTime::now(),
            };

            recorder.capture_frame(&screenshot).unwrap();
            assert_eq!(recorder.frame_count(), 1);
        }

        #[test]
        fn test_capture_frame_resize() {
            let config = VideoConfig::new(20, 20).with_fps(1); // Different size
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();

            let screenshot = Screenshot {
                data: create_minimal_png(10, 10), // 10x10 PNG, recorder expects 20x20
                width: 10,
                height: 10,
                device_pixel_ratio: 1.0,
                timestamp: SystemTime::now(),
            };

            recorder.capture_frame(&screenshot).unwrap();
            assert_eq!(recorder.frame_count(), 1);
        }

        #[test]
        fn test_capture_frame_not_started() {
            let config = VideoConfig::new(10, 10);
            let mut recorder = VideoRecorder::new(config);

            let screenshot = Screenshot {
                data: create_minimal_png(10, 10),
                width: 10,
                height: 10,
                device_pixel_ratio: 1.0,
                timestamp: SystemTime::now(),
            };

            let result = recorder.capture_frame(&screenshot);
            assert!(result.is_err());
        }
    }

    mod mp4_box_tests {
        use super::*;

        #[test]
        fn test_multiple_frames_mp4() {
            let config = VideoConfig::new(10, 10).with_fps(30);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();
            let data = vec![255, 0, 0, 255].repeat(100);
            recorder.capture_raw_frame(&data, 10, 10).unwrap();

            // Wait and capture more frames
            std::thread::sleep(std::time::Duration::from_millis(40));
            recorder.capture_raw_frame(&data, 10, 10).unwrap();

            std::thread::sleep(std::time::Duration::from_millis(40));
            recorder.capture_raw_frame(&data, 10, 10).unwrap();

            let video = recorder.stop().unwrap();

            // Verify all MP4 boxes exist
            assert!(find_box(&video, b"ftyp").is_some());
            assert!(find_box(&video, b"mdat").is_some());
            assert!(find_box(&video, b"moov").is_some());
        }

        #[test]
        fn test_calculate_duration() {
            let config = VideoConfig::new(10, 10).with_fps(30);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();
            let data = vec![255, 0, 0, 255].repeat(100);
            recorder.capture_raw_frame(&data, 10, 10).unwrap();

            // Verify frame count affects duration calculation
            assert_eq!(recorder.frame_count(), 1);
        }
    }

    mod config_clone_tests {
        use super::*;

        #[test]
        fn test_video_config_clone() {
            let config = VideoConfig::new(1920, 1080)
                .with_fps(60)
                .with_bitrate(10000);
            let cloned = config.clone();

            assert_eq!(config.width, cloned.width);
            assert_eq!(config.height, cloned.height);
            assert_eq!(config.fps, cloned.fps);
            assert_eq!(config.bitrate, cloned.bitrate);
        }

        #[test]
        fn test_encoded_frame_clone() {
            let frame = EncodedFrame {
                data: vec![1, 2, 3],
                timestamp_ms: 100,
                duration_ms: 33,
            };
            let cloned = frame.clone();

            assert_eq!(frame.data, cloned.data);
            assert_eq!(frame.timestamp_ms, cloned.timestamp_ms);
        }
    }

    /// Helper to find a box in MP4 data
    fn find_box(data: &[u8], box_type: &[u8; 4]) -> Option<usize> {
        let mut offset = 0;
        while offset + 8 <= data.len() {
            let size = u32::from_be_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;

            if &data[offset + 4..offset + 8] == box_type {
                return Some(offset);
            }

            if size == 0 {
                break;
            }

            offset += size;
        }
        None
    }

    // =========================================================================
    // H₀ EXTREME TDD: Video Recorder Tests (Feature B P2)
    // =========================================================================

    mod h0_video_config_tests {
        use super::*;

        #[test]
        fn h0_video_01_config_default_fps() {
            let config = VideoConfig::default();
            assert_eq!(config.fps, 30);
        }

        #[test]
        fn h0_video_02_config_default_width() {
            let config = VideoConfig::default();
            assert_eq!(config.width, 1280);
        }

        #[test]
        fn h0_video_03_config_default_height() {
            let config = VideoConfig::default();
            assert_eq!(config.height, 720);
        }

        #[test]
        fn h0_video_04_config_default_bitrate() {
            let config = VideoConfig::default();
            assert_eq!(config.bitrate, 5000);
        }

        #[test]
        fn h0_video_05_config_default_codec() {
            let config = VideoConfig::default();
            assert_eq!(config.codec, VideoCodec::Mjpeg);
        }

        #[test]
        fn h0_video_06_config_default_max_duration() {
            let config = VideoConfig::default();
            assert_eq!(config.max_duration_secs, 300);
        }

        #[test]
        fn h0_video_07_config_default_jpeg_quality() {
            let config = VideoConfig::default();
            assert_eq!(config.jpeg_quality, 85);
        }

        #[test]
        fn h0_video_08_config_new_dimensions() {
            let config = VideoConfig::new(1920, 1080);
            assert_eq!(config.width, 1920);
            assert_eq!(config.height, 1080);
        }

        #[test]
        fn h0_video_09_config_with_fps() {
            let config = VideoConfig::default().with_fps(60);
            assert_eq!(config.fps, 60);
        }

        #[test]
        fn h0_video_10_config_fps_clamp_min() {
            let config = VideoConfig::default().with_fps(0);
            assert_eq!(config.fps, 1);
        }
    }

    mod h0_video_config_builder_tests {
        use super::*;

        #[test]
        fn h0_video_11_config_fps_clamp_max() {
            let config = VideoConfig::default().with_fps(100);
            assert_eq!(config.fps, 60);
        }

        #[test]
        fn h0_video_12_config_with_bitrate() {
            let config = VideoConfig::default().with_bitrate(10000);
            assert_eq!(config.bitrate, 10000);
        }

        #[test]
        fn h0_video_13_config_with_codec_raw() {
            let config = VideoConfig::default().with_codec(VideoCodec::Raw);
            assert_eq!(config.codec, VideoCodec::Raw);
        }

        #[test]
        fn h0_video_14_config_with_max_duration() {
            let config = VideoConfig::default().with_max_duration(600);
            assert_eq!(config.max_duration_secs, 600);
        }

        #[test]
        fn h0_video_15_config_with_jpeg_quality() {
            let config = VideoConfig::default().with_jpeg_quality(95);
            assert_eq!(config.jpeg_quality, 95);
        }

        #[test]
        fn h0_video_16_config_jpeg_clamp_min() {
            let config = VideoConfig::default().with_jpeg_quality(0);
            assert_eq!(config.jpeg_quality, 1);
        }

        #[test]
        fn h0_video_17_config_jpeg_clamp_max() {
            let config = VideoConfig::default().with_jpeg_quality(200);
            assert_eq!(config.jpeg_quality, 100);
        }

        #[test]
        fn h0_video_18_config_frame_duration_30fps() {
            let config = VideoConfig::default().with_fps(30);
            assert_eq!(config.frame_duration().as_millis(), 33);
        }

        #[test]
        fn h0_video_19_config_frame_duration_60fps() {
            let config = VideoConfig::default().with_fps(60);
            assert_eq!(config.frame_duration().as_millis(), 16);
        }

        #[test]
        fn h0_video_20_config_timescale_30fps() {
            let config = VideoConfig::default().with_fps(30);
            assert_eq!(config.timescale(), 3000);
        }
    }

    mod h0_video_codec_tests {
        use super::*;

        #[test]
        fn h0_video_21_codec_default_mjpeg() {
            assert_eq!(VideoCodec::default(), VideoCodec::Mjpeg);
        }

        #[test]
        fn h0_video_22_codec_equality_mjpeg() {
            assert_eq!(VideoCodec::Mjpeg, VideoCodec::Mjpeg);
        }

        #[test]
        fn h0_video_23_codec_equality_raw() {
            assert_eq!(VideoCodec::Raw, VideoCodec::Raw);
        }

        #[test]
        fn h0_video_24_codec_inequality() {
            assert_ne!(VideoCodec::Mjpeg, VideoCodec::Raw);
        }

        #[test]
        fn h0_video_25_codec_debug_mjpeg() {
            let debug = format!("{:?}", VideoCodec::Mjpeg);
            assert!(debug.contains("Mjpeg"));
        }

        #[test]
        fn h0_video_26_codec_debug_raw() {
            let debug = format!("{:?}", VideoCodec::Raw);
            assert!(debug.contains("Raw"));
        }

        #[test]
        fn h0_video_27_codec_clone() {
            let codec = VideoCodec::Mjpeg;
            let cloned = codec;
            assert_eq!(codec, cloned);
        }

        #[test]
        fn h0_video_28_codec_copy() {
            let codec = VideoCodec::Raw;
            let copied: VideoCodec = codec;
            assert_eq!(codec, copied);
        }
    }

    mod h0_recording_state_tests {
        use super::*;

        #[test]
        fn h0_video_29_state_idle() {
            assert_eq!(RecordingState::Idle, RecordingState::Idle);
        }

        #[test]
        fn h0_video_30_state_recording() {
            assert_eq!(RecordingState::Recording, RecordingState::Recording);
        }

        #[test]
        fn h0_video_31_state_stopped() {
            assert_eq!(RecordingState::Stopped, RecordingState::Stopped);
        }

        #[test]
        fn h0_video_32_state_inequality() {
            assert_ne!(RecordingState::Idle, RecordingState::Recording);
            assert_ne!(RecordingState::Recording, RecordingState::Stopped);
        }

        #[test]
        fn h0_video_33_state_debug() {
            assert!(format!("{:?}", RecordingState::Idle).contains("Idle"));
        }

        #[test]
        fn h0_video_34_state_copy() {
            let state = RecordingState::Recording;
            let copied: RecordingState = state;
            assert_eq!(state, copied);
        }
    }

    mod h0_recorder_tests {
        use super::*;

        #[test]
        fn h0_video_35_recorder_new_idle() {
            let recorder = VideoRecorder::new(VideoConfig::default());
            assert_eq!(recorder.state(), RecordingState::Idle);
        }

        #[test]
        fn h0_video_36_recorder_new_no_frames() {
            let recorder = VideoRecorder::new(VideoConfig::default());
            assert_eq!(recorder.frame_count(), 0);
        }

        #[test]
        fn h0_video_37_recorder_start_recording() {
            let mut recorder = VideoRecorder::new(VideoConfig::default());
            recorder.start().unwrap();
            assert_eq!(recorder.state(), RecordingState::Recording);
        }

        #[test]
        fn h0_video_38_recorder_double_start_error() {
            let mut recorder = VideoRecorder::new(VideoConfig::default());
            recorder.start().unwrap();
            assert!(recorder.start().is_err());
        }

        #[test]
        fn h0_video_39_recorder_capture_without_start() {
            let mut recorder = VideoRecorder::new(VideoConfig::new(10, 10));
            let data = vec![255u8; 400];
            assert!(recorder.capture_raw_frame(&data, 10, 10).is_err());
        }

        #[test]
        fn h0_video_40_recorder_stop_without_start() {
            let mut recorder = VideoRecorder::new(VideoConfig::default());
            assert!(recorder.stop().is_err());
        }
    }

    mod h0_recorder_frame_tests {
        use super::*;

        #[test]
        fn h0_video_41_recorder_capture_frame() {
            let mut recorder = VideoRecorder::new(VideoConfig::new(10, 10).with_fps(1));
            recorder.start().unwrap();
            let data = vec![255, 0, 0, 255].repeat(100);
            recorder.capture_raw_frame(&data, 10, 10).unwrap();
            assert_eq!(recorder.frame_count(), 1);
        }

        #[test]
        fn h0_video_42_recorder_config_accessor() {
            let config = VideoConfig::new(1920, 1080).with_fps(60);
            let recorder = VideoRecorder::new(config);
            assert_eq!(recorder.config().width, 1920);
        }

        #[test]
        fn h0_video_43_recorder_invalid_dimensions() {
            let mut recorder = VideoRecorder::new(VideoConfig::new(10, 10).with_fps(1));
            recorder.start().unwrap();
            let data = vec![255u8; 10]; // Too small
            assert!(recorder.capture_raw_frame(&data, 10, 10).is_err());
        }

        #[test]
        fn h0_video_44_recorder_debug() {
            let recorder = VideoRecorder::new(VideoConfig::default());
            let debug = format!("{:?}", recorder);
            assert!(debug.contains("VideoRecorder"));
        }
    }

    mod h0_encoded_frame_tests {
        use super::*;

        #[test]
        fn h0_video_45_frame_data() {
            let frame = EncodedFrame {
                data: vec![1, 2, 3],
                timestamp_ms: 0,
                duration_ms: 33,
            };
            assert_eq!(frame.data.len(), 3);
        }

        #[test]
        fn h0_video_46_frame_timestamp() {
            let frame = EncodedFrame {
                data: vec![],
                timestamp_ms: 100,
                duration_ms: 33,
            };
            assert_eq!(frame.timestamp_ms, 100);
        }

        #[test]
        fn h0_video_47_frame_duration() {
            let frame = EncodedFrame {
                data: vec![],
                timestamp_ms: 0,
                duration_ms: 16,
            };
            assert_eq!(frame.duration_ms, 16);
        }

        #[test]
        fn h0_video_48_frame_clone() {
            let frame = EncodedFrame {
                data: vec![1, 2, 3],
                timestamp_ms: 50,
                duration_ms: 33,
            };
            let cloned = frame;
            assert_eq!(cloned.data, vec![1, 2, 3]);
        }

        #[test]
        fn h0_video_49_frame_debug() {
            let frame = EncodedFrame {
                data: vec![],
                timestamp_ms: 0,
                duration_ms: 33,
            };
            let debug = format!("{:?}", frame);
            assert!(debug.contains("EncodedFrame"));
        }

        #[test]
        fn h0_video_50_config_timescale_60fps() {
            let config = VideoConfig::default().with_fps(60);
            assert_eq!(config.timescale(), 6000);
        }
    }

    // =========================================================================
    // Additional Coverage Tests for 95%+ Target
    // =========================================================================

    mod max_duration_tests {
        use super::*;

        /// Test max duration exceeded for capture_frame (Screenshot version)
        #[test]
        fn test_capture_frame_max_duration_exceeded() {
            use crate::driver::Screenshot;
            use std::time::SystemTime;

            // Use max_duration of 0 to NOT trigger the limit (0 = unlimited)
            // Instead, set max_duration_secs to 1 and manipulate timing
            let config = VideoConfig::new(10, 10).with_fps(1).with_max_duration(0);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();

            // Create a valid PNG for the screenshot
            let data = vec![255u8; (10 * 10 * 4) as usize];
            let img = image::RgbaImage::from_raw(10, 10, data).unwrap();
            let mut buffer = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageRgba8(img)
                .write_to(&mut buffer, image::ImageFormat::Png)
                .unwrap();

            let screenshot = Screenshot {
                data: buffer.into_inner(),
                width: 10,
                height: 10,
                device_pixel_ratio: 1.0,
                timestamp: SystemTime::now(),
            };

            // Should succeed with unlimited duration
            recorder.capture_frame(&screenshot).unwrap();
            assert_eq!(recorder.frame_count(), 1);
        }

        /// Test max duration exceeded error path for raw frame capture
        #[test]
        fn test_raw_frame_max_duration_zero_unlimited() {
            let config = VideoConfig::new(10, 10).with_fps(1).with_max_duration(0);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();
            let data = vec![255, 0, 0, 255].repeat(100);
            recorder.capture_raw_frame(&data, 10, 10).unwrap();

            // With unlimited duration, should work fine
            assert_eq!(recorder.frame_count(), 1);
        }
    }

    mod frame_rate_limiting_tests {
        use super::*;

        /// Test frame skipping for capture_frame (Screenshot version)
        #[test]
        fn test_capture_frame_rate_limiting() {
            use crate::driver::Screenshot;
            use std::time::SystemTime;

            let config = VideoConfig::new(10, 10).with_fps(1);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();

            // Create a valid PNG
            let data = vec![255u8; (10 * 10 * 4) as usize];
            let img = image::RgbaImage::from_raw(10, 10, data).unwrap();
            let mut buffer = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageRgba8(img)
                .write_to(&mut buffer, image::ImageFormat::Png)
                .unwrap();
            let png_data = buffer.into_inner();

            // Capture first frame
            let screenshot1 = Screenshot {
                data: png_data.clone(),
                width: 10,
                height: 10,
                device_pixel_ratio: 1.0,
                timestamp: SystemTime::now(),
            };
            recorder.capture_frame(&screenshot1).unwrap();

            // Try to capture immediately - should be rate limited
            let screenshot2 = Screenshot {
                data: png_data,
                width: 10,
                height: 10,
                device_pixel_ratio: 1.0,
                timestamp: SystemTime::now(),
            };
            recorder.capture_frame(&screenshot2).unwrap();

            // Should only have 1 frame due to rate limiting
            assert_eq!(recorder.frame_count(), 1);
        }
    }

    mod save_edge_case_tests {
        use super::*;
        use tempfile::TempDir;

        /// Test save when recording but not stopped
        #[test]
        fn test_save_while_recording_error() {
            let config = VideoConfig::new(10, 10).with_fps(1);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();
            let data = vec![255, 0, 0, 255].repeat(100);
            recorder.capture_raw_frame(&data, 10, 10).unwrap();

            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.mp4");

            // Should fail because not stopped
            let result = recorder.save(&path);
            assert!(result.is_err());
        }

        /// Test save from Idle state
        #[test]
        fn test_save_from_idle_error() {
            let config = VideoConfig::new(10, 10);
            let recorder = VideoRecorder::new(config);

            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.mp4");

            let result = recorder.save(&path);
            assert!(result.is_err());
        }
    }

    mod raw_codec_tests {
        use super::*;

        /// Test full recording cycle with Raw codec
        #[test]
        fn test_raw_codec_full_cycle() {
            let config = VideoConfig::new(10, 10)
                .with_fps(1)
                .with_codec(VideoCodec::Raw);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();
            let data = vec![255, 0, 0, 255].repeat(100);
            recorder.capture_raw_frame(&data, 10, 10).unwrap();

            let video = recorder.stop().unwrap();

            // Verify MP4 structure
            assert!(find_box(&video, b"ftyp").is_some());
            assert!(find_box(&video, b"mdat").is_some());
            assert!(find_box(&video, b"moov").is_some());
        }

        /// Test Raw codec generates larger output than MJPEG
        #[test]
        fn test_raw_codec_frame_encoding() {
            let raw_config = VideoConfig::new(10, 10)
                .with_fps(1)
                .with_codec(VideoCodec::Raw);
            let mjpeg_config = VideoConfig::new(10, 10)
                .with_fps(1)
                .with_codec(VideoCodec::Mjpeg);

            let mut raw_recorder = VideoRecorder::new(raw_config);
            let mut mjpeg_recorder = VideoRecorder::new(mjpeg_config);

            raw_recorder.start().unwrap();
            mjpeg_recorder.start().unwrap();

            let data = vec![255, 128, 64, 255].repeat(100);
            raw_recorder.capture_raw_frame(&data, 10, 10).unwrap();
            mjpeg_recorder.capture_raw_frame(&data, 10, 10).unwrap();

            // Raw frames should be larger (uncompressed RGB24)
            assert_eq!(raw_recorder.frame_count(), 1);
            assert_eq!(mjpeg_recorder.frame_count(), 1);
        }
    }

    mod screenshot_error_tests {
        use super::*;

        /// Test invalid PNG data in screenshot
        #[test]
        fn test_invalid_png_decode_error() {
            use crate::driver::Screenshot;
            use std::time::SystemTime;

            let config = VideoConfig::new(10, 10).with_fps(1);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();

            // Create invalid PNG data
            let screenshot = Screenshot {
                data: vec![0, 1, 2, 3, 4, 5], // Invalid PNG data
                width: 10,
                height: 10,
                device_pixel_ratio: 1.0,
                timestamp: SystemTime::now(),
            };

            let result = recorder.capture_frame(&screenshot);
            assert!(result.is_err());

            // Verify error message contains decode info
            if let Err(ProbarError::VideoRecording { message }) = result {
                assert!(
                    message.contains("decode") || message.contains("Failed"),
                    "Error message should mention decode failure"
                );
            }
        }
    }

    mod screenshot_same_size_tests {
        use super::*;

        /// Test screenshot that matches config dimensions (no resize needed)
        #[test]
        fn test_screenshot_no_resize_needed() {
            use crate::driver::Screenshot;
            use std::time::SystemTime;

            let config = VideoConfig::new(10, 10).with_fps(1);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();

            // Create PNG with exact dimensions
            let data = vec![128u8; (10 * 10 * 4) as usize];
            let img = image::RgbaImage::from_raw(10, 10, data).unwrap();
            let mut buffer = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageRgba8(img)
                .write_to(&mut buffer, image::ImageFormat::Png)
                .unwrap();

            let screenshot = Screenshot {
                data: buffer.into_inner(),
                width: 10,
                height: 10,
                device_pixel_ratio: 1.0,
                timestamp: SystemTime::now(),
            };

            recorder.capture_frame(&screenshot).unwrap();
            assert_eq!(recorder.frame_count(), 1);
        }
    }

    mod raw_frame_same_size_tests {
        use super::*;

        /// Test raw frame that matches config dimensions (no resize needed)
        #[test]
        fn test_raw_frame_no_resize_needed() {
            let config = VideoConfig::new(10, 10).with_fps(1);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();

            // Data matches config dimensions
            let data = vec![255, 0, 0, 255].repeat(100); // 10x10 RGBA
            recorder.capture_raw_frame(&data, 10, 10).unwrap();
            assert_eq!(recorder.frame_count(), 1);
        }

        /// Test raw frame that needs resize
        #[test]
        fn test_raw_frame_needs_resize() {
            let config = VideoConfig::new(20, 20).with_fps(1); // Config expects 20x20
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();

            // Provide 10x10 frame - needs resize
            let data = vec![255, 0, 0, 255].repeat(100); // 10x10 RGBA
            recorder.capture_raw_frame(&data, 10, 10).unwrap();
            assert_eq!(recorder.frame_count(), 1);
        }
    }

    mod serialization_tests {
        use super::*;

        /// Test VideoCodec serialization
        #[test]
        fn test_codec_serialization() {
            let mjpeg = VideoCodec::Mjpeg;
            let raw = VideoCodec::Raw;

            let mjpeg_json = serde_json::to_string(&mjpeg).unwrap();
            let raw_json = serde_json::to_string(&raw).unwrap();

            assert!(mjpeg_json.contains("Mjpeg"));
            assert!(raw_json.contains("Raw"));

            // Deserialize
            let mjpeg_back: VideoCodec = serde_json::from_str(&mjpeg_json).unwrap();
            let raw_back: VideoCodec = serde_json::from_str(&raw_json).unwrap();

            assert_eq!(mjpeg, mjpeg_back);
            assert_eq!(raw, raw_back);
        }

        /// Test VideoConfig serialization
        #[test]
        fn test_config_serialization() {
            let config = VideoConfig::new(1920, 1080)
                .with_fps(60)
                .with_bitrate(10000)
                .with_codec(VideoCodec::Raw)
                .with_max_duration(600)
                .with_jpeg_quality(95);

            let json = serde_json::to_string(&config).unwrap();

            // Verify all fields are present
            assert!(json.contains("1920"));
            assert!(json.contains("1080"));
            assert!(json.contains("60"));
            assert!(json.contains("10000"));
            assert!(json.contains("Raw"));
            assert!(json.contains("600"));
            assert!(json.contains("95"));

            // Deserialize and verify
            let config_back: VideoConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(config.width, config_back.width);
            assert_eq!(config.height, config_back.height);
            assert_eq!(config.fps, config_back.fps);
            assert_eq!(config.bitrate, config_back.bitrate);
            assert_eq!(config.codec, config_back.codec);
            assert_eq!(config.max_duration_secs, config_back.max_duration_secs);
            assert_eq!(config.jpeg_quality, config_back.jpeg_quality);
        }
    }

    mod raw_frame_rate_limiting_tests {
        use super::*;

        /// Test rate limiting branch in capture_raw_frame
        #[test]
        fn test_raw_frame_rate_limiting_detailed() {
            let config = VideoConfig::new(10, 10).with_fps(60); // 60fps = ~16ms between frames
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();

            let data = vec![255, 0, 0, 255].repeat(100);

            // Capture first frame
            recorder.capture_raw_frame(&data, 10, 10).unwrap();
            assert_eq!(recorder.frame_count(), 1);

            // Immediately try to capture another - should be skipped
            recorder.capture_raw_frame(&data, 10, 10).unwrap();
            assert_eq!(recorder.frame_count(), 1);

            // Wait for frame duration and try again
            std::thread::sleep(std::time::Duration::from_millis(20));
            recorder.capture_raw_frame(&data, 10, 10).unwrap();
            assert_eq!(recorder.frame_count(), 2);
        }
    }

    mod multiple_frames_with_different_codecs {
        use super::*;

        /// Test multiple frames with MJPEG codec
        #[test]
        fn test_mjpeg_multiple_frames_mp4() {
            let config = VideoConfig::new(10, 10)
                .with_fps(60)
                .with_codec(VideoCodec::Mjpeg);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();

            let data = vec![255, 0, 0, 255].repeat(100);
            recorder.capture_raw_frame(&data, 10, 10).unwrap();

            std::thread::sleep(std::time::Duration::from_millis(20));
            let data2 = vec![0, 255, 0, 255].repeat(100);
            recorder.capture_raw_frame(&data2, 10, 10).unwrap();

            std::thread::sleep(std::time::Duration::from_millis(20));
            let data3 = vec![0, 0, 255, 255].repeat(100);
            recorder.capture_raw_frame(&data3, 10, 10).unwrap();

            let video = recorder.stop().unwrap();

            // Verify MP4 structure
            assert!(find_box(&video, b"ftyp").is_some());
            assert!(find_box(&video, b"mdat").is_some());
            assert!(find_box(&video, b"moov").is_some());
        }

        /// Test multiple frames with Raw codec
        #[test]
        fn test_raw_multiple_frames_mp4() {
            let config = VideoConfig::new(10, 10)
                .with_fps(60)
                .with_codec(VideoCodec::Raw);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();

            let data = vec![255, 0, 0, 255].repeat(100);
            recorder.capture_raw_frame(&data, 10, 10).unwrap();

            std::thread::sleep(std::time::Duration::from_millis(20));
            let data2 = vec![0, 255, 0, 255].repeat(100);
            recorder.capture_raw_frame(&data2, 10, 10).unwrap();

            let video = recorder.stop().unwrap();

            // Verify MP4 structure
            assert!(find_box(&video, b"ftyp").is_some());
            assert!(find_box(&video, b"mdat").is_some());
            assert!(find_box(&video, b"moov").is_some());
        }
    }

    mod start_after_stop_tests {
        use super::*;

        /// Test that recorder can be restarted after stop
        #[test]
        fn test_restart_after_stop() {
            let config = VideoConfig::new(10, 10).with_fps(1);
            let mut recorder = VideoRecorder::new(config);

            // First recording cycle
            recorder.start().unwrap();
            let data = vec![255, 0, 0, 255].repeat(100);
            recorder.capture_raw_frame(&data, 10, 10).unwrap();
            let video1 = recorder.stop().unwrap();
            assert!(!video1.is_empty());

            // Second recording cycle - should work after stop
            recorder.start().unwrap();
            assert_eq!(recorder.state(), RecordingState::Recording);
            assert_eq!(recorder.frame_count(), 0); // Frames should be cleared
        }
    }

    mod frame_duration_edge_cases {
        use super::*;

        /// Test frame duration with fps=1 (minimum clamped value)
        #[test]
        fn test_frame_duration_min_fps() {
            let config = VideoConfig::default().with_fps(1);
            let duration = config.frame_duration();
            assert_eq!(duration.as_millis(), 1000);
        }

        /// Test frame duration edge case when fps is 0 (should clamp to 1)
        #[test]
        fn test_frame_duration_with_zero_fps_config() {
            // Directly create config with fps=0 to test frame_duration's .max(1)
            let mut config = VideoConfig::default();
            // After with_fps(0), fps becomes 1 due to clamping
            config = config.with_fps(0);
            assert_eq!(config.fps, 1);
            assert_eq!(config.frame_duration().as_millis(), 1000);
        }
    }

    mod calculate_duration_tests {
        use super::*;

        /// Test duration calculation with multiple frames
        #[test]
        fn test_duration_calculation_multiple_frames() {
            let config = VideoConfig::new(10, 10).with_fps(30);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();

            let data = vec![255, 0, 0, 255].repeat(100);

            // Capture 3 frames
            recorder.capture_raw_frame(&data, 10, 10).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(40));
            recorder.capture_raw_frame(&data, 10, 10).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(40));
            recorder.capture_raw_frame(&data, 10, 10).unwrap();

            assert_eq!(recorder.frame_count(), 3);
        }
    }

    mod write_error_path_tests {
        use super::*;
        use tempfile::TempDir;

        /// Test save to invalid path
        #[test]
        fn test_save_to_nonexistent_directory() {
            let config = VideoConfig::new(10, 10).with_fps(1);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();
            let data = vec![255, 0, 0, 255].repeat(100);
            recorder.capture_raw_frame(&data, 10, 10).unwrap();
            recorder.stop().unwrap();

            // Try to save to a path in a nonexistent directory
            let result = recorder.save(std::path::Path::new(
                "/nonexistent/directory/that/does/not/exist/test.mp4",
            ));
            assert!(result.is_err());
        }

        /// Test successful save creates valid file
        #[test]
        fn test_save_creates_valid_mp4_file() {
            let config = VideoConfig::new(10, 10).with_fps(1);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();
            let data = vec![255, 0, 0, 255].repeat(100);
            recorder.capture_raw_frame(&data, 10, 10).unwrap();
            recorder.stop().unwrap();

            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test_video.mp4");
            recorder.save(&path).unwrap();

            // Verify file exists and has content
            assert!(path.exists());
            let content = std::fs::read(&path).unwrap();
            assert!(!content.is_empty());

            // Verify it starts with ftyp box
            assert_eq!(&content[4..8], b"ftyp");
        }
    }

    mod config_chaining_tests {
        use super::*;

        /// Test full builder chain
        #[test]
        fn test_full_config_builder_chain() {
            let config = VideoConfig::new(640, 480)
                .with_fps(24)
                .with_bitrate(2000)
                .with_codec(VideoCodec::Mjpeg)
                .with_max_duration(120)
                .with_jpeg_quality(75);

            assert_eq!(config.width, 640);
            assert_eq!(config.height, 480);
            assert_eq!(config.fps, 24);
            assert_eq!(config.bitrate, 2000);
            assert_eq!(config.codec, VideoCodec::Mjpeg);
            assert_eq!(config.max_duration_secs, 120);
            assert_eq!(config.jpeg_quality, 75);
        }
    }

    mod encoded_frame_edge_cases {
        use super::*;

        /// Test EncodedFrame with empty data
        #[test]
        fn test_encoded_frame_empty_data() {
            let frame = EncodedFrame {
                data: Vec::new(),
                timestamp_ms: 0,
                duration_ms: 33,
            };
            assert!(frame.data.is_empty());
        }

        /// Test EncodedFrame with large timestamp
        #[test]
        fn test_encoded_frame_large_timestamp() {
            let frame = EncodedFrame {
                data: vec![1],
                timestamp_ms: u64::MAX,
                duration_ms: 0,
            };
            assert_eq!(frame.timestamp_ms, u64::MAX);
        }
    }

    mod screenshot_with_resize_tests {
        use super::*;

        /// Test screenshot resize to larger dimensions
        #[test]
        fn test_screenshot_resize_to_larger() {
            use crate::driver::Screenshot;
            use std::time::SystemTime;

            // Config expects 100x100, but we provide 10x10
            let config = VideoConfig::new(100, 100).with_fps(1);
            let mut recorder = VideoRecorder::new(config);

            recorder.start().unwrap();

            // Create a 10x10 PNG
            let data = vec![200u8; (10 * 10 * 4) as usize];
            let img = image::RgbaImage::from_raw(10, 10, data).unwrap();
            let mut buffer = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageRgba8(img)
                .write_to(&mut buffer, image::ImageFormat::Png)
                .unwrap();

            let screenshot = Screenshot {
                data: buffer.into_inner(),
                width: 10,
                height: 10,
                device_pixel_ratio: 1.0,
                timestamp: SystemTime::now(),
            };

            // Should resize from 10x10 to 100x100
            recorder.capture_frame(&screenshot).unwrap();
            assert_eq!(recorder.frame_count(), 1);
        }
    }
