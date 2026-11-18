
    fn brick_with_gap(budget_us: f64, actual_us: f64) -> BrickTiming {
        let mut b = BrickTiming::new("t", budget_us);
        b.actual_us = actual_us;
        b
    }

    #[test]
    fn score_brick_gap_table() {
        // (budget_us, actual_us, expected score, expected grade)
        let cases: [(f64, f64, u32, &str); 8] = [
            (10.0, 10.0, 100, "A"),
            (10.0, 5.0, 100, "A"),
            (10.0, 11.0, 95, "A"),
            (10.0, 12.0, 90, "A"),
            (10.0, 15.0, 50, "F"),
            (10.0, 30.0, 0, "F"),
            (10.0, 1000.0, 0, "F"),
            (0.0, 42.0, 100, "A"),
        ];
        for (budget, actual, want_score, want_grade) in cases {
            let s = score_brick(&brick_with_gap(budget, actual));
            assert_eq!(s.score, want_score, "budget={budget} actual={actual}");
            assert_eq!(s.grade, want_grade, "budget={budget} actual={actual}");
        }
    }

    #[test]
    fn score_brick_copies_fields() {
        let s = score_brick(&brick_with_gap(10.0, 15.0));
        assert_eq!(s.name, "t");
        assert_eq!(s.budget_us, 10.0);
        assert_eq!(s.actual_us, 15.0);
        assert!((s.gap_factor - 1.5).abs() < 1e-9);
    }

    #[test]
    fn score_brick_grade_boundaries() {
        // Slow branch (gap > 1.2): score = 100 - (gap-1)*100.
        // gap 1.3 -> 70 (C), 1.4 -> 60 (D), 1.41 -> 59 (F), 1.2 -> 90 (A).
        let a = score_brick(&brick_with_gap(100.0, 120.0));
        assert_eq!((a.score, a.grade.as_str()), (90, "A"));
        let c = score_brick(&brick_with_gap(100.0, 130.0));
        assert_eq!(c.grade, "C");
        let d = score_brick(&brick_with_gap(100.0, 140.0));
        assert_eq!(d.grade, "D");
        let f = score_brick(&brick_with_gap(100.0, 141.0));
        assert_eq!(f.grade, "F");
    }

    fn scored(score: u32) -> BrickScore {
        BrickScore {
            name: "x".to_string(),
            score,
            grade: score_to_grade(score),
            budget_us: 1.0,
            actual_us: 1.0,
            gap_factor: 1.0,
        }
    }

    #[test]
    fn weighted_brick_score_empty_is_zero() {
        assert_eq!(weighted_brick_score(&[]), 0);
    }

    #[test]
    fn weighted_brick_score_averages_and_truncates() {
        assert_eq!(weighted_brick_score(&[scored(100), scored(50)]), 75);
        assert_eq!(weighted_brick_score(&[scored(100), scored(51)]), 75); // 75.5 truncates
        assert_eq!(weighted_brick_score(&[scored(90)]), 90);
        // Bricks beyond the 7th are no longer dropped (GH-422 B6b).
        let eight = [100, 100, 100, 100, 100, 100, 100, 0].map(scored);
        assert_eq!(weighted_brick_score(&eight), 87);
    }

    #[test]
    fn score_to_grade_boundaries() {
        for (s, g) in [(100, "A"), (90, "A"), (89, "B"), (80, "B"), (79, "C"), (70, "C"), (69, "D"), (60, "D"), (59, "F"), (0, "F")] {
            assert_eq!(score_to_grade(s), g, "score {s}");
        }
    }
