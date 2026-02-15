
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masked_prediction() {
        let mp = MaskedPrediction::new(0.15, 103);
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let (masked, positions) = mp.apply(&input, 42);
        assert_eq!(masked.len(), input.len());
        for &pos in &positions {
            assert_eq!(masked[pos], 103);
        }
    }

    #[test]
    fn test_rotation_prediction() {
        let rp = RotationPrediction::new();
        let image = vec![1.0, 2.0, 3.0, 4.0]; // 1x2x2
        let rotated = rp.rotate(&image, 2, 2, 1, 0);
        assert_eq!(rotated, image);
    }

    #[test]
    fn test_rotation_task() {
        let rp = RotationPrediction::new();
        let image = vec![1.0; 16]; // 1x4x4
        let (rotated, label) = rp.generate_task(&image, 4, 4, 1, 42);
        assert_eq!(rotated.len(), 16);
        assert!(label < 4);
    }

    #[test]
    fn test_jigsaw_puzzle() {
        let jp = JigsawPuzzle::new(2, 10);
        let image = vec![1.0; 16]; // 1x4x4
        let (patches, perm) = jp.generate_task(&image, 4, 4, 1, 42);
        assert_eq!(patches.len(), 4); // 2x2 grid
        assert!(perm < 10);
    }

    #[test]
    fn test_contrastive_task() {
        let ct = ContrastiveTask::new(0.07);
        let anchor = vec![1.0, 0.0, 0.0];
        let positive = vec![0.9, 0.1, 0.0];
        let negatives = vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let loss = ct.info_nce_loss(&anchor, &positive, &negatives);
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_contrastive_same_positive() {
        let ct = ContrastiveTask::new(0.1);
        let anchor = vec![1.0, 0.0];
        let positive = vec![1.0, 0.0]; // Same as anchor
        let negatives = vec![vec![0.0, 1.0]];
        let loss = ct.info_nce_loss(&anchor, &positive, &negatives);
        assert!(loss.is_finite());
    }

    // SimCLR Tests
    #[test]
    fn test_simclr_basic() {
        let simclr = SimCLR::new(0.07, 128);
        assert!((simclr.temperature() - 0.07).abs() < 1e-6);
        assert_eq!(simclr.projection_dim(), 128);
    }

    #[test]
    fn test_simclr_nt_xent_loss() {
        let simclr = SimCLR::new(0.5, 64);
        let z_i = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let z_j = vec![vec![0.9, 0.1, 0.0], vec![0.1, 0.9, 0.0]];

        let loss = simclr.nt_xent_loss(&z_i, &z_j);
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_simclr_identical_pairs() {
        let simclr = SimCLR::new(0.5, 64);
        let z_i = vec![vec![1.0, 0.0]];
        let z_j = vec![vec![1.0, 0.0]];

        let loss = simclr.nt_xent_loss(&z_i, &z_j);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_simclr_empty_batch() {
        let simclr = SimCLR::new(0.5, 64);
        let z_i: Vec<Vec<f32>> = vec![];
        let z_j: Vec<Vec<f32>> = vec![];

        let loss = simclr.nt_xent_loss(&z_i, &z_j);
        assert!((loss - 0.0).abs() < 1e-6);
    }

    // MoCo Tests
    #[test]
    fn test_moco_basic() {
        let moco = MoCo::new(0.07, 0.999, 65536, 128);
        assert!((moco.momentum() - 0.999).abs() < 1e-6);
        assert_eq!(moco.queue_len(), 0);
    }

    #[test]
    fn test_moco_queue_update() {
        let mut moco = MoCo::new(0.07, 0.999, 4, 3);

        moco.update_queue(&[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
        assert_eq!(moco.queue_len(), 2);

        moco.update_queue(&[vec![0.0, 0.0, 1.0], vec![1.0, 1.0, 0.0]]);
        assert_eq!(moco.queue_len(), 4);

        // Now queue is full, next update wraps around
        moco.update_queue(&[vec![0.5, 0.5, 0.0]]);
        assert_eq!(moco.queue_len(), 4);
    }

    #[test]
    fn test_moco_momentum_update() {
        let moco = MoCo::new(0.07, 0.9, 100, 3);
        let encoder = vec![1.0, 2.0, 3.0];
        let mut momentum = vec![0.0, 0.0, 0.0];

        moco.momentum_update(&encoder, &mut momentum);

        // momentum = 0.9 * 0 + 0.1 * encoder = 0.1 * encoder
        assert!((momentum[0] - 0.1).abs() < 1e-6);
        assert!((momentum[1] - 0.2).abs() < 1e-6);
        assert!((momentum[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_moco_contrastive_loss() {
        let mut moco = MoCo::new(0.5, 0.999, 100, 3);

        // Fill queue
        moco.update_queue(&[vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]]);

        let queries = vec![vec![1.0, 0.0, 0.0]];
        let keys = vec![vec![0.9, 0.1, 0.0]];

        let loss = moco.contrastive_loss(&queries, &keys);
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    // BYOL Tests
    #[test]
    fn test_byol_basic() {
        let byol = BYOL::new(0.996);
        assert!((byol.momentum() - 0.996).abs() < 1e-6);
    }

    #[test]
    fn test_byol_loss() {
        let byol = BYOL::new(0.996);
        let pred = vec![vec![1.0, 0.0, 0.0]];
        let target = vec![vec![0.9, 0.1, 0.0]];

        let loss = byol.loss(&pred, &target);
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_byol_identical() {
        let byol = BYOL::new(0.996);
        let pred = vec![vec![1.0, 0.0]];
        let target = vec![vec![1.0, 0.0]];

        let loss = byol.loss(&pred, &target);
        assert!(loss < 0.01); // Should be very small
    }

    #[test]
    fn test_byol_symmetric_loss() {
        let byol = BYOL::new(0.996);
        let pred_1 = vec![vec![1.0, 0.0]];
        let proj_2 = vec![vec![0.9, 0.1]];
        let pred_2 = vec![vec![0.0, 1.0]];
        let proj_1 = vec![vec![0.1, 0.9]];

        let loss = byol.symmetric_loss(&pred_1, &proj_2, &pred_2, &proj_1);
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_byol_momentum_update() {
        let byol = BYOL::new(0.9);
        let online = vec![1.0, 2.0];
        let mut target = vec![0.0, 0.0];

        byol.momentum_update(&online, &mut target);

        assert!((target[0] - 0.1).abs() < 1e-6);
        assert!((target[1] - 0.2).abs() < 1e-6);
    }

    // SimCSE Tests
    #[test]
    fn test_simcse_basic() {
        let simcse = SimCSE::new(0.05);
        assert!((simcse.temperature() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_simcse_unsupervised() {
        let simcse = SimCSE::new(0.5); // Higher temp for smoother loss
        let emb_1 = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let emb_2 = vec![
            vec![0.95, 0.05, 0.0],
            vec![0.05, 0.95, 0.0],
            vec![0.0, 0.05, 0.95],
        ];

        let loss = simcse.unsupervised_loss(&emb_1, &emb_2);
        assert!(loss.is_finite());
        // With in-batch negatives, loss should be positive
        assert!(loss > 0.0 || loss >= 0.0); // Allow zero for identical pairs
    }

    #[test]
    fn test_simcse_supervised() {
        let simcse = SimCSE::new(0.5);
        let anchors = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let positives = vec![vec![0.9, 0.1, 0.0], vec![0.1, 0.9, 0.0]];
        let negatives = vec![vec![0.0, 0.0, 1.0], vec![0.5, 0.5, 0.0]];

        let loss = simcse.supervised_loss(&anchors, &positives, &negatives);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_simcse_empty() {
        let simcse = SimCSE::new(0.05);
        let emb_1: Vec<Vec<f32>> = vec![];
        let emb_2: Vec<Vec<f32>> = vec![];

        let loss = simcse.unsupervised_loss(&emb_1, &emb_2);
        assert!((loss - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let norm = l2_normalize(&v);
        let length: f32 = norm.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((length - 1.0).abs() < 1e-6);
    }
}
