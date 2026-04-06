
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alibi_get_bias_shape() {
        let alibi = ALiBi::new(4).expect("alibi");
        let bias = alibi.get_bias(8).expect("bias");
        assert_eq!(bias.shape(), &[8, 8, 4]);
    }

    #[test]
    fn test_alibi_get_bias_zero_seq_error() {
        let alibi = ALiBi::new(4).expect("alibi");
        let result = alibi.get_bias(0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("seq_len must be > 0"));
    }

    #[test]
    fn test_alibi_get_bias_diagonal_is_zero() {
        let alibi = ALiBi::new(2).expect("alibi");
        let bias = alibi.get_bias(3).expect("bias");
        let data = bias.data();
        // At (i, i), distance = 0, so bias = 0
        // Bias is [seq_len, seq_len, num_heads], linearized as [i][j][h]
        // (0,0,0): data[0*3*2 + 0*2 + 0] = data[0]
        // (1,1,0): data[1*3*2 + 1*2 + 0] = data[8]
        // (2,2,0): data[2*3*2 + 2*2 + 0] = data[16]
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[8] - 0.0).abs() < 1e-6);
        assert!((data[16] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_get_bias_values() {
        let alibi = ALiBi::new(1).expect("alibi");
        let bias = alibi.get_bias(3).expect("bias");
        let data = bias.data();
        let slope = alibi.slopes()[0];
        // With 1 head: bias[i,j,0] = -slope * |i-j|
        // (0,0): 0, (0,1): -slope*1, (0,2): -slope*2
        // (1,0): -slope*1, (1,1): 0, (1,2): -slope*1
        // (2,0): -slope*2, (2,1): -slope*1, (2,2): 0
        assert!((data[0] - 0.0).abs() < 1e-6); // (0,0)
        assert!((data[1] - (-slope * 1.0)).abs() < 1e-6); // (0,1)
        assert!((data[2] - (-slope * 2.0)).abs() < 1e-6); // (0,2)
        assert!((data[3] - (-slope * 1.0)).abs() < 1e-6); // (1,0)
        assert!((data[4] - 0.0).abs() < 1e-6); // (1,1)
        assert!((data[5] - (-slope * 1.0)).abs() < 1e-6); // (1,2)
    }

    #[test]
    fn test_alibi_bias_symmetry() {
        let alibi = ALiBi::new(2).expect("alibi");
        let bias = alibi.get_bias(4).expect("bias");
        let data = bias.data();
        // bias[i,j,h] should have |bias[i,j,h]| == |bias[j,i,h]|
        // Since we use |i-j|, bias[i,j,h] == bias[j,i,h] (both negative)
        for i in 0..4 {
            for j in 0..4 {
                for h in 0..2 {
                    let idx_ij = i * 4 * 2 + j * 2 + h;
                    let idx_ji = j * 4 * 2 + i * 2 + h;
                    assert!((data[idx_ij] - data[idx_ji]).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_alibi_clone() {
        let alibi = ALiBi::new(4).expect("alibi");
        let cloned = alibi.clone();
        assert_eq!(alibi.num_heads(), cloned.num_heads());
        assert_eq!(alibi.slopes(), cloned.slopes());
    }

    #[test]
    fn test_alibi_large_seq_len() {
        let alibi = ALiBi::new(8).expect("alibi");
        let bias = alibi.get_bias(256).expect("bias");
        assert_eq!(bias.shape(), &[256, 256, 8]);
        // Check a corner case: (0, 255) should be -slope * 255
        let data = bias.data();
        let idx = 255 * 8; // (0, 255, head=0)
        let expected = -alibi.slopes()[0] * 255.0;
        assert!((data[idx] - expected).abs() < 1e-4);
    }

    #[test]
    fn test_alibi_non_power_of_two_slopes() {
        // For 6 heads: closest power of 2 is 4
        // First 4 slopes: 2^(-8h/4) = 2^(-2h) for h=0..4
        // Extra 2 slopes: 2^(-(2h+1)*4/4) = 2^(-2h-1) for h=0..2
        let alibi = ALiBi::new(6).expect("alibi");
        let slopes = alibi.slopes();
        assert_eq!(slopes.len(), 6);
        // First 4: [1.0, 0.25, 0.0625, 0.015625]
        assert!((slopes[0] - 1.0).abs() < 1e-6);
        assert!((slopes[3] - 0.015625).abs() < 1e-6);
        // Extra 2 have different computation
        assert!(slopes[4] > 0.0);
        assert!(slopes[5] > 0.0);
    }
include!("position_rope.rs");
}
