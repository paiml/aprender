use super::*;

#[test]
fn test_gru_creation() {
    let gru = GRU::new(10, 20);
    assert_eq!(gru.input_size(), 10);
    assert_eq!(gru.hidden_size(), 20);
}

#[test]
fn test_gru_forward_step() {
    let gru = GRU::new(4, 8);
    let x = Tensor::ones(&[2, 4]);
    let h = Tensor::zeros(&[2, 8]);

    let h_new = gru.forward_step(&x, &h);
    assert_eq!(h_new.shape(), &[2, 8]);
}

#[test]
fn test_gru_forward_sequence() {
    let gru = GRU::new(4, 8);
    let x = Tensor::ones(&[2, 5, 4]); // batch=2, seq=5, input=4

    let (output, h_final) = gru.forward_sequence(&x, None);
    assert_eq!(output.shape(), &[2, 5, 8]);
    assert_eq!(h_final.shape(), &[2, 8]);
}

#[test]
fn test_gru_module_forward() {
    let gru = GRU::new(4, 8);
    let x = Tensor::ones(&[2, 5, 4]);

    let output = gru.forward(&x);
    assert_eq!(output.shape(), &[2, 5, 8]);
}

#[test]
fn test_gru_parameters() {
    let gru = GRU::new(4, 8);
    let params = gru.parameters();
    // 6 linear layers * 2 (weight + bias) = 12
    assert_eq!(params.len(), 12);
}

#[test]
fn test_gru_train_eval() {
    let mut gru = GRU::new(4, 8);
    assert!(gru.training());
    gru.eval();
    assert!(!gru.training());
}

#[test]
fn test_gru_with_initial_hidden() {
    let gru = GRU::new(4, 8);
    let x = Tensor::ones(&[2, 3, 4]);
    let h0 = Tensor::ones(&[2, 8]);

    let (output, _) = gru.forward_sequence(&x, Some(&h0));
    assert_eq!(output.shape(), &[2, 3, 8]);
}

#[test]
fn test_sigmoid_tensor() {
    let x = Tensor::new(&[0.0, 10.0, -10.0], &[3]);
    let y = sigmoid_tensor(&x);

    assert!((y.data()[0] - 0.5).abs() < 1e-5);
    assert!(y.data()[1] > 0.99);
    assert!(y.data()[2] < 0.01);
}

#[test]
fn test_tanh_tensor() {
    let x = Tensor::new(&[0.0, 10.0, -10.0], &[3]);
    let y = tanh_tensor(&x);

    assert!((y.data()[0] - 0.0).abs() < 1e-5);
    assert!((y.data()[1] - 1.0).abs() < 1e-5);
    assert!((y.data()[2] + 1.0).abs() < 1e-5);
}

// Bidirectional tests

#[test]
fn test_bidirectional_creation() {
    let bi = Bidirectional::new(4, 8);
    assert_eq!(bi.hidden_size(), 8);
    assert_eq!(bi.output_size(), 16);
}

#[test]
fn test_bidirectional_forward() {
    let bi = Bidirectional::new(4, 8);
    let x = Tensor::ones(&[2, 5, 4]); // batch=2, seq=5, input=4

    let output = bi.forward(&x);
    assert_eq!(output.shape(), &[2, 5, 16]); // hidden*2
}

#[test]
fn test_bidirectional_forward_sequence() {
    let bi = Bidirectional::new(4, 8);
    let x = Tensor::ones(&[2, 5, 4]);

    let (output, fwd_h, bwd_h) = bi.forward_sequence(&x);
    assert_eq!(output.shape(), &[2, 5, 16]);
    assert_eq!(fwd_h.shape(), &[2, 8]);
    assert_eq!(bwd_h.shape(), &[2, 8]);
}

#[test]
fn test_bidirectional_parameters() {
    let bi = Bidirectional::new(4, 8);
    let params = bi.parameters();
    // 2 GRUs * 12 params each = 24
    assert_eq!(params.len(), 24);
}

#[test]
fn test_bidirectional_train_eval() {
    let mut bi = Bidirectional::new(4, 8);
    assert!(bi.training());
    bi.eval();
    assert!(!bi.training());
}

#[test]
fn test_reverse_sequence() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 3, 2]);
    let rev = reverse_sequence(&x);

    // [1,2], [3,4], [5,6] -> [5,6], [3,4], [1,2]
    assert_eq!(rev.data(), &[5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
}

// LSTM Tests

#[test]
fn test_lstm_creation() {
    let lstm = LSTM::new(10, 20);
    assert_eq!(lstm.input_size(), 10);
    assert_eq!(lstm.hidden_size(), 20);
}

#[test]
fn test_lstm_forward_step() {
    let lstm = LSTM::new(4, 8);
    let x = Tensor::ones(&[2, 4]);
    let h = Tensor::zeros(&[2, 8]);
    let c = Tensor::zeros(&[2, 8]);

    let (h_new, c_new) = lstm.forward_step(&x, &h, &c);
    assert_eq!(h_new.shape(), &[2, 8]);
    assert_eq!(c_new.shape(), &[2, 8]);
}

#[test]
fn test_lstm_forward_sequence() {
    let lstm = LSTM::new(4, 8);
    let x = Tensor::ones(&[2, 5, 4]); // batch=2, seq=5, input=4

    let (output, h_final, c_final) = lstm.forward_sequence(&x, None, None);
    assert_eq!(output.shape(), &[2, 5, 8]);
    assert_eq!(h_final.shape(), &[2, 8]);
    assert_eq!(c_final.shape(), &[2, 8]);
}

#[test]
fn test_lstm_module_forward() {
    let lstm = LSTM::new(4, 8);
    let x = Tensor::ones(&[2, 5, 4]);

    let output = lstm.forward(&x);
    assert_eq!(output.shape(), &[2, 5, 8]);
}

#[test]
fn test_lstm_parameters() {
    let lstm = LSTM::new(4, 8);
    let params = lstm.parameters();
    // 8 linear layers * 2 (weight + bias) = 16
    assert_eq!(params.len(), 16);
}

#[test]
fn test_lstm_train_eval() {
    let mut lstm = LSTM::new(4, 8);
    assert!(lstm.training());
    lstm.eval();
    assert!(!lstm.training());
}

#[test]
fn test_lstm_with_initial_state() {
    let lstm = LSTM::new(4, 8);
    let x = Tensor::ones(&[2, 3, 4]);
    let h0 = Tensor::ones(&[2, 8]);
    let c0 = Tensor::ones(&[2, 8]);

    let (output, _, _) = lstm.forward_sequence(&x, Some(&h0), Some(&c0));
    assert_eq!(output.shape(), &[2, 3, 8]);
}

#[test]
fn test_lstm_cell_state_changes() {
    let lstm = LSTM::new(4, 8);
    let x = Tensor::ones(&[1, 4]);
    let h = Tensor::zeros(&[1, 8]);
    let c = Tensor::zeros(&[1, 8]);

    let (h_new, c_new) = lstm.forward_step(&x, &h, &c);

    // Cell state should have changed from initial zeros
    let c_sum: f32 = c_new.data().iter().sum();
    assert!(c_sum.abs() > 1e-6, "Cell state should change");

    // Hidden state bounded by tanh
    for &val in h_new.data() {
        assert!((-1.0..=1.0).contains(&val), "Hidden state bounded");
    }
}

// ==========================================================================
// Additional Coverage Tests
// ==========================================================================

// GRU Debug and parameters_mut tests
#[test]
fn test_gru_debug() {
    let gru = GRU::new(4, 8);
    let debug_str = format!("{:?}", gru);
    assert!(debug_str.contains("GRU"));
    assert!(debug_str.contains("input_size"));
    assert!(debug_str.contains("hidden_size"));
}

#[test]
fn test_gru_parameters_mut() {
    let mut gru = GRU::new(4, 8);
    let params = gru.parameters_mut();
    assert_eq!(params.len(), 12);
}

#[test]
fn test_gru_train() {
    let mut gru = GRU::new(4, 8);
    gru.eval();
    assert!(!gru.training());
    gru.train();
    assert!(gru.training());
}

// LSTM Debug and parameters_mut tests
#[test]
fn test_lstm_debug() {
    let lstm = LSTM::new(4, 8);
    let debug_str = format!("{:?}", lstm);
    assert!(debug_str.contains("LSTM"));
    assert!(debug_str.contains("input_size"));
    assert!(debug_str.contains("hidden_size"));
}

#[test]
fn test_lstm_parameters_mut() {
    let mut lstm = LSTM::new(4, 8);
    let params = lstm.parameters_mut();
    assert_eq!(params.len(), 16);
}

#[test]
fn test_lstm_train() {
    let mut lstm = LSTM::new(4, 8);
    lstm.eval();
    assert!(!lstm.training());
    lstm.train();
    assert!(lstm.training());
}

#[test]
fn test_lstm_with_h0_only() {
    let lstm = LSTM::new(4, 8);
    let x = Tensor::ones(&[2, 3, 4]);
    let h0 = Tensor::ones(&[2, 8]);

    let (output, _, _) = lstm.forward_sequence(&x, Some(&h0), None);
    assert_eq!(output.shape(), &[2, 3, 8]);
}

#[test]
fn test_lstm_with_c0_only() {
    let lstm = LSTM::new(4, 8);
    let x = Tensor::ones(&[2, 3, 4]);
    let c0 = Tensor::ones(&[2, 8]);

    let (output, _, _) = lstm.forward_sequence(&x, None, Some(&c0));
    assert_eq!(output.shape(), &[2, 3, 8]);
}

// Bidirectional Debug and parameters_mut tests
#[test]
fn test_bidirectional_debug() {
    let bi = Bidirectional::new(4, 8);
    let debug_str = format!("{:?}", bi);
    assert!(debug_str.contains("Bidirectional"));
    assert!(debug_str.contains("input_size"));
    assert!(debug_str.contains("hidden_size"));
}

#[test]
fn test_bidirectional_parameters_mut() {
    let mut bi = Bidirectional::new(4, 8);
    let params = bi.parameters_mut();
    assert_eq!(params.len(), 24);
}

#[test]
fn test_bidirectional_train() {
    let mut bi = Bidirectional::new(4, 8);
    bi.eval();
    assert!(!bi.training());
    bi.train();
    assert!(bi.training());
}

// Helper function tests
#[test]
fn test_add_tensors() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[4.0, 5.0, 6.0], &[3]);
    let result = add_tensors(&a, &b);
    assert_eq!(result.data(), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_mul_tensors() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[4.0, 5.0, 6.0], &[3]);
    let result = mul_tensors(&a, &b);
    assert_eq!(result.data(), &[4.0, 10.0, 18.0]);
}

#[test]
fn test_sub_from_one() {
    let x = Tensor::new(&[0.2, 0.5, 0.8], &[3]);
    let result = sub_from_one(&x);
    assert!((result.data()[0] - 0.8).abs() < 1e-5);
    assert!((result.data()[1] - 0.5).abs() < 1e-5);
    assert!((result.data()[2] - 0.2).abs() < 1e-5);
}

#[test]
fn test_slice_timestep() {
    // [batch=2, seq=3, input=2]
    let x = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 3, 2],
    );

    // Get timestep 0 (should be [1,2] and [7,8])
    let t0 = slice_timestep(&x, 0);
    assert_eq!(t0.shape(), &[2, 2]);
    assert_eq!(t0.data(), &[1.0, 2.0, 7.0, 8.0]);

    // Get timestep 1 (should be [3,4] and [9,10])
    let t1 = slice_timestep(&x, 1);
    assert_eq!(t1.data(), &[3.0, 4.0, 9.0, 10.0]);
}

#[test]
fn test_concat_last_dim() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let b = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[1, 2, 2]);

    let result = concat_last_dim(&a, &b, 1, 2, 2);
    assert_eq!(result.shape(), &[1, 2, 4]);
    // [1,2,5,6], [3,4,7,8]
    assert_eq!(result.data(), &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn test_reverse_sequence_batch() {
    // Test with batch_size > 1
    let x = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0: [1,2], [3,4], [5,6]
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 1: [7,8], [9,10], [11,12]
        ],
        &[2, 3, 2],
    );

    let rev = reverse_sequence(&x);
    // batch 0: [5,6], [3,4], [1,2]
    // batch 1: [11,12], [9,10], [7,8]
    assert_eq!(
        rev.data(),
        &[5.0, 6.0, 3.0, 4.0, 1.0, 2.0, 11.0, 12.0, 9.0, 10.0, 7.0, 8.0]
    );
}

#[test]
fn test_gru_long_sequence() {
    let gru = GRU::new(4, 8);
    let x = Tensor::ones(&[1, 20, 4]); // longer sequence

    let (output, h_final) = gru.forward_sequence(&x, None);
    assert_eq!(output.shape(), &[1, 20, 8]);
    assert_eq!(h_final.shape(), &[1, 8]);
}

#[test]
fn test_lstm_long_sequence() {
    let lstm = LSTM::new(4, 8);
    let x = Tensor::ones(&[1, 20, 4]); // longer sequence

    let (output, h_final, c_final) = lstm.forward_sequence(&x, None, None);
    assert_eq!(output.shape(), &[1, 20, 8]);
    assert_eq!(h_final.shape(), &[1, 8]);
    assert_eq!(c_final.shape(), &[1, 8]);
}

#[test]
fn test_bidirectional_long_sequence() {
    let bi = Bidirectional::new(4, 8);
    let x = Tensor::ones(&[1, 20, 4]);

    let output = bi.forward(&x);
    assert_eq!(output.shape(), &[1, 20, 16]);
}

#[test]
fn test_gru_single_timestep_sequence() {
    let gru = GRU::new(4, 8);
    let x = Tensor::ones(&[2, 1, 4]); // seq_len = 1

    let (output, h_final) = gru.forward_sequence(&x, None);
    assert_eq!(output.shape(), &[2, 1, 8]);
    assert_eq!(h_final.shape(), &[2, 8]);
}

#[test]
fn test_lstm_single_timestep_sequence() {
    let lstm = LSTM::new(4, 8);
    let x = Tensor::ones(&[2, 1, 4]); // seq_len = 1

    let (output, h_final, c_final) = lstm.forward_sequence(&x, None, None);
    assert_eq!(output.shape(), &[2, 1, 8]);
    assert_eq!(h_final.shape(), &[2, 8]);
    assert_eq!(c_final.shape(), &[2, 8]);
}

#[test]
fn test_gru_num_parameters() {
    let gru = GRU::new(10, 20);
    // 6 Linear layers:
    // w_ir: 10->20, w_hr: 20->20
    // w_iz: 10->20, w_hz: 20->20
    // w_in: 10->20, w_hn: 20->20
    // Each Linear has (in*out + out) params
    // 3*(10*20+20) + 3*(20*20+20) = 3*220 + 3*420 = 660 + 1260 = 1920
    assert_eq!(gru.num_parameters(), 1920);
}

#[test]
fn test_lstm_num_parameters() {
    let lstm = LSTM::new(10, 20);
    // 8 Linear layers:
    // 4 input layers: 10->20 each = 4*(10*20+20) = 4*220 = 880
    // 4 hidden layers: 20->20 each = 4*(20*20+20) = 4*420 = 1680
    // Total: 880 + 1680 = 2560
    assert_eq!(lstm.num_parameters(), 2560);
}
