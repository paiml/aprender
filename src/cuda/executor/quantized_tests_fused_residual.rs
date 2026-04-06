
#[test]
fn test_fused_residual_rmsnorm_with_harness() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Create gamma buffer directly
    let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; config.hidden_dim]).expect("gamma");
    let residual = GpuBuffer::from_host(&exec.context, &vec![1.0f32; config.hidden_dim]).expect("residual");
    let input = GpuBuffer::from_host(&exec.context, &vec![0.5f32; config.hidden_dim]).expect("input");
    let result =
        exec.fused_residual_rmsnorm_gpu(&residual, &input, &gamma, config.hidden_dim as u32, 1e-5);
    assert!(result.is_ok());
}

#[test]
fn test_batched_rope_with_harness() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let m = 4u32;
    let total_dim = (m as usize) * config.num_heads * config.head_dim;
    let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; total_dim]).expect("input");
    let output = GpuBuffer::<f32>::new(&exec.context, total_dim).expect("output");
    let positions = GpuBuffer::from_host(&exec.context, &vec![0u32, 1, 2, 3]).expect("positions");

    let result = exec.batched_rope_into(
        &input,
        &output,
        &positions,
        config.num_heads as u32,
        config.head_dim as u32,
        m,
        exec.rope_theta,
    );
    assert!(result.is_ok());
}

#[test]
fn test_batched_swiglu_with_harness() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let m = 4u32;
    let size = (m as usize) * config.intermediate_dim;
    let gate = GpuBuffer::from_host(&exec.context, &vec![1.0f32; size]).expect("gate");
    let up = GpuBuffer::from_host(&exec.context, &vec![2.0f32; size]).expect("up");
    let output = GpuBuffer::<f32>::new(&exec.context, size).expect("output");

    let result = exec.batched_swiglu_into(&gate, &up, &output, config.intermediate_dim as u32, m);
    assert!(result.is_ok());
}

#[test]
fn test_batched_residual_with_harness() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let m = 4u32;
    let size = (m as usize) * config.hidden_dim;
    let input1 = GpuBuffer::from_host(&exec.context, &vec![1.0f32; size]).expect("input1");
    let input2 = GpuBuffer::from_host(&exec.context, &vec![0.5f32; size]).expect("input2");
    let output = GpuBuffer::<f32>::new(&exec.context, size).expect("output");

    let result =
        exec.batched_residual_add_into(&input1, &input2, &output, config.hidden_dim as u32, m);
    assert!(result.is_ok());
}
