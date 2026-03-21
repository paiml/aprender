//! Kernel operation dispatch tables for each kernel class.

use super::{Constraints, KernelClass, KernelOp};

/// Get kernel ops for a class, optionally enriched with constraint-specific ops.
pub(crate) fn kernel_ops_for_family(
    class: KernelClass,
    constraints: &Constraints,
) -> Vec<KernelOp> {
    let mut ops = kernel_ops_for_class(class);
    // Add RoPE op for families that use RoPE but whose class doesn't include it
    // (e.g., Phi: Class B with RoPE positional encoding)
    let has_rope = ops.iter().any(|o| o.kernel == "rope_forward");
    if !has_rope && constraints.positional_encoding == "rope" {
        ops.push(KernelOp {
            op: "Position Encoding",
            kernel: "rope_forward",
            contract: "rope-kernel-v1",
        });
    }
    ops
}

pub(crate) fn kernel_ops_for_class(class: KernelClass) -> Vec<KernelOp> {
    // Base ops: MatVec always present. Softmax only for attention-based models (not SSM).
    let mut ops = vec![
        KernelOp {
            op: "MatVec (Q4K)",
            kernel: "fused_q4k_parallel_matvec",
            contract: "matvec-kernel-v1",
        },
        KernelOp {
            op: "MatVec (Q6K)",
            kernel: "fused_q6k_parallel_matvec",
            contract: "matvec-kernel-v1",
        },
    ];

    // Softmax is attention-specific — SSM and linear attention models don't use it
    if class != KernelClass::Ssm && class != KernelClass::Linear {
        ops.push(KernelOp {
            op: "Softmax",
            kernel: "softmax",
            contract: "softmax-kernel-v1",
        });
    }

    ops.push(KernelOp {
        op: "Kernel Fusion",
        kernel: "fused_matvec_activation",
        contract: "kernel-fusion-v1",
    });

    match class {
        KernelClass::A => {
            ops.push(KernelOp {
                op: "Attention (GQA)",
                kernel: "gqa_forward",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Normalization",
                kernel: "rms_norm",
                contract: "normalization-kernel-v1",
            });
            ops.push(KernelOp {
                op: "Activation",
                kernel: "silu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "MLP",
                kernel: "swiglu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Position Encoding",
                kernel: "rope_forward",
                contract: "rope-kernel-v1",
            });
        }
        KernelClass::B => {
            ops.push(KernelOp {
                op: "Attention (MHA)",
                kernel: "mha_forward",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Normalization",
                kernel: "layer_norm",
                contract: "normalization-kernel-v1",
            });
            ops.push(KernelOp {
                op: "Activation",
                kernel: "gelu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "MLP",
                kernel: "gelu_mlp",
                contract: "element-wise-ops-v1",
            });
        }
        KernelClass::C => {
            ops.push(KernelOp {
                op: "Attention (MQA)",
                kernel: "mqa_forward",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Normalization",
                kernel: "layer_norm",
                contract: "normalization-kernel-v1",
            });
            ops.push(KernelOp {
                op: "Activation",
                kernel: "gelu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Position Encoding",
                kernel: "alibi",
                contract: "element-wise-ops-v1",
            });
        }
        KernelClass::D => {
            ops.push(KernelOp {
                op: "Attention (GQA/MHA)",
                kernel: "gqa_forward",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Normalization",
                kernel: "layer_norm",
                contract: "normalization-kernel-v1",
            });
            ops.push(KernelOp {
                op: "Activation",
                kernel: "silu/gelu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "MLP",
                kernel: "gated_mlp",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Position Encoding",
                kernel: "rope_forward",
                contract: "rope-kernel-v1",
            });
        }
        KernelClass::E => {
            ops.push(KernelOp {
                op: "Attention (GQA)",
                kernel: "gqa_forward",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Normalization",
                kernel: "rms_norm",
                contract: "normalization-kernel-v1",
            });
            ops.push(KernelOp {
                op: "MoE Router",
                kernel: "moe_routing",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Activation",
                kernel: "silu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "MLP",
                kernel: "swiglu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Position Encoding",
                kernel: "rope_forward",
                contract: "rope-kernel-v1",
            });
        }
        KernelClass::F => {
            ops.push(KernelOp {
                op: "Attention (GQA)",
                kernel: "gqa_forward",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Normalization",
                kernel: "rms_norm",
                contract: "normalization-kernel-v1",
            });
            ops.push(KernelOp {
                op: "Activation",
                kernel: "gelu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "MLP",
                kernel: "gated_mlp",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Position Encoding",
                kernel: "rope_forward",
                contract: "rope-kernel-v1",
            });
        }
        KernelClass::Ssm => {
            ops.push(KernelOp {
                op: "SSM Scan",
                kernel: "selective_scan",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Normalization",
                kernel: "rms_norm",
                contract: "normalization-kernel-v1",
            });
            ops.push(KernelOp {
                op: "Activation",
                kernel: "silu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "MLP",
                kernel: "gated_mlp",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Conv1d",
                kernel: "depthwise_conv1d",
                contract: "element-wise-ops-v1",
            });
        }
        KernelClass::Linear => {
            ops.push(KernelOp {
                op: "WKV Recurrence",
                kernel: "wkv_forward",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Token Shift",
                kernel: "token_shift",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Normalization",
                kernel: "layer_norm",
                contract: "normalization-kernel-v1",
            });
            ops.push(KernelOp {
                op: "Activation",
                kernel: "gelu",
                contract: "element-wise-ops-v1",
            });
            ops.push(KernelOp {
                op: "Channel Mixing",
                kernel: "channel_mix",
                contract: "element-wise-ops-v1",
            });
        }
        KernelClass::Unknown => {}
    }

    ops
}
