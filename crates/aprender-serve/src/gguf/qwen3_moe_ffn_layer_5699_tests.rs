// #5699 — `moe_ffn_forward_layer` coverage: input guards and the routed
// weighted-sum path (one expert selected, so output == that expert's SwiGLU).
//
// Named `moe_ffn_layer_5699_*` so `cargo test -p aprender-serve --lib
// moe_ffn_layer_5699` selects exactly this file.

#[cfg(test)]
mod moe_ffn_layer_5699_tests {
    use super::*;
    use crate::gguf::types::GGUF_TYPE_Q4_K;

    const H: usize = 256;
    const I: usize = 256;
    const E: usize = 2;
    const SB: usize = 144;

    /// Q4_K bytes for `n_blocks` super-blocks: d = 1.0 (f16), dmin = 0, the
    /// rest a pattern that depends on `seed`, so different experts differ.
    fn q4k_blocks(n_blocks: usize, seed: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n_blocks * SB);
        for b in 0..n_blocks {
            out.extend_from_slice(&0x3C00u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            for i in 4..SB {
                out.push(((b * 31 + i * 7 + seed * 13 + 3) % 251) as u8);
            }
        }
        out
    }

    struct Fixture {
        layer: Qwen3MoeQuantizedLayer,
        data: Vec<u8>,
    }

    /// Layout: [router F32 E×H][gate E×I rows][up E×I rows][down E×H rows].
    /// `favoured` is the expert the router scores highest for an all-ones input.
    fn fixture(favoured: usize) -> Fixture {
        let mut data = Vec::new();
        let router_off = data.len();
        for e in 0..E {
            let w: f32 = if e == favoured { 1.0 } else { -1.0 };
            for _ in 0..H {
                data.extend_from_slice(&w.to_le_bytes());
            }
        }
        let router_len = data.len() - router_off;
        let mut tensor = |rows_per_expert: usize, seed: usize| {
            let off = data.len();
            for e in 0..E {
                data.extend(q4k_blocks(rows_per_expert, seed + e));
            }
            QuantizedTensorRef {
                offset: off,
                byte_size: E * rows_per_expert * SB,
                num_elements: E * rows_per_expert * 256,
                qtype: GGUF_TYPE_Q4_K,
            }
        };
        let gate_exps = tensor(I, 1);
        let up_exps = tensor(I, 5);
        let down_exps = tensor(H, 9);
        let router = QuantizedTensorRef {
            offset: router_off,
            byte_size: router_len,
            num_elements: E * H,
            qtype: crate::gguf::types::GGUF_TYPE_F32,
        };
        Fixture {
            layer: Qwen3MoeQuantizedLayer {
                router,
                gate_exps,
                up_exps,
                down_exps,
            },
            data,
        }
    }

    #[test]
    fn moe_ffn_layer_5699_rejects_hidden_dim_mismatch() {
        let f = fixture(0);
        let err =
            moe_ffn_forward_layer(&vec![1.0; H - 1], &f.layer, E, 1, I, H, &f.data).unwrap_err();
        assert!(
            format!("{err}").contains(&format!("hidden.len() = {} but hidden_dim = {H}", H - 1)),
            "{err}"
        );
    }

    #[test]
    fn moe_ffn_layer_5699_rejects_quantized_router() {
        let mut f = fixture(0);
        f.layer.router.qtype = GGUF_TYPE_Q4_K;
        let err = moe_ffn_forward_layer(&vec![1.0; H], &f.layer, E, 1, I, H, &f.data).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("moe_router_quantized_read") && msg.contains("not F32"),
            "{msg}"
        );
    }

    #[test]
    fn moe_ffn_layer_5699_rejects_router_byte_size_mismatch() {
        let mut f = fixture(0);
        f.layer.router.byte_size -= 4;
        let err = moe_ffn_forward_layer(&vec![1.0; H], &f.layer, E, 1, I, H, &f.data).unwrap_err();
        assert!(format!("{err}").contains("router byte_size"), "{err}");
    }

    #[test]
    fn moe_ffn_layer_5699_k1_output_is_the_routed_experts_swiglu() {
        let hidden = vec![1.0f32; H];
        for favoured in 0..E {
            let f = fixture(favoured);
            let out = moe_ffn_forward_layer(&hidden, &f.layer, E, 1, I, H, &f.data).unwrap();
            let want =
                expert_swiglu_quantized(&hidden, &f.layer, favoured, E, I, H, &f.data).unwrap();
            assert_eq!(out.len(), H);
            assert!(out.iter().all(|v| v.is_finite()));
            assert_eq!(
                out, want,
                "k=1 weight is 1.0, so output must equal expert {favoured}"
            );
            let other =
                expert_swiglu_quantized(&hidden, &f.layer, 1 - favoured, E, I, H, &f.data).unwrap();
            assert_ne!(
                out, other,
                "the fixture's experts must differ or routing is untested"
            );
        }
    }
}
