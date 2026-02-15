```mermaid
graph TD
  root[Model]
  root --> root_0{{ decoder }}
  root_0 --> root_0_0{{ layers }}
  root_0_0 --> root_0_0_0{{ 0 }}
  root_0_0_0 --> root_0_0_0_0{{ cross_attn }}
  root_0_0_0_0 --> root_0_0_0_0_0{{ k_proj }}
  root_0_0_0_0_0 --> root_0_0_0_0_0_0[weight<br/>[384, 384]]
  root_0_0_0_0 --> root_0_0_0_0_1{{ out_proj }}
  root_0_0_0_0_1 --> root_0_0_0_0_1_0[weight<br/>[384, 384]]
  root_0_0_0_0 --> root_0_0_0_0_2{{ q_proj }}
  root_0_0_0_0_2 --> root_0_0_0_0_2_0[weight<br/>[384, 384]]
  root_0_0_0_0 --> root_0_0_0_0_3{{ v_proj }}
  root_0_0_0_0_3 --> root_0_0_0_0_3_0[weight<br/>[384, 384]]
  root_0_0_0 --> root_0_0_0_1{{ self_attn }}
  root_0_0_0_1 --> root_0_0_0_1_0{{ k_proj }}
  root_0_0_0_1_0 --> root_0_0_0_1_0_0[weight<br/>[384, 384]]
  root_0_0_0_1 --> root_0_0_0_1_1{{ out_proj }}
  root_0_0_0_1_1 --> root_0_0_0_1_1_0[weight<br/>[384, 384]]
  root_0_0_0_1 --> root_0_0_0_1_2{{ q_proj }}
  root_0_0_0_1_2 --> root_0_0_0_1_2_0[weight<br/>[384, 384]]
  root_0_0_0_1 --> root_0_0_0_1_3{{ v_proj }}
  root_0_0_0_1_3 --> root_0_0_0_1_3_0[weight<br/>[384, 384]]
  root_0_0 --> root_0_0_1{{ 1 }}
  root_0_0_1 --> root_0_0_1_0{{ cross_attn }}
  root_0_0_1_0 --> root_0_0_1_0_0{{ k_proj }}
  root_0_0_1_0_0 --> root_0_0_1_0_0_0[weight<br/>[384, 384]]
  root_0_0_1_0 --> root_0_0_1_0_1{{ out_proj }}
  root_0_0_1_0_1 --> root_0_0_1_0_1_0[weight<br/>[384, 384]]
  root_0_0_1_0 --> root_0_0_1_0_2{{ q_proj }}
  root_0_0_1_0_2 --> root_0_0_1_0_2_0[weight<br/>[384, 384]]
  root_0_0_1_0 --> root_0_0_1_0_3{{ v_proj }}
  root_0_0_1_0_3 --> root_0_0_1_0_3_0[weight<br/>[384, 384]]
  root_0_0_1 --> root_0_0_1_1{{ self_attn }}
  root_0_0_1_1 --> root_0_0_1_1_0{{ k_proj }}
  root_0_0_1_1_0 --> root_0_0_1_1_0_0[weight<br/>[384, 384]]
  root_0_0_1_1 --> root_0_0_1_1_1{{ out_proj }}
  root_0_0_1_1_1 --> root_0_0_1_1_1_0[weight<br/>[384, 384]]
  root_0_0_1_1 --> root_0_0_1_1_2{{ q_proj }}
  root_0_0_1_1_2 --> root_0_0_1_1_2_0[weight<br/>[384, 384]]
  root_0_0_1_1 --> root_0_0_1_1_3{{ v_proj }}
  root_0_0_1_1_3 --> root_0_0_1_1_3_0[weight<br/>[384, 384]]
  root_0_0 --> root_0_0_2{{ 2 }}
  root_0_0_2 --> root_0_0_2_0{{ cross_attn }}
  root_0_0_2_0 --> root_0_0_2_0_0{{ k_proj }}
  root_0_0_2_0_0 --> root_0_0_2_0_0_0[weight<br/>[384, 384]]
  root_0_0_2_0 --> root_0_0_2_0_1{{ out_proj }}
  root_0_0_2_0_1 --> root_0_0_2_0_1_0[weight<br/>[384, 384]]
  root_0_0_2_0 --> root_0_0_2_0_2{{ q_proj }}
  root_0_0_2_0_2 --> root_0_0_2_0_2_0[weight<br/>[384, 384]]
  root_0_0_2_0 --> root_0_0_2_0_3{{ v_proj }}
  root_0_0_2_0_3 --> root_0_0_2_0_3_0[weight<br/>[384, 384]]
  root_0_0_2 --> root_0_0_2_1{{ self_attn }}
  root_0_0_2_1 --> root_0_0_2_1_0{{ k_proj }}
  root_0_0_2_1_0 --> root_0_0_2_1_0_0[weight<br/>[384, 384]]
  root_0_0_2_1 --> root_0_0_2_1_1{{ out_proj }}
  root_0_0_2_1_1 --> root_0_0_2_1_1_0[weight<br/>[384, 384]]
  root_0_0_2_1 --> root_0_0_2_1_2{{ q_proj }}
  root_0_0_2_1_2 --> root_0_0_2_1_2_0[weight<br/>[384, 384]]
  root_0_0_2_1 --> root_0_0_2_1_3{{ v_proj }}
  root_0_0_2_1_3 --> root_0_0_2_1_3_0[weight<br/>[384, 384]]
  root_0_0 --> root_0_0_3{{ 3 }}
  root_0_0_3 --> root_0_0_3_0{{ cross_attn }}
  root_0_0_3_0 --> root_0_0_3_0_0{{ k_proj }}
  root_0_0_3_0_0 --> root_0_0_3_0_0_0[weight<br/>[384, 384]]
  root_0_0_3_0 --> root_0_0_3_0_1{{ out_proj }}
  root_0_0_3_0_1 --> root_0_0_3_0_1_0[weight<br/>[384, 384]]
  root_0_0_3_0 --> root_0_0_3_0_2{{ q_proj }}
  root_0_0_3_0_2 --> root_0_0_3_0_2_0[weight<br/>[384, 384]]
  root_0_0_3_0 --> root_0_0_3_0_3{{ v_proj }}
  root_0_0_3_0_3 --> root_0_0_3_0_3_0[weight<br/>[384, 384]]
  root_0_0_3 --> root_0_0_3_1{{ self_attn }}
  root_0_0_3_1 --> root_0_0_3_1_0{{ k_proj }}
  root_0_0_3_1_0 --> root_0_0_3_1_0_0[weight<br/>[384, 384]]
  root_0_0_3_1 --> root_0_0_3_1_1{{ out_proj }}
  root_0_0_3_1_1 --> root_0_0_3_1_1_0[weight<br/>[384, 384]]
  root_0_0_3_1 --> root_0_0_3_1_2{{ q_proj }}
  root_0_0_3_1_2 --> root_0_0_3_1_2_0[weight<br/>[384, 384]]
  root_0_0_3_1 --> root_0_0_3_1_3{{ v_proj }}
  root_0_0_3_1_3 --> root_0_0_3_1_3_0[weight<br/>[384, 384]]
  root --> root_1{{ encoder }}
  root_1 --> root_1_0{{ layers }}
  root_1_0 --> root_1_0_0{{ 0 }}
  root_1_0_0 --> root_1_0_0_0{{ self_attn }}
  root_1_0_0_0 --> root_1_0_0_0_0{{ k_proj }}
  root_1_0_0_0_0 --> root_1_0_0_0_0_0[weight<br/>[384, 384]]
  root_1_0_0_0 --> root_1_0_0_0_1{{ out_proj }}
  root_1_0_0_0_1 --> root_1_0_0_0_1_0[weight<br/>[384, 384]]
  root_1_0_0_0 --> root_1_0_0_0_2{{ q_proj }}
  root_1_0_0_0_2 --> root_1_0_0_0_2_0[weight<br/>[384, 384]]
  root_1_0_0_0 --> root_1_0_0_0_3{{ v_proj }}
  root_1_0_0_0_3 --> root_1_0_0_0_3_0[weight<br/>[384, 384]]
  root_1_0 --> root_1_0_1{{ 1 }}
  root_1_0_1 --> root_1_0_1_0{{ self_attn }}
  root_1_0_1_0 --> root_1_0_1_0_0{{ k_proj }}
  root_1_0_1_0_0 --> root_1_0_1_0_0_0[weight<br/>[384, 384]]
  root_1_0_1_0 --> root_1_0_1_0_1{{ out_proj }}
  root_1_0_1_0_1 --> root_1_0_1_0_1_0[weight<br/>[384, 384]]
  root_1_0_1_0 --> root_1_0_1_0_2{{ q_proj }}
  root_1_0_1_0_2 --> root_1_0_1_0_2_0[weight<br/>[384, 384]]
  root_1_0_1_0 --> root_1_0_1_0_3{{ v_proj }}
  root_1_0_1_0_3 --> root_1_0_1_0_3_0[weight<br/>[384, 384]]
  root_1_0 --> root_1_0_2{{ 2 }}
  root_1_0_2 --> root_1_0_2_0{{ self_attn }}
  root_1_0_2_0 --> root_1_0_2_0_0{{ k_proj }}
  root_1_0_2_0_0 --> root_1_0_2_0_0_0[weight<br/>[384, 384]]
  root_1_0_2_0 --> root_1_0_2_0_1{{ out_proj }}
  root_1_0_2_0_1 --> root_1_0_2_0_1_0[weight<br/>[384, 384]]
  root_1_0_2_0 --> root_1_0_2_0_2{{ q_proj }}
  root_1_0_2_0_2 --> root_1_0_2_0_2_0[weight<br/>[384, 384]]
  root_1_0_2_0 --> root_1_0_2_0_3{{ v_proj }}
  root_1_0_2_0_3 --> root_1_0_2_0_3_0[weight<br/>[384, 384]]
  root_1_0 --> root_1_0_3{{ 3 }}
  root_1_0_3 --> root_1_0_3_0{{ self_attn }}
  root_1_0_3_0 --> root_1_0_3_0_0{{ k_proj }}
  root_1_0_3_0_0 --> root_1_0_3_0_0_0[weight<br/>[384, 384]]
  root_1_0_3_0 --> root_1_0_3_0_1{{ out_proj }}
  root_1_0_3_0_1 --> root_1_0_3_0_1_0[weight<br/>[384, 384]]
  root_1_0_3_0 --> root_1_0_3_0_2{{ q_proj }}
  root_1_0_3_0_2 --> root_1_0_3_0_2_0[weight<br/>[384, 384]]
  root_1_0_3_0 --> root_1_0_3_0_3{{ v_proj }}
  root_1_0_3_0_3 --> root_1_0_3_0_3_0[weight<br/>[384, 384]]
```
