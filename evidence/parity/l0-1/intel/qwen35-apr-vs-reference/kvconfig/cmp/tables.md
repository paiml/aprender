## logits (apr vs llama per-token reference)

| prompt | apr | ref | min cos | mean cos | median cos | n<0.98 | argmax mismatches | Frobenius rel L2 | gap removed vs OFF-vs-A |
|---|---|---|---|---|---|---|---|---|---|
| orig | OFF | A | 0.994599 | 0.998995 | 0.999282 | 0 | 4 | 0.044677 | 0.0% |
| orig | OFF | B | 0.994599 | 0.998995 | 0.999282 | 0 | 4 | 0.044677 | 0.0% |
| orig | OFF | C | 0.996167 | 0.999032 | 0.999256 | 0 | 4 | 0.043681 | 2.2% |
| orig | ON | A | 0.998157 | 0.999266 | 0.999319 | 0 | 3 | 0.037667 | 15.7% |
| orig | ON | B | 0.998157 | 0.999266 | 0.999319 | 0 | 3 | 0.037667 | 15.7% |
| orig | ON | C | 0.998536 | 0.99927 | 0.999319 | 0 | 3 | 0.037928 | 15.1% |
| p1 | OFF | A | 0.997448 | 0.999509 | 0.99958 | 0 | 4 | 0.031619 | 0.0% |
| p1 | OFF | B | 0.997448 | 0.999509 | 0.99958 | 0 | 4 | 0.031619 | 0.0% |
| p1 | OFF | C | 0.997383 | 0.999512 | 0.999611 | 0 | 6 | 0.030891 | 2.3% |
| p1 | ON | A | 0.998868 | 0.999528 | 0.999565 | 0 | 8 | 0.031376 | 0.8% |
| p1 | ON | B | 0.998868 | 0.999528 | 0.999565 | 0 | 8 | 0.031376 | 0.8% |
| p1 | ON | C | 0.998796 | 0.999544 | 0.999575 | 0 | 6 | 0.030473 | 3.6% |
| p2 | OFF | A | 0.99537 | 0.998371 | 0.998489 | 0 | 2 | 0.056340 | 0.0% |
| p2 | OFF | B | 0.99537 | 0.998371 | 0.998489 | 0 | 2 | 0.056340 | 0.0% |
| p2 | OFF | C | 0.995651 | 0.998352 | 0.998444 | 0 | 3 | 0.056881 | -1.0% |
| p2 | ON | A | 0.993047 | 0.998519 | 0.998728 | 0 | 2 | 0.053926 | 4.3% |
| p2 | ON | B | 0.993047 | 0.998519 | 0.998728 | 0 | 2 | 0.053926 | 4.3% |
| p2 | ON | C | 0.995296 | 0.998572 | 0.99867 | 0 | 3 | 0.052848 | 6.2% |
| p3 | OFF | A | 0.995687 | 0.999133 | 0.999353 | 0 | 5 | 0.040229 | 0.0% |
| p3 | OFF | B | 0.995687 | 0.999133 | 0.999353 | 0 | 5 | 0.040229 | 0.0% |
| p3 | OFF | C | 0.995828 | 0.999105 | 0.999342 | 0 | 4 | 0.040604 | -0.9% |
| p3 | ON | A | 0.997929 | 0.999302 | 0.999432 | 0 | 5 | 0.036117 | 10.2% |
| p3 | ON | B | 0.997929 | 0.999302 | 0.999432 | 0 | 5 | 0.036117 | 10.2% |
| p3 | ON | C | 0.994753 | 0.999252 | 0.999441 | 0 | 4 | 0.036772 | 8.6% |
| p4 | OFF | A | 0.961309 | 0.994771 | 0.998415 | 4 | 6 | 0.092010 | 0.0% |
| p4 | OFF | B | 0.961309 | 0.994771 | 0.998415 | 4 | 6 | 0.092010 | 0.0% |
| p4 | OFF | C | 0.957286 | 0.995217 | 0.998602 | 2 | 8 | 0.087292 | 5.1% |
| p4 | ON | A | 0.984332 | 0.997562 | 0.998841 | 0 | 3 | 0.062323 | 32.3% |
| p4 | ON | B | 0.984332 | 0.997562 | 0.998841 | 0 | 3 | 0.062323 | 32.3% |
| p4 | ON | C | 0.984187 | 0.997733 | 0.99885 | 0 | 4 | 0.060027 | 34.8% |

## sub-layer, p4 pos 1: apr ON vs config C (residual stream steps)

| point | layer | type | rel before | rel after | step | sub-layer out | sub rel |
|---|---|---|---|---|---|---|---|
| attn_residual-0 | 0 | linear_gated_deltanet | 0.000000 | 0.000002 | 0.000002 | linear_attn_out-0 | 0.000002 |
| l_out-0 | 0 | linear_gated_deltanet | 0.000002 | 0.000001 | -0.000001 | ffn_out-0 | 0.000001 |
| attn_residual-1 | 1 | linear_gated_deltanet | 0.000001 | 0.000001 | 0.000000 | linear_attn_out-1 | 0.000000 |
| l_out-1 | 1 | linear_gated_deltanet | 0.000001 | 0.000001 | 0.000000 | ffn_out-1 | 0.000001 |
| attn_residual-2 | 2 | linear_gated_deltanet | 0.000001 | 0.000001 | 0.000000 | linear_attn_out-2 | 0.000001 |
| l_out-2 | 2 | linear_gated_deltanet | 0.000001 | 0.000001 | 0.000000 | ffn_out-2 | 0.000001 |
| attn_residual-3 | 3 | full_attention | 0.000001 | 0.000001 | 0.000000 | attn_output-3 | 0.000001 |
| l_out-3 | 3 | full_attention | 0.000001 | 0.000001 | 0.000000 | ffn_out-3 | 0.000000 |
| attn_residual-4 | 4 | linear_gated_deltanet | 0.000001 | 0.000001 | 0.000000 | linear_attn_out-4 | 0.000001 |
| l_out-4 | 4 | linear_gated_deltanet | 0.000001 | 0.000001 | 0.000000 | ffn_out-4 | 0.000001 |
| attn_residual-5 | 5 | linear_gated_deltanet | 0.000001 | 0.000001 | 0.000000 | linear_attn_out-5 | 0.000001 |
| l_out-5 | 5 | linear_gated_deltanet | 0.000001 | 0.000001 | 0.000000 | ffn_out-5 | 0.000001 |
| attn_residual-6 | 6 | linear_gated_deltanet | 0.000001 | 0.000001 | 0.000000 | linear_attn_out-6 | 0.000001 |
| l_out-6 | 6 | linear_gated_deltanet | 0.000001 | 0.000001 | 0.000000 | ffn_out-6 | 0.000000 |
| attn_residual-7 | 7 | full_attention | 0.000001 | 0.000001 | 0.000000 | attn_output-7 | 0.000001 |
| l_out-7 | 7 | full_attention | 0.000001 | 0.000001 | 0.000000 | ffn_out-7 | 0.000000 |
| attn_residual-8 | 8 | linear_gated_deltanet | 0.000001 | 0.000002 | 0.000001 | linear_attn_out-8 | 0.000001 |
| l_out-8 | 8 | linear_gated_deltanet | 0.000002 | 0.000002 | 0.000000 | ffn_out-8 | 0.000001 |
| attn_residual-9 | 9 | linear_gated_deltanet | 0.000002 | 0.000002 | 0.000000 | linear_attn_out-9 | 0.000001 |
| l_out-9 | 9 | linear_gated_deltanet | 0.000002 | 0.000002 | 0.000000 | ffn_out-9 | 0.000000 |
| attn_residual-10 | 10 | linear_gated_deltanet | 0.000002 | 0.000002 | 0.000000 | linear_attn_out-10 | 0.000001 |
| l_out-10 | 10 | linear_gated_deltanet | 0.000002 | 0.000001 | -0.000001 | ffn_out-10 | 0.000000 |
| attn_residual-11 | 11 | full_attention | 0.000001 | 0.000002 | 0.000001 | attn_output-11 | 0.000001 |
| l_out-11 | 11 | full_attention | 0.000002 | 0.000001 | -0.000001 | ffn_out-11 | 0.000000 |
| attn_residual-12 | 12 | linear_gated_deltanet | 0.000001 | 0.000006 | 0.000005 | linear_attn_out-12 | 0.000032 |
| l_out-12 | 12 | linear_gated_deltanet | 0.000006 | 0.000553 | 0.000547 | ffn_out-12 | 0.001413 |
| attn_residual-13 | 13 | linear_gated_deltanet | 0.000553 | 0.001732 | 0.001179 | linear_attn_out-13 | 0.015309 |
| l_out-13 | 13 | linear_gated_deltanet | 0.001732 | 0.008744 | 0.007012 | ffn_out-13 | 0.018542 |
| attn_residual-14 | 14 | linear_gated_deltanet | 0.008744 | 0.011367 | 0.002623 | linear_attn_out-14 | 0.033760 |
| l_out-14 | 14 | linear_gated_deltanet | 0.011367 | 0.015952 | 0.004585 | ffn_out-14 | 0.027603 |
| attn_residual-15 | 15 | full_attention | 0.015952 | 0.022204 | 0.006252 | attn_output-15 | 0.052921 |
| l_out-15 | 15 | full_attention | 0.022204 | 0.028698 | 0.006494 | ffn_out-15 | 0.038125 |
| attn_residual-16 | 16 | linear_gated_deltanet | 0.028698 | 0.031721 | 0.003023 | linear_attn_out-16 | 0.042357 |
| l_out-16 | 16 | linear_gated_deltanet | 0.031721 | 0.035815 | 0.004094 | ffn_out-16 | 0.047589 |
| attn_residual-17 | 17 | linear_gated_deltanet | 0.035815 | 0.037459 | 0.001644 | linear_attn_out-17 | 0.060042 |
| l_out-17 | 17 | linear_gated_deltanet | 0.037459 | 0.041553 | 0.004094 | ffn_out-17 | 0.053313 |
| attn_residual-18 | 18 | linear_gated_deltanet | 0.041553 | 0.044190 | 0.002637 | linear_attn_out-18 | 0.066913 |
| l_out-18 | 18 | linear_gated_deltanet | 0.044190 | 0.049276 | 0.005086 | ffn_out-18 | 0.067262 |
| attn_residual-19 | 19 | full_attention | 0.049276 | 0.054288 | 0.005012 | attn_output-19 | 0.076934 |
| l_out-19 | 19 | full_attention | 0.054288 | 0.061095 | 0.006807 | ffn_out-19 | 0.084398 |
| attn_residual-20 | 20 | linear_gated_deltanet | 0.061095 | 0.064029 | 0.002934 | linear_attn_out-20 | 0.081146 |
| l_out-20 | 20 | linear_gated_deltanet | 0.064029 | 0.072864 | 0.008835 | ffn_out-20 | 0.094633 |
| attn_residual-21 | 21 | linear_gated_deltanet | 0.072864 | 0.078382 | 0.005518 | linear_attn_out-21 | 0.116437 |
| l_out-21 | 21 | linear_gated_deltanet | 0.078382 | 0.088171 | 0.009789 | ffn_out-21 | 0.103655 |
| attn_residual-22 | 22 | linear_gated_deltanet | 0.088171 | 0.088234 | 0.000063 | linear_attn_out-22 | 0.110306 |
| l_out-22 | 22 | linear_gated_deltanet | 0.088234 | 0.088952 | 0.000718 | ffn_out-22 | 0.106485 |
| attn_residual-23 | 23 | full_attention | 0.088952 | 0.094506 | 0.005554 | attn_output-23 | 0.069169 |
| l_out-23 | 23 | full_attention | 0.094506 | 0.099840 | 0.005334 | ffn_out-23 | 0.090657 |
| result_norm | -1 | - | 0.099840 | 0.101369 | 0.001529 | - | - |

```
{
 "largest_step_on_vs_C_p4_pos1": "l_out-21 (layer 21 linear_gated_deltanet, ffn) 0.078382 -> 0.088171 step 0.009789",
 "largest_step_off_vs_C_p4_pos1": "attn_residual-0 (layer 0 linear_gated_deltanet, mixer) step 0.039963",
 "first_nonzero_on_vs_C_p4_pos1": "linear_attn_out-0 (layer 0 linear_gated_deltanet) rel_l2 0.000002",
 "first_nonzero_on_vs_A_p4_pos1": "linear_attn_out-0 (layer 0 linear_gated_deltanet) rel_l2 0.000002",
 "first_nonzero_on_vs_C_p4_pos0": "linear_attn_out-0 (layer 0 linear_gated_deltanet) rel_l2 0.000001",
 "first_nonzero_on_vs_A_p4_pos0": "linear_attn_out-0 (layer 0 linear_gated_deltanet) rel_l2 0.000001",
 "final_rel_on_vs_C_p4_pos1": "0.101369"
}
```
# largest step p4 pos 0: attn_residual-15 (layer 15 full_attention, mixer) rel_l2 0.001664 -> 0.030645 step 0.028981
# largest step p4 pos 1: l_out-21 (layer 21 linear_gated_deltanet, ffn) rel_l2 0.078382 -> 0.088171 step 0.009789
# largest step p4 pos 2: attn_residual-23 (layer 23 full_attention, mixer) rel_l2 0.065460 -> 0.074741 step 0.009281
# largest step p4 pos 3: l_out-23 (layer 23 full_attention, ffn) rel_l2 0.072849 -> 0.096479 step 0.023630
# largest step orig pos 4: l_out-2 (layer 2 linear_gated_deltanet, ffn) rel_l2 0.002427 -> 0.014620 step 0.012193
# largest step orig pos 28: l_out-2 (layer 2 linear_gated_deltanet, ffn) rel_l2 0.014734 -> 0.027309 step 0.012575
