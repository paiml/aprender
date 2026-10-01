load: read 0.07 s, total 0.16 s (F16 -> f32), threads 6, local window |i-j| <= 64

| ladder block (158 tokens) | max abs diff | ref rms |
|---|---|---|
| emb | 9.537e-7 | 0.548 |
| layer0 | 1.526e-5 | 0.990 |
| layer1 | 2.289e-5 | 1.210 |
| layer2 | 3.815e-5 | 1.347 |
| layer3 | 3.815e-5 | 1.449 |
| layer5 | 2.747e-4 | 2.004 |
| layer11 | 7.324e-4 | 11.693 |
| layer17 | 1.123e-2 | 26.707 |
| layer23 | 5.664e-2 | 127.028 |
| layer26 | 5.664e-2 | 127.739 |
| layer27 | 5.664e-2 | 127.741 |
| final | 3.052e-5 | 1.115 |
| head0 | 1.465e-3 | 209.282 |
| head1 | 1.587e-3 | 296.598 |

| rec | qid | type | tokens | ids == python | rust ms | torch ms | max abs dm | max abs dz | max abs dp | argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | department | choice | 158 | yes | 383 | 121 | 1.22e-3 | 5.01e-6 | 1.03e-6 | same |
| 0 | escalate | noul | 144 | yes | 344 | 82 | 2.50e-3 | 1.07e-5 | 9.86e-7 | same |
| 0 | frustration | score | 143 | yes | 344 | 82 | 7.63e-4 | 4.29e-6 | 1.65e-7 | same |
| 1 | label | choice | 94 | yes | 245 | 62 | 7.63e-4 | 4.77e-6 | 4.27e-7 | same |
| 2 | label | choice | 85 | yes | 227 | 62 | 7.32e-4 | 1.67e-6 | 2.01e-7 | same |
| 3 | label | choice | 88 | yes | 229 | 62 | 4.64e-3 | 2.61e-5 | 3.80e-6 | same |
| 4 | label | choice | 88 | yes | 228 | 64 | 1.10e-3 | 2.15e-6 | 3.42e-7 | same |
| 5 | label | choice | 93 | yes | 241 | 63 | 1.83e-3 | 1.03e-5 | 9.08e-7 | same |
| 6 | label | choice | 88 | yes | 227 | 61 | 3.42e-3 | 1.55e-5 | 1.17e-6 | same |
| 7 | label | choice | 39 | yes | 127 | 48 | 1.28e-3 | 4.77e-6 | 6.35e-9 | same |
| 8 | label | choice | 65 | yes | 181 | 64 | 2.72e-3 | 1.58e-5 | 1.78e-7 | same |
| 9 | safe | noul | 48 | yes | 140 | 49 | 1.28e-3 | 5.48e-6 | 2.11e-7 | same |
| 10 | lang | choice | 52 | yes | 151 | 50 | 2.08e-3 | 6.32e-6 | 3.80e-7 | same |
| 11 | billing | noul | 512 | yes | 1244 | 229 | 1.95e-3 | 1.10e-5 | 2.29e-6 | same |

ids identical 14/14; worst |dz| 2.611e-5; worst |dp| 3.801e-6; argmax agree 14/14
