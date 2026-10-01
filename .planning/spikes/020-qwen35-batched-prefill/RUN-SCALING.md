## RAYON_NUM_THREADS=1
load 418 ms, rayon threads 1

| row | tokens | prefill ms | token-at-a-time ms | speedup | max|dh| prefill vs step | max|dh| decide vs torch | max|dp| vs torch | argmax |
|---|---|---|---|---|---|---|---|---|
| 1 | 87 | 1685 | NaN | NaNx | NaN | 7.63e-6 | 1.01e-6 | same |
| 2 | 81 | 1565 | NaN | NaNx | NaN | 8.58e-6 | 4.77e-7 | same |
| 3 | 84 | 1567 | NaN | NaNx | NaN | 1.53e-5 | 1.01e-6 | same |
| 4 | 85 | 1639 | NaN | NaNx | NaN | 8.58e-6 | 2.24e-7 | same |
| 5 | 84 | 1563 | NaN | NaNx | NaN | 1.91e-5 | 5.36e-7 | same |
| 6 | 85 | 1611 | NaN | NaNx | NaN | 7.63e-6 | 1.31e-6 | same |
| 7 | 64 | 1227 | NaN | NaNx | NaN | 8.11e-6 | 1.28e-6 | same |
| 8 | 37 | 814 | NaN | NaNx | NaN | 1.14e-5 | 5.96e-8 | same |
| 9 | 44 | 924 | NaN | NaNx | NaN | 1.53e-5 | 7.15e-7 | same |
| 10 | 36 | 775 | NaN | NaNx | NaN | 6.68e-6 | 2.98e-8 | same |
| 11 | 42 | 882 | NaN | NaNx | NaN | 8.11e-6 | 6.26e-7 | same |
| 12 | 915 | 17437 | NaN | NaNx | NaN | 8.58e-6 | 2.24e-8 | same |

worst |dp| vs torch 1.311e-6, argmax 12/12, total prefill 31688 ms vs token-at-a-time NaN ms

87 tok: gemm 1329 ms, transpose 38 ms, deltanet-loop 281 ms, attention 21 ms, other 15 ms
81 tok: gemm 1233 ms, transpose 35 ms, deltanet-loop 264 ms, attention 18 ms, other 14 ms
84 tok: gemm 1228 ms, transpose 35 ms, deltanet-loop 270 ms, attention 20 ms, other 14 ms
85 tok: gemm 1297 ms, transpose 35 ms, deltanet-loop 272 ms, attention 20 ms, other 14 ms
84 tok: gemm 1225 ms, transpose 35 ms, deltanet-loop 269 ms, attention 20 ms, other 15 ms
85 tok: gemm 1274 ms, transpose 33 ms, deltanet-loop 270 ms, attention 20 ms, other 14 ms
64 tok: gemm 977 ms, transpose 26 ms, deltanet-loop 203 ms, attention 11 ms, other 11 ms
37 tok: gemm 672 ms, transpose 15 ms, deltanet-loop 116 ms, attention 4 ms, other 7 ms
44 tok: gemm 755 ms, transpose 18 ms, deltanet-loop 138 ms, attention 6 ms, other 8 ms
36 tok: gemm 632 ms, transpose 16 ms, deltanet-loop 116 ms, attention 4 ms, other 7 ms
42 tok: gemm 714 ms, transpose 19 ms, deltanet-loop 135 ms, attention 5 ms, other 8 ms
915 tok: gemm 11828 ms, transpose 441 ms, deltanet-loop 2906 ms, attention 2123 ms, other 139 ms

## RAYON_NUM_THREADS=6
load 395 ms, rayon threads 6

| row | tokens | prefill ms | token-at-a-time ms | speedup | max|dh| prefill vs step | max|dh| decide vs torch | max|dp| vs torch | argmax |
|---|---|---|---|---|---|---|---|---|
| 1 | 87 | 376 | NaN | NaNx | NaN | 7.63e-6 | 1.01e-6 | same |
| 2 | 81 | 362 | NaN | NaNx | NaN | 8.58e-6 | 4.77e-7 | same |
| 3 | 84 | 358 | NaN | NaNx | NaN | 1.53e-5 | 1.01e-6 | same |
| 4 | 85 | 372 | NaN | NaNx | NaN | 8.58e-6 | 2.24e-7 | same |
| 5 | 84 | 360 | NaN | NaNx | NaN | 1.91e-5 | 5.36e-7 | same |
| 6 | 85 | 377 | NaN | NaNx | NaN | 7.63e-6 | 1.31e-6 | same |
| 7 | 64 | 293 | NaN | NaNx | NaN | 8.11e-6 | 1.28e-6 | same |
| 8 | 37 | 203 | NaN | NaNx | NaN | 1.14e-5 | 5.96e-8 | same |
| 9 | 44 | 225 | NaN | NaNx | NaN | 1.53e-5 | 7.15e-7 | same |
| 10 | 36 | 194 | NaN | NaNx | NaN | 6.68e-6 | 2.98e-8 | same |
| 11 | 42 | 216 | NaN | NaNx | NaN | 8.11e-6 | 6.26e-7 | same |
| 12 | 915 | 3897 | NaN | NaNx | NaN | 8.58e-6 | 2.24e-8 | same |

worst |dp| vs torch 1.311e-6, argmax 12/12, total prefill 7233 ms vs token-at-a-time NaN ms

87 tok: gemm 252 ms, transpose 20 ms, deltanet-loop 91 ms, attention 5 ms, other 9 ms
81 tok: gemm 237 ms, transpose 27 ms, deltanet-loop 85 ms, attention 5 ms, other 9 ms
84 tok: gemm 237 ms, transpose 19 ms, deltanet-loop 88 ms, attention 5 ms, other 8 ms
85 tok: gemm 250 ms, transpose 20 ms, deltanet-loop 88 ms, attention 5 ms, other 8 ms
84 tok: gemm 240 ms, transpose 20 ms, deltanet-loop 88 ms, attention 5 ms, other 8 ms
85 tok: gemm 255 ms, transpose 20 ms, deltanet-loop 88 ms, attention 5 ms, other 8 ms
64 tok: gemm 198 ms, transpose 18 ms, deltanet-loop 67 ms, attention 3 ms, other 7 ms
37 tok: gemm 141 ms, transpose 14 ms, deltanet-loop 40 ms, attention 2 ms, other 6 ms
44 tok: gemm 155 ms, transpose 15 ms, deltanet-loop 47 ms, attention 2 ms, other 6 ms
36 tok: gemm 131 ms, transpose 15 ms, deltanet-loop 40 ms, attention 1 ms, other 7 ms
42 tok: gemm 145 ms, transpose 16 ms, deltanet-loop 46 ms, attention 2 ms, other 7 ms
915 tok: gemm 2364 ms, transpose 139 ms, deltanet-loop 960 ms, attention 383 ms, other 51 ms

## RAYON_NUM_THREADS=14
load 406 ms, rayon threads 14

| row | tokens | prefill ms | token-at-a-time ms | speedup | max|dh| prefill vs step | max|dh| decide vs torch | max|dp| vs torch | argmax |
|---|---|---|---|---|---|---|---|---|
| 1 | 87 | 336 | NaN | NaNx | NaN | 7.63e-6 | 1.01e-6 | same |
| 2 | 81 | 317 | NaN | NaNx | NaN | 8.58e-6 | 4.77e-7 | same |
| 3 | 84 | 321 | NaN | NaNx | NaN | 1.53e-5 | 1.01e-6 | same |
| 4 | 85 | 326 | NaN | NaNx | NaN | 8.58e-6 | 2.24e-7 | same |
| 5 | 84 | 313 | NaN | NaNx | NaN | 1.91e-5 | 5.36e-7 | same |
| 6 | 85 | 332 | NaN | NaNx | NaN | 7.63e-6 | 1.31e-6 | same |
| 7 | 64 | 275 | NaN | NaNx | NaN | 8.11e-6 | 1.28e-6 | same |
| 8 | 37 | 202 | NaN | NaNx | NaN | 1.14e-5 | 5.96e-8 | same |
| 9 | 44 | 216 | NaN | NaNx | NaN | 1.53e-5 | 7.15e-7 | same |
| 10 | 36 | 184 | NaN | NaNx | NaN | 6.68e-6 | 2.98e-8 | same |
| 11 | 42 | 199 | NaN | NaNx | NaN | 8.11e-6 | 6.26e-7 | same |
| 12 | 915 | 2954 | NaN | NaNx | NaN | 8.58e-6 | 2.24e-8 | same |

worst |dp| vs torch 1.311e-6, argmax 12/12, total prefill 5976 ms vs token-at-a-time NaN ms

87 tok: gemm 188 ms, transpose 35 ms, deltanet-loop 80 ms, attention 4 ms, other 28 ms
81 tok: gemm 178 ms, transpose 34 ms, deltanet-loop 75 ms, attention 4 ms, other 25 ms
84 tok: gemm 183 ms, transpose 34 ms, deltanet-loop 77 ms, attention 4 ms, other 23 ms
85 tok: gemm 186 ms, transpose 34 ms, deltanet-loop 77 ms, attention 4 ms, other 24 ms
84 tok: gemm 174 ms, transpose 33 ms, deltanet-loop 76 ms, attention 4 ms, other 26 ms
85 tok: gemm 188 ms, transpose 34 ms, deltanet-loop 78 ms, attention 4 ms, other 27 ms
64 tok: gemm 156 ms, transpose 33 ms, deltanet-loop 62 ms, attention 3 ms, other 22 ms
37 tok: gemm 110 ms, transpose 29 ms, deltanet-loop 37 ms, attention 2 ms, other 24 ms
44 tok: gemm 118 ms, transpose 30 ms, deltanet-loop 42 ms, attention 2 ms, other 25 ms
36 tok: gemm 97 ms, transpose 28 ms, deltanet-loop 35 ms, attention 2 ms, other 23 ms
42 tok: gemm 106 ms, transpose 28 ms, deltanet-loop 40 ms, attention 2 ms, other 22 ms
915 tok: gemm 1709 ms, transpose 145 ms, deltanet-loop 807 ms, attention 231 ms, other 62 ms

