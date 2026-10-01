## f32
load: 429 ms (mmap GGUF + owned layers + head)

| rec | row | tokens | rust ms | torch-cpu ms | max|dh| decide (rel rms) | max|dh| opts | max|dlogit| | max|dp| | argmax |
|---|---|---|---|---|---|---|---|---|---|
ladder row0: final hidden at all 87 positions, max|dh| 1.221e-4 at pos 25 (rms of h 2.096)
| 0 | 0 | 87 (state 35) | 6884 | 4225 | 5.15e-5 (4.2e-5) | 4.39e-5 | 8.64e-6 | 1.13e-6 | same |
| 1 | 0 | 81 (state 29) | 6398 | 4193 | 5.05e-5 (4.2e-5) | 4.01e-5 | 8.17e-6 | 3.58e-7 | same |
| 2 | 0 | 84 (state 32) | 6622 | 4239 | 3.05e-5 (2.5e-5) | 2.86e-5 | 1.34e-5 | 1.52e-6 | same |
| 3 | 0 | 85 (state 33) | 6686 | 4204 | 2.05e-5 (1.7e-5) | 2.57e-5 | 2.15e-6 | 1.19e-7 | same |
| 4 | 0 | 84 (state 32) | 6601 | 4182 | 3.24e-5 (2.7e-5) | 3.91e-5 | 1.97e-5 | 1.25e-6 | same |
| 5 | 0 | 85 (state 33) | 6749 | 4793 | 2.19e-5 (1.8e-5) | 2.00e-5 | 1.07e-5 | 6.26e-7 | same |
| 6 | 0 | 64 (state 22) | 5064 | 2797 | 2.19e-5 (1.8e-5) | 3.91e-5 | 9.42e-6 | 1.10e-6 | same |
| 6 | 1 | 37 (state 22) | 2901 | 2797 | 3.10e-5 (2.6e-5) | 2.38e-5 | 6.79e-6 | 5.36e-7 | same |
| 6 | 2 | 44 (state 22) | 3478 | 2797 | 1.43e-5 (1.2e-5) | 3.34e-5 | 3.34e-6 | 2.98e-7 | same |
| 7 | 0 | 36 (state 19) | 2830 | 4388 | 6.29e-5 (5.2e-5) | 4.20e-5 | 4.53e-6 | 1.19e-7 | same |
| 8 | 0 | 42 (state 24) | 3309 | 4657 | 1.49e-5 (1.2e-5) | 5.44e-5 | 1.22e-5 | 8.34e-7 | same |

worst |dlogit| 1.967e-5, worst |dp| 1.520e-6, argmax agree 11/11
## bf16
load: 2340 ms (mmap GGUF + owned layers + head)

| rec | row | tokens | rust ms | torch-cpu ms | max|dh| decide (rel rms) | max|dh| opts | max|dlogit| | max|dp| | argmax |
|---|---|---|---|---|---|---|---|---|---|
ladder row0: final hidden at all 87 positions, max|dh| 7.914e-2 at pos 55 (rms of h 1.760)
| 0 | 0 | 87 (state 35) | 3877 | 4225 | 1.43e-2 (1.2e-2) | 1.66e-2 | 2.01e-2 | 1.12e-3 | same |
| 1 | 0 | 81 (state 29) | 3513 | 4193 | 1.49e-2 (1.2e-2) | 1.51e-2 | 5.00e-3 | 3.82e-4 | same |
| 2 | 0 | 84 (state 32) | 3608 | 4239 | 1.39e-2 (1.1e-2) | 1.53e-2 | 5.62e-3 | 2.21e-4 | same |
| 3 | 0 | 85 (state 33) | 3663 | 4204 | 1.38e-2 (1.1e-2) | 1.33e-2 | 1.13e-2 | 2.88e-4 | same |
| 4 | 0 | 84 (state 32) | 3669 | 4182 | 1.26e-2 (1.0e-2) | 1.29e-2 | 8.97e-3 | 9.70e-4 | same |
| 5 | 0 | 85 (state 33) | 3648 | 4793 | 1.19e-2 (9.8e-3) | 1.14e-2 | 1.02e-2 | 4.92e-4 | same |
| 6 | 0 | 64 (state 22) | 2746 | 2797 | 1.35e-2 (1.1e-2) | 1.38e-2 | 2.34e-2 | 1.79e-3 | same |
| 6 | 1 | 37 (state 22) | 1598 | 2797 | 1.13e-2 (9.3e-3) | 1.35e-2 | 2.76e-2 | 1.92e-3 | same |
| 6 | 2 | 44 (state 22) | 1888 | 2797 | 1.27e-2 (1.0e-2) | 1.29e-2 | 8.50e-3 | 1.75e-4 | same |
| 7 | 0 | 36 (state 19) | 1543 | 4388 | 1.62e-2 (1.3e-2) | 1.40e-2 | 1.62e-2 | 1.69e-3 | same |
| 8 | 0 | 42 (state 24) | 1727 | 4657 | 1.46e-2 (1.2e-2) | 1.47e-2 | 2.07e-2 | 1.17e-3 | same |

worst |dlogit| 2.761e-2, worst |dp| 1.920e-3, argmax agree 11/11
## f16
load: 2340 ms (mmap GGUF + owned layers + head)

| rec | row | tokens | rust ms | torch-cpu ms | max|dh| decide (rel rms) | max|dh| opts | max|dlogit| | max|dp| | argmax |
|---|---|---|---|---|---|---|---|---|---|
ladder row0: final hidden at all 87 positions, max|dh| 1.032e-2 at pos 22 (rms of h 2.511)
| 0 | 0 | 87 (state 35) | 5587 | 4225 | 2.72e-3 (2.2e-3) | 2.90e-3 | 3.45e-3 | 1.74e-4 | same |
| 1 | 0 | 81 (state 29) | 5261 | 4193 | 3.23e-3 (2.7e-3) | 2.65e-3 | 9.79e-4 | 8.49e-5 | same |
| 2 | 0 | 84 (state 32) | 4900 | 4239 | 2.83e-3 (2.3e-3) | 2.80e-3 | 1.43e-3 | 6.18e-5 | same |
| 3 | 0 | 85 (state 33) | 5122 | 4204 | 3.43e-3 (2.8e-3) | 2.79e-3 | 1.32e-3 | 6.64e-5 | same |
| 4 | 0 | 84 (state 32) | 5503 | 4182 | 2.91e-3 (2.4e-3) | 2.83e-3 | 2.67e-3 | 1.62e-4 | same |
| 5 | 0 | 85 (state 33) | 5515 | 4793 | 2.72e-3 (2.2e-3) | 2.52e-3 | 1.24e-3 | 1.50e-4 | same |
| 6 | 0 | 64 (state 22) | 4147 | 2797 | 2.56e-3 (2.1e-3) | 2.52e-3 | 3.35e-3 | 3.41e-4 | same |
| 6 | 1 | 37 (state 22) | 2397 | 2797 | 2.98e-3 (2.5e-3) | 2.53e-3 | 2.64e-3 | 1.85e-4 | same |
| 6 | 2 | 44 (state 22) | 2811 | 2797 | 2.90e-3 (2.4e-3) | 2.50e-3 | 3.99e-3 | 1.64e-4 | same |
| 7 | 0 | 36 (state 19) | 2350 | 4388 | 3.93e-3 (3.3e-3) | 3.44e-3 | 3.89e-4 | 1.67e-5 | same |
| 8 | 0 | 42 (state 24) | 2716 | 4657 | 3.06e-3 (2.5e-3) | 3.17e-3 | 3.31e-3 | 2.85e-4 | same |

worst |dlogit| 3.987e-3, worst |dp| 3.405e-4, argmax agree 11/11
## q8_0
load: 4680 ms (mmap GGUF + owned layers + head)

| rec | row | tokens | rust ms | torch-cpu ms | max|dh| decide (rel rms) | max|dh| opts | max|dlogit| | max|dp| | argmax |
|---|---|---|---|---|---|---|---|---|---|
ladder row0: final hidden at all 87 positions, max|dh| 4.429e-1 at pos 27 (rms of h 2.215)
| 0 | 0 | 87 (state 35) | 3726 | 4225 | 1.18e-1 (9.6e-2) | 1.98e-1 | 1.55e-1 | 7.13e-3 | same |
| 1 | 0 | 81 (state 29) | 3495 | 4193 | 2.05e-1 (1.7e-1) | 2.26e-1 | 4.74e-2 | 6.26e-4 | same |
| 2 | 0 | 84 (state 32) | 3668 | 4239 | 1.63e-1 (1.3e-1) | 2.36e-1 | 1.69e-1 | 9.74e-3 | same |
| 3 | 0 | 85 (state 33) | 3681 | 4204 | 1.58e-1 (1.3e-1) | 1.33e-1 | 4.14e-2 | 4.91e-4 | same |
| 4 | 0 | 84 (state 32) | 3626 | 4182 | 1.28e-1 (1.1e-1) | 1.69e-1 | 1.27e-1 | 6.26e-3 | same |
| 5 | 0 | 85 (state 33) | 3649 | 4793 | 1.18e-1 (9.7e-2) | 1.72e-1 | 8.91e-2 | 7.23e-3 | same |
| 6 | 0 | 64 (state 22) | 2831 | 2797 | 1.62e-1 (1.3e-1) | 1.64e-1 | 1.30e-1 | 7.21e-3 | same |
| 6 | 1 | 37 (state 22) | 1614 | 2797 | 1.36e-1 (1.1e-1) | 1.61e-1 | 1.39e-1 | 1.34e-2 | same |
| 6 | 2 | 44 (state 22) | 1901 | 2797 | 1.92e-1 (1.6e-1) | 2.43e-1 | 6.16e-2 | 2.18e-3 | same |
| 7 | 0 | 36 (state 19) | 1520 | 4388 | 1.07e-1 (8.9e-2) | 1.11e-1 | 3.10e-2 | 3.66e-3 | same |
| 8 | 0 | 42 (state 24) | 1857 | 4657 | 1.17e-1 (9.5e-2) | 1.62e-1 | 8.89e-2 | 4.23e-3 | same |

worst |dlogit| 1.693e-1, worst |dp| 1.340e-2, argmax agree 11/11
