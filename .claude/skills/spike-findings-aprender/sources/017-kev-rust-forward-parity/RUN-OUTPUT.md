load: 3089 ms (mmap GGUF + owned layers + head)

| rec | row | tokens | rust ms | torch-cpu ms | max|dh| decide (rel rms) | max|dh| opts | max|dlogit| | max|dp| | argmax |
|---|---|---|---|---|---|---|---|---|---|
ladder row0: final hidden at all 87 positions, max|dh| 1.221e-4 at pos 25 (rms of h 2.096)
| 0 | 0 | 87 (state 35) | 6276 | 4225 | 5.15e-5 (4.2e-5) | 4.39e-5 | 8.64e-6 | 1.13e-6 | same |
| 1 | 0 | 81 (state 29) | 5844 | 4193 | 5.05e-5 (4.2e-5) | 4.01e-5 | 8.17e-6 | 3.58e-7 | same |
| 2 | 0 | 84 (state 32) | 6545 | 4239 | 3.05e-5 (2.5e-5) | 2.86e-5 | 1.34e-5 | 1.52e-6 | same |
| 3 | 0 | 85 (state 33) | 6617 | 4204 | 2.05e-5 (1.7e-5) | 2.57e-5 | 2.15e-6 | 1.19e-7 | same |
| 4 | 0 | 84 (state 32) | 6581 | 4182 | 3.24e-5 (2.7e-5) | 3.91e-5 | 1.97e-5 | 1.25e-6 | same |
| 5 | 0 | 85 (state 33) | 6613 | 4793 | 2.19e-5 (1.8e-5) | 2.00e-5 | 1.07e-5 | 6.26e-7 | same |
| 6 | 0 | 64 (state 22) | 5005 | 2797 | 2.19e-5 (1.8e-5) | 3.91e-5 | 9.42e-6 | 1.10e-6 | same |
| 6 | 1 | 37 (state 22) | 2986 | 2797 | 3.10e-5 (2.6e-5) | 2.38e-5 | 6.79e-6 | 5.36e-7 | same |
| 6 | 2 | 44 (state 22) | 3491 | 2797 | 1.43e-5 (1.2e-5) | 3.34e-5 | 3.34e-6 | 2.98e-7 | same |
| 7 | 0 | 36 (state 19) | 2786 | 4388 | 6.29e-5 (5.2e-5) | 4.20e-5 | 4.53e-6 | 1.19e-7 | same |
| 8 | 0 | 42 (state 24) | 3258 | 4657 | 1.49e-5 (1.2e-5) | 5.44e-5 | 1.22e-5 | 8.34e-7 | same |
| 9 | 0 | 915 (state 901) | 72250 | 5417 | 2.00e-5 (1.6e-5) | 4.96e-5 | 1.43e-5 | 4.77e-7 | same |

worst |dlogit| 1.967e-5, worst |dp| 1.520e-6, argmax agree 12/12
