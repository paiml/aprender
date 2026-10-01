# Spike 010 — spike-004 server under concurrent requests (14 cores)

# Configuration: pool = 1 (spike-004 as shipped: one `Arc<Mutex<Server>>` behind pmcp's router)

Server in-process, streamable-HTTP, 8 requests: peyton/prophet, peyton/neuralprophet-lags30, air/prophet, air/prophet-multiplicative, synth1/prophet, synth1/neuralprophet-lags30, synth2/prophet, synth2/neuralprophet-lags30

## 1. Sequential baseline

| request | seconds |
|---|---|
| peyton/prophet | 1.46 |
| peyton/neuralprophet-lags30 | 1.21 |
| air/prophet | 0.02 |
| air/prophet-multiplicative | 0.01 |
| synth1/prophet | 0.17 |
| synth1/neuralprophet-lags30 | 1.21 |
| synth2/prophet | 0.25 |
| synth2/neuralprophet-lags30 | 1.21 |
| **total** | **5.53** |

## 2. Concurrent rounds (all 8 at once)

| round | wall s | speed-up vs sequential | responses identical to baseline | max |Δ yhat| |
|---|---|---|---|---|
| 1 | 5.41 | 1.0× | 8/8 | 0.0e0 |
| 2 | 5.40 | 1.0× | 8/8 | 0.0e0 |
| 3 | 5.43 | 1.0× | 8/8 | 0.0e0 |

## 3. Stress: 16 simultaneous NeuralProphet n_lags=30 fits

16/16 identical to their sequential result, 0 errors, wall 19.42 s (one such fit alone: 1.21 s)

Correctness: every concurrent response equals its sequential result; the thread-local tape and per-request seeds hold under load

# Configuration: pool = 8 (K independent pmcp routers, round-robin front handler)

Server in-process, streamable-HTTP, 8 requests: peyton/prophet, peyton/neuralprophet-lags30, air/prophet, air/prophet-multiplicative, synth1/prophet, synth1/neuralprophet-lags30, synth2/prophet, synth2/neuralprophet-lags30

## 1. Sequential baseline

| request | seconds |
|---|---|
| peyton/prophet | 1.41 |
| peyton/neuralprophet-lags30 | 1.21 |
| air/prophet | 0.02 |
| air/prophet-multiplicative | 0.01 |
| synth1/prophet | 0.19 |
| synth1/neuralprophet-lags30 | 1.22 |
| synth2/prophet | 0.26 |
| synth2/neuralprophet-lags30 | 1.23 |
| **total** | **5.56** |

## 2. Concurrent rounds (all 8 at once)

| round | wall s | speed-up vs sequential | responses identical to baseline | max |Δ yhat| |
|---|---|---|---|---|
| 1 | 1.44 | 3.9× | 8/8 | 0.0e0 |
| 2 | 1.44 | 3.9× | 8/8 | 0.0e0 |
| 3 | 1.43 | 3.9× | 8/8 | 0.0e0 |

## 3. Stress: 16 simultaneous NeuralProphet n_lags=30 fits

16/16 identical to their sequential result, 0 errors, wall 3.16 s (one such fit alone: 1.21 s)

Correctness: every concurrent response equals its sequential result; the thread-local tape and per-request seeds hold under load
