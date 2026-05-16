# Audit Report: Albor 370M Roadmap (`albor-370m-roadmap.md`)

## 1. Executive Summary & Convergence Failure Analysis

The `albor-370m-roadmap.md` describes the final push to ship MODEL-2 (`aprender/albor-370m`). Despite achieving a `val_loss` of 4.71, the model is still failing to converge to a usable state, producing repetitive token gibberish (e.g., `def fibonacci(n):` → ` č č č č ...`).

**Why it is failing to converge:**
1. **Severe Data Starvation (The Binding Constraint):** The roadmap explicitly notes that the Chinchilla gate warning fires because $D=22M$ tokens versus $N=494M$ parameters (which implies a ~370M-494M param scale depending on vocab/embeddings). This is an empirical ratio of $0.04\times$, which is catastrophically below the $20\times$ target required for compute-optimal training. The model does not have enough unique data to learn generalizable syntax, leading to mode collapse and repetition.
2. **Compute Misallocation Risk:** The roadmap prioritizes `P2-A2` (running more steps on the same small corpus) over `P2-C` (widening the corpus). Burning 3-8 hours of GPU time on the same data will merely overfit more deeply, not break the generalization plateau necessary to drop `val_loss` below 3.0.

## 2. Popperian Falsification Assessment

The roadmap introduces clear, falsifiable hypotheses, specifically around the `val_loss` plateau and the need for distillation vs. pretraining:

*   **Hypothesis (P2-A2):** A longer run (20K-50K steps) on the same corpus (`qwen-v2` subset) will drop `val_loss` below 3.5.
*   **Falsification:** If the model hits a hard validation loss plateau early in the run (as observed in the previous Two-Model spec audit), then the step-count hypothesis is falsified, proving the dataset diversity is the binding constraint.
*   **Conclusion:** We can predict *a priori* based on the 0.04x Chinchilla ratio that P2-A2 will fail to reach the target, validating the need for the P2-C (Wider corpus) pipeline.

## 3. Literature Support (ArXiv Citations)

*   **Chinchilla Scaling Laws:** *Training Compute-Optimal Large Language Models* (Hoffmann et al., 2022 - [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)). The roadmap explicitly references this, but the execution queue does not enforce it. To train a ~370M-494M parameter model, the dataset must be in the 7.4B to 10B token range. $22M$ tokens is insufficient.
*   **Repetitive Degeneration:** *The Curious Case of Neural Text Degeneration* (Holtzman et al., 2019 - [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)). The "repetitive token gibberish" observed at `val_loss=4.71` is a classic symptom of an under-trained or poorly-regularized language model falling into a high-confidence, low-entropy loop. Without sufficient data diversity to shape the probability distribution of the long tail, the model collapses into predicting the same token infinitely.

## 4. Specific Code/Process Examples & Five-Whys Analysis

### Case: Repetitive Token Gibberish at `val_loss=4.71`
*   **Observation:** The model generates `č č č č ...` despite reaching a seemingly "okay" pre-convergence validation loss of 4.71.
*   **Why 1:** Why is it generating gibberish? Because `val_loss=4.71` corresponds to a perplexity of $e^{4.71} \approx 111$, which means the model is highly uncertain and falls back to mode collapse (predicting the most frequent or degenerate token loops).
*   **Why 2:** Why did the validation loss plateau at 4.71? Because the model has extracted all the structural signal it can from the tiny 22M token dataset.
*   **Why 3:** Why is the model using only 22M tokens? Because the `qwen-v2` subset currently used is a small sample, not a full-scale pretraining corpus.
*   **Why 4:** Why was the training allowed to proceed on such a small dataset? Because the Chinchilla constraint ($D \approx 20N$) was only recently added as a warning (`P1-A2`), not a hard blocker.
*   **Why 5:** Why was it not a hard blocker? Infrastructure focused on verifying pipeline mechanics (getting the loss to go down *at all*) rather than ensuring theoretical convergence feasibility.

## 5. Engineering Recommendations for Rapid Convergence

To ensure the ALBOR-370M model reaches a `val_loss < 3.0` and avoids wasted compute, the following engineering interventions are strongly recommended:

### Recommendation 1: Re-rank P2-C over P2-A2 Immediately
**Abandon P2-A2 (longer run on the same small data).** Given the Chinchilla ratio of $0.04\times$, P2-A2 is guaranteed to result in overfitting and will not break the `val_loss` plateau. 
*   **Action:** Immediately promote **P2-C (Wider corpus: codeparrot-python permissive + the-stack-v2 Python)** to the highest priority dispatch.
*   **Target:** Ensure the re-tokenized corpus is strictly $> 2$ Billion tokens before dispatching any further multi-hour GPU training runs.

### Recommendation 2: Make the Chinchilla Gate a Hard Blocker
Currently, `P1-A2` only verifies that the Chinchilla gate *warns* the user. A warning is insufficient and leads to wasted RTX 4090 compute hours.
*   **Action:** Update the `apr pretrain` CLI to fail fast with a fatal error if the $D/N$ ratio is less than $10\times$ (as a minimum floor) unless overridden with a explicit `--force-under-provisioned` flag.

### Recommendation 3: Defer Downstream Evals (P1-B, P1-C, P3-A)
*   **Action:** Do not waste CPU/GPU hours or developer time running HumanEval (`P1-B`), AST parsing (`P1-C`), or Quality scoring (`P3-A`) while `val_loss > 3.0`.
*   **Reasoning:** At a perplexity of $> 20$ (`val_loss > 3.0`), the model is mathematically incapable of demonstrating zero-shot reasoning or complex syntax generation. All testing resources should be entirely redirected to the data engineering pipeline (P2-C).