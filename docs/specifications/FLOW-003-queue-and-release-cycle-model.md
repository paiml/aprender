# FLOW-003: Merge-queue and release-cycle model: equations, proofs, and queueing policy

**Spec id:** `FLOW-003` · **Rows:** `QM-00..QM-07` (the traffic cop mints one `pmat` ticket per row; single-minter rule applies)
**Target repo:** `paiml/aprender` (`docs/specifications/FLOW-003-queue-and-release-cycle-model.md`). Ruleset and runner changes are filed as issues and not done here (§0.6).
**Runner:** the aprender traffic cop (`aprender-traffic-cop-prompt.md`).
**Launch:** from `~/src/aprender`, run `Implement docs/specifications/FLOW-003-queue-and-release-cycle-model.md autonomously.`
**Related:** FLOW-001 (fleet flow rules), FLOW-002 (dev → rc → release ladder), `release-flow-rules.md` (cop, 2026-09-27; open questions 1, 3, 5), APR-RELEASE-001, merge-queue ruleset 17836320, FLAKE-0 (owner a2), CI sharding (a2/49/5d).
**Status:** spec v1.1, not implemented. Dated 2026-09-27. v1.1 applies review findings R-1..R-12 (§12) and adds §5.1 service classes (cheap vs full lane, QM-08); ticket PMAT-4514.

**Provenance marks:** `[V]` verified at the cited time · `[C]` computed · `[A]` asserted · `[U]` unmeasured · `[X]` third-party.

**Ontology (ONT-001, `ont.paiml.dev/v1alpha1`):** this spec is the entity `spec:FLOW-003` (ONT-001 Appendix B.11). Each result carries a `pv:` marker naming the contract clause that holds it. The contracts are in §11. A result gets credit only when its contract's `proof.status` is `discharged`, and every contract here stays `declared` until QM-00 lands.

---

## §0 Operating assumptions

1. **Purpose.** Replace judgement calls about the merge queue and the release PR with equations whose inputs are measured. Four decisions follow from the model: group size, retry count, fold size, and where cycle time pays.
2. **Every decision in §7 is a theorem or proposition in §4–§5 evaluated at measured inputs.** A decision whose inputs are still `[U]` holds the value in §7, which is derived from the robust results (Theorems 2 and 6). The robust results hold for every value of the inputs.
3. **Proofs are part of the spec.** A change to §3 (assumptions) or §4–§5 (results) is a new spec version. The xtask (QM-00) implements the closed forms, and its golden tests pin the §6 values.
4. **Simulation checks the model; it never overrides a proof.** Where the proofs give only a necessary condition (Theorem 8), the simulation sets the operating threshold, which is marked `[C]` and recalibrated from measured data (QM-05).
5. **No Python in the repo.** The analysis that produced §6 was done outside the repo. The model, the evaluator and the simulator in the repo are Rust (`cargo xtask queue-model`, `cargo xtask queue-sim`).
6. **Cross-repo and settings work is filed, not done.** Ruleset 17836320 changes and runner changes are filed as issues with checkable acceptance criteria.
7. **The release train wins.** No row here changes CI settings on the release PR while a cut is in progress (§8 S-1).

---

## §1 Ground truth (baseline, frozen 2026-09-27; never quote as current)

| # | Fact | Value | Mark | Source of truth |
|---|---|---|---|---|
| G1 | Merge-queue build time *T* | 14 min | [V] n=1 | `release-flow-rules.md` "Where the time goes" |
| G2 | Release-PR cycle *C* (review beside full CI) | 75–90 min | [A] | same |
| G3 | Merge-queue lanes *L* / group size *B* | 8 / 8 | [V] | ruleset 17836320 |
| G4 | CI retries *r* | 2 | [V] | nextest profile; FLAKE-0 plans 2 → 0 |
| G5 | Open aprender PRs | 19 (cap 10) | [V] | `gh pr list` |
| G6 | Per-entry defect rate in the merge queue *q* | unknown | [U] | QM-01 |
| G7 | Flake rate per CI run *φ* | unknown | [U] | a2's flake ledger, QM-01 |
| G8 | PR arrival rate *λ* | unknown | [U] | QM-01 |
| G9 | Fix success rate *f*, fix time *F*, retry cost *ρ*, re-arm delay *R* | unknown | [U] | QM-01 |
| G10 | GitHub behaviour when a group of size > 1 fails (bisect vs eject whole), and whether GitHub builds one commit per group at all (R-3) | unknown | [U] | QM-02 |

---

## §2 Notation

| Symbol | Meaning | Unit |
|---|---|---|
| $C$ | release-PR cycle time (review beside CI; CI is the long pole) | min |
| $T$ | merge-queue build time | min |
| $\rho$ | extra time per CI retry (rerun of the failing shard). Two measured values: $\rho_{\text{rel}}$ on the release PR (§4) and $\rho_{\text{MQ}}$ in the merge queue (§5). A bare $\rho$ in §4 means $\rho_{\text{rel}}$, and in §5 it means $\rho_{\text{MQ}}$ (R-11) | min |
| $r$ | retries per CI run ($r+1$ attempts) | — |
| $\varphi$ | probability that one attempt fails spuriously (flake) | — |
| $q$ | probability that one PR carries a defect that CI detects | — |
| $f$ | probability that a fix removes the defect | — |
| $F$ | time from a red cycle to the fix being pushed | min |
| $R$ | delay before a falsely ejected PR is re-armed | min |
| $k$ | number of PRs folded into one release-PR push | — |
| $L$ | merge-queue build concurrency (speculative lanes) | — |
| $B = 2^h$ | maximum merge-queue group size | — |
| $\lambda$ | PR arrival rate into the merge queue | 1/min |
| $S_r$ | $\sum_{j=1}^{r}\varphi^{j}$ (expected retries used by a clean run) | — |

---

## §3 Model assumptions

**Release PR**

- **A1 Detection.** A cycle whose head contains an unfixed defect is red. Every retry also fails, so such a cycle costs $C + r\rho$.
- **A2 Identification.** A red cycle names every defective PR through its failing tests. QM-01 measures the identification rate. Below 0.9 → §8 S-4.
- **A3 Parallel fixes.** All named defects are fixed in parallel within $F$. Each fix succeeds independently with probability $f$.
- **A4 Flakes.** Attempts flake independently with probability $\varphi$. A flake-only failure is rerun at once with no fix time.
- **A5 Independence.** Defects are independent Bernoulli($q$) across PRs.

**Merge queue**

> **Q1–Q3 are `[U]` against GitHub's actual merge queue (R-3).** They describe *a* grouped queue that bisects. GitHub may instead give every entry its own `merge_group` build of *main + all entries ahead + itself*: build concurrency caps how many of those builds run, "max PRs to merge" lets a passing build merge the entries ahead of it, and a failing entry is removed while the builds behind it are recreated. In that case Lemma 5's tree is not the mechanism. QM-02 records the observed semantics. Until it does, the §6.3 rows with $B > 1$ are model rows, not a description of the live configuration. The §7 group-size decision does not depend on this: at $B = 1$ every semantics coincides (Theorem 6, Remark).

- **Q1 Speculation.** Entries form groups of up to $B$ consecutive entries. Up to $L$ builds run at once, each on top of every build ahead of it.
- **Q2 Outcome.** A build fails if and only if its group contains a defect ($\varphi$ is handled in Proposition 10). It takes $T$ if it passes and $T + r\rho$ if it fails.
- **Q3 Failure handling.** When the head build fails, every build behind it is cancelled. A failed group of size > 1 is split into halves, which go to the front of the queue left half first (bisection). A failed group of size 1 is ejected.
- **Q4 Re-entry.** An ejected defective PR re-enters after its fix, and the fix succeeds with probability $f$. Per original PR, the expected number of entries is $1 + q/f$, so the effective arrival rate and the per-entry defect rate are
$$\lambda' = \lambda\left(1+\frac{q}{f}\right), \qquad q' = \frac{q}{f+q}.$$

*Derivation of Q4.* A defective PR fails once per attempt until a fix succeeds. The number of failing entries is Geometric($f$), with mean $1/f$, and one passing entry follows. So the expected number of entries is $(1-q)\cdot 1 + q(1/f + 1) = 1 + q/f$, and the expected number of defective entries is $q/f$. Hence $q' = (q/f)/(1+q/f) = q/(f+q)$. ∎

`pv: queue-model-v1 / DEF-Q4`

- **Q5 Negligible false re-entry (R-7).** A good entry that is falsely ejected re-enters after $R$ and adds $(1-q')\varphi^{r+1}$ entries per entry, a term that Q4 leaves out. The model assumes $\varphi^{r+1} \ll q/f$. With $\varphi \le 0.01$ (the §7 flake budget) the omitted term is under 1% of $\lambda'$. If measured inputs violate Q5, the xtask adds the term to $\lambda'$ and reports that it did.

---

## §4 Release PR

### Lemma 1 (number of defect cycles)
Let $N_i$ be the number of red cycles caused by PR $i$ ($N_i = 0$ if PR $i$ is clean). Under A1–A3 and A5, $P(N_i > n) = q(1-f)^n$ for $n \ge 0$. The number of defect cycles in a fold of $k$ PRs is $M = \max_i N_i$, with
$$\mathbb{E}[M] = \sum_{n=0}^{\infty}\Big[1 - \big(1 - q(1-f)^n\big)^{k}\Big].$$

*Proof.* A clean PR causes no red cycles. A defective PR (probability $q$) is red in cycle 1 (A1) and stays red in each later cycle until a fix succeeds. Fixes succeed independently with probability $f$ (A3), so $N_i \mid \text{defective} \sim$ Geometric($f$) on $\{1,2,\dots\}$, and $P(N_i > n) = q(1-f)^n$. Fixes run in parallel (A3), so cycle $n$ is a defect cycle if and only if some $N_i \ge n$. The number of defect cycles is therefore $M = \max_i N_i$. By independence (A5), $P(M \le n) = \prod_i P(N_i \le n) = (1 - q(1-f)^n)^k$. For a non-negative integer variable, $\mathbb{E}[M] = \sum_{n \ge 0} P(M > n)$. ∎

`pv: queue-model-v1 / POST-L1`

### Theorem 1 (expected time to a green release head)
Under A1–A5, the expected time until the release PR is green with all $k$ folded PRs is
$$\boxed{\;\mathbb{E}[T_{\text{fold}}(k,r)] \;=\; \mathbb{E}[M]\,\big(C + r\rho + F\big) \;+\; \frac{C + \rho\, S_r}{1-\varphi^{\,r+1}}\;}$$

*Proof.* Split the timeline into a defect phase and a clean phase.

- **Defect phase.** By Lemma 1 there are $M$ defect cycles. Each is red regardless of flakes (A1), costs $C + r\rho$ because every retry fails, and is followed by a fix of duration $F$ (A3). Its expected cost is $\mathbb{E}[M](C + r\rho + F)$.
- **Clean phase.** With no defects left, attempts are i.i.d. An attempt fails only if all $r+1$ tries flake, which has probability $\varphi^{r+1}$ (A4). So the number of attempts $G$ is Geometric($s$) with $s = 1 - \varphi^{r+1}$.
  - An attempt uses $\min(X, r)$ retries, where $X$ is the number of leading flakes and $P(X \ge j) = \varphi^j$. Its expected cost is $C + \rho\,\mathbb{E}[\min(X,r)] = C + \rho\sum_{j=1}^{r} P(X \ge j) = C + \rho S_r$.
  - $G$ is a stopping time for the attempt sequence: the event $\{G \ge i\}$ depends only on attempts $1,\dots,i-1$. So Wald's identity gives an expected clean-phase cost of $\mathbb{E}[G]\,(C + \rho S_r) = (C + \rho S_r)/s$.

Adding the two phases gives the result. ∎

`pv: queue-model-v1 / POST-T1`

### Theorem 2 (one fold dominates any sequential split)
Partition the $k$ PRs into $m \ge 2$ batches $\mathcal B_1,\dots,\mathcal B_m$, each pushed and driven to green before the next. Then
$$\mathbb{E}[T_{\text{fold}}] < \mathbb{E}\Big[\sum_{j=1}^{m} T_{\mathcal B_j}\Big].$$

*Proof.* Couple the two strategies on the same realisation of $(N_i)$. By Theorem 1's decomposition, batch $j$ costs $M_j(C + r\rho + F) + \Gamma_j$, where $M_j = \max_{i \in \mathcal B_j} N_i$ and the clean-phase costs $\Gamma_j$ are i.i.d. with mean $\gamma = (C + \rho S_r)/(1 - \varphi^{r+1}) \ge C > 0$.

- **Defect phases.** Pathwise, $M = \max_i N_i \le \sum_j M_j$, because the maximum over a union of non-negative sets is at most the sum of the per-set maxima. So the fold's defect phase costs no more than the batches' defect phases combined.
- **Clean phases.** The fold has one clean phase, with expected cost $\gamma$. The partition has $m$, with expected cost $m\gamma$.

Taking expectations, $\mathbb{E}[T_{\text{fold}}] \le \mathbb{E}\big[\sum_j T_{\mathcal B_j}\big] - (m-1)\gamma$. Since $\gamma > 0$ and $m \ge 2$, the inequality is strict. ∎

`pv: queue-model-v1 / POST-T2`

*Remark.* Theorem 2 needs A2. If culprits cannot be identified, a red fold costs up to $\lceil \log_2 k\rceil$ extra bisection cycles. QM-01 measures the identification rate, and S-4 fires if it falls below 0.9.

*Scope (R-8).* The comparison is against **sequential** batches, where each batch is driven to green before the next is pushed. Pipelined pushes are not covered: under those, batch 2 is pushed while batch 1's fix is in flight. A5 also rules out interaction defects, where PRs $i$ and $j$ are each clean but break together. Those defects are the main practical argument for splitting a fold. If QM-07 finds interaction defects among red release cycles, A5 is falsified and the fold policy is re-derived.

### Corollary 3 (break-even for retries on the release PR)
Going from $r = 0$ to $r = 1$ lowers $\mathbb{E}[T_{\text{fold}}]$ if and only if
$$\boxed{\;\frac{\varphi\,(C-\rho)}{1-\varphi^{2}} \;>\; \rho\,\mathbb{E}[M]\;}$$

*Proof.* From Theorem 1, $\mathbb{E}[T(k,1)] - \mathbb{E}[T(k,0)] = \rho\,\mathbb{E}[M] + \dfrac{C + \rho\varphi}{1-\varphi^2} - \dfrac{C}{1-\varphi}$. Write $\dfrac{C}{1-\varphi} = \dfrac{C(1+\varphi)}{1-\varphi^2}$. The difference becomes $\rho\,\mathbb{E}[M] - \dfrac{\varphi(C-\rho)}{1-\varphi^2}$, which is negative if and only if the stated inequality holds. ∎

`pv: queue-model-v1 / POST-C3`

For $r \ge 2$ the xtask evaluates Theorem 1 exactly for every $r \in \{0,\dots,3\}$ and picks the argmin. No monotonicity claim is needed.

### Corollary 4 (cycle time is the dominant lever)
$\mathbb{E}[T_{\text{fold}}]$ is affine in $C$ with slope $\mathbb{E}[M] + 1/(1-\varphi^{r+1}) \ge 1$. Every minute cut from the cycle saves at least one minute per release, and $(1+\mathbb{E}[M])$ minutes in expectation when $\varphi$ is small. The cycle also enters Corollary 3's break-even: shorter cycles make retries worth even less.

*Proof.* Differentiate Theorem 1 with respect to $C$. ∎

`pv: queue-model-v1 / POST-C4`

*Scope (R-9).* Corollary 4 is a partial derivative: it holds $F$, $q$ and $\varphi$ fixed. Sharding also changes those inputs. A shorter feedback loop plausibly lowers $F$, and more jobs can raise $\varphi$. The −58% in §7 is therefore the effect of $C$ alone at `[A]` inputs. QM-06 records $F$ and $\varphi$ before and after sharding, together with $C$.

---

## §5 Merge queue

### Lemma 5 (the bisection tree)
*Scope.* Lemma 5 and Theorem 6 hold for $B = 2^h$ with **full groups** (the backlog holds at least $B$ entries when the group forms). A partial group, or a $B$ that is not a power of two, gives an unbalanced tree that these closed forms do not cover.

Under Q2–Q3, the builds triggered by one full group of size $B = 2^h$ are the tested nodes of a complete binary tree over its entries. A node at depth $d$ covers $B/2^d$ entries. Let $E_t$, $E_f$ and $E_{ff}$ be the expected number of tested builds, failing builds, and failing builds that immediately follow a failing build at the head. With per-entry defect rate $q'$:
$$E_t(B) = 1 + \sum_{d=1}^{h} 2^{d}\Big[1-(1-q')^{B/2^{d-1}}\Big], \qquad E_f(B) = \sum_{d=0}^{h} 2^{d}\Big[1-(1-q')^{B/2^{d}}\Big],$$
$$E_{ff}(B) \;\ge\; \sum_{d=0}^{h-1} 2^{d}\Big[1-(1-q')^{B/2^{d+1}}\Big].$$

*Proof.*

- **Tested builds.** The root is always tested. A non-root node is tested if and only if its parent failed (Q3), and by Q2 the parent fails if and only if it contains a defect. A parent at depth $d-1$ covers $B/2^{d-1}$ entries, so it contains a defect with probability $1 - (1-q')^{B/2^{d-1}}$. Summing indicators over the $2^d$ nodes at each depth (linearity of expectation) gives $E_t$.
- **Failing builds.** A node fails if and only if it contains a defect. A defect in a node is also in its parent, so a failing node is always tested. Summing over nodes gives $E_f$.
- **Consecutive failures.** When an internal node fails at the head, Q3 cancels everything behind it and puts its left child first, so the next resolution at the head is the left child. The left child fails if and only if it contains a defect. That event implies the parent failed, so its probability is $1-(1-q')^{B/2^{d+1}}$. This counts only parent → left-child pairs. Other consecutive failures (such as a leaf failure followed by a failing sibling subtree) only add to the count, so the sum is a lower bound. ∎

`pv: queue-model-v1 / POST-L5`

### Theorem 6 (group size: B = 1 minimises failures per entry)
The expected failing builds per entry have the closed form
$$\boxed{\;\frac{E_f(B)}{B} \;=\; \sum_{j=0}^{h}\frac{1-(1-q')^{2^{j}}}{2^{j}}\;}$$
It is strictly increasing in $h$ for every $q' \in (0,1)$ and equals $q'$ at $B = 1$. The **lower bound** on $E_{ff}(B)/B$, which is $\underline{E}_{ff}(B)/B = \sum_{j=0}^{h-1} \big(1-(1-q')^{2^j}\big)/2^{j+1}$, is also strictly increasing in $h$ and is 0 at $B = 1$. Lemma 5 bounds $E_{ff}$ only from below, so this is **not** a claim that $E_{ff}(B)/B$ itself is monotone (R-2). Theorem 8 uses only the bound.

*Proof.* Substitute $j = h - d$ in $E_f$. Then $2^d/B = 2^{-j}$ and $B/2^d = 2^j$. Going from $h$ to $h+1$ adds the term $j = h+1$, which is strictly positive when $0 < q' < 1$. For $E_{ff}$, substitute $j = h - d - 1$, so that $2^d/B = 2^{-(j+1)}$ and $B/2^{d+1} = 2^j$. The monotonicity argument is the same. ∎

`pv: queue-model-v1 / POST-T6` (the $E_f$ closed form and the monotonicity of $\underline{E}_{ff}$; nothing about $E_{ff}$ itself)

*Remark (robust to unknown GitHub semantics).* If GitHub ejects a failed group whole instead of bisecting it (G10), every good entry in a failed group is falsely ejected. Per entry, the expected number of false ejections is $(1-q')\big[1-(1-q')^{B-1}\big]$: the entry is good, and at least one of the other $B-1$ entries is defective. This is also increasing in $B$ and is 0 at $B = 1$. **At $B = 1$ the two semantics coincide,** so the §7 policy doesn't depend on QM-02.

### Lemma 7 (head-of-line windows)
Let $t_1 < t_2 < \dots$ be the times of failing resolutions at the head, and let $W_k$ run from $t_k$ to the next resolution. The windows are pairwise disjoint, no entry merges inside any of them, and
$$|W_k| \;\ge\; T + r\rho\cdot \mathbf 1[\text{the next resolution is a failure}].$$

*Proof.* At $t_k$ every build behind the head is cancelled (Q3), so the next head build starts at $t_k$ or later. It lasts at least $T$, and exactly $T + r\rho$ if it fails (Q2). Nothing merges before a head resolution, because merges happen in order. The windows lie between consecutive resolutions, so they are disjoint. ∎

`pv: queue-model-v1 / POST-L7`

### Theorem 8 (necessary condition for a stable queue)
Define the head-of-line load
$$\boxed{\;\rho_{\text{HOL}}(B,r) \;=\; \lambda'\,\frac{T\,E_f(B) + r\rho\,\underline{E}_{ff}(B)}{B}\;}$$
**Assumptions.** (i) Q1–Q5. (ii) Full groups are i.i.d.: successive full groups draw their defects independently with rate $q'$ (A5 applied per entry, and groups do not share entries). (iii) The queue state is a Markov chain on (backlog, in-flight builds). (iv) Per-window arrivals have finite variance, which holds for Poisson arrivals.

**Claim.** If $\rho_{\text{HOL}} \ge 1$, the queue is **not positive recurrent**: it has no stationary distribution with finite mean backlog.

*Proof.*

1. Suppose the backlog holds at least $LB$ entries. Then groups are full-sized.
2. By Lemma 7, the head spends disjoint windows of at least $T$ after every failure, plus $r\rho$ after every failure that follows a failure, with no departures.
3. By Lemma 5 and renewal-reward over i.i.d. full groups, the long-run head time per departing entry is at least $h(B) = [T\,E_f(B) + r\rho\,E_{ff}(B)]/B$. The departure rate is therefore at most $1/h(B)$.
4. Arrivals come at rate $\lambda'$ (Q4). If $\lambda' h(B) \ge 1$, the expected drift of the backlog is non-negative whenever the backlog is at least $LB$.
5. A chain whose drift is non-negative outside a finite set, and whose increments have bounded second moments (assumption iv), is not positive recurrent. This is the converse drift criterion (Tweedie 1976; Meyn & Tweedie, ch. 11). Pakes' lemma and Foster–Lyapunov run the other way (negative drift ⇒ positive recurrence) and do not give this claim (R-4). ∎

*What the claim does not say.* At $\rho_{\text{HOL}} = 1$ exactly the drift is zero. The chain can then be null recurrent, which is still unstable in the operational sense (unbounded mean wait), but "the backlog grows without bound" is only claimed for $\rho_{\text{HOL}} > 1$.

`pv: queue-model-v1 / POST-T8`

*Tightness.* $h(B)$ leaves out three costs: passing resolutions after a pass, cancelled speculative work, and refilling the pipeline. So $\rho_{\text{HOL}}$ is a **lower bound** on the real load, and the simulation (§6.3) shows queues degrading well before $\rho_{\text{HOL}} = 1$. The operating threshold is therefore set empirically in §7.

### Proposition 9 (lane capacity, and when B > 1 is justified)
Every tested build takes at least $T$ lane-minutes, and every failing build $T + r\rho$. A necessary condition for stability is
$$\rho_{\text{lane}}(B,r) \;=\; \lambda'\,\frac{T\,E_t(B) + r\rho\,E_f(B)}{L\,B} \;<\; 1.$$
At $B = 1$ this reduces to $\lambda'(T + r\rho\,q') < L$. **So $B > 1$ can be necessary only when $\lambda' \ge L/(T + r\rho q')$**, which is about 34 entries per hour at $L = 8$, $T = 14$, $r = 0$.

*Proof.* The lanes supply $L$ lane-minutes per minute, and each entry consumes $[T\,E_t(B) + r\rho\,E_f(B)]/B$ lane-minutes in expectation (Lemma 5). The same drift argument as Theorem 8 applies. For $B = 1$, $E_t = 1$ and $E_f = q'$. ∎

`pv: queue-model-v1 / POST-P9` · same assumptions as Theorem 8. Speculative builds cancelled by a head failure (up to $L-1$ per failure) also consume lanes. $\rho_{\text{lane}}$ omits them, which is why it is only a necessary condition.

By Theorem 6, $B = 1$ minimises the head-of-line bound. By Proposition 9 it is feasible for lanes whenever $\lambda' < L/(T + r\rho q')$. So **$B^\* = 1$ whenever the lane condition holds with margin**, and otherwise $B^\*$ is the smallest $B$ satisfying both conditions.

### Proposition 10 (retries in the merge queue at B = 1)
Measure the cost of an entry as the head time it adds (Lemma 7) plus $R$ for each false ejection. Going from $r = 0$ to $r = 1$ lowers the expected cost per entry if and only if
$$\boxed{\;(1-q')\,\varphi\,\big[(1-\varphi)(T+R) - \rho\big] \;>\; q'\rho\;}$$

*Proof.* The expected per-entry cost is as follows.

- **At $r = 0$:** $q' T$ from real failures (a restart window), plus $(1-q')\varphi\,(T + R)$ from false ejections (a restart plus the re-arm delay).
- **At $r = 1$:** $q'(T + \rho)$ from real failures, which now retry and fail again. Every good entry whose first attempt flakes (probability $\varphi$) pays the retry $\rho$, giving $(1-q')\varphi\rho$. Those whose retry also flakes (probability $\varphi^2$) are still false ejections and pay the restart plus re-arm, giving $(1-q')\varphi^2(T + R)$. Their $\rho$ is already in the previous term.

Subtracting the $r = 0$ cost from the $r = 1$ cost gives $q'\rho + (1-q')\varphi\big[\rho + \varphi(T+R) - (T+R)\big] = q'\rho - (1-q')\varphi\big[(1-\varphi)(T+R) - \rho\big]$. This is negative if and only if the stated inequality holds. ∎

`pv: queue-model-v1 / POST-P10`

*Correction (v1.1, R-1).* v1.0 charged double flakes $(1-q')\varphi^2(T+\rho+R)$ on top of $(1-q')\varphi\rho$ for every first flake, which counted the retry's $\rho$ twice on the double-flake path. At $q' = 1/9$, $T = 14$, $R = 15$, $\rho = 8$, the break-even moves from $\varphi^\* = 0.0525$ (v1.0) to $\varphi^\* = 0.0512$, the smaller root of $29\varphi^2 - 21\varphi + 1 = 0$. The §7 decision ($r = 0$) does not change.

### §5.1 Service classes: a cheap lane and a full lane (v1.1, operator question 2026-09-27)

*Motivation.* aprender#4462: fork PRs have no self-service path to PR review. aprender#4467, a docs-only PR, paid a full 3/3 quorum plus full CI. Both are one question: should every PR pay the same service, whatever its measured defect rate?

**Classes.** Every entry belongs to exactly one class $c$. The class is decided by a mechanical predicate, never by the author's own claim.

- **$d$, docs-only.** The diff touches no path that any build, test or doctest reads. That rules out any file reached by `include_str!`, `include_bytes!` or `#[doc = include_str!(…)]`, anything a `build.rs` reads, and any mdBook chapter that is tested. A README pulled in as crate docs is therefore **full**, not docs.
- **$a$, maintainer-attested fork.** A fork PR on which a maintainer has recorded an attestation, for example a label applied by an account in the owners list.
- **$x$, full.** Every other PR.

Each class has an arrival share $\pi_c$ (so $\sum_c \pi_c = 1$), a per-PR defect rate $q_c$ with per-entry rate $q'_c$ (Q4 applied per class), a `merge_group` build time $T_c$, and a review cost $Q_c$ in minutes. A cheap lane serves a class with a smaller build ($T_c < T$) and a smaller review ($Q_c < Q_x$). The full lane is the class-blind service in use today.

**Assumption C1 (class does not change the build that follows).** A cheap build applies only when the entry's `merge_group` base contains no un-merged full-class entry. Under GitHub's cumulative per-entry builds (G10, R-3), a docs entry queued behind a code entry is building code, so it is served as full. QM-08 measures the fraction of cheap entries that actually receive a cheap build and records it as $\pi_c^{\text{eff}} \le \pi_c$. The formulas below use $\pi_c^{\text{eff}}$.

**Proposition 11 (head-of-line load with classes, $B = 1$, $r = 0$).** Suppose classes are assigned i.i.d. per entry and the restart window after a failure is the build time of the entry behind the failure (Lemma 7). Then
$$\boxed{\;\rho_{\text{HOL}} \;=\; \lambda'\,\bar q'\,\bar T, \qquad \bar q' = \sum_c \pi_c^{\text{eff}} q'_c, \qquad \bar T = \sum_c \pi_c^{\text{eff}} T_c\;}$$
with $\rho_{\text{lane}} = \lambda' \bar T / L$.

*Proof.* At $B = 1$ and $r = 0$, a failure happens at the head with rate $\bar q'$ per entry. After it, the entry behind rebuilds from scratch. That entry's class is independent of the failed entry's class, so the expected restart window is $\bar T$. Renewal-reward over entries gives $\lambda' \bar q' \bar T$. The lane statement is Prop. 9 with $E_t = 1$ and service time $\bar T$. ∎

`pv: queue-model-v1 / POST-P11`

*Consequence.* Routing entries does not change any defect rate, so $\bar q'$ is the same with or without a cheap lane. The cheap lane changes only the restart window:
$$\frac{\rho_{\text{HOL}}^{\text{split}}}{\rho_{\text{HOL}}^{\text{blind}}} \;=\; \frac{\bar T}{T} \;=\; 1 - \sum_{c \ne x} \pi_c^{\text{eff}}\Big(1 - \frac{T_c}{T}\Big).$$
Illustration, all inputs `[A]`: with $\pi_d^{\text{eff}} = 0.3$, $T_d = 3$ and $T = 14$, the ratio is $10.7/14 = 0.764$. That lowers $\rho_{\text{HOL}}$ and $\rho_{\text{lane}}$ by 24%. At $r > 0$, the $r\rho$ term of Theorem 8 splits by class in the same way.

**Proposition 12 (when a class may skip the full quorum).** The full quorum detects a defect in a class-$c$ PR with probability $\kappa_x$, and the cheap review with probability $\kappa_c$. A defect that escapes review costs $D$ minutes downstream: finding it, reverting and re-landing. Skipping the full quorum for class $c$ lowers the expected cost per PR if and only if
$$\boxed{\;q_c \;<\; q_c^\* \;=\; \frac{Q_x - Q_c}{(\kappa_x - \kappa_c)\,D}\;}$$

*Proof.* Moving from the full quorum to the cheap review saves $Q_x - Q_c$ per PR. It adds $q_c(\kappa_x - \kappa_c)D$ in expected escape cost. The move lowers the total cost exactly when the saving is larger than the added cost. ∎

`pv: queue-model-v1 / POST-P12`

Illustration, all inputs `[A]`: with $Q_x - Q_c = 30$ min, $\kappa_x - \kappa_c = 0.5$ and $D = 240$ min, $q_d^\* = 0.25$. The threshold is per class. A maintainer attestation earns a fork class the cheap lane only through a measured $q_a < q_a^\*$, never through the attestation alone. A class whose $n$ is too small to put the upper end of a 95% interval on $q_c$ below $q_c^\*$ stays in the full lane (S-2: an unmeasured input never decides).

*Scope.* Prop. 12 prices review only. The cheap lane's build is a separate decision, governed by C1 and Prop. 11. Neither proposition authorises a gate change: the §7 row stays "no change" until QM-08 has receipted $q_c$, $T_c$ and $\pi_c^{\text{eff}}$ and the operator gives a go.

---

## §6 Validation

### §6.1 Closed forms against Monte Carlo (analysis run 2026-09-27, 200k trials each)

| Quantity | Parameters | Formula | Monte Carlo |
|---|---|---|---|
| $E_t, E_f$ (Lemma 5) | $B=8,\ q'=0.1$ | 5.0347, 2.8173 | 5.0371, 2.8187 |
| $E_t, E_f$ | $B=4,\ q'=0.2$ | 3.6208, 2.1104 | 3.6284, 2.1160 |
| $E_{ff}$ lower bound | $B=8,\ q'=0.1$ | 1.1239 (bound) | 1.2355 (all consecutive failures) |
| $E_f/B$ closed form (Thm 6) | $B=8,\ q'=0.1$ | 0.3522 | = $E_f/8$ |
| $\mathbb{E}[T_{\text{fold}}]$ (Thm 1) | $k=5,q=.1,f=.8,C=80,F=20,\varphi=.02,r=0,\rho=10$ | 134.67 | 134.44 |
| same | same, $r = 2$ | 143.86 | 143.60 |
| same | $k=5,q=.2,f=.8,C=30,F=20,\varphi=.1,r=1,\rho=10$ | 85.69 | 85.78 |
| same | $k=8,q=.1,f=.8,C=80,F=20,\varphi=.05,r=1,\rho=10$ | 164.12 | 164.20 |
| $\varphi^\*_{\text{MQ}}$ (Prop. 10, v1.1) | $q'=1/9,T=14,R=15,\rho=8$ | 0.0512 | — |

*Provenance (R-5, R-6).* Every row now states every parameter, so no value depends on a reader inheriting one from the row above. The formula column was recomputed independently on 2026-09-27 from the closed forms, and every value in it agrees to 4 significant figures `[C]`. The Monte Carlo column came from the analysis outside the repo (§0.5) and has no in-repo generator, so it is `[A]` until QM-04 reproduces it. **The prose is not the oracle:** QM-00's golden tests must agree with QM-04's simulator on every row (§9). A golden test that pins these numbers with no independent generator would enforce an error in the prose as readily as a correct value, and v1.0's Prop. 10 was such an error.

### §6.2 Release-PR numbers (inputs `[A]` until QM-01)

With $k=5$, $q=0.1$, $f=0.8$, $F=20$, $\rho_{\text{rel}}=10$, $\varphi=0.02$:

| Cycle $C$ | $\mathbb{E}[M]$ | One fold (Thm 1) | Two pushes (2+3) | Retry break-even $\varphi^\*$ (Cor. 3) |
|---|---|---|---|---|
| 80 min | 0.530 | **134.7 min** | 221.7 min | 0.075 |
| 30 min (sharded) | 0.530 | **57.1 min** | 90.4 min | 0.249 |

### §6.3 Merge-queue loads (inputs `[A]`: $T = 14$, $\rho_{\text{MQ}} = 8$, $f = 0.8$, $L = 8$) against simulation

The simulation is a discrete-event model of Q1–Q4 with flakes, re-arm and fix re-entry. It ran 10 × 5-day runs per configuration from the 19-PR backlog. Waits are in minutes, and p90 includes defective PRs' fix loops. Rows with $B > 1$ assume the Q1–Q3 bisecting-group semantics, which is `[U]` against GitHub (R-3). They are model rows. "(current)" means the ruleset values at v1.0, not a claim that GitHub behaves as Q1–Q3 describe.

| λ/h, $q$ | $B$, $r$ | $\rho_{\text{HOL}}$ | $\rho_{\text{lane}}$ | Sim p50 / p90 |
|---|---|---|---|---|
| 6, 0.10 | **1, 0** | **0.175** | 0.197 | **14 / 118** |
| 6, 0.10 | 8, 0 | 0.608 | 0.133 | 14 / 118 |
| 6, 0.10 | 8, 2 (current) | **0.887** | 0.220 | **37 / 1528** |
| 6, 0.20 | **1, 0** | **0.350** | 0.219 | **17 / 129** |
| 6, 0.20 | 2, 0 | 0.665 | 0.188 | 19 / 134 |
| 6, 0.20 | 8, 0 | 1.105 | 0.216 | 30 / 1258 |
| 6, 0.20 | 8, 2 (current) | 1.633 | 0.374 | 2302 / 4874 |
| 10, 0.10 | **1, 0** | **0.292** | 0.328 | **14 / 82** |
| 10, 0.10 | 8, 0 | 1.014 | 0.222 | 24 / 651 |
| 10, 0.10 | 8, 2 (current) | 1.479 | 0.366 | 2299 / 4306 |

- **Knee `[C]`:** every configuration with $\rho_{\text{HOL}} \le 0.61$ stayed at p90 ≤ 134 min. Every configuration at $\ge 0.89$ degraded by more than 10×.
- **Retries at $B = 1$** (Prop. 10) with $q' = 0.111$, $T = 14$, $R = 15$, $\rho_{\text{MQ}} = 8$: the break-even is $\varphi^\* = 0.0512$ (v1.1; v1.0 said 0.053, see R-1). The simulation agrees: at $\varphi = 0.05$, $r = 0$ gave p90 123 min against 160 min for $r = 2$.

---

## §7 Policy (the decisions)

| Decision | Value | Basis | Holds until |
|---|---|---|---|
| Merge-queue group size | **$B = 1$** (ruleset max group size 1). Operator ruling 2026-09-27: "queue group 1"; applied by QM-02 (aprender#4515) after rc.1, per S-1 | Thm 6; Remark (robust to G10) | measured $\lambda' \ge 0.8\,L/T$ (Prop. 9) |
| Merge-queue lanes | **$L = 8$** (keep) | speculation keeps $\rho_{\text{lane}} \le 0.33$ at 10/h | $\rho_{\text{lane}} > 0.6$ measured |
| Merge-queue retries | **$r = 0$ now** | Prop. 10 at $B = 1$ ($\varphi^\*_{\text{MQ}} = 0.0512$ at `[A]` inputs) and the §6.3 $B = 1$ simulation line. Theorem 8's 0.89 is the $B = 8$, $r = 2$ row and is context, not the basis (R-10) | measured $\varphi > \varphi^\*_{\text{MQ}}$ |
| Release-PR fold | **one fold per push** | Thm 2 (strict dominance) | identification rate < 0.9 (S-4) |
| Release-PR retries | **$r = 0$** once a2's disable list lands | Cor. 3 ($\varphi^\* = 0.075$ at $C = 80$; 0.25 at $C = 30$) | measured $\varphi > \varphi^\*$ |
| Cycle time | **shard CI to $C \le 30$** as the first post-rc investment | Cor. 4 (−58% expected time to green) | — |
| Queue health | GREEN $\rho_{\text{HOL}} \le 0.4$ · AMBER $\le 0.6$ · RED $> 0.6$ `[C]` | Thm 8 + §6.3 knee | recalibrated by QM-05 |
| Flake budget | $\varphi \le 0.01$ per run | keeps both break-evens far away; FLAKE-0 | — |
| Service classes (cheap vs full lane) | **no change now**: every PR pays the full quorum and full CI | Prop. 11 ($\rho_{\text{HOL}}$ scales by $\bar T/T$) and Prop. 12 ($q_c < q_c^\*$), both at `[A]` inputs | QM-08 receipts $q_c$, $T_c$, $\pi_c^{\text{eff}}$ with the 95% upper bound on $q_c$ below $q_c^\*$, **and** an operator go (a gate change) |

**Correction recorded.** The advice given on 2026-09-27 at 08:34 to keep 2 retries on the release PR until the publish is withdrawn. Corollary 3 shows retries lose unless $\varphi > 0.075$ at $C = 80$.

---

## §8 STOP conditions (stop, write the §10 report, do not work around)

- **S-1** A row would change CI or ruleset settings on the release PR while a cut is in progress.
- **S-2** A decision would be taken from a `[U]` input when §7 doesn't hold a value for it.
- **S-3** Measured $\rho_{\text{HOL}} > 0.6$ under the §7 policy. That means the policy is failing, so re-derive the model; don't raise the threshold.
- **S-4** The measured identification rate of red release cycles (A2) is below 0.9. Theorem 2's premise is weakened, so re-evaluate the fold policy with the bisection term.
- **S-5** A harness hook refuses the session's writes for a reason other than a missing ticket.
- **S-6** Two consecutive non-passing quorums on the same PR.
- **S-7** Budget $K$ reached, or the andon crossed with more than one row incomplete.

---

## §9 Tickets (EV-ordered)

`K̂` in minutes `[A]`.

| EV | Row | Work | Contract | Done when (all must hold) | K̂ |
|---|---|---|---|---|---|
| 0 | **QM-00** model xtask | `cargo xtask queue-model`: Lemma 1, Thm 1, Cor. 3, Lemma 5, Thm 6, Thm 8, Prop. 9, Prop. 10 as pure functions. Reads a measured-inputs JSON; prints §7 decisions and $\rho_{\text{HOL}}$, $\rho_{\text{lane}}$ | `queue-model-v1` | **Depends on QM-04 (the oracle).** Every §6.1–§6.2 formula value is reproduced to 4 significant figures **and** agrees with QM-04's Monte Carlo within 3 standard errors on every row. The simulator is the oracle, not the prose (R-6). Planted mutants turn RED: drop the $d = 0$ term of $E_f$; replace $\max$ by $\sum$ in Lemma 1; use $1/(1-\varphi)$ for every $r$; use $\varphi$ instead of $\varphi(1-\varphi)$ for the rescued-flake probability in Prop. 10 (the v1.0 error, R-1) | 150 |
| 0 | **QM-01** measure inputs | Receipted 7-day measurement of $T$, $C$, $q$, $\varphi$, $\lambda$, $f$, $F$, $\rho$, $R$ and the identification rate (commands in §9.1) | `queue-inputs-v1` | every §2 input has a value, $n$, window and command; 0 `[U]` fields; an empty window is RED | 120 |
| 1 | **QM-02** settings (filed) | Issue against ruleset 17836320: max group size 1, build concurrency 8. PR: nextest `merge_group` profile retries = 0. Record the observed GitHub group-failure behaviour (G10) | `mq-settings-v1` | the ruleset export shows group size 1 / concurrency 8; the profile lint fails on a planted `retries = 2` in the merge-queue profile | 45 |
| 1 | **QM-03** release retries | After FLAKE-0's disable list lands and no cut is in progress: evaluate Cor. 3 at measured inputs; set release-PR retries to the argmin of Thm 1 over $r \in \{0,\dots,3\}$ | extends `queue-model-v1` | decision receipted with its inputs; setting matches the xtask output | 30 |
| 0 | **QM-04** queue simulator (QM-00's oracle) | `cargo xtask queue-sim`: Rust port of the §6.3 discrete-event model (Q1–Q5, flakes, re-arm). It also runs Monte Carlo of the release-PR fold (A1–A5) and of the bisection tree (Lemma 5), so every §6.1 row has an independent generator. Its merge-queue semantics follow the G10 behaviour QM-02 observes | `queue-sim-v1` | reproduces §6.3 p50 within ±10% (10 seeds); a planted "no cancellation on failure" mutant changes the current-config p90 by > 5× | 180 |
| 2 | **QM-05** queue-health gate | Nightly: compute $\rho_{\text{HOL}}$, $\rho_{\text{lane}}$ from the trailing 7-day inputs; RED > 0.6 opens one issue with inputs attached; recalibrate the knee from measured waits every 30 days | `queue-health-v1` | a planted input set with $B = 8$, $r = 2$, $q = 0.1$, 6/h turns RED; an empty input window turns RED | 90 |
| 3 | **QM-06** cycle-time link | Record measured $C$ before and after the CI-sharding work; evaluate Cor. 4 savings | — | before/after receipts with $n \ge 5$ cycles each, recording $C$, $F$ and $\varphi$ (Cor. 4 holds $F$ and $\varphi$ fixed; R-9) | 30 |
| 4 | **QM-07** fold identification | Measure the A2 identification rate on red release cycles; if < 0.9, extend Thm 1 with the bisection term and re-prove Thm 2's bound | extends `queue-model-v1` | rate receipted with $n$; S-4 evaluated | 60 |
| 2 | **QM-08** per-class measurement | Classify every PR in the window by the §5.1 predicate ($d$, $a$, $x$). Per class, measure $\pi_c$, $\pi_c^{\text{eff}}$ (C1), $q_c$ and $q'_c$ (the §9.1 $q$ command split by class), $T_c$, the review cost $Q_c$, and the escape cost $D$ from revert timelines. Evaluate Props. 11 and 12 | extends `queue-inputs-v1` (`classes`) | every class has $n$, window and command; the classifier sends a planted docs PR that edits an `include_str!` target to class $x$; a class with too little $n$ is reported as not decided, never as passing | 60 |

**K̂ = 765 `[A]` · K = 800 · andon at 640.**

### §9.1 Measurement commands (QM-01)

| Input | Command (7-day window) |
|---|---|
| $T$ | `gh run list -R paiml/aprender --event merge_group --status completed --limit 500 --json createdAt,updatedAt,conclusion` → duration of successful runs, median |
| $q$ | Same list, counting only runs whose `conclusion` is success or failure; a cancelled run is neither. After removing FLAKE-0 flakes, failures ÷ (successes + failures) measures the **per-entry** rate $q'$, which includes re-entries. Record it as `q_eff` and back out $q = q'f/(1-q')$ (Q4; R-12) |
| $\varphi$ | FLAKE-0 ledger: flaky failures ÷ runs |
| $\lambda$ | `gh pr list -R paiml/aprender --state all --limit 1000 --json createdAt` → PR creations per hour, which is $\lambda$. Queue-entry events, when available, measure $\lambda'$, which includes re-entries. Record which one was measured, and never store both under one key (R-12) |
| $C$ | release-PR push → both review and `ci / gate` green, per cycle |
| $f$, $F$ | ejected-PR timelines: ejection → re-entry; the re-entry passes (success) or fails again |
| $\rho_{\text{rel}}$, $\rho_{\text{MQ}}$ | nextest retry durations from job logs, split by event into the release PR (`pull_request`) and `merge_group` (R-11) |
| $R$ | false ejection → re-arm timestamps |
| Identification rate | red release cycles whose failing tests name the culprit PR ÷ red release cycles |

---

## §10 Final report schema (one per session)

```yaml
spec: FLOW-003
session: {date_utc, host, tree: {head, origin_main, worktree}, model}
selected_row: QM-NN
ticket: PMAT-NNNN
outcome: merged | stopped | already-done | premise-falsified
stop: {id: S-n, evidence: "..."}
inputs: {T, C, q, q_eff, phi, lambda_per_h, lambda_eff_per_h, f, F, rho_rel, rho_mq, R, ident_rate, window_days, n_runs}  # each: {value, n, window, command, mark}
classes: {d: {pi, pi_eff, q, q_eff, T, Q}, a: {…}, x: {…}}  # §5.1; each field: {value, n, window, command, mark}
review: {kappa_full, kappa_cheap, D}                       # Prop. 12; [U] until QM-08
derived: {q_eff, lambda_eff, EM, E_T_fold_by_r: {0,1,2,3}, rho_hol, rho_lane, phi_star_release, phi_star_mq}
decisions: {B, L, r_mq, r_release, fold: one}
issues_filed: [{repo, url, purpose}]
scoreboard_moved: [{metric, before, after, command}]
budget: {k_hat_row, k_actual_row, cumulative, andon_crossed: bool}
next_row: QM-NN
```

### §10.1 Scoreboard (targets and probes only; state is rendered, never written here)

| Metric | Target | Rendered by |
|---|---|---|
| $\rho_{\text{HOL}}$ at measured inputs | **≤ 0.4** | QM-05 |
| $\rho_{\text{lane}}$ at measured inputs | **≤ 0.6** | QM-05 |
| Merge-queue wait p50 / p90 | **≤ 15 / ≤ 120 min** | QM-04 vs measured |
| Release-PR pushes per green head | **≤ 1.5** | QM-01 |
| Release cycle $C$ | **≤ 30 min** | QM-06 |
| Flake rate $\varphi$ | **≤ 0.01** | FLAKE-0 |
| False ejections | **≤ 1 per day** | QM-01 |
| Inputs marked `[U]` | **0** | QM-01 |
| Python lines in files this spec touches | **0** | `grep -rl python3` over the diff |

---

## §11 Ontology: the contracts that hold this spec (`pv`, ONT-001 v1alpha1)

These are the YAML `pv` contracts that QM-00, QM-01, QM-04 and QM-05 land under `contracts/`. The ids match the Contract column of §9. They are written here so that each row's done-when is a contract, not prose. Every contract stays `proof.status: declared` until its row lands, so `pv` reports `Unknown`, which never arms. Nothing here claims `discharged`.

### §11.1 The spec itself (`entity: spec`, ONT-001 B.11)

```yaml
id: DOC-spec-flow-003
metadata: { ontology_version: ont.paiml.dev/v1alpha1 }
entity: { type: spec, ref: docs/specifications/FLOW-003-queue-and-release-cycle-model.md }
shape:
  closed: true
  properties:
    - { path: spec:id,         minCount: 1, maxCount: 1, pattern: "^FLOW-003$" }
    - { path: spec:row,        minCount: 8, pattern: "^QM-0[0-7]$" }
    - { path: spec:stop,       minCount: 7, pattern: "^S-[1-7]$" }
    - { path: spec:falsifier,  minCount: 1 }                       # every row's done-when names a RED case
    - { path: spec:contract,   minCount: 5, resolves: pv-contract } # §9 Contract column → a file under contracts/
proof: { status: not_applicable }
evidence: { level: L1, provenance: { wasGeneratedBy: { command: "pv extract docs/specifications/FLOW-003-queue-and-release-cycle-model.md", git_sha: "<census sha>" }, wasAttributedTo: orchestrator, generatedAtTime: "<from git>" } }
# spec extractor is `declared` in ONT-001 (Q12): validates today as Unknown{ExtractorMissing: spec}.
```

### §11.2 The model (`queue-model-v1`, bound to QM-00's code)

Each §3–§5 result becomes a clause. `formal:` is given where the statement is a closed form, and `prose` otherwise. The proof obligation is the one this spec already carries: the written proof, plus a golden test agreeing with QM-04's simulator.

```yaml
id: queue-model-v1
metadata: { kind: kernel, ontology_version: ont.paiml.dev/v1alpha1 }
entity: { type: code, ref: xtask::queue_model }        # resolves once QM-00 lands; Unknown{SymbolMissing} until then
relations: { depends_on: [queue-sim-v1] }              # the oracle (R-6)
requires:
  - { id: PRE-1, statement: "probabilities in the open unit interval", formal: "0 < q < 1 && 0 < f <= 1 && 0 <= phi < 1", formal_status: parsed }
  - { id: PRE-2, statement: "times non-negative, C > 0", formal: "C > 0 && T > 0 && F >= 0 && rho >= 0 && R >= 0", formal_status: parsed }
  - { id: PRE-3, statement: "group size is a power of two with full groups (Lemma 5 scope)", formal: "exists h: Nat. B = 2^h", formal_status: parsed }
  - { id: PRE-4, statement: "Q5: false re-entry negligible, else lambda_eff carries the term", formal_status: prose }
ensures:
  - { id: DEF-Q4,   statement: "effective arrival and defect rate", formal: "lambda_eff = lambda*(1+q/f) && q_eff = q/(f+q)", formal_status: parsed }
  - { id: POST-L1,  statement: "expected defect cycles", formal: "EM(k) = sum_{n>=0} 1 - (1 - q*(1-f)^n)^k", formal_status: parsed }
  - { id: POST-T1,  statement: "expected time to green", formal: "ET(k,r) = EM(k)*(C + r*rho + F) + (C + rho*S_r)/(1 - phi^(r+1))", formal_status: parsed }
  - { id: POST-T2,  statement: "one fold strictly dominates any SEQUENTIAL split (m >= 2)", formal_status: prose }
  - { id: POST-C3,  statement: "r=1 beats r=0 iff phi*(C-rho)/(1-phi^2) > rho*EM", formal_status: parsed }
  - { id: POST-C4,  statement: "dET/dC = EM + 1/(1-phi^(r+1)) >= 1, holding F, q, phi fixed", formal_status: parsed }
  - { id: POST-L5,  statement: "E_t, E_f closed forms; E_ff >= lower bound", formal_status: parsed }
  - { id: POST-T6,  statement: "E_f(B)/B and the E_ff LOWER BOUND /B strictly increase in h; nothing claimed for E_ff itself (R-2)", formal_status: parsed }
  - { id: POST-L7,  statement: "head-of-line windows disjoint, no merge inside, |W| >= T + r*rho*[next fails]", formal_status: prose }
  - { id: POST-T8,  statement: "rho_hol >= 1 => not positive recurrent, under i.i.d. full groups and finite-variance arrivals (R-4)", formal_status: prose }
  - { id: POST-P9,  statement: "rho_lane < 1 necessary; at B=1 reduces to lambda_eff*(T + r*rho*q_eff) < L", formal_status: parsed }
  - { id: POST-P10, statement: "at B=1, r=1 beats r=0 iff (1-q_eff)*phi*((1-phi)*(T+R) - rho) > q_eff*rho (v1.1, R-1)", formal_status: parsed }
  - { id: POST-P11, statement: "classes at B=1, r=0: rho_hol = lambda_eff * sum(pi_eff*q_eff_c) * sum(pi_eff*T_c)", formal_status: parsed }
  - { id: POST-P12, statement: "class c may skip the full quorum iff q_c < (Q_x - Q_c) / ((kappa_x - kappa_c) * D)", formal_status: parsed }
invariants:
  - { id: INV-1, statement: "every §7 decision is a pure function of the inputs JSON; no decision reads a [U] input without a §7 default (S-2)", formal_status: prose }
shape:
  properties:
    - { path: ont:binds,      minCount: 1, resolves: symbol }
    - { path: ont:falsifier,  minCount: 4 }            # the four planted mutants of QM-00
proof:
  status: declared
  tests: { golden: "§6.1–§6.2 rows, 4 s.f.", oracle: queue-sim-v1, tolerance: "3 standard errors" }
evidence: { level: L1, provenance: { wasGeneratedBy: { command: "cargo xtask queue-model --self-test", git_sha: "<QM-00 merge sha>" }, wasAttributedTo: "lane:aprender", generatedAtTime: "<from git>" } }
```

### §11.3 The inputs (`queue-inputs-v1`, QM-01's receipt; `entity: json`)

```yaml
id: queue-inputs-v1
entity: { type: json, ref: docs/receipts/flow-003/queue-inputs.json }
shape:
  closed: true
  properties:
    # one node per §10 `inputs` key; each node carries value, n, window, command, mark
    - { path: qin:input,        minCount: 15, maxCount: 15 }
    - { path: qin:value,        minCount: 1, datatype: xsd:decimal }
    - { path: qin:n,            minCount: 1, datatype: xsd:integer, minInclusive: 1 }      # an empty window is RED
    - { path: qin:command,      minCount: 1, resolves: ci-step }
    - { path: qin:mark,         in: ["[V]", "[C]", "[A]"] }                                 # "[U]" is not in the set: 0 unmeasured (QM-01)
    - { path: qin:measuredRate, in: [q, q_eff, lambda, lambda_eff] }                        # which rate was measured (R-12)
proof: { status: not_applicable }
evidence: { level: L2, provenance: { wasGeneratedBy: { command: "<the §9.1 command per input>", git_sha: "<sha>" }, wasAttributedTo: "lane:aprender", generatedAtTime: "<window end>" } }
```

### §11.4 The policy (`POLICY-flow-003-queue`, unanchored; ONT-001 B.2)

```yaml
id: POLICY-flow-003-queue
# unanchored: a rule about the repository's process
relations: { depends_on: [queue-model-v1, queue-inputs-v1] }
requires:
  - { id: PRE-1, statement: "no row changes release-PR CI or the ruleset while a cut is in progress (S-1)", formal_status: prose }
ensures:
  - { id: POST-B,     statement: "merge-queue max group size = 1 (Thm 6; operator ruling 2026-09-27)", formal_status: prose }
  - { id: POST-L,     statement: "merge-queue build concurrency = 8 while rho_lane <= 0.6", formal_status: prose }
  - { id: POST-RMQ,   statement: "merge_group nextest retries = 0 while phi <= phi*_MQ (Prop. 10)", formal_status: prose }
  - { id: POST-FOLD,  statement: "one fold per release-PR push while ident_rate >= 0.9 (Thm 2, S-4)", formal_status: prose }
  - { id: POST-RREL,  statement: "release-PR retries = argmin_r ET(k,r), r in 0..3 (QM-03)", formal_status: prose }
  - { id: POST-HOL,   statement: "rho_hol GREEN <= 0.4, AMBER <= 0.6, RED > 0.6 [C] (QM-05)", formal_status: prose }
  - { id: POST-CLS,   statement: "no class leaves the full lane without QM-08 receipts and an operator go (§5.1)", formal_status: prose }
shape:
  closed: true
  properties:
    - { path: ont:falsifier, minCount: 2, resolves: ci-step }   # mq-settings-v1 profile lint; queue-health-v1 planted RED input set
proof: { status: not_applicable }
evidence: { level: L1, provenance: { wasGeneratedBy: { command: "cargo xtask queue-model --inputs docs/receipts/flow-003/queue-inputs.json", git_sha: "<sha>" }, wasAttributedTo: orchestrator, generatedAtTime: "<from git>" } }
```

`mq-settings-v1` (QM-02), `queue-sim-v1` (QM-04) and `queue-health-v1` (QM-05) follow the same pattern. Each is written by its own row, with its done-when as `ont:falsifier` and a `resolves: ci-step` to the fixture that must turn RED.

---

## §12 Review record (v1.0 → v1.1)

Every closed-form value in §6.1 was recomputed independently from the formulas, and all of them reproduce. The findings:

| # | Where | Finding | v1.1 change |
|---|---|---|---|
| R-1 | Prop. 10 | The retry cost $\rho$ was counted twice on the double-flake path: the rescued-flake term used $\varphi$ where it needed $\varphi(1-\varphi)$. $\varphi^\*_{\text{MQ}}$ changes from 0.0525 to **0.0512**. | Condition and proof corrected. Fourth planted mutant added to QM-00. The §7 decision is unchanged. |
| R-2 | Thm 6 | The $E_{ff}$ monotonicity was proved only for the lower bound. | Claim scoped to $\underline{E}_{ff}$. |
| R-3 | Q1–Q3, G10 | The bisecting-group model may not be how GitHub's merge queue works (per-entry cumulative builds). | Q1–Q3 marked `[U]`. §6.3 rows with $B>1$ labelled as model rows. QM-04 follows QM-02's observation. |
| R-4 | Thm 8 | Step 5 cited the negative-drift criterion (Pakes, Foster–Lyapunov) for its converse. At $\rho_{\text{HOL}}=1$ the "grows without bound" claim was too strong. | Restated as "≥ 1 ⇒ not positive recurrent", with the i.i.d.-full-group and finite-variance assumptions listed. |
| R-5 | §6.1 rows 3–4 | $f$, $F$, $\rho$ were omitted and had to be inherited from row 1. | Every parameter is stated. |
| R-6 | §6, QM-00 | The golden values had no in-repo generator, so the golden tests would have enforced errors in the prose. | QM-04 is QM-00's oracle (3 standard errors). The Monte Carlo column is marked `[A]` until QM-04 reproduces it. |
| R-7 | Q4 | False-ejection re-entries were left out of $\lambda'$. | Assumption Q5 added. |
| R-8 | Thm 2 | The theorem covers sequential splits only. A5 excludes interaction defects. | Title and scope corrected. QM-07 falsifies A5. |
| R-9 | Cor. 4 | The derivative holds $F$ and $\varphi$ fixed, but sharding moves both. | QM-06 records $C$, $F$ and $\varphi$. |
| R-10 | §7 | The MQ-retries row cited the $B=8$, $r=2$ load as its basis. | Basis changed to Prop. 10 at $B=1$. |
| R-11 | §2, §6 | $\rho$ held two values (10 and 8). | Split into $\rho_{\text{rel}}$ and $\rho_{\text{MQ}}$ in §2, §9.1 and §10. |
| R-12 | §9.1 | merge_group failed ÷ completed measures $q'$, not $q$, and counts cancelled runs. PR creations measure $\lambda$, not queue entries. | Measurement commands corrected. `q_eff` and `lambda_eff` added to the inputs schema. |
| A-1 | §5.1 (addition) | Operator question 2026-09-27, prompted by aprender#4462 (fork PRs have no self-service review path) and #4467 (a docs-only PR paid a full 3/3 quorum and full CI). | Service classes added: Prop. 11 (ρ_HOL scales by $\bar T/T$), Prop. 12 (quorum-skip threshold $q_c^\*$), assumption C1, QM-08, a §7 row that stays "no change" until QM-08 has receipts and the operator gives a go. |
