# FLOW-003 v2.1: CI flow model: queue, fleet hardware and tiered testing (equations, proofs, simulation, optimisation)

**Spec id:** `FLOW-003` · **Version:** 2.1 (2026-09-27) · **Rows:** `QM-00..QM-16` (the traffic cop mints one `pmat` ticket per row; single-minter rule applies)
**Target repo:** `paiml/aprender` (`docs/specifications/FLOW-003-queue-and-release-cycle-model.md`). Ruleset, runner and forjar changes are filed as issues and not done here (§0.6).
**Runner:** the aprender traffic cop (`aprender-traffic-cop-prompt.md`).
**Launch:** from `~/src/aprender`, run `Implement docs/specifications/FLOW-003-queue-and-release-cycle-model.md autonomously.`
**Related:** FLOW-001, FLOW-002, `release-flow-rules.md` (cop, 2026-09-27), APR-RELEASE-001 (rules 14, 16, 17), ruleset 17836320, FLAKE-0 (a2), CI sharding (a2/49/5d), #4462 (vouched-contributor path), #4513 (7-day input measurement), infra#670 (shared sccache).
**Status:** spec, not implemented.

**Changes from v1.0:**
- Part I: the three v1.1 fixes known to the author are applied: Prop. 10's double count, Theorem 8 restated as non-positive-recurrence with its assumptions listed, and Lemma 5 / Theorem 6 scoped. infra-5a applies its remaining v1.1 fixes on top of this version.
- Part II (fleet hardware) is new.
- Part III (tiered testing) is new.
- §7 policy, §9 tickets and the scoreboard are extended.

**Changes from v2.0 (v2.1, infra-5a; full record in §12):**
- Part I: the rest of the v1.1 review is applied: R-3 (Q1–Q3 are `[U]` against GitHub), R-4 (Theorem 8's citations and its $\underline E_{ff}$), R-5 (§6 rows fully parameterised), R-6 (QM-04 is QM-00's oracle), R-7 (Q5), R-8 (Theorem 2's scope), R-9 (Cor. 4), R-10 (§7 retries basis), R-11 ($\rho_{\text{rel}}$ vs $\rho_{\text{MQ}}$) and R-12 (the measurements behind $q'$ and $\lambda'$).
- Parts II–III: review findings R2-1..R2-14 are applied, and the v2.1 quorum added R2-15 (QM-00's mutant list restored). Two of them affect soundness: $\mathcal I$ goes stale between nightlies (R2-8), and directory listings and failed lookups are invisible to an `openat` trace (R2-9). A third, the doctests that nextest archives cannot run, can drop doctests without anyone noticing (R2-10). Prop. 18 measured the wrong escape rate (R2-12).
- New: Prop. 19 (what tiering does to the merge queue), §11 (the `pv` contracts) and §12 (the review record).
- Operator ruling 2026-09-27: the lighter review tier (Prop. 18, §7 row 12) is **approved** for docs-only PRs and vouched contributors. It is gated on the measured escape bound and reverts automatically on an escape.
- Operator amendment 2026-09-27: **rebase before arm** joins the author checklist (§7 row 14), and **PR age** becomes a model input. Prop. 20 gives $P(\text{conflict}) \approx 1 - e^{-\mu_D a}$. Short PRs keep the queue model's $q$ stationary. Group 1, retries 0, compile once and tiered tests are unchanged.

**Ontology.** This document is the entity `spec:FLOW-003` (ONT-001 B.11). A `pv: <contract> / <clause>` marker binds a result to a clause of a §11 contract. A clause earns credit only when its proof or falsifier is **discharged**; until then `pv` reports `Unknown`, and `Unknown` never arms.

**Provenance marks:** `[V]` verified at the cited time · `[C]` computed · `[A]` asserted · `[U]` unmeasured · `[X]` third-party result, cited but not proved here.

---

## §0 Operating assumptions

1. **Purpose.** Replace judgement calls about CI flow with equations whose inputs are measured. There are three layers:
   - **Part I, the queue:** merge-queue group size and retries, and release-PR fold size.
   - **Part II, the hardware:** compile width, job shape, runner shapes and capacity.
   - **Part III, the tests:** which PR runs which tier, and in what order.
2. **Every decision in §7 is a theorem or proposition evaluated at measured inputs.** When an input is still `[U]`, the decision holds the §7 value, which rests on results that hold for all inputs (Theorems 2, 6, 11, 13, 14 and 15).
3. **Proofs are part of the spec.** A change to an assumption or a result is a new spec version. The xtask (QM-00) implements every closed form, and golden tests pin every §6 value.
4. **Simulation checks the model; it never overrides a proof.** Where a proof gives only a bound (Theorems 8 and 12), the simulation sets the operating threshold. Such thresholds are marked `[C]` and recalibrated from measured data.
5. **No Python in the repo.** The analysis that produced §6 and §H.6 was done outside the repo. In the repo, the model is `cargo xtask queue-model`, the fleet LP is `cargo xtask fleet-lp` (`good_lp` + HiGHS), and the simulators are `cargo xtask queue-sim` and `cargo xtask fleet-sim`.
6. **Cross-repo work is filed, not done.** Ruleset 17836320, forjar runner declarations, cgroup slices and the shared cache are filed as infra issues with checkable acceptance criteria.
7. **The release train wins.** No row changes CI or runner settings while a cut is in progress (S-1).
8. **Tiers are chosen by what a PR changes, never by who wrote it (R-T1).** Who wrote it may affect only the review tier, and a review-tier change needs an operator ruling (Prop. 18).

---

## §1 Ground truth (baseline, frozen 2026-09-27; never quote as current)

| # | Fact | Value | Mark | Source of truth |
|---|---|---|---|---|
| G1 | Merge-queue build time $T$ | 14 min | [V] n=1 | `release-flow-rules.md` |
| G2 | Release-PR cycle $C$ | 75–90 min | [A] | same |
| G3 | Merge-queue lanes $L$ / group size $B$ | 8 / 8 | [V] | ruleset 17836320 |
| G4 | CI retries $r$ | 2 | [V] | nextest profile |
| G5 | Ordinary PR CI latency | 34–44 min | [V] samples | `release-flow-rules.md` |
| G6 | B2 (clean-room `test --lib`) | 2055 s, of which compile 1208 s at `-j8` | [V] 2026-09-18 | infra#663 |
| G7 | Hosts in the PR pool | intel (x86, 32 thr, RAM [U]); framework16 (x86, 16 thr, 60 GB); yoga (x86, 22 thr, RAM [U]); gx10 (aarch64, 20 cores, 120 GB unified); mini (macOS arm64, 10 cores, 30 GB) | [V] except RAM | `infra/machines/*/forjar.yaml`, status page 2026-09-26 |
| G8 | lambda | final release-asset build only, never PR CI | [V] doctrine | `sovereign-ai-stack.md` §6 |
| G9 | Observed squander | yoga at load 92 on 22 threads while framework16 had 53 of 60 GB free; gx10 CPU idle outside serve work | [V] 2026-09-26 | status page |
| G10 | Docs-only PR (Alfredo) paid for the full quorum and full CI | — | [V] cop report | #4462 |
| G11 | Per-entry defect rate $q$, flake rate $\varphi$, arrivals $\lambda$, fix rate $f$, fix time $F$, retry cost $\rho$, re-arm delay $R$ | unknown | [U] | #4513 / QM-01 |
| G12 | Compile work $W$, critical path $L_c$, peak compile RAM, per-test durations | unknown | [U] | QM-08 (`cargo build --timings`, cgroup anon) |

---

## §2 Notation

| Symbol | Meaning |
|---|---|
| $C, T$ | release-PR cycle time; merge-queue build time (min) |
| $\rho, r$ | extra time per retry; retries per run. There are two measured values of $\rho$: $\rho_{\text{rel}}$ on the release PR (§4) and $\rho_{\text{MQ}}$ in the merge queue (§5). A bare $\rho$ means $\rho_{\text{rel}}$ in §4 and $\rho_{\text{MQ}}$ in §5 (R-11) |
| $\varphi, q, f, F, R$ | flake rate per attempt; per-PR defect rate; fix success rate; fix time; re-arm delay |
| $q', \lambda'$ | per-**entry** defect rate and entry rate, re-entries included (Q4). A merge-queue failure ratio measures $q'$, not $q$ (R-12) |
| $a, \mu_D, \kappa, w$ | base age: hours from the PR head's last base on `main` to its merge-group build; rate (per hour) of `main` commits touching the PR's changed files $D$; share of such commits that conflict with or break the PR; arm-to-entry window (Prop. 20) |
| $k$ | PRs folded into one release push |
| $L, B=2^h$ | merge-queue lanes; merge-queue group size |
| $\lambda$ | PR (or CI-run) arrival rate |
| $S_r$ | $\sum_{j=1}^{r}\varphi^j$ |
| $P_h, R_h, a_h$ | threads, RAM (GB) and architecture of host $h$ |
| $t_j, m_j, d_j, A_j, n_j$ | threads, peak RAM, duration, admissible architectures, and count per PR of job type $j$ |
| $W, L_c$ | total compile work (thread-seconds) and critical path of the crate/unit DAG (seconds) |
| $\mathcal I$ | build-and-test input set of the repo at base (Def. 14) |
| $\mathcal C$ | configuration set: manifests, lockfile, toolchain, `.cargo/`, workflows, build scripts, proc-macro packages |

---

# Part I: queue and release cycle

## §3 Model assumptions

**Release PR**
- **A1 Detection.** A cycle whose head contains an unfixed defect is red. Every retry also fails, so the cycle costs $C + r\rho$.
- **A2 Identification.** A red cycle names every defective PR through its failing tests. This is measured by QM-07, and a rate below 0.9 triggers S-4.
- **A3 Parallel fixes.** All named defects are fixed in parallel within $F$, each succeeding independently with probability $f$.
- **A4 Flakes.** Attempts flake independently with probability $\varphi$. A flake-only failure is rerun at once with no fix.
- **A5 Independence.** Defects are independent Bernoulli($q$).

**Merge queue**

> **Q1–Q3 are `[U]` against GitHub's actual merge queue (R-3).** They describe *a* grouped queue that bisects. GitHub may instead give every entry its own `merge_group` build of *main + every entry ahead + itself*. In that design, build concurrency caps how many of those builds run at once, and a failing entry is removed while the builds behind it are recreated, so Lemma 5's tree is not the mechanism. QM-02 records the observed semantics. Until it does, the $B > 1$ rows of §6 are model rows, not a description of the live queue. The group-size decision does not depend on this, because at $B = 1$ every semantics coincides (Theorem 6, Remark).

- **Q1 Speculation.** Entries form groups of up to $B$ consecutive entries. Up to $L$ builds run at once, each on top of every build ahead of it.
- **Q2 Outcome.** A build fails if and only if its group contains a defect ($\varphi$ is handled in Prop. 10). It takes $T$ if it passes and $T + r\rho$ if it fails.
- **Q3 Failure handling.** When the head build fails, every build behind it is cancelled. A failed group of size > 1 is split into halves, which go to the front of the queue left half first. A failed group of size 1 is ejected.
- **Q4 Re-entry.** An ejected defective PR re-enters after its fix, succeeding with probability $f$. Therefore $\lambda' = \lambda(1 + q/f)$ and $q' = q/(f+q)$.

*Derivation of Q4.* The number of failing entries of a defective PR is Geometric($f$) with mean $1/f$, followed by one passing entry. So the expected number of entries per original PR is $1 + q/f$, of which $q/f$ are defective. ∎ `pv: queue-model-v1 / DEF-Q4`
- **Q5 Negligible false re-entry (R-7).** A falsely ejected good entry re-enters after $R$ and adds $(1-q')\varphi^{r+1}$ entries per entry, which Q4 leaves out. The model assumes $\varphi^{r+1} \ll q/f$. At $\varphi \le 0.01$ (the flake budget) the omitted term is under 1% of $\lambda'$. If measured inputs violate Q5, the xtask adds the term to $\lambda'$ and says so.

## §4 Release PR

### Lemma 1 (number of defect cycles)
$P(N_i > n) = q(1-f)^n$. The number of defect cycles is $M = \max_i N_i$, with
$$\mathbb{E}[M] = \sum_{n=0}^{\infty}\Big[1 - \big(1 - q(1-f)^n\big)^{k}\Big].$$

*Proof.* A defective PR is red in cycle 1 (A1), and each fix succeeds with probability $f$ (A3). So $N_i \mid \text{defective} \sim$ Geometric($f$). Fixes run in parallel, so cycle $n$ is a defect cycle if and only if $\max_i N_i \ge n$. By independence (A5), $P(M \le n) = (1 - q(1-f)^n)^k$, and $\mathbb{E}[M] = \sum_{n \ge 0} P(M > n)$. ∎ `pv: queue-model-v1 / POST-L1`

### Theorem 1 (expected time to a green release head)
$$\boxed{\;\mathbb{E}[T_{\text{fold}}(k,r)] = \mathbb{E}[M]\,(C + r\rho + F) + \frac{C + \rho S_r}{1-\varphi^{r+1}}\;}$$

*Proof.*
- **Defect phase.** There are $M$ red cycles (Lemma 1), each costing $C + r\rho$ (A1) plus a fix of $F$ (A3).
- **Clean phase.** Attempts are i.i.d. and each fails with probability $\varphi^{r+1}$, so the number of attempts is Geometric($1 - \varphi^{r+1}$). An attempt's expected cost is $C + \rho\,\mathbb{E}[\min(X, r)] = C + \rho S_r$, where $P(X \ge j) = \varphi^j$ counts leading flakes.
- The event {an attempt happens} depends only on earlier attempts, so Wald's identity gives the clean-phase cost as the expected number of attempts times the expected cost per attempt. ∎ `pv: queue-model-v1 / POST-T1`

### Theorem 2 (one fold strictly dominates any split into sequential pushes)
For any partition of the $k$ PRs into $m \ge 2$ sequential batches, $\mathbb{E}[T_{\text{fold}}] < \mathbb{E}\big[\sum_j T_{\mathcal B_j}\big]$. **Precondition:** A2.

*Proof.* Couple both strategies on the same $(N_i)$.
- **Defect phases.** Pathwise, $\max_i N_i \le \sum_j \max_{i \in \mathcal B_j} N_i$, because every $N_i \ge 0$.
- **Clean phases.** The fold has one clean phase and the partition has $m$ i.i.d. clean phases, each with mean $\gamma \ge C > 0$.

So $\mathbb{E}[T_{\text{fold}}] \le \mathbb{E}\big[\sum_j T_j\big] - (m-1)\gamma$, which is a strict inequality. ∎ `pv: queue-model-v1 / POST-T2`

*Scope (R-8).* The comparator is sequential pushes. Pipelined pushes, where batch $j+1$ is pushed while batch $j$'s CI runs, are not covered. A5 also excludes interaction defects, where two PRs are only red together, and those can favour smaller folds. QM-07 measures A2 and is the row that falsifies A5.

### Corollary 3 (retry break-even on the release PR)
Going from $r = 0$ to $r = 1$ helps if and only if $\varphi(C - \rho)/(1-\varphi^2) > \rho\,\mathbb{E}[M]$.

*Proof.* Take the difference of Theorem 1 at $r = 1$ and $r = 0$, writing $C/(1-\varphi) = C(1+\varphi)/(1-\varphi^2)$. ∎ `pv: queue-model-v1 / POST-C3`

For $r \ge 2$ the xtask evaluates Theorem 1 over $r \in \{0,\dots,3\}$ and takes the argmin.

### Corollary 4 (cycle time is the dominant release lever)
$\partial\,\mathbb{E}[T_{\text{fold}}]/\partial C = \mathbb{E}[M] + 1/(1-\varphi^{r+1}) \ge 1$.

*Proof.* Differentiate Theorem 1. ∎ `pv: queue-model-v1 / POST-C4`

*Scope (R-9).* The derivative holds $F$ and $\varphi$ fixed. Sharding moves both: more shards can mean more flake exposure, and faster CI can mean faster fixes. So QM-06 records $C$, $F$ and $\varphi$ before and after.

**Consequence after sharding.** With CI at 16–26 min (§H.4), the quorum review (about 30 min) becomes the long pole of the release cycle. The next release lever is review latency, not CI.

## §5 Merge queue

### Lemma 5 (the bisection tree; scope: one full group of size $B = 2^h$)
$$E_t(B) = 1 + \sum_{d=1}^{h}2^d\big[1-(1-q')^{B/2^{d-1}}\big], \qquad E_f(B) = \sum_{d=0}^{h}2^d\big[1-(1-q')^{B/2^{d}}\big],$$
$$E_{ff}(B) \ge \sum_{d=0}^{h-1}2^d\big[1-(1-q')^{B/2^{d+1}}\big].$$

*Proof.*
- **Tested builds.** A non-root node is tested if and only if its parent contains a defect (Q2, Q3).
- **Failing builds.** A node fails if and only if it contains a defect, and a failing node is always tested.
- **Consecutive failures.** After an internal node fails, the next resolution at the head is its left child (Q3). Counting only these parent → left-child pairs gives a lower bound.

Linearity of expectation over the $2^d$ nodes at each depth gives all three. ∎ (Checked against Monte Carlo in §6.1.) `pv: queue-model-v1 / POST-L5`

### Theorem 6 (group size: $B = 1$ minimises failures per entry)
$$\boxed{\;\frac{E_f(B)}{B} = \sum_{j=0}^{h}\frac{1-(1-q')^{2^j}}{2^j}\;}$$
This is strictly increasing in $h$ for every $q' \in (0,1)$, and equal to $q'$ at $B = 1$. Likewise $E_{ff}(B)/B$ is at least $\sum_{j=0}^{h-1}\big(1-(1-q')^{2^j}\big)/2^{j+1}$, which is non-decreasing in $h$ and 0 at $B = 1$.

*Proof.* Substitute $j = h - d$; each increase of $h$ adds a strictly positive term. ∎ `pv: queue-model-v1 / POST-T6`

*Remark (robust to GitHub's failure semantics).* If a failed group is ejected whole instead of bisected, the expected false ejections per entry are $(1-q')\big[1-(1-q')^{B-1}\big]$. That is also increasing in $B$ and 0 at $B = 1$. At $B = 1$ the two semantics coincide.

### Lemma 7 (head-of-line windows)
After each failing resolution at $t_k$, no entry merges until the next resolution, which comes at least $T$ later, or $T + r\rho$ later if it is a failure. These windows are pairwise disjoint.

*Proof.* Q3 cancels every build behind the head, so the next head build starts at $t_k$ or later and lasts $T$, or $T + r\rho$ if it fails (Q2). Merges happen in order. ∎ `pv: queue-model-v1 / POST-L7`

### Theorem 8 (necessary condition for positive recurrence)
Let
$$\rho_{\text{HOL}}(B,r) = \lambda'\,\frac{T E_f(B) + r\rho\,\underline{E}_{ff}(B)}{B},$$
where $\underline{E}_{ff}$ is Lemma 5's lower bound. Only the bound has a closed form, so $\rho_{\text{HOL}}$ is computed with it, and every conclusion below needs only a lower bound (R-4).
Assume (i) groups formed under saturation are full and their defect indicators are i.i.d. Bernoulli($q'$), and (ii) arrivals are Poisson($\lambda'$). **If $\rho_{\text{HOL}} \ge 1$, the backlog process is not positive recurrent.** At $\rho_{\text{HOL}} > 1$ it is transient; at exactly 1 it can at best be null recurrent.

*Proof.* When the backlog is at least $LB$, groups are full. By Lemma 7 and renewal-reward over i.i.d. groups (assumption i), the long-run head time per departure is at least $h(B) = [T E_f + r\rho E_{ff}]/B$, so the departure rate is at most $1/h(B) \le \lambda'$. The backlog then has non-negative drift outside a finite set, This rules out positive recurrence by the converse drift criterion for chains with bounded second moments of increments (Tweedie 1976; Meyn & Tweedie ch. 11 `[X]`). With drift bounded below by $\lambda' - 1/h(B) > 0$ the chain is transient (Lamperti's criterion `[X]`). Foster–Lyapunov and Pakes' lemma run the other way, from negative drift to positive recurrence, and are not the tool here (R-4). ∎ `pv: queue-model-v1 / POST-T8`

*Tightness.* $h(B)$ leaves out passing resolutions after a pass, cancelled speculative work and pipeline refill, so it is a lower bound. The operating threshold comes from simulation (§7).

### Proposition 9 (lane capacity, and when $B > 1$ is justified)
A necessary condition is $\rho_{\text{lane}} = \lambda'\,[T E_t(B) + r\rho E_f(B)]/(LB) < 1$. At $B = 1$ this is $\lambda'(T + r\rho q') < L$. **So $B > 1$ can be needed only when $\lambda' \ge L/(T + r\rho q')$**, which is about 34 per hour at $L = 8$, $T = 14$, $r = 0$.

*Proof.* Each tested build occupies a lane for at least $T$, or $T + r\rho$ if it fails, and there are $L$ lanes. The drift argument is the same as in Theorem 8. ∎ `pv: queue-model-v1 / POST-P9`

### Proposition 10 (merge-queue retries at $B = 1$; corrected in v2.0)
Measure an entry's cost as the head time it adds plus $R$ for each false ejection. Going from $r = 0$ to $r = 1$ lowers the expected cost if and only if
$$\boxed{\;(1-q')\,\varphi\,\big[(T+R)(1-\varphi) - \rho\big] > q'\rho\;}$$

*Proof.*
- **At $r = 0$:** $q'T + (1-q')\varphi(T + R)$.
- **At $r = 1$:** $q'(T+\rho) + (1-q')\varphi\rho + (1-q')\varphi^2(T+R)$. Every first-attempt flake pays $\rho$ once. A double flake additionally pays $T + R$ and **not** a second $\rho$, which v1.0 double-counted.

The difference is $q'\rho + (1-q')\varphi\big[\rho - (T+R)(1-\varphi)\big]$, which is negative if and only if the stated inequality holds. ∎ `pv: queue-model-v1 / POST-P10`

With $q' = 0.111$, $T = 14$, $R = 15$, $\rho_{\text{MQ}} = 8$, the break-even is $\varphi^\* = 0.0512$ (v1.0 had 0.0525). No decision changes.

### Proposition 20 (PR age and the conflict hazard; operator amendment 2026-09-27, v2.1)
Let the `main` commits that touch a PR's changed files $D$ arrive as a Poisson process of rate $\mu_D$. Suppose a share $\kappa \in [0,1]$ of them conflict with the PR textually or break it semantically, and let the PR reach its merge-group build $a$ hours after its head was last based on `main`. Then
$$\boxed{\;P(\text{conflict}) = 1 - e^{-\kappa\mu_D a} \;\le\; 1 - e^{-\mu_D a}\;}$$
and the per-PR defect rate that A5 and Q4 treat as a constant is really a function of age, $q(a) = q_0 + (1-q_0)\,(1 - e^{-\kappa\mu_D a})$. Here $q_0$ is the age-free defect rate, which includes semantic breaks from files outside $D$.

*Proof.* Thinning a Poisson process of rate $\mu_D$ with independent marks of probability $\kappa$ gives a Poisson process of rate $\kappa\mu_D$. The probability of at least one event in an interval of length $a$ is $1 - e^{-\kappa\mu_D a}$. A textual conflict needs a change to a file in $D$, so $\kappa = 1$ is the worst case, and that gives the bound. A conflict and an age-free defect are independent by assumption, so $q(a) = 1 - (1-q_0)\,e^{-\kappa\mu_D a}$, which is the stated form. ∎ `pv: queue-model-v1 / POST-P20`

**Consequences.**
- **The queue model holds only while ages are short and stable.** Q4 and every Part I result take $q$ as one number. With age in play, they hold with $q$ replaced by $\bar q = \mathbb E_a[q(a)]$ over the base-age distribution. That distribution drifts as PRs linger, so $\bar q$ drifts with it, and a queue that looked stable at the last measurement can cross Theorem 8's boundary with no change in code quality. Keeping $a$ short keeps $\bar q \approx q_0$, the value QM-01 measures.
- **Rebase before arm resets $a$ to $w$.** After `gh pr update-branch N --rebase`, $a$ is the window $w$ from that rebase to the merge-group build: CI plus quorum plus queue wait. A push after the quorum lapses it, so the order is **rebase → CI → quorum → arm**, and the rebase never comes after the quorum. The force-push ban stands. The rebase is GitHub-side, and on a conflict the author commits a merge of `main` with a per-hunk resolution table (cop ruling 2026-09-27).
- **Illustration `[C]` on assumed inputs** ($\mu_D$ is `[U]` until QM-01): take $\mu_D = 0.05$/h, one change to the PR's files every 20 h, and $\kappa = 1$. A PR based 24 h ago has $P = 1 - e^{-1.2} = 0.70$. The same PR rebased with $w = 1$ h has $P = 0.049$.

## §6 Part I validation (analysis run 2026-09-27)

| Quantity | Parameters | Closed form | Monte Carlo (200k) |
|---|---|---|---|
| $E_t, E_f$ | $B=8,\ q'=0.1$ | 5.0347, 2.8173 | 5.0371, 2.8187 |
| $E_t, E_f$ | $B=4,\ q'=0.2$ | 3.6208, 2.1104 | 3.6284, 2.1160 |
| $\mathbb{E}[T_{\text{fold}}]$ | $k=5,q=.1,f=.8,C=80,F=20,\varphi=.02,\rho_{\text{rel}}=10$; $r=0$ / $r=2$ | 134.67 / 143.86 | 134.44 / 143.60 |
| $\mathbb{E}[T_{\text{fold}}]$ | $k=8,q=.1,f=.8,C=80,F=20,\varphi=.05,\rho_{\text{rel}}=10,r=1$ | 164.12 | 164.20 |

**Release fold** ($k=5$): at $C=80$, one fold takes **134.7 min** against 221.7 min for two pushes, with $\varphi^\* = 0.075$. At $C=30$ it takes **57.1 min** against 90.4, with $\varphi^\* = 0.249$.

**Merge-queue load against simulation** ($T=14$, $\rho_{\text{MQ}}=8$, $f=0.8$, $L=8$; 10 runs of 5 days each; p90 includes fix loops):

| λ/h, $q$ | $B$, $r$ | $\rho_{\text{HOL}}$ | Sim p50 / p90 (min) |
|---|---|---|---|
| 6, 0.10 | **1, 0** | 0.175 | **14 / 118** |
| 6, 0.10 | 8, 2 (v1.0 ruleset) | 0.887 | 37 / 1528 |
| 6, 0.20 | **1, 0** | 0.350 | **17 / 129** |
| 6, 0.20 | 8, 2 | 1.633 | 2302 / 4874 |
| 10, 0.10 | **1, 0** | 0.292 | **14 / 82** |
| 10, 0.10 | 8, 2 | 1.479 | 2299 / 4306 |

Knee `[C]`: $\rho_{\text{HOL}} \le 0.61$ stayed at p90 ≤ 134 min, and $\ge 0.89$ degraded more than 10×.

*Provenance (R-5, R-6).* The closed-form column was recomputed independently for v1.1 and v2.1, and every value agrees to 4 significant figures. The Monte Carlo and simulation columns come from the author's analysis outside the repo. They stay `[A]` until QM-04 reproduces them, because QM-00's golden tests must not enforce the prose. The $B > 1$ rows are model rows (R-3).

---

# Part II: fleet hardware

## §H.1 Model

- **Hosts.** Host $h$ has $P_h$ threads, $R_h$ GB and architecture $a_h$.
- **Jobs.** Job type $j$ needs $t_j$ threads and $m_j$ GB for $d_j$ seconds, may run only on architectures in $A_j$, and occurs $n_j$ times per PR.
- **PR CI shapes.**
  - **P0 (today):** 4 jobs per PR, each compiling the workspace itself at `-j8`, then running its own workload.
  - **P2 (compile-once):** one workspace test build at width $t^\*$, archived (`cargo nextest archive`), then $K$ test shards at 8 threads each, run from the archive.
  - **P3 (P2 + tiers):** Part III routing on top of P2.

**Calibration anchor.** P0's unloaded PR latency in the model is 34.4 min, against 34–44 min measured (G5). The work constants are $W = 9700$ thread-s, $L_c = 600$ s and post-compile work of $2150 \times 8$ thread-s. All are `[A]` until QM-08 measures them.

## §H.2 Compile width

### Theorem 11 (Graham's bound for a greedy parallel build)
For a unit DAG with total work $W$ and critical path $L_c$, built by a greedy scheduler with $P$ threads (cargo's jobserver starts any ready unit when a token is free), the build time $\tau(P)$ satisfies
$$\max\!\big(W/P,\; L_c\big) \;\le\; \tau(P) \;\le\; W/P + L_c.$$

*Proof.*
- **Lower bound.** $P$ threads deliver at most $P$ thread-seconds per second, and the critical path is a chain of dependencies.
- **Upper bound.** Split $[0, \tau]$ into busy instants (all $P$ threads working) and non-busy instants. The busy instants total at most $W/P$.
- Take the unit $u_1$ that finishes last and walk backwards. At any non-busy instant before $u_1$ starts, a greedy scheduler would have started $u_1$ if it were ready. So some predecessor of $u_1$ is running at that instant. Pick $u_2$ as the predecessor that finishes last before $u_1$ starts, and repeat.
- This builds a dependency chain $u_k \to \dots \to u_1$ in which some chain unit is running at every non-busy instant. So the non-busy instants total at most the chain's length, which is at most $L_c$. ∎ (Graham 1966 `[X]`, proved here in full.) `pv: fleet-model-v1 / POST-T11`

*Assumption (H-a, R2-2).* Every unit holds one token for a duration that does not depend on $P$. rustc's parallel codegen and the linker take extra jobserver tokens or threads, and memory-bandwidth contention stretches durations as $P$ grows. So $W$ is measured as the sum over units of duration × threads held (§9.1), at each width $\{8, 16, 24\}$ separately. The bound is then applied per width, never extrapolated from one.

### Corollary 11a (compile width rule)
Set $t^\* = \lceil W/L_c \rceil$.
- At $P = t^\*$: $\tau \le 2L_c$, within 2× of the best possible time at any width.
- For $P > t^\*$: the lower bound stays at $L_c$, and the upper bound improves by at most $W/t^\* - W/P \le L_c$.

**Rule:** compile slots are about $t^\*$ threads wide. A host with $P_h \ge 2t^\*$ runs $\lfloor P_h/t^\* \rfloor$ compile slots in parallel rather than one wider build.

**With the §H.1 constants (R2-1).** $W/L_c = 9700/600 = 16.17$, so $\lceil W/L_c\rceil = 17$, not 16. At $P = 16$ the upper bound is $606 + 600 = 1206$ s $= 2.01\,L_c$, so the $2L_c$ guarantee misses by 6 s. At $P = 17$ it holds: 1171 s. The choice matters for packing. Under the rule, intel's 32 threads hold one 17-wide slot, but two 16-wide slots lose only 0.5% of the guarantee and double intel's compile throughput. **§7 uses 16, written as a deliberate rounding down, not as $\lceil W/L_c \rceil$.** QM-08 re-derives it from measured $W$ and $L_c$.

| Threads | Lower bound (s) | Upper bound (s) |
|---|---|---|
| 8 | 1212 | 1812 |
| 16 | 606 | 1206 |
| 32 | 600 | 903 |

### Proposition 12a (compile-once work identity)
Suppose $k$ jobs each build the workspace, with a shared-cache hit rate $\eta$. Replacing them with one build plus $k$ archive consumers saves exactly $(k-1)W(1-\eta)$ thread-seconds per PR, minus the archive transfer cost $k\,o\,t_s$. Here $o$ is the per-consumer transfer and startup time and $t_s$ the threads a consumer holds meanwhile.

**Precondition (R2-3).** The $k$ jobs build the *same* artifacts: same profile, feature set, target and `RUSTFLAGS`. A `clippy`, `check` or `doc` job builds different artifacts and is not one of the $k$. At QM-09, the workflow lists which of P0's jobs share a fingerprint.

*Proof.* Under the precondition the build work is identical per job, and the consumers run only test execution. ∎ `pv: ci-compile-once-v1 / POST-P12a`

## §H.3 Shard balancing

### Theorem 13 (list-scheduling bound for test shards)
Assign tests (or test groups) with durations $p_i$ to $K$ shards greedily: whenever a shard is free, it takes the next test. Then
$$\text{makespan} \le \frac{\sum_i p_i}{K} + \max_i p_i.$$

*Proof.* Let test $\ell$ finish last, having started at time $s$. Before $s$ every shard was busy, otherwise $\ell$ would have started earlier. So $K s \le \sum_{i \ne \ell} p_i$, and the finish time $s + p_\ell \le \sum_i p_i / K + p_\ell$. ∎ Longest-first (LPT) ordering tightens this to $(4/3 - 1/(3K))\cdot\text{OPT}$ (Graham 1969 `[X]`).

### Corollary 13a (shard count)
To keep each shard within a budget $\tau_s$ given per-shard overhead $o$ (archive download plus startup), use
$$K = \left\lceil \frac{\sum_i p_i}{\tau_s - o - \max_i p_i} \right\rceil,$$
and split any test group with $p_i > \tau_s/2$. With $\sum p = 17{,}200$ thread-s / 8 threads = 2150 s, $\tau_s = 300$ s, $o = 60$ s and $\max p = 60$ s `[A]`: $K = 12$.

## §H.4 Capacity (linear programme)

### Theorem 12 (fluid capacity bound)
If the fleet sustains PR throughput $\lambda$ with bounded queues, then the linear programme
$$\begin{aligned}
\max\ \lambda \quad \text{s.t.}\quad
& \textstyle\sum_h x_{jh} = \lambda\, n_j \quad \forall j, \\
& \textstyle\sum_j x_{jh}\, t_j\, d_j \le U P_h \quad \forall h, \\
& \textstyle\sum_j x_{jh}\, m_j\, d_j \le U R_h \quad \forall h, \\
& x_{jh} = 0 \ \text{if}\ a_h \notin A_j, \qquad x \ge 0
\end{aligned}$$
is feasible at that $\lambda$ with $U = 1$. So $\lambda \le \lambda^\*$.

*Proof.* Let $x_{jh}$ be the long-run completion rate of type-$j$ jobs on host $h$. Stability makes completions equal arrivals, which gives the equality constraints. By Little's law applied to jobs of type $j$ in service on $h$, the mean number in service is $x_{jh}d_j$. So the time-average threads in use on $h$ are $\sum_j x_{jh} t_j d_j$, which is at most $P_h$ because usage never exceeds $P_h$. The same argument bounds RAM. Architecture eligibility holds by construction. ∎ `pv: fleet-model-v1 / POST-T12` The design capacity uses headroom $U = 0.85$ `[A]`.

**What the two rows measure (R2-4).**
- **Threads.** "Usage never exceeds $P_h$" is true of **CPU-seconds**, not of requested threads. G9, load 92 on 22 threads, is requested threads exceeding $P_h$: jobs stretch, and $d_j$ grows with the load. So $t_j d_j$ is the measured CPU-seconds of a type-$j$ job, from cgroup `cpu.stat usage_usec`, not the `-j` value × wall time.
- **RAM.** The row uses peak $m_j$ for the whole $d_j$. That is implied by stability only under **reservation**: each job holds $m_j$ for its lifetime, as a `MemoryMax` slice does (P1–P3). For P0, which enforces no RAM, the row is not a necessary condition, and P0's valid bound is the thread row alone. The crude cross-check is 70 x86 threads × 0.85 × 3600 / 56,000 ≈ **3.8 PR/h** (P0 threads only), against 3.4 with the RAM row. Both are `[C]` from `[A]` inputs.

**Solved (HiGHS, $U = 0.85$, intel RAM 192 GB `[A]`, yoga RAM 64 GB `[A]`):**

| Shape | Thread-s per PR | $\lambda^\*$ (PR/h) | Binding resource |
|---|---|---|---|
| P0 today | 56,000 | **3.4** | x86 threads, plus framework16/yoga RAM |
| P2 without cache ($\eta = 0$) | 30,740 | 7.0 | x86 threads |
| P2 ($\eta = 0.6$ `[A]`) | 24,920 | 8.6 | x86 threads |
| P3 ($\eta = 0.6$) | 11,053 | **21.5** | x86 threads; gx10 33% busy (T0/T1 only) |
| P3 ($\eta = 0$) | 17,018 | 15.0 | x86 threads |

Compile-once alone doubles capacity (3.4 → 7.0). The shared cache adds 23%, and tiering multiplies capacity by 2.5 on top.

**Full-tier CI bound (R2-5).** Under P2/P3, with a free slot, the full tier's CI latency is compile plus shards. Compile is 606–1206 s at width 16 (Thm 11). Shards are at most $2150/12 + 60 + 60 = 299$ s (Thm 13 + Cor. 13a at $K = 12$); v2.0's 329 s had no derivation. That gives **15.1–25.1 min**. It bounds **CI given a free slot**, not $C$. $C$ is the release cycle: review runs beside CI, plus runner queue wait, T0/T1 and the doctest job (R2-10). By §4 the review, at about 30 min, is then the long pole. So the claim is "CI is no longer the long pole, by proof". It is not "$C \le 30$ by proof", and QM-06 measures $C$ itself.

## §H.5 Dispatch and runner shapes

### Proposition 14a (zero squander under resource-aware, work-conserving dispatch)
Define squander $S(t) = 1$ if some job is queued while an eligible host has at least $t_j$ free threads and $m_j$ free GB. A dispatcher that, at every arrival and completion, places queued jobs until none fits anywhere has $S(t) = 0$ for all $t$.

*Proof.* Resources change only at events. After each event's dispatch pass, no queued job fits any host by construction, and between events free resources can't grow. ∎

`pv: runner-shapes-v1 / POST-P14a`

*What zero squander does not give (R2-6).* It bounds idle capacity, not waiting. A work-conserving dispatcher that backfills small jobs can starve a large one indefinitely: an `x86-c16-m48` compile never fits while shards keep taking the freed threads. The dispatcher therefore also needs an age rule. A job queued longer than its shape's budget reserves the next host that frees enough room. QM-12 reports the maximum queue age per shape beside squander.

*Corollary.* Static per-host slots or labels have $S > 0$ whenever a job's label maps only to busy slots while another eligible host has room. G9 is exactly this.

**Implementation on GitHub Actions.** Runners are slots, so exact zero squander isn't available. The spec approximates it three ways:
1. **Org-wide capability-and-size labels** (`x86-c16-m48` compile, `x86-c8-m8` shard, `any-c8-m16` check, `any-c1-m1` lint), pooled across hosts (rule 17). GitHub then places a job on any idle matching runner, which is work-conserving within a shape.
2. **Runner counts per shape** $N_c = \lceil a_c + \beta\sqrt{a_c}\,\rceil$, where $a_c = \lambda \sum_{j \in c} n_j d_j$ is the offered load in Erlangs and $\beta = 1$ (square-root staffing, Halfin–Whitt `[X]`). Runners are placed onto hosts by first-fit-decreasing on (threads, RAM) within $U$.
3. **Hard cgroup limits.** Each runner gets a systemd slice with `CPUQuota` and `MemoryMax` equal to its shape, so shapes can't oversubscribe a host. That is what makes P0's hidden RAM overcommit visible (§H.6).

**Residual squander** is cross-shape only (for example compile slots idle while shards queue). It is measured by QM-12 and targeted at ≤ 5% of host-time.

**Hosts that can't run x86 work** (gx10, mini) get arch-neutral work: T0 lint, T1 `cargo check`/clippy, the aarch64 and macOS legs, and serving/Prometheus on gx10. The LP puts gx10 at 33% busy at $\lambda^\*$ (P3). Its target is utilisation of the arch-neutral pool, not x86 parity.

## §H.6 Simulation (discrete-event, analysis run 2026-09-27)

The simulation models PR CI runs arriving as a Poisson process with class mix docs / code-small / cross-crate = 25 / 55 / 20% `[A]`, 3 runs per cell, times in minutes.
- **P0:** static 8-thread slots (intel 4, yoga 3, framework16 2) with RAM **not** enforced, as today.
- **P1:** P0's job shapes with resource-aware pooling and RAM enforced.
- **P2:** compile-once plus cache.
- **P3:** P2 plus tiers.

| λ/h | Shape | All p50 / p90 | Docs p50 | Code-small p50 | Cross-crate p50 | CPU % intel / fw16 / yoga / gx10 |
|---|---|---|---|---|---|---|
| 2 | P0 | 34.4 / 46.8 | 34.4 | 34.4 | 34.4 | 46 / 58 / 30 / 0 |
| 2 | P3 | **6.7 / 9.5** | **2.0** | 6.7 | 9.5 | 5 / 14 / 11 / 0 |
| 4 | P0 | 50.0 / 100.9 | 50.4 | 48.7 | 48.8 | 85 / 86 / 78 / 0 |
| 4 | P1 (RAM 32 GB/compile) | 312.6 / 426.6 | 316.0 | 308.3 | 308.9 | 98 / 49 / 71 / 0 |
| 4 | P2 | 12.1 / 20.1 | 12.8 | 11.6 | 12.4 | 37 / 54 / 29 / 0 |
| 4 | P3 | **6.7 / 10.4** | **2.0** | 6.7 | 9.5 | 10 / 27 / 19 / 1 |
| 6 | P0 | 613.9 / 1117.6 | — | — | — | 100 / 99 / 99 / 0 |
| 6 | P2 | 15.1 / 26.6 | 15.3 | 15.2 | 15.3 | 58 / 77 / 45 / 0 |
| 6 | P3 | **6.7 / 11.4** | **2.0** | 6.7 | 9.9 | 18 / 38 / 27 / 3 |

*Rows above capacity are transients (R2-7).* P0 at 4 and 6 per hour, and P1 at 4, run above that shape's $\lambda^\*$. A finite simulation reports finite waits there, but they grow with the horizon and are not steady-state values. Read them as "saturated", not as latencies. P0 enforces no RAM, so its rows also leave out swap and OOM, which would make them worse.

Findings `[C]`:
1. **Today's fleet saturates at about 3.4 PR CI runs per hour** (the LP and the simulation agree). Above that, waits grow without bound.
2. **Pooling alone (P1) doesn't help.** With RAM enforced it looks worse, because P0's slots overcommit RAM: two 32 GB compiles on framework16's 60 GB. **The job shape is the lever, not the dispatcher.**
3. **Compile-once (P2)** cuts p50 by 3–40×, depending on load.
4. **Tiers (P3)** keep docs PRs at 2 min and code at 7–10 min up to 6 runs per hour, with capacity to about 21 per hour.

---

# Part III: tiered testing

## §T.1 Tiers

| Tier | Contents | Budget (timeout = andon) | Runs on |
|---|---|---|---|
| **T0 lint** | `cargo fmt --check`; the diff classifier (Def. 14); markdown/link/SHACL doc validation; `actionlint`; `bashrs` on changed shell; `mdbook build` if book files changed | ≤ 2 min | `any-c1-m1` (mini, gx10, x86) |
| **T1 check** | `cargo check --workspace --all-targets`; clippy `-D warnings` on $\mathcal R$ (Thm 15); shared cache | ≤ 5 min | `any-c8-m16` (gx10 preferred) |
| **T2 affected** | one `--workspace` test build at width $t^\*$, archived; run tests and doctests of $\mathcal R'$ (Thm 15) as shards | ≤ 15 min | `x86-c16-m48` compile, `x86-c8-m8` shards |
| **T3 full** | every test, doctest, feature matrix and example; Mode C GPU on the CUDA pool; sharded (Cor. 13a) | ≤ 30 min | same, with $K$ from Cor. 13a |
| Clean-room | release tag only (unchanged; hard gate) | — | clean-room pool |

## §T.2 Soundness of skipping

### Definition 14 (input set)
$\mathcal I$ is the union of three sets:
- $\mathcal I_{\text{dep}}$: every file listed in rustc dep-info (`target/**/*.d`) from a full T3 build on base. This covers every source file rustc reads, including `include_str!`/`include_bytes!` targets and `#[doc = include_str!(…)]` READMEs.
- $\mathcal F$: every file opened by any test process during a full T3 run on base, traced and unioned over the last 7 nightlies. The trace records three kinds of access (R2-9). (i) Every path opened or stat'd. (ii) Every directory **enumerated** (`getdents64`, `readdir`, globs, `walkdir`, `include_dir!`), recorded as the whole prefix `dir/**`. (iii) Every **failed** lookup (`ENOENT` on `openat`/`stat`/`access`), recorded as the path itself. Without (ii) and (iii), a PR that *adds* a file to a globbed directory, or creates a file a test probes for, can change an outcome while $D \cap \mathcal F = \emptyset$. v2.0 only traced `openat,open,stat` and covered this with an ad hoc "fixtures directory" rule, which is dropped. Use `strace -f -e trace=openat,open,stat,newfstatat,statx,access,getdents64` or fanotify with `FAN_OPEN|FAN_ACCESS|FAN_ONDIR`.
- $\mathcal C$: manifests, lockfile, toolchain file, `.cargo/`, workflow files, every build script's package when the script declares no `rerun-if-changed`, and every proc-macro package.

**Validity at a base (R2-8).** $\mathcal I$ is computed on the nightly commit $n$, but a PR is tested against its own base $b$. In the merge queue, $b$ is `merge_group.base_sha`, which is `main` plus the entries ahead. $\mathcal I(n)$ may be used at $b$ only if **every commit in $n..b$ itself took the docs lane**, since then $\mathcal I(b) = \mathcal I(n)$ by Thm 14 applied to the files the build reads. Otherwise the router refreshes $\mathcal I_{\text{dep}}$ from T1's `cargo check` dep-info on $b$, which it pays for anyway, and adds to $\mathcal R'$ the packages in $\mathcal R(D_{n..b})$ whose run-time reads it cannot refresh without a trace. Counterexample if skipped: commit $c_1$ (code, merged after $n$) adds `include_str!("../docs/x.md")`, and PR $c_2$ edits only `docs/x.md`. $\mathcal I(n)$ does not contain `docs/x.md`, so the router sends $c_2$ to the docs lane unsoundly. `pv: ci-input-set-v1 / INV-BASE`

### Theorem 14 (the docs lane is sound)
Assume:
- **(T-a)** test outcomes are deterministic functions of (compiled artifacts, files read at run time, environment); flakes are excluded, per FLAKE-0;
- **(T-b)** $\mathcal F$ is complete for base, including enumerated directories and failed lookups, and $\mathcal I$ is valid at base (Def. 14);
- **(T-c)** artifacts are a function of $\mathcal I_{\text{dep}} \cup \mathcal C$ and the environment.

If a PR's diff $D$ satisfies $D \cap \mathcal I = \emptyset$ and the environment is unchanged, **every test outcome on head equals its outcome on base.** Base is green `main`, so head is green, and T1–T3 can be skipped for that PR.

*Proof.* Every file in $\mathcal I$ is byte-identical between base and head. By (T-c) the compiled artifacts are identical. By (T-b) every test reads only files in $\mathcal F \subseteq \mathcal I$, all of them unchanged. Every directory it lists has the same entries, and every path it failed to find is still absent. By (T-a) each outcome is the same function of the same inputs. ∎ `pv: ci-tier-soundness-v1 / POST-T14`

*Consequences.* A README that is `include_str!`'d into rustdoc is in $\mathcal I_{\text{dep}}$, so changing it is **not** docs-only; the theorem handles this automatically. A spec under `docs/specifications/` read by no build and no test is docs-only. That covers Alfredo's case (G10).

*Backstop.* Nightly T3 on `main` stays mandatory. A nightly failure first reachable through a commit that took the docs lane is a **soundness escape**: STOP (S-8), five-whys, and fix $\mathcal I$.

### Theorem 15 (affected-closure selection is sound)
Let $D \cap \mathcal C = \emptyset$. Let $S$ be the packages owning the changed files, where a file is owned by every package whose dep-info lists it. Let $\mathcal R$ be the closure of $S$ under reverse normal, dev and build dependencies, and let
$$\mathcal R' = \mathcal R \cup \{X : D \cap \mathcal F_X \neq \emptyset\}.$$
**Build the whole workspace** (`--workspace`, the same feature resolution as T3) and run only the tests of $\mathcal R'$. Then every skipped test has the same outcome as on base.

*Proof.* Take $X \notin \mathcal R'$. $X$ doesn't depend, transitively, on any package in $S$. Its compilation inputs are files owned by $X$ and its dependencies, all unchanged. Feature unification is a function of the manifests in $\mathcal C$, which are unchanged, and the build set (`--workspace`), which is identical. So $X$'s test artifacts are identical. $X$'s run-time reads $\mathcal F_X$ don't meet $D$. By (T-a) its outcomes are equal. ∎ `pv: ci-tier-soundness-v1 / POST-T15`

*Edges (R2-11).* "Files owned by $X$" means $X$'s dep-info plus, for a build script, its `rerun-if-changed` paths. A build script without them is in $\mathcal C$. Doctests read the crate's source and its `include_str!` targets, which are in its dep-info. They need the doctest job of R2-10 to actually run.

*Doctests (R2-10).* `cargo nextest archive` does not carry doctests, and nextest does not run them `[X]` (re-verify at QM-09). T2 and T3 therefore run a separate `cargo test --doc` job for $\mathcal R'$ or the workspace. That job re-compiles. It is outside Prop. 12a's saving and outside the §H.4 latency bound, so QM-09 measures it separately. QM-09's never-down invariant counts tests and doctests **separately**, so doctests cannot vanish silently when the compile-once workflow lands.

*Why `--workspace` is required.* Building only $\mathcal R$ would change Cargo's feature unification and could produce artifacts that differ from T3's. The build stays whole and only the **run** is filtered.

## §T.3 Order and overlap of cheap stages

### Theorem 16 (fail-fast ordering; Smith's rule)
Stages $i$ have costs $c_i$ and independent failure probabilities $p_i$, and run serially, stopping at the first failure. The expected cost $\sum_i c_i \prod_{j<i}(1-p_j)$ is minimised by ordering stages by $c_i/p_i$ ascending.

*Proof.* Swap adjacent stages $i$ and $i+1$. The expected cost changes by $\prod_{j<i}(1-p_j)\,[\,p_{i+1}c_i - p_i c_{i+1}\,]$, which is non-positive if and only if $c_i/p_i \le c_{i+1}/p_{i+1}$. Any order that isn't sorted has an improving adjacent swap. ∎ `pv: ci-tier-router-v1 / POST-T16`

The xtask sorts T0/T1 sub-steps (fmt, classifier, clippy, check) by measured $c_i/p_i$ from QM-13 data.

### Theorem 17 (run cheap and full in parallel, with cancellation)
A cheap stage takes time $c$ on 1 runner, and a full stage takes $F \ge c$ on $k$ runners. The cheap stage fails with probability $p$, and in that case the full stage would fail too. Compare:
- **staged:** full starts only after cheap passes;
- **parallel-cancel:** both start together, and full is cancelled the moment cheap fails.

Pathwise, parallel-cancel's latency is at most staged latency, and strictly lower by $c$ on every pass. It costs exactly $kc$ extra runner-seconds on every cheap failure. In expectation:
$$\Delta\text{latency} = -(1-p)\,c, \qquad \Delta\text{runner-seconds} = +k\,p\,c.$$

*Proof.*
- **On a pass:** staged takes $c + F$ and parallel takes $F$; runner time is equal.
- **On a failure:** both finish at $c$; parallel has additionally held $k$ runners for $c$. ∎ `pv: ci-tier-router-v1 / POST-T17`

**Rule.** Use parallel-cancel while the shape's pool utilisation stays ≤ 0.6 after adding $\lambda k p c$ `[C]`, and staged otherwise. With today's squander (§H.6), parallel-cancel is the default.

### Proposition 18 (review tier threshold; operator-approved 2026-09-27, gated on measurement)
Take a PR class $\kappa$ with $n_\kappa$ merged PRs observed, of which $e_\kappa$ later caused a regression. Let $\bar p_\kappa$ be the Clopper–Pearson 95% upper bound (with $e_\kappa = 0$, $\bar p = 1 - 0.05^{1/n}$, about $3/n$). A lighter review tier for $\kappa$ has lower expected cost at 95% confidence if and only if
$$\bar p_\kappa\, c_{\text{escape}} < c_{\text{review saved}}.$$

*Proof.* With 95% confidence, the expected escape cost per PR is at most $\bar p_\kappa c_{\text{escape}}$. Compare it with the review cost saved per PR. ∎ `pv: review-tier-evidence-v1 / POST-P18`

*The measurand (R2-12).* For a class whose PRs all had full review, "$e_\kappa$ later caused a regression" counts only what full review **missed**. The lighter tier is judged on what it would miss, which also includes defects the full quorum **caught**. So $e_\kappa$ counts both: (a) post-merge regressions traced to a class-$\kappa$ PR, and (b) class-$\kappa$ PRs where a full-quorum FAIL finding led to a code change that the lighter tier's lanes would not have run. (b) is the $(\kappa_x - \kappa_c)$ term of v1.1 Prop. 12, and it is measurable from quorum receipts. Counting (a) alone biases $\bar p_\kappa$ low and makes the lighter tier look safe by construction. The bound stays the one-sided Clopper–Pearson upper bound.

For example, with $n = 30$ docs PRs and 0 escapes, $\bar p = 0.095$. So lighter docs review pays whenever an escape costs less than about 10.5× the review time saved. **The test tier is fixed by Theorem 14. This proposition decides the review tier only.** It applies to docs-only PRs and to #4462-vouched contributors. **Operator ruling 2026-09-27 (approved):** the lighter tier is approved for both classes. It switches on per class once $\bar p_\kappa$, measured with the R2-12 measurand, clears the threshold above. It switches back to full review **automatically** on the first escape in that class. An escape is a post-merge regression traced to a lighter-reviewed PR, and it resets that class's $n_\kappa$ and $e_\kappa$ window.

## §T.4 Tiering and the merge queue

### Proposition 19 (tier routing lowers the head-of-line load; v2.1)
At $B = 1$, $r = 0$, let the merge-group CI latency depend on the entry's tier class: T0-only, selective and full with shares $s_0, s_1, s_3$ and mean latencies $T_0, T_1, T_3$. Let $\bar T = \sum_k s_k T_k$. Then
$$\rho_{\text{HOL}} = \lambda'\,\bar q'\,\bar T,$$
where $\bar q'$ is the defective fraction of entries, which routing does not change (Thm 14 and 15: a skipped test has its base outcome). The ratio against all-T3 routing is $\bar T / T_3$.

*Proof.* At $B = 1$ an entry's failure blocks the lane for the length of its own run (Lemma 7). A sound router changes an entry's run length, not its outcome. So the expected blocked time per entry is $\sum_k s_k\,q'_k\,T_k$. If defects are independent of class, $q'_k = \bar q'$, which gives the result. If they are not, $\rho_{\text{HOL}} = \lambda' \sum_k s_k q'_k T_k$, and QM-15's per-class measurement supplies $q'_k$. ∎ `pv: queue-model-v1 / POST-P19`

*Illustration `[C]` from `[A]`.* The v2.0 mix of 25 / 55 / 20% at 2 / 6.7 / 9.5 min (§H.6 P3) gives $\bar T = 6.085$ min, a ratio of 0.64 to $T_3$. Against today's 14 min run (§1) it is 0.435, so $\rho_{\text{HOL}}$ falls about 57%. This needs R2-13: the merge-group diff is taken against `merge_group.base_sha`, and $\mathcal I$ must be valid there (Def. 14). Without that, the router is unsound in exactly the setting where it pays most. This proposition supersedes v1.1 §5.1 Prop. 11. The review side (v1.1 Prop. 12) is Prop. 18 with the R2-12 measurand.

---

## §7 Policy (the decisions)

| # | Decision | Value | Basis | Holds until |
|---|---|---|---|---|
| 1 | Merge-queue group size | **$B = 1$**, $L = 8$ | Thm 6 (+ Remark) | $\lambda' \ge 0.8\,L/T$ (Prop. 9) |
| 2 | Merge-queue retries | **0** | Thm 8; Prop. 10 at $B = 1$ ($\varphi^\*_{\text{MQ}} = 0.0512$; R-10) | measured $\varphi > \varphi^\*$ |
| 3 | Release PR | one fold per push; retries 0 after FLAKE-0's disable list lands | Thm 2; Cor. 3 | identification rate < 0.9 (S-4) |
| 4 | **Job shape** | **compile once per PR at width $t^\* \approx \lceil W/L_c\rceil$: 16 `[A]`, a deliberate round-down from 17 so that two slots fit intel's 32 threads (R2-1), archive, fan out as 8-thread shards; shared cache** | Thm 11; Prop. 12a; Thm 12 table | — |
| 5 | Shard count | $K$ from Cor. 13a, longest-first by measured test time; split any test group > $\tau_s/2$ | Thm 13 | — |
| 6 | Runner shapes | org-wide `arch-cN-mM` labels; $N_c$ by square-root staffing; systemd slices with CPUQuota/MemoryMax; forjar-declared | Prop. 14a; §H.5 | QM-12 residual squander > 5% |
| 7 | Arch-neutral work | T0 and T1 on gx10 and mini first; aarch64 and macOS legs; serving on gx10 | §H.5 | — |
| 8 | **Tiers by diff** | $D \cap \mathcal I = \emptyset$ → **T0 only**. $D \cap \mathcal C \ne \emptyset$ → T3. Otherwise T0 ∥ T1 ∥ T2 (parallel-cancel), escalating to T3 when $\mathcal R'$ covers ≥ 50% of test time `[A]` | Thms 14, 15, 17 | a soundness escape (S-8) |
| 9 | Merge-queue CI | same tier routing on the merge-group diff **against `merge_group.base_sha`** (the entry ahead, not `main`), with $\mathcal I$ valid at that base (Def. 14; R2-13) | Thms 14, 15; Prop. 19 | — |
| 10 | Nightly `main` | T3 full + $\mathcal F$ re-trace + selection audit | Thm 14 backstop | never removed |
| 11 | Release PR and tag | T3 + clean-room, always | doctrine | never removed |
| 12 | Review tier | **approved (operator, 2026-09-27):** a lighter tier for docs-only PRs and #4462-vouched contributors, switched on per class when Prop. 18 clears on measured $n, e$ (R2-12 measurand); full quorum until then | Prop. 18 | automatic revert to full review on the first escape in the class |
| 13 | Queue health | GREEN $\rho_{\text{HOL}} \le 0.4$ · AMBER ≤ 0.6 · RED > 0.6 `[C]` | Thm 8; §6 | QM-05 recalibration |
| 14 | **Rebase before arm** (operator amendment 2026-09-27) | author checklist: `gh pr update-branch N --rebase` (GitHub-side, never a local force-push) → CI → quorum on the rebased head → arm. On a conflict, a merge-`main` commit with a per-hunk resolution table, re-reviewed by the owner. Targets: PR age p90 < 24 h; 0 PRs DIRTY > 4 h | Prop. 20 | measured $1 - e^{-\kappa\mu_D w} > q_0$: conflicts inside the window itself dominate, so shorten $w$ |

## §8 STOP conditions

- **S-1** A change to CI, ruleset or runner settings would land while a release cut is in progress.
- **S-2** A decision would be taken from a `[U]` input without a §7 fallback.
- **S-3** Measured $\rho_{\text{HOL}} > 0.6$ under the §7 policy.
- **S-4** The release-cycle identification rate is below 0.9.
- **S-5** A harness hook refuses writes for a reason other than a missing ticket.
- **S-6** Two consecutive non-passing quorums on the same PR.
- **S-7** Budget $K$ reached, or the andon crossed with more than one row incomplete.
- **S-8** **Soundness escape:** nightly T3 fails on a test that a selective or docs-lane run on the introducing commit skipped. Freeze tier skipping repo-wide (every PR runs T3) until $\mathcal I$ or $\mathcal R'$ is fixed and a planted reproduction goes RED.
- **S-9** A tier budget timeout fires 3 times in 24 h: the budget or the shard count is wrong, so re-derive it (Cor. 13a).
- **S-10** A runner shape would be declared outside forjar, or without a cgroup limit.

## §9 Tickets (EV-ordered)

| EV | Row | Work | Contract | Done when (all must hold) | K̂ |
|---|---|---|---|---|---|
| 0 | **QM-00** model xtask | `cargo xtask queue-model`: Lemma 1, Thm 1, Cor. 3, Lemma 5, Thm 6, Thm 8, Prop. 9, Prop. 10 (v2.0 form), Prop. 19, Prop. 20 | `queue-model-v1` | golden tests reproduce §6 to 4 significant figures **and agree with QM-04's simulation within 3 standard errors** on every §6 row (R-6: §6 has no other oracle); mutants RED (six; R2-15): drop $d = 0$ in $E_f$; `max` → `sum` in Lemma 1; use $1/(1-\varphi)$ for every $r$ (Thm 1's $S_r$); use $\varphi$ instead of $\varphi(1-\varphi)$ for the rescued-flake probability in Prop. 10 (the v1.0 error, R-1); re-add the double-counted $\varphi^2\rho$ (v2.0 correction); linearise Prop. 20 to $\kappa\mu_D a$ (RED at $\mu_D = 0.05$/h, $a = 24$ h: 1.2 against 0.70) | 150 |
| 0 | **QM-01** queue inputs | #4513: 7-day $T, C, q, q', \varphi, \lambda, \lambda', f, F, \rho_{\text{rel}}, \rho_{\text{MQ}}, R$ and identification rate (§9.1), plus $\mu_D$ (per merged PR: `main` commits touching its files while it was open), PR age p90 and base age at merge-group entry p90 (Prop. 20) | `queue-inputs-v1` | 0 `[U]`; empty window RED | 120 |
| 0 | **QM-04** queue sim (moved from EV 3; QM-00's oracle, R-6) | `cargo xtask queue-sim` (§6 model), independent code path from QM-00 | `queue-sim-v1` | §6 p50 within ±10%; "no cancellation" mutant changes the current-config p90 by > 5× | 180 |
| 0 | **QM-08** fleet inputs | Per host: threads, RAM, arch (forjar facts). Per workload: $W$ and $L_c$ from `cargo build --timings` (unit graph), peak cgroup anon RAM per compile width {8, 16, 24}, per-test durations from nextest JUnit, cache hit rate $\eta$ | `fleet-inputs-v1` | G7 RAM and G12 filled; ≥ 5 samples each; receipts carry host and sha | 150 |
| 1 | **QM-02** queue settings (filed) | Ruleset 17836320 → group 1 / concurrency 8; `merge_group` nextest profile retries 0; lint | `mq-settings-v1` | ruleset export receipt; planted `retries = 2` is RED | 45 |
| 1 | **QM-09** compile-once | Workflow: one `--workspace` test build at width $t^\*$ → `cargo nextest archive` → $K$ shards from the archive (Cor. 13a, LPT); shared cache wired (infra#670) | `ci-compile-once-v1` | Σ tests passed equals the pre-change count, **with doctests counted separately** (never-down invariant; R2-10); latency and thread-s receipts over ≥ 5 PRs; a planted test deleted from the archive is RED | 240 |
| 1 | **QM-10** input set $\mathcal I$ | Nightly: collect dep-info, trace test run-time reads, compute $\mathcal C$; publish `ci/input-set.json` (single writer: nightly) | `ci-input-set-v1` | a planted `include_str!` of a docs file puts that file in $\mathcal I$; a planted test reading `docs/x.md` puts it in $\mathcal F$; a planted test that globs `docs/specifications/*.md` puts the prefix in $\mathcal F$, and a PR adding a file there is **not** docs-only; a planted `Path::exists` probe of an absent file puts that path in $\mathcal F$ (R2-9); a planted code commit between nightly and base that `include_str!`s a docs file forces an $\mathcal I_{\text{dep}}$ refresh (R2-8) | 180 |
| 2 | **QM-11** tier router | Classifier: T0-only / T3 / selective by Def. 14 and Thm 15; T0 ∥ T1 ∥ T2 parallel-cancel (Thm 17); T1 sub-steps ordered by Thm 16 | `ci-tier-router-v1` | planted docs-only PR runs T0 only; planted README (include_str'd) change runs selective; planted `Cargo.toml` change runs T3; planted change to crate X runs exactly $\mathcal R'(X)$ | 240 |
| 2 | **QM-12** runner shapes (filed) | Infra issue: shape labels, square-root staffing counts from the LP at the design $\lambda$, first-fit-decreasing placement, systemd slices, forjar declarations for intel/framework16/yoga/gx10/mini; residual-squander telemetry | `runner-shapes-v1` | forjar-declared; cgroup limits verified live (`systemctl show`); squander ≤ 5% over 7 days; max queue age per shape reported, with an aging rule so no shape starves (R2-6) | 180 |
| 2 | **QM-13** fleet LP + sim | `cargo xtask fleet-lp` (Thm 12) and `cargo xtask fleet-sim` (§H.6 model) in Rust | `fleet-model-v1` | reproduces the §H.4 $\lambda^\*$ values to 3 significant figures and §H.6 p50 within ±10%; mutant "RAM not enforced" changes the P1 row by > 3× | 240 |
| 3 | **QM-03** release retries | Cor. 3 at measured inputs, after FLAKE-0 lands | extends `queue-model-v1` | receipt; setting matches the xtask | 30 |
| 3 | **QM-05** queue-health gate | Nightly $\rho_{\text{HOL}}$, $\rho_{\text{lane}}$ from trailing inputs; RED > 0.6 | `queue-health-v1` | planted B=8, r=2, q=0.1, 6/h is RED; empty input window is RED | 90 |
| 4 | **QM-14** soundness audit | Nightly: for every commit that took T0-only or selective, compare against T3 outcomes; any mismatch fires S-8 | `ci-tier-soundness-v1` | planted unsound classifier (fixtures dir removed from $\mathcal F$) is caught within 1 nightly | 120 |
| 4 | **QM-15** review-tier evidence | Measure $n_\kappa, e_\kappa$ for docs-only and vouched-contributor (#4462) classes; evaluate Prop. 18; present to the operator | `review-tier-evidence-v1` | $\bar p_\kappa$ with $n$ receipted, $e_\kappa$ counting quorum-caught findings as well as post-merge regressions (R2-12); per-class $q'_k$ for Prop. 19; the class switches on only when $\bar p_\kappa$ clears; a planted escape flips the class back to full review within one PR | 60 |
| 5 | **QM-06** cycle time | $C$ before and after QM-09 (≥ 5 cycles each), recording $C$, $F$ and $\varphi$ per cycle (R-9); confirm the 15–25 min CI-given-a-free-slot bound (R2-5); measure review latency, queue wait and the doctest job as the new long pole | — | receipts | 30 |
| 5 | **QM-07** identification rate | A2 measurement; S-4 evaluation | extends `queue-model-v1` | rate with $n$ | 60 |
| 6 | **QM-16** arch-neutral utilisation | Route T0/T1 to gx10 and mini; report their pool utilisation | extends `runner-shapes-v1` | gx10 CPU ≥ 30% during PR hours when serve is idle `[A]`; mini running T0 | 60 |

**K̂ = 2,175 `[A]` · K = 2,400 · andon at 1,920.**

### §9.1 Measurement commands

| Input | Command |
|---|---|
| $T$, $q'$, $q$ | `gh run list -R paiml/aprender --event merge_group --status completed --limit 500 --json createdAt,updatedAt,conclusion`. Count only `success`/`failure` conclusions (drop `cancelled`, `skipped`), R-11. The failure ratio is $q'$ (per entry), not $q$. Back out $q = q'f/(1-q')$ (Q4, R-12) |
| $\varphi$ | FLAKE-0 ledger |
| $\lambda$, $\lambda'$ | $\lambda$ from `gh pr list -R paiml/aprender --state all --limit 1000 --json createdAt`. $\lambda'$ from merge-queue entry events (`added_to_merge_queue` timeline items), re-entries included. Cross-check $\lambda' = \lambda(1+q/f)$ |
| $\rho_{\text{rel}}$, $\rho_{\text{MQ}}$ | retry-attempt durations from job `run_attempt` > 1, split by event (`pull_request` on the release PR vs `merge_group`); never pooled (R-5) |
| $W$, $L_c$ | `cargo build --workspace --tests --timings=json` on a clean target, per compile width; $W$ = Σ unit durations × unit threads, $L_c$ = longest path in the unit graph |
| Compile RAM | cgroup `memory.stat` anon peak of the build slice (not `memory.peak`) |
| $p_i$ | nextest JUnit durations from the last 7 nightly T3 runs |
| $\eta$ | sccache `--show-stats` per PR build |
| Host facts | forjar facts per machine (`nproc`, `MemTotal`, arch) |
| Squander | runner API job `queued_at`/`started_at` joined with per-host free-thread/RAM samples at 60 s |

## §10 Final report schema (one per session)

```yaml
spec: FLOW-003
version: 2.1
session: {date_utc, host, tree: {head, origin_main, worktree}, model}
selected_row: QM-NN
ticket: PMAT-NNNN
outcome: merged | stopped | already-done | premise-falsified
stop: {id: S-n, evidence: "..."}
queue_inputs: {T, C, q, q_eff, phi, lambda_per_h, lambda_eff_per_h, f, F, rho_rel, rho_mq, R, ident_rate, mu_D_per_h, pr_age_p90_h, base_age_p90_h, window_days}
fleet_inputs: {hosts: [{name, arch, threads, ram_gb}], W, L_c, compile_ram_by_width, eta, test_time_sum, test_time_max}
derived: {rho_hol_by_tier_mix, EM, E_T_fold_by_r, rho_hol, rho_lane, t_star, K_shards, lambda_star_by_shape, squander_pct}
tiers: {docs_lane_share, selective_share, full_share, soundness_escapes}
decisions: {B, L, r_mq, r_release, shape, labels, tier_routing}
issues_filed: [{repo, url, purpose}]
scoreboard_moved: [{metric, before, after, command}]
budget: {k_hat_row, k_actual_row, cumulative, andon_crossed: bool}
next_row: QM-NN
```

### §10.1 Scoreboard (targets and probes only; state is rendered, never written here)

| Metric | Target | Rendered by |
|---|---|---|
| Fleet capacity $\lambda^\*$ (LP at measured inputs) | **≥ 20 PR CI runs/h** (today ≈ 3.4 `[C]` with the RAM row, 3.8 on threads alone; R2-4) | QM-13 |
| Docs-only PR CI latency | **≤ 2 min** | QM-11 |
| Code PR CI latency p50 / p90 | **≤ 10 / ≤ 15 min** | QM-11 |
| Full tier (T3) latency | **≤ 26 min** CI given a free slot (Thm 11 + Cor. 13a bound: 25.1; R2-5) | QM-06 |
| Thread-s per PR (mean) | **≤ 12,000** (today ≈ 56,000 `[C]`) | QM-13 |
| x86 host CPU during PR hours | **60–85%**, none < 40% while jobs queue | QM-12 |
| Residual squander | **≤ 5%** of host-time | QM-12 |
| gx10 arch-neutral utilisation when serve is idle | **≥ 30%** | QM-16 |
| Soundness escapes | **0** | QM-14 |
| $\rho_{\text{HOL}}$ | **≤ 0.4** | QM-05 |
| Merge-queue wait p50 / p90 | **≤ 15 / ≤ 120 min** | QM-04 vs measured |
| Release pushes per green head | **≤ 1.5** | QM-01 |
| PR age p90 (open → merge) | **< 24 h** (Prop. 20; §7 row 14) | QM-01 |
| PRs DIRTY (merge conflict) for > 4 h | **0** | QM-05 |
| Flake rate $\varphi$ | **≤ 0.01** | FLAKE-0 |
| Inputs marked `[U]` | **0** | QM-01, QM-08 |
| Python lines in files this spec touches | **0** | `grep -rl python3` over the diff |

---

## §11 Ontology: the contracts that hold this spec (`pv`, ONT-001 v1alpha1)

These are the YAML `pv` contracts that the §9 rows land under `contracts/`. The ids match §9's Contract column. Every contract stays `proof.status: declared` until its row lands, so `pv` reports `Unknown`, and `Unknown` never arms. Nothing here claims `discharged`.

### §11.1 The spec itself (`entity: spec`, ONT-001 B.11)

```yaml
id: DOC-spec-flow-003
metadata: { ontology_version: ont.paiml.dev/v1alpha1 }
entity: { type: spec, ref: docs/specifications/FLOW-003-queue-and-release-cycle-model.md }
shape:
  closed: true
  properties:
    - { path: spec:id,        minCount: 1, maxCount: 1, pattern: "^FLOW-003$" }
    - { path: spec:row,       minCount: 17, maxCount: 17, pattern: "^QM-(0[0-9]|1[0-6])$" }
    - { path: spec:stop,      minCount: 10, pattern: "^S-([1-9]|10)$" }
    - { path: spec:falsifier, minCount: 1 }                        # every row's done-when names a RED case
    - { path: spec:contract,  minCount: 12, resolves: pv-contract } # §9 Contract column → a file under contracts/
proof: { status: not_applicable }
evidence: { level: L1, provenance: { wasGeneratedBy: { command: "pv extract docs/specifications/FLOW-003-queue-and-release-cycle-model.md", git_sha: "<census sha>" }, wasAttributedTo: orchestrator, generatedAtTime: "<from git>" } }
# The spec extractor is `declared` in ONT-001 (Q12), so today this validates as Unknown{ExtractorMissing: spec}.
```

### §11.2 The queue model (`queue-model-v1`, bound to QM-00's code)

```yaml
id: queue-model-v1
metadata: { kind: kernel, ontology_version: ont.paiml.dev/v1alpha1 }
entity: { type: code, ref: xtask::queue_model }        # Unknown{SymbolMissing} until QM-00 lands
relations: { depends_on: [queue-sim-v1] }              # the oracle (R-6)
requires:
  - { id: PRE-1, statement: "probabilities in the open unit interval", formal: "0 < q < 1 && 0 < f <= 1 && 0 <= phi < 1", formal_status: parsed }
  - { id: PRE-2, statement: "times non-negative, C > 0", formal: "C > 0 && T > 0 && F >= 0 && rho_rel >= 0 && rho_mq >= 0 && R >= 0", formal_status: parsed }
  - { id: PRE-3, statement: "B = 2^h with full groups (Lemma 5 scope)", formal: "exists h: Nat. B = 2^h", formal_status: parsed }
  - { id: PRE-4, statement: "Q5: false re-entry negligible, else lambda_eff carries the term", formal_status: prose }
ensures:
  - { id: DEF-Q4,   statement: "effective arrival and defect rate", formal: "lambda_eff = lambda*(1+q/f) && q_eff = q/(f+q)", formal_status: parsed }
  - { id: POST-L1,  statement: "expected defect cycles", formal: "EM(k) = sum_{n>=0} 1 - (1 - q*(1-f)^n)^k", formal_status: parsed }
  - { id: POST-T1,  statement: "expected time to green", formal: "ET(k,r) = EM(k)*(C + r*rho_rel + F) + (C + rho_rel*S_r)/(1 - phi^(r+1))", formal_status: parsed }
  - { id: POST-T2,  statement: "one fold strictly dominates any SEQUENTIAL split, m >= 2 (R-8)", formal_status: prose }
  - { id: POST-C3,  statement: "r=1 beats r=0 iff phi*(C-rho_rel)/(1-phi^2) > rho_rel*EM", formal_status: parsed }
  - { id: POST-C4,  statement: "dET/dC = EM + 1/(1-phi^(r+1)) >= 1, holding F, q, phi fixed (R-9)", formal_status: parsed }
  - { id: POST-L5,  statement: "E_t, E_f closed forms; E_ff >= lower bound", formal_status: parsed }
  - { id: POST-T6,  statement: "E_f(B)/B and the E_ff lower bound /B strictly increase in h", formal_status: parsed }
  - { id: POST-L7,  statement: "head-of-line windows disjoint, no merge inside", formal_status: prose }
  - { id: POST-T8,  statement: "rho_hol >= 1 => not positive recurrent (converse drift; Lamperti for > 1) (R-4)", formal_status: prose }
  - { id: POST-P9,  statement: "rho_lane < 1 necessary; at B=1: lambda_eff*(T + r*rho_mq*q_eff) < L", formal_status: parsed }
  - { id: POST-P10, statement: "at B=1, r=1 beats r=0 iff (1-q_eff)*phi*((1-phi)*(T+R) - rho_mq) > q_eff*rho_mq", formal_status: parsed }
  - { id: POST-P19, statement: "at B=1, r=0 under a sound router: rho_hol = lambda_eff * sum_k s_k*q_eff_k*T_k", formal_status: parsed }
  - { id: POST-P20, statement: "P(conflict) = 1 - exp(-kappa*mu_D*a) <= 1 - exp(-mu_D*a); q(a) = 1 - (1-q0)*exp(-kappa*mu_D*a)", formal_status: parsed }
invariants:
  - { id: INV-1, statement: "every §7 decision is a pure function of the inputs JSON; no decision reads a [U] input without a §7 default (S-2)", formal_status: prose }
shape:
  properties:
    - { path: ont:binds,     minCount: 1, resolves: symbol }
    - { path: ont:falsifier, minCount: 6 }             # QM-00's planted mutants (R2-15)
proof:
  status: declared
  tests: { golden: "§6 rows, 4 s.f.", oracle: queue-sim-v1, tolerance: "3 standard errors" }
evidence: { level: L1, provenance: { wasGeneratedBy: { command: "cargo xtask queue-model --self-test", git_sha: "<QM-00 merge sha>" }, wasAttributedTo: "lane:aprender", generatedAtTime: "<from git>" } }
```

### §11.3 The queue inputs (`queue-inputs-v1`, QM-01's receipt)

```yaml
id: queue-inputs-v1
entity: { type: json, ref: docs/receipts/flow-003/queue-inputs.json }
shape:
  closed: true
  properties:
    - { path: qin:input,        minCount: 17, maxCount: 17 }   # the §10 queue_inputs keys
    - { path: qin:value,        minCount: 1, datatype: xsd:decimal }
    - { path: qin:n,            minCount: 1, datatype: xsd:integer, minInclusive: 1 }  # an empty window is RED
    - { path: qin:command,      minCount: 1, resolves: ci-step }
    - { path: qin:mark,         in: ["[V]", "[C]", "[A]"] }  # "[U]" is excluded: 0 unmeasured
    - { path: qin:measuredRate, in: [q, q_eff, lambda, lambda_eff] }  # which rate was measured (R-12)
    - { path: qin:conclusions,  in: [success, failure] }     # cancelled/skipped runs are not counted (R-11)
proof: { status: not_applicable }
evidence: { level: L2, provenance: { wasGeneratedBy: { command: "<the §9.1 command per input>", git_sha: "<sha>" }, wasAttributedTo: "lane:aprender", generatedAtTime: "<window end>" } }
```

### §11.4 The input set (`ci-input-set-v1`, QM-10; the soundness hinge)

Theorems 14 and 15 are only as sound as this contract, so it is written out in full.

```yaml
id: ci-input-set-v1
metadata: { ontology_version: ont.paiml.dev/v1alpha1 }
entity: { type: json, ref: ci/input-set.json }          # single writer: the nightly
shape:
  closed: true
  properties:
    - { path: cis:baseSha,     minCount: 1, maxCount: 1, pattern: "^[0-9a-f]{40}$" }
    - { path: cis:dep,         minCount: 1 }            # I_dep: rustc dep-info paths, full T3 build
    - { path: cis:read,        minCount: 1 }            # F (i): opened / stat'd paths
    - { path: cis:dirPrefix,   minCount: 0 }            # F (ii): enumerated directories, as dir/** (R2-9)
    - { path: cis:absent,      minCount: 0 }            # F (iii): ENOENT lookups (R2-9)
    - { path: cis:config,      minCount: 1 }            # C
    - { path: cis:traceSyscalls, minCount: 1, in: [openat, open, newfstatat, statx, access, getdents64, fanotify] }
    - { path: cis:nightlies,   minCount: 1, datatype: xsd:integer, minInclusive: 7 }
requires:
  - { id: PRE-T3,   statement: "the trace is of a full T3 run on baseSha, green", formal_status: prose }
invariants:
  - { id: INV-DIR,  statement: "a path under any dirPrefix counts as in F, including paths that do not yet exist", formal_status: prose }
  - { id: INV-BASE, statement: "input-set(n) is used at base b only if every commit in n..b took the docs lane; otherwise I_dep is refreshed from T1 dep-info on b and R(D_{n..b}) joins R' (R2-8)", formal_status: prose }
ensures:
  - { id: POST-COVER, statement: "every file, listing and failed lookup a test performs on baseSha is covered by read ∪ dirPrefix ∪ absent", formal_status: prose }
# ont:falsifier (QM-10 done-when; each resolves to a ci-step fixture that must go RED):
#   - "planted include_str! of docs/x.md puts docs/x.md in dep"
#   - "planted test reading docs/x.md puts it in read"
#   - "planted glob of docs/specifications/*.md puts docs/specifications/** in dirPrefix"
#   - "planted Path::exists probe of an absent file puts it in absent"
#   - "planted code commit between nightly and base forces a refresh (INV-BASE)"
proof: { status: declared }
evidence: { level: L1, provenance: { wasGeneratedBy: { command: "cargo xtask input-set --nightly", git_sha: "<QM-10 merge sha>" }, wasAttributedTo: "lane:aprender", generatedAtTime: "<nightly end>" } }
```

### §11.5 The policy (`POLICY-flow-003-queue`, unanchored; ONT-001 B.2)

```yaml
id: POLICY-flow-003-queue
relations: { depends_on: [queue-model-v1, queue-inputs-v1, fleet-model-v1, ci-input-set-v1, ci-tier-soundness-v1] }
requires:
  - { id: PRE-1, statement: "no row changes release-PR CI, the ruleset or runner settings while a cut is in progress (S-1)", formal_status: prose }
ensures:   # one clause per §7 row
  - { id: POST-B,     statement: "merge-queue group size 1, concurrency 8 (§7.1)", formal_status: prose }
  - { id: POST-RMQ,   statement: "merge_group retries 0 while phi <= phi*_MQ = 0.0512 (§7.2, Prop. 10 at B=1)", formal_status: prose }
  - { id: POST-FOLD,  statement: "one fold per release push while ident_rate >= 0.9 (§7.3)", formal_status: prose }
  - { id: POST-SHAPE, statement: "compile once at width ~t*, archive, 8-thread shards (§7.4)", formal_status: prose }
  - { id: POST-K,     statement: "K from Cor. 13a, LPT (§7.5)", formal_status: prose }
  - { id: POST-RUN,   statement: "runner shapes forjar-declared with cgroup limits and an aging rule (§7.6, S-10, R2-6)", formal_status: prose }
  - { id: POST-TIER,  statement: "tier by diff per Thms 14/15, merge-group diff against merge_group.base_sha (§7.8–7.9)", formal_status: prose }
  - { id: POST-NIGHT, statement: "nightly T3 + trace + audit never removed (§7.10)", formal_status: prose }
  - { id: POST-REL,   statement: "release PR and tag: T3 + clean-room always (§7.11)", formal_status: prose }
  - { id: POST-REV,   statement: "lighter review for docs-only / vouched classes (operator-approved 2026-09-27) only while Prop. 18 clears; automatic revert to full on the first escape (§7.12)", formal_status: prose }
  - { id: POST-HOL,   statement: "rho_hol GREEN <= 0.4, AMBER <= 0.6, RED > 0.6 (§7.13)", formal_status: prose }
  - { id: POST-REBASE, statement: "rebase (GitHub-side) -> CI -> quorum on the rebased head -> arm; no force-push; PR age p90 < 24 h; 0 PRs DIRTY > 4 h (§7.14, Prop. 20)", formal_status: prose }
shape:
  closed: true
  properties:
    - { path: ont:falsifier, minCount: 3, resolves: ci-step }  # mq-settings-v1 lint; queue-health-v1 planted RED; ci-tier-soundness-v1 planted unsound classifier
proof: { status: not_applicable }
evidence: { level: L1, provenance: { wasGeneratedBy: { command: "cargo xtask queue-model --inputs docs/receipts/flow-003/queue-inputs.json", git_sha: "<sha>" }, wasAttributedTo: orchestrator, generatedAtTime: "<from git>" } }
```

### §11.6 The other contracts (written by their rows)

Each has the same pattern: its §9 done-when is its `ont:falsifier`, with `resolves: ci-step` to the fixture that must go RED.

| Contract | Row | Entity | Clauses it must carry |
|---|---|---|---|
| `fleet-inputs-v1` | QM-08 | json receipt | per host threads/RAM/arch; $W$, $L_c$ per width {8, 16, 24} as CPU-seconds (H-a, R2-4); anon RAM peak; $\eta$; ≥ 5 samples; no `[U]` |
| `ci-compile-once-v1` | QM-09 | workflow | PRE same-fingerprint artifacts (R2-3); INV never-down for tests **and** doctests separately (R2-10); POST-P12a |
| `ci-tier-router-v1` | QM-11 | code | POST-T16, POST-T17; routes by `ci-input-set-v1` at the entry's base (R2-8, R2-13) |
| `runner-shapes-v1` | QM-12, QM-16 | forjar | POST-P14a; INV cgroup limits live; max queue age per shape bounded (R2-6) |
| `fleet-model-v1` | QM-13 | code | POST-T11 under (H-a); POST-T12 with the reservation note (R2-4); Cor. 13a; §H.6 rows above $\lambda^\*$ flagged transient (R2-7) |
| `ci-tier-soundness-v1` | QM-14 | code | POST-T14, POST-T15; any mismatch → S-8 |
| `review-tier-evidence-v1` | QM-15 | json receipt | POST-P18 with $e_\kappa$ = post-merge escapes + quorum-caught findings (R2-12); per-class $q'_k$ |
| `mq-settings-v1`, `queue-sim-v1`, `queue-health-v1` | QM-02, QM-04, QM-05 | as v1.1 | unchanged |

---

## §12 Review record

### §12.1 v1.0 → v1.1 findings, and where they stand in v2.1

| # | Finding (v1.1) | v2.0 | v2.1 |
|---|---|---|---|
| R-1 | Prop. 10 counted $\rho$ twice; $\varphi^\*_{\text{MQ}}$ is 0.0512, not 0.0525 | applied | kept |
| R-2 | Thm 6 monotonicity proved only for $\underline E_{ff}$ | applied | kept |
| R-3 | Q1–Q3 (bisecting groups) may not match GitHub | — | `[U]` note in §3 |
| R-4 | Thm 8 cited forward drift criteria for a converse | partly (restated) | converse drift (Tweedie; Meyn–Tweedie) + Lamperti cited `[X]`; $\underline E_{ff}$ in $\rho_{\text{HOL}}$ |
| R-5 | §6 rows 3–4 not fully parameterised | — | applied |
| R-6 | §6 golden values had no in-repo oracle | — | QM-04 at EV 0 as QM-00's oracle (3 s.e.) |
| R-7 | false re-entries missing from $\lambda'$ | — | Q5 |
| R-8 | Thm 2 covers sequential splits only | — | scope paragraph |
| R-9 | Cor. 4 holds $F$, $\varphi$ fixed | — | scope paragraph; QM-06 records $C$, $F$, $\varphi$ |
| R-10 | §7 retries row cited the $B=8$ load | — | Prop. 10 at $B=1$ |
| R-11 | $\rho$ held two values | — | $\rho_{\text{rel}}$ / $\rho_{\text{MQ}}$ in §2, §6, §9.1, §10 |
| R-12 | merge_group failure ratio measures $q'$; PR creations measure $\lambda$ | — | §2 row, §9.1, §10, `queue-inputs-v1` |
| A-1 | v1.1 §5.1 service classes (operator question, #4462/#4467) | superseded by Part III | Prop. 11 → Prop. 19; Prop. 12 → Prop. 18 with the R2-12 measurand; v1.1's per-class-$q$ row → QM-15 (see R2-14) |

### §12.2 v2.0 → v2.1 findings (Parts II–III)

All arithmetic in §H.1–§H.6 and §T was recomputed. The LP ratios (7.0, 8.6, ×2.5, +23%) and the proofs of Theorems 11, 13, 16 and 17 check out. Theorem numbering (12a before 12; 14a beside 14) is irregular but left as is, so the ticket references stay stable.

| # | Sev. | Where | Finding | v2.1 change |
|---|---|---|---|---|
| R2-1 | minor | Cor. 11a, §7.4 | $\lceil 9700/600 \rceil = 17$, not 16. At 16 the bound is $2.01\,L_c$ | 16 kept as a stated round-down (two slots on 32 threads) |
| R2-2 | minor | Thm 11 | assumes unit durations independent of $P$ and one token per unit | assumption (H-a); $W$ measured per width |
| R2-3 | minor | Prop. 12a | "exactly" needs identical artifacts; $t_s$ undefined | precondition; $o$, $t_s$ defined |
| R2-4 | major | Thm 12 | the RAM row is necessary only under reservation (P1–P3), and the thread row must be CPU-seconds, not requested threads | note; P0 thread-only 3.8/h `[C]` |
| R2-5 | major | §H.4 | the 329 s shard figure had no derivation (Cor. 13a gives ≤ 299 s), and "$C \le 30$ by proof" ignores review, queue wait and T0/T1 | 15.1–25.1 min, restated as a CI-given-a-free-slot bound; QM-06 |
| R2-6 | minor | Prop. 14a | zero squander doesn't bound waiting: backfill can starve large jobs | aging rule; QM-12 max queue age |
| R2-7 | minor | §H.6 | rows above $\lambda^\*$ are transients | labelled |
| R2-8 | **soundness** | Def. 14, Thm 14 | $\mathcal I$ from the nightly is stale at a later base | validity-at-base rule, INV-BASE, QM-10 falsifier |
| R2-9 | **soundness** | Def. 14 | an `openat`/`stat` trace can't see directory listings or ENOENT lookups, so adding a file to a globbed directory escapes | trace `getdents64` and failed lookups; fixtures rule dropped; QM-10 falsifiers |
| R2-10 | major | T2/T3, QM-09 | nextest archives don't run doctests `[X]` | a separate doctest job; never-down invariant counts it separately |
| R2-11 | nit | Thm 15 | build-script `rerun-if-changed` and doctest inputs | edges paragraph |
| R2-12 | major | Prop. 18 | $e_\kappa$ counted only what full review missed, which biases the bound low | measurand includes quorum-caught findings |
| R2-13 | major | §7.9 | the merge-group diff must be taken against `merge_group.base_sha` | row 9; Prop. 19 |
| R2-14 | process | §9 | QM-08 means "fleet inputs" in v2.0, but aprender#4519 was minted as QM-08 for v1.1's per-class $q$, which maps to v2.0's QM-15 | **not renumbered here.** Flagged to the cop, the single minter, for re-mapping |
| R2-15 | major | §9 QM-00; §11.2 | v2.0 cut QM-00's planted mutants from v1.1's four to three (dropping the $1/(1-\varphi)$ retry mutant and the Prop. 10 rescued-flake mutant), and nothing recorded it: a gate weakened silently. Found by the v2.1 quorum (lane 2) | restored as the union of v1.1's four and v2.0's double-count mutant, plus one for Prop. 20; `ont:falsifier minCount` 3 → 6 |
