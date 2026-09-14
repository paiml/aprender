# `breaks` goldens — provenance

`manifest.json` holds the tick placements that `tests/breaks_golden.rs` compares
`aprender_viz::breaks::extended` against. They are not hand-written and they are not this
crate's own output: they are **the reference implementation's output**, and this file is the
record of where they came from.

## Why goldens and not properties

Talbot, Lin & Hanrahan's objective weights are stated two different ways by the same author:

| source | simplicity | coverage |
|---|---|---|
| CRAN `labeling::extended` — `w[1]*s + w[2]*c`, `w = c(0.25, 0.2, 0.5, 0.05)` | 0.25 | 0.2 |
| `jtalbot/Labeling` `ExtendedAxisLabeler.cs` — `w[0]*s + w[1]*c`, `w = {0.25, 0.2, 0.5, 0.05}` | 0.25 | 0.2 |
| the paper's prose, §3.2 | 0.2 | 0.25 |

The two implementations agree with each other and disagree with the prose, so the
implementations are normative and `W_DEFAULT` is `[0.25, 0.2, 0.5, 0.05]`.

**A property suite cannot catch the swap.** Exchanging the first two weights leaves every
property intact — the labeling is still inside a sane range, still strictly increasing, still has
a uniform step whose mantissa is in `Q`, still carries about `m` labels — and changes only *which*
candidate labeling wins. Only a comparison against the reference's actual numbers detects it, which
is what these goldens are for, and
`mutation_swapping_simplicity_and_coverage_breaks_the_goldens` demonstrates that they do.

## The reference

Below is a **literal, line-for-line transcription** of the CRAN `labeling::extended` R source —
Justin Talbot's own code, the first author's. It is kept verbatim on purpose: the only thing that
makes `manifest.json` evidence rather than 36 arrays of magic numbers is that a reader can diff
this block against the R source and see that nothing was reinterpreted on the way through. It is
deliberately **not** refactored to match `src/breaks.rs` — an oracle that shares its structure with
the implementation under test can share its mistakes.

It lives in Markdown rather than in a `.py` file because it is not this project's source code: it
is an external artefact reproduced for provenance. Nothing in the build imports it and no CI job
runs it. It does not rot silently, though — `generate.py` **extracts and executes this exact
block**, so if it stops parsing or stops producing the committed goldens, regeneration fails.

```python
import math

EPS = 2.220446049250313e-16 * 100  # .Machine$double.eps * 100


def _simplicity(q, Q, j, lmin, lmax, lstep):
    n = len(Q)
    i = Q.index(q) + 1                      # match(q, Q)[1], 1-based
    v = 1 if ((lmin % lstep < EPS or lstep - (lmin % lstep) < EPS)
              and lmin <= 0 <= lmax) else 0
    return 1.0 - (i - 1) / (n - 1) - j + v


def _simplicity_max(q, Q, j):
    n = len(Q)
    i = Q.index(q) + 1
    return 1.0 - (i - 1) / (n - 1) - j + 1.0


def _coverage(dmin, dmax, lmin, lmax):
    rng = dmax - dmin
    return 1.0 - 0.5 * ((dmax - lmax) ** 2 + (dmin - lmin) ** 2) / ((0.1 * rng) ** 2)


def _coverage_max(dmin, dmax, span):
    rng = dmax - dmin
    if span > rng:
        half = (span - rng) / 2.0
        return 1.0 - 0.5 * (half * half + half * half) / ((0.1 * rng) ** 2)
    return 1.0


def _density(k, m, dmin, dmax, lmin, lmax):
    r = (k - 1) / (lmax - lmin)
    rt = (m - 1) / (max(lmax, dmax) - min(dmin, lmin))
    return 2.0 - max(r / rt, rt / r)


def _density_max(k, m):
    if k >= m:
        return 2.0 - (k - 1) / (m - 1)
    return 1.0


def _legibility(lmin, lmax, lstep):
    return 1.0


def extended(dmin, dmax, m, Q=(1, 5, 2, 2.5, 4, 3), only_loose=False,
             w=(0.25, 0.2, 0.5, 0.05)):
    Q = list(Q)
    if dmin > dmax:
        dmin, dmax = dmax, dmin
    if dmax - dmin < EPS:
        return [dmin + i * (dmax - dmin) / (m - 1) for i in range(m)]

    best = {"score": -2.0, "lmin": dmin, "lmax": dmax, "lstep": 0.0}

    j = 1
    while j < math.inf:
        stop_j = False
        for q in Q:
            sm = _simplicity_max(q, Q, j)
            if (w[0] * sm + w[1] + w[2] + w[3]) < best["score"]:
                stop_j = True
                break
            k = 2
            while k < math.inf:
                dm = _density_max(k, m)
                if (w[0] * sm + w[1] + w[2] * dm + w[3]) < best["score"]:
                    break
                delta = (dmax - dmin) / (k + 1) / j / q
                z = math.ceil(math.log10(delta)) if delta > 0 else 0
                while z < math.inf:
                    step = j * q * (10.0 ** z)
                    cm = _coverage_max(dmin, dmax, step * (k - 1))
                    if (w[0] * sm + w[1] * cm + w[2] * dm + w[3]) < best["score"]:
                        break
                    min_start = int(math.floor(dmax / step) * j - (k - 1) * j)
                    max_start = int(math.ceil(dmin / step) * j)
                    if min_start > max_start:
                        z += 1
                        continue
                    for start in range(min_start, max_start + 1):
                        lmin = start * (step / j)
                        lmax = lmin + step * (k - 1)
                        lstep = step
                        s = _simplicity(q, Q, j, lmin, lmax, lstep)
                        c = _coverage(dmin, dmax, lmin, lmax)
                        g = _density(k, m, dmin, dmax, lmin, lmax)
                        el = _legibility(lmin, lmax, lstep)
                        score = w[0] * s + w[1] * c + w[2] * g + w[3] * el
                        if score > best["score"] and (
                            not only_loose or (lmin <= dmin and lmax >= dmax)
                        ):
                            best = {"lmin": lmin, "lmax": lmax, "lstep": lstep,
                                    "k": k, "q": q, "j": j, "score": score}
                    z += 1
                k += 1
            if stop_j:
                break
        if stop_j:
            break
        j += 1

    n = int(round((best["lmax"] - best["lmin"]) / best["lstep"])) + 1
    return [best["lmin"] + i * best["lstep"] for i in range(n)]
```

## Regenerating

```sh
WT=$(git rev-parse --show-toplevel) python3 crates/aprender-viz/fixtures/breaks/generate.py
```

`generate.py` reads the block above out of this file, executes it, and rewrites `manifest.json`.
Regenerating must leave `manifest.json` byte-identical; a diff means either the transcription or
the domain list moved, and both are the reader's business.
