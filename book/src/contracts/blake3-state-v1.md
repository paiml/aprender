# blake3-state-v1

**Version:** 1.0.0

BLAKE3 content-addressed state hashing — tripwire integrity foundation

## References

- O'Connor et al. (2019) BLAKE3: One function, fast everywhere

## Equations

### composite_hash

$$
H(c₁, ..., cₙ) = 'blake3:' || hex(BLAKE3(c₁ || NUL || c₂ || NUL || ... || cₙ || NUL))
$$

**Domain:** $components \in [&str]$

**Codomain:** $String matching /^blake3:[0-9a-f]{64}$/$

**Invariants:**

- $Output always has prefix 'blake3:'$
- $Order-sensitive: H(a, b) \neq H(b, a) in general$
- $Deterministic: same inputs \to same hash$

### hash_file

$$
H(path) = 'blake3:' || hex(BLAKE3(read_all(path)))
$$

**Domain:** $path \in Path where path.exists() ∧ path.is_file()$

**Codomain:** $Result<String, String>$

**Invariants:**

- $Output always has prefix 'blake3:' on success$
- $Deterministic: same file contents \to same hash$
- $Returns Err for non-existent paths$

### hash_string

$$
H(s) = 'blake3:' || hex(BLAKE3(s.as_bytes()))
$$

**Domain:** $s \in String$

**Codomain:** $String matching /^blake3:[0-9a-f]{64}$/$

**Invariants:**

- $Output always has prefix 'blake3:'$
- $Output length = 71 (7 prefix + 64 hex)$
- $Deterministic: H(s) = H(s) for all s$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | All hashes have blake3: prefix | $\forall input: output.starts_with('blake3:')$ |
| 2 | invariant | Deterministic hashing | $\forall s: hash_string(s) = hash_string(s)$ |
| 3 | ordering | Composite hash is order-sensitive | $\exists a, b: composite_hash([a, b]) \neq composite_hash([b, a])$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-B3-001 | Prefix format | hash_string(s) starts with 'blake3:' and has length 71 for any s | Format string bug in hash_string |
| FALSIFY-B3-002 | Determinism | hash_string(s) = hash_string(s) for any random s | Non-deterministic state in hasher |
| FALSIFY-B3-003 | Order sensitivity | composite_hash([a, b]) ≠ composite_hash([b, a]) for distinct a, b | Missing NUL separator or sorting in composite_hash |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-B3-001 | Prefix format | 32 | exhaustive |

## QA Gate

**BLAKE3 State Contract** (F-B3-001)

Content-addressed hashing quality gate

**Checks:** prefix_format, determinism, order_sensitivity

**Pass criteria:** All 3 falsification tests pass

