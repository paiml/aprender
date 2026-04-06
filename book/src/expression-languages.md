# Contract Expression Languages

Provable-contracts supports three expression language families for
preconditions, postconditions, and invariants. The language is
auto-detected from the YAML field structure.

## 1. Rust Expressions (Default)

Raw Rust boolean expressions. Variables in scope match equation
parameters; `result` is available in postconditions.

```yaml
preconditions:
  - 'x.iter().all(|v| v.is_finite())'
  - 'a.len() == m * k'
postconditions:
  - '(result.iter().sum::<f32>() - 1.0).abs() < 1e-5'
```

**Codegen:** `pv codegen` emits `debug_assert!` macros.

**Verification chain:** `debug_assert!` (L3) -> Kani bounded check (L4) -> Lean theorem (L5)

## 2. Regex Patterns

For string-producing functions (CLI output, serialization, log format,
protocol messages). Use `regex:` key in postcondition objects.

```yaml
postconditions:
  - regex: '^(PMAT|GH|EPIC)-\d+$'
    target: result
    description: Ticket ID matches canonical format
regex_invariants:
  - pattern: '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
    target: timestamp
    description: ISO 8601 timestamp
```

### Use Cases

| Domain | Pattern | Purpose |
|--------|---------|---------|
| CLI exit code | `^\d+$` | Valid integer exit |
| JSON output | `^\{"version":"\d+\.\d+\.\d+"` | Schema-conforming serialization |
| Log format | `^\d{4}-\d{2}-\d{2}T` | Structured timestamp |
| Commit message | `^(feat\|fix\|docs):` | Conventional commits |
| Semver | `^\d+\.\d+\.\d+` | Version string validity |
| Module path | `^[a-z_][a-z0-9_]*(::[a-z_]+)*$` | Rust module naming |

### Verification Chain

- **L3:** `debug_assert!(Regex::new(pattern).unwrap().is_match(&target))`
- **L4:** Kani bounded regex — exhaustively verify for all inputs up to bound
- **L5:** Lean language containment proof: `forall x, P(x) -> output(x) in L(regex)`

## 3. Refinement Types (Haskell/F# Style)

Type-level contracts using the newtype/phantom pattern. Invalid states
are **unrepresentable** — the Rust compiler enforces the contract.

```yaml
type_enforcement:
  principle: "Poka-Yoke via refinement types (Shingo 1986)"
  validated_types:
    NonEmptyVec:
      inner: Vec<T>
      refinement: "self.len() > 0"
      constructor: "fn new(v: Vec<T>) -> Result<Self, EmptyError>"
      eliminates: "index-out-of-bounds on first/last"
    BoundedFloat:
      inner: f32
      refinement: "self >= 0.0 && self <= 1.0"
      constructor: "fn new(val: f32) -> Result<Self, OutOfRange>"
      eliminates: "NaN propagation, division by zero"
```

### Type Class Contracts

Algebraic laws that verified types must satisfy:

```yaml
type_class_contracts:
  Invertible:
    laws: ["forall x. inverse(forward(x)) = x"]
    instances: [Tokenizer, Encoder, Serializer]
    verification: lean
  Idempotent:
    laws: ["forall x. f(f(x)) = f(x)"]
    instances: [normalize, constrain_layout]
    verification: kani
```

### Verification Chain

- **L2:** Rust compiler (private inner field + Result constructor)
- **L3:** `debug_assert!` on refinement predicate in constructor
- **L4:** Kani proof that constructor rejects all invalid values
- **L5:** Lean theorem for type class laws and refinement soundness

## Dual Expressions

Any postcondition can embed both Rust and Lean expressions, verified
independently:

```yaml
postconditions:
  - rust: '(result.iter().sum::<f32>() - 1.0).abs() < 1e-5'
    lean: '(Array.sum (softmax x) - 1.0).abs < 1e-5'
    description: Softmax output sums to 1
```

The Lean proof covers all inputs universally. The Rust assertion catches
regressions at runtime on actual data.

## References

- Meyer (1988). Design by Contract.
- Vazou et al. (2014). Liquid Haskell: Refinement types via SMT.
- Swamy et al. (2016). F*: Dependent types for program verification.
- Brady (2013). Idris: Dependent types for systems programming.
