# cpp-type-preservation-v1

**Version:** 1.0.0

C++ to Rust type preservation contract for Decy transpiler

## References

- CROWN: Ownership Guided C to Rust Translation [2303.10515] (CAV 2023)
- Scylla: Compiling C to Safe Rust [2412.15042] (Fromherz 2024)
- CRUST-Bench [2504.15254] (COLM 2025)

## Equations

### class_to_struct

$$
forall c in C++ classes: transpile(c) = struct + impl + (Drop if destructor)
$$

**Domain:** $C++ class declarations with fields, methods, constructors, destructors$

**Codomain:** $Rust struct + impl block + optional Drop trait$

**Invariants:**

- $Field count preserved (|fields(class)| = |fields(struct)|)$
- $Field types mapped correctly (int -> i32, float -> f32, etc.)$
- `Constructor maps to pub fn new() -> Self`
- $Destructor maps to impl Drop$
- $Const methods get &self, non-const get &mut self$

### inheritance_to_composition

$$
forall derived : base: transpile(derived) = struct { base: Base, ...fields } + Deref
$$

**Domain:** $C++ single inheritance (class Derived : public Base)$

**Codomain:** $Rust composition with Deref/DerefMut$

**Invariants:**

- $Base class embedded as first field named 'base'$
- $impl Deref with Target = BaseClass$
- $impl DerefMut for mutable base access$

### namespace_to_mod

$$
forall ns in C++ namespaces: transpile(ns) = pub mod { contents }
$$

**Domain:** $C++ namespace declarations$

**Codomain:** $Rust pub mod blocks$

**Invariants:**

- $Namespace name preserved as module name$
- $Nested namespaces become nested modules$
- $Functions, structs, classes within namespace appear inside mod$

### operator_to_trait

```
forall op in overloaded operators: transpile(op) = impl std::ops::Trait
```

**Domain:** `C++ operator overloading methods (operator+, operator==, etc.)`

**Codomain:** `Rust std::ops trait implementations`

**Invariants:**

- $operator+ maps to impl Add with Output type$
- `operator== maps to impl PartialEq`
- $operator+= maps to impl AddAssign$
- $Regular methods remain in impl block (not moved to traits)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Field count preservation (class fields = struct fields) |  |
| 2 | postcondition | Output compiles with rustc |  |
| 3 | invariant | Constructor parameter mapping (name match or positional fallback) |  |
| 4 | equivalence | Method bodies preserve semantic intent (implicit this -> self) |  |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-CPP-001 | Class with fields transpiles to struct | class Point { int x; int y; } -> struct Point { pub x: i32, pub y: i32 } | Class extraction or struct codegen broken |
| FALSIFY-CPP-002 | Constructor maps to new() | Point(int a, int b) -> pub fn new(a: i32, b: i32) -> Self | Constructor extraction or new() codegen broken |
| FALSIFY-CPP-003 | Destructor maps to Drop | ~Point() -> impl Drop for Point | Destructor detection or Drop codegen broken |
| FALSIFY-CPP-004 | Namespace maps to mod | namespace math { int add(); } -> pub mod math { fn add() } | Namespace extraction or mod codegen broken |
| FALSIFY-CPP-005 | operator+ maps to impl Add | operator+(T rhs) -> impl std::ops::Add<T> for Self | Operator detection or trait impl codegen broken |
| FALSIFY-CPP-006 | Inheritance maps to composition + Deref | class Dog : public Animal -> struct Dog { base: Animal } + impl Deref | Base specifier extraction or Deref codegen broken |
| FALSIFY-CPP-007 | Implicit this access maps to self | return x; (in method) -> return self.x; | CXXThisExpr handling or field access fallback broken |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-CPP-001 | Field count preservation | 8 | bounded_int |

## QA Gate

**cpp-type-preservation-v1 Contract** (F-CPP-001)

C++ type preservation quality gate for Decy transpiler

**Checks:** validation, falsification

