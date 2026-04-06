# Phase 3 Traits: Executive Summary

## Prioritized Implementation List

### MUST HAVE - Phase 3.1 (Week 1)

| Trait | Priority | Why | Implementation | Tests | Risk |
|-------|----------|-----|-----------------|-------|------|
| **Deref<Target=[T]>** | ⭐⭐⭐⭐⭐ | Enables automatic slice coercion - idiomatic Rust | `fn deref(&self) -> &[T] { self.as_slice() }` | 20 | Very Low |
| **DerefMut** | ⭐⭐⭐⭐⭐ | Mutable slice access | `fn deref_mut(&mut self) -> &mut [T] { self.as_slice_mut() }` | 15 | Very Low |
| **AsRef<[T]>** | ⭐⭐⭐⭐⭐ | Generic function bounds | `fn as_ref(&self) -> &[T] { self.as_slice() }` | 15 | Very Low |
| **AsMut<[T]>** | ⭐⭐⭐⭐⭐ | Mutable generic bounds | `fn as_mut(&mut self) -> &mut [T] { self.as_slice_mut() }` | 15 | Very Low |

**Estimated Effort**: 150-200 lines of code (implementation + tests)  
**Expected Outcome**: Fundamental ergonomics, high user utility

---

### HIGHLY VALUABLE - Phase 3.2 (Week 2)

| Trait | Priority | Why | Implementation | Tests | Risk |
|-------|----------|-----|-----------------|-------|------|
| **PartialOrd** | ⭐⭐⭐⭐ | Enables `<`, `>`, `<=`, `>=` comparisons | Lexicographic comparison | 25 | Low |
| **Ord** | ⭐⭐⭐⭐ | Total ordering, BTreeMap keys | Requires Eq + full ordering | 20 | Low |
| **Hash** | ⭐⭐⭐ | HashMap/HashSet keys, caching | Hash all elements in sequence | 20 | Low |

**Estimated Effort**: 200-250 lines of code  
**Expected Outcome**: Sorting, comparison operations, hash-based collections support

---

### OPTIONAL - Phase 3.3 (Week 3+)

| Trait | Priority | Why | Implementation | Tests | Risk |
|-------|----------|-----|-----------------|-------|------|
| **Display** | ⭐⭐ | User-friendly formatting | Format as `[elem1, elem2, ...]` | 15 | Very Low |
| **Borrow<[T]>** | ⭐⭐ | Advanced borrowing patterns | Delegate to AsRef | 10 | Very Low |
| **BorrowMut<[T]>** | ⭐⭐ | Advanced mutable borrowing | Delegate to AsMut | 10 | Very Low |

**Estimated Effort**: 150 lines of code  
**Expected Outcome**: Polish, advanced use cases

---

## Why This Order?

### Deref/DerefMut → Ubiquitous in Rust
Every container type provides `Deref` coercion. It's the first thing users expect:
```rust
vec![1,2,3].len()  // Works because of Deref coercion to slice
```

### AsRef/AsMut → Interoperability
Standard library functions use these bounds extensively:
```rust
fn process<T: AsRef<[u8]>>(data: T) { /* ... */ }
process(&vec);  // TruenoVec must implement AsRef<[u8]>
```

### PartialOrd/Ord → Fundamental Collections Use Case
Sorting is ubiquitous:
```rust
let mut vectors = vec![...];
vectors.sort();  // Requires PartialOrd/Ord
```

### Hash → Specific Use Case but Important
Enables using TruenoVec as keys:
```rust
let mut map = HashMap::new();
map.insert(vec, value);  // Requires Hash
```

### Display → Cosmetic but Valuable
Better user experience:
```rust
println!("{}", vec);  // [1, 2, 3] instead of TruenoVec { ... }
```

---

## Implementation Complexity Breakdown

### PHASE 3.1 (Deref, DerefMut, AsRef, AsMut)

**Total Implementation Lines**: ~120  
**Total Test Lines**: ~600  
**Complexity**: Very Low - All delegate to existing `as_slice()` methods

```rust
// Deref implementation (2 lines)
impl<T> Deref for TruenoVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] { self.as_slice() }
}

// DerefMut implementation (2 lines)
impl<T> DerefMut for TruenoVec<T> {
    fn deref_mut(&mut self) -> &mut [T] { self.as_slice_mut() }
}

// AsRef implementation (2 lines)
impl<T> AsRef<[T]> for TruenoVec<T> {
    fn as_ref(&self) -> &[T] { self.as_slice() }
}

// AsMut implementation (2 lines)
impl<T> AsMut<[T]> for TruenoVec<T> {
    fn as_mut(&mut self) -> &mut [T] { self.as_slice_mut() }
}
```

### PHASE 3.2 (PartialOrd, Ord, Hash)

**Total Implementation Lines**: ~180  
**Total Test Lines**: ~550  
**Complexity**: Low - Delegate to T's implementations

```rust
// PartialOrd (requires lexicographic comparison)
impl<T: PartialOrd> PartialOrd for TruenoVec<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

// Ord (total ordering)
impl<T: Ord> Ord for TruenoVec<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

// Hash (hash all elements)
impl<T: Hash> Hash for TruenoVec<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}
```

---

## Testing Requirements

### Unit Tests by Trait

| Trait | Number | Categories |
|-------|--------|------------|
| Deref/DerefMut | 15 | Coercion, method delegation, mutability |
| AsRef/AsMut | 15 | Type conversion, interop, bounds |
| PartialOrd | 20 | Comparisons, edge cases, consistency |
| Ord | 15 | Sorting, total order, BTreeMap compat |
| Hash | 12 | Hash stability, HashMap/HashSet |
| Display | 10 | Formatting, edge cases |
| **Total** | **87** | - |

### Property-Based Tests

| Category | Tests | Properties |
|----------|-------|-----------|
| Deref consistency | 5 | Coercion equivalence with slice |
| Ordering semantics | 10 | Transitivity, antisymmetry, totality |
| Hash stability | 7 | Consistency, equivalence with Eq |
| Display semantics | 3 | Round-trip, formatting consistency |
| **Total** | **25** | - |

### Integration Tests

- Deref/AsRef with standard library functions
- PartialOrd/Ord with Vec::sort(), BTreeMap
- Hash with HashMap/HashSet
- Display with format! macros

---

## Mutation Testing Strategy

### High-Risk Areas

1. **Deref implementations**
   - Mutant: Return wrong slice bounds
   - Mutant: Return self.ptr instead of dereferenced memory
   
2. **PartialOrd/Ord implementations**
   - Mutant: Wrong comparison operator
   - Mutant: Skip first element comparison
   - Mutant: Return Ordering::Equal incorrectly

3. **Hash implementation**
   - Mutant: Hash only first element
   - Mutant: Hash in wrong order
   - Mutant: Skip hashing entire slice

### Target Score: 85%+ mutation score per trait

---

## Code Metrics Projection

### Current State (End of Phase 2)
- **Total LOC**: 3,072
- **Impl blocks**: 26
- **Test coverage**: 97.7% mutation score

### After Phase 3.1 (Deref + AsRef)
- **Added LOC**: ~720 (120 impl + 600 tests)
- **New impl blocks**: 4
- **Projected coverage**: 96%+ mutation score

### After Phase 3.2 (PartialOrd + Ord + Hash)
- **Added LOC**: ~730 (180 impl + 550 tests)
- **New impl blocks**: 3
- **Projected coverage**: 86%+ mutation score

### After Phase 3.3 (Display + Borrow)
- **Added LOC**: ~330 (90 impl + 240 tests)
- **New impl blocks**: 3
- **Projected coverage**: 85%+ mutation score

### Final Projected State (Full Phase 3)
- **Total LOC**: ~4,850
- **Impl blocks**: 36
- **Test count**: 250+
- **Mutation score**: 85%+ (comprehensive)

---

## Concurrent vs. Sequential Implementation

### Recommended: Concurrent Implementation

**Phase 3.1 can be done by different developers in parallel**:
- Developer A: Deref/DerefMut + tests
- Developer B: AsRef/AsMut + tests
- Developer C: Core functionality integration tests

**Phase 3.2 should proceed after 3.1 is merged**:
- Dependency: All 3.1 traits must be stable before 3.2
- Reason: 3.1 provides foundation for ergonomic usage

### Dependencies Between Traits

```
Phase 3.1 (Fundamental)
├── Deref
├── DerefMut  
├── AsRef
└── AsMut

Phase 3.2 (Comparison & Hashing) [Depends on 3.1]
├── PartialOrd (independent)
├── Ord (depends on PartialOrd + Eq)
└── Hash (independent)

Phase 3.3 (Optional Polish) [Depends on 3.1]
├── Display (independent)
├── Borrow (complements AsRef)
└── BorrowMut (complements AsMut)
```

---

## Success Criteria Checklist

### Phase 3.1
- [ ] Deref/DerefMut implementation with docs
- [ ] AsRef/AsMut implementation with docs
- [ ] 95%+ line coverage
- [ ] 85%+ mutation score
- [ ] All property-based tests pass
- [ ] Integration tests with std library
- [ ] No SATD comments (zero-tolerance)

### Phase 3.2
- [ ] PartialOrd implementation
- [ ] Ord implementation
- [ ] Hash implementation with Eq consistency proof
- [ ] Sorting tests pass
- [ ] HashMap/HashSet integration tests
- [ ] BTreeMap/BTreeSet integration tests
- [ ] 85%+ mutation score
- [ ] Property tests for ordering semantics

### Phase 3.3 (If Resources Allow)
- [ ] Display implementation
- [ ] Borrow/BorrowMut (optional)
- [ ] User-friendly formatting tests
- [ ] Backward compatibility verified

---

## Appendix: Why Not Other Traits?

### Rejected Candidates

| Trait | Reason |
|-------|--------|
| **Into** | Superseded by IntoIterator; not idiomatic for collections |
| **TryFrom/TryInto** | Requires error handling; not applicable here |
| **Serialize/Deserialize** | Requires serde dependency; defer to Phase 4+ |
| **CoerceUnsized** | Unstable feature; not production-ready |
| **Unpin** | Auto-implemented; not needed |
| **Visitor Pattern** | Over-engineering for simple collections |

### Partially Implemented (Don't Need Phase 3)

| Trait | Status |
|-------|--------|
| **IntoIterator** | Complete (3 variants) |
| **Iterator** | Complete (3 implementations) |
| **ExactSizeIterator** | Complete (on IntoIter, partial on references) |
| **DoubleEndedIterator** | Complete (on all iterators) |

---

## References & Best Practices

### Rust Standard Library Patterns

- **std::Vec**: Has all Phase 3.1-3.2 traits ✓
- **std::collections::VecDeque**: Has all Phase 3.1-3.2 traits ✓
- **std::collections::LinkedList**: Has all Phase 3.1-3.2 traits ✓

### Certification Standards

Aligns with PMAT compliance:
- Coverage: 95%+
- Complexity: Cyclomatic ≤ 10
- SATD: Zero tolerance
- Mutation: 85%+ score
- Documentation: 90%+ rustdoc

---

Generated: 2025-11-17  
Framework: certeza - Asymptotic Test Effectiveness
