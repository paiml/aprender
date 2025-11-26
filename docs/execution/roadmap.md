# PMAT Development Roadmap

## Current Sprint: v0.9.0 Autograd Engine - PyTorch-Compatible Automatic Differentiation
- **Duration**: 2025-11-25 to 2025-12-16
- **Priority**: P0
- **Quality Gates**: Complexity ≤ 20, SATD = 0, Coverage ≥ 80%

### Tasks
| ID | Description | Status | Complexity | Priority |
|----|-------------|--------|------------|----------|
| AG-001 | Create `src/autograd/mod.rs` module structure | pending | low | P0 |
| AG-002 | Implement `Tensor` struct with gradient tracking | pending | high | P0 |
| AG-003 | Implement `GradFn` trait for backward operations | pending | high | P0 |
| AG-004 | Implement `ComputationGraph` tape-based recorder | pending | high | P0 |
| AG-005 | Implement element-wise ops: add, sub, mul, div, neg | pending | medium | P0 |
| AG-006 | Implement transcendental ops: exp, log, pow, sqrt | pending | medium | P0 |
| AG-007 | Implement reduction ops: sum, mean, sum_dim, mean_dim | pending | medium | P0 |
| AG-008 | Implement matmul with trueno backend | pending | high | P0 |
| AG-009 | Implement activation gradients: relu, sigmoid, tanh | pending | medium | P0 |
| AG-010 | Implement softmax and log_softmax gradients | pending | high | P0 |
| AG-011 | Implement backward() with topological sort | pending | high | P0 |
| AG-012 | Implement no_grad context and detach() | pending | low | P1 |
| AG-013 | Add gradient verification tests (numerical check) | pending | medium | P0 |
| AG-014 | Integrate with existing optim module | pending | medium | P1 |
| AG-015 | Documentation and examples | pending | low | P1 |

### Definition of Done
- [ ] All tasks completed
- [ ] Quality gates passed
- [ ] Documentation updated
- [ ] Tests passing
- [ ] Changelog updated

