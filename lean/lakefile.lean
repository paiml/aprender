import Lake
open Lake DSL

package provableContracts where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib ProvableContracts where
  srcDir := "."

@[default_target, test_driver]
lean_exe test where
  root := `Test
  srcDir := "test"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"
