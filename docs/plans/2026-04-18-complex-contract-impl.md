# Complex Contract Rollout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Define a source-level complex contract for `C32` / `C64`, implement the supported parts consistently on real backends, and make unsupported backend/feature combinations fail through centralized validation instead of missing code paths.

**Architecture:** Split the frontend surface into `ComplexCore`, `ComplexCompare`, and `ComplexMath`; add a backend capability registry for those families in `cubecl-ir`; validate complex use at expansion time; and keep the existing `runtime_tests` macro framework by splitting complex tests into contract groups plus negative validation tests. CUDA is the reference positive backend, CPU is a conditional positive backend for `Core` only if the MLIR path is reviewable, and WGPU / HIP stay validation-only for now.

**Tech Stack:** `cubecl-core` frontend and runtime tests, `cubecl-ir` features/properties, `cubecl-cpp` shared/CUDA/HIP dialects, `cubecl-cpu` MLIR backend, `cubecl-wgpu` WGSL backend, `num_complex`, `cargo check`, `cargo test`.

---

## Contract decisions to keep fixed during implementation

- `C32` and `C64` are both part of the contract.
- `ComplexCore` is the ML-centric minimum:
  - `+`, `-`, `*`, `/`
  - unary `-`
  - `conj`
  - `real_val`
  - `imag_val`
- `ComplexCompare` is optional and only covers `eq` / `ne`.
- `ComplexMath` is optional and covers:
  - `abs`
  - `exp`
  - `log`
  - `sin`
  - `cos`
  - `sqrt`
  - `tanh`
  - `powf`
- Ordering comparisons, `min` / `max`, `clamp`, bitwise ops, and scalar ABI shortcuts are not part of the complex contract.
- `InputScalar` / dynamic scalar complex arguments are explicitly out of scope for this PR.
- No snapshot / golden tests.
- No new test harness.
- Source of truth is code:
  - frontend trait split
  - IR capability registry
  - centralized validation
  - shared runtime test macros

## Backend target matrix for this plan

- CUDA:
  - `C32`: `Core`, `Compare`, `Math`
  - `C64`: `Core`, `Compare`, `Math`
- CPU:
  - target `C32` / `C64` `Core`
  - keep `Compare` / `Math` explicitly unsupported unless MLIR support is trivial and reviewable
- WGPU:
  - no positive complex support in this PR
  - must fail through centralized validation, not `unimplemented!`
- HIP:
  - no positive complex support in this PR
  - must not advertise complex support accidentally

### Task 1: Split the runtime contract into positive and negative test groups

**Files:**
- Modify: `crates/cubecl-core/src/runtime_tests/complex.rs`
- Modify: `crates/cubecl-core/src/runtime_tests/mod.rs`

**Step 1: Replace the monolithic export macro layout**

In `crates/cubecl-core/src/runtime_tests/complex.rs`, replace the single exported `testgen_complex!()` macro with four exported macros:

```rust
#[macro_export]
macro_rules! testgen_complex_core { () => { /* add/sub/mul/div/neg/conj/real/imag */ }; }

#[macro_export]
macro_rules! testgen_complex_compare { () => { /* eq/ne */ }; }

#[macro_export]
macro_rules! testgen_complex_math { () => { /* abs/exp/log/sin/cos/sqrt/tanh/powf */ }; }

#[macro_export]
macro_rules! testgen_complex_validation { () => { /* unsupported capability tests */ }; }
```

Keep helper functions in the same file so backends continue using the existing `cubecl_core::runtime_tests::*` pattern.

**Step 2: Add the missing `ComplexCore` positive kernels and helpers**

Add kernels and test helpers for:

- complex subtraction
- complex division
- complex negation
- `real_val()`
- `imag_val()`

Follow the same shape as the existing `kernel_complex_add`, `kernel_complex_mul`, and `kernel_complex_conj` helpers.

**Step 3: Add the `ComplexCompare` positive kernels and helpers**

Add kernels and test helpers for `==` and `!=` that write `bool` outputs into `Array<bool>`.

Use exact equality, not epsilon, because the contract is component-wise equality for compare.

**Step 4: Remove scalar ABI from the contract surface**

Delete `kernel_complex_constant`, `test_complex_constant_cf32`, and `test_complex_constant_cf64` from the exported contract macros.

Do not replace them with ad hoc tests. If the helpers are kept temporarily, gate them behind a comment that they are not part of the v1 contract.

**Step 5: Add negative validation helpers**

Add small kernels that each exercise exactly one capability family:

- `kernel_complex_validation_core`
- `kernel_complex_validation_compare`
- `kernel_complex_validation_math`

Each helper should launch a one-element kernel and then assert that `client.flush()` returns `ServerError::ServerUnhealthy` with a validation error when the relevant capability is absent.

**Step 6: Check that the test module still parses**

Run: `cargo check -p cubecl-core --tests`

Expected: compile may still fail on missing trait names or capability helpers, but the new test module structure itself should parse cleanly enough to move to Task 2.

**Step 7: Commit**

```bash
git add crates/cubecl-core/src/runtime_tests/complex.rs crates/cubecl-core/src/runtime_tests/mod.rs
git commit -m "test: split complex runtime tests by contract family"
```

---

### Task 2: Split the frontend traits into `Core`, `Compare`, and `Math`

**Files:**
- Modify: `crates/cubecl-core/src/frontend/element/complex.rs`
- Modify: `crates/cubecl-core/src/runtime_tests/complex.rs`

**Step 1: Replace the current `Complex` trait**

In `crates/cubecl-core/src/frontend/element/complex.rs`, replace the single public trait with:

```rust
pub trait ComplexCore:
    CubePrimitive
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Copy
    + Clone
    + PartialEq
    + core::fmt::Debug
    + Send
    + Sync
    + 'static
{
    type FloatElem: Scalar;

    fn conj(self) -> Self { unexpanded!() }
    fn real_val(self) -> Self::FloatElem { unexpanded!() }
    fn imag_val(self) -> Self::FloatElem { unexpanded!() }
}

pub trait ComplexCompare: ComplexCore {}

pub trait ComplexMath:
    ComplexCore
    + Abs<AbsElem = Self::FloatElem>
    + Exp
    + Log
    + Sin
    + Cos
    + Sqrt
    + Tanh
    + Powf
{}
```

Do not keep a public umbrella `Complex` trait in the final diff. The call sites should show which family they depend on.

**Step 2: Rename the expansion trait to match `Core`**

Rename `ComplexExpand` to `ComplexCoreExpand` and keep it responsible only for:

- `conj`
- `real_val`
- `imag_val`

**Step 3: Implement the split traits for both native complex types**

Keep the existing `impl_complex!` macro, but change it so the concrete impls become:

```rust
impl ComplexCore for num_complex::Complex<f32> { type FloatElem = f32; }
impl ComplexCompare for num_complex::Complex<f32> {}
impl ComplexMath for num_complex::Complex<f32> {}
```

and the same for `num_complex::Complex<f64>`.

**Step 4: Update the runtime test kernel bounds**

In `crates/cubecl-core/src/runtime_tests/complex.rs`, update each kernel signature:

- `ComplexCore` for add/sub/mul/div/neg/conj/real/imag
- `ComplexCompare` for eq/ne
- `ComplexMath` for math ops

Do not use broader bounds than required.

**Step 5: Check the frontend crate**

Run: `cargo check -p cubecl-core --tests`

Expected: compile should move forward; failures should now be about missing validation or capability APIs rather than missing trait names.

**Step 6: Commit**

```bash
git add crates/cubecl-core/src/frontend/element/complex.rs crates/cubecl-core/src/runtime_tests/complex.rs
git commit -m "refactor: split complex frontend surface into core compare and math"
```

---

### Task 3: Add a complex capability registry to `cubecl-ir`

**Files:**
- Modify: `crates/cubecl-ir/src/features.rs`
- Modify: `crates/cubecl-ir/src/properties.rs`
- Modify: `crates/cubecl-core/src/frontend/element/complex.rs`

**Step 1: Add a new complex usage enum**

In `crates/cubecl-ir/src/features.rs`, add:

```rust
#[derive(Debug, Hash, PartialOrd, Ord, EnumSetType)]
pub enum ComplexUsage {
    Core,
    Compare,
    Math,
}
```

**Step 2: Add complex usage storage beside normal type usage**

Extend `Types` with a dedicated map:

```rust
pub complex: BTreeMap<StorageType, EnumSet<ComplexUsage>>,
```

Do not overload `TypeUsage::Arithmetic` for this. That enum is too coarse for complex.

**Step 3: Add feature and property helpers**

In `Features` and `DeviceProperties`, add:

```rust
pub fn complex_usage(&self, ty: StorageType) -> EnumSet<ComplexUsage> { ... }
pub fn register_complex_usage(&mut self, ty: impl Into<StorageType>, uses: impl Into<EnumSet<ComplexUsage>>) { ... }
pub fn supports_complex_usage(&self, ty: impl Into<StorageType>, usage: ComplexUsage) -> bool { ... }
```

Use an empty enum set as the default when a type is missing.

**Step 4: Add a frontend query helper**

In `crates/cubecl-core/src/frontend/element/complex.rs`, add a helper on `ComplexCore` so tests can query runtime support directly:

```rust
fn supported_complex_uses<R: Runtime>(client: &ComputeClient<R>) -> EnumSet<ComplexUsage> {
    client.properties().complex_usage(Self::as_type_native_unchecked().storage_type())
}
```

Keep it parallel to `CubePrimitive::supported_uses`.

**Step 5: Check the IR and frontend**

Run: `cargo check -p cubecl-ir -p cubecl-core`

Expected: compile succeeds or fails only on the new call sites that still need validation wiring.

**Step 6: Commit**

```bash
git add crates/cubecl-ir/src/features.rs crates/cubecl-ir/src/properties.rs crates/cubecl-core/src/frontend/element/complex.rs
git commit -m "feat: add complex capability registry to device properties"
```

---

### Task 4: Centralize complex validation in expansion, not in backend panics

**Files:**
- Modify: `crates/cubecl-core/src/frontend/element/complex.rs`
- Modify: `crates/cubecl-core/src/frontend/validation.rs`
- Modify: `crates/cubecl-core/src/frontend/operation/cmp.rs`
- Modify: `crates/cubecl-core/src/frontend/operation/unary.rs`
- Modify: `crates/cubecl-core/src/frontend/operation/binary.rs`

**Step 1: Add one helper for capability requirements**

In `crates/cubecl-core/src/frontend/validation.rs` or `complex.rs`, add a helper with a concrete error shape:

```rust
fn require_complex_usage(
    scope: &mut Scope,
    elem: ElemType,
    usage: ComplexUsage,
    op_name: &'static str,
) {
    let ty = StorageType::Scalar(elem);
    let Some(props) = scope.properties.as_ref() else {
        return;
    };

    if !props.supports_complex_usage(ty, usage) {
        scope.push_error(format!(
            "Complex operation `{op_name}` requires {:?} support for `{elem}` on this backend",
            usage
        ));
    }
}
```

**Step 2: Validate `ComplexCore` methods**

Call the helper from the `ComplexCoreExpand` implementations for:

- `conj`
- `real_val`
- `imag_val`

Use `ComplexUsage::Core`.

**Step 3: Validate `eq` / `ne` and reject ordering comparisons**

In `crates/cubecl-core/src/frontend/operation/cmp.rs`:

- detect complex operands
- allow only `eq` and `ne`
- require `ComplexUsage::Compare`
- for `lt`, `le`, `gt`, `ge` on complex, push a validation error immediately

Do not let unsupported ordering comparisons reach backend codegen.

**Step 4: Validate complex math through the frontend op surface**

In `unary.rs` and `binary.rs`, keep real-only generic macros as they are, but make complex math explicit and validated:

- `abs`, `exp`, `log`, `sin`, `cos`, `sqrt`, `tanh`, `powf` require `ComplexUsage::Math` when the operand type is complex
- `min`, `max`, `clamp`, remainder, bitwise ops stay rejected for complex

If a generic helper already works, keep it, but add the complex guard before emitting IR.

**Step 5: Replace backend `unimplemented!` as the primary contract signal**

Do not remove all backend `unimplemented!` sites yet, but make sure normal frontend kernels now fail earlier with validation errors on unsupported backends.

**Step 6: Check the frontend again**

Run: `cargo check -p cubecl-core --tests`

Expected: runtime test helpers should now compile cleanly against the new contract model.

**Step 7: Commit**

```bash
git add crates/cubecl-core/src/frontend/element/complex.rs crates/cubecl-core/src/frontend/validation.rs crates/cubecl-core/src/frontend/operation/cmp.rs crates/cubecl-core/src/frontend/operation/unary.rs crates/cubecl-core/src/frontend/operation/binary.rs
git commit -m "feat: validate complex contract centrally in frontend expansion"
```

---

### Task 5: Stop shared C++ code from leaking CUDA assumptions and accidental HIP support

**Files:**
- Modify: `crates/cubecl-cpp/src/shared/dialect.rs`
- Modify: `crates/cubecl-cpp/src/shared/instruction.rs`
- Modify: `crates/cubecl-cpp/src/shared/variable.rs`
- Modify: `crates/cubecl-cpp/src/shared/base.rs`
- Modify: `crates/cubecl-cpp/src/cuda/dialect.rs`
- Modify: `crates/cubecl-cpp/src/hip/dialect.rs`
- Modify: `crates/cubecl-cpp/src/metal/dialect.rs`

**Step 1: Remove complex storage registration from the shared helper**

In `crates/cubecl-cpp/src/shared/base.rs::register_supported_types`, remove:

- `ElemType::Complex(C32)`
- `ElemType::Complex(C64)`

from the unconditional shared supported type list.

This stops HIP from advertising complex support just because it shares the C++ base layer.

**Step 2: Add dialect-owned complex formatting hooks**

In `crates/cubecl-cpp/src/shared/dialect.rs`, add dialect methods for:

- complex constant construction
- complex conjugation
- complex real extraction
- complex imaginary extraction

Example shape:

```rust
fn compile_complex_make(...) -> std::fmt::Result;
fn compile_complex_conj(...) -> std::fmt::Result;
fn compile_complex_real(...) -> std::fmt::Result;
fn compile_complex_imag(...) -> std::fmt::Result;
```

**Step 3: Replace `.x` / `.y` / `make_cu*` in the shared formatter**

In `shared/instruction.rs` and `shared/variable.rs`, replace the current CUDA-specific formatting with calls to the new dialect hooks.

The shared layer must not assume:

- `.x` is real
- `.y` is imaginary
- `make_cuFloatComplex` exists

**Step 4: Keep CUDA as the only positive dialect**

In `crates/cubecl-cpp/src/cuda/dialect.rs`, implement the new hooks using:

- `make_cuFloatComplex`
- `make_cuDoubleComplex`
- `cuCrealf` / `cuCimagf`
- `cuCreal` / `cuCimag`

Reuse the already-added `cubecl_abs`, `cubecl_exp`, `cubecl_log`, `cubecl_sin`, `cubecl_cos`, `cubecl_sqrt`, `cubecl_tanh`, and `cubecl_powf` helpers.

**Step 5: Keep HIP and Metal explicitly unsupported**

In `hip/dialect.rs` and `metal/dialect.rs`, keep the output as explicit `#error Complex not supported ...` for now.

Do not advertise more than the dialect really implements.

**Step 6: Check the C++ layer**

Run: `cargo check -p cubecl-cpp -p cubecl-cuda -p cubecl-hip`

Expected: CUDA compiles; HIP still compiles as a crate but does not advertise complex support.

**Step 7: Commit**

```bash
git add crates/cubecl-cpp/src/shared/dialect.rs crates/cubecl-cpp/src/shared/instruction.rs crates/cubecl-cpp/src/shared/variable.rs crates/cubecl-cpp/src/shared/base.rs crates/cubecl-cpp/src/cuda/dialect.rs crates/cubecl-cpp/src/hip/dialect.rs crates/cubecl-cpp/src/metal/dialect.rs
git commit -m "refactor: make complex formatting dialect owned in cpp backends"
```

---

### Task 6: Register the backend capability matrix explicitly

**Files:**
- Modify: `crates/cubecl-cuda/src/runtime.rs`
- Modify: `crates/cubecl-cpu/src/runtime.rs`
- Modify: `crates/cubecl-cpu/src/compiler/visitor/elem.rs`
- Modify: `crates/cubecl-hip/src/runtime.rs`
- Modify: `crates/cubecl-wgpu/src/backend/wgsl.rs`
- Modify: `crates/cubecl-wgpu/src/backend/vulkan.rs`
- Modify: `crates/cubecl-wgpu/src/backend/metal.rs`

**Step 1: Register CUDA complex type usage and complex usage**

In `crates/cubecl-cuda/src/runtime.rs`, after the normal type registration, add:

```rust
device_props.register_type_usage(ElemType::Complex(ComplexKind::C32), TypeUsage::all());
device_props.register_type_usage(ElemType::Complex(ComplexKind::C64), TypeUsage::all());
device_props.register_complex_usage(ElemType::Complex(ComplexKind::C32), ComplexUsage::Core | ComplexUsage::Compare | ComplexUsage::Math);
device_props.register_complex_usage(ElemType::Complex(ComplexKind::C64), ComplexUsage::Core | ComplexUsage::Compare | ComplexUsage::Math);
```

**Step 2: Register CPU only after the lowering actually exists**

Do not register CPU complex support yet in this task. Wait until Task 8 or 9 proves what the CPU path can really do.

**Step 3: Keep HIP and WGPU empty**

In HIP and WGPU backends, do not register any complex storage or complex usage in this PR.

If a backend advertises zero support, the validation tests must be the proof, not documentation prose.

**Step 4: Check the runtime property code**

Run: `cargo check -p cubecl-cuda -p cubecl-cpu -p cubecl-hip -p cubecl-wgpu`

Expected: no backend should accidentally advertise complex support except CUDA.

**Step 5: Commit**

```bash
git add crates/cubecl-cuda/src/runtime.rs crates/cubecl-cpu/src/runtime.rs crates/cubecl-cpu/src/compiler/visitor/elem.rs crates/cubecl-hip/src/runtime.rs crates/cubecl-wgpu/src/backend/wgsl.rs crates/cubecl-wgpu/src/backend/vulkan.rs crates/cubecl-wgpu/src/backend/metal.rs
git commit -m "feat: register backend complex capability matrix explicitly"
```

---

### Task 7: Wire backend test macros without changing the test framework

**Files:**
- Modify: `crates/cubecl-cuda/src/lib.rs`
- Modify: `crates/cubecl-cpu/src/lib.rs`
- Modify: `crates/cubecl-wgpu/src/lib.rs`
- Modify: `crates/cubecl-hip/src/lib.rs`

**Step 1: Update CUDA tests**

Replace `cubecl_core::testgen_complex!();` with:

```rust
cubecl_core::testgen_complex_core!();
cubecl_core::testgen_complex_compare!();
cubecl_core::testgen_complex_math!();
cubecl_core::testgen_complex_validation!();
```

The validation macro should become a no-op for supported CUDA capabilities because the helpers check the runtime capability map first.

**Step 2: Update CPU tests**

For now, wire only:

```rust
cubecl_core::testgen_complex_validation!();
```

Add `testgen_complex_core!()` only after CPU positive support is real.

**Step 3: Update WGPU and HIP tests**

Add:

```rust
cubecl_core::testgen_complex_validation!();
```

to WGPU and HIP test modules so unsupported complex use is exercised through the same framework.

**Step 4: Check test compilation**

Run: `cargo check -p cubecl-cuda --tests -p cubecl-cpu --tests -p cubecl-wgpu --tests`

Expected: all three test modules compile with the new macro layout.

**Step 5: Commit**

```bash
git add crates/cubecl-cuda/src/lib.rs crates/cubecl-cpu/src/lib.rs crates/cubecl-wgpu/src/lib.rs crates/cubecl-hip/src/lib.rs
git commit -m "test: wire backend complex contract macros by capability"
```

---

### Task 8: Make a hard go/no-go decision on CPU positive support

**Files:**
- Read: `crates/cubecl-cpu/src/compiler/visitor/elem.rs`
- Read: `crates/cubecl-cpu/src/compiler/visitor/variables.rs`
- Read: `crates/cubecl-cpu/src/compiler/visitor/operation/arithmetic.rs`
- Read: `crates/cubecl-cpu/src/compiler/visitor/operation/operator.rs`
- Read: `crates/cubecl-cpu/src/compiler/visitor/operation/comparison.rs`

**Step 1: Check whether the existing MLIR wrapper exposes a reviewable complex path**

Verify whether `tracel-mlir-rs` already exposes:

- complex scalar type construction
- constant construction or a clean equivalent
- real / imag extraction
- conjugation

If this requires raw string-based op assembly or large raw-FFI glue, stop here and do not land CPU positive support in this PR.

**Step 2: Record the branch decision**

Use one of these decisions and keep it explicit in the implementation branch:

- `Decision A`: CPU positive support is reviewable, continue to Task 9.
- `Decision B`: CPU positive support needs dependency work, keep CPU validation-only in this PR and open a follow-up.

**Step 3: Check current crate status**

Run: `cargo check -p cubecl-cpu`

Expected: current CPU crate still builds before any positive complex changes.

**Step 4: Commit the decision note if needed**

If `Decision B`, commit only the comment / TODO / issue reference that makes the stop point explicit.

```bash
git add relevant/files
git commit -m "docs: record cpu complex support gate"
```

---

### Task 9: Implement CPU `ComplexCore` only if the lowering path is reviewable

**Files:**
- Modify: `crates/cubecl-cpu/src/compiler/visitor/elem.rs`
- Modify: `crates/cubecl-cpu/src/compiler/visitor/variables.rs`
- Modify: `crates/cubecl-cpu/src/compiler/visitor/operation/arithmetic.rs`
- Modify: `crates/cubecl-cpu/src/compiler/visitor/operation/operator.rs`
- Modify: `crates/cubecl-cpu/src/runtime.rs`
- Modify: `crates/cubecl-cpu/src/lib.rs`

**Step 1: Add `ElemType::Complex` type lowering**

In `elem.rs`, add lowering for:

- `Complex<f32>`
- `Complex<f64>`

If the wrapper exposes a native complex type, use that. Do not lower complex values as ad hoc `vector<2xf32>` unless the whole backend is being converted consistently, which is out of scope here.

**Step 2: Add constant loading**

In `variables.rs`, add constant handling for `ConstantValue::Complex(re, im)`.

Keep the representation consistent with the chosen type lowering from Step 1.

**Step 3: Add the `ComplexCore` arithmetic and operator pieces**

In `operation/arithmetic.rs`, add support for:

- add
- sub
- mul
- div
- neg
- conj

In `operation/operator.rs`, add support for:

- `Operator::Real`
- `Operator::Imag`

Do not add compare or math in this PR unless the same lowering path makes them trivially obvious and reviewable.

**Step 4: Register only the CPU capabilities that are really implemented**

In `crates/cubecl-cpu/src/runtime.rs`, add:

```rust
device_props.register_type_usage(ElemType::Complex(ComplexKind::C32), TypeUsage::all());
device_props.register_type_usage(ElemType::Complex(ComplexKind::C64), TypeUsage::all());
device_props.register_complex_usage(ElemType::Complex(ComplexKind::C32), ComplexUsage::Core);
device_props.register_complex_usage(ElemType::Complex(ComplexKind::C64), ComplexUsage::Core);
```

Do not register `Compare` or `Math`.

**Step 5: Enable only the matching positive tests**

In `crates/cubecl-cpu/src/lib.rs`, add:

```rust
cubecl_core::testgen_complex_core!();
cubecl_core::testgen_complex_validation!();
```

Do not enable compare or math macros.

**Step 6: Run CPU tests**

Run: `cargo test -p cubecl-cpu test_complex_core -- --nocapture`

Expected: `ComplexCore` tests pass; validation tests fail only for compare / math kernels as intended.

**Step 7: Commit**

```bash
git add crates/cubecl-cpu/src/compiler/visitor/elem.rs crates/cubecl-cpu/src/compiler/visitor/variables.rs crates/cubecl-cpu/src/compiler/visitor/operation/arithmetic.rs crates/cubecl-cpu/src/compiler/visitor/operation/operator.rs crates/cubecl-cpu/src/runtime.rs crates/cubecl-cpu/src/lib.rs
git commit -m "feat: add cpu complex core support"
```

---

### Task 10: Run the full verification matrix and stop on the first inconsistency

**Files:**
- No code changes required unless a verification failure exposes a real bug

**Step 1: Run the crate-level compile matrix**

Run:

```bash
cargo check -p cubecl-ir -p cubecl-core -p cubecl-cpp -p cubecl-cuda -p cubecl-cpu -p cubecl-wgpu -p cubecl-hip
```

Expected: all listed crates compile.

**Step 2: Run CPU positive tests if Task 9 landed**

Run:

```bash
cargo test -p cubecl-cpu test_complex_core -- --nocapture
```

Expected: `ComplexCore` passes on CPU.

**Step 3: Run CUDA compile-level test coverage**

Run:

```bash
cargo test -p cubecl-cuda test_complex_ --no-run
```

Expected: CUDA complex tests compile.

If a compatible CUDA driver/toolchain is available, then also run:

```bash
cargo test -p cubecl-cuda test_complex_ -- --nocapture
```

Expected: `Core`, `Compare`, and `Math` pass on CUDA.

**Step 4: Run WGPU validation tests locally**

Run:

```bash
cargo test -p cubecl-wgpu complex_validation -- --nocapture
```

Expected: unsupported complex kernels fail through validation errors, not WGSL `unimplemented!`.

**Step 5: Keep HIP at compile-only until AMD hardware is available**

Run:

```bash
cargo check -p cubecl-hip
```

Expected: HIP crate compiles and does not advertise complex support.

Do not block this PR on HIP runtime execution if no AMD runtime is available yet.

**Step 6: Final cleanup commit**

```bash
git add -A
git commit -m "chore: verify complex contract rollout across backends"
```

---

## Notes for the implementing engineer

- The critical review point is not “does one backend happen to work”, but “can every backend either advertise the capability or reject it from the same frontend path”.
- If a backend still reaches `unimplemented!` for a normal frontend complex kernel after Task 4, treat that as a bug.
- Do not reintroduce `kernel_complex_constant` as proof of complex support. Complex scalar ABI is a separate contract.
- If CPU positive support requires broad raw-FFI MLIR work, split that into a follow-up PR rather than hiding it inside this contract PR.

---

## Current Status (2026-04-18, before reboot)

- Branch: `codex/add-expm1-origin-main`
- Repo state: working tree is dirty with the complex-contract rollout changes; nothing has been committed in this execution pass.
- Contract decision implemented in code:
  - Frontend split into `ComplexCore`, `ComplexCompare`, `ComplexMath`
  - IR/device capability registry has `ComplexUsage::{Core, Compare, Math}`
  - Centralized frontend validation is active on shared expand paths
  - Unsupported complex operations are rejected from the frontend with validation errors
- Backend policy currently implemented:
  - CUDA: positive support advertised for `C32` and `C64` with `Core | Compare | Math`
  - CPU: validation-only in this branch
  - WGPU: validation-only in this branch
  - HIP: validation-only in this branch, and accidental complex advertisement from `cubecl-cpp` shared registration was removed
- Important implementation detail:
  - CPU launch-time compilation errors are now propagated as normal stream/server errors instead of panicking in `unwrap()`

### Fresh verification already run

```bash
cargo fmt --all
cargo check --tests -p cubecl-core -p cubecl-cuda -p cubecl-cpu -p cubecl-wgpu -p cubecl-hip
cargo test -p cubecl-cpu complex_validation -- --nocapture
cargo test -p cubecl-wgpu complex_validation -- --nocapture
```

- Result:
  - all listed `cargo check` targets passed
  - CPU `complex_validation` tests passed
  - WGPU `complex_validation` tests passed locally

### Local environment notes

- WGPU tests ran on local Vulkan `llvmpipe` CPU adapter, so the frontend validation path is exercised locally.
- CUDA runtime smoke tests were not run in this session because `nvidia-smi -L` reported `Failed to initialize NVML: Driver/library version mismatch`.
- HIP/AMD runtime execution is still unverified locally because no AMD runtime/hardware is available on this machine.

### Remaining work after reboot

- If CUDA driver state is fixed, run:

```bash
cargo test -p cubecl-cuda test_complex_ --no-run
```

- If CUDA runtime execution is available, then run:

```bash
cargo test -p cubecl-cuda test_complex_ -- --nocapture
```

- If AMD hardware/runtime becomes available later, run HIP validation/compile checks and then decide whether any HIP positive implementation belongs in a follow-up.
- Before opening an upstream PR, re-read the diff for reviewability:
  - make sure the centralized validation table matches the intended v1 contract
  - make sure no backend advertises a capability family it does not implement
