# Host-Visible WGPU Primary Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in WGPU primary-memory mode whose whole-resource allocations can be mapped by the host and used by Metal-backed WGPU without copying tensor bytes.

**Architecture:** Keep the default device-local WGPU path unchanged. An opt-in `PrimaryMemoryMode::HostVisible` requests `MAPPABLE_PRIMARY_BUFFERS`, forces the main pool to `ExclusivePages`, allocates main buffers with `MAP_READ | MAP_WRITE` in addition to their GPU usages, and returns guarded mapped ranges from `WgpuResource`. Every allocation carries shared access state so a CPU mapping and GPU binding are mutually exclusive; dropping a mapping unmaps the resource and restores GPU access.

**Tech Stack:** Rust, CubeCL 0.10 fork, wgpu 26, Metal-backed WGPU, Cargo tests.

## Global Constraints

- Default behavior and non-Metal adapters remain unchanged.
- Host-visible primary memory is explicit; no runtime fallback or hidden copy is allowed.
- Host-visible allocations use whole-resource pooling (`MemoryConfiguration::ExclusivePages`) and stable shared allocation identity.
- A host mapping waits for submitted GPU work through wgpu mapping completion.
- GPU binding while a host guard is live is rejected with a typed error; overlapping host guards are rejected with a typed error.
- Dropping a host guard always unmaps and makes the resource GPU-usable again.
- Do not publish or release crates in this task; a git revision is sufficient.
- Keep fork-only changes small and compatible with the upstream 0.10 API shape.

---

### Task 1: Opt-in host-visible WGPU allocations and guarded mapping

**Files:**
- Modify: `crates/cubecl-wgpu/src/runtime.rs`
- Modify: `crates/cubecl-wgpu/src/backend/metal.rs`
- Modify: `crates/cubecl-wgpu/src/backend/base.rs`
- Modify: `crates/cubecl-wgpu/src/compute/mem_manager.rs`
- Modify: `crates/cubecl-wgpu/src/compute/storage.rs`
- Modify: `crates/cubecl-wgpu/src/compute/server.rs`
- Modify: `crates/cubecl-wgpu/src/compute/schedule.rs`
- Test: `crates/cubecl-wgpu/src/compute/storage.rs`
- Test: `crates/cubecl-wgpu/tests/host_visible_primary.rs`

**Interfaces:**
- Produces: public `PrimaryMemoryMode::{DeviceLocal, HostVisible}`.
- Produces: `RuntimeOptions { tasks_max, memory_config, primary_memory }`, with `DeviceLocal` as the default.
- Produces: `WgpuResource::allocation_id() -> u64` returning an identity shared by clones/views of one physical allocation.
- Produces: `WgpuResource::map_read() -> Result<WgpuMappedReadGuard, HostAccessError>` and `WgpuResource::map_write() -> Result<WgpuMappedWriteGuard, HostAccessError>`.
- Produces: read and write guards implementing `Deref<Target = [u8]>`; the write guard also implements `DerefMut`.
- Produces: public `HostAccessError` variants that distinguish a device-local allocation, an overlapping host mapping, a mapped resource requested for GPU use, a map callback failure, and device polling failure.
- Consumes later: tenferro obtains a `WgpuResource` through `ComputeClient::get_resource`, maps it synchronously, and records `allocation_id()` for zero-copy assertions.

- [ ] **Step 1: Add failing unit tests for allocation identity and access-state transitions**

  Add tests around the access-state object used by `WgpuResource`. Assert that resource clones retain the same nonzero allocation ID, one host access excludes another, GPU access is rejected while a host token exists, and dropping the token restores GPU access. Keep these tests independent of physical GPU availability.

- [ ] **Step 2: Run the focused unit test and confirm it fails**

  Run: `cargo test -p t4a-cubecl-wgpu compute::storage::tests --features std`

  Expected: compilation fails because the identity/access-state API does not exist.

- [ ] **Step 3: Implement stable identity, typed access errors, and RAII mapped guards**

  Store each physical buffer in a cloneable allocation object containing the `wgpu::Buffer`, `wgpu::Device`, a process-unique nonzero `u64` allocation ID, a host-visible flag, and an atomic access state. `WgpuStorage::get` must create offset/size views that share this object. Mapping must compare-and-swap GPU-idle to host-mapped before calling `map_async`; every error path must restore GPU-idle. After callback completion, poll with `wgpu::PollType::Wait`, obtain the exact logical `[offset, offset + size)` mapped range, and return an owning guard. Guard drop must release the mapped view before calling `unmap`, then restore GPU-idle. The GPU binding path must call a fallible `ensure_gpu_access()` before producing a binding resource.

- [ ] **Step 4: Add opt-in runtime configuration and Metal feature negotiation**

  Add `PrimaryMemoryMode` to `RuntimeOptions`, defaulting to `DeviceLocal`. Thread the choice through WGPU device/setup creation and `WgpuServer`/`WgpuMemManager`. When `HostVisible` is selected, require `wgpu::Features::MAPPABLE_PRIMARY_BUFFERS`; return or panic with a precise unsupported-feature message at setup rather than silently falling back. The Metal backend must request the feature only in this mode. Other backends must either request it when supported or reject setup explicitly.

- [ ] **Step 5: Allocate host-visible main buffers as exclusive whole resources**

  In host-visible mode, force only the main memory pool to `MemoryConfiguration::ExclusivePages` and add `MAP_READ | MAP_WRITE` to its existing `STORAGE | COPY_SRC | COPY_DST | INDIRECT` usages. Mark those resources host-visible. Staging and uniform pools remain unchanged. Default mode must retain its caller-provided memory configuration and usages.

- [ ] **Step 6: Reject GPU scheduling while a host guard is live**

  Make WGPU binding preparation validate every `WgpuResource` with `ensure_gpu_access()`. Convert `HostAccessError::MappedForHost` into the existing asynchronous server error path without panicking. Cover direct kernel buffers, dynamic dispatch buffers, writes, and other scheduled tasks that use primary resources.

- [ ] **Step 7: Add a macOS Metal integration test**

  Under `#[cfg(target_os = "macos")]`, initialize a Metal WGPU device with host-visible primary memory and `ExclusivePages`. Create one `u32` buffer, map-write values, drop the guard, launch a kernel that increments them, map-read the result, and assert values and an unchanged allocation ID. While a write guard is live, attempt GPU use and assert a typed/server error; after dropping it, assert the same resource remains usable. Skip with a clear message only when no Metal adapter is available.

- [ ] **Step 8: Verify formatting, focused tests, and the WGPU crate**

  Run:

  ```bash
  cargo fmt --all -- --check
  cargo test -p t4a-cubecl-wgpu --features std
  cargo clippy -p t4a-cubecl-wgpu --all-targets --features std -- -D warnings
  ```

  Expected: all commands pass, including the macOS host-visible transition test on an available Metal adapter.

- [ ] **Step 9: Commit**

  ```bash
  git add crates/cubecl-wgpu docs/superpowers/plans/2026-07-19-host-visible-primary.md
  git commit -m "feat(wgpu): add guarded host-visible primary memory"
  ```
