# Fresh Graphics-API Device Initialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sync and async WGPU initialization helpers that create independently registered clients under fresh device IDs for a selected graphics API.

**Architecture:** Select the adapter and create `WgpuSetup` without registering the selector, then reuse `init_device` for fresh-ID generation and client registration. Preserve all existing initialization entry points.

**Tech Stack:** Rust, CubeCL WGPU runtime, wgpu Metal backend, cargo test.

## Global Constraints

- Preserve the behavior and signatures of `init_setup`, `init_setup_async`, and `init_device`.
- Return a fresh `WgpuDevice::Existing` ID on every successful call.
- Support `PrimaryMemoryMode::HostVisible` and both synchronous native and asynchronous initialization.
- Keep the fork minimal and upstream-close.

---

### Task 1: Fresh Graphics-API Initialization

**Files:**
- Modify: `crates/cubecl-wgpu/src/runtime.rs`
- Test: `crates/cubecl-wgpu/tests/host_visible_primary.rs`

**Interfaces:**
- Consumes: `create_setup_for_device`, `init_device`, `GraphicsApi`, `RuntimeOptions`, `WgpuDevice`.
- Produces: `init_device_for_graphics_api<G>(&WgpuDevice, RuntimeOptions) -> WgpuDevice` and `init_device_for_graphics_api_async<G>(&WgpuDevice, RuntimeOptions) -> WgpuDevice`.

- [ ] **Step 1: Write the failing integration test**

Add a macOS Metal test that calls `init_device_for_graphics_api::<Metal>` twice with host-visible options, asserts the IDs differ, resolves `WgpuRuntime::client` for both IDs, allocates one handle per client, and resolves both through `ComputeClient::get_resource`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p t4a-cubecl-wgpu --test host_visible_primary independent_host_visible_metal_devices_use_fresh_context_ids --features msl -- --exact --nocapture`

Expected: compilation fails because `init_device_for_graphics_api` does not exist.

- [ ] **Step 3: Implement the async helper**

Add:

```rust
pub async fn init_device_for_graphics_api_async<G: GraphicsApi>(
    selector: &WgpuDevice,
    options: RuntimeOptions,
) -> WgpuDevice {
    let setup = create_setup_for_device(selector, G::backend(), options.primary_memory).await;
    init_device(setup, options)
}
```

- [ ] **Step 4: Implement the synchronous wrapper**

Add a native wrapper using `future::block_on(init_device_for_graphics_api_async::<G>(selector, options))`; on WebAssembly, panic with guidance to use the async function, matching `init_setup`.

- [ ] **Step 5: Run focused and API validation**

Run the focused integration command from Step 2, then `cargo fmt --all -- --check`, `cargo clippy -p t4a-cubecl-wgpu --all-targets --features std,msl -- -D warnings`, and `RUSTDOCFLAGS='-D warnings' cargo doc -p t4a-cubecl-wgpu --features std,msl --no-deps`.

Expected: all commands pass; on a host without Metal the integration test reports its existing skip path.

- [ ] **Step 6: Commit with the related review fixes**

Stage the runtime and integration changes together with the guarded host-visible primary-memory review fixes and commit with an intentional fix message.
