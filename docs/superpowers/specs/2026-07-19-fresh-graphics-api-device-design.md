# Fresh Graphics-API Device Initialization Design

## Goal

Allow callers to select a WGPU graphics API repeatedly without registering the selector device globally, returning a fresh `WgpuDevice::Existing` ID for each independent runtime client.

## API

Add these public functions beside the existing initialization helpers:

```rust
pub fn init_device_for_graphics_api<G: GraphicsApi>(
    selector: &WgpuDevice,
    options: RuntimeOptions,
) -> WgpuDevice;

pub async fn init_device_for_graphics_api_async<G: GraphicsApi>(
    selector: &WgpuDevice,
    options: RuntimeOptions,
) -> WgpuDevice;
```

The synchronous function is unavailable on WebAssembly in the same way as `init_setup`; callers there use the async function.

## Data Flow

The async function calls `create_setup_for_device(selector, G::backend(), options.primary_memory)` directly, so the selector is used only for adapter selection. It then passes the setup and options to the existing `init_device`, which generates a globally unique `WgpuDevice::Existing` ID, creates the server, and registers the client under that ID. The synchronous function blocks on the async function outside WebAssembly.

Existing `init_setup`, `init_setup_async`, and `init_device` behavior remains unchanged.

## Errors and Compatibility

Adapter/device setup keeps the existing panic behavior because the lower-level setup helpers are unchanged. Host-visible feature validation remains in `create_server`. No existing public signature or default changes.

## Testing

On macOS with an available Metal adapter, initialize two host-visible Metal devices through the new API. Assert the returned IDs are distinct, obtain a client for each ID, allocate resources independently, and verify both resource paths resolve. The existing host-visible integration test remains the downstream-path and mapping behavior test.
