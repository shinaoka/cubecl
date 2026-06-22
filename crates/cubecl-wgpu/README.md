> **tensor4all temporary fork.** This `t4a-cubecl*` package is a temporary tensor4all fork used by `tenferro-rs` while required patches are being upstreamed to the official CubeCL project. It is not a replacement for upstream CubeCL.

# CubeCL WGPU Runtime

[CubeCL](https://github.com/tracel-ai/cubecl) WGPU runtime.

## Configuration

You can set `CUBECL_WGPU_MAX_TASKS` to a positive integer that determines how many computing tasks are submitted in batches to the graphics API.

## Platform Support

| Option    | CPU | GPU | Linux | MacOS | Windows | Android | iOS | WASM |
| :-------- | :-: | :-: | :---: | :---: | :-----: | :-----: | :-: | :--: |
| Metal     | No  | Yes |  No   |  Yes  |   No    |   No    | Yes |  No  |
| Vulkan    | Yes | Yes |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| OpenGL    | No  | Yes |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| WebGpu    | No  | Yes |  No   |  No   |   No    |   No    | No  | Yes  |
| Dx11/Dx12 | No  | Yes |  No   |  No   |   Yes   |   No    | No  |  No  |
