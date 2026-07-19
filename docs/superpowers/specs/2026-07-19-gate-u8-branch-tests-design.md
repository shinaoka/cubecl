# Gate `u8` Branch Runtime Tests by Supported Types

## Problem

`testgen_branch!` generates the `u8` literal branch tests once for every
floating-point test module. WGSL does not support `u8`, so the WGPU test suite
generates four runtime tests that can only panic in the WGSL compiler.

## Design

Keep the shared `u8` kernel and runtime assertion, but move test generation to
a new helper macro that accepts one unsigned type token. The macro generates
the two tests only for the `u8` token and expands to nothing for other unsigned
types. `testgen_all!` invokes this helper from each generated unsigned-type
module.

This uses each backend's existing unsigned-type list as the capability source:
CPU, CUDA, and SPIR-V retain one true/false pair under `u8_ty`, while WGSL and
MSL generate no `u8` runtime tests. No backend-specific conditionals or `u8`
emulation are introduced.

## Verification

- Before the change, the focused WGPU command must reproduce four failures.
- After the change, the same filter must find zero WGPU tests and exit cleanly.
- Macro expansion must compile for both a list containing `u8` and a list that
  omits it.
- The full WGPU library suite must pass on the local Metal backend.
