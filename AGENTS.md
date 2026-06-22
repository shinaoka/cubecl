# AGENTS.md

This repository is the `tensor4all/cubecl` fork of upstream
`tracel-ai/cubecl`.

## Fork Purpose

- This fork exists so `tenferro-rs` can be published on crates.io while the
  CubeCL changes it needs are pending upstream.
- Maintain this fork only until the required changes are merged upstream and
  `tenferro-rs` can depend on upstream CubeCL again.
- Use lowercase `tensor4all` in repository-facing text and package
  descriptions.

## Upstream Tracking

- Keep this fork as close to upstream CubeCL as possible.
- Prefer merging upstream release branches used by `tenferro-rs`, starting
  with `upstream/release/0.10`, instead of tracking unstable upstream `main`
  by default.
- Keep fork-only changes small, reviewable, and grouped by purpose:
  upstream merge commits, package-publication metadata, and tenferro-required
  implementation changes.
- Do not rebrand source modules, Rust crate import names, directory names, or
  internal API paths unless required for publication. Rebranding source code
  makes future upstream merges harder.
- When updating from upstream, merge the selected upstream release branch, then
  reapply or adjust the minimal `tensor4all` packaging delta.

## Crates.io Publishing Scope

- Published packages must use a `t4a-` package prefix so they are clearly
  distinct from upstream CubeCL packages on crates.io.
- Keep dependency keys and code import paths close to upstream. For example,
  tenferro may depend on:

  ```toml
  cubecl = { package = "t4a-cubecl", version = "=0.10.0", features = ["cuda"] }
  cubecl-cuda = { package = "t4a-cubecl-cuda", version = "=0.10.0" }
  cubecl-runtime = { package = "t4a-cubecl-runtime", version = "=0.10.0" }
  ```

- Publish only the first-party crate dependency closure required by
  `tenferro-rs`. Do not publish unused CubeCL workspace crates unless tenferro
  starts requiring them.
- For the current CUDA and WebGPU paths, the expected publish set is:
  `t4a-cubecl`, `t4a-cubecl-common`, `t4a-cubecl-core`,
  `t4a-cubecl-ir`, `t4a-cubecl-macros`,
  `t4a-cubecl-macros-internal`, `t4a-cubecl-runtime`,
  `t4a-cubecl-zspace`, `t4a-cubecl-std`, `t4a-cubecl-cuda`,
  `t4a-cubecl-wgpu`, `t4a-cubecl-cpp`, and `t4a-cubecl-opt`.
- Do not publish `cubecl-cpu`, `cubecl-hip`, `cubecl-spirv`, examples, or
  `xtask` unless they become required by tenferro.

## Public Messaging

- The repository README and crates.io package pages must state that these are
  `tensor4all`-maintained fork packages.
- Public messaging must say the fork is maintained to support `tenferro-rs`
  until the required changes are merged upstream.
- Public messaging must not imply that `tensor4all` owns upstream CubeCL.
