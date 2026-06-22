> **tensor4all temporary fork.** This `t4a-cubecl*` package is a temporary tensor4all fork used by `tenferro-rs` while required patches are being upstreamed to the official CubeCL project. It is not a replacement for upstream CubeCL.

# CubeCL Common

The `cubecl-common` package hosts code that _must_ be shared between cubecl packages (with `std` or
`no_std` enabled). No other code should be placed in this package unless unavoidable.

The package must build with `cargo build --no-default-features` as well.
