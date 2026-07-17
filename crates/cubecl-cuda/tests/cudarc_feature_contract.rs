#[test]
fn cudarc_uses_the_tensor4all_cuda_floor_without_fallback_detection() {
    let manifest = include_str!("../../../Cargo.toml");
    let normalized = manifest.split_whitespace().collect::<Vec<_>>().join(" ");

    assert!(normalized.contains(
        r#"cudarc = { version = "0.19.0", default-features = false, features = [ "std", "driver", "runtime", "nvrtc", "nccl", "dynamic-loading", "cuda-12080", ] }"#
    ));
    assert!(!normalized.contains("fallback-dynamic-loading"));
    assert!(!normalized.contains("cuda-version-from-build-system"));
    assert!(!normalized.contains("fallback-latest"));
}
