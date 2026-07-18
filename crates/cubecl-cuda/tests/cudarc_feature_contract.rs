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

#[test]
fn cudarc_build_dependency_only_enables_build_script_requirements() {
    let manifest = include_str!("../Cargo.toml");
    let normalized = manifest.split_whitespace().collect::<Vec<_>>().join(" ");

    assert!(normalized.contains(
        r#"[build-dependencies] cudarc = { version = "0.19.0", default-features = false, features = ["std", "driver", "dynamic-loading", "cuda-12080"] }"#
    ));
}

#[test]
fn cuda_12_8_tensor_map_symbol_is_runtime_guarded() {
    let runtime = include_str!("../src/runtime.rs");
    let server = include_str!("../src/compute/server.rs");

    assert!(runtime.contains("cuDriverGetVersion"));
    assert!(runtime.contains("nvrtcVersion"));

    let tensor_map_loop = server
        .find("for TensorMapBinding")
        .expect("tensor map encoding loop must exist");
    let symbol_call = server[tensor_map_loop..]
        .find("cuTensorMapEncodeIm2colWide(")
        .expect("Im2colWide symbol call must exist");
    let guarded_prefix = &server[tensor_map_loop..tensor_map_loop + symbol_call];
    assert!(guarded_prefix.contains("supports_cuda_12_8"));
}
