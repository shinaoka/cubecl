#![cfg(target_os = "macos")]

use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_runtime::server::ServerError;
use cubecl_wgpu::{
    HostAccessError, MemoryConfiguration, Metal, PrimaryMemoryMode, RuntimeOptions, WgpuDevice,
    WgpuRuntime, init_setup,
};

#[cube(launch)]
fn increment(values: &mut Array<u32>) {
    if ABSOLUTE_POS < values.len() {
        values[ABSOLUTE_POS] += 1;
    }
}

#[test]
fn compute_client_handle_resolves_to_guarded_host_visible_resource() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapters =
        cubecl_common::future::block_on(instance.enumerate_adapters(wgpu::Backends::METAL));
    if adapters.is_empty() {
        eprintln!("skipping host-visible primary test: no Metal adapter is available");
        return;
    }

    let device = WgpuDevice::DefaultDevice;
    let _setup = init_setup::<Metal>(
        &device,
        RuntimeOptions {
            tasks_max: 32,
            memory_config: MemoryConfiguration::ExclusivePages,
            primary_memory: PrimaryMemoryMode::HostVisible,
        },
    );
    let client = WgpuRuntime::client(&device);
    let handle = client.empty(4 * core::mem::size_of::<u32>());

    // This is the public downstream path used by tenferro: Handle -> ComputeClient ->
    // ManagedResource<WgpuResource> -> &WgpuResource.
    let managed = client.get_resource(handle.clone()).unwrap();
    let resource = managed.resource();
    let allocation_id = resource.allocation_id();
    let cloned_resource = resource.clone();
    assert_ne!(allocation_id, 0);
    assert_eq!(cloned_resource.allocation_id(), allocation_id);

    {
        let mut guard = resource.map_write().unwrap();
        guard.copy_from_slice(u32::as_bytes(&[1, 2, 3, 4]));

        assert_eq!(
            resource.map_read().unwrap_err(),
            HostAccessError::OverlappingHostMapping
        );

        assert_eq!(
            resource.ensure_gpu_access(),
            Err(HostAccessError::MappedForHost)
        );

        increment::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(4),
            unsafe { ArrayArg::from_raw_parts(handle.clone(), 4) },
        );
        let error = client.flush().unwrap_err();
        assert!(matches!(error, ServerError::ServerUnhealthy { .. }));
        assert!(format!("{error:?}").contains("mapped for host access"));
    }

    increment::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(4),
        unsafe { ArrayArg::from_raw_parts(handle.clone(), 4) },
    );
    client.flush().unwrap();

    let guard = resource.map_read().unwrap();
    assert_eq!(u32::from_bytes(&guard), &[2, 3, 4, 5]);
    assert_eq!(resource.allocation_id(), allocation_id);
}
