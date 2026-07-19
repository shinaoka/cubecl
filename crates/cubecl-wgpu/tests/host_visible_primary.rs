#![cfg(target_os = "macos")]

use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_runtime::server::ServerError;
use cubecl_wgpu::{
    HostAccessError, MemoryConfiguration, Metal, PrimaryMemoryMode, RuntimeOptions, WgpuDevice,
    WgpuRuntime, init_device_for_graphics_api, init_setup,
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
    // Barrier on the server channel: binding preparation and scheduler registration have completed,
    // but the queued encoder has not been submitted yet.
    client.exclusive(|| ()).unwrap();
    let pending_resource = resource.clone();
    let host_attempt = std::thread::spawn(move || pending_resource.map_read().unwrap_err());
    assert_eq!(
        host_attempt.join().unwrap(),
        HostAccessError::GpuAccessInProgress
    );
    client.flush().unwrap();

    let guard = resource.map_read().unwrap();
    assert_eq!(u32::from_bytes(&guard), &[2, 3, 4, 5]);
    assert_eq!(resource.allocation_id(), allocation_id);
    drop(guard);
    drop(managed);
    drop(handle);

    // A cloned resource can escape the ManagedResource borrow. It must retain the managed-memory
    // binding so cleanup cannot recycle its slice for a new allocation while the clone is alive.
    let replacement_handle = client.empty(4 * core::mem::size_of::<u32>());
    let replacement_managed = client.get_resource(replacement_handle).unwrap();
    let replacement = replacement_managed.resource();
    assert_ne!(replacement.allocation_id(), cloned_resource.allocation_id());

    {
        let mut replacement_guard = replacement.map_write().unwrap();
        replacement_guard.copy_from_slice(u32::as_bytes(&[10, 20, 30, 40]));
    }
    let stale_guard = cloned_resource.map_read().unwrap();
    drop(cloned_resource);

    // The mapped guard is now the only remaining lease for the original allocation.
    let guard_replacement_handle = client.empty(4 * core::mem::size_of::<u32>());
    let guard_replacement_managed = client.get_resource(guard_replacement_handle).unwrap();
    assert_ne!(
        guard_replacement_managed.resource().allocation_id(),
        allocation_id
    );
    assert_eq!(u32::from_bytes(&stale_guard), &[2, 3, 4, 5]);
}

#[test]
fn independent_host_visible_metal_devices_use_fresh_context_ids() {
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

    let options = || RuntimeOptions {
        tasks_max: 32,
        memory_config: MemoryConfiguration::ExclusivePages,
        primary_memory: PrimaryMemoryMode::HostVisible,
    };
    let selector = WgpuDevice::DefaultDevice;
    let first = init_device_for_graphics_api::<Metal>(&selector, options());
    let second = init_device_for_graphics_api::<Metal>(&selector, options());

    assert_ne!(first, second);
    assert!(matches!(first, WgpuDevice::Existing(_)));
    assert!(matches!(second, WgpuDevice::Existing(_)));

    let first_client = WgpuRuntime::client(&first);
    let second_client = WgpuRuntime::client(&second);
    let first_handle = first_client.empty(core::mem::size_of::<u32>());
    let second_handle = second_client.empty(core::mem::size_of::<u32>());
    let first_resource = first_client.get_resource(first_handle).unwrap();
    let second_resource = second_client.get_resource(second_handle).unwrap();

    assert_ne!(
        first_resource.resource().allocation_id(),
        second_resource.resource().allocation_id()
    );
}
