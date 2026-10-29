//! Compute on the GPU (Adreno) through Vulkan: device setup, buffers and
//! compute kernels (`--features vulkan`; IMPLEMENTATION.md L.3).
//!
//! The pieces, in the order a run needs them:
//!
//! - [`Gpu`]: the Vulkan instance, the physical device (the GPU), a logical
//!   device with one compute queue, and a command pool.
//! - [`Buffer`]: GPU memory the CPU can also see. Phone GPUs share the
//!   phone's RAM with the CPU (unified memory), so a buffer is one
//!   allocation, mapped into our address space for its whole life: no
//!   copies between "host" and "device" memory.
//! - [`Kernel`]: one compute shader (SPIR-V, compiled from `shaders/*.comp`
//!   by `build.rs`), with its storage-buffer bindings and push constants.
//! - [`Gpu::run`]: records `reps` dispatches of a kernel into a command
//!   buffer with timestamps around them, submits, waits, and returns the GPU
//!   time the dispatches took.
//!
//! Every Vulkan call is `unsafe` (raw handles, lifetimes the API cannot
//! check); the types here own their handles and free them in `Drop`.

use crate::quant::BlockQ4_0;
use ash::vk;
use std::ffi::CStr;

/// The SPIR-V of `shaders/NAME.comp`, compiled by `build.rs`.
#[macro_export]
macro_rules! spirv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

pub type Result<T> = std::result::Result<T, String>;

/// Descriptor sets the pool behind [`Kernel::bind`] holds (and eight times as
/// many buffer descriptors).
const MAX_SETS: u32 = 2048;

fn vkr(what: &str) -> impl FnOnce(vk::Result) -> String + '_ {
    move |e| format!("{what}: {e}")
}

/// What the device says about itself (the parts that matter for compute).
#[derive(Debug, Clone)]
pub struct Info {
    pub name: String,
    pub api_version: (u32, u32, u32),
    pub driver_version: u32,
    /// Invocations that run in lockstep (a "wave"); 64 or 128 on Adreno.
    pub subgroup_size: u32,
    /// The subgroup size every pipeline asks for, on devices that may
    /// otherwise pick fewer than 32 lanes for a shader (Intel compiles
    /// compute shaders for 8, 16 or 32 lanes as it sees fit). The shaders
    /// assume 32-lane groups stay inside one subgroup. None elsewhere
    /// (Adreno: 64 or 128, never fewer than 32).
    pub required_subgroup_size: Option<u32>,
    /// Which subgroup operations shaders may use (arithmetic, clustered, …).
    pub subgroup_ops: vk::SubgroupFeatureFlags,
    pub max_workgroup_invocations: u32,
    pub max_storage_buffer_range: u32,
    pub max_shared_memory: u32,
    /// Byte offsets of buffer views ([`Kernel::bind_at`]) are multiples of this.
    pub storage_offset_align: u64,
    /// The device's extensions (e.g. VK_KHR_shader_float16_int8,
    /// VK_KHR_cooperative_matrix).
    pub extensions: Vec<String>,
    /// Nanoseconds per timestamp tick.
    pub timestamp_period: f32,
    /// (size in bytes, flags) per memory heap, and (heap, flags) per type.
    pub heaps: Vec<(u64, vk::MemoryHeapFlags)>,
    pub types: Vec<(u32, vk::MemoryPropertyFlags)>,
}

pub struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    phys: vk::PhysicalDevice,
    pub device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    /// Descriptor sets for [`Kernel::bind`] (a model binds a few hundred).
    sets: vk::DescriptorPool,
    pub info: Info,
}

impl Gpu {
    /// Opens the first GPU that has a compute queue.
    pub fn new() -> Result<Gpu> {
        // SAFETY: loading libvulkan.so runs its initialisers; it is the
        // system's Vulkan loader.
        let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("load libvulkan: {e}"))?;
        let app = vk::ApplicationInfo::default().application_name(c"aether").api_version(vk::API_VERSION_1_1);
        // SAFETY: valid create info; no layers or extensions.
        let instance = unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None) }.map_err(vkr("create instance"))?;
        // SAFETY: `instance` is live for every call below.
        unsafe {
            let devices = instance.enumerate_physical_devices().map_err(vkr("enumerate devices"))?;
            let (phys, family) = devices
                .iter()
                .find_map(|&d| {
                    let fams = instance.get_physical_device_queue_family_properties(d);
                    fams.iter().position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE)).map(|i| (d, i as u32))
                })
                .ok_or("no GPU with a compute queue")?;
            let props = instance.get_physical_device_properties(phys);
            let extensions: Vec<String> = instance
                .enumerate_device_extension_properties(phys)
                .map_err(vkr("device extensions"))?
                .iter()
                .map(|e| CStr::from_ptr(e.extension_name.as_ptr()).to_string_lossy().into_owned())
                .collect();
            let size_control = extensions.iter().any(|e| e == "VK_EXT_subgroup_size_control");
            let mut sub = vk::PhysicalDeviceSubgroupProperties::default();
            let mut sizes = vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sub);
            if size_control {
                p2 = p2.push_next(&mut sizes);
            }
            instance.get_physical_device_properties2(phys, &mut p2);
            let required_subgroup_size = (size_control
                && sizes.min_subgroup_size < 32
                && sizes.max_subgroup_size >= 32
                && sizes.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE))
            .then_some(32);
            let mem = instance.get_physical_device_memory_properties(phys);
            let heaps: Vec<_> = mem.memory_heaps[..mem.memory_heap_count as usize].iter().map(|h| (h.size, h.flags)).collect();
            let types: Vec<_> = mem.memory_types[..mem.memory_type_count as usize].iter().map(|t| (t.heap_index, t.property_flags)).collect();
            let v = props.api_version;
            let info = Info {
                name: CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned(),
                api_version: (vk::api_version_major(v), vk::api_version_minor(v), vk::api_version_patch(v)),
                driver_version: props.driver_version,
                subgroup_size: required_subgroup_size.unwrap_or(sub.subgroup_size),
                required_subgroup_size,
                subgroup_ops: sub.supported_operations,
                max_workgroup_invocations: props.limits.max_compute_work_group_invocations,
                max_storage_buffer_range: props.limits.max_storage_buffer_range,
                max_shared_memory: props.limits.max_compute_shared_memory_size,
                storage_offset_align: props.limits.min_storage_buffer_offset_alignment,
                extensions,
                timestamp_period: props.limits.timestamp_period,
                heaps,
                types,
            };
            let prio = [1.0f32];
            let qinfo = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&prio)];
            let ext_names = [c"VK_EXT_subgroup_size_control".as_ptr()];
            let mut size_feature = vk::PhysicalDeviceSubgroupSizeControlFeaturesEXT::default().subgroup_size_control(true);
            let mut dinfo = vk::DeviceCreateInfo::default().queue_create_infos(&qinfo);
            if required_subgroup_size.is_some() {
                dinfo = dinfo.enabled_extension_names(&ext_names).push_next(&mut size_feature);
            }
            let device = instance.create_device(phys, &dinfo, None).map_err(vkr("create device"))?;
            let queue = device.get_device_queue(family, 0);
            let pool = device
                .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
                .map_err(vkr("create command pool"))?;
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 8 * MAX_SETS }];
            let sets = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(MAX_SETS).pool_sizes(&sizes), None).map_err(vkr("descriptor pool"))?;
            Ok(Gpu { _entry: entry, instance, phys, device, queue, pool, sets, info })
        }
    }

    /// A storage buffer of `size` bytes, mapped for the CPU, in memory that
    /// is device-local, host-visible and coherent (unified memory: the GPU's
    /// fast memory is also the CPU's; coherent: no flushes needed).
    pub fn buffer(&self, size: usize) -> Result<Buffer> {
        use vk::MemoryPropertyFlags as M;
        self.buffer_with(size, M::DEVICE_LOCAL | M::HOST_VISIBLE | M::HOST_COHERENT)
    }

    /// [`Gpu::buffer`] from the first memory type that has all of `flags`
    /// (which must include HOST_VISIBLE). Adreno also offers HOST_CACHED
    /// types: the CPU reads them through its caches, which the GPU then has
    /// to keep coherent with (snooping).
    pub fn buffer_with(&self, size: usize, flags: vk::MemoryPropertyFlags) -> Result<Buffer> {
        assert!(flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE), "buffers are mapped for the CPU");
        let d = &self.device;
        // SAFETY: `d` is live; the buffer is bound to its own allocation
        // before it is mapped, and both are freed together in Drop (or here,
        // on an error).
        unsafe {
            let buf = d
                .create_buffer(&vk::BufferCreateInfo::default().size(size as u64).usage(vk::BufferUsageFlags::STORAGE_BUFFER).sharing_mode(vk::SharingMode::EXCLUSIVE), None)
                .map_err(vkr("create buffer"))?;
            let req = d.get_buffer_memory_requirements(buf);
            let Some(mem_type) = self.info.types.iter().enumerate().position(|(i, &(_, t))| req.memory_type_bits & (1 << i) != 0 && t.contains(flags)) else {
                d.destroy_buffer(buf, None);
                return Err(format!("no memory type with {flags:?} for this buffer"));
            };
            let mem = match d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mem_type as u32), None) {
                Ok(m) => m,
                Err(e) => {
                    d.destroy_buffer(buf, None);
                    return Err(format!("allocate {size} bytes: {e}"));
                }
            };
            let mapped = d.bind_buffer_memory(buf, mem, 0).and_then(|_| d.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()));
            match mapped {
                Ok(ptr) => Ok(Buffer { device: d.clone(), buf, mem, ptr: ptr as *mut u8, size, mem_type: mem_type as u32, allowed_types: req.memory_type_bits }),
                Err(e) => {
                    d.destroy_buffer(buf, None);
                    d.free_memory(mem, None);
                    Err(format!("bind and map memory: {e}"))
                }
            }
        }
    }

    /// Runs `kernel` on `buffers` (binding i = buffers[i]) with push
    /// constants `push`, as `groups` workgroups, `reps` times back to back
    /// (a barrier between runs, so each sees the previous one's writes).
    /// Returns the GPU time of all `reps` in seconds, from timestamps.
    pub fn run(&self, kernel: &Kernel, buffers: &[&Buffer], push: &[u8], groups: [u32; 3], reps: u32) -> Result<f64> {
        assert_eq!(buffers.len(), kernel.bindings, "one buffer per binding");
        let d = &self.device;
        // SAFETY: every handle is live; the descriptor set is not in use
        // (the previous run was waited for); the command buffer, query pool
        // and fence are freed before returning.
        unsafe {
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers.iter().map(|b| [vk::DescriptorBufferInfo { buffer: b.buf, offset: 0, range: vk::WHOLE_SIZE }]).collect();
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(kernel.set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info))
                .collect();
            d.update_descriptor_sets(&writes, &[]);

            let cb = d
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))
                .map_err(vkr("allocate command buffer"))?[0];
            let qp = d.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None).map_err(vkr("create query pool"))?;
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).map_err(vkr("create fence"))?;
            let result = (|| {
                d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(vkr("begin"))?;
                d.cmd_reset_query_pool(cb, qp, 0, 2);
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
                d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, kernel.layout, 0, &[kernel.set], &[]);
                if !push.is_empty() {
                    d.cmd_push_constants(cb, kernel.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
                }
                d.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
                let barrier = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
                for r in 0..reps {
                    if r > 0 {
                        d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier, &[], &[]);
                    }
                    d.cmd_dispatch(cb, groups[0], groups[1], groups[2]);
                }
                d.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1);
                // Make the shader's writes visible to the CPU's reads.
                let host = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)];
                d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &host, &[], &[]);
                d.end_command_buffer(cb).map_err(vkr("end"))?;
                let cbs = [cb];
                d.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], fence).map_err(vkr("submit"))?;
                d.wait_for_fences(&[fence], true, u64::MAX).map_err(vkr("wait"))?;
                let mut ts = [0u64; 2];
                d.get_query_pool_results(qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT).map_err(vkr("timestamps"))?;
                Ok(ts[1].wrapping_sub(ts[0]) as f64 * self.info.timestamp_period as f64 * 1e-9)
            })();
            d.destroy_fence(fence, None);
            d.destroy_query_pool(qp, None);
            d.free_command_buffers(self.pool, &[cb]);
            result
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: nothing runs on the device any more (every run waits for
        // its fence); buffers and kernels hold clones of `device` and must
        // be dropped first, which owning them after the Gpu ensures.
        unsafe {
            self.device.destroy_descriptor_pool(self.sets, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        let _ = self.phys;
    }
}

/// GPU memory, mapped for the CPU (see the module docs).
pub struct Buffer {
    device: ash::Device,
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    ptr: *mut u8,
    pub size: usize,
    /// Index into [`Info::types`], and the bit mask of the types this
    /// buffer could have used.
    pub mem_type: u32,
    pub allowed_types: u32,
}

impl Buffer {
    /// The buffer as `T`s for the CPU to read or fill. Only call it while no
    /// GPU run uses the buffer ([`Gpu::run`] waits for its runs to finish).
    pub fn as_mut_slice<T: Copy>(&mut self) -> &mut [T] {
        // SAFETY: the mapping covers `size` bytes and lives as long as the
        // buffer; Vulkan maps at an alignment of at least 64 bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.size / size_of::<T>()) }
    }

    pub fn as_slice<T: Copy>(&self) -> &[T] {
        // SAFETY: as in `as_mut_slice`.
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.size / size_of::<T>()) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: no run uses the buffer once it is dropped.
        unsafe {
            self.device.unmap_memory(self.mem);
            self.device.destroy_buffer(self.buf, None);
            self.device.free_memory(self.mem, None);
        }
    }
}

/// A compute shader with `bindings` storage buffers (bindings 0, 1, …),
/// `push_bytes` of push constants, and u32 specialization constants
/// (`constant_id` 0, 1, … = `spec[0]`, `spec[1]`, …: fixed when the pipeline
/// is built, so the driver compiles them in as constants).
pub struct Kernel {
    device: ash::Device,
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    desc_pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    bindings: usize,
    pub spec: Vec<u32>,
}

impl Kernel {
    pub fn new(gpu: &Gpu, spirv: &[u8], bindings: usize, push_bytes: u32) -> Result<Kernel> {
        Kernel::with_spec(gpu, spirv, bindings, push_bytes, &[])
    }

    pub fn with_spec(gpu: &Gpu, spirv: &[u8], bindings: usize, push_bytes: u32, spec: &[u32]) -> Result<Kernel> {
        let d = &gpu.device;
        let code = ash::util::read_spv(&mut std::io::Cursor::new(spirv)).map_err(|e| format!("SPIR-V: {e}"))?;
        // SAFETY: `d` is live; the objects are freed in Drop, in reverse.
        unsafe {
            let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(vkr("create shader module"))?;
            let binds: Vec<_> = (0..bindings as u32)
                .map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
                .collect();
            let set_layout = d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds), None).map_err(vkr("set layout"))?;
            let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: push_bytes }];
            let sets = [set_layout];
            let mut li = vk::PipelineLayoutCreateInfo::default().set_layouts(&sets);
            if push_bytes > 0 {
                li = li.push_constant_ranges(&ranges);
            }
            let layout = d.create_pipeline_layout(&li, None).map_err(vkr("pipeline layout"))?;
            let entries: Vec<vk::SpecializationMapEntry> = (0..spec.len() as u32).map(|i| vk::SpecializationMapEntry { constant_id: i, offset: 4 * i, size: 4 }).collect();
            let data: Vec<u8> = spec.iter().flat_map(|v| v.to_le_bytes()).collect();
            let si = vk::SpecializationInfo::default().map_entries(&entries).data(&data);
            let mut stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main").specialization_info(&si);
            let mut required = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT::default().required_subgroup_size(gpu.info.required_subgroup_size.unwrap_or(0));
            if gpu.info.required_subgroup_size.is_some() {
                stage = stage.push_next(&mut required);
            }
            let pipeline = d
                .create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None)
                .map_err(|(_, e)| format!("create pipeline: {e}"))?[0];
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: bindings.max(1) as u32 }];
            let desc_pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None).map_err(vkr("descriptor pool"))?;
            let set = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&sets)).map_err(vkr("descriptor set"))?[0];
            Ok(Kernel { device: d.clone(), module, set_layout, layout, pipeline, desc_pool, set, bindings, spec: spec.to_vec() })
        }
    }
}

impl Kernel {
    /// A descriptor set binding `buffers` (binding i = buffers[i]) for this
    /// kernel, to use in recorded commands ([`Gpu::record`]). Sets come from
    /// the GPU's pool and live as long as the [`Gpu`].
    pub fn bind(&self, gpu: &Gpu, buffers: &[&Buffer]) -> Result<Bound> {
        let at: Vec<(&Buffer, usize)> = buffers.iter().map(|&b| (b, 0)).collect();
        self.bind_at(gpu, &at)
    }

    /// [`Kernel::bind`] where binding i starts `at[i].1` bytes into its
    /// buffer (a multiple of `Info::storage_offset_align`): one buffer can
    /// hold several outputs, e.g. q, k and v of one fused matrix product.
    pub fn bind_at(&self, gpu: &Gpu, at: &[(&Buffer, usize)]) -> Result<Bound> {
        assert_eq!(at.len(), self.bindings, "one buffer per binding");
        let align = gpu.info.storage_offset_align as usize;
        assert!(at.iter().all(|&(b, o)| o.is_multiple_of(align) && o < b.size), "view offsets must be aligned to {align} and inside the buffer");
        let d = &gpu.device;
        let layouts = [self.set_layout];
        // SAFETY: live device, pool and layout; the set is written before
        // any command buffer uses it.
        unsafe {
            let set = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(gpu.sets).set_layouts(&layouts)).map_err(vkr("descriptor set"))?[0];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = at.iter().map(|&(b, o)| [vk::DescriptorBufferInfo { buffer: b.buf, offset: o as u64, range: vk::WHOLE_SIZE }]).collect();
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info))
                .collect();
            d.update_descriptor_sets(&writes, &[]);
            Ok(Bound { pipeline: self.pipeline, layout: self.layout, set })
        }
    }
}

/// A kernel with its buffers bound ([`Kernel::bind`]).
#[derive(Clone, Copy)]
pub struct Bound {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set: vk::DescriptorSet,
}

/// Commands being recorded by [`Gpu::record`].
pub struct Recorder<'a> {
    device: &'a ash::Device,
    cb: vk::CommandBuffer,
    /// With [`Gpu::record_timed`]: the query pool and how many stamps.
    queries: Option<(vk::QueryPool, u32)>,
}

impl Recorder<'_> {
    /// One dispatch of `k` as `groups` workgroups, with push constants.
    pub fn dispatch(&mut self, k: &Bound, push: &[u8], groups: [u32; 3]) {
        // SAFETY: recording into a live command buffer; the handles in `k`
        // live as long as the Gpu and its kernels.
        unsafe {
            let d = self.device;
            d.cmd_bind_pipeline(self.cb, vk::PipelineBindPoint::COMPUTE, k.pipeline);
            d.cmd_bind_descriptor_sets(self.cb, vk::PipelineBindPoint::COMPUTE, k.layout, 0, &[k.set], &[]);
            if !push.is_empty() {
                d.cmd_push_constants(self.cb, k.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            }
            d.cmd_dispatch(self.cb, groups[0], groups[1], groups[2]);
        }
    }

    /// A GPU timestamp once everything before it has finished (only in
    /// [`Gpu::record_timed`] commands; otherwise nothing). Returns its index.
    pub fn stamp(&mut self) -> usize {
        let Some((qp, n)) = &mut self.queries else { return 0 };
        assert!(*n < MAX_STAMPS, "more than {MAX_STAMPS} stamps");
        // SAFETY: recording into a live command buffer; the query is reset
        // at the start of the buffer.
        unsafe { self.device.cmd_write_timestamp(self.cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, *qp, *n) };
        *n += 1;
        *n as usize - 1
    }

    /// Everything dispatched so far finishes, and its writes are visible,
    /// before anything dispatched later starts.
    pub fn barrier(&mut self) {
        let b = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        // SAFETY: recording into a live command buffer.
        unsafe { self.device.cmd_pipeline_barrier(self.cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &b, &[], &[]) };
    }
}

/// A recorded command buffer that can be submitted any number of times
/// ([`Gpu::submit`]): a model's whole decode step, recorded once.
pub struct Commands {
    device: ash::Device,
    pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    queries: Option<(vk::QueryPool, u32)>,
}

/// Timestamps one [`Gpu::record_timed`] command buffer may hold.
const MAX_STAMPS: u32 = 1024;

impl Drop for Commands {
    fn drop(&mut self) {
        // SAFETY: every submit waited for its fence, so nothing is running.
        unsafe {
            if let Some((qp, _)) = self.queries {
                self.device.destroy_query_pool(qp, None);
            }
            self.device.destroy_fence(self.fence, None);
            self.device.free_command_buffers(self.pool, &[self.cb]);
        }
    }
}

impl Gpu {
    /// Records commands with `f` into a command buffer to submit later.
    /// The host's writes to mapped buffers before a submit are visible to
    /// the commands, and the commands' writes are visible to the host after
    /// the submit returns.
    pub fn record(&self, f: impl FnOnce(&mut Recorder)) -> Result<Commands> {
        self.record_with(false, f)
    }

    /// [`Gpu::record`] where [`Recorder::stamp`] writes GPU timestamps, read
    /// after a submit with [`Gpu::stamps`]. A stamp is written at the start
    /// too, so stamp i's time since the start is `stamps()[i + 1]`.
    pub fn record_timed(&self, f: impl FnOnce(&mut Recorder)) -> Result<Commands> {
        self.record_with(true, f)
    }

    /// Seconds from the start of the last submit of `c` to each stamp
    /// (the first entry is the start itself, 0).
    pub fn stamps(&self, c: &Commands) -> Result<Vec<f64>> {
        let Some((qp, n)) = c.queries else { return Ok(Vec::new()) };
        let mut ts = vec![0u64; n as usize];
        // SAFETY: the last submit was waited for, so all n queries are written.
        unsafe { self.device.get_query_pool_results(qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT) }.map_err(vkr("timestamps"))?;
        Ok(ts.iter().map(|&t| t.wrapping_sub(ts[0]) as f64 * self.info.timestamp_period as f64 * 1e-9).collect())
    }

    fn record_with(&self, timed: bool, f: impl FnOnce(&mut Recorder)) -> Result<Commands> {
        let d = &self.device;
        // SAFETY: live device and pool; the buffer is recorded completely
        // before it is returned.
        unsafe {
            let cb = d
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))
                .map_err(vkr("allocate command buffer"))?[0];
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).map_err(vkr("create fence"))?;
            let qp = if timed { Some(d.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(MAX_STAMPS), None).map_err(vkr("create query pool"))?) } else { None };
            let mut cmds = Commands { device: d.clone(), pool: self.pool, cb, fence, queries: qp.map(|q| (q, 0)) };
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default()).map_err(vkr("begin"))?;
            if let Some(q) = qp {
                d.cmd_reset_query_pool(cb, q, 0, MAX_STAMPS);
            }
            let host_in = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::HOST_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ)];
            d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::HOST, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &host_in, &[], &[]);
            let mut rec = Recorder { device: d, cb, queries: qp.map(|q| (q, 0)) };
            rec.stamp();
            f(&mut rec);
            cmds.queries = rec.queries;
            let host_out = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)];
            d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &host_out, &[], &[]);
            d.end_command_buffer(cb).map_err(vkr("end"))?;
            Ok(cmds)
        }
    }

    /// Runs recorded commands and waits until they are done. With `spin`,
    /// the CPU polls the fence (for up to 100 ms, then sleeps on it) instead
    /// of sleeping until the driver wakes it: the next step starts sooner.
    pub fn submit(&self, c: &Commands, spin: bool) -> Result<()> {
        let d = &self.device;
        let cbs = [c.cb];
        // SAFETY: the command buffer is fully recorded and not running (the
        // previous submit waited); the fence is reset before reuse.
        unsafe {
            d.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], c.fence).map_err(vkr("submit"))?;
            let t = std::time::Instant::now();
            while spin && t.elapsed().as_millis() < 100 && !d.get_fence_status(c.fence).map_err(vkr("fence status"))? {
                std::hint::spin_loop();
            }
            d.wait_for_fences(&[c.fence], true, u64::MAX).map_err(vkr("wait"))?;
            d.reset_fences(&[c.fence]).map_err(vkr("reset fence"))
        }
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: no run uses the kernel once it is dropped.
        unsafe {
            let d = &self.device;
            d.destroy_descriptor_pool(self.desc_pool, None);
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
            d.destroy_shader_module(self.module, None);
        }
    }
}

/// A Q4_0 matrix in GPU memory, in the layout `shaders/q4_0_gemv.comp`
/// reads: the nibble bytes of all blocks, then the f16 scales.
pub struct MatQ4_0 {
    pub qs: Buffer,
    pub d: Buffer,
    pub rows: usize,
    /// Blocks per row (columns / 32).
    pub nb: usize,
}

impl MatQ4_0 {
    /// Uploads `blocks` (row-major, `rows` rows).
    pub fn new(gpu: &Gpu, blocks: &[BlockQ4_0], rows: usize) -> Result<MatQ4_0> {
        assert!(rows > 0 && blocks.len().is_multiple_of(rows));
        let mut qs = gpu.buffer(blocks.len() * 16)?;
        // Scales two per u32: round up to a whole one.
        let mut d = gpu.buffer(blocks.len().div_ceil(2) * 4)?;
        for (dst, b) in qs.as_mut_slice::<[u8; 16]>().iter_mut().zip(blocks) {
            *dst = b.qs;
        }
        for (dst, b) in d.as_mut_slice::<u16>().iter_mut().zip(blocks) {
            *dst = b.d;
        }
        Ok(MatQ4_0 { qs, d, rows, nb: blocks.len() / rows })
    }
}

/// Workgroups for `n` items in groups of `per` (rows per workgroup), split
/// over x and y because one dimension allows at most 65535.
pub fn groups_for(n: usize, per: usize) -> [u32; 3] {
    let g = n.div_ceil(per);
    let x = g.min(65535);
    [x as u32, g.div_ceil(x) as u32, 1]
}

impl Gpu {
    /// `out[r] = W[r] · x` for a Q4_0 matrix on the GPU, `reps` times;
    /// `kernel` is `q4_0_gemv` (see [`Gpu::q4_0_gemv_kernel`]). Returns the
    /// GPU time of all reps in seconds.
    pub fn q4_0_gemv(&self, kernel: &Kernel, w: &MatQ4_0, x: &Buffer, out: &Buffer, reps: u32) -> Result<f64> {
        assert!(x.size >= w.nb * 32 * 4 && out.size >= w.rows * 4);
        let push: Vec<u8> = [w.rows as u32, w.nb as u32].iter().flat_map(|v| v.to_le_bytes()).collect();
        // 128 lanes = 4 groups of 32, each R = 4 · spec[0] rows.
        // Binding 4 (the norm's weights) is unused without NORM.
        self.run(kernel, &[&w.qs, &w.d, x, out, x], &push, groups_for(w.rows, 16 * kernel.spec[0] as usize), reps)
    }

    /// The shader for `rows_per_32` rows per 32 lanes, a multiple of 4 (see
    /// the shader; 8 is fastest for big matrices, 4 for small ones). It
    /// adds partial sums across 32 subgroup lanes by shuffles.
    pub fn q4_0_gemv_kernel(&self, rows_per_32: u32) -> Result<Kernel> {
        self.q4_0_gemv_kernel_fused(rows_per_32, false, false, false)
    }

    /// [`Gpu::q4_0_gemv_kernel`] that, with `accum`, adds the product to
    /// `out` (`out += W x`: the residual add fused in) and, with `glu`,
    /// reads x as gate then up (twice the columns) and multiplies by
    /// silu(gate) · up (the SwiGLU fused in), and with `norm`, RMS-normalizes
    /// x first with the weights in binding 4 and the epsilon pushed after
    /// rows and nb (the RMS norm fused in).
    pub fn q4_0_gemv_kernel_fused(&self, rows_per_32: u32, accum: bool, glu: bool, norm: bool) -> Result<Kernel> {
        assert!(!(glu && norm), "a GLU input is not normalized");
        assert!(rows_per_32 > 0 && rows_per_32.is_multiple_of(4));
        if self.info.subgroup_size < 32 || !self.info.subgroup_ops.contains(vk::SubgroupFeatureFlags::SHUFFLE) {
            return Err(format!("q4_0_gemv needs subgroup shuffles and subgroups of at least 32 lanes ({} lanes, {:?})", self.info.subgroup_size, self.info.subgroup_ops));
        }
        Kernel::with_spec(self, crate::spirv!("q4_0_gemv"), 5, 12, &[rows_per_32 / 4, accum.into(), glu.into(), norm.into()])
    }
}

impl Gpu {
    /// `out[t][r] = W[r] · x[t]` for `tokens` activations (row-major in
    /// `x`, `out` is tokens × rows), `reps` times; `kernel` from
    /// [`Gpu::q4_0_gemm_kernel`]. Returns the GPU time of all reps.
    pub fn q4_0_gemm(&self, kernel: &Kernel, w: &MatQ4_0, x: &Buffer, tokens: usize, out: &Buffer, reps: u32) -> Result<f64> {
        assert!(x.size >= tokens * w.nb * 32 * 4 && out.size >= tokens * w.rows * 4);
        let push: Vec<u8> = [w.rows as u32, w.nb as u32, tokens as u32].iter().flat_map(|v| v.to_le_bytes()).collect();
        // Tiles of 64 rows × 64 tokens.
        let [gx, gy] = [w.rows.div_ceil(64) as u32, tokens.div_ceil(64) as u32];
        assert!(gx <= 65535 && gy <= 65535, "too many rows or tokens for one dispatch");
        self.run(kernel, &[&w.qs, &w.d, x, out], &push, [gx, gy, 1], reps)
    }

    /// The prefill shader (tiles in shared memory; see the shader).
    pub fn q4_0_gemm_kernel(&self) -> Result<Kernel> {
        self.q4_0_gemm_kernel_fused(false, false)
    }

    /// [`Gpu::q4_0_gemm_kernel`] with `accum` (`out += X Wᵀ`) and `glu`
    /// (X's rows are gate then up, multiplied as silu(gate) · up), as for
    /// [`Gpu::q4_0_gemv_kernel_fused`].
    pub fn q4_0_gemm_kernel_fused(&self, accum: bool, glu: bool) -> Result<Kernel> {
        Kernel::with_spec(self, crate::spirv!("q4_0_gemm"), 4, 12, &[accum.into(), glu.into()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// llama.cpp's Q4_0 blocks as written by tools/gen_quant.py: 18 bytes
    /// each, f16 scale first.
    fn read_q4_0(path: &std::path::Path) -> Vec<BlockQ4_0> {
        let bytes = std::fs::read(path).unwrap();
        bytes.as_chunks::<18>().0.iter().map(|c| BlockQ4_0 { d: u16::from_le_bytes([c[0], c[1]]), qs: c[2..].try_into().unwrap() }).collect()
    }

    /// The copy shader copies (a buffer round trip through the GPU: setup,
    /// mapping, bindings, push constants and barriers all have to work).
    /// Skipped where there is no Vulkan GPU (e.g. a PC without drivers).
    #[test]
    fn copy_shader_copies() {
        let Ok(gpu) = Gpu::new() else { return };
        let k = Kernel::new(&gpu, crate::spirv!("copy"), 2, 4).unwrap();
        let n = 1000 * 4 + 3; // not a multiple of the workgroup
        let mut a = gpu.buffer(n * 4).unwrap();
        let mut b = gpu.buffer(n * 4).unwrap();
        a.as_mut_slice::<f32>().iter_mut().enumerate().for_each(|(i, v)| *v = i as f32 * 0.5);
        b.as_mut_slice::<f32>().fill(-1.0);
        gpu.run(&k, &[&a, &b], &((n / 4) as u32).to_le_bytes(), [3, 1, 1], 2).unwrap();
        let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
        assert_eq!(&b[..n / 4 * 4], &a[..n / 4 * 4]);
        assert_eq!(&b[n / 4 * 4..], &[-1.0; 3], "past n4 vec4s nothing is written");
    }

    /// The Q4_0 GEMV shader on llama.cpp's own Q4_0 bytes (`w_q4_0.bin`)
    /// against dequant(W) @ x in float64 (`out_q4_0_a`, tools/gen_quant.py).
    #[test]
    fn q4_0_gemv_matches_reference() {
        let Some(root) = crate::testutil::golden("quant") else { return };
        let Ok(gpu) = Gpu::new() else { return };
        let ks: Vec<Kernel> = [4, 8, 12].map(|r| gpu.q4_0_gemv_kernel(r).unwrap()).into();
        let mut cases = 0;
        for dir in std::fs::read_dir(root).unwrap().map(|e| e.unwrap().path()) {
            cases += 1;
            let (ws, _) = crate::testutil::read_npy(&dir.join("w.npy"));
            let (_, xv) = crate::testutil::read_npy(&dir.join("x.npy"));
            let (_, want) = crate::testutil::read_npy(&dir.join("out_q4_0_a.npy"));
            let blocks = read_q4_0(&dir.join("w_q4_0.bin"));
            let w = MatQ4_0::new(&gpu, &blocks, ws[0]).unwrap();
            let mut x = gpu.buffer(xv.len() * 4).unwrap();
            x.as_mut_slice::<f32>().copy_from_slice(&xv);
            let mut out = gpu.buffer(ws[0] * 4).unwrap();
            for k in &ks {
                out.as_mut_slice::<f32>().fill(f32::NAN);
                gpu.q4_0_gemv(k, &w, &x, &out, 1).unwrap();
                let (_, rel) = crate::testutil::max_errors(&out.as_slice::<f32>()[..ws[0]], &want);
                assert!(rel < 1e-6 * (ws[1] as f32).sqrt().max(4.0), "{} R={}: rel {rel:.2e}", dir.display(), 4 * k.spec[0]);
            }
        }
        assert!(cases >= 12, "only {cases} golden cases");
    }

    /// The Q4_0 prefill shader on llama.cpp's Q4_0 bytes against
    /// X @ dequant(W)ᵀ in float64 (`out_q4_0_a`, the batch goldens: odd row
    /// and token counts, so tiles have tails).
    #[test]
    fn q4_0_gemm_matches_reference() {
        let Some(root) = crate::testutil::golden("quant_batch") else { return };
        let Ok(gpu) = Gpu::new() else { return };
        let ks = [gpu.q4_0_gemm_kernel().unwrap()];
        let mut cases = 0;
        for dir in std::fs::read_dir(root).unwrap().map(|e| e.unwrap().path()) {
            cases += 1;
            let (ws, _) = crate::testutil::read_npy(&dir.join("w.npy"));
            let (xs, xv) = crate::testutil::read_npy(&dir.join("x.npy"));
            let (_, want) = crate::testutil::read_npy(&dir.join("out_q4_0_a.npy"));
            let blocks = read_q4_0(&dir.join("w_q4_0.bin"));
            let w = MatQ4_0::new(&gpu, &blocks, ws[0]).unwrap();
            let mut x = gpu.buffer(xv.len() * 4).unwrap();
            x.as_mut_slice::<f32>().copy_from_slice(&xv);
            let mut out = gpu.buffer(want.len() * 4).unwrap();
            for k in &ks {
                out.as_mut_slice::<f32>().fill(f32::NAN);
                gpu.q4_0_gemm(k, &w, &x, xs[0], &out, 1).unwrap();
                let (_, rel) = crate::testutil::max_errors(out.as_slice::<f32>(), &want);
                assert!(rel < 1e-6 * (ws[1] as f32).sqrt().max(4.0), "{}: rel {rel:.2e}", dir.display());
            }
        }
        assert!(cases >= 11, "only {cases} golden cases");
    }
}
