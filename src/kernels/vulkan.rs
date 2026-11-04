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

use crate::quant::{self, BlockQ4_0};
use crate::weight::Weight;
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
const MAX_SETS: u32 = 4096;

fn desc_type(texel: &[usize], binding: usize) -> vk::DescriptorType {
    if texel.contains(&binding) {
        vk::DescriptorType::UNIFORM_TEXEL_BUFFER
    } else {
        vk::DescriptorType::STORAGE_BUFFER
    }
}

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
    /// Texels a texel buffer view may span (`maxTexelBufferElements`).
    pub max_texel_elements: u32,
    pub max_shared_memory: u32,
    /// Byte offsets of buffer views ([`Kernel::bind_at`]) are multiples of this.
    pub storage_offset_align: u64,
    /// The device's extensions (e.g. VK_KHR_shader_float16_int8,
    /// VK_KHR_cooperative_matrix).
    pub extensions: Vec<String>,
    /// Nanoseconds per timestamp tick.
    pub timestamp_period: f32,
    /// fp16 arithmetic and 16-bit storage in shaders are on
    /// (`shaderFloat16` and `storageBuffer16BitAccess`), as the f16
    /// prefill shader needs ([`Gpu::q4_0_gemm_h`]).
    pub fp16: bool,
    /// Shaders may use int8 dot products (`shaderIntegerDotProduct`,
    /// VK_KHR_shader_integer_dot_product), and whether the device says the
    /// signed 4 × int8 packed form is accelerated (faster than the same
    /// done with ordinary integer instructions).
    pub int_dot: bool,
    pub int_dot_accelerated: bool,
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
    /// The buffer views of [`Kernel::bind_at`]'s texel bindings, freed with the device.
    views: std::sync::Mutex<Vec<vk::BufferView>>,
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
            // fp16 arithmetic (VK_KHR_shader_float16_int8) and f16 values in
            // storage buffers (16-bit storage, core in Vulkan 1.1).
            let f16_ext = extensions.iter().any(|e| e == "VK_KHR_shader_float16_int8");
            let mut f16_feat = vk::PhysicalDeviceShaderFloat16Int8Features::default();
            let mut s16_feat = vk::PhysicalDevice16BitStorageFeatures::default();
            if f16_ext {
                let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f16_feat).push_next(&mut s16_feat);
                instance.get_physical_device_features2(phys, &mut f2);
            }
            let fp16 = f16_ext && f16_feat.shader_float16 == vk::TRUE && s16_feat.storage_buffer16_bit_access == vk::TRUE;
            let dot_ext = extensions.iter().any(|e| e == "VK_KHR_shader_integer_dot_product");
            let mut dot_feat = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
            let mut dot_props = vk::PhysicalDeviceShaderIntegerDotProductProperties::default();
            if dot_ext {
                let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut dot_feat);
                instance.get_physical_device_features2(phys, &mut f2);
                let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut dot_props);
                instance.get_physical_device_properties2(phys, &mut p2);
            }
            let int_dot = dot_ext && dot_feat.shader_integer_dot_product == vk::TRUE;
            let int_dot_accelerated = int_dot && dot_props.integer_dot_product4x8_bit_packed_signed_accelerated == vk::TRUE;
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
                max_texel_elements: props.limits.max_texel_buffer_elements,
                max_shared_memory: props.limits.max_compute_shared_memory_size,
                storage_offset_align: props.limits.min_storage_buffer_offset_alignment,
                extensions,
                timestamp_period: props.limits.timestamp_period,
                fp16,
                int_dot,
                int_dot_accelerated,
                heaps,
                types,
            };
            let prio = [1.0f32];
            let qinfo = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&prio)];
            let mut ext_names = Vec::new();
            let mut size_feature = vk::PhysicalDeviceSubgroupSizeControlFeaturesEXT::default().subgroup_size_control(true);
            let mut f16_on = vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(true);
            let mut s16_on = vk::PhysicalDevice16BitStorageFeatures::default().storage_buffer16_bit_access(true);
            let mut dot_on = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default().shader_integer_dot_product(true);
            let mut dinfo = vk::DeviceCreateInfo::default().queue_create_infos(&qinfo);
            if required_subgroup_size.is_some() {
                ext_names.push(c"VK_EXT_subgroup_size_control".as_ptr());
                dinfo = dinfo.push_next(&mut size_feature);
            }
            if fp16 {
                ext_names.push(c"VK_KHR_shader_float16_int8".as_ptr());
                dinfo = dinfo.push_next(&mut f16_on).push_next(&mut s16_on);
            }
            if int_dot {
                ext_names.push(c"VK_KHR_shader_integer_dot_product".as_ptr());
                dinfo = dinfo.push_next(&mut dot_on);
            }
            dinfo = dinfo.enabled_extension_names(&ext_names);
            let device = instance.create_device(phys, &dinfo, None).map_err(vkr("create device"))?;
            let queue = device.get_device_queue(family, 0);
            let pool = device
                .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
                .map_err(vkr("create command pool"))?;
            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 8 * MAX_SETS },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_TEXEL_BUFFER, descriptor_count: MAX_SETS },
            ];
            let sets = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(MAX_SETS).pool_sizes(&sizes), None).map_err(vkr("descriptor pool"))?;
            Ok(Gpu { _entry: entry, instance, phys, device, queue, pool, sets, views: Default::default(), info })
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
                .create_buffer(&vk::BufferCreateInfo::default().size(size as u64).usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER).sharing_mode(vk::SharingMode::EXCLUSIVE), None)
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
        assert!(kernel.texel.is_empty(), "texel bindings are bound with Kernel::bind");
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
            for v in self.views.get_mut().unwrap().drain(..) {
                self.device.destroy_buffer_view(v, None);
            }
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
    /// Bindings that are uniform texel buffers (`samplerBuffer`, f16 × 4 texels).
    texel: Vec<usize>,
    pub spec: Vec<u32>,
}

impl Kernel {
    pub fn new(gpu: &Gpu, spirv: &[u8], bindings: usize, push_bytes: u32) -> Result<Kernel> {
        Kernel::with_spec(gpu, spirv, bindings, push_bytes, &[])
    }

    pub fn with_spec(gpu: &Gpu, spirv: &[u8], bindings: usize, push_bytes: u32, spec: &[u32]) -> Result<Kernel> {
        Kernel::with_texel(gpu, spirv, bindings, push_bytes, spec, &[])
    }

    /// [`Kernel::with_spec`] where the bindings in `texel` are uniform texel
    /// buffers, read with `texelFetch` as 4 × f16 texels
    /// (R16G16B16A16_SFLOAT), the way an image would be.
    pub fn with_texel(gpu: &Gpu, spirv: &[u8], bindings: usize, push_bytes: u32, spec: &[u32], texel: &[usize]) -> Result<Kernel> {
        let d = &gpu.device;
        let code = ash::util::read_spv(&mut std::io::Cursor::new(spirv)).map_err(|e| format!("SPIR-V: {e}"))?;
        // SAFETY: `d` is live; the objects are freed in Drop, in reverse.
        unsafe {
            let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(vkr("create shader module"))?;
            let binds: Vec<_> = (0..bindings as u32)
                .map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(desc_type(texel, i as usize)).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
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
            let mut sizes = vec![vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: (bindings - texel.len()).max(1) as u32 }];
            if !texel.is_empty() {
                sizes.push(vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_TEXEL_BUFFER, descriptor_count: texel.len() as u32 });
            }
            let desc_pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None).map_err(vkr("descriptor pool"))?;
            let set = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&sets)).map_err(vkr("descriptor set"))?[0];
            Ok(Kernel { device: d.clone(), module, set_layout, layout, pipeline, desc_pool, set, bindings, texel: texel.to_vec(), spec: spec.to_vec() })
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
            let mut views = Vec::new();
            for &i in &self.texel {
                let (b, o) = at[i];
                let range = ((b.size - o) / 8 * 8) as u64;
                if range / 8 > gpu.info.max_texel_elements as u64 {
                    return Err(format!("{range} bytes are more than a texel buffer view spans ({} texels)", gpu.info.max_texel_elements));
                }
                let info = vk::BufferViewCreateInfo::default().buffer(b.buf).format(vk::Format::R16G16B16A16_SFLOAT).offset(o as u64).range(range);
                views.push([d.create_buffer_view(&info, None).map_err(vkr("create buffer view"))?]);
            }
            gpu.views.lock().unwrap().extend(views.iter().map(|v| v[0]));
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| match self.texel.iter().position(|&t| t == i) {
                    Some(v) => vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::UNIFORM_TEXEL_BUFFER).texel_buffer_view(&views[v]),
                    None => vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info),
                })
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

/// A matrix in any of the formats of a llama.cpp Q4_K_M file (Q4_0, Q5_0,
/// Q8_0, Q4_K, Q6_K), in the layout `shaders/qx_gemv.comp` reads: every
/// 32-value piece as 16 bytes of nibble pairs (32 int8s for Q8_0), then
/// the Q5_0 fifth bits or Q6_K high bits, and the scales in `sc`.
pub struct MatQ {
    /// ggml's type id (the shader's FMT).
    pub ty: u32,
    pub qs: Buffer,
    pub sc: Buffer,
    pub rows: usize,
    /// 32-value pieces per row (columns / 32).
    pub nb: usize,
}

/// 32 values 0..15 as 16 bytes, byte k = value k | value k + 16 << 4.
fn nibble_pairs(v: &[u8]) -> [u8; 16] {
    std::array::from_fn(|k| v[k] | v[k + 16] << 4)
}

impl MatQ {
    /// Uploads `w` (`rows` rows, row-major blocks).
    pub fn new(gpu: &Gpu, w: &Weight, rows: usize) -> Result<MatQ> {
        let (ty, pieces) = match w {
            Weight::Q4_0(b) => (2, b.len()),
            Weight::Q5_0(b) => (6, b.len()),
            Weight::Q8_0(b) => (8, b.len()),
            Weight::Q4_K(b) => (12, 8 * b.len()),
            Weight::Q6_K(b) => (14, 8 * b.len()),
            _ => return Err(format!("{} matrices are not supported on the GPU", w.format_name())),
        };
        assert!(rows > 0 && pieces.is_multiple_of(rows));
        let qs_bytes = match ty {
            6 => pieces * 20,
            8 => pieces * 32,
            14 => pieces * 24,
            _ => pieces * 16,
        };
        let sc_words = if ty >= 12 { 5 * pieces / 8 } else { pieces.div_ceil(2) };
        let (mut qs, mut sc) = (gpu.buffer(qs_bytes)?, gpu.buffer(sc_words * 4)?);
        let (q, s) = (qs.as_mut_slice::<u8>(), sc.as_mut_slice::<u32>());
        // The f16 scales of the 32-value formats, two per u32 (a new
        // buffer is not zeroed).
        let mut halves = |ds: &mut dyn Iterator<Item = u16>| {
            s.fill(0);
            for (i, d) in ds.enumerate() {
                s[i / 2] |= (d as u32) << (16 * (i % 2));
            }
        };
        match w {
            Weight::Q4_0(b) => {
                b.iter().zip(q.as_chunks_mut::<16>().0).for_each(|(b, q)| *q = b.qs);
                halves(&mut b.iter().map(|b| b.d));
            }
            Weight::Q5_0(b) => {
                let (nib, high) = q.split_at_mut(16 * pieces);
                b.iter().zip(nib.as_chunks_mut::<16>().0).for_each(|(b, q)| *q = b.qs);
                b.iter().zip(high.as_chunks_mut::<4>().0).for_each(|(b, h)| *h = b.qh);
                halves(&mut b.iter().map(|b| b.d));
            }
            Weight::Q8_0(b) => {
                b.iter().zip(q.as_chunks_mut::<32>().0).for_each(|(b, q)| *q = b.qs.map(|v| v as u8));
                halves(&mut b.iter().map(|b| b.d));
            }
            Weight::Q4_K(b) => {
                for (k, (b, q)) in b.iter().zip(q.as_chunks_mut::<128>().0).enumerate() {
                    let v = quant::q4_k_values(b);
                    for (j, p) in q.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                        *p = nibble_pairs(&v[32 * j..]);
                    }
                    let (scales, mins) = quant::q4_k_scales_mins(&b.scales);
                    let word = |a: &[u8]| u32::from_le_bytes(a.try_into().unwrap());
                    s[5 * k..5 * k + 5].copy_from_slice(&[b.d as u32 | (b.dmin as u32) << 16, word(&scales[..4]), word(&scales[4..]), word(&mins[..4]), word(&mins[4..])]);
                }
            }
            Weight::Q6_K(b) => {
                let (nib, high) = q.split_at_mut(16 * pieces);
                let high = high.as_chunks_mut::<4>().0;
                for (k, b) in b.iter().enumerate() {
                    // 0..63: four low bits, two high.
                    let v = quant::q6_k_values(b).map(|v| (v + 32) as u8);
                    for j in 0..8 {
                        let (i, p) = (8 * k + j, &v[32 * j..32 * j + 32]);
                        let low: Vec<u8> = p.iter().map(|v| v & 15).collect();
                        nib[16 * i..16 * i + 16].copy_from_slice(&nibble_pairs(&low));
                        for half in 0..2 {
                            let h = (0..16).fold(0u32, |h, l| h | ((p[16 * half + l] >> 4) as u32) << (2 * l));
                            high[2 * i + half] = h.to_le_bytes();
                        }
                    }
                    let sb = b.scales.map(|v| v as u8);
                    s[5 * k] = b.d as u32;
                    for (n, c) in sb.as_chunks::<4>().0.iter().enumerate() {
                        s[5 * k + 1 + n] = u32::from_le_bytes(*c);
                    }
                }
            }
            _ => unreachable!(),
        }
        Ok(MatQ { ty, qs, sc, rows, nb: pieces / rows })
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
    /// `out[r] = W[r] · x` for a [`MatQ`] of any format, `reps` times;
    /// `kernel` from [`Gpu::qx_gemv_kernel`] for that format. Returns the
    /// GPU time of all reps in seconds.
    pub fn qx_gemv(&self, kernel: &Kernel, w: &MatQ, x: &Buffer, out: &Buffer, reps: u32) -> Result<f64> {
        assert!(x.size >= w.nb * 32 * 4 && out.size >= w.rows * 4);
        assert_eq!(kernel.spec[4], w.ty, "a kernel for another format");
        let push: Vec<u8> = [w.rows as u32, w.nb as u32].iter().flat_map(|v| v.to_le_bytes()).collect();
        self.run(kernel, &[&w.qs, &w.sc, x, out, x], &push, groups_for(w.rows, 16 * kernel.spec[0] as usize), reps)
    }

    /// The decode shader for matrices of ggml type `ty` (see [`MatQ`]),
    /// with the options of [`Gpu::q4_0_gemv_kernel_fused`].
    pub fn qx_gemv_kernel(&self, ty: u32, rows_per_32: u32, accum: bool, glu: bool, norm: bool) -> Result<Kernel> {
        assert!(!(glu && norm), "a GLU input is not normalized");
        assert!(rows_per_32 > 0 && rows_per_32.is_multiple_of(4));
        assert!([2, 6, 8, 12, 14].contains(&ty), "ggml type {ty} has no GPU kernel");
        if self.info.subgroup_size < 32 || !self.info.subgroup_ops.contains(vk::SubgroupFeatureFlags::SHUFFLE) {
            return Err(format!("qx_gemv needs subgroup shuffles and subgroups of at least 32 lanes ({} lanes, {:?})", self.info.subgroup_size, self.info.subgroup_ops));
        }
        Kernel::with_spec(self, crate::spirv!("qx_gemv"), 5, 20, &[rows_per_32 / 4, accum.into(), glu.into(), norm.into(), ty, 1])
    }

    /// The decode shader for up to `max_tokens` tokens per dispatch (4 rows
    /// per 32 lanes): each weight is read once for all of them. Pushes
    /// rows, nb, eps (unused), tokens and the output row length (0: rows);
    /// with `glu` each token's x is gate then up. No norm.
    pub fn qx_gemv_n_kernel(&self, ty: u32, accum: bool, glu: bool, max_tokens: u32) -> Result<Kernel> {
        assert!((2..=8).contains(&max_tokens), "2 to 8 tokens");
        assert!([2, 6, 8, 12, 14].contains(&ty), "ggml type {ty} has no GPU kernel");
        if self.info.subgroup_size < 32 || !self.info.subgroup_ops.contains(vk::SubgroupFeatureFlags::SHUFFLE) {
            return Err(format!("qx_gemv needs subgroup shuffles and subgroups of at least 32 lanes ({} lanes, {:?})", self.info.subgroup_size, self.info.subgroup_ops));
        }
        Kernel::with_spec(self, crate::spirv!("qx_gemv"), 5, 20, &[1, accum.into(), glu.into(), 0, ty, max_tokens])
    }

    /// `out[t][r] = W[r] · x[t]` for `tokens` (at most the kernel's)
    /// activations of a [`MatQ`], with [`Gpu::qx_gemv_n_kernel`], `reps`
    /// times. Returns the GPU time of all reps in seconds.
    pub fn qx_gemv_n(&self, kernel: &Kernel, w: &MatQ, x: &Buffer, tokens: usize, out: &Buffer, reps: u32) -> Result<f64> {
        assert!(tokens >= 1 && tokens <= kernel.spec[5] as usize && x.size >= tokens * w.nb * 32 * 4 && out.size >= tokens * w.rows * 4);
        let push: Vec<u8> = [w.rows as u32, w.nb as u32, 0, tokens as u32, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
        self.run(kernel, &[&w.qs, &w.sc, x, out, x], &push, groups_for(w.rows, 16), reps)
    }

    /// `quant_x.comp`: each token's x (with `glu`, silu(gate) · up) to 8
    /// bits per 32 values, for [`Gpu::qx_gemv_q8_kernel`]. Pushes nb and
    /// tokens.
    pub fn quant_x_kernel(&self, glu: bool) -> Result<Kernel> {
        Kernel::with_spec(self, crate::spirv!("quant_x"), 2, 8, &[glu.into()])
    }

    /// [`Gpu::quant_x_kernel`] storing the codes in the order of
    /// [`Gpu::q4_0_gemm_q8_kernel`] (`PAIRS` in `quant_x.comp`).
    pub fn quant_xp_kernel(&self, glu: bool) -> Result<Kernel> {
        Kernel::with_spec(self, crate::spirv!("quant_x"), 2, 8, &[glu.into(), 1])
    }

    /// Quantizes `tokens` rows of `nb` 32-value pieces of `x` into `xq`
    /// (48 bytes a piece), `reps` times. Returns the GPU time.
    pub fn quant_x(&self, kernel: &Kernel, x: &Buffer, nb: usize, tokens: usize, xq: &Buffer, reps: u32) -> Result<f64> {
        let glu = kernel.spec[0] != 0;
        assert!(x.size >= tokens * nb * 32 * 4 * if glu { 2 } else { 1 } && xq.size >= tokens * nb * 48);
        let push: Vec<u8> = [nb as u32, tokens as u32].iter().flat_map(|v| v.to_le_bytes()).collect();
        self.run(kernel, &[x, xq], &push, groups_for(tokens * nb, 64), reps)
    }

    /// `qx_gemv_q8.comp`: the few-token product of [`Gpu::qx_gemv_n_kernel`]
    /// with x quantized by [`Gpu::quant_x`] and int8 dot products, for up
    /// to `max_tokens` tokens (1 to 8). Needs [`Info::int_dot`]. Pushes
    /// rows, nb, tokens and the output row length (0: rows).
    pub fn qx_gemv_q8_kernel(&self, ty: u32, accum: bool, max_tokens: u32) -> Result<Kernel> {
        assert!((1..=8).contains(&max_tokens), "1 to 8 tokens");
        assert!([2, 6, 8, 12, 14].contains(&ty), "ggml type {ty} has no GPU kernel");
        if !self.info.int_dot {
            return Err("qx_gemv_q8 needs int8 dot products (VK_KHR_shader_integer_dot_product)".into());
        }
        if self.info.subgroup_size < 32 || !self.info.subgroup_ops.contains(vk::SubgroupFeatureFlags::SHUFFLE) {
            return Err(format!("qx_gemv_q8 needs subgroup shuffles and subgroups of at least 32 lanes ({} lanes, {:?})", self.info.subgroup_size, self.info.subgroup_ops));
        }
        Kernel::with_spec(self, crate::spirv!("qx_gemv_q8"), 4, 16, &[accum.into(), ty, max_tokens])
    }

    /// `out[t][r] = W[r] · x[t]` for `tokens` quantized activations `xq`
    /// of a [`MatQ`], `reps` times. Returns the GPU time.
    pub fn qx_gemv_q8(&self, kernel: &Kernel, w: &MatQ, xq: &Buffer, tokens: usize, out: &Buffer, reps: u32) -> Result<f64> {
        assert!(tokens >= 1 && tokens <= kernel.spec[2] as usize && xq.size >= tokens * w.nb * 48 && out.size >= tokens * w.rows * 4);
        assert_eq!(kernel.spec[1], w.ty, "a kernel for another format");
        let push: Vec<u8> = [w.rows as u32, w.nb as u32, tokens as u32, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
        self.run(kernel, &[&w.qs, &w.sc, xq, out], &push, groups_for(w.rows, 16), reps)
    }
}

impl Gpu {
    /// `out[t][r] = W[r] · x[t]` for a [`MatQ`] of any format and `tokens`
    /// activations, `reps` times; `kernel` from [`Gpu::qx_gemm_kernel`].
    /// Each token's row of `out` is `ostride` floats long (0: the matrix's
    /// rows). Returns the GPU time of all reps.
    pub fn qx_gemm(&self, kernel: &Kernel, w: &MatQ, x: &Buffer, tokens: usize, out: &Buffer, ostride: usize, reps: u32) -> Result<f64> {
        assert!(x.size >= tokens * w.nb * 32 * 4 && out.size >= tokens * ostride.max(w.rows) * 4);
        assert_eq!(kernel.spec[2], w.ty, "a kernel for another format");
        let push: Vec<u8> = [w.rows as u32, w.nb as u32, tokens as u32, ostride as u32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let [gx, gy] = [w.rows.div_ceil(64) as u32, tokens.div_ceil(64) as u32];
        assert!(gx <= 65535 && gy <= 65535, "too many rows or tokens for one dispatch");
        self.run(kernel, &[&w.qs, &w.sc, x, out], &push, [gx, gy, 1], reps)
    }

    /// The prefill shader for matrices of ggml type `ty` (see [`MatQ`]),
    /// with `accum` and `glu` as in [`Gpu::q4_0_gemm_kernel_fused`].
    pub fn qx_gemm_kernel(&self, ty: u32, accum: bool, glu: bool) -> Result<Kernel> {
        assert!([2, 6, 8, 12, 14].contains(&ty), "ggml type {ty} has no GPU kernel");
        Kernel::with_spec(self, crate::spirv!("qx_gemm"), 4, 16, &[accum.into(), glu.into(), ty])
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

/// A Q4_0 matrix in the layout `shaders/q4_0_gemm_h.comp` reads: rows
/// rounded up to a multiple of 4 (`rows4`, zero weights past `rows`);
/// word h of block b of row r (8 inputs, one nibble each, input 8h + j in
/// bits 4j..) at u32 `(4b + h) · rows4 + r`, the f16 scales at
/// `b · rows4 + r`. Neighbouring rows are neighbouring words, so the
/// threads of a workgroup read memory in one sweep.
pub struct MatQ4_0T {
    pub q: Buffer,
    pub d: Buffer,
    pub rows: usize,
    pub nb: usize,
}

impl MatQ4_0T {
    /// Uploads `blocks` (row-major, `rows` rows).
    pub fn new(gpu: &Gpu, blocks: &[BlockQ4_0], rows: usize) -> Result<MatQ4_0T> {
        assert!(rows > 0 && blocks.len().is_multiple_of(rows));
        let nb = blocks.len() / rows;
        let rows4 = rows.div_ceil(4) * 4;
        let mut q = gpu.buffer(nb * 4 * rows4 * 4)?;
        let mut d = gpu.buffer(nb * rows4 * 2)?;
        let (qv, dv) = (q.as_mut_slice::<u32>(), d.as_mut_slice::<u16>());
        qv.fill(0);
        dv.fill(0);
        for r in 0..rows {
            for b in 0..nb {
                let blk = &blocks[r * nb + b];
                dv[b * rows4 + r] = blk.d;
                // Input i of the block: byte i's low nibble for i < 16,
                // byte i - 16's high nibble after.
                let nib = |i: usize| if i < 16 { blk.qs[i] & 15 } else { blk.qs[i - 16] >> 4 } as u32;
                for h in 0..4 {
                    qv[(4 * b + h) * rows4 + r] = (0..8).map(|j| nib(8 * h + j) << (4 * j)).sum();
                }
            }
        }
        Ok(MatQ4_0T { q, d, rows, nb })
    }
}

impl Gpu {
    /// The f16 prefill shader (`shaders/q4_0_gemm_h.comp`); with `accum`
    /// it adds to `out`. Needs [`Info::fp16`].
    pub fn q4_0_gemm_h_kernel(&self, accum: bool) -> Result<Kernel> {
        if !self.info.fp16 {
            return Err("q4_0_gemm_h needs fp16 arithmetic and 16-bit storage buffers".into());
        }
        Kernel::with_spec(self, crate::spirv!("q4_0_gemm_h"), 4, 16, &[accum.into()])
    }

    /// [`Gpu::q4_0_gemm_h_kernel`] with the activations (binding 2) as a
    /// texel buffer (`shaders/q4_0_gemm_ht.comp`): the texture path, as
    /// llama.cpp's Adreno kernel reads them.
    pub fn q4_0_gemm_ht_kernel(&self, accum: bool) -> Result<Kernel> {
        if !self.info.fp16 {
            return Err("q4_0_gemm_ht needs fp16 arithmetic and 16-bit storage buffers".into());
        }
        Kernel::with_texel(self, crate::spirv!("q4_0_gemm_ht"), 4, 16, &[accum.into()], &[2])
    }

    /// `q4_0_gemm_q8.comp`: the prefill product of [`Gpu::q4_0_gemm_h_kernel`]
    /// with the activations quantized to 8 bits by [`Gpu::quant_xp_kernel`]
    /// and int8 dot products. Same buffers, same pushes. Needs
    /// [`Info::int_dot`].
    pub fn q4_0_gemm_q8_kernel(&self, accum: bool) -> Result<Kernel> {
        if !self.info.int_dot {
            return Err("q4_0_gemm_q8 needs int8 dot products (VK_KHR_shader_integer_dot_product)".into());
        }
        Kernel::with_spec(self, crate::spirv!("q4_0_gemm_q8"), 4, 16, &[accum.into()])
    }

    /// Records the quantization of `tokens` rows of `cols` activations
    /// (`quant_x.comp`), the int8 counterpart of [`Gpu::record_f16t`].
    pub fn record_quant_x(r: &mut Recorder, t: &Bound, cols: usize, tokens: usize) {
        r.dispatch(t, &push_u32(&[(cols / 32) as u32, tokens as u32]), groups_for(tokens * (cols / 32), 64));
    }

    /// Its activations' transpose to f16 (`shaders/to_f16t.comp`); with
    /// `glu`, from gate and up rows as silu(gate) · up.
    pub fn to_f16t_kernel(&self, glu: bool) -> Result<Kernel> {
        if !self.info.fp16 {
            return Err("to_f16t needs fp16 arithmetic and 16-bit storage buffers".into());
        }
        Kernel::with_spec(self, crate::spirv!("to_f16t"), 2, 12, &[glu.into()])
    }

    /// Tokens per row of the transposed activations: a multiple of 8.
    pub fn f16t_pad(tokens: usize) -> usize {
        tokens.div_ceil(8) * 8
    }

    /// Records `out[t][r] = W[r] · x[t]` for `tokens` rows of `x`
    /// (f32, `w.nb` · 32 each): the transpose into `xt` (at least
    /// `w.nb · 32 · f16t_pad(tokens) · 2` bytes), a barrier, the product.
    pub fn record_q4_0_gemm_h(r: &mut Recorder, t: &Bound, mm: &Bound, w: &MatQ4_0T, tokens: usize) {
        Gpu::record_f16t(r, t, w.nb * 32, tokens);
        r.barrier();
        Gpu::record_q4_0_gemm_h_mm(r, mm, w, tokens);
    }

    /// Records the transpose of `tokens` rows of `cols` activations to f16
    /// (`to_f16t.comp`).
    pub fn record_f16t(r: &mut Recorder, t: &Bound, cols: usize, tokens: usize) {
        let npad = Gpu::f16t_pad(tokens);
        r.dispatch(t, &push_u32(&[cols as u32, tokens as u32, npad as u32]), [cols as u32, npad.div_ceil(64) as u32, 1]);
    }

    /// Records the product alone, from activations already transposed.
    pub fn record_q4_0_gemm_h_mm(r: &mut Recorder, mm: &Bound, w: &MatQ4_0T, tokens: usize) {
        let npad = Gpu::f16t_pad(tokens);
        r.dispatch(mm, &push_u32(&[w.rows as u32, w.nb as u32, tokens as u32, npad as u32]), [w.rows.div_ceil(256) as u32, (npad / 8) as u32, 1]);
    }

    /// [`Gpu::record_q4_0_gemm_h`] run once on its own, plus the product
    /// `reps` more times; returns the GPU time of one product (the
    /// median of those reps' stamps, or of the one if `reps` is 0).
    pub fn q4_0_gemm_h(&self, t: &Kernel, mm: &Kernel, w: &MatQ4_0T, x: &Buffer, xt: &Buffer, tokens: usize, out: &Buffer, reps: u32) -> Result<f64> {
        let npad = Gpu::f16t_pad(tokens);
        assert!(x.size >= tokens * w.nb * 32 * 4 && xt.size >= w.nb * 32 * npad * 2 && out.size >= tokens * w.rows * 4);
        let (tb, mb) = (t.bind(self, &[x, xt])?, mm.bind(self, &[&w.q, &w.d, xt, out])?);
        let cmds = self.record_timed(|r| {
            Gpu::record_q4_0_gemm_h(r, &tb, &mb, w, tokens);
            r.barrier();
            r.stamp();
            let push: Vec<u8> = [w.rows as u32, w.nb as u32, tokens as u32, npad as u32].iter().flat_map(|x| x.to_le_bytes()).collect();
            for _ in 0..reps {
                r.dispatch(&mb, &push, [w.rows.div_ceil(256) as u32, (npad / 8) as u32, 1]);
                r.barrier();
                r.stamp();
            }
        })?;
        self.submit(&cmds, false)?;
        let st = self.stamps(&cmds)?;
        let mut each: Vec<f64> = st.windows(2).skip(1).map(|w| w[1] - w[0]).collect();
        if each.is_empty() {
            return Ok(st[1]);
        }
        each.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(each[each.len() / 2])
    }
}

fn push_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A matrix in any format of a llama.cpp Q4_K_M file in the layout
/// `shaders/qx_gemm_h.comp` reads (see there): per 32-value piece of a
/// row, [`MatQT::words`] u32 words and f16 scales, row-interleaved as in
/// [`MatQ4_0T`]. The K-quants' scales are multiplied out per piece (a
/// Q4_K piece's d · scale and dmin · min, a Q6_K piece's two d · scale),
/// so its scales take 1 bit a weight against the file's 0.19.
pub struct MatQT {
    /// ggml's type id (the shader's FMT).
    pub ty: u32,
    pub q: Buffer,
    pub s: Buffer,
    pub rows: usize,
    /// 32-value pieces per row.
    pub nb: usize,
}

impl MatQT {
    /// Words and f16 scales per piece of format `ty`.
    pub fn words(ty: u32) -> (usize, usize) {
        match ty {
            2 => (4, 1),
            6 => (5, 1),
            8 => (8, 1),
            12 => (4, 2),
            14 => (6, 2),
            _ => panic!("ggml type {ty} has no GPU kernel"),
        }
    }

    /// Uploads `w` (`rows` rows, row-major blocks).
    pub fn new(gpu: &Gpu, w: &Weight, rows: usize) -> Result<MatQT> {
        MatQT::build(gpu, w, rows, false)
    }

    /// [`MatQT::new`] for `qx_gemm_q8.comp`: the high bits of Q5_0 and Q6_K
    /// and the bytes of Q8_0 in the order of the pairs of 8 values
    /// `quant_x.comp` (PAIRS) stores, as that shader's header says.
    pub fn new_paired(gpu: &Gpu, w: &Weight, rows: usize) -> Result<MatQT> {
        MatQT::build(gpu, w, rows, true)
    }

    fn build(gpu: &Gpu, w: &Weight, rows: usize, paired: bool) -> Result<MatQT> {
        let (ty, pieces) = match w {
            Weight::Q4_0(b) => (2, b.len()),
            Weight::Q5_0(b) => (6, b.len()),
            Weight::Q8_0(b) => (8, b.len()),
            Weight::Q4_K(b) => (12, 8 * b.len()),
            Weight::Q6_K(b) => (14, 8 * b.len()),
            _ => return Err(format!("{} matrices are not supported on the GPU", w.format_name())),
        };
        assert!(rows > 0 && pieces.is_multiple_of(rows));
        let nb = pieces / rows;
        let rows4 = rows.div_ceil(4) * 4;
        let (nw, ns) = MatQT::words(ty);
        let (mut q, mut s) = (gpu.buffer(nb * nw * rows4 * 4)?, gpu.buffer(nb * ns * rows4 * 2)?);
        let (qv, sv) = (q.as_mut_slice::<u32>(), s.as_mut_slice::<u16>());
        qv.fill(0);
        sv.fill(0);
        // Piece i (= row · nb + piece of the row): its words and scales.
        let mut put = |i: usize, words: &[u32], scales: &[u16]| {
            let (r, b) = (i / nb, i % nb);
            words.iter().enumerate().for_each(|(h, &v)| qv[(nw * b + h) * rows4 + r] = v);
            scales.iter().enumerate().for_each(|(k, &v)| sv[(ns * b + k) * rows4 + r] = v);
        };
        // The low 4 bits of 32 values as 4 words, value 8h + j in bits 4j
        // of word h.
        let nib = |v: &dyn Fn(usize) -> u8| -> [u32; 4] { std::array::from_fn(|h| (0..8).map(|j| ((v(8 * h + j) & 15) as u32) << (4 * j)).sum()) };
        // Value k of a Q4_0 / Q5_0 block's nibbles: byte k's low nibble,
        // then byte k − 16's high one.
        let lo_hi = |qs: &[u8; 16], k: usize| if k < 16 { qs[k] & 15 } else { qs[k - 16] >> 4 };
        let half = quant::f32_to_f16;
        match w {
            Weight::Q4_0(b) => b.iter().enumerate().for_each(|(i, b)| put(i, &nib(&|k| lo_hi(&b.qs, k)), &[b.d])),
            Weight::Q5_0(b) => b.iter().enumerate().for_each(|(i, b)| {
                let n = nib(&|k| lo_hi(&b.qs, k));
                let qh = u32::from_le_bytes(b.qh);
                // Paired: nibble 2h the fifth bits of values 8h + 0, 2, 4, 6
                // (bit i: value 8h + 2i), nibble 2h + 1 of 8h + 1, 3, 5, 7.
                let qh = if paired { (0..32).fold(0u32, |a, v| a | (qh >> v & 1) << (8 * (v / 8) + 4 * (v % 2) + (v % 8) / 2)) } else { qh };
                put(i, &[n[0], n[1], n[2], n[3], qh], &[b.d]);
            }),
            Weight::Q8_0(b) => b.iter().enumerate().for_each(|(i, b)| {
                // Paired: word 2g the values 8g + 0, 2, 4, 6, word 2g + 1 the odd ones.
                let at = |h: usize, j: usize| if paired { 8 * (h / 2) + 2 * j + h % 2 } else { 4 * h + j };
                let words: [u32; 8] = std::array::from_fn(|h| u32::from_le_bytes(std::array::from_fn(|j| b.qs[at(h, j)] as u8)));
                put(i, &words, &[b.d]);
            }),
            Weight::Q4_K(b) => b.iter().enumerate().for_each(|(k, b)| {
                let (v, d, dmin) = (quant::q4_k_values(b), quant::f16_to_f32(b.d), quant::f16_to_f32(b.dmin));
                for j in 0..8 {
                    let (sc, m) = quant::q4_k_scale_min(&b.scales, j);
                    put(8 * k + j, &nib(&|e| v[32 * j + e]), &[half(d * sc as f32), half(dmin * m as f32)]);
                }
            }),
            Weight::Q6_K(b) => b.iter().enumerate().for_each(|(k, b)| {
                // 0..63: four low bits, two high.
                let (v, d) = (quant::q6_k_values(b).map(|v| (v + 32) as u8), quant::f16_to_f32(b.d));
                for j in 0..8 {
                    let p = &v[32 * j..32 * j + 32];
                    let n = nib(&|e| p[e]);
                    let high = |g: usize| (0..16).fold(0u32, |h, l| h | ((p[16 * g + l] >> 4) as u32) << (2 * l));
                    // Paired: byte 2h the high bits of values 8h + 0, 2, 4, 6
                    // (value 8h + 2i in bits 2i), byte 2h + 1 of the odd ones.
                    let paired_high = |w: usize| (0..2).flat_map(|hh| [0, 1].map(|o| (hh, o))).fold(0u32, |a, (hh, o)| {
                        let h = 2 * w + hh;
                        let byte = (0..4).fold(0u32, |x, i| x | ((p[8 * h + 2 * i + o] >> 4) as u32) << (2 * i));
                        a | byte << (8 * (2 * hh + o))
                    });
                    let (h0, h1) = if paired { (paired_high(0), paired_high(1)) } else { (high(0), high(1)) };
                    put(8 * k + j, &[n[0], n[1], n[2], n[3], h0, h1], &[half(d * b.scales[2 * j] as f32), half(d * b.scales[2 * j + 1] as f32)]);
                }
            }),
            _ => unreachable!(),
        }
        Ok(MatQT { ty, q, s, rows, nb })
    }
}

impl Gpu {
    /// `qx_gemm_q8.comp`: the prefill product for matrices of ggml type `ty`
    /// (a [`MatQT::new_paired`]) with the activations quantized to 8 bits by
    /// [`Gpu::quant_xp_kernel`] and int8 dot products; the buffers and
    /// pushes of [`Gpu::qx_gemm_h_kernel`]. Needs [`Info::int_dot`].
    pub fn qx_gemm_q8_kernel(&self, ty: u32, accum: bool) -> Result<Kernel> {
        if !self.info.int_dot {
            return Err("qx_gemm_q8 needs int8 dot products (VK_KHR_shader_integer_dot_product)".into());
        }
        MatQT::words(ty);
        Kernel::with_spec(self, crate::spirv!("qx_gemm_q8"), 4, 20, &[accum.into(), ty])
    }

    /// The f16 prefill shader for matrices of ggml type `ty`
    /// (`shaders/qx_gemm_h.comp`, a [`MatQT`]); with `accum` it adds to
    /// `out`. Its input is transposed by [`Gpu::to_f16t_kernel`]. Needs
    /// [`Info::fp16`].
    pub fn qx_gemm_h_kernel(&self, ty: u32, accum: bool) -> Result<Kernel> {
        if !self.info.fp16 {
            return Err("qx_gemm_h needs fp16 arithmetic and 16-bit storage buffers".into());
        }
        MatQT::words(ty);
        Kernel::with_spec(self, crate::spirv!("qx_gemm_h"), 4, 20, &[accum.into(), ty])
    }

    /// Records the product from activations already transposed by
    /// [`Gpu::record_f16t`]; each token's row of `out` is `ostride` floats
    /// long (0: the matrix's rows).
    pub fn record_qx_gemm_h(r: &mut Recorder, mm: &Bound, w: &MatQT, tokens: usize, ostride: usize) {
        let npad = Gpu::f16t_pad(tokens);
        r.dispatch(mm, &push_u32(&[w.rows as u32, w.nb as u32, tokens as u32, npad as u32, ostride as u32]), [w.rows.div_ceil(256) as u32, (npad / 8) as u32, 1]);
    }

    /// The transpose and the product once, then the product `reps` more
    /// times; returns the GPU time of one product as
    /// [`Gpu::q4_0_gemm_h`] does.
    pub fn qx_gemm_h(&self, t: &Kernel, mm: &Kernel, w: &MatQT, x: &Buffer, xt: &Buffer, tokens: usize, out: &Buffer, reps: u32) -> Result<f64> {
        let npad = Gpu::f16t_pad(tokens);
        assert!(x.size >= tokens * w.nb * 32 * 4 && xt.size >= w.nb * 32 * npad * 2 && out.size >= tokens * w.rows * 4);
        assert_eq!(mm.spec[1], w.ty, "a kernel for another format");
        let (tb, mb) = (t.bind(self, &[x, xt])?, mm.bind(self, &[&w.q, &w.s, xt, out])?);
        let cmds = self.record_timed(|r| {
            Gpu::record_f16t(r, &tb, w.nb * 32, tokens);
            r.barrier();
            Gpu::record_qx_gemm_h(r, &mb, w, tokens, 0);
            r.barrier();
            r.stamp();
            for _ in 0..reps {
                Gpu::record_qx_gemm_h(r, &mb, w, tokens, 0);
                r.barrier();
                r.stamp();
            }
        })?;
        self.submit(&cmds, false)?;
        let st = self.stamps(&cmds)?;
        let mut each: Vec<f64> = st.windows(2).skip(1).map(|w| w[1] - w[0]).collect();
        if each.is_empty() {
            return Ok(st[1]);
        }
        each.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(each[each.len() / 2])
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

    /// The multi-format decode shader on llama.cpp's bytes against
    /// X @ dequant(W)ᵀ in float64: Q4_0, Q5_0, Q8_0 from llama.cpp's
    /// quantizers (`quant/`), Q4_K and Q6_K random blocks dequantized by
    /// the gguf package (`quant_k/`), every token of x in turn.
    #[test]
    fn qx_gemv_matches_reference() {
        let (Some(root), Some(root_k)) = (crate::testutil::golden("quant"), crate::testutil::golden("quant_k")) else { return };
        let Ok(gpu) = Gpu::new() else { return };
        let mut cases = 0;
        let dirs = |r: &std::path::Path| std::fs::read_dir(r).unwrap().map(|e| e.unwrap().path()).collect::<Vec<_>>();
        let check = |dir: &std::path::Path, name: &str, w: Weight, rows: usize| {
            let (xs, xv) = crate::testutil::read_npy(&dir.join("x.npy"));
            let (_, want) = crate::testutil::read_npy(&dir.join(format!("out_{name}_a.npy")));
            let cols = *xs.last().unwrap();
            let m = MatQ::new(&gpu, &w, rows).unwrap();
            let mut x = gpu.buffer(cols * 4).unwrap();
            let mut out = gpu.buffer(rows * 4).unwrap();
            for r in [4, 8] {
                let k = gpu.qx_gemv_kernel(m.ty, r, false, false, false).unwrap();
                // quant/ has x as one vector and out as [rows]; quant_k/ has
                // tokens × cols and tokens × rows.
                for (t, x_t) in xv.chunks_exact(cols).enumerate() {
                    x.as_mut_slice::<f32>().copy_from_slice(x_t);
                    out.as_mut_slice::<f32>().fill(f32::NAN);
                    gpu.qx_gemv(&k, &m, &x, &out, 1).unwrap();
                    let (_, rel) = crate::testutil::max_errors(&out.as_slice::<f32>()[..rows], &want[t * rows..(t + 1) * rows]);
                    assert!(rel < 1e-6 * (cols as f32).sqrt().max(4.0), "{} {name} R={r} token {t}: rel {rel:.2e}", dir.display());
                }
            }
        };
        for dir in dirs(&root) {
            let (ws, _) = crate::testutil::read_npy(&dir.join("w.npy"));
            let bytes = |n: &str| std::fs::read(dir.join(format!("w_{n}.bin"))).unwrap();
            check(&dir, "q4_0", Weight::Q4_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q4_0")))), ws[0]);
            check(&dir, "q5_0", Weight::Q5_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q5_0")))), ws[0]);
            check(&dir, "q8_0", Weight::Q8_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q8_0")))), ws[0]);
            cases += 1;
        }
        for dir in dirs(&root_k) {
            let bytes = |n: &str| std::fs::read(dir.join(format!("w_{n}.bin"))).unwrap();
            let (ds, _) = crate::testutil::read_npy(&dir.join("deq_q4_k.npy"));
            check(&dir, "q4_k", Weight::Q4_K(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q4_k")))), ds[0]);
            check(&dir, "q6_k", Weight::Q6_K(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q6_k")))), ds[0]);
            cases += 1;
        }
        assert!(cases >= 21, "only {cases} golden cases");
    }

    /// The decode shader for several tokens at once (`qx_gemv_n_kernel`,
    /// up to 8 per dispatch) on the batch goldens, as the prefill shaders:
    /// each run of up to 8 tokens against X @ dequant(W)ᵀ in float64.
    /// (`MatQ4_0` holds the same bytes as `MatQ`'s Q4_0, so the model binds
    /// either.)
    #[test]
    fn qx_gemv_n_matches_reference() {
        let (Some(root), Some(root_k)) = (crate::testutil::golden("quant_batch"), crate::testutil::golden("quant_k")) else { return };
        let Ok(gpu) = Gpu::new() else { return };
        let mut cases = 0;
        let dirs = |r: &std::path::Path| std::fs::read_dir(r).unwrap().map(|e| e.unwrap().path()).collect::<Vec<_>>();
        let check = |dir: &std::path::Path, name: &str, w: Weight, rows: usize| {
            let (xs, xv) = crate::testutil::read_npy(&dir.join("x.npy"));
            let (_, want) = crate::testutil::read_npy(&dir.join(format!("out_{name}_a.npy")));
            let (tokens, cols) = (xs[0], xs[1]);
            let m = MatQ::new(&gpu, &w, rows).unwrap();
            for nt in [2, 3, 8] {
                let k = gpu.qx_gemv_n_kernel(m.ty, false, false, nt).unwrap();
                for t0 in (0..tokens).step_by(nt as usize) {
                    let n = (tokens - t0).min(nt as usize);
                    let mut x = gpu.buffer(n * cols * 4).unwrap();
                    x.as_mut_slice::<f32>().copy_from_slice(&xv[t0 * cols..(t0 + n) * cols]);
                    let out = gpu.buffer(n * rows * 4).unwrap();
                    gpu.qx_gemv_n(&k, &m, &x, n, &out, 1).unwrap();
                    let (_, rel) = crate::testutil::max_errors(&out.as_slice::<f32>()[..n * rows], &want[t0 * rows..(t0 + n) * rows]);
                    assert!(rel < 1e-6 * (cols as f32).sqrt().max(4.0), "{} {name}, {nt} a dispatch, tokens {t0}..{}: rel {rel:.2e}", dir.display(), t0 + n);
                }
            }
        };
        for dir in dirs(&root) {
            let (ws, _) = crate::testutil::read_npy(&dir.join("w.npy"));
            let bytes = |n: &str| std::fs::read(dir.join(format!("w_{n}.bin"))).unwrap();
            check(&dir, "q4_0", Weight::Q4_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q4_0")))), ws[0]);
            check(&dir, "q5_0", Weight::Q5_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q5_0")))), ws[0]);
            check(&dir, "q8_0", Weight::Q8_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q8_0")))), ws[0]);
            cases += 1;
        }
        for dir in dirs(&root_k) {
            let bytes = |n: &str| std::fs::read(dir.join(format!("w_{n}.bin"))).unwrap();
            let (ds, _) = crate::testutil::read_npy(&dir.join("deq_q4_k.npy"));
            check(&dir, "q4_k", Weight::Q4_K(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q4_k")))), ds[0]);
            check(&dir, "q6_k", Weight::Q6_K(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q6_k")))), ds[0]);
            cases += 1;
        }
        assert!(cases >= 20, "only {cases} golden cases");
    }

    /// The int8 products on any target (the phone included; the test below
    /// needs Python): x's Q8_0 blocks as gguf-py wrote them (`x_q8_0.bin`)
    /// are laid out as `quant_x.comp` would, plain for `qx_gemv_q8.comp`
    /// (all five formats, 1, 3 and 8 tokens a dispatch) and in nibble-pair
    /// order for `q4_0_gemm_q8.comp` (plain and accumulating), and the
    /// products are compared with the goldens' `out_*_b.npy`
    /// (dequant(X_q8_0) @ dequant(W)ᵀ); `quant_x.comp`'s own codes are
    /// compared with gguf-py's.
    #[test]
    fn int8_products_match_goldens() {
        let (Some(root), Some(root_k)) = (crate::testutil::golden("quant_batch"), crate::testutil::golden("quant_k")) else { return };
        let Ok(gpu) = Gpu::new() else { return };
        if !gpu.info.int_dot {
            return;
        }
        // x_q8_0.bin as 48-byte pieces: the 32 codes (value order, or for
        // `paired` the even/odd words of each group of 8), d as f32, and
        // the sums of the codes of values 0–15 and 16–31.
        let pieces = |dir: &std::path::Path, tokens: usize, nb: usize, paired: bool| -> Buffer {
            let raw = std::fs::read(dir.join("x_q8_0.bin")).expect("x_q8_0.bin: regenerate with tools/gen_quant.py");
            assert_eq!(raw.len(), tokens * nb * 34);
            let mut buf = gpu.buffer(Gpu::f16t_pad(tokens) * nb * 48).unwrap();
            let out = buf.as_mut_slice::<u8>();
            out.fill(0);
            for p in 0..tokens * nb {
                let blk = &raw[34 * p..34 * p + 34];
                let d = crate::format::safetensors::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let codes = &blk[2..];
                let o = &mut out[48 * p..48 * p + 48];
                for g in 0..4 {
                    for j in 0..4 {
                        let (even, odd) = (codes[8 * g + 2 * j], codes[8 * g + 2 * j + 1]);
                        if paired {
                            o[8 * g + j] = even;
                            o[8 * g + 4 + j] = odd;
                        } else {
                            o[8 * g + 2 * j] = even;
                            o[8 * g + 2 * j + 1] = odd;
                        }
                    }
                }
                o[32..36].copy_from_slice(&d.to_le_bytes());
                let sum = |r: std::ops::Range<usize>| codes[r].iter().map(|&c| c as i8 as i32).sum::<i32>();
                o[36..40].copy_from_slice(&sum(0..16).to_le_bytes());
                o[40..44].copy_from_slice(&sum(16..32).to_le_bytes());
            }
            buf
        };
        let mut cases = 0;
        let check = |dir: &std::path::Path, name: &str, w: Weight, rows: usize| {
            let (xs, xv) = crate::testutil::read_npy(&dir.join("x.npy"));
            let (_, want) = crate::testutil::read_npy(&dir.join(format!("out_{name}_b.npy")));
            let (tokens, cols) = (xs[0], xs[1]);
            let nb = cols / 32;
            let tol = 1e-6 * (cols as f32).sqrt().max(4.0);
            // quant_x.comp's codes against gguf-py's, plain and paired.
            let mut xb = gpu.buffer(xv.len() * 4).unwrap();
            xb.as_mut_slice::<f32>().copy_from_slice(&xv);
            for paired in [false, true] {
                let want_x = pieces(dir, tokens, nb, paired);
                let got_x = gpu.buffer(Gpu::f16t_pad(tokens) * nb * 48).unwrap();
                let k = if paired { gpu.quant_xp_kernel(false) } else { gpu.quant_x_kernel(false) }.unwrap();
                gpu.quant_x(&k, &xb, nb, tokens, &got_x, 1).unwrap();
                let (a, b) = (got_x.as_slice::<u8>(), want_x.as_slice::<u8>());
                let mut differ = 0;
                for p in 0..tokens * nb {
                    let (d, dr) = (f32::from_le_bytes(a[48 * p + 32..48 * p + 36].try_into().unwrap()), f32::from_le_bytes(b[48 * p + 32..48 * p + 36].try_into().unwrap()));
                    assert!((d - dr).abs() <= dr * 2f32.powi(-11), "{} {name} piece {p}: d {d}, gguf-py {dr}", dir.display());
                    for k in 0..32 {
                        let (q, qr) = (a[48 * p + k] as i8 as i32, b[48 * p + k] as i8 as i32);
                        assert!((q - qr).abs() <= 1, "{} {name} piece {p} byte {k} (paired {paired}): code {q}, gguf-py {qr}", dir.display());
                        differ += (q != qr) as usize;
                    }
                }
                assert!(differ * 1000 <= tokens * cols, "{} {name}: {differ} of {} codes differ from gguf-py", dir.display(), tokens * cols);
            }
            // qx_gemv_q8.comp on gguf-py's codes.
            let m = MatQ::new(&gpu, &w, rows).unwrap();
            let plain = pieces(dir, tokens, nb, false);
            for nt in [1usize, 3, 8] {
                let k = gpu.qx_gemv_q8_kernel(m.ty, false, nt as u32).unwrap();
                for t0 in (0..tokens).step_by(nt) {
                    let n = (tokens - t0).min(nt);
                    let mut part = gpu.buffer(n * nb * 48).unwrap();
                    part.as_mut_slice::<u8>().copy_from_slice(&plain.as_slice::<u8>()[t0 * nb * 48..(t0 + n) * nb * 48]);
                    let out = gpu.buffer(n * rows * 4).unwrap();
                    gpu.qx_gemv_q8(&k, &m, &part, n, &out, 1).unwrap();
                    let (_, rel) = crate::testutil::max_errors(&out.as_slice::<f32>()[..n * rows], &want[t0 * rows..(t0 + n) * rows]);
                    assert!(rel < tol, "{} {name} gemv, {nt} a dispatch, tokens {t0}..{}: rel {rel:.2e}", dir.display(), t0 + n);
                }
            }
            // qx_gemm_q8.comp on every format, set and accumulating, with
            // a row stride longer than the rows.
            {
                let mp = MatQT::new_paired(&gpu, &w, rows).unwrap();
                let xp = pieces(dir, tokens, nb, true);
                let npad = Gpu::f16t_pad(tokens);
                let stride = rows + 3;
                let push: Vec<u8> = [rows as u32, nb as u32, tokens as u32, npad as u32, stride as u32].iter().flat_map(|v| v.to_le_bytes()).collect();
                let groups = [rows.div_ceil(256) as u32, (npad / 8) as u32, 1];
                for accum in [false, true] {
                    let k = gpu.qx_gemm_q8_kernel(mp.ty, accum).unwrap();
                    let mut out = gpu.buffer(tokens * stride * 4).unwrap();
                    out.as_mut_slice::<f32>().fill(if accum { 1.0 } else { f32::NAN });
                    gpu.run(&k, &[&mp.q, &mp.s, &xp, &out], &push, groups, 1).unwrap();
                    let got: Vec<f32> = (0..tokens).flat_map(|t| out.as_slice::<f32>()[t * stride..t * stride + rows].to_vec()).collect();
                    let wanted: Vec<f32> = want.iter().map(|v| v + if accum { 1.0 } else { 0.0 }).collect();
                    let (_, rel) = crate::testutil::max_errors(&got, &wanted[..tokens * rows]);
                    // MatQT keeps a K-quant's per-piece scales (d · scale, dmin ·
                    // min) as f16. The random blocks of these goldens have scales
                    // that nearly cancel, so that rounding shows as up to 5e-3
                    // (numpy with the scales rounded to f16 gives the same
                    // 4.66e-3 on quant_k_1x256_t1); a wrong offset or scale is
                    // off by several percent.
                    let bound = if mp.ty >= 12 { 1e-2 } else { tol };
                    assert!(rel < bound, "{} {name} qx gemm (accum {accum}): rel {rel:.2e}", dir.display());
                }
            }
            // q4_0_gemm_q8.comp, set and accumulating.
            if let Weight::Q4_0(blocks) = &w {
                let mt = MatQ4_0T::new(&gpu, blocks, rows).unwrap();
                let xp = pieces(dir, tokens, nb, true);
                let npad = Gpu::f16t_pad(tokens);
                let push: Vec<u8> = [rows as u32, nb as u32, tokens as u32, npad as u32].iter().flat_map(|v| v.to_le_bytes()).collect();
                let groups = [rows.div_ceil(256) as u32, (npad / 8) as u32, 1];
                for accum in [false, true] {
                    let k = gpu.q4_0_gemm_q8_kernel(accum).unwrap();
                    let mut out = gpu.buffer(tokens * rows * 4).unwrap();
                    out.as_mut_slice::<f32>().fill(if accum { 1.0 } else { f32::NAN });
                    gpu.run(&k, &[&mt.q, &mt.d, &xp, &out], &push, groups, 1).unwrap();
                    let wanted: Vec<f32> = want.iter().map(|v| v + if accum { 1.0 } else { 0.0 }).collect();
                    let (_, rel) = crate::testutil::max_errors(&out.as_slice::<f32>()[..tokens * rows], &wanted[..tokens * rows]);
                    assert!(rel < tol, "{} {name} gemm (accum {accum}): rel {rel:.2e}", dir.display());
                }
            }
        };
        let dirs = |r: &std::path::Path| std::fs::read_dir(r).unwrap().map(|e| e.unwrap().path()).collect::<Vec<_>>();
        for dir in dirs(&root) {
            let (ws, _) = crate::testutil::read_npy(&dir.join("w.npy"));
            let bytes = |n: &str| std::fs::read(dir.join(format!("w_{n}.bin"))).unwrap();
            check(&dir, "q4_0", Weight::Q4_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q4_0")))), ws[0]);
            check(&dir, "q5_0", Weight::Q5_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q5_0")))), ws[0]);
            check(&dir, "q8_0", Weight::Q8_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q8_0")))), ws[0]);
            cases += 1;
        }
        for dir in dirs(&root_k) {
            let bytes = |n: &str| std::fs::read(dir.join(format!("w_{n}.bin"))).unwrap();
            let (ds, _) = crate::testutil::read_npy(&dir.join("deq_q4_k.npy"));
            check(&dir, "q4_k", Weight::Q4_K(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q4_k")))), ds[0]);
            check(&dir, "q6_k", Weight::Q6_K(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q6_k")))), ds[0]);
            cases += 1;
        }
        assert!(cases >= 20, "only {cases} golden cases");
    }

    /// The int8 path (`quant_x`, `qx_gemv_q8`). First, x's codes and
    /// scales against gguf-py's Q8_0 quantizer (llama.cpp's reference:
    /// the same codes; d within f16 rounding, as ours stays f32), plain
    /// and through SwiGLU. Then the products, 1, 3 and 8 tokens a
    /// dispatch, all five formats, against numpy's X' @ dequant(W)ᵀ in
    /// float64, where X' is x as our codes say (code · d) and dequant is
    /// gguf-py's.
    #[cfg(not(target_os = "android"))]
    #[test]
    fn qx_gemv_q8_matches_reference() {
        let (Some(root), Some(root_k)) = (crate::testutil::golden("quant_batch"), crate::testutil::golden("quant_k")) else { return };
        let Ok(gpu) = Gpu::new() else { return };
        if !gpu.info.int_dot {
            return;
        }
        let py = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/.venv/bin/python");
        let gguf_py = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("llama.cpp/gguf-py");
        assert!(py.exists() && gguf_py.join("gguf/quants.py").exists(), "needs tools/.venv and ~/llama.cpp/gguf-py");
        let script = "import sys\nsys.path.insert(0, sys.argv[1])\nimport numpy as np\nfrom gguf import GGMLQuantizationType as T\nfrom gguf.quants import dequantize, Q8_0\nmode = sys.argv[2]\nif mode == 'quant':\n    rows, cols, glu = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6] == '1'\n    x = np.fromfile(sys.argv[3], dtype='<f4').reshape(rows, -1)\n    if glu:\n        g, u = x[:, :cols], x[:, cols:]\n        x = (g / (np.float32(1) + np.exp(-g)) * u).astype(np.float32)\n    b = Q8_0.quantize_blocks(x.reshape(-1, 32))\n    b[:, 2:].tofile(sys.argv[7])\n    b[:, :2].copy().view(np.float16).astype(np.float32).tofile(sys.argv[8])\nelse:\n    typ = {'q4_0': T.Q4_0, 'q5_0': T.Q5_0, 'q8_0': T.Q8_0, 'q4_k': T.Q4_K, 'q6_k': T.Q6_K}[sys.argv[3]]\n    rows, cols = int(sys.argv[4]), int(sys.argv[5])\n    w = dequantize(np.fromfile(sys.argv[6], dtype=np.uint8).reshape(rows, -1), typ).astype(np.float64)\n    x = np.fromfile(sys.argv[7], dtype='<f4').astype(np.float64).reshape(-1, cols)\n    (x @ w.T).astype(np.float32).tofile(sys.argv[8])\n";
        let dir_out = std::env::temp_dir().join(format!("aether-q8-{}", std::process::id()));
        std::fs::create_dir_all(&dir_out).unwrap();
        let run_py = |args: &[&str]| {
            let st = std::process::Command::new(&py).args(["-c", script, gguf_py.to_str().unwrap()]).args(args).output().unwrap();
            assert!(st.status.success(), "{}", String::from_utf8_lossy(&st.stderr));
        };
        let f32s = |b: &[u8]| b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
        let path = |n: &str| dir_out.join(n).to_str().unwrap().to_string();
        // Quantizes `tokens` rows of x (gate then up with `glu`) on the GPU,
        // checks the codes and scales against gguf-py, and returns the GPU's
        // xq buffer and x as its codes say.
        let quantize = |x: &[f32], tokens: usize, cols: usize, glu: bool| -> (Buffer, Vec<f32>) {
            let nb = cols / 32;
            let mut xb = gpu.buffer(x.len() * 4).unwrap();
            xb.as_mut_slice::<f32>().copy_from_slice(x);
            let xq = gpu.buffer(tokens * nb * 48).unwrap();
            gpu.quant_x(&gpu.quant_x_kernel(glu).unwrap(), &xb, nb, tokens, &xq, 1).unwrap();
            std::fs::write(path("x.bin"), x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
            run_py(&["quant", &path("x.bin"), &tokens.to_string(), &cols.to_string(), if glu { "1" } else { "0" }, &path("q.bin"), &path("d.bin")]);
            let (want_q, want_d) = (std::fs::read(path("q.bin")).unwrap(), f32s(&std::fs::read(path("d.bin")).unwrap()));
            let raw = xq.as_slice::<u8>();
            let (mut differ, mut deq) = (0, Vec::with_capacity(tokens * cols));
            for p in 0..tokens * nb {
                let piece = &raw[48 * p..48 * p + 48];
                let d = f32::from_le_bytes(piece[32..36].try_into().unwrap());
                let dr = want_d[p];
                assert!((d - dr).abs() <= dr * 2f32.powi(-11), "piece {p}: d {d}, gguf-py {dr}");
                for k in 0..32 {
                    let (q, qr) = (piece[k] as i8, want_q[32 * p + k] as i8);
                    assert!((q as i32 - qr as i32).abs() <= 1, "piece {p} value {k}: code {q}, gguf-py {qr}");
                    differ += (q != qr) as usize;
                    deq.push(q as f32 * d);
                }
            }
            // A code may differ only where x / d is within rounding of a
            // half: our 1 / d and exp are not numpy's to the last bit.
            assert!(differ * 1000 <= tokens * cols * if glu { 10 } else { 1 }, "{differ} of {} codes differ from gguf-py", tokens * cols);
            (xq, deq)
        };
        let mut cases = 0;
        let check = |dir: &std::path::Path, name: &str, w: Weight, rows: usize| {
            let (xs, xv) = crate::testutil::read_npy(&dir.join("x.npy"));
            let (tokens, cols) = (xs[0], xs[1]);
            let (xq, xd) = quantize(&xv, tokens, cols, false);
            std::fs::write(path("xd.bin"), xd.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
            run_py(&["ref", name, &rows.to_string(), &cols.to_string(), dir.join(format!("w_{name}.bin")).to_str().unwrap(), &path("xd.bin"), &path("want.bin")]);
            let want = f32s(&std::fs::read(path("want.bin")).unwrap());
            let m = MatQ::new(&gpu, &w, rows).unwrap();
            let nb = cols / 32;
            for nt in [1usize, 3, 8] {
                let k = gpu.qx_gemv_q8_kernel(m.ty, false, nt as u32).unwrap();
                for t0 in (0..tokens).step_by(nt) {
                    let n = (tokens - t0).min(nt);
                    let mut part = gpu.buffer(n * nb * 48).unwrap();
                    part.as_mut_slice::<u8>().copy_from_slice(&xq.as_slice::<u8>()[t0 * nb * 48..(t0 + n) * nb * 48]);
                    let out = gpu.buffer(n * rows * 4).unwrap();
                    gpu.qx_gemv_q8(&k, &m, &part, n, &out, 1).unwrap();
                    let (_, rel) = crate::testutil::max_errors(&out.as_slice::<f32>()[..n * rows], &want[t0 * rows..(t0 + n) * rows]);
                    assert!(rel < 1e-6 * (cols as f32).sqrt().max(4.0), "{} {name}, {nt} a dispatch, tokens {t0}..{}: rel {rel:.2e}", dir.display(), t0 + n);
                }
            }
            // The prefill product: the same codes in the order of nibble
            // pairs, then q4_0_gemm_q8.comp against the same reference.
            if let Weight::Q4_0(blocks) = &w {
                let mut xb = gpu.buffer(xv.len() * 4).unwrap();
                xb.as_mut_slice::<f32>().copy_from_slice(&xv);
                let npad = Gpu::f16t_pad(tokens);
                let xp = gpu.buffer(npad * nb * 48).unwrap();
                gpu.quant_x(&gpu.quant_xp_kernel(false).unwrap(), &xb, nb, tokens, &xp, 1).unwrap();
                let (plain, paired) = (xq.as_slice::<u8>(), xp.as_slice::<u8>());
                for p in 0..tokens * nb {
                    assert_eq!(plain[48 * p + 32..48 * p + 48], paired[48 * p + 32..48 * p + 48], "piece {p}: scale and sums");
                    for g in 0..4 {
                        for j in 0..4 {
                            assert_eq!(paired[48 * p + 8 * g + j], plain[48 * p + 8 * g + 2 * j], "piece {p} group {g}: even value {j}");
                            assert_eq!(paired[48 * p + 8 * g + 4 + j], plain[48 * p + 8 * g + 2 * j + 1], "piece {p} group {g}: odd value {j}");
                        }
                    }
                }
                let mt = MatQ4_0T::new(&gpu, blocks, rows).unwrap();
                let mm = gpu.q4_0_gemm_q8_kernel(false).unwrap();
                let out = gpu.buffer(tokens * rows * 4).unwrap();
                let push: Vec<u8> = [rows as u32, nb as u32, tokens as u32, npad as u32].iter().flat_map(|v| v.to_le_bytes()).collect();
                gpu.run(&mm, &[&mt.q, &mt.d, &xp, &out], &push, [rows.div_ceil(256) as u32, (npad / 8) as u32, 1], 1).unwrap();
                let (_, rel) = crate::testutil::max_errors(&out.as_slice::<f32>()[..tokens * rows], &want[..tokens * rows]);
                assert!(rel < 1e-6 * (cols as f32).sqrt().max(4.0), "{} q4_0 prefill product: rel {rel:.2e}", dir.display());
            }
        };
        let dirs = |r: &std::path::Path| std::fs::read_dir(r).unwrap().map(|e| e.unwrap().path()).collect::<Vec<_>>();
        for dir in dirs(&root) {
            let (ws, _) = crate::testutil::read_npy(&dir.join("w.npy"));
            let bytes = |n: &str| std::fs::read(dir.join(format!("w_{n}.bin"))).unwrap();
            check(&dir, "q4_0", Weight::Q4_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q4_0")))), ws[0]);
            check(&dir, "q5_0", Weight::Q5_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q5_0")))), ws[0]);
            check(&dir, "q8_0", Weight::Q8_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q8_0")))), ws[0]);
            // SwiGLU's input: gate x, up a different row order of x.
            let (xs, xv) = crate::testutil::read_npy(&dir.join("x.npy"));
            let (tokens, cols) = (xs[0], xs[1]);
            let glu: Vec<f32> = (0..tokens).flat_map(|t| xv[t * cols..(t + 1) * cols].iter().chain(&xv[(tokens - 1 - t) * cols..(tokens - t) * cols]).copied().collect::<Vec<_>>()).collect();
            quantize(&glu, tokens, cols, true);
            cases += 1;
        }
        for dir in dirs(&root_k) {
            let bytes = |n: &str| std::fs::read(dir.join(format!("w_{n}.bin"))).unwrap();
            let (ds, _) = crate::testutil::read_npy(&dir.join("deq_q4_k.npy"));
            check(&dir, "q4_k", Weight::Q4_K(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q4_k")))), ds[0]);
            check(&dir, "q6_k", Weight::Q6_K(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q6_k")))), ds[0]);
            cases += 1;
        }
        let _ = std::fs::remove_dir_all(&dir_out);
        assert!(cases >= 20, "only {cases} golden cases");
    }

    /// The multi-format prefill shader against X @ dequant(W)ᵀ in float64:
    /// Q4_0, Q5_0, Q8_0 on llama.cpp's bytes (`quant_batch/`, odd row and
    /// token counts, so tiles have tails), Q4_K and Q6_K on random blocks
    /// dequantized by the gguf package (`quant_k/`).
    #[test]
    fn qx_gemm_matches_reference() {
        let (Some(root), Some(root_k)) = (crate::testutil::golden("quant_batch"), crate::testutil::golden("quant_k")) else { return };
        let Ok(gpu) = Gpu::new() else { return };
        let mut cases = 0;
        let dirs = |r: &std::path::Path| std::fs::read_dir(r).unwrap().map(|e| e.unwrap().path()).collect::<Vec<_>>();
        let check = |dir: &std::path::Path, name: &str, w: Weight, rows: usize| {
            let (xs, xv) = crate::testutil::read_npy(&dir.join("x.npy"));
            let (_, want) = crate::testutil::read_npy(&dir.join(format!("out_{name}_a.npy")));
            let (tokens, cols) = (xs[0], xs[1]);
            let m = MatQ::new(&gpu, &w, rows).unwrap();
            let mut x = gpu.buffer(xv.len() * 4).unwrap();
            x.as_mut_slice::<f32>().copy_from_slice(&xv);
            let mut out = gpu.buffer(want.len() * 4).unwrap();
            out.as_mut_slice::<f32>().fill(f32::NAN);
            let k = gpu.qx_gemm_kernel(m.ty, false, false).unwrap();
            gpu.qx_gemm(&k, &m, &x, tokens, &out, 0, 1).unwrap();
            let (_, rel) = crate::testutil::max_errors(out.as_slice::<f32>(), &want);
            assert!(rel < 1e-6 * (cols as f32).sqrt().max(4.0), "{} {name}: rel {rel:.2e}", dir.display());
        };
        for dir in dirs(&root) {
            let (ws, _) = crate::testutil::read_npy(&dir.join("w.npy"));
            let bytes = |n: &str| std::fs::read(dir.join(format!("w_{n}.bin"))).unwrap();
            check(&dir, "q4_0", Weight::Q4_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q4_0")))), ws[0]);
            check(&dir, "q5_0", Weight::Q5_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q5_0")))), ws[0]);
            check(&dir, "q8_0", Weight::Q8_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q8_0")))), ws[0]);
            cases += 1;
        }
        for dir in dirs(&root_k) {
            let bytes = |n: &str| std::fs::read(dir.join(format!("w_{n}.bin"))).unwrap();
            let (ds, _) = crate::testutil::read_npy(&dir.join("deq_q4_k.npy"));
            check(&dir, "q4_k", Weight::Q4_K(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q4_k")))), ds[0]);
            check(&dir, "q6_k", Weight::Q6_K(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q6_k")))), ds[0]);
            cases += 1;
        }
        assert!(cases >= 20, "only {cases} golden cases");
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

    /// The f16 prefill shader on llama.cpp's Q4_0 bytes against X @
    /// dequant(W)ᵀ in float64 (the batch goldens). X, W's scales and each
    /// product are rounded to f16 (2⁻¹¹ each), and each block's 32
    /// products are summed in f16 before the f32 sum: the bound is a few
    /// f16 roundings of the largest output.
    #[test]
    fn q4_0_gemm_h_matches_reference() {
        let Some(root) = crate::testutil::golden("quant_batch") else { return };
        let Ok(gpu) = Gpu::new() else { return };
        let (Ok(t), Ok(mm)) = (gpu.to_f16t_kernel(false), gpu.q4_0_gemm_h_kernel(false)) else { return };
        // The texel-buffer variant reads the same activations through texelFetch.
        let mmt = gpu.q4_0_gemm_ht_kernel(false).unwrap();
        let mut cases = 0;
        for dir in std::fs::read_dir(root).unwrap().map(|e| e.unwrap().path()) {
            cases += 1;
            let (ws, _) = crate::testutil::read_npy(&dir.join("w.npy"));
            let (xs, xv) = crate::testutil::read_npy(&dir.join("x.npy"));
            let (_, want) = crate::testutil::read_npy(&dir.join("out_q4_0_a.npy"));
            let blocks = read_q4_0(&dir.join("w_q4_0.bin"));
            let w = MatQ4_0T::new(&gpu, &blocks, ws[0]).unwrap();
            let mut x = gpu.buffer(xv.len() * 4).unwrap();
            x.as_mut_slice::<f32>().copy_from_slice(&xv);
            let xt = gpu.buffer(w.nb * 32 * Gpu::f16t_pad(xs[0]) * 2).unwrap();
            let mut out = gpu.buffer(want.len() * 4).unwrap();
            out.as_mut_slice::<f32>().fill(f32::NAN);
            for k in [&mm, &mmt] {
                out.as_mut_slice::<f32>().fill(f32::NAN);
                gpu.q4_0_gemm_h(&t, k, &w, &x, &xt, xs[0], &out, 0).unwrap();
                let (_, rel) = crate::testutil::max_errors(out.as_slice::<f32>(), &want);
                assert!(rel < 4.0 * 2f32.powi(-11), "{}: rel {rel:.2e}", dir.display());
            }
        }
        assert!(cases >= 11, "only {cases} golden cases");
    }

    /// The f16 multi-format prefill shader against X @ dequant(W)ᵀ in
    /// float64, on the goldens of `qx_gemm_matches_reference`, with the
    /// bound of `q4_0_gemm_h_matches_reference` (the K-quants' per-piece
    /// scales are also rounded to f16 at upload).
    #[test]
    fn qx_gemm_h_matches_reference() {
        let (Some(root), Some(root_k)) = (crate::testutil::golden("quant_batch"), crate::testutil::golden("quant_k")) else { return };
        let Ok(gpu) = Gpu::new() else { return };
        let Ok(t) = gpu.to_f16t_kernel(false) else { return };
        let mut cases = 0;
        let dirs = |r: &std::path::Path| std::fs::read_dir(r).unwrap().map(|e| e.unwrap().path()).collect::<Vec<_>>();
        let check = |dir: &std::path::Path, name: &str, w: Weight, rows: usize| {
            let (xs, xv) = crate::testutil::read_npy(&dir.join("x.npy"));
            let (_, want) = crate::testutil::read_npy(&dir.join(format!("out_{name}_a.npy")));
            let tokens = xs[0];
            let m = MatQT::new(&gpu, &w, rows).unwrap();
            let mut x = gpu.buffer(xv.len() * 4).unwrap();
            x.as_mut_slice::<f32>().copy_from_slice(&xv);
            let xt = gpu.buffer(m.nb * 32 * Gpu::f16t_pad(tokens) * 2).unwrap();
            let mut out = gpu.buffer(want.len() * 4).unwrap();
            out.as_mut_slice::<f32>().fill(f32::NAN);
            let k = gpu.qx_gemm_h_kernel(m.ty, false).unwrap();
            gpu.qx_gemm_h(&t, &k, &m, &x, &xt, tokens, &out, 0).unwrap();
            let deq = dir.join(format!("deq_{name}.npy"));
            if !deq.exists() {
                let (_, rel) = crate::testutil::max_errors(out.as_slice::<f32>(), &want);
                assert!(rel < 4.0 * 2f32.powi(-11), "{} {name}: rel {rel:.2e}", dir.display());
                return;
            }
            // The K-quant cases: random blocks, where Q4_K's mins can make
            // a whole row lean one way and a single output a sum that
            // cancels (1x256: 2.3e-2 of itself). Each output is checked
            // against the sum of |w| · |x| that its roundings scale with
            // (W the gguf package's dequantization); measured up to 0.53
            // · 2⁻¹¹.
            let (_, wd) = crate::testutil::read_npy(&deq);
            let cols = wd.len() / rows;
            for (i, (&got, &r)) in out.as_slice::<f32>().iter().zip(&want).enumerate() {
                let (t, row) = (i / rows, i % rows);
                let mag: f32 = (0..cols).map(|k| (wd[row * cols + k] * xv[t * cols + k]).abs()).sum();
                assert!((got - r).abs() < 2f32.powi(-11) * mag, "{} {name} token {t} row {row}: {got} against {r}", dir.display());
            }
        };
        for dir in dirs(&root) {
            let (ws, _) = crate::testutil::read_npy(&dir.join("w.npy"));
            let bytes = |n: &str| std::fs::read(dir.join(format!("w_{n}.bin"))).unwrap();
            check(&dir, "q4_0", Weight::Q4_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q4_0")))), ws[0]);
            check(&dir, "q5_0", Weight::Q5_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q5_0")))), ws[0]);
            check(&dir, "q8_0", Weight::Q8_0(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q8_0")))), ws[0]);
            cases += 1;
        }
        for dir in dirs(&root_k) {
            let bytes = |n: &str| std::fs::read(dir.join(format!("w_{n}.bin"))).unwrap();
            let (ds, _) = crate::testutil::read_npy(&dir.join("deq_q4_k.npy"));
            check(&dir, "q4_k", Weight::Q4_K(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q4_k")))), ds[0]);
            check(&dir, "q6_k", Weight::Q6_K(crate::weight::Store::Owned(quant::blocks_from_bytes(&bytes("q6_k")))), ds[0]);
            cases += 1;
        }
        assert!(cases >= 20, "only {cases} golden cases");
    }

    /// The prompt's attention shaders on causal, grouped-query batches
    /// (`attention_causal_<nh>x<nkv>x<hs>_p<pos0>_t<nt>`, tools/gen_kernels.py).
    /// q, k and v are laid out and dispatched as `GpuModel::record_pass`
    /// does for a batch; `flash` runs `attention_flash.comp` (into `xb2`),
    /// else `attention_tile.comp` + `attention_merge.comp` (the `chunked`
    /// option, through `parts`).
    ///
    /// The keys and values pass through f16 in shared memory (2⁻¹¹
    /// relative each). The output is a weighted average of values, so its
    /// absolute error is about one value's rounding of max |v|, plus the
    /// scores' error carried through exp: a score is hs products each off
    /// by 2⁻¹¹ of its size, so δs ≈ √hs·2⁻¹¹ (≈ 8·2⁻¹¹ at hs = 64 for
    /// unit-variance inputs), and the weights follow by about w·δs. The
    /// bound is 4·2⁻¹¹ of max |v| — the largest value, not the largest
    /// output, whose size falls as the average takes in more positions.
    fn prompt_attention_matches(gpu: &Gpu, flash: bool) -> Option<(f32, usize)> {
        let root = crate::testutil::golden("kernels")?;
        let mut dirs: Vec<_> = std::fs::read_dir(root)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.file_name().unwrap().to_str().unwrap().starts_with("attention_causal_"))
            .collect();
        dirs.sort();
        let (mut worst, mut tested) = (0.0f32, 0);
        for dir in &dirs {
            let name = dir.file_name().unwrap().to_str().unwrap();
            let (shape, tail) = name.strip_prefix("attention_causal_").unwrap().split_once("_p").unwrap();
            let (pos0, nt) = tail.split_once("_t").unwrap();
            let dims: Vec<usize> = shape.split('x').map(|s| s.parse().unwrap()).collect();
            let (nh, nkv, hs) = (dims[0], dims[1], dims[2]);
            // The shaders stop at head size 64 (LFM3's 128 is plans/lfm3.md A8).
            if hs > 64 {
                continue;
            }
            let (pos0, nt): (usize, usize) = (pos0.parse().unwrap(), nt.parse().unwrap());
            let (_, q) = crate::testutil::read_npy(&dir.join("q.npy"));
            let (_, k) = crate::testutil::read_npy(&dir.join("k.npy"));
            let (_, v) = crate::testutil::read_npy(&dir.join("v.npy"));
            let (_, want) = crate::testutil::read_npy(&dir.join("out.npy"));
            let qstride = nh * hs + 2 * nkv * hs;
            let chunks = (pos0 + nt).div_ceil(64);
            // q at tok * qstride + h * hs, k and v at pos * kv_dim + kvh * hs
            // (layer_off 0); parts (m, l, out) per token, head and chunk.
            let mut qb = gpu.buffer(nt * qstride * 4).unwrap();
            qb.as_mut_slice::<f32>().fill(0.0);
            for tok in 0..nt {
                let (d, s) = (tok * qstride, tok * nh * hs);
                qb.as_mut_slice::<f32>()[d..d + nh * hs].copy_from_slice(&q[s..s + nh * hs]);
            }
            let mut kb = gpu.buffer(k.len() * 4).unwrap();
            kb.as_mut_slice::<f32>().copy_from_slice(&k);
            let mut vb = gpu.buffer(v.len() * 4).unwrap();
            vb.as_mut_slice::<f32>().copy_from_slice(&v);
            // Only the chunked path reads parts; NaN in it would fail the
            // test if a shader read a chunk it did not write.
            let mut prts = (!flash).then(|| {
                let mut b = gpu.buffer(nt * nh * chunks * (hs + 2) * 4).unwrap();
                b.as_mut_slice::<f32>().fill(f32::NAN);
                b
            });
            let mut out = gpu.buffer(nt * nh * hs * 4).unwrap();
            out.as_mut_slice::<f32>().fill(f32::NAN);
            let mut params = gpu.buffer(4).unwrap();
            params.as_mut_slice::<u32>()[0] = pos0 as u32;
            let cmds = if flash {
                let k = Kernel::new(gpu, crate::spirv!("attention_flash"), 5, 24).unwrap();
                let b = k.bind(gpu, &[&qb, &kb, &vb, &out, &params]).unwrap();
                let p = push_u32(&[nh as u32, nkv as u32, hs as u32, 0, qstride as u32, nt as u32]);
                gpu.record(|r| r.dispatch(&b, &p, [nh as u32, nt.div_ceil(32) as u32, 1])).unwrap()
            } else {
                let parts = prts.as_mut().unwrap();
                let tile = Kernel::new(gpu, crate::spirv!("attention_tile"), 5, 28).unwrap();
                let merge = Kernel::new(gpu, crate::spirv!("attention_merge"), 3, 8).unwrap();
                let tb = tile.bind(gpu, &[&qb, &kb, &vb, parts, &params]).unwrap();
                let mb = merge.bind(gpu, &[parts, &out, &params]).unwrap();
                let tp = push_u32(&[nh as u32, nkv as u32, hs as u32, chunks as u32, 0, qstride as u32, nt as u32]);
                let mp = push_u32(&[hs as u32, chunks as u32]);
                gpu.record(|r| {
                    r.dispatch(&tb, &tp, [nh as u32, chunks as u32, nt.div_ceil(32) as u32]);
                    r.barrier();
                    r.dispatch(&mb, &mp, [nh as u32, nt as u32, 1]);
                })
                .unwrap()
            };
            gpu.submit(&cmds, false).unwrap();
            // The error is relative to the largest value, not to the largest
            // output: every value is rounded to f16 in shared memory (2⁻¹¹
            // relative), so the weighted average's absolute error is about
            // 2⁻¹¹ of max |v| whatever the context, while the output itself
            // shrinks as the average takes in more values (at 4000 positions
            // its largest component is 0.12 against v's 4.2). The scores'
            // f16 error moves the weights by a similar amount. Measured:
            // 1.6 · 2⁻¹¹ of max |v| on the Adreno, 0.73 on the Intel GPU.
            let (abs, _) = crate::testutil::max_errors(out.as_slice::<f32>(), &want);
            let vmax = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let bound = 4.0 * 2f32.powi(-11) * vmax;
            assert!(abs <= bound, "{name}: abs {abs:.2e}, {:.2e} of max |v| (bound {bound:.2e})", abs / vmax);
            worst = worst.max(abs / vmax);
            tested += 1;
        }
        Some((worst, tested))
    }

    #[test]
    fn attention_chunked_matches_reference() {
        let Ok(gpu) = Gpu::new() else { return };
        let Some((worst, cases)) = prompt_attention_matches(&gpu, false) else { return };
        println!("attention chunked: {cases} cases vs numpy, worst {worst:.2e} of max |v|");
        assert!(cases >= 60, "only {cases} golden cases");
    }

    #[test]
    fn attention_flash_matches_reference() {
        let Ok(gpu) = Gpu::new() else { return };
        let Some((worst, cases)) = prompt_attention_matches(&gpu, true) else { return };
        println!("attention flash: {cases} cases vs numpy, worst {worst:.2e} of max |v|");
        assert!(cases >= 60, "only {cases} golden cases");
    }
}
