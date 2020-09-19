use renderdoc::{RenderDoc, V100};
use std::fs::File;
use std::io::BufReader;
use std::mem::size_of;
use std::mem::size_of_val;
use std::path::PathBuf;
use structopt::StructOpt;
use svgtypes::PathParser;
use xml::reader::{EventReader, XmlEvent};
use ash::vk;
use ash::version::EntryV1_0;
use ash::version::DeviceV1_0;
use ash::version::InstanceV1_0;
use vk_mem;

const WORKGROUP_X: u32 = 8;
const WORKGROUP_Y: u32 = 8;
const WORKGROUP_Z: u32 = 16;

pub const FIX_THRESHOLD: f32 = 0.01;

mod bezier;
use bezier::*;

#[macro_use]
mod shader_include;

fn num_groups(x: u32, y: u32) -> u32 {
    (x + y - 1) / y
}

fn round_up(x: u32, y: u32) -> u32 {
    num_groups(x, y) * y
}

fn num_workgroups(opt: &Options) -> [u32; 3] {
    [
        num_groups(opt.width, WORKGROUP_X),
        num_groups(opt.height, WORKGROUP_Y),
        1,
    ]
}

fn get_max_local_memory(props: &vk::PhysicalDeviceMemoryProperties) -> vk::DeviceSize {
    props.memory_heaps
        .iter()
        .zip(props.memory_types.iter())
        .filter(|(_, t)| t.property_flags.contains(
                vk::MemoryPropertyFlags::DEVICE_LOCAL))
        .map(|(h, _)| h.size)
        .max()
        .unwrap_or(0)
}

fn evaluate_physical_device(instance: &ash::Instance, pd: vk::PhysicalDevice) -> (bool, vk::DeviceSize) {
    let (props, mem_props) = unsafe {(
        instance.get_physical_device_properties(pd),
        instance.get_physical_device_memory_properties(pd),
    )};
    (
        props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
        get_max_local_memory(&mem_props),
    )
}

#[derive(Debug, StructOpt)]
#[structopt(name = "render", about = "Render an SVG file to image.")]
struct Options {
    /// The input file name
    #[structopt(parse(from_os_str))]
    input: PathBuf,
    /// The output file name
    #[structopt(short, parse(from_os_str))]
    output: Option<PathBuf>,
    /// Output width
    #[structopt(short, long)]
    width: u32,
    /// Output height
    #[structopt(short, long)]
    height: u32,
    /// Scale factor
    #[structopt(short, long, default_value="1.0")]
    scale: f32,
    /// Origin X position
    #[structopt(short = "x", long, default_value="0.0")]
    origin_x: f32,
    /// Origin Y position
    #[structopt(short = "y", long, default_value="0.0")]
    origin_y: f32,
    /// SDF scale (larger = more spread out)
    #[structopt(long, default_value="1.0")]
    sdf_scale: f32,
    /// Gradient scale (larger = steeper)
    #[structopt(long, default_value="0.70710678118")]
    grad_scale: f32,
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct FillVertex {
    pos_params: [f32; 4],
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct FillPushConstants {
    size: [f32; 2],
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct SdfPushConstants {
    start_index: u32,
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct SdfCurve {
    from_to: [f32; 4],
    control: [f32; 4],
}

#[derive(Debug)]
enum SomeError {
    Vulkan(vk::Result),
    VkMem(vk_mem::error::Error),
}

impl core::convert::From<vk::Result> for SomeError {
    fn from(x: vk::Result) -> Self { Self::Vulkan(x) }
}

impl core::convert::From<vk_mem::error::Error> for SomeError {
    fn from(x: vk_mem::error::Error) -> Self { Self::VkMem(x) }
}

macro_rules! cstr {
    ($s: literal) => { std::ffi::CStr::from_bytes_with_nul_unchecked($s) }
}

unsafe fn upload_buffer<T: Copy>(
        device: &ash::Device,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        allocator: &vk_mem::Allocator,
        dest: vk::Buffer,
        data: &[T]) -> Result<(), SomeError> {
    let size = size_of_val(data) as vk::DeviceSize;
    let (buf, alloc, info) = allocator.create_buffer(
        &vk::BufferCreateInfo::builder()
            .size(size as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC),
        &vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::CpuOnly,
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            ..Default::default()
        }
    )?;

    let mapped = info.get_mapped_data();
    std::ptr::copy_nonoverlapping(data.as_ptr(), mapped.cast(), data.len());

    let cmd = device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1)
    )?.pop().unwrap();

    device.begin_command_buffer(
        cmd,
        &vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
    )?;
    device.cmd_copy_buffer(cmd, buf, dest, &[vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size,
    }]);
    device.end_command_buffer(cmd)?;

    let command_buffers = &[cmd];
    device.queue_submit(
        queue,
        &[vk::SubmitInfo::builder().command_buffers(command_buffers).build()],
        vk::Fence::null(),
    )?;

    device.device_wait_idle().expect("error waiting for idle");
    allocator.destroy_buffer(buf, &alloc).unwrap();

    Ok(())
}

unsafe fn download_image<Pixel: Copy>(
        device: &ash::Device,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        allocator: &vk_mem::Allocator,
        src: vk::Image,
        layout: vk::ImageLayout,
        width: u32,
        height: u32) -> Result<Vec<Pixel>, SomeError> {
    let bytes_per_pixel = size_of::<Pixel>() as vk::DeviceSize;
    let num_pixels = width as vk::DeviceSize * height as vk::DeviceSize;
    let size = num_pixels * bytes_per_pixel;
    let (buf, alloc, info) = allocator.create_buffer(
        &vk::BufferCreateInfo::builder()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST),
        &vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::CpuOnly,
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            ..Default::default()
        }
    )?;

    let cmd = device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1)
    )?.pop().unwrap();

    device.begin_command_buffer(
        cmd,
        &vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
    )?;
    device.cmd_copy_image_to_buffer(cmd,
        src, layout,
        buf,
        &[
            vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D {
                    x: 0,
                    y: 0,
                    z: 0,
                },
                image_extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
            }
        ]
    );
    device.end_command_buffer(cmd)?;

    let command_buffers = &[cmd];
    device.queue_submit(
        queue,
        &[vk::SubmitInfo::builder().command_buffers(command_buffers).build()],
        vk::Fence::null(),
    )?;

    device.device_wait_idle()?;

    let mapped = info.get_mapped_data() as *const Pixel;
    let vec = std::slice::from_raw_parts(mapped, num_pixels as usize).to_vec();

    allocator.destroy_buffer(buf, &alloc)?;

    Ok(vec)
}

unsafe fn create_shader(device: &ash::Device, bytes: &[u8]) -> ash::prelude::VkResult<vk::ShaderModule> {
    let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap();
    device.create_shader_module(
        &vk::ShaderModuleCreateInfo::builder().code(&words),
        None
    )
}

struct MainUnsafeResults {
    sdf_data: Vec<[f32; 4]>,
}

unsafe fn main_unsafe(
    opt: &Options,
    mut rd: Option<RenderDoc<V100>>,
    vertices: &[FillVertex],
    curves: &[SdfCurve]
) -> Result<MainUnsafeResults, SomeError> {
    let gpu_w = round_up(opt.width, WORKGROUP_X);
    let gpu_h = round_up(opt.height, WORKGROUP_Y);

    let entry = ash::Entry::new().expect("error loading Vulkan");
    let instance = entry.create_instance(
        &vk::InstanceCreateInfo::builder()
            .application_info(
                &vk::ApplicationInfo::builder()
                    .application_name(cstr!(b"render_svg\0"))
                    .application_version(vk::make_version(0, 1, 0))
            ),
        None,
    ).expect("error creating instance");
    let physical = instance.enumerate_physical_devices()?
        .into_iter()
        .max_by_key(|pd| evaluate_physical_device(&instance, *pd))
        .expect("no suitable physical device");
    let pd_name = std::ffi::CStr::from_ptr(
        &instance.get_physical_device_properties(physical)
            .device_name[0] as *const std::os::raw::c_char
    ).to_owned();
    let mem_size = get_max_local_memory(
        &instance.get_physical_device_memory_properties(physical));
    println!("using physical device {} with {} GiB of memory",
             pd_name.to_string_lossy(),
             mem_size as f64 / 2usize.pow(30) as f64);

    // find queue family
    let queue_family = instance.get_physical_device_queue_family_properties(physical)
        .into_iter()
        .position(|props| {
            props.queue_flags.contains(
                vk::QueueFlags::empty()
                | vk::QueueFlags::COMPUTE
                | vk::QueueFlags::GRAPHICS
                | vk::QueueFlags::TRANSFER
            )
        })
        .expect("no queue family supports everything") as u32;

    let device = {
        let priorities = [0.5f32];
        instance.create_device(
            physical,
            &vk::DeviceCreateInfo::builder()
                .queue_create_infos(&[
                    vk::DeviceQueueCreateInfo {
                        queue_family_index: queue_family,
                        queue_count: 1,
                        p_queue_priorities: &priorities[0] as *const f32,
                        ..Default::default()
                    }
                ])
                .enabled_features(
                    &vk::PhysicalDeviceFeatures::builder()
                        .logic_op(true)
                )
                .enabled_extension_names(&[
                    vk::KhrStorageBufferStorageClassFn::name().as_ptr()
                ]),
            None,
        )?
    };

    let queue = device.get_device_queue(queue_family, 0);

    match rd.as_mut() {
        Some(r) => {
            println!("starting frame capture");
            r.start_frame_capture(std::ptr::null(), std::ptr::null());
        },
        None => (),
    }

    let mut allocator = vk_mem::Allocator::new(&vk_mem::AllocatorCreateInfo {
        physical_device: physical,
        device: device.clone(),
        instance: instance.clone(),
        ..Default::default()
    }).expect("error creating allocator");

    let command_pool = device.create_command_pool(
        &vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(queue_family),
        None,
    )?;

    let (fill_image, fill_image_alloc, _) = allocator.create_image(
        &vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8_UNORM)
            .extent(vk::Extent3D {
                width: gpu_w,
                height: gpu_h,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::empty()
                   | vk::ImageUsageFlags::TRANSFER_SRC
                   | vk::ImageUsageFlags::TRANSFER_DST
                   | vk::ImageUsageFlags::COLOR_ATTACHMENT
                   | vk::ImageUsageFlags::STORAGE
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&[queue_family])
            .initial_layout(vk::ImageLayout::UNDEFINED)
        ,
        &vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        },
    )?;

    let (sdf_image, sdf_image_alloc, _) = allocator.create_image(
        &vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .extent(vk::Extent3D {
                width: gpu_w,
                height: gpu_h,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::empty()
                   | vk::ImageUsageFlags::TRANSFER_SRC
                   | vk::ImageUsageFlags::TRANSFER_DST
                   | vk::ImageUsageFlags::STORAGE
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&[queue_family])
            .initial_layout(vk::ImageLayout::UNDEFINED)
        ,
        &vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        },
    )?;

    let (vertex_buf, vertex_buf_alloc, _) = allocator.create_buffer(
        &vk::BufferCreateInfo::builder()
            .size(size_of_val(vertices) as u64)
            .usage(vk::BufferUsageFlags::empty()
                   | vk::BufferUsageFlags::TRANSFER_DST
                   | vk::BufferUsageFlags::VERTEX_BUFFER
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&[queue_family])
        ,
        &vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        },
    )?;

    upload_buffer(
        &device,
        queue,
        command_pool,
        &allocator,
        vertex_buf,
        vertices,
    )?;

    let (curves_buf, curves_buf_alloc, _) = allocator.create_buffer(
        &vk::BufferCreateInfo::builder()
            .size(size_of_val(curves) as u64)
            .usage(vk::BufferUsageFlags::empty()
                   | vk::BufferUsageFlags::TRANSFER_DST
                   | vk::BufferUsageFlags::STORAGE_BUFFER
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&[queue_family])
        ,
        &vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        },
    )?;

    upload_buffer(
        &device,
        queue,
        command_pool,
        &allocator,
        vertex_buf,
        vertices,
    )?;

    upload_buffer(
        &device,
        queue,
        command_pool,
        &allocator,
        curves_buf,
        curves,
    )?;

    // create pipelines

    let desc_pool = device.create_descriptor_pool(
        &vk::DescriptorPoolCreateInfo::builder()
            .max_sets(2)
            .pool_sizes(&[
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 2,
                },
            ])
        ,
        None
    )?;

    let fill_set_layout = device.create_descriptor_set_layout(
        &Default::default(),
        None
    )?;

    let fill_pipeline_layout = device.create_pipeline_layout(
        &vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&[
                fill_set_layout,
            ])
            .push_constant_ranges(&[
                vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    offset: 0,
                    size: size_of::<FillPushConstants>() as u32,
                }
            ])
        ,
        None,
    )?;

    let fill_renderpass = device.create_render_pass(
        &vk::RenderPassCreateInfo::builder()
            .attachments(&[
                vk::AttachmentDescription {
                    flags: Default::default(),
                    format: vk::Format::R8_UNORM,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::STORE,
                    stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                    stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                }
            ])
            .subpasses(&[
                vk::SubpassDescription::builder()
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .color_attachments(&[
                        vk::AttachmentReference {
                            attachment: 0,
                            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        }
                    ])
                    .build()
            ]),
        None,
    )?;

    let fill_vs = create_shader(&device, include_shader!("fill.vert")).unwrap();
    let fill_fs = create_shader(&device, include_shader!("fill.frag")).unwrap();

    let fill_pipeline = device.create_graphics_pipelines(
        vk::PipelineCache::null(),
        &[
            vk::GraphicsPipelineCreateInfo::builder()
                .stages(&[
                    vk::PipelineShaderStageCreateInfo::builder()
                        .stage(vk::ShaderStageFlags::VERTEX)
                        .module(fill_vs)
                        .name(cstr!(b"main\0"))
                        .build(),
                    vk::PipelineShaderStageCreateInfo::builder()
                        .stage(vk::ShaderStageFlags::FRAGMENT)
                        .module(fill_fs)
                        .name(cstr!(b"main\0"))
                        .build(),
                ])
                .vertex_input_state(
                    &vk::PipelineVertexInputStateCreateInfo::builder()
                        .vertex_binding_descriptions(&[
                            vk::VertexInputBindingDescription {
                                binding: 0,
                                stride: size_of::<FillVertex>() as u32,
                                input_rate: vk::VertexInputRate::VERTEX,
                            },
                        ])
                        .vertex_attribute_descriptions(&[
                            vk::VertexInputAttributeDescription {
                                location: 0,
                                binding: 0,
                                format: vk::Format::R32G32B32A32_SFLOAT,
                                offset: 0,
                            },
                        ])
                )
                .input_assembly_state(
                    &vk::PipelineInputAssemblyStateCreateInfo::builder()
                        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                )
                .viewport_state(
                    &vk::PipelineViewportStateCreateInfo::builder()
                        .viewports(&[
                            vk::Viewport {
                                x: 0.0,
                                y: 0.0,
                                width: gpu_w as f32,
                                height: gpu_h as f32,
                                min_depth: 0.0,
                                max_depth: 1.0,
                            },
                        ])
                        .scissors(&[
                            vk::Rect2D {
                                offset: vk::Offset2D {
                                    x: 0, y: 0,
                                },
                                extent: vk::Extent2D {
                                    width: gpu_w, height: gpu_h,
                                },
                            },
                        ])
                )
                .rasterization_state(
                    &vk::PipelineRasterizationStateCreateInfo::builder()
                        .polygon_mode(vk::PolygonMode::FILL)
                        .cull_mode(vk::CullModeFlags::NONE)
                        .line_width(1.0)
                )
                .multisample_state(
                    &vk::PipelineMultisampleStateCreateInfo::builder()
                        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                )
                .color_blend_state(
                    &vk::PipelineColorBlendStateCreateInfo::builder()
                        .logic_op_enable(true)
                        .logic_op(vk::LogicOp::XOR)
                        .attachments(&[
                            vk::PipelineColorBlendAttachmentState::builder()
                                .blend_enable(false)
                                .color_write_mask(vk::ColorComponentFlags::all())
                                .build()
                        ])
                )
                .layout(fill_pipeline_layout)
                .render_pass(fill_renderpass)
                .subpass(0)
                .build(),
        ],
        None
    ).map_err(|x| x.1)?.pop().unwrap();
    device.destroy_shader_module(fill_vs, None);
    device.destroy_shader_module(fill_fs, None);

    let fill_image_view = device.create_image_view(
        &vk::ImageViewCreateInfo::builder()
            .image(fill_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8_UNORM)
            .subresource_range(
                vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }
            )
        ,
        None
    )?;

    let framebuffer = device.create_framebuffer(
        &vk::FramebufferCreateInfo::builder()
            .render_pass(fill_renderpass)
            .attachments(&[fill_image_view])
            .width(gpu_w)
            .height(gpu_h)
            .layers(1)
        ,
        None
    )?;

    let sdf_set_layout = device.create_descriptor_set_layout(
        &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
        ]),
        None,
    )?;

    let sdf_pipeline_layout = device.create_pipeline_layout(
        &vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&[
                sdf_set_layout,
            ])
            .push_constant_ranges(&[
                vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    offset: 0,
                    size: size_of::<SdfPushConstants>() as u32,
                }
            ])
        ,
        None,
    )?;

    let sdf_cs = create_shader(&device, include_shader!("sdf.comp"))?;

    let sdf_pipeline = device.create_compute_pipelines(
        vk::PipelineCache::null(),
        &[
            vk::ComputePipelineCreateInfo::builder()
                .stage(
                    vk::PipelineShaderStageCreateInfo::builder()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(sdf_cs)
                        .name(cstr!(b"main\0"))
                        .build()
                )
                .layout(sdf_pipeline_layout)
                .build(),
        ],
        None,
    ).map_err(|x| x.1)?.pop().unwrap();
    device.destroy_shader_module(sdf_cs, None);

    let sdf_image_view = device.create_image_view(
        &vk::ImageViewCreateInfo::builder()
            .image(sdf_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .subresource_range(
                vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }
            )
        ,
        None
    )?;

    // create descriptor sets

    let desc_sets = device.allocate_descriptor_sets(
        &vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(desc_pool)
            .set_layouts(&[
                fill_set_layout,
                sdf_set_layout,
            ])
    )?;

    let fill_desc_set = desc_sets[0];
    let sdf_desc_set = desc_sets[1];

    device.update_descriptor_sets(
        &[
            vk::WriteDescriptorSet::builder()
                .dst_set(sdf_desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[
                    vk::DescriptorBufferInfo {
                        buffer: curves_buf,
                        offset: 0,
                        range: vk::WHOLE_SIZE,
                    },
                ])
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(sdf_desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&[
                    vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: fill_image_view,
                        image_layout: vk::ImageLayout::GENERAL,
                    },
                ])
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(sdf_desc_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&[
                    vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: sdf_image_view,
                        image_layout: vk::ImageLayout::GENERAL,
                    },
                ])
                .build(),
        ],
        &[], // copies
    );

    // execute render pass and dispatch

    let cmd = device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1)
    )?.pop().unwrap();

    device.begin_command_buffer(
        cmd,
        &vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
    )?;

    // draw fill texture

    device.cmd_begin_render_pass(cmd,
        &vk::RenderPassBeginInfo::builder()
            .render_pass(fill_renderpass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: gpu_w, height: gpu_h },
            })
            .clear_values(&[
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0]
                    }
                },
            ])
        ,
        vk::SubpassContents::INLINE,
    );
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, fill_pipeline);
    device.cmd_bind_vertex_buffers(
        cmd,
        0,
        &[
            vertex_buf,
        ],
        &[
            0,
        ],
    );
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        fill_pipeline_layout,
        0,
        &[
            fill_desc_set,
        ],
        &[],
    );
    device.cmd_push_constants(
        cmd,
        fill_pipeline_layout,
        vk::ShaderStageFlags::VERTEX,
        0,
        std::slice::from_raw_parts(
            &FillPushConstants {
                size: [gpu_w as f32, gpu_h as f32],
            } as *const FillPushConstants as *const u8,
            size_of::<FillPushConstants>(),
        ),
    );
    device.cmd_draw(cmd, vertices.len() as u32, 1, 0, 0);
    device.cmd_end_render_pass(cmd);

    // transition and clear sdf texture
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::empty()
        | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
        | vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[
            vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(fill_image)
                .subresource_range(
                    vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }
                )
                .build(),
            vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(sdf_image)
                .subresource_range(
                    vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }
                )
                .build(),
        ],
    );
    device.cmd_clear_color_image(
        cmd,
        sdf_image,
        vk::ImageLayout::GENERAL,
        &vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, std::f32::INFINITY]
        },
        &[
            vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        ],
    );

    // dispatch compute to generate sdf
    device.cmd_bind_pipeline(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        sdf_pipeline
    );
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        sdf_pipeline_layout,
        0,
        &[
            sdf_desc_set,
        ],
        &[],
    );

    let [wg_x, wg_y, wg_z] = num_workgroups(opt);
    for dispatch_index in 0..(curves.len() as u32 / WORKGROUP_Z) {
        if dispatch_index > 0 {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[
                    vk::MemoryBarrier::builder()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .build(),
                ],
                &[],
                &[],
            );
        }

        device.cmd_push_constants(
            cmd,
            sdf_pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            std::slice::from_raw_parts(
                &SdfPushConstants {
                    start_index: WORKGROUP_Z * dispatch_index,
                } as *const SdfPushConstants as *const u8,
                size_of::<SdfPushConstants>(),
            ),
        );
        device.cmd_dispatch(
            cmd,
            wg_x,
            wg_y,
            wg_z,
        );
    }

    device.end_command_buffer(cmd)?;

    let command_buffers = &[cmd];
    device.queue_submit(
        queue,
        &[vk::SubmitInfo::builder().command_buffers(command_buffers).build()],
        vk::Fence::null(),
    )?;

    device.device_wait_idle()?;

    // copy texture to CPU

    let sdf_data = download_image(
        &device,
        queue,
        command_pool,
        &allocator,
        sdf_image,
        vk::ImageLayout::GENERAL,
        opt.width,
        opt.height,
    )?;

    // cleanup:

    device.destroy_image_view(sdf_image_view, None);
    device.destroy_pipeline(sdf_pipeline, None);
    device.destroy_pipeline_layout(sdf_pipeline_layout, None);
    device.destroy_descriptor_set_layout(sdf_set_layout, None);

    device.destroy_image_view(fill_image_view, None);
    device.destroy_framebuffer(framebuffer, None);
    device.destroy_pipeline(fill_pipeline, None);
    device.destroy_render_pass(fill_renderpass, None);
    device.destroy_pipeline_layout(fill_pipeline_layout, None);
    device.destroy_descriptor_set_layout(fill_set_layout, None);

    device.destroy_descriptor_pool(desc_pool, None);

    allocator.destroy_buffer(curves_buf, &curves_buf_alloc).unwrap();
    allocator.destroy_buffer(vertex_buf, &vertex_buf_alloc).unwrap();
    allocator.destroy_image(sdf_image, &sdf_image_alloc).unwrap();
    allocator.destroy_image(fill_image, &fill_image_alloc).unwrap();
    allocator.destroy();

    match rd.as_mut() {
        Some(r) => {
            println!("ending frame capture");
            r.end_frame_capture(std::ptr::null(), std::ptr::null());
        },
        None => (),
    }

    device.destroy_command_pool(command_pool, None);
    device.destroy_device(None);
    instance.destroy_instance(None);

    Ok(MainUnsafeResults {
        sdf_data,
    })
}

fn main() {
    let opt = Options::from_args();

    let rd: Option<RenderDoc<V100>> = RenderDoc::new().ok();

    println!("*** loading SVG...");
    let file = File::open(&opt.input).expect("failed to open input file");
    let reader = BufReader::new(file);
    let parser = EventReader::new(reader);
    let mut path_d = None;
    'xml: for event in parser {
        match event.expect("XML error reading input file") {
            XmlEvent::StartElement { name, attributes, .. } => {
                if &name.local_name == "path" {
                    for attr in attributes {
                        if &attr.name.local_name == "d" {
                            path_d = Some(attr.value);
                            break 'xml;
                        }
                    }
                }
            }
            _ => ()
        }
    }
    let path_d = path_d.expect("no path with d attribute");
    let path = QuadraticPath::from_svg_path(
        PathParser::from(&path_d[..]).map(|x| x.expect("failed to parse path"))
    );
    println!("*** done; {} loops with {} quadratic segments total",
             path.loops.len(), path.loops.iter().map(|l| l.segments.len()).sum::<usize>());

    fn translate(vx: Vec2, u: f32, v: f32, opt: &Options) -> FillVertex {
        FillVertex {
            pos_params: [
                vx.x as f32 * opt.scale + opt.origin_x,
                vx.y as f32 * opt.scale + opt.origin_y,
                u,
                v],
        }
    }

    let mut vertices = Vec::new();
    for l in &path.loops {
        for seg in &l.segments {
            vertices.push(translate(l.start, 0.0, 1.0, &opt));
            vertices.push(translate(seg.from, 0.0, 1.0, &opt));
            vertices.push(translate(seg.to, 0.0, 1.0, &opt));
            vertices.push(translate(seg.from, 0.0, 0.0, &opt));
            vertices.push(translate(seg.c, 0.5, 0.0, &opt));
            vertices.push(translate(seg.to, 1.0, 1.0, &opt));
        }
    }

    let mut curves = Vec::new();
    for l in &path.loops {
        for seg in &l.segments {
            curves.push(SdfCurve {
                from_to: [
                    seg.from.x as f32 * opt.scale + opt.origin_x,
                    seg.from.y as f32 * opt.scale + opt.origin_y,
                    seg.to.x as f32 * opt.scale + opt.origin_x,
                    seg.to.y as f32 * opt.scale + opt.origin_y,
                ],
                control: [
                    seg.c.x as f32 * opt.scale + opt.origin_x,
                    seg.c.y as f32 * opt.scale + opt.origin_y,
                    1.0,
                    1.0,
                ],
            });
        }
    }
    let num_curves = curves.len() as u32;
    for _ in 0..(WORKGROUP_Z - num_curves % WORKGROUP_Z) {
        curves.push(SdfCurve {
            from_to: [0.0f32, 0.0, 0.0, 0.0],
            control: [0.0f32, 0.0, 0.0, 0.0],
        });
    }

    if rd.is_some() {
        println!("found RD!");
    }

    let mut results = unsafe {
        main_unsafe(&opt, rd, &vertices, &curves).expect("error in unsafe main")
    };

    let mut out_normal = image::RgbaImage::new(opt.width, opt.height);

    // fix pixels close to edge
    let xy_scale = opt.grad_scale.max(0.0).min(1.0);
    let z_scale = (1.0 - xy_scale * xy_scale).max(0.0).sqrt();
    let inv_sdf_scale = 1.0 / opt.sdf_scale;
    for y in 0..opt.height as usize {
        for x in 0..opt.width as usize {
            let i = y * opt.width as usize + x;
            let pix = &mut results.sdf_data[i];
            if pix[3].abs() < FIX_THRESHOLD {
                let p = 1.0 / opt.scale as f64 * Vec2 {
                    x: x as f64 + 0.5 - opt.origin_x as f64,
                    y: y as f64 + 0.5 - opt.origin_y as f64,
                };
                let (normal, inside, dist) = path.calculate_true_normal(p);
                pix[0] = normal.x as f32;
                pix[1] = normal.y as f32;
                pix[2] = opt.scale * if inside { -dist } else { dist } as f32;
                pix[3] = opt.scale * dist as f32;
            }
            let grad_x = pix[0];
            let grad_y = pix[1];
            let signed_distance = pix[2];
            fn normalize(x: f32) -> u8 {
                if x >= 1.0 {
                    return 255;
                }
                if x <= 0.0 {
                    return 0;
                }
                (x * 255.0).round() as u8
            }
            fn saturate(x: f32) -> u8 {
                if x >= 255.0 {
                    return 255;
                }
                if x <= 0.0 {
                    return 0;
                }
                x.round() as u8
            }
            out_normal.put_pixel(x as u32, y as u32, image::Rgba([
                normalize(0.5 * xy_scale * grad_x + 0.5),
                normalize(0.5 * xy_scale * grad_y + 0.5),
                normalize(0.5 * z_scale + 0.5),
                saturate(inv_sdf_scale * signed_distance + 127.5),
            ]));
        }
    }

    let out_filename = opt.output.clone().unwrap_or_else(|| {
        let mut out = opt.input.clone();
        out.set_extension("png");
        out
    });

    out_normal.save_with_format(out_filename, image::ImageFormat::Png).unwrap();
}
