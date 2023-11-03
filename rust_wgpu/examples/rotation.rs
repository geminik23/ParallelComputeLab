use image::{DynamicImage, EncodableLayout, GenericImageView};
use rust_wgpu::{save_img, WgpuState};
use std::error::Error;
use wgpu::{util::DeviceExt, BufferAddress, ImageDataLayout};

const IMAGE_GRAY_PATH: &str = "data/cat.png";
const IMAGE_OUT_PATH: &str = "data/cat_rotation_out.png";
async fn run(image: DynamicImage, cols: u32, rows: u32, theta: f32) -> Option<Vec<f32>> {
    let image = image.to_rgba32f();
    log::info!("Rotation - cols({cols}), rows({rows}), theta({theta:.3})");
    log::info!("bytes size is {}", image.as_bytes().len());

    let init_wgpu = WgpuState::init()
        .await
        .expect("Failed to initialize the wgpu");

    let device = &init_wgpu.device;
    let queue = &init_wgpu.queue;

    //
    // load shader
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../wgsl/rotation.wgsl").into()),
    });

    //
    // create the texture
    let texture_size = wgpu::Extent3d {
        width: cols,
        height: rows,
        depth_or_array_layers: 1,
    };

    let input_texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            size: texture_size,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            label: None,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            view_formats: &[],
        },
        &image.as_bytes(),
    );
    let input_texture_view = input_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        size: texture_size,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        label: None,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        view_formats: &[],
    });
    let output_texture_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // load buffer for size and theta
    let img_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[cols, rows]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let theta_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[theta]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE,
    });

    // create bind groups
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Image rotation Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&input_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&output_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: img_size_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: theta_buffer.as_entire_binding(),
            },
        ],
        label: None,
    });

    // create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    // create command encoder
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(150, 150, 1);
    }

    // create buffer to copy the result image from texture
    let read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (cols * rows * 4 * 4) as BufferAddress,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &output_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &read_buffer,
            layout: ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(cols * 4 * 4),
                rows_per_image: Some(rows),
            },
        },
        texture_size,
    );

    queue.submit(Some(encoder.finish()));

    // read the result buffer
    let read_buffer_slice = read_buffer.slice(..);

    let (sender, receiver) = futures_channel::oneshot::channel();

    read_buffer_slice.map_async(wgpu::MapMode::Read, |result| {
        sender.send(result).ok();
    });

    device.poll(wgpu::Maintain::Wait);
    let recv = receiver.await.expect("failed to communication");
    match recv {
        Ok(_) => {
            let data = read_buffer_slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            read_buffer.unmap();
            Some(result)
        }
        Err(err) => {
            log::error!("Buffer MapRead error - {err:?}");
            None
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    dotenv::dotenv().ok();
    env_logger::init();

    let degree = 60.0;
    let theta: f32 = std::f32::consts::PI / 180.0 * degree;

    // load the imge
    let img = image::open(IMAGE_GRAY_PATH)?;

    let width = 1488;
    let height = 1488;

    //
    // to copy texture to buffer : Bytes per row needs to respect to 'COPY_BYTES_PER_ROW_ALIGNMENT' = 256u
    // rgba (4) channel * 4 bytes(float) * 16 = 256
    let img = img.resize(width, height, image::imageops::FilterType::Triangle);
    let (img_cols, img_rows) = img.dimensions();

    // img.into_iter().cloned().collect::<Vec<f32>>();
    if let Some(out) = pollster::block_on(run(img, img_cols, img_rows, theta)) {
        save_img(IMAGE_OUT_PATH, &out, img_cols, img_rows)?;
    }
    Ok(())
}
