#![allow(dead_code)]
use wgpu::{util::DeviceExt, InstanceDescriptor};

async fn run(in1: Vec<f32>, in2: Vec<f32>) -> Option<Vec<f32>> {
    // initialize the instance, adapter and device.
    let instance = wgpu::Instance::new(InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                // limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .unwrap();

    //
    let vector_size = in1.len();

    // load shader
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../wgsl/vectoradd.wgsl").into()),
    });

    // init buffer
    let in1_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input A Buffer"),
        contents: bytemuck::cast_slice(&in1),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let in2_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input B Buffer"),
        contents: bytemuck::cast_slice(&in2),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let result_buffer_size: u64 = (vector_size * 4) as u64; // f32
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: result_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: result_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        label: Some("Uniform Bind Group"),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in1_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: in2_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: result_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.insert_debug_marker("compute add operation");
        compute_pass.dispatch_workgroups(256, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&result_buffer, 0, &read_buffer, 0, result_buffer_size);
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

fn main() {
    dotenv::dotenv().ok();
    env_logger::init();

    let in1: Vec<f32> = (0..1024).into_iter().map(|v| v as f32).collect();
    let in2: Vec<f32> = (1..1025).into_iter().map(|v| v as f32).collect();
    if let Some(result) = pollster::block_on(run(in1, in2)) {
        log::info!("Result is {result:?}");
    }
}
