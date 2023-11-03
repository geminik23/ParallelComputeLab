use wgpu::InstanceDescriptor;

use image::EncodableLayout;

pub fn save_img(
    path: &str,
    out: &Vec<f32>,
    width: u32,
    height: u32,
) -> Result<(), image::error::ImageError> {
    let out = out.iter().map(|v| (v * 255.0) as u8).collect::<Vec<u8>>();
    image::save_buffer(path, out.as_bytes(), width, height, image::ColorType::Rgba8)?;
    Ok(())
}

pub struct WgpuState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WgpuState {
    pub async fn init() -> Result<Self, wgpu::RequestDeviceError> {
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

        Ok(Self { device, queue })
    }
}

/// Generate the gaussian kernel
pub fn generate_gaussian_kernel(radius: i32, sigma: f32) -> Vec<f32> {
    let kernel_size = 2 * radius as usize + 1;
    let mut kernel = vec![0.0; kernel_size * kernel_size];
    let mut sum = 0.0;

    for y in -radius..=radius {
        for x in -radius..=radius {
            let value = (1.0 / (2.0 * std::f32::consts::PI * sigma.powi(2)))
                * (-((x.pow(2) + y.pow(2)) as f32) / (2.0 * sigma.powi(2))).exp();
            kernel[(y + radius) as usize * kernel_size + (x + radius) as usize] = value;
            sum += value;
        }
    }

    // Normalize the kernel
    for ele in kernel.iter_mut() {
        *ele /= sum;
    }

    kernel
}

/// Generate the uniform kernel
pub fn generate_uniform_kernel(filter_size: u32) -> Vec<f32> {
    let len_filter = filter_size * filter_size;
    let value = 1.0 / (len_filter as f32);
    vec![value; len_filter as usize]
}
