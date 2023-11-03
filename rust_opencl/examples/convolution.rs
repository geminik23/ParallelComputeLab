use std::error::Error;

use image::{EncodableLayout, GenericImageView};
use lab_opencl::utils::generate_gaussian_kernel;
use ocl::{
    core::{ImageDescriptor, ImageFormat},
    enums::{AddressingMode, FilterMode, ImageChannelOrder, MemObjectType},
    flags, Buffer, Context, Device, Image, Kernel, Platform, Program, Queue, Sampler,
};

const IMAGE_PATH: &str = "data/cat.png";
const IMAGE_GRAY_PATH: &str = "data/cat_gray.png";
const IMAGE_OUT_PATH: &str = "data/cat_convolution.png";

fn save_img(
    path: &str,
    out: &Vec<f32>,
    width: u32,
    height: u32,
) -> Result<(), image::error::ImageError> {
    let out = out.iter().map(|v| (v * 255.0) as u8).collect::<Vec<u8>>();
    image::save_buffer(path, out.as_bytes(), width, height, image::ColorType::L8)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    dotenv::dotenv().ok();
    env_logger::init();

    // load the image
    let img = image::open(IMAGE_PATH)?;
    let (img_cols, img_rows) = img.dimensions();
    let img: Vec<f32> = img
        .into_luma8()
        .into_iter()
        .map(|&v| v as f32 / 255.0)
        .collect();
    let no_elements = img_cols * img_rows;

    save_img(IMAGE_GRAY_PATH, &img, img_cols, img_rows)?;

    // Kernel filter
    let filter_size = 5;
    let kernel = generate_gaussian_kernel(filter_size / 2, 1.0);

    // vector for result image.
    let mut out = vec![0f32; no_elements as usize];

    log::info!("Image Loaded from ({})", IMAGE_PATH);
    log::info!("======================");
    log::info!("Image size : Width({img_cols}), Height({img_rows})");
    log::info!("Image bytes size : {}", img.len());

    // initialize host-side program.
    log::info!("Initialize the host-side program");
    let platform = Platform::default();
    let device = Device::first(platform)?;
    let context = Context::builder().devices(device).build()?;
    let queue = Queue::new(&context, device, None)?;
    let source = std::fs::read_to_string("./kernels/convolution.cl")?;

    // create the image buffer
    log::info!("Create the image buffers");
    let img_desc = ImageDescriptor::new(
        MemObjectType::Image2d,
        img_cols as usize,
        img_rows as usize,
        0,
        0,
        0,
        0,
        None,
    );

    let img_format = ImageFormat::new(
        ImageChannelOrder::R,
        ocl::enums::ImageChannelDataType::Float,
    );

    unsafe {
        let input_img = Image::<f32>::new(
            &queue,
            flags::MEM_READ_ONLY,
            img_format.clone(),
            img_desc.clone(),
            None,
        )?;

        let output_img = Image::<f32>::new(
            &queue,
            flags::MEM_WRITE_ONLY,
            img_format.clone(),
            img_desc.clone(),
            None,
        )?;

        log::info!("Copy data to input image buffer..");
        // write image
        input_img
            .write(&img)
            .region((img_cols, img_rows, 1))
            .enq()?;

        // buffer for filter
        log::info!("Create the filter buffer and sampler.");
        let filter_size = 5;
        let filter_buffer = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .len(kernel.len())
            .copy_host_slice(&kernel)
            .build()?;

        let sampler = Sampler::new(
            &context,
            false,
            AddressingMode::ClampToEdge,
            FilterMode::Nearest,
        )?;

        log::info!("Create the program");
        // create a program
        let program = Program::builder()
            .devices(device)
            .src(source.as_str())
            .build(&context)?;

        //
        let kernel = Kernel::builder()
            .program(&program)
            .name("convolution")
            .queue(queue.clone())
            .global_work_size((1500, 1500))
            .local_work_size((4, 4))
            .arg(&input_img)
            .arg(&output_img)
            .arg(&filter_buffer)
            .arg(&filter_size)
            .arg_sampler(&sampler)
            .build()?;
        kernel.enq()?;

        output_img
            .read(&mut out)
            .region((img_cols, img_rows, 1))
            .enq()?;
    }

    save_img(IMAGE_OUT_PATH, &out, img_cols, img_rows)?;
    Ok(())
}
