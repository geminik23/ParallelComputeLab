use std::error::Error;

use image::{EncodableLayout, GenericImageView};
use ocl::{
    core::{ImageDescriptor, ImageFormat},
    enums::{ImageChannelOrder, MemObjectType},
    flags, Context, Device, Image, Kernel, Platform, Program, Queue,
};

const IMAGE_PATH: &str = "data/cat.png";
const IMAGE_GRAY_PATH: &str = "data/cat_gray.png";
const IMAGE_OUT_PATH: &str = "data/cat_out.png";

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

    let degree = 30.0;
    let theta: f32 = std::f32::consts::PI / 180.0 * degree;

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

    let mut out = vec![0f32; no_elements as usize];

    log::info!("Image Loaded from ({})", IMAGE_PATH);
    log::info!("======================");
    log::info!("Image size : Width({img_cols}), Height({img_rows})");
    log::info!("Image bytes size : {}", img.len());

    // histogram
    // initialize host-side program.
    log::info!("Initialize the host-side program");
    let platform = Platform::default();
    let device = Device::first(platform)?;
    let context = Context::builder().devices(device).build()?;
    let queue = Queue::new(&context, device, None)?;
    let source = std::fs::read_to_string("./kernels/rotation.cl")?;

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

        log::info!("Create the program");
        // create a program
        let program = Program::builder()
            .devices(device)
            .src(source.as_str())
            .build(&context)?;

        //
        let kernel = Kernel::builder()
            .program(&program)
            .name("rotation")
            .queue(queue.clone())
            .global_work_size((1500, 1500))
            .local_work_size((4, 4))
            .arg(&input_img)
            .arg(&output_img)
            .arg(&img_cols)
            .arg(&img_rows)
            .arg(&theta)
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
