use std::error::Error;

use image::GenericImageView;
use ocl::{flags, Buffer, Context, Device, Kernel, Platform, Program, Queue};

const HIST_BINS: usize = 256;
fn main() -> Result<(), Box<dyn Error>> {
    // load the image
    let img = image::open("data/cat.png")?;
    let (img_cols, img_rows) = img.dimensions();
    let img: Vec<i32> = img
        .grayscale()
        .into_luma8()
        .into_iter()
        .map(|v| *v as i32)
        .collect();

    let no_elements = img_cols * img_rows;
    println!("Image size : Cols({img_cols}), Rows({img_rows})");
    println!("Image bytes size : {}", img.len());

    // histogram
    let mut histogram: Vec<i32> = vec![0; HIST_BINS];

    // initialize host-side program.
    let platform = Platform::default();
    let device = Device::first(platform)?;
    let context = Context::builder().devices(device).build()?;
    let queue = Queue::new(&context, device, None)?;
    let source = std::fs::read_to_string("./kernels/histogram.cl")?;

    // buffers
    // buffer input image
    let buff_in = Buffer::<i32>::builder()
        .queue(queue.clone())
        .len(img.len())
        .copy_host_slice(&img)
        .build()?;

    // initialize the histogram buffer with zero
    let buff_out = Buffer::<i32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_HOST_READ_ONLY)
        .len(histogram.len())
        .fill_val(0)
        .build()?;

    // create a program
    let program = Program::builder()
        .devices(device)
        .src(source.as_str())
        .build(&context)?;

    //
    let kernel = Kernel::builder()
        .program(&program)
        .name("histogram")
        .queue(queue.clone())
        .global_work_size(1024)
        .local_work_size(64)
        .arg(&buff_in)
        .arg(&no_elements)
        .arg(&buff_out)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    buff_out.read(&mut histogram).enq()?;
    println!("Result: {histogram:?}");

    Ok(())
}
