use ocl::{Buffer, Context, Device, Kernel, Platform, Program, Queue};
use std::error::Error;

const VECTOR_SIZE: usize = 1024;
const WORK_SIZE: usize = 256;

fn main() -> Result<(), Box<dyn Error>> {
    // generate  random
    let in1: Vec<f32> = (0..VECTOR_SIZE).map(|v| v as f32).collect();
    let in2: Vec<f32> = (0..VECTOR_SIZE).map(|v| v as f32).collect();
    let mut out: Vec<f32> = vec![0.0; VECTOR_SIZE];

    // let platform_list = Platform::list();
    // let platform = platform_list[0];
    let platform = Platform::default();
    // platform infos
    // println!("{}", platform.version()?);

    // let device_list = Device::list_all(platform)?;
    // let device = device_list[0];
    let device = Device::first(platform)?;

    // device infos
    println!("{} - {}", device.name()?, device.vendor()?);

    let context = Context::builder().devices(device).build()?;
    let queue = Queue::new(&context, device, None)?;

    let source = std::fs::read_to_string("./kernels/vecadd_kernel.cl")?;

    let buff_x = Buffer::<f32>::builder()
        .queue(queue.clone())
        .len(VECTOR_SIZE)
        .copy_host_slice(&in1)
        .build()?;

    let buff_y = Buffer::<f32>::builder()
        .queue(queue.clone())
        .len(VECTOR_SIZE)
        .copy_host_slice(&in2)
        .build()?;

    let buff_z = Buffer::<f32>::builder()
        .queue(queue.clone())
        .len(VECTOR_SIZE)
        .build()?;

    let program = Program::builder()
        .devices(device)
        .src(source.as_str())
        .build(&context)?;

    let kernel = Kernel::builder()
        .program(&program)
        .name("add_vectors")
        .queue(queue.clone())
        .global_work_size(VECTOR_SIZE)
        .local_work_size(WORK_SIZE)
        .arg(&buff_x)
        .arg(&buff_y)
        .arg(&buff_z)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    buff_z.read(&mut out).enq()?;

    println!("Result: {out:?}");

    Ok(())
}
