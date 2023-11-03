#![allow(dead_code)]
use wgpu::{Adapter, InstanceDescriptor};

async fn run() {
    let instance = wgpu::Instance::new(InstanceDescriptor::default());

    let adapters: Vec<Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).collect();
    for adapter in adapters.into_iter() {
        let info = adapter.get_info();
        log::info!("Adapter name : {}", info.name);
        log::info!("\tdevice_type : {:?}", info.device_type);
        log::info!("\tbackend : {:?}", info.backend);
        log::info!("\tdriver : {:?}", info.driver);
        log::info!("\tdriver_info: {:?}", info.driver_info);

        let feature = adapter.features();
        log::info!("\tfeatures : {:?}", feature);
    }
}

fn main() {
    dotenv::dotenv().ok();
    env_logger::init();
    pollster::block_on(run());
}
