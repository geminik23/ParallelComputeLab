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

#[cfg(test)]
mod tests {
    use super::generate_gaussian_kernel;

    #[test]
    pub fn test_generate_kernel() {
        dotenv::dotenv().ok();
        env_logger::init();

        let kernel_size: usize = 5;

        let kernel = generate_gaussian_kernel(kernel_size as i32 / 2, 1.0);
        assert_eq!(kernel.len(), kernel_size * kernel_size);
        assert!((1.0 - kernel.iter().sum::<f32>()).abs() < 1e-6);
    }
}
