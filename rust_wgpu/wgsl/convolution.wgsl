@group(0) @binding(0) var input_img: texture_2d<f32>;
@group(0) @binding(1) var output_img: texture_storage_2d<rgba32float,write>;
@group(0) @binding(2) var<storage, read> kernel: array<f32>;


@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // load the grobal id
    let x = f32(global_id.x);
    let y = f32(global_id.y);

    let texture_dim = textureDimensions(input_img, 0);

    let kernel_size = sqrt(f32(arrayLength(&kernel)));
    let half_kernel_size = i32(kernel_size) / 2;

    // Initialize the sum as zero
    var sum: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // filter index
    var filt_idx: i32 = 0;

    // iterate height
    for (var i = -half_kernel_size; i <= half_kernel_size; i = i + 1) {
        // iterate width
        for (var j = -half_kernel_size; j <= half_kernel_size; j = j + 1) {
            // Get the neighboring pixel coordinates
            let pixel_pos_x = x + f32(j);
            let pixel_pos_y = y + f32(i);

            // check if the position is within the texture boundaries
            // ignore the outer image.
            if pixel_pos_x >= 0.0 && pixel_pos_x < f32(texture_dim.x) && pixel_pos_y >= 0.0 && pixel_pos_y < f32(texture_dim.y) {
                let pixel_val: vec4<f32> = textureLoad(input_img, vec2<i32>(i32(pixel_pos_x), i32(pixel_pos_y)), 0);
                sum = sum + pixel_val * kernel[filt_idx];
                filt_idx = filt_idx + 1;
            }
        }
    }

    // write the result to the output texture
    textureStore(output_img, vec2<i32>(i32(x), i32(y)), sum);
}