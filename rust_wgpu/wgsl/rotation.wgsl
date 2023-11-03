@group(0) @binding(0) var input_img: texture_2d<f32>;
@group(0) @binding(1) var output_img: texture_storage_2d<rgba32float,write>;
@group(0) @binding(2) var<storage, read> img_size: array<u32>;
@group(0) @binding(3) var<uniform> theta: f32;


@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var x = f32(global_id.x);
    var y = f32(global_id.y);

    // Pivot point. In this code, center of image.
    let x0 = f32(img_size[0]) / 2.0;
    let y0 = f32(img_size[1]) / 2.0;

    // Relative position
    let x_ = x - x0;
    let y_ = y - y0;

    let sin_theta = sin(theta);
    let cos_theta = cos(theta);

    // Compute the original location
    let read_coord = vec2<f32>(
        x_ * cos_theta - y_ * sin_theta + x0,
        x_ * sin_theta + y_ * cos_theta + y0
    );

    //
    // Read the pixel from input

    //
    // !! floor the position
    //
    // let coord = vec2<i32>(read_coord);
    // let value = textureLoad(input_img, coord, 0);

    // 
    // !! Bilinear interpolation
    //

    // Extract integer and fractional parts of (u, v)
    let u_int = i32(read_coord.x);
    let v_int = i32(read_coord.y);
    let u_frac = fract(read_coord.x);
    let v_frac = fract(read_coord.y);

    let texel_00 = textureLoad(input_img, vec2<i32>(u_int, v_int), 0);
    let texel_10 = textureLoad(input_img, vec2<i32>(u_int + 1, v_int), 0);
    let texel_01 = textureLoad(input_img, vec2<i32>(u_int, v_int + 1), 0);
    let texel_11 = textureLoad(input_img, vec2<i32>(u_int + 1, v_int + 1), 0);

    // compute the bilinear interpolation
    let interp_u0 = texel_00 * (1.0 - u_frac) + texel_10 * u_frac;
    let interp_u1 = texel_01 * (1.0 - u_frac) + texel_11 * u_frac;
    let value = interp_u0 * (1.0 - v_frac) + interp_u1 * v_frac;

    // Write to the output
    textureStore(output_img, vec2<i32>(i32(x), i32(y)), value);
}