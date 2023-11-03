
__constant sampler_t sampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP;

__kernel void rotation(__read_only image2d_t input_img,
                       __write_only image2d_t output_img, int img_width,
                       int img_height, float theta) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  // fivot point. in this code, center of image.
  float x0 = img_width / 2.0f;
  float y0 = img_height / 2.0f;

  // relative position
  int x_ = x - x0;
  int y_ = y - y0;

  float sin_theta = sin(theta);
  float cos_theta = cos(theta);

  // compute the orig location
  float2 read_coord;
  read_coord.x = x_ * cos_theta - y_ * sin_theta + x0;
  read_coord.y = x_ * sin_theta + y_ * cos_theta + y0;

  // read the pixel from input
  float value = read_imagef(input_img, sampler, read_coord).x; // dim 1.

  // write to the output
  write_imagef(output_img, (int2)(x, y), (float4)(value, 0.f, 0.f, 0.f));
}