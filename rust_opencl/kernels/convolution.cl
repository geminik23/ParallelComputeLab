
__kernel void convolution(__read_only image2d_t input_img,
                          __write_only image2d_t output_img,
                          __constant float *filter, int filter_size,
                          sampler_t sampler) {
  //
  int column = get_global_id(0);
  int row = get_global_id(1);

  int half_filter_size = (int)(filter_size / 2);

  float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};

  // filter index
  int filt_idx = 0;
  int2 pixel_pos;

  // kernel height
  for (int i = -half_filter_size; i <= half_filter_size; ++i) {
    pixel_pos.y = row + i;

    // kernel width
    for (int j = -half_filter_size; j <= half_filter_size; ++j) {
      pixel_pos.x = column + j;

      sum.x +=
          read_imagef(input_img, sampler, pixel_pos).x * filter[filt_idx++];
    }
  }

  pixel_pos.x = column;
  pixel_pos.y = row;
  write_imagef(output_img, pixel_pos, sum);
}
