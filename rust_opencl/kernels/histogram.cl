#define HIST_BINS 256

__kernel void histogram(__global int *data, int num_data,
                        __global int *histogram) {
  __local int local_histogram[HIST_BINS];
  int lid = get_local_id(0);
  int gid = get_global_id(0);

  // initialize the local histogram
  for (int i = lid; i < HIST_BINS; i += get_local_size(0)) {
    local_histogram[i] = 0;
  }

  // wait until all work-items within the work-group have completed
  barrier(CLK_LOCAL_MEM_FENCE);

  // compute the local histogram
  for (int i = gid; i < num_data; i += get_global_size(0)) {
    atomic_add(&local_histogram[data[i]], 1);
  }

  // wait until all work-items within the work-group have completed
  barrier(CLK_LOCAL_MEM_FENCE);

  // write the local histogram to global one.
  for (int i = lid; i < HIST_BINS; i += get_local_size(0)) {
    atomic_add(&histogram[i], local_histogram[i]);
  }
}
