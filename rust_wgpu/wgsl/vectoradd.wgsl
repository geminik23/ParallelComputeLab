struct Buffer {
    data: array<f32>,
};

@group(0) @binding(0) var<storage, read> in_a: array<f32>;

@group(0) @binding(1) var<storage, read> in_b: array<f32>;

@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let gid = GlobalInvocationID.x;
    out[gid] = in_a[gid] + in_b[gid];
}
