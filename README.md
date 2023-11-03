# Parallel Compute Lab
 
This project contains the course labs for course "Introduction to OpenCL on FPGAs" on Coursera and an additional small side project. Although the course focuses on FPGAs, the code here has been implemented and tested using a GPU device. The given lab exercises are in C++, but I have also added Rust implementations for the host language out of personal interest in Rust. The purpose of this repository is to demonstrate proficiency in OpenCL and parallel programming.

## Contents

three projects:

1. [OpenCL implementations in C++](./cpp_opencl/)
2. [OpenCL implementations in Rust](./rust_opencl/)
3. [wgpu implementations in Rust](./rust_wgpu/)


## Execution Environment

The code has been developed and tested on Windows 11 and kali-linux (WSL)

## Building and Running the Projects

### C++ 

To build and run a C++ project, follow instructions:

1. Navigate to `cpp_opencl` directory.
2. Run the `build.sh` script to build the project:
   ```bash
   ./build.sh
   ```

3. Run the `run.sh` script to execute the compiled binary:
   ```bash
   ./run.sh
   ```


### Rust

To build and run a Rust project, follow instructions:

1. Install Rust form [rustup.rs](https://rustup.rs/).
2. Navigate to `rust_opencl` or `rust_wgpu` directory.
3. Run the next command to build the project:
   ```bash
   cargo build
   ```
4. Run the next command to execute the compiled binary:
   ```bash
   cargo run --example {name "in examples folder"}
   ```


