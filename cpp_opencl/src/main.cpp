// home/gemin/.local/share/nvim/swap//%home%gemin%project%ocl%main.cpp.swp/ #define CL_PLATFORM_NUMERIC_VERSION 220

#include <CL/cl_platform.h>
#include <iostream>
#include <fstream>
#include <iterator>

//  #define CL_TARGET_OPENCL_VERSION 220
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <cassert>
#include <algorithm>
#include <vector>
#include <random>

void print_device_info(const std::vector<cl::Device> &device_list);

static const cl_uint VECTOR_SIZE = 1024;
static const cl_uint WORK_SIZE = 256;

int main()
{

	size_t datasize = sizeof(float_t) * VECTOR_SIZE;

	std::vector<float_t> in1(VECTOR_SIZE);
	std::vector<float_t> in2(VECTOR_SIZE);
	std::vector<float_t> out(VECTOR_SIZE);

	// fill vectors
	for (auto i = 0; i < VECTOR_SIZE; ++i)
	{
		in1[i] = i;
		in2[i] = i;
	}

	try
	{
		// Used exceptions

		// gEt a list of platforms
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		// get a list of devices
		std::vector<cl::Device> devices;
		platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

		// print devices
		print_device_info(devices);

		// create a context for the devices
		cl::Context context(devices);

		// create a command-queue for the device.
		// one command_queue per one device. used first device.
		cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

		// create the memory objects(buffers).
		cl::Buffer buffer_a = cl::Buffer(context, CL_MEM_READ_ONLY, datasize);
		cl::Buffer buffer_b = cl::Buffer(context, CL_MEM_READ_ONLY, datasize);
		cl::Buffer buffer_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, datasize);

		// copy the input vectors into buffer objects
		// via command-queue for first device.
		queue.enqueueWriteBuffer(buffer_a, CL_TRUE, 0, datasize, &in1);
		queue.enqueueWriteBuffer(buffer_b, CL_TRUE, 0, datasize, &in2);

		// read the program source from file.
		std::ifstream source("src/vecadd_kernel.cl");
		auto begin = std::istreambuf_iterator<char>{source};
		auto end = std::istreambuf_iterator<char>{};
		std::string source_code(begin, end);
		std::cout << "Source code: " << source_code << std::endl;
		source.close();

		// create a program from source.
		cl::Program program{context, source_code};
		program.build(devices);

		// create the kernel
		cl::Kernel kernel(program, "add_vectors");

		// set the kernel arguments
		kernel.setArg(0, buffer_a);
		kernel.setArg(1, buffer_b);
		kernel.setArg(2, buffer_c);

		//
		// execute the kernel
		cl::NDRange global(VECTOR_SIZE);
		cl::NDRange local(WORK_SIZE);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
		//
		//

		// copy the output data to the host.
		queue.enqueueReadBuffer(buffer_c, CL_TRUE, 0, datasize, &out);

		// print out the result
		std::cout << "[ ";
		std::for_each(std::begin(out), std::end(out), [](auto &&v)
					  { std::cout << v << ", "; });
		std::cout << "]" << std::endl;
	}
	catch (cl::Error err)
	{
		std::cout << err.what() << " ERROR CODE (" << err.err() << ")" << std::endl;
	}

	return 0;
}

void print_device_info(const std::vector<cl::Device> &device_list)
{
	size_t num_devices = device_list.size();
	std::cout << "Num devices : " << num_devices << std::endl;

	for (size_t i = 0; i < num_devices; i++)
	{
		std::cout << "Device [" << i << "]" << std::endl;

		std::cout << "\tname: " << device_list[i].getInfo<CL_DEVICE_NAME>() << std::endl;
		std::cout << "\tavailability: " << device_list[i].getInfo<CL_DEVICE_AVAILABLE>() << std::endl;
		std::cout << "\tmax compute units: " << device_list[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
		std::cout << "\tmax work item dimensions: " << device_list[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;
		std::cout << "\tmax work group size: " << device_list[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
		std::cout << "\tmax frequency: " << device_list[i].getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
		std::cout << "\tmax mem alloc size: " << device_list[i].getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl
				  << std::endl;
	}
}
