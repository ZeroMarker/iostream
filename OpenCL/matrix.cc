#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <random>

int main() {
    // 设置OpenCL环境
    cl::Context context;
    std::vector<cl::Device> devices;
    cl::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform defaultPlatform = platforms.front();
    defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    context = cl::Context(devices);

    // 创建命令队列
    cl::CommandQueue queue(context, devices[0]);

    // 读取内核源码
    std::ifstream kernelFile("vector_add.cl");
    std::string kernelSource(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(kernelSource.c_str(), kernelSource.length() + 1));

    // 创建程序并构建
    cl::Program program(context, source);
    program.build(devices);

    // 创建内核
    cl::Kernel kernel(program, "vector_add");

    // 设置输入数据
    const int DATA_SIZE = 1024;
    std::vector<float> input_x(DATA_SIZE, 1.0f);
    std::vector<float> input_y(DATA_SIZE, 2.0f);
    std::vector<float> output(DATA_SIZE, 0.0f);

    // 创建内存对象
    cl::Buffer mem_object_x(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * DATA_SIZE, input_x.data());
    cl::Buffer mem_object_y(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * DATA_SIZE, input_y.data());
    cl::Buffer mem_object_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * DATA_SIZE);

    // 设置内核参数
    kernel.setArg(0, mem_object_x);
    kernel.setArg(1, mem_object_y);
    kernel.setArg(2, mem_object_output);

    // 执行内核
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(DATA_SIZE), cl::NDRange(1));

    // 读取结果
    queue.enqueueReadBuffer(mem_object_output, CL_TRUE, 0, sizeof(float) * DATA_SIZE, output.data());

    // 输出结果
    for (int i = 0; i < DATA_SIZE; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}