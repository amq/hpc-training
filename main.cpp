#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#define _VARIADIC_MAX 10
#include "external/filters.h"
#include "external/tga.h"
#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

int main(void) {

    tga::TGAImage image, output;
    tga::LoadTGA(&image, "test/input.tga");

    output.imageData.resize(image.imageData.size());
    output.bpp = image.bpp;
    output.height = image.height;
    output.width = image.width;
    output.type = image.type;

    int filterSize = 5;
    std::vector<float> filter = filters::gaussian(filterSize);

    std::ifstream file("kernel.cl");
    std::stringstream source;
    source << file.rdbuf();

    cl::Program program;

    try {
        program = cl::Program(cl::STRING_CLASS(source.str()));
        program.build();

        auto GaussianBlurKernel =
            cl::make_kernel<
                cl::Buffer &,
                cl::Buffer &,
                cl::Buffer &,
                int,
                int,
                int>(program, "GaussianBlur");

        cl::Buffer inputBuffer(begin(image.imageData), end(image.imageData), true);
        cl::Buffer filterBuffer(begin(filter), end(filter), true);
        cl::Buffer outputBuffer(begin(output.imageData), end(output.imageData), false);

        GaussianBlurKernel(
            cl::EnqueueArgs(
                cl::NDRange(image.width * image.height)),
            inputBuffer,
            filterBuffer,
            outputBuffer,
            filterSize,
            image.width,
            image.height);

        cl::copy(outputBuffer, begin(output.imageData), end(output.imageData));

    } catch (cl::Error err) {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;

        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
            std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault());
            std::cerr << log << std::endl;
        }
    }

    tga::saveTGA(output, "test/output.tga");

    return EXIT_SUCCESS;
}
