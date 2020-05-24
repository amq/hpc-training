#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#define _VARIADIC_MAX 10
#include "filters.hpp"
#include "lodepng.h"
#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#define FILTER_SIZE 5

int main(void) {

    std::vector<unsigned char> image, output;
    unsigned width, height;
    unsigned error;

    // always returns RGBA
    error = lodepng::decode(image, width, height, "test/input.png");

    if (error) {
        std::cerr << "PNG ERROR: " << error << ": " << lodepng_error_text(error) << std::endl;
        return EXIT_FAILURE;
    }

    output.resize(image.size());

    int filterSize = FILTER_SIZE;
    std::vector<float> filter = filters::gaussian(filterSize);

    std::ifstream file("kernel.cl");
    std::stringstream source;
    if (!(source << file.rdbuf())) {
        std::cerr << "Could not load source" << std::endl;
        return EXIT_FAILURE;
    }

    cl::Program program;

    try {
        program = cl::Program(cl::STRING_CLASS(source.str()));
        program.build();

        auto GaussianBlurKernel =
            cl::make_kernel<
                cl::Buffer &,
                cl::Buffer &,
                cl::Buffer &,
                int>(program, "GaussianBlur");

        cl::Buffer inputBuffer(begin(image), end(image), true);
        cl::Buffer filterBuffer(begin(filter), end(filter), true);
        cl::Buffer outputBuffer(begin(output), end(output), false);

        GaussianBlurKernel(
            cl::EnqueueArgs(
                cl::NDRange(width, height)),
            inputBuffer,
            outputBuffer,
            filterBuffer,
            filterSize);

        cl::copy(outputBuffer, begin(output), end(output));

    } catch (cl::Error err) {
        std::cerr << "OPENCL ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;

        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
            std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault());
            std::cerr << log << std::endl;
        }

        return EXIT_FAILURE;
    }

    error = lodepng::encode("test/output.png", output, width, height);

    if (error) {
        std::cerr << "PNG ERROR: " << error << ": " << lodepng_error_text(error) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
