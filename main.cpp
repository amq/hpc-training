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

#define INPUT "test/input.png"
#define OUTPUT "test/output.png"
#define FILTER_SIZE 5

int main(void) {

    std::vector<unsigned char> image;
    unsigned width, height;

    // always returns RGBA
    if (auto err = lodepng::decode(image, width, height, INPUT)) {
        std::cerr << "PNG ERROR: " << lodepng_error_text(err) << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<float> input(image.begin(), image.end());
    std::vector<float> output(input.size());

    int filterSize = FILTER_SIZE;
    std::vector<float> filter = filters::gaussian2d(filterSize);

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
                int>(program, "GaussianBlur2D");

        cl::Buffer inputBuffer(begin(input), end(input), true);
        cl::Buffer outputBuffer(begin(output), end(output), false);
        cl::Buffer filterBuffer(begin(filter), end(filter), true);

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

    image.assign(output.begin(), output.end());

    if (auto err = lodepng::encode(OUTPUT, image, width, height)) {
        std::cerr << "PNG ERROR: " << lodepng_error_text(err) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
