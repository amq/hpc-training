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
#define LOCAL_SIZE 16

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
    std::vector<float> filter = filters::gaussian1d(filterSize);

    std::ifstream file("kernel.cl");
    std::stringstream source;
    if (!(source << file.rdbuf())) {
        std::cerr << "FILE ERROR: Could not load source" << std::endl;
        return EXIT_FAILURE;
    }

    cl::Program program;

    try {
        auto device = cl::Device::getDefault();
        std::vector<::size_t> itemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

        if (itemSizes[0] < LOCAL_SIZE || itemSizes[1] < LOCAL_SIZE) {
            std::cerr << "OPENCL ERROR: Local size not supported by device" << std::endl;
            return EXIT_FAILURE;
        }

        program = cl::Program(cl::STRING_CLASS(source.str()));
        program.build();

        cl::Buffer inputBuffer(begin(input), end(input), false);
        cl::Buffer intermediaryBuffer(begin(output), end(output), false);
        cl::Buffer outputBuffer(begin(output), end(output), false);
        cl::Buffer filterBuffer(begin(filter), end(filter), true);

        auto GaussianBlur1DKernel =
            cl::make_kernel<
                cl::Buffer &,
                cl::Buffer &,
                cl::Buffer &,
                int,
                int>(program, "GaussianBlur1D_local");

        GaussianBlur1DKernel(
            cl::EnqueueArgs(
                cl::NDRange(width, height),
                cl::NDRange(LOCAL_SIZE, LOCAL_SIZE)),
            inputBuffer,
            intermediaryBuffer,
            filterBuffer,
            filterSize,
            0);

        GaussianBlur1DKernel(
            cl::EnqueueArgs(
                cl::NDRange(width, height),
                cl::NDRange(LOCAL_SIZE, LOCAL_SIZE)),
            intermediaryBuffer,
            outputBuffer,
            filterBuffer,
            filterSize,
            1);

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
