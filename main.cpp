#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#define _VARIADIC_MAX 10
#include <CL/cl.hpp>
#include <iostream>
#include "external/tga.h"

int main(void) {

    tga::TGAImage image;
    tga::LoadTGA(&image, "test/input.tga");

    try {
        cl::Program GaussianBlur(
            cl::STRING_CLASS(
                "kernel void GaussianBlur(\
                    global const float *input,\
                    global float *output,\
                    const int width) {\
                        float value = input[get_global_id(0)];\
                        output[get_global_id(0)] = value;\
            }"),
            false);

        GaussianBlur.build();

        auto GaussianBlurKernel =
            cl::make_kernel<
                cl::Buffer &,
                cl::Buffer &,
                int>(GaussianBlur, "GaussianBlur");

        cl::Buffer inputBuffer(begin(image.imageData), end(image.imageData), true);
        cl::Buffer outputBuffer(begin(image.imageData), end(image.imageData), false);

        GaussianBlurKernel(
            cl::EnqueueArgs(
                cl::NDRange(image.width * image.height)),
            inputBuffer,
            outputBuffer,
            image.width);

        cl::copy(outputBuffer, begin(image.imageData), end(image.imageData));

    } catch (cl::Error err) {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")";
    }

    tga::saveTGA(image, "test/output.tga");

    return EXIT_SUCCESS;
}
