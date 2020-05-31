kernel void GaussianBlur1D(
    global const float4 *input,
    global float4 *output,
    constant float *filter,
    int filter_size,
    int horizontal) {

    const int2 pos   = (int2)(get_global_id(0), get_global_id(1)); // x, y
    const int2 size  = (int2)(get_global_size(0), get_global_size(1)); // w, h
    const int radius = filter_size / 2;

    float4 pixel = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    if (horizontal) {
        for (int column = -radius, i = 0; column <= radius; column++) {
            int x = min(max(pos.x + column, 0), size.x);
            pixel += input[x + pos.y * size.x] * filter[i++];
        }
    } else {
        for (int row = -radius, i = 0; row <= radius; row++) {
            int y = min(max(pos.y + row, 0), size.y);
            pixel += input[pos.x + y * size.x] * filter[i++];
        }
    }


    output[pos.x + pos.y * size.x] = pixel;
}

kernel void GaussianBlur2D(
    global const float4 *input,
    global float4 *output,
    constant float *filter,
    int filter_size) {

    const int2 pos   = (int2)(get_global_id(0), get_global_id(1)); // x, y
    const int2 size  = (int2)(get_global_size(0), get_global_size(1)); // w, h
    const int radius = filter_size / 2;

    float4 pixel = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (int row = -radius, i = 0; row <= radius; row++) {
        int y = min(max(pos.y + row, 0), size.y);

        for (int column = -radius; column <= radius; column++) {
            int x = min(max(pos.x + column, 0), size.x);

            pixel += input[x + y * size.x] * filter[i++];
        }
    }

    output[pos.x + pos.y * size.x] = pixel;
}
