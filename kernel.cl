kernel void GaussianBlur1D_local(
    global const float4 *input,
    global float4 *output,
    constant float *filter,
    int diameter,
    int horizontal) {

    const int2 block = (int2)(get_group_id(0), get_group_id(1)) * 16; // x, y
    const int2 size  = (int2)(get_global_size(0), get_global_size(1)); // w, h
    const int radius = diameter / 2;

    const int2 loc   = (int2)(get_local_id(0), get_local_id(1)); // x, y
    const int offset = block.x + block.y * size.x;
    
    local float4 tmp[16][16];
    tmp[loc.x][loc.y] = input[offset + (loc.x + loc.y * size.x)];

    barrier(CLK_LOCAL_MEM_FENCE);

    float4 pixel = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    if (horizontal) {
        for (int column = -radius, i = 0; column <= radius; column++) {
            int x = min(max(loc.x + column, 0), 15);
            pixel += tmp[x][loc.y] * filter[i++];
        }
    } else {
        for (int row = -radius, i = 0; row <= radius; row++) {
            int y = min(max(loc.y + row, 0), 15);
            pixel += tmp[loc.x][y]* filter[i++];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    output[offset + (loc.x + loc.y * size.x)] = pixel;
}

kernel void GaussianBlur1D_global(
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

kernel void GaussianBlur2D_global(
    global const float4 *input,
    global float4 *output,
    constant float *filter,
    int diameter) {

    const int2 pos   = (int2)(get_global_id(0), get_global_id(1)); // x, y
    const int2 size  = (int2)(get_global_size(0), get_global_size(1)); // w, h
    const int radius = diameter / 2;

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
