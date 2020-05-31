kernel void GaussianBlur(
    global const float4 *input,
    global float4 *output,
    global const float *filter,
    const int size) {

    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int width  = get_global_size(0);
    const int height = get_global_size(1);
    const int radius = size / 2;

    float4 pixel = (float4)(0.0, 0.0, 0.0, 0.0);

    for (int row = -radius; row <= radius; row++) {
        int y = min(max(pos.y + row, 0), height);
        for (int column = -radius; column <= radius; column++) {
            int x = min(max(pos.x + column, 0), width);

            pixel += input[x + y * width] * filter[(column + radius) + (row + radius) * size];
        }
    }

    output[pos.x + pos.y * width] = pixel;
}