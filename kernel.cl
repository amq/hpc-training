kernel void GaussianBlur(
    global const unsigned char *input,
    global unsigned char *output,
    global const float *filter,
    const int size) {

    int width = get_global_size(0);
    int height = get_global_size(1);
    int radius = size / 2;

    int column = get_global_id(0); // x
    int row = get_global_id(1);    // y
    int idx = (column * 3) + (row * width * 3);

    if (
        column - radius < 0 ||
        column + radius >= width ||
        row - radius < 0 ||
        row + radius >= height) {

        output[idx] = input[idx];
        output[idx + 1] = input[idx + 1];
        output[idx + 2] = input[idx + 2];
        return;
    }

    int fidx = 0;
    output[idx] = 0.0;
    output[idx + 1] = 0.0;
    output[idx + 2] = 0.0;

    for (int r = -radius; r <= radius; r++) {
        int y = row + r;
        for (int c = -radius; c <= radius; c++) {
            int x = column + c;
            int byte = (x * 3) + (y * width * 3);

            output[idx] += input[byte] * filter[fidx];
            output[idx + 1] += input[byte + 1] * filter[fidx + 1];
            output[idx + 2] += input[byte + 2] * filter[fidx + 2];

            fidx++;
        }
    }
}
