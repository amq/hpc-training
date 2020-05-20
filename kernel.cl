kernel void GaussianBlur(
    global const char *input,
    global const float *filter,
    global char *output,
    const int size,
    const int width,
    const int height) {

    int idx = 3 * get_global_id(0);
    int x = (idx / 3) % width;
    int y = (idx / 3) / width;
    int halfSize = size / 2;

    float r = 0.0;
    float g = 0.0;
    float b = 0.0;

    int i, j;
    for (j = -halfSize; j <= halfSize; j++) {
        for (i = -halfSize; i <= halfSize; i++) {
            r += input[idx] * filter[(j * size) + i];
            g += input[idx + 1] * filter[(j * size) + i];
            b += input[idx + 2] * filter[(j * size) + i];
        }
    }

    output[idx] = r;
    output[idx + 1] = g;
    output[idx + 2] = b;
}
