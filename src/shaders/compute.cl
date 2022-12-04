__kernel void compute(
    __global uchar* in_buffer,
    __global uchar* out_buffer,
    __global long* stencil_buffer,
    uint stencil_size,
    int array_size)
{
    int neighbors = 0;

    for (int i = 0; i < stencil_size; i++) {
        int index = stencil_buffer[i] + get_global_id(0);
        if (index >= 0 && index <= array_size) {
            neighbors += in_buffer[index];
        }
    }

    if (neighbors == 2) {
        out_buffer[get_global_id(0)] = in_buffer[get_global_id(0)];
        return;
    }

    if (neighbors == 3) {
        out_buffer[get_global_id(0)] = 1;
        return;
    }

    out_buffer[get_global_id(0)] = 0;
}