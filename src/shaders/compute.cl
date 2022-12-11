__kernel void compute(
    __global uchar* in_buffer,
    __global uchar* out_buffer
) {
    uint neighbors = 0;
    ulong index = get_global_id(0);

    ulong dim2 = index / 100;
    index -= dim2 * 100;

    ulong dim1 = index;

    if ((dim1 != 99) && (dim2 != 99) && (101 + get_global_id(0) >= 0)) {
        neighbors += in_buffer[101 + get_global_id(0)];
    }

    if ((dim1 != 99) && (1 + get_global_id(0) >= 0)) {
        neighbors += in_buffer[1 + get_global_id(0)];
    }

    if ((dim1 != 99) && (dim2 != 0) && (-99 + get_global_id(0) >= 0)) {
        neighbors += in_buffer[-99 + get_global_id(0)];
    }

    if ((dim2 != 99) && (100 + get_global_id(0) >= 0)) {
        neighbors += in_buffer[100 + get_global_id(0)];
    }

    if ((dim2 != 0) && (-100 + get_global_id(0) >= 0)) {
        neighbors += in_buffer[-100 + get_global_id(0)];
    }

    if ((dim1 != 0) && (dim2 != 99) && (99 + get_global_id(0) >= 0)) {
        neighbors += in_buffer[99 + get_global_id(0)];
    }

    if ((dim1 != 0) && (-1 + get_global_id(0) >= 0)) {
        neighbors += in_buffer[-1 + get_global_id(0)];
    }

    if ((dim1 != 0) && (dim2 != 0) && (-101 + get_global_id(0) >= 0)) {
        neighbors += in_buffer[-101 + get_global_id(0)];
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
