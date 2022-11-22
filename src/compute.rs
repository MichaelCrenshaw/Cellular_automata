pub(crate) mod compute {
    extern crate ocl;

    use std::ops::Index;
    use ocl::{flags, Platform, Device, Context, Queue, Program, Buffer, Kernel};

    /// Returns a buffer containing computed 2d grid
    pub(crate) fn compute_2d(array_vec: &Vec<u8>, dimensions: &[u32; 2]) -> ocl::Result<(Buffer<u8>, Buffer<u8>)> {
        let src = r#"
            __kernel void add(
                __global uchar* in_buffer,
                __global uchar* out_buffer,
                __global int* stencil_buffer,
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

                // out_buffer[get_global_id(0)] = neighbors;
                // return;

                if (neighbors == 2) {
                    out_buffer[get_global_id(0)] = in_buffer[get_global_id(0)];
                    return;
                }

                if (neighbors == 3) {
                    out_buffer[get_global_id(0)] = 1;
                    return;
                }

                out_buffer[get_global_id(0)] = 0;
            }"#;

        let stencil: [[i32; 2]; 8] = [
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [1, -1],
            [1, 0],
            [1, 1],
            [0, -1],
            [0, 1],
        ];

        let platform = Platform::default();
        let device = Device::first(platform)?;
        let context = Context::builder()
            .platform(platform)
            .devices(device.clone())
            .build()?;

        let program = Program::builder()
            .devices(device)
            .src(src)
            .build(&context)?;

        let queue = Queue::new(&context, device, None)?;
        let dims = array_vec.len();

        let stencil_buffer = Buffer::<i32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .len(stencil.len())
            .fill_val(0)
            .build()?;

        let mut stencil_vec: Vec<i32> = Vec::with_capacity(stencil.len());

        for s in stencil.iter() {
            let mut index: i32 = s[0];
            for (d, z) in dimensions.iter().enumerate() {
                if d == 0 {
                    continue;
                }

                let mut offset = 1i32;
                for x in dimensions[0..d].into_iter() { offset *= *x as i32 }
                index += s[d] * offset;
            }
            stencil_vec.push(index);
        }

        unsafe {
            stencil_buffer.write(&stencil_vec).enq().unwrap();
        }

        let mut in_buffer = Buffer::<u8>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_WRITE_ONLY)
            .len(dims)
            .build()?;

        unsafe {
            in_buffer.write(array_vec).enq().unwrap();
        }

        let out_buffer = Buffer::<u8>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_WRITE_ONLY)
            .len(dims)
            .fill_val(0)
            .build()?;

        let kernel = Kernel::builder()
            .program(&program)
            .name("add")
            .queue(queue.clone())
            .global_work_size(dims)
            .arg(&in_buffer)
            .arg(&out_buffer)
            .arg(&stencil_buffer)
            .arg(&(stencil.len() as u32))
            .arg(&dimensions[0] * dimensions[1])
            .build()?;

        unsafe {
            kernel.cmd()
                .queue(&queue)
                .global_work_offset(kernel.default_global_work_offset())
                .global_work_size(dims)
                .local_work_size(kernel.default_local_work_size())
                .enq()?;
        }

        Ok((in_buffer, out_buffer))
    }
}
