pub mod compute {
    extern crate ocl;

    use std::ops::Index;
    use ocl::{flags, Platform, Device, Context, Queue, Program, Buffer, Kernel};
    use glium::buffer::Buffer as GLBuffer;
    use glium::GlObject;

    /// Computes next game-stage from one buffer into another buffer, does not return values or move any memory
    pub fn compute_2d_gl(
        in_buffer_id: u32,
        out_buffer_id: u32,
        stencil_buffer: &Buffer<i32>,
        context: &Context,
        device: &Device,
        queue: &Queue,
        worker_dims: usize,
    ) -> ocl::Result<()> {

        // Get OpenGL buffers by id and acquire them for use by OpenCL
        let mut in_buffer_cl = unsafe { Buffer::<u8>::from_gl_buffer(context, Some(flags::MEM_READ_ONLY), in_buffer_id)? };
        let mut out_buffer_cl = unsafe { Buffer::<u8>::from_gl_buffer(context, Some(flags::MEM_WRITE_ONLY), out_buffer_id)? };
        in_buffer_cl.set_default_queue(queue.clone());
        out_buffer_cl.set_default_queue(queue.clone());
        in_buffer_cl.cmd().gl_acquire().enq()?;
        out_buffer_cl.cmd().gl_acquire().enq()?;


        // Kernel code run on the gpu by each compute unit
        let kernel_command = r#"
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

        let program = Program::builder()
            .devices(device)
            .src(kernel_command)
            .build(&context)?;

        // Create kernel
        let kernel = Kernel::builder()
            .program(&program)
            .name("add")
            .queue(queue.clone())
            .global_work_size(worker_dims)
            .arg(&in_buffer_cl)
            .arg(&out_buffer_cl)
            .arg(stencil_buffer)
            .arg(&(stencil_buffer.len() as u32))
            .arg(&(in_buffer_cl.len() as u32))
            .build()?;

        // Run kernel on GPU with default local work size and global work offset
        unsafe {
            kernel.cmd()
                .queue(&queue)
                .global_work_offset(kernel.default_global_work_offset())
                .global_work_size(worker_dims)
                .local_work_size(kernel.default_local_work_size())
                .enq()?;
        }

        // Release buffers for OpenCL use
        in_buffer_cl.cmd().gl_release().enq()?;
        out_buffer_cl.cmd().gl_release().enq()?;

        Ok(())
    }
}