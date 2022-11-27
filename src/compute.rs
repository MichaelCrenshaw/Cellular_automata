pub mod compute {
    extern crate ocl;

    use std::ops::Index;
    use std::rc::Rc;
    use ocl::{flags, Platform, Device, Context, Queue, Program, Buffer, Kernel};
    use glium::buffer::Buffer as GLBuffer;
    use glium::GlObject;
    use ocl::builders::{KernelBuilder, KernelCmd};

    // Base compute kernel, INFO: currently for 2d calculations using 1d index wrapping
    pub static DEFAULT_KERNEL: &'static str = r#"
            __kernel void compute(
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

    /// Computes next step of automata from one buffer into another buffer, does not return values or move any memory
    pub fn enqueue_kernel_command(
        kernel_command: KernelCmd,
        in_buffer: &Buffer<u8>,
        out_buffer: &Buffer<u8>,
    ) -> ocl::Result<()> {

        // Get OpenGL buffers by id and acquire them for use by OpenCL
        in_buffer.cmd().gl_acquire().enq()?;
        out_buffer.cmd().gl_acquire().enq()?;

        // Run kernel on GPU with default local work size and global work offset
        unsafe {
            kernel_command.enq()?;
        }

        // Release buffers for OpenGL use
        in_buffer.cmd().gl_release().enq()?;
        out_buffer.cmd().gl_release().enq()?;

        Ok(())
    }

    /// Create OCL Program with provided kernel function, or default
    pub fn create_program(context: &Context, device: Device, kernel_func: Option<&str>) -> Program {
        Program::builder()
            .devices(device)
            .src({ if let Some(func) = kernel_func { func } else { DEFAULT_KERNEL } })
            .build(context).expect("Could not create program")
    }

    // TODO: Remove this after initial commit, included in initial commit of new branch as a reminder:
    //       Check lifetimes of ALL involved structs before basing your architecture on an assumed borrow pattern
    // /// Create OCL Kernel, containing kernel code, work size defaults, and buffer references
    // pub fn create_kernel (
    //     in_buffer_id: u32,
    //     out_buffer_id: u32,
    //     stencil_buffer: Buffer<i32>,
    //     program: &Program,
    //     context: &Context,
    //     queue: Queue,
    //     worker_dims: usize,
    // ) -> KernelBuilder {
    //     // Create OpenCL buffer objects from OpenGL buffer objects (technically structs but it walks like a duck)
    //     let mut in_buffer_cl = unsafe {
    //         Buffer::<u8>::from_gl_buffer(context, Some(flags::MEM_READ_WRITE), in_buffer_id)
    //             .expect("Could not create in CLBuffer")
    //     };
    //     let mut out_buffer_cl = unsafe {
    //         Buffer::<u8>::from_gl_buffer(context, Some(flags::MEM_READ_WRITE), out_buffer_id)
    //             .expect("Could not create in CLBuffer")
    //     };
    //     in_buffer_cl.set_default_queue(queue.clone());
    //     out_buffer_cl.set_default_queue(queue.clone());
    //
    //     // TODO: remove this note-to-self
    //     //       There are no lifetimes on the final produced Kernel object, so (along with other evidence) I have very strong reason to believe:
    //     //         A. The kernel generated is essentially a struct which is formatted down to a string of OpenCL C code, and then compiled
    //     //         B. Creating two kernels, one for the input cycle and one for the output cycle (rename), will have a negligible memory overhead
    //     //         C. There should be a point somewhere in the OCL library where everything is compiled and sent to the GPU and calls to a function-
    //     //            -can simply be run without the need for any compiling or CPU-bound operations.
    //     //              TODO: Look into either offloading kernel calls to an async-only thread or simply finding said breakpoint in OCL (most likely KernelCmd.enq)
    //     // WARNING: Finally, we are throwing away the OpenCL buffer objects (more or less) above the kernel if this causes issues THIS IS WHERE TO LOOK.
    //     //          Specifically, ensure `out_buffer_cl.cmd().gl_acquire().enq()?;` is still being used before compute. (reference the async_cycles.rs example for OCL)
    //     let k = Kernel::builder()
    //         .program(program)
    //         .name("compute")
    //         .queue(queue.clone())
    //         .global_work_size(worker_dims)
    //         .arg(in_buffer_cl)
    //         .arg(out_buffer_cl)
    //         .arg(stencil_buffer)
    //         .arg(stencil_buffer.len() as u32)
    //         .arg(in_buffer_cl.len() as u32)
    //         .build()
    //         .expect("Could not create program kernel");
    // }

    /// Generate enqueue-able kernel command from kernel defaults and queue
    pub fn create_kernel_command<'a> (
        kernel: &'a Kernel,
        queue: &'a Queue,
    ) -> KernelCmd<'a> {
        kernel.cmd()
            .queue(queue)
            .global_work_offset(kernel.default_global_work_offset())
            .global_work_size(kernel.default_global_work_size())
            .local_work_size(kernel.default_local_work_size())
    }
}