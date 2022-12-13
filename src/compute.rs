/// All code directly relating to OpenCL, unless the code must be in main loop for various reasons
extern crate ocl;

use ocl::{Device, Context, Queue, Program, Buffer, Kernel, EventList};
use ocl::builders::KernelCmd;
use crate::game_objects::LastComputed;

// Base compute kernel, INFO: currently for 2d calculations using 1d index wrapping
pub static DEFAULT_KERNEL: &'static str = include_str!("./shaders/compute.cl");

/// Create OCL Program with provided kernel function, or default
pub fn create_program(
    context: &Context,
    device: Device,
    kernel_func: Option<&str>,
    compiler_options:Option<&str>
) -> Program {
    Program::builder()
        .devices(device)
        .cmplr_opt(if let Some(opts) = compiler_options { opts } else { "" })
        .src(if let Some(func) = kernel_func { func } else { DEFAULT_KERNEL })
        .build(context).expect("Could not create program")
}

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

/// Computes next step of automata from one buffer into another buffer, does not return values or move any memory
pub fn enqueue_kernel_command(
    kernel_command: KernelCmd,
    in_buffer: &Buffer<u8>,
    out_buffer: &Buffer<u8>,
    event_list: &mut EventList,
) -> ocl::Result<()> {

    // Get OpenGL buffers by id and acquire them for use by OpenCL
    in_buffer.cmd().gl_acquire().enq()?;
    out_buffer.cmd().gl_acquire().enq()?;

    // Schedule event to run after current events in queue have finished
    unsafe {
        kernel_command.enew(event_list).enq()?;
    }

    // Release buffers for OpenGL use
    in_buffer.cmd().gl_release().enq()?;
    out_buffer.cmd().gl_release().enq()?;

    Ok(())
}

/// Decides correct buffer cycle based on flag, creates command based on related builder, and updates flag
pub(crate) fn compute_game_state(
    in_kernel: &Kernel,
    out_kernel: &Kernel,
    queue: &Queue,
    last_computed: LastComputed,
    in_buffer_cl: &Buffer<u8>,
    out_buffer_cl: &Buffer<u8>,
    event_list: &mut EventList,
) -> LastComputed {

    if last_computed == LastComputed::IN {
        enqueue_kernel_command(
            create_kernel_command(in_kernel, queue),
            in_buffer_cl,
            out_buffer_cl,
            event_list,
        ).expect("Could not compute game state from kernel");
    } else {
        enqueue_kernel_command(
            create_kernel_command(out_kernel, queue),
            in_buffer_cl,
            out_buffer_cl,
            event_list,
        ).expect("Could not compute game state from kernel");
    }

    if last_computed == LastComputed::IN {LastComputed::OUT} else {LastComputed::IN}
}

