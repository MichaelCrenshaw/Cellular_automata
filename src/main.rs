mod compute;
mod dtypes;

extern crate ocl;
extern crate ocl_interop;
extern crate core;
extern crate glium;

use ocl::*;
use ocl::Buffer as Buffer;
use glium::glutin::{event_loop, window, dpi};
use glium::glutin::ContextBuilder;
use glium::{Display, GlObject};
use glium::buffer::{Buffer as GLBuffer, BufferType, BufferMode};

fn main() {
    // Init dimensions
    let dimensions: [u32; 2] = [25, 25];
    let array_len = dimensions[0] * dimensions[1];

    // Init board
    let mut array_vec: Vec<u8> = Vec::with_capacity(array_len as usize);
    for index in 0..array_len {
        if index % 5 == 1 || index % 7 == 0 {
            array_vec.push(1);
        } else {
            array_vec.push(0);
        }
    }

    // Create Glium eventloop, window, and builders (including opengl context)
    let mut events_loop = event_loop::EventLoop::new();
    let wb = window::WindowBuilder::new()
        .with_inner_size(dpi::LogicalSize::new(1024.0, 768.0))
        .with_title("2d Cellular Automata");
    let cb = ContextBuilder::new();
    let display = Display::new(wb, cb, &events_loop).expect("Could not create display");

    // Init ocl interop context from active opengl context
    let mut context = ocl_interop::get_context().expect("Cannot find valid OpenGL context");

    // init ocl objects
    let platform = Platform::default();
    let device = Device::first(platform).expect("No valid OpenCL device found");
    let queue = Queue::new(&context, device, None).unwrap();
    let worker_dims = array_vec.len();

    // Create OpenGL buffers, which will be used for each game-update; then swapped to compute the next stage
    let mut in_buffer = GLBuffer::<[u8]>::new(&display, &array_vec[..], BufferType::ArrayBuffer, BufferMode::Dynamic).unwrap();
    let mut out_buffer = GLBuffer::<[u8]>::new(&display, &array_vec[..], BufferType::ArrayBuffer, BufferMode::Dynamic).unwrap();

    // Create OpenCL buffer containing index offsets for each cell's neighbors
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
    let mut stencil_buffer = Buffer::<i32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY)
        .len(stencil.len())
        .fill_val(0)
        .build().unwrap();

    // Fill stencil_buffer with stencil offsets
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

    // TODO: move code above this to other functions or constants
    // Main Loop

    // main loop
    for _ in 0..10 {
        let mut vec = &out_buffer.read().unwrap();

        for x in 0..vec.len() {
            if x.rem_euclid(dimensions[1] as usize) == 0 {
                print!("\n");
            }
            print!("{:^8?}", vec[x as usize]);
        }
        println!("\n");

        compute::compute::compute_2d_gl(
            in_buffer.get_id(),
            out_buffer.get_id(),
            &stencil_buffer,
            &context,
            &device,
            &queue,
            worker_dims,
        ).unwrap();

        // Swap input and output buffers for the next gpu cycle
        (in_buffer, out_buffer) = (out_buffer, in_buffer);
    }

}

