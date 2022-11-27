mod compute;
mod dtypes;

extern crate ocl;
extern crate ocl_interop;
extern crate core;
extern crate glium;

use ocl::*;
use ocl::Buffer as Buffer;

use compute::compute::*;
use glium::{Display, GlObject, Surface, uniform};
use glium::buffer::{ Buffer as GLBuffer, BufferType, BufferMode };
use glium::glutin::{ event_loop, window, dpi, event };
use glium::glutin::ContextBuilder;
use glium::texture::buffer_texture::{BufferTexture, BufferTextureType};

use dtypes::dtypes::{ Quad, LastComputed };

fn main() {
    // Init dimensions
    let dimensions: [u32; 2] = [100, 100];
    let array_len = dimensions[0] * dimensions[1];

    // Init board
    let mut array_vec: Vec<u8> = Vec::with_capacity(array_len as usize);
    for index in 0..array_len {
        if index % 100 == 0 || index % 11 == 3 {
            array_vec.push(1);
        } else {
            array_vec.push(0);
        }
    }

    // Create Glium event loop, window, and builders (including opengl context)
    let events_loop = event_loop::EventLoop::new();
    let wb = window::WindowBuilder::new()
        .with_inner_size(dpi::LogicalSize::new(1024.0, 768.0))
        .with_title("Cellular Automata")
        .with_transparent(true);
    let cb = ContextBuilder::new().with_depth_buffer(24);
    let display = Display::new(wb, cb, &events_loop).expect("Could not create display");

    // Init ocl interop context from active opengl context
    let context = ocl_interop::get_context().expect("Cannot find valid OpenGL context");

    // init ocl objects
    let platform = Platform::default();
    let device = Device::first(platform).expect("No valid OpenCL device found");
    let queue = Queue::new(&context, device, None).unwrap();
    let program = create_program(&context, device, None);

    // TODO: Fix work size to never overflow, and always batch at high-efficiency
    let worker_dims = array_vec.len();

    // TODO: When using these buffers as TextureBuffers inevitably becomes both too slow and too cumbersome, look into BufferType::UniformBuffer
    // Create OpenGL buffers, which will be used for each game-update; then swapped to compute the next stage
    let in_buffer = GLBuffer::<[u8]>::new(&display, &array_vec[..], BufferType::ArrayBuffer, BufferMode::Dynamic).unwrap();
    let out_buffer = GLBuffer::<[u8]>::new(&display, &array_vec[..], BufferType::ArrayBuffer, BufferMode::Dynamic).unwrap();

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
    let stencil_buffer = Buffer::<i32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY)
        .len(stencil.len())
        .fill_val(0)
        .build().unwrap();

    // Fill stencil_buffer with stencil offsets
    let mut stencil_vec: Vec<i32> = Vec::with_capacity(stencil.len());
    for s in stencil.iter() {
        let mut index: i32 = s[0];
        for d in 0..dimensions.len() {
            if d == 0 {
                continue;
            }

            let mut offset = 1i32;
            for x in dimensions[0..d].into_iter() { offset *= *x as i32 }
            index += s[d] * offset;
        }
        stencil_vec.push(index);
    }
    stencil_buffer.write(&stencil_vec).enq().unwrap();

    // TODO: move code above this to other functions or constants

    // I've tried every way I can think to make the KernelBuilder cloneable, movable, heap-allocated, or whatever works to generate this outside of main.
    // Annoyingly the library authors have clearly outdated documentation on the process, and haven't responded to other people's issues with this on GitHub.
    // If I ever figure out this dark magic then I'll write up new docs, initiate a pull request, and move the code below.
    // Create KernelBuffer to generate kernels from, rather than just reusing existing immutable kernels because OCL doesn't like that other threads "could mutate them"
    let mut in_buffer_cl = Buffer::<u8>::from_gl_buffer(
        &context,
        Some(flags::MEM_READ_WRITE),
        in_buffer.get_id()
    ).expect("Could not create in CLBuffer");

    let mut out_buffer_cl = Buffer::<u8>::from_gl_buffer(
        &context,
        Some(flags::MEM_READ_WRITE),
        out_buffer.get_id()
    ).expect("Could not create in CLBuffer");

    in_buffer_cl.set_default_queue(queue.clone());
    out_buffer_cl.set_default_queue(queue.clone());

    let in_cycle_kernel = Kernel::builder()
        .program(&program)
        .name("compute")
        .queue(queue.clone())
        .global_work_size(worker_dims)
        .arg(&in_buffer_cl)
        .arg(&out_buffer_cl)
        .arg(&stencil_buffer)
        .arg(stencil_buffer.len() as u32)
        .arg(in_buffer_cl.len() as u32)
        .build()
        .expect("Could not create out kernel from builder");

    let out_cycle_kernel = Kernel::builder()
        .program(&program)
        .name("compute")
        .queue(queue.clone())
        .global_work_size(worker_dims)
        .arg(&out_buffer_cl)
        .arg(&in_buffer_cl)
        .arg(&stencil_buffer)
        .arg(stencil_buffer.len() as u32)
        .arg(out_buffer_cl.len() as u32)
        .build()
        .expect("Could not create out kernel from builder");

    let quad = Quad::new_rect(2.0, 2.0, &[0.0f32, 0.0f32]);

    let vertex_buffer = quad.get_vertex_buffer(&display);
    let indices = quad.get_index_buffer(&display);

    // BufferTexture initialization
    let texture_in_cycle: BufferTexture<u8> = BufferTexture::from_buffer(&display, in_buffer, BufferTextureType::Unsigned).unwrap();
    let texture_out_cycle: BufferTexture<u8> = BufferTexture::from_buffer(&display, out_buffer, BufferTextureType::Unsigned).unwrap();

    let triangle_shader_src = r#"
        #version 140

        in vec3 position;
        in vec2 tex_coords;
        out vec2 v_tex_coords;

        uniform mat4 perspective;
        uniform mat4 transform_matrix;

        void main() {
            v_tex_coords = tex_coords;
            gl_Position = perspective * transform_matrix * vec4(position, 1.0);
        }
    "#;

    let fragment_shader_src = r#"
        #version 140

        in vec2 v_tex_coords;
        out vec4 color;

        uniform usamplerBuffer tex;

        void main() {
            int buffer_index = int(floor(v_tex_coords[0] * 100) + floor(v_tex_coords[1] * 100) * 100 );
            bool alive = false;
            if (texelFetch(tex, buffer_index)[0] > 0.5) alive = true;
            color = alive ? vec4(1.0, 1.0, 1.0, 1.0) : vec4(0.1, 0.1, 0.1, 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, triangle_shader_src, fragment_shader_src, None).unwrap();


    // Main loop
    let mut computed_buffer_flag = LastComputed::IN;
    events_loop.run(move |event, _, control_flow| {

        // todo: force calculations (and related) to occur even if another event is clogging the loop
        let next_frame_time = std::time::Instant::now() + std::time::Duration::from_nanos(250_000);
        *control_flow = event_loop::ControlFlow::WaitUntil(next_frame_time);

        let mut target = display.draw();
        target.clear_color_and_depth((0.1, 0.1, 0.1, 1.0), 1.0);

        let perspective = {
            let (width, height) = target.get_dimensions();
            let aspect_ratio = height as f32 / width as f32;

            let fov: f32 = 3.141592 / 3.0;
            let zfar = 1024.0;
            let znear = 0.1;

            let f = 1.0 / (fov / 2.0).tan();

            [
                [f *   aspect_ratio   ,    0.0,              0.0              ,   0.0],
                [         0.0         ,     f ,              0.0              ,   0.0],
                [         0.0         ,    0.0,  (zfar+znear)/(zfar-znear)    ,   1.0],
                [         0.0         ,    0.0, -(2.0*zfar*znear)/(zfar-znear),   0.0],
            ]
        };

        let params = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            .. Default::default()
        };

        let texture_buffer = {
            if LastComputed::IN == computed_buffer_flag {
                &texture_in_cycle
            } else {
                &texture_out_cycle
            }
        };

        target.draw(
            &vertex_buffer,
            &indices,
            &program,
            &uniform! {
                transform_matrix: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [ 0.0 , 0.0, 2.0, 1.0f32]
                ],
                tex: texture_buffer,
                perspective: perspective,
            },
            &params
        ).unwrap();
        target.finish().unwrap();

        match event {
            event::Event::WindowEvent { event, .. } => match event {
                event::WindowEvent::KeyboardInput { device_id: _, input, is_synthetic: _ } => match input.virtual_keycode {
                    Some(event::VirtualKeyCode::Tab) => {
                        computed_buffer_flag = compute_game_state(
                            &in_cycle_kernel,
                            &out_cycle_kernel,
                            &queue,
                            computed_buffer_flag,
                            &in_buffer_cl,
                            &out_buffer_cl,
                        );
                    },
                    None | Some(_) => return,
                }
                event::WindowEvent::CloseRequested => {
                    *control_flow = event_loop::ControlFlow::Exit;
                    return;
                },
                _ => return,
            },
            event::Event::NewEvents(cause) => match cause {
                event::StartCause::ResumeTimeReached { .. } => (
                    computed_buffer_flag = compute_game_state(
                        &in_cycle_kernel,
                        &out_cycle_kernel,
                        &queue,
                        computed_buffer_flag,
                        &in_buffer_cl,
                        &out_buffer_cl,
                    )
                ),
                event::StartCause::Init => (),
                _ => return,
            },
            _ => return
        }
    });



}

