mod compute;
mod dtypes;

extern crate ocl;
extern crate ocl_interop;
extern crate core;
extern crate glium;

use ocl::*;
use ocl::Buffer as Buffer;

use glium::{Display, GlObject, Surface, uniform, VertexBuffer};
use glium::buffer::{ Buffer as GLBuffer, BufferType, BufferMode };
use glium::glutin::{ event_loop, window, dpi, event };
use glium::glutin::ContextBuilder;

use dtypes::dtypes::{ Vertex, Quad };

// TODO: remove temp imports
extern crate image;
use std::io::Cursor;
use glium::texture::buffer_texture::{BufferTexture, BufferTextureRef, BufferTextureType};

fn main() {
    // Init dimensions
    let dimensions: [u32; 2] = [100, 100];
    let array_len = dimensions[0] * dimensions[1];

    // Init board
    let mut array_vec: Vec<u8> = Vec::with_capacity(array_len as usize);
    for index in 0..array_len {
        if index % 100 == 0 || index % 100 == 99 {
            array_vec.push(1);
        } else {
            array_vec.push(0);
        }
    }

    // Create Glium event loop, window, and builders (including opengl context)
    let mut events_loop = event_loop::EventLoop::new();
    let wb = window::WindowBuilder::new()
        .with_inner_size(dpi::LogicalSize::new(1024.0, 768.0))
        .with_title("2d Cellular Automata")
        .with_transparent(true);
    let cb = ContextBuilder::new().with_depth_buffer(24);
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


    // TODO: remove testcode below
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
            color = vec4(0.0, 0.0, 0.0, vec4(texelFetch(tex, buffer_index))[0]);
        }
    "#;

    let program = glium::Program::from_source(&display, triangle_shader_src, fragment_shader_src, None).unwrap();


    // Main loop
    let start_time = std::time::Instant::now();
    events_loop.run(move |event, _, control_flow| {

        // todo: change frame logic to not wait the event loop, but only draw at the correct rate
        //       this will allow key-responsiveness outside of frame draw intervals
        let next_frame_time = std::time::Instant::now() + std::time::Duration::from_nanos(1000);
        *control_flow = event_loop::ControlFlow::WaitUntil(next_frame_time);

        let mut target = display.draw();
        target.clear_color_and_depth((0.1, 0.1, 0.1, 0.1), 1.0);

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
                tex: &texture_in_cycle,
                perspective: perspective,
            },
            &params
        ).unwrap();
        target.finish().unwrap();

        match event {
            event::Event::WindowEvent { event, .. } => match event {
                event::WindowEvent::CloseRequested => {
                    *control_flow = event_loop::ControlFlow::Exit;
                    return;
                },
                _ => return,
            },
            event::Event::NewEvents(cause) => match cause {
                event::StartCause::ResumeTimeReached { .. } => (),
                event::StartCause::Init => (),
                _ => return,
            },
            _ => return
        }
    });

    // TODO: remove testcode below
    for _ in 0..10 {
        let mut vec = &out_buffer.read().unwrap();

        for x in 0..vec.len() {
            if x.rem_euclid(dimensions[1] as usize) == 0 {
                print!("\n");
            }
            print!("{:^8?}", vec[x as usize]);
        }
        println!("\n");
        // Compute next step of game
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

