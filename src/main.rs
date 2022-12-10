mod compute;
mod render;
mod game_objects;

extern crate ocl;
extern crate ocl_interop;
extern crate glium;

use std::time::{Duration, Instant};
use ocl::*;
use ocl::Buffer as Buffer;

use compute::*;
use render::*;
use game_objects::*;
use glium::{CapabilitiesSource, Display, GlObject, PolygonMode, Surface, uniform};
use glium::backend::Facade;
use glium::glutin::{ event_loop, window, dpi, event };
use glium::glutin::ContextBuilder;
use glium::glutin::event_loop::ControlFlow;
use glium::texture::buffer_texture::{BufferTexture, BufferTextureType};


fn main() {
    // TODO: Limit dimension total size to not go past buffer texture limit
    // Init dimensions
    let dimensions = GridDimensions::new(&[20, 20, 20]);
    let array_len = dimensions.dimension_size();

    // Game settings
    let target_fps = 60;
    let survive_rules = vec![3, 4];
    let spawn_rules = vec![5];

    // Init camera and settings
    let mut camera = Camera::default();
    camera.pass_rotate();
    if dimensions.z() == 1 {
        camera.center();
    }

    // TODO: Optimize this to only count and add live cells
    // TODO: Add user-defined input for starting the simulation
    // Init board
    let mut array_vec: Vec<u8> = Vec::with_capacity(array_len as usize);
    for index in 0..array_len {
        if index % 11 == 0 || index % 12 == 1 || index % 10 == 9 {
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

    // Init ocl objects
    let kernel_source = &dimensions.generate_program_string(
        survive_rules,
        spawn_rules
    );
    let platform = Platform::default();
    let device = Device::first(platform).expect("No valid OpenCL device found");
    let queue = Queue::new(&context, device, Some(CommandQueueProperties::new().out_of_order())).unwrap();
    let program = create_program(
        &context,
        device,
        Some(kernel_source),
        Some("-cl-no-signed-zeros -cl-fast-relaxed-math -cl-mad-enable -cl-strict-aliasing"));

    // TODO: Fix work size to never overflow, and always batch at high-efficiency
    let worker_dims = array_vec.len();

    // TODO: When using these buffers as TextureBuffers inevitably becomes both too slow and too cumbersome, look into BufferType::UniformBuffer
    // Create OpenGL buffers, which will be used for each game-update; then swapped to compute the next stage
    let buffers = dimensions.generate_grid_buffers(&display, Some(array_vec)).unwrap();
    let in_buffer = buffers.0;
    let out_buffer = buffers.1;

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
        .build()
        .expect("Could not create in kernel from builder");

    let out_cycle_kernel = Kernel::builder()
        .program(&program)
        .name("compute")
        .queue(queue.clone())
        .global_work_size(worker_dims)
        .arg(&out_buffer_cl)
        .arg(&in_buffer_cl)
        .build()
        .expect("Could not create out kernel from builder");

    let board: &dyn Bufferable = &SpacedCubeVertexGrid::new(&[dimensions.x(), dimensions.y(), dimensions.z()]);

    // TODO: Offload both of these to a simplified buffer and basic instancing (with vertices added by compute kernels...)
    let vertex_buffer = board.get_vertex_buffer(&display);
    let indices = board.get_index_buffer(&display);

    // INFO: To get past the buffer texture length limit I'll need to offload an enormous amount of logic to compute shaders,
    //        creating vertices during the compute stage and giving them explicit values that don't require a buffer lookup in other shaders
    // BufferTexture initialization
    let texture_in_cycle: BufferTexture<u8> = BufferTexture::from_buffer(&display, in_buffer, BufferTextureType::Unsigned).unwrap();
    let texture_out_cycle: BufferTexture<u8> = BufferTexture::from_buffer(&display, out_buffer, BufferTextureType::Unsigned).unwrap();

    // Init Glium shaders and program
    // let triangle_shader_src = include_str!("./shaders/vertex_shader.glsl");
    let geometry_shader = include_str!("./shaders/generate_cubes.geom");
    let triangle_shader_src = include_str!("./shaders/vertex_shader.glsl");
    let fragment_shader_src = include_str!("./shaders/fragment_shader.glsl");
    let program = glium::Program::new(
        &display,
        glium::program::SourceCode {
            vertex_shader: triangle_shader_src,
            tessellation_control_shader: None,
            tessellation_evaluation_shader: None,
            geometry_shader: Some(geometry_shader),
            fragment_shader: fragment_shader_src,
        }
    ).unwrap();

    // Main loop
    let mut resume = false;
    let mut last_toggle = Instant::now();

    let mut computed_buffer_flag = LastComputed::IN;
    let mut last_frame_time = Instant::now();


    events_loop.run(move |event, _, control_flow| {
        let start_time = Instant::now();
        let mut no_wait = false;

        match event {
            event::Event::WindowEvent { event, .. } => match event {
                event::WindowEvent::Resized(_) => {
                    no_wait = true;
                },
                event::WindowEvent::KeyboardInput { device_id: _, input, is_synthetic: _ } => match input.virtual_keycode {
                    // If key is tab, calculate game step
                    Some(event::VirtualKeyCode::Tab) => {
                        // Add delay to toggling state, so repeating keys don't screw with the user
                        if last_toggle + Duration::from_millis(250) > Instant::now() {return}
                        resume = !resume;
                        last_toggle = Instant::now();
                    },
                    _ => return,
                }
                event::WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                    return;
                },
                _ => return,
            },
            event::Event::NewEvents(cause) => match cause {
                // If event is the loop's refresh interval expiring, calculate game step
                event::StartCause::ResumeTimeReached { .. } => {
                    if !resume {
                        return
                    }
                    computed_buffer_flag = compute_game_state(
                        &in_cycle_kernel,
                        &out_cycle_kernel,
                        &queue,
                        computed_buffer_flag,
                        &in_buffer_cl,
                        &out_buffer_cl,
                    )
                },
                event::StartCause::Init => (),
                _ => return,
            },
            _ => return
        }

        // If refresh interval interrupts the program, redraw the frame and step the game once
        let mut target = display.draw();
        target.clear_color_and_depth((0.0, 0.0, 0.0, 0.0), 1.0);

        // TODO: Add camera controls, GUI, and dimension controls
        // Magic matrix that handles incredibly complex perspective transformations for me
        let perspective = {
            let (width, height) = target.get_dimensions();
            let aspect_ratio = height as f32 / width as f32;

            let fov: f32 = std::f32::consts::PI / 3.0;
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

        // Basic depth parameters
        let params = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            // polygon_mode: PolygonMode::Line,
            .. Default::default()
        };

        // Use last computed buffer as input for fragment shader
        let texture_buffer = {
            if LastComputed::IN == computed_buffer_flag {
                &texture_in_cycle
            } else {
                &texture_out_cycle
            }
        };

        // Draw new frame with uniforms
        target.draw(
            &vertex_buffer,
            &indices,
            &program,
            &uniform! {
                model: [
                    [ 1.0, 0.0, 0.0, 0.0 ],
                    [ 0.0, 1.0, 0.0, 0.0 ],
                    [ 0.0, 0.0, 1.0, 0.0 ],
                    [ 0.0 , 0.0, 0.0, 1.0f32]
                ],
                tex: texture_buffer,
                perspective: perspective,
                view: camera.view_matrix(),
                tess_level_x: dimensions.x(),
                tess_level_y: dimensions.y(),
                tess_level_z: dimensions.z(),
                offset: 0u32,
            },
            &params
        ).unwrap();
        target.finish().unwrap();

        // Don't wait to redraw if event flags nowait
        if no_wait { return }

        // Wait until next frame redraw should occur
        let elapsed_time = Instant::now().duration_since(start_time).as_millis() as u64;
        let wait_milliseconds = match 1000 / target_fps >= elapsed_time {
            true => 1000 / target_fps - elapsed_time,
            false => 0
        };

        camera.pass_rotate();
        let next_interval = start_time + Duration::from_millis(wait_milliseconds);
        *control_flow = ControlFlow::WaitUntil(next_interval);
        last_frame_time = Instant::now()
    });
}

