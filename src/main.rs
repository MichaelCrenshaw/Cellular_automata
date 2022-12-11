mod compute;
mod render;
mod game_objects;

extern crate ocl;
extern crate ocl_interop;
extern crate glium;
extern crate core;

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
    // Game settings
    let survive_rules = vec![3, 4];
    let spawn_rules = vec![5];

    // TODO: Limit dimension total size to not go past buffer texture limit
    // Init game objects
    let dimensions = GridDimensions::new(vec![20, 20, 20]);
    let mut camera = Camera::default();
    let mut bindings = KeyBindings::default();
    let mut window_state = GUIState::Menu;
    let mut game_state = GameOptions::default();

    let mut manager = GameManager::new(
        dimensions.clone(),
        camera,
        bindings,
        window_state,
        game_state,
    );

    // TODO: Optimize this to only count and add live cells
    // TODO: Add user-defined input for starting the simulation
    // Init board
    let array_len = &dimensions.dimension_size();
    let mut array_vec: Vec<u8> = Vec::with_capacity(*array_len as usize);
    for index in 0..*array_len {
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
    let mut computed_buffer_flag = LastComputed::IN;
    let mut last_frame_time = Instant::now();


    events_loop.run(move |event, _, control_flow| {
        let start_time = Instant::now();

        // Use last computed buffer as texture input
        let texture_buffer = {
            if LastComputed::IN == computed_buffer_flag {
                &texture_in_cycle
            } else {
                &texture_out_cycle
            }
        };

        match event {
            event::Event::WindowEvent { event, .. } => match event {
                // When resizing the window, redraw frame immediately
                event::WindowEvent::Resized(_) => {
                    manager.draw_frame(
                        &display,
                        &program,
                        &vertex_buffer,
                        &indices,
                        &texture_buffer,
                        0u32
                    );
                    return;
                },
                event::WindowEvent::KeyboardInput { device_id: _, input, is_synthetic: _ } => {
                     &manager.handle_keypress(input);
                    return;
                }
                event::WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                    return;
                },
                _ => return,
            },
            event::Event::NewEvents(cause) => match cause {
                // If event is the loop's refresh interval expiring, calculate game step
                event::StartCause::ResumeTimeReached { .. } => {}
                event::StartCause::Init => (),
                _ => return,
            },
            _ => return
        }

        // Compute next game step at step interval if game isn't paused
        if manager.step_wait_over() {
            computed_buffer_flag = compute_game_state(
                &in_cycle_kernel,
                &out_cycle_kernel,
                &queue,
                computed_buffer_flag,
                &in_buffer_cl,
                &out_buffer_cl,
            );
        }

        // Draw new frame at framerate interval
        if manager.frame_wait_over() {
            manager.draw_frame(
                &display,
                &program,
                &vertex_buffer,
                &indices,
                &texture_buffer,
                0u32
            );
        }

        // Wait until next tick to run loop (barring keyboard events etc)
        *control_flow = ControlFlow::WaitUntil(manager.next_tick_time(start_time));
        last_frame_time = Instant::now()
    });
}

