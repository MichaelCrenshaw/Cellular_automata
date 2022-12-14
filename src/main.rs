mod compute;
mod render;
mod game_objects;

extern crate ocl;
extern crate ocl_interop;
extern crate glium;
extern crate egui;
extern crate egui_winit;
extern crate egui_glium;

use std::time::Instant;
use ocl::*;
use ocl::Buffer as Buffer;

use compute::*;
use render::*;
use game_objects::*;
use glium::{Display, GlObject};
use glium::glutin::{ event_loop, window, dpi, event };
use glium::glutin::ContextBuilder;
use glium::glutin::event_loop::ControlFlow;
use glium::texture::buffer_texture::{BufferTexture, BufferTextureType};


fn main() {
    // Game settings
    let survive_rules = vec![3, 4];
    let spawn_rules = vec![5];

    // TODO v1.2: Limit board size, enable dynamic resizing
    // Init game objects
    let dimensions = GridDimensions::new(vec![20, 20, 20]);
    let camera = Camera::default();
    let bindings = KeyBindings::default();
    let window_state = GUIState::ClearView;
    let game_state = GameOptions::default();


    // TODO v1.2: Move to game manager with dynamic rules, and/or typed positions
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
    let cb = ContextBuilder::new().with_depth_buffer(24).with_vsync(true);
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
    let queue = Queue::new(&context, device, Some(CommandQueueProperties::new())).unwrap();
    let program = create_program(
        &context,
        device,
        Some(kernel_source),
        Some("-cl-no-signed-zeros -cl-fast-relaxed-math -cl-mad-enable -cl-strict-aliasing"));

    // TODO v1.1: Move to game manager and set upper limit to work size
    let worker_dims = array_vec.len();

    // INFO: While buffers aren't generally memory-capped, a texture buffer has hardware-dependant memory caps;
    //        The only way around this will be changing how these buffers are read in the future
    // Create OpenGL buffers, which will be used for each game-update; then swapped to compute the next stage
    let buffers = dimensions.generate_grid_buffers(&display, Some(array_vec)).unwrap();
    let in_buffer = buffers.0;
    let out_buffer = buffers.1;

    // TODO v1.1: Move all of this to the game manager, and set up dynamic recompiling
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

    // TODO v1.4: Offload to a smaller vertex buffer containing only cells in the current view, to bypass tex-buffer size restrictions
    //        Doing this will require much more complex compute code, but allows much more memory for the board itself 
    let vertex_buffer = board.get_vertex_buffer(&display);
    let indices = board.get_index_buffer(&display);

    // INFO: To get past the buffer texture length limit I'll need to offload an enormous amount of logic to compute shaders,
    //        creating vertices during the compute stage and giving them explicit values that don't require a buffer lookup in other shaders
    // BufferTexture initialization
    let texture_in_cycle: BufferTexture<u8> = BufferTexture::from_buffer(&display, in_buffer, BufferTextureType::Unsigned).unwrap();
    let texture_out_cycle: BufferTexture<u8> = BufferTexture::from_buffer(&display, out_buffer, BufferTextureType::Unsigned).unwrap();

    // Init Glium shaders and program
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

    // Game Loop variables
    let mut computed_buffer_flag = LastComputed::IN;
    let mut manager = GameManager::new(
        dimensions.clone(),
        camera,
        bindings,
        window_state,
        game_state,
        egui_glium::EguiGlium::new(&display, &events_loop),
    );


    // Main loop
    events_loop.run(move |event, _, control_flow| {
        let start_time = Instant::now();

        // TODO v1.1: Move to game manager, it's just messy here
        // Use last computed buffer as texture input
        let texture_buffer = {
            if LastComputed::IN == computed_buffer_flag {
                &texture_in_cycle
            } else {
                &texture_out_cycle
            }
        };

        // TODO v1.0: Allow traversing through other dimension-slices via the offset variable
        // Draw new frame at framerate interval, located here to play nicely with egui's antics
        if manager.frame_wait_over() {
            manager.draw_frame(
                &display,
                &program,
                &vertex_buffer,
                &indices,
                &texture_buffer,
                0u32,
            );
            // Wait until next frame tick to run loop (barring keyboard events etc)
            *control_flow = ControlFlow::WaitUntil(manager.next_frame_time(start_time));
        }

        match event {
            event::Event::WindowEvent { event, .. } => {
                // Send events to our gui handler
                let response = &manager.egui.on_event(&event);
                if response.repaint { display.gl_window().window().request_redraw(); };

                // If the gui didn't notice any changes from our event, send it to game manager
                if response.consumed { return; }
                match event {
                    // When resizing the window, redraw frame immediately
                    event::WindowEvent::Resized(_) => {
                        manager.draw_frame(
                            &display,
                            &program,
                            &vertex_buffer,
                            &indices,
                            &texture_buffer,
                            0u32,
                        );
                        return;
                    },
                    event::WindowEvent::KeyboardInput { device_id: _, input, is_synthetic: _ } => {
                        manager.handle_keypress(input);
                        return;
                    }
                    event::WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                        return;
                    },
                    _ => return,

                }
            },
            event::Event::NewEvents(cause) => match cause {
                // If event is the loop's refresh interval expiring, calculate game step
                event::StartCause::ResumeTimeReached { .. } => {}
                event::StartCause::Init => (),
                _ => return,
            },
            _ => return
        }

        // Compute next game step at step interval
        if manager.step_wait_over() {
            // Compute if not paused
            if !manager.is_paused() {
                computed_buffer_flag = compute_game_state(
                    &in_cycle_kernel,
                    &out_cycle_kernel,
                    &queue,
                    computed_buffer_flag,
                    &in_buffer_cl,
                    &out_buffer_cl,
                    manager.get_event_list()
                );
            }
            // Wait until next compute tick to run loop (barring keyboard events etc)
            *control_flow = ControlFlow::WaitUntil(manager.next_step_time(start_time));
        }

        // Run camera tick at tick interval
        if manager.tick_wait_over() {
            manager.tick_camera();
            // Wait until next frame tick to run loop (barring keyboard events etc)
            *control_flow = ControlFlow::WaitUntil(manager.next_tick_time(start_time));
        }
    });
}

