use std::time::{Duration, Instant};
use egui_glium::EguiGlium;
use glium::backend::Facade;
use glium::buffer::Buffer as GLBuffer;
use glium::buffer::{  BufferType, BufferMode };
use glium::glutin::event::{ElementState, KeyboardInput, VirtualKeyCode};
use glium::{Display, Frame, IndexBuffer, Program, Surface, uniform, VertexBuffer};
use glium::texture::buffer_texture::BufferTexture;
use crate::render::{Camera, Vertex};


/// Game-State objects
#[derive(PartialEq, Copy, Clone)]
pub(crate) enum LastComputed {
    IN,
    OUT,
}

/// Struct which contains methods for dimension-specific logic of the game board
pub struct GridDimensions {
    dimensions: Vec<u8>,
}

impl GridDimensions {
    pub fn new(dimensions: Vec<u8>) -> GridDimensions {
        GridDimensions {
            dimensions
        }
    }

    /// Gather the [x, y, z, w, ...] address offset for each position in which a neighbor cell would lie
    pub fn cartesian_neighbors(&self) -> Vec<Vec<i64>> {
        let dimensions = self.dimensions.len();
        if dimensions == 0 {
            return vec![vec![0]]
        }

        // The cartesian neighbors of a point in space of n-dimensions are the product of [-1, 0, 1]^n
        // Generate the factors, and pre-allocate the correct space in a vec
        let factors = (0..dimensions).map(|_| vec![-1, 0, 1]).collect::<Vec<Vec<i64>>>();
        let mut cartesian_product: Vec<Vec<i64>> = Vec::with_capacity(3_u32.pow(factors.len() as u32) as usize);

        // Push the digits in first list as individual vecs, which in this case will always be -1, 0, and 1
        cartesian_product.append(&mut vec![vec![-1], vec![0], vec![1]]);

        // Generate cartesian power by appending remaining digits to existing partial powers repeatedly
        for list in &factors[1..] {
            let mut temp_product = Vec::with_capacity(dimensions);
            for partial_product in &cartesian_product {
                for digit in list {
                    let mut tmp = partial_product.clone();
                    tmp.push(*digit);
                    temp_product.push(tmp)
                }
            }
            cartesian_product = temp_product.to_owned();
        }

        // Remove origin point of (0, 0, ...) so cells do not count as their own neighbors
        cartesian_product.into_iter().filter(|x| x != {&vec![0_i64; dimensions]}).collect::<Vec<Vec<i64>>>()
    }

    /// Gather the index offsets for a spacial GridDimension object
    ///  such that a cell's index applied to the returned values will return the indices of each of that cell's neighbors
    pub fn cartesian_neighbor_offsets(&self) -> Vec<i64> {
        let neighbors = self.cartesian_neighbors();
        let dimensions = &self.dimensions;

        // Collect the index offsets for each cell around a given point of origin
        let temp = neighbors.into_iter()
            .map(|address| {
                let mut index = 0;

                for dim in 0..dimensions.len() {

                    let mut offset = 1_i64;
                    for n in dimensions[0..dim].into_iter() { offset *= (*n) as i64 }
                    index += address[dim] * offset;
                }
                index
            })
            .collect::<Vec<i64>>();

        temp
    }

    /// Get cell count of grid
    pub fn dimension_size(&self) -> u64 {
        let mut size: u64 = 1;
        for dim in self.dimensions.as_slice() {
            size *= *dim as u64
        }
        size
    }

    // TODO v2.0: Uncomment function and add test cases when bitwise logic is added
    // /// Get size of grid in bytes required to store the data
    // pub fn dimension_byte_size(&self) -> u64 {
    //     let mut size: u64 = 1;
    //     for dim in self.dimensions {
    //         size *= *dim as u64
    //     }
    //     (size as f64 / 8.0).ceil() as u64
    // }

    // Trio of basic getter functions
    pub fn x(&self) -> u8 {
        self.dimensions[0]
    }
    pub fn y(&self) -> u8 {
        self.dimensions[1]
    }
    pub fn z(&self) -> u8 {
        self.dimensions[2]
    }

    /// Generate in and out OpenGL buffers with capacity to hold the cell data for our dimensions
    pub fn generate_grid_buffers<T>(&self, display: &T, starting_vec: Option<Vec<u8>>) -> Result<(GLBuffer<[u8]>, GLBuffer<[u8]>), &str>
    where T: Facade
    {
        let in_buffer = GLBuffer::<[u8]>::empty_array(
            display,
            BufferType::ArrayBuffer,
            self.dimension_size() as usize,
            BufferMode::Persistent
        ).expect("Could not generate in_buffer");

        // If the user supplied a starting vector, ensure the size is correct for the dimension grid and write it to in_buffer
        if let Some(vec) = starting_vec {
            if vec.len() != self.dimension_size() as usize {
                return Err("Starting vec of invalid size")
            }

            in_buffer.write(&vec);
        }

        let out_buffer = GLBuffer::<[u8]>::empty_array(
            display,
            BufferType::ArrayBuffer,
            self.dimension_size() as usize,
            BufferMode::Persistent
        ).expect("Could not generate out_buffer");

        Ok((in_buffer, out_buffer))
    }

    /// Generate code for OpenCL kernel with hard-coded neighbor logic
    pub fn generate_program_string(&self, survive: Vec<u32>, spawn: Vec<u32>) -> String {
        // Get neighbor indexes
        let offsets = self.cartesian_neighbor_offsets();
        let neighbors = self.cartesian_neighbors();
        let mut result = String::new();

        // Open & name function, include parameters, open function body, declare neighbors variable
        let func_start = "
        __kernel void compute(
            __global uchar* in_buffer,
            __global uchar* out_buffer
        ) {
            uint neighbors = 0;
            ulong index = get_global_id(0);
        ";
        result += &func_start;

        // Gather code conditionally adding neighbors
        let mut dimensions = Vec::with_capacity(self.dimensions.len());
        let mut offset: u64 = 1;
        for (index, dim) in self.dimensions.iter().enumerate() {
            if index == 0 {
                // If final dimension (x dimension) then directly take remainder index
                dimensions.push(
                    format!(r#"
                    ulong dim{} = index;
                    "#,
                    index + 1,
                    )
                );
            } else {
                // Otherwise get max quotient from running tally and reduce the tally accordingly
                dimensions.push(
                 format!(r#"
                    ulong dim{} = index / {1};
                    index -= dim{0} * {1};
                    "#,
                    index + 1,
                    offset,
                    )
                );
            }

            offset *= *dim as u64;
        }

        result += &dimensions.into_iter().rev().collect::<Vec<String>>().join("");

        // Generate conditional lookups for each neighbor
        let mut neighbor_logic = Vec::with_capacity(neighbors.len());
        for (address, offset) in neighbors.into_iter().zip(offsets.iter()) {
            let mut conditions: Vec<String> = vec![];
            for (dim, num) in address.iter().enumerate() {
                // Near side neighbor logic
                if *num == -1 {
                    conditions.push(
                        format!(
                            "(dim{} != 0)",
                            dim + 1
                        )
                    );
                }
                // Far side neighbor logic
                if *num == 1 {
                    conditions.push(
                        format!(
                            "(dim{} != {})",
                            dim + 1,
                            self.dimensions[dim] - 1
                        )
                    );
                }
            }
            conditions.push(
                format!(
                    "({} + get_global_id(0) >= 0)",
                    offset
                )
            );

            neighbor_logic.push(
                format!(r#"
                    if ({}) {{
                        neighbors += in_buffer[{} + get_global_id(0)];
                    }}
                    "#,
                    conditions.join(" && "),
                    offset
                )
            );

        }
        result += &neighbor_logic.into_iter().rev().collect::<Vec<String>>().join("");

        // Generate survival rules
        if survive.len() == 1 {
            result += &format!(r#"
                if (neighbors == {}) {{
                    out_buffer[get_global_id(0)] = in_buffer[get_global_id(0)];
                    return;
                }}
                "#,
                survive[0]
            );
        } else if survive.len() > 1 {
            result += &format!(r#"
                if ({}) {{
                    out_buffer[get_global_id(0)] = in_buffer[get_global_id(0)];
                    return;
                }}
                "#,
                survive.into_iter()
                   .map(|x| format!("(neighbors == {})", x))
                   .collect::<Vec<String>>()
                   .join(" || "),
            );
        }

        // Generate spawning rules
        if spawn.len() == 1 {
            result += &format!(r#"
                if (neighbors == {}) {{
                    out_buffer[get_global_id(0)] = 1;
                    return;
                }}
                "#,
                spawn[0]
            );
        } else if spawn.len() > 1 {
            result += &format!(r#"
                if ({}) {{
                    out_buffer[get_global_id(0)] = 1;
                    return;
                }}
                "#,
                spawn.into_iter()
                   .map(|x| format!("(neighbors == {})", x))
                   .collect::<Vec<String>>()
                   .join(" || "),
            );
        }

        // Default if no other rules are met, and close function
        let func_end = r#"
            out_buffer[get_global_id(0)] = 0;
        }
        "#;
        result += func_end;

        result
    }
}

impl Clone for GridDimensions {
    fn clone(&self) -> Self {
        GridDimensions {
            dimensions: self.dimensions.clone()
        }
    }
}

/// Stores and modifies the game's gui state
#[derive(PartialEq, Copy, Clone)]
pub enum GUIState {
    Menu,
    ClearView,
}

impl GUIState {
    pub fn menu(&mut self) {
        *self = GUIState::Menu
    }
    
    pub fn clear(&mut self) {
        *self = GUIState::ClearView
    }
}

/// Stores and modifies state varI'iables for the game
pub struct GameOptions {
    fps: u8,
    sps: u8,
    tps: u8,
    spinning: bool,
    paused: bool,
}

impl GameOptions {
    /// Create GameOptions object with default values
    pub fn default() -> Self {
        GameOptions {
            fps: 165,
            sps: 20,
            tps: 165,
            spinning: false,
            paused: false,
        }
    }

    /// Create new GameOptions object from params
    pub fn new(
        frames_per_second: u8,
        steps_per_second: u8,
        ticks_per_second: u8,
        spinning: bool,
        paused: bool,
    ) -> Self {
        GameOptions {
            fps: frames_per_second,
            sps: steps_per_second,
            tps: ticks_per_second,
            spinning,
            paused,
        }
    }
}

/// Handle basic game settings like fps,
pub struct KeyBindings {
    menu: VirtualKeyCode,
    pause: VirtualKeyCode,
    step_simulation: VirtualKeyCode,
    strafe_up: VirtualKeyCode,
    strafe_left: VirtualKeyCode,
    strafe_down: VirtualKeyCode,
    strafe_right: VirtualKeyCode,
    zoom_in: VirtualKeyCode,
    zoom_out: VirtualKeyCode,
    toggle_spin: VirtualKeyCode,
}

impl KeyBindings {
    /// Instantiate with default keybindings
    pub fn default() -> Self {
        KeyBindings {
            menu: VirtualKeyCode::Escape,
            pause: VirtualKeyCode::Space,
            step_simulation: VirtualKeyCode::E,
            strafe_up: VirtualKeyCode::W,
            strafe_left: VirtualKeyCode::A,
            strafe_down: VirtualKeyCode::S,
            strafe_right: VirtualKeyCode::D,
            zoom_in: VirtualKeyCode::R,
            zoom_out: VirtualKeyCode::F,
            toggle_spin: VirtualKeyCode::Q,
        }
    }
}

/// Data backbone of GameManager, holds values for logical calculations
struct GameCache {
    event_list: ocl::EventList,
    last_frame: Instant,
    last_step: Instant,
    last_tick: Instant,
    queued_allowed_steps: u32,
}

/// Coordinates game actions through various game objects
pub struct GameManager {
    dimensions: GridDimensions,
    camera: Camera,
    settings: KeyBindings,
    window: GUIState,
    state: GameOptions,
    pub egui: EguiGlium,
    cache: GameCache,
}

impl GameManager {
    pub fn new(
        dimensions: GridDimensions,
        camera: Camera,
        settings: KeyBindings,
        window: GUIState,
        state: GameOptions,
        egui_glium: EguiGlium,
    ) -> GameManager {
        GameManager {
            dimensions,
            camera,
            settings,
            window,
            state,
            egui: egui_glium,
            cache: GameCache {
                event_list: ocl::EventList::new(),
                last_frame: Instant::now(),
                last_step: Instant::now(),
                last_tick: Instant::now(),
                queued_allowed_steps: 0,
            }
        }
    }

    /// Returns a mutable reference to game's event list
    pub fn get_event_list(&mut self) -> &mut ocl::EventList {
        &mut self.cache.event_list
    }

    /// Return whether or not the game is currently paused
    pub fn is_paused(&mut self) -> bool {
        if self.cache.queued_allowed_steps > 0 {
            self.cache.queued_allowed_steps -= 1;
            self.cache.last_step = Instant::now();
            return false;
        }

        self.state.paused
    }

    /// Return the configured tick interval in milliseconds
    fn tick_interval(&self) -> u64 {
        1000 / (self.state.tps as u64)
    }

    /// Return the configured framerate interval in milliseconds
    fn frame_interval(&self) -> u64 {
        1000 / (self.state.fps as u64)
    }

    /// Return the configured compute step interval in milliseconds
    fn step_interval(&self) -> u64 {
        1000 / (self.state.sps as u64)
    }

    /// Returns whether or not it is time to step the game again, and if so updates the previous tick time
    pub fn tick_wait_over(&mut self) -> bool {
        if Instant::now().duration_since(self.cache.last_tick).as_millis() as u64 >= self.tick_interval() {
            self.cache.last_tick = Instant::now();
            return true;
        }
        false
    }

    /// Returns whether or not it is time to draw the next frame, and if so updates the previous frame draw time
    pub fn frame_wait_over(&mut self) -> bool {
        if Instant::now().duration_since(self.cache.last_frame).as_millis() as u64 >= self.frame_interval() {
            self.cache.last_frame = Instant::now();
            return true;
        }
        false
    }

    /// Returns whether or not it is time to run the next step, and if so updates the previous step time
    pub fn step_wait_over(&mut self) -> bool {
        if Instant::now().duration_since(self.cache.last_step).as_millis() as u64 >= self.step_interval() {
            self.cache.last_step = Instant::now();
            return true;
        }
        false
    }

    /// Returns the correct wait time before the next tick should be run
    pub fn next_tick_time(&self, start_time: Instant) -> Instant {
        let elapsed_time = Instant::now().duration_since(start_time).as_millis() as u64;
        let wait_milliseconds = match self.tick_interval() >= elapsed_time {
            true => self.tick_interval() - elapsed_time,
            false => 0,
        };

        start_time + Duration::from_millis(wait_milliseconds)
    }

    /// Returns the correct wait time before the next tick should be run
    pub fn next_frame_time(&self, start_time: Instant) -> Instant {
        let elapsed_time = Instant::now().duration_since(start_time).as_millis() as u64;
        let wait_milliseconds = match self.frame_interval() >= elapsed_time {
            true => self.frame_interval() - elapsed_time,
            false => 0
        };

        start_time + Duration::from_millis(wait_milliseconds)
    }

    /// Returns the correct wait time before the next tick should be run
    pub fn next_step_time(&self, start_time: Instant) -> Instant {
        let elapsed_time = Instant::now().duration_since(start_time).as_millis() as u64;
        let wait_milliseconds = match self.step_interval() >= elapsed_time {
            true => self.step_interval() - elapsed_time,
            false => 0
        };

        start_time + Duration::from_millis(wait_milliseconds)
    }

    /// Allows an additional game step to be calculated without waiting for the step interval
    fn allow_one_step(&mut self) {
        self.cache.queued_allowed_steps += 1;
    }

    /// Compute a new camera tick, changing its position
    pub fn tick_camera(&mut self) {
        if self.state.spinning {
            self.camera.pass_rotate();
        }
        self.camera.calculate_position();
    }

    /// Handles a keypress from the main loop; changing state, camera angle, and settings as needed
    pub fn handle_keypress(&mut self, key: KeyboardInput) {
        match key {
            KeyboardInput { state, virtual_keycode, .. } => {
                if let Some(key) = virtual_keycode { match key {
                    key if key == self.settings.menu && state == ElementState::Pressed => {
                        if self.window == GUIState::Menu {
                            self.window.clear();
                        } else {
                            self.window.menu();
                        }
                    },
                    key if key == self.settings.pause && state == ElementState::Pressed => {
                        self.state.paused = !self.state.paused;
                    },
                    key if key == self.settings.step_simulation && state == ElementState::Pressed => {
                        self.allow_one_step();
                    },
                    key if key == self.settings.strafe_up => {
                        if state == ElementState::Pressed {
                            self.camera.start_strafe_up();
                        } else {
                            self.camera.end_strafe_vertical();
                        }
                    },
                    key if key == self.settings.strafe_left => {
                        if state == ElementState::Pressed {
                            self.camera.start_strafe_left();
                        } else {
                            self.camera.end_strafe_horizontal();
                        }
                    },
                    key if key == self.settings.strafe_down => {
                        if state == ElementState::Pressed {
                            self.camera.start_strafe_down();
                        } else {
                            self.camera.end_strafe_vertical();
                        }
                    },
                    key if key == self.settings.strafe_right => {
                        if state == ElementState::Pressed {
                            self.camera.start_strafe_right();
                        } else {
                            self.camera.end_strafe_horizontal();
                        }
                    },
                    key if key == self.settings.zoom_in => {
                        if state == ElementState::Pressed {
                            self.camera.start_zoom_in();
                        } else {
                            self.camera.end_zoom();
                        }
                    },
                    key if key == self.settings.zoom_out => {
                        if state == ElementState::Pressed {
                            self.camera.start_zoom_out();
                        } else {
                            self.camera.end_zoom();
                        }
                    },
                    key if key == self.settings.toggle_spin && state == ElementState::Pressed => {
                        self.state.spinning = !self.state.spinning;
                    }
                    _ => ()
                }}
            }
        }
    }

    /// Draws a new frame with current variables for all objects
    pub fn draw_frame(
        &mut self,
        display: &Display,
        program: &Program,
        vertex_buffer: &VertexBuffer<Vertex>,
        index_buffer: &IndexBuffer<u32>,
        texture_buffer: &BufferTexture<u8>,
        offset: u32,
    ) {
        let mut target = display.draw();
        target.clear_color_and_depth((0.0, 0.0, 0.0, 0.0), 1.0);

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

        // Drawing parameters, only needs to be changed for testing really
        let params = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            .. Default::default()
        };

        target.draw(
            vertex_buffer,
            index_buffer,
            &program,
            &uniform! {
                // INFO: should other scaling methods become too cumbersome, this matrix would be the correct place to scale the board's size
                model: [
                    [ 1.0, 0.0, 0.0, 0.0 ],
                    [ 0.0, 1.0, 0.0, 0.0 ],
                    [ 0.0, 0.0, 1.0, 0.0 ],
                    [ 0.0 , 0.0, 0.0, 1.0f32]
                ],
                tex: texture_buffer,
                perspective: perspective,
                view: self.camera.view_matrix(),
                tess_level_x: self.dimensions.x(),
                tess_level_y: self.dimensions.y(),
                tess_level_z: self.dimensions.z(),
                offset: offset,
            },
            &params
        ).unwrap();

        if self.window == GUIState::Menu {
            self.draw_menu(display, &mut target);
        }

        target.finish().unwrap();
    }

    /// Draws game menu and modifies state based on input
    fn draw_menu(&mut self, display: &Display, target: &mut Frame) {
        let _ = self.egui.run(&display, |egui_ctx| {
            // The layout, logic, and variables of the menu gui as the user sees it
            egui::SidePanel::left("menu").default_width(50.0).show(egui_ctx, |ui| {
                // Display options
                ui.label("Camera ticks per second");
                let tps_slider = egui::widgets::Slider::new(&mut self.state.tps, 60..=165);
                ui.add(tps_slider);

                ui.label("Frames per second");
                let fps_slider = egui::widgets::Slider::new(&mut self.state.fps, 30..=165);
                ui.add(fps_slider);

                ui.label("Board steps per second");
                let sps_slider = egui::widgets::Slider::new(&mut self.state.sps, 1..=60);
                ui.add(sps_slider);

                // TODO: v1.1: Allow altering game logic, and recompiling the compute kernel dynamically
                // Past this point all changes require a recompile of some part of the game

                // TODO v1.2: Allow altering starting cells and grid size
                // Past this point all changes require restarting the game completely

                // TODO v1.3: Allow windows behavior changes via the menu
                // Past this point all changes require restarting the entire program
            });
        });

        self.egui.paint(display, target);
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::hash::Hash;
    use std::fmt::Debug;
    use glium::glutin::ContextBuilder;
    use glium::glutin::dpi::PhysicalSize;
    use glium::glutin::event_loop::EventLoopBuilder;
    use glium::glutin::platform::windows::EventLoopBuilderExtWindows;

    // Helper functions
    /// Removes some busywork around checking vec equivalency
    fn assert_vec_eq<T>(case: Vec<T>, condition: Vec<T>)
        where T: Eq + Hash + Debug
    {
        let hash1 = HashSet::<&T>::from_iter(case.iter());
        let hash2 = HashSet::<&T>::from_iter(condition.iter());
        assert_eq!(hash1, hash2)
    }

    /// This function avoids incorrect vec equivalency checks by converting supported vectors into a hashset instead
    fn assert_nested_vec_eq<T>(case: Vec<Vec<T>>, condition: Vec<Vec<T>>)
    where T: Eq + Hash + Debug
    {
        let hash1 = HashSet::<&[T]>::from_iter(case.iter().map(|x| x.as_slice()));
        let hash2 = HashSet::<&[T]>::from_iter(condition.iter().map(|x| x.as_slice()));
        assert_eq!(hash1, hash2)
    }

    // Test GridDimensions
    #[test]
    fn test_cartesian_neighbors() {
        let cases = [
            vec![0, 0],
            vec![10, 10],
            vec![255, 255, 255],
        ];

        let conditions = [
            vec![vec![-1, -1], vec![-1, 0], vec![-1, 1], vec![0, -1], vec![0, 1], vec![1, -1], vec![1, 0], vec![1, 1]],
            vec![vec![-1, -1], vec![-1, 0], vec![-1, 1], vec![0, -1], vec![0, 1], vec![1, -1], vec![1, 0], vec![1, 1]],
            vec![vec![0, -1, -1], vec![0, -1, 0], vec![-1, 0, -1], vec![0, -1, 1], vec![1, -1, 1], vec![-1, 1, 1], vec![1, 0, -1], vec![-1, 0, 1], vec![1, 0, 0], vec![1, 0, 1], vec![1, 1, -1], vec![0, 0, -1], vec![0, 1, -1], vec![0, 0, 1], vec![0, 1, 0], vec![1, -1, 0], vec![-1, -1, 1], vec![1, -1, -1], vec![-1, -1, 0], vec![0, 1, 1], vec![1, 1, 1], vec![1, 1, 0], vec![-1, -1, -1], vec![-1, 0, 0], vec![-1, 1, -1], vec![-1, 1, 0]],
        ];

        for test in cases.into_iter().zip(conditions.into_iter()) {
            let dims = test.0;
            let dimensions = GridDimensions::new(dims);

            let neighbors = dimensions.cartesian_neighbors();

            assert_nested_vec_eq(neighbors, test.1);
        }
    }

    #[test]
    fn test_cartesian_neighbor_indexes() {
        let cases = [
            vec![5, 5],
            vec![10, 10],
            vec![255, 255, 255],
        ];

        let conditions = [
            vec![-6, -1, 4, -5, 5, -4, 1, 6],
            vec![-11, -1, 9, -10, 10, -9, 1, 11],
            vec![-65281, -256, 64769, -65026, -1, 65024, -64771, 254, 65279, -65280, -255, 64770, -65025, 65025, -64770, 255, 65280, -65279, -254, 64771, -65024, 1, 65026, -64769, 256, 65281],
        ];

        for test in cases.into_iter().zip(conditions.into_iter()) {
            let dims = test.0;
            let dimensions = GridDimensions::new(dims);

            let neighbors = dimensions.cartesian_neighbor_offsets();

            assert_vec_eq(neighbors, test.1);
        }
    }

    #[test]
    fn test_dim_size() {
        let cases = [
            vec![0, 0],
            vec![10, 10],
            vec![255, 255, 255],
        ];

        let conditions = [
            0,
            100,
            16_581_375,
        ];

        for test in cases.into_iter().zip(conditions.into_iter()) {
            let dimensions = GridDimensions::new(test.0);

            assert_eq!(test.1, dimensions.dimension_size());
        }
    }

    #[test]
    fn test_board_buffers_size() {
        let cases = [
            vec![5, 5],
            vec![10, 10],
            vec![255, 255, 255],
            vec![10, 10, 10, 10],
        ];

        let conditions = [
            25,
            100,
            16_581_375,
            10_000,
        ];

        // I don't normally document test cases as it is most often the purist form of code duplication...
        //  but do not, under any circumstance, change this initialization code. The test code that the glium library itself-
        //  -uses isn't available to library users and importing glutin directly for dev-dependencies doesn't work

        // This case will most likely only pass on Windows, Linux doesn't allow the screwy architecture created below
        let evl = EventLoopBuilder::new().with_any_thread(true).build();
        let context = ContextBuilder::new().build_headless(&evl, PhysicalSize::new(1000, 1000)).unwrap();
        let display = glium::HeadlessRenderer::new(context).unwrap();

        for test in cases.into_iter().zip(conditions.into_iter()) {
            let dimensions = GridDimensions::new(test.0);
            let buffers = dimensions.generate_grid_buffers(&display, None).unwrap();
            assert_eq!(buffers.0.len(), buffers.1.len());
            assert_eq!(buffers.0.len(), test.1);
        }
    }

    // TODO v1.3: revise this test
    #[test]
    fn test_create_compute_shader() {
        let cases = [
            vec![100, 100],
        ];

        let conditions = [
            include_str!("./test/compute_programs/compute_100x100.cl"),
        ];


        for test in cases.into_iter().zip(conditions.into_iter()) {
            let dimensions = GridDimensions::new(test.0);
            let mut program = dimensions.generate_program_string(
                vec![2],
                vec![3]
            );

            assert_eq!(
                program.retain(|x| !x.is_whitespace()),
                String::from(test.1).retain(|x| !x.is_whitespace()));
        }
    }
}
