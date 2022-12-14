use glium::*;
use glium::index::PrimitiveType;


/// Rendering objects
#[derive(Copy, Clone)]
pub struct Vertex {
    pub(crate) position: [f32; 3],
    pub(crate) tex_coords: [f32; 3],
}

implement_vertex!(Vertex, position, tex_coords);

pub trait Bufferable {
    fn get_vertex_buffer(&self, display: &Display) -> VertexBuffer<Vertex>;
    fn get_index_buffer(&self, display: &Display) -> IndexBuffer<u32>;
}


pub struct SpacedCubeVertexGrid {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl SpacedCubeVertexGrid {
    pub fn default() -> Self {
        SpacedCubeVertexGrid {
            vertices: vec![Vertex {
                position: [0.0, 0.0, 0.0],
                tex_coords: [0.0, 0.0, 0.0],
            }],
            indices: vec![0],
        }
    }

    pub fn new(dims: &[u8; 3]) -> Self {
        let mut verts = vec![];
        let mut indices = vec![];

        let num_cells: u64 = dims.into_iter().fold(1, | res, x| res * *x as u64);

        // Generate the positions of all vertices required to display a 3d slice of the given dimension array
        for x in 0..num_cells {
            // Get place in each dimension by backtracking
            let mut tally = x.clone();
            let z = tally / (dims[1] as u64 * dims[0] as u64);
            tally -= z * (dims[1] as u64 * dims[0] as u64);
            let y = tally / dims[0] as u64;
            tally -= y * dims[0] as u64;

            let tex_coords = [tally as f32 / dims[0] as f32, y as f32 / dims[1] as f32, z as f32 / dims[2] as f32];
            // Center the vertices around 0.0 0.0 0.0
            let position = tex_coords.into_iter()
                .map(|position| position - 0.5)
                .collect::<Vec<f32>>();

            verts.push(Vertex {
                position: [position[0], position[1], position[2]],
                tex_coords,
            });
            indices.push(x as u32);
        }

        SpacedCubeVertexGrid {
            vertices: verts,
            indices
        }
    }
}

impl Bufferable for SpacedCubeVertexGrid {
    fn get_vertex_buffer(&self, display: &Display) -> VertexBuffer<Vertex> {
        VertexBuffer::new(display, self.vertices.as_slice()).unwrap()
    }

    fn get_index_buffer(&self, display: &Display) -> IndexBuffer<u32> {
        IndexBuffer::new(display, PrimitiveType::Points, self.indices.as_slice()).unwrap()
    }
}

/// Camera object, apply changes to this object and read the view matrix as uniform for each frame.
pub struct Camera {
    altitude: f32,
    azimuth: f32,
    direction: [f32; 3],
    up: [f32; 3],
    speed: f32,
    scale: f32,
    friction: f32,
    active_direction: [i8; 3],
    inertia_direction: [f32; 3],
}

impl Camera {
    /// Mainly for testing, get camera at a generic position and
    pub fn default() -> Self {
        Camera {
            altitude: 0.0,
            azimuth: 0.0,
            direction: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            speed: 0.025,
            scale: 1.5,
            friction: 0.05,
            active_direction: [0, 0, 0],
            inertia_direction: [0.0, 0.0, 0.0],
        }
    }

    /// Creates a new Camera object
    pub fn new(
        altitude: f32,
        azimuth: f32,
        direction: [f32; 3],
        up: [f32; 3],
        speed: f32,
        scale: f32,
        friction: f32,
        active_direction: [i8; 3],
        inertia_direction: [f32; 3],
    ) -> Self {
        Camera {
            altitude,
            azimuth,
            direction,
            up,
            speed,
            scale,
            friction,
            active_direction,
            inertia_direction,
        }
    }

    /// Converts camera's spherical coordinates to cartesian [x, y, z] coordinates
    fn as_cartesian(&self) -> [f32; 3] {
        let radius = self.scale;
        let theta = self.altitude;
        let phi = self.azimuth;
        let x = radius * phi.sin() * theta.cos();
        let y = radius * phi.sin() * theta.sin();
        let z = radius * phi.cos();
        [x, y, z]
    }

    /// Calculates next camera position based on active directions, smoothing logic, friction, and existing inertia
    pub fn calculate_position(&mut self) -> &mut Self {
        // Handle camera inertia
        for (direction, inertia) in self.active_direction.iter_mut().zip(self.inertia_direction.iter_mut()) {
            // If user is moving camera in this axis, set inertia to max
            if *direction != 0 {
                *inertia = *direction as f32;
                continue;
            }

            // Increment camera inertia towards 0
            if inertia.abs() < self.friction {
                *inertia = 0.0;
            } else {
                *inertia = inertia.signum() * (inertia.abs() - self.friction);
            }
        }
        self.azimuth += self.inertia_direction[0] * self.speed;
        self.altitude += self.inertia_direction[1] * self.speed;
        self.scale += self.inertia_direction[2] * self.speed;

        self
    }

    /// Begins to move camera right along sphere's logical surface
    pub fn start_strafe_right(&mut self) {
        self.active_direction[0] = -1;
    }

    /// Begins to move camera left along sphere's logical surface
    pub fn start_strafe_left(&mut self) {
        self.active_direction[0] = 1;
    }

    /// Begins to move camera up along sphere's logical surface
    pub fn start_strafe_up(&mut self) {
        self.active_direction[1] = 1;
    }

    /// Begins to move camera down along sphere's logical surface
    pub fn start_strafe_down(&mut self) {
        self.active_direction[1] = -1;
    }

    /// Begins to scale the grid's size up
    pub fn start_zoom_in(&mut self) {
        self.active_direction[2] = -1;
    }

    /// Begins to scale the grid's size down
    pub fn start_zoom_out(&mut self) {
        self.active_direction[2] = 1;
    }

    /// Begins to move camera down along sphere's logical surface
    pub fn end_strafe_horizontal(&mut self) {
        self.active_direction[0] = 0;
    }

    /// Begins to move camera right along sphere's logical surface
    pub fn end_strafe_vertical(&mut self) {
        self.active_direction[1] = 0;
    }

    /// Mark current zoom direction as none
    pub fn end_zoom(&mut self) {
        self.active_direction[2] = 0;
    }

    /// Perform a single rotation step on the camera
    pub fn pass_rotate(&mut self) {
        self.azimuth += self.speed / 5.0;
    }

    /// Centers camera for viewing the board, meant for 2d boards
    pub fn center(&mut self) {
        self.altitude = 0.0;
        self.azimuth = 0.0;
        self.scale = 1.0;
    }

    /// Get view matrix based on the camera's current position and direction
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        let position = self.as_cartesian();
        let direction = self.as_cartesian().into_iter().map(|x| -x).collect::<Vec<f32>>();
        let up = self.up;

        // Normalize forward direction
        let fwd = {
            let f = direction;
            let len = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
            let len = len.sqrt();
            [f[0] / len, f[1] / len, f[2] / len]
        };

        // Generate plane from up and forward
        let c_plane = [up[1] * fwd[2] - up[2] * fwd[1],
            up[2] * fwd[0] - up[0] * fwd[2],
            up[0] * fwd[1] - up[1] * fwd[0]];

        // Generate normal of camera plane (faces right by convention)
        let right = {
            let len = c_plane[0] * c_plane[0] + c_plane[1] * c_plane[1] + c_plane[2] * c_plane[2];
            let len = len.sqrt();
            [c_plane[0] / len, c_plane[1] / len, c_plane[2] / len]
        };

        // Normalized and transformed up direction
        let up_norm = [fwd[1] * right[2] - fwd[2] * right[1],
            fwd[2] * right[0] - fwd[0] * right[2],
            fwd[0] * right[1] - fwd[1] * right[0]];

        // Normalized and transformed position
        let pos = [-position[0] * right[0] - position[1] * right[1] - position[2] * right[2],
            -position[0] * up_norm[0] - position[1] * up_norm[1] - position[2] * up_norm[2],
            -position[0] * fwd[0] - position[1] * fwd[1] - position[2] * fwd[2]];

        // View matrix
        [
            [right[0], up_norm[0], fwd[0], self.direction[0]],
            [right[1], up_norm[1], fwd[1], self.direction[1]],
            [right[2], up_norm[2], fwd[2], self.direction[2]],
            [pos[0], pos[1], pos[2], 1.0],
        ]
    }
}
