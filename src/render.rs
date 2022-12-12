use std::f32::consts;
use glium::*;
use glium::index::PrimitiveType;
use glium::glutin::event::VirtualKeyCode;


/// Normalize a vector to a length of 1 with the same proportions
fn unit_vector_reduce(vec: &Vec<f32>) -> Vec<f32> {
    let mut vec_magnitude = 0f32;
    vec.iter().for_each(|x| vec_magnitude += x.powf(2f32));
    vec_magnitude = f32::sqrt(vec_magnitude);
    vec.into_iter().map(|x| x / vec_magnitude).collect::<Vec<f32>>()
}

/// Normalize a vector, then scale the result to produce an artificial transform effect
fn scaled_vector_reduce(vec: Vec<f32>, scale: f32) -> Vec<f32> {
    let mut vec_magnitude = 0f32;
    vec.iter().for_each(|x| vec_magnitude += x.powf(2f32));
    vec_magnitude = f32::sqrt(vec_magnitude);
    vec.into_iter().map(|x| x / (vec_magnitude / scale)).collect::<Vec<f32>>()
}

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

        let num_cells: u64 = dims.into_iter().fold(1, |mut res, x| res * *x as u64);

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

// TODO: Implement smoothed camera movement using movement direction and friction
/// Camera object, apply changes to this object and read the view matrix as uniform for each frame.
pub struct Camera {
    altitude: f32,
    azimuth: f32,
    direction: [f32; 3],
    up: [f32; 3],
    speed: f32,
    scale: f32,
    friction: f32,
    movement_direction: [f32; 3],
}

impl Camera {
    /// Mainly for testing, get camera at a generic position and
    pub fn default() -> Self {
        Camera {
            altitude: 0.0,
            azimuth: 0.0,
            direction: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            speed: 0.05,
            scale: 1.0,
            friction: 0.001,
            movement_direction: [0.0, 0.0, 0.0],
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
        movement_direction: [f32; 3],
    ) -> Self {
        Camera {
            altitude,
            azimuth,
            direction,
            up,
            speed,
            scale,
            friction,
            movement_direction,
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

    /// Move camera up along sphere's logical surface
    pub fn strafe_up(&mut self) {
        self.altitude += self.speed;
    }

    /// Move camera left along sphere's logical surface
    pub fn strafe_left(&mut self) {
        self.azimuth -= self.speed;
    }

    /// Move camera down along sphere's logical surface
    pub fn strafe_down(&mut self) {
        self.altitude -= self.speed;
    }

    /// Move camera right along sphere's logical surface
    pub fn strafe_right(&mut self) {
        self.azimuth += self.speed;
    }

    /// Scale the grid's size up
    pub fn zoom_in(&mut self) {
        let new_scale = self.scale - self.speed;
        if new_scale < 1f32 { self.scale = 1f32 } else { self.scale = new_scale }
    }

    /// Scale the grid's size down
    pub fn zoom_out(&mut self) {
        self.scale += self.speed;
    }

    /// Perform a passive rotation on the camera
    pub fn pass_rotate(&mut self) {
        self.strafe_right()
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
            [right[0], up_norm[0], fwd[0], 0.0],
            [right[1], up_norm[1], fwd[1], 0.0],
            [right[2], up_norm[2], fwd[2], 0.0],
            [pos[0], pos[1], pos[2], 1.0],
        ]
    }
}
