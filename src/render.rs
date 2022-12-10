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


pub struct Camera {
    position: [f32; 3],
    direction: [f32; 3],
    up: [f32; 3],
    speed: f32,
    friction: f32,
    movement_direction: [f32; 3],
}

/// Camera object, apply changes to this object and read the view matrix as uniform for each frame.
impl Camera {
    /// Mainly for testing, get camera at a generic position and
    pub fn default() -> Self {
        Camera {
            position: [0.0, 0.0, 0.0],
            direction: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            speed: 0.005,
            friction: 0.001,
            movement_direction: [0.0, 0.0, 0.0],
        }
    }

    /// Creates a new Camera object
    pub fn new(
        position: [f32; 3],
        direction: [f32; 3],
        up: [f32; 3],
        speed: f32,
        friction: f32,
        movement_direction: [f32; 3],
    ) -> Self {
        Camera {
            position,
            direction,
            up,
            speed,
            friction,
            movement_direction,
        }
    }

    /// Perform a passive rotation on the camera
    pub fn pass_rotate(&mut self) {
        self.friction += 0.006;

        let x = f32::sin(self.friction) * 2.0;
        let z = f32::cos(self.friction) * 2.0;

        let y = f32::sin((x + z) / 150.0) * 50.0;

        self.position[0] = x;
        self.position[1] = y;
        self.position[2] = z;
    }

    /// Centers camera for viewing the board, optimized for 2d boards
    pub fn center(&mut self) {
        self.position[0] = 0.0;
        self.position[1] = 0.0;
        self.position[2] = 0.5;
    }

    /// Get view matrix based on the camera's current position and direction
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {

        let position = self.position;
        let direction = self.position.into_iter().map(|x| -x).collect::<Vec<f32>>();
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
