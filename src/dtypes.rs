use glium::*;
use glium::backend::Facade;
use glium::buffer::Buffer as GLBuffer;

/// Game-State objects
#[derive(PartialEq, Copy, Clone)]
pub(crate) enum LastComputed {
    IN,
    OUT,
}

/// Struct which contains methods for dimension-specific logic of the game board
pub struct GridDimensions<'a> {
    dimensions: &'a [u8],
}

impl<'a> GridDimensions<'a> {
    pub fn new(dimensions: &[u8]) -> GridDimensions {
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
        let dimensions = self.dimensions;

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
        for dim in self.dimensions {
            size *= *dim as u64
        }
        size
    }

    // TODO: Uncomment function and add test cases when bitwise logic is added
    // /// Get size of grid in bytes required to store the data
    // pub fn dimension_byte_size(&self) -> u64 {
    //     let mut size: u64 = 1;
    //     for dim in self.dimensions {
    //         size *= *dim as u64
    //     }
    //     (size as f64 / 8.0).ceil() as u64
    // }

    /// Generate in and out OpenGL buffers with capacity to hold the cell data for our dimensions
    pub fn generate_grid_buffers<T>(&self, display: &T, starting_vec: Option<Vec<u8>>) -> Result<(GLBuffer<[u8]>, GLBuffer<[u8]>), &str>
    where T: Facade
    {
        let in_buffer = GLBuffer::<[u8]>::empty_array(
            display,
            buffer::BufferType::ArrayBuffer,
            self.dimension_size() as usize,
            buffer::BufferMode::Dynamic
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
            buffer::BufferType::ArrayBuffer,
            self.dimension_size() as usize,
            buffer::BufferMode::Dynamic
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

        // Gather the index of each dimension describing where a given cell is for each axis
        let mut dimensions = Vec::with_capacity(self.dimensions.len());
        let mut offset: u64 = 1;
        for (index, dim) in self.dimensions.iter().enumerate() {
            // If final dimension (x dimension) then directly take remainder index
            if index == 0 {
                dimensions.push(
                    format!(r#"
                    ulong dim{} = index;
                    "#,
                    index + 1,
                    )
                );
            } else {
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

/// Rendering objects
#[derive(Copy, Clone)]
pub struct Vertex {
    pub(crate) position: [f32; 3],
    pub(crate) tex_coords: [f32; 3],
}
implement_vertex!(Vertex, position, tex_coords);

pub trait Bufferable {
    fn get_vertex_buffer(&self, display: &Display) -> VertexBuffer<Vertex>;
    fn get_index_buffer(&self, display: &Display) -> IndexBuffer<u16>;
}

#[derive(Copy, Clone)]
pub struct Quad {
    vertices: [Vertex; 4],
    indices: [u16; 6],
}

impl Quad {
    pub fn new_rect(height: f32, width: f32, center: &[f32; 2]) -> Self {

        let positions = [
            [center[0] - width / 2.0, center[1] - height / 2.0, 0.0],
            [center[0] + width / 2.0, center[1] - height / 2.0, 0.0],
            [center[0] - width / 2.0, center[1] + height / 2.0, 0.0],
            [center[0] + width / 2.0, center[1] + height / 2.0, 0.0],
        ];

        let vertices = [
            Vertex { position: positions[0], tex_coords: [0.0, 0.0, 0.0] },
            Vertex { position: positions[1], tex_coords: [1.0, 0.0, 0.0] },
            Vertex { position: positions[2], tex_coords: [0.0, 1.0, 0.0] },
            Vertex { position: positions[3], tex_coords: [1.0, 1.0, 0.0] },
        ];

        Quad {
            vertices,
            indices: [
                0, 1, 2,
                1, 2, 3,
            ],
        }
    }
}

impl Bufferable for Quad {
    fn get_vertex_buffer(&self, display: &Display) -> VertexBuffer<Vertex> {
        VertexBuffer::new(display, &self.vertices).unwrap()
    }

    fn get_index_buffer(&self, display: &Display) -> IndexBuffer<u16> {
        IndexBuffer::new(display, index::PrimitiveType::TriangleStrip, &self.indices).unwrap()
    }
}

pub struct Cube {
    vertices: [Vertex; 8],
    indices: [u16; 36],
}

impl Cube {
    pub fn new(height: f32, width: f32, depth: f32, center: &[f32; 3]) -> Self  {

        let positions = [
            [center[0] - width / 2.0, center[1] - height / 2.0, center[2] - depth / 2.0],
            [center[0] + width / 2.0, center[1] - height / 2.0, center[2] - depth / 2.0],
            [center[0] - width / 2.0, center[1] + height / 2.0, center[2] - depth / 2.0],
            [center[0] + width / 2.0, center[1] + height / 2.0, center[2] - depth / 2.0],
            [center[0] - width / 2.0, center[1] - height / 2.0, center[2] + depth / 2.0],
            [center[0] + width / 2.0, center[1] - height / 2.0, center[2] + depth / 2.0],
            [center[0] - width / 2.0, center[1] + height / 2.0, center[2] + depth / 2.0],
            [center[0] + width / 2.0, center[1] + height / 2.0, center[2] + depth / 2.0],
        ];

        let vertices = [
            Vertex { position: positions[0], tex_coords: [0.0, 0.0, 0.0] },
            Vertex { position: positions[1], tex_coords: [1.0, 0.0, 0.0] },
            Vertex { position: positions[2], tex_coords: [0.0, 1.0, 0.0] },
            Vertex { position: positions[3], tex_coords: [1.0, 1.0, 0.0] },
            Vertex { position: positions[4], tex_coords: [0.0, 0.0, 1.0] },
            Vertex { position: positions[5], tex_coords: [1.0, 0.0, 1.0] },
            Vertex { position: positions[6], tex_coords: [0.0, 1.0, 1.0] },
            Vertex { position: positions[7], tex_coords: [1.0, 1.0, 1.0] },
        ];

        Cube {
            vertices,
            indices: [
                0, 1, 2,
                1, 2, 3,
                2, 0, 4,
                4, 2, 6,
                6, 4, 7,
                7, 4, 5,
                7, 3, 5,
                3, 5, 1,
                0, 4, 2,
                2, 4, 5,
                2, 3, 6,
                3, 6, 7,
            ],
        }
    }
}

impl Bufferable for Cube {
    fn get_vertex_buffer(&self, display: &Display) -> VertexBuffer<Vertex> {
        VertexBuffer::new(display, &self.vertices).unwrap()
    }

    fn get_index_buffer(&self, display: &Display) -> IndexBuffer<u16> {
        IndexBuffer::new(display, index::PrimitiveType::TriangleStrip, &self.indices).unwrap()
    }
}

pub struct Camera {
    position: [f32; 3],
    direction: [f32; 3],
    up: [f32; 3],
    speed: f32,
    friction: f32,
}

impl Camera {
    pub fn default() -> Self {
        Camera {
            position: [0.0, 0.0, 0.0],
            direction: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            speed: 0.005,
            friction: 0.001,
        }
    }

    pub fn new(position: [f32; 3], direction: [f32; 3], up: [f32; 3], speed: f32, friction: f32) -> Self {
        Camera {
            position,
            direction,
            up,
            speed,
            friction,
        }
    }

    /// Perform a passive rotation on the camera
    pub fn pass_rotate(&mut self) {
        self.friction += 0.01;
        self.position[0] = f32::sin(self.friction) * 5.0;
        self.position[2] = f32::cos(self.friction) * 5.0;
    }

    /// Get view matrix based on the camera's current position and direction
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {

        let position = self.position;
        let direction = self.position.into_iter().map(|x| -x).collect::<Vec<f32>>();
        let up = self.up;

        let f = {
            let f = direction;
            let len = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
            let len = len.sqrt();
            [f[0] / len, f[1] / len, f[2] / len]
        };

        let s = [up[1] * f[2] - up[2] * f[1],
            up[2] * f[0] - up[0] * f[2],
            up[0] * f[1] - up[1] * f[0]];

        let s_norm = {
            let len = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
            let len = len.sqrt();
            [s[0] / len, s[1] / len, s[2] / len]
        };

        let u = [f[1] * s_norm[2] - f[2] * s_norm[1],
            f[2] * s_norm[0] - f[0] * s_norm[2],
            f[0] * s_norm[1] - f[1] * s_norm[0]];

        let p = [-position[0] * s_norm[0] - position[1] * s_norm[1] - position[2] * s_norm[2],
            -position[0] * u[0] - position[1] * u[1] - position[2] * u[2],
            -position[0] * f[0] - position[1] * f[1] - position[2] * f[2]];

        [
            [s_norm[0], u[0], f[0], 0.0],
            [s_norm[1], u[1], f[1], 0.0],
            [s_norm[2], u[2], f[2], 0.0],
            [p[0], p[1], p[2], 1.0],
        ]
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
            let dimensions = GridDimensions::new(dims.as_slice());

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
            let dimensions = GridDimensions::new(dims.as_slice());

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
            let dimensions = GridDimensions::new(test.0.as_slice());

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
            let dimensions = GridDimensions::new(test.0.as_slice());
            let buffers = dimensions.generate_grid_buffers(&display, None).unwrap();
            assert_eq!(buffers.0.len(), buffers.1.len());
            assert_eq!(buffers.0.len(), test.1);
        }
    }

    // TODO: improve this test
    #[test]
    fn test_create_compute_shader() {
        let cases = [
            vec![100, 100],
        ];

        let conditions = [
            include_str!("./test/compute_programs/compute_100x100.cl"),
        ];


        for test in cases.into_iter().zip(conditions.into_iter()) {
            let dimensions = GridDimensions::new(test.0.as_slice());
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
