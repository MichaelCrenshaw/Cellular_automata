use std::fmt::Debug;
use glium::*;
use glium::backend::Facade;
use glium::buffer::Buffer as GLBuffer;
use ocl::Queue;

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

    /// Generate an OpenCL buffer with neighbor offset stencil as data
    pub fn generate_stencil_buffer(&self, queue: &Queue) -> ocl::Buffer<i64> {
        let stencil = self.cartesian_neighbor_offsets();

        let buffer = ocl::Buffer::<i64>::builder()
            .queue(queue.clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(stencil.len())
            .fill_val(0)
            .build()
            .expect("Could not create OpenGL stencil buffer");

        buffer.write(&stencil).enq().unwrap();
        buffer
    }

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

    // TODO: Uncomment function and add test cases when bitwise logic is added
    // /// Get size of grid in bytes required to store the data
    // pub fn dimension_byte_size(&self) -> u64 {
    //     let mut size: u64 = 1;
    //     for dim in self.dimensions {
    //         size *= *dim as u64
    //     }
    //     (size as f64 / 8.0).ceil() as u64
    // }
}

/// Rendering objects
#[derive(Copy, Clone)]
pub struct Vertex {
    pub(crate) position: [f32; 3],
    pub(crate) tex_coords: [f32; 2],
}
implement_vertex!(Vertex, position, tex_coords);

#[derive(Copy, Clone)]
pub struct Quad {
    vertices: [Vertex; 4],
    indices: [u16;6],
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
            Vertex { position: positions[0], tex_coords: [0.0, 0.0] },
            Vertex { position: positions[1], tex_coords: [1.0, 0.0] },
            Vertex { position: positions[2], tex_coords: [0.0, 1.0] },
            Vertex { position: positions[3], tex_coords: [1.0, 1.0] },
        ];

        Quad {
            vertices,
            indices: [
                0, 1, 2,
                1, 2, 3,
            ],
        }
    }

    pub fn get_vertex_buffer(&self, display: &Display) -> VertexBuffer<Vertex> {
        VertexBuffer::new(display, &self.vertices).unwrap()
    }

    pub fn get_index_buffer(&self, display: &Display) -> IndexBuffer<u16> {
        IndexBuffer::new(display, index::PrimitiveType::TrianglesList, &self.indices).unwrap()
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::hash::Hash;
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
    fn test_stencil_buffer() {
        let cases = [
            vec![5, 5],
            vec![10, 10],
            vec![255, 255, 255],
            vec![10, 10, 10, 10],
        ];

        let conditions = [
            vec![-6, -1, 4, -5, 5, -4, 1, 6],
            vec![-11, -1, 9, -10, 10, -9, 1, 11],
            vec![-65281, -256, 64769, -65026, -1, 65024, -64771, 254, 65279, -65280, -255, 64770, -65025, 65025, -64770, 255, 65280, -65279, -254, 64771, -65024, 1, 65026, -64769, 256, 65281],
            vec![10, 991, -99, -999, 901, 99, 9, 1089, 1090, 989, 1000, -991, -1110, 1110, 1109, -111, 899, 891, 111, -1090, 91, 1010, -889, 11, 1111, -1099, 100, -1009, 889, 101, -989, -9, -900, -11, -1001, -89, -901, 1101, 999, 1011, -110, -10, -891, 110, 1009, -1101, -1091, 1001, -90, 900, -899, 89, 109, -91, 90, -1100, -100, -1, 909, -1109, -890, -990, -109, 990, 911, -101, -909, 1091, -1089, 1, 1099, -911, 890, -1111, -1011, -1010, -910, -1000, 1100, 910],
        ];

        let platform = ocl::Platform::default();
        let device = ocl::Device::first(platform).expect("No valid OpenCL device found");
        let context = ocl::Context::builder()
            .platform(platform)
            .devices(device.clone())
            .build()
            .expect("Could not create OpenCL context");
        let queue = Queue::new(&context, device, None).unwrap();

        for test in cases.into_iter().zip(conditions.into_iter()) {
            let dimensions = GridDimensions::new(test.0.as_slice());

            let stencil_buffer = dimensions.generate_stencil_buffer(&queue);

            let mut stencil_vec = vec![0i64; test.1.len()];
            stencil_buffer.read(&mut stencil_vec).enq().expect("Could not read from stencil buffer");

            assert_vec_eq(stencil_vec, test.1);
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
}
