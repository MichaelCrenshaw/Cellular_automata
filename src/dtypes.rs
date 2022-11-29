pub mod dtypes {
    use std::fmt::{Debug, Formatter};
    use glium::*;

    /// Game-State objects
    #[derive(PartialEq, Copy, Clone)]
    #[allow(unused)]
    pub(crate) enum LastComputed {
        IN,
        OUT,
    }

    // TODO: Entire struct should be revised due to new compute structure, possibly unneeded;
    //       prefer a "Dimension" struct to combine stencil and bitboard helper functions
    struct BitBoard2D<'a> {
        data: Vec<u8>,
        dims: &'a [u8]
    }

    impl BitBoard2D<'_> {
        pub fn new(dims: &[u8]) -> Box<BitBoard2D<'_>> {
            let board = (0..(dims[0] * dims[1])).into_iter().map(|_| 0b_0000_0000).collect::<Vec<u8>>();
            Box::new(BitBoard2D {
                data: board,
                dims,
            })
        }

        pub fn len(&self) -> u32 {
            self.data.len() as u32
        }

        pub fn bitwise_len(&self) -> usize {
            (self.data.len() * 8) as usize
        }

        pub fn get_byte(&self, address: &[u8; 2]) -> Option<&u8> {
            let index = address[0] + (address[1] * self.dims[0]);

            match self.data.get::<usize>(index as usize) {
                None => {None}
                Some(x) => {Some(x as &u8)}
            }
        }

        pub fn get(&self, address: &[u8; 2]) -> Option<bool> {
            let u8_group: &u8 = self.get_byte(&[(address[0] / 8), (address[1] / 8)])?;

            let rem = (address[0] + (address[1] * self.dims[0])) % 8;

            let mask: u8 = 0b_0000_0001 << rem;

            Some((mask & u8_group) != 0b_0000_0000)
        }

        pub fn set_byte(&mut self, address: &[u8; 2], value: u8) -> Result<(), &str> {
            let index = address[0] + (address[1] * self.dims[0]);
            if index as u32 > self.len() {
                return Err("Out of index range");
            }

            self.data[index as usize] = value;

            Ok(())
        }

        pub fn set(&mut self, address: &[u8; 2], value: bool) -> Result<(), &str> {
            // validity of address is only checked once, in set_byte
            // as such this function assumes a valid index until the very end
            let byte_index = &[(address[0] / 8), (address[1] / 8)];

            let mut byte = match self.get_byte(byte_index) {
                None => { return Err("Address is out of index") }
                Some(b) => {b.to_owned()}
            };

            let rem = (address[0] + (address[1] * self.dims[0])) % 8;

            if value {
                byte &= !(0b_1000_0000 >> rem)
            } else {
                byte |= 0b_1000_0000 >> rem
            }

            self.set_byte(byte_index, byte)?;

            Ok(())
        }
    }

    impl Debug for BitBoard2D<'_> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            let mut out = String::new();
            // print header
            out += &*format!("{:^8}", '+');
            for x in 0..(self.dims[0]*8) {out += &*format!("{:^8}", x)}

            for y in 0..(self.dims[0]) as u8 {
                out += "\r\n";
                out += &*format!("{:^8}", y);
                for x in 0..(self.dims[1]*8) as u8 {
                    println!("{}, {}", x, y);
                    out += &*format!("{:^8}", self.get(&[x, y]).unwrap())
                }
            }

            f.write_str(&*out).expect("Debug print failed");
            Ok(())
        }
    }

    /// Struct which contains methods for dimension-specific logic of the game board
    pub struct GridDimsensions<'a> {
        dimensions: &'a [u8],
    }

    impl GridDimsensions<'_> {
        pub fn new(dimensions: &[u8]) -> GridDimsensions {
            GridDimsensions {
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
            let dim_count = dimensions.len();

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

        // TODO: when bitwise logic is added, add function to generate corrected size in bytes
        // Get cell count of grid
        pub fn dimension_size(&self) -> u64 {
            let mut size: u64 = 1;
            for dim in self.dimensions {
                size *= *dim as u64
            }
            size
        }
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
        positions: [[f32; 3]; 4],
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
                positions,
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
        use std::ptr::eq;

        fn assert_vec_eq<T>(case: Vec<T>, condition: Vec<T>)
            where T: Eq + Hash + Debug
        {
            let hash1 = HashSet::<&T>::from_iter(case.iter());
            let hash2 = HashSet::<&T>::from_iter(condition.iter());
            assert_eq!(hash1, hash2)
        }

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
                let dimensions = GridDimsensions::new(dims.as_slice());

                let neighbors = dimensions.cartesian_neighbors();
                
                assert_nested_vec_eq(neighbors, test.1)

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
                let dimensions = GridDimsensions::new(dims.as_slice());

                let neighbors = dimensions.cartesian_neighbor_offsets();

                assert_vec_eq(neighbors, test.1)

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
                let dimensions = GridDimsensions::new(test.0.as_slice());

                assert_eq!(test.1, dimensions.dimension_size())
            }
        }

        // Test 2d board
        #[test]
        fn test_board_inst() {
            let board = BitBoard2D::new(&[6, 10]);

            assert_eq!(board.data.len(), 60);
            assert_eq!(board.dims, [6, 10]);
        }

        #[test]
        fn test_board_set_read() {
            let mut board = BitBoard2D::new(&[10, 10]);

            let mut cases: Vec<([u8; 2], u8)> = vec![
                ([0, 0], 0),
                ([9, 9], 0),
                ([9, 0], 0),
                ([0, 9], 0),
                // ensure bytes can be overwritten
                ([9, 9], 0b_1111_1111),
            ];

            for (x, y) in &cases {
                match board.set_byte(x, *y) {
                    Err(_) => {panic!("Attempted to set to out of index")},
                    Ok(_) => {}
                }
            }

            // remove overwritten byte
            cases.remove(1);
            for (x, y) in &cases {
                match board.get_byte(x) {
                    None => {panic!("Attempted to get from out of index")}
                    Some(b) => { if y != b {panic!("Byte does not set value")} }
                }
            }
        }

        // #[test]
        fn test_board_set_read_bitwise() {
            let mut board = BitBoard2D::new(&[10, 10]);

            let mut cases: Vec<([u8; 2], bool)> = vec![
                ([0, 0], false),
                ([79, 79], false),
                ([79, 0], false),
                ([0, 79], false),
                // ensure bits can be overwritten
                ([0, 0], true),
            ];

            for (x, y) in &cases {
                match board.set(x, *y) {
                    Err(_) => {panic!("{}", format!("Could not set bit at {:?} to {}", x, y))}
                    Ok(_) => {}
                }
            }

            // remove overwritten byte
            cases.remove(0);
            for (x, y) in &cases {
                match board.get(x) {
                    None => {panic!("{}" , format!("Attempted to get from out of index at {:?}", x))}
                    Some(b) => { if y != &b {panic!("{}", format!("Bit at {:?} not equal to {}", x, y))} }
                }
            }
        }
    }
}
