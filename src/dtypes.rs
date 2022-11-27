pub mod dtypes {
    use std::fmt::{Debug, Formatter};
    use glium::*;

    /// Game-State objects
    #[derive(PartialEq, Copy, Clone)]
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
            println!("{}", index);

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

    // TODO: Review this struct architecture, should be altered to prepare for 3d+
    struct NeighborStencil<'a> {
        positions: Vec<&'a[i8]>,
    }

    impl<'a> NeighborStencil<'a> {
        fn len(&self) -> usize {
            self.positions.len()
        }
    }

    impl<'a> Iterator for NeighborStencil<'a> {
        type Item = &'a[i8];

        fn next(&mut self) -> Option<Self::Item> {
            todo!()
        }
    }

    #[cfg(test)]
    pub mod tests {
        use super::*;

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

            println!("{}", format!("{:?}", board));

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

            println!("{}", format!("{:?}", board));

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
}