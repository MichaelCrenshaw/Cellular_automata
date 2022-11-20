pub(crate) mod dtypes {
    use std::fmt::{Debug, Formatter};
    use std::ops::Index;

    struct BitBoard2D<'a> {
        data: Vec<u8>,
        dims: &'a [u32]
    }

    impl BitBoard2D<'_> {
        pub fn new(dims: &[u32]) -> Box<BitBoard2D<'_>> {
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

        pub fn get_byte(&self, address: &[u32; 2]) -> Option<&u8> {
            let index = address[0] + (address[1] * self.dims[0]);
            println!("{}", index);

            match self.data.get::<usize>(index as usize) {
                None => {None}
                Some(x) => {Some(x as &u8)}
            }
        }

        pub fn get(&self, address: &[u32; 2]) -> Option<bool> {
            let u8_group: &u8 = self.get_byte(&[(address[0] / 8) as u32, (address[1] / 8) as u32])?;

            let rem = (address[0] + (address[1] * self.dims[0])) % 8;

            let mask: u8 = 0b_0000_0001 << rem;

            Some((mask & u8_group) != 0b_0000_0000)
        }

        pub fn set_byte(&mut self, address: &[u32; 2], value: u8) -> Result<(), &str> {
            let index = address[0] + (address[1] * self.dims[0]);
            if index > self.len() {
                return Err("Out of index range");
            }

            self.data[index as usize] = value;

            Ok(())
        }

        pub fn set(&mut self, address: &[u32; 2], value: bool) -> Result<(), &str> {
            // validity of address is only checked once, in set_byte
            // as such this function assumes a valid index until the very end
            let byte_index = &[(address[0] / 8) as u32, (address[1] / 8) as u32];

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

            for y in 0..(self.dims[0]) as u32 {
                out += "\r\n";
                out += &*format!("{:^8}", y);
                for x in 0..(self.dims[1]*8) as u32 {
                    println!("{}, {}", x, y);
                    out += &*format!("{:^8}", self.get(&[x, y]).unwrap())
                }
            }

            f.write_str(&*out);
            Ok(())
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

            let mut cases: Vec<([u32; 2], u8)> = vec![
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

            let mut cases: Vec<([u32; 2], bool)> = vec![
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
}