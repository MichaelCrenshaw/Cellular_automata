mod trivial;
mod dtypes;

extern crate ocl;
extern crate core;

use ocl::*;
use ocl::Kernel;
use ocl::Buffer;

fn main() {
    trivial::trivial::trivial_fn().unwrap();

}

