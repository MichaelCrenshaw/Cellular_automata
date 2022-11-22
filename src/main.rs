mod compute;
mod dtypes;

extern crate ocl;
extern crate core;
extern crate glium;

use ocl::*;
use ocl::Kernel;
use ocl::Buffer;
use glium::glutin::{event_loop, window, dpi};
use glium::glutin::ContextBuilder;
use glium::Display;

fn main() {
    // init dimensions, board array, and helpers
    let dimensions: [u32; 2] = [25, 25];
    let array_len = dimensions[0] * dimensions[1];
    let mut array_vec = Vec::with_capacity(array_len as usize);

    for index in 0..array_len {
        if index % 5 == 1 || index % 7 == 0 {
            array_vec.push(1);
        } else {
            array_vec.push(0);
        }
    }

    let mut buffer = compute::compute::compute_2d(&array_vec, &dimensions).unwrap().1;

    // init glium objects
    let mut events_loop = event_loop::EventLoop::new();
    let wb = window::WindowBuilder::new()
        .with_inner_size(dpi::LogicalSize::new(1024.0, 768.0))
        .with_title("2d Cellular Automata");
    let cb = ContextBuilder::new();
    let display = Display::new(wb, cb, &events_loop).unwrap();

    // main loop
    for _ in 0..10 {
        let mut vec = vec![0; array_len as usize];
        buffer.read(&mut vec).enq().unwrap();

        for x in 0..vec.len() {
            if x.rem_euclid(dimensions[1] as usize) == 0 {
                print!("\n");
            }
            print!("{:^8?}", vec[x as usize]);
        }
        println!("\n");

        buffer = compute::compute::compute_2d(&vec, &dimensions).unwrap().1;
    }

}

