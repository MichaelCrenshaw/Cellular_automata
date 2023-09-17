<h1>N-Dimensional Cellular Automata</h1>

Please note that while this project is still under development, the principle goals of this project have been met and all other features will be considered quality of life updates— when I have the time.

https://github.com/MichaelCrenshaw/Cellular_automata/assets/57016022/68bfe5d3-0db9-4e73-b7be-5d89a86c46a7

<h2>Abstract</h2>
<hr>
<p>
Cellular automata are known to exhibit different behaviors under different rule-sets and different dimensions, but dynamically changing both of these variables at once is not a feature I found in any existing cellular automata simulations.

This program, and the framework underlying it, aim to solve that issue by making few to no assumptions about the number of dimensions which the simulation will include.

The principle goal of this project is a "learning-out-loud" approach to Rust, OpenCL, and OpenGL.

As a programmer I had, previous to this project, little to no direct experience with rendering or gpu-acceleration of calculations. This project exists mainly due to my desire to refine my skills in Rust, as well as learn the basic and unusual use of gpu rendering/acceleration.
</p>

<h3>Planned Features</h3>

1. [ ] v2.0: Optimize memory use with bitwise logic
2. [ ] v1.4: Add user-defined starting conditions to simulation, and forced simulation restarts
3. [ ] v1.3: Add runtime changes to window behavior
4. [ ] v1.2: Add user control over grid-sizing
5. [ ] v1.1: Add user control over game rules, and recompiling of the compute kernel during runtime
6. [ ] v1.0: Add user control over dimension-traversal, basic QOL, cross-platform support

<h2>Overview</h2>
<h3>Design Choices</h3>
<hr>
<br>

<h4>Data Structures</h4>
<p>
The immediate question when handling highly-variable and/or high-dimension data seems pretty simple: "How do you plan to represent that?"
In this particular case I decided to use a Vec to store data and helper functions to handle the main issue, "Where is cell X?"
This decision in beta-test stages of the project translated very well to the final solution, as buffers can't easily stack in multiple dimensions like a more pre-mature approach might have worked.
</p>
<br>

<h4>Libraries</h4>
<p>
Let it be clear, I'm extremely aware that libraries would have made this project much easier.
My preferred data-structure is one of the exact reasons nalgebra was created.
Camera operations would have been far simpler in any number of libraries, and higher-level libraries would have been easier to work with than OpenGL and OpenCL.

My choice in library is based on two simple questions:
    
1. Is this concept something I already know very well?
2. Is this concept something I want to learn for this particular project?

All libraries used in this project represent where those two questions led me:

While I wanted to learn Rust data structures, I didn't feel a need to go through the tedium of writing a custom GUI; hence egui.

While I wanted to learn rendering, I didn't feel a need to learn platform-specific initialization techniques; hence glium.
</p>
<br>

<h4>Calculations</h4>
<p>
This program uses a fairly standard method of calculation, it simply swaps two buffers back and forth between read and write states for the compute kernel.
Simply put, the program computes the next stage of the simulation by reading from buffer "A" as a reference, and writing the next state of each cell to buffer "B".

That, unfortunately, is where the simplicity ends.
Calculating the neighbors for a given cell isn't a difficult task, but making it dimension-agnostic without extreme performance loss is definitely a challenge.
Given the variability in size of an array of neighbors no compiler would be able to unroll operations for efficiency, and the already steep big-O notation would be even steeper.
<br>(As a note, the lowest big-O I've been able to set up as of yet is roughly O(n<sup>3</sup> - 1 + k))

As a current solution, this program uses the set dimension-rules to generate OpenCL C code with hard-coded neighbor bound checking.
This has a few advantages, and will remain a feature of this project well into the future, but the generated code itself could likely be optimized further; sadly profiling OpenCL code is cumbersome at best.

There are, admittedly some notable inadequacies in the implementation of GPU calculations here, but the primary goal of this project was met; and rehashing async code and threading would be a point of diminishing returns in my learning.
</p>
<br>

<h4>Rendering</h4>
<p>
Rendering was a notable but outwardly uninteresting portion of this project; as such I'll touch on it only briefly.
This project, in its current state, uses rendering techniques which are notably different from how the final version will need to render.
Most notably, the current system of using a texture to determine live and dead cells to render will need to be completely replaced with OpenCL directly altering vertex data.
</p>
<br>

<h4>Concerns</h4>
<p>
A few things are not addressed in this readme, as any project with more than a few hundred lines of code will have seemingly strange decisions behind it; here are a few valid complaints I could see:

* Segmentation of cells looks bad for uneven axis lengths
* "N-dimensions" actually means between 1 and 255 (good luck getting the memory for a dimension complexity of 255)
* The game loop could benefit from async processes since the game steps will always be deterministic
* Please instance your rendering calls
* Moving ana and kata for 4d grids could be shown in a more interesting manner than just slices of the game board

All valid concerns, but this project is a learning out loud approach to Rust and lower-level apis; perhaps these will be resolved in the future.
</p>

<hr>
<p>
As a final note, if you've read this far you likely noticed some errors in both my code and my reasoning. I'm acutely aware of a few of the issues, and have either pointedly ignored them or noted them for future development; if you don't think I brought an issue up then feel free to create an issue on GitHub.
</p>
