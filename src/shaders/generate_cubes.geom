#version 460 core

layout (points) in;
layout (triangle_strip, max_vertices = 24) out;

uniform mat4 perspective;
uniform mat4 view;
uniform mat4 model;

uniform uint tess_level_x;
uniform uint tess_level_y;
uniform uint tess_level_z;
uniform usamplerBuffer tex;
uniform uint offset;

in vec3 v_tex_coords[];

out vec4 cube_color;

void createVertex(vec3 offset, vec3 scale){
    vec4 actualOffset = vec4(offset * scale, 0.0);
    vec4 worldPosition = gl_in[0].gl_Position + actualOffset;
    gl_Position = (perspective * (view * model)) * worldPosition;
    EmitVertex();
}

void main() {
    // Get correct cube color from texture buffer
    uint z = uint(floor(v_tex_coords[0][0] * tess_level_x));
    uint y = uint(floor(v_tex_coords[0][1] * tess_level_x)) * tess_level_y;
    uint x = uint(floor(v_tex_coords[0][2] * tess_level_x)) * tess_level_y * tess_level_z;
    bool alive = bool(texelFetch(tex, int(z + y + x + offset)));
    cube_color = alive ? vec4(normalize(vec3(0.1 + v_tex_coords[0][0], 0.1 + v_tex_coords[0][1], 0.1 + v_tex_coords[0][2])), 1.0) : vec4(0.0, 0.0, 0.0, 0.0);

    // Dont render dead cells
    if (!alive) {
        return;
    }

    // Width
    float space_x = 0.2 / tess_level_x;
    float space_y = 0.2 / tess_level_y;
    float space_z = 0.2 / tess_level_z;

    // Positions
    vec4 near_bottom_left = gl_in[0].gl_Position + vec4(-space_x, -space_y, -space_z, 0.0);
    vec4 near_bottom_right = gl_in[0].gl_Position + vec4(space_x, -space_y, -space_z, 0.0);
    vec4 near_top_left = gl_in[0].gl_Position + vec4(-space_x, space_y, -space_z, 0.0);
    vec4 near_top_right = gl_in[0].gl_Position + vec4(space_x, space_y, -space_z, 0.0);
    vec4 far_bottom_left = gl_in[0].gl_Position + vec4(-space_x, -space_y, space_z, 0.0);
    vec4 far_bottom_right = gl_in[0].gl_Position + vec4(space_x, -space_y, space_z, 0.0);
    vec4 far_top_left = gl_in[0].gl_Position + vec4(-space_x, space_y, space_z, 0.0);
    vec4 far_top_right = gl_in[0].gl_Position + vec4(space_x, space_y, space_z, 0.0);


    // Generate faces
    createVertex(vec3(-1.0, 1.0, 1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(-1.0, -1.0, 1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(1.0, 1.0, 1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(1.0, -1.0, 1.0), vec3(space_x, space_y, space_z));

    EndPrimitive();

    createVertex(vec3(1.0, 1.0, 1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(1.0, -1.0, 1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(1.0, 1.0, -1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(1.0, -1.0, -1.0), vec3(space_x, space_y, space_z));

    EndPrimitive();

    createVertex(vec3(1.0, 1.0, -1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(1.0, -1.0, -1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(-1.0, 1.0, -1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(-1.0, -1.0, -1.0), vec3(space_x, space_y, space_z));

    EndPrimitive();

    createVertex(vec3(-1.0, 1.0, -1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(-1.0, -1.0, -1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(-1.0, 1.0, 1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(-1.0, -1.0, 1.0), vec3(space_x, space_y, space_z));

    EndPrimitive();

    createVertex(vec3(1.0, 1.0, 1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(1.0, 1.0, -1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(-1.0, 1.0, 1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(-1.0, 1.0, -1.0), vec3(space_x, space_y, space_z));

    EndPrimitive();

    createVertex(vec3(-1.0, -1.0, 1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(-1.0, -1.0, -1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(1.0, -1.0, 1.0), vec3(space_x, space_y, space_z));
    createVertex(vec3(1.0, -1.0, -1.0), vec3(space_x, space_y, space_z));

    EndPrimitive();
}