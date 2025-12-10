#version 450

layout (location = 0) in vec3 v_position;

layout (push_constant) uniform PushConstants {
    mat4 light_view_projection;
    mat4 model;
} push;

void main() {
    gl_Position = push.light_view_projection * push.model * vec4(v_position, 1.0);
}
