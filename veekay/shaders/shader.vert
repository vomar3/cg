#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;
layout (location = 3) out vec4 f_light_space_pos;
layout (location = 4) out vec4 f_spotlight_pos_0;
layout (location = 5) out vec4 f_spotlight_pos_1;

layout (binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    float _pad0;
    vec3 directional_light_direction;
    float _pad1;
    vec3 directional_light_color;
    float directional_light_intensity;
    vec3 ambient_light;
    uint num_point_lights;
    uint num_spotlights;
    float time;
    float effect_strength;
    float _pad4;
    mat4 directional_light_space_matrix;
    mat4 spotlight_space_matrices[2];
    uint active_shadow_sources;
    float shadow_bias;
    float _pad5;
    float _pad6;
};

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float _pad_m0;
    vec3 specular_color;
    float shininess;
};

void main() {
    vec4 world_position = model * vec4(v_position, 1.0f);
    mat3 normal_matrix = mat3(transpose(inverse(model)));

    gl_Position = view_projection * world_position;
    f_position = world_position.xyz;
    f_normal = normal_matrix * v_normal;
    f_uv = v_uv;

    // Вычисляем позиции для shadow mapping
    f_light_space_pos = directional_light_space_matrix * world_position;
    f_spotlight_pos_0 = spotlight_space_matrices[0] * world_position;
    f_spotlight_pos_1 = spotlight_space_matrices[1] * world_position;
}
