#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

struct PointLight {
    vec3 position;
    float _pad0;
    vec3 color;
    float intensity;
};

struct SpotLight {
    vec3 position;
    float _pad0;
    vec3 direction;
    float _pad1;
    vec3 color;
    float intensity;
    float innerCutoff;
    float outerCutoff;
    float _pad2;
    float _pad3;
};

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
    float _pad2;
    float _pad3;
    float _pad4;
};

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float _pad_m0;
    vec3 specular_color;
    float shininess;
};

layout (std430, binding = 2) readonly buffer PointLightsBuffer {
    PointLight point_lights[];
};

layout (std430, binding = 3) readonly buffer SpotLightsBuffer {
    SpotLight spotlights[];
};

vec3 calculate_blinn_phong(vec3 light_dir, vec3 light_color, vec3 normal, vec3 view_dir) {
    // Diffuse component (main color)
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = diff * light_color * albedo_color;

    // Specular component (Blinn-Phong)
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), shininess);
    vec3 specular = spec * light_color * specular_color;

    return diffuse + specular;
}

void main() {
    vec3 normal = normalize(f_normal);
    vec3 view_dir = normalize(camera_position - f_position);

    // Ambient lighting
    vec3 result = ambient_light * albedo_color;

    // Directional light
    vec3 light_dir = normalize(-directional_light_direction);
    result += calculate_blinn_phong(light_dir,
                                    directional_light_color * directional_light_intensity,
                                    normal, view_dir);

    // Point lights
    for (uint i = 0; i < num_point_lights; ++i) {
        vec3 light_vec = point_lights[i].position - f_position;
        float distance = length(light_vec);
        vec3 light_dir = normalize(light_vec);

        float attenuation = point_lights[i].intensity / (distance * distance);
        vec3 light_color = point_lights[i].color * attenuation;

        result += calculate_blinn_phong(light_dir, light_color, normal, view_dir);
    }

    // Spotlights
    for (uint i = 0; i < num_spotlights; ++i) {
        vec3 light_vec = spotlights[i].position - f_position;
        float distance = length(light_vec);
        vec3 light_dir = normalize(light_vec);

        // Spotlight cone calculation
        float theta = dot(light_dir, normalize(-spotlights[i].direction));
        float epsilon = spotlights[i].innerCutoff - spotlights[i].outerCutoff;
        float intensity = clamp((theta - spotlights[i].outerCutoff) / epsilon, 0.0, 1.0);

        // Inverse square law attenuation
        float attenuation = spotlights[i].intensity / (distance * distance);
        vec3 light_color = spotlights[i].color * attenuation * intensity;

        result += calculate_blinn_phong(light_dir, light_color, normal, view_dir);
    }

    final_color = vec4(result, 1.0f);
}
