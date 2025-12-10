#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;
layout (location = 3) in vec4 f_light_space_pos;
layout (location = 4) in vec4 f_spotlight_pos_0;
layout (location = 5) in vec4 f_spotlight_pos_1;

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

layout (std430, binding = 2) readonly buffer PointLightsBuffer {
    PointLight point_lights[];
};

layout (std430, binding = 3) readonly buffer SpotLightsBuffer {
    SpotLight spotlights[];
};

// Текстуры материалов (set 1)
layout (set = 1, binding = 0) uniform sampler2D albedo_texture;
layout (set = 1, binding = 1) uniform sampler2D specular_texture;
layout (set = 1, binding = 2) uniform sampler2D emissive_texture;

// Shadow maps (set 2)
layout (set = 2, binding = 0) uniform sampler2D shadow_map_directional;
layout (set = 2, binding = 1) uniform sampler2D shadow_map_spotlight_0;
layout (set = 2, binding = 2) uniform sampler2D shadow_map_spotlight_1;

// Нетривиальное сэмплирование текстур
vec4 sample_trippy_texture(sampler2D tex, vec2 uv) {
    if (effect_strength < 0.001) {
        return texture(tex, uv);
    }

    // Волновое искажение координат
    float wave_strength = 0.02 * effect_strength;
    vec2 distorted_uv = uv;
    distorted_uv.x += sin(uv.y * 20.0 + time * 2.0) * wave_strength;
    distorted_uv.y += cos(uv.x * 20.0 + time * 1.5) * wave_strength;

    // 3D-анаглиф эффект
    float offset = 0.003 * effect_strength;
    vec2 uv_red = distorted_uv + vec2(offset, 0.0);
    vec2 uv_blue = distorted_uv - vec2(offset, 0.0);
    vec2 uv_green = distorted_uv;

    float red = texture(tex, uv_red).r;
    float green = texture(tex, uv_green).g;
    float blue = texture(tex, uv_blue).b;
    float alpha = texture(tex, distorted_uv).a;

    // Радиальное искажение
    vec2 center_offset = uv - vec2(0.5);
    float dist = length(center_offset);
    float radial_distortion = sin(dist * 10.0 - time * 3.0) * 0.01 * effect_strength;
    vec2 radial_uv = distorted_uv + center_offset * radial_distortion;
    vec4 radial_sample = texture(tex, radial_uv);

    vec4 anaglyph = vec4(red, green, blue, alpha);
    float mix_factor = 0.3 * clamp(effect_strength, 0.0, 1.0);
    return mix(anaglyph, radial_sample, mix_factor);
}

// Расчет тени с PCF фильтрацией
float calculate_shadow_pcf(vec4 light_space_pos, sampler2D shadow_map) {
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;

    // Проверка, находимся ли мы внутри shadow map
    if (proj_coords.z > 1.0 || proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 1.0; // Вне области shadow map - нет тени
    }

    float current_depth = proj_coords.z;
    vec2 texel_size = 1.0 / textureSize(shadow_map, 0);
    float shadow = 0.0;

    // PCF фильтрация 3x3
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 offset = vec2(x, y) * texel_size;
            float pcf_depth = texture(shadow_map, proj_coords.xy + offset).r;
            shadow += (current_depth - shadow_bias > pcf_depth) ? 0.0 : 1.0;
        }
    }
    shadow /= 9.0;

    return shadow;
}

// Модель освещения Блинна-Фонга
vec3 calculate_blinn_phong(vec3 light_dir, vec3 light_color, vec3 normal,
                           vec3 view_dir, vec3 albedo, vec3 specular) {
    // Diffuse component
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = diff * light_color * albedo;

    // Specular component (Blinn-Phong)
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), shininess);
    vec3 specular_result = spec * light_color * specular;

    return diffuse + specular_result;
}

void main() {
    vec3 normal = normalize(f_normal);
    vec3 view_dir = normalize(camera_position - f_position);

    // Сэмплируем текстуры
    vec4 albedo_tex = sample_trippy_texture(albedo_texture, f_uv);
    vec4 specular_tex = sample_trippy_texture(specular_texture, f_uv);
    vec4 emissive_tex = sample_trippy_texture(emissive_texture, f_uv);

    vec3 albedo = albedo_tex.rgb;
    vec3 specular = specular_tex.rgb * specular_color;
    vec3 emissive = emissive_tex.rgb;

    // Ambient lighting
    vec3 result = ambient_light * albedo;

    // Directional light с тенями
    vec3 light_dir = normalize(-directional_light_direction);
    float directional_shadow = 1.0;

    // Проверяем, включены ли тени для направленного света
    if ((active_shadow_sources & 1u) != 0u) {
        directional_shadow = calculate_shadow_pcf(f_light_space_pos, shadow_map_directional);
    }

    result += directional_shadow * calculate_blinn_phong(
        light_dir,
        directional_light_color * directional_light_intensity,
        normal, view_dir, albedo, specular
    );

    // Point lights (без теней)
    for (uint i = 0; i < num_point_lights; ++i) {
        vec3 light_vec = point_lights[i].position - f_position;
        float distance = length(light_vec);
        vec3 light_dir_point = normalize(light_vec);
        float attenuation = point_lights[i].intensity / (distance * distance);
        vec3 light_color = point_lights[i].color * attenuation;

        result += calculate_blinn_phong(light_dir_point, light_color,
                                       normal, view_dir, albedo, specular);
    }

    // Spotlights с тенями
    for (uint i = 0; i < num_spotlights; ++i) {
        vec3 light_vec = spotlights[i].position - f_position;
        float distance = length(light_vec);
        vec3 light_dir_spot = normalize(light_vec);

        // Spotlight cone calculation
        float theta = dot(light_dir_spot, normalize(-spotlights[i].direction));
        float epsilon = spotlights[i].innerCutoff - spotlights[i].outerCutoff;
        float intensity = clamp((theta - spotlights[i].outerCutoff) / epsilon, 0.0, 1.0);

        // Inverse square law attenuation
        float attenuation = spotlights[i].intensity / (distance * distance);
        vec3 light_color = spotlights[i].color * attenuation * intensity;

        // Проверяем тени для прожекторов
        float spotlight_shadow = 1.0;
        if (i == 0 && (active_shadow_sources & 2u) != 0u) {
            spotlight_shadow = calculate_shadow_pcf(f_spotlight_pos_0, shadow_map_spotlight_0);
        } else if (i == 1 && (active_shadow_sources & 4u) != 0u) {
            spotlight_shadow = calculate_shadow_pcf(f_spotlight_pos_1, shadow_map_spotlight_1);
        }

        result += spotlight_shadow * calculate_blinn_phong(
            light_dir_spot, light_color, normal, view_dir, albedo, specular
        );
    }

    // Добавляем свечение (emissive)
    result += emissive;

    final_color = vec4(result, 1.0);
}
