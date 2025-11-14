#include "veekay/input.hpp"
#include <veekay/veekay.hpp>
#include <veekay/graphics.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>
#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace {

size_t aligned_sizeof;
constexpr uint32_t max_models = 1024;
constexpr uint32_t max_point_lights = 16;
constexpr uint32_t max_spotlights = 16;

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct Material {
    glm::vec3 albedo; // материал
    float _pad0;
    glm::vec3 specular; // блик
    float shininess; // ширина блика
};

struct PointLight {
    glm::vec3 position;
    float _pad0;
    glm::vec3 color;
    float intensity;
};

struct SpotLight {
    glm::vec3 position;
    float _pad0;
    glm::vec3 direction;
    float _pad1;
    glm::vec3 color;
    float intensity;
    float innerCutoff;
    float outerCutoff;
    float _pad2;
    float _pad3;
};

struct SceneUniforms {
    glm::mat4 view_projection;
    glm::vec3 camera_position;
    float _pad0;
    glm::vec3 directional_light_direction;
    float _pad1;
    glm::vec3 directional_light_color;
    float directional_light_intensity;
    glm::vec3 ambient_light;
    uint32_t num_point_lights;
    uint32_t num_spotlights;
    float _pad2;
    float _pad3;
    float _pad4;
};

struct ModelUniforms {
    glm::mat4 model;
    glm::vec3 albedo_color;
    float _pad_m0;
    glm::vec3 specular_color;
    float shininess;
};

struct Mesh {
    veekay::graphics::Buffer* vertex_buffer;
    veekay::graphics::Buffer* index_buffer;
    uint32_t indices;
};

struct 	Transform {
    glm::vec3 position = {0.0f, 0.0f, 0.0f};
    glm::vec3 scale = {1.0f, 1.0f, 1.0f};
    glm::vec3 rotation = {0.0f, 0.0f, 0.0f};
    glm::mat4 matrix() const;
};

struct Model {
    Mesh mesh;
    Transform transform;
    Material material;
};

enum class CameraMode {
    LookAt,
    Transform
};

struct Camera {
    constexpr static float default_fov = 60.0f;
    constexpr static float default_near_plane = 0.01f;
    constexpr static float default_far_plane = 100.0f;

    CameraMode mode = CameraMode::LookAt;

    // Look-At mode parameters
    glm::vec3 position = {0.0f, 0.0f, 0.0f};
    glm::vec3 front = {0.0f, 0.0f, 1.0f};
    glm::vec3 up = {0.0f, 1.0f, 0.0f};
    float yaw = 90.0f; // left right
    float pitch = 0.0f; // up down

    // Transform mode parameters
    glm::vec3 transform_position = {0.0f, 0.0f, 0.0f};
    glm::vec3 transform_rotation = {0.0f, 0.0f, 0.0f};

    // Common parameters
    float fov = default_fov;
    float near_plane = default_near_plane;
    float far_plane = default_far_plane;
    float speed = 2.5f;
    float sensitivity = 0.1f;
    float rotation_speed = 1.0f;

    glm::mat4 view() const;
    glm::mat4 view_projection(float aspect_ratio) const;
    glm::vec3 get_camera_position() const;
    void update_vectors();
    void save_state();
    void restore_state();
    void switch_mode(CameraMode new_mode);

private:
    // Saved state for mode switching
    glm::vec3 saved_position = {0.0f, 0.0f, 0.0f};
    glm::vec3 saved_front = {0.0f, 0.0f, 1.0f};
    float saved_yaw = 0.0f;
    float saved_pitch = 0.0f;
    glm::vec3 saved_transform_position = {0.0f, 0.0f, 0.0f};
    glm::vec3 saved_transform_rotation = {0.0f, 0.0f, 0.0f};
};

// Scene objects
inline namespace {
    Camera camera;
    std::vector<Model> models;
    std::vector<PointLight> point_lights;
    std::vector<SpotLight> spotlights;

    glm::vec3 directional_light_direction = {-0.2f, -1.0f, -0.3f};
    glm::vec3 directional_light_color = {1.0f, 1.0f, 1.0f};
    float directional_light_intensity = 0.5f;
    glm::vec3 ambient_light = {0.2f, 0.2f, 0.2f};

    bool camera_enabled = false;
    float last_mouse_x = 640.0f;
    float last_mouse_y = 360.0f;
    bool first_mouse = true;
}

// Vulkan objects
inline namespace {
    VkShaderModule vertex_shader_module;
    VkShaderModule fragment_shader_module;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorSet descriptor_set;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    veekay::graphics::Buffer* scene_uniforms_buffer;
    veekay::graphics::Buffer* model_uniforms_buffer;
    veekay::graphics::Buffer* point_lights_buffer;
    veekay::graphics::Buffer* spotlights_buffer;
    Mesh plane_mesh;
    Mesh cube_mesh;
    Mesh sphere_mesh;
}

float toRadians(float degrees) {
    return glm::radians(degrees);
}

glm::mat4 Transform::matrix() const {
    glm::mat4 t = glm::translate(glm::mat4(1.0f), position);
    glm::mat4 r = glm::mat4(1.0f);
    r = glm::rotate(r, rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
    r = glm::rotate(r, rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
    r = glm::rotate(r, rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
    glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
    return t * r * s;
}

void Camera::save_state() {
    saved_position = position;
    saved_front = front;
    saved_yaw = yaw;
    saved_pitch = pitch;
    saved_transform_position = transform_position;
    saved_transform_rotation = transform_rotation;
}

void Camera::restore_state() {
    position = saved_position;
    front = saved_front;
    yaw = saved_yaw;
    pitch = saved_pitch;
    transform_position = saved_transform_position;
    transform_rotation = saved_transform_rotation;
}

void Camera::switch_mode(CameraMode new_mode) {
    if (mode == new_mode) return;

    if (mode == CameraMode::LookAt && new_mode == CameraMode::Transform) {
        transform_position = position;
        transform_rotation.y = -toRadians(yaw);
        transform_rotation.x = toRadians(pitch);
        transform_rotation.z = 0.0f;
    } else if (mode == CameraMode::Transform && new_mode == CameraMode::LookAt) {
        position = transform_position;
        yaw = -glm::degrees(transform_rotation.y);
        pitch = glm::degrees(transform_rotation.x);
        update_vectors();
    }

    mode = new_mode;
}

void Camera::update_vectors() {
    if (mode == CameraMode::LookAt) {
        glm::vec3 direction;
        direction.x = std::cos(toRadians(yaw)) * std::cos(toRadians(pitch));
        direction.y = std::sin(toRadians(pitch));
        direction.z = std::sin(toRadians(yaw)) * std::cos(toRadians(pitch));
        front = glm::normalize(direction);
    }
}

glm::mat4 Camera::view() const {
    if (mode == CameraMode::LookAt) {
        return glm::lookAt(position, position + front, up);
    } else {
        glm::mat4 translation = glm::translate(glm::mat4(1.0f), transform_position);
        glm::mat4 rotation = glm::mat4(1.0f);
        rotation = glm::rotate(rotation, transform_rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
        rotation = glm::rotate(rotation, transform_rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
        rotation = glm::rotate(rotation, transform_rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));

        glm::mat4 camera_matrix = translation * rotation;
        return glm::inverse(camera_matrix);
    }
}

glm::mat4 Camera::view_projection(float aspect_ratio) const {
    glm::mat4 projection = glm::perspective(toRadians(fov), aspect_ratio, near_plane, far_plane);
    return projection * view();
}

glm::vec3 Camera::get_camera_position() const {
    if (mode == CameraMode::LookAt) {
        return position;
    } else {
        return transform_position;
    }
}

VkShaderModule loadShaderModule(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << path << std::endl;
        return nullptr;
    }
    size_t size = file.tellg();
    std::vector<uint32_t> buffer(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    file.close();
    VkShaderModuleCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = buffer.data(),
    };
    VkShaderModule result;
    if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
        return nullptr;
    }
    return result;
}

Mesh create_sphere_mesh(int stacks, int slices) {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    for (int i = 0; i <= stacks; ++i) {
        float V = float(i) / float(stacks);
        float phi = V * float(M_PI);

        for (int j = 0; j <= slices; ++j) {
            float U = float(j) / float(slices);
            float theta = U * 2.0f * float(M_PI);

            float x = std::cos(theta) * std::sin(phi);
            float y = std::cos(phi);
            float z = std::sin(theta) * std::sin(phi);

            vertices.push_back({
                glm::vec3(x * 0.5f, y * 0.5f, z * 0.5f),
                glm::vec3(x, y, z),
                glm::vec2(U, V)
            });
        }
    }

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            int first = i * (slices + 1) + j;
            int second = first + slices + 1;

            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }

    Mesh mesh;
    mesh.vertex_buffer = new veekay::graphics::Buffer(
        vertices.size() * sizeof(Vertex), vertices.data(),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    mesh.index_buffer = new veekay::graphics::Buffer(
        indices.size() * sizeof(uint32_t), indices.data(),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    mesh.indices = uint32_t(indices.size());

    return mesh;
}

void initialize(VkCommandBuffer cmd) {
    VkDevice& device = veekay::app.vk_device;
    VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device, &props);
    uint32_t alignment = props.limits.minUniformBufferOffsetAlignment;
    aligned_sizeof = ((sizeof(ModelUniforms) + alignment - 1) / alignment) * alignment;

    { // Build graphics pipeline
        vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
        if (!vertex_shader_module) {
            std::cerr << "Failed to load Vulkan vertex shader from file\n";
            veekay::app.running = false;
            return;
        }

        fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
        if (!fragment_shader_module) {
            std::cerr << "Failed to load Vulkan fragment shader from file\n";
            veekay::app.running = false;
            return;
        }

        VkPipelineShaderStageCreateInfo stage_infos[2];
        stage_infos[0] = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertex_shader_module,
            .pName = "main",
        };
        stage_infos[1] = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragment_shader_module,
            .pName = "main",
        };

        VkVertexInputBindingDescription buffer_binding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        VkVertexInputAttributeDescription attributes[] = {
            {
                .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, position),
            },
            {
                .location = 1,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, normal),
            },
            {
                .location = 2,
                .binding = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(Vertex, uv),
            },
        };

        VkPipelineVertexInputStateCreateInfo input_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &buffer_binding,
            .vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
            .pVertexAttributeDescriptions = attributes,
        };

        VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };

        VkPipelineRasterizationStateCreateInfo raster_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .lineWidth = 1.0f,
        };

        VkPipelineMultisampleStateCreateInfo sample_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = false,
            .minSampleShading = 1.0f,
        };

        VkViewport viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(veekay::app.window_width),
            .height = static_cast<float>(veekay::app.window_height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        VkRect2D scissor{
            .offset = {0, 0},
            .extent = {veekay::app.window_width, veekay::app.window_height},
        };

        VkPipelineViewportStateCreateInfo viewport_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
        };

        VkPipelineDepthStencilStateCreateInfo depth_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = true,
            .depthWriteEnable = true,
            .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        };

        VkPipelineColorBlendAttachmentState attachment_info{
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                            VK_COLOR_COMPONENT_G_BIT |
                            VK_COLOR_COMPONENT_B_BIT |
                            VK_COLOR_COMPONENT_A_BIT,
        };

        VkPipelineColorBlendStateCreateInfo blend_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = false,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &attachment_info
        };

        VkDescriptorPoolSize pools[] = {
            {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 8,
            },
            {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .descriptorCount = 8,
            },
            {
                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 16,
            }
        };

        VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = sizeof(pools) / sizeof(pools[0]),
            .pPoolSizes = pools,
        };

        if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan descriptor pool\n";
            veekay::app.running = false;
            return;
        }

        VkDescriptorSetLayoutBinding bindings[] = {
            { // scene
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            { // Models
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            { // point lights
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            { // spotlights
                .binding = 3,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            }
        };

        VkDescriptorSetLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
            .pBindings = bindings,
        };

        if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan descriptor set layout\n";
            veekay::app.running = false;
            return;
        }

        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptor_set_layout,
        };

        if (vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan descriptor set\n";
            veekay::app.running = false;
            return;
        }

        VkPipelineLayoutCreateInfo pipeline_layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_set_layout,
        };

        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan pipeline layout\n";
            veekay::app.running = false;
            return;
        }

        VkGraphicsPipelineCreateInfo pipeline_info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = stage_infos,
            .pVertexInputState = &input_state_info,
            .pInputAssemblyState = &assembly_state_info,
            .pViewportState = &viewport_info,
            .pRasterizationState = &raster_info,
            .pMultisampleState = &sample_info,
            .pDepthStencilState = &depth_info,
            .pColorBlendState = &blend_info,
            .layout = pipeline_layout,
            .renderPass = veekay::app.vk_render_pass,
        };

        if (vkCreateGraphicsPipelines(device, nullptr, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan pipeline\n";
            veekay::app.running = false;
            return;
        }
    }

    scene_uniforms_buffer = new veekay::graphics::Buffer(
        sizeof(SceneUniforms),
        nullptr,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    model_uniforms_buffer = new veekay::graphics::Buffer(
        max_models * aligned_sizeof,
        nullptr,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    point_lights_buffer = new veekay::graphics::Buffer(
        max_point_lights * sizeof(PointLight),
        nullptr,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    spotlights_buffer = new veekay::graphics::Buffer(
        max_spotlights * sizeof(SpotLight),
        nullptr,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    VkDescriptorBufferInfo buffer_infos[] = {
        {
            .buffer = scene_uniforms_buffer->buffer,
            .offset = 0,
            .range = sizeof(SceneUniforms),
        },
        {
            .buffer = model_uniforms_buffer->buffer,
            .offset = 0,
            .range = sizeof(ModelUniforms),
        },
        {
            .buffer = point_lights_buffer->buffer,
            .offset = 0,
            .range = max_point_lights * sizeof(PointLight),
        },
        {
            .buffer = spotlights_buffer->buffer,
            .offset = 0,
            .range = max_spotlights * sizeof(SpotLight),
        }
    };

    VkWriteDescriptorSet write_infos[] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &buffer_infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .pBufferInfo = &buffer_infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_infos[2],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 3,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_infos[3],
        }
    };

    vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]), write_infos, 0, nullptr);

    // Create meshes
    {
        std::vector<Vertex> vertices = {
            {glm::vec3(-5.0f, 0.0f,  5.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(0.0f, 0.0f)},
            {glm::vec3( 5.0f, 0.0f,  5.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(5.0f, 0.0f)},
            {glm::vec3( 5.0f, 0.0f, -5.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(5.0f, 5.0f)},
            {glm::vec3(-5.0f, 0.0f, -5.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(0.0f, 5.0f)},
        };
        std::vector<uint32_t> indices = {
            0, 1, 2, 2, 3, 0
        };
        plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
            vertices.size() * sizeof(Vertex), vertices.data(),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        plane_mesh.index_buffer = new veekay::graphics::Buffer(
            indices.size() * sizeof(uint32_t), indices.data(),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        plane_mesh.indices = uint32_t(indices.size());
    }

    {
        std::vector<Vertex> vertices = {
            {glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec2(0.0f, 0.0f)},
            {glm::vec3(+0.5f, -0.5f, -0.5f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec2(1.0f, 0.0f)},
            {glm::vec3(+0.5f, +0.5f, -0.5f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec2(1.0f, 1.0f)},
            {glm::vec3(-0.5f, +0.5f, -0.5f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec2(0.0f, 1.0f)},
            {glm::vec3(+0.5f, -0.5f, -0.5f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec2(0.0f, 0.0f)},
            {glm::vec3(+0.5f, -0.5f, +0.5f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec2(1.0f, 0.0f)},
            {glm::vec3(+0.5f, +0.5f, +0.5f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec2(1.0f, 1.0f)},
            {glm::vec3(+0.5f, +0.5f, -0.5f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec2(0.0f, 1.0f)},
            {glm::vec3(+0.5f, -0.5f, +0.5f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec2(0.0f, 0.0f)},
            {glm::vec3(-0.5f, -0.5f, +0.5f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec2(1.0f, 0.0f)},
            {glm::vec3(-0.5f, +0.5f, +0.5f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec2(1.0f, 1.0f)},
            {glm::vec3(+0.5f, +0.5f, +0.5f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec2(0.0f, 1.0f)},
            {glm::vec3(-0.5f, -0.5f, +0.5f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec2(0.0f, 0.0f)},
            {glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec2(1.0f, 0.0f)},
            {glm::vec3(-0.5f, +0.5f, -0.5f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec2(1.0f, 1.0f)},
            {glm::vec3(-0.5f, +0.5f, +0.5f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec2(0.0f, 1.0f)},
            {glm::vec3(-0.5f, +0.5f, -0.5f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(0.0f, 0.0f)},
            {glm::vec3(+0.5f, +0.5f, -0.5f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(1.0f, 0.0f)},
            {glm::vec3(+0.5f, +0.5f, +0.5f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(1.0f, 1.0f)},
            {glm::vec3(-0.5f, +0.5f, +0.5f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(0.0f, 1.0f)},
            {glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec2(0.0f, 0.0f)},
            {glm::vec3(+0.5f, -0.5f, -0.5f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec2(1.0f, 0.0f)},
            {glm::vec3(+0.5f, -0.5f, +0.5f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec2(1.0f, 1.0f)},
            {glm::vec3(-0.5f, -0.5f, +0.5f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec2(0.0f, 1.0f)},
        };
        std::vector<uint32_t> indices = {
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 22, 21, 22, 20, 23
        };
        cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
            vertices.size() * sizeof(Vertex), vertices.data(),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        cube_mesh.index_buffer = new veekay::graphics::Buffer(
            indices.size() * sizeof(uint32_t), indices.data(),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        cube_mesh.indices = uint32_t(indices.size());
    }

    sphere_mesh = create_sphere_mesh(30, 30);

    // Create scene
    models.push_back({
        .mesh = plane_mesh,
        .transform = {.position = {0.0f, 5.0f, 0.0f}},
        .material = {
            .albedo = {0.8f, 0.8f, 0.8f},
            .specular = {0.3f, 0.3f, 0.3f},
            .shininess = 32.0f
        }
    });

    for (int i = 0; i < 3; ++i) {
        models.push_back({
            .mesh = cube_mesh,
            .transform = {
                .position = {-2.0f + i * 2.0f, 1.0f, 0.0f},
                .scale = {0.8f, 0.8f, 0.8f}
            },
            .material = {
                .albedo = {0.2f + i * 0.3f, 0.3f, 0.8f - i * 0.2f},
                .specular = {0.8f, 0.8f, 0.8f},
                .shininess = 64.0f
            }
        });
    }

    models.push_back({
        .mesh = sphere_mesh,
        .transform = {
            .position = {0.0f, 2.5f, 2.0f},
            .scale = {1.2f, 1.2f, 1.2f}
        },
        .material = {
            .albedo = {0.9f, 0.3f, 0.2f},
            .specular = {1.0f, 1.0f, 1.0f},
            .shininess = 128.0f
        }
    });

    // Create lights
    point_lights.push_back({
        .position = {2.0f, 2.0f, 2.0f},
        .color = {1.0f, 0.8f, 0.6f},
        .intensity = 5.0f
    });

    point_lights.push_back({
        .position = {-2.0f, 1.5f, -1.0f},
        .color = {0.3f, 0.6f, 1.0f},
        .intensity = 3.0f
    });

    spotlights.push_back({
        .position = {0.0f, 4.0f, 0.0f},
        .direction = {0.0f, -1.0f, 0.0f},
        .color = {1.0f, 1.0f, 1.0f},
        .intensity = 10.0f,
        .innerCutoff = std::cos(toRadians(12.5f)),
        .outerCutoff = std::cos(toRadians(17.5f))
    });

    // Initialize camera
    camera.position = {0.0f, 2.0f, -5.0f};
    camera.transform_position = {0.0f, 2.0f, -5.0f};
    camera.update_vectors();
}

void shutdown() {
    VkDevice& device = veekay::app.vk_device;

    delete sphere_mesh.vertex_buffer;
    delete sphere_mesh.index_buffer;
    delete cube_mesh.vertex_buffer;
    delete cube_mesh.index_buffer;
    delete plane_mesh.vertex_buffer;
    delete plane_mesh.index_buffer;
    delete spotlights_buffer;
    delete point_lights_buffer;
    delete model_uniforms_buffer;
    delete scene_uniforms_buffer;
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    vkDestroyShaderModule(device, fragment_shader_module, nullptr);
    vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
    static double last_time = time;
    float delta_time = float(time - last_time);
    last_time = time;

    // ImGui UI
    ImGui::Begin("Lighting & Camera Control");

    // Camera Mode Section
    ImGui::SeparatorText("Camera Mode");
    const char* camera_modes[] = { "Look-At", "Transform" };
    int current_mode = static_cast<int>(camera.mode);
    if (ImGui::Combo("Mode", &current_mode, camera_modes, 2)) {
        camera.switch_mode(static_cast<CameraMode>(current_mode));
    }

    if (ImGui::Button("Save Camera State")) {
        camera.save_state();
        ImGui::OpenPopup("State Saved");
    }
    if (ImGui::BeginPopup("State Saved")) {
        ImGui::Text("Camera state saved!");
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Restore Camera State")) {
        camera.restore_state();
    }

    ImGui::Separator();
    ImGui::Text("Press C to toggle camera control");
    ImGui::Text("Controls: WASD + Mouse + Space/Shift");

    if (camera.mode == CameraMode::LookAt) {
        ImGui::Text("Mode: Look-At (FPS Style)");
        ImGui::Text("Position: %.1f, %.1f, %.1f", camera.position.x, camera.position.y, camera.position.z);
        ImGui::Text("Yaw: %.1f, Pitch: %.1f", camera.yaw, camera.pitch);
    } else {
        ImGui::Text("Mode: Transform (Object-based)");
        ImGui::Text("Position: %.1f, %.1f, %.1f",
                    camera.transform_position.x, camera.transform_position.y, camera.transform_position.z);
        ImGui::Text("Rotation: %.1f, %.1f, %.1f",
                    glm::degrees(camera.transform_rotation.x),
                    glm::degrees(camera.transform_rotation.y),
                    glm::degrees(camera.transform_rotation.z));
    }

    ImGui::SliderFloat("Speed", &camera.speed, 0.5f, 10.0f);
    ImGui::SliderFloat("Sensitivity", &camera.sensitivity, 0.01f, 0.5f);
    ImGui::SliderFloat("Rotation Speed", &camera.rotation_speed, 0.1f, 3.0f);

    // Lighting Section
    ImGui::SeparatorText("Lighting");
    ImGui::ColorEdit3("Ambient Light", &ambient_light.x);

    ImGui::Separator();
    ImGui::Text("Directional Light");
    ImGui::SliderFloat3("Direction", &directional_light_direction.x, -1.0f, 1.0f);
    ImGui::ColorEdit3("Dir Color", &directional_light_color.x);
    ImGui::SliderFloat("Dir Intensity", &directional_light_intensity, 0.0f, 2.0f);

    // Point Lights Section
    ImGui::SeparatorText("Point Lights");
    if (ImGui::Button("Add Point Light") && point_lights.size() < max_point_lights) {
        point_lights.push_back({
            .position = {0.0f, 2.0f, 0.0f},
            .color = {1.0f, 1.0f, 1.0f},
            .intensity = 5.0f
        });
    }

    for (size_t i = 0; i < point_lights.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        std::string node_name = "Point Light " + std::to_string(i);
        if (ImGui::TreeNode(node_name.c_str())) {
            ImGui::SliderFloat3("Position", &point_lights[i].position.x, -10.0f, 10.0f);
            ImGui::ColorEdit3("Color", &point_lights[i].color.x);
            ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.0f, 20.0f);
            if (ImGui::Button("Remove")) {
                point_lights.erase(point_lights.begin() + i);
                ImGui::TreePop();
                ImGui::PopID();
                break;
            }
            ImGui::TreePop();
        }
        ImGui::PopID();
    }

    // Spotlights Section
    ImGui::SeparatorText("Spotlights");
    if (ImGui::Button("Add Spotlight") && spotlights.size() < max_spotlights) {
        spotlights.push_back({
            .position = {0.0f, 3.0f, 0.0f},
            .direction = {0.0f, -1.0f, 0.0f},
            .color = {1.0f, 1.0f, 1.0f},
            .intensity = 10.0f,
            .innerCutoff = std::cos(toRadians(12.5f)),
            .outerCutoff = std::cos(toRadians(17.5f))
        });
    }

    for (size_t i = 0; i < spotlights.size(); ++i) {
        ImGui::PushID(1000 + static_cast<int>(i));
        std::string node_name = "Spotlight " + std::to_string(i);
        if (ImGui::TreeNode(node_name.c_str())) {
            ImGui::SliderFloat3("Position", &spotlights[i].position.x, -10.0f, 10.0f);
            ImGui::SliderFloat3("Direction", &spotlights[i].direction.x, -1.0f, 1.0f);
            ImGui::ColorEdit3("Color", &spotlights[i].color.x);
            ImGui::SliderFloat("Intensity", &spotlights[i].intensity, 0.0f, 30.0f);
            float inner_angle = std::acos(spotlights[i].innerCutoff) * 180.0f / float(M_PI);
            float outer_angle = std::acos(spotlights[i].outerCutoff) * 180.0f / float(M_PI);
            if (ImGui::SliderFloat("Inner Angle", &inner_angle, 0.0f, 60.0f)) {
                spotlights[i].innerCutoff = std::cos(toRadians(inner_angle));
            }
            if (ImGui::SliderFloat("Outer Angle", &outer_angle, 0.0f, 60.0f)) {
                spotlights[i].outerCutoff = std::cos(toRadians(outer_angle));
            }
            if (ImGui::Button("Remove")) {
                spotlights.erase(spotlights.begin() + i);
                ImGui::TreePop();
                ImGui::PopID();
                break;
            }
            ImGui::TreePop();
        }
        ImGui::PopID();
    }

    ImGui::End();

    // Camera control
    if (veekay::input::keyboard::isKeyPressed(veekay::input::keyboard::Key::c)) {
        camera_enabled = !camera_enabled;
        veekay::input::mouse::setCaptured(camera_enabled);
        first_mouse = true;
    }

    if (camera_enabled) {
        float velocity = camera.speed * delta_time;

        if (camera.mode == CameraMode::LookAt) {
            // Look-At mode: FPS-style movement
            if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::w)) {
                camera.position += camera.front * velocity;
            }
            if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::s)) {
                camera.position -= camera.front * velocity;
            }
            if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::a)) {
                auto right = glm::normalize(glm::cross(camera.front, camera.up));
                camera.position -= right * velocity;
            }
            if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::d)) {
                auto right = glm::normalize(glm::cross(camera.front, camera.up));
                camera.position += right * velocity;
            }
            if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::space)) {
                camera.position.y -= velocity;
            }
            if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::left_shift)) {
                camera.position.y += velocity;
            }

            auto mouse_pos = veekay::input::mouse::cursorPosition();
            if (first_mouse) {
                last_mouse_x = mouse_pos.x;
                last_mouse_y = mouse_pos.y;
                first_mouse = false;
            }

            float xoffset = (mouse_pos.x - last_mouse_x);
            float yoffset = -(last_mouse_y - mouse_pos.y);
            last_mouse_x = mouse_pos.x;
            last_mouse_y = mouse_pos.y;

            xoffset *= camera.sensitivity;
            yoffset *= camera.sensitivity;

            camera.yaw += xoffset;
            camera.pitch += yoffset;

            if (camera.pitch > 89.0f) camera.pitch = 89.0f;
            if (camera.pitch < -89.0f) camera.pitch = -89.0f;

            camera.update_vectors();
        } else {
            // Transform mode: Object-based movement
            glm::mat4 rotation = glm::mat4(1.0f);
            rotation = glm::rotate(rotation, camera.transform_rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
            glm::vec3 forward = glm::vec3(rotation * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f));
            glm::vec3 right = glm::vec3(rotation * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));

            if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::w)) {
                camera.transform_position -= forward * velocity;
            }
            if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::s)) {
                camera.transform_position += forward * velocity;
            }
            if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::a)) {
                camera.transform_position -= right * velocity;
            }
            if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::d)) {
                camera.transform_position += right * velocity;
            }
            if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::space)) {
                camera.transform_position.y -= velocity;
            }
            if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::left_shift)) {
                camera.transform_position.y += velocity;
            }

            auto mouse_pos = veekay::input::mouse::cursorPosition();
            if (first_mouse) {
                last_mouse_x = mouse_pos.x;
                last_mouse_y = mouse_pos.y;
                first_mouse = false;
            }

            float xoffset = mouse_pos.x - last_mouse_x;
            float yoffset = last_mouse_y - mouse_pos.y;
            last_mouse_x = mouse_pos.x;
            last_mouse_y = mouse_pos.y;

            camera.transform_rotation.y -= xoffset * camera.sensitivity * 0.01f;
            camera.transform_rotation.x -= yoffset * camera.sensitivity * 0.01f;

            if (camera.transform_rotation.x > toRadians(89.0f)) camera.transform_rotation.x = toRadians(89.0f);
            if (camera.transform_rotation.x < toRadians(-89.0f)) camera.transform_rotation.x = toRadians(-89.0f);
        }
    }

    // Animate some objects
    if (models.size() > 1) models[1].transform.rotation.y = float(time) * 0.5f;
    if (models.size() > 2) models[2].transform.position.y = 1.0f + std::sin(float(time)) * 0.3f;

    // Animate a point light
    if (point_lights.size() > 0) {
        point_lights[0].position.x = std::cos(float(time)) * 3.0f;
        point_lights[0].position.z = std::sin(float(time)) * 3.0f;
    }
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &begin_info);

    VkClearValue clear_values[2];
    clear_values[0].color = {0.1f, 0.1f, 0.15f, 1.0f};
    clear_values[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo render_pass_info{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = veekay::app.vk_render_pass,
        .framebuffer = framebuffer,
        .renderArea = {
            .extent = {veekay::app.window_width, veekay::app.window_height},
        },
        .clearValueCount = 2,
        .pClearValues = clear_values,
    };

    vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

    // Update scene uniforms
    float aspect = float(veekay::app.window_width) / float(veekay::app.window_height);
    glm::vec3 cam_pos = camera.get_camera_position();

    SceneUniforms scene_uniforms{
        .view_projection = camera.view_projection(aspect),
        .camera_position = cam_pos,
        ._pad0 = 0.0f,
        .directional_light_direction = directional_light_direction,
        ._pad1 = 0.0f,
        .directional_light_color = directional_light_color,
        .directional_light_intensity = directional_light_intensity,
        .ambient_light = ambient_light,
        .num_point_lights = static_cast<uint32_t>(point_lights.size()),
        .num_spotlights = static_cast<uint32_t>(spotlights.size()),
        ._pad2 = 0.0f,
        ._pad3 = 0.0f,
        ._pad4 = 0.0f
    };
    std::memcpy(scene_uniforms_buffer->mapped_region, &scene_uniforms, sizeof(SceneUniforms));

    // Update lights buffers
    if (!point_lights.empty()) {
        std::memcpy(point_lights_buffer->mapped_region, point_lights.data(),
                   point_lights.size() * sizeof(PointLight));
    }
    if (!spotlights.empty()) {
        std::memcpy(spotlights_buffer->mapped_region, spotlights.data(),
                   spotlights.size() * sizeof(SpotLight));
    }

    // Update model uniforms
    for (size_t i = 0; i < models.size(); ++i) {
        ModelUniforms model_uniforms{
            .model = models[i].transform.matrix(),
            .albedo_color = models[i].material.albedo,
            ._pad_m0 = 0.0f,
            .specular_color = models[i].material.specular,
            .shininess = models[i].material.shininess
        };
        char* dst = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * aligned_sizeof;
        std::memcpy(dst, &model_uniforms, sizeof(ModelUniforms));
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
    VkBuffer current_index_buffer = VK_NULL_HANDLE;
    VkDeviceSize zero_offset = 0;

    for (size_t i = 0; i < models.size(); ++i) {
        const auto& model = models[i];
        const auto& mesh = model.mesh;

        if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
            current_vertex_buffer = mesh.vertex_buffer->buffer;
            vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
        }

        if (current_index_buffer != mesh.index_buffer->buffer) {
            current_index_buffer = mesh.index_buffer->buffer;
            vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
        }

        uint32_t offset = uint32_t(i * aligned_sizeof);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                               0, 1, &descriptor_set, 1, &offset);
        vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
    std::srand(static_cast<unsigned int>(time(nullptr)));
    return veekay::run({
        .init = initialize,
        .shutdown = shutdown,
        .update = update,
        .render = render,
    });
}
