#include "veekay/graphics.hpp"
#include "veekay/input.hpp"
#include "veekay/veekay.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <algorithm>

#include "stb_image.h"

double GetTimeAsDouble() {
    using namespace std::chrono;
    using SecondsFP = std::chrono::duration<double>;
    return duration_cast<SecondsFP>(high_resolution_clock::now().time_since_epoch())
        .count();
}

namespace {

constexpr uint32_t max_models       = 1024;
constexpr uint32_t max_point_lights = 16;
constexpr uint32_t max_spotlights   = 16;
constexpr uint32_t shadow_resolution = 4096;
constexpr uint32_t max_shadow_spotlights = 2;

size_t aligned_sizeof_model_ubo = 0;

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct Material {
    glm::vec3 albedo;
    float     _pad0;
    glm::vec3 specular;
    float     shininess;
};

struct MaterialTextures {
    veekay::graphics::Texture* albedo  = nullptr;
    veekay::graphics::Texture* specular = nullptr;
    veekay::graphics::Texture* emissive = nullptr;
    VkSampler                  sampler  = VK_NULL_HANDLE;
    VkDescriptorSet            descriptor_set = VK_NULL_HANDLE;
};

struct PointLight {
    glm::vec3 position;
    float     _pad0;
    glm::vec3 color;
    float     intensity;
};

struct SpotLight {
    glm::vec3 position;
    float     _pad0;
    glm::vec3 direction;
    float     _pad1;
    glm::vec3 color;
    float     intensity;
    float     innerCutoff;
    float     outerCutoff;
    float     _pad2;
    float     _pad3;
};

struct SceneUniforms {
    glm::mat4 view_projection;
    glm::vec3 camera_position;
    float     _pad0;
    glm::vec3 directional_light_direction;
    float     _pad1;
    glm::vec3 directional_light_color;
    float     directional_light_intensity;
    glm::vec3 ambient_light;
    uint32_t  num_point_lights;
    uint32_t  num_spotlights;
    float     time;
    float     effect_strength;
    float     _pad4;
    glm::mat4 directional_light_space_matrix;
    glm::mat4 spotlight_space_matrices[max_shadow_spotlights];
    uint32_t  active_shadow_sources;
    float     shadow_bias;
    float     _pad5;
    float     _pad6;
};

struct ModelUniforms {
    glm::mat4 model;
    glm::vec3 albedo_color;
    float     _pad_m0;
    glm::vec3 specular_color;
    float     shininess;
};

struct ShadowPassPC {
    glm::mat4 light_view_projection;
    glm::mat4 model;
};

struct Mesh {
    veekay::graphics::Buffer* vertex_buffer = nullptr;
    veekay::graphics::Buffer* index_buffer  = nullptr;
    uint32_t                  index_count   = 0;
};

struct Transform {
    glm::vec3 position {0.0f};
    glm::vec3 scale    {1.0f};
    glm::vec3 rotation {0.0f};

    glm::mat4 matrix() const {
        glm::mat4 t = glm::translate(glm::mat4(1.0f), position);
        glm::mat4 r = glm::mat4(1.0f);
        r = glm::rotate(r, rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
        r = glm::rotate(r, rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
        r = glm::rotate(r, rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
        glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
        return t * r * s;
    }
};

struct Model {
    Mesh            mesh;
    Transform       transform;
    Material        material;
    MaterialTextures textures;
};

enum class CameraMode {
    LookAt,
    Transform
};

struct Camera {
    static constexpr float default_fov        = 60.0f;
    static constexpr float default_near_plane = 0.01f;
    static constexpr float default_far_plane  = 100.0f;

    CameraMode mode       = CameraMode::LookAt;
    glm::vec3 position    {0.0f, 0.0f, -5.0f};
    glm::vec3 front       {0.0f, 0.0f, 1.0f};
    glm::vec3 up          {0.0f, -1.0f, 0.0f};
    float      yaw        = 90.0f;
    float      pitch      = 0.0f;

    glm::vec3 transform_position {0.0f, 0.0f, -5.0f};
    glm::vec3 transform_rotation {0.0f};

    float fov        = default_fov;
    float near_plane = default_near_plane;
    float far_plane  = default_far_plane;
    float speed      = 2.5f;
    float sensitivity    = 0.1f;
    float rotation_speed = 1.0f;

    glm::mat4 view() const {
        if (mode == CameraMode::LookAt) {
            return glm::lookAt(position, position + front, up);
        } else {
            glm::mat4 T = glm::translate(glm::mat4(1.0f), transform_position);
            glm::mat4 R = glm::mat4(1.0f);
            R = glm::rotate(R, transform_rotation.y, glm::vec3(0,1,0));
            R = glm::rotate(R, transform_rotation.x, glm::vec3(1,0,0));
            R = glm::rotate(R, transform_rotation.z, glm::vec3(0,0,1));
            glm::mat4 cam = T * R;
            return glm::inverse(cam);
        }
    }

    glm::mat4 view_projection(float aspect) const {
        glm::mat4 proj = glm::perspective(glm::radians(fov), aspect, near_plane, far_plane);
        return proj * view();
    }

    glm::vec3 get_position() const {
        return (mode == CameraMode::LookAt) ? position : transform_position;
    }

    void update_vectors() {
        if (mode != CameraMode::LookAt) return;
        glm::vec3 dir;
        dir.x = std::cos(glm::radians(yaw)) * std::cos(glm::radians(pitch));
        dir.y = std::sin(glm::radians(pitch));
        dir.z = std::sin(glm::radians(yaw)) * std::cos(glm::radians(pitch));
        front = glm::normalize(dir);
    }

    void switch_mode(CameraMode new_mode) {
        if (mode == new_mode) return;
        if (mode == CameraMode::LookAt && new_mode == CameraMode::Transform) {
            transform_position = position;
            transform_rotation.y = -glm::radians(yaw);
            transform_rotation.x = glm::radians(pitch);
            transform_rotation.z = 0.0f;
        } else if (mode == CameraMode::Transform && new_mode == CameraMode::LookAt) {
            position = transform_position;
            yaw   = -glm::degrees(transform_rotation.y);
            pitch =  glm::degrees(transform_rotation.x);
            update_vectors();
        }
        mode = new_mode;
    }

    void save_state() {
        saved_position          = position;
        saved_front             = front;
        saved_yaw               = yaw;
        saved_pitch             = pitch;
        saved_transform_pos     = transform_position;
        saved_transform_rot     = transform_rotation;
    }

    void restore_state() {
        position          = saved_position;
        front             = saved_front;
        yaw               = saved_yaw;
        pitch             = saved_pitch;
        transform_position = saved_transform_pos;
        transform_rotation = saved_transform_rot;
        update_vectors();
    }

private:
    glm::vec3 saved_position {0.0f};
    glm::vec3 saved_front    {0.0f, 0.0f, 1.0f};
    float     saved_yaw   = 0.0f;
    float     saved_pitch = 0.0f;
    glm::vec3 saved_transform_pos {0.0f};
    glm::vec3 saved_transform_rot {0.0f};
};

// Shadow map holder
struct ShadowMap {
    VkImage        image  = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView    view   = VK_NULL_HANDLE;
    VkSampler      sampler = VK_NULL_HANDLE;
    VkFramebuffer  framebuffer = VK_NULL_HANDLE;
    uint32_t       resolution = 0;
};

// ====== глобальное состояние сцены ======
inline namespace {
    Camera camera;

    std::vector<Model>      models;
    std::vector<PointLight> point_lights;
    std::vector<SpotLight>  spotlights;

    glm::vec3 dir_light_dir   {-0.2f, -1.0f, -0.3f};
    glm::vec3 dir_light_color {1.0f, 1.0f, 1.0f};
    float     dir_light_int   = 0.8f;
    glm::vec3 ambient_light   {0.2f, 0.2f, 0.2f};

    float texture_effect_strength = 0.0f;

    bool camera_enabled = false;
    bool first_mouse    = true;
    float last_mouse_x  = 640.0f;
    float last_mouse_y  = 360.0f;

    // Vulkan objects
    VkShaderModule vert_module  = VK_NULL_HANDLE;
    VkShaderModule frag_module  = VK_NULL_HANDLE;
    VkShaderModule shadow_vert_module = VK_NULL_HANDLE;
    VkShaderModule shadow_frag_module = VK_NULL_HANDLE;

    VkDescriptorPool        global_desc_pool      = VK_NULL_HANDLE;
    VkDescriptorSetLayout   global_desc_layout    = VK_NULL_HANDLE;
    VkDescriptorSet         global_desc_set       = VK_NULL_HANDLE;

    VkDescriptorSetLayout   material_desc_layout  = VK_NULL_HANDLE;
    VkDescriptorPool        material_desc_pool    = VK_NULL_HANDLE;

    VkDescriptorSetLayout   shadow_desc_layout    = VK_NULL_HANDLE;
    VkDescriptorSet         shadow_desc_set       = VK_NULL_HANDLE;

    VkPipelineLayout        pipeline_layout       = VK_NULL_HANDLE;
    VkPipeline              pipeline              = VK_NULL_HANDLE;

    VkPipelineLayout        shadow_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline              shadow_pipeline        = VK_NULL_HANDLE;
    VkRenderPass            shadow_render_pass     = VK_NULL_HANDLE;

    veekay::graphics::Buffer* scene_ubo  = nullptr;
    veekay::graphics::Buffer* model_ubos = nullptr;
    veekay::graphics::Buffer* point_ssbo = nullptr;
    veekay::graphics::Buffer* spot_ssbo  = nullptr;

    Mesh plane_mesh;
    Mesh cube_mesh;
    Mesh sphere_mesh;

    veekay::graphics::Texture* white_tex = nullptr;
    veekay::graphics::Texture* black_tex = nullptr;

    ShadowMap directional_shadow;
    ShadowMap spotlight_shadows[max_shadow_spotlights];

    bool  enable_shadows          = true;
    bool  enable_dir_shadow       = true;
    bool  enable_spotlight_shadow = true;
    float shadow_bias             = 0.004f;
}

// ========== Вспомогательные функции ==========
VkShaderModule load_shader_module(const char* path) {
    VkDevice device = veekay::app.vk_device;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader: " << path << "\n";
        return VK_NULL_HANDLE;
    }
    size_t size = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> data(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), size);
    file.close();

    VkShaderModuleCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = data.data(),
    };
    VkShaderModule mod;
    if (vkCreateShaderModule(device, &info, nullptr, &mod) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module: " << path << "\n";
        return VK_NULL_HANDLE;
    }
    return mod;
}

veekay::graphics::Texture* load_texture(VkCommandBuffer cmd, const char* path) {
    int w, h, ch;
    unsigned char* pixels = stbi_load(path, &w, &h, &ch, 4);
    if (!pixels) {
        std::cerr << "Failed to load texture: " << path << "\n";
        return nullptr;
    }
    auto* tex = new veekay::graphics::Texture(
        cmd, w, h, VK_FORMAT_R8G8B8A8_UNORM, pixels
    );
    stbi_image_free(pixels);
    return tex;
}

VkSampler create_sampler(VkFilter filter, VkSamplerAddressMode mode) {
    VkDevice device = veekay::app.vk_device;
    VkPhysicalDevice phys = veekay::app.vk_physical_device;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys, &props);

    VkSamplerCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = filter,
        .minFilter = filter,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = mode,
        .addressModeV = mode,
        .addressModeW = mode,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_TRUE,
        .maxAnisotropy = props.limits.maxSamplerAnisotropy,
        .compareEnable = VK_FALSE,                  // или VK_TRUE, если вам нужен compare
        .compareOp = VK_COMPARE_OP_ALWAYS,          // или другое при необходимости
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
        .unnormalizedCoordinates = VK_FALSE,
    };

    VkSampler s;
    if (vkCreateSampler(device, &info, nullptr, &s) != VK_SUCCESS) {
        std::cerr << "Failed to create sampler\n";
        return VK_NULL_HANDLE;
    }
    return s;
}

Mesh create_plane() {
    std::vector<Vertex> v = {
        {{-5.0f, 0.0f, -5.0f}, {0,1,0}, {0,0}},
        {{ 5.0f, 0.0f, -5.0f}, {0,1,0}, {5,0}},
        {{ 5.0f, 0.0f,  5.0f}, {0,1,0}, {5,5}},
        {{-5.0f, 0.0f,  5.0f}, {0,1,0}, {0,5}},
    };
    std::vector<uint32_t> idx = {0,1,2, 0,2,3};

    auto* vb = new veekay::graphics::Buffer(
        v.size() * sizeof(Vertex), v.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    );
    auto* ib = new veekay::graphics::Buffer(
        idx.size() * sizeof(uint32_t), idx.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    );

    Mesh m;
    m.vertex_buffer = vb;
    m.index_buffer  = ib;
    m.index_count   = static_cast<uint32_t>(idx.size());
    return m;
}

Mesh create_cube() {
    std::vector<Vertex> v;
    std::vector<uint32_t> idx;

    float s = 0.5f;
    glm::vec3 p[8] = {
        {-s,-s,-s}, { s,-s,-s}, { s, s,-s}, {-s, s,-s},
        {-s,-s, s}, { s,-s, s}, { s, s, s}, {-s, s, s}
    };
    glm::vec3 n[6] = {
        {0,0,-1}, {0,0,1}, {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}
    };
    int faces[6][4] = {
        {0,1,2,3}, {5,4,7,6}, {4,0,3,7},
        {1,5,6,2}, {4,5,1,0}, {3,2,6,7}
    };
    glm::vec2 u[4] = {{0,0},{1,0},{1,1},{0,1}};

    for (int f=0; f<6; ++f) {
        uint32_t base = static_cast<uint32_t>(v.size());
        for (int i=0;i<4;++i) {
            v.push_back({p[faces[f][i]], n[f], u[i]});
        }
        idx.insert(idx.end(), {base,base+1,base+2, base,base+2,base+3});
    }

    auto* vb = new veekay::graphics::Buffer(
        v.size()*sizeof(Vertex), v.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    );
    auto* ib = new veekay::graphics::Buffer(
        idx.size()*sizeof(uint32_t), idx.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    );

    Mesh m;
    m.vertex_buffer = vb;
    m.index_buffer  = ib;
    m.index_count   = static_cast<uint32_t>(idx.size());
    return m;
}

Mesh create_sphere(int stacks, int slices) {
    std::vector<Vertex> v;
    std::vector<uint32_t> idx;

    for (int i=0;i<=stacks;++i) {
        float V   = float(i)/float(stacks);
        float phi = V * float(M_PI);
        for (int j=0;j<=slices;++j) {
            float U    = float(j)/float(slices);
            float theta = U * 2.0f * float(M_PI);

            float x = std::cos(theta) * std::sin(phi);
            float y = std::cos(phi);
            float z = std::sin(theta) * std::sin(phi);

            v.push_back({
                glm::vec3(x*0.5f, y*0.5f, z*0.5f),
                glm::vec3(x,y,z),
                glm::vec2(U,V)
            });
        }
    }

    for (int i=0;i<stacks;++i) {
        for (int j=0;j<slices;++j) {
            int first  = i*(slices+1) + j;
            int second = first + slices + 1;

            idx.push_back(first);
            idx.push_back(second);
            idx.push_back(first+1);
            idx.push_back(second);
            idx.push_back(second+1);
            idx.push_back(first+1);
        }
    }

    auto* vb = new veekay::graphics::Buffer(
        v.size()*sizeof(Vertex), v.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    );
    auto* ib = new veekay::graphics::Buffer(
        idx.size()*sizeof(uint32_t), idx.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    );

    Mesh m;
    m.vertex_buffer = vb;
    m.index_buffer  = ib;
    m.index_count   = static_cast<uint32_t>(idx.size());
    return m;
}

VkDescriptorSet create_material_set(MaterialTextures& tex) {
    VkDevice device = veekay::app.vk_device;

    VkDescriptorSetAllocateInfo alloc{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = material_desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &material_desc_layout,
    };

    VkDescriptorSet set;
    if (vkAllocateDescriptorSets(device, &alloc, &set) != VK_SUCCESS) {
        std::cerr << "Failed to alloc material set\n";
        return VK_NULL_HANDLE;
    }

    auto* albedo   = tex.albedo   ? tex.albedo   : white_tex;
    auto* specular = tex.specular ? tex.specular : white_tex;
    auto* emissive = tex.emissive ? tex.emissive : black_tex;

    VkDescriptorImageInfo infos[3] = {
        {tex.sampler, albedo->view,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {tex.sampler, specular->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {tex.sampler, emissive->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
    };

    VkWriteDescriptorSet writes[3]{};
    for (int i=0;i<3;++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
    return set;
}

ShadowMap create_shadow_map(uint32_t res) {
    VkDevice device = veekay::app.vk_device;
    VkPhysicalDevice phys = veekay::app.vk_physical_device;

    ShadowMap sm{};
    sm.resolution = res;

    VkImageCreateInfo img{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_D32_SFLOAT,
        .extent = {res,res,1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling  = VK_IMAGE_TILING_OPTIMAL,
        .usage   = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    vkCreateImage(device, &img, nullptr, &sm.image);

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device, sm.image, &req);

    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);

    uint32_t type_index = UINT32_MAX;
    for (uint32_t i=0;i<mp.memoryTypeCount;++i) {
        if ((req.memoryTypeBits & (1u<<i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            type_index = i;
            break;
        }
    }

    VkMemoryAllocateInfo alloc{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = type_index,
    };
    vkAllocateMemory(device, &alloc, nullptr, &sm.memory);
    vkBindImageMemory(device, sm.image, sm.memory, 0);

    VkImageViewCreateInfo view{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = sm.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_D32_SFLOAT,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    vkCreateImageView(device, &view, nullptr, &sm.view);

    VkSamplerCreateInfo samp{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_TRUE,
        .compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        .minLod = 0.0f,
        .maxLod = 1.0f,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
        .unnormalizedCoordinates = VK_FALSE,
    };
    vkCreateSampler(device, &samp, nullptr, &sm.sampler);

    return sm;
}

glm::mat4 calc_dir_light_matrix() {
    glm::vec3 dir = glm::normalize(dir_light_dir);

    const float scene_half_size = 15.0f;
    const float scene_height    = 10.0f;

    glm::vec3 scene_center(0.0f, 2.0f, 0.0f);

    float light_distance = scene_half_size + scene_height;
    glm::vec3 light_pos = scene_center - dir * light_distance;

    glm::vec3 up = (glm::abs(dir.y) > 0.9f)
        ? glm::vec3(1.0f, 0.0f, 0.0f)
        : glm::vec3(0.0f, 1.0f, 0.0f);

    glm::mat4 view = glm::lookAt(light_pos, scene_center, up);

    float ortho = scene_half_size;
    float near_plane = 1.0f;
    float far_plane  = light_distance + scene_height;

    glm::mat4 proj = glm::ortho(-ortho, ortho,
                                -ortho, ortho,
                                near_plane, far_plane);

    return proj * view;
}



glm::mat4 calc_spot_matrix(const SpotLight& s) {
    glm::vec3 d = glm::normalize(s.direction);

    glm::vec3 up;
    if (std::abs(d.y) > 0.99f) {
        up = glm::vec3(1, 0, 0);
    } else {
        up = glm::vec3(0, -1, 0);
    }

    glm::mat4 view = glm::lookAt(s.position, s.position + d, up);

    float fov = 2.0f * std::acos(s.outerCutoff);
    glm::mat4 proj = glm::perspective(fov, 1.0f, 0.1f, 50.0f);

    return proj * view;
}

std::vector<uint32_t> select_spotlights(const glm::vec3& camera_pos) {
    struct Score { uint32_t idx; float score; };
    std::vector<Score> scores;
    for (uint32_t i=0;i<spotlights.size();++i) {
        const auto& s = spotlights[i];
        float dist = glm::length(s.position - camera_pos);
        glm::vec3 to_cam = glm::normalize(camera_pos - s.position);
        float facing = glm::dot(glm::normalize(s.direction), to_cam); // >0 значит в сторону камеры
        float sc = s.intensity / (dist+1.0f) * (1.0f + glm::max(facing,0.0f));
        scores.push_back({i, sc});
    }
    std::sort(scores.begin(), scores.end(),
              [](auto& a, auto& b){ return a.score > b.score; });

    std::vector<uint32_t> result;
    for (size_t i=0;i<scores.size() && i<max_shadow_spotlights;++i)
        result.push_back(scores[i].idx);
    return result;
}

// ====== Shadow pass ======
void render_shadow_pass(VkCommandBuffer cmd, const ShadowMap& sm, const glm::mat4& light_view_proj) {
    VkDevice device = veekay::app.vk_device;

    // 1. Барьер: Shadow map -> DEPTH_ATTACHMENT
    VkImageMemoryBarrier to_depth_barrier = {};
    to_depth_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_depth_barrier.srcAccessMask = 0;
    to_depth_barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    to_depth_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_depth_barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    to_depth_barrier.image = sm.image;
    to_depth_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    to_depth_barrier.subresourceRange.baseMipLevel = 0;
    to_depth_barrier.subresourceRange.levelCount = 1;
    to_depth_barrier.subresourceRange.baseArrayLayer = 0;
    to_depth_barrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        0, 0, nullptr, 0, nullptr, 1, &to_depth_barrier);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)sm.resolution;
    viewport.height = (float)sm.resolution;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {sm.resolution, sm.resolution};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // shadow render pass
    VkClearValue clear_value{};
    clear_value.depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rp_info{};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_info.renderPass = shadow_render_pass;
    rp_info.framebuffer = sm.framebuffer;
    rp_info.renderArea.offset = {0, 0};
    rp_info.renderArea.extent = {sm.resolution, sm.resolution};
    rp_info.clearValueCount = 1;
    rp_info.pClearValues = &clear_value;

    vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);

    // shadow pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);

    // рендерим все модели
    for (size_t i = 0; i < models.size(); ++i) {
        const auto& m = models[i];

        // Push constants для shadow pass
        ShadowPassPC pc{};
        pc.light_view_projection = light_view_proj;
        pc.model = m.transform.matrix();
        vkCmdPushConstants(cmd, shadow_pipeline_layout,
            VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPassPC), &pc);

        // Bind vertex/index buffers
        VkBuffer vb[] = {m.mesh.vertex_buffer->buffer};
        VkDeviceSize off[] = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
        vkCmdBindIndexBuffer(cmd, m.mesh.index_buffer->buffer, 0, VK_INDEX_TYPE_UINT32);

        // Draw
        vkCmdDrawIndexed(cmd, m.mesh.index_count, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cmd);

    // DEPTH_ATTACHMENT -> SHADER_READ (для основного рендера)
    VkImageMemoryBarrier to_shader_read_barrier = {};
    to_shader_read_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_shader_read_barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    to_shader_read_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    to_shader_read_barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    to_shader_read_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    to_shader_read_barrier.image = sm.image;
    to_shader_read_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    to_shader_read_barrier.subresourceRange.baseMipLevel = 0;
    to_shader_read_barrier.subresourceRange.levelCount = 1;
    to_shader_read_barrier.subresourceRange.baseArrayLayer = 0;
    to_shader_read_barrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &to_shader_read_barrier);
}


// ====== Инициализация / обновление / рендер / очистка ======

void initialize(VkCommandBuffer cmd) {
    VkDevice device = veekay::app.vk_device;
    VkPhysicalDevice phys = veekay::app.vk_physical_device;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys, &props);
    uint32_t align = props.limits.minUniformBufferOffsetAlignment;
    aligned_sizeof_model_ubo =
        ((sizeof(ModelUniforms) + align - 1) / align) * align;

    // --- Shadow render pass ---
    {
        VkAttachmentDescription depth{
            .format = VK_FORMAT_D32_SFLOAT,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp= VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        VkAttachmentReference depth_ref{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.pDepthStencilAttachment = &depth_ref;

        VkRenderPassCreateInfo rp{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &depth,
            .subpassCount = 1,
            .pSubpasses = &sub,
        };
        if (vkCreateRenderPass(device, &rp, nullptr, &shadow_render_pass) != VK_SUCCESS) {
            std::cerr << "Failed shadow render pass\n";
            veekay::app.running = false;
            return;
        }
    }

    // --- Shadow maps + framebuffers ---
    directional_shadow = create_shadow_map(shadow_resolution);
    for (uint32_t i=0;i<max_shadow_spotlights;++i)
        spotlight_shadows[i] = create_shadow_map(shadow_resolution);

    {
        VkFramebufferCreateInfo fb{
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = shadow_render_pass,
            .attachmentCount = 1,
            .width  = shadow_resolution,
            .height = shadow_resolution,
            .layers = 1,
        };
        VkImageView att = directional_shadow.view;
        fb.pAttachments = &att;
        vkCreateFramebuffer(device, &fb, nullptr, &directional_shadow.framebuffer);

        for (uint32_t i=0;i<max_shadow_spotlights;++i) {
            att = spotlight_shadows[i].view;
            fb.pAttachments = &att;
            vkCreateFramebuffer(device, &fb, nullptr, &spotlight_shadows[i].framebuffer);
        }
    }

    // --- Основные шейдеры ---
    vert_module = load_shader_module("./shaders/shader.vert.spv");
    frag_module = load_shader_module("./shaders/shader.frag.spv");
    shadow_vert_module = load_shader_module("./shaders/shadow.vert.spv");
    shadow_frag_module = load_shader_module("./shaders/shadow.frag.spv");
    if (!vert_module || !frag_module || !shadow_vert_module || !shadow_frag_module) {
        veekay::app.running = false;
        return;
    }

    // --- Descriptor set layouts ---
    {
        VkDescriptorSetLayoutBinding global_bindings[4]{};
        // 0: scene UBO
        global_bindings[0].binding = 0;
        global_bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        global_bindings[0].descriptorCount = 1;
        global_bindings[0].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        // 1: model UBO dynamic
        global_bindings[1].binding = 1;
        global_bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        global_bindings[1].descriptorCount = 1;
        global_bindings[1].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        // 2: point lights SSBO
        global_bindings[2].binding = 2;
        global_bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        global_bindings[2].descriptorCount = 1;
        global_bindings[2].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        // 3: spot lights SSBO
        global_bindings[3].binding = 3;
        global_bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        global_bindings[3].descriptorCount = 1;
        global_bindings[3].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 4,
            .pBindings = global_bindings,
        };
        vkCreateDescriptorSetLayout(device, &info, nullptr, &global_desc_layout);

        // material layout
        VkDescriptorSetLayoutBinding mb[3]{};
        for (uint32_t i=0;i<3;++i) {
            mb[i].binding = i;
            mb[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            mb[i].descriptorCount = 1;
            mb[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo minfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3,
            .pBindings = mb,
        };
        vkCreateDescriptorSetLayout(device, &minfo, nullptr, &material_desc_layout);

        // shadow layout
        VkDescriptorSetLayoutBinding sb[3]{};
        for (uint32_t i=0;i<3;++i) {
            sb[i].binding = i;
            sb[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            sb[i].descriptorCount = 1;
            sb[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo sinfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3,
            .pBindings = sb,
        };
        vkCreateDescriptorSetLayout(device, &sinfo, nullptr, &shadow_desc_layout);
    }

    // --- Descriptor pools ---
    {
        VkDescriptorPoolSize sizes[4]{};
        sizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,          8};
        sizes[1] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,  8};
        sizes[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         16};
        sizes[3] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,128};

        VkDescriptorPoolCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 64,
            .poolSizeCount = 4,
            .pPoolSizes = sizes,
        };
        vkCreateDescriptorPool(device, &info, nullptr, &global_desc_pool);

        VkDescriptorPoolSize mat_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 128};
        VkDescriptorPoolCreateInfo minfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 128,
            .poolSizeCount = 1,
            .pPoolSizes = &mat_size,
        };
        vkCreateDescriptorPool(device, &minfo, nullptr, &material_desc_pool);
    }

    // --- Global descriptor set ---
    {
        VkDescriptorSetAllocateInfo alloc{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = global_desc_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &global_desc_layout,
        };
        vkAllocateDescriptorSets(device, &alloc, &global_desc_set);
    }

    // --- Shadow descriptor set ---
    {
        VkDescriptorSetAllocateInfo alloc{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = material_desc_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &shadow_desc_layout,
        };
        vkAllocateDescriptorSets(device, &alloc, &shadow_desc_set);

        VkDescriptorImageInfo img[3] = {
            {directional_shadow.sampler, directional_shadow.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {spotlight_shadows[0].sampler, spotlight_shadows[0].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {spotlight_shadows[1].sampler, spotlight_shadows[1].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        };
        VkWriteDescriptorSet w[3]{};
        for (int i=0;i<3;++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = shadow_desc_set;
            w[i].dstBinding = static_cast<uint32_t>(i);
            w[i].descriptorCount = 1;
            w[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[i].pImageInfo      = &img[i];
        }
        vkUpdateDescriptorSets(device, 3, w, 0, nullptr);
    }

    // --- UBO/SSBO ---
    scene_ubo = new veekay::graphics::Buffer(
        sizeof(SceneUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    );
    model_ubos = new veekay::graphics::Buffer(
        aligned_sizeof_model_ubo * max_models, nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    );
    point_ssbo = new veekay::graphics::Buffer(
        sizeof(PointLight)*max_point_lights, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    spot_ssbo = new veekay::graphics::Buffer(
        sizeof(SpotLight)*max_spotlights, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    // Обновление глобального descriptor set
    {
        VkDescriptorBufferInfo sbi{scene_ubo->buffer, 0, sizeof(SceneUniforms)};
        VkDescriptorBufferInfo mbi{model_ubos->buffer, 0, sizeof(ModelUniforms)};
        VkDescriptorBufferInfo pbi{point_ssbo->buffer, 0, sizeof(PointLight)*max_point_lights};
        VkDescriptorBufferInfo spbi{spot_ssbo->buffer, 0, sizeof(SpotLight)*max_spotlights};

        VkWriteDescriptorSet w[4]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = global_desc_set;
        w[0].dstBinding = 0;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[0].pBufferInfo = &sbi;

        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = global_desc_set;
        w[1].dstBinding = 1;
        w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        w[1].pBufferInfo = &mbi;

        w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[2].dstSet = global_desc_set;
        w[2].dstBinding = 2;
        w[2].descriptorCount = 1;
        w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[2].pBufferInfo = &pbi;

        w[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[3].dstSet = global_desc_set;
        w[3].dstBinding = 3;
        w[3].descriptorCount = 1;
        w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[3].pBufferInfo = &spbi;

        vkUpdateDescriptorSets(device, 4, w, 0, nullptr);
    }

    // --- Pipelines (shadow и основной) ---
    {
        // shadow pipeline layout
        VkPushConstantRange pc{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset     = 0,
            .size       = sizeof(ShadowPassPC),
        };
        VkPipelineLayoutCreateInfo li{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pc,
        };
        vkCreatePipelineLayout(device, &li, nullptr, &shadow_pipeline_layout);

        // shadow pipeline
        VkPipelineShaderStageCreateInfo sh_stages[2]{};
        sh_stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        sh_stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        sh_stages[0].module = shadow_vert_module;
        sh_stages[0].pName  = "main";
        sh_stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        sh_stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        sh_stages[1].module = shadow_frag_module;
        sh_stages[1].pName  = "main";

        VkVertexInputBindingDescription bind{0,sizeof(Vertex),VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attr{
            0,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,position)
        };
        VkPipelineVertexInputStateCreateInfo vi{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bind,
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions = &attr,
        };
        VkPipelineInputAssemblyStateCreateInfo ia{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };
        VkPipelineViewportStateCreateInfo vp{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount  = 1,
        };
        VkPipelineRasterizationStateCreateInfo rs{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = VK_TRUE,
            .depthBiasConstantFactor = 2.0f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 3.0f,
            .lineWidth = 1.0f,
        };
        VkPipelineMultisampleStateCreateInfo ms{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        };
        VkPipelineDepthStencilStateCreateInfo ds{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable  = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL,
        };
        VkDynamicState dyn_states[3] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS};
        VkPipelineDynamicStateCreateInfo dyn{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = 3,
            .pDynamicStates = dyn_states,
        };

        VkGraphicsPipelineCreateInfo pi{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages    = sh_stages,
            .pVertexInputState   = &vi,
            .pInputAssemblyState = &ia,
            .pViewportState      = &vp,
            .pRasterizationState = &rs,
            .pMultisampleState   = &ms,
            .pDepthStencilState  = &ds,
            .pDynamicState       = &dyn,
            .layout              = shadow_pipeline_layout,
            .renderPass          = shadow_render_pass,
            .subpass             = 0,
        };
        vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &shadow_pipeline);
    }

    {
        // основной pipeline layout
        VkDescriptorSetLayout sets[3] = {global_desc_layout, material_desc_layout, shadow_desc_layout};
        VkPipelineLayoutCreateInfo li{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 3,
            .pSetLayouts = sets,
        };
        vkCreatePipelineLayout(device, &li, nullptr, &pipeline_layout);

        VkPipelineShaderStageCreateInfo st[2]{};
        st[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        st[0].module= vert_module;
        st[0].pName = "main";
        st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        st[1].module= frag_module;
        st[1].pName = "main";

        VkVertexInputBindingDescription bind{0,sizeof(Vertex),VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attrs[3]{};
        attrs[0] = {0,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,position)};
        attrs[1] = {1,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,normal)};
        attrs[2] = {2,0,VK_FORMAT_R32G32_SFLOAT,offsetof(Vertex,uv)};
        VkPipelineVertexInputStateCreateInfo vi{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bind,
            .vertexAttributeDescriptionCount = 3,
            .pVertexAttributeDescriptions = attrs,
        };
        VkPipelineInputAssemblyStateCreateInfo ia{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };
        VkPipelineViewportStateCreateInfo vp{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount  = 1,
        };
        VkPipelineRasterizationStateCreateInfo rs{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f,
        };
        VkPipelineMultisampleStateCreateInfo ms{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        };
        VkPipelineDepthStencilStateCreateInfo ds{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable  = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL,
        };
        VkPipelineColorBlendAttachmentState ca{
            .blendEnable = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|
                              VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT,
        };
        VkPipelineColorBlendStateCreateInfo cb{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &ca,
        };
        VkDynamicState dyn_states[3] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS};
        VkPipelineDynamicStateCreateInfo dyn{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = 3,
            .pDynamicStates = dyn_states,
        };

        VkGraphicsPipelineCreateInfo pi{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages    = st,
            .pVertexInputState   = &vi,
            .pInputAssemblyState = &ia,
            .pViewportState      = &vp,
            .pRasterizationState = &rs,
            .pMultisampleState   = &ms,
            .pDepthStencilState  = &ds,
            .pColorBlendState    = &cb,
            .pDynamicState       = &dyn,
            .layout              = pipeline_layout,
            .renderPass          = veekay::app.vk_render_pass,
            .subpass             = 0,
        };
        vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline);
    }

    // ---- Текстуры и сцена ----
    {
        unsigned char white_px[] = {255,255,255,255};
        unsigned char black_px[] = {0,0,0,255};
        white_tex = new veekay::graphics::Texture(cmd, 1,1, VK_FORMAT_R8G8B8A8_UNORM, white_px);
        black_tex = new veekay::graphics::Texture(cmd, 1,1, VK_FORMAT_R8G8B8A8_UNORM, black_px);

        VkSampler linear_sampler = create_sampler(VK_FILTER_LINEAR,  VK_SAMPLER_ADDRESS_MODE_REPEAT);
        VkSampler nearest_sampler= create_sampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

        auto* meme  = load_texture(cmd, "assets/meme.png");
        auto* marmot = load_texture(cmd, "assets/marmot.png");
        auto* pig = load_texture(cmd, "assets/pig.png");

        plane_mesh  = create_plane();
        cube_mesh   = create_cube();
        sphere_mesh = create_sphere(30,30);

        // models.clear();
        // models.push_back({
        //     .mesh = plane_mesh,
        //     .transform = { {0,0,0}, {1,1,1}, {0,0,0} },
        //     .material = {{0.8f,0.8f,0.8f},0,{0.3f,0.3f,0.3f},32.0f},
        //     .textures = {meme,nullptr,nullptr,linear_sampler}
        // });
        // models.push_back({
        //     .mesh = cube_mesh,
        //     .transform = {{-2.0f,1.0f,0.0f},{0.8f,0.8f,0.8f},{0,0,0}},
        //     .material = {{1,1,1},0,{0.5f,0.5f,0.5f},64.0f},
        //     .textures = {pig,pig,nullptr,linear_sampler}
        // });
        // models.push_back({
        //     .mesh = cube_mesh,
        //     .transform = {{0.0f,1.0f,0.0f},{0.8f,0.8f,0.8f},{0,0,0}},
        //     .material = {{0.5f,0.3f,0.8f},0,{0.8f,0.8f,0.8f},64.0f},
        //     .textures = {meme,meme,nullptr,linear_sampler}
        // });
        // models.push_back({
        //     .mesh = sphere_mesh,
        //     .transform = {{0.0f,2.5f,2.0f},{1.2f,1.2f,1.2f},{0,0,0}},
        //     .material = {{0.9f,0.3f,0.2f},0,{1,1,1},128.0f},
        //     .textures = {nullptr,meme,marmot,nearest_sampler}
        // });
        models.push_back({
    .mesh = plane_mesh,
    .transform = { {0,0,0}, {4.0f,4.0f,1.0f}, {0,0,0} },
    .material = {{1.0f,1.0f,1.0f}, 0, {0.1f,0.1f,0.1f}, 8.0f},
    .textures = {meme,nullptr,nullptr,linear_sampler}
});

        // Центральная высокая сфера (основная тень)
        models.push_back({
            .mesh = create_sphere(40,40),
            .transform = { {0.0f,3.0f,1.0f}, {1.2f,1.2f,1.2f}, {0,0,0} },
            .material = {{0.9f,0.3f,0.2f},0,{1.0f,1.0f,1.0f},128.0f},
            .textures = {marmot,pig,nullptr,linear_sampler}
        });

        // Кубы для разных источников
        models.push_back({
            .mesh = cube_mesh,
            .transform = {{-3.0f,1.0f,-1.0f},{0.8f,0.8f,0.8f},{0,0,0}},
            .material = {{0.2f,0.6f,1.0f},0,{0.8f,0.8f,0.8f},64.0f},
            .textures = {pig,meme,nullptr,linear_sampler}
        });

        models.push_back({
            .mesh = cube_mesh,
            .transform = {{2.5f,1.5f,2.0f},{1.0f,1.0f,1.0f},{0,0,0}},
            .material = {{1.0f,0.5f,0.2f},0,{0.9f,0.9f,0.9f},32.0f},
            .textures = {meme,marmot,nullptr,linear_sampler}
        });

        // Маленькие сферы для градиента
        models.push_back({
            .mesh = create_sphere(20,20),
            .transform = {{-1.5f,0.8f,3.5f},{0.6f,0.6f,0.6f},{0,0,0}},
            .material = {{0.3f,0.9f,0.4f},0,{0.5f,0.5f,0.5f},16.0f},
            .textures = {nullptr,pig,nullptr,nearest_sampler}
        });

        for (auto& m : models)
            m.textures.descriptor_set = create_material_set(m.textures);

        // point_lights.clear();
        // point_lights.push_back({{2,2,2},0,{1.0f,0.8f,0.6f},5.0f});
        // point_lights.push_back({{-2,1.5f,-1},0,{0.3f,0.6f,1.0f},3.0f});
        //
        // spotlights.clear();
        // spotlights.push_back({
        //     {0,4,0},0,{0,-1,0},0,
        //     {1,1,1},10.0f,
        //     std::cos(glm::radians(12.5f)),
        //     std::cos(glm::radians(17.5f)),
        //     0,0
        // });
        // spotlights.push_back({
        //     {3,3,2},0,{-0.5f,-1, -0.3f},0,
        //     {1,0.5f,0.3f},8.0f,
        //     std::cos(glm::radians(15.0f)),
        //     std::cos(glm::radians(20.0f)),
        //     0,0
        // });
        point_lights.clear();
        // point_lights.push_back({{4.0f,3.0f,1.0f},0,{1.0f,0.7f,0.3f},8.0f});
        // point_lights.push_back({{-2.0f,2.5f,2.5f},0,{0.4f,0.8f,1.0f},6.0f});

        spotlights.clear();
        spotlights.push_back({
            {1.0f,4.5f,-2.0f},0,{0.0f,-1.0f,0.2f},0,
            {1.0f,1.0f,1.0f},12.0f,
            std::cos(glm::radians(15.0f)),
            std::cos(glm::radians(25.0f)),
            0,0
        });
        spotlights.push_back({
            {-2.5f,3.5f,4.0f},0,{0.3f,-0.8f,-0.4f},0,
            {0.9f,0.3f,0.6f},9.0f,
            std::cos(glm::radians(12.0f)),
            std::cos(glm::radians(22.0f)),
            0,0
        });
        camera.position           = {0.0f,2.0f,-5.0f};
        camera.transform_position = camera.position;
        camera.update_vectors();
    }
}

void update(double time) {
    static double last = time;
    float dt = float(time - last);
    last = time;

    ImGui::Begin("Controls");
    {
        ImGui::SeparatorText("Camera");
        int mode = (camera.mode == CameraMode::LookAt) ? 0 : 1;
        const char* modes[] = {"Look-At","Transform"};
        if (ImGui::Combo("Mode", &mode, modes, 2)) {
            camera.switch_mode(mode == 0 ? CameraMode::LookAt : CameraMode::Transform);
        }
        if (ImGui::Button("Save camera")) camera.save_state();
        ImGui::SameLine();
        if (ImGui::Button("Restore")) camera.restore_state();
        ImGui::SliderFloat("Speed", &camera.speed, 0.5f, 10.0f);
        ImGui::SliderFloat("Sensitivity", &camera.sensitivity, 0.01f, 0.5f);

        ImGui::SeparatorText("Shadows");
        ImGui::Checkbox("Enable shadows", &enable_shadows);
        ImGui::Checkbox("Dir light shadow", &enable_dir_shadow);
        ImGui::Checkbox("Spotlight shadows", &enable_spotlight_shadow);
        ImGui::SliderFloat("Shadow bias", &shadow_bias, 0.0005f, 0.02f, "%.4f");

        ImGui::SeparatorText("Directional light");
        ImGui::SliderFloat3("Direction", &dir_light_dir.x, -1.0f, 1.0f);
        ImGui::ColorEdit3("Color", &dir_light_color.x);
        ImGui::SliderFloat("Intensity", &dir_light_int, 0.0f, 3.0f);

        ImGui::SeparatorText("Point lights");
        for (size_t i = 0; i < point_lights.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            ImGui::Text("Point %zu", i);
            ImGui::SliderFloat3("Position", &point_lights[i].position.x, -10.0f, 10.0f);
            ImGui::ColorEdit3("Color", &point_lights[i].color.x);
            ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.0f, 30.0f);
            ImGui::PopID();
        }

        ImGui::SeparatorText("Spotlights");
        for (size_t i = 0; i < spotlights.size(); ++i) {
            ImGui::PushID(static_cast<int>(i) + 100);
            ImGui::Text("Spot %zu", i);
            ImGui::SliderFloat3("Position", &spotlights[i].position.x, -10.0f, 10.0f);
            ImGui::SliderFloat3("Direction", &spotlights[i].direction.x, -1.0f, 1.0f);
            ImGui::ColorEdit3("Color", &spotlights[i].color.x);
            ImGui::SliderFloat("Intensity", &spotlights[i].intensity, 0.0f, 40.0f);

            float inner_deg = glm::degrees(std::acos(spotlights[i].innerCutoff));
            float outer_deg = glm::degrees(std::acos(spotlights[i].outerCutoff));
            if (ImGui::SliderFloat("Inner angle", &inner_deg, 1.0f, 45.0f)) {
                spotlights[i].innerCutoff = std::cos(glm::radians(inner_deg));
            }
            if (ImGui::SliderFloat("Outer angle", &outer_deg, 1.0f, 60.0f)) {
                spotlights[i].outerCutoff = std::cos(glm::radians(outer_deg));
            }
            ImGui::PopID();
        }

        ImGui::SeparatorText("Texture FX");
        ImGui::SliderFloat("Effect strength", &texture_effect_strength, 0.0f, 1.0f);
    }
    ImGui::End();

    // Camera control
    if (veekay::input::keyboard::isKeyPressed(veekay::input::keyboard::Key::c)) {
        camera_enabled = !camera_enabled;
        veekay::input::mouse::setCaptured(camera_enabled);
        first_mouse = true;
    }

    if (camera_enabled) {
        float velocity = camera.speed * dt;
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
                camera.position.y += velocity;
            }
            if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::left_shift)) {
                camera.position.y -= velocity;
            }

            auto mouse_pos = veekay::input::mouse::cursorPosition();
            if (first_mouse) {
                last_mouse_x = mouse_pos.x;
                last_mouse_y = mouse_pos.y;
                first_mouse = false;
            }

            float xoffset = -(mouse_pos.x - last_mouse_x);
            float yoffset = (last_mouse_y - mouse_pos.y);
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

            if (camera.transform_rotation.x > glm::radians(89.0f)) camera.transform_rotation.x = glm::radians(89.0f);
            if (camera.transform_rotation.x < glm::radians(-89.0f)) camera.transform_rotation.x = glm::radians(-89.0f);
        }
    }
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
    VkDevice device = veekay::app.vk_device;

    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &bi);

    glm::mat4 dir_mat{};
    glm::mat4 spot_mats[max_shadow_spotlights]{};
    std::vector<uint32_t> selected;

    if (enable_shadows) {
        if (enable_dir_shadow) {
            dir_mat = calc_dir_light_matrix();
            render_shadow_pass(cmd, directional_shadow, dir_mat);
        }
        if (enable_spotlight_shadow && !spotlights.empty()) {
            selected = select_spotlights(camera.get_position());
            for (size_t i=0;i<selected.size();++i) {
                spot_mats[i] = calc_spot_matrix(spotlights[selected[i]]);
                render_shadow_pass(cmd, spotlight_shadows[i], spot_mats[i]);
            }
        }
    }

    // === Layout transition для shadow map ===
    auto transition_shadow = [&](VkImage image) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
    };

    if (enable_shadows) {
        if (enable_dir_shadow) {
            transition_shadow(directional_shadow.image);
        }
        if (enable_spotlight_shadow) {
            for (uint32_t i = 0; i < max_shadow_spotlights; ++i) {
                transition_shadow(spotlight_shadows[i].image);
            }
        }
    }

    float aspect = float(veekay::app.window_width) / float(veekay::app.window_height);
    SceneUniforms su{};
    su.view_projection = camera.view_projection(aspect);
    su.camera_position = camera.get_position();
    su.directional_light_direction = glm::normalize(dir_light_dir);
    su.directional_light_color     = dir_light_color;
    su.directional_light_intensity = dir_light_int;
    su.ambient_light = ambient_light;
    su.num_point_lights = static_cast<uint32_t>(point_lights.size());
    su.num_spotlights   = static_cast<uint32_t>(spotlights.size());
    su.time             = float(GetTimeAsDouble());
    su.effect_strength  = texture_effect_strength;
    su.directional_light_space_matrix = dir_mat;
    su.active_shadow_sources = 0;
    su.shadow_bias = shadow_bias;

    if (enable_shadows && enable_dir_shadow)
        su.active_shadow_sources |= (1u << 0);  // Бит 0 для directional

    if (enable_shadows && enable_spotlight_shadow) {
        auto selected = select_spotlights(camera.get_position());
        for (size_t i = 0; i < selected.size() && i < max_shadow_spotlights; i++) {
            su.spotlight_space_matrices[i] = spot_mats[i];
            su.active_shadow_sources |= (1u << (i + 1));
        }
    }
    scene_ubo->copy_to(&su, sizeof(su));

    if (!point_lights.empty())
        point_ssbo->copy_to(point_lights.data(),
                            point_lights.size()*sizeof(PointLight));

    if (!spotlights.empty())
        spot_ssbo->copy_to(spotlights.data(),
                           spotlights.size()*sizeof(SpotLight));

    for (size_t i=0;i<models.size();++i) {
        ModelUniforms mu{};
        mu.model         = models[i].transform.matrix();
        mu.albedo_color  = models[i].material.albedo;
        mu.specular_color= models[i].material.specular;
        mu.shininess     = models[i].material.shininess;
        model_ubos->copy_to(&mu, sizeof(mu), i*aligned_sizeof_model_ubo);
    }

    VkClearValue clr[2];
    clr[0].color = {{0.1f,0.1f,0.15f,1.0f}};
    clr[1].depthStencil = {1.0f,0};

    VkRenderPassBeginInfo rp{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = veekay::app.vk_render_pass,
        .framebuffer = framebuffer,
        .renderArea = {{0,0},{veekay::app.window_width,veekay::app.window_height}},
        .clearValueCount = 2,
        .pClearValues = clr,
    };
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{
        0,0,
        float(veekay::app.window_width),
        float(veekay::app.window_height),
        0.0f,1.0f
    };
    vkCmdSetViewport(cmd,0,1,&vp);
    VkRect2D sc{{0,0},{veekay::app.window_width,veekay::app.window_height}};
    vkCmdSetScissor(cmd,0,1,&sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    for (size_t i=0;i<models.size();++i) {
        uint32_t offset = static_cast<uint32_t>(i*aligned_sizeof_model_ubo);

        VkDescriptorSet sets[3] = {
            global_desc_set,
            models[i].textures.descriptor_set,
            shadow_desc_set
        };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout, 0, 3, sets, 1, &offset);

        VkBuffer vb[] = {models[i].mesh.vertex_buffer->buffer};
        VkDeviceSize off[] = {0};
        vkCmdBindVertexBuffers(cmd,0,1,vb,off);
        vkCmdBindIndexBuffer(cmd, models[i].mesh.index_buffer->buffer,
                             0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, models[i].mesh.index_count, 1, 0,0,0);
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

void shutdown() {
    VkDevice device = veekay::app.vk_device;

    // Shadow maps
    vkDestroyFramebuffer(device, directional_shadow.framebuffer, nullptr);
    for (uint32_t i=0;i<max_shadow_spotlights;++i)
        vkDestroyFramebuffer(device, spotlight_shadows[i].framebuffer, nullptr);

    vkDestroySampler(device, directional_shadow.sampler, nullptr);
    vkDestroyImageView(device, directional_shadow.view, nullptr);
    vkFreeMemory(device, directional_shadow.memory, nullptr);
    vkDestroyImage(device, directional_shadow.image, nullptr);
    for (uint32_t i=0;i<max_shadow_spotlights;++i) {
        vkDestroySampler(device, spotlight_shadows[i].sampler, nullptr);
        vkDestroyImageView(device, spotlight_shadows[i].view, nullptr);
        vkFreeMemory(device, spotlight_shadows[i].memory, nullptr);
        vkDestroyImage(device, spotlight_shadows[i].image, nullptr);
    }
    vkDestroyRenderPass(device, shadow_render_pass, nullptr);

    delete scene_ubo;
    delete model_ubos;
    delete point_ssbo;
    delete spot_ssbo;

    delete plane_mesh.vertex_buffer;
    delete plane_mesh.index_buffer;
    delete cube_mesh.vertex_buffer;
    delete cube_mesh.index_buffer;
    delete sphere_mesh.vertex_buffer;
    delete sphere_mesh.index_buffer;

    delete white_tex;
    delete black_tex;

    vkDestroyPipeline(device, shadow_pipeline, nullptr);
    vkDestroyPipelineLayout(device, shadow_pipeline_layout, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);

    vkDestroyDescriptorPool(device, material_desc_pool, nullptr);
    vkDestroyDescriptorSetLayout(device, material_desc_layout, nullptr);

    vkDestroyDescriptorPool(device, global_desc_pool, nullptr);
    vkDestroyDescriptorSetLayout(device, global_desc_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, shadow_desc_layout, nullptr);

    vkDestroyShaderModule(device, vert_module, nullptr);
    vkDestroyShaderModule(device, frag_module, nullptr);
    vkDestroyShaderModule(device, shadow_vert_module, nullptr);
    vkDestroyShaderModule(device, shadow_frag_module, nullptr);
}

} // namespace

int main() {
    veekay::ApplicationInfo info;
    info.init     = initialize;
    info.update   = update;
    info.render   = render;
    info.shutdown = shutdown;
    return veekay::run(info);
}
