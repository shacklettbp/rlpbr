#pragma once

#include "core.hpp"

#include <glm/glm.hpp>

#include <string>
#include <vector>

namespace RLpbr {
namespace vk {

namespace Shader {
using namespace glm;
using uint = uint32_t;

#include "shaders/shader_common.h"
};

using Shader::PackedInstance;
using Shader::PackedCamera;
using Shader::PackedLight;
using Shader::PackedEnv;
using Shader::RTPushConstant;

namespace VulkanConfig {

constexpr uint32_t max_materials = MAX_MATERIALS;
constexpr uint32_t textures_per_material = 8;
constexpr uint32_t max_lights = MAX_LIGHTS;
constexpr uint32_t max_instances = 10000000;
constexpr uint32_t compute_workgroup_size = WORKGROUP_SIZE;
constexpr uint32_t localWorkgroupX = LOCAL_WORKGROUP_X;
constexpr uint32_t localWorkgroupY = LOCAL_WORKGROUP_Y;
constexpr uint32_t localWorkgroupZ = LOCAL_WORKGROUP_Z;

}

struct BindingOverride {
    uint32_t setID;
    uint32_t bindingID;
    VkSampler sampler;
    uint32_t descriptorCount;
    VkDescriptorBindingFlags flags;
};

class ShaderPipeline {
public:
    ShaderPipeline(const DeviceState &dev,
                   const std::vector<std::string> &shader_paths,
                   const std::vector<BindingOverride> &binding_overrides,
                   const std::vector<std::string> &defines,
                   const char *shader_dir);
    ShaderPipeline(const ShaderPipeline &) = delete;
    ShaderPipeline(ShaderPipeline &&) = default;
    ~ShaderPipeline();

    static void initCompiler();

    inline VkShaderModule getShader(uint32_t idx) const
    {
        return shaders_[idx];
    }

    inline VkDescriptorSetLayout getLayout(uint32_t idx) const
    {
        return layouts_[idx];
    }

    VkDescriptorPool makePool(uint32_t set_id, uint32_t max_sets) const;

private:
    const DeviceState &dev;
    std::vector<VkShaderModule> shaders_;
    std::vector<VkDescriptorSetLayout> layouts_;
    std::vector<std::vector<VkDescriptorPoolSize>> base_pool_sizes_;
};

}
}
