#pragma once

#include <rlpbr/config.hpp>
#include <rlpbr_core/scene.hpp>

#include <filesystem>
#include <list>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_map>

#include "descriptors.hpp"
#include "utils.hpp"
#include "core.hpp"
#include "memory.hpp"
#include "shader.hpp"

// Forward declare ktxTexture as kind of an opaque backing data type
struct ktxTexture;

namespace RLpbr {
namespace vk {

struct VulkanScene;

struct BLAS {
    VkAccelerationStructureKHR hdl;
    VkDeviceAddress devAddr;
};

struct BLASData {
public:
    BLASData(const BLASData &) = delete;
    BLASData(BLASData &&) = default;
    ~BLASData();

    const DeviceState &dev;
    std::vector<BLAS> accelStructs;
    LocalBuffer storage;
};

struct TLAS {
    VkAccelerationStructureKHR hdl;
    std::optional<HostBuffer> buildStorage;
    uint32_t numBuildInstances;

    std::optional<LocalBuffer> tlasStorage;
    VkDeviceAddress tlasStorageDevAddr;
    size_t numStorageBytes;

    void build(const DeviceState &dev,
               MemoryAllocator &alloc,
               const std::vector<ObjectInstance> &instances,
               const std::vector<InstanceTransform> &instance_transforms,
               const std::vector<InstanceFlags> &instance_flags,
               const std::vector<ObjectInfo> &objects,
               const BLASData &blases,
               VkCommandBuffer build_cmd);

    void free(const DeviceState &dev);
};

struct ReservoirGrid {
    AABB bbox;
    VkDeviceMemory storage;
    VkDeviceAddress devAddr;
    LocalBuffer grid;
};

struct VulkanEnvironment : public EnvironmentBackend {
    VulkanEnvironment(const DeviceState &dev, MemoryAllocator &alloc,
                      const VulkanScene &scene);
    ~VulkanEnvironment();

    uint32_t addLight(const glm::vec3 &position, const glm::vec3 &color);

    void removeLight(uint32_t light_idx);

    std::vector<PackedLight> lights;

    const DeviceState &dev;
    TLAS tlas;

    ReservoirGrid reservoirGrid;
};

struct TextureData {
    TextureData(const DeviceState &d, MemoryAllocator &a);
    TextureData(const TextureData &) = delete;
    TextureData(TextureData &&);
    ~TextureData();

    const DeviceState &dev;
    MemoryAllocator &alloc;

    VkDeviceMemory memory;
    std::vector<LocalTexture> textures;
    std::vector<VkImageView> views;
};

struct SceneDescriptorBindings {
    uint32_t vertexBinding;
    uint32_t indexBinding;
    uint32_t textureBinding;
    uint32_t materialBinding;
    uint32_t meshInfoBinding;
};

struct VulkanScene : public Scene {
    TextureData textures;
    DescriptorSet descSet;

    LocalBuffer data;
    VkDeviceSize indexOffset;
    uint32_t numMeshes;

    BLASData blases;
};

class VulkanLoader : public LoaderBackend {
public:
    VulkanLoader(const DeviceState &dev,
                 MemoryAllocator &alloc,
                 const QueueState &transfer_queue,
                 const QueueState &render_queue,
                 const ShaderPipeline &shader,
                 const SceneDescriptorBindings &binding_defn,
                 uint32_t render_qf,
                 uint32_t max_texture_resolution);

    std::shared_ptr<Scene> loadScene(SceneLoadData &&load_info);

private:
    const DeviceState &dev;
    MemoryAllocator &alloc;
    const QueueState &transfer_queue_;
    const QueueState &render_queue_;

    VkCommandPool transfer_cmd_pool_;
    VkCommandBuffer transfer_cmd_;
    VkCommandPool render_cmd_pool_;
    VkCommandBuffer render_cmd_;

    VkSemaphore transfer_sema_;
    VkFence fence_;

    DescriptorManager desc_mgr_;
    SceneDescriptorBindings binding_defn_;

    uint32_t render_qf_;
    uint32_t max_texture_resolution_;
};

}
}
