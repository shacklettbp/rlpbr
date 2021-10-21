#pragma once

#include "common.hpp"
#include "utils.hpp"
#include "physics.hpp"
#include "device.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <filesystem>
#include <fstream>
#include <functional>
#include <string_view>
#include <variant>
#include <vector>

namespace RLpbr {

struct LoaderBackend;

struct alignas(16) Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct alignas(16) PackedVertex {
    glm::vec3 position;
    glm::vec3 normalTangentPacked;
    glm::vec2 uv;
};

struct alignas(16) MaterialParams {
    glm::u8vec3 baseColor; // 0 - 1
    uint8_t baseTransmission; // 0 - 1
    uint8_t specularScale; // 0 - 1
    uint8_t ior; // 1.0 - 2.5 (float(i) / 170.f + 1.f)
    uint8_t baseMetallic;
    uint8_t baseRoughness;
    glm::u16vec3 baseSpecular; // fp16, could cut down to RGBM
    uint16_t flags;

    uint8_t clearcoat; 
    uint8_t clearcoatRoughness;
    glm::u8vec3 attenuationColor;
    uint8_t anisoScale;
    uint8_t anisoRotation;
    uint16_t attenuationDistance;
    glm::u16vec3 baseEmittance;
};

enum class LightType : uint32_t {
    Sphere,
    Triangle,
    Portal,
};

struct LightProperties {
    LightType type;
    union {
        struct {
            uint32_t sphereVertIdx;
            uint32_t sphereMatIdx;
            float radius;
        };
        struct {
            uint32_t triIdxOffset;
            uint32_t triMatIdx;
        };
        struct {
            uint32_t portalIdxOffset;
        };
    };
};

struct EnvironmentInit {
    EnvironmentInit(const AABB &bbox,
                    std::vector<ObjectInstance> instances,
                    std::vector<uint32_t> instance_materials,
                    std::vector<InstanceTransform> transforms,
                    std::vector<InstanceFlags> instance_flags,
                    std::vector<LightProperties> lights);

    AABB defaultBBox;
    std::vector<ObjectInstance> defaultInstances;
    std::vector<uint32_t> defaultInstanceMaterials;
    std::vector<InstanceTransform> defaultTransforms;
    std::vector<InstanceFlags> defaultInstanceFlags;

    std::vector<uint32_t> indexMap;
    std::vector<uint32_t> reverseIDMap;

    std::vector<LightProperties> lights;
    std::vector<uint32_t> lightIDs;
    std::vector<uint32_t> lightReverseIDs;
};

struct ObjectInfo {
    uint32_t meshIndex;
    uint32_t numMeshes;
};

struct alignas(16) MeshInfo {
    uint32_t indexOffset;
    uint32_t numTriangles;
    uint32_t numVertices;
};

struct TextureInfo {
    std::string textureDir;
    std::vector<std::string> base;
    std::vector<std::string> metallicRoughness;
    std::vector<std::string> specular;
    std::vector<std::string> normal;
    std::vector<std::string> emittance;
    std::vector<std::string> transmission;
    std::vector<std::string> clearcoat;
    std::vector<std::string> anisotropic;
    std::string envMap;
};

struct MaterialTextures {
    uint32_t baseColorIdx; // SRGB 4 Component (RGB + transmission)
    uint32_t metallicRoughnessIdx; // Linear 2 Component (BC 5)
    uint32_t specularIdx; // SRGB 4 Component
    uint32_t normalIdx; // Linear 2 Component (BC 5)
    uint32_t emittanceIdx; // SRGB 3 Component
    uint32_t transmissionIdx; // Linear 1 Component
    uint32_t clearcoatIdx; // Linear 2 Component (BC 5)
    uint32_t anisoIdx; // Linear 2 Component (BC 5) (Vector Map)
};

struct MaterialMetadata {
    TextureInfo textureInfo;
    std::vector<MaterialParams> materialParams;
    std::vector<MaterialTextures> materialTextures;
};

struct StagingHeader {
    uint32_t numMeshes;
    uint32_t numObjects;
    uint32_t numVertices;
    uint32_t numIndices;
    uint32_t numMaterials;

    uint64_t indexOffset;
    uint64_t meshOffset;
    uint64_t objectOffset;
    uint64_t materialOffset;

    uint64_t physicsOffset;
    
    uint64_t totalBytes;
};

struct SceneLoadData {
    StagingHeader hdr;
    std::vector<MeshInfo> meshInfo;
    std::vector<ObjectInfo> objectInfo;
    TextureInfo textureInfo;
    std::vector<MaterialTextures> textureIndices;
    EnvironmentInit envInit;
    PhysicsMetadata physics;

    std::variant<std::ifstream, std::vector<char>> data;

    static SceneLoadData loadFromDisk(std::string_view scene_path,
                                      bool load_full_file = false);
};

struct Scene {
    std::vector<MeshInfo> meshInfo;
    std::vector<ObjectInfo> objectInfo;
    EnvironmentInit envInit;
};

}
