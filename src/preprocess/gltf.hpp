#pragma once

#include "import.hpp"

#include <rlpbr_core/scene.hpp>

#include <glm/glm.hpp>
#include <simdjson.h>

#include <cstdint>
#include <optional>
#include <filesystem>
#include <string_view>
#include <vector>

namespace RLpbr {
namespace SceneImport {

struct GLTFBuffer {
    const uint8_t *dataPtr;
    std::string_view filePath;
};

struct GLTFBufferView {
    uint32_t bufferIdx;
    uint32_t offset;
    uint32_t stride;
    uint32_t numBytes;
};

enum class GLTFComponentType {
    UINT32,
    UINT16,
    FLOAT
};

struct GLTFAccessor {
    uint32_t viewIdx;
    uint32_t offset;
    uint32_t numElems;
    GLTFComponentType type;
};
    
enum class GLTFImageType {
    JPEG,
    PNG,
    BASIS,
    EXTERNAL
};

struct GLTFImage {
    GLTFImageType type;
    union {
        std::string_view filePath;
        uint32_t viewIdx;
    };
};

struct GLTFTexture {
    uint32_t sourceIdx;
    uint32_t samplerIdx;
};

struct GLTFMaterial {
    std::string name;
    uint32_t baseColorIdx;
    uint32_t metallicRoughnessIdx;
    uint32_t specularIdx;
    uint32_t normalIdx;
    uint32_t emittanceIdx;
    uint32_t transmissionIdx;
    uint32_t clearcoatIdx;
    uint32_t clearcoatNormalIdx;
    uint32_t anisoIdx;
    glm::vec4 baseColor;
    float transmissionFactor;
    glm::vec3 baseSpecular;
    float specularFactor;
    float metallic;
    float roughness;
    float ior;
    float clearcoat;
    float clearcoatRoughness;
    glm::vec3 attenuationColor;
    float attenuationDistance;
    float anisoScale;
    glm::vec3 anisoDir;
    glm::vec3 baseEmittance;
    bool thinwalled;
};

struct GLTFPrimitive {
    std::optional<uint32_t> positionIdx;
    std::optional<uint32_t> normalIdx;
    std::optional<uint32_t> uvIdx;
    std::optional<uint32_t> colorIdx;
    uint32_t indicesIdx;
    uint32_t materialIdx;
};

struct GLTFMesh {
    std::string name;
    std::vector<GLTFPrimitive> primitives;
};

struct GLTFNode {
    std::vector<uint32_t> children;
    uint32_t meshIdx;
    glm::mat4 transform;
};

struct GLTFScene {
    std::string sceneName;
    std::filesystem::path sceneDirectory;
    simdjson::dom::parser jsonParser;
    simdjson::dom::element root;
    std::vector<uint8_t> internalData;

    std::vector<GLTFBuffer> buffers;
    std::vector<GLTFBufferView> bufferViews;
    std::vector<GLTFAccessor> accessors;
    std::vector<GLTFImage> images;
    std::vector<GLTFTexture> textures;
    std::vector<GLTFMaterial> materials;
    std::vector<GLTFMesh> meshes;
    std::vector<GLTFNode> nodes;
    std::vector<uint32_t> rootNodes;
};

GLTFScene gltfLoad(const std::string_view gltf_path) noexcept;

template <typename VertexType, typename MaterialType>
SceneDescription<VertexType, MaterialType> parseGLTF(
    std::filesystem::path scene_path, const glm::mat4 &base_txfm,
    std::optional<std::string_view> texture_dir);

}
}
