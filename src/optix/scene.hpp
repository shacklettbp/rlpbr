#pragma once

#include <rlpbr_core/common.hpp>
#include <rlpbr_core/scene.hpp>
#include <rlpbr_core/utils.hpp>

#include "shader.hpp"
#include "physics.hpp"
#include "texture.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

namespace RLpbr {
namespace optix {

struct TLASIntermediate {
    void *instanceTransforms;
    void *buildScratch;

    void free();
};

struct TLAS {
    OptixTraversableHandle hdl;
    CUdeviceptr storage;
    size_t numBytes;
    OptixTraversableHandle *instanceBLASes;
};

struct LoadedTextures {
    std::vector<Texture> base;
    std::vector<Texture> metallicRoughness;
    std::vector<Texture> specular;
    std::vector<Texture> normal;
    std::vector<Texture> emittance;
    std::vector<Texture> transmission;
    std::vector<Texture> clearcoat;
    std::vector<Texture> anisotropic;
    std::optional<Texture> envMap;
};


struct OptixScene : public Scene {
    OptixScene(const OptixScene &) = delete;
    OptixScene(OptixScene &&) = delete;
    ~OptixScene();

    CUdeviceptr sceneStorage;
    const DevicePackedVertex *vertexPtr;
    const uint32_t *indexPtr;
    const PackedMaterial *materialPtr;
    const PackedMeshInfo *meshPtr;

    std::vector<CUdeviceptr> blasStorage;
    std::vector<OptixTraversableHandle> blases;

    TLAS defaultTLAS;
    InstanceTransform *defaultTransformBuffer;

    LoadedTextures textures;
    const cudaTextureObject_t *texturePtr;
    const TextureSize *textureDimsPtr;

    std::optional<ScenePhysicsData> physics;
};

class OptixEnvironment : public EnvironmentBackend {
public:
    static OptixEnvironment make(OptixDeviceContext ctx,
                                 cudaStream_t build_stream,
                                 const OptixScene &scene);

    OptixEnvironment(const OptixEnvironment &) = delete;
    OptixEnvironment(OptixEnvironment &&) = delete;
    ~OptixEnvironment();

    uint32_t addLight(const glm::vec3 &position,
                      const glm::vec3 &color);

    void removeLight(uint32_t light_idx);

    TLASIntermediate queueTLASRebuild(const Environment &env,
        OptixDeviceContext ctx, cudaStream_t strm);

    CUdeviceptr tlasStorage;
    OptixTraversableHandle tlas;
    InstanceTransform *transformBuffer;

    std::vector<PackedLight> lights;
    std::vector<InstanceFlags> instanceFlags;

    std::optional<PhysicsEnvironment> physics;
};

class OptixLoader : public LoaderBackend {
public:
    OptixLoader(OptixDeviceContext ctx, TextureManager &texture_mgr,
                bool need_physics);

    std::shared_ptr<Scene> loadScene(SceneLoadData &&load_info);

private:
    cudaStream_t stream_;
    OptixDeviceContext ctx_;
    TextureManager &texture_mgr_;

    bool need_physics_;
};

}
}
