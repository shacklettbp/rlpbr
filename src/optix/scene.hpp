#pragma once

#include <rlpbr_core/common.hpp>
#include <rlpbr_core/scene.hpp>
#include <rlpbr_core/utils.hpp>

#include "shader.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

#include <atomic>
#include <mutex>
#include <unordered_map>

namespace RLpbr {
namespace optix {

class TextureManager;

struct TextureBacking {
    TextureBacking(cudaArray_t m, cudaTextureObject_t h);

    cudaArray_t mem;
    cudaTextureObject_t hdl;

    std::atomic_uint32_t refCount;
};

using TextureRefType =
    std::unordered_map<std::string, TextureBacking>::iterator;

class Texture {
public:
    Texture(TextureManager &mgr, const TextureRefType &r,
            cudaTextureObject_t hdl);
    Texture(const Texture &) = delete;
    Texture(Texture &&);
    ~Texture();

    inline cudaTextureObject_t getHandle() const { return hdl_; }

private:
    TextureManager &mgr_;
    TextureRefType ref_;
    cudaTextureObject_t hdl_;
};

class TextureManager {
public:
    TextureManager();
    ~TextureManager();

    std::vector<Texture> load(
        const TextureInfo &tex_info, cudaStream_t copy_strm);

private:
    void decrementTextureRef(const TextureRefType &tex_ref);

    std::mutex cache_lock_;
    std::unordered_map<std::string, TextureBacking> loaded_;

    friend class Texture;
};

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

struct OptixScene : public Scene {
    OptixScene(const OptixScene &) = delete;
    OptixScene(OptixScene &&) = delete;
    ~OptixScene();

    CUdeviceptr sceneStorage;
    const PackedVertex *vertexPtr;
    const uint32_t *indexPtr;
    const PackedMaterial *materialPtr;

    std::vector<CUdeviceptr> blasStorage;
    std::vector<OptixTraversableHandle> blases;
    TLAS defaultTLAS;

    std::vector<Texture> textures;
    const cudaTextureObject_t *texturePtr;
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

    TLASIntermediate queueTLASRebuild(const Environment &env, OptixDeviceContext ctx,
                          cudaStream_t strm);

    CUdeviceptr tlasStorage;
    OptixTraversableHandle tlas;

    std::vector<PackedLight> lights;
};

class OptixLoader : public LoaderBackend {
public:
    OptixLoader(OptixDeviceContext ctx, TextureManager &texture_mgr);

    std::shared_ptr<Scene> loadScene(SceneLoadData &&load_info);

private:
    cudaStream_t stream_;
    OptixDeviceContext ctx_;
    TextureManager &texture_mgr_;
};

}
}
