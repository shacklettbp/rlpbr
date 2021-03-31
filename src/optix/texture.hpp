#pragma once

#include <rlpbr_core/scene.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

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

enum class TextureFormat {
    R8G8B8A8_SRGB,
    R8G8B8A8_UNORM,
    R8G8_UNORM,
    R8_UNORM,
    R32G32B32A32_SFLOAT,
    R32_SFLOAT,
};

class TextureManager {
public:
    TextureManager();
    ~TextureManager();

    template <typename LoadFn>
    inline Texture load(const std::string &lookup_key,
                        TextureFormat fmt,
                        cudaTextureAddressMode edge_mode,
                        cudaStream_t copy_strm,
                        LoadFn load_fn);

private:
    void decrementTextureRef(const TextureRefType &tex_ref);

    std::mutex cache_lock_;
    std::unordered_map<std::string, TextureBacking> loaded_;

    friend class Texture;
};

}
}

#include "texture.inl"
