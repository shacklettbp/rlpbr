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

struct TextureMemory {
    bool mipmapped;
    cudaArray_t arr;
    cudaMipmappedArray_t mipArr;
};

struct TextureBacking {
    TextureBacking(TextureMemory m, cudaTextureObject_t handle,
                   uint32_t w, uint32_t h, uint32_t d);

    TextureMemory mem;
    cudaTextureObject_t hdl;
    uint32_t width;
    uint32_t height;
    uint32_t depth;

    std::atomic_uint32_t refCount;
};

using TextureRefType =
    std::unordered_map<std::string, TextureBacking>::iterator;

class Texture {
public:
    Texture(TextureManager &mgr, const TextureRefType &r,
            cudaTextureObject_t hdl, uint32_t width,
            uint32_t height, uint32_t depth);
    Texture(const Texture &) = delete;
    Texture(Texture &&);
    ~Texture();

    inline cudaTextureObject_t getHandle() const { return hdl_; }
    inline uint32_t getWidth() const { return width_; }
    inline uint32_t getHeight() const { return height_; }
    inline uint32_t getDepth() const { return depth_; }

private:
    TextureManager &mgr_;
    TextureRefType ref_;
    cudaTextureObject_t hdl_;
    uint32_t width_;
    uint32_t height_;
    uint32_t depth_;
};

enum class TextureFormat {
    R8G8B8A8_SRGB,
    R8G8B8A8_UNORM,
    R8G8_UNORM,
    R8_UNORM,
    R32G32B32A32_SFLOAT,
    R32_SFLOAT,
    BC7,
    BC5,
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
