#pragma once

#include "utils.hpp"

#include <cuda_fp16.h>
#include <iostream>
#include <cassert>

namespace RLpbr {
namespace optix {

[[maybe_unused]]
static std::pair<TextureMemory , cudaTextureObject_t> load1DTexture(
    void *data, float len, uint32_t num_levels, TextureFormat fmt, 
    cudaTextureAddressMode edge_mode, cudaStream_t cpy_strm)
{
    if (fmt != TextureFormat::R32_SFLOAT) {
        std::cerr << 
            "Only floating point 1D textures currently supported" <<
            std::endl;
        std::abort();
    }
    if (num_levels != 1) {
        std::cerr << "1D mipmaps not currently supported" << std::endl;
        std::abort();
    }

    auto channel_desc = cudaCreateChannelDesc<float>();

    cudaArray_t tex_mem;
    REQ_CUDA(cudaMallocArray(&tex_mem, &channel_desc, len, 0,
                             cudaArrayDefault));

    REQ_CUDA(cudaMemcpy2DToArrayAsync(tex_mem, 0, 0, data, sizeof(float) * len,
        sizeof(float) * len, 1, cudaMemcpyHostToDevice, cpy_strm));

    cudaResourceDesc res_desc {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = tex_mem;

    cudaTextureDesc tex_desc {};
    tex_desc.addressMode[0] = edge_mode;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = true;
    tex_desc.sRGB = false;

    cudaTextureObject_t hdl;
    REQ_CUDA(cudaCreateTextureObject(&hdl, &res_desc, &tex_desc,
                                     nullptr));
    return {
        TextureMemory {
            false,
            tex_mem,
            0,
        },
        hdl,
    };
}

[[maybe_unused]]
static std::pair<TextureMemory, cudaTextureObject_t> load2DTexture(
    void *data, glm::u32vec2 dims, uint32_t num_levels,
    TextureFormat fmt, cudaTextureAddressMode edge_mode,
    cudaStream_t cpy_strm)
{
    cudaChannelFormatDesc channel_desc;
    uint32_t num_bytes_per_elem;
    bool normalized = true;
    bool compressed = false;
    bool srgb = false;
    if (fmt == TextureFormat::R8G8B8A8_SRGB) {
        channel_desc = cudaCreateChannelDesc<uchar4>();
        num_bytes_per_elem = 4;
        srgb = true;
    } else if (fmt == TextureFormat::R8G8B8A8_UNORM) {
        channel_desc = cudaCreateChannelDesc<uchar4>();
        num_bytes_per_elem = 4;
    } else if (fmt == TextureFormat::R8G8_UNORM) {
        channel_desc = cudaCreateChannelDesc<uchar2>();
        num_bytes_per_elem = 2;
    } else if (fmt == TextureFormat::R8_UNORM) {
        channel_desc = cudaCreateChannelDesc<uchar1>();
        num_bytes_per_elem = 1;
    } else if (fmt == TextureFormat::R32G32B32A32_SFLOAT) {
        channel_desc = cudaCreateChannelDesc<float4>();
        num_bytes_per_elem = 4 * sizeof(float);
        normalized = false;
    } else if (fmt == TextureFormat::R32_SFLOAT) {
        channel_desc = cudaCreateChannelDesc<float>();
        num_bytes_per_elem = sizeof(float);
        normalized = false;
    } else if (fmt == TextureFormat::BC7) {
        channel_desc = cudaCreateChannelDesc<uint4>();
        num_bytes_per_elem = 16;
        compressed = true;
        srgb = true;
    } else if (fmt == TextureFormat::BC5) {
        channel_desc = cudaCreateChannelDesc<uint4>();
        num_bytes_per_elem = 16;
        compressed = true;
    } else {
        assert(false);
    }

    if (compressed) {
        // Cuda block compressed support is broken, can't use mip tail
        uint32_t new_num_levels = 0;
        uint32_t cur_x = dims.x;
        uint32_t cur_y = dims.y;
        for (int i = 0; i < (int)num_levels; i++) {
            new_num_levels++;

            cur_x /= 2;
            cur_y /= 2;
            if (cur_x < 4 || cur_y < 4) break;
        }

        num_levels = new_num_levels;

        dims.x = (dims.x + 3) / 4;
        dims.y = (dims.y + 3) / 4;
    } 

    TextureMemory tex_mem;
    cudaResourceDesc res_desc {};

    if (num_levels > 1) {
        cudaExtent tex_extent = make_cudaExtent(dims.x, dims.y, 0);
        REQ_CUDA(cudaMallocMipmappedArray(&tex_mem.mipArr, &channel_desc,
            tex_extent, num_levels, cudaArrayDefault));
        tex_mem.mipmapped = true;

        uint8_t *cur_data = (uint8_t *)data;
        for (int i = 0; i < (int)num_levels; i++) {
            cudaArray_t level_arr;
            REQ_CUDA(cudaGetMipmappedArrayLevel(
                &level_arr, tex_mem.mipArr, i));

            cudaChannelFormatDesc fmt_desc;
            cudaExtent level_ext;
            REQ_CUDA(cudaArrayGetInfo(&fmt_desc, &level_ext, nullptr,
                                      level_arr));

            REQ_CUDA(cudaMemcpy2DToArrayAsync(level_arr, 0, 0, cur_data,
                num_bytes_per_elem * level_ext.width,
                num_bytes_per_elem * level_ext.width,
                level_ext.height, cudaMemcpyHostToDevice, cpy_strm));

            cur_data += num_bytes_per_elem * level_ext.width *
                level_ext.height;
        }

        res_desc.resType = cudaResourceTypeMipmappedArray;
        res_desc.res.mipmap.mipmap = tex_mem.mipArr;
    } else {
        REQ_CUDA(cudaMallocArray(&tex_mem.arr, &channel_desc, dims.x, dims.y,
            cudaArrayDefault));
        tex_mem.mipmapped = false;

        REQ_CUDA(cudaMemcpy2DToArrayAsync(tex_mem.arr, 0, 0, data,
            num_bytes_per_elem * dims.x, num_bytes_per_elem * dims.x, dims.y,
            cudaMemcpyHostToDevice, cpy_strm)); 

        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = tex_mem.arr;
    }

    cudaResourceViewDesc view_desc {};
    if (compressed) {
        if (fmt == TextureFormat::BC7) {
            view_desc.format = cudaResViewFormatUnsignedBlockCompressed7;
        } else if (fmt == TextureFormat::BC5) {
            view_desc.format = cudaResViewFormatUnsignedBlockCompressed5;
        }

        // FIXME: non multiple of 4 textures?
        view_desc.width = dims.x * 4;
        view_desc.height = dims.y * 4;
        view_desc.depth = 0;
        view_desc.firstMipmapLevel = 0;
        view_desc.lastMipmapLevel = num_levels - 1;
    }

    cudaTextureDesc tex_desc {};
    tex_desc.addressMode[0] = edge_mode;
    tex_desc.addressMode[1] = edge_mode;
    tex_desc.filterMode = compressed ?
        cudaFilterModePoint : cudaFilterModeLinear;
    tex_desc.readMode = normalized && !compressed ?
        cudaReadModeNormalizedFloat : cudaReadModeElementType;
    tex_desc.normalizedCoords = true;
    if (num_levels > 1 && !compressed) {
        tex_desc.mipmapFilterMode = cudaFilterModeLinear;
    }
    tex_desc.sRGB = srgb;

    cudaTextureObject_t hdl;
    REQ_CUDA(cudaCreateTextureObject(&hdl, &res_desc, &tex_desc,
                                     compressed ? &view_desc : nullptr));
    return {
        tex_mem,
        hdl,
    };
}

[[maybe_unused]]
static std::pair<TextureMemory, cudaTextureObject_t> load3DTexture(
    void *data, glm::u32vec3 dims, uint32_t num_levels, TextureFormat fmt,
    cudaTextureAddressMode edge_mode, cudaStream_t cpy_strm)
{
    if (fmt != TextureFormat::R32_SFLOAT) {
        std::cerr << 
            "Only floating point volume textures currently supported" <<
            std::endl;
        std::abort();
    }

    if (num_levels != 1) {
        std::cerr << "3D mipmaps not currently supported" <<
            std::endl;
        std::abort();
    }

    auto tex_extent = make_cudaExtent(dims.x, dims.y, dims.z);

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();

    cudaArray_t tex_mem;
    REQ_CUDA(cudaMalloc3DArray(&tex_mem, &channel_desc, tex_extent));

    cudaMemcpy3DParms cpy_params {};
    cpy_params.srcPtr = make_cudaPitchedPtr(data, dims.x * sizeof(float),
                                            dims.x, dims.y);
    cpy_params.dstArray = tex_mem;
    cpy_params.extent = tex_extent;
    cpy_params.kind = cudaMemcpyHostToDevice;

    REQ_CUDA(cudaMemcpy3DAsync(&cpy_params, cpy_strm));

    cudaResourceDesc res_desc {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = tex_mem;

    cudaTextureDesc tex_desc {};
    tex_desc.addressMode[0] = edge_mode;
    tex_desc.addressMode[1] = edge_mode;
    tex_desc.addressMode[2] = edge_mode;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = true;
    tex_desc.sRGB = false;

    cudaTextureObject_t hdl;
    REQ_CUDA(cudaCreateTextureObject(&hdl, &res_desc, &tex_desc,
                                     nullptr));

    return {
        TextureMemory {
            false,
            tex_mem,
            0,
        },
        hdl,
    };
}

template <typename LoadFn>
Texture TextureManager::load(const std::string &lookup_key,
                             TextureFormat fmt,
                             cudaTextureAddressMode edge_mode,
                             cudaStream_t copy_strm,
                             LoadFn load_fn)
{
    using namespace std;

    cache_lock_.lock();
    auto iter = loaded_.find(lookup_key);

    if (iter == loaded_.end()) {
        cache_lock_.unlock();

        auto [host_ptr, dims, num_levels] = load_fn(lookup_key);

        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t depth = 0;

        TextureMemory tex_mem;
        cudaTextureObject_t tex_hdl;
        if constexpr (is_same_v<decltype(dims), glm::u32vec3>) {
            tie(tex_mem, tex_hdl) =
                load3DTexture(host_ptr, dims, num_levels, fmt, edge_mode,
                              copy_strm);

            width = dims.x;
            height = dims.y;
            depth = dims.z;
        } else if constexpr (is_same_v<decltype(dims), glm::u32vec1>) {
            tie(tex_mem, tex_hdl) =
                load1DTexture(host_ptr, dims.x, num_levels, fmt, edge_mode,
                              copy_strm);

            width = dims.x;
        } else {
            tie(tex_mem, tex_hdl) =
                load2DTexture(host_ptr, dims, num_levels, fmt, edge_mode,
                              copy_strm);

            width = dims.x;
            height = dims.y;
        }

        cache_lock_.lock();

        auto res = loaded_.emplace(piecewise_construct,
                                   forward_as_tuple(lookup_key),
                                   forward_as_tuple(tex_mem, tex_hdl, width,
                                                    height, depth));
        iter = res.first;
        iter->second.refCount.fetch_add(1, memory_order_acq_rel);

        cache_lock_.unlock();

        if (!res.second) {
            cudaDestroyTextureObject(tex_hdl);
            if (tex_mem.mipmapped) {
                cudaFreeMipmappedArray(tex_mem.mipArr);
            } else {
                cudaFreeArray(tex_mem.arr);
            }
        }

    } else {
        iter->second.refCount.fetch_add(1, memory_order_acq_rel);

        cache_lock_.unlock();
    }

    return Texture(*this, iter, iter->second.hdl, iter->second.width,
                   iter->second.height, iter->second.depth);
}

}
}
