#pragma once

#include "utils.hpp"

#include <cuda_fp16.h>
#include <iostream>

namespace RLpbr {
namespace optix {

[[maybe_unused]]
static std::pair<cudaArray_t, cudaTextureObject_t> load1DTexture(
    void *data, float len, TextureFormat fmt, 
    cudaTextureAddressMode edge_mode, cudaStream_t cpy_strm)
{
    if (fmt != TextureFormat::R32_SFLOAT) {
        std::cerr << 
            "Only floating point 1D textures currently supported" <<
            std::endl;
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
        tex_mem,
        hdl,
    };
}

[[maybe_unused]]
static std::pair<cudaArray_t, cudaTextureObject_t> load2DTexture(
    void *data, glm::u32vec2 dims, TextureFormat fmt,
    cudaTextureAddressMode edge_mode, cudaStream_t cpy_strm)
{
    cudaChannelFormatDesc channel_desc;
    uint32_t num_bytes_per_elem;
    bool normalized;
    if (fmt == TextureFormat::R8G8B8A8_SRGB) {
        channel_desc = cudaCreateChannelDesc<uchar4>();
        num_bytes_per_elem = 4;
        normalized = true;
    } else if (fmt == TextureFormat::R8G8B8A8_UNORM) {
        channel_desc = cudaCreateChannelDesc<uchar4>();
        num_bytes_per_elem = 4;
        normalized = true;
    } else if (fmt == TextureFormat::R8G8_UNORM) {
        channel_desc = cudaCreateChannelDesc<uchar2>();
        num_bytes_per_elem = 2;
        normalized = true;
    } else if (fmt == TextureFormat::R8_UNORM) {
        channel_desc = cudaCreateChannelDesc<uchar1>();
        num_bytes_per_elem = 1;
        normalized = true;
    } else if (fmt == TextureFormat::R32G32B32A32_SFLOAT) {
        channel_desc = cudaCreateChannelDesc<float4>();
        num_bytes_per_elem = 4 * sizeof(float);
        normalized = false;
    } else if (fmt == TextureFormat::R32_SFLOAT) {
        channel_desc = cudaCreateChannelDesc<float>();
        num_bytes_per_elem = sizeof(float);
        normalized = false;
    }

    cudaArray_t tex_mem;
    REQ_CUDA(cudaMallocArray(&tex_mem, &channel_desc, dims.x, dims.y,
                             cudaArrayDefault));

    REQ_CUDA(cudaMemcpy2DToArrayAsync(tex_mem, 0, 0, data,
        num_bytes_per_elem * dims.x, num_bytes_per_elem * dims.x, dims.y,
        cudaMemcpyHostToDevice, cpy_strm)); 

    cudaResourceDesc res_desc {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = tex_mem;

    cudaTextureDesc tex_desc {};
    tex_desc.addressMode[0] = edge_mode;
    tex_desc.addressMode[1] = edge_mode;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = normalized ? cudaReadModeNormalizedFloat :
        cudaReadModeElementType;
    tex_desc.normalizedCoords = true;
    tex_desc.sRGB = (fmt == TextureFormat::R8G8B8A8_SRGB);

    cudaTextureObject_t hdl;
    REQ_CUDA(cudaCreateTextureObject(&hdl, &res_desc, &tex_desc,
                                     nullptr));
    return {
        tex_mem,
        hdl,
    };
}

[[maybe_unused]]
static std::pair<cudaArray_t, cudaTextureObject_t> load3DTexture(
    void *data, glm::u32vec3 dims, TextureFormat fmt,
    cudaTextureAddressMode edge_mode, cudaStream_t cpy_strm)
{
    if (fmt != TextureFormat::R32_SFLOAT) {
        std::cerr << 
            "Only floating point volume textures currently supported" <<
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
        tex_mem,
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

        auto [host_ptr, dims] = load_fn(lookup_key);

        cudaArray_t tex_mem;
        cudaTextureObject_t tex_hdl;
        if constexpr (is_same_v<decltype(dims), glm::u32vec3>) {
            tie(tex_mem, tex_hdl) =
                load3DTexture(host_ptr, dims, fmt, edge_mode, copy_strm);
        } else if constexpr (is_same_v<decltype(dims), glm::u32vec1>) {
            tie(tex_mem, tex_hdl) =
                load1DTexture(host_ptr, dims.x, fmt, edge_mode, copy_strm);
        } else {
            tie(tex_mem, tex_hdl) =
                load2DTexture(host_ptr, dims, fmt, edge_mode, copy_strm);
        }

        cache_lock_.lock();

        auto res = loaded_.emplace(piecewise_construct,
                                   forward_as_tuple(lookup_key),
                                   forward_as_tuple(tex_mem, tex_hdl));
        iter = res.first;
        iter->second.refCount.fetch_add(1, memory_order_acq_rel);

        cache_lock_.unlock();

        if (!res.second) {
            cudaDestroyTextureObject(tex_hdl);
            cudaFreeArray(tex_mem);
        }

    } else {
        iter->second.refCount.fetch_add(1, memory_order_acq_rel);

        cache_lock_.unlock();
    }

    return Texture(*this, iter, iter->second.hdl);
}

}
}
