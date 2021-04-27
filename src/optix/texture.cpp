#include "texture.hpp"
#include "utils.hpp"

#include <iostream>

using namespace std;

namespace RLpbr {
namespace optix {

TextureBacking::TextureBacking(TextureMemory m, cudaTextureObject_t handle,
                               uint32_t w, uint32_t h, uint32_t d)
    : mem(m),
      hdl(handle),
      width(w),
      height(h),
      depth(d),
      refCount(0)
{}

Texture::Texture(TextureManager &mgr, const TextureRefType &r,
                 cudaTextureObject_t hdl, uint32_t width,
                 uint32_t height, uint32_t depth)
    : mgr_(mgr),
      ref_(r),
      hdl_(hdl),
      width_(width),
      height_(height),
      depth_(depth)
{}

Texture::Texture(Texture &&o)
    : mgr_(o.mgr_),
      ref_(o.ref_),
      hdl_(o.hdl_),
      width_(o.width_),
      height_(o.height_),
      depth_(o.depth_)
{
    o.hdl_ = 0;
}

Texture::~Texture()
{
    if (hdl_ == 0) return;
    mgr_.decrementTextureRef(ref_);
}

TextureManager::TextureManager()
    : cache_lock_(),
      loaded_()
{}

TextureManager::~TextureManager()
{
    if (!loaded_.empty()) {
        cerr << "Dangling references to textures" << endl;
        abort();
    }
}

void TextureManager::decrementTextureRef(const TextureRefType &tex_ref)
{
    TextureBacking &tex = tex_ref->second;

    if (tex.refCount.fetch_sub(1, memory_order_acq_rel) == 1) {
        cudaDestroyTextureObject(tex.hdl);
        if (tex.mem.mipmapped) {
            cudaFreeMipmappedArray(tex.mem.mipArr);
        } else {
            cudaFreeArray(tex.mem.arr);
        }

        cache_lock_.lock();

        if (tex.refCount.load(memory_order_acquire) == 0) {
            loaded_.erase(tex_ref);
        }

        cache_lock_.unlock();
    }
}

}
}
