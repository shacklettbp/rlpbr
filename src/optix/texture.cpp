#include "texture.hpp"
#include "utils.hpp"

#include <iostream>

using namespace std;

namespace RLpbr {
namespace optix {

TextureBacking::TextureBacking(cudaArray_t m, cudaTextureObject_t h)
    : mem(m),
      hdl(h),
      refCount(0)
{}

Texture::Texture(TextureManager &mgr, const TextureRefType &r,
                 cudaTextureObject_t hdl)
    : mgr_(mgr),
      ref_(r),
      hdl_(hdl)
{}

Texture::Texture(Texture &&o)
    : mgr_(o.mgr_),
      ref_(o.ref_),
      hdl_(o.hdl_)
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
        cudaFreeArray(tex.mem);

        cache_lock_.lock();

        if (tex.refCount.load(memory_order_acquire) == 0) {
            loaded_.erase(tex_ref);
        }

        cache_lock_.unlock();
    }
}

}
}
