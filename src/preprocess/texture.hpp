#pragma once

#include <cstdint>
#include <string_view>
#include <texutil.hpp>

namespace RLpbr {
namespace SceneImport {

class TextureCallback {
public:
    using CBType = void (*)(std::string_view texture_name,
                            texutil::TextureType type,
                            const uint8_t *data,
                            uint64_t num_bytes,
                            void *cb_data);

    inline TextureCallback(CBType cb, void *cb_data)
        : cb_(cb),
          cb_data_(cb_data)
    {}

    inline void operator()(std::string_view texture_name,
                           texutil::TextureType texture_type,
                           const uint8_t *data,
                           uint64_t num_bytes) const
    {
        cb_(texture_name, texture_type, data, num_bytes, cb_data_);
    }

private:
    CBType cb_;
    void *cb_data_;
};


}
}
