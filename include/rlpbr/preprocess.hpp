#pragma once

#include <rlpbr/utils.hpp>
#include <glm/glm.hpp>

#include <optional>
#include <string_view>

namespace RLpbr {

struct PreprocessData;

class ScenePreprocessor {
public:
    ScenePreprocessor(std::string_view gltf_path,
                      const glm::mat4 &base_txfm,
                      std::optional<std::string_view> texture_dir);

    void dump(std::string_view out_path);

private:
    Handle<PreprocessData> scene_data_;
};

}
