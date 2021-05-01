#pragma once 

#include <rlpbr_core/scene.hpp>
#include <vector>

namespace RLpbr {

template <typename VertexType>
struct ProcessedGeometry {
    std::vector<VertexType> vertices;
    std::vector<uint32_t> indices;
    std::vector<MeshInfo> meshInfos;
    std::vector<ObjectInfo> objectInfos;
    std::vector<std::string> objectNames;
};

}
