#pragma once

#include <rlpbr/config.hpp>
#include <rlpbr_core/common.hpp>

#include "scene.hpp"
#include "shader.hpp"

namespace RLpbr {
namespace optix {

struct Pipeline {
    OptixModule shader;
    std::array<OptixProgramGroup, 3> groups;
    OptixPipeline hdl;
};

struct SBT {
    CUdeviceptr storage;
    OptixShaderBindingTable hdl;
};

struct ShaderBuffers {
    half *outputBuffer;
    PackedEnv *envs;
    LaunchInput *launchInput;
    PackedInstance *instanceBuffer;
    uint32_t *instanceMaterialBuffer;
    PackedLight *lightBuffer;
};

struct RenderState {
    half *output;
    void *paramBuffer;
    std::array<ShaderBuffers, 2> shaderBuffers;
};

struct BSDFLookupTables {
    Texture diffuseAverageAlbedo;
    Texture diffuseDirectionalAlbedo;
    Texture ggxAverageAlbedo;
    Texture ggxDirectionalAlbedo;
    Texture ggxDirectionalInverse;
    BSDFPrecomputed deviceHandles;
};

class OptixBackend : public RenderBackend {
public:
    OptixBackend(const RenderConfig &cfg, bool validate);
    LoaderImpl makeLoader();

    EnvironmentImpl makeEnvironment(const std::shared_ptr<Scene> &scene);

    uint32_t render(const Environment *envs);

    void waitForFrame(uint32_t frame_idx);

    half *getOutputPointer(uint32_t frame_idx);

private:
    const uint32_t batch_size_;
    const glm::u32vec2 img_dims_;
    uint32_t active_idx_;
    uint32_t frame_counter_;
    const uint32_t frame_mask_;
    std::array<cudaStream_t, 2> streams_;
    cudaStream_t tlas_strm_;
    OptixDeviceContext ctx_;
    RenderState render_state_;
    Pipeline pipeline_;
    SBT sbt_;
    TextureManager texture_mgr_;
    BSDFLookupTables bsdf_luts_;
    std::optional<PhysicsSimulator> physics_;
};

}
}
