#pragma once

#include <rlpbr/config.hpp>

#include "common.hpp"
#include "optix_scene.hpp"
#include "optix_shader.hpp"

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

struct RenderState {
    half *output;
    void *paramBuffer;
    ShaderParams *deviceParams;
    std::array<ShaderParams, 2> hostParams;
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
    uint32_t cur_frame_;
    const uint32_t num_frames_;
    std::array<cudaStream_t, 2> streams_;
    cudaStream_t tlas_strm_;
    OptixDeviceContext ctx_;
    Pipeline pipeline_;
    SBT sbt_;
    RenderState render_state_;

};

}
}
