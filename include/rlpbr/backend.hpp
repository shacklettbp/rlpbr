#pragma once

#include <rlpbr/fwd.hpp>

#include <glm/glm.hpp>

#include <cuda_fp16.h>

#include <memory>
#include <string_view>

namespace RLpbr {

class EnvironmentImpl {
public:
    typedef void(*DestroyType)(EnvironmentBackend *);
    typedef uint32_t(EnvironmentBackend::*AddLightType)(
        const glm::vec3 &, const glm::vec3 &);
    typedef void(EnvironmentBackend::*RemoveLightType)(uint32_t);

    EnvironmentImpl(DestroyType destroy_ptr, AddLightType add_light_ptr,
                           RemoveLightType remove_light_ptr,
                           EnvironmentBackend *state);
    EnvironmentImpl(const EnvironmentImpl &) = delete;
    EnvironmentImpl(EnvironmentImpl &&);

    EnvironmentImpl & operator=(const EnvironmentImpl &) = delete;
    EnvironmentImpl & operator=(EnvironmentImpl &&);

    ~EnvironmentImpl();

    inline uint32_t addLight(const glm::vec3 &position,
                             const glm::vec3 &color);
    inline void removeLight(uint32_t idx);

    inline EnvironmentBackend *getState() { return state_; };
    inline const EnvironmentBackend *getState() const  { return state_; };

private:
    DestroyType destroy_ptr_;
    AddLightType add_light_ptr_;
    RemoveLightType remove_light_ptr_;
    EnvironmentBackend *state_;
};

class LoaderImpl {
public:
    typedef void(*DestroyType)(LoaderBackend *);
    typedef std::shared_ptr<Scene>(LoaderBackend::*LoadSceneType)(
        SceneLoadData &&);

    LoaderImpl(DestroyType destroy_ptr, LoadSceneType load_scene_ptr,
               LoaderBackend *state);
    LoaderImpl(const LoaderImpl &) = delete;
    LoaderImpl(LoaderImpl &&);

    LoaderImpl & operator=(const LoaderImpl &) = delete;
    LoaderImpl & operator=(LoaderImpl &&);

    ~LoaderImpl();

    inline std::shared_ptr<Scene> loadScene(SceneLoadData &&scene_data);

private:
    DestroyType destroy_ptr_;
    LoadSceneType load_scene_ptr_;
    LoaderBackend *state_;
};

struct AuxiliaryOutputs {
    half *normal;
    half *albedo;
};

class RendererImpl {
public:
    typedef void(*DestroyType)(RenderBackend *);
    typedef LoaderImpl(RenderBackend::*MakeLoaderType)();
    typedef EnvironmentImpl(RenderBackend::*MakeEnvironmentType)(
        const std::shared_ptr<Scene> &, const Camera &);
    typedef std::unique_ptr<BatchBackend, BatchDeleter>(
        RenderBackend::*MakeBatchType)();
    typedef void(RenderBackend::*RenderType)(RenderBatch &batch);
    typedef void(RenderBackend::*WaitType)(RenderBatch &batch);
    typedef half *(RenderBackend::*GetOutputType)(RenderBatch &batch);
    typedef AuxiliaryOutputs(RenderBackend::*GetAuxType)(RenderBatch &batch);

    RendererImpl(DestroyType destroy_ptr,
        MakeLoaderType make_loader_ptr, MakeEnvironmentType make_env_ptr,
        MakeBatchType make_batch_ptr, RenderType render_ptr, WaitType wait_ptr,
        GetOutputType get_output_ptr, GetAuxType get_aux_ptr,
        RenderBackend *state);
    RendererImpl(const RendererImpl &) = delete;
    RendererImpl(RendererImpl &&);

    RendererImpl & operator=(const RendererImpl &) = delete;
    RendererImpl & operator=(RendererImpl &&);

    ~RendererImpl();

    inline LoaderImpl makeLoader();

    inline EnvironmentImpl makeEnvironment(
        const std::shared_ptr<Scene> &scene, const Camera &) const;

    inline void render(RenderBatch &batch);
    inline std::unique_ptr<BatchBackend, BatchDeleter> makeRenderBatch() const;

    inline void waitForBatch(RenderBatch &batch);

    inline half *getOutputPointer(RenderBatch &batch) const;

    inline AuxiliaryOutputs getAuxiliaryOutputs(RenderBatch &batch) const;

private:
    DestroyType destroy_ptr_;
    MakeLoaderType make_loader_ptr_;
    MakeEnvironmentType make_env_ptr_;
    MakeBatchType make_batch_ptr_;
    RenderType render_ptr_;
    WaitType wait_ptr_;
    GetOutputType get_output_ptr_;
    GetAuxType get_aux_ptr_;
    RenderBackend *state_;
};

}
