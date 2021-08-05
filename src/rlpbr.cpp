#include <rlpbr.hpp>
#include <rlpbr_core/common.hpp>
#include <rlpbr_core/scene.hpp>
#include <rlpbr_core/utils.hpp>

#include "optix/render.hpp"
#include "vulkan/render.hpp"

#include <functional>
#include <iostream>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/transform.hpp>

using namespace std;

namespace RLpbr {

AssetLoader::AssetLoader(LoaderImpl &&backend)
    : backend_(move(backend))
{}

shared_ptr<Scene> AssetLoader::loadScene(string_view scene_path)
{
    SceneLoadData load_data =
        SceneLoadData::loadFromDisk(scene_path);

    return backend_.loadScene(move(load_data));
}

static bool enableValidation()
{
    char *enable_env = getenv("RLPBR_VALIDATE");
    if (!enable_env || enable_env[0] == '0')
        return false;

    return true;
}

static RendererImpl makeBackend(const RenderConfig &cfg)
{
    bool validate = enableValidation();

    switch(cfg.backend) {
        case BackendSelect::Optix: {
            auto *renderer = new optix::OptixBackend(cfg, validate);
            return makeRendererImpl<optix::OptixBackend>(renderer);
        }
        case BackendSelect::Vulkan: {
            auto *renderer = new vk::VulkanBackend(cfg, validate);
            return makeRendererImpl<vk::VulkanBackend>(renderer);
        }
    }

    cerr << "Unknown backend" << endl;
    abort();
}

void Renderer::BatchInitializer::addEnvironment(shared_ptr<Scene> scene)
{
    scenes_.emplace_back(move(scene));
}

Renderer::Renderer(const RenderConfig &cfg)
    : backend_(makeBackend(cfg)),
      aspect_ratio_(float(cfg.imgWidth) / float(cfg.imgHeight))
{}

AssetLoader Renderer::makeLoader()
{
    return AssetLoader(backend_.makeLoader());
}

Environment Renderer::makeEnvironment(const shared_ptr<Scene> &scene)
{
    Camera cam(glm::vec3(0.f), glm::vec3(0.f, 0.f, 1.f),
               glm::vec3(0.f, 1.f, 0.f), 90.f, aspect_ratio_);

    return Environment(backend_.makeEnvironment(scene, cam), scene, cam);
}

Environment Renderer::makeEnvironment(const shared_ptr<Scene> &scene,
                                      const glm::vec3 &eye, const glm::vec3 &target,
                                      const glm::vec3 &up, float vertical_fov,
                                      float aspect_ratio)
{
    Camera cam(eye, target, up, vertical_fov,
               aspect_ratio == 0.f ? aspect_ratio_ : aspect_ratio);

    return Environment(backend_.makeEnvironment(scene, cam), scene, cam);
}

Environment Renderer::makeEnvironment(const shared_ptr<Scene> &scene,
                                      const glm::mat4 &camera_to_world,
                                      float vertical_fov, float aspect_ratio)
{
    Camera cam(camera_to_world, vertical_fov, 
               aspect_ratio == 0.f ? aspect_ratio_ : aspect_ratio);

    return Environment(backend_.makeEnvironment(scene, cam), scene, cam);
}

Environment Renderer::makeEnvironment(const std::shared_ptr<Scene> &scene,
                                      const glm::vec3 &pos,
                                      const glm::vec3 &fwd,
                                      const glm::vec3 &up,
                                      const glm::vec3 &right,
                                      float vertical_fov, float aspect_ratio)
{
    Camera cam(pos, fwd, up, right, vertical_fov,
               aspect_ratio == 0.f ? aspect_ratio_ : aspect_ratio);

    return Environment(backend_.makeEnvironment(scene, cam), scene, cam);
}

RenderBatch::RenderBatch(Handle &&backend, vector<Environment> &&envs)
    : backend_(move(backend)),
      envs_(move(envs))
{
}

RenderBatch Renderer::makeRenderBatch(BatchInitializer &&init)
{
    int num_envs = init.scenes_.size();

    vector<Environment> envs;

    for (int i = 0; i < num_envs; i++) {
        envs.emplace_back(makeEnvironment(move(init.scenes_[i])));
    }

    return RenderBatch(backend_.makeRenderBatch(), move(envs));
}

void Renderer::render(RenderBatch &batch)
{
    backend_.render(batch);
}

void Renderer::waitForBatch(RenderBatch &batch)
{
    backend_.waitForBatch(batch);
}

half *Renderer::getOutputPointer(RenderBatch &batch) const
{
    return backend_.getOutputPointer(batch);
}

AuxiliaryOutputs Renderer::getAuxiliaryOutputs(RenderBatch &batch) const
{
    return backend_.getAuxiliaryOutputs(batch);
}

Environment::Environment(EnvironmentImpl &&backend,
                         const shared_ptr<Scene> &scene,
                         const Camera &cam)
    : backend_(move(backend)),
      scene_(scene),
      camera_(cam),
      instances_(scene_->envInit.defaultInstances),
      instance_materials_(scene->envInit.defaultInstanceMaterials),
      transforms_(scene_->envInit.defaultTransforms),
      instance_flags_(scene_->envInit.defaultInstanceFlags),
      index_map_(scene_->envInit.indexMap),
      reverse_id_map_(scene_->envInit.reverseIDMap),
      free_ids_(),
      free_light_ids_(),
      light_ids_(scene_->envInit.lightIDs),
      light_reverse_ids_(scene_->envInit.lightReverseIDs),
      dirty_(true)
{
    // FIXME use EnvironmentInit lights
}

void Environment::reset()
{
    instances_ = scene_->envInit.defaultInstances;
    instance_materials_ = scene_->envInit.defaultInstanceMaterials;
    transforms_ = scene_->envInit.defaultTransforms;
    instance_flags_ = scene_->envInit.defaultInstanceFlags;
    index_map_ = scene_->envInit.indexMap;
    reverse_id_map_ = scene_->envInit.reverseIDMap;
    free_ids_.clear();
    free_light_ids_.clear();
    light_ids_ = scene_->envInit.lightIDs;
    light_reverse_ids_ = scene_->envInit.lightReverseIDs;

    setDirty();
}

void Environment::deleteInstance(uint32_t inst_id)
{
    // FIXME, deal with instance_materials_
    uint32_t instance_idx = index_map_[inst_id];
    if (instances_.size() > 1) {
        // Keep contiguous
        instances_[instance_idx] = instances_.back();
        transforms_[instance_idx] = transforms_.back();
        reverse_id_map_[instance_idx] = reverse_id_map_.back();
        index_map_[reverse_id_map_[instance_idx]] = instance_idx;
    }
    instances_.pop_back();
    transforms_.pop_back();
    reverse_id_map_.pop_back();

    free_ids_.push_back(inst_id);
}

uint32_t Environment::addLight(const glm::vec3 &position,
                               const glm::vec3 &color)
{
    backend_.addLight(position, color);
    uint32_t light_idx = light_reverse_ids_.size();

    uint32_t light_id;
    if (free_light_ids_.size() > 0) {
        uint32_t free_id = free_light_ids_.back();
        free_light_ids_.pop_back();
        light_ids_[free_id] = light_idx;

        light_id = free_id;
    } else {
        light_ids_.push_back(light_idx);
        light_id = light_ids_.size() - 1;
    }

    light_reverse_ids_.push_back(light_idx);
    return light_id;
}

void Environment::removeLight(uint32_t light_id)
{
    uint32_t light_idx = light_ids_[light_id];
    backend_.removeLight(light_idx);

    if (light_reverse_ids_.size() > 1) {
        light_reverse_ids_[light_idx] = light_reverse_ids_.back();
        light_ids_[light_reverse_ids_[light_idx]] = light_idx;
    }
    light_reverse_ids_.pop_back();

    free_light_ids_.push_back(light_id);
}

uint32_t EnvironmentImpl::addLight(const glm::vec3 &position,
                                   const glm::vec3 &color)
{
    return invoke(add_light_ptr_, state_, position, color);
}

void EnvironmentImpl::removeLight(uint32_t idx)
{
    invoke(remove_light_ptr_, state_, idx);
}

shared_ptr<Scene> LoaderImpl::loadScene(SceneLoadData &&scene_data)
{
    return invoke(load_scene_ptr_, state_, move(scene_data));
}

LoaderImpl RendererImpl::makeLoader()
{
    return invoke(make_loader_ptr_, state_);
}

EnvironmentImpl RendererImpl::makeEnvironment(
    const std::shared_ptr<Scene> &scene, const Camera &cam) const
{
    return invoke(make_env_ptr_, state_, scene, cam);
}

RenderBatch::Handle RendererImpl::makeRenderBatch() const
{
    return invoke(make_batch_ptr_, state_);
}

void RendererImpl::render(RenderBatch &batch)
{
    return invoke(render_ptr_, state_, batch);
}

void RendererImpl::waitForBatch(RenderBatch &batch)
{
    invoke(wait_ptr_, state_, batch);
}

half *RendererImpl::getOutputPointer(RenderBatch &batch) const
{
    return invoke(get_output_ptr_, state_, batch);
}

AuxiliaryOutputs RendererImpl::getAuxiliaryOutputs(RenderBatch &batch) const
{
    return invoke(get_aux_ptr_, state_, batch);
}

void BatchDeleter::operator()(BatchBackend *ptr) const
{
    deletePtr(state, ptr);
}

}

