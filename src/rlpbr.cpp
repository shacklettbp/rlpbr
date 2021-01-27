#include <rlpbr.hpp>
#include "common.hpp"
#include "scene.hpp"
#include "utils.hpp"
#include "optix_render.hpp"

#include <functional>
#include <iostream>

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
    }

    cerr << "Unknown backend" << endl;
    abort();
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
    return Environment(backend_.makeEnvironment(scene), scene);
}

Environment Renderer::makeEnvironment(const shared_ptr<Scene> &scene,
                                      const glm::vec3 &eye, const glm::vec3 &target,
                                      const glm::vec3 &up, float vertical_fov,
                                      float aspect_ratio)
{
    return Environment(backend_.makeEnvironment(scene),
                       scene, eye, target, up, vertical_fov,
                       aspect_ratio == 0.f ? aspect_ratio_ : aspect_ratio);
}

Environment Renderer::makeEnvironment(const shared_ptr<Scene> &scene,
                                      const glm::mat4 &camera_to_world,
                                      float vertical_fov, float aspect_ratio)
{
    return Environment(backend_.makeEnvironment(scene),
                       scene, camera_to_world, vertical_fov,
                       aspect_ratio == 0.f ? aspect_ratio_ : aspect_ratio);
}

Environment Renderer::makeEnvironment(const std::shared_ptr<Scene> &scene,
                                      const glm::vec3 &pos,
                                      const glm::vec3 &fwd,
                                      const glm::vec3 &up,
                                      const glm::vec3 &right,
                                      float vertical_fov, float aspect_ratio)
{
    return Environment(backend_.makeEnvironment(scene),
                       scene, pos, fwd, up, right,
                       vertical_fov,
                       aspect_ratio == 0.f ? aspect_ratio_ : aspect_ratio);
}

uint32_t Renderer::render(const Environment *envs)
{
    return backend_.render(envs);
}

void Renderer::waitForFrame(uint32_t frame_idx)
{
    backend_.waitForFrame(frame_idx);
}

half *Renderer::getOutputPointer(uint32_t frame_idx)
{
    return backend_.getOutputPointer(frame_idx);
}

Environment::Environment(EnvironmentImpl &&backend,
                         const shared_ptr<Scene> &scene,
                         const Camera &cam)
    : backend_(move(backend)),
      scene_(scene),
      camera_(cam),
      transforms_(scene_->envInit.transforms),
      materials_(scene_->envInit.materials),
      index_map_(scene_->envInit.indexMap),
      reverse_id_map_(scene_->envInit.reverseIDMap),
      free_ids_(),
      free_light_ids_(),
      light_ids_(scene_->envInit.lightIDs),
      light_reverse_ids_(scene_->envInit.lightReverseIDs)
{
    // FIXME use EnvironmentInit lights
}

Environment::Environment(EnvironmentImpl &&backend,
                         const shared_ptr<Scene> &scene)
    : Environment(move(backend), scene,
                  Camera(glm::vec3(0.f), glm::vec3(0.f, 0.f, 1.f),
                         glm::vec3(0.f, 1.f, 0.f), 90.f, 1.f))
{}

Environment::Environment(EnvironmentImpl &&backend,
                         const shared_ptr<Scene> &scene,
                         const glm::vec3 &eye, const glm::vec3 &target,
                         const glm::vec3 &up, float vertical_fov,
                         float aspect_ratio)
    : Environment(move(backend), scene,
                  Camera(eye, target, up, vertical_fov, aspect_ratio))
{}

Environment::Environment(EnvironmentImpl &&backend,
                         const shared_ptr<Scene> &scene,
                         const glm::mat4 &camera_to_world, float vertical_fov,
                         float aspect_ratio)
    : Environment(move(backend), scene,
                  Camera(camera_to_world, vertical_fov, aspect_ratio))
{}

Environment::Environment(EnvironmentImpl &&backend,
                         const shared_ptr<Scene> &scene,
                         const glm::vec3 &position_vec,
                         const glm::vec3 &forward_vec,
                         const glm::vec3 &up_vec,
                         const glm::vec3 &right_vec,
                         float vertical_fov, float aspect_ratio)
    : Environment(move(backend), scene,
                  Camera(position_vec, forward_vec, up_vec, right_vec,
                         vertical_fov, aspect_ratio))
{}

uint32_t Environment::addInstance(uint32_t model_idx, uint32_t material_idx,
                                  const glm::mat4x3 &model_matrix)
{
    transforms_[model_idx].emplace_back(model_matrix);
    materials_[model_idx].emplace_back(material_idx);
    uint32_t instance_idx = transforms_[model_idx].size() - 1;

    uint32_t outer_id;
    if (free_ids_.size() > 0) {
        uint32_t free_id = free_ids_.back();
        free_ids_.pop_back();
        index_map_[free_id].first = model_idx;
        index_map_[free_id].second = instance_idx;

        outer_id = free_id;
    } else {
        index_map_.emplace_back(model_idx, instance_idx);
        outer_id = index_map_.size() - 1;
    }

    reverse_id_map_[model_idx].emplace_back(outer_id);

    return outer_id;
}

void Environment::deleteInstance(uint32_t inst_id)
{
    auto [model_idx, instance_idx] = index_map_[inst_id];
    auto &transforms = transforms_[model_idx];
    auto &materials = materials_[model_idx];
    auto &reverse_ids = reverse_id_map_[model_idx];

    if (transforms.size() > 1) {
        // Keep contiguous
        transforms[instance_idx] = transforms.back();
        materials[instance_idx] = materials.back();
        reverse_ids[instance_idx] = reverse_ids.back();
        index_map_[reverse_ids[instance_idx]] = { model_idx, instance_idx };
    }
    transforms.pop_back();
    materials.pop_back();
    reverse_ids.pop_back();

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

EnvironmentImpl::EnvironmentImpl(
    DestroyType destroy_ptr, AddLightType add_light_ptr,
    RemoveLightType remove_light_ptr,
    EnvironmentBackend *state)
    : destroy_ptr_(destroy_ptr),
      add_light_ptr_(add_light_ptr),
      remove_light_ptr_(remove_light_ptr),
      state_(state)
{}

EnvironmentImpl::EnvironmentImpl(EnvironmentImpl &&o)
    : destroy_ptr_(o.destroy_ptr_),
      add_light_ptr_(o.add_light_ptr_),
      remove_light_ptr_(o.remove_light_ptr_),
      state_(o.state_)
{
    o.state_ = nullptr;
}

EnvironmentImpl::~EnvironmentImpl()
{
    if (state_) {
        invoke(destroy_ptr_, state_);
    }
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

LoaderImpl::LoaderImpl(DestroyType destroy_ptr, LoadSceneType load_scene_ptr,
                       LoaderBackend *state)
    : destroy_ptr_(destroy_ptr),
      load_scene_ptr_(load_scene_ptr),
      state_(state)
{}

LoaderImpl::LoaderImpl(LoaderImpl &&o)
    : destroy_ptr_(o.destroy_ptr_),
      load_scene_ptr_(o.load_scene_ptr_),
      state_(o.state_)
{
    o.state_ = nullptr;
}

LoaderImpl::~LoaderImpl()
{
    if (state_) {
        invoke(destroy_ptr_, state_);
    }
}

shared_ptr<Scene> LoaderImpl::loadScene(SceneLoadData &&scene_data)
{
    return invoke(load_scene_ptr_, state_, move(scene_data));
}

RendererImpl::RendererImpl(DestroyType destroy_ptr,
                           MakeLoaderType make_loader_ptr,
                           MakeEnvironmentType make_env_ptr,
                           RenderType render_ptr,
                           WaitType wait_ptr,
                           GetOutputType get_output_ptr,
                           RenderBackend *state)
    : destroy_ptr_(destroy_ptr),
      make_loader_ptr_(make_loader_ptr),
      make_env_ptr_(make_env_ptr),
      render_ptr_(render_ptr),
      wait_ptr_(wait_ptr),
      get_output_ptr_(get_output_ptr),
      state_(state)
{}

RendererImpl::RendererImpl(RendererImpl &&o)
    : destroy_ptr_(o.destroy_ptr_),
      make_loader_ptr_(o.make_loader_ptr_),
      make_env_ptr_(o.make_env_ptr_),
      render_ptr_(o.render_ptr_),
      wait_ptr_(o.wait_ptr_),
      get_output_ptr_(o.get_output_ptr_),
      state_(o.state_)
{
    o.state_ = nullptr;
}

RendererImpl::~RendererImpl()
{
    if (state_) {
        invoke(destroy_ptr_, state_);
    }
}

LoaderImpl RendererImpl::makeLoader()
{
    return invoke(make_loader_ptr_, state_);
}

EnvironmentImpl RendererImpl::makeEnvironment(
    const std::shared_ptr<Scene> &scene) const
{
    return invoke(make_env_ptr_, state_, scene);
}

uint32_t RendererImpl::render(const Environment *envs)
{
    return invoke(render_ptr_, state_, envs);
}

void RendererImpl::waitForFrame(uint32_t frame_idx)
{
    invoke(wait_ptr_, state_, frame_idx);
}

half *RendererImpl::getOutputPointer(uint32_t frame_idx)
{
    return invoke(get_output_ptr_, state_, frame_idx);
}

}
