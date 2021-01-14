#include <rlpbr.hpp>
#include "common.hpp"
#include "scene.hpp"
#include "utils.hpp"

#include <functional>

using namespace std;

namespace RLpbr {

AssetLoader::AssetLoader(Handle<LoaderBackend> &&backend)
    : backend_(move(backend))
{}

shared_ptr<Scene> AssetLoader::loadScene(string_view scene_path)
{
    SceneLoadData load_data =
        SceneLoadData::loadFromDisk(scene_path, *backend_);

    return invoke(backend_->loadScene, backend_, load_data);
}

Renderer::Renderer(const RenderConfig &cfg)
    : backend_(nullptr)
{}

AssetLoader Renderer::makeLoader()
{
    return AssetLoader(invoke(backend_->makeLoader, backend_));
}

Environment Renderer::makeEnvironment(const shared_ptr<Scene> &scene)
{
    return makeEnvironment(scene, glm::vec3(0.f), glm::vec3(0.f, 0.f, 1.f),
                           glm::vec3(0.f, 1.f, 0.f));
}

Environment Renderer::makeEnvironment(const shared_ptr<Scene> &scene,
                                      const glm::vec3 &eye, const glm::vec3 &look,
                                      const glm::vec3 &up)
{
    return Environment(invoke(backend_->makeEnvironment, backend_, scene),
                       scene->envInit, eye, look, up);
}

Environment::Environment(Handle<EnvironmentState> &&renderer_state,
                         const EnvironmentInit &init,
                         const glm::vec3 &eye, const glm::vec3 &look,
                         const glm::vec3 &up)
    : renderer_state_(move(renderer_state)),
      camera_(eye, look, up),
      transforms_(init.transforms),
      materials_(init.materials),
      index_map_(init.indexMap),
      reverse_id_map_(init.reverseIDMap),
      free_ids_(),
      free_light_ids_(),
      light_ids_(init.lightIDs),
      light_reverse_ids_(init.lightReverseIDs)
{
    // FIXME use EnvironmentInit lights
}

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
    invoke(renderer_state_->addLight, renderer_state_, position, color);
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

void Environment::deleteLight(uint32_t light_id)
{
    uint32_t light_idx = light_ids_[light_id];
    invoke(renderer_state_->deleteLight, renderer_state_, light_idx);

    if (light_reverse_ids_.size() > 1) {
        light_reverse_ids_[light_idx] = light_reverse_ids_.back();
        light_ids_[light_reverse_ids_[light_idx]] = light_idx;
    }
    light_reverse_ids_.pop_back();

    free_light_ids_.push_back(light_id);
}

LoaderBackend::~LoaderBackend()
{
    invoke(destroy, this);
}

EnvironmentState::~EnvironmentState()
{
    invoke(destroy, this);
}

RenderBackend::~RenderBackend()
{
    invoke(destroy, this);
}

template struct HandleDeleter<LoaderBackend>;
template struct HandleDeleter<RenderBackend>;
template struct HandleDeleter<EnvironmentState>;

}
