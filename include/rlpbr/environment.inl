#pragma once

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace RLpbr {

namespace CameraHelper {

static inline glm::vec3 computeViewVector(const glm::vec3 &eye,
                                          const glm::vec3 &target)
{
    return glm::normalize(target - eye);
}

static inline glm::vec3 computeRightVector(const glm::vec3 &view,
                                           const glm::vec3 &up)
{
    return glm::normalize(glm::cross(view, up));
}

static inline glm::vec3 extractPosition(
    const glm::mat4 &camera_to_world)
{
    return camera_to_world[3];
}

static inline glm::vec3 extractViewVector(
    const glm::mat4 &camera_to_world)
{
    return -glm::normalize(glm::vec3(camera_to_world[2]));
}

static inline glm::vec3 extractUpVector(
    const glm::mat4 &camera_to_world)
{
    return glm::normalize(glm::vec3(camera_to_world[1]));
}

static inline glm::vec3 extractRightVector(
    const glm::mat4 &camera_to_world)
{
    return glm::normalize(glm::vec3(camera_to_world[0]));
}

}

static inline float convertFOV(float fov)
{
    return tanf(glm::radians(fov) / 2.f);
}

Camera::Camera(const glm::vec3 &eye, const glm::vec3 &target,
               const glm::vec3 &up_vec, float vertical_fov,
               float aspect_ratio)
    : position(eye),
      view(CameraHelper::computeViewVector(eye, target)),
      up(up_vec),
      right(CameraHelper::computeRightVector(view, up)),
      tanFOV(convertFOV(vertical_fov)),
      aspectRatio(aspect_ratio)
{}

Camera::Camera(const glm::mat4 &camera_to_world,
               float vertical_fov, float aspect_ratio)
    : position(CameraHelper::extractPosition(camera_to_world)),
      view(CameraHelper::extractViewVector(camera_to_world)),
      up(CameraHelper::extractUpVector(camera_to_world)),
      right(CameraHelper::extractRightVector(camera_to_world)),
      tanFOV(convertFOV(vertical_fov)),
      aspectRatio(aspect_ratio)
{}

Camera::Camera(const glm::vec3 &position_vec,
               const glm::vec3 &forward_vec,
               const glm::vec3 &up_vec,
               const glm::vec3 &right_vec,
               float vertical_fov,
               float aspect_ratio)
    : position(position_vec),
      view(forward_vec),
      up(up_vec),
      right(right_vec),
      tanFOV(convertFOV(vertical_fov)),
      aspectRatio(aspect_ratio)
{}

void Camera::updateView(const glm::vec3 &eye, const glm::vec3 &target,
                        const glm::vec3 &up_vec)
{
    using namespace CameraHelper;
    position = eye;
    view = computeViewVector(eye, target);
    up = up_vec;
    right = computeRightVector(view, up);
}

void Camera::updateView(const glm::mat4 &camera_to_world)
{
    using namespace CameraHelper;
    position = extractPosition(camera_to_world);
    view = extractViewVector(camera_to_world);
    up = extractUpVector(camera_to_world);
    right = extractRightVector(camera_to_world);
}

void Camera::updateView(const glm::vec3 &position_vec,
                        const glm::vec3 &forward_vec,
                        const glm::vec3 &up_vec,
                        const glm::vec3 &right_vec)
{
    position = position_vec;
    view = forward_vec;
    up = up_vec;
    right = right_vec;
}

uint32_t Environment::addInstance(uint32_t obj_idx,
                                  const uint32_t *material_idxs,
                                  uint32_t num_mat_indices,
                                  const glm::vec3 &position,
                                  const glm::quat &rotation,
                                  bool dynamic,
                                  bool kinematic)
{
    setDirty();
    // FIXME
    (void)dynamic;
    (void)kinematic;

    glm::mat4 rot_matrix = glm::mat4_cast(rotation);

    glm::mat4 model_matrix = glm::translate(position) * rot_matrix;
    glm::mat4 inv_model = glm::transpose(rot_matrix) *
        glm::translate(-position);

    instances_.push_back({
        obj_idx,
        uint32_t(instance_materials_.size()),
    });

    for (int i = 0; i < (int)num_mat_indices; i++) {
        instance_materials_.push_back(material_idxs[i]);
    }

    transforms_.push_back({model_matrix, inv_model});
    instance_flags_.push_back(InstanceFlags {});

    return instances_.size() - 1;

    // FIXME
#if 0
    uint32_t instance_idx = instances_.size() - 1;

    uint32_t outer_id;
    if (free_ids_.size() > 0) {
        uint32_t free_id = free_ids_.back();
        free_ids_.pop_back();
        index_map_[free_id] = instance_idx;

        outer_id = free_id;
    } else {
        index_map_.push_back(instance_idx);
        outer_id = index_map_.size() - 1;
    }

    reverse_id_map_.emplace_back(outer_id);

    return outer_id;
#endif
}

void Environment::moveInstance(uint32_t inst_id, const glm::vec3 &delta)
{
    (void)inst_id;
    (void)delta;
}

void Environment::rotateInstance(uint32_t inst_id, const glm::quat &rot)
{
    (void)inst_id;
    (void)rot;
}

template <int N>
void Environment::setInstanceMaterial(uint32_t inst_id,
                                      const std::array<uint32_t, N> &material_idxs)
{
    uint32_t idx = index_map_[inst_id];
    uint32_t *mats = &instance_materials_[instances_[idx].materialOffset];
    for (int i = 0; i < N; i++) {
        mats[i] = material_idxs[i];
    }
}

void Environment::setCameraView(const glm::vec3 &eye, const glm::vec3 &target,
                                const glm::vec3 &up)
{
    camera_.updateView(eye, target, up);
}

void Environment::setCameraView(const glm::mat4 &camera_to_world)
{
    camera_.updateView(camera_to_world);
}

void Environment::setCameraView(const glm::vec3 &position,
                                const glm::vec3 &fwd,
                                const glm::vec3 &up,
                                const glm::vec3 &right)
{
    camera_.updateView(position, fwd, up, right);
}

const std::shared_ptr<Scene> &Environment::getScene() const
{
    return scene_;
}

const EnvironmentBackend *Environment::getBackend() const
{
    return backend_.getState();
}

EnvironmentBackend *Environment::getBackend()
{
    return backend_.getState();
}

const Camera &Environment::getCamera() const
{
    return camera_;
}

const std::vector<ObjectInstance> &
    Environment::getInstances() const
{
    return instances_;
}

const std::vector<uint32_t> &
    Environment::getInstanceMaterials() const
{
    return instance_materials_;
}

const std::vector<InstanceTransform> &
    Environment::getTransforms() const
{
    return transforms_;
}

const std::vector<InstanceFlags> &
    Environment::getInstanceFlags() const
{
    return instance_flags_;
}

uint32_t Environment::getNumInstances() const
{
    return instances_.size();
}

bool Environment::isDirty() const
{
    return dirty_;
}

void Environment::setDirty() const
{
    dirty_ = true;
}

void Environment::clearDirty() const
{
    dirty_ = false;
}

}
