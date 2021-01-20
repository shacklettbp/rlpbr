#pragma once

#include <glm/gtc/matrix_transform.hpp>

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
    return glm::normalize(-glm::vec3(camera_to_world[2]));
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

Camera::Camera(const glm::vec3 &eye, const glm::vec3 &target,
               const glm::vec3 &up_vec, float vertical_fov,
               float aspect_ratio)
    : position(eye),
      view(CameraHelper::computeViewVector(eye, target)),
      up(up_vec),
      right(CameraHelper::computeRightVector(view, up)),
      tanFOV(tanf(vertical_fov / 2.f)),
      aspectRatio(aspect_ratio)
{}

Camera::Camera(const glm::mat4 &camera_to_world,
               float vertical_fov, float aspect_ratio)
    : position(CameraHelper::extractPosition(camera_to_world)),
      view(CameraHelper::extractViewVector(camera_to_world)),
      up(CameraHelper::extractUpVector(camera_to_world)),
      right(CameraHelper::extractRightVector(camera_to_world)),
      tanFOV(tanf(vertical_fov / 2.f)),
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
      tanFOV(tanf(vertical_fov / 2.f)),
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

uint32_t Environment::addInstance(uint32_t model_idx, uint32_t material_idx,
                                  const glm::mat4x4 &matrix)
{
    return addInstance(model_idx, material_idx, glm::mat4x3(matrix));
}

const glm::mat4x3 & Environment::getInstanceTransform(uint32_t inst_id) const
{

    const auto &p = index_map_[inst_id];
    return transforms_[p.first][p.second];
}

void Environment::updateInstanceTransform(uint32_t inst_id,
                                          const glm::mat4x3 &mat)
{
    const auto &p = index_map_[inst_id];
    transforms_[p.first][p.second] = mat;
}

void Environment::updateInstanceTransform(uint32_t inst_id,
                                          const glm::mat4 &mat)
{
    updateInstanceTransform(inst_id, glm::mat4x3(mat));
}

void Environment::setInstanceMaterial(uint32_t inst_id,
                                      uint32_t material_idx)
{
    const auto &p = index_map_[inst_id];
    materials_[p.first][p.second] = material_idx;
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

const std::shared_ptr<Scene> Environment::getScene() const
{
    return scene_;
}

const Camera &Environment::getCamera() const
{
    return camera_;
}

}
