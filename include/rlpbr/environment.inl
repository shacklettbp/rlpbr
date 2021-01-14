#pragma once

#include <glm/gtc/matrix_transform.hpp>

namespace RLpbr {

Camera::Camera(const glm::vec3 &eye, const glm::vec3 &look_vec,
               const glm::vec3 &up_vec)
    : state(CameraVectors { eye, up_vec, glm::cross(up_vec, look_vec) })
{}

Camera::Camera(const glm::mat4 &view_matrix)
    : state(view_matrix)
{}

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

void Environment::setCameraView(const glm::vec3 &eye, const glm::vec3 &look,
                                const glm::vec3 &up)
{
    camera_ = Camera(eye, look, up);
}

}
