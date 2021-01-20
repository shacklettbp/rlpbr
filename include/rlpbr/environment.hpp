#pragma once 

#include <rlpbr/fwd.hpp>
#include <rlpbr/backend.hpp>
#include <rlpbr/utils.hpp>

#include <glm/glm.hpp>
#include <vector>

namespace RLpbr {

struct EnvironmentInit;

struct Camera {
    inline Camera(const glm::vec3 &eye, const glm::vec3 &target,
                  const glm::vec3 &up_vec, float vertical_fov,
                  float aspect_ratio);

    inline Camera(const glm::mat4 &camera_to_world,
                  float vertical_fov, float aspect_ratio);

    inline Camera(const glm::vec3 &position_vec,
                  const glm::vec3 &forward_vec,
                  const glm::vec3 &up_vec,
                  const glm::vec3 &right_vec,
                  float vertical_fov,
                  float aspect_ratio);

    inline void updateView(const glm::vec3 &eye, const glm::vec3 &target,
                           const glm::vec3 &up_vec);

    inline void updateView(const glm::mat4 &camera_to_world);

    inline void updateView(const glm::vec3 &position_vec,
                           const glm::vec3 &forward_vec,
                           const glm::vec3 &up_vec,
                           const glm::vec3 &right_vec);

    glm::vec3 position;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;

    float tanFOV;
    float aspectRatio;
};

class Environment {
public:
    Environment(EnvironmentImpl &&backend,
                const std::shared_ptr<Scene> &scene,
                const glm::vec3 &eye, const glm::vec3 &target,
                const glm::vec3 &up, float vertical_fov,
                float aspect_ratio);

    Environment(EnvironmentImpl &&backend,
                const std::shared_ptr<Scene> &scene,
                const glm::mat4 &camera_to_world,
                float vertical_fov, float aspect_ratio);

    Environment(EnvironmentImpl &&backend,
                const std::shared_ptr<Scene> &scene,
                const glm::vec3 &position_vec,
                const glm::vec3 &forward_vec,
                const glm::vec3 &up_vec,
                const glm::vec3 &right_vec,
                float vertical_fov, float aspect_ratio);

    Environment(EnvironmentImpl &&backend,
                const std::shared_ptr<Scene> &scene);

    Environment(const Environment &) = delete;
    Environment & operator=(const Environment &) = delete;

    Environment(Environment &&) = default;
    Environment & operator=(Environment &&) = default;

    // Instance transformations
    inline uint32_t addInstance(uint32_t model_idx, uint32_t material_idx,
                                const glm::mat4 &model_matrix);

    uint32_t addInstance(uint32_t model_idx, uint32_t material_idx,
                         const glm::mat4x3 &model_matrix);

    void deleteInstance(uint32_t inst_id);

    inline const glm::mat4x3 & getInstanceTransform(uint32_t inst_id) const;

    inline void updateInstanceTransform(uint32_t inst_id,
                                        const glm::mat4 &model_matrix);

    inline void updateInstanceTransform(uint32_t inst_id,
                                        const glm::mat4x3 &model_matrix);

    inline void setInstanceMaterial(uint32_t inst_id, uint32_t material_idx);

    inline void setCameraView(const glm::vec3 &eye, const glm::vec3 &target,
                              const glm::vec3 &up);
    inline void setCameraView(const glm::mat4 &camera_to_world);

    uint32_t addLight(const glm::vec3 &position, const glm::vec3 &color);
    void removeLight(uint32_t light_id);

    inline const std::shared_ptr<Scene> getScene() const;
    inline const Camera &getCamera() const;

private:
    Environment(EnvironmentImpl &&backend,
                const std::shared_ptr<Scene> &scene,
                const Camera &cam);

    EnvironmentImpl backend_;
    std::shared_ptr<Scene> scene_;

    Camera camera_;

    std::vector<std::vector<glm::mat4x3>> transforms_;
    std::vector<std::vector<uint32_t>> materials_;

    std::vector<std::pair<uint32_t, uint32_t>> index_map_;
    std::vector<std::vector<uint32_t>> reverse_id_map_;
    std::vector<uint32_t> free_ids_;

    std::vector<uint32_t> free_light_ids_;
    std::vector<uint32_t> light_ids_;
    std::vector<uint32_t> light_reverse_ids_;
};

}

#include <rlpbr/environment.inl>
