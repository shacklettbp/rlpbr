#pragma once 

#include <rlpbr/fwd.hpp>
#include <rlpbr/utils.hpp>

#include <glm/glm.hpp>
#include <variant>
#include <vector>

namespace RLpbr {

struct EnvironmentInit;

struct CameraVectors {
    glm::vec3 position;
    glm::vec3 up;
    glm::vec3 right;
};

struct Camera {
    inline Camera(const glm::vec3 &eye, const glm::vec3 &look_vec,
                  const glm::vec3 &up_vec);

    inline Camera(const glm::mat4 &view_matrix);

    std::variant<CameraVectors, glm::mat4> state;
};

class Environment {
public:
    Environment(Handle<EnvironmentState> &&renderer_state_,
                const EnvironmentInit &init,
                const glm::vec3 &eye, const glm::vec3 &look,
                const glm::vec3 &up);

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

    inline void setCameraView(const glm::vec3 &eye, const glm::vec3 &look,
                              const glm::vec3 &up);

    uint32_t addLight(const glm::vec3 &position, const glm::vec3 &color);
    void deleteLight(uint32_t light_id);

private:
    Handle<EnvironmentState> renderer_state_;

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
