#pragma once 

#include <rlpbr/fwd.hpp>
#include <rlpbr/backend.hpp>
#include <rlpbr/utils.hpp>

#include <glm/glm.hpp>
#include <vector>

namespace RLpbr {

struct EnvironmentInit;

struct ObjectInstance {
    uint32_t objectIndex;
    uint32_t materialOffset;
};

struct InstanceTransform {
    glm::mat4x3 mat;
    glm::mat4x3 inv;
};

enum class InstanceFlags : uint32_t {
    Transparent = 1 << 0,
};

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
                const Camera &cam);

    Environment(const Environment &) = delete;
    Environment & operator=(const Environment &) = delete;

    Environment(Environment &&) = default;
    Environment & operator=(Environment &&) = default;

    // Instance transformations
    inline uint32_t addInstance(uint32_t obj_idx,
                                const uint32_t *material_idxs,
                                uint32_t num_mat_indices,
                                const glm::vec3 &position, 
                                const glm::quat &rotation,
                                bool dynamic = true,
                                bool kinematic = false);

    void deleteInstance(uint32_t inst_id);

    inline void moveInstance(uint32_t inst_id, const glm::vec3 &delta);
    inline void rotateInstance(uint32_t inst_id, const glm::quat &rot);

    template <int N>
    inline void setInstanceMaterial(uint32_t inst_id,
                                    const std::array<uint32_t, N> &material_idxs);

    inline void setCameraView(const glm::vec3 &eye, const glm::vec3 &target,
                              const glm::vec3 &up);

    inline void setCameraView(const glm::vec3 &position,
                              const glm::vec3 &fwd,
                              const glm::vec3 &up,
                              const glm::vec3 &right);

    inline void setCameraView(const glm::mat4 &camera_to_world);

    uint32_t addLight(const glm::vec3 &position, const glm::vec3 &color);
    void removeLight(uint32_t light_id);

    inline const std::shared_ptr<Scene> &getScene() const;
    inline const EnvironmentBackend *getBackend() const;
    inline EnvironmentBackend *getBackend();
    inline const Camera &getCamera() const;

    inline const std::vector<ObjectInstance> &
        getInstances() const;

    inline const std::vector<uint32_t> &
        getInstanceMaterials() const;

    inline const std::vector<InstanceTransform> &
        getTransforms() const;

    inline const std::vector<InstanceFlags> &
        getInstanceFlags() const;

    inline uint32_t getNumInstances() const;

    inline bool isDirty() const;
    inline void setDirty() const;
    inline void clearDirty() const;

    // Reset environment to default instances / materials
    void reset();

private:
    EnvironmentImpl backend_;
    std::shared_ptr<Scene> scene_;

    Camera camera_;

    std::vector<ObjectInstance> instances_;
    std::vector<uint32_t> instance_materials_;
    std::vector<InstanceTransform> transforms_;
    std::vector<InstanceFlags> instance_flags_;

    std::vector<uint32_t> index_map_;
    std::vector<uint32_t> reverse_id_map_;
    std::vector<uint32_t> free_ids_;

    std::vector<uint32_t> free_light_ids_;
    std::vector<uint32_t> light_ids_;
    std::vector<uint32_t> light_reverse_ids_;

    mutable bool dirty_;
};

inline InstanceFlags & operator|=(InstanceFlags &a, InstanceFlags b)
{
    a = InstanceFlags(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    return a;
}

inline bool operator&(InstanceFlags a, InstanceFlags b)
{
    return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) > 0;
}


}

#include <rlpbr/environment.inl>
