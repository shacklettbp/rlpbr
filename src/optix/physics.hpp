#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <rlpbr_core/physics.hpp>

#include "shader.hpp"
#include "texture.hpp"

namespace RLpbr {
namespace optix {

struct PhysicsConfig {
    uint32_t batchSize;
};

struct ScenePhysicsData {
    ScenePhysicsData(std::vector<Texture> &&sdf_volumes,
                     void *default_instances,
                     void *default_transforms,
                     uint32_t num_default_static,
                     uint32_t num_default_dynamic,
                     const PhysicsObject *objects_ptr,
                     const cudaTextureObject_t *sdf_hdls);

    ScenePhysicsData(const ScenePhysicsData &) = delete;
    ScenePhysicsData(ScenePhysicsData &&);
    ~ScenePhysicsData();

    static ScenePhysicsData make(const PhysicsMetadata &metadata,
                                 const StagingHeader &hdr,
                                 void *start_staging_ptr,
                                 void *post_staging_ptr,
                                 cudaStream_t cpy_strm,
                                 TextureManager &tex_mgr);

    std::vector<Texture> sdfVolumes;
    void *defaultInstances;
    void *defaultTransforms;
    uint32_t numDefaultStatic;
    uint32_t numDefaultDynamic;

    const PhysicsObject *objectsPtr;
    const cudaTextureObject_t *sdfHandles;
};

struct CollisionCandidate {
    uint32_t dynInstance;
    uint32_t otherInstance;
};

struct Contact {
    glm::vec3 normal;
    glm::vec3 posA;
    glm::vec3 posB;
    uint32_t aDynamicID;
    uint32_t bDynamicID;
};

struct alignas(16) PhysicsScratch {
    CollisionCandidate *collisionCandidates;
    Contact *contacts;

    uint32_t numContacts;
};

struct PhysicsEnvironment {
    PhysicsEnvironment(const ScenePhysicsData &scene_data,
                       cudaStream_t cpy_strm);
    PhysicsEnvironment(const PhysicsEnvironment &) = delete;
    PhysicsEnvironment(PhysicsEnvironment &&);
    ~PhysicsEnvironment();

    PhysicsInstance *instances;
    PhysicsTransform *transforms;

    PhysicsScratch scratchHost;
    PhysicsScratch *scratch;

    uint32_t numStatic;
    uint32_t numDynamic;
    uint32_t freeInstanceCapacity;
};

// FIXME rename to DevicePhysicsEnv
struct alignas(16) PackedPhysicsEnv {
    PhysicsInstance *instances;

    const DevicePackedVertex *vertexBuffer;
    const uint32_t *indexBuffer;

    const PhysicsObject *objects;
    const cudaTextureObject_t *sdfHandles;

    const InstanceTransform *transforms;
    const PhysicsTransform *decomposedTransforms;

    PhysicsScratch *scratch;

    uint32_t numStatic;
    uint32_t numDynamic;
};

class PhysicsSimulator {
public:
    PhysicsSimulator(const PhysicsConfig &cfg);
    PhysicsSimulator(const PhysicsSimulator &) = delete;
    PhysicsSimulator(PhysicsSimulator &&);
    ~PhysicsSimulator();

    void simulate(const Environment *envs);

private:
    //void processCollisions();

    void sdfDebug(const Environment *envs);

    PhysicsConfig cfg_;
    cudaStream_t stream_;
    PackedPhysicsEnv *env_input_;
};

}
}
