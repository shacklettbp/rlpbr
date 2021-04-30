#include "physics.hpp"
#include "scene.hpp"
#include "rlpbr_core/physics.hpp"
#include "utils.hpp"

#include <iostream>
#include <fstream>

using namespace std;

namespace RLpbr {
namespace optix {

struct PhysicsConstants {
    static constexpr int maxContactsPerDynamic = 500;
    static constexpr int numExtraInstances = 20;
};

ScenePhysicsData::ScenePhysicsData(vector<Texture> &&sdf_volumes,
                                   void *default_instances,
                                   void *default_transforms,
                                   uint32_t num_default_static,
                                   uint32_t num_default_dynamic,
                                   const PhysicsObject *objects_ptr,
                                   const cudaTextureObject_t *sdf_hdls)
    : sdfVolumes(move(sdf_volumes)),
      defaultInstances(default_instances),
      defaultTransforms(default_transforms),
      numDefaultStatic(num_default_static),
      numDefaultDynamic(num_default_dynamic),
      objectsPtr(objects_ptr),
      sdfHandles(sdf_hdls)
{}

ScenePhysicsData::ScenePhysicsData(ScenePhysicsData &&o)
    : sdfVolumes(move(o.sdfVolumes)),
      defaultInstances(o.defaultInstances),
      defaultTransforms(o.defaultTransforms),
      numDefaultStatic(o.numDefaultStatic),
      numDefaultDynamic(o.numDefaultDynamic),
      objectsPtr(o.objectsPtr),
      sdfHandles(o.sdfHandles)
{
    o.defaultInstances = nullptr;
    o.defaultTransforms = nullptr;
}

ScenePhysicsData::~ScenePhysicsData()
{
    REQ_CUDA(cudaFree(defaultTransforms));
    REQ_CUDA(cudaFree(defaultInstances));
}

// start_staging_ptr is used to determine offset pointers for
// PhysicsObjects and MeshInfos
// post_staging_ptr is the start of extra space in the scene buffer
// allocated for SDF texture handles
ScenePhysicsData ScenePhysicsData::make(const PhysicsMetadata &metadata,
                                        const StagingHeader &hdr,
                                        void *start_staging_ptr,
                                        void *post_staging_ptr,
                                        cudaStream_t cpy_strm,
                                        TextureManager &tex_mgr)
{
    vector<void *> host_data;
    host_data.reserve(metadata.sdfPaths.size());

    vector<Texture> sdf_volumes;
    sdf_volumes.reserve(metadata.sdfPaths.size());

    for (const auto &sdf_path : metadata.sdfPaths) {
        Texture volume = tex_mgr.load(sdf_path, TextureFormat::R32_SFLOAT,
            cudaAddressModeClamp, cpy_strm,
            [&host_data](const string &tex_path) {
                ifstream sdf_file(tex_path);

                glm::u32vec3 sdf_dims;
                sdf_file.read(reinterpret_cast<char *>(&sdf_dims),
                              sizeof(glm::u32vec3));
                size_t sdf_bytes = sizeof(float) * sdf_dims.x *
                    sdf_dims.y * sdf_dims.z;

                void *sdf_data = allocCUHost(sdf_bytes,
                    cudaHostAllocMapped | cudaHostAllocWriteCombined);

                sdf_file.read(reinterpret_cast<char *>(sdf_data), sdf_bytes);

                host_data.push_back(sdf_data);

                return make_tuple(sdf_data, sdf_dims, 1, 0.f);
            });

        sdf_volumes.emplace_back(move(volume));
    }

    vector<cudaTextureObject_t> sdf_hdls;
    sdf_hdls.reserve(sdf_volumes.size());
    for (const auto &vol : sdf_volumes) {
        sdf_hdls.push_back(vol.getHandle());
    }

    REQ_CUDA(cudaMemcpyAsync(post_staging_ptr, sdf_hdls.data(),
        sizeof(cudaTextureObject_t) * sdf_hdls.size(),
        cudaMemcpyHostToDevice, cpy_strm));

    size_t static_inst_bytes =
        metadata.staticInstances.size() * sizeof(PhysicsInstance);
    size_t dynamic_inst_bytes =
        metadata.dynamicInstances.size() * sizeof(PhysicsInstance);

    size_t dynamic_txfm_bytes =
        sizeof(PhysicsTransform) * metadata.dynamicInstances.size();

    void *instance_buffer = allocCU(static_inst_bytes + dynamic_inst_bytes);
    void *transform_buffer = allocCU(dynamic_txfm_bytes);

    REQ_CUDA(cudaMemcpyAsync(instance_buffer, metadata.staticInstances.data(),
        static_inst_bytes, cudaMemcpyHostToDevice, cpy_strm));

    REQ_CUDA(cudaMemcpyAsync((char *)instance_buffer + static_inst_bytes,
        metadata.dynamicInstances.data(),
        dynamic_inst_bytes, cudaMemcpyHostToDevice, cpy_strm));

    REQ_CUDA(cudaMemcpyAsync(transform_buffer, metadata.dynamicTransforms.data(),
                             dynamic_txfm_bytes, cudaMemcpyHostToDevice,
                             cpy_strm));

    REQ_CUDA(cudaStreamSynchronize(cpy_strm));

    return ScenePhysicsData(move(sdf_volumes), instance_buffer,
                            transform_buffer,
                            metadata.staticInstances.size(),
                            metadata.dynamicInstances.size(),
                            reinterpret_cast<PhysicsObject *>(
                                reinterpret_cast<char *>(start_staging_ptr) +
                                    hdr.physicsOffset),
                            reinterpret_cast<cudaTextureObject_t *>(
                                post_staging_ptr));
}

PhysicsEnvironment::PhysicsEnvironment(const ScenePhysicsData &scene_data,
                                       cudaStream_t cpy_strm)
    : instances((PhysicsInstance *)allocCU(
            (scene_data.numDefaultStatic + scene_data.numDefaultDynamic + 
             PhysicsConstants::numExtraInstances) * sizeof(PhysicsInstance))),
      transforms((PhysicsTransform *)allocCU(
              (scene_data.numDefaultStatic + scene_data.numDefaultDynamic +
               PhysicsConstants::numExtraInstances) *
              sizeof(PhysicsTransform))),
      scratchHost({
          (CollisionCandidate *)allocCU(scene_data.numDefaultDynamic * 
              (scene_data.numDefaultDynamic + scene_data.numDefaultStatic) *
              sizeof(CollisionCandidate)),
          (Contact *)allocCU(scene_data.numDefaultDynamic *
              PhysicsConstants::maxContactsPerDynamic * sizeof(Contact)),
          0,
      }),
      scratch((PhysicsScratch *)allocCU(sizeof(PhysicsScratch))),
      numStatic(scene_data.numDefaultStatic),
      numDynamic(scene_data.numDefaultDynamic),
      freeInstanceCapacity(PhysicsConstants::numExtraInstances)
{
    if (numStatic + numDynamic > 0) {
        REQ_CUDA(cudaMemcpyAsync(instances,
            scene_data.defaultInstances,
            (numStatic + numDynamic) * sizeof(PhysicsInstance),
            cudaMemcpyDeviceToDevice, cpy_strm));

        REQ_CUDA(cudaMemcpyAsync(transforms,
            scene_data.defaultTransforms,
            numDynamic * sizeof(PhysicsTransform),
            cudaMemcpyDeviceToDevice, cpy_strm));
    }

    REQ_CUDA(cudaMemcpyAsync(scratch, &scratchHost, sizeof(PhysicsScratch),
                             cudaMemcpyHostToDevice,cpy_strm));

    REQ_CUDA(cudaStreamSynchronize(cpy_strm));
}

PhysicsEnvironment::PhysicsEnvironment(PhysicsEnvironment &&o)
    : instances(o.instances),
      scratchHost(move(o.scratchHost)),
      scratch(o.scratch),
      numStatic(o.numStatic),
      numDynamic(o.numDynamic)
{
    o.instances = nullptr;
    o.transforms = nullptr;
    o.scratchHost.collisionCandidates = nullptr;
    o.scratchHost.contacts = nullptr;
    o.scratch = nullptr;
}

PhysicsEnvironment::~PhysicsEnvironment()
{
    REQ_CUDA(cudaFree(instances));
    REQ_CUDA(cudaFree(transforms));
    REQ_CUDA(cudaFree(scratch));
    REQ_CUDA(cudaFree(scratchHost.contacts));
    REQ_CUDA(cudaFree(scratchHost.collisionCandidates));
}

PhysicsSimulator::PhysicsSimulator(const PhysicsConfig &cfg)
    : cfg_(cfg),
      stream_([]() {
          cudaStream_t strm;
          REQ_CUDA(cudaStreamCreate(&strm));
          return strm;
      }()), 
      env_input_((PackedPhysicsEnv *)allocCUHost(
        sizeof(PackedPhysicsEnv) * cfg.batchSize))
{}

PhysicsSimulator::PhysicsSimulator(PhysicsSimulator &&o)
    : cfg_(o.cfg_),
      stream_(o.stream_),
      env_input_(o.env_input_)
{
    o.stream_ = nullptr;
    o.env_input_ = nullptr;
}

PhysicsSimulator::~PhysicsSimulator()
{
    cudaStreamDestroy(stream_);
    cudaFreeHost(env_input_);
}

static PackedPhysicsEnv packPhysicsEnv(const Environment &env)
{
    const OptixEnvironment &env_backend =
        *static_cast<const OptixEnvironment *>(env.getBackend());
    const OptixScene &scene = 
        *static_cast<const OptixScene *>(env.getScene().get());
    const ScenePhysicsData &scene_physics = *scene.physics;

    const PhysicsEnvironment &physics_env = *env_backend.physics;

    PackedPhysicsEnv packed;
    packed.instances = physics_env.instances;
    packed.vertexBuffer = scene.vertexPtr;
    packed.indexBuffer = scene.indexPtr;
    packed.objects = scene_physics.objectsPtr;
    packed.sdfHandles = scene_physics.sdfHandles;
    packed.transforms = env_backend.transformBuffer;
    packed.decomposedTransforms = physics_env.transforms;
    packed.scratch = physics_env.scratch;
    packed.numStatic = physics_env.numStatic;
    packed.numDynamic = physics_env.numDynamic;

    return packed;
}


void PhysicsSimulator::simulate(const Environment *envs)
{
    for (int i = 0; i < (int)cfg_.batchSize; i++) {
        env_input_[i] = packPhysicsEnv(envs[i]);
    }

    //sdfDebug(envs);

    processCollisions();
}

}
}
