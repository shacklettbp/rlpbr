#define GLM_FORCE_CUDA

#include "physics.hpp"
#include "device.cuh"
#include "cuda_utils.hpp"

#include <glm/gtx/norm.hpp>

using namespace std;

namespace RLpbr {
namespace optix {

// Numbers vaguely optimized around batch size = 1024 on an
// RTX 3090: 82 SMs, 48 warps per SM
struct CollisionConfig {
    static constexpr int warpWidth = 32;
    static constexpr int broadTBP = 128;
    static constexpr int narrowTBP = 96;
    static constexpr float broadExpansion = 2;
    static constexpr float deltaT = 1.f / 30.f;
    static constexpr int numSubsteps = 4;
    static constexpr float subDeltaT = deltaT / numSubsteps;
    static constexpr int frankWolfeIters = 8;
    static constexpr float contactThreshold = 1e-5f;
};

static __device__ bool rayBBoxIntersect(const AABB &bbox,
                                        glm::vec3 o,
                                        glm::vec3 d,
                                        float max_t,
                                        float *close_hit,
                                        float *far_hit)
{
    float t0 = 0, t1 = max_t;

    for (int i = 0; i < 3; ++i) {
        float inv_dir = 1.f / d[i];
        float t_near = (bbox.pMin[i] - o[i]) * inv_dir;
        float t_far = (bbox.pMax[i] - o[i]) * inv_dir;

        if (t_near > t_far) {
            float tmp = t_near;
            t_near = t_far;
            t_far = tmp;
        }

        t0 = t_near > t0 ? t_near : t0;
        t1 = t_far < t1 ? t_far : t1;
        if (t0 > t1) {
            return false;
        }
    }
    if (close_hit)
        *close_hit = t0;
    if (far_hit)
        *far_hit = t1;
    return true;
}

static __forceinline__ __device__ bool sphereTrace(const SDFBoundingBox &bounds,
                                   cudaTextureObject_t sdf_volume,
                                   glm::vec3 o,
                                   glm::vec3 d,
                                   float start_t,
                                   float hit_epsilon,
                                   bool force_outside,
                                   float *hit_t)
{
    glm::vec3 expanded_bbox_size = bounds.aabb.pMax - bounds.aabb.pMin +
        bounds.edgeOffset * 2.f;

    bool outside;

    float cur_t = start_t;
    for (int i = 0; i < 10000; i++) {
        glm::vec3 cur_pos = o + d * cur_t;

        glm::vec3 expanded_bbox_offset =
            cur_pos - (bounds.aabb.pMin - bounds.edgeOffset);

        glm::vec3 sdf_coords = expanded_bbox_offset / expanded_bbox_size;

        // Once point has exited bbox + edge offset, it's a miss
        if (sdf_coords.x < 0.f || sdf_coords.y < 0.f || sdf_coords.z < 0.f ||
            sdf_coords.x > 1.f || sdf_coords.y > 1.f || sdf_coords.z > 1.f) {
            return false;
        }

        float dist = tex3DLod<float>(sdf_volume, sdf_coords.x, sdf_coords.y,
                                     sdf_coords.z, 0);

        if (i == 0) {
            if (force_outside) {
                outside = true;
            } else {
                if (dist < 0.f) {
                    outside = false;
                } else {
                    outside = true;
                }
            }
        }

        if (outside) {
            cur_t += dist;
        } else {
            cur_t -= dist;
        }

        if (fabs(dist) <= hit_epsilon) {
            *hit_t = cur_t;
            return true;
        }
    }

    return false;
}

static __global__ void sdfTrace(const PackedPhysicsEnv &env,
                                Camera cam, float *out)
{
    PhysicsInstance *instances = env.instances;
    //DynamicInstance *dynamic_instances = env.dynamicPtr;

    dim3 global_coords = globalThreadIdx();

    size_t linear_idx = global_coords.y * 512 + global_coords.x;

    glm::vec3 ray_origin = cam.position;

    glm::vec2 screen((2.f * global_coords.x) / 512.f - 1.f,
                     (2.f * global_coords.y) / 512.f -1.f);

    glm::vec3 right = cam.aspectRatio * cam.tanFOV * cam.right;

    glm::vec3 up = -cam.tanFOV * cam.up;
    glm::vec3 view = cam.view;

    glm::vec3 ray_direction = right * screen.x + up * screen.y + view;

    float min_depth = 100000.f;
    for (int i = 0; i < (int)env.numStatic; i++) {
        const PhysicsInstance &inst = instances[i];
        const PhysicsObject &obj = env.objects[inst.objectID];
        const InstanceTransform &txfm = env.transforms[inst.instanceID];
        cudaTextureObject_t sdf_volume = env.sdfHandles[obj.sdfID];

        auto txfm_expanded = glm::mat4(txfm.mat);
        auto inv_expanded = glm::mat4(txfm.inv);

        auto txfm_o = inv_expanded * glm::vec4(ray_origin, 1.f);
        auto txfm_d =
            glm::normalize(inv_expanded * glm::vec4(ray_direction, 0.f));


        // FIXME, not really principled
        glm::vec3 hit_epsilon = obj.bounds.edgeOffset / 100.f; 
        float inst_epsilon = max(hit_epsilon.x, max(hit_epsilon.y, hit_epsilon.z));

        float bbox_hit_t;
        if (rayBBoxIntersect(obj.bounds.aabb, txfm_o, txfm_d, 100000.f,
                             &bbox_hit_t, nullptr)) {
            float hit_t = 0.f;
            bool force_outside = bbox_hit_t > 0.f;
            if (sphereTrace(obj.bounds, sdf_volume, txfm_o, txfm_d, bbox_hit_t,
                            inst_epsilon, force_outside, &hit_t)) {
                glm::vec3 hit_pos = txfm_o + txfm_d * hit_t;
                glm::vec3 world_hit_pos = txfm_expanded * glm::vec4(hit_pos, 1.f);

                float depth = glm::length(world_hit_pos - ray_origin);
                if (depth < min_depth) {
                    min_depth = depth;
                }
            } 
        }
    }

    out[linear_idx] = min_depth != 100000.f ? min_depth : 0.f ;
}

__host__ void PhysicsSimulator::sdfDebug(const Environment *host_envs)
{
    float *out_tmp =
        (float *)allocCUHost(512 * 512 * sizeof(float),
                             cudaHostAllocMapped);

    dim3 thread_dims { 32, 32, 1 };
    dim3 block_dims = computeBlockDims({512, 512, 1}, thread_dims);

    sdfTrace<<<block_dims, thread_dims, 0, stream_>>>(
        env_input_[0], host_envs[0].getCamera(), out_tmp);

    REQ_CUDA(cudaStreamSynchronize(stream_));

    ofstream dump("/tmp/spheretrace");
    dump.write((char *)out_tmp, 512*512*sizeof(float));

    cudaFreeHost(out_tmp);
}

static __device__ glm::vec3 transformPoint(const glm::mat4x3 &txfm,
                                           const glm::vec3 &point)
{
    return glm::mat3(txfm) * point + txfm[3];
}

static __device__ glm::vec3 transformVector(const glm::mat4x3 &txfm,
                                            const glm::vec3 &vec)
{
    return glm::mat3(txfm) * vec;
}

static __device__ glm::vec3 transformNormal(const glm::mat4x3 &txfm,
                                            const glm::vec3 &vec)
{
    return glm::transpose(glm::mat3(txfm)) * vec;
}

static __device__ AABB transformAABB(const glm::mat4x3 &txfm, const AABB &bbox)
{
    return {
        transformPoint(txfm, bbox.pMin),
        transformPoint(txfm, bbox.pMax),
    };
}

static __device__ bool checkAABBOverlap(const AABB &a, const AABB &b,
                                        const glm::vec3 &eps)
{
    return a.pMin.x - eps.x < b.pMax.x && b.pMin.x - eps.x < a.pMax.x &&
        a.pMin.y - eps.y < b.pMax.y && b.pMin.y - eps.y < a.pMax.y &&
        a.pMin.z - eps.z < b.pMax.z && b.pMin.z - eps.z < a.pMax.z;
}

static __device__ glm::vec3 unpackPosition(const DevicePackedVertex &v)
{
    float4 data0 = v.data[0];
    return glm::vec3(data0.x, data0.y, data0.z);
}

struct SDFTransform {
    glm::vec3 expandedOffset;
    glm::vec3 expandedSize;
};

static __device__ SDFTransform getSDFTransform(const SDFBoundingBox &bounds)
{
    return SDFTransform {
        bounds.aabb.pMin - bounds.edgeOffset,
        bounds.aabb.pMax - bounds.aabb.pMin + bounds.edgeOffset * 2.f,
    };
}

static __device__ glm::vec3 getSDFCoords(const SDFTransform &txfm,
                                         const glm::vec3 &pos)
{
    glm::vec3 offset = pos - txfm.expandedOffset;
    return offset / txfm.expandedSize;
}

static __device__ float sampleVolume(cudaTextureObject_t volume,
                                     const glm::vec3 &coords)
{
    return tex3DLod<float>(volume, coords.x, coords.y, coords.z, 0);
}

static __device__ float sampleSDF(cudaTextureObject_t sdf_volume,
                                  const SDFTransform &txfm,
                                  const glm::vec3 &pos)
{
    glm::vec3 sdf_coords = getSDFCoords(txfm, pos);

    return sampleVolume(sdf_volume, sdf_coords);
}

// Tetrahedron method:
// https://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
template <bool normalize = true>
static __device__ glm::vec3 computeSDFGradient(cudaTextureObject_t sdf_volume,
                                               const SDFTransform &txfm,
                                               const glm::vec3 &pos,
                                               float h)
{
    constexpr glm::vec3 k0(1, -1, -1);
    constexpr glm::vec3 k1(-1, -1, 1);
    constexpr glm::vec3 k2(-1, 1, -1);
    constexpr glm::vec3 k3(1, 1, 1);

    glm::vec3 base_coords = getSDFCoords(txfm, pos);

    glm::vec3 m = 
        k0 * sampleVolume(sdf_volume, base_coords + k0 * h) +
        k1 * sampleVolume(sdf_volume, base_coords + k1 * h) +
        k2 * sampleVolume(sdf_volume, base_coords + k2 * h) +
        k3 * sampleVolume(sdf_volume, base_coords + k3 * h);

    if constexpr (normalize) {
        return rsqrtf(glm::length2(m)) * m;
    } else {
        return m / 4.f;
    }
}

template <typename T, typename U>
struct DevicePair {
    T first;
    U second;
};

static __device__ DevicePair<glm::vec3, float> triangleCentroidRadius(
    const glm::vec3 *tri)
{
    glm::vec3 centroid = (tri[0] + tri[1] + tri[2]) / 3.f;
    float max_squared = max(glm::length2(tri[0] - centroid), max(
            glm::length2(tri[1] - centroid), glm::length2(tri[2] - centroid)));

    return {
        centroid,
        sqrtf(max_squared),
    };
}

static __global__ void generateContacts(const PackedPhysicsEnv &env)
{
    int check_idx = blockIdx.y;

    PhysicsScratch &scratch = *env.scratch;

    const CollisionCandidate &candidate = scratch.collisionCandidates[check_idx];

    const PhysicsInstance &dyn_inst = env.instances[candidate.dynInstance];
    const PhysicsObject &dyn_obj = env.objects[dyn_inst.objectID];

    const PhysicsInstance &o_inst = env.instances[candidate.otherInstance];
    bool other_is_static = candidate.otherInstance < env.numStatic;

    glm::mat4x3 dyn_o2w = env.transforms[dyn_inst.instanceID].mat;
    glm::mat4x3 other_o2w = env.transforms[o_inst.instanceID].mat;
    glm::mat4x3 other_w2o = env.transforms[o_inst.instanceID].inv;
    glm::mat4x3 to_other = glm::mat4(other_w2o) * glm::mat4(dyn_o2w);

    // FIXME should really force uniform scale...
    glm::vec3 scaled_threshold = glm::mat3(other_w2o) *
        glm::vec3(CollisionConfig::contactThreshold);
    float contact_threshold = min(scaled_threshold.x, min(
            scaled_threshold.y, scaled_threshold.z));

    const PhysicsObject &other_obj = env.objects[o_inst.objectID];
    cudaTextureObject_t sdf = env.sdfHandles[other_obj.sdfID];
    auto sdf_transform = getSDFTransform(other_obj.bounds);

    for (int tri_idx = threadIdx.x; tri_idx < (int)dyn_obj.numTriangles;
         tri_idx += CollisionConfig::narrowTBP) {
        uint32_t base_index = dyn_obj.indexOffset + 3 * tri_idx;
        glm::vec3 tri[] {
            transformPoint(to_other, unpackPosition(
                env.vertexBuffer[env.indexBuffer[base_index]])),
            transformPoint(to_other, unpackPosition(
                env.vertexBuffer[env.indexBuffer[base_index + 1]])),
            transformPoint(to_other, unpackPosition(
                env.vertexBuffer[env.indexBuffer[base_index + 2]])),
        };

        auto [tri_centroid, tri_radius] = triangleCentroidRadius(tri);

        float centroid_dist = sampleSDF(sdf, sdf_transform, tri_centroid);
        if (centroid_dist > tri_radius) {
            continue;
        }

        glm::vec3 xi;
        float min_dist = INFINITY;
        for (int tri_vert = 0; tri_vert < 3; tri_vert++) {
            float dist = sampleSDF(sdf, sdf_transform, tri[tri_vert]);
            if (dist < min_dist) {
                min_dist = dist;
                xi = tri[tri_vert];
            }
        }

        for (int i = 0; i < CollisionConfig::frankWolfeIters; i++) {
            glm::vec3 cur_gradient = computeSDFGradient(sdf, sdf_transform,
                xi, other_obj.bounds.derivativeOffset);

            float min_dot = INFINITY;
            glm::vec3 si;
            for (int tri_vert = 0; tri_vert < 3; tri_vert++) {
                float dot = glm::dot(tri[tri_vert], cur_gradient);
                if (dot < min_dot) {
                    min_dot = dot;
                    si = tri[tri_vert];
                }
            }

            float alpha = 2.f / (float(i) + 2.f);
            xi = xi + alpha * (si - xi);
        }

        float final_dist = sampleSDF(sdf, sdf_transform, xi);
        if (final_dist < contact_threshold) {
            uint32_t contact_idx = atomicAdd(&scratch.numContacts, 1);

            // obj_normal points towards xi (if outside shape)
            glm::vec3 contact_normal = computeSDFGradient<true>(sdf,
                sdf_transform, xi, other_obj.bounds.derivativeOffset);

            glm::vec3 pos_b = xi - final_dist * contact_normal;

            glm::vec3 pos_a = transformPoint(other_o2w, xi);
            pos_b = transformPoint(other_o2w, pos_b);

            contact_normal = transformNormal(other_w2o, contact_normal);
            
            scratch.contacts[contact_idx] = {
                contact_normal,
                pos_a,
                pos_b,
                candidate.dynInstance,
                other_is_static ? ~0u : candidate.otherInstance,
            };
        }
    }
}

static __global__ void physicsEntry(const PackedPhysicsEnv *envs)
{
    // FIXME: maybe also sum up total triangles that need to be
    // checked? Could use it to dispatch more blocks
    __shared__ uint32_t num_checks;

    if (threadIdx.x == 0) {
        num_checks = 0;
    }
    __syncthreads();

    const PackedPhysicsEnv &env = envs[blockIdx.y];
    PhysicsScratch &scratch = *env.scratch;

    // Each block operates on one environment.
    // Each warp is assigned to one dynamic instance: warp loops over
    // all other static and dynamic objects, checking bboxes.
    // Block emits kernel launch for an environment's collisions

    int thread_id = threadIdx.x;
    int warp_id = thread_id / CollisionConfig::warpWidth;
    int lane_id = thread_id % CollisionConfig::warpWidth;

    int total_insts = env.numDynamic + env.numStatic;

    for (int inst_idx = warp_id; inst_idx < (int)env.numDynamic; inst_idx += 
         CollisionConfig::broadTBP / CollisionConfig::warpWidth) {
        const PhysicsInstance &inst = env.instances[env.numStatic + inst_idx];
        const PhysicsObject &obj = env.objects[inst.objectID];
        const InstanceTransform &txfm = env.transforms[inst.instanceID];

        AABB bbox = transformAABB(txfm.mat, obj.bounds.aabb);

        for (int o_idx = lane_id; o_idx < total_insts;
             o_idx += CollisionConfig::warpWidth) {

            const PhysicsInstance &o_inst = env.instances[o_idx];

            const InstanceTransform &o_txfm = env.transforms[o_inst.instanceID];
            const PhysicsObject &o_obj = env.objects[o_inst.objectID];

            AABB o_bbox = transformAABB(o_txfm.mat, o_obj.bounds.aabb);

            // FIXME change edgeOffset to half cell width, not 1.5 cell width
            glm::vec3 eps =
                transformVector(o_txfm.mat, o_obj.bounds.edgeOffset / 3.f);

            if (checkAABBOverlap(bbox, o_bbox, eps)) {
                uint32_t out_idx = atomicAdd_block(&num_checks, 1);
                scratch.collisionCandidates[out_idx] = {
                    uint32_t(inst_idx),
                    uint32_t(o_idx),
                };
            }
        }
    }


    for (int substep = 0; substep < CollisionConfig::numSubsteps; substep++) {
        __syncthreads();
        if (threadIdx.x == 0) {
            scratch.numContacts = 0;

            dim3 thread_dims { CollisionConfig::narrowTBP, 1, 1};
            dim3 block_dims { 1, num_checks, 1 };
            generateContacts<<<block_dims, thread_dims, 0, 0>>>(env);

            cudaDeviceSynchronize();
            //printf("%u %u\n", num_checks, scratch.numContacts);
        }
        __syncthreads();

        // Update positions

        __syncthreads();


    }

}

__host__ void PhysicsSimulator::processCollisions()
{
    //sdfDebug(env_input_, envs, stream_);

    // Launch one block per environment
    dim3 thread_dims { CollisionConfig::broadTBP, 1, 1 };
    dim3 block_dims = { 1, cfg_.batchSize, 1};
    physicsEntry<<<block_dims, thread_dims, 4, stream_>>>(env_input_);

    REQ_CUDA(cudaStreamSynchronize(stream_));
}

}
}
