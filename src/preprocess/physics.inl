#include "physics.hpp"

#include <iostream>
#include <fstream>
#include <meshoptimizer.h>
#include <igl/signed_distance.h>

#include <glm/gtx/string_cast.hpp>

#include <rlpbr_core/utils.hpp>


#include "preprocess.hpp"

namespace RLpbr {

template <typename VertexType>
static AABB computeAABB(const VertexType *vertices,
                        const uint32_t *indices,
                        uint32_t num_indices)
{
    AABB bounds {
        glm::vec3(INFINITY, INFINITY, INFINITY),
        glm::vec3(-INFINITY, -INFINITY, -INFINITY),
    };

    auto updateBounds = [&bounds](const glm::vec3 &point) {
        if (point.x < bounds.pMin.x) {
            bounds.pMin.x = point.x;
        }

        if (point.y < bounds.pMin.y) {
            bounds.pMin.y = point.y;
        }

        if (point.z < bounds.pMin.z) {
            bounds.pMin.z = point.z;
        }

        if (point.x > bounds.pMax.x) {
            bounds.pMax.x = point.x;
        }

        if (point.y > bounds.pMax.y) {
            bounds.pMax.y = point.y;
        }

        if (point.z > bounds.pMax.z) {
            bounds.pMax.z = point.z;
        }
    };

    for (int i = 0; i < (int)num_indices; i++) {
        uint32_t idx = indices[i];
        updateBounds(vertices[idx].position);
    }

    return bounds;
}

template <typename VertexType>
static PhysicsMeshProperties getMeshProperties(const VertexType *vertices,
                                               const uint32_t *indices,
                                               uint32_t num_indices)
{
    const float density = 1.f;

    // Blow, Binstock, 2004
    const glm::mat3 covar_canonical(1.f / 60.f, 1.f / 120.f, 1.f / 120.f,
                                    1.f / 120.f, 1.f / 60.f, 1.f / 120.f,
                                    1.f / 120.f, 1.f / 120.f, 1.f / 60.f);

    float mass_total = 0.f;
    glm::vec3 com_total(0.f);
    glm::mat3 covar_total(0.f);
    for (int index_idx = 0; index_idx < (int)num_indices; index_idx += 3) {
        glm::u32vec3 tri_indices(indices[index_idx], indices[index_idx + 1],
                                 indices[index_idx + 2]);

        glm::vec3 w1 = vertices[tri_indices.x].position;
        glm::vec3 w2 = vertices[tri_indices.y].position;
        glm::vec3 w3 = vertices[tri_indices.z].position;

        glm::mat3 A(w1, w2, w3);
        float detA = glm::determinant(A);

        glm::mat3 covar_target =
            detA * A * covar_canonical * glm::transpose(A);
        float mass = density / 6.f * detA;
        glm::vec3 com = (w1 + w2 + w3) / 4.f;

        float old_mass = mass_total;
        mass_total += mass;
        com_total = (com_total * old_mass + com * mass) / mass_total;
        covar_total += covar_target;
    }

    covar_total -= mass_total * glm::outerProduct(com_total, com_total);

    float trace = covar_total[0][0] + covar_total[1][1] + covar_total[2][2];
    glm::mat3 inertia_mat = trace * glm::mat3(1.f) - covar_total;

    // FIXME this inertia diagonalization is only valid if the object
    // is rotated around its principle axes

    return PhysicsMeshProperties {
        glm::vec3(inertia_mat[0][0], inertia_mat[1][1], inertia_mat[2][2]),
        com_total,
        mass_total,
    };
}

// https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
[[maybe_unused]]
static glm::vec3 closestTrianglePoint(const glm::vec3 &x,
                                      const glm::vec3 &p0,
                                      const glm::vec3 &p1,
                                      const glm::vec3 &p2)
{
    glm::vec3 diff = x - p0;
    glm::vec3 e0 = p1 - p0;
    glm::vec3 e1 = p2 - p0;

    float a00 = glm::dot(e0, e0);
    float a01 = glm::dot(e0, e1);
    float a11 = glm::dot(e1, e1);
    float b0 = -glm::dot(diff, e0);
    float b1 = -glm::dot(diff, e1);
    float det = a00 * a11 - a01 * a01;
    float t0 = a01 * b1 - a11 * b0;
    float t1 = a01 * b0 - a00 * b1;

    if (t0 + t1 <= det) {
        if (t0 < 0.f) {
            if (t1 < 0.f) { // region 4
                if (b0 < 0.f) {
                    t1 = 0.f;
                    if (-b0 >= a00) { // V1
                        t0 = 1.f;
                    } else { // E01
                        t0 = -b0 / a00;
                    }
                } else {
                    t0 = 0.f;
                    if (b1 >= 0.f) { // V0
                        t1 = 0.f;
                    } else if (-b1 >= a11) { // V2
                        t1 = 1.f;
                    } else { // E20
                        t1 = -b1 / a11;
                    }
                }
            } else { // region 3
                t0 = 0.f;
                if (b1 >= 0.f) { // V0
                    t1 = 0.f;
                } else if (-b1 >= a11) { // V2
                    t1 = 1.f;
                } else { // E20
                    t1 = -b1 / a11;
                }
            }
        } else if (t1 < 0.f) { // region 5
            t1 = 0.f;
            if (b0 >= 0.f) { // V0
                t0 = 0.f;
            } else if (-b0 >= a00) { // V1
                t0 = 1.f;
            } else { // E01
                t0 = -b0 / a00;
            }
        } else { // region 0, interior
            float invDet = 1.f / det;
            t0 *= invDet;
            t1 *= invDet;
        }
    } else {
        float tmp0, tmp1, numer, denom;

        if (t0 < 0.f) { // region 2
            tmp0 = a01 + b0;
            tmp1 = a11 + b1;
            if (tmp1 > tmp0) {
                numer = tmp1 - tmp0;
                denom = a00 - 2.f * a01 + a11;
                if (numer >= denom) { // V1
                    t0 = 1.f;
                    t1 = 0.f;
                } else { // E12
                    t0 = numer / denom;
                    t1 = 1.f - t0;
                }
            } else {
                t0 = 0.f;
                if (tmp1 <= 0.f) { // V2
                    t1 = 1.f;
                } else if (b1 >= 0.f) { // V0
                    t1 = 0.f;
                } else { // E20
                    t1 = -b1 / a11;
                }
            }
        } else if (t1 < 0.f) { // region 6
            tmp0 = a01 + b1;
            tmp1 = a00 + b0;
            if (tmp1 > tmp0) {
                numer = tmp1 - tmp0;
                denom = a00 - 2.f * a01 + a11;
                if (numer >= denom) { // V2
                    t1 = 1.f;
                    t0 = 0.f;
                } else { // E12
                    t1 = numer / denom;
                    t0 = 1.f - t1;
                }
            } else {
                t1 = 0.f;
                if (tmp1 <= 0.f) { // V1
                    t0 = 1.f;
                } else if (b0 >= 0.f) { // V0
                    t0 = 0.f;
                } else { // E01
                    t0 = -b0 / a00;
                }
            }
        } else { // region 1
            numer = a11 + b1 - a01 - b0;
            if (numer <= 0.f) { // V2
                t0 = 0.f;
                t1 = 1.f;
            } else {
                denom = a00 - 2.f * a01 + a11;
                if (numer >= denom) { // V1
                    t0 = 1.f;
                    t1 = 0.f;
                } else { // 12
                    t0 = numer / denom;
                    t1 = 1.f - t0;
                }
            }
        }
    }

    return p0 + t0 * e0 + t1 * e1;
}

template <typename VertexType>
static SDF computeSDF(const VertexType *src_vertices,
                      const uint32_t *src_indices,
                      uint32_t num_indices,
                      const AABB &bbox)
{
    using namespace std;

    // Delete duplicate vertices (by position), convert to Eigen's format
    vector<glm::vec3> vertex_positions;
    vertex_positions.reserve(num_indices);
    for (int index_idx = 0; index_idx < (int)num_indices; index_idx++) {
        uint32_t idx = src_indices[index_idx];
        vertex_positions.push_back(src_vertices[idx].position);
    }

    DynArray<uint32_t> remap(num_indices);
    uint32_t num_vertices = meshopt_generateVertexRemap(
        remap.data(), nullptr, num_indices, vertex_positions.data(),
        num_indices, sizeof(glm::vec3));
    DynArray<glm::vec3> vertices(num_vertices);
    DynArray<uint32_t> indices(num_indices);
    meshopt_remapIndexBuffer(indices.data(), nullptr, num_indices,
                             remap.data());
    meshopt_remapVertexBuffer(vertices.data(), vertex_positions.data(),
                              num_indices, sizeof(glm::vec3), remap.data());

    uint32_t num_triangles = num_indices / 3;

    Eigen::MatrixXf igl_verts(num_vertices, 3);
    Eigen::MatrixXi igl_faces(num_triangles, 3);

    for (int i = 0; i < (int)num_vertices; i++) {
        igl_verts(i, 0) = vertices[i].x;
        igl_verts(i, 1) = vertices[i].y;
        igl_verts(i, 2) = vertices[i].z;
    }

    for (int tri_idx = 0; tri_idx < (int)num_triangles; tri_idx++) {
        igl_faces(tri_idx, 0) = indices[3 * tri_idx];
        igl_faces(tri_idx, 1) = indices[3 * tri_idx + 1];
        igl_faces(tri_idx, 2) = indices[3 * tri_idx + 2];
    }

    [[maybe_unused]]
    auto meshDistance = [&vertices, &indices, num_indices](
        const glm::vec3 &p) {
        float min_squared_dist = INFINITY;

        for (int index_idx = 0; index_idx < (int)num_indices; index_idx += 3) {
            const glm::vec3 &a = vertices[indices[index_idx]];
            const glm::vec3 &b = vertices[indices[index_idx + 1]];
            const glm::vec3 &c = vertices[indices[index_idx + 2]];

            glm::vec3 closest_candidate = closestTrianglePoint(p, a, b, c);

            glm::vec3 to_candidate = closest_candidate - p;
            float squared_dist = glm::dot(to_candidate, to_candidate);

            if (squared_dist < min_squared_dist) {
                min_squared_dist = squared_dist;
            }
        }

        return sqrtf(min_squared_dist);
    };

    glm::vec3 dist_per_sample = SDFConfig::sdfSampleResolution;

    glm::vec3 bbox_dims = bbox.pMax - bbox.pMin;
    glm::vec3 num_samples_frac = (bbox_dims / dist_per_sample) + 1.f;
    glm::u32vec3 num_samples(ceilf(num_samples_frac.x),
                             ceilf(num_samples_frac.y),
                             ceilf(num_samples_frac.z));

    printf("Generating SDF: %u %u %u\n", num_samples.x, num_samples.y,
           num_samples.z);

    dist_per_sample = bbox_dims / glm::vec3(num_samples - 1u);

    // Add extra layer of outside bbox samples
    num_samples += 2u;
    glm::vec3 expanded_min = bbox.pMin - dist_per_sample;

    uint32_t total_samples = num_samples.x * num_samples.y * num_samples.z;

    Eigen::MatrixXf grid_positions(total_samples, 3);

    for (int k = 0; k < (int)num_samples.z; k++) {
        for (int j = 0; j < (int)num_samples.y; j++) {
            for (int i = 0; i < (int)num_samples.x; i++) {
                glm::vec3 cell_offset(dist_per_sample.x * i, dist_per_sample.y * j,
                                      dist_per_sample.z * k);

                glm::vec3 pos = expanded_min + cell_offset;

                int linear_idx = (k * num_samples.y + j) * num_samples.x + i;
                grid_positions(linear_idx, 0) = pos.x;
                grid_positions(linear_idx, 1) = pos.y;
                grid_positions(linear_idx, 2) = pos.z;
            }
        }
    }

    Eigen::VectorXf signed_distances;
    Eigen::VectorXi tri_indices;
    Eigen::MatrixXf closest_points;
    Eigen::MatrixXf closest_normals;

    igl::signed_distance(grid_positions, igl_verts, igl_faces,
                         igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER,
                         signed_distances,
                         tri_indices, closest_points, closest_normals);

    vector<float> grid(num_samples.x * num_samples.y * num_samples.z);
    memcpy(grid.data(), signed_distances.data(), sizeof(float) * total_samples);

    cout << "SDF generated" << endl;

    float min_dist_per_sample = dist_per_sample.x;
    float min_dist_samples = num_samples.x;

    if (dist_per_sample.y < min_dist_per_sample) {
        min_dist_per_sample = dist_per_sample.y;
        min_dist_samples = num_samples.y;
    }

    if (dist_per_sample.z < min_dist_per_sample) {
        min_dist_per_sample = dist_per_sample.z;
        min_dist_samples = num_samples.z;
    }

    return SDF {
        num_samples,
        move(grid),
        dist_per_sample * 1.5f,
        (min_dist_per_sample / min_dist_samples) / 10.f,
    };
}

template <typename VertexType>
PhysicsMeshInfo PhysicsMeshInfo::make(const VertexType *vertices,
                                      const uint32_t *indices,
                                      uint32_t num_indices,
                                      bool skip_sdf)
{
    using namespace std;

    auto mesh_props = getMeshProperties(vertices, indices, num_indices);

    AABB bbox = computeAABB(vertices, indices, num_indices);

    SDF sdf {};
    if (!skip_sdf) {
        sdf = computeSDF(vertices, indices, num_indices, bbox);
    }

    return PhysicsMeshInfo {
        mesh_props,
        bbox,
        move(sdf),
    };
}

template <typename VertexType>
ProcessedPhysicsState ProcessedPhysicsState::make(
    const ProcessedGeometry<VertexType> &geometry,
    bool skip_sdfs)
{
    using namespace std;

    vector<SDF> sdfs;
    vector<PhysicsObject> physics_objects;

    for (uint32_t obj_id = 0; obj_id < geometry.objectInfos.size();
         obj_id++) {
        const auto &obj_info = geometry.objectInfos[obj_id];
        uint32_t index_offset =
            geometry.meshInfos[obj_info.meshIndex].indexOffset;
        uint32_t num_triangles = 0;
        for (int mesh_idx = 0; mesh_idx < (int)obj_info.numMeshes;
             mesh_idx++) {
            num_triangles +=
                geometry.meshInfos[obj_info.meshIndex + mesh_idx].numTriangles;
        }

        auto physics_info = PhysicsMeshInfo::make(geometry.vertices.data(),
            geometry.indices.data() + index_offset,
            num_triangles * 3, skip_sdfs);

        sdfs.emplace_back(move(physics_info.sdf));
        uint32_t sdf_id = sdfs.size() - 1;

        physics_objects.push_back({
            {
                physics_info.bbox,
                sdfs.back().edgeOffset,
                sdfs.back().derivativeOffset,
            },
            sdf_id,
            physics_info.meshProps.interia,
            physics_info.meshProps.com,
            physics_info.meshProps.mass,
            index_offset,
            num_triangles,
        });
    }

    return ProcessedPhysicsState {
        move(sdfs),
        move(physics_objects),
    };
}

void SDF::dump(const std::string_view dump_path) const
{
    std::ofstream dump_file(std::filesystem::path{dump_path});
    dump_file.write(reinterpret_cast<const char *>(&numCells),
                    sizeof(glm::u32vec3));
    dump_file.write(reinterpret_cast<const char *>(grid.data()),
                    grid.size() * sizeof(float));

    dump_file.close();
}

}  // namespace RLpbr
