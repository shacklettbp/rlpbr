#include <glm/gtx/quaternion.hpp>
#include <rlpbr/preprocess.hpp>
#include <rlpbr_core/utils.hpp>

#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>

#include <glm/gtc/type_precision.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/transform.hpp>
#include <meshoptimizer.h>
#include <mikktspace.h>

#include "import.hpp"
#include "physics.hpp"
#include "physics.inl"
#include "rlpbr_core/scene.hpp"

using namespace std;

namespace RLpbr {

using namespace SceneImport;

struct PreprocessData {
    SceneDescription<Vertex, Material> desc;
    string dataDir;
    bool dumpSDFs;
};

static PreprocessData parseSceneData(string_view scene_path,
                                     const glm::mat4 &base_txfm,
                                     optional<string_view> data_dir,
                                     bool dump_textures,
                                     bool dump_sdfs)
{
    string serialized_data_dir;
    if (!data_dir.has_value()) {
        serialized_data_dir = "./";
    } else {
        serialized_data_dir = data_dir.value();
    }
    return PreprocessData {
        SceneDescription<Vertex, Material>::parseScene(scene_path, base_txfm,
            dump_textures ? data_dir : optional<string_view>()),
        serialized_data_dir,
        dump_sdfs,
    };
}

ScenePreprocessor::ScenePreprocessor(string_view gltf_path,
                                     const glm::mat4 &base_txfm,
                                     optional<string_view> data_dir,
                                     bool dump_textures,
                                     bool dump_sdfs)
    : scene_data_(new PreprocessData(parseSceneData(gltf_path,
        base_txfm, data_dir, dump_textures, dump_sdfs)))
{}

template <typename VertexType>
static vector<uint32_t> filterDegenerateTriangles(
    const vector<VertexType> &vertices,
    const vector<uint32_t> &orig_indices)
{
    vector<uint32_t> new_indices;
    new_indices.reserve(orig_indices.size());

    uint32_t num_indices = orig_indices.size();
    uint32_t tri_align = orig_indices.size() % 3;
    if (tri_align != 0) {
        cerr << "Warning: non multiple of 3 indices in mesh" << endl;
        num_indices -= tri_align;
    }
    assert(num_indices % 3 == 0);

    for (uint32_t i = 0; i < num_indices;) {
        uint32_t a_idx = orig_indices[i++];
        uint32_t b_idx = orig_indices[i++];
        uint32_t c_idx = orig_indices[i++];

        if (a_idx >= vertices.size() || b_idx >= vertices.size() ||
            c_idx >= vertices.size()) {
            continue;
        }

        glm::vec3 a = vertices[a_idx].position;
        glm::vec3 b = vertices[b_idx].position;
        glm::vec3 c = vertices[c_idx].position;

        glm::vec3 ab = a - b;
        glm::vec3 bc = b - c;
        float check = glm::length2(glm::cross(ab, bc));

        if (check < 1e-20f) {
            continue;
        }

        new_indices.push_back(a_idx);
        new_indices.push_back(b_idx);
        new_indices.push_back(c_idx);
    }

    uint32_t num_degenerate = orig_indices.size() - new_indices.size();

    if (num_degenerate > 0) {
        cout << "Filtered: " << num_degenerate
             << " degenerate triangles" << endl;
    }

    return new_indices;
}

static glm::vec3 encodeNormalTangent(const glm::vec3 &normal,
                                     const glm::vec4 &tangent_plussign)
{
    // https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
    auto octWrap = [](const glm::vec2 &v) {
        return glm::vec2 {
            (1.f - fabsf(v.y)) * (v.x >= 0.f ? 1.f : -1.f),
            (1.f - fabsf(v.x)) * (v.y >= 0.f? 1.f : -1.f),
        };
    };
 
    auto octEncode = [&octWrap](glm::vec3 n) {
        n /= (fabsf(n.x) + fabsf(n.y) + fabsf(n.z));

        glm::vec2 nxy(n.x, n.y);

        nxy = n.z >= 0.0f ? nxy : octWrap(nxy);
        nxy = nxy * 0.5f + 0.5f;
        return nxy;
    };

    glm::vec3 tangent = tangent_plussign;
    float bitangent_sign = tangent_plussign.w;

    uint32_t nxy = glm::packHalf2x16(glm::vec2(normal.x, normal.y));
    uint32_t nzsign = glm::packHalf2x16(glm::vec2(normal.z, bitangent_sign));

    glm::vec2 octtan = octEncode(tangent);
    uint32_t octtan_snorm = glm::packSnorm2x16(octtan);

    // FIXME: c++20 bit_cast
    
    glm::vec3 packed;
    memcpy(&packed.x, &nxy, sizeof(float));
    memcpy(&packed.y, &nzsign, sizeof(float));
    memcpy(&packed.z, &octtan_snorm, sizeof(float));

    return packed;
}

template <typename VertexType>
optional<Mesh<PackedVertex>> processMesh(const Mesh<VertexType> &orig_mesh)
{
    const vector<VertexType> &orig_vertices = orig_mesh.vertices;
    const vector<uint32_t> &orig_indices = orig_mesh.indices;

    vector<uint32_t> filtered_indices =
        filterDegenerateTriangles(orig_vertices, orig_indices);

    if (filtered_indices.size() == 0) {
        cerr << "Warning: removing entire degenerate mesh" << endl;
        return optional<Mesh<PackedVertex>>();
    }

    uint32_t num_indices = filtered_indices.size();

    // Unweld mesh and normalize vertices (maybe have been unnormalized
    // by static rescaling)
    vector<VertexType> unwelded_vertices;
    unwelded_vertices.reserve(num_indices);
    for (uint32_t idx : filtered_indices) {
        Vertex new_vert = orig_vertices[idx];
        new_vert.normal = normalize(new_vert.normal);
        unwelded_vertices.push_back(new_vert);
    }

    DynArray<glm::vec4> raw_tangents(unwelded_vertices.size());

    struct TSpaceUserData {
        uint32_t numTris;
        const VertexType *vertices;
        glm::vec4 *tangents;

        const VertexType & getVertex(int32_t face, int32_t vert) {
            return vertices[face * 3 + vert];
        };
    };

    TSpaceUserData tspace_user {
        num_indices / 3,
        unwelded_vertices.data(),
        raw_tangents.data(),
    };

    // Generate tangents
    SMikkTSpaceInterface mikktspace = {};

    mikktspace.m_getNumFaces = [](const SMikkTSpaceContext* ctx) {
        return (int32_t)((TSpaceUserData *)(ctx->m_pUserData))->numTris;
    };

    mikktspace.m_getNumVerticesOfFace = [](const SMikkTSpaceContext*, int32_t) {
        return 3;
    };

    mikktspace.m_getPosition = [](const SMikkTSpaceContext* ctx,
                                  float position[],
                                  int32_t face,
                                  int32_t vert) {
        auto user = ((TSpaceUserData*)(ctx->m_pUserData));
        const glm::vec3 &pos = user->getVertex(face, vert).position;
        position[0] = pos.x;
        position[1] = pos.y;
        position[2] = pos.z;
    };

    mikktspace.m_getNormal = [](const SMikkTSpaceContext* ctx,
                                float normal[],
                                int32_t face,
                                int32_t vert) {
        auto user = ((TSpaceUserData*)(ctx->m_pUserData));
        const glm::vec3 &n = user->getVertex(face, vert).normal;
        normal[0] = n.x;
        normal[1] = n.y;
        normal[2] = n.z;
    };

    mikktspace.m_getTexCoord = [](const SMikkTSpaceContext* ctx,
                                  float uv[],
                                  int32_t face,
                                  int32_t vert) {
        auto user = ((TSpaceUserData*)(ctx->m_pUserData));
        const glm::vec2 &u = user->getVertex(face, vert).uv;
        uv[0] = u.x;
        uv[1] = u.y;
    };

    mikktspace.m_setTSpaceBasic = [](const SMikkTSpaceContext* ctx,
                                     const float tangent[],
                                     float sign,
                                     int32_t face,
                                     int32_t vert) {
        auto user = ((TSpaceUserData*)(ctx->m_pUserData));

        user->tangents[face * 3 + vert] =
            glm::vec4(tangent[0], tangent[1], tangent[2], sign);
    };

    SMikkTSpaceContext tspace_ctx = {};
    tspace_ctx.m_pInterface = &mikktspace;
    tspace_ctx.m_pUserData = &tspace_user;

    if (!genTangSpaceDefault(&tspace_ctx)) {
        cerr << "Failed to generate tangents for mesh" << endl;
        abort();
    }

    vector<PackedVertex> packed_vertices;
    packed_vertices.reserve(unwelded_vertices.size());
    for (int v_idx = 0; v_idx < (int)unwelded_vertices.size(); v_idx++) {
        const auto &o_v = unwelded_vertices[v_idx];
        const glm::vec4 &tangent_plussign = raw_tangents[v_idx];
        packed_vertices.push_back({
            o_v.position,
            encodeNormalTangent(o_v.normal, tangent_plussign),
            o_v.uv,
        });
    }

    vector<uint32_t> index_remap(num_indices);
    size_t new_vertex_count =
        meshopt_generateVertexRemap(index_remap.data(),
                                    nullptr, num_indices,
                                    packed_vertices.data(),
                                    num_indices,
                                    sizeof(PackedVertex));

    vector<uint32_t> new_indices(num_indices);
    vector<PackedVertex> new_vertices(new_vertex_count);

    meshopt_remapIndexBuffer(new_indices.data(), nullptr,
                             num_indices, index_remap.data());

    meshopt_remapVertexBuffer(new_vertices.data(), packed_vertices.data(),
                              num_indices, sizeof(PackedVertex),
                              index_remap.data());

    meshopt_optimizeVertexCache(new_indices.data(), new_indices.data(),
                                num_indices, new_vertex_count);

    new_vertex_count = meshopt_optimizeVertexFetch(new_vertices.data(),
                                                   new_indices.data(),
                                                   num_indices,
                                                   new_vertices.data(),
                                                   new_vertex_count,
                                                   sizeof(PackedVertex));
    new_vertices.resize(new_vertex_count);

    return Mesh<PackedVertex> {
        move(new_vertices),
        move(new_indices),
    };
}

template <typename VertexType>
optional<pair<Object<PackedVertex>, vector<uint32_t>>>
processObject(const Object<VertexType> &orig_obj)
{
    Object<PackedVertex> obj;
    obj.name = orig_obj.name;

    vector<uint32_t> removed_meshes;

    for (uint32_t mesh_idx = 0; mesh_idx < orig_obj.meshes.size(); mesh_idx++) {
        const auto &mesh = orig_obj.meshes[mesh_idx];
        auto opt_mesh = processMesh(mesh);

        if (opt_mesh.has_value()) {
            obj.meshes.emplace_back(move(*opt_mesh));
        } else {
            removed_meshes.push_back(mesh_idx);
        }
    }

    if (!obj.meshes.empty()) {
        return make_pair(move(obj), move(removed_meshes));
    } else {
        return optional<pair<Object<PackedVertex>, vector<uint32_t>>>();
    }
}

template <typename VertexType, typename MaterialType>
static tuple<ProcessedGeometry<PackedVertex>,
            vector<unordered_map<glm::vec3, uint32_t>>,
            vector<vector<uint32_t>>>
processGeometry(const vector<Object<VertexType>> &orig_objects,
                const vector<unordered_set<glm::vec3>> &obj_scales)
{
    vector<Object<PackedVertex>> processed_objects;
    // For each processed object, a list of the object's removed mesh indices
    vector<vector<uint32_t>> removed_meshes;

    vector<unordered_map<glm::vec3, uint32_t>> obj_id_remap(orig_objects.size());
    {
        vector<Object<VertexType>> scaled_objects;

        // Map from scaled object to orig + scale
        vector<pair<uint32_t, glm::vec3>> reverse_scale_map;

        // Bake scales
        for (int obj_idx = 0; obj_idx < (int)orig_objects.size(); obj_idx++) {
            const auto &scales = obj_scales[obj_idx];
            auto &obj = orig_objects[obj_idx];

            if (scales.empty()) { // This is a not currently instanced object
                scaled_objects.emplace_back(move(obj));
                reverse_scale_map.emplace_back(obj_idx, glm::vec3(1.f));
            } else {
                int scale_idx = 0;
                for (const glm::vec3 &scale : scales) {
                    Object<VertexType> scaled_obj;
                    scaled_obj.name = obj.name + "_" + to_string(scale_idx);
                    for (const auto &mesh : obj.meshes) {
                        Mesh<VertexType> scaled_mesh;
                        for (const VertexType &vert : mesh.vertices) {
                            VertexType new_vert = vert;
                            new_vert.position *= scale;
                            new_vert.normal *= 1.f / scale;
                            new_vert.normal = normalize(new_vert.normal);
                            scaled_mesh.vertices.push_back(new_vert);
                        }
                        scaled_mesh.indices = mesh.indices;

                        scaled_obj.meshes.emplace_back(move(scaled_mesh));
                    }
                    scaled_objects.emplace_back(move(scaled_obj));
                    reverse_scale_map.emplace_back(obj_idx, scale);

                    scale_idx++;
                }
            }
        }

        // Process objects, potentially culling degenerate objects
        vector<pair<uint32_t, glm::vec3>> culled_reverse_map;
        culled_reverse_map.reserve(reverse_scale_map.size());

        for (int scaled_idx = 0; scaled_idx < (int)scaled_objects.size();
             scaled_idx++) {
            const auto &orig_obj = scaled_objects[scaled_idx];
            auto processed = processObject<VertexType>(orig_obj);

            if (processed.has_value()) {
                auto &[obj, removed] = *processed;
                processed_objects.emplace_back(move(obj));
                removed_meshes.emplace_back(move(removed));

                const auto &[orig_id, scale] = reverse_scale_map[scaled_idx];
                culled_reverse_map.emplace_back(orig_id, scale);
            } 
        }
        assert(processed_objects.size() > 0);

        // Build obj_id_remap
        for (int processed_idx = 0;
             processed_idx < (int)processed_objects.size();
             processed_idx++) {
            const auto &[orig_id, scale] = culled_reverse_map[processed_idx];
            obj_id_remap[orig_id].emplace(scale, processed_idx);
        }
    }

    vector<PackedVertex> vertices;
    vector<uint32_t> indices;
    vector<MeshInfo> mesh_infos;
    vector<ObjectInfo> obj_infos;
    vector<string> obj_names;

    for (auto &obj : processed_objects) {
        uint32_t mesh_offset = mesh_infos.size();
        for (auto &mesh : obj.meshes) {
            mesh_infos.push_back(MeshInfo {
                uint32_t(indices.size()),
                uint32_t(mesh.indices.size() / 3),
                uint32_t(mesh.vertices.size())
            });

            // Rewrite indices to refer to the global vertex array
            // (Note this only really matters for RT to allow gl_CustomIndexEXT
            // to simply hold the base index of a mesh)
            for (uint32_t idx : mesh.indices) {
                indices.push_back(idx + vertices.size());
            }

            for (const auto &vert : mesh.vertices) {
                vertices.push_back(vert);
            }
        }

        obj_infos.push_back({
            mesh_offset,
            uint32_t(obj.meshes.size()),
        });

        obj_names.emplace_back(move(obj.name));
    }

    return {
        ProcessedGeometry<PackedVertex> {
            move(vertices),
            move(indices),
            move(mesh_infos),
            move(obj_infos),
            move(obj_names),
        },
        move(obj_id_remap),
        move(removed_meshes),
    };
}

template <typename VertexType, typename MaterialType>
SceneDescription<VertexType, MaterialType> mergeStaticInstances(
    const SceneDescription<VertexType, MaterialType> &orig_desc)
{
    using SceneDesc = SceneDescription<VertexType, MaterialType>;
    constexpr int duplication_threshold = 4;
    SceneDesc static_desc;
    SceneDesc static_transparent_desc;
    SceneDesc new_desc;

    vector<uint32_t> static_object_usage(orig_desc.objects.size());

    for (const auto &inst : orig_desc.defaultInstances) {
        if (!inst.dynamic) {
            static_object_usage[inst.objectIndex]++;
        } 
    }

    vector<uint32_t> obj_remap(orig_desc.objects.size(), ~0u);
    vector<bool> obj_merged(orig_desc.objects.size(), false);

    for (const auto &inst : orig_desc.defaultInstances) {
        if (!inst.dynamic &&
            static_object_usage[inst.objectIndex] < duplication_threshold) {
            if (inst.transparent) {
                static_transparent_desc.objects.push_back(
                    orig_desc.objects[inst.objectIndex]);
                static_transparent_desc.defaultInstances.push_back(inst);
                static_transparent_desc.defaultInstances.back().objectIndex =
                    static_transparent_desc.objects.size() - 1;
            } else {
                static_desc.objects.push_back(orig_desc.objects[inst.objectIndex]);
                static_desc.defaultInstances.push_back(inst);
                static_desc.defaultInstances.back().objectIndex =
                    static_desc.objects.size() - 1;
            }
            obj_merged[inst.objectIndex] = true;
        } else {
            uint32_t remapped = obj_remap[inst.objectIndex];
            new_desc.defaultInstances.push_back(inst);
            if (remapped == ~0u) {
                new_desc.objects.push_back(orig_desc.objects[inst.objectIndex]);
                uint32_t new_idx = new_desc.objects.size() - 1;
                new_desc.defaultInstances.back().objectIndex = new_idx;
                obj_remap[inst.objectIndex] = new_idx;
            } else {
                new_desc.defaultInstances.back().objectIndex = remapped;
            }
        }
    }

    // Include non instanced objects
    for (int orig_obj_idx = 0; orig_obj_idx < (int)orig_desc.objects.size();
         orig_obj_idx++) {
        if (obj_remap[orig_obj_idx] == ~0u && !obj_merged[orig_obj_idx]) {
            new_desc.objects.push_back(orig_desc.objects[orig_obj_idx]);
        }
    }

    if (static_desc.defaultInstances.size() > 0) {
        auto [static_obj, static_mat_ids] = SceneDesc::mergeScene(static_desc, 0);
        static_obj.name = "static_opaque_merged";

        new_desc.objects.emplace_back(move(static_obj));
        new_desc.defaultInstances.push_back({
            "static_opaque_merged",
            uint32_t(new_desc.objects.size() - 1),
            move(static_mat_ids),
            glm::vec3(0.f),
            glm::quat(1.f, 0.f, 0.f, 0.f),
            glm::vec3(1.f),
            false,
            false,
        });
    }

    if (static_transparent_desc.defaultInstances.size() > 0) {
        auto [static_transparent_obj, static_transparent_mat_ids] =
            SceneDesc::mergeScene(static_transparent_desc, 0);
        static_transparent_obj.name = "static_transparent_merged";

        new_desc.objects.emplace_back(move(static_transparent_obj));
        new_desc.defaultInstances.push_back({
            "static_transparent_merged",
            uint32_t(new_desc.objects.size() - 1),
            move(static_transparent_mat_ids),
            glm::vec3(0.f),
            glm::quat(1.f, 0.f, 0.f, 0.f),
            glm::vec3(1.f),
            false,
            true,
        });
    }

    new_desc.materials = orig_desc.materials;
    new_desc.defaultLights = orig_desc.defaultLights;

    return new_desc;
}

static vector<LightProperties> processLights(
    const vector<LightProperties> &initial_lights,
    ProcessedGeometry<PackedVertex> &geo,
    vector<InstanceProperties> &instances,
    vector<Material> &materials,
    const AABB &scene_bbox,
    const filesystem::path &lights_path)
{
    vector<LightProperties> lights = initial_lights;

    materials.push_back(Material {
        "light_material",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        glm::vec3(0.f),
        0.f,
        glm::vec3(0.f),
        0.f,
        0.f,
        0.f,
        1.4f,
        0.f,
        0.f,
        glm::vec3(0.f),
        0.f,
        0.f,
        0.f,
        glm::vec3(100.f, 90.f, 90.f),
        false,
    });

    auto addLight = [&](glm::vec3 *verts, glm::vec3 translate) {
         geo.vertices.push_back(PackedVertex {
             verts[0] + translate,
             encodeNormalTangent(glm::vec3(0.f, -1.f, 0.f),
                                 glm::vec4(1.f, 0.f, 0.f, 1.f)),
             glm::vec2(0.f),
         });
         geo.vertices.push_back(PackedVertex {
             verts[1] + translate,
             encodeNormalTangent(glm::vec3(0.f, -1.f, 0.f),
                                 glm::vec4(1.f, 0.f, 0.f, 1.f)),
             glm::vec2(0.f),
         });
         geo.vertices.push_back(PackedVertex {
             verts[2] + translate,
             encodeNormalTangent(glm::vec3(0.f, -1.f, 0.f),
                                 glm::vec4(1.f, 0.f, 0.f, 1.f)),
             glm::vec2(0.f),
         });
         geo.vertices.push_back(PackedVertex {
             verts[3] + translate,
             encodeNormalTangent(glm::vec3(0.f, -1.f, 0.f),
                                 glm::vec4(1.f, 0.f, 0.f, 1.f)),
             glm::vec2(0.f),
         });

         uint32_t idx_offset = geo.indices.size();
         geo.indices.push_back(geo.vertices.size() - 3);
         geo.indices.push_back(geo.vertices.size() - 1);
         geo.indices.push_back(geo.vertices.size() - 2);

         geo.indices.push_back(geo.vertices.size() - 3);
         geo.indices.push_back(geo.vertices.size() - 2);
         geo.indices.push_back(geo.vertices.size() - 4);

         geo.meshInfos.push_back(MeshInfo {
             idx_offset,
             2,
             4,
         });

         string name = 
             string("light_") + to_string(lights.size());

         geo.objectInfos.push_back({
             uint32_t(geo.meshInfos.size() - 1),
             1,
         });
         geo.objectNames.push_back(name);

         instances.push_back(InstanceProperties {
             name,
             uint32_t(geo.objectInfos.size() - 1),
             { uint32_t(materials.size() - 1) },
             glm::vec3(0.f),
             glm::identity<glm::quat>(),
             glm::vec3(1.f),
             true,
             false,
         });

         LightProperties tri_light;
         tri_light.type = LightType::Triangle;
         tri_light.triIdxOffset = idx_offset;
         tri_light.triMatIdx = materials.size() - 1;

         lights.push_back(tri_light);

         tri_light.triIdxOffset = idx_offset + 3;
         lights.push_back(tri_light);
    };

    ifstream serialized_lights(lights_path, ios::binary);

    if (!serialized_lights.is_open()) return lights;

    {
        uint32_t num_lights;
        serialized_lights.read((char *)&num_lights, sizeof(uint32_t));

        for (int i = 0; i < (int)num_lights; i++) {
            glm::vec3 verts[4];
            serialized_lights.read((char *)&verts, sizeof(glm::vec3) * 4);
            glm::vec3 translate;
            serialized_lights.read((char *)&translate, sizeof(glm::vec3));
            addLight(verts, translate);
        }
    }

#if 0
    {
        glm::vec3 ceiling_min(INFINITY, INFINITY, INFINITY);
        glm::vec3 ceiling_max(-INFINITY, -INFINITY, -INFINITY);

        bool ceiling_found = false;
        for (const auto &inst : instances) {
            vector<int> ceiling_meshes;
            for (int mesh_idx = 0; mesh_idx < (int)inst.materials.size();
                 mesh_idx++) {
                int mat_idx = inst.materials[mesh_idx];
                if (materials[mat_idx].name == "ceiling") {
                    ceiling_meshes.push_back(mesh_idx);
                }
            }

            if (ceiling_meshes.empty()) continue;

            ceiling_found = true;

            const auto &object_info = geo.objectInfos[inst.objectIndex];
            for (int mesh_offset : ceiling_meshes) {
                int mesh_idx = object_info.meshIndex + mesh_offset;
                const auto &mesh = geo.meshInfos[mesh_idx];

                for (int i = 0; i < (int)(mesh.numTriangles * 3); i++) {
                    const PackedVertex vert =
                        geo.vertices[geo.indices[mesh.indexOffset + i]];

                    if (vert.position.x < ceiling_min.x) {
                        ceiling_min.x = vert.position.x;
                    }
                    if (vert.position.y < ceiling_min.y) {
                        ceiling_min.y = vert.position.y;
                    }
                    if (vert.position.z < ceiling_min.z) {
                        ceiling_min.z = vert.position.z;
                    }

                    if (vert.position.x > ceiling_min.x) {
                        ceiling_max.x = vert.position.x;
                    }
                    if (vert.position.y > ceiling_min.y) {
                        ceiling_max.y = vert.position.y;
                    }
                    if (vert.position.z > ceiling_min.z) {
                        ceiling_max.z = vert.position.z;
                    }
                }
            }
        }

        float light_height;
        if (ceiling_found) {
            light_height = ceiling_min.y - 0.05f;
        } else {
            light_height = scene_bbox.pMax.y - 1.5f;
        }

        const int num_init_lights = 5;
        float bbox_width = scene_bbox.pMax.x - scene_bbox.pMin.x;
        float bbox_depth = scene_bbox.pMax.z - scene_bbox.pMin.z;

        for (int i = 0; i < num_init_lights; i++) {
            for (int j = 0; j < num_init_lights; j++) {
                glm::vec3 light_position(
                    (bbox_width / num_init_lights) * i +
                        scene_bbox.pMin.x,
                    light_height,
                    (bbox_depth / num_init_lights) * j +
                        scene_bbox.pMin.z);

                addLight(light_position);
            }
        }
    }
#endif

#if 0
    for (const auto &inst : instances) {
        if (inst.transparent) {
            const auto &object_info = geo.objectInfos[inst.objectIndex];
            for (int mesh_offset = 0; mesh_offset < (int)object_info.numMeshes;
                 mesh_offset++) {
                int mesh_idx = object_info.meshIndex + mesh_offset;
                const auto &mesh = geo.meshInfos[mesh_idx];
                glm::vec3 bbox_min(INFINITY, INFINITY, INFINITY);
                glm::vec3 bbox_max(-INFINITY, -INFINITY, -INFINITY);
                for (int i = 0; i < (int)(mesh.numTriangles * 3); i++) {
                    const PackedVertex vert =
                        geo.vertices[geo.indices[mesh.indexOffset + i]];

                    const glm::vec3 &p = vert.position;
                    if (p.x < bbox_min.x) {
                        bbox_min.x = p.x;
                    }
                    if (p.y < bbox_min.y) {
                        bbox_min.y = p.y;
                    }

                    if (p.z < bbox_min.z) {
                        bbox_min.z = p.z;
                    }

                    if (p.x > bbox_max.x) {
                        bbox_max.x = p.x;
                    }

                    if (p.y > bbox_max.y) {
                        bbox_max.y = p.y;
                    }

                    if (p.z > bbox_max.z) {
                        bbox_max.z = p.z;
                    }
                }

                glm::vec3 delta = bbox_max - bbox_min;
                float min_comp = delta.x;
                int min_idx = 0;
                if (delta.y < min_comp) {
                    min_comp = delta.y;
                    min_idx = 1;
                }
                if (delta.z < min_comp) {
                    min_comp = delta.z;
                    min_idx = 2;
                }

                int left_idx;
                int up_idx;
                if (min_idx == 0) {
                    left_idx = 1;
                    up_idx = 2;
                } else if (min_idx == 1) {
                    left_idx = 0;
                    up_idx = 2;
                } else {
                    left_idx = 0;
                    up_idx = 1;
                }

                LightProperties portal;
                portal.type = LightType::Portal;
                portal.corners[0][0] = bbox_min.x;
                portal.corners[0][1] = bbox_min.y;
                portal.corners[0][2] = bbox_min.z;

                portal.corners[1][0] = bbox_min.x;
                portal.corners[1][1] = bbox_min.y;
                portal.corners[1][2] = bbox_min.z;

                portal.corners[1][up_idx] += delta[up_idx];

                portal.corners[2][0] = bbox_min.x;
                portal.corners[2][1] = bbox_min.y;
                portal.corners[2][2] = bbox_min.z;

                portal.corners[2][left_idx] += delta[left_idx];
                portal.corners[2][up_idx] += delta[up_idx];

                portal.corners[3][0] = bbox_min.x;
                portal.corners[3][1] = bbox_min.y;
                portal.corners[3][2] = bbox_min.z;

                portal.corners[3][left_idx] += delta[left_idx];

                lights.push_back(portal);
            }
        }
    }
#endif

    return lights;
}

struct ProcessedScene {
    ProcessedGeometry<PackedVertex> geometry;
    vector<InstanceProperties> instances;
    AABB bbox;
};

template <typename VertexType, typename MaterialType>
static ProcessedScene
processScene(const SceneDescription<VertexType, MaterialType> &orig_desc)
{
    SceneDescription<VertexType, MaterialType> desc =
        mergeStaticInstances(orig_desc);
    
    vector<unordered_set<glm::vec3>> obj_scales(desc.objects.size());

    for (const auto &inst : desc.defaultInstances) {
        obj_scales[inst.objectIndex].emplace(inst.scale);
    }

    auto [geometry, obj_remap, removed_meshes] =
        processGeometry<VertexType, MaterialType>(desc.objects, obj_scales);

    vector<InstanceProperties> new_insts;
    for (const auto &inst : desc.defaultInstances) {
        if (obj_remap[inst.objectIndex].empty()) continue;

        auto iter = obj_remap[inst.objectIndex].find(inst.scale);
        string new_name = inst.name;
        if (inst.scale != glm::vec3(1.f)) {
            new_name += "_" + to_string(inst.scale.x) + "_" +
                to_string(inst.scale.y) + "_" + to_string(inst.scale.z);
        }

        InstanceProperties new_inst {
            move(new_name),
            iter->second,
            {},
            inst.position,
            inst.rotation,
            glm::vec3(1.f),
            inst.dynamic,
            inst.transparent,
        };

        const auto &obj_removed_meshes = removed_meshes[new_inst.objectIndex];
        uint32_t cur_removed_idx = 0;
        uint32_t cur_removed_mesh_idx =
            obj_removed_meshes.empty() ? inst.materials.size() : 
            obj_removed_meshes[cur_removed_idx];

        for (int inst_mat_idx = 0; inst_mat_idx < (int)inst.materials.size();
             inst_mat_idx++) {
            if (uint32_t(inst_mat_idx) == cur_removed_mesh_idx) {
                cur_removed_idx++;
                cur_removed_mesh_idx =
                    obj_removed_meshes.size() > cur_removed_idx ?
                        obj_removed_meshes[cur_removed_idx] :
                        inst.materials.size();
            } else {
                new_inst.materials.push_back(inst.materials[inst_mat_idx]);
            }
        }

        new_insts.emplace_back(new_inst);
    }

    AABB bbox {
        glm::vec3(INFINITY, INFINITY, INFINITY),
        glm::vec3(-INFINITY, -INFINITY, -INFINITY),
    };

    auto updateBounds = [&bbox](const glm::vec3 &point) {
        bbox.pMin = glm::min(bbox.pMin, point);
        bbox.pMax = glm::max(bbox.pMax, point);
    };

    for (int inst_idx = 0; inst_idx < (int)new_insts.size(); inst_idx++) {
        const InstanceProperties &inst = new_insts[inst_idx];

        const ObjectInfo &obj = geometry.objectInfos[inst.objectIndex];
        for (int mesh_offset = 0; mesh_offset < (int)obj.numMeshes;
             mesh_offset++) {
            uint32_t mesh_idx = obj.meshIndex + mesh_offset;
            const MeshInfo &mesh = geometry.meshInfos[mesh_idx];

            for (int tri_idx = 0; tri_idx < (int)mesh.numTriangles;
                 tri_idx++) {
                uint32_t base_idx = tri_idx * 3 + mesh.indexOffset;

                glm::u32vec3 tri_indices(geometry.indices[base_idx],
                                         geometry.indices[base_idx + 1],
                                         geometry.indices[base_idx + 2]);

                auto a = geometry.vertices[tri_indices.x].position;
                auto b = geometry.vertices[tri_indices.y].position;
                auto c = geometry.vertices[tri_indices.z].position;

                // FIXME remove redundant calculation
                glm::mat4 rot_mat = glm::mat4_cast(inst.rotation);
                glm::mat4 txfm = glm::translate(inst.position) *
                    rot_mat;

                a = txfm * glm::vec4(a, 1.f);
                b = txfm * glm::vec4(b, 1.f);
                c = txfm * glm::vec4(c, 1.f);

                updateBounds(a);
                updateBounds(b);
                updateBounds(c);
            }
        }
    }

    return {
        move(geometry),
        move(new_insts),
        bbox,
    };
}

static MaterialMetadata stageMaterials(const vector<Material> &materials,
                                       const string &texture_dir,
                                       const string &env_map)
{
    auto packNonlinearUnorm = [&](float v) {
        float s = sqrtf(v);

        return glm::packUnorm1x8(s);
    };

    auto packNonlinearUnormVec3 = [&](glm::vec3 v) {
        return glm::u8vec3(
            packNonlinearUnorm(v.x),
            packNonlinearUnorm(v.y),
            packNonlinearUnorm(v.z));
    };

    vector<string> base_textures;
    vector<string> mr_textures;
    vector<string> spec_textures;
    vector<string> normal_textures;
    vector<string> emittance_textures;
    vector<string> transmission_textures;
    vector<string> clearcoat_textures;
    vector<string> aniso_textures;
    unordered_map<string, size_t> tracker;

    auto mapTexName = [&](const string &orig_tex, vector<string> &tex_list) {
        string new_name = orig_tex.substr(0, orig_tex.rfind(".")) +
            string(".tex");
        auto [iter, inserted] =
            tracker.emplace(new_name, tex_list.size());

        if (inserted) {
            tex_list.emplace_back(new_name);
        }

        return iter->second;
    };

    vector<MaterialParams> mat_params;
    vector<MaterialTextures> tex_indices;
    for (const Material &material : materials) {
        uint16_t mat_flags = 0;

        uint32_t base_color_idx = -1;
        if (!material.baseColorTexture.empty()) {
            base_color_idx =
                mapTexName(material.baseColorTexture, base_textures);

            mat_flags |= uint16_t(MaterialFlags::HasBaseTexture);
        }

        uint32_t mr_idx = -1;
        if (!material.metallicRoughnessTexture.empty()) {
            mr_idx = mapTexName(material.metallicRoughnessTexture,
                                 mr_textures);

            mat_flags |= uint16_t(MaterialFlags::HasMRTexture);
        }

        uint32_t spec_idx = -1;
        if (!material.specularTexture.empty()) {
            spec_idx = mapTexName(material.specularTexture,
                                   spec_textures);

            mat_flags |= uint16_t(MaterialFlags::HasSpecularTexture);
        }

        uint32_t normal_idx = -1;
        if (!material.normalMapTexture.empty()) {
            normal_idx = mapTexName(material.normalMapTexture,
                                     normal_textures);
            mat_flags |= uint16_t(MaterialFlags::HasNormalMap);
        }

        uint32_t emittance_idx = -1;
        if (!material.emittanceTexture.empty()) {
            emittance_idx = mapTexName(material.emittanceTexture,
                                        emittance_textures);
            mat_flags |= uint16_t(MaterialFlags::HasEmittanceTexture);
        }

        uint32_t transmission_idx = -1;
        if (!material.transmissionTexture.empty()) {
            transmission_idx = mapTexName(material.transmissionTexture,
                                           transmission_textures);
            mat_flags |= uint16_t(MaterialFlags::HasTransmissionTexture);
        }

        uint32_t clearcoat_idx = -1;
        if (!material.clearcoatTexture.empty()) {
            clearcoat_idx = mapTexName(material.clearcoatTexture,
                                        clearcoat_textures);
            mat_flags |= uint16_t(MaterialFlags::HasClearcoatTexture);
        }

        uint32_t aniso_idx = -1;
        if (!material.anisoTexture.empty()) {
            aniso_idx = mapTexName(material.anisoTexture,
                                    aniso_textures);
            mat_flags |= uint16_t(MaterialFlags::HasAnisotropicTexture);
        }

        if (material.thinwalled) {
            mat_flags |= uint16_t(MaterialFlags::ThinWalled);
        }

        float offset_ior = material.ior - 1;
        uint8_t packed_ior = uint8_t(roundf(offset_ior * 170.f));

        if (material.clearcoat != 0.f ||
            material.attenuationColor != glm::vec3(0.f) ||
            material.attenuationDistance != 0.f ||
            material.anisoScale != 0.f ||
            material.baseEmittance != glm::vec3(0.f)) {
            mat_flags |= uint16_t(MaterialFlags::Complex);
        }

        mat_params.emplace_back(MaterialParams {
            packNonlinearUnormVec3(material.baseColor),
            glm::packUnorm1x8(material.baseTransmission),
            glm::packUnorm1x8(material.specularScale),
            packed_ior,
            glm::packUnorm1x8(material.baseMetallic),
            glm::packUnorm1x8(material.baseRoughness),
            glm::u16vec3(
                glm::packHalf1x16(material.baseSpecular.r),
                glm::packHalf1x16(material.baseSpecular.g),
                glm::packHalf1x16(material.baseSpecular.b)),
            mat_flags,
            glm::packUnorm1x8(material.clearcoat),
            glm::packUnorm1x8(material.clearcoatRoughness),
            packNonlinearUnormVec3(material.attenuationColor),
            glm::packUnorm1x8(material.anisoScale),
            glm::packUnorm1x8(material.anisoRotation),
            glm::packHalf1x16(material.attenuationDistance),
            glm::u16vec3(
                glm::packHalf1x16(material.baseEmittance.r),
                glm::packHalf1x16(material.baseEmittance.g),
                glm::packHalf1x16(material.baseEmittance.b)),
        });

        tex_indices.emplace_back(MaterialTextures {
            base_color_idx,
            mr_idx,
            spec_idx,
            normal_idx,
            emittance_idx,
            transmission_idx,
            clearcoat_idx,
            aniso_idx,
        });
    }

    return {
        {
            texture_dir,
            move(base_textures),
            move(mr_textures),
            move(spec_textures),
            move(normal_textures),
            move(emittance_textures),
            move(transmission_textures),
            move(clearcoat_textures),
            move(aniso_textures),
            env_map,
        },
        move(mat_params),
        move(tex_indices),
    };
}

static void dumpIDMap(string_view scene_path_base,
                      const ProcessedGeometry<PackedVertex> &geo,
                      const vector<InstanceProperties> &insts,
                      const vector<Material> &materials)
{
    ofstream out(string(scene_path_base) + "_ids.json");
    string_view tab = "    ";
    out << "{\n";
    out << tab << "\"instances\": {\n";
    for (int inst_idx = 0; inst_idx < (int)insts.size(); inst_idx++) {
        const auto &inst = insts[inst_idx];
        out << tab << tab << "\"" << inst.name << "\": " << inst_idx;
        if (inst_idx != (int)insts.size() - 1) {
            out << ",";
        }
        out << "\n";
    }

    out << tab << "},\n";

    out << tab << "\"objects\": {\n";
    for (int obj_idx = 0; obj_idx < (int)geo.objectNames.size(); obj_idx++) {
        const auto &name = geo.objectNames[obj_idx];
        out << tab << tab << "\"" << name << "\": " << obj_idx;
        if (obj_idx != (int)geo.objectNames.size() - 1) {
            out << ",";
        }
        out << "\n";
    }

    out << tab << "},\n";

    out << tab << "\"materials\": {\n";
    for (int mat_idx = 0; mat_idx < (int)materials.size(); mat_idx++) {
        out << tab << tab << "\"" << materials[mat_idx].name <<
            "\": " << mat_idx;
        if (mat_idx != (int)materials.size() - 1) {
            out << ",";
        }
        out << "\n";
    }

    out << tab << "}\n";

    out << "}";
}

void ScenePreprocessor::dump(string_view out_path_name)
{
    auto [processed_geometry, processed_instances, default_bbox] =
        processScene(scene_data_->desc);

    vector<Material> materials = scene_data_->desc.materials;

    auto lights_path =
        filesystem::path(out_path_name).replace_extension("lights");
    auto processed_lights = processLights(scene_data_->desc.defaultLights,
        processed_geometry, processed_instances, materials, default_bbox,
        lights_path);

    auto processed_physics_state =
        ProcessedPhysicsState::make(processed_geometry, !scene_data_->dumpSDFs);

    filesystem::path out_path(out_path_name);
    string basename = out_path;
    basename.resize(basename.rfind('.'));

    dumpIDMap(basename, processed_geometry, processed_instances, materials);

    ofstream out(out_path, ios::binary);
    auto write = [&](auto val) {
        out.write(reinterpret_cast<const char *>(&val), sizeof(decltype(val)));
    };

    // Pad to 256 (maximum uniform / storage buffer alignment requirement)
    auto write_pad = [&](size_t align_req = 256) {
        static char pad_buffer[64] = { 0 };
        size_t cur_bytes = out.tellp();
        size_t align = cur_bytes % align_req;
        if (align != 0) {
            out.write(pad_buffer, align_req - align);
        }
    };

    auto align_offset = [](size_t offset) {
        return (offset + 255) & ~255;
    };

    auto make_staging_header = [&](const auto &geometry,
                                   const MaterialMetadata &material_metadata) {

        constexpr uint64_t vertex_size =
            sizeof(typename decltype(geometry.vertices)::value_type);
        uint64_t vertex_bytes = vertex_size * geometry.vertices.size();
        uint64_t index_bytes = sizeof(uint32_t) * geometry.indices.size();

        StagingHeader hdr;
        hdr.numMeshes = geometry.meshInfos.size();
        hdr.numObjects = geometry.objectInfos.size();
        hdr.numVertices = geometry.vertices.size();
        hdr.numIndices = geometry.indices.size();

        hdr.numMaterials = material_metadata.materialParams.size();

        hdr.indexOffset = align_offset(vertex_bytes);
        hdr.meshOffset = align_offset(hdr.indexOffset + index_bytes);
        hdr.objectOffset = align_offset(hdr.meshOffset + sizeof(MeshInfo) *
                                        hdr.numMeshes);
        hdr.materialOffset = align_offset(hdr.objectOffset +
            sizeof(ObjectInfo) * hdr.numObjects);
        hdr.physicsOffset = align_offset(hdr.materialOffset +
            sizeof(MaterialParams) * hdr.numMaterials);

        hdr.totalBytes =
            hdr.physicsOffset + hdr.numObjects * sizeof(PhysicsObject);

        return hdr;
    };

    auto write_staging = [&](const auto &geometry,
                             const MaterialMetadata &materials,
                             const ProcessedPhysicsState &physics_state,
                             const StagingHeader &hdr) {
        write_pad(256);

        auto stage_beginning = out.tellp();
        // Write all vertices
        constexpr uint64_t vertex_size =
            sizeof(typename decltype(geometry.vertices)::value_type);
        out.write(reinterpret_cast<const char *>(geometry.vertices.data()),
                  vertex_size * geometry.vertices.size());

        write_pad(256);
        // Write all indices
        out.write(reinterpret_cast<const char *>(geometry.indices.data()),
                  geometry.indices.size() * sizeof(uint32_t));

        write_pad(256);
        out.write(reinterpret_cast<const char *>(geometry.meshInfos.data()),
                  geometry.meshInfos.size() * sizeof(MeshInfo));

        write_pad(256);
        out.write(reinterpret_cast<const char *>(geometry.objectInfos.data()),
                  geometry.objectInfos.size() * sizeof(ObjectInfo));

        write_pad(256);
        out.write(reinterpret_cast<const char *>(
                materials.materialParams.data()),
                materials.materialParams.size() *
                sizeof(MaterialParams));

        write_pad(256);
        out.write(reinterpret_cast<const char *>(physics_state.objects.data()),
                  sizeof(PhysicsObject) * physics_state.objects.size());

        assert(out.tellp() == int64_t(hdr.totalBytes + stage_beginning));
    };

    auto write_lights = [&](const auto &lights) {
        write(uint32_t(lights.size()));
        for (const auto &light : lights) {
            write(light);
        }
    };

    auto write_materials = [&](const MaterialMetadata &metadata) {
        filesystem::path root_dir = filesystem::path(out_path_name).parent_path();
        filesystem::path relative_path =
            filesystem::path(metadata.textureInfo.textureDir).
                lexically_relative(root_dir);

        string relative_path_str = relative_path.string() + "/";

        out.write(relative_path_str.data(),
                  relative_path_str.size());
        out.put(0);
        
        out.write(metadata.textureInfo.envMap.data(),
                  metadata.textureInfo.envMap.size());
        out.put(0);

        auto writeNameArray = [&](const auto &names) {
            write(uint32_t(names.size()));
            for (const auto &tex_name : names) {
                out.write(tex_name.data(), tex_name.size());
                out.put(0);
            }
        };

        writeNameArray(metadata.textureInfo.base);
        writeNameArray(metadata.textureInfo.metallicRoughness);
        writeNameArray(metadata.textureInfo.specular);
        writeNameArray(metadata.textureInfo.normal);
        writeNameArray(metadata.textureInfo.emittance);
        writeNameArray(metadata.textureInfo.transmission);
        writeNameArray(metadata.textureInfo.clearcoat);
        writeNameArray(metadata.textureInfo.anisotropic);

        out.write(reinterpret_cast<const char *>(
            metadata.materialTextures.data()),
            sizeof(MaterialTextures) * metadata.materialTextures.size());
    };

    auto write_instances = [&](const auto &instance_props,
                               const AABB &bbox) {
        uint32_t num_instances = instance_props.size();

        vector<PhysicsInstance> static_instances;
        vector<PhysicsInstance> dynamic_instances;
        vector<PhysicsTransform> dynamic_transforms;
        vector<InstanceTransform> default_transforms;

        vector<ObjectInstance> instances;
        vector<uint32_t> instance_materials;
        vector<InstanceFlags> instance_flags;

        for (int inst_id = 0; inst_id < (int)num_instances; inst_id++) {
            const InstanceProperties &inst_props = instance_props[inst_id];

            instances.push_back({
                inst_props.objectIndex,
                uint32_t(instance_materials.size()),
            });

            InstanceFlags flags {};
            if (inst_props.transparent) {
                flags |= InstanceFlags::Transparent;
            }
            instance_flags.push_back(flags);

            for (uint32_t mat_idx : inst_props.materials) {
                instance_materials.push_back(mat_idx);
            }

            if (inst_props.dynamic) {
                dynamic_instances.push_back({
                    uint32_t(inst_id),
                    inst_props.objectIndex,
                });

                dynamic_transforms.push_back({
                    inst_props.position,
                    inst_props.rotation,
                });
            } else {
                static_instances.push_back({
                    uint32_t(inst_id),
                    inst_props.objectIndex,
                });
            }

            glm::mat4 rot_mat = glm::mat4_cast(inst_props.rotation);

            glm::mat4 txfm = glm::translate(inst_props.position) *  rot_mat;

            glm::mat4 inv = glm::transpose(rot_mat) * 
                glm::translate(-inst_props.position);

            glm::mat4x3 reduced(txfm);
            glm::mat4x3 inv_reduced(inv);

            default_transforms.push_back({
                reduced,
                inv_reduced,
            });
        }

        write(uint32_t(instance_materials.size()));
        out.write(reinterpret_cast<const char *>(instance_materials.data()),
                  sizeof(uint32_t) * instance_materials.size());

        out.write(reinterpret_cast<const char *>(&bbox),
                  sizeof(AABB));

        write(uint32_t(num_instances));
        out.write(reinterpret_cast<const char *>(instances.data()),
                  sizeof(ObjectInstance) * instances.size());

        // Write out default transforms
        for (const auto &txfm : default_transforms) {
            write(txfm);
        }

        out.write(reinterpret_cast<const char *>(instance_flags.data()),
                  sizeof(InstanceFlags) * instance_flags.size());

        write(uint32_t(static_instances.size()));
        write(uint32_t(dynamic_instances.size()));

        // Physics instance information
        out.write(reinterpret_cast<const char *>(static_instances.data()),
                  sizeof(PhysicsInstance) * static_instances.size());

        out.write(reinterpret_cast<const char *>(dynamic_instances.data()),
            sizeof(PhysicsInstance) * dynamic_instances.size());

        out.write(reinterpret_cast<const char *>(dynamic_transforms.data()),
            sizeof(PhysicsTransform) * dynamic_transforms.size());
    };

    auto write_sdfs = [&](const auto &physics_state,
                          const auto &data_dir,
                          bool dump_sdfs) {
        write(uint32_t(physics_state.sdfs.size()));
        for (int i = 0; i < (int)physics_state.sdfs.size(); i++) {
            const auto &sdf = physics_state.sdfs[i];
            string sdf_path = data_dir + "/sdf_" + to_string(i) + ".bin";

            if (dump_sdfs) {
                sdf.dump(sdf_path);
            }

            out.write(sdf_path.data(), sdf_path.size());
            out.put(0);
        }
    };

    auto write_objects = [&](const auto &geometry) {
        // Write mesh infos
        out.write(reinterpret_cast<const char *>(geometry.meshInfos.data()),
                  geometry.meshInfos.size() * sizeof(MeshInfo));

        // Write object infos
        out.write(reinterpret_cast<const char *>(geometry.objectInfos.data()),
                  geometry.objectInfos.size() * sizeof(ObjectInfo));

    };

    auto write_scene = [&](const auto &geometry,
                           const auto &instances,
                           const AABB &bbox,
                           const auto &lights,
                           const auto &materials,
                           const auto &env_map,
                           const auto &data_dir) {
        auto material_metadata =
            stageMaterials(materials, data_dir, env_map);

        StagingHeader hdr = make_staging_header(geometry, material_metadata);
        write(hdr);
        write_pad();

        write_objects(geometry);

        write_lights(lights);

        write_materials(material_metadata);

        write_instances(instances, bbox);

        write_sdfs(processed_physics_state, data_dir, scene_data_->dumpSDFs);

        write_staging(geometry, material_metadata, processed_physics_state,
                      hdr);
    };

    // Header: magic
    write(uint32_t(0x55555555));
    write_scene(processed_geometry, processed_instances, default_bbox,
                processed_lights, materials, scene_data_->desc.envMap,
                scene_data_->dataDir);
    out.close();
}

template struct HandleDeleter<PreprocessData>;

}
