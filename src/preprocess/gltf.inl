#include "gltf.hpp"
#include "import.hpp"

#include <rlpbr_core/utils.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include <fstream>
#include <iostream>
#include <type_traits>
#include <unordered_set>

using namespace std;

namespace RLpbr {
namespace SceneImport {

struct GLBHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t length;
};

struct ChunkHeader {
    uint32_t chunkLength;
    uint32_t chunkType;
};

template <typename T>
T jsonReadVec(const simdjson::dom::array &arr)
{
    T v;
    float *data = glm::value_ptr(v);
    int offset = 0;
    for (double comp : arr) {
        data[offset] = comp;
        offset++;

        if (offset >= T::length()) break;
    }

    return v;
}
    
template <typename T, typename U>
T jsonGetOr(const simdjson::simdjson_result<U> &e, T default_val)
{
    using ReadType = conditional_t<
        is_same_v<T, float>, double, conditional_t<
        is_same_v<T, uint32_t>, uint64_t, conditional_t<
        is_same_v<T, glm::vec3>, simdjson::dom::array, conditional_t<
        is_same_v<T, glm::vec4>, simdjson::dom::array, void
    >>>>;

    static_assert(!is_same_v<ReadType, void>);

    ReadType tmp;
    auto err = e.get(tmp);

    if (!err) {
        if constexpr (is_same_v<ReadType, simdjson::dom::array>) {
            return jsonReadVec<T>(tmp);
        } else {
            return T(tmp);
        }
    } else {
        return default_val;
    }
}

GLTFScene gltfLoad(filesystem::path gltf_path) noexcept
{
    GLTFScene scene;
    scene.sceneName = gltf_path.stem();
    scene.sceneDirectory = gltf_path.parent_path();

    auto suffix = gltf_path.extension();
    bool binary = suffix == ".glb";
    if (binary) {
        ifstream binary_file(string(gltf_path),
                                  ios::in | ios::binary);

        GLBHeader glb_header;
        binary_file.read(reinterpret_cast<char *>(&glb_header),
                         sizeof(GLBHeader));

        uint32_t total_length = glb_header.length;

        ChunkHeader json_header;
        binary_file.read(reinterpret_cast<char *>(&json_header),
                         sizeof(ChunkHeader));

        vector<uint8_t> json_buffer(json_header.chunkLength +
                                         simdjson::SIMDJSON_PADDING);

        binary_file.read(reinterpret_cast<char *>(json_buffer.data()),
                         json_header.chunkLength);

        try {
            scene.root = scene.jsonParser.parse(
                json_buffer.data(), json_header.chunkLength, false);
        } catch (const simdjson::simdjson_error &e) {
            cerr << "GLTF loading '" << gltf_path
                      << "' failed: " << e.what() << endl;
            abort();
        }

        if (json_header.chunkLength < total_length) {
            ChunkHeader bin_header;
            binary_file.read(reinterpret_cast<char *>(&bin_header),
                             sizeof(ChunkHeader));

            assert(bin_header.chunkType == 0x004E4942);

            scene.internalData.resize(bin_header.chunkLength);

            binary_file.read(
                reinterpret_cast<char *>(scene.internalData.data()),
                bin_header.chunkLength);
        }
    } else {
        scene.root = scene.jsonParser.load(string(gltf_path));
    }

    try {
        for (const auto &buffer : scene.root["buffers"]) {
            string_view uri {};
            const uint8_t *data_ptr = nullptr;

            auto uri_elem = buffer.at_key("uri");
            if (uri_elem.error() != simdjson::NO_SUCH_FIELD) {
                uri = uri_elem.get_string();
            } else {
                data_ptr = scene.internalData.data();
            }
            scene.buffers.push_back(GLTFBuffer {
                data_ptr,
                uri,
            });
        }

        cout << "Buffers" << endl;

        for (const auto &view : scene.root["bufferViews"]) {
            uint64_t stride_res;
            auto stride_error = view["byteStride"].get(stride_res);
            if (stride_error) {
                stride_res = 0;
            }
            scene.bufferViews.push_back(GLTFBufferView {
                static_cast<uint32_t>(view["buffer"].get_uint64()),
                static_cast<uint32_t>(view["byteOffset"].get_uint64()),
                static_cast<uint32_t>(stride_res),
                static_cast<uint32_t>(view["byteLength"].get_uint64()),
            });
        }

        cout << "bufferViews" << endl;

        for (const auto &accessor : scene.root["accessors"]) {
            GLTFComponentType type;
            uint64_t component_type = accessor["componentType"];
            if (component_type == 5126) {
                type = GLTFComponentType::FLOAT;
            } else if (component_type == 5125) {
                type = GLTFComponentType::UINT32;
            } else if (component_type == 5123) {
                type = GLTFComponentType::UINT16;
            } else {
                cerr << "GLTF loading '" << gltf_path
                          << "' failed: unknown component type" << endl;
                abort();
            }

            uint64_t byte_offset;
            auto offset_error = accessor["byteOffset"].get(byte_offset);
            if (offset_error) {
                byte_offset = 0;
            }

            scene.accessors.push_back(GLTFAccessor {
                static_cast<uint32_t>(accessor["bufferView"].get_uint64()),
                static_cast<uint32_t>(byte_offset),
                static_cast<uint32_t>(accessor["count"].get_uint64()),
                type,
            });
        }

        cout << "accessors" << endl;

        auto images_elem = scene.root.at_key("images");
        if (images_elem.error() != simdjson::NO_SUCH_FIELD) {
            for (const auto &json_image : images_elem.get_array()) {
                GLTFImage img {};
                string_view uri {};
                auto uri_err = json_image["uri"].get(uri);
                if (!uri_err) {
                    img.type = GLTFImageType::EXTERNAL;
                    img.filePath = uri;
                } else {
                    uint64_t view_idx = json_image["bufferView"];
                    string_view mime = json_image["mimeType"];
                    if (mime == "image/jpeg") {
                        img.type = GLTFImageType::JPEG;
                    } else if (mime == "image/png") {
                        img.type = GLTFImageType::PNG;
                    } else if (mime == "image/x-basis") {
                        img.type = GLTFImageType::BASIS;
                    } else {
                        cerr << "Unsupported mime type: " << mime << endl;
                        img.type = GLTFImageType::JPEG;
                        //abort();
                    }

                    img.viewIdx = view_idx;
                }

                scene.images.push_back(img);
            }
        }

        cout << "images" << endl;

        auto textures_elem = scene.root.at_key("textures");
        if (textures_elem.error() != simdjson::NO_SUCH_FIELD) {
            for (const auto &texture : textures_elem.get_array()) {
                uint64_t source_idx;
                auto src_err = texture["source"].get(source_idx);
                if (src_err) {
                    auto ext_err =
                        texture["extensions"]["GOOGLE_texture_basis"]["source"]
                            .get(source_idx);
                    if (ext_err) {
                        cerr << "GLTF loading '" << gltf_path
                                  << "' failed: texture without source"
                                  << endl;
                        abort();
                    }
                }

                uint64_t sampler_idx;
                auto sampler_error = texture["sampler"].get(sampler_idx);
                if (sampler_error) {
                    sampler_idx = 0;
                }

                scene.textures.push_back(GLTFTexture {
                    static_cast<uint32_t>(source_idx),
                    static_cast<uint32_t>(sampler_idx),
                });
            }
        }

        cout << "textures" << endl;

        for (const auto &material : scene.root["materials"]) {
            uint32_t tex_missing = scene.textures.size();
            const auto &exts = material["extensions"];

            const auto &pbr = material["pbrMetallicRoughness"];
            uint32_t base_color_idx = jsonGetOr(
                pbr["baseColorTexture"]["index"], tex_missing);

            uint32_t metallic_roughness_idx = jsonGetOr(
                pbr["metallicRoughnessTexture"]["index"], tex_missing);

            uint32_t bc_coord =
                jsonGetOr(pbr["baseColorTexture"]["texCoord"], 0u);


            uint32_t mr_coord = 
                jsonGetOr(pbr["metallicRoughnessTexture"]["texCoord"], 0u);

            if (bc_coord != 0 || mr_coord != 0) {
                cerr << "Multiple UVs not supported" << endl;
                abort();
            }

            glm::vec4 base_color =
                jsonGetOr(pbr["baseColorFactor"], glm::vec4(1.f));
            simdjson::dom::array base_color_json;

            float metallic = jsonGetOr(pbr["metallicFactor"], 1.f);

            float roughness = jsonGetOr(pbr["roughnessFactor"], 1.f);

            auto transmission_ext =
                exts["KHR_materials_transmission"];

            uint32_t transmission_idx = jsonGetOr(
                transmission_ext["transmissionTexture"]["index"], tex_missing);

            float transmission_factor =
                jsonGetOr(transmission_ext["transmissionFactor"], 0.f);

            auto specular_ext = exts["KHR_materials_specular"];

            glm::vec3 base_specular = 
                jsonGetOr(specular_ext["specularColorFactor"], glm::vec3(1.f));

            float specular_factor =
                jsonGetOr(specular_ext["specularFactor"], 1.f);

            uint32_t spec_idx = jsonGetOr(
                specular_ext["specularTexture"]["index"], tex_missing);

            uint32_t spec_color_idx = jsonGetOr(
                specular_ext["specularColorTexture"]["index"], tex_missing);

            if (spec_idx != spec_color_idx) {
                cerr << "Specular textures must be packed together" << endl;
                abort();
            }

            float ior = jsonGetOr(
                exts["KHR_materials_ior"]["ior"], 1.5f);

            auto clearcoat_ext = exts["KHR_materials_clearcoat"];

            float clearcoat = jsonGetOr(
                clearcoat_ext["clearcoatFactor"], 0.f);

            float clearcoat_roughness = jsonGetOr(
                clearcoat_ext["clearcoatRoughnessFactor"], 0.f);

            uint32_t clearcoat_idx = jsonGetOr(
                clearcoat_ext["clearcoatTexture"]["index"], tex_missing);

            uint32_t clearcoat_roughness_idx = jsonGetOr(
                clearcoat_ext["clearcoatRoughnessTexture"]["index"],
                tex_missing);

            uint32_t clearcoat_normal_idx = jsonGetOr(
                clearcoat_ext["clearcoatNormalTexture"]["index"], tex_missing);

            if (clearcoat_idx != clearcoat_roughness_idx) {
                cerr << "Clearcoat textures must be packed together" << endl;
                abort();
            }

            auto volume_ext = exts["KHR_materials_volume"];

            float thickness = jsonGetOr(
                volume_ext["thicknessFactor"], 0.f);
            bool thinwalled = thickness == 0.f;

            float attenuation_distance = jsonGetOr(
                volume_ext["attenuationDistance"], INFINITY);
            glm::vec3 attenuation_color = jsonGetOr(
                volume_ext["attenuationColor"], glm::vec3(1.f));

            auto aniso_ext = exts["KHR_materials_anisotropy"];

            float aniso_scale = jsonGetOr(
                aniso_ext["anisotropy"], 0.f);
            glm::vec3 aniso_dir = jsonGetOr(
                aniso_ext["anisotropyDirection"], glm::vec3(1.f, 0.f, 0.f));

            uint32_t aniso_idx = jsonGetOr(
                aniso_ext["anisotropyTexture"], tex_missing);

            uint32_t aniso_rot_idx = jsonGetOr(
                aniso_ext["anisotropyDirectionTexture"], tex_missing);

            if (aniso_idx != aniso_rot_idx) {
                cerr << "Anisotropy textures must be packed together" << endl;
                abort();
            }

            uint32_t normal_idx = jsonGetOr(
                material["normalTexture"]["index"], tex_missing);

            glm::vec3 base_emittance = jsonGetOr(
                material["emissiveFactor"], glm::vec3(0.f));

            uint32_t emissive_idx = jsonGetOr(
                material["emissiveTexture"]["index"], tex_missing);

            string_view material_name_view;
            string material_name;
            auto name_err = material["name"].get(material_name_view);
            if (name_err) {
                material_name = to_string(scene.materials.size());
            } else {
                material_name = material_name_view;
            }

            scene.materials.push_back(GLTFMaterial {
                move(material_name),
                base_color_idx,
                metallic_roughness_idx,
                spec_idx,
                normal_idx,
                emissive_idx,
                transmission_idx,
                clearcoat_idx,
                clearcoat_normal_idx,
                aniso_idx,
                base_color,
                transmission_factor,
                base_specular,
                specular_factor,
                metallic,
                roughness,
                ior,
                clearcoat,
                clearcoat_roughness,
                attenuation_color,
                attenuation_distance,
                aniso_scale,
                aniso_dir,
                base_emittance,
                thinwalled,
            });
        }

        cout << "materials" << endl;

        for (const auto &mesh : scene.root["meshes"]) {
            simdjson::dom::array gltf_prims = mesh["primitives"];
            vector<GLTFPrimitive> prims;

            for (const simdjson::dom::element &prim : gltf_prims) {
                simdjson::dom::element attrs = prim["attributes"];

                optional<uint32_t> position_idx;
                optional<uint32_t> normal_idx;
                optional<uint32_t> uv_idx;
                optional<uint32_t> color_idx;

                uint64_t position_res;
                auto position_error = attrs["POSITION"].get(position_res);
                if (!position_error) {
                    position_idx = position_res;
                }

                uint64_t normal_res;
                auto normal_error = attrs["NORMAL"].get(normal_res);
                if (!normal_error) {
                    normal_idx = normal_res;
                }

                uint64_t uv_res;
                auto uv_error = attrs["TEXCOORD_0"].get(uv_res);
                if (!uv_error) {
                    uv_idx = uv_res;
                }

                uint64_t color_res;
                auto color_error = attrs["COLOR_0"].get(color_res);
                if (!color_error) {
                    color_idx = color_res;
                }
                
                uint64_t material_idx;
                auto mat_error = prim["material"].get(material_idx);
                if (mat_error) {
                    material_idx = 0;
                }

                prims.push_back({
                    position_idx,
                    normal_idx,
                    uv_idx,
                    color_idx,
                    static_cast<uint32_t>(prim["indices"].get_uint64()),
                    static_cast<uint32_t>(material_idx),
                });
            }

            string_view mesh_name_view;
            string mesh_name;
            auto name_err = mesh["name"].get(mesh_name_view);
            if (name_err) {
                mesh_name = to_string(scene.meshes.size());
            } else {
                mesh_name = mesh_name_view;
            }

            scene.meshes.push_back(GLTFMesh {
                move(mesh_name),
                move(prims),
            });
        }

        cout << "meshes" << endl;

        for (const auto &node : scene.root["nodes"]) {
            vector<uint32_t> children;
            simdjson::dom::array json_children;
            auto children_error = node["children"].get(json_children);

            if (!children_error) {
                for (uint64_t child : json_children) {
                    children.push_back(child);
                }
            }

            uint64_t mesh_idx;
            auto mesh_error = node["mesh"].get(mesh_idx);
            if (mesh_error) {
                mesh_idx = scene.meshes.size();
            }

            glm::mat4 txfm(1.f);

            simdjson::dom::array matrix;
            auto matrix_error = node["matrix"].get(matrix);
            if (!matrix_error) {
                float *txfm_data = glm::value_ptr(txfm);
                for (double mat_elem : matrix) {
                    *txfm_data = mat_elem;
                    txfm_data++;
                }
            } else {
                glm::mat4 translation(1.f);
                simdjson::dom::array translate_raw;
                auto translate_error = node["translation"].get(translate_raw);
                if (!translate_error) {
                    glm::vec3 translate_vec;
                    float *translate_ptr = glm::value_ptr(translate_vec);
                    for (double vec_elem : translate_raw) {
                        *translate_ptr = vec_elem;
                        translate_ptr++;
                    }
                    translation = glm::translate(translate_vec);
                }

                glm::mat4 rotation(1.f);
                simdjson::dom::array quat_raw;
                auto quat_error = node["rotation"].get(quat_raw);
                if (!quat_error) {
                    glm::quat quat_vec;
                    float *quat_ptr = glm::value_ptr(quat_vec);
                    for (double vec_elem : quat_raw) {
                        *quat_ptr = vec_elem;
                        quat_ptr++;
                    }
                    rotation = glm::mat4_cast(quat_vec);
                }

                glm::mat4 scale(1.f);
                simdjson::dom::array scale_raw;
                auto scale_error = node["scale"].get(scale_raw);
                if (!scale_error) {
                    glm::vec3 scale_vec;
                    float *scale_ptr = glm::value_ptr(scale_vec);
                    for (double vec_elem : scale_raw) {
                        *scale_ptr = vec_elem;
                        scale_ptr++;
                    }
                    scale = glm::scale(scale_vec);
                }

                txfm = translation * rotation * scale;
            }

            scene.nodes.push_back(GLTFNode {
                move(children), static_cast<uint32_t>(mesh_idx), txfm});
        }

        cout << "nodes" << endl;

        simdjson::dom::array scenes = scene.root["scenes"];
        if (scenes.size() > 1) {
            cerr << "GLTF loading '" << gltf_path
                      << "' failed: Multiscene files not supported"
                      << endl;
            abort();
        }

        for (uint64_t node_idx : scenes.at(0)["nodes"]) {
            scene.rootNodes.push_back(node_idx);
        }

    } catch (const simdjson::simdjson_error &e) {
        cerr << "GLTF loading '" << gltf_path << "' failed: " << e.what()
                  << endl;
        abort();
    }

    return scene;
}

template <typename T>
static StridedSpan<T> getGLTFBufferView(const GLTFScene &scene,
                                        uint32_t view_idx,
                                        uint32_t start_offset = 0,
                                        uint32_t num_elems = 0)
{
    const GLTFBufferView &view = scene.bufferViews[view_idx];
    const GLTFBuffer &buffer = scene.buffers[view.bufferIdx];

    if (buffer.dataPtr == nullptr) {
        cerr << "GLTF loading failed: external references not supported"
             << endl;
    }

    size_t total_offset = start_offset + view.offset;
    const uint8_t *start_ptr = buffer.dataPtr + total_offset;
    ;

    uint32_t stride = view.stride;
    if (stride == 0) {
        stride = sizeof(T);
    }

    if (num_elems == 0) {
        num_elems = view.numBytes / stride;
    }

    return StridedSpan<T>(start_ptr, num_elems, stride);
}

template <typename T>
static StridedSpan<T> getGLTFAccessorView(const GLTFScene &scene,
                                          uint32_t accessor_idx)
{
    const GLTFAccessor &accessor = scene.accessors[accessor_idx];

    return getGLTFBufferView<T>(scene, accessor.viewIdx, accessor.offset,
                                accessor.numElems);
}

static void dumpGLTFTexture(const GLTFScene &scene, const GLTFImage &img,
                            string_view texture_dir, string_view texture_name)
{
    const GLTFBufferView &buffer_view = scene.bufferViews[img.viewIdx];
    if (buffer_view.stride > 1) {
        cerr << "GLTF import: cannot dump strided texture" << endl;
        abort();
    }

    ofstream tex_dump(filesystem::path(texture_dir) / texture_name);

    const uint8_t *tex_ptr = scene.internalData.data() + buffer_view.offset;

    tex_dump.write(reinterpret_cast<const char *>(tex_ptr), buffer_view.numBytes);
}

template <typename MaterialType>
vector<MaterialType> gltfParseMaterials(const GLTFScene &scene,
    optional<string_view> texture_dir)
{
    vector<MaterialType> materials;
    materials.reserve(scene.materials.size());

    unordered_set<uint32_t> internal_tracker;

    auto extractTex = [&](uint32_t tex_idx) {
        string tex_name = "";
        if (tex_idx < scene.textures.size()) {
            const GLTFImage &img =
                scene.images[scene.textures[tex_idx].sourceIdx];

            if (img.type == GLTFImageType::EXTERNAL) {
                tex_name = img.filePath;
            } else {
                const char *ext;
                if (img.type == GLTFImageType::JPEG) {
                    ext = ".jpg";
                } else if (img.type == GLTFImageType::PNG) {
                    ext = ".png";
                } else {
                    cerr << "GLTF: Unsupported internal image type"
                         << endl;
                    abort();
                }

                tex_name = scene.sceneName + "_" + to_string(tex_idx) + ext;

                if (texture_dir.has_value()) {
                    auto inserted = internal_tracker.emplace(tex_idx);
                    if (inserted.second) {
                        dumpGLTFTexture(scene, img, texture_dir.value(), tex_name);
                    }
                }
            }
        }

        return tex_name;
    };

    for (const auto &gltf_mat : scene.materials) {
        float alpha_transparency = gltf_mat.baseColor.a;

        // Hack: use alpha as proper transparency for assets with
        // hardcoded alpha in the baseColorFactor
        float transmission = gltf_mat.transmissionFactor;
        if (transmission == 0.f) {
            transmission = 1.f - alpha_transparency;
        }

        glm::vec3 base_color(gltf_mat.baseColor.r,
                             gltf_mat.baseColor.g,
                             gltf_mat.baseColor.b);

        // FIXME
        float aniso_rotation = atan2(gltf_mat.anisoDir.y, gltf_mat.anisoDir.x);

        materials.push_back({
            gltf_mat.name,
            extractTex(gltf_mat.baseColorIdx),
            extractTex(gltf_mat.metallicRoughnessIdx),
            extractTex(gltf_mat.specularIdx),
            extractTex(gltf_mat.normalIdx),
            extractTex(gltf_mat.emittanceIdx),
            extractTex(gltf_mat.transmissionIdx),
            extractTex(gltf_mat.clearcoatIdx),
            extractTex(gltf_mat.anisoIdx),
            base_color,
            transmission,
            gltf_mat.baseSpecular,
            gltf_mat.specularFactor,
            gltf_mat.metallic,
            gltf_mat.roughness,
            gltf_mat.ior,
            gltf_mat.clearcoat,
            gltf_mat.clearcoatRoughness,
            gltf_mat.attenuationColor,
            gltf_mat.attenuationDistance,
            gltf_mat.anisoScale,
            aniso_rotation,
            gltf_mat.baseEmittance,
            gltf_mat.thinwalled,
        });
    }

    return materials;
}

template <typename T, typename = int>
struct HasPosition : std::false_type { };
template <typename T>
struct HasPosition <T, decltype((void) T::position, 0)> : std::true_type { };

template <typename T, typename = int>
struct HasNormal : std::false_type { };
template <typename T>
struct HasNormal <T, decltype((void) T::normal, 0)> : std::true_type { };

template <typename T, typename = int>
struct HasUV : std::false_type { };
template <typename T>
struct HasUV <T, decltype((void) T::uv, 0)> : std::true_type { };

template <typename T, typename = int>
struct HasColor : std::false_type { };
template <typename T>
struct HasColor <T, decltype((void) T::color, 0)> : std::true_type { };

template <typename VertexType>
static vector<Mesh<VertexType>> gltfParseMesh(
    const GLTFScene &scene,
    uint32_t mesh_idx)
{
    const GLTFMesh &gltf_mesh = scene.meshes[mesh_idx];

    vector<Mesh<VertexType>> meshes;

    for (const GLTFPrimitive &prim : gltf_mesh.primitives) {
        vector<VertexType> vertices;
        vector<uint32_t> indices;

        optional<StridedSpan<const glm::vec3>> position_accessor;
        optional<StridedSpan<const glm::vec3>> normal_accessor;
        optional<StridedSpan<const glm::vec2>> uv_accessor;
        optional<StridedSpan<const glm::u8vec3>> color_accessor;

        constexpr bool has_position = HasPosition<VertexType>::value;
        constexpr bool has_normal = HasNormal<VertexType>::value;
        constexpr bool has_uv = HasUV<VertexType>::value;
        constexpr bool has_color = HasColor<VertexType>::value;

        if constexpr (has_position) {
            position_accessor = getGLTFAccessorView<const glm::vec3>(
                scene, prim.positionIdx.value());
        }

        if constexpr (has_normal) {
            if (prim.normalIdx.has_value()) {
                normal_accessor = getGLTFAccessorView<const glm::vec3>(
                    scene, prim.normalIdx.value());
            }
        }

        if constexpr (has_uv) {
            if (prim.uvIdx.has_value()) {
                uv_accessor = getGLTFAccessorView<const glm::vec2>(scene,
                    prim.uvIdx.value());
            }
        }

        if constexpr (has_color) {
            if (prim.colorIdx.has_value()) {
                color_accessor = getGLTFAccessorView<const glm::u8vec3>(
                    scene, prim.colorIdx.value());
            }
        }

        uint32_t max_idx = 0;

        auto index_type = scene.accessors[prim.indicesIdx].type;

        if (index_type == GLTFComponentType::UINT32) {
            auto idx_accessor =
                getGLTFAccessorView<const uint32_t>(scene, prim.indicesIdx);
            indices.reserve(idx_accessor.size());

            for (uint32_t idx : idx_accessor) {
                if (idx > max_idx) {
                    max_idx = idx;
                }

                indices.push_back(idx);
            }
        } else if (index_type == GLTFComponentType::UINT16) {
            auto idx_accessor =
                getGLTFAccessorView<const uint16_t>(scene, prim.indicesIdx);
            indices.reserve(idx_accessor.size());

            for (uint16_t idx : idx_accessor) {
                if (idx > max_idx) {
                    max_idx = idx;
                }

                indices.push_back(idx);
            }
        } else {
            cerr << "GLTF loading failed: unsupported index type"
                      << endl;
            abort();
        }

        max_idx = min(uint32_t(position_accessor->size()), max_idx);

        vertices.reserve(max_idx + 1);
        for (uint32_t vert_idx = 0; vert_idx <= max_idx; vert_idx++) {
            VertexType vert {};

            if constexpr (has_position) {
                vert.position = (*position_accessor)[vert_idx];
                if (isnan(vert.position.x) || isinf(vert.position.x)) {
                    vert.position.x = 0;
                }

                if (isnan(vert.position.y) || isinf(vert.position.y)) {
                    vert.position.y = 0;
                }

                if (isnan(vert.position.z) || isinf(vert.position.z)) {
                    vert.position.z = 0;
                }
            }

            if constexpr (has_normal) {
                if (normal_accessor.has_value()) {
                    vert.normal = (*normal_accessor)[vert_idx];
                }

                if (isnan(vert.normal.x) || isinf(vert.normal.x)) {
                    vert.normal.x = 0;
                }

                if (isnan(vert.normal.y) || isinf(vert.normal.y)) {
                    vert.normal.y = 0;
                }

                if (isnan(vert.normal.z) || isinf(vert.normal.z)) {
                    vert.normal.z = 0;
                }
            }

            if constexpr (has_uv) {
                if (uv_accessor.has_value()) {
                    vert.uv = (*uv_accessor)[vert_idx];
                }

                if (isnan(vert.uv.x) || isinf(vert.uv.x)) {
                    vert.uv.x = 0;
                }

                if (isnan(vert.uv.y) || isinf(vert.uv.y)) {
                    vert.uv.y = 0;
                }
            }

            if constexpr (has_color) {
                if (color_accessor.has_value()) {
                    vert.color = glm::u8vec4((*color_accessor)[vert_idx], 255);
                }
            }

            vertices.push_back(vert);
        }

        Mesh<VertexType> mesh {
            move(vertices),
            move(indices),
        };

        meshes.emplace_back(move(mesh));
    }

    return meshes;
}

template <typename MaterialType>
static std::vector<InstanceProperties> gltfParseInstances(
    const GLTFScene &scene,
    const glm::mat4 &coordinate_txfm,
    const vector<MaterialType> &materials)
{
    vector<pair<uint32_t, glm::mat4>> node_stack;
    for (uint32_t root_node : scene.rootNodes) {
        node_stack.emplace_back(root_node, coordinate_txfm);
    }

    vector<InstanceProperties> instances;
    while (!node_stack.empty()) {
        auto [node_idx, parent_txfm] = node_stack.back();
        node_stack.pop_back();

        const GLTFNode &cur_node = scene.nodes[node_idx];
        glm::mat4 cur_txfm = parent_txfm * cur_node.transform;

        for (const uint32_t child_idx : cur_node.children) {
            node_stack.emplace_back(child_idx, cur_txfm);
        }

        if (cur_node.meshIdx < scene.meshes.size()) {
            // Decompose transform
            glm::vec3 position(cur_txfm[3]);

            glm::vec3 scale(glm::length(cur_txfm[0]),
                            glm::length(cur_txfm[1]),
                            glm::length(cur_txfm[2]));

            if (glm::dot(glm::cross(glm::vec3(cur_txfm[0]),
                                    glm::vec3(cur_txfm[1])),
                         glm::vec3(cur_txfm[2])) < 0.f) {
                scale.x *= -1.f;
            }

            glm::vec3 v1 = glm::normalize(cur_txfm[0] / scale.x);
            glm::vec3 v2 = cur_txfm[1] / scale.y;
            glm::vec3 v3 = cur_txfm[2] / scale.z;

            v2 = glm::normalize(v2 - dot(v2, v1) * v1);
            v3 = v3 - glm::dot(v3, v1) * v1;
            v3 -= glm::dot(v3, v2) * v2;
            v3 = glm::normalize(v3);

            glm::mat3 rot(v1, v2, v3);

            glm::quat rot_quat = glm::quat_cast(rot);

            bool is_transparent = false;
            vector<uint32_t> instance_materials;
            for (const auto &prim : scene.meshes[cur_node.meshIdx].primitives) {
                instance_materials.push_back(prim.materialIdx);

                if (materials[prim.materialIdx].baseTransmission > 0.f) {
                    is_transparent = true;
                }
            }

            instances.push_back({
                to_string(instances.size()),
                cur_node.meshIdx,
                move(instance_materials),
                position,
                rot_quat,
                scale,
                false,
                is_transparent,
            });
        }
    }

    return instances;
}

template <typename VertexType, typename MaterialType>
SceneDescription<VertexType, MaterialType> parseGLTF(
    filesystem::path scene_path, const glm::mat4 &base_txfm,
    optional<string_view> texture_dir)
{
    auto raw_scene = gltfLoad(scene_path);

    vector<MaterialType> materials =
        gltfParseMaterials<MaterialType>(raw_scene, texture_dir);

    vector<Object<VertexType>> geometry;

    for (uint32_t mesh_idx = 0; mesh_idx < raw_scene.meshes.size();
         mesh_idx++) {
        vector<Mesh<VertexType>> meshes =
            gltfParseMesh<VertexType>(raw_scene, mesh_idx);

        Object<VertexType> obj {
            raw_scene.meshes[mesh_idx].name,
            move(meshes),
        };

        geometry.emplace_back(move(obj));
    }

    vector<InstanceProperties> instances =
        gltfParseInstances(raw_scene, base_txfm, materials);

    return SceneDescription<VertexType, MaterialType> {
        move(geometry),
        move(materials),
        move(instances),
        {},
        "",
    };
}

}
}
