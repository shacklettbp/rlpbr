#include "scene.hpp"
#include "common.hpp"
#include <rlpbr_core/utils.hpp>
#include <rlpbr_core/physics.hpp>

#include <fstream>
#include <iostream>

#include <glm/gtx/string_cast.hpp>

using namespace std;

namespace RLpbr {

SceneLoadData SceneLoadData::loadFromDisk(string_view scene_path_name,
                                          bool load_full_file)
{
    filesystem::path scene_path(scene_path_name);
    filesystem::path scene_dir = scene_path.parent_path();

    ifstream scene_file(scene_path, ios::binary);

    auto read_uint = [&]() {
        uint32_t val;
        scene_file.read(reinterpret_cast<char *>(&val), sizeof(uint32_t));

        return val;
    };

    uint32_t magic = read_uint();
    if (magic != 0x55555555) {
        cerr << "Invalid preprocessed scene" << endl;
        abort();
    }

    auto alignSkip = [&](uint32_t pad = 256) {
        auto cur_pos = scene_file.tellg();
        auto alignment = cur_pos % pad;
        if (alignment != 0) {
            scene_file.seekg(pad - alignment, ios::cur);
        }
    };

    StagingHeader hdr;
    scene_file.read(reinterpret_cast<char *>(&hdr), sizeof(StagingHeader));

    alignSkip();

    vector<MeshInfo> mesh_infos(hdr.numMeshes);
    scene_file.read(reinterpret_cast<char *>(mesh_infos.data()),
                    sizeof(MeshInfo) * hdr.numMeshes);

    vector<ObjectInfo> obj_infos(hdr.numObjects);
    scene_file.read(reinterpret_cast<char *>(obj_infos.data()),
                    sizeof(ObjectInfo) * hdr.numObjects);

    uint32_t num_lights = read_uint();
    vector<LightProperties> light_props(num_lights);
    scene_file.read(reinterpret_cast<char *>(light_props.data()),
                    sizeof(LightProperties) * num_lights);

    TextureInfo textures;
    vector<char> name_buffer;
    do {
        name_buffer.push_back(scene_file.get());
    } while (name_buffer.back() != 0);
    textures.textureDir = scene_dir / name_buffer.data();
    name_buffer.clear();

    do {
        name_buffer.push_back(scene_file.get());
    } while (name_buffer.back() != 0);
    textures.envMap = name_buffer.data();
    name_buffer.clear();

    auto readTextureNames = [&]() {
        uint32_t num_tex = read_uint();

        vector<string> names;
        for (int tex_idx = 0; tex_idx < (int)num_tex; tex_idx++) {
            do {
                name_buffer.push_back(scene_file.get());
            } while (name_buffer.back() != 0);

            names.emplace_back(name_buffer.data());
            name_buffer.clear();
        }

        return names;
    };

    textures.base = readTextureNames();
    textures.metallicRoughness = readTextureNames();
    textures.specular = readTextureNames();
    textures.normal = readTextureNames();
    textures.emittance = readTextureNames();
    textures.transmission = readTextureNames();
    textures.clearcoat = readTextureNames();
    textures.anisotropic = readTextureNames();

    vector<MaterialTextures> texture_indices(hdr.numMaterials);
    scene_file.read(reinterpret_cast<char *>(texture_indices.data()),
                    sizeof(MaterialTextures) * hdr.numMaterials);

    uint32_t num_instance_materials = read_uint();

    vector<uint32_t> instance_materials(num_instance_materials);
    scene_file.read(reinterpret_cast<char *>(instance_materials.data()),
                    sizeof(uint32_t) * num_instance_materials);

    AABB default_bbox;
    scene_file.read(reinterpret_cast<char *>(&default_bbox),
                    sizeof(AABB));

    uint32_t num_instances = read_uint();

    vector<ObjectInstance> instances(num_instances);
    scene_file.read(reinterpret_cast<char *>(instances.data()),
                    sizeof(ObjectInstance) * num_instances);

    vector<InstanceTransform> default_transforms(num_instances);
    scene_file.read(reinterpret_cast<char *>(default_transforms.data()),
                    sizeof(InstanceTransform) * num_instances);

    vector<InstanceFlags> default_inst_flags(num_instances);
    scene_file.read(reinterpret_cast<char *>(default_inst_flags.data()),
                    sizeof(InstanceFlags) * num_instances);

    uint32_t num_static = read_uint();
    uint32_t num_dynamic = read_uint();

    DynArray<PhysicsInstance> static_instances(num_static);
    scene_file.read(reinterpret_cast<char *>(static_instances.data()),
                    sizeof(PhysicsInstance) * num_static);

    DynArray<PhysicsInstance> dynamic_instances(num_dynamic);
    scene_file.read(reinterpret_cast<char *>(dynamic_instances.data()),
                    sizeof(PhysicsInstance) * num_dynamic);


    DynArray<PhysicsTransform> dynamic_transforms(num_dynamic);
    scene_file.read(reinterpret_cast<char *>(dynamic_transforms.data()),
                    sizeof(PhysicsTransform) * num_dynamic);

    uint32_t num_sdfs = read_uint();
    vector<string> sdf_paths;
    sdf_paths.reserve(num_sdfs);
    for (int sdf_idx = 0; sdf_idx < (int)num_sdfs; sdf_idx++) {
        do {
            name_buffer.push_back(scene_file.get());
        } while (name_buffer.back() != 0);

        sdf_paths.emplace_back(name_buffer.data());
        name_buffer.clear();
    }

    alignSkip();

    auto loadRemainingData = [&]() {
        vector<char> file_data(hdr.totalBytes);
        scene_file.read(file_data.data(), hdr.totalBytes);

        return file_data;
    };

    return SceneLoadData {
        hdr,
        move(mesh_infos),
        move(obj_infos),
        move(textures),
        move(texture_indices),
        EnvironmentInit(default_bbox,
                        move(instances), 
                        move(instance_materials),
                        move(default_transforms),
                        move(default_inst_flags),
                        move(light_props)),
        PhysicsMetadata {
            move(sdf_paths),
            move(static_instances),
            move(dynamic_instances),
            move(dynamic_transforms),
        },
        load_full_file ? 
            variant<ifstream, vector<char>>(loadRemainingData()) :
            variant<ifstream, vector<char>>(move(scene_file)),
    };
}

EnvironmentInit::EnvironmentInit(const AABB &bbox,
    vector<ObjectInstance> instances,
    vector<uint32_t> instance_materials,
    vector<InstanceTransform> transforms,
    vector<InstanceFlags> instance_flags,
    vector<LightProperties> l)
    : defaultBBox(bbox),
      defaultInstances(move(instances)),
      defaultInstanceMaterials(move(instance_materials)),
      defaultTransforms(move(transforms)),
      defaultInstanceFlags(move(instance_flags)),
      indexMap(),
      reverseIDMap(),
      lights(move(l)),
      lightIDs(),
      lightReverseIDs()
{
    indexMap.reserve(instances.size());
    reverseIDMap.reserve(instances.size());

    for (uint32_t cur_id = 0; cur_id < instances.size(); cur_id++) {
        indexMap.emplace_back(cur_id);
        reverseIDMap.push_back(cur_id);
    }

    lightIDs.reserve(lights.size());
    lightReverseIDs.reserve(lights.size());
    for (uint32_t light_idx = 0; light_idx < lights.size(); light_idx++) {
        lightIDs.push_back(light_idx);
        lightReverseIDs.push_back(light_idx);
    }
}

}
