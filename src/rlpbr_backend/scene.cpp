#include "scene.hpp"
#include "common.hpp"

#include <fstream>
#include <iostream>

using namespace std;

namespace RLpbr {

SceneLoadData SceneLoadData::loadFromDisk(string_view scene_path_name)
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

    StagingHeader hdr;
    scene_file.read(reinterpret_cast<char *>(&hdr), sizeof(StagingHeader));

    auto cur_pos = scene_file.tellg();
    auto post_hdr_alignment = cur_pos % 256;
    if (post_hdr_alignment != 0) {
        scene_file.seekg(256 - post_hdr_alignment, ios::cur);
    }

    vector<MeshInfo> mesh_infos(hdr.numMeshes);
    scene_file.read(reinterpret_cast<char *>(mesh_infos.data()),
                    sizeof(MeshInfo) * hdr.numMeshes);

    uint32_t num_lights = read_uint();
    vector<LightProperties> light_props(num_lights);
    scene_file.read(reinterpret_cast<char *>(light_props.data()),
                    sizeof(LightProperties) * num_lights);

    TextureInfo textures;
    uint32_t num_textures = read_uint();
    vector<char> name_buffer;
    for (uint32_t tex_idx = 0; tex_idx < num_textures; tex_idx++) {
        do {
            name_buffer.push_back(scene_file.get());
        } while (name_buffer.back() != 0);

        textures.albedo.emplace_back(scene_dir / name_buffer.data());
        name_buffer.clear();
    }

    uint32_t num_instances = read_uint();

    // FIXME this should just be baked in...
    vector<InstanceProperties> instances;
    instances.reserve(num_instances);

    for (uint32_t inst_idx = 0; inst_idx < num_instances; inst_idx++) {
        uint32_t mesh_index = read_uint();
        uint32_t material_index = read_uint();
        glm::mat4x3 txfm;
        scene_file.read(reinterpret_cast<char *>(&txfm), sizeof(glm::mat4x3));
        instances.push_back({
            mesh_index,
            material_index,
            txfm,
        });
    }

    cur_pos = scene_file.tellg();
    auto post_inst_alignment = cur_pos % 256;
    if (post_inst_alignment != 0) {
        scene_file.seekg(256 - post_inst_alignment, ios::cur);
    }

    return SceneLoadData {
        hdr,
        move(mesh_infos),
        move(textures),
        EnvironmentInit(move(instances), move(light_props), hdr.numMeshes),
        variant<ifstream, vector<char>>(move(scene_file)),
    };
}

EnvironmentInit::EnvironmentInit(const vector<InstanceProperties> &instances,
                                 const vector<LightProperties> &l,
                                 uint32_t num_meshes)
    : transforms(num_meshes),
      materials(num_meshes),
      indexMap(),
      reverseIDMap(num_meshes),
      lights(l),
      lightIDs(),
      lightReverseIDs()
{
    indexMap.reserve(instances.size());

    for (uint32_t cur_id = 0; cur_id < instances.size(); cur_id++) {
        const auto &inst = instances[cur_id];
        uint32_t mesh_idx = inst.meshIndex;

        uint32_t inst_idx = transforms[mesh_idx].size();

        transforms[mesh_idx].push_back(inst.txfm);
        materials[mesh_idx].push_back(inst.materialIndex);
        reverseIDMap[mesh_idx].push_back(cur_id);
        indexMap.emplace_back(mesh_idx, inst_idx);
    }

    lightIDs.reserve(lights.size());
    lightReverseIDs.reserve(lights.size());
    for (uint32_t light_idx = 0; light_idx < lights.size(); light_idx++) {
        lightIDs.push_back(light_idx);
        lightReverseIDs.push_back(light_idx);
    }
}

}
