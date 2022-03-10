#include <rlpbr.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <random>

#include "navmesh.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <cuda_runtime.h>
#include <../../bin/oiio_bridge.hpp>

using namespace std;
using namespace RLpbr;

struct DataPointers {
    half *color;
    half *albedo;
    half *normal;
    half *colorHost;
    half *albedoHost;
    half *normalHost;
};

int main(int argc, char *argv[]) {
    if (argc < 10) {
        cerr << argv[0] << " scene_list out_dir batch_size res src_spp src_path_depth ref_spp ref_path_depth points_per_env" << endl;
        exit(EXIT_FAILURE);
    }

    const char *scene_list_path = argv[1];
    const char *out_dir = argv[2];

    uint32_t batch_size = stoul(argv[3]);
    uint32_t res = stoul(argv[4]);
    uint32_t src_spp = stoul(argv[5]);
    uint32_t src_path_depth = stoul(argv[6]);
    uint32_t ref_spp = stoul(argv[7]);
    uint32_t ref_path_depth = stoul(argv[8]);
    uint32_t points_per_env = stoul(argv[9]);

    uint32_t batches_per_env = vk::divideRoundUp(points_per_env, batch_size);

    Renderer src_renderer({0, 1, batch_size, res, res, src_spp, src_path_depth,
        128, RenderMode::PathTracer, RenderFlags::AuxiliaryOutputs, 0.f,
        BackendSelect::Vulkan});

    Renderer ref_renderer({0, 1, batch_size, res, res, ref_spp, ref_path_depth,
        128, RenderMode::PathTracer, RenderFlags::AuxiliaryOutputs, 0.f,
        BackendSelect::Vulkan});

    auto src_loader = src_renderer.makeLoader();
    auto ref_loader = ref_renderer.makeLoader();

    vector<pair<string, string>> scenes;
    {
        ifstream scene_list(scene_list_path);
        if (!scene_list.is_open()) {
            cerr << "Failed to open " << argv[1] << endl;
            abort();
        }

        string line;
        while (true) {
            getline(scene_list, line);
            if (scene_list.eof()) {
                break;
            }

            auto pos = line.find(";");
            scenes.emplace_back(
                line.substr(0, pos),
                line.substr(pos + 1));
        }
    }

    RenderBatch src_batch = src_renderer.makeRenderBatch();
    RenderBatch ref_batch = ref_renderer.makeRenderBatch();
    {
        auto init_scene = src_loader.loadScene(scenes[0].first);
        auto ref_scene = ref_loader.loadScene(scenes[0].first);
        for (int i = 0; i < (int)batch_size; i++) {
            src_batch.initEnvironment(i, src_renderer.makeEnvironment(init_scene));
            ref_batch.initEnvironment(i, ref_renderer.makeEnvironment(ref_scene));
        }
    }

    uint64_t num_pixels = batch_size * res * res;

    auto getRendererPointers = [num_pixels](auto &renderer, auto &batch) {
        half *ref_color = renderer.getOutputPointer(batch);
        AuxiliaryOutputs aux_outputs = renderer.getAuxiliaryOutputs(batch);

        half *color_host;
        cudaHostAlloc(&color_host,
                      num_pixels * sizeof(half) * 3, cudaHostAllocDefault);
        half *albedo_host;
        cudaHostAlloc(&albedo_host,
                      num_pixels * sizeof(half) * 3, cudaHostAllocDefault);
        half *normal_host;
        cudaHostAlloc(&normal_host,
                      num_pixels * sizeof(half) * 3, cudaHostAllocDefault);

        return DataPointers {
            ref_color,
            aux_outputs.albedo,
            aux_outputs.normal,
            color_host,
            albedo_host,
            normal_host,
        };
    };

    auto src_ptrs = getRendererPointers(src_renderer, src_batch);
    auto ref_ptrs = getRendererPointers(ref_renderer, ref_batch);

    mt19937 mt(1792582337);
    uniform_real_distribution<float> rot_dist(0.f, 1.f);
    srand(1792582337);

    auto saveBatch = [num_pixels, batch_size, res, out_dir](
            const DataPointers &ptrs,
            const char *prefix,
            int offset, bool save_normal) {
        cudaMemcpy2D(ptrs.colorHost, sizeof(half) * 3,
                     ptrs.color, sizeof(half) * 4,
                     sizeof(half) * 3, num_pixels,
                     cudaMemcpyDeviceToHost);
        cudaMemcpy(ptrs.albedoHost, ptrs.albedo,
                   num_pixels * sizeof(half) * 3, cudaMemcpyDeviceToHost);

        if (save_normal) {
            cudaMemcpy(ptrs.normalHost, ptrs.normal,
                       num_pixels * sizeof(half) * 3, cudaMemcpyDeviceToHost);
        }

        for (int batch_idx = 0; batch_idx < (int)batch_size; batch_idx++) {
            string color_name = out_dir + string(prefix) +
                "color_" + to_string(offset + batch_idx) + ".exr";
            string albedo_name = out_dir + string(prefix) +
                "albedo_" + to_string(offset + batch_idx) + ".exr";
            string normal_name = out_dir + string(prefix) +
                "normal_" + to_string(offset + batch_idx) + ".exr";

            saveHDR(color_name.c_str(), res, res,
                    ptrs.colorHost + batch_idx * res * res * 3, true);
            saveHDR(albedo_name.c_str(), res, res,
                    ptrs.albedoHost + batch_idx * res * res * 3, true);

            if (save_normal) {
                saveHDR(normal_name.c_str(), res, res,
                        ptrs.normalHost + batch_idx * res * res * 3, true);
            }
        }
    };

    int img_count = 0;
    for (const auto &[scene_path, navmesh_path] : scenes) {
        auto src_scene = src_loader.loadScene(scene_path);
        auto ref_scene = ref_loader.loadScene(scene_path);
        const char *navmesh_err;
        auto navmesh_opt = editor::loadNavmesh(navmesh_path.c_str(),
                                               &navmesh_err);

        if (!navmesh_opt.has_value()) {
            cerr << "Failed to load navmesh: " << navmesh_err << endl;
            abort();
        }

        auto navmesh = move(*navmesh_opt);

        for (int env_idx = 0; env_idx < (int)batch_size; env_idx++) {
            src_batch.getEnvironment(env_idx) =
                src_renderer.makeEnvironment(src_scene);
            ref_batch.getEnvironment(env_idx) =
                ref_renderer.makeEnvironment(ref_scene);
        }

        for (int i = 0; i < (int)batches_per_env; i++) {
            cout << scene_path << ": " << i << "/" << batches_per_env << endl;
            for (int env_idx = 0; env_idx < (int)batch_size; env_idx++) {
                glm::vec3 pos = navmesh.getRandomPoint();

                // Elevate
                pos += glm::vec3(0, 1, 0);

                float angle = rot_dist(mt) * 2.f * M_PI;
                auto rot = 
                    glm::mat3(glm::angleAxis(angle, glm::vec3(0, 1, 0)));

                glm::vec3 fwd = rot * glm::vec3(0, 0, 1);
                glm::vec3 up = rot * glm::vec3(0, 1, 0);
                glm::vec3 right = rot * glm::vec3(-1, 0, 0);

                src_batch.getEnvironment(env_idx).setCameraView(
                    pos, fwd, up, right);

                ref_batch.getEnvironment(env_idx).setCameraView(
                    pos, fwd, up, right);
            }

            src_renderer.render(src_batch);
            src_renderer.waitForBatch(src_batch);

            saveBatch(src_ptrs, "src_", img_count, true);

            ref_renderer.render(ref_batch);
            ref_renderer.waitForBatch(ref_batch);

            saveBatch(ref_ptrs, "ref_", img_count, false);

            img_count += batch_size;
        }
    }
}
