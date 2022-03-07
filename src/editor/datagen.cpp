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

int main(int argc, char *argv[]) {
    if (argc < 8) {
        cerr << argv[0] << " scene_list out_dir batch_size res spp path_depth points_per_env" << endl;
        exit(EXIT_FAILURE);
    }

    const char *scene_list_path = argv[1];
    const char *out_dir = argv[2];

    uint32_t batch_size = stoul(argv[3]);
    uint32_t res = stoul(argv[4]);
    uint32_t spp = stoul(argv[5]);
    uint32_t path_depth = stoul(argv[6]);
    uint32_t points_per_env = stoul(argv[7]);

    uint32_t batches_per_env = vk::divideRoundUp(points_per_env, batch_size);

    Renderer renderer({0, 1, batch_size, res, res, spp, path_depth, 128,
        RenderMode::PathTracer, RenderFlags::AuxiliaryOutputs, 0.f, BackendSelect::Vulkan});

    auto loader = renderer.makeLoader();

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

    RenderBatch batch = renderer.makeRenderBatch();
    {
        auto init_scene = loader.loadScene(scenes[0].first);
        for (int i = 0; i < (int)batch_size; i++) {
            batch.initEnvironment(i, renderer.makeEnvironment(init_scene));
        }
    }

    half *output = renderer.getOutputPointer(batch);
    AuxiliaryOutputs aux_outputs = renderer.getAuxiliaryOutputs(batch);

    uint64_t num_pixels = batch_size * res * res;
    half *output_host;
    cudaHostAlloc(&output_host, num_pixels * sizeof(half) * 3, cudaHostAllocDefault);
    half *albedo_host;
    cudaHostAlloc(&albedo_host, num_pixels * sizeof(half) * 3, cudaHostAllocDefault);
    half *normal_host;
    cudaHostAlloc(&normal_host, num_pixels * sizeof(half) * 3, cudaHostAllocDefault);

    mt19937 mt(1292482335);
    uniform_real_distribution<float> rot_dist(0.f, 1.f);
    srand(1292482335);

    auto saveBatch = [&](const char *prefix, int offset) {
        cudaMemcpy2D(output_host, sizeof(half) * 3,
                     output, sizeof(half) * 4,
                     sizeof(half) * 3, num_pixels,
                     cudaMemcpyDeviceToHost);
        cudaMemcpy(albedo_host, aux_outputs.albedo, num_pixels * sizeof(half) * 3, cudaMemcpyDeviceToHost);
        cudaMemcpy(normal_host, aux_outputs.normal, num_pixels * sizeof(half) * 3, cudaMemcpyDeviceToHost);

        for (int batch_idx = 0; batch_idx < (int)batch_size; batch_idx++) {
            string color_name = out_dir + string(prefix) + "color_" + to_string(offset + batch_idx) + ".exr";
            string albedo_name = out_dir + string(prefix) + "albedo_" + to_string(offset + batch_idx) + ".exr";
            string normal_name = out_dir + string(prefix) + "normal_" + to_string(offset + batch_idx) + ".exr";

            saveHDR(color_name.c_str(), res, res, output_host + batch_idx * res * res * 3, true);
            saveHDR(albedo_name.c_str(), res, res, albedo_host + batch_idx * res * res * 3, true);
            saveHDR(normal_name.c_str(), res, res, normal_host + batch_idx * res * res * 3, true);
        }
    };

    int img_count = 0;
    for (const auto &[scene_path, navmesh_path] : scenes) {
        auto scene = loader.loadScene(scene_path);
        const char *navmesh_err;
        auto navmesh_opt = editor::loadNavmesh(navmesh_path.c_str(),
                                               &navmesh_err);

        if (!navmesh_opt.has_value()) {
            cerr << "Failed to load navmesh: " << navmesh_err << endl;
            abort();
        }

        auto navmesh = move(*navmesh_opt);

        for (int env_idx = 0; env_idx < (int)batch_size; env_idx++) {
            batch.getEnvironment(env_idx) = renderer.makeEnvironment(scene);
        }

        for (int i = 0; i < (int)batches_per_env; i++) {
            cout << scene_path << ": " << i << "/" << batches_per_env << endl;
            for (int env_idx = 0; env_idx < (int)batch_size; env_idx++) {
                glm::vec3 pos = navmesh.getRandomPoint();

                // Elevate
                pos += glm::vec3(0, 1, 0);

                float angle = rot_dist(mt) * 2.f * M_PI;
                glm::quat rot = glm::angleAxis(angle, glm::vec3(0, 1, 0));

                //batch.getEnvironment(env_idx).setCameraView(
                //    pos, rot * glm::vec3(0, 0, 1), rot * glm::vec3(0, 1, 0),
                //    rot * glm::vec3(1, 0, 0));

                batch.getEnvironment(env_idx).setCameraView(
                    pos, glm::vec3(0, 0, 1), glm::vec3(0, 1, 0),
                    glm::vec3(1, 0, 0));
            }

            renderer.render(batch);
            renderer.waitForBatch(batch);

            saveBatch("out_", img_count);

            renderer.render(batch);
            renderer.waitForBatch(batch);

            saveBatch("ref_", img_count);

            img_count += batch_size;
        }
    }
}
