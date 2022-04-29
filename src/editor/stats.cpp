#include <rlpbr.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <random>

#include "navmesh.hpp"
#include "stats.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

using namespace std;
using namespace RLpbr;

static tuple<glm::dvec3, glm::dvec3> updateMeanAndVarDouble(
    glm::dvec3 iter_mean, glm::dvec3 iter_var, double iter_count,
    glm::dvec3 running_mean, glm::dvec3 running_var, double running_count,
    double new_count)
{
    glm::dvec3 m_a = running_var * running_count;
    glm::dvec3 m_b = iter_var * iter_count;

    glm::dvec3 mean_diff = iter_mean - running_mean;

    glm::dvec3 m2 = m_a + m_b + (mean_diff * mean_diff) * running_count * 
        iter_count / new_count; 

    glm::dvec3 new_var = m2 / new_count;
    glm::dvec3 new_mean = (running_count * running_mean +
                           iter_count * iter_mean) / new_count;

    return {
        new_mean,
        new_var,
    };
}

static tuple<glm::vec3, glm::vec3, uint64_t> updateMeanAndVar(
    glm::vec3 iter_mean, glm::vec3 iter_var, uint64_t iter_count,
    glm::vec3 running_mean, glm::vec3 running_var, uint64_t running_count)
{
    uint64_t new_count = iter_count + running_count;

    auto [new_mean, new_var] = updateMeanAndVarDouble(
        iter_mean, iter_var, iter_count, running_mean, running_var,
        running_count, new_count);

    return {
        glm::vec3(new_mean),
        glm::vec3(new_var),
        new_count,
    };
}

int main(int argc, char *argv[]) {
    if (argc < 9) {
        cerr << argv[0] << " scene_list env_maps batch_size res spp path_depth points_per_env mode" << endl;
        exit(EXIT_FAILURE);
    }

    uint32_t batch_size = stoul(argv[3]);
    uint32_t res = stoul(argv[4]);
    uint32_t spp = stoul(argv[5]);
    uint32_t path_depth = stoul(argv[6]);
    uint32_t points_per_env = stoul(argv[7]);
    RenderMode mode {};
    if (!strcmp(argv[8], "PathTracer")) {
        mode = RenderMode::PathTracer;
    } else if (!strcmp(argv[8], "Biased")) {
        mode = RenderMode::Biased;
    }

    RenderFlags flags {};
    flags |= RenderFlags::Tonemap;
    flags |= RenderFlags::Randomize;
    flags |= RenderFlags::Denoise;
    flags |= RenderFlags::AuxiliaryOutputs;

    Renderer renderer({0, 1, batch_size, res, res, spp, path_depth, 128,
        mode, flags, 0.f, BackendSelect::Vulkan});

    auto loader = renderer.makeLoader();

     // Load environment maps
    {
        auto env_dir = filesystem::path(argv[2]);
    
        vector<string> env_paths;
        vector<const char *> env_cstrs;
        for (auto &entry : filesystem::directory_iterator(env_dir)) {
            const string filename = entry.path().string();
            if (string_view(filename).substr(
                    filename.size() - strlen("bpsenv")) == "bpsenv") {
                env_paths.emplace_back(move(filename));
                env_cstrs.push_back(env_paths.back().c_str());
            }
        }
    
        auto env_maps = loader.loadEnvironmentMaps(env_cstrs.data(), env_cstrs.size());
        renderer.setActiveEnvironmentMaps(move(env_maps));
    }

    vector<pair<string, string>> scenes;
    {
        ifstream scene_list(argv[1]);
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

    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<float> rot_dist(0.f, 1.f);
    srand(time(nullptr));

    glm::vec3 mean { 0, 0, 0 };
    glm::vec3 var { 1, 1, 1 };
    uint64_t total_count = 0;

    MeanVarContext mv_ctx = getMeanVarContext(batch_size, res, res);

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

        for (int i = 0; i < (int)points_per_env; i++) {
            for (int env_idx = 0; env_idx < (int)batch_size; env_idx++) {
                glm::vec3 pos = navmesh.getRandomPoint() + glm::vec3(0, 1, 0);
                float angle = rot_dist(mt) * 2.f * M_PI;
                glm::quat rot = glm::angleAxis(angle, glm::vec3(0, 1, 0));

                batch.getEnvironment(env_idx).setCameraView(
                    pos, rot * glm::vec3(0, 0, 1), glm::vec3(0, 1, 0),
                    rot * glm::vec3(1, 0, 0));
            }

            renderer.render(batch);
            renderer.waitForBatch(batch);

            auto [cur_mean, cur_var] = computeMeanAndVar(output, mv_ctx);

            tie(mean, var, total_count) = updateMeanAndVar(
                glm::make_vec3(cur_mean.data()),
                glm::make_vec3(cur_var.data()),
                1, mean, var, total_count);
        }

        cout << glm::to_string(mean) << endl;
        cout << glm::to_string(var) << endl;
    }
}
