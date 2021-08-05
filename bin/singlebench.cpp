#include <rlpbr.hpp>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <fstream>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#define STRINGIFY_HELPER(m) #m
#define STRINGIFY(m) STRINGIFY_HELPER(m)


using namespace std;
using namespace RLpbr;

constexpr uint32_t num_frames = 100000;

vector<pair<glm::vec3, glm::quat>> readViews(const string &dump_path)
{
    ifstream dump_file(dump_path, ios::binary);
    uint32_t num_views;
    dump_file.read((char *)&num_views, sizeof(uint32_t));

    vector<pair<glm::vec3, glm::quat>> views;

    for (int i = 0; i < (int)num_views; i++) {
        glm::vec3 position;
        glm::quat rotation;
        dump_file.read((char *)glm::value_ptr(position), sizeof(glm::vec3));
        dump_file.read((char *)glm::value_ptr(rotation), sizeof(glm::quat));

        views.emplace_back(position, rotation);
    }

    return views;
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        cerr << argv[0] << " scene batch_size res spp path depth views" << endl;
        exit(EXIT_FAILURE);
    }

    uint32_t batch_size = stoul(argv[2]);
    uint32_t res = stoul(argv[3]);
    uint32_t spp = stoul(argv[4]);
    uint32_t path_depth = stoul(argv[5]);
    vector<pair<glm::vec3, glm::quat>> views;
    if (argc > 6) {
        views = readViews(argv[6]);
    } else {
        views = readViews(string(STRINGIFY(RLPBR_DATA_DIR)) + "/test_cams.bin");
    }

    cout << "Loaded " << views.size() << " views" << endl;

    Renderer renderer({0, 1, batch_size, res, res, spp, path_depth, 0, false,
        false, false, 0.f, BackendSelect::Vulkan});

    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);

    vector<Environment> envs;

    Renderer::BatchInitializer init;

    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        init.addEnvironment(scene);
    }

    RenderBatch batch = renderer.makeRenderBatch(move(init));

    renderer.render(batch);
    renderer.waitForFrame();

    auto start = chrono::steady_clock::now();

    uint32_t num_iters = num_frames / batch_size;

    uint32_t cur_view = 0;

    for (uint32_t i = 0; i < num_iters; i++) {
        for (int env_idx = 0; env_idx < (int)batch_size; env_idx++) {
            auto [position, rotation] = views[cur_view];
            cur_view = (cur_view + 1) % views.size();

            auto &env = batch.getEnvironment(env_idx);

            env.setCameraView(position,
                rotation * glm::vec3(0.f, 0.f, 1.f),
                glm::vec3(0.f, 1.f, 0.f),
                rotation * glm::vec3(1.f, 0.f, 0.f));
        }
        renderer.render(batch);
        renderer.waitForFrame();
    }

    auto end = chrono::steady_clock::now();

    auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Batch size " << batch_size << ", Resolution " << res << ", FPS: " << ((double)num_iters * (double)batch_size /
            (double)diff.count()) * 1000.0 << endl;
}
