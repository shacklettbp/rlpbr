#include <rlpbr.hpp>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <fstream>

using namespace std;
using namespace RLpbr;

constexpr uint32_t num_frames = 1000000;

constexpr uint32_t max_load_frames = 10000;

vector<glm::mat4> readViews(const char *dump_path)
{
    ifstream dump_file(dump_path, ios::binary);

    vector<glm::mat4> views;

    for (size_t i = 0; i < max_load_frames; i++) {
        float raw[16];
        dump_file.read((char *)raw, sizeof(float)*16);
        views.emplace_back(
                glm::mat4(raw[0], raw[1], raw[2], raw[3],
                          raw[4], raw[5], raw[6], raw[7],
                          raw[8], raw[9], raw[10], raw[11],
                          raw[12], raw[13], raw[14], raw[15]));
    }

    return views;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        cerr << argv[0] << " scene batch_size res views" << endl;
        exit(EXIT_FAILURE);
    }

    uint32_t batch_size = stoul(argv[2]);
    uint32_t res = stoul(argv[3]);
    vector<glm::mat4> init_views;
    if (argc > 4) {
        init_views = readViews(argv[4]);
    } else {
        init_views = {glm::mat4(
            -1.19209e-07, 0, 1, 0,
            0, 1, 0, 0,
            -1, 0, -1.19209e-07, 0,
            -3.38921, 1.62114, -3.34509, 1)
        };
    }

    Renderer renderer({0, 1, batch_size, res, res, 4, 4, false, false,
                       0, BackendSelect::Optix});

    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);

    vector<Environment> envs;

    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        envs.emplace_back(renderer.makeEnvironment(scene,
            glm::vec3(10.184777, 0.994812, -2.762561),
            glm::vec3(10.812518, 0.927049, -1.987069),
            glm::vec3(-0.000107, 0.996203, 0.087133)));
    }

    auto start = chrono::steady_clock::now();

    uint32_t num_iters = num_frames / batch_size;

    //uint32_t cur_view = 0;

    for (uint32_t i = 0; i < num_iters; i++) {
        //for (auto &env : envs) {
        //    env.setDirty();
        //}
        renderer.render(envs.data());
        renderer.waitForFrame();
    }

    auto end = chrono::steady_clock::now();

    auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Batch size " << batch_size << ", Resolution " << res << ", FPS: " << ((double)num_iters * (double)batch_size /
            (double)diff.count()) * 1000.0 << endl;
}
