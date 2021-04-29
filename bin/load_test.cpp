#include <rlpbr.hpp>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <atomic>
#include <thread>
#include <optional>

using namespace std;
using namespace RLpbr;

constexpr int num_batches = 10000;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << argv[0] << "num_loaders scenes..." << endl;
        exit(EXIT_FAILURE);
    }

    int num_loaders = atoi(argv[1]);
    Renderer renderer({0, (uint32_t)num_loaders, (uint32_t)num_loaders,
                      64, 64, 4, 4, 0, false, false,
                      BackendSelect::Optix});

    uint32_t num_scenes = argc - 2;

    auto getSceneName = [&](int i) {
        return argv[2 + (i % num_scenes)];
    };
    
    vector<Environment> active_envs;
    active_envs.reserve(num_loaders);
    atomic<Environment *> *loader_envs = new atomic<Environment *>[num_loaders];

    vector<thread> threads;
    threads.reserve(num_loaders);
    vector<AssetLoader> loaders;
    loaders.reserve(num_loaders);


    for (int i = 0; i < num_loaders; i++) {
        auto loader = renderer.makeLoader();
        auto init_scene = loader.loadScene(getSceneName(i));
        active_envs.emplace_back(renderer.makeEnvironment(move(init_scene)));
        loader_envs[i].store(nullptr);

        loaders.emplace_back(move(loader));
    }

    atomic_bool should_exit = false;
    for (int i = 0; i < num_loaders; i++) {
        threads.emplace_back([i, &loader_envs, &renderer, &getSceneName, &should_exit](auto &&loader) {
            for (int iter = 0; should_exit.load(memory_order_relaxed) != true;
                 iter++) {
                auto scene = loader.loadScene(getSceneName(i + iter));

                Environment *env = new Environment(renderer.makeEnvironment(scene));
                Environment *old_env = loader_envs[i].exchange(env, memory_order_acq_rel);
                delete old_env;
            }

            Environment *cleanup = loader_envs[i];
            delete cleanup;
        }, move(loaders[i]));
    }

    for (int i = 0; i < num_batches; i++) {
        for (int e = 0; e < num_loaders; e++) {
            Environment *env = loader_envs[e].exchange(nullptr, memory_order_acq_rel);
            if (env != nullptr) {
                active_envs[e] = move(*env);
                delete env;
            }
        }
        renderer.render(active_envs.data());
    }

    should_exit = true;
    for (auto &t : threads) {
        t.join();
    }

    delete[] loader_envs;
}
