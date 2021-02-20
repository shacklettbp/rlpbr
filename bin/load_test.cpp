#include <rlpbr.hpp>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <fstream>

#include <thread>

using namespace std;
using namespace RLpbr;

constexpr int num_iters = 1000;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << argv[0] << "num_loaders scenes..." << endl;
        exit(EXIT_FAILURE);
    }

    int num_loaders = atoi(argv[1]);
    Renderer renderer({0, (uint32_t)num_loaders, 1, 4, 4, 4, 4, false,
                       BackendSelect::Optix});

    uint32_t num_scenes = argc - 2;
    
    vector<thread> threads;
    threads.reserve(num_loaders);

    for (int i = 0; i < num_loaders; i++) {
        auto l = renderer.makeLoader();

        threads.emplace_back([&](auto &&loader) {
            for (int iter = 0; iter < num_iters; iter++) {
                auto scene = loader.loadScene(argv[2 + (iter % num_scenes)]);
                (void)scene;
            }
        }, move(l));
    }

    for (auto &t : threads) {
        t.join();
    }
}
