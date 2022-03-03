#include <rlpbr.hpp>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace RLpbr;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << argv[0] << " scene" << endl;
        exit(EXIT_FAILURE);
    }

    Renderer renderer({0, 1, 1, 1, 1, 1, 1,
                       1, RenderMode::Biased, {},
                       0.f, BackendSelect::Vulkan});

    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);
}
