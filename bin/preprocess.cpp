#include <iostream>
#include <cstdlib>

#include <rlpbr/preprocess.hpp>

using namespace std;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << argv[0] << " src dst" << endl;
        exit(EXIT_FAILURE);
    }

    RLpbr::ScenePreprocessor dumper(
        argv[1],
        glm::mat4(
            1, 0, 0, 0,
            0, -1.19209e-07, -1, 0,
            0, 1, -1.19209e-07, 0, 0, 0, 0, 1)
    );

    dumper.dump(argv[2]);

    return 0;
}
