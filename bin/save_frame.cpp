#include <rlpbr.hpp>
#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>

// FIXME
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
using namespace RLpbr;

template<typename T>
static vector<T> copyToHost(const T *dev_ptr, uint32_t width,
                            uint32_t height, uint32_t num_channels)
{
    uint64_t num_pixels = width * height * num_channels;

    vector<T> buffer(num_pixels);

    cudaMemcpy(buffer.data(), dev_ptr, sizeof(T) * num_pixels,
               cudaMemcpyDeviceToHost);

    return buffer;
}

void saveFrame(const char *fname, const half *dev_ptr,
               uint32_t width, uint32_t height, uint32_t num_channels)
{
    auto buffer = copyToHost(dev_ptr, width, height, num_channels);

    vector<uint8_t> sdr_buffer(buffer.size());
    for (unsigned i = 0; i < buffer.size(); i++) {
        half v = buffer[i];
        if (v < 0) v = half(0.f);
        if (v > 1) v = half(1.f);
        sdr_buffer[i] = v * 255;
    }

    stbi_write_bmp(fname, width, height, num_channels, sdr_buffer.data());
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << argv[0] << "scene batch_size" << endl;
        exit(EXIT_FAILURE);
    }

    uint32_t batch_size = stoul(argv[2]);

    glm::u32vec2 out_dim(256, 256);

    Renderer renderer({0, 1, batch_size, out_dim.x, out_dim.y, 1, 1, false,
                       BackendSelect::Optix});

    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);
    vector<Environment> envs;

    glm::mat4 base(-1.19209e-07, 0, 1, 0,
                   0, 1, 0, 0,
                   -1, 0, -1.19209e-07, 0,
                   -3.38921, 1.62114, -3.34509, 1);
    
    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        envs.emplace_back(renderer.makeEnvironment(scene, 
            glm::rotate(base, glm::radians(10.f * batch_idx),
                            glm::vec3(0.f, 0.f, 1.f))));
    }

    renderer.render(envs.data());
    renderer.waitForFrame();

    half *base_out_ptr = renderer.getOutputPointer();

    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        saveFrame(("/tmp/out_color_" + to_string(batch_idx) + ".bmp").c_str(),
                  base_out_ptr + batch_idx * out_dim.x * out_dim.y * 3,
                  out_dim.x, out_dim.y, 3);
    }
}
