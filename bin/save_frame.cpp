#include <rlpbr.hpp>
#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>

#include <glm/gtx/transform.hpp>

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
        assert(v >= 0);
        float f = v / (v + 1.f);
        f = powf(f, 1.f/2.2f);
        if (f < 0) f = 0.f;
        if (f > 1) f = 1.f;
        sdr_buffer[i] = uint8_t(f * 255.f);
    }

    stbi_write_bmp(fname, width, height, num_channels, sdr_buffer.data());
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << argv[0] << "scene batch_size [spp] [depth]" << endl;
        exit(EXIT_FAILURE);
    }

    uint32_t batch_size = atoi(argv[2]);

    glm::u32vec2 out_dim(1920, 1080);

    uint32_t spp = 1;

    if (argc > 3) {
        spp = atoi(argv[3]);
    }

    uint32_t depth = 1;
    if (argc > 4) {
        depth = atoi(argv[4]);
    }

    Renderer renderer({0, 1, batch_size, out_dim.x, out_dim.y, spp, depth,
                       false, BackendSelect::Optix});

    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);
    vector<Environment> envs;

    glm::vec3 eye(12.796778, 1.331334, -0.135939);
    glm::vec3 look(12.622172, 1.227377, 0.843202);
    glm::vec3 up(-0.101932, 0.990977, 0.087038);

    glm::vec3 to_look = look - eye;
    
    
    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        glm::mat3 r = glm::rotate(glm::radians(10.f * batch_idx), up);
        r = glm::mat3(1.f);
        envs.emplace_back(renderer.makeEnvironment(scene, 
            eye, eye + r * to_look, up, 60.f));
    }
    envs.back().addLight(glm::vec3(-1.950218, 1.623819, 0.863453), glm::vec3(10.f));
    envs.back().addLight(glm::vec3(1.762336, 1.211801, -4.574429), glm::vec3(10.f));
    envs.back().addLight(glm::vec3(8.107919, 1.345027, -1.867001), glm::vec3(10.f));
    envs.back().addLight(glm::vec3(12.499360, 2.102839, 1.691340), glm::vec3(10.f));

    renderer.render(envs.data());
    renderer.waitForFrame();

    half *base_out_ptr = renderer.getOutputPointer();

    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        saveFrame(("/tmp/out_color_" + to_string(batch_idx) + ".bmp").c_str(),
                  base_out_ptr + batch_idx * out_dim.x * out_dim.y * 3,
                  out_dim.x, out_dim.y, 3);
    }
}
