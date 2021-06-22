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

float toSRGB(float v)
{
    if (v <= 0.00031308f) {
        return 12.92f * v;
    } else {
        return 1.055f*powf(v,(1.f / 2.4f)) - 0.055f;
    }
}

glm::vec3 tonemap(glm::vec3 v)
{
    v *= 0.6;

    float A = 2.51f;
    float B = 0.03f;
    float C = 2.43f;
    float D = 0.59f;
    float E = 0.14f;

    v = clamp((v*(A*v+B))/(v*(C*v+D)+E), 0.f, 1.f);
    return v;
}

void saveFrame(const char *fname, const half *dev_ptr,
               uint32_t width, uint32_t height, uint32_t num_channels)
{
    auto buffer = copyToHost(dev_ptr, width, height, num_channels);

    uint32_t num_pixels = buffer.size() / num_channels;
    vector<uint8_t> sdr_buffer(num_pixels * 3);

    for (unsigned pixel = 0; pixel < num_pixels; pixel++) {
        uint32_t i = pixel * num_channels;
        uint32_t k = pixel * 3;

        glm::vec3 rgb {
            float(buffer[i]),
            float(buffer[i + 1]),
            float(buffer[i + 2]),
        };

        assert(rgb.r >= 0 && rgb.g >= 0 && rgb.b >= 0);

        glm::vec3 tonemapped = tonemap(rgb);
        for (int j = 0; j < 3; j++) {
            float v = toSRGB(tonemapped[j]);
            if (v < 0) v = 0.f;
            if (v > 1) v = 1.f;
            sdr_buffer[k + j] = uint8_t(v * 255.f);
        }
    }

    stbi_write_bmp(fname, width, height, 3, sdr_buffer.data());
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
                       0, false, false, false, 0.f, BackendSelect::Vulkan});

    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);
    vector<Environment> envs;

    // Vase
    //glm::vec3 eye(10.573854, 1.332727, -2.085712);
    //glm::vec3 look(11.193787, 1.303279, -1.301609);
    //glm::vec3 up(-0.018044, 0.998499, 0.051766);


    // Mirror table
    glm::vec3 eye(-6.207120, 0.825648, 0.911869);
    glm::vec3 look(-6.698653, 0.796494, 0.041498);
    glm::vec3 up(-0.030218, 0.999409, -0.016411);

    // workout
    //glm::vec3 eye(-0.933040, 0.442653, -1.826718);
    //glm::vec3 look(-1.161190, 0.458737, -0.853224);
    //glm::vec3 up(-0.035431, 0.999065, -0.024810);

    glm::vec3 to_look = look - eye;
    
    
    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        glm::mat3 r = glm::rotate(glm::radians(10.f * batch_idx), up);
        envs.emplace_back(renderer.makeEnvironment(scene, 
            eye, eye + r * to_look, up, 60.f));
    }
    //envs.back().addLight(glm::vec3(-1.950218, 1.623819, 0.863453), glm::vec3(10.f));
    //envs.back().addLight(glm::vec3(1.762336, 1.211801, -4.574429), glm::vec3(10.f));
    //envs.back().addLight(glm::vec3(8.107919, 1.345027, -1.867001), glm::vec3(10.f));
    //envs.back().addLight(glm::vec3(12.499360, 2.102839, 1.691340), glm::vec3(10.f));

    renderer.render(envs.data());
    renderer.waitForFrame();

    half *base_out_ptr = renderer.getOutputPointer();

    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        saveFrame(("/tmp/out_color_" + to_string(batch_idx) + ".bmp").c_str(),
                  base_out_ptr + batch_idx * out_dim.x * out_dim.y * 4,
                  out_dim.x, out_dim.y, 4);
    }
}
