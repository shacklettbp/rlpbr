#include <rlpbr.hpp>
#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>

#include <glm/gtx/transform.hpp>

#include "oiio_bridge.hpp"

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
    v *= 0.8;

    float A = 2.51f;
    float B = 0.03f;
    float C = 2.43f;
    float D = 0.59f;
    float E = 0.14f;

    v = clamp((v*(A*v+B))/(v*(C*v+D)+E), 0.f, 1.f);
    return v;
    
    //return v * (1.f + (v / 0.15f)) / (1.f + v);
}

void saveFrame(string fprefix, const half *dev_ptr,
               uint32_t width, uint32_t height, uint32_t num_channels)
{
    auto buffer = copyToHost(dev_ptr, width, height, num_channels);

    uint32_t num_pixels = buffer.size() / num_channels;
    vector<uint8_t> sdr_buffer(num_pixels * 3);
    vector<half> hdr_buffer(num_pixels * 3);

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
            hdr_buffer[k + j] = buffer[i + j];
        }
    }

    saveSDR((fprefix + ".png").c_str(), width, height, sdr_buffer.data());
    saveHDR((fprefix + ".exr").c_str(), width, height, hdr_buffer.data(), true);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << argv[0] << "scene batch_size [spp] [depth]" << endl;
        exit(EXIT_FAILURE);
    }

    uint32_t batch_size = atoi(argv[2]);

    glm::u32vec2 out_dim(1920, 1080);
    //glm::u32vec2 out_dim(1920, 1080);

    uint32_t spp = 1;

    if (argc > 3) {
        spp = atoi(argv[3]);
    }

    uint32_t depth = 1;
    if (argc > 4) {
        depth = atoi(argv[4]);
    }

    Renderer renderer({0, 1, batch_size, out_dim.x, out_dim.y, spp, depth,
                       0, RenderMode::PathTracer, {}, 0.f, BackendSelect::Vulkan});

    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);

    //glm::vec3 eye (-2.206948, 1.599559, -5.020905);
    //glm::vec3 look(-2.287329, 1.336111, -5.982229);
    //glm::vec3 up(0.011677, 0.964129, -0.265193);

    //glm::vec3 eye(-5.805865, 1.189658, 0.623973);
    //glm::vec3 look(-6.281745, 1.006185, 1.484137);
    //glm::vec3 up(-0.116739, 0.982527, 0.144987);

    //glm::vec3 eye(-7.037555, 1.062551, 8.410369);
    //glm::vec3 look(-7.775107, 0.833322, 7.775162);
    //glm::vec3 up(-0.207893, 0.972026, -0.109381);

    // Vase
    //glm::vec3 eye(10.573854, 1.332727, -2.085712);
    //glm::vec3 look(11.193787, 1.303279, -1.301609);
    //glm::vec3 up(-0.018044, 0.998499, 0.051766);

    //glm::vec3 eye(3.032244, 1.832471, 4.601660);
    //glm::vec3 look(3.729160, 1.781724, 3.886291);
    //glm::vec3 up(0, 1, 0);

    //glm::vec3 eye(10.748793, 1.411647, -2.430134);
    //glm::vec3 look(11.293303, 1.406617, -1.591388);
    //glm::vec3 up(0, 1, 0);
    
    //glm::vec3 eye(-2.396987, 1.257544, 5.563847);
    //glm::vec3 look(-1.796038, 1.336225, 4.768434);
    //glm::vec3 up(-0.026884, 0.996575, 0.078269);

    // BUG : 103997730_171030885
    //glm::vec3 eye(4.519275, 1.042895, -3.621908);
    //glm::vec3 look(4.926752, 0.944727, -4.529839);
    //glm::vec3 up(0.034596, 0.995155, -0.092071);

#if 0
    glm::vec3 eye(-6.209174, 1.363060, 4.240532);
    glm::vec3 look(-6.396608, 1.146945, 3.282313);
    glm::vec3 up(-0.025357, 0.976241, -0.215219);
#endif

    glm::vec3 eye(4.744188, 2.347717, -1.483009);
    glm::vec3 look(4.023069, 2.075568, -2.120136);
    glm::vec3 up(-0.187018, 0.961949, -0.199220);


    //glm::vec3 eye(7.1, 1.673785, -0.217743);
    //glm::vec3 look(7.655676, 1.512004, -0.946787);
    //glm::vec3 up(0.174367, 0.982914, -0.059044);

    // Mirror table
    //glm::vec3 eye(-6.207120, 0.825648, 0.911869);
    //glm::vec3 look(-6.698653, 0.796494, 0.041498);
    //glm::vec3 up(-0.030218, 0.999409, -0.016411);

    // workout
    //glm::vec3 eye(-0.933040, 0.442653, -1.826718);
    //glm::vec3 look(-1.161190, 0.458737, -0.853224);
    //glm::vec3 up(-0.035431, 0.999065, -0.024810);

    glm::vec3 to_look = look - eye;
    
    RenderBatch batch = renderer.makeRenderBatch();

    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        glm::mat3 r = glm::rotate(glm::radians(10.f * batch_idx), up);
        batch.initEnvironment(batch_idx,
            renderer.makeEnvironment(scene, eye, eye + r * to_look, up, 60.f));
    }
    //envs.back().addLight(glm::vec3(-1.950218, 1.623819, 0.863453), glm::vec3(10.f));
    //envs.back().addLight(glm::vec3(1.762336, 1.211801, -4.574429), glm::vec3(10.f));
    //envs.back().addLight(glm::vec3(8.107919, 1.345027, -1.867001), glm::vec3(10.f));
    //envs.back().addLight(glm::vec3(12.499360, 2.102839, 1.691340), glm::vec3(10.f));

    renderer.render(batch);
    renderer.waitForBatch(batch);

    half *base_out_ptr = renderer.getOutputPointer(batch);

    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        saveFrame("/tmp/out_color_" + to_string(batch_idx),
                  base_out_ptr + batch_idx * out_dim.x * out_dim.y * 4,
                  out_dim.x, out_dim.y, 4);
    }
}
