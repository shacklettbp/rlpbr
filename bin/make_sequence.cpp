#include <rlpbr.hpp>
#include <iostream>
#include <fstream>
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
    //v *= 0.6;

    //float A = 2.51f;
    //float B = 0.03f;
    //float C = 2.43f;
    //float D = 0.59f;
    //float E = 0.14f;

    //v = clamp((v*(A*v+B))/(v*(C*v+D)+E), 0.f, 1.f);
    //return v;
    
    return v * (1.f + (v / 0.15f)) / (1.f + v);
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
    if (argc < 5) {
        cerr << argv[0] << "scene cam_pos spp depth" << endl;
        exit(EXIT_FAILURE);
    }

    ifstream cam_file(argv[2], ios::binary);
    if (!cam_file.is_open()) {
        cerr << "Failed to open " << argv[2] << endl;
        exit(EXIT_FAILURE);
    }

    vector<glm::vec3> positions;
    vector<glm::quat> rotations;
    while (!cam_file.eof()) {
        glm::vec3 pos;
        glm::quat rotation;
        cam_file.read((char *)&pos, sizeof(glm::vec3));
        cam_file.read((char *)&rotation, sizeof(glm::quat));

        positions.push_back(pos);
        rotations.push_back(rotation);
    }

    {
        vector<glm::vec3> new_positions;
        vector<glm::quat> new_rotations;
        for (int i = 0; i < (int)positions.size() - 1; i++) {
            auto pos = positions[i];
            auto rot = rotations[i];

            auto pos_next = positions[i + 1];
            auto rot_next = rotations[i + 1];

            if (rot != rot_next) {
                int rot_substeps = 10;
                float rot_delta = 1.f / (float)rot_substeps;
                for (int j = 0; j < rot_substeps; j++) {
                    float v = rot_delta * j;
                    new_positions.push_back(pos);
                    new_rotations.push_back(rot * (1.f - v) + rot_next * v);
                }
            } else if (pos != pos_next) {
                int pos_substeps = 10;
                float pos_delta = 1.f / (float)pos_substeps;

                for (int j = 0; j < pos_substeps; j++) {
                    float v = pos_delta * j;
                    new_positions.push_back(pos * (1.f - v) + pos_next * v);
                    new_rotations.push_back(rot);
                }
            } else {
                new_positions.push_back(pos);
                new_rotations.push_back(rot);
            }
        }
        new_positions.push_back(positions.back());
        new_rotations.push_back(rotations.back());

        positions = move(new_positions);
        rotations = move(new_rotations);
    }

    //glm::u32vec2 out_dim(128, 128);
    glm::u32vec2 out_dim(1920, 1080);

    uint32_t spp = 1;

    if (argc > 3) {
        spp = atoi(argv[3]);
    }

    uint32_t depth = 1;
    if (argc > 4) {
        depth = atoi(argv[4]);
    }

    Renderer renderer({0, 1, 1, out_dim.x, out_dim.y, spp, depth,
                       0, RenderMode::PathTracer, {}, 0.f, BackendSelect::Vulkan});

    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);

    RenderBatch batch = renderer.makeRenderBatch();

    batch.initEnvironment(0,
        renderer.makeEnvironment(scene, glm::vec3(0), glm::vec3(0, 0, 1),
                                 glm::vec3(0, 1, 0), 60.f));
    
    for (int i = 0; i < (int)positions.size(); i++) {
        glm::vec3 pos = positions[i];
        glm::quat rotation = rotations[i];

        glm::vec3 UP_VECTOR(0.f, 1.f, 0.f);
        glm::vec3 FWD_VECTOR(0.f, 0.f, 1.f);
        glm::vec3 RIGHT_VECTOR(-1.f, 0.f, 0.f);

        glm::vec3 up = rotation * UP_VECTOR;
        glm::vec3 fwd = rotation * FWD_VECTOR;
        glm::vec3 right = rotation * RIGHT_VECTOR;

        batch.getEnvironment(0).setCameraView(pos, fwd, up, right);
        renderer.render(batch);
        renderer.waitForBatch(batch);

        half *base_out_ptr = renderer.getOutputPointer(batch);

        uint32_t batch_idx = 0;
        saveFrame(("/tmp/out_color_" + to_string(i) + ".bmp").c_str(),
            base_out_ptr + batch_idx * out_dim.x * out_dim.y * 4,
            out_dim.x, out_dim.y, 4);
    }
}
