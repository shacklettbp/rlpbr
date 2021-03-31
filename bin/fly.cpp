#include <rlpbr.hpp>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <chrono>

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace std;
using namespace RLpbr;

const float mouse_speed = 2e-4;
const float movement_speed = 1.5;
const float rotate_speed = 1.25;

static GLFWwindow * makeWindow(const glm::u32vec2 &dim)
{
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 0);

    auto window = glfwCreateWindow(dim.x, dim.y, "RLPBR", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        cerr << "Failed to create window" << endl;
        abort();
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    return window;
}

struct CameraState {
    glm::vec3 eye;
    glm::vec3 look;
    glm::vec3 up;
};

static glm::i8vec3 key_movement(0, 0, 0);

void windowKeyHandler(GLFWwindow *window, int key, int, int action, int)
{
    if (action == GLFW_REPEAT) return;

    glm::i8vec3 cur_movement(0, 0, 0);
    switch (key) {
        case GLFW_KEY_ESCAPE: {
            if (action == GLFW_PRESS) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
            break;
        }
        case GLFW_KEY_ENTER: {
            if (action == GLFW_PRESS) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            }
            break;
        }
        case GLFW_KEY_W: {
            cur_movement.y += 1;
            break;
        }
        case GLFW_KEY_A: {
            cur_movement.x -= 1;
            break;
        }
        case GLFW_KEY_S: {
            cur_movement.y -= 1;
            break;
        }
        case GLFW_KEY_D: {
            cur_movement.x += 1;
            break;
        }
        case GLFW_KEY_Q: {
            cur_movement.z -= 1;
            break;
        }
        case GLFW_KEY_E: {
            cur_movement.z += 1;
            break;
        }
    }

    if (action == GLFW_PRESS) {
        key_movement += cur_movement;
    } else {
        key_movement -= cur_movement;
    }
}

static glm::vec2 cursorPosition(GLFWwindow *window)
{
    double mouse_x, mouse_y;
    glfwGetCursorPos(window, &mouse_x, &mouse_y);

    return glm::vec2(mouse_x, -mouse_y);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << argv[0] << " scene [spp] [depth] [verbose]" << endl;
        exit(EXIT_FAILURE);
    }

    if (!glfwInit()) {
        cerr << "GLFW failed to initialize" << endl;
        exit(EXIT_FAILURE);
    }

    glm::u32vec2 img_dims(1920, 1080);

    GLFWwindow *window = makeWindow(img_dims);
    if (glewInit() != GLEW_OK) {
        cerr << "GLEW failed to initialize" << endl;
        exit(EXIT_FAILURE);
    }

    array<GLuint, 2> read_fbos;
    glCreateFramebuffers(2, read_fbos.data());

    array<GLuint, 2> render_textures;
    glCreateTextures(GL_TEXTURE_2D, 2, render_textures.data());

    uint32_t spp = 1;
    uint32_t depth = 1;

    if (argc > 2) {
        spp = atoi(argv[2]);
    }

    if (argc > 3) {
        depth = atoi(argv[3]);
    }

    bool show_camera = false;
    if (argc > 4) {
        if (!strcmp(argv[4], "--cam")) {
            show_camera = true;
        }
    }

    Renderer renderer({0, 1, 1, img_dims.x, img_dims.y, spp, depth, true, false,
                       BackendSelect::Optix});

    array<cudaStream_t, 2> copy_streams;

    array<cudaGraphicsResource_t, 2> dst_imgs;
    array<void *, 2> cuda_intermediates;

    cudaError_t res;
    for (int i = 0; i < 2; i++) {
        glTextureStorage2D(render_textures[i], 1, GL_RGBA16F, img_dims.x, img_dims.y);

        res = cudaStreamCreate(&copy_streams[i]);
        if (res != cudaSuccess) {
            cerr << "CUDA stream initialization failed" << endl;
            abort();
        }

        res = cudaGraphicsGLRegisterImage(&dst_imgs[i], render_textures[i],
            GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
        if (res != cudaSuccess) {
            cerr << "Failed to map texture into CUDA" << endl;
            abort();
        }

        res = cudaMalloc(&cuda_intermediates[i], sizeof(half) * img_dims.x * img_dims.y * 4);
        if (res != cudaSuccess) {
            cerr << "Cuda intermediate buffer allocation failed" << endl;
        }
    }

    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);

    CameraState cam {
        glm::vec3(0, 0, 0),
        glm::vec3(0, 0, 1),
        glm::vec3(0, 1, 0)
    };
    glm::vec2 mouse_prev = cursorPosition(window);

    vector<Environment> envs;
    envs.emplace_back(
        renderer.makeEnvironment(scene, cam.eye, cam.look, cam.up, 60.f));
    envs.back().addLight(glm::vec3(-1.950218, 1.623819, 0.863453), glm::vec3(10.f));
    envs.back().addLight(glm::vec3(1.762336, 1.211801, -4.574429), glm::vec3(10.f));
    envs.back().addLight(glm::vec3(8.107919, 1.345027, -1.867001), glm::vec3(10.f));
    envs.back().addLight(glm::vec3(12.499360, 2.102839, 1.691340), glm::vec3(10.f));
    envs.back().addLight(glm::vec3(1.f, 10.f, 1.f), glm::vec3(10.f));
    envs.back().addLight(glm::vec3(-1.f, 10.f, -1.f), glm::vec3(10.f));

    glfwSetKeyCallback(window, windowKeyHandler);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    if (glfwRawMouseMotionSupported()) {
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }

    uint32_t prev_frame = renderer.render(envs.data());

    auto time_prev = chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window)) {
        auto time_cur = chrono::steady_clock::now();
        chrono::duration<float> elapsed_duration = time_cur - time_prev;
        time_prev = time_cur;
        float time_delta = elapsed_duration.count();

        glfwPollEvents();
        glm::vec2 mouse_delta; 
        if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {
            glm::vec2 mouse_cur = cursorPosition(window);
            mouse_delta = mouse_cur - mouse_prev;
            mouse_prev = mouse_cur;
        } else {
            mouse_delta = glm::vec2(0, 0);
            mouse_prev = cursorPosition(window);
        }

        glm::vec3 to_look = cam.look - cam.eye;
        glm::vec3 right = glm::cross(to_look, cam.up);
        glm::mat3 around_right(glm::angleAxis(mouse_delta.y * mouse_speed,
                                              right));

        cam.up = around_right * cam.up;

        glm::mat3 around_up(glm::angleAxis(-mouse_delta.x * mouse_speed,
                                           cam.up));

        to_look = around_up * around_right * to_look;

        glm::mat3 around_look(glm::angleAxis(
            float(key_movement.z) * rotate_speed * time_delta, to_look));
        cam.up = around_look * cam.up;
        right = around_look * around_up * right;

        glm::vec2 movement = movement_speed * time_delta *
            glm::vec2(key_movement.x, key_movement.y);
        cam.eye += right * movement.x + to_look * movement.y;

        cam.look = cam.eye + to_look;

        envs[0].setCameraView(cam.eye, cam.look, cam.up);
        if (show_camera) {
            cout << "E: " << glm::to_string(cam.eye) << "\n"
                 << "L: " << glm::to_string(cam.look) << "\n"
                 << "U: " << glm::to_string(cam.up) << "\n";
        }

        uint32_t new_frame = renderer.render(envs.data());
        renderer.waitForFrame(prev_frame);

        half *output = renderer.getOutputPointer(prev_frame);

        glNamedFramebufferTexture(read_fbos[prev_frame], GL_COLOR_ATTACHMENT0,
                                  0, 0);

        res = cudaGraphicsMapResources(1, &dst_imgs[prev_frame],
                                       copy_streams[prev_frame]);
        if (res != cudaSuccess) {
            cerr << "Failed to map opengl resource" << endl;
            abort();
        }

        cudaArray_t dst_arr;
        res = cudaGraphicsSubResourceGetMappedArray(&dst_arr,
            dst_imgs[prev_frame], 0, 0);
        if (res != cudaSuccess) {
           cerr << "Failed to get cuda array from opengl" << endl;
            abort();
        }

        res = cudaMemcpy2DAsync(cuda_intermediates[prev_frame],
            sizeof(half) * 4, output, sizeof(half) * 3, sizeof(half) * 3,
            img_dims.x * img_dims.y, cudaMemcpyDeviceToDevice,
            copy_streams[prev_frame]);

        if (res != cudaSuccess) {
            cerr << "buffer to intermediate buffer copy failed " << endl;
        }

        res = cudaMemcpy2DToArrayAsync(dst_arr, 0, 0,
            cuda_intermediates[prev_frame],
            img_dims.x * sizeof(half) * 4, img_dims.x * sizeof(half) * 4,
            img_dims.y, cudaMemcpyDeviceToDevice,
            copy_streams[prev_frame]);

        if (res != cudaSuccess) {
            cerr << "buffer to image copy failed " << endl;
        }

        // Seems like it shouldn't be necessary but bad tearing otherwise
        cudaStreamSynchronize(copy_streams[prev_frame]);

        res = cudaGraphicsUnmapResources(1, &dst_imgs[prev_frame],
                                         copy_streams[prev_frame]);
        if (res != cudaSuccess) {
            cerr << "Failed to unmap opengl resource" << endl;
            abort();
        }

        glNamedFramebufferTexture(read_fbos[prev_frame], GL_COLOR_ATTACHMENT0,
                                  render_textures[prev_frame], 0);

        glBlitNamedFramebuffer(read_fbos[prev_frame], 0,
                               0, img_dims.y, img_dims.x, 0,
                               0, 0, img_dims.x, img_dims.y,
                               GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glfwSwapBuffers(window);

        prev_frame = new_frame;
    }

    cudaFree(cuda_intermediates[0]);
    cudaFree(cuda_intermediates[1]);

    cudaGraphicsUnregisterResource(dst_imgs[0]);
    cudaGraphicsUnregisterResource(dst_imgs[1]);

    cudaStreamDestroy(copy_streams[0]);
    cudaStreamDestroy(copy_streams[1]);

    glDeleteTextures(2, render_textures.data());
    glDeleteFramebuffers(2, read_fbos.data());

    glfwDestroyWindow(window);
}
