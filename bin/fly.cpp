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

const float mouse_speed = 1e-4;
const float movement_speed = 1;
const float rotate_speed = 0.5;

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
            cur_movement.z += 1;
            break;
        }
        case GLFW_KEY_E: {
            cur_movement.z -= 1;
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

    return glm::vec2(mouse_x, mouse_y);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << argv[0] << " scene" << endl;
        exit(EXIT_FAILURE);
    }

    if (!glfwInit()) {
        cerr << "GLFW failed to initialize" << endl;
        exit(EXIT_FAILURE);
    }

    glm::u32vec2 img_dims(1024, 1024);

    GLFWwindow *window = makeWindow(img_dims);
    if (glewInit() != GLEW_OK) {
        cerr << "GLEW failed to initialize" << endl;
        exit(EXIT_FAILURE);
    }

    GLuint read_fbo;
    glCreateFramebuffers(1, &read_fbo);

    GLuint render_texture;
    glCreateTextures(GL_TEXTURE_2D, 1, &render_texture);
    glTextureStorage2D(render_texture, 1, GL_R32F, img_dims.x, img_dims.y);

    Renderer renderer({0, 1, 1, img_dims.x, img_dims.y, true,
                       BackendSelect::Optix});

    cudaStream_t copy_stream;
    auto res =  cudaStreamCreate(&copy_stream);
    if (res != cudaSuccess) {
        cerr << "CUDA stream initialization failed" << endl;
        abort();
    }

    cudaGraphicsResource_t dst_img;
    res = cudaGraphicsGLRegisterImage(&dst_img, render_texture,
        GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);

    if (res != cudaSuccess) {
        cerr << "Failed to map renderbuffer into CUDA" << endl;
        abort();
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
        renderer.makeEnvironment(scene, cam.eye, cam.look, cam.up));

    envs[0].addLight(glm::vec3(5, 6, 0), glm::vec3(1, 2, 2));
    envs[0].addLight(glm::vec3(4, 6, -4), glm::vec3(2, 2, 1));

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

        uint32_t new_frame = renderer.render(envs.data());
        renderer.waitForFrame(prev_frame);

        float *output = renderer.getOutputPointer(prev_frame);

        glNamedFramebufferTexture(read_fbo, GL_COLOR_ATTACHMENT0, 0, 0);

        res = cudaGraphicsMapResources(1, &dst_img, copy_stream);
        if (res != cudaSuccess) {
            cerr << "Failed to map opengl resource" << endl;
            abort();
        }

        cudaArray_t dst_arr;
        res = cudaGraphicsSubResourceGetMappedArray(&dst_arr, dst_img, 0, 0);
        if (res != cudaSuccess) {
            cerr << "Failed to get cuda array from opengl" << endl;
            abort();
        }

        res = cudaMemcpy2DToArrayAsync(dst_arr, 0, 0, output, img_dims.x * sizeof(float),
            img_dims.x * sizeof(float), img_dims.y, cudaMemcpyDeviceToDevice, copy_stream);
        if (res != cudaSuccess) {
            cerr << "buffer to image copy failed " << endl;
        }

        res = cudaGraphicsUnmapResources(1, &dst_img, copy_stream);
        if (res != cudaSuccess) {
            cerr << "Failed to unmap opengl resource" << endl;
            abort();
        }

        glNamedFramebufferTexture(read_fbo, GL_COLOR_ATTACHMENT0,
                                  render_texture, 0);

        glBlitNamedFramebuffer(read_fbo, 0,
                               0, 0, img_dims.x, img_dims.y,
                               0, 0, img_dims.x, img_dims.y,
                               GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glfwSwapBuffers(window);

        prev_frame = new_frame;
    }

    cudaGraphicsUnregisterResource(dst_img);
}
