#include "editor.hpp"
#include "navmesh.hpp"
#include "file_select.hpp"

#include <rlpbr/environment.hpp>
#include "rlpbr_core/utils.hpp"
#include "rlpbr_core/scene.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include "imgui_extensions.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <thread>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

using namespace std;

namespace RLpbr {
namespace editor {

namespace InternalConfig {
inline constexpr float cameraMoveSpeed = 5.f;
inline constexpr float mouseSpeed = 2e-4f;

inline constexpr auto nsPerFrame = chrono::nanoseconds(8333333);
inline constexpr auto nsPerFrameLongWait =
    chrono::nanoseconds(7000000);
inline constexpr float secondsPerFrame =
    chrono::duration<float>(nsPerFrame).count();
}

struct SceneProperties {
    AABB aabb;
    uint32_t totalTriangles; // Post transform
};

static SceneProperties computeSceneProperties(
    const PackedVertex *vertices,
    const uint32_t *indices,
    const std::vector<ObjectInfo> &objects,
    const std::vector<MeshInfo> &meshes,
    const std::vector<ObjectInstance> &instances,
    const std::vector<InstanceTransform> &transforms)
{
    AABB bounds {
        glm::vec3(INFINITY, INFINITY, INFINITY),
        glm::vec3(-INFINITY, -INFINITY, -INFINITY),
    };

    auto updateBounds = [&bounds](const glm::vec3 &point) {
        bounds.pMin = glm::min(bounds.pMin, point);
        bounds.pMax = glm::max(bounds.pMax, point);
    };

    uint32_t total_triangles = 0;
    for (int inst_idx = 0; inst_idx < (int)instances.size(); inst_idx++) {
        const ObjectInstance &inst = instances[inst_idx];
        const InstanceTransform &txfm = transforms[inst_idx];

        const ObjectInfo &obj = objects[inst.objectIndex];
        for (int mesh_offset = 0; mesh_offset < (int)obj.numMeshes;
             mesh_offset++) {
            uint32_t mesh_idx = obj.meshIndex + mesh_offset;
            const MeshInfo &mesh = meshes[mesh_idx];

            for (int tri_idx = 0; tri_idx < (int)mesh.numTriangles; tri_idx++) {
                uint32_t base_idx = tri_idx * 3 + mesh.indexOffset;

                glm::u32vec3 tri_indices(indices[base_idx],
                                         indices[base_idx + 1],
                                         indices[base_idx + 2]);

                auto a = vertices[tri_indices.x].position;
                auto b = vertices[tri_indices.y].position;
                auto c = vertices[tri_indices.z].position;

                a = txfm.mat * glm::vec4(a, 1.f);
                b = txfm.mat * glm::vec4(b, 1.f);
                c = txfm.mat * glm::vec4(c, 1.f);

                updateBounds(a);
                updateBounds(b);
                updateBounds(c);

                total_triangles++;
            }
        }
    }

    return {
        bounds,
        total_triangles,
    };
}

void Editor::loadScene(const char *scene_name)
{
    SceneLoadData load_data = SceneLoadData::loadFromDisk(scene_name, true);
    const vector<char> &loaded_gpu_data = *get_if<vector<char>>(&load_data.data);
    vector<char> cpu_data(loaded_gpu_data);

    PackedVertex *verts = (PackedVertex *)cpu_data.data();
    assert((uintptr_t)verts % std::alignment_of_v<PackedVertex> == 0);
    uint32_t *indices =
        (uint32_t *)(cpu_data.data() + load_data.hdr.indexOffset);

    auto render_data = renderer_.loadScene(move(load_data));

    auto [scene_aabb, total_triangles] = computeSceneProperties(verts, indices,
        render_data->objectInfo, render_data->meshInfo,
        render_data->envInit.defaultInstances,
        render_data->envInit.defaultTransforms);


    EditorCam default_cam;
    default_cam.position = glm::vec3(0.f, 10.f, 0.f);
    default_cam.view = glm::vec3(0.f, -1.f, 0.f);
    default_cam.up = glm::vec3(0.f, 0.f, 1.f);
    default_cam.right = glm::cross(default_cam.view, default_cam.up);

    scenes_.emplace_back(EditorScene {
        string(scene_name),
        move(render_data),
        move(cpu_data),
        verts,
        indices,
        scene_aabb,
        total_triangles,
        default_cam,
        Renderer::FrameConfig(),
        NavmeshConfig {
            scene_aabb,
        },
        optional<Navmesh>(),
        false,
        optional<EditorEpisodes>(),
        EpisodeConfig(),
        {},
        -1,
    });
}

static void updateNavmeshSettings(NavmeshConfig *cfg,
                                  const AABB &full_bbox,
                                  bool build_inprogress,
                                  bool has_navmesh,
                                  bool *build_navmesh,
                                  bool *save_navmesh,
                                  bool *load_navmesh)
{
    ImGui::Begin("Navmesh Build", nullptr, ImGuiWindowFlags_None);
    ImGui::PushItemWidth(ImGui::GetFontSize() * 10.f);

    ImGui::TextUnformatted("Agent Settings:");
    ImGui::InputFloat("Agent Height", &cfg->agentHeight);
    ImGui::InputFloat("Agent Radius", &cfg->agentRadius);
    ImGui::InputFloat("Max Slope", &cfg->maxSlope);
    ImGui::InputFloat("Max Climb", &cfg->agentMaxClimb);

    ImGui::TextUnformatted("Voxelization Settings:");
    ImGui::InputFloat("Cell Size", &cfg->cellSize);
    ImGui::InputFloat("Cell Height", &cfg->cellHeight);

    ImGui::TextUnformatted("Meshification Settings:");
    ImGui::InputFloat("Max Edge Length", &cfg->maxEdgeLen);
    ImGui::InputFloat("Max Edge Error", &cfg->maxError);
    ImGui::InputFloat("Minimum Region Size", &cfg->regionMinSize);
    ImGui::InputFloat("Region Merge Size", &cfg->regionMergeSize);
    ImGui::InputFloat("Detail Sampling Distance", &cfg->detailSampleDist);
    ImGui::InputFloat("Detail Sampling Max Error", &cfg->detailSampleMaxError);
    ImGui::PopItemWidth();

    ImGui::PushItemWidth(ImGui::GetFontSize() * 15.f);
    glm::vec3 speed(0.05f);
    ImGuiEXT::DragFloat3SeparateRange("Bounding box min",
        glm::value_ptr(cfg->bbox.pMin),
        glm::value_ptr(speed),
        glm::value_ptr(full_bbox.pMin),
        glm::value_ptr(full_bbox.pMax),
        "%.2f",
        ImGuiSliderFlags_AlwaysClamp);

    ImGuiEXT::DragFloat3SeparateRange("Bounding box max",
        glm::value_ptr(cfg->bbox.pMax),
        glm::value_ptr(speed),
        glm::value_ptr(full_bbox.pMin),
        glm::value_ptr(full_bbox.pMax),
        "%.2f",
        ImGuiSliderFlags_AlwaysClamp);
    ImGui::PopItemWidth();

    bool reset_settings = false;
    if (!build_inprogress) {
        *build_navmesh = ImGui::Button("Build Navmesh");
        ImGui::SameLine();
        reset_settings = ImGui::Button("Reset to Defaults");

        ImGuiEXT::PushDisabled(!has_navmesh);
        *save_navmesh = ImGui::Button("Save Navmesh");
        ImGuiEXT::PopDisabled();
        ImGui::SameLine();
        *load_navmesh = ImGui::Button("Load Navmesh");
    } else {
        ImGui::TextUnformatted("Navmesh building...");
        *build_navmesh = false;
        *save_navmesh = false;
        *load_navmesh = false;
    }

    if (reset_settings) {
        *cfg = NavmeshConfig {
            full_bbox,
        };
    }

    ImGui::End();
}

static void handleCamera(GLFWwindow *window, EditorScene &scene)
{
    auto keyPressed = [&](uint32_t key) {
        return glfwGetKey(window, key) == GLFW_PRESS;
    };

    glm::vec3 translate(0.f);

    auto cursorPosition = [window]() {
        double mouse_x, mouse_y;
        glfwGetCursorPos(window, &mouse_x, &mouse_y);

        return glm::vec2(mouse_x, -mouse_y);
    };


    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        glm::vec2 mouse_cur = cursorPosition();
        glm::vec2 mouse_delta = mouse_cur - scene.cam.mousePrev;

        auto around_right = glm::angleAxis(
            mouse_delta.y * InternalConfig::mouseSpeed, scene.cam.right);

        auto around_up = glm::angleAxis(
            -mouse_delta.x * InternalConfig::mouseSpeed, glm::vec3(0, 1, 0));

        auto rotation = around_up * around_right;

        scene.cam.up = rotation * scene.cam.up;
        scene.cam.view = rotation * scene.cam.view;
        scene.cam.right = rotation * scene.cam.right;

        if (keyPressed(GLFW_KEY_W)) {
            translate += scene.cam.view;
        }

        if (keyPressed(GLFW_KEY_A)) {
            translate -= scene.cam.right;
        }

        if (keyPressed(GLFW_KEY_S)) {
            translate -= scene.cam.view;
        }

        if (keyPressed(GLFW_KEY_D)) {
            translate += scene.cam.right;
        }

        scene.cam.mousePrev = mouse_cur;
    } else {
        if (keyPressed(GLFW_KEY_W)) {
            translate += scene.cam.up;
        }

        if (keyPressed(GLFW_KEY_A)) {
            translate -= scene.cam.right;
        }

        if (keyPressed(GLFW_KEY_S)) {
            translate -= scene.cam.up;
        }

        if (keyPressed(GLFW_KEY_D)) {
            translate += scene.cam.right;
        }

        scene.cam.mousePrev = cursorPosition();
    }

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_RELEASE) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }

    scene.cam.position += translate * InternalConfig::cameraMoveSpeed *
        InternalConfig::secondsPerFrame;
}

static void handleNavmesh(EditorScene &scene)
{
    bool should_build, should_save, should_load;
    updateNavmeshSettings(&scene.navmeshCfg, scene.bbox,
                          scene.navmeshBuilding, scene.navmesh.has_value(),
                          &should_build, &should_save, &should_load);

    if (should_build) {
        const char *err_msg;
        scene.navmesh = buildNavmesh(scene.navmeshCfg,
                     scene.totalTriangles,
                     scene.verts,
                     scene.indices,
                     scene.hdl->objectInfo,
                     scene.hdl->meshInfo,
                     scene.hdl->envInit.defaultInstances,
                     scene.hdl->envInit.defaultTransforms,
                     &err_msg);

        if (scene.navmesh.has_value()) {
            cout << "Built navmesh with "
                 <<  scene.navmesh->renderData.triIndices.size() / 3
                 << " triangles" << endl;
        } else {
            cerr << err_msg << endl;
        }
    } else if (should_save) {
        assert(scene.navmesh.has_value());
        filesystem::path navmesh_path =
            filesystem::path(scene.scenePath).replace_extension("navmesh");

        saveNavmesh(navmesh_path.c_str(), *scene.navmesh);
    } else if (should_load) {
        // FIXME: make this path selectable
        filesystem::path navmesh_path =
            filesystem::path(scene.scenePath).replace_extension("navmesh");

        const char *err_msg;
        auto new_navmesh = loadNavmesh(navmesh_path.c_str(), &err_msg);
        if (!new_navmesh) {
            cerr << err_msg << endl;
        } else {
            scene.navmesh = move(new_navmesh);
            scene.navmeshCfg.bbox = scene.navmesh->bbox;
        }
    }
}

static glm::vec3 randomEpisodeColor(uint32_t idx) {
    auto rand = [](glm::vec2 co) {
        const float a  = 12.9898f;
        const float b  = 78.233f;
        const float c  = 43758.5453f;
        float dt = glm::dot(co, glm::vec2(a, b));
        float sn = fmodf(dt, 3.14);
        return glm::fract(glm::sin(sn) * c);
    };

    return glm::vec3(rand(glm::vec2(idx, idx)),
                rand(glm::vec2(idx + 1, idx)),
                rand(glm::vec2(idx, idx + 1)));
}

static optional<EditorEpisodes> loadEpisodes(const char *filename,
    simdjson::ondemand::parser &&json_parser,
    const Navmesh &navmesh)
{
    auto json = JSONReader::loadGZipped(filename, move(json_parser));
    if (!json.has_value()) {
        return optional<EditorEpisodes>();
    }

    auto &doc = json->getDocument();

    EditorEpisodes data;
    vector<Episode> episodes;
    // FIXME remove dependence on renderData here
    vector<glm::vec3> path_tmp(navmesh.renderData.triIndices.size() / 3);
    UniqueMallocPtr path_scratch(
        malloc(Navmesh::scratchBytesPerTri() * path_tmp.size()));

    for (auto episode_json : doc["episodes"]) {
        auto start_pos =
            JSONReader::parseVec3(episode_json, "start_position");
        auto end_pos =
            JSONReader::parseVec3(episode_json, "target_position");

        if (!start_pos.has_value() || !end_pos.has_value()) {
            cout << "Failed to get positions" << endl;
            return optional<EditorEpisodes>();
        }

        glm::vec3 color = randomEpisodeColor(data.episodes.size());
        glm::u8vec4 quantized = glm::u8vec4(glm::clamp(glm::round(255.f *
            color), 0.f, 255.f), 255);

        uint32_t path_len = navmesh.findPath(*start_pos, *end_pos,
            path_tmp.size(), path_tmp.data(), path_scratch.get());

        uint32_t idx_offset = data.indices.size();
        for (int i = 0; i < (int)path_len; i++) {
            data.vertices.push_back({
                path_tmp[i],
                quantized,
            });

            if (i != (int)path_len - 1) {
                data.indices.push_back(data.vertices.size() - 1);
                data.indices.push_back(data.vertices.size());
            }
        }

        data.episodes.push_back({
            *start_pos,
            *end_pos,
            idx_offset,
            (uint32_t)data.indices.size() - idx_offset,
        });
    }

    return move(data);
}

static void handleEpisodes(EditorScene &scene)
{
    ImGui::Begin("Episode Overlay");

    ImGuiEXT::PushDisabled(!scene.navmesh);
    ImGui::PushItemWidth(10.f * ImGui::GetFontSize());
    ImGuiEXT::PushDisabled(!scene.episodes.has_value());
    ImGui::DragInt("# Visualized", (int *)&scene.episodeCfg.numVisualized,
                   10.f, 0.f, scene.episodes->episodes.size(),
                   "%0.f", ImGuiSliderFlags_AlwaysClamp);

    ImGui::DragInt("Episode Offset", (int *)&scene.episodeCfg.offset,
        1.f, 0.f,
        scene.episodes->episodes.size() - scene.episodeCfg.numVisualized,
        "%0.f", ImGuiSliderFlags_AlwaysClamp);
    ImGuiEXT::PopDisabled();
    ImGui::PopItemWidth();

    bool error = false;
    if (ImGui::Button("Load Episode JSON") && scene.navmesh.has_value()) {
        const char *filename = fileDialog();
        if (filename != nullptr) {
            scene.episodes = loadEpisodes(filename,
                simdjson::ondemand::parser(),
                *scene.navmesh);
            if (!scene.episodes.has_value()) {
                error = true;
            }

            delete[] filename;
        }
    }
    ImGuiEXT::PopDisabled();

    if (error) {
        ImGui::OpenPopup("EpisodeError");
    }
    if (ImGui::BeginPopupModal("EpisodeError", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize |
                               ImGuiWindowFlags_NoTitleBar)) {
        ImGui::Text("Failed to load episodes");
        if (ImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    ImGui::End();
}

static void handleLights(EditorScene &scene)
{
    ImGui::Begin("Lights");
    ImGui::Separator();
    if (ImGui::Button("Add Light")) {
        const float area_offset = 0.25f;
        AreaLight new_light {
            {
                glm::vec3(-area_offset, 0, -area_offset),
                glm::vec3(-area_offset, 0, area_offset),
                glm::vec3(area_offset, 0, -area_offset),
                glm::vec3(area_offset, 0, area_offset),
            },
            glm::vec3(0),
        };

        if (scene.selectedLight == -1) {
            new_light.translate = scene.cam.position + 0.5f * scene.cam.view;
        } else {
            new_light.translate = scene.lights[scene.selectedLight].translate +
                glm::vec3(2.f * area_offset, 0, 2.f * area_offset);
        }

        scene.lights.push_back(new_light);
        scene.selectedLight = scene.lights.size() - 1;
    }

    ImGuiEXT::PushDisabled(scene.lights.size() <= 1);
    ImGui::PushItemWidth(4.f * ImGui::GetFontSize());

    ImGui::DragInt("Selected Light", &scene.selectedLight, 0.05f,
                   0, scene.lights.size() - 1,
                   "%d", ImGuiSliderFlags_AlwaysClamp);

    ImGui::PopItemWidth();
    ImGuiEXT::PopDisabled();

    ImGuiEXT::PushDisabled(scene.selectedLight == -1);

    AreaLight *cur_light = nullptr;
    if (scene.selectedLight != -1) {
        cur_light = &scene.lights[scene.selectedLight];
    }

    glm::vec3 fake(0);
    auto pos_ptr = glm::value_ptr(fake);
    if (cur_light) {
        pos_ptr = glm::value_ptr(cur_light->translate);
    }

    ImGui::PushItemWidth(ImGui::GetFontSize() * 15.f);
    glm::vec3 speed(0.01f);
    ImGuiEXT::DragFloat3SeparateRange("Position",
        pos_ptr,
        glm::value_ptr(speed),
        glm::value_ptr(scene.bbox.pMin),
        glm::value_ptr(scene.bbox.pMax),
        "%.3f",
        ImGuiSliderFlags_AlwaysClamp);

    ImGuiEXT::PopDisabled();


    bool save = ImGui::Button("Save Lights");
    ImGui::SameLine();
    bool load = ImGui::Button("Load Lights");

    if (save || load) {
        filesystem::path lights_path =
            filesystem::path(scene.scenePath).replace_extension("lights");
        if (save) {
            ofstream lights_out(lights_path, ios::binary);
            uint32_t num_lights = scene.lights.size();
            lights_out.write((char *)&num_lights, sizeof(uint32_t));
            for (const AreaLight &light : scene.lights) {
                lights_out.write((char *)&light, sizeof(AreaLight));
            }
        }

        if (load) {
            ifstream lights_in(lights_path, ios::binary);
            uint32_t num_lights;
            lights_in.read((char *)&num_lights, sizeof(uint32_t));
            scene.lights.clear();
            scene.lights.reserve(num_lights);
            for (int i = 0; i < (int)num_lights; i++) {
                AreaLight light;
                lights_in.read((char *)&light, sizeof(AreaLight));
                scene.lights.push_back(light);
            }
        }
    }

    ImGui::End();
}

static float throttleFPS(chrono::time_point<chrono::steady_clock> start) {
    using namespace chrono;
    using namespace chrono_literals;
    
    auto end = steady_clock::now();
    while (end - start <
           InternalConfig::nsPerFrameLongWait) {
        this_thread::sleep_for(1ms);
    
        end = steady_clock::now();
    }
    
    while (end - start < InternalConfig::nsPerFrame) {
        this_thread::yield();
    
        end = steady_clock::now();
    }

    return duration<float>(end - start).count();
}

void Editor::loop()
{
    auto window = renderer_.getWindow();

    float frame_duration = InternalConfig::secondsPerFrame;
    while (!glfwWindowShouldClose(window)) {
        EditorScene &scene = scenes_[cur_scene_idx_];

        auto start_time = chrono::steady_clock::now();

        startFrame();

        handleNavmesh(scene);
        handleEpisodes(scene);
        handleCamera(window, scene);
        handleLights(scene);
        render(scene, frame_duration);

        frame_duration = throttleFPS(start_time);
    }

    renderer_.waitForIdle();
}

void Editor::startFrame()
{
    renderer_.waitUntilFrameReady();

    glfwPollEvents();

    renderer_.startFrame();
    ImGui::NewFrame();
}

static void renderCFGUI(Renderer::FrameConfig &cfg,
                        EditorCam &cam)
{
    ImGui::Begin("Render Settings");

    ImGui::TextUnformatted("Camera");
    ImGui::Separator();

    auto side_size = ImGui::CalcTextSize(" Bottom " );
    side_size.y *= 1.4f;
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign,
                        ImVec2(0.5f, 0.f));

    if (ImGui::Button("Top", side_size)) {
        cam.position = glm::vec3(0.f, 10.f, 0.f);
        cam.view = glm::vec3(0, -1, 0.f);
        cam.up = glm::vec3(0, 0, 1.f);
        cam.right = glm::cross(cam.view, cam.up);
    }

    ImGui::SameLine();

    if (ImGui::Button("Left", side_size)) {
        cam.position = glm::vec3(-10.f, 0, 0);
        cam.view = glm::vec3(1, 0, 0);
        cam.up = glm::vec3(0, 1, 0);
        cam.right = glm::cross(cam.view, cam.up);
    }

    ImGui::SameLine();

    if (ImGui::Button("Right", side_size)) {
        cam.position = glm::vec3(10.f, 0, 0);
        cam.view = glm::vec3(-1, 0, 0);
        cam.up = glm::vec3(0, 1, 0);
        cam.right = glm::cross(cam.view, cam.up);
    }

    ImGui::SameLine();

    if (ImGui::Button("Bottom", side_size)) {
        cam.position = glm::vec3(0, -10, 0);
        cam.view = glm::vec3(0, 1, 0);
        cam.up = glm::vec3(0, 0, 1);
        cam.right = glm::cross(cam.view, cam.up);
    }

    ImGui::PopStyleVar();

    auto ortho_size = ImGui::CalcTextSize(" Orthographic ");
    ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign,
                        ImVec2(0.5f, 0.f));
    if (ImGui::Selectable("Perspective", cam.perspective, 0,
                          ortho_size)) {
        cam.perspective = true;
    }
    ImGui::SameLine();

    if (ImGui::Selectable("Orthographic", !cam.perspective, 0,
                          ortho_size)) {
        cam.perspective = false;
    }

    ImGui::SameLine();

    ImGui::PopStyleVar();

    ImGui::TextUnformatted("Projection");

    float digit_width = ImGui::CalcTextSize("0").x;
    ImGui::SetNextItemWidth(digit_width * 6);
    if (cam.perspective) {
        ImGui::DragFloat("FOV", &cam.fov, 1.f, 1.f, 179.f, "%.0f");
    } else {
        ImGui::DragFloat("View Size", &cam.orthoHeight,
                          0.5f, 0.f, 100.f, "%0.1f");
    }

    ImGui::NewLine();
    ImGui::TextUnformatted("General");
    ImGui::Separator();

    ImGui::SetNextItemWidth(digit_width * 6);
    ImGui::DragFloat("Line Width", &cfg.lineWidth, 0.1f, 1.f, 10.f,
                     "%0.1f");
    ImGui::Checkbox("Edges always visible", &cfg.linesNoDepthTest);

    ImGui::NewLine();
    ImGui::TextUnformatted("Navmesh Rendering");
    ImGui::Separator();

    ImGui::Checkbox("Show Navmesh", &cfg.showNavmesh);
    ImGui::Checkbox("No Edges", &cfg.navmeshNoEdges);
    ImGui::Checkbox("Edges only", &cfg.wireframeOnly);

#if 0
    static const char *navmesh_options[] = {
        "Wireframe",
        "Wireframe Overlay",
        "Filled",
    };

    if (ImGui::Button(
            navmesh_options[(uint32_t)cfg.navmeshStyle])) {
        ImGui::OpenPopup("navmesh_render_popup");
    }
    ImGui::SameLine();
    ImGui::TextUnformatted("Navmesh rendering style");

    if (ImGui::BeginPopup("navmesh_render_popup",
                          ImGuiWindowFlags_NoMove)) {
        for (int i = 0; i < IM_ARRAYSIZE(navmesh_options); i++) {
            if (ImGui::Selectable(navmesh_options[i],
                (uint32_t)cfg.navmeshStyle == (uint32_t)i)) {
                cfg.navmeshStyle = NavmeshStyle(i);
            }
        }
        ImGui::EndPopup();
    }
#endif

    ImGui::End();
}

static void fpsCounterUI(float frame_duration)
{
    auto viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkSize.x, 0.f),
                            0, ImVec2(1.f, 0.f));
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5f);
    ImGui::Begin("FPS Counter", nullptr,
                 ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoInputs |
                 ImGuiWindowFlags_NoTitleBar |
                 ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopStyleVar();
    ImGui::Text("%.3f ms per frame (%.1f FPS)",
                1000.f * frame_duration, 1.f / frame_duration);

    ImGui::End();
}

void Editor::render(EditorScene &scene, float frame_duration)
{
    renderCFGUI(scene.renderCfg, scene.cam);
    fpsCounterUI(frame_duration);

    ImGui::Render();

    uint32_t total_vertices = 0;
    uint32_t total_tri_indices = 0;
    uint32_t total_line_indices = 0;

    if (scene.renderCfg.showNavmesh && scene.navmesh.has_value()) {
        total_vertices += scene.navmesh->renderData.vertices.size();
        total_tri_indices += scene.navmesh->renderData.triIndices.size();

        if (!scene.renderCfg.navmeshNoEdges) {
            total_line_indices += scene.navmesh->renderData.boundaryLines.size();
            total_line_indices += scene.navmesh->renderData.internalLines.size();
        }
    }

    if (scene.episodes.has_value()) {
        const EditorEpisodes &episode_data = *scene.episodes;
        uint32_t episode_offset = scene.episodeCfg.offset;
        uint32_t num_episodes = scene.episodeCfg.numVisualized;
        total_vertices += episode_data.vertices.size();

        for (int i = 0; i < (int)num_episodes; i++) {
            uint32_t episode_idx = episode_offset + i;
            const Episode &episode = episode_data.episodes[episode_idx];

            total_line_indices += episode.numIndices;
        }
    }

    total_vertices += scene.lights.size() * 4;
    uint32_t total_light_indices = scene.lights.size() * 6;

    vector<OverlayVertex> tmp_verts(total_vertices);
    vector<uint32_t> tmp_indices(total_tri_indices + total_line_indices +
                                 total_light_indices);

    OverlayVertex *vert_ptr = tmp_verts.data();
    uint32_t *idx_ptr = tmp_indices.data();

    if (scene.renderCfg.showNavmesh && scene.navmesh.has_value()) {
        memcpy(vert_ptr, scene.navmesh->renderData.vertices.data(),
            sizeof(OverlayVertex) * scene.navmesh->renderData.vertices.size());

        vert_ptr += scene.navmesh->renderData.vertices.size();

        memcpy(idx_ptr, scene.navmesh->renderData.triIndices.data(),
               sizeof(uint32_t) * scene.navmesh->renderData.triIndices.size());
        idx_ptr += scene.navmesh->renderData.triIndices.size();

        if (!scene.renderCfg.navmeshNoEdges) {
            memcpy(idx_ptr, scene.navmesh->renderData.internalLines.data(),
                   sizeof(uint32_t) * scene.navmesh->renderData.internalLines.size());
            idx_ptr += scene.navmesh->renderData.internalLines.size();

            memcpy(idx_ptr, scene.navmesh->renderData.boundaryLines.data(),
                   sizeof(uint32_t) *scene.navmesh->renderData.boundaryLines.size());
            idx_ptr += scene.navmesh->renderData.boundaryLines.size();
        }
    }

    for (int i = 0; i < (int)scene.lights.size(); i++) {
        auto &light = scene.lights[i];
        bool is_selected = i == scene.selectedLight;
        uint32_t idx_offset = i * 4;

        for (int j = 0; j < 4; j++) {
            *vert_ptr++ = OverlayVertex {
                light.vertices[j] + light.translate,
                is_selected ?
                    glm::u8vec4(130, 130, 255, 255) :
                    glm::u8vec4(255, 255, 255, 255),
            };
        }

        *idx_ptr++ = idx_offset + 1;
        *idx_ptr++ = idx_offset + 3;
        *idx_ptr++ = idx_offset + 2;

        *idx_ptr++ = idx_offset + 1;
        *idx_ptr++ = idx_offset + 2;
        *idx_ptr++ = idx_offset;
    }

    if (scene.episodes.has_value()) {
        const EditorEpisodes &episode_data = *scene.episodes;
        uint32_t base_episode = scene.episodeCfg.offset;
        uint32_t num_episodes = scene.episodeCfg.numVisualized;
        uint32_t base_idx_offset = vert_ptr - tmp_verts.data();

        memcpy(vert_ptr, episode_data.vertices.data(),
               sizeof(OverlayVertex) * episode_data.vertices.size());
        vert_ptr += episode_data.vertices.size();

        for (int episode_offset = 0; episode_offset < (int)num_episodes;
             episode_offset++) {
            uint32_t episode_idx = episode_offset + base_episode;
            const Episode &episode = episode_data.episodes[episode_idx];

            for (int index_idx = 0; index_idx < (int)episode.numIndices;
                 index_idx++) {
                *idx_ptr =
                    episode_data.indices[episode.indexOffset + index_idx];
                if (*idx_ptr != ~0u) {
                    *idx_ptr += base_idx_offset;
                }

                idx_ptr++;
            }
        }
    }

    renderer_.render(scene.hdl.get(), scene.cam, scene.renderCfg,
                     tmp_verts.data(), tmp_indices.data(),
                     tmp_verts.size(), 
                     total_tri_indices, total_line_indices,
                     total_light_indices);
}

Editor::Editor(uint32_t gpu_id, uint32_t img_width, uint32_t img_height)
    : renderer_(gpu_id, img_width, img_height),
      cur_scene_idx_(0),
      scenes_()
{}

}
}

using namespace RLpbr;
using namespace RLpbr::editor;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "%s scene.bps\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    Editor editor(0, 3840, 2160);
    editor.loadScene(argv[1]);

    editor.loop();
}
