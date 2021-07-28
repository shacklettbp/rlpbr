#pragma once

#include "renderer.hpp"
#include "utils.hpp"
#include "navmesh.hpp"
#include "json.hpp"

namespace RLpbr {
namespace editor {

struct EpisodeConfig {
    uint32_t numVisualized = 10;
    uint32_t offset = 0;
};

struct Episode {
    glm::vec3 startPos;
    glm::vec3 endPos;

    uint32_t indexOffset;
    uint32_t numIndices;
};

struct EditorEpisodes {
    std::vector<Episode> episodes;
    std::vector<OverlayVertex> vertices;
    std::vector<uint32_t> indices;
};

struct EditorScene {
    std::string scenePath;
    std::shared_ptr<Scene> hdl;

    std::vector<char> cpuData;
    PackedVertex *verts;
    uint32_t *indices;
    AABB bbox;
    uint32_t totalTriangles;

    EditorCam cam;
    Renderer::FrameConfig renderCfg;
    NavmeshConfig navmeshCfg;
    std::optional<Navmesh> navmesh;
    bool navmeshBuilding;

    std::optional<EditorEpisodes> episodes;
    EpisodeConfig episodeCfg;
};

class Editor {
public:
    Editor(uint32_t gpu_id, uint32_t img_width, uint32_t img_height);

    void loadScene(const char *scene_name);

    void loop();

private:
    void startFrame();
    void render(EditorScene &scene, float frame_duration);

    Renderer renderer_;
    uint32_t cur_scene_idx_;
    std::vector<EditorScene> scenes_;
};

}
}
