#include "navmesh.hpp"
#include "rlpbr_core/utils.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <meshoptimizer.h>
#include <Recast.h>
#include <DetourNavMesh.h>
#include <DetourNavMeshBuilder.h>
#include <DetourNavMeshQuery.h>

#include <cstring>
#include <iostream>
#include <unordered_map>

using namespace std;

namespace RLpbr {
namespace editor {

enum PolyAreas {
    POLYAREA_GROUND,
    POLYAREA_DOOR,
};

enum PolyFlags {
    POLYFLAGS_WALK     = 0x01,  // Ability to walk (ground, grass, road)
    POLYFLAGS_SWIM     = 0x02,  // Ability to swim (water).
    POLYFLAGS_DOOR     = 0x04,  // Ability to move through doors.
    POLYFLAGS_JUMP     = 0x08,  // Ability to jump.
    POLYFLAGS_DISABLED = 0x10,  // Disabled polygon
    POLYFLAGS_ALL      = 0xffff // All abilities.
};

struct NavmeshInternal {
    dtNavMesh *detourMesh;
    dtNavMeshQuery *detourQuery;
};

void NavmeshDeleter::operator()(NavmeshInternal *ptr) const
{
    dtFreeNavMesh(ptr->detourMesh);
    dtFreeNavMeshQuery(ptr->detourQuery);
    delete ptr;
}

uint32_t Navmesh::scratchBytesPerTri()
{
    return sizeof(dtPolyRef);
}

uint32_t Navmesh::findPath(const glm::vec3 &start, const glm::vec3 &end,
                  uint32_t max_verts, glm::vec3 *verts,
                  void *scratch) const
{
    dtNavMeshQuery *query = internal->detourQuery;
    dtQueryFilter filter;
    filter.setIncludeFlags(POLYFLAGS_WALK);
    filter.setExcludeFlags(0);

    glm::vec3 extents(0.f);
    dtPolyRef start_ref;
    glm::vec3 nearest_start;
    dtStatus status = query->findNearestPoly(glm::value_ptr(start),
                                             glm::value_ptr(extents),
                                             &filter,
                                             &start_ref,
                                             glm::value_ptr(nearest_start));

    if (dtStatusFailed(status)) {
        return ~0u;
    }

    dtPolyRef end_ref;
    glm::vec3 nearest_end;
    status = query->findNearestPoly(glm::value_ptr(end),
                                    glm::value_ptr(extents),
                                    &filter,
                                    &end_ref,
                                    glm::value_ptr(nearest_end));

    if (dtStatusFailed(status)) {
        return ~0u;
    }

    dtPolyRef *scratch_polys = (dtPolyRef *)scratch;
    int scratch_path_size;
    status = query->findPath(start_ref, end_ref,
                             glm::value_ptr(nearest_start),
                             glm::value_ptr(nearest_end),
                             &filter,
                             scratch_polys,
                             &scratch_path_size,
                             max_verts);

    if (dtStatusFailed(status)) {
        return ~0u;
    }

    int path_size;
    status = query->findStraightPath(glm::value_ptr(nearest_start),
                                     glm::value_ptr(nearest_end),
                                     scratch_polys, scratch_path_size,
                                     &verts[0].x, nullptr, nullptr,
                                     &path_size, max_verts);

    if (dtStatusFailed(status)) {
        return ~0u;
    }

    return (uint32_t)path_size;
}

glm::vec3 Navmesh::getRandomPoint()
{
    constexpr int max_tries = 10;

    dtQueryFilter filter;
    filter.setIncludeFlags(POLYFLAGS_WALK);
    filter.setExcludeFlags(0);

    auto rand_cb = []() {
        return (float)rand() / float(RAND_MAX + 1u);
    };

    glm::vec3 result(0.f);
    int i;
    for (i = 0; i < max_tries; i++) {
        dtPolyRef ref = 0;
        dtStatus status =
            internal->detourQuery->findRandomPoint(&filter, rand_cb, &ref,
                                                  glm::value_ptr(result));

        if (dtStatusSucceed(status)) {
            break;
        }
    }

    if (i == max_tries) {
        cerr << "Failed to get random point on navmesh" << endl;
        abort();
    }

    return result;
}

static NavmeshRenderData buildRenderData(
    const vector<glm::vec3> &orig_vertices)
{
    vector<uint32_t> index_remap(orig_vertices.size());
    size_t new_vertex_count =
        meshopt_generateVertexRemap(index_remap.data(),
                                    nullptr, orig_vertices.size(),
                                    orig_vertices.data(),
                                    orig_vertices.size(),
                                    sizeof(glm::vec3));

    vector<uint32_t> new_indices(orig_vertices.size());
    vector<glm::vec3> new_vertices(new_vertex_count);

    meshopt_remapIndexBuffer(new_indices.data(), nullptr,
                             orig_vertices.size(), index_remap.data());

    meshopt_remapVertexBuffer(new_vertices.data(), orig_vertices.data(),
                              orig_vertices.size(), sizeof(glm::vec3),
                              index_remap.data());

    unordered_map<uint64_t, uint32_t> edge_map;
    for (int tri_idx = 0; tri_idx < (int)new_indices.size() / 3; tri_idx++) {
        glm::u32vec3 tri_indices {
            new_indices[3 * tri_idx],
            new_indices[3 * tri_idx + 1],
            new_indices[3 * tri_idx + 2],
        };

        auto edgePair = [&](uint32_t a, uint32_t b) {
            uint32_t a_idx = tri_indices[a];
            uint32_t b_idx = tri_indices[b];

            if (a_idx > b_idx) {
                swap(a_idx, b_idx);
            }

            return ((uint64_t)a_idx << 32) + b_idx;
        };

        uint64_t ab = edgePair(0, 1);
        uint64_t bc = edgePair(1, 2);
        uint64_t ca = edgePair(2, 0);

        edge_map[ab]++;
        edge_map[bc]++;
        edge_map[ca]++;
    }

    vector<uint32_t> boundary_indices;
    vector<uint32_t> internal_indices;

    for (const auto [edge, count] : edge_map) {
        uint32_t a = edge >> 32;
        uint32_t b = (uint32_t)edge;

        if (count > 1) {
            internal_indices.push_back(a);
            internal_indices.push_back(b);
        } else {
            boundary_indices.push_back(a);
            boundary_indices.push_back(b);
        }
    }

    static const glm::u8vec4 fill_color = glm::clamp(glm::round(255.f *
        glm::vec4(0.066f, 0.620f, 0.730f, 0.5f)), 0.f, 255.f);
    static const glm::u8vec4 boundary_color = glm::clamp(glm::round(255.f *
        glm::vec4(0.1f, 0.1f, 0.1f, 1.f)), 0.f, 255.f);
    static const glm::u8vec4 internal_color = glm::clamp(glm::round(255.f *
        glm::vec4(0.066f, 0.620f, 0.730f, 1.f)), 0.f, 255.f);

    vector<OverlayVertex> colored_verts;
    colored_verts.reserve(new_vertices.size());

    for (const glm::vec3 &pos : new_vertices) {
        colored_verts.push_back({
            pos,
            fill_color,
        });
    }

    for (uint32_t &idx : boundary_indices) {
        colored_verts.push_back({
            new_vertices[idx],
            boundary_color,
        });

        idx = colored_verts.size() - 1;
    }

    for (uint32_t &idx : internal_indices) {
        colored_verts.push_back({
            new_vertices[idx],
            internal_color,
        });

        idx = colored_verts.size() - 1;
    }

    return NavmeshRenderData {
        move(colored_verts),
        move(new_indices),
        move(boundary_indices),
        move(internal_indices),
    };
}

static vector<glm::vec3> collectNavmeshVertices(const dtNavMesh *dt_navmesh)
{
    vector<glm::vec3> navmesh_vertices;

     // Iterate over all tiles
    for (int tile_idx = 0; tile_idx < dt_navmesh->getMaxTiles(); tile_idx++) {
        const dtMeshTile* tile =
            const_cast<const dtNavMesh*>(dt_navmesh)->getTile(tile_idx);
        if (!tile)
            continue;

        // Iterate over all polygons in a tile
        for (int poly_idx = 0; poly_idx < tile->header->polyCount;
             poly_idx++) {
            // Get the polygon reference from the tile and polygon id
            dtPolyRef polyRef =
                dt_navmesh->encodePolyId(tile->salt, tile_idx, poly_idx);
            const dtPoly* poly = nullptr;
            const dtMeshTile* tmp = nullptr;
            dt_navmesh->getTileAndPolyByRefUnsafe(polyRef, &tmp, &poly);

            assert(poly != nullptr);
            assert(tmp != nullptr);

            const std::ptrdiff_t ip = poly - tile->polys;
            const dtPolyDetail* pd = &tile->detailMeshes[ip];

            for (int j = 0; j < pd->triCount; ++j) {
                const unsigned char* t =
                    &tile->detailTris[(pd->triBase + j) * 4];
                for (int k = 0; k < 3; ++k) {
                    glm::vec3 v;
                    float *src;
                    if (t[k] < poly->vertCount) {
                        src = &tile->verts[poly->verts[t[k]] * 3];
                    } else {
                        src = &tile->detailVerts[(pd->vertBase +
                            (t[k] - poly->vertCount)) * 3];
                    }
                    memcpy(glm::value_ptr(v), src, sizeof(float) * 3);
                    navmesh_vertices.push_back(v);
                }
            }
        }
    }

    return navmesh_vertices;
}

static bool initQuery(dtNavMeshQuery *dt_query, dtNavMesh *dt_navmesh)
{
    dtStatus status = dt_query->init(dt_navmesh, 2048);
    if (dtStatusFailed(status)) {
        return false;
    }

    return true;
}

optional<Navmesh> buildNavmesh(const NavmeshConfig &cfg,
                     uint32_t total_triangles,
                     const PackedVertex *vertices,
                     const uint32_t *indices,
                     const std::vector<ObjectInfo> &objects,
                     const std::vector<MeshInfo> &meshes,
                     const std::vector<ObjectInstance> &instances,
                     const std::vector<InstanceTransform> &transforms,
                     const char **err_msg)
{
    struct Intermediate {
        rcHeightfield *solid = nullptr;
        uint8_t *triareas = nullptr;
        rcCompactHeightfield *chf = nullptr;
        rcContourSet *cset = nullptr;
        rcPolyMesh *pmesh = nullptr;
        rcPolyMeshDetail *dmesh = nullptr;

        ~Intermediate()
        {
            rcFreeHeightField(solid);
            delete[] triareas;
            rcFreeCompactHeightfield(chf);
            rcFreeContourSet(cset);
            rcFreePolyMesh(pmesh);
            rcFreePolyMeshDetail(dmesh);
        }
    } inter;

    rcConfig rc_cfg = {};
    rc_cfg.cs = cfg.cellSize;
    rc_cfg.ch = cfg.cellHeight;
    rc_cfg.walkableSlopeAngle = cfg.maxSlope;
    rc_cfg.walkableHeight = (int)ceilf(cfg.agentHeight / cfg.cellHeight);
    rc_cfg.walkableClimb = (int)floorf(cfg.agentMaxClimb / cfg.cellHeight);
    rc_cfg.walkableRadius = (int)ceilf(cfg.agentRadius / cfg.cellSize);
    rc_cfg.maxEdgeLen = (int)(cfg.maxEdgeLen / cfg.cellSize);
    rc_cfg.maxSimplificationError = cfg.maxError;
    // Note: area = size*size
    rc_cfg.minRegionArea = (int)rcSqr(cfg.regionMinSize);
    // Note: area = size*size
    rc_cfg.mergeRegionArea = (int)rcSqr(cfg.regionMergeSize);
    rc_cfg.maxVertsPerPoly = 3;
    rc_cfg.detailSampleDist =
        cfg.detailSampleDist < 0.9f ? 0 : cfg.cellSize * cfg.detailSampleDist;
    rc_cfg.detailSampleMaxError = cfg.cellHeight * cfg.detailSampleMaxError;

    const float *min_ptr = glm::value_ptr(cfg.bbox.pMin);
    const float *max_ptr = glm::value_ptr(cfg.bbox.pMax);

    // Set the area where the navigation will be build.
    // Here the bounds of the input mesh are used, but the
    // area could be specified by an user defined box, etc.
    int rc_width, rc_height;
    rcCalcGridSize(min_ptr, max_ptr, cfg.cellSize, &rc_width, &rc_height);

    rcContext rc_ctx(false);

    //
    // Step 2. Rasterize input polygon soup.
    //

    // Allocate voxel heightfield where we rasterize our input data to.
    inter.solid = rcAllocHeightfield();
    if (!inter.solid)
    {
        *err_msg = "OOM while allocating heightfield";
        return optional<Navmesh>();
    }

    if (!rcCreateHeightfield(&rc_ctx, *inter.solid, rc_width, rc_height,
                             min_ptr, max_ptr, cfg.cellSize, cfg.cellHeight)) {
        *err_msg = "failed to create heightfield";
        return optional<Navmesh>();
    }

    // Allocate array that can hold triangle area types.
    // If you have multiple meshes you need to process, allocate
    // and array which can hold the max number of triangles you need to process.
    inter.triareas = new uint8_t[total_triangles];
    if (!inter.triareas)
    {
        *err_msg = "OOM for triangle area scratch space";
        return optional<Navmesh>();
    }

    // Find triangles which are walkable based on their slope and rasterize them.
    // If your input data is multiple meshes, you can transform them here, calculate
    // the are type for each of the meshes and rasterize them.
    memset(inter.triareas, 0, total_triangles*sizeof(unsigned char));

    uint32_t tri_offset = 0;
    vector<float> transformed_verts;
    vector<int> transformed_indices;

    for (int inst_idx = 0; inst_idx < (int)instances.size(); inst_idx++) {
        const ObjectInstance &inst = instances[inst_idx];
        const InstanceTransform &txfm = transforms[inst_idx];

        const ObjectInfo &obj = objects[inst.objectIndex];
        for (int mesh_offset = 0; mesh_offset < (int)obj.numMeshes;
             mesh_offset++) {
            uint32_t mesh_idx = obj.meshIndex + mesh_offset;
            const MeshInfo &mesh = meshes[mesh_idx];

            // FIXME: this just makes an unindexed mesh for now
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

                transformed_verts.push_back(a.x);
                transformed_verts.push_back(a.y);
                transformed_verts.push_back(a.z);
                transformed_indices.push_back(transformed_verts.size() / 3 - 1);
                transformed_verts.push_back(b.x);
                transformed_verts.push_back(b.y);
                transformed_verts.push_back(b.z);
                transformed_indices.push_back(transformed_verts.size() / 3 - 1);
                transformed_verts.push_back(c.x);
                transformed_verts.push_back(c.y);
                transformed_verts.push_back(c.z);
                transformed_indices.push_back(transformed_verts.size() / 3 - 1);
            }

            rcMarkWalkableTriangles(&rc_ctx, rc_cfg.walkableSlopeAngle,
                transformed_verts.data(), transformed_verts.size(),
                transformed_indices.data(), mesh.numTriangles,
                inter.triareas + tri_offset);

            if (!rcRasterizeTriangles(&rc_ctx, transformed_verts.data(),
                    transformed_verts.size(), transformed_indices.data(),
                    inter.triareas + tri_offset, mesh.numTriangles,
                    *inter.solid, rc_cfg.walkableClimb)) {
                *err_msg = "triangle rasterization failed";
                return optional<Navmesh>();
            }

            transformed_verts.clear();
            transformed_indices.clear();

            tri_offset += mesh.numTriangles;
        }
    }

    delete[] inter.triareas;
    inter.triareas = nullptr;

    //
    // Step 3. Filter walkables surfaces.
    //

    // Once all geoemtry is rasterized, we do initial pass of filtering to
    // remove unwanted overhangs caused by the conservative rasterization
    // as well as filter spans where the character cannot possibly stand.
    rcFilterLowHangingWalkableObstacles(&rc_ctx, rc_cfg.walkableClimb,
                                        *inter.solid);
    rcFilterLedgeSpans(&rc_ctx, rc_cfg.walkableHeight,
                       rc_cfg.walkableClimb, *inter.solid);
    rcFilterWalkableLowHeightSpans(&rc_ctx, rc_cfg.walkableHeight,
                                   *inter.solid);


    //
    // Step 4. Partition walkable surface to simple regions.
    //

    // Compact the heightfield so that it is faster to handle from now on.
    // This will result more cache coherent data as well as the neighbours
    // between walkable cells will be calculated.
    inter.chf = rcAllocCompactHeightfield();
    if (!inter.chf) {
        *err_msg = "OOM while allocating compact heightfield";
        return optional<Navmesh>();
    }

    if (!rcBuildCompactHeightfield(&rc_ctx, rc_cfg.walkableHeight,
            rc_cfg.walkableClimb, *inter.solid, *inter.chf)) {
        *err_msg = "failed to build compact heightfield";
        return optional<Navmesh>();
    }

    rcFreeHeightField(inter.solid);
    inter.solid = nullptr;

    // Erode the walkable area by agent radius.
    if (!rcErodeWalkableArea(&rc_ctx, rc_cfg.walkableRadius, *inter.chf)) {
        *err_msg = "failed to erode walkable area";
        return optional<Navmesh>();
    }

    if (!rcBuildDistanceField(&rc_ctx, *inter.chf)) {
        *err_msg = "failed to build distance field";
        return optional<Navmesh>();
    }

    if (!rcBuildRegions(&rc_ctx, *inter.chf, 0, rc_cfg.minRegionArea,
                        rc_cfg.mergeRegionArea)) {
        *err_msg = "failed to build watershed regions";
        return optional<Navmesh>();
    }

    //
    // Step 5. Trace and simplify region contours.
    //

    // Create contours.
    inter.cset = rcAllocContourSet();
    if (!inter.cset) {
        *err_msg = "OOM while allocating contour set";
        return optional<Navmesh>();
    }

    if (!rcBuildContours(&rc_ctx, *inter.chf, rc_cfg.maxSimplificationError,
                         rc_cfg.maxEdgeLen, *inter.cset)) {
        *err_msg = "failed to build contours";
        return optional<Navmesh>();
    }

    //
    // Step 6. Build polygons mesh from contours.
    //

    // Build polygon navmesh from the contours.
    inter.pmesh = rcAllocPolyMesh();
    if (!inter.pmesh) {
        *err_msg = "OOM while allocating polygon mesh";
        return optional<Navmesh>();
    }

    if (!rcBuildPolyMesh(&rc_ctx, *inter.cset, rc_cfg.maxVertsPerPoly,
                         *inter.pmesh)) {
        *err_msg = "failed to build polygon mesh";
        return optional<Navmesh>();
    }

    //
    // Step 7. Create detail mesh which allows to access approximate height on each polygon.
    //

    inter.dmesh = rcAllocPolyMeshDetail();
    if (!inter.dmesh) {
        *err_msg = "OOM while allocating detail mesh";
        return optional<Navmesh>();
    }

    if (!rcBuildPolyMeshDetail(&rc_ctx, *inter.pmesh, *inter.chf,
                               rc_cfg.detailSampleDist,
                               rc_cfg.detailSampleMaxError, *inter.dmesh)) {
        *err_msg = "failed to build detail mesh";
        return optional<Navmesh>();
    }

    rcFreeCompactHeightfield(inter.chf);
    inter.chf = nullptr;
    rcFreeContourSet(inter.cset);
    inter.cset = nullptr;

    unsigned char* nav_data = nullptr;
    int nav_data_size = 0;

    // Update poly flags from areas.
    for (int i = 0; i < inter.pmesh->npolys; ++i) {
      if (inter.pmesh->areas[i] == RC_WALKABLE_AREA) {
        inter.pmesh->areas[i] = POLYAREA_GROUND;
      }
      if (inter.pmesh->areas[i] == POLYAREA_GROUND) {
        inter.pmesh->flags[i] = POLYFLAGS_WALK;
      } else if (inter.pmesh->areas[i] == POLYAREA_DOOR) {
        inter.pmesh->flags[i] = POLYFLAGS_WALK | POLYFLAGS_DOOR;
      }
    }

    dtNavMeshCreateParams params {};
    memset(&params, 0, sizeof(params));
    params.verts = inter.pmesh->verts;
    params.vertCount = inter.pmesh->nverts;
    params.polys = inter.pmesh->polys;
    params.polyAreas = inter.pmesh->areas;
    params.polyFlags = inter.pmesh->flags;
    params.polyCount = inter.pmesh->npolys;
    params.nvp = inter.pmesh->nvp;
    params.detailMeshes = inter.dmesh->meshes;
    params.detailVerts = inter.dmesh->verts;
    params.detailVertsCount = inter.dmesh->nverts;
    params.detailTris = inter.dmesh->tris;
    params.detailTriCount = inter.dmesh->ntris;
    params.walkableHeight = cfg.agentHeight;
    params.walkableRadius = cfg.agentRadius;
    params.walkableClimb = cfg.agentMaxClimb;
    rcVcopy(params.bmin, inter.pmesh->bmin);
    rcVcopy(params.bmax, inter.pmesh->bmax);

    params.cs = rc_cfg.cs;
    params.ch = rc_cfg.ch;
    params.buildBvTree = true;

    if (!dtCreateNavMeshData(&params, &nav_data, &nav_data_size)) {
      *err_msg = "failed to create detour navmesh data";
      return optional<Navmesh>();
    }

    unique_ptr<NavmeshInternal, NavmeshDeleter> dt_data(new NavmeshInternal {
        dtAllocNavMesh(),
        dtAllocNavMeshQuery(),
    });

    if (!dt_data->detourMesh || !dt_data->detourQuery) {
      dtFree(nav_data);
      *err_msg = "OOM while allocating detour data";
      return optional<Navmesh>();
    }

    dtStatus status =
        dt_data->detourMesh->init(nav_data, nav_data_size, DT_TILE_FREE_DATA);
    if (dtStatusFailed(status)) {
        dtFree(nav_data);
        *err_msg = "failed to initialize detour navmesh";
        return optional<Navmesh>();
    }

    rcFreePolyMesh(inter.pmesh);
    inter.pmesh = nullptr;
    rcFreePolyMeshDetail(inter.dmesh);
    inter.dmesh = nullptr;

    vector<glm::vec3> navmesh_vertices =
        collectNavmeshVertices(dt_data->detourMesh);
    for (int i = 0; i < (int)navmesh_vertices.size(); i += 3) {
        glm::vec3 a = navmesh_vertices[i];
        glm::vec3 b = navmesh_vertices[i + 1];
        glm::vec3 c = navmesh_vertices[i + 2];

        glm::vec3 ba = b - a;
        glm::vec3 cb = c - b;

        float area = glm::length(glm::cross(ba, cb));
        if (area < 1e-6f) {
            *err_msg = "Generated navmesh had invalid zero area triangles";
            return optional<Navmesh>();
        }
    }

    
    if (!initQuery(dt_data->detourQuery, dt_data->detourMesh)) {
        *err_msg = "failed to initialize detour query engine";
        return optional<Navmesh>();
    }

    NavmeshRenderData render_data = buildRenderData(navmesh_vertices);
    return Navmesh {
        move(dt_data),
        cfg.bbox,
        move(render_data),
    };
}

// Saving and loading code largely copied from Habitat for compat
namespace HabitatNavmeshFormat {
    constexpr int NAVMESHSET_MAGIC =
        'M' << 24 | 'S' << 16 | 'E' << 8 | 'T';  //'MSET';
    constexpr int NAVMESHSET_VERSION = 1;

    struct NavMeshSetHeader {
        int magic;
        int version;
        int numTiles;
        dtNavMeshParams params;
    };

    struct NavMeshTileHeader {
        dtTileRef tileRef;
        int dataSize;
    };
}

optional<Navmesh> loadNavmesh(const char *file_path,
                              const char **err_msg)
{
    using namespace HabitatNavmeshFormat;
    ifstream file(file_path, ios::binary);

    if (!file.is_open()) {
        *err_msg = "failed to open navmesh";
        return optional<Navmesh>();
    }

    auto read = [&file](void *data, size_t num_bytes) {
        file.read((char *)data, num_bytes);

        return (size_t)file.gcount() == num_bytes;
    };

    // Read header.
    NavMeshSetHeader header;
    if (!read(&header, sizeof(NavMeshSetHeader))) {
        *err_msg = "navmesh has partial header";
        return optional<Navmesh>();
    }

    if (header.version != NAVMESHSET_VERSION) {
        *err_msg = "navmesh has wrong version";
        return optional<Navmesh>();
    }

    glm::vec3 bmin(INFINITY);
    glm::vec3 bmax(-INFINITY);

    unique_ptr<NavmeshInternal, NavmeshDeleter> internal(new NavmeshInternal {
        dtAllocNavMesh(),
        dtAllocNavMeshQuery(),
    });

    if (!internal->detourMesh || !internal->detourQuery) {
        *err_msg = "OOM while allocating detour data";
        return optional<Navmesh>();
    }
    dtStatus status = internal->detourMesh->init(&header.params);
    if (dtStatusFailed(status)) {
        *err_msg = "failed to initialize navmesh";
        return optional<Navmesh>();
    }

    if (!initQuery(internal->detourQuery, internal->detourMesh)) {
        *err_msg = "failed to initialize navmesh query engine";
        return optional<Navmesh>();
    }

    // Read tiles.
    for (int i = 0; i < header.numTiles; ++i) {
        NavMeshTileHeader tileHeader;
        if (!read(&tileHeader, sizeof(NavMeshTileHeader))) {
            *err_msg = "navmesh has partial tile header";
            return optional<Navmesh>();
        }

        if ((tileHeader.tileRef == 0u) || (tileHeader.dataSize == 0)) {
            break;
        }

        auto data = (uint8_t *)dtAlloc(tileHeader.dataSize, DT_ALLOC_PERM);
        if (!data) {
            break;
        }
        memset(data, 0, tileHeader.dataSize);
        if (!read(data, tileHeader.dataSize)) {
            dtFree(data);
            *err_msg = "navmesh has incomplete tile";
            return optional<Navmesh>();
        }

        internal->detourMesh->addTile(data, tileHeader.dataSize,
            DT_TILE_FREE_DATA, tileHeader.tileRef, nullptr);
        const dtMeshTile* tile =
            internal->detourMesh->getTileByRef(tileHeader.tileRef);
        bmin = glm::min(bmin, glm::make_vec3(tile->header->bmin));
        bmax = glm::max(bmax, glm::make_vec3(tile->header->bmax));
    }

    auto navmesh_vertices = collectNavmeshVertices(internal->detourMesh);
    auto render_data = buildRenderData(navmesh_vertices);

    return Navmesh {
        move(internal),
        AABB {
            bmin,
            bmax,
        },
        move(render_data),
    };
}

void saveNavmesh(const char *file_path, const Navmesh &navmesh)
{
    using namespace HabitatNavmeshFormat;
    ofstream file(file_path, ios::binary);

    if (!file.is_open()) {
        cerr << "Failed to open navmesh" << endl;
        return;
    }

    auto write = [&file](void *data, size_t num_bytes) {
        file.write((const char *)data, num_bytes);
    };

    const dtNavMesh *dt_nav = navmesh.internal->detourMesh;

    NavMeshSetHeader header {};
    header.magic = NAVMESHSET_MAGIC;
    header.version = NAVMESHSET_VERSION;
    header.numTiles = 0;
    for (int i = 0; i < dt_nav->getMaxTiles(); ++i) {
        const dtMeshTile* tile = dt_nav->getTile(i);
        if (!tile || !tile->header || (tile->dataSize == 0))
            continue;
        header.numTiles++;
    }
    memcpy(&header.params, dt_nav->getParams(), sizeof(dtNavMeshParams));

    write(&header, sizeof(NavMeshSetHeader));

    // Store tiles.
    for (int i = 0; i < dt_nav->getMaxTiles(); ++i) {
        const dtMeshTile* tile = dt_nav->getTile(i);
        if (!tile || !tile->header || (tile->dataSize == 0))
            continue;

        NavMeshTileHeader tileHeader{};
        tileHeader.tileRef = dt_nav->getTileRef(tile);
        tileHeader.dataSize = tile->dataSize;
        write(&tileHeader, sizeof(NavMeshTileHeader));

        write(tile->data, tile->dataSize);
    }

    cout << "Navmesh written to " << file_path << endl;
}

}
}
