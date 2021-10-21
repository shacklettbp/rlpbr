#ifndef RLPBR_VK_GEOMETRY_GLSL_INCLUDED
#define RLPBR_VK_GEOMETRY_GLSL_INCLUDED

#include "inputs.glsl"
#include "unpack.glsl"

struct HitInfo {
    vec3 position;
    vec3 geoNormal;
    float triArea;
    TangentFrame tangentFrame;
    Material material;
};

struct RayCone {
    vec3 curOrigin;
    float totalDistance;
    float pixelSpread;
};

u32vec3 fetchTriangleIndices(IdxRef idx_ref, uint32_t index_offset)
{
    // FIXME: maybe change all this to triangle offset
    return u32vec3(
        idx_ref[nonuniformEXT(index_offset)].idx,
        idx_ref[nonuniformEXT(index_offset + 1)].idx,
        idx_ref[nonuniformEXT(index_offset + 2)].idx);
}

Triangle fetchTriangle(VertRef vert_ref, IdxRef idx_ref, uint32_t index_offset)
{
    u32vec3 indices = fetchTriangleIndices(idx_ref, index_offset);

    return Triangle(
        unpackVertex(vert_ref, indices.x),
        unpackVertex(vert_ref, indices.y),
        unpackVertex(vert_ref, indices.z));
}

#define INTERPOLATE_ATTR(a, b, c, barys) \
    (a + barys.x * (b - a) + \
     barys.y * (c - a))

vec3 interpolatePosition(vec3 a, vec3 b, vec3 c, vec2 barys)
{
    return INTERPOLATE_ATTR(a, b, c, barys);
}

vec3 interpolateNormal(vec3 a, vec3 b, vec3 c, vec2 barys)
{
    return INTERPOLATE_ATTR(a, b, c, barys);
}

vec4 interpolateCombinedTangent(vec4 a, vec4 b, vec4 c, vec2 barys)
{
    return INTERPOLATE_ATTR(a, b, c, barys);
}

vec2 interpolateUV(vec2 a, vec2 b, vec2 c, vec2 barys)
{
    return INTERPOLATE_ATTR(a, b, c, barys);
}

#undef INTERPOLATE_ATTR

void computeTriangleProperties(in vec3 a, in vec3 b, in vec3 c,
                               out vec3 geo_normal,
                               out float area)
{
    vec3 v1 = b - a;
    vec3 v2 = c - a;

    vec3 cp = cross(v1, v2);
    float len = length(cp);

    geo_normal = cp / len;
    area = 0.5f * len;
}

TangentFrame computeTangentFrame(Triangle hit_tri,
                                 vec2 barys,
                                 MaterialParams mat_params,
                                 uint32_t base_tex_idx,
                                 vec2 uv,
                                 vec4 uv_derivs)
{
    vec3 n = normalize(interpolateNormal(hit_tri.a.normal,
                                         hit_tri.b.normal,
                                         hit_tri.c.normal,
                                         barys));

    vec4 combined = interpolateCombinedTangent(hit_tri.a.tangentAndSign,
                                               hit_tri.b.tangentAndSign,
                                               hit_tri.c.tangentAndSign,
                                               barys);

    vec3 t = combined.xyz;
    float bitangent_sign = combined.w;

    vec3 b = cross(n, t) * bitangent_sign;

    vec3 perturb = vec3(0, 0, 1);
    if (bool(mat_params.flags & MaterialFlagsHasNormalMap)) {
        vec2 xy = textureGrad(sampler2D(
            textures[base_tex_idx + TextureConstantsNormalOffset],
            repeatSampler), uv, uv_derivs.xy, uv_derivs.zw).xy;

        vec2 centered = xy * 2.0 - 1.0;
        float length2 = clamp(dot(centered, centered), 0.0, 1.0);

        perturb = vec3(centered.x, centered.y, sqrt(1.0 - length2));
    } 

    // Perturb normal
    n = normalize(t * perturb.x + b * perturb.y + n * perturb.z);
    // Ensure perpendicular (if new normal is parallel to old tangent... boom)
    t = normalize(t - n * dot(n, t));
    b = cross(n, t) * bitangent_sign;

    return TangentFrame(t, b, n);
}

TangentFrame tangentFrameToWorld(mat4x3 o2w, mat4x3 w2o, TangentFrame frame,
                                 vec3 geo_normal)
{
    frame.tangent = normalize(transformVector(o2w, frame.tangent));
    frame.bitangent = normalize(transformVector(o2w, frame.bitangent));
    frame.normal = normalize(transformNormal(w2o, frame.normal));

    // There's some stupidity with frame.normal being backfacing relative
    // to the outgoing vector when it shouldn't be due to normal
    // interpolation / mapping. Therefore instead of flipping the tangent
    // frame based on ray direction, flip it based on the geo normal,
    // which has been aligned with the outgoing vector already
    if (dot(frame.normal, geo_normal) < 0.f) {
        frame.tangent *= -1.f;
        frame.bitangent *= -1.f;
        frame.normal *= -1.f;
    }
                                                               
    return frame;                                              
}

void getHitParams(in rayQueryEXT ray_query, out vec2 barys,
                  out uint32_t tri_idx, out uint32_t material_offset,
                  out uint32_t geo_idx, out uint32_t mesh_offset,
                  out mat4x3 o2w, out mat4x3 w2o)
{
    barys = rayQueryGetIntersectionBarycentricsEXT(ray_query, true);

    tri_idx =
        uint32_t(rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true));

    material_offset = uint32_t(
        rayQueryGetIntersectionInstanceCustomIndexEXT(ray_query, true));

    geo_idx = 
        uint32_t(rayQueryGetIntersectionGeometryIndexEXT(ray_query, true));

    mesh_offset = uint32_t(
        rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(
            ray_query, true));

    o2w = rayQueryGetIntersectionObjectToWorldEXT(ray_query, true);
    w2o = rayQueryGetIntersectionWorldToObjectEXT(ray_query, true);
}

uint32_t getHitInstance(in rayQueryEXT ray_query)
{
    return uint32_t(rayQueryGetIntersectionInstanceIdEXT(ray_query, true));
}

// Ray Tracing Gems II Chapter 7
vec4 UVDerivsFromRayCone(vec3 ray_dir,
                       vec3 world_normal,
                       float tri_area,
                       float cone_width,
                       vec2 a_uv,
                       vec2 b_uv,
                       vec2 c_uv)
{
    vec2 vUV10 = b_uv - a_uv;
	vec2 vUV20 = c_uv - a_uv;
	float fQuadUVArea = abs(vUV10.x * vUV20.y - vUV20.x * vUV10.y);

	// Since the ray cone's width is in world-space, we need to compute the quad
	// area in world-space as well to enable proper ratio calculation
	float fQuadArea = 2.f * tri_area;

	float fDistTerm = abs(cone_width);
	float fNormalTerm = abs(dot(ray_dir, world_normal));
	float fProjectedConeWidth = cone_width / fNormalTerm;
	float fVisibleAreaRatio =
        (fProjectedConeWidth * fProjectedConeWidth) / fQuadArea;

	float fVisibleUVArea = fQuadUVArea * fVisibleAreaRatio;
	float fULength = sqrt(fVisibleUVArea);
	return vec4(fULength, 0, 0, fULength);
}

RayCone initRayCone(Camera cam)
{
    return RayCone(cam.origin, 0, atan(2.f * cam.upScale / float(RES_Y)));
}

float updateRayCone(in vec3 position, inout RayCone cone)
{
    float new_dist = length(position - cone.curOrigin);

    cone.totalDistance += new_dist;
    cone.curOrigin = position;

    return cone.pixelSpread * cone.totalDistance;
}

HitInfo processHit(in rayQueryEXT ray_query, in Environment env,
                   in vec3 outgoing_dir, inout RayCone ray_cone)
{
    vec2 barys;
    uint32_t tri_idx, material_offset, geo_idx, mesh_offset;
    mat4x3 o2w, w2o;
    getHitParams(ray_query, barys, tri_idx,
                 material_offset, geo_idx, mesh_offset, o2w, w2o);

    SceneAddresses scene_addrs = sceneAddrs[env.sceneID];

    MeshInfo mesh_info =
        unpackMeshInfo(scene_addrs.meshAddr, mesh_offset + geo_idx);

    uint32_t index_offset = mesh_info.indexOffset + tri_idx * 3;
    Triangle hit_tri =
        fetchTriangle(scene_addrs.vertAddr, scene_addrs.idxAddr, index_offset);
    vec3 world_a = transformPosition(o2w, hit_tri.a.position);
    vec3 world_b = transformPosition(o2w, hit_tri.b.position);
    vec3 world_c = transformPosition(o2w, hit_tri.c.position);
    vec3 world_geo_normal;
    float world_tri_area;
    computeTriangleProperties(world_a, world_b, world_c, world_geo_normal,
                              world_tri_area);
    vec3 world_position =
        interpolatePosition(world_a, world_b, world_c, barys);

    if (dot(world_geo_normal, -outgoing_dir) < 0.f) {
        world_geo_normal *= -1.f;
    }

    float cone_width = updateRayCone(world_position, ray_cone);
    vec4 uv_derivs = UVDerivsFromRayCone(outgoing_dir,
                                         world_geo_normal,
                                         world_tri_area,
                                         cone_width,
                                         hit_tri.a.uv, hit_tri.b.uv,
                                         hit_tri.c.uv);

    vec2 uv = interpolateUV(hit_tri.a.uv, hit_tri.b.uv, hit_tri.c.uv,
                            barys);
    // Unpack materials
    uint32_t material_id = unpackMaterialID(
        env.baseMaterialOffset + material_offset + geo_idx);

    MaterialParams material_params =
        unpackMaterialParams(scene_addrs.matAddr, material_id);

    uint32_t mat_texture_offset = env.baseTextureOffset + 1 +
        material_id * TextureConstantsTexturesPerMaterial;

    TangentFrame obj_tangent_frame =
        computeTangentFrame(hit_tri, barys, material_params,
                            mat_texture_offset, uv, uv_derivs);

    TangentFrame world_tangent_frame =
        tangentFrameToWorld(o2w, w2o, obj_tangent_frame, world_geo_normal);

    Material material = processMaterial(material_params,
        mat_texture_offset, uv, uv_derivs);

    return HitInfo(world_position, world_geo_normal,
                   world_tri_area, world_tangent_frame, material);
}

vec3 worldToLocalIncoming(vec3 v, TangentFrame frame) 
{
    return vec3(dot(v, frame.tangent), dot(v, frame.bitangent),
        dot(v, frame.normal));
}

vec3 worldToLocalOutgoing(vec3 v, TangentFrame frame)
{
    // Hack from frostbite / filament
    // Consider Falcor strategy if can find reference
    return vec3(dot(v, frame.tangent), dot(v, frame.bitangent),
                abs(dot(v, frame.normal)) + 1e-5f);
}

vec3 localToWorld(vec3 v, TangentFrame frame)
{
    return v.x * frame.tangent + v.y * frame.bitangent +
        v.z * frame.normal;
}

#endif
