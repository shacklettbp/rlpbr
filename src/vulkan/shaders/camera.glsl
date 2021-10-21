#ifndef RLPBR_VK_CAMERA_GLSL_INCLUDED
#define RLPBR_VK_CAMERA_GLSL_INCLUDED

#include "inputs.glsl"

void computeCameraRay(in Camera camera, in u32vec3 idx, in vec2 jitter,
                      out vec3 ray_origin, out vec3 ray_dir)
{
    vec2 jittered_raster = vec2(idx.x, idx.y) + jitter;

    vec2 screen = vec2((2.f * jittered_raster.x) / RES_X - 1,
                       (2.f * jittered_raster.y) / RES_Y - 1);

    vec3 right = camera.right * camera.rightScale;
    vec3 up = camera.up * camera.upScale;

    ray_origin = camera.origin;
    ray_dir = normalize(
        right * screen.x + up * screen.y + camera.view);
}

vec2 getScreenSpacePosition(Camera camera, vec3 world_pos)
{
    vec3 to_pos = world_pos - camera.origin;

    vec3 camera_space = vec3(dot(to_pos, camera.right),
                             dot(to_pos, camera.up),
                             dot(to_pos, camera.view));

    return vec2(camera_space.x / camera.rightScale / camera_space.z,
                camera_space.y / camera.upScale / camera_space.z);
}

i32vec2 getPixelCoords(vec2 screen_space)
{
    vec2 offset = (screen_space + 1.f) / 2.f;
    return i32vec2(offset.x * RES_X, offset.y * RES_Y);
}

#endif
