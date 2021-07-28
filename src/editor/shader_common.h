#ifndef EDITOR_SHADER_COMMON_H_INCLUDED
#define EDITOR_SHADER_COMMON_H_INCLUDED

struct DrawPushConst {
    mat4 proj;
    mat4 view;
};

struct NavmeshPushConst {
    DrawPushConst base;
    vec4 color;
};

#endif
