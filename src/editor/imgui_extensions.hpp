#pragma once

#include <imgui.h>

namespace ImGuiEXT {

bool DragScalarNSeparateRange(const char* label,
                              ImGuiDataType data_type,
                              void* p_data,
                              int components,
                              float *v_speed,
                              const void* p_min,
                              const void* p_max,
                              const char* format,
                              ImGuiSliderFlags flags);

bool DragFloat3SeparateRange(const char* label,
                             float* p_data,
                             float *v_speed,
                             const float* p_min,
                             const float* p_max,
                             const char* format,
                             ImGuiSliderFlags flags = ImGuiSliderFlags_None);


void PushDisabled(bool disabled = true);
void PopDisabled();

}
